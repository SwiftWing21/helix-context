"""
Dimensional-lock benchmark — measure how recall scales with axis count.

Standard NIAH (bench_needle_1000.py) generates one query per needle and
grades recall@1. That works for single-axis RAG systems but
under-measures multi-axis retrieval engines like helix, where every
gene is addressed by 12 retrieval signals + 4-5 attribution axes
simultaneously. Single-axis recall ignores 11 of the 12 narrowing
dimensions the system actually has.

This bench generates **four query variants per needle** with
progressively more axis information, then measures recall@k at each
variant level. The *shape of the curve* IS the diagnostic:

    flat    → retrieval pipeline doesn't compose axes (broken)
    rising  → multi-axis index works as designed (healthy)
    plateau → over-specification doesn't penalize (good)
    falls   → axis-weighting bug at high specificity

See docs/BENCHMARK_RATIONALE.md for the full discovery story.

Variant grid (per needle):

    1-axis: "What is the value of {key}?"
    2-axis: "What is the value of {key} in {project}?"
    3-axis: "What is the {key} configured in {project} {module}?"
    4-axis: "What is the {key} value in {project}/{module}/{filename}?"

Reuses harvest_needles + categorize from bench_needle_1000.py so
the needles are directly comparable to prior NIAH runs.

Usage:
    python benchmarks/bench_dimensional_lock.py
    N=50 SEED=42 HELIX_MODEL=qwen3:8b python benchmarks/bench_dimensional_lock.py
    GENOME_DB=F:/Projects/helix-context/genome-bench-N1000.db \\
        python benchmarks/bench_dimensional_lock.py

Output:
    benchmarks/dimensional_lock_results.json — per-needle × per-variant
        retrieval/answer/latency rows + summary curve
    benchmarks/dimensional_lock_results.incremental.jsonl — streamed for
        crash-resilience
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import httpx

# Reuse the harvest, categorize, and quality-filter helpers from the
# existing N=1000 bench so the needle population is identical and runs
# are directly comparable. Keeps the dimensional-lock numbers honest
# against historical NIAH baselines.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bench_needle_1000 import (  # noqa: E402
    harvest_needles,
    categorize,
)


HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
GENOME_DB = os.environ.get(
    "GENOME_DB",
    "F:/Projects/helix-context/genome-bench-N1000.db",
)
MODEL = os.environ.get("HELIX_MODEL", "qwen3:8b")
N_TOTAL = int(os.environ.get("N", "50"))
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_PATH = os.environ.get(
    "OUTPUT",
    str(_HERE / "dimensional_lock_results.json"),
)
INCREMENTAL_PATH = OUTPUT_PATH.replace(".json", ".incremental.jsonl")

REQUEST_TIMEOUT_S = 90.0
ASK_PROXY = os.environ.get("ASK_PROXY", "1") not in ("0", "false", "False")


# -- Path → (project, module, filename) extraction ---------------------

def _split_source(source_id: str) -> tuple[str, str, str]:
    """Extract (project, module, filename) heuristic from a path.

    project  = first dir-name after a "Projects" / "SteamLibrary" /
               "OpenModels" anchor (or first dir if no anchor)
    module   = the immediate parent dir of the file
    filename = basename without extension

    These are the natural axes that a real query would specify.
    """
    if not source_id:
        return ("", "", "")

    # Normalize separators
    norm = source_id.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]

    project = ""
    anchors = ("projects", "steamlibrary", "openmodels", "documents")
    for i, p in enumerate(parts):
        if p.lower() in anchors and i + 1 < len(parts):
            project = parts[i + 1]
            # The "module" is the next directory after the project
            # (or the file's parent dir if there are no intermediates)
            module = ""
            for j in range(i + 2, len(parts) - 1):
                module = parts[j]
                break  # just the first sub-dir
            filename = parts[-1].rsplit(".", 1)[0]
            return (project, module, filename)

    # No anchor matched — use first dir + parent + basename
    if len(parts) >= 3:
        return (parts[0], parts[-2], parts[-1].rsplit(".", 1)[0])
    if len(parts) >= 2:
        return (parts[0], "", parts[-1].rsplit(".", 1)[0])
    if parts:
        return ("", "", parts[-1].rsplit(".", 1)[0])
    return ("", "", "")


# -- Variant query generation -----------------------------------------

def make_variants(needle: dict) -> list[dict]:
    """Generate 4 query variants of progressively higher axis specificity.

    Returns a list of {variant_id, axes, query} dicts. The variant_id is
    1..4; axes is a string describing what's specified (used for
    debug/visualization).

    ORDER MODES (selectable via DEWEY env var):
      DEWEY=0 (default, original): project → module → filename
        "outside-in" — starts broad, narrows via classification.
      DEWEY=1: filename → project → module
        "Dewey Decimal" — filename as primary anchor (like a call number),
        project/module layered on as classification metadata. Hypothesis
        test: does filename-first produce a rising curve where project-first
        produces a falling one? (See 2026-04-14 dim-lock N=200 run that
        showed recall@1 = [7.0, 5.5, 4.0, 17.5] under the default order —
        falling until filename rescued at axis 4.)
    """
    key = needle["key"]
    src = needle.get("source", "")
    project, module, filename = _split_source(src)

    # Display-friendly key for natural-language phrasing
    key_phrase = key.replace("_", " ")

    dewey = os.environ.get("DEWEY", "0") == "1"

    variants = []

    # Variant 1: just the key (1 axis) — identical in both modes so the
    # 1-axis baseline is directly comparable across runs.
    variants.append({
        "variant_id": 1,
        "axes": "key",
        "query": f"What is the value of {key_phrase}?",
    })

    if dewey:
        # Dewey / filename-first ordering.
        # Variant 2: key + filename (2 axes)
        if filename:
            variants.append({
                "variant_id": 2,
                "axes": "key+filename",
                "query": f"What is the value of {key_phrase} in {filename}?",
            })
        else:
            cat = needle.get("category", "")
            variants.append({
                "variant_id": 2,
                "axes": "key+category (no filename)",
                "query": f"What is the value of {key_phrase} in the {cat} source?",
            })

        # Variant 3: key + filename + project (3 axes)
        if filename and project:
            variants.append({
                "variant_id": 3,
                "axes": "key+filename+project",
                "query": f"What is the value of {key_phrase} in {filename} ({project})?",
            })
        else:
            variants.append({
                "variant_id": 3,
                "axes": variants[-1]["axes"] + " (no project)",
                "query": variants[-1]["query"],
            })

        # Variant 4: key + filename + project + module (4 axes)
        full_locator = "/".join(p for p in (project, module, filename) if p)
        if filename and full_locator:
            variants.append({
                "variant_id": 4,
                "axes": "key+filename+project+module",
                "query": f"What is the {key_phrase} value in {full_locator}?",
            })
        else:
            variants.append({
                "variant_id": 4,
                "axes": variants[-1]["axes"] + " (partial locator)",
                "query": variants[-1]["query"],
            })
    else:
        # Original / project-first ordering.
        # Variant 2: key + project (2 axes)
        if project:
            variants.append({
                "variant_id": 2,
                "axes": "key+project",
                "query": f"What is the value of {key_phrase} in {project}?",
            })
        else:
            # No project token available — fall back to category as project hint
            cat = needle.get("category", "")
            variants.append({
                "variant_id": 2,
                "axes": "key+category",
                "query": f"What is the value of {key_phrase} in the {cat} source?",
            })

        # Variant 3: key + project + module (3 axes)
        locator = " ".join(p for p in (project, module) if p)
        if locator:
            variants.append({
                "variant_id": 3,
                "axes": "key+project+module",
                "query": f"What is the {key_phrase} configured in {locator}?",
            })
        else:
            # No module — duplicate variant 2 to keep the grid square; flag it
            variants.append({
                "variant_id": 3,
                "axes": "key+project (no module)",
                "query": variants[-1]["query"],
            })

        # Variant 4: key + project + module + filename (4 axes)
        full_locator = "/".join(p for p in (project, module, filename) if p)
        if full_locator and filename:
            variants.append({
                "variant_id": 4,
                "axes": "key+project+module+filename",
                "query": f"What is the {key_phrase} value in {full_locator}?",
            })
        else:
            variants.append({
                "variant_id": 4,
                "axes": "key+project+module (no filename)",
                "query": variants[-1]["query"],
            })

    return variants


# -- Single-variant evaluation ----------------------------------------

def eval_variant(
    client: httpx.Client,
    needle: dict,
    variant: dict,
) -> dict:
    """Run one query variant through /context (and optionally the proxy).

    Returns a row dict with retrieval / answer / latency.
    """
    result = {
        "gene_id": needle["gene_id"],
        "category": needle.get("category", ""),
        "key": needle["key"],
        "value": needle["value"],
        "variant_id": variant["variant_id"],
        "axes": variant["axes"],
        "query": variant["query"],
        "retrieved": False,
        "answered": False,
        "answer_in_context": False,
        "context_latency_s": 0.0,
        "proxy_latency_s": 0.0,
        "genes_expressed": 0,
        "ellipticity": 0.0,
        "error": None,
    }

    # Step 1: /context retrieval
    t0 = time.time()
    try:
        resp = client.post(
            f"{HELIX_URL}/context",
            json={
                "query": variant["query"],
                "decoder_mode": "none",
                "verbose": True,
                "clean": True,  # fresh state per variant
            },
            timeout=REQUEST_TIMEOUT_S,
        )
    except Exception as e:
        result["error"] = f"context: {e}"
        result["context_latency_s"] = time.time() - t0
        return result

    result["context_latency_s"] = time.time() - t0
    if resp.status_code != 200:
        result["error"] = f"context HTTP {resp.status_code}"
        return result

    body = resp.json()
    if isinstance(body, list):
        body = body[0] if body else {}

    content = body.get("content", "") or ""
    health = body.get("context_health", {}) or {}
    agent = body.get("agent", {}) or {}

    result["genes_expressed"] = health.get("genes_expressed", 0)
    result["ellipticity"] = health.get("ellipticity", 0.0)

    # Retrieval grade: did the needle gene_id appear in expressed citations?
    cits = agent.get("citations", []) or []
    cit_gene_ids = {c.get("gene_id") for c in cits}
    if needle["gene_id"] in cit_gene_ids:
        result["retrieved"] = True

    # Answer-in-context grade: even if exact gene wasn't returned, did
    # the needle's value appear anywhere in the expressed context?
    # This is the FAIR retrieval metric for multi-axis systems — it
    # measures "did we surface enough information to answer" rather
    # than "did we guess the exact gene the bench picked."
    if needle["value"].lower() in content.lower():
        result["answer_in_context"] = True

    # Step 2 (optional): proxy answer extraction. Skip for speed in the
    # primary diagnostic; the in-context check above is the structurally
    # honest measurement. Proxy adds the "can the model extract" layer.
    if ASK_PROXY:
        prompt = (
            f"Using the context provided, answer concisely: {variant['query']}\n"
            f"If the answer isn't present, say 'unknown'."
        )
        t1 = time.time()
        try:
            proxy_resp = client.post(
                f"{HELIX_URL}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.0,
                },
                timeout=REQUEST_TIMEOUT_S,
            )
        except Exception as e:
            result["error"] = f"proxy: {e}"
            result["proxy_latency_s"] = time.time() - t1
            return result

        result["proxy_latency_s"] = time.time() - t1
        if proxy_resp.status_code == 200:
            try:
                pj = proxy_resp.json()
                ans = pj["choices"][0]["message"]["content"]
            except Exception:
                ans = ""
            if needle["value"].lower() in (ans or "").lower():
                result["answered"] = True

    return result


# -- Main -------------------------------------------------------------

def main() -> int:
    print(f"=== Dimensional-lock benchmark ===")
    print(f"Genome:  {GENOME_DB}")
    print(f"Server:  {HELIX_URL}")
    print(f"Model:   {MODEL}")
    print(f"Seed:    {SEED}")
    print(f"N:       {N_TOTAL} needles × 4 variants = {N_TOTAL * 4} queries")
    print()

    # Sanity ping
    try:
        h = httpx.get(f"{HELIX_URL}/health", timeout=5).json()
        print(f"Server: ok, ribosome={h.get('ribosome')}, genes={h.get('genes')}")
    except Exception as e:
        print(f"Server unreachable: {e}", file=sys.stderr)
        return 2

    print("Harvesting needles...")
    needles = harvest_needles(GENOME_DB, N_TOTAL, SEED)
    print(f"Selected {len(needles)} needles\n")

    # Truncate any prior incremental file
    Path(INCREMENTAL_PATH).write_text("")

    rows: list[dict] = []
    start = time.time()

    with httpx.Client() as client:
        for i, needle in enumerate(needles, 1):
            variants = make_variants(needle)
            for variant in variants:
                row = eval_variant(client, needle, variant)
                rows.append(row)
                with open(INCREMENTAL_PATH, "a") as f:
                    f.write(json.dumps(row) + "\n")

            done = i
            elapsed_min = (time.time() - start) / 60
            if done % 10 == 0 or done == len(needles):
                print(
                    f"  [{done:>3}/{len(needles)}] elapsed={elapsed_min:5.1f}m  "
                    f"queries={done * 4}"
                )

    elapsed_min = (time.time() - start) / 60
    print()
    print("=" * 64)
    print(f"N = {len(needles)} needles × 4 variants = {len(rows)} queries")
    print(f"Total time: {elapsed_min:.1f} min")
    print()

    # Aggregate by variant_id
    print("== Recall curve (the diagnostic) ==")
    print()
    print(f"  {'axes':10s}  {'variant':12s}  {'recall@1':10s}  "
          f"{'in-context':12s}  {'answered':10s}  {'avg ctx s':10s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")

    summary = {}
    prev_recall = None
    for v in (1, 2, 3, 4):
        sub = [r for r in rows if r["variant_id"] == v]
        if not sub:
            continue
        n = len(sub)
        retr = sum(1 for r in sub if r["retrieved"])
        in_ctx = sum(1 for r in sub if r["answer_in_context"])
        ans = sum(1 for r in sub if r["answered"])
        ctx_lat = sum(r["context_latency_s"] for r in sub) / n
        axes_label = sub[0]["axes"][:24]

        recall_pct = 100 * retr / n
        in_ctx_pct = 100 * in_ctx / n
        ans_pct = 100 * ans / n

        lift = ""
        if prev_recall is not None:
            delta = recall_pct - prev_recall
            sign = "+" if delta >= 0 else ""
            lift = f"  ({sign}{delta:.1f}pp)"
        prev_recall = recall_pct

        print(
            f"  {v} axis{' s' if v>1 else '  '}  {axes_label:12s}  "
            f"{recall_pct:5.1f}%   {in_ctx_pct:5.1f}%        "
            f"{ans_pct:5.1f}%      {ctx_lat:5.1f}{lift}"
        )

        summary[f"variant_{v}"] = {
            "n": n,
            "axes": axes_label,
            "retrieval_pct": round(recall_pct, 1),
            "in_context_pct": round(in_ctx_pct, 1),
            "answer_pct": round(ans_pct, 1),
            "avg_context_latency_s": round(ctx_lat, 2),
        }

    # Curve shape diagnostic
    print()
    print("== Curve shape diagnostic ==")
    recalls = [summary.get(f"variant_{v}", {}).get("retrieval_pct", 0)
               for v in (1, 2, 3, 4)]
    if all(r is not None for r in recalls) and len(recalls) == 4:
        deltas = [recalls[i+1] - recalls[i] for i in range(3)]
        print(f"  Recall@1 by axis count: {recalls}")
        print(f"  Deltas (pp):            {[round(d, 1) for d in deltas]}")
        if all(d > 0 for d in deltas):
            print(f"  [OK] MONOTONICALLY RISING — multi-axis index composing correctly")
        elif all(abs(r - recalls[0]) < 5 for r in recalls):
            print(f"  [X] FLAT — retrieval not composing axes (broken)")
        elif deltas[0] > 0 and deltas[1] > 0 and deltas[2] < -5:
            print(f"  [!] OVER-FIT PENALTY at variant 4 — axis weighting bug")
        else:
            print(f"  [!] MIXED — see deltas for partial composition")

    # Save full output
    out = {
        "timestamp": time.time(),
        "genome_snapshot": GENOME_DB,
        "model": MODEL,
        "seed": SEED,
        "n_needles": len(needles),
        "n_queries": len(rows),
        "summary": summary,
        "rows": rows,
    }
    Path(OUTPUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Incremental log:  {INCREMENTAL_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

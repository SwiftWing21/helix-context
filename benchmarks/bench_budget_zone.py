r"""
Benchmark: Budget-zone gene-cap spike.

Re-runs the needle set from ``bench_needle.py`` at several simulated
``prompt_tokens`` values to measure how a budget-zone cap on the expressed
gene count affects retrieval recall, gene count, and token footprint.

Usage:

    # 1. Start the server with the flag ON:
    HELIX_BUDGET_ZONE=1 python -m uvicorn helix_context.server:app \
        --host 127.0.0.1 --port 11437

    # 2. Run the bench:
    python benchmarks/bench_budget_zone.py

The bench calls ``POST /context`` directly (no proxy) so the only variable
is ``prompt_tokens``. One row per (zone, needle). Two summary tables:

    1. Per-zone aggregates (R@k, mean genes, mean ellipticity, est tokens)
    2. Per-needle delta vs. baseline (which queries got hurt)

If the server has HELIX_BUDGET_ZONE unset or false, the cap is a no-op and
every zone should look identical to the baseline — a useful sanity check.
"""

from __future__ import annotations

import json
import os
import sys
import time
from statistics import mean
from typing import List, Optional

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")

# Reuse the needle set — same structure as bench_needle.py but inlined so
# we can edit this bench without touching the upstream one.
NEEDLES = [
    {"name": "helix_port",              "query": "What port does the Helix proxy server listen on?",                             "accept": ["11437"]},
    {"name": "scorerift_threshold",     "query": "What is the divergence threshold that triggers alerts in ScoreRift?",          "accept": ["0.15", ".15"]},
    {"name": "biged_skills_count",      "query": "How many skills does the BigEd fleet have?",                                    "accept": ["125", "129"]},
    {"name": "bookkeeper_monetary",     "query": "What type should be used for monetary values in BookKeeper instead of float?", "accept": ["decimal"]},
    {"name": "helix_pipeline_steps",    "query": "How many steps are in the Helix expression pipeline?",                          "accept": ["6", "six"]},
    {"name": "biged_rust_binary_size",  "query": "What is the binary size of the Rust BigEd build in MB?",                       "accept": ["11", "11mb", "11 mb"]},
    {"name": "genome_compression",      "query": "What is the target compression ratio for Helix Context?",                       "accept": ["5x", "5:1", "5 to 1"]},
    {"name": "scorerift_preset_dims",   "query": "How many dimensions does the Python preset in ScoreRift check?",               "accept": ["8", "eight"]},
    {"name": "helix_ribosome_budget",   "query": "How many tokens are allocated for the ribosome decoder prompt?",                "accept": ["3000", "3k", "3,000"]},
    {"name": "biged_default_model",     "query": "What is the default local model used by BigEd for conductor tasks?",           "accept": ["qwen3", "qwen"]},
]

# Simulated prompt-size sweeps. Each one lands in a distinct zone at a
# 128k window so we can see every band of behavior in one run.
#   None    — no signal provided (baseline, no cap regardless of env flag)
#   5_000   — clean zone      (<25%)  — expected no-op
#   40_000  — soft zone       (25-40%) — cap 12 (no-op vs default)
#   60_000  — pressure zone   (40-60%) — cap 6
#   90_000  — cap zone        (60-80%) — cap 3
#   120_000 — emergency zone  (80%+)   — cap 1
SWEEP_POINTS: List[Optional[int]] = [None, 5_000, 40_000, 60_000, 90_000, 120_000]

ZONE_LABELS = {
    None:    "baseline",
    5_000:   "clean",
    40_000:  "soft",
    60_000:  "pressure",
    90_000:  "cap",
    120_000: "emergency",
}


def query_needle(client: httpx.Client, needle: dict, prompt_tokens: Optional[int]) -> dict:
    body = {
        "query": needle["query"],
        "decoder_mode": "none",
        "clean": True,  # reset per-session caches between independent needles
    }
    if prompt_tokens is not None:
        body["prompt_tokens"] = prompt_tokens

    t0 = time.time()
    try:
        resp = client.post(f"{HELIX_URL}/context", json=body)
    except Exception as exc:
        return {"name": needle["name"], "ok": False, "error": str(exc),
                "prompt_tokens": prompt_tokens}

    latency_s = round(time.time() - t0, 3)
    if resp.status_code != 200:
        return {"name": needle["name"], "ok": False,
                "error": f"HTTP {resp.status_code}", "latency_s": latency_s,
                "prompt_tokens": prompt_tokens}

    data = resp.json()
    entry = data[0] if data else {}
    content = entry.get("content", "")
    health = entry.get("context_health", {})

    accept = needle["accept"]
    found = any(a.lower() in content.lower() for a in accept)

    return {
        "name": needle["name"],
        "ok": True,
        "prompt_tokens": prompt_tokens,
        "zone": ZONE_LABELS.get(prompt_tokens, f"{prompt_tokens}"),
        "found": found,
        "genes_expressed": health.get("genes_expressed", 0),
        "ellipticity": round(health.get("ellipticity", 0.0), 3),
        "status": health.get("status", "unknown"),
        "content_chars": len(content),
        "latency_s": latency_s,
    }


def summarize_zone(rows: List[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    total = len(rows)
    if not ok:
        return {"count": total, "recall": 0.0, "mean_genes": 0,
                "mean_ellipticity": 0.0, "mean_chars": 0, "mean_latency_s": 0.0}
    return {
        "count": total,
        "recall": round(sum(1 for r in ok if r["found"]) / total, 3),
        "mean_genes": round(mean(r["genes_expressed"] for r in ok), 2),
        "mean_ellipticity": round(mean(r["ellipticity"] for r in ok), 3),
        "mean_chars": int(mean(r["content_chars"] for r in ok)),
        "mean_latency_s": round(mean(r["latency_s"] for r in ok), 3),
    }


def main() -> int:
    client = httpx.Client(timeout=120)

    try:
        stats = client.get(f"{HELIX_URL}/stats", timeout=10).json()
    except Exception as exc:
        print(f"Cannot reach Helix at {HELIX_URL}: {exc}")
        return 1

    print(f"Genome: {stats.get('total_genes', '?')} genes, "
          f"{stats.get('compression_ratio', 0):.1f}x compression")
    server_flag = stats.get("config", {}).get("budget_zone_enabled")
    if server_flag is None:
        print("Server /stats does not expose budget_zone_enabled — "
              "upgrade the server image (added with this spike).")
    elif not server_flag:
        print("⚠  Server-side HELIX_BUDGET_ZONE is OFF. All zones will "
              "produce identical numbers (no-op). Restart the server with "
              "the flag enabled for a meaningful Phase 2 run.")
    else:
        print("Server-side HELIX_BUDGET_ZONE is ON — caps are active.")
    print()

    all_rows: List[dict] = []
    per_zone_rows: dict = {}

    for prompt_tokens in SWEEP_POINTS:
        zone = ZONE_LABELS.get(prompt_tokens, f"{prompt_tokens}")
        zone_rows = []
        hits = 0
        for needle in NEEDLES:
            row = query_needle(client, needle, prompt_tokens)
            zone_rows.append(row)
            all_rows.append(row)
            if row.get("ok") and row.get("found"):
                hits += 1
        per_zone_rows[zone] = zone_rows
        summary = summarize_zone(zone_rows)
        print(
            f"[{zone:>9}] prompt_tokens={str(prompt_tokens):<7}  "
            f"R@k={summary['recall']*100:5.1f}%  "
            f"genes={summary['mean_genes']:>4.1f}  "
            f"ell={summary['mean_ellipticity']:>5.3f}  "
            f"chars={summary['mean_chars']:>5}  "
            f"lat={summary['mean_latency_s']:>5.2f}s"
        )

    # Per-needle delta table — which queries lose recall under pressure?
    print("\n=== Per-needle recall by zone ===")
    header_zones = [ZONE_LABELS.get(p, f"{p}") for p in SWEEP_POINTS]
    print(f"{'needle':<26}  " + "  ".join(f"{z:>9}" for z in header_zones))
    for needle in NEEDLES:
        cells = []
        for prompt_tokens in SWEEP_POINTS:
            zone = ZONE_LABELS.get(prompt_tokens, f"{prompt_tokens}")
            row = next((r for r in per_zone_rows[zone] if r["name"] == needle["name"]), None)
            if row and row.get("ok"):
                cells.append("+" if row["found"] else "-")
            else:
                cells.append("e")
        cell_str = "  ".join(f"{c:>9}" for c in cells)
        print(f"{needle['name']:<26}  {cell_str}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "helix_budget_zone_client_env": os.environ.get("HELIX_BUDGET_ZONE"),
        "genome_genes": stats.get("total_genes"),
        "sweep_points": SWEEP_POINTS,
        "per_zone_summary": {
            ZONE_LABELS.get(p, f"{p}"): summarize_zone(per_zone_rows[ZONE_LABELS.get(p, f"{p}")])
            for p in SWEEP_POINTS
        },
        "rows": all_rows,
    }
    out_path = os.path.join(os.path.dirname(__file__), "budget_zone_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

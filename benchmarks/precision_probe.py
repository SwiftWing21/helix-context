"""
Precision probe — Stage 1: determinism check + per-tier contribution capture.

Question: is helix calculator-grade? Given identical query + identical genome
+ identical config, does it return bitwise-identical top-k gene IDs and
scores on back-to-back runs?

If yes: helix is already deterministic at the coarse level, and Stage 2
(Decimal vs float fusion) builds on a clean baseline.

If no: there's nondeterminism somewhere (numpy parallel reductions, dict
iteration order, concurrent futures) that would muddy any decimal signal.
We'd need to pin the cause before the decimal A/B is meaningful.

Also captures per-query per-gene tier contribution vectors — raw material
for later stages: Decimal fusion A/B (Pass 2), pairwise-agreement term
analysis (Pass 2.5), tier-distribution shape on successful retrievals
(Pass 3). Two experiments for the price of one run.

Target genome: genome-bench-2026-04-14.db (18,254 genes).
Query set:     benchmarks/needles_50_for_claude.json (first 25 with non-empty results).

Output: benchmarks/precision_probe_2026-04-15.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Force LLM-free config path for this probe so we're not timing Ollama warmup
# or hitting a backend we don't need. The /context retrieval loop doesn't
# touch the ribosome anyway when these are off.
os.environ["HELIX_DISABLE_HEADROOM"] = "1"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Windows cp1252 can't encode Δ/Σ/etc; force UTF-8 for console output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from helix_context import HelixContextManager, load_config  # noqa: E402


GENOME_DB = "genome-bench-2026-04-14.db"
NEEDLES_JSON = REPO / "benchmarks" / "needles_50_for_claude.json"
OUTPUT_JSON = REPO / "benchmarks" / "precision_probe_2026-04-15.json"
N_QUERIES = 25
TOP_K = 12


def load_needles(path: Path, n: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        needles = json.load(f)
    return needles[:n]


def snapshot_run(hcm: HelixContextManager, query: str) -> Dict:
    """Run a query and snapshot the raw retrieval state (scores + tier contribs).

    We bypass build_context()'s assembly/splice overhead and hit the genome
    directly via query_genes(), which is what all the +=/fusion code runs.
    This is the tightest possible measurement of the fusion arithmetic —
    no ribosome, no assembly, no session-delivery bookkeeping.
    """
    cfg = hcm.config
    # query_genes takes an expanded_query + extracted signals. Pull those
    # via the same path build_context() uses so we exercise the real flow.
    expanded = hcm._expand_query_intent(query)  # no-op when disabled
    domains, entities = hcm._extract_query_signals(expanded)

    # Call query_genes with the extracted signals. The genome's last_*
    # attributes get populated as a side effect.
    t0 = time.perf_counter()
    genes = hcm.genome.query_genes(
        domains=domains,
        entities=entities,
        max_genes=TOP_K,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    gene_ids = [g.gene_id for g in genes]
    scores = dict(hcm.genome.last_query_scores)  # snapshot
    tier_contrib = {
        gid: dict(contribs)
        for gid, contribs in hcm.genome.last_tier_contributions.items()
    }

    # Reduce scores to just the top-k we returned, but keep full tier_contrib
    # for those genes so Stage 2 can post-process fusion alternatives.
    top_k_scores = [
        {"gene_id": gid, "score": scores.get(gid, 0.0)}
        for gid in gene_ids
    ]
    top_k_contribs = {
        gid: tier_contrib.get(gid, {}) for gid in gene_ids
    }

    return {
        "gene_ids_ordered": list(gene_ids),
        "scores": top_k_scores,
        "tier_contributions": top_k_contribs,
        "elapsed_ms": elapsed_ms,
    }


def compare_runs(run_a: Dict, run_b: Dict) -> Dict:
    """Return a diff report between two runs of the same query."""
    ids_a = run_a["gene_ids_ordered"]
    ids_b = run_b["gene_ids_ordered"]

    set_a, set_b = set(ids_a), set(ids_b)
    set_identical = set_a == set_b
    order_identical = ids_a == ids_b

    # Score deltas — per gene_id in the intersection, compare scores
    score_a = {s["gene_id"]: s["score"] for s in run_a["scores"]}
    score_b = {s["gene_id"]: s["score"] for s in run_b["scores"]}
    score_deltas = []
    for gid in set_a & set_b:
        delta = score_b[gid] - score_a[gid]
        score_deltas.append({"gene_id": gid, "delta": delta, "abs_delta": abs(delta)})

    max_abs_delta = max((d["abs_delta"] for d in score_deltas), default=0.0)
    n_nonzero_deltas = sum(1 for d in score_deltas if d["abs_delta"] > 0.0)

    # Bitwise-identical check on scores for genes in the intersection
    bitwise_identical_scores = max_abs_delta == 0.0

    return {
        "set_identical": set_identical,
        "order_identical": order_identical,
        "bitwise_identical_scores": bitwise_identical_scores,
        "in_a_not_b": sorted(set_a - set_b),
        "in_b_not_a": sorted(set_b - set_a),
        "max_abs_score_delta": max_abs_delta,
        "n_genes_with_score_drift": n_nonzero_deltas,
        "elapsed_ms_a": run_a["elapsed_ms"],
        "elapsed_ms_b": run_b["elapsed_ms"],
    }


def main() -> int:
    print(f"[probe] repo: {REPO}")
    print(f"[probe] target genome: {GENOME_DB}")
    print(f"[probe] needles: {NEEDLES_JSON.name} (first {N_QUERIES})")
    print(f"[probe] top-k: {TOP_K}")
    print()

    # Load config, override genome path
    cfg = load_config()
    cfg.genome.path = GENOME_DB
    # Double-check LLM-free defaults are in effect for this probe
    cfg.ribosome.query_expansion_enabled = False
    cfg.ingestion.rerank_enabled = False

    print(f"[probe] initializing HelixContextManager...")
    t0 = time.perf_counter()
    hcm = HelixContextManager(cfg)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[probe] init: {init_ms:.0f} ms\n")

    needles = load_needles(NEEDLES_JSON, N_QUERIES)
    print(f"[probe] loaded {len(needles)} needles\n")

    results = []
    n_fully_identical = 0
    n_order_drift = 0
    n_set_drift = 0
    n_empty = 0

    for needle in needles:
        idx = needle["idx"]
        query = needle["query"]

        # Run twice, same config, same inputs
        run_a = snapshot_run(hcm, query)
        run_b = snapshot_run(hcm, query)

        diff = compare_runs(run_a, run_b)

        if not run_a["gene_ids_ordered"]:
            n_empty += 1
            status = "EMPTY"
        elif diff["order_identical"] and diff["bitwise_identical_scores"]:
            n_fully_identical += 1
            status = "IDENTICAL"
        elif diff["set_identical"]:
            n_order_drift += 1
            status = "ORDER_DRIFT"
        else:
            n_set_drift += 1
            status = "SET_DRIFT"

        print(
            f"  [{idx:2d}] {status:12s} "
            f"top={len(run_a['gene_ids_ordered']):2d} "
            f"a={run_a['elapsed_ms']:6.1f}ms "
            f"b={run_b['elapsed_ms']:6.1f}ms "
            f"max_d={diff['max_abs_score_delta']:.2e} "
            f"q='{query[:55]}'"
        )

        results.append({
            "idx": idx,
            "query": query,
            "category": needle.get("category"),
            "key": needle.get("key"),
            "value": needle.get("value"),
            "status": status,
            "run_a": run_a,
            "run_b": run_b,
            "diff": diff,
        })

    n_usable = N_QUERIES - n_empty
    summary = {
        "genome_db": GENOME_DB,
        "needles_file": NEEDLES_JSON.name,
        "n_queries": N_QUERIES,
        "top_k": TOP_K,
        "n_empty": n_empty,
        "n_usable": n_usable,
        "n_fully_identical": n_fully_identical,
        "n_order_drift": n_order_drift,
        "n_set_drift": n_set_drift,
        "determinism_verdict": (
            "CALCULATOR_GRADE" if n_fully_identical == n_usable and n_usable > 0
            else "DRIFT_DETECTED"
        ),
    }

    output = {"summary": summary, "results": results}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"[probe] summary")
    print(f"  queries run:           {N_QUERIES}")
    print(f"  empty (no results):    {n_empty}")
    print(f"  usable:                {n_usable}")
    print(f"  fully identical A==B:  {n_fully_identical}/{n_usable}")
    print(f"  set-equal, order drift: {n_order_drift}/{n_usable}")
    print(f"  set drift:             {n_set_drift}/{n_usable}")
    print()
    print(f"[probe] verdict: {summary['determinism_verdict']}")
    print(f"[probe] wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

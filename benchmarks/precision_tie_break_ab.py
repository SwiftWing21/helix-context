"""
Pass 3c — A/B test: baseline insertion-order tie-break vs walking tie-break.

Runs the same 25 queries twice against genome-bench-2026-04-14.db:
    Run A (baseline):  HELIX_WALKING_TIEBREAK unset — current behaviour
    Run B (walking):   HELIX_WALKING_TIEBREAK=1 — associative-graph ordering

Compares per-query top-k orderings. For any query whose top-k changed,
emits the specific rank-swaps plus a per-rule explanation of why walking
chose its ordering (from tie_break.explain_pair).

Metrics:
    - # of queries with any head-tie reordering (rank 0-3)
    - # of queries with any top-k reordering (all ranks)
    - Per-rank swap counts
    - Rule usage histogram (which ladder rule fired how often)

Output: benchmarks/precision_tie_break_ab_2026-04-15.json
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HELIX_DISABLE_HEADROOM"] = "1"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from helix_context import HelixContextManager, load_config  # noqa: E402
from helix_context import tie_break  # noqa: E402


GENOME_DB = "genome-bench-2026-04-14.db"
NEEDLES_JSON = REPO / "benchmarks" / "needles_50_for_claude.json"
OUTPUT_JSON = REPO / "benchmarks" / "precision_tie_break_ab_2026-04-15.json"
N_QUERIES = 25
TOP_K = 12


def load_needles(path: Path, n: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[:n]


def run_query(hcm: HelixContextManager, query: str) -> Tuple[List[str], Dict[str, float]]:
    expanded = hcm._expand_query_intent(query)
    domains, entities = hcm._extract_query_signals(expanded)
    genes = hcm.genome.query_genes(domains=domains, entities=entities, max_genes=TOP_K)
    return [g.gene_id for g in genes], dict(hcm.genome.last_query_scores)


def find_swaps(ids_a: List[str], ids_b: List[str]) -> List[Dict]:
    """Find positions where A and B disagree."""
    swaps = []
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            swaps.append({
                "rank": i,
                "baseline_id": ids_a[i],
                "walking_id": ids_b[i],
            })
    return swaps


def main() -> int:
    print(f"[ab] target genome: {GENOME_DB}")
    print(f"[ab] running {N_QUERIES} queries in two modes\n")

    cfg = load_config()
    cfg.genome.path = GENOME_DB
    cfg.ribosome.query_expansion_enabled = False
    cfg.ingestion.rerank_enabled = False

    print(f"[ab] initializing HelixContextManager...")
    t0 = time.perf_counter()
    hcm = HelixContextManager(cfg)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[ab] init: {init_ms:.0f} ms\n")

    needles = load_needles(NEEDLES_JSON, N_QUERIES)

    # Shared SQLite connection for tie_break.explain_pair calls
    explain_conn = sqlite3.connect(str(REPO / GENOME_DB))

    results = []
    rule_usage = {
        "strong_edge_freshness": 0,
        "neighborhood_size": 0,
        "nli_entailment": 0,
        "freshness_fallback": 0,
        "lexical_gene_id": 0,
    }
    n_head_changed = 0
    n_any_changed = 0

    for needle in needles:
        query = needle["query"]
        idx = needle["idx"]

        # Run A — baseline
        os.environ.pop("HELIX_WALKING_TIEBREAK", None)
        ids_a, scores_a = run_query(hcm, query)

        # Run B — walking
        os.environ["HELIX_WALKING_TIEBREAK"] = "1"
        ids_b, scores_b = run_query(hcm, query)

        # Clean up env so subsequent work isn't affected
        os.environ.pop("HELIX_WALKING_TIEBREAK", None)

        # Scores should be identical — tie_break only reorders within ties,
        # doesn't change the values. Verify.
        score_mismatch = False
        for gid in set(ids_a) & set(ids_b):
            if scores_a.get(gid) != scores_b.get(gid):
                score_mismatch = True
                break

        swaps = find_swaps(ids_a, ids_b)
        head_swaps = [s for s in swaps if s["rank"] < 4]

        if swaps:
            n_any_changed += 1
        if head_swaps:
            n_head_changed += 1

        # For each swap, get rule-by-rule explanation for why walking chose
        # its gene. We explain the pair (baseline_id, walking_id) — both
        # were candidates at this rank and walking preferred walking_id.
        swap_explanations = []
        for s in swaps:
            trace = tie_break.explain_pair(explain_conn, s["baseline_id"], s["walking_id"])
            # Determine which rule decided — first non-abstain wins.
            decisive_rule = None
            for rule_name in ["strong_edge_freshness", "neighborhood_size",
                              "nli_entailment", "freshness_fallback", "lexical_gene_id"]:
                v = trace.get(rule_name)
                if v != "abstain":
                    decisive_rule = rule_name
                    break
            if decisive_rule:
                rule_usage[decisive_rule] += 1
            swap_explanations.append({
                "rank": s["rank"],
                "baseline_id": s["baseline_id"],
                "walking_id": s["walking_id"],
                "decisive_rule": decisive_rule,
                "trace": trace,
            })

        # Status line
        if not swaps:
            status = "UNCHANGED"
        elif head_swaps:
            status = f"HEAD_CHANGED ({len(head_swaps)}h/{len(swaps)}t)"
        else:
            status = f"TAIL_CHANGED (0h/{len(swaps)}t)"

        if score_mismatch:
            status += " SCORE_MISMATCH"

        print(
            f"  [{idx:2d}] {status:25s} "
            f"top={len(ids_a):2d} "
            f"q='{query[:55]}'"
        )

        results.append({
            "idx": idx,
            "query": query,
            "status": status,
            "baseline_ids": ids_a,
            "walking_ids": ids_b,
            "swaps": swaps,
            "head_swaps": head_swaps,
            "score_mismatch": score_mismatch,
            "swap_explanations": swap_explanations,
        })

    summary = {
        "genome_db": GENOME_DB,
        "n_queries": N_QUERIES,
        "top_k": TOP_K,
        "n_queries_any_reorder": n_any_changed,
        "n_queries_head_reorder": n_head_changed,
        "rule_usage": rule_usage,
    }

    total_swaps = sum(rule_usage.values())
    print()
    print(f"[ab] summary")
    print(f"  queries with ANY reorder:    {n_any_changed}/{N_QUERIES}")
    print(f"  queries with HEAD reorder:   {n_head_changed}/{N_QUERIES}")
    print(f"  total rank positions changed: {total_swaps}")
    print()
    print(f"[ab] rule usage (decisive rule per swap):")
    for rule, n in sorted(rule_usage.items(), key=lambda x: -x[1]):
        pct = 100.0 * n / total_swaps if total_swaps else 0.0
        print(f"  {rule:25s} {n:4d}  ({pct:5.1f}%)")

    output = {"summary": summary, "results": results}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"[ab] wrote {OUTPUT_JSON}")
    explain_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

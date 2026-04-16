"""
Precision sensitivity analysis — post-processing pass over precision_probe data.

Question: on 25 real queries against the benchmark genome, how often are
adjacent top-k scores close enough that float imprecision could plausibly
flip their order?

Method:
  - Load the per-query top-k score lists captured by precision_probe.py
  - For each adjacent pair (rank i, rank i+1), compute the score gap
  - Classify each gap against precision-sensitivity thresholds:
      * float32 relative epsilon  ~1.19e-7   (SEMA uses np.float32)
      * float64 relative epsilon  ~2.22e-16  (Python native float)
  - A gap is "precision-sensitive" at threshold T when gap/max(|a|,|b|) < T
  - Report: per-query flag, aggregate counts, full distribution

Decision rule for "should we build the Decimal path":
  - If >10%% of queries have any adjacent pair within float32 epsilon,
    the sidebar is earning its keep and the Decimal path is worth building.
  - If <5%% do, the precision question is empirically a non-issue on
    real workloads — ship the sensitivity flag only, shelve Decimal.
  - In between: judgment call, but lean toward the flag-only solution
    since it's cheaper and can be upgraded later.

Input:  benchmarks/precision_probe_2026-04-15.json
Output: benchmarks/precision_sensitivity_2026-04-15.json
        (plus human-readable summary to stdout)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
INPUT_JSON = REPO / "benchmarks" / "precision_probe_2026-04-15.json"
OUTPUT_JSON = REPO / "benchmarks" / "precision_sensitivity_2026-04-15.json"

# Windows cp1252 can't encode >=, pct, etc; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# IEEE-754 machine epsilons (relative precision floor for adjacent-value math)
EPS_FLOAT32 = 1.19e-7
EPS_FLOAT64 = 2.22e-16

# Sensitivity thresholds, expressed as relative gap: gap / max(|a|, |b|).
# Gap == 0 (exact tie) is treated separately — it's not a float-precision
# concern, it's a "multiple genes with the same score" phenomenon that
# Decimal wouldn't change. Ties are resolved by dict/sort stability.
THRESHOLDS = {
    "catastrophic": 1e-15,     # below float64 epsilon — ranking is luck
    "float32_risky": 1e-6,     # within ~10x float32 epsilon — SEMA could flip
    "close_but_safe": 1e-3,    # visibly close but well above any epsilon
    "comfortable": float("inf"),
}


def classify_gap(abs_gap: float, rel_gap: float, score_a: float, score_b: float) -> str:
    # Both scores zero → padding (helix returned more slots than it had real
    # hits for). Not a ranking decision worth analyzing.
    if score_a == 0.0 and score_b == 0.0:
        return "padding"
    # Exact tie between two non-zero scores → ordering decided by insertion
    # order / sort stability, not arithmetic. Decimal changes nothing here.
    if abs_gap == 0.0:
        return "exact_tie"
    for name, threshold in THRESHOLDS.items():
        if rel_gap < threshold:
            return name
    return "comfortable"


def analyze_query(result: Dict) -> Dict:
    """Classify every adjacent score pair in this query's top-k."""
    scores = result["run_a"]["scores"]  # list of {gene_id, score}
    pairs: List[Dict] = []
    for i in range(len(scores) - 1):
        a = scores[i]["score"]
        b = scores[i + 1]["score"]
        abs_gap = abs(a - b)
        max_mag = max(abs(a), abs(b))
        rel_gap = abs_gap / max_mag if max_mag > 0 else 0.0
        pairs.append({
            "rank_a": i,
            "rank_b": i + 1,
            "score_a": a,
            "score_b": b,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            "classification": classify_gap(abs_gap, rel_gap, a, b),
        })

    # Any-pair classification: the riskiest adjacent pair wins (padding
    # excluded — padding is not a ranking decision).
    priority = ["catastrophic", "float32_risky", "close_but_safe", "exact_tie", "comfortable"]
    worst = "comfortable"
    for p in pairs:
        cls = p["classification"]
        if cls == "padding":
            continue
        if priority.index(cls) < priority.index(worst):
            worst = cls

    return {
        "idx": result["idx"],
        "query": result["query"],
        "n_scores": len(scores),
        "n_pairs": len(pairs),
        "worst_adjacent_class": worst,
        "min_rel_gap": min((p["rel_gap"] for p in pairs), default=0.0),
        "min_abs_gap": min((p["abs_gap"] for p in pairs), default=0.0),
        "pairs": pairs,
    }


def main() -> int:
    print(f"[sensitivity] loading {INPUT_JSON.name}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    print(f"[sensitivity] analyzing {len(results)} queries")
    print(f"[sensitivity] thresholds (relative gap):")
    for name, t in THRESHOLDS.items():
        print(f"              {name:18s} < {t}")
    print()

    analyzed = [analyze_query(r) for r in results]

    # Aggregate
    all_classes = ["catastrophic", "float32_risky", "close_but_safe", "exact_tie", "comfortable", "padding"]
    class_counts = {c: 0 for c in all_classes}
    for a in analyzed:
        class_counts[a["worst_adjacent_class"]] += 1

    n_total = len(analyzed)
    n_risky = class_counts["catastrophic"] + class_counts["float32_risky"]
    pct_risky_or_worse = 100.0 * n_risky / n_total if n_total else 0.0

    # Full distribution of ALL adjacent pairs (not just worst-per-query).
    # Also compute stats excluding padding (which is not a ranking decision).
    all_pair_classes = {c: 0 for c in all_classes}
    all_rel_gaps_real: List[float] = []  # excludes padding
    for a in analyzed:
        for p in a["pairs"]:
            all_pair_classes[p["classification"]] += 1
            if p["classification"] != "padding":
                all_rel_gaps_real.append(p["rel_gap"])
    n_pairs_total = sum(all_pair_classes.values())
    n_pairs_real = n_pairs_total - all_pair_classes["padding"]

    # Percentiles on relative gaps of REAL (non-padding) adjacent pairs
    srt = sorted(all_rel_gaps_real)
    def pct(p):
        if not srt:
            return 0.0
        k = max(0, min(len(srt) - 1, int(p * (len(srt) - 1))))
        return srt[k]

    distribution = {
        "p01_rel_gap": pct(0.01),
        "p05_rel_gap": pct(0.05),
        "p25_rel_gap": pct(0.25),
        "p50_rel_gap": pct(0.50),
        "p75_rel_gap": pct(0.75),
        "p95_rel_gap": pct(0.95),
        "p99_rel_gap": pct(0.99),
    }

    # Decision — based on queries where precision could actually matter
    if pct_risky_or_worse >= 10.0:
        decision = "BUILD_DECIMAL_PATH - 10pct+ of queries have precision-risky top-k boundaries"
    elif pct_risky_or_worse >= 5.0:
        decision = "JUDGMENT_CALL - 5-10pct risky; flag-only is probably enough, Decimal is optional polish"
    else:
        decision = "FLAG_ONLY - <5pct risky on real workload; Decimal path is a solution to a non-problem"

    # Per-query print — annotate the worst-adjacent class (excluding padding)
    marker_map = {
        "catastrophic": "!!!",
        "float32_risky": " !!",
        "close_but_safe": "  .",
        "exact_tie": " ~=",
        "comfortable": "   ",
    }
    for a in analyzed:
        marker = marker_map.get(a["worst_adjacent_class"], "   ")
        print(
            f"  {marker} [{a['idx']:2d}] "
            f"{a['worst_adjacent_class']:15s} "
            f"min_rel_gap={a['min_rel_gap']:.2e} "
            f"min_abs_gap={a['min_abs_gap']:.4f} "
            f"q='{a['query'][:50]}'"
        )

    print()
    print(f"[sensitivity] per-query worst-adjacent-pair class counts (N={n_total}):")
    for cls in ["catastrophic", "float32_risky", "close_but_safe", "exact_tie", "comfortable"]:
        n = class_counts[cls]
        print(f"  {cls:18s} {n:3d}  ({100.0 * n / n_total:5.1f}%)")

    print()
    print(f"[sensitivity] adjacent-pair class counts (all pairs N={n_pairs_total}, real N={n_pairs_real}):")
    for cls in ["catastrophic", "float32_risky", "close_but_safe", "exact_tie", "comfortable", "padding"]:
        n = all_pair_classes[cls]
        pct_all = 100.0 * n / n_pairs_total if n_pairs_total else 0.0
        pct_real = 100.0 * n / n_pairs_real if n_pairs_real and cls != "padding" else 0.0
        if cls == "padding":
            print(f"  {cls:18s} {n:4d}  ({pct_all:5.1f}% of all)")
        else:
            print(f"  {cls:18s} {n:4d}  ({pct_all:5.1f}% of all, {pct_real:5.1f}% of real)")

    print()
    print(f"[sensitivity] relative-gap distribution (all adjacent pairs):")
    for k, v in distribution.items():
        print(f"  {k:14s} {v:.2e}")

    print()
    print(f"[sensitivity] pct_risky_or_worse: {pct_risky_or_worse:.1f}%")
    print(f"[sensitivity] decision: {decision}")

    output = {
        "input": INPUT_JSON.name,
        "thresholds": THRESHOLDS,
        "per_query_worst_class_counts": class_counts,
        "all_pair_class_counts": all_pair_classes,
        "distribution": distribution,
        "pct_risky_or_worse": pct_risky_or_worse,
        "decision": decision,
        "queries": analyzed,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[sensitivity] wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

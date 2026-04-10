"""
A/B comparison helper for Headroom integration benchmarking.

Reads two bench_needle_1000.py result JSONs (baseline = truncation,
treatment = Kompress) and prints a delta report with gate evaluation.

Usage:
    python benchmarks/compare_ab.py BASELINE_JSON TREATMENT_JSON

Gate (from the integration plan):
    - denatured rate drops >= 15 percentage points, OR
    - aligned count at least doubles, OR
    - answer accuracy rate improves >= 10 percentage points
"""

import json
import sys
from pathlib import Path


def load_summary(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    summary = data.get("summary", {})
    return {
        "path": path,
        "n": summary.get("n", 0),
        "retrieval_rate": summary.get("retrieval_rate", 0),
        "answer_rate": summary.get("answer_accuracy_rate", 0),
        "retrieved": summary.get("retrieved", 0),
        "answered": summary.get("answered", 0),
        "errors": summary.get("errors", 0),
        "by_category": summary.get("by_category", {}),
        "latency_p50": summary.get("latency", {}).get("proxy_p50_s", 0),
        "latency_p95": summary.get("latency", {}).get("proxy_p95_s", 0),
        "context_p95": summary.get("latency", {}).get("context_p95_s", 0),
        "total_time_min": summary.get("total_time_min", 0),
        "genome_genes": data.get("genome_genes", 0),
    }


def format_pp(baseline_val: float, treatment_val: float) -> str:
    """Format a percentage-point delta with sign and color hints."""
    delta = (treatment_val - baseline_val) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}pp"


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_ab.py BASELINE_JSON TREATMENT_JSON")
        sys.exit(2)

    baseline = load_summary(sys.argv[1])
    treatment = load_summary(sys.argv[2])

    print("=" * 70)
    print("Headroom A/B Benchmark Comparison")
    print("=" * 70)
    print(f"  baseline  : {baseline['path']}")
    print(f"  treatment : {treatment['path']}")
    print(f"  n         : {baseline['n']} / {treatment['n']}")
    print(f"  genes     : {baseline['genome_genes']} / {treatment['genome_genes']}")
    print()

    print("-- Top-line --")
    print(
        f"  retrieval : {baseline['retrieval_rate']:>6.1%} -> {treatment['retrieval_rate']:>6.1%}  "
        f"({format_pp(baseline['retrieval_rate'], treatment['retrieval_rate'])})"
    )
    print(
        f"  answers   : {baseline['answer_rate']:>6.1%} -> {treatment['answer_rate']:>6.1%}  "
        f"({format_pp(baseline['answer_rate'], treatment['answer_rate'])})"
    )
    print(f"  errors    : {baseline['errors']} -> {treatment['errors']}")
    print()

    print("-- Latency --")
    print(
        f"  context p95 : {baseline['context_p95']:>6.2f}s -> {treatment['context_p95']:>6.2f}s  "
        f"({treatment['context_p95'] - baseline['context_p95']:+.2f}s)"
    )
    print(
        f"  proxy   p50 : {baseline['latency_p50']:>6.2f}s -> {treatment['latency_p50']:>6.2f}s  "
        f"({treatment['latency_p50'] - baseline['latency_p50']:+.2f}s)"
    )
    print(
        f"  proxy   p95 : {baseline['latency_p95']:>6.2f}s -> {treatment['latency_p95']:>6.2f}s  "
        f"({treatment['latency_p95'] - baseline['latency_p95']:+.2f}s)"
    )
    print(f"  total min   : {baseline['total_time_min']:.1f} -> {treatment['total_time_min']:.1f}")
    print()

    print("-- By category --")
    all_cats = sorted(set(baseline["by_category"].keys()) | set(treatment["by_category"].keys()))
    print(f"  {'category':<20}  {'n':>3}  {'retr base':>9}  {'retr treat':>10}  {'ans base':>8}  {'ans treat':>9}")
    for cat in all_cats:
        b = baseline["by_category"].get(cat, {})
        t = treatment["by_category"].get(cat, {})
        n = max(b.get("n", 0), t.get("n", 0))
        print(
            f"  {cat:<20}  {n:>3}  "
            f"{b.get('retrieval_rate', 0):>8.1%}  "
            f"{t.get('retrieval_rate', 0):>9.1%}  "
            f"{b.get('answer_rate', 0):>7.1%}  "
            f"{t.get('answer_rate', 0):>8.1%}"
        )
    print()

    # Gate evaluation (from integration plan)
    print("-- Gate evaluation --")

    retr_delta_pp = (treatment["retrieval_rate"] - baseline["retrieval_rate"]) * 100
    ans_delta_pp = (treatment["answer_rate"] - baseline["answer_rate"]) * 100
    ans_base = baseline["answered"]
    ans_treat = treatment["answered"]

    gate_answer_pp = ans_delta_pp >= 10
    gate_answer_double = ans_base > 0 and ans_treat >= ans_base * 2
    gate_latency_ok = treatment["latency_p95"] <= baseline["latency_p95"] * 1.5

    print(f"  answer rate delta : {ans_delta_pp:+.1f}pp  (gate: >=+10pp -> {'PASS' if gate_answer_pp else 'FAIL'})")
    print(f"  answer 2x guard   : {ans_base} -> {ans_treat}  (gate: 2x -> {'PASS' if gate_answer_double else 'FAIL'})")
    print(f"  retrieval delta   : {retr_delta_pp:+.1f}pp  (info only — retrieval untouched)")
    print(f"  latency p95 guard : {treatment['latency_p95']:.1f}s <= {baseline['latency_p95'] * 1.5:.1f}s  ({'PASS' if gate_latency_ok else 'FAIL'})")
    print()

    passed_quality = gate_answer_pp or gate_answer_double
    passed_latency = gate_latency_ok

    if passed_quality and passed_latency:
        print("  OVERALL: PASS — ship v0.3.0b5")
        sys.exit(0)
    elif passed_quality and not passed_latency:
        print("  OVERALL: QUALITY PASS / LATENCY REGRESSION — needs investigation before ship")
        sys.exit(1)
    elif not passed_quality and passed_latency:
        print("  OVERALL: NO QUALITY GAIN — hold push, consider option B (re-ingest with density gate)")
        sys.exit(2)
    else:
        print("  OVERALL: BOTH REGRESSED — revert Phase 2 and escalate")
        sys.exit(3)


if __name__ == "__main__":
    main()

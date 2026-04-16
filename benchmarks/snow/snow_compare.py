#!/usr/bin/env python3
"""SNOW benchmark multi-model comparison table.

Reads all snow_*.json result files from benchmarks/snow/results/
and prints a formatted comparison table to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows Unicode safety
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).parent / "results"
TIERS = ["T0", "T1", "T2", "T3", "T4"]


def load_results() -> list[dict]:
    """Load all snow_*.json files from results dir."""
    results = []
    if not RESULTS_DIR.exists():
        return results
    for p in sorted(RESULTS_DIR.glob("snow_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_file"] = p.name
            results.append(data)
        except Exception as e:
            print(f"  [warn] skipping {p.name}: {e}", file=sys.stderr)
    return results


def pct_str(count: int, total: int) -> str:
    """Format count as percentage of total."""
    if total == 0:
        return "0%"
    return f"{count / total * 100:.0f}%"


def main() -> None:
    results = load_results()
    if not results:
        print("No SNOW result files found in", RESULTS_DIR)
        sys.exit(1)

    # --- Oracle floor (deterministic, take from first file) ---
    oracle = None
    genome_label = "genome"
    for r in results:
        if r.get("oracle"):
            oracle = r["oracle"]
            meta = r.get("meta", {})
            gene_count = meta.get("gene_count", "?")
            genome_label = f"{meta.get('genome', 'genome')} ({gene_count} genes)"
            break

    n_queries = results[0].get("meta", {}).get("n_queries", "?")

    print(f"SNOW Benchmark \u2014 {genome_label}")
    print("\u2550" * 60)
    print()

    if oracle:
        orc_total = sum(oracle.get("cascade_profile", {}).get(t, 0) for t in TIERS)
        profile_parts = []
        for t in TIERS:
            c = oracle.get("cascade_profile", {}).get(t, 0)
            profile_parts.append(f"{t}:{pct_str(c, orc_total)}")
        profile_str = "  ".join(profile_parts)

        print("Oracle floor:")
        print(f"  Tier (avg): {oracle.get('avg_tier', '?'):.1f}"
              f"    Tokens: {oracle.get('avg_tokens', '?'):.0f}"
              f"    Latency: {oracle.get('avg_latency_s', '?'):.4f}s"
              f"    Miss rate: {oracle.get('miss_rate', 0):.0%}")
        print(f"  Profile: {profile_str}")
        print()

    # --- Collect LLM results ---
    llm_rows = []
    for r in results:
        llm = r.get("llm")
        if not llm:
            continue
        model = r.get("meta", {}).get("model", "unknown")
        llm_rows.append({
            "model": model,
            "hops": llm.get("avg_hops", 0),
            "tokens": llm.get("avg_tokens", 0),
            "latency": llm.get("avg_latency_s", 0),
            "triage": llm.get("triage_accuracy", 0),
            "miss_rate": llm.get("miss_rate", 0),
            "waste": llm.get("hop_waste", 0),
            "token_oh": llm.get("token_overhead_x", 0),
            "latency_oh": llm.get("latency_overhead_x", 0),
            "per_step": llm.get("per_step_latency", {}),
            "file": r.get("_file", ""),
        })

    # Sort by avg hops ascending
    llm_rows.sort(key=lambda x: x["hops"])

    if llm_rows:
        # --- Model comparison table ---
        # Find max model name length for alignment
        name_w = max(len(r["model"]) for r in llm_rows)
        name_w = max(name_w, 5)  # minimum "Model"

        print(f"Model comparison (sorted by avg hops ascending, n={n_queries}):")
        header = (f"  {'Model':<{name_w}}   Hops   Tokens   Latency  Triage%"
                  f"  Miss%  Waste  Tok OH  Lat OH")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for r in llm_rows:
            print(f"  {r['model']:<{name_w}}"
                  f"   {r['hops']:4.1f}"
                  f"   {r['tokens']:7.0f}"
                  f"   {r['latency']:6.2f}s"
                  f"   {r['triage']:5.0%}"
                  f"   {r['miss_rate']:4.0%}"
                  f"   {r['waste']:+4.1f}"
                  f"   {r['token_oh']:5.1f}x"
                  f"   {r['latency_oh']:5.1f}x")
        print()

        # --- Per-step latency table ---
        # Collect all tier labels that appear
        all_tiers = set()
        for r in llm_rows:
            all_tiers.update(r["per_step"].keys())
        tier_labels = [t for t in TIERS if t in all_tiers]

        if tier_labels:
            print("Per-step latency (avg):")
            tier_header = "  " + f"{'Model':<{name_w}}"
            for t in tier_labels:
                tier_header += f"   {t:>8}"
            print(tier_header)
            print("  " + "-" * (len(tier_header) - 2))

            for r in llm_rows:
                line = f"  {r['model']:<{name_w}}"
                for t in tier_labels:
                    step = r["per_step"].get(t, {})
                    if isinstance(step, dict):
                        avg = step.get("avg", step.get("mean", 0))
                    else:
                        avg = step
                    line += f"   {avg:7.3f}s"
                print(line)
            print()
    else:
        print("No LLM results found (oracle-only runs).")
        print("Run bench_snow.py with a model to generate comparison data.")
        print()

    # --- File inventory ---
    print(f"Result files ({len(results)}):")
    for r in results:
        model = r.get("meta", {}).get("model", "?")
        ts = r.get("meta", {}).get("timestamp", "?")[:10]
        has_llm = "LLM" if r.get("llm") else "oracle-only"
        print(f"  {r.get('_file', '?'):<45}  {model:<20}  {has_llm}  {ts}")


if __name__ == "__main__":
    main()

"""Headroom E2E latency — measure the compress_text() seam with headroom
enabled vs disabled (HELIX_DISABLE_HEADROOM=1 fallback).

Samples real gene content from the running genome across content types
(code / config / doc / benchmark / log-ish), runs N trials per sample
in each mode, and reports:
    - mean / p50 / p95 latency per mode
    - compression ratio per call (output_chars / input_chars)
    - per-content-type breakdown

Runs in-process via the library — no HTTP, no separate server. The
headroom_bridge ``_disabled()`` check reads env on every call, so we
can toggle between trials without restarting anything.

Usage:
    python benchmarks/bench_headroom_latency.py [--trials=20]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helix_context.config import load_config  # noqa: E402
from helix_context.genome import Genome        # noqa: E402
from helix_context.headroom_bridge import compress_text  # noqa: E402


# Target budgets mimic the hot path in context_manager when genes are
# expressed. Values span the three regimes the expression pipeline uses.
BUDGETS = [200, 500, 1000]


def _pick_sample(genes, content_kind: str, n: int = 5) -> list:
    """Sample `n` genes whose source_id hints at the given content_kind."""
    pool = [
        g for g in genes
        if g.content and len(g.content) > 400 and (
            (content_kind == "code" and (g.source_id or "").endswith(
                (".py", ".rs", ".ts", ".js", ".go")
            ))
            or (content_kind == "config" and (g.source_id or "").endswith(
                (".toml", ".yaml", ".yml", ".ini", ".json")
            ))
            or (content_kind == "doc" and (g.source_id or "").endswith(
                (".md", ".rst", ".txt")
            ))
            or (content_kind == "mixed" and True)
        )
    ]
    if not pool:
        return []
    random.seed(42)
    return random.sample(pool, min(n, len(pool)))


def _time_compress(text: str, budget: int) -> tuple[float, int]:
    """Run compress_text once, return (elapsed_s, output_chars)."""
    t0 = time.perf_counter()
    out = compress_text(text, target_chars=budget)
    elapsed = time.perf_counter() - t0
    return elapsed, len(out or "")


def _run_one_mode(
    samples: list,
    budget: int,
    trials: int,
) -> dict:
    """Run all samples × trials in the *current* env mode."""
    latencies: list[float] = []
    ratios: list[float] = []
    out_chars_total = 0
    in_chars_total = 0

    # Warmup — first call loads the specialist model; not part of measurement.
    if samples:
        _time_compress(samples[0].content[:2000], budget)

    for sample in samples:
        text = sample.content[:4000]  # cap input to a realistic gene-body size
        for _ in range(trials):
            elapsed_s, n_out = _time_compress(text, budget)
            latencies.append(elapsed_s)
            if len(text) > 0:
                ratios.append(n_out / len(text))
            out_chars_total += n_out
            in_chars_total += len(text)

    if not latencies:
        return {"n": 0}

    latencies_ms = [x * 1000 for x in latencies]
    return {
        "n": len(latencies),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "median_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(statistics.quantiles(latencies_ms, n=20)[-1], 2)
        if len(latencies_ms) >= 20 else round(max(latencies_ms), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
        "mean_ratio": round(statistics.mean(ratios), 3) if ratios else 0.0,
        "total_in_chars": in_chars_total,
        "total_out_chars": out_chars_total,
        "aggregate_compression": round(out_chars_total / max(in_chars_total, 1), 3),
    }


def _format_table(rows: list[dict]) -> str:
    headers = ["kind", "budget", "n", "mode", "mean_ms", "p95_ms",
               "mean_ratio", "compression"]
    lines = [
        f"{headers[0]:<8} {headers[1]:>6} {headers[2]:>4} "
        f"{headers[3]:<12} {headers[4]:>8} {headers[5]:>7} "
        f"{headers[6]:>10} {headers[7]:>11}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"{r['kind']:<8} {r['budget']:>6} {r['n']:>4} "
            f"{r['mode']:<12} {r['mean_ms']:>8.2f} {r['p95_ms']:>7.2f} "
            f"{r['mean_ratio']:>10.3f} {r['aggregate_compression']:>11.3f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per (sample × budget × mode)")
    parser.add_argument("--samples-per-kind", type=int, default=3)
    parser.add_argument("--out", type=str,
                        default=f"benchmarks/results/headroom_latency_"
                        f"{time.strftime('%Y-%m-%d')}.json")
    args = parser.parse_args(argv)

    print("Loading genome…", flush=True)
    cfg = load_config()
    genome = Genome(cfg.genome.path)
    try:
        all_genes = list(genome.iter_genes()) if hasattr(genome, "iter_genes") \
            else [genome._row_to_gene(r) for r in genome.conn.execute(
                "SELECT * FROM genes LIMIT 2000"
            )]
    finally:
        genome.close()

    print(f"Sampled from {len(all_genes)} genes", flush=True)

    kinds_samples = {
        kind: _pick_sample(all_genes, kind, args.samples_per_kind)
        for kind in ("code", "config", "doc", "mixed")
    }
    for k, s in kinds_samples.items():
        print(f"  {k}: {len(s)} samples")

    results: List[dict] = []
    rows_table: List[dict] = []

    print("\nRunning benchmark (this will take a minute)…\n", flush=True)

    for budget in BUDGETS:
        for kind, samples in kinds_samples.items():
            if not samples:
                continue
            # A: headroom ON
            os.environ.pop("HELIX_DISABLE_HEADROOM", None)
            on_stats = _run_one_mode(samples, budget, args.trials)
            on_row = {"kind": kind, "budget": budget, "mode": "headroom_on",
                      **on_stats}

            # B: headroom OFF (character-level fallback)
            os.environ["HELIX_DISABLE_HEADROOM"] = "1"
            off_stats = _run_one_mode(samples, budget, args.trials)
            off_row = {"kind": kind, "budget": budget, "mode": "headroom_off",
                       **off_stats}

            results.extend([on_row, off_row])
            rows_table.extend([on_row, off_row])

    os.environ.pop("HELIX_DISABLE_HEADROOM", None)

    print(_format_table(rows_table))

    # Deltas per (kind, budget)
    print("\nDeltas (headroom_on - headroom_off):")
    print(f"{'kind':<8} {'budget':>6} {'d_ mean_ms':>12} "
          f"{'d_ ratio':>10} {'d_ out_chars':>14}")
    print("-" * 60)
    for budget in BUDGETS:
        for kind in kinds_samples:
            on = next((r for r in results
                       if r["kind"] == kind and r["budget"] == budget
                       and r["mode"] == "headroom_on"), None)
            off = next((r for r in results
                        if r["kind"] == kind and r["budget"] == budget
                        and r["mode"] == "headroom_off"), None)
            if not on or not off or on.get("n", 0) == 0:
                continue
            d_ms = on["mean_ms"] - off["mean_ms"]
            d_ratio = on["mean_ratio"] - off["mean_ratio"]
            d_out = on["total_out_chars"] - off["total_out_chars"]
            print(f"{kind:<8} {budget:>6} {d_ms:>+12.2f} "
                  f"{d_ratio:>+10.3f} {d_out:>+14d}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {
            "trials": args.trials,
            "samples_per_kind": args.samples_per_kind,
            "budgets": BUDGETS,
        },
        "results": results,
    }, indent=2))
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

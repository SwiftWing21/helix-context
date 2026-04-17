"""SNOW oracle-only ablation sweep across helix.toml toggles.

Runs N=65 SNOW oracle-only against the same genome under several
config variants, writes per-run results to benchmarks/snow/results/,
and prints a comparison table vs the locked baseline.

Waude diagnostic 2026-04-17 scope — measures Genome-layer
retrieval-dimension contribution only. D4 (access rate) has no
toggle, D7 (party bonus) is not yet code, D9 (TCM tiebreaker) lives
in context_manager._express which SNOW oracle bypasses — those three
dimensions are deliberately not in this sweep.

Usage:
    python scripts/snow_ablation_sweep.py \\
        --genome F:/Projects/helix-context/genomes/main/genome.db
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# tomli-w is available in the env; fallback to manual serialisation
# would be ugly and error-prone for nested dicts.
try:
    import tomli_w
except ImportError:  # pragma: no cover
    tomli_w = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "helix.toml"
DEFAULT_GENOME = "F:/Projects/helix-context/genomes/main/genome.db"
RESULTS_DIR = REPO / "benchmarks" / "snow" / "results"


# Config variants — each one is (label, nested helix.toml patch dict).
# The patch is deep-merged onto the base helix.toml.
VARIANTS: list[tuple[str, dict]] = [
    (
        "baseline",
        {},  # No patch — baseline is current helix.toml
    ),
    (
        "d6_cymatics_off",
        {"cymatics": {"enabled": False}},
    ),
    (
        "d8_harmonic_off",
        {"cymatics": {"harmonic_links": False}},
    ),
    (
        "d5_cold_tier_on",
        {"context": {"cold_tier_enabled": True}},
    ),
]


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def write_variant_config(base: dict, patch: dict, dest: Path) -> None:
    merged = deep_merge(base, patch)
    if tomli_w is None:
        raise RuntimeError(
            "tomli_w not installed — pip install tomli-w (needed to "
            "write the ablation variant config)"
        )
    with open(dest, "wb") as f:
        tomli_w.dump(merged, f)


def run_bench(config_path: Path, genome: str) -> Path:
    env = os.environ.copy()
    env["HELIX_CONFIG"] = str(config_path)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable,
        str(REPO / "benchmarks" / "snow" / "bench_snow.py"),
        "--model", "oracle-only",
        "--genome", genome,
    ]
    proc = subprocess.run(
        cmd, env=env, cwd=str(REPO), capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:] if proc.stdout else "")
        print(proc.stderr[-2000:] if proc.stderr else "")
        raise RuntimeError(f"bench_snow failed with rc={proc.returncode}")
    # SNOW writes to results/snow_oracle-only_<date>.json — find the
    # most recent file matching that pattern.
    latest = max(
        RESULTS_DIR.glob("snow_oracle-only_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return latest


def tag_result(src: Path, label: str) -> Path:
    dest = RESULTS_DIR / f"ablation_{label}_{time.strftime('%Y-%m-%d')}.json"
    dest.write_bytes(src.read_bytes())
    return dest


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    oracle = data.get("oracle", {})
    return {
        "avg_tier": oracle.get("avg_tier"),
        "avg_tokens": oracle.get("avg_tokens"),
        "miss_rate": oracle.get("miss_rate"),
        "cascade": oracle.get("cascade_profile", {}),
    }


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "min": None, "max": None, "stddev": None}
    mean = sum(xs) / len(xs)
    variance = sum((x - mean) ** 2 for x in xs) / len(xs) if len(xs) > 1 else 0.0
    stddev = variance ** 0.5
    return {
        "mean": mean, "min": min(xs), "max": max(xs), "stddev": stddev,
        "values": list(xs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default=DEFAULT_GENOME)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--repeats", type=int, default=1,
                    help="Run each variant N times to estimate variance")
    args = ap.parse_args()

    base_path = Path(args.config)
    with open(base_path, "rb") as f:
        base_config = tomllib.load(f)

    print(f"[ablation] genome: {args.genome}")
    print(f"[ablation] base config: {base_path}")
    print(f"[ablation] variants: {len(VARIANTS)}, repeats per variant: {args.repeats}")
    print()

    results: dict[str, list[dict]] = {label: [] for label, _ in VARIANTS}
    with tempfile.TemporaryDirectory() as td:
        for label, patch in VARIANTS:
            variant_path = Path(td) / f"helix_{label}.toml"
            write_variant_config(base_config, patch, variant_path)
            for rep in range(args.repeats):
                tag = label if args.repeats == 1 else f"{label}_r{rep + 1}"
                print(f"[{tag}] running SNOW oracle-only N=65...")
                t0 = time.time()
                raw_result = run_bench(variant_path, args.genome)
                tagged = tag_result(raw_result, tag)
                r = summarize(tagged)
                r["file"] = str(tagged)
                results[label].append(r)
                dt = time.time() - t0
                print(
                    f"  done in {dt:.1f}s: avg_tier={r['avg_tier']:.2f}  "
                    f"miss_rate={r['miss_rate']:.3f}  avg_tokens={r['avg_tokens']:.0f}"
                )

    # Aggregate per-variant stats
    agg: dict[str, dict] = {}
    for label, runs in results.items():
        agg[label] = {
            "n_runs": len(runs),
            "avg_tier": _stats([r["avg_tier"] for r in runs]),
            "miss_rate": _stats([r["miss_rate"] for r in runs]),
            "avg_tokens": _stats([r["avg_tokens"] for r in runs]),
            "runs": runs,
        }

    # Comparison table
    print()
    if args.repeats == 1:
        print("=" * 85)
        print(f"{'variant':<22} {'avg_tier':>10} {'miss_rate':>12} {'avg_tokens':>12} {'delta vs base':>22}")
        print("-" * 85)
        base_tier = agg["baseline"]["avg_tier"]["mean"]
        base_miss = agg["baseline"]["miss_rate"]["mean"]
        for label, _ in VARIANTS:
            a = agg[label]
            t = a["avg_tier"]["mean"]
            m = a["miss_rate"]["mean"]
            tok = a["avg_tokens"]["mean"]
            dt = t - base_tier
            dm = m - base_miss
            delta_str = "" if label == "baseline" else f"Δtier={dt:+.3f} Δmiss={dm:+.3f}"
            print(f"{label:<22} {t:>10.2f} {m:>12.3f} {tok:>12.0f} {delta_str:>22}")
        print("=" * 85)
    else:
        # Multi-repeat view: mean ± stddev, range
        print("=" * 100)
        print(f"{'variant':<22} {'avg_tier (mean±sd)':>22} {'miss_rate (mean±sd)':>24} {'min-max miss':>16}")
        print("-" * 100)
        for label, _ in VARIANTS:
            a = agg[label]
            t = a["avg_tier"]
            m = a["miss_rate"]
            t_str = f"{t['mean']:.3f} ± {t['stddev']:.3f}"
            m_str = f"{m['mean']:.3f} ± {m['stddev']:.3f}"
            range_str = f"[{m['min']:.3f}, {m['max']:.3f}]"
            print(f"{label:<22} {t_str:>22} {m_str:>24} {range_str:>16}")
        print("-" * 100)
        # Also show deltas vs baseline mean
        base_t = agg["baseline"]["avg_tier"]["mean"]
        base_m = agg["baseline"]["miss_rate"]["mean"]
        base_t_sd = agg["baseline"]["avg_tier"]["stddev"]
        base_m_sd = agg["baseline"]["miss_rate"]["stddev"]
        print(f"\n(baseline variance — avg_tier σ={base_t_sd:.3f}, miss_rate σ={base_m_sd:.3f})")
        print("delta vs baseline mean:")
        for label, _ in VARIANTS:
            if label == "baseline":
                continue
            a = agg[label]
            dt = a["avg_tier"]["mean"] - base_t
            dm = a["miss_rate"]["mean"] - base_m
            # Signal-to-noise check — is |Δ| > 2σ of baseline?
            noise_t = 2 * base_t_sd
            noise_m = 2 * base_m_sd
            t_signal = "✓" if abs(dt) > noise_t else "✗"
            m_signal = "✓" if abs(dm) > noise_m else "✗"
            print(
                f"  {label:<22} Δtier={dt:+.3f} ({t_signal} >2σ) "
                f"Δmiss={dm:+.3f} ({m_signal} >2σ)"
            )
        print("=" * 100)

    # Save summary
    summary_path = RESULTS_DIR / f"ablation_sweep_{time.strftime('%Y-%m-%d')}_r{args.repeats}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "genome": args.genome,
                "repeats": args.repeats,
                "variants": agg,
            },
            f, indent=2,
        )
    print(f"\n[ablation] summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

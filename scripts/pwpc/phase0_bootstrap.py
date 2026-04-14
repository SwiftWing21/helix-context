"""Phase 0 PWPC bootstrap on existing cwola_log export.

Computes per-dimension precision Π = 1/(var(score) + ε_floor) for each of
the 9 retrieval tiers, conditional on the tier firing (score present).

Splits by bucket (A = accepted, B = re-queried) to check whether
per-dimension precision shows content-dependent structure — the Phase 0
gate from PWPC_EXPERIMENT_SPEC.md adapted to helix's substrate.

Usage:
    python scripts/pwpc/phase0_bootstrap.py <export_json> [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

TIER_KEYS = [
    "fts5", "splade", "sema_boost", "lex_anchor",
    "tag_exact", "tag_prefix", "pki", "harmonic", "sr",
]

EPSILON_FLOOR = 1e-9


def load_export(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_scores(rows: list[dict[str, Any]], bucket_filter: str | None = None) -> dict[str, list[float]]:
    """Return {tier_key: [scores...]} for rows matching bucket_filter (or all if None)."""
    out: dict[str, list[float]] = {k: [] for k in TIER_KEYS}
    for row in rows:
        if bucket_filter is not None and row.get("bucket") != bucket_filter:
            continue
        feats = row.get("tier_features") or {}
        for k in TIER_KEYS:
            v = feats.get(k)
            if v is None:
                continue
            try:
                out[k].append(float(v))
            except (TypeError, ValueError):
                continue
    return out


def summarize(scores: dict[str, list[float]], total_rows: int) -> dict[str, dict[str, float]]:
    """For each tier: n, fire_rate, mean, var, precision (1/var), median, min, max."""
    result: dict[str, dict[str, float]] = {}
    for k in TIER_KEYS:
        s = scores[k]
        n = len(s)
        if n == 0:
            result[k] = {
                "n": 0, "fire_rate": 0.0, "mean": float("nan"),
                "var": float("nan"), "precision": float("nan"),
                "median": float("nan"), "min": float("nan"), "max": float("nan"),
            }
            continue
        mean = statistics.fmean(s)
        var = statistics.pvariance(s, mu=mean) if n > 1 else 0.0
        precision = 1.0 / (var + EPSILON_FLOOR)
        result[k] = {
            "n": n,
            "fire_rate": n / total_rows if total_rows else 0.0,
            "mean": mean,
            "var": var,
            "precision": precision,
            "median": statistics.median(s),
            "min": min(s),
            "max": max(s),
        }
    return result


def fmt_num(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "  n/a"
    if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:>10.3e}"
    return f"{x:>10.3f}"


def print_table(title: str, summary: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        f"\n### {title}",
        "",
        "| tier | n | fire_rate | mean | var | Π (=1/var) | median | min | max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in TIER_KEYS:
        r = summary[k]
        lines.append(
            f"| {k} | {r['n']} | {r['fire_rate']:.3f} | "
            f"{fmt_num(r['mean']).strip()} | {fmt_num(r['var']).strip()} | "
            f"{fmt_num(r['precision']).strip()} | {fmt_num(r['median']).strip()} | "
            f"{fmt_num(r['min']).strip()} | {fmt_num(r['max']).strip()} |"
        )
    return lines


def precision_ratio(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]]) -> list[str]:
    """Compare A-bucket vs B-bucket precision — the PWPC Phase 0 gate."""
    lines = [
        "",
        "### Per-tier precision ratio (A vs B)",
        "",
        "Content-dependent structure test from PWPC spec. If Π differs meaningfully by bucket, ",
        "per-dimension precision carries information about retrieval outcome. Ratio > 1 means ",
        "A-bucket is *more* precise (more consistent score) for that dimension; ratio < 1 means ",
        "B-bucket is more precise.",
        "",
        "| tier | Π_A | Π_B | Π_A / Π_B | n_A | n_B | mean_A | mean_B | mean Δ (A-B) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in TIER_KEYS:
        ra, rb = a[k], b[k]
        if ra["n"] == 0 or rb["n"] == 0:
            lines.append(
                f"| {k} | {fmt_num(ra['precision']).strip()} | "
                f"{fmt_num(rb['precision']).strip()} | n/a | "
                f"{ra['n']} | {rb['n']} | "
                f"{fmt_num(ra['mean']).strip()} | {fmt_num(rb['mean']).strip()} | n/a |"
            )
            continue
        ratio = ra["precision"] / rb["precision"] if rb["precision"] > 0 else float("inf")
        mean_delta = ra["mean"] - rb["mean"]
        lines.append(
            f"| {k} | {fmt_num(ra['precision']).strip()} | "
            f"{fmt_num(rb['precision']).strip()} | {ratio:.3f} | "
            f"{ra['n']} | {rb['n']} | "
            f"{fmt_num(ra['mean']).strip()} | {fmt_num(rb['mean']).strip()} | "
            f"{fmt_num(mean_delta).strip()} |"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path, help="cwola_export_*.json path")
    ap.add_argument("--out", type=Path, default=Path("docs/collab/comms/pwpc_phase0_bootstrap.md"))
    args = ap.parse_args()

    rows = load_export(args.export)
    total = len(rows)
    a_rows = [r for r in rows if r.get("bucket") == "A"]
    b_rows = [r for r in rows if r.get("bucket") == "B"]

    all_scores = collect_scores(rows)
    a_scores = collect_scores(rows, "A")
    b_scores = collect_scores(rows, "B")

    all_summary = summarize(all_scores, total)
    a_summary = summarize(a_scores, len(a_rows))
    b_summary = summarize(b_scores, len(b_rows))

    out_lines: list[str] = []
    out_lines += [
        "# Phase 0 PWPC Bootstrap — helix side",
        "",
        f"**Source:** `{args.export.name}` (N={total} rows; {len(a_rows)} A, {len(b_rows)} B)",
        f"**Party:** swift_wing21 (single-party — no cross-party generalization possible on this slice)",
        "**Method:** per-tier variance + precision Π = 1/(var + 1e-9), conditional on tier firing (score present in `tier_features`).",
        "**Caveat:** 95.3% B-rate is inflated by 5-min synthetic-session windowing on burst traffic; treat A-vs-B differences here as methodology validation, not load-bearing findings until organic data accumulates (~2–3 weeks) or schema is enriched per Phase 1.",
        "",
        "## Interpretation guide",
        "",
        "- **fire_rate**: fraction of retrievals where this tier produced a score. Low fire_rate (sema_boost 17%, sr 59%) means the tier is gated — most rows won't inform its precision.",
        "- **Π (precision)**: higher = more consistent score across rows = tier is firing in a narrow band. Lower = tier's output has high variance across rows.",
        "- **Π_A / Π_B**: if a tier's precision differs between A-bucket and B-bucket, that tier carries content-dependent signal. Ratio ≈ 1 means no differentiation on this slice.",
        "",
    ]

    out_lines += print_table("All rows (N = {})".format(total), all_summary)
    out_lines += print_table("A-bucket only (N = {})".format(len(a_rows)), a_summary)
    out_lines += print_table("B-bucket only (N = {})".format(len(b_rows)), b_summary)
    out_lines += precision_ratio(a_summary, b_summary)

    # Reader's-digest verdict
    out_lines += [
        "",
        "## Phase 0 gate — verdict",
        "",
    ]

    differentiating = []
    for k in TIER_KEYS:
        ra, rb = a_summary[k], b_summary[k]
        if ra["n"] < 5 or rb["n"] < 5:
            continue
        if rb["precision"] <= 0:
            continue
        ratio = ra["precision"] / rb["precision"]
        # Meaningful = ratio outside [0.5, 2.0]
        if ratio < 0.5 or ratio > 2.0:
            differentiating.append((k, ratio, ra["mean"], rb["mean"]))

    if differentiating:
        out_lines.append(
            f"**{len(differentiating)} of 9 tiers show meaningful A-vs-B precision ratio** "
            f"(outside [0.5, 2.0]):"
        )
        out_lines.append("")
        for k, ratio, ma, mb in differentiating:
            direction = "A more precise" if ratio > 1 else "B more precise"
            out_lines.append(
                f"- **{k}**: Π_A/Π_B = {ratio:.3f} ({direction}); mean_A = {ma:.3f}, mean_B = {mb:.3f}"
            )
        out_lines.append("")
        out_lines.append(
            "This is a methodology-level positive signal — per-dimension precision is not "
            "uniform across bucket labels. With the B-rate caveat above, treat this as "
            "'PWPC Phase 0 analysis is viable on helix data' rather than 'these tiers predict retrieval failure'."
        )
    else:
        out_lines.append(
            "**No tier shows meaningful A-vs-B precision ratio** on this slice. "
            "Likely causes (in decreasing order of probability): (1) 95.3% B-rate is a "
            "synthetic-session artifact and labels don't reflect retrieval outcome; "
            "(2) precision needs to be computed on raw scores, not the normalized ones "
            "already in tier_features; (3) D1–D9 don't carry content-dependent precision "
            "structure (would be surprising and worth investigating)."
        )

    out_lines += [
        "",
        "## Next moves",
        "",
        "1. Schema enrichment (this week, Phase 1 prerequisite): add per-tier **raw** scores + `query_sema[20]` + `top_candidate_sema[20]` columns to `cwola_log`; re-export after a few days of organic traffic.",
        "2. When enriched data arrives: re-run this script. If A-vs-B precision ratio becomes more meaningful on organic data, PWPC Phase 0 gate passes cleanly.",
        "3. Per-dimension precision as a live telemetry signal: add `helix_tier_precision{tier}` gauges to OTel dashboard (Raude's Sprint 5A infrastructure makes this trivial).",
        "4. Sparse-firing tiers (sema_boost 17%, sr 59%, pki 92%) — per-class firing-rate breakdown is the Sprint 5/6 instrumentation item from Raude's Council Triage; directly informs which tiers have enough data for meaningful Π.",
        "",
        "— Laude (generated by `scripts/pwpc/phase0_bootstrap.py`)",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  total rows: {total}")
    print(f"  A-bucket: {len(a_rows)}  B-bucket: {len(b_rows)}")
    print(f"  differentiating tiers (|Pi_A/Pi_B| outside [0.5, 2.0]): {len(differentiating)}")


if __name__ == "__main__":
    main()

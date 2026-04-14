"""Lockstep-as-failure empirical test on enriched cwola_log export.

The Phase 0 v2 drilldown flagged that the top-10 B-bucket sema_boost queries
had *all 9 tiers firing* with high cos(query_sema, top_candidate_sema). This
script tests whether the pattern holds on the full 1865-row dataset via
multiple candidate lockstep scalars.

Candidate scalars (per row):

  n_tiers_fired   Simplest — how many of the 9 tiers produced a score
  mean_z          Mean z-score across tiers that fired (standardised
                  per tier over the full dataset)
  min_z           Min z-score across fired tiers (high = ALL tiers
                  simultaneously above average = lockstep)
  all9_fired      Binary: did every one of the 9 tiers fire this query?

For each scalar we compute:
  - Mean per bucket A vs B
  - Mean delta (A - B) with sign convention "negative = B is more locked"
  - Point-biserial correlation (equivalent to Pearson r when one var is 0/1)

Gate from the batman-task proposal: does any scalar produce
|r| >= 0.2 against bucket label on this data?

Usage:
    python scripts/pwpc/lockstep_test.py <v2_export.json> [--out <path>]
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


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tier_zscore_stats(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Compute (mean, stdev) per tier, conditional on the tier firing."""
    stats: dict[str, tuple[float, float]] = {}
    for tier in TIER_KEYS:
        vals: list[float] = []
        for r in rows:
            v = (r.get("tier_features") or {}).get(tier)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals:
            stats[tier] = (0.0, 1.0)
            continue
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 1.0
        stats[tier] = (mean, std if std > 1e-9 else 1.0)
    return stats


def per_row_scalars(
    row: dict[str, Any], zstats: dict[str, tuple[float, float]]
) -> dict[str, float | int]:
    feats = row.get("tier_features") or {}
    n_fired = 0
    z_values: list[float] = []
    for tier in TIER_KEYS:
        v = feats.get(tier)
        if v is None:
            continue
        n_fired += 1
        mean, std = zstats[tier]
        try:
            z = (float(v) - mean) / std
        except (TypeError, ValueError):
            continue
        z_values.append(z)

    return {
        "n_tiers_fired": n_fired,
        "all9_fired": 1 if n_fired == 9 else 0,
        "mean_z": statistics.fmean(z_values) if z_values else 0.0,
        "min_z": min(z_values) if z_values else 0.0,
        "max_z": max(z_values) if z_values else 0.0,
    }


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation. None if stdev is zero."""
    n = len(xs)
    if n < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-9 or sy < 1e-9:
        return None
    return num / (sx * sy)


def split_by_bucket(
    rows: list[dict[str, Any]], scalars: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    a, b = [], []
    for r, s in zip(rows, scalars):
        if r.get("bucket") == "A":
            a.append(s)
        elif r.get("bucket") == "B":
            b.append(s)
    return a, b


def mean_on(scalars: list[dict[str, Any]], key: str) -> float:
    if not scalars:
        return float("nan")
    return statistics.fmean(float(s[key]) for s in scalars)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/collab/comms/LOCKSTEP_TEST.md"))
    args = ap.parse_args()

    rows = load_rows(args.export)
    n = len(rows)
    a_rows = [r for r in rows if r.get("bucket") == "A"]
    b_rows = [r for r in rows if r.get("bucket") == "B"]

    zstats = tier_zscore_stats(rows)
    scalars = [per_row_scalars(r, zstats) for r in rows]
    a_sc, b_sc = split_by_bucket(rows, scalars)

    # Bucket label as 0/1 for Pearson (A=0, B=1). Restrict to A+B rows.
    bucket_labels: list[float] = []
    filtered_scalars: list[dict[str, Any]] = []
    for r, s in zip(rows, scalars):
        b = r.get("bucket")
        if b == "A":
            bucket_labels.append(0.0)
            filtered_scalars.append(s)
        elif b == "B":
            bucket_labels.append(1.0)
            filtered_scalars.append(s)

    scalar_keys = ["n_tiers_fired", "all9_fired", "mean_z", "min_z", "max_z"]

    # Correlations
    correlations: dict[str, float | None] = {}
    for k in scalar_keys:
        xs = [float(s[k]) for s in filtered_scalars]
        correlations[k] = pearson_r(xs, bucket_labels)

    # Also: correlation of cos(query_sema, top_candidate_sema) with bucket
    cos_vals: list[float] = []
    cos_labels: list[float] = []
    for r in rows:
        qs = r.get("query_sema")
        cs = r.get("top_candidate_sema")
        if not qs or not cs or len(qs) != len(cs):
            continue
        dot = sum(a * b for a, b in zip(qs, cs))
        na = math.sqrt(sum(a * a for a in qs))
        nb = math.sqrt(sum(b * b for b in cs))
        if na < 1e-9 or nb < 1e-9:
            continue
        bucket = r.get("bucket")
        if bucket not in ("A", "B"):
            continue
        cos_vals.append(dot / (na * nb))
        cos_labels.append(0.0 if bucket == "A" else 1.0)
    cos_r = pearson_r(cos_vals, cos_labels)

    # Build artifact
    lines: list[str] = [
        "# Lockstep-as-failure empirical test",
        "",
        f"**Source:** `{args.export.name}` (N = {n}; A = {len(a_rows)}, B = {len(b_rows)})",
        "",
        "Testing whether the Phase 0 v2 drilldown finding — that the top-10 "
        "B-bucket sema_boost queries had all 9 tiers firing with high cosine "
        "agreement — generalises from that 10-row slice to the full 1865-row "
        "dataset.",
        "",
        "**Gate:** does any candidate lockstep scalar produce |Pearson r| ≥ 0.2 "
        "against the 0/1 bucket label? If yes, the antiresonance signature is a "
        "load-bearing training signal, not a top-10 curiosity.",
        "",
        "## Candidate scalars",
        "",
        "| scalar | definition | meaning |",
        "|---|---|---|",
        "| `n_tiers_fired` | how many of the 9 tiers produced a score | higher = more dimensions active |",
        "| `all9_fired` | binary: did every tier fire? | 1 = full lockstep possible |",
        "| `mean_z` | mean z-score across fired tiers | high = all fired tiers are above their own mean |",
        "| `min_z` | min z-score across fired tiers | high = the weakest fired tier is still above average (strongest lockstep signal) |",
        "| `max_z` | max z-score across fired tiers | high = at least one tier is strong |",
        "",
        "## Results",
        "",
        "### Per-bucket means",
        "",
        "| scalar | mean_A | mean_B | Δ (A−B) |",
        "|---|---|---|---|",
    ]
    for k in scalar_keys:
        ma = mean_on(a_sc, k)
        mb = mean_on(b_sc, k)
        lines.append(f"| {k} | {ma:.4f} | {mb:.4f} | {ma - mb:+.4f} |")

    lines += [
        "",
        "### Pearson r vs bucket label (A=0, B=1)",
        "",
        "| scalar | r | |r| ≥ 0.2? |",
        "|---|---|---|",
    ]
    any_passed = False
    for k in scalar_keys:
        r = correlations[k]
        if r is None:
            lines.append(f"| {k} | n/a | — |")
            continue
        passed = abs(r) >= 0.2
        if passed:
            any_passed = True
        lines.append(f"| {k} | {r:+.4f} | {'**YES**' if passed else 'no'} |")
    if cos_r is not None:
        passed = abs(cos_r) >= 0.2
        if passed:
            any_passed = True
        lines.append(f"| cos(q,c) | {cos_r:+.4f} | {'**YES**' if passed else 'no'} |")

    lines += [
        "",
        "## Verdict",
        "",
    ]
    if any_passed:
        lines.append(
            "**Gate PASSES.** At least one lockstep scalar produces |r| ≥ 0.2 "
            "against bucket label on the full 1865-row dataset. The "
            "antiresonance signature is a load-bearing signal on helix's "
            "substrate, not an artifact of the top-10 drilldown."
        )
    else:
        lines.append(
            "**Gate FAILS.** No lockstep scalar exceeds |r| = 0.2 on this "
            "dataset. Possible reasons:"
        )
        lines.append("")
        lines += [
            "1. The top-10 drilldown pattern is real but narrow — it holds "
            "   for the extreme high-score tail but not across the full "
            "   score distribution.",
            "2. The 95% B-rate is diluting signal by mislabelling many "
            "   low-lockstep queries as B.",
            "3. A scalar summary of the 9-tier pattern loses the structure; "
            "   the 9×9 matrix may still carry the signal but not any single "
            "   projection of it.",
            "4. The pattern requires interaction with query length or query "
            "   type, not captured by these scalars.",
        ]

    lines += [
        "",
        "## Signed interpretation",
        "",
        "Regardless of whether the gate passes, the **sign of each correlation** "
        "is the interesting part:",
        "",
        "- **Positive r** means higher lockstep → more likely to be bucket B "
        "  (failed retrieval). This is the antiresonance signature: agreement "
        "  = failure mode.",
        "- **Negative r** means higher lockstep → more likely to be bucket A "
        "  (accepted). This is the conventional prior: agreement = confidence.",
        "",
        "The conventional prior predicts all correlations should be **negative**. "
        "If correlations are positive (even if small), that directionally supports "
        "the antiresonance hypothesis even without clearing the 0.2 threshold.",
        "",
        "## Next moves",
        "",
        "1. If the gate passes: bake the best-performing scalar into batman's "
        "   agreement head as a baseline before the 9×9 matrix takes over.",
        "2. If the gate fails but signs are consistently antiresonance-aligned: "
        "   still a useful direction, but needs the full matrix to surface.",
        "3. If signs are mixed: the top-10 drilldown was content-specific — "
        "   dig into query-type segmentation before designing the head.",
        "",
        "— Laude (generated by `scripts/pwpc/lockstep_test.py`)",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"N = {n}; A = {len(a_rows)}; B = {len(b_rows)}")
    print("Correlations:")
    for k in scalar_keys:
        r = correlations[k]
        print(f"  {k:16s}  r = {r:+.4f}" if r is not None else f"  {k:16s}  r = n/a")
    if cos_r is not None:
        print(f"  {'cos(q,c)':16s}  r = {cos_r:+.4f}")


if __name__ == "__main__":
    main()

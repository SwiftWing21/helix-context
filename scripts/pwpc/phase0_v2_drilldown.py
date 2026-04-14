"""Phase 0 v2 drilldown — enriched-export analysis on cwola_log.

Builds on phase0_bootstrap.py. Adds two things the v1 bootstrap couldn't do:

1. **Per-row semantic agreement** — cosine(query_sema, top_candidate_sema).
   A direct intra-query agreement signal at the semantic level, orthogonal
   to the tier score distribution. Split by bucket A vs B.

2. **sema_boost outlier drill** — the v1 bootstrap flagged
   mean(sema_boost | B) ≈ 2× mean(sema_boost | A) and
   Pi_A / Pi_B ≈ 2.49. That matches the antiresonance hypothesis
   (high agreement on template-shape queries is the failure mode).
   This script surfaces the actual queries behind those numbers so we
   can eyeball whether the B-bucket high-sema_boost queries really are
   template-shaped.

Usage:
    python scripts/pwpc/phase0_v2_drilldown.py <v2_export.json> [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cosine(a: list[float], b: list[float]) -> float | None:
    if a is None or b is None or len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def summarise_cosines(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    sorted_v = sorted(values)
    n = len(values)

    def pct(p: float) -> float:
        i = max(0, min(n - 1, int(n * p)))
        return sorted_v[i]

    return {
        "n": n,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values) if n > 1 else 0.0,
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "min": sorted_v[0],
        "max": sorted_v[-1],
    }


def top_outliers(
    rows: list[dict[str, Any]], tier: str, bucket: str, k: int = 10
) -> list[dict[str, Any]]:
    """Rows with highest `tier` score in the given bucket, plus their embedding cosine."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        if r.get("bucket") != bucket:
            continue
        feats = r.get("tier_features") or {}
        v = feats.get(tier)
        if v is None:
            continue
        try:
            scored.append((float(v), r))
        except (TypeError, ValueError):
            continue
    scored.sort(key=lambda p: p[0], reverse=True)
    out = []
    for score, r in scored[:k]:
        cos = cosine(r.get("query_sema"), r.get("top_candidate_sema"))
        out.append({
            "retrieval_id": r.get("retrieval_id"),
            "query": r.get("query"),
            "score": score,
            "cos_q_c": cos,
            "requery_delta_s": r.get("requery_delta_s"),
            "top_gene_id": r.get("top_gene_id"),
            "n_tiers_fired": sum(1 for v in (r.get("tier_features") or {}).values() if v is not None),
        })
    return out


def format_outlier_table(outliers: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| retrieval_id | score | cos(q,c) | n_tiers | requery_Δs | query |",
        "|---|---|---|---|---|---|",
    ]
    for o in outliers:
        q = (o["query"] or "").replace("|", "\\|").replace("\n", " ")
        if len(q) > 80:
            q = q[:80] + "..."
        cos_str = f"{o['cos_q_c']:.3f}" if o["cos_q_c"] is not None else "n/a"
        rq = f"{o['requery_delta_s']:.1f}" if o["requery_delta_s"] is not None else "—"
        lines.append(
            f"| {o['retrieval_id']} | {o['score']:.1f} | {cos_str} | "
            f"{o['n_tiers_fired']} | {rq} | {q} |"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path, help="cwola_export_*.json (v2, enriched)")
    ap.add_argument("--out", type=Path, default=Path("docs/collab/comms/PHASE0_V2_DRILLDOWN.md"))
    ap.add_argument("--k", type=int, default=10, help="top-K outliers per bucket")
    args = ap.parse_args()

    rows = load_rows(args.export)
    n_total = len(rows)
    a_rows = [r for r in rows if r.get("bucket") == "A"]
    b_rows = [r for r in rows if r.get("bucket") == "B"]

    # ── Part 1: semantic agreement (cos sim) ───────────────────────
    cos_all: list[float] = []
    cos_a: list[float] = []
    cos_b: list[float] = []
    missing_embed = 0
    for r in rows:
        c = cosine(r.get("query_sema"), r.get("top_candidate_sema"))
        if c is None:
            missing_embed += 1
            continue
        cos_all.append(c)
        if r.get("bucket") == "A":
            cos_a.append(c)
        elif r.get("bucket") == "B":
            cos_b.append(c)

    cos_all_s = summarise_cosines(cos_all)
    cos_a_s = summarise_cosines(cos_a)
    cos_b_s = summarise_cosines(cos_b)

    # ── Part 2: sema_boost outliers ────────────────────────────────
    top_a = top_outliers(rows, "sema_boost", "A", args.k)
    top_b = top_outliers(rows, "sema_boost", "B", args.k)

    # ── Build the artifact ─────────────────────────────────────────
    lines: list[str] = []
    lines += [
        "# Phase 0 v2 drilldown — semantic agreement + sema_boost outliers",
        "",
        f"**Source:** `{args.export.name}` (N = {n_total}; A = {len(a_rows)}, B = {len(b_rows)})",
        f"**Missing embeddings:** {missing_embed} rows (should be 0 post-backfill)",
        "",
        "Two questions this doc answers:",
        "",
        "1. Does **cos(query_sema, top_candidate_sema)** differentiate A vs B buckets? "
        "This is the simplest possible intra-query semantic-agreement signal — "
        "orthogonal to the 9 tier scores.",
        "2. What do the **top-k sema_boost queries** actually look like? "
        "Phase 0 v1 found `mean(sema_boost | B) ≈ 2× mean(sema_boost | A)` — "
        "are the high-sema_boost B queries template-shaped as the antiresonance "
        "hypothesis predicts?",
        "",
        "## Part 1 — semantic agreement (cos q,c)",
        "",
        "| split | n | mean | median | p10 | p25 | p75 | p90 | stdev |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, s in (("all", cos_all_s), ("A", cos_a_s), ("B", cos_b_s)):
        if s.get("n", 0) == 0:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | "
            f"{s['p10']:.3f} | {s['p25']:.3f} | {s['p75']:.3f} | {s['p90']:.3f} | "
            f"{s['stdev']:.3f} |"
        )

    if cos_a_s.get("n", 0) and cos_b_s.get("n", 0):
        delta = cos_a_s["mean"] - cos_b_s["mean"]
        lines += [
            "",
            f"**A-bucket mean cosine − B-bucket mean cosine = {delta:+.4f}**",
            "",
        ]
        if abs(delta) < 0.01:
            lines.append(
                "Delta is within noise — on this slice, embedding-space "
                "agreement does not differentiate A from B. Either (a) the "
                "20d ΣĒMA projection is too coarse to pick up bucket-level "
                "differences, (b) the bucket labels are mostly noise at this "
                "B-rate, or (c) semantic agreement is genuinely not the right "
                "axis — tier-score agreement (Part 2 + 9×9 matrix) is."
            )
        else:
            direction = "A queries are more semantically aligned with their top candidate" if delta > 0 else "B queries are more semantically aligned with their top candidate"
            lines.append(
                f"Non-trivial shift. {direction}. "
                f"Worth checking whether this is driven by query-length bias "
                f"(short queries project into a sparser 20d region so cosines "
                f"get inflated) before drawing conclusions."
            )

    # ── Part 2 ─────────────────────────────────────────────────────
    lines += [
        "",
        "## Part 2 — sema_boost outliers",
        "",
        "### Top-k sema_boost scores in **A-bucket** (accepted retrievals)",
        "",
    ]
    lines += format_outlier_table(top_a)
    lines += [
        "",
        "### Top-k sema_boost scores in **B-bucket** (re-queried within 60s)",
        "",
    ]
    lines += format_outlier_table(top_b)

    # ── Interpretation prompt ──────────────────────────────────────
    lines += [
        "",
        "## Antiresonance read",
        "",
        "The antiresonance hypothesis: high tier agreement on template-shape "
        "queries is the failure mode (every tier reads the same surface "
        "feature; confidence is lockstep but wrong). If this is what "
        "sema_boost is picking up, the top-B queries above should show:",
        "",
        "- Template structure ('what is X of Y', 'how does A relate to B') "
        "more often than the top-A queries",
        "- cos(q, c) similar or higher than A-bucket (dimensions agree "
        "semantically AND score highly — all-green failure mode)",
        "- Multiple tiers fired simultaneously (n_tiers_fired high)",
        "",
        "Inspect the tables above. If the pattern holds, sema_boost is a "
        "concrete lever for tuning: either down-weight it when other tiers "
        "also score high (lockstep detection), or invert its usage (high "
        "sema_boost = suspicious, not confident).",
        "",
        "## Next moves",
        "",
        "1. Label the top-B queries by hand (template / natural / mixed) and "
        "   quantify whether the antiresonance signature holds past eyeballing.",
        "2. Full 9×9 tier correlation matrix (not in this drilldown — that "
        "   needs the per-tier raw scores and belongs with the batman follow-up).",
        "3. If the cosine delta in Part 1 is non-trivial, investigate "
        "   query-length bias before crediting the signal.",
        "",
        "— Laude (generated by `scripts/pwpc/phase0_v2_drilldown.py`)",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  rows: {n_total}  A: {len(a_rows)}  B: {len(b_rows)}")
    print(f"  cos(q,c) available on {len(cos_all)} rows ({missing_embed} missing)")
    if cos_a_s.get("n") and cos_b_s.get("n"):
        print(f"  mean cos A: {cos_a_s['mean']:.4f}   mean cos B: {cos_b_s['mean']:.4f}   delta: {cos_a_s['mean'] - cos_b_s['mean']:+.4f}")
    print(f"  sema_boost outliers: top-{args.k} per bucket")


if __name__ == "__main__":
    main()

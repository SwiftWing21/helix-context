"""
Per-gene legibility header — Sprint 1 of the AI-consumer roadmap.

The downstream LLM consuming /context today sees a block like:

    <expressed_context>
    spliced text from gene 1
    ---
    spliced text from gene 2
    </expressed_context>

It has no way to tell WHICH tier fired each gene, WHICH genes are strong
hits vs reaches, or HOW MUCH content was compressed away. The consumer
council (2026-04-14) flagged this as the biggest single-shot legibility
win available.

This module emits one metadata line per gene block, consolidating all
three Sprint 1 asks from `docs/FUTURE/AI_CONSUMER_ROADMAP_2026-04-14.md`
into a single ~50-60 token prefix:

    [gene=abc12345 ◆ fired=harmonic:2.3,lex_anchor:1.1 1000→200c]
    <spliced content>
    ---
    [gene=def67890 ◇ fired=sema_boost:1.8 180c]
    <spliced content>

Covers:
    1. Fired-tier tags per gene (top-3 by contribution)
    2. Hash preview — short gene_id links back to the full content + the
       raw→compressed char ratio signals whether to re-query via the
       forthcoming /context/expand endpoint (Sprint 3)
    3. Confidence marker derived from z-normalized retrieval score across
       the expressed gene set: ◆ strong (z≥1), ◇ moderate (0≤z<1), ⬦ weak

The format is pure-text; no markup changes, no schema migration. Any
downstream consumer can strip it by skipping lines starting with `[gene=`.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Strong retrieval signal (above-average for this response set)
SYMBOL_STRONG = "◆"
# Moderate — at or near mean
SYMBOL_MODERATE = "◇"
# Weak — below mean; the consumer should treat this as a reach
SYMBOL_WEAK = "⬦"


def _confidence_symbol(z_score: float) -> str:
    """Map a z-normalized score to a visual trust marker.

    Thresholds are chosen so that in a response with 12 genes normally
    distributed, roughly the top ~15% surface as ◆, middle ~35% as ◇,
    and bottom half as ⬦. Consumers can treat weak genes as candidates
    to skip or re-expand.
    """
    if z_score >= 1.0:
        return SYMBOL_STRONG
    if z_score >= 0.0:
        return SYMBOL_MODERATE
    return SYMBOL_WEAK


def _z_normalize(score: float, mean: float, std: float) -> float:
    """Z-score with a guard for degenerate (zero or near-zero) std.

    Single-gene responses or runs where every gene produced an identical
    score would otherwise divide by zero. Returning 0.0 collapses the
    marker to ◇ (moderate) for those cases — honest: we genuinely can't
    tell which is stronger.
    """
    if std <= 1e-9:
        return 0.0
    return (score - mean) / std


def _format_fired_tiers(
    tier_contrib: Dict[str, float],
    max_tiers: int = 3,
) -> str:
    """Render the top-N tiers by contribution as `tier:X.X,tier:X.X,...`.

    Empty contrib → "none" (e.g. a gene pulled by co-activation bleed with
    no direct tier hit). Rounding to one decimal keeps the line compact
    without losing signal at typical retrieval-score scales (0.1 - 5.0).
    """
    if not tier_contrib:
        return "none"
    sorted_tiers = sorted(
        tier_contrib.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:max_tiers]
    return ",".join(f"{name}:{score:.1f}" for name, score in sorted_tiers)


def compute_score_stats(scores: Dict[str, float]) -> Tuple[float, float]:
    """Compute (mean, std) over the expressed gene set for z-normalization.

    Stats are ALWAYS computed per-response over ONLY the genes included in
    the response — not genome-wide. That way a response full of strong
    hits still has differentiation across its members rather than every
    gene showing ◆.

    Uses sample std (n-1 denominator). For n < 2 or uniform scores,
    returns std=0 so downstream z-normalize falls back to moderate.
    """
    if not scores:
        return 0.0, 0.0
    values = list(scores.values())
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def format_gene_header(
    gene_id: str,
    raw_chars: int,
    compressed_chars: int,
    combined_score: float,
    tier_contrib: Dict[str, float],
    score_stats: Tuple[float, float],
    *,
    id_width: int = 12,
    max_tiers: int = 3,
) -> str:
    """Render one per-gene metadata header line.

    Args:
        gene_id: Full gene_id (will be truncated to `id_width` chars for display)
        raw_chars: Length of g.content before splicing
        compressed_chars: Length of the final spliced text that reached the consumer
        combined_score: Aggregate retrieval score for this gene
        tier_contrib: Per-tier contribution dict (from genome.last_tier_contributions)
        score_stats: (mean, std) over all expressed genes in THIS response
        id_width: How many leading chars of gene_id to show (default 12)
        max_tiers: Cap on tiers listed in `fired=` (default top 3)

    Returns:
        Single-line bracketed header, no trailing newline.
    """
    short_id = gene_id[:id_width]
    mean, std = score_stats
    z = _z_normalize(combined_score, mean, std)
    symbol = _confidence_symbol(z)
    fired = _format_fired_tiers(tier_contrib, max_tiers)
    if raw_chars == compressed_chars:
        size_str = f"{compressed_chars}c"
    else:
        size_str = f"{raw_chars}→{compressed_chars}c"
    return f"[gene={short_id} {symbol} fired={fired} {size_str}]"

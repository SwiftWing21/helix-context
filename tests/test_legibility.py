"""
Sprint 1 legibility pack — unit tests for per-gene header helpers.

Covers:
- _confidence_symbol: ◆/◇/⬦ thresholds on z-score
- _z_normalize: handles zero/tiny std gracefully
- _format_fired_tiers: sort-by-score, top-N cap, empty case
- compute_score_stats: mean/std over expressed genes
- format_gene_header: full header assembly, edge cases (raw==compressed,
  empty tiers, short gene_ids, missing scores)

See docs/FUTURE/AI_CONSUMER_ROADMAP_2026-04-14.md Sprint 1.
"""

from __future__ import annotations

import pytest

from helix_context.legibility import (
    _confidence_symbol,
    _format_fired_tiers,
    _z_normalize,
    compute_score_stats,
    format_gene_header,
)


# ── _confidence_symbol ────────────────────────────────────────────────

def test_confidence_symbol_strong_at_z_1():
    assert _confidence_symbol(1.0) == "◆"


def test_confidence_symbol_strong_above_z_1():
    assert _confidence_symbol(2.5) == "◆"


def test_confidence_symbol_moderate_at_z_0():
    assert _confidence_symbol(0.0) == "◇"


def test_confidence_symbol_moderate_below_1():
    assert _confidence_symbol(0.5) == "◇"


def test_confidence_symbol_weak_negative():
    assert _confidence_symbol(-0.1) == "⬦"


def test_confidence_symbol_weak_very_negative():
    assert _confidence_symbol(-3.0) == "⬦"


# ── _z_normalize ──────────────────────────────────────────────────────

def test_z_normalize_basic():
    assert _z_normalize(score=10.0, mean=5.0, std=2.5) == pytest.approx(2.0)


def test_z_normalize_zero_std_returns_zero():
    # All scores equal → std=0 → z undefined → fallback to 0 (→ ◇ moderate)
    assert _z_normalize(score=5.0, mean=5.0, std=0.0) == 0.0


def test_z_normalize_tiny_std_returns_zero():
    # Prevent divide-by-zero blowup for nearly-identical scores
    assert _z_normalize(score=5.0, mean=5.0, std=1e-12) == 0.0


def test_z_normalize_negative():
    assert _z_normalize(score=2.0, mean=5.0, std=2.0) == pytest.approx(-1.5)


# ── _format_fired_tiers ───────────────────────────────────────────────

def test_format_fired_tiers_sorted_descending():
    contrib = {"sema_boost": 0.5, "harmonic": 2.3, "lex_anchor": 1.1}
    out = _format_fired_tiers(contrib)
    # top 3, by score desc
    assert out == "harmonic:2.3,lex_anchor:1.1,sema_boost:0.5"


def test_format_fired_tiers_caps_at_top_n():
    contrib = {f"tier_{i}": float(i) for i in range(9)}
    out = _format_fired_tiers(contrib, max_tiers=3)
    # top 3 should be tier_8, tier_7, tier_6
    assert out == "tier_8:8.0,tier_7:7.0,tier_6:6.0"


def test_format_fired_tiers_custom_max_tiers():
    contrib = {"a": 3.0, "b": 2.0, "c": 1.0}
    out = _format_fired_tiers(contrib, max_tiers=2)
    assert out == "a:3.0,b:2.0"


def test_format_fired_tiers_empty_returns_none_sentinel():
    assert _format_fired_tiers({}) == "none"


def test_format_fired_tiers_rounds_to_one_decimal():
    contrib = {"harmonic": 2.345678}
    out = _format_fired_tiers(contrib)
    assert "harmonic:2.3" in out


# ── compute_score_stats ───────────────────────────────────────────────

def test_compute_score_stats_empty():
    assert compute_score_stats({}) == (0.0, 0.0)


def test_compute_score_stats_single_returns_zero_std():
    # Single observation can't produce a std — return mean and 0 std
    mean, std = compute_score_stats({"a": 5.0})
    assert mean == 5.0
    assert std == 0.0


def test_compute_score_stats_basic():
    mean, std = compute_score_stats({"a": 2.0, "b": 4.0, "c": 6.0})
    assert mean == pytest.approx(4.0)
    # sample std with n-1 denom: sqrt(((2-4)^2+(4-4)^2+(6-4)^2)/2) = sqrt(4) = 2
    assert std == pytest.approx(2.0)


def test_compute_score_stats_uniform_scores_zero_std():
    mean, std = compute_score_stats({"a": 3.0, "b": 3.0, "c": 3.0})
    assert mean == 3.0
    assert std == 0.0


# ── format_gene_header ────────────────────────────────────────────────

def test_format_gene_header_basic_shape():
    out = format_gene_header(
        gene_id="abc123456789xyz",
        raw_chars=1000,
        compressed_chars=200,
        combined_score=10.0,
        tier_contrib={"harmonic": 2.3, "lex_anchor": 1.1},
        score_stats=(5.0, 2.5),
    )
    # Expect the fields in a single bracketed line
    assert out.startswith("[")
    assert out.endswith("]")
    assert "gene=abc123456789" in out       # short_id, 12 chars default
    assert "◆" in out                       # z = (10-5)/2.5 = 2.0 → strong
    assert "fired=harmonic:2.3,lex_anchor:1.1" in out
    assert "1000→200c" in out


def test_format_gene_header_short_gene_id():
    # Short gene_id shorter than id_width → pass through as-is
    out = format_gene_header(
        gene_id="abc123",
        raw_chars=10,
        compressed_chars=10,
        combined_score=0.0,
        tier_contrib={},
        score_stats=(0.0, 0.0),
    )
    assert "gene=abc123" in out


def test_format_gene_header_raw_equals_compressed_no_arrow():
    out = format_gene_header(
        gene_id="abc12345",
        raw_chars=150,
        compressed_chars=150,
        combined_score=0.0,
        tier_contrib={},
        score_stats=(0.0, 0.0),
    )
    assert "150c" in out
    assert "→" not in out  # no compression arrow when unchanged


def test_format_gene_header_empty_tiers():
    out = format_gene_header(
        gene_id="abc12345",
        raw_chars=100,
        compressed_chars=50,
        combined_score=1.0,
        tier_contrib={},
        score_stats=(1.0, 0.0),
    )
    assert "fired=none" in out


def test_format_gene_header_weak_score_marker():
    out = format_gene_header(
        gene_id="abc12345",
        raw_chars=100,
        compressed_chars=50,
        combined_score=1.0,
        tier_contrib={"harmonic": 0.1},
        score_stats=(5.0, 1.0),   # z = (1-5)/1 = -4 → weak
    )
    assert "⬦" in out


def test_format_gene_header_moderate_score_marker():
    out = format_gene_header(
        gene_id="abc12345",
        raw_chars=100,
        compressed_chars=50,
        combined_score=5.5,
        tier_contrib={"harmonic": 0.5},
        score_stats=(5.0, 1.0),   # z = 0.5 → moderate
    )
    assert "◇" in out


def test_format_gene_header_zero_std_falls_back_to_moderate():
    # All genes have identical score → std=0 → every gene gets ◇
    out = format_gene_header(
        gene_id="abc12345",
        raw_chars=100,
        compressed_chars=50,
        combined_score=5.0,
        tier_contrib={"harmonic": 1.0},
        score_stats=(5.0, 0.0),
    )
    assert "◇" in out


def test_format_gene_header_compressed_longer_than_raw_handled():
    # Shouldn't happen in practice but don't crash — just show compressed size
    out = format_gene_header(
        gene_id="abc12345",
        raw_chars=50,
        compressed_chars=60,
        combined_score=0.0,
        tier_contrib={},
        score_stats=(0.0, 0.0),
    )
    # Report as-is with arrow; downstream inspection can spot the anomaly
    assert "50→60c" in out or "60c" in out


# ── Integration: confirm header is a single line ────────────────────

def test_format_gene_header_is_one_line():
    out = format_gene_header(
        gene_id="abc123456789",
        raw_chars=1000,
        compressed_chars=200,
        combined_score=10.0,
        tier_contrib={"harmonic": 2.3, "lex_anchor": 1.1, "sema_boost": 0.5},
        score_stats=(5.0, 2.5),
    )
    assert "\n" not in out
    # Keep the header budget in check — shouldn't exceed ~120 chars for
    # typical genes (12 genes × 120c = ~1440c or ~360 tokens; well within
    # the 6% of expression budget we planned).
    assert len(out) < 160, f"Header too long ({len(out)} chars): {out}"


def test_format_gene_header_includes_all_three_sprint1_fields():
    """Smoke check that one header covers all 3 roadmap items."""
    out = format_gene_header(
        gene_id="deadbeefcafe",
        raw_chars=2000,
        compressed_chars=150,
        combined_score=15.0,
        tier_contrib={"harmonic": 3.0, "tag_exact": 1.5},
        score_stats=(5.0, 5.0),
    )
    # 1. Fired-tier tags (roadmap item 1)
    assert "fired=" in out
    assert "harmonic" in out
    # 2. Hash preview (roadmap item 2) — short gene_id + compression ratio
    assert "gene=deadbeefcafe" in out
    assert "2000→150c" in out
    # 3. Confidence marker (roadmap item 3)
    assert any(sym in out for sym in ("◆", "◇", "⬦"))


# ── Integration: _assemble emits headers when flag on ─────────────────

from helix_context.config import HelixConfig, BudgetConfig, GenomeConfig, RibosomeConfig
from helix_context.context_manager import HelixContextManager
from tests.conftest import make_gene


def _make_manager(legibility_enabled: bool) -> HelixContextManager:
    cfg = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(
            max_genes_per_turn=4,
            splice_aggressiveness=0.5,
            legibility_enabled=legibility_enabled,
        ),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        synonym_map={},
    )
    return HelixContextManager(cfg)


def test_assemble_emits_headers_when_flag_on():
    mgr = _make_manager(legibility_enabled=True)
    try:
        g1 = make_gene(content="alpha content", gene_id="aaaa1111bbbb2222")
        g2 = make_gene(content="beta content", gene_id="cccc3333dddd4444")
        # Seed retrieval state as if the pipeline had run
        mgr.genome.last_query_scores = {g1.gene_id: 10.0, g2.gene_id: 2.0}
        mgr.genome.last_tier_contributions = {
            g1.gene_id: {"harmonic": 3.0, "lex_anchor": 1.5},
            g2.gene_id: {"sema_boost": 0.8},
        }
        window = mgr._assemble(
            query="any",
            candidates=[g1, g2],
            spliced_map={g1.gene_id: "spliced-alpha", g2.gene_id: "spliced-beta"},
        )
        ec = window.expressed_context
        # Every gene got a header line
        assert ec.count("[gene=") == 2
        assert "gene=aaaa1111bbbb" in ec         # short-id 12-char truncation
        assert "gene=cccc3333dddd" in ec
        # Fired-tier annotation present for both
        assert "harmonic" in ec
        assert "sema_boost" in ec
        # Divider still present between the two blocks
        assert "\n---\n" in ec
    finally:
        mgr.close()


def test_assemble_no_headers_when_flag_off():
    mgr = _make_manager(legibility_enabled=False)
    try:
        g1 = make_gene(content="alpha content", gene_id="aaaa1111bbbb2222")
        mgr.genome.last_query_scores = {g1.gene_id: 10.0}
        mgr.genome.last_tier_contributions = {g1.gene_id: {"harmonic": 3.0}}
        window = mgr._assemble(
            query="any",
            candidates=[g1],
            spliced_map={g1.gene_id: "spliced-alpha"},
        )
        ec = window.expressed_context
        # Pre-Sprint-1 plain-text format: no per-gene header
        assert "[gene=" not in ec
        # But spliced content is still present
        assert "spliced-alpha" in ec
    finally:
        mgr.close()


def test_assemble_confidence_calibrated_against_response_set():
    """Strong gene relative to the set gets ◆; weak one gets ⬦."""
    mgr = _make_manager(legibility_enabled=True)
    try:
        # Three genes with wide score spread — highest should be ◆, lowest ⬦
        g1 = make_gene(content="first", gene_id="gene_strong00000")
        g2 = make_gene(content="second", gene_id="gene_moderate000")
        g3 = make_gene(content="third", gene_id="gene_weakzzzzzzz")
        mgr.genome.last_query_scores = {
            g1.gene_id: 10.0,
            g2.gene_id: 5.0,
            g3.gene_id: 1.0,
        }
        mgr.genome.last_tier_contributions = {
            g1.gene_id: {"harmonic": 3.0},
            g2.gene_id: {"harmonic": 1.5},
            g3.gene_id: {"harmonic": 0.3},
        }
        window = mgr._assemble(
            query="any",
            candidates=[g1, g2, g3],
            spliced_map={
                g1.gene_id: "a",
                g2.gene_id: "b",
                g3.gene_id: "c",
            },
        )
        ec = window.expressed_context
        # Each of the three symbols appears at least once
        assert "◆" in ec
        assert "⬦" in ec
    finally:
        mgr.close()


def test_assemble_gene_missing_from_scores_gets_none_fired():
    """A candidate without scores (e.g. co-activation bleed) still gets a
    header, just with `fired=none` and a neutral marker."""
    mgr = _make_manager(legibility_enabled=True)
    try:
        g1 = make_gene(content="scored", gene_id="scored0000000000")
        g2 = make_gene(content="unscored", gene_id="unscored00000000")
        mgr.genome.last_query_scores = {g1.gene_id: 5.0}   # g2 absent
        mgr.genome.last_tier_contributions = {
            g1.gene_id: {"harmonic": 2.0},
        }
        window = mgr._assemble(
            query="any",
            candidates=[g1, g2],
            spliced_map={g1.gene_id: "a", g2.gene_id: "b"},
        )
        ec = window.expressed_context
        # Unscored gene should still appear (truncated to 12-char prefix),
        # with `fired=none`
        assert "gene=unscored0000" in ec        # 12-char truncation
        assert "fired=none" in ec
    finally:
        mgr.close()

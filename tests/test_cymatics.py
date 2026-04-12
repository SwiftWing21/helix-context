"""Gate 0 — Cymatics tests (no model calls, pure CPU math).

Tests frequency spectrum construction, resonance scoring,
interference-based splice, harmonic co-activation, and Q-factor mapping.
"""

import math
import pytest

from helix_context.cymatics import (
    N_BINS,
    MATH_BACKEND,
    term_to_frequency,
    build_spectrum,
    query_spectrum,
    gene_spectrum,
    cached_gene_spectrum,
    clear_spectrum_cache,
    resonance_score,
    resonance_rank,
    interference_splice,
    harmonic_weight,
    compute_harmonic_weights,
    aggressiveness_to_peak_width,
    cymatics_info,
)

from tests.conftest import make_gene


# ── Frequency Space ──────────────────────────────────────────────


class TestTermToFrequency:
    def test_deterministic(self):
        """Same term always maps to the same frequency bin."""
        assert term_to_frequency("auth") == term_to_frequency("auth")

    def test_different_terms_different_bins(self):
        """Different terms should generally map to different bins."""
        f1 = term_to_frequency("authentication")
        f2 = term_to_frequency("database")
        # Could theoretically collide, but extremely unlikely
        assert f1 != f2

    def test_case_insensitive(self):
        """Term mapping should be case-insensitive."""
        assert term_to_frequency("Auth") == term_to_frequency("auth")

    def test_range(self):
        """Frequency must be within [0, N_BINS)."""
        for term in ["hello", "world", "test", "αβγ", "🧬"]:
            f = term_to_frequency(term)
            assert 0 <= f < N_BINS


class TestBuildSpectrum:
    def test_empty_terms(self):
        spec = build_spectrum([])
        assert len(spec) == N_BINS
        assert all(v == 0.0 for v in spec)

    def test_single_term_creates_peak(self):
        spec = build_spectrum(["auth"])
        assert len(spec) == N_BINS
        # Should have a non-zero peak somewhere
        assert max(spec) > 0.0

    def test_peak_at_correct_bin(self):
        freq = term_to_frequency("database")
        spec = build_spectrum(["database"], peak_width=1.0)
        # The maximum should be at or very near the hashed bin
        peak_bin = spec.index(max(spec))
        assert abs(peak_bin - freq) <= 1

    def test_weights_scale_amplitude(self):
        spec_full = build_spectrum(["auth"], weights=[1.0])
        spec_half = build_spectrum(["auth"], weights=[0.5])
        # Half-weight should produce half-amplitude peak
        assert max(spec_half) == pytest.approx(max(spec_full) * 0.5, rel=0.01)

    def test_decay_damps_amplitude(self):
        spec_fresh = build_spectrum(["auth"], decay=1.0)
        spec_stale = build_spectrum(["auth"], decay=0.5)
        assert max(spec_stale) == pytest.approx(max(spec_fresh) * 0.5, rel=0.01)

    def test_constructive_interference(self):
        """Two terms at the same word should produce higher peak than one."""
        spec_one = build_spectrum(["auth"])
        spec_two = build_spectrum(["auth", "auth"])
        assert max(spec_two) > max(spec_one)

    def test_spectrum_length(self):
        spec = build_spectrum(["a", "b", "c"])
        assert len(spec) == N_BINS

    def test_narrow_peak_width(self):
        """Narrow peaks should have less spectral energy spread."""
        spec_narrow = build_spectrum(["auth"], peak_width=1.0)
        spec_broad = build_spectrum(["auth"], peak_width=5.0)
        # Both have same max (1.0 at center), but broad has more total energy
        energy_narrow = sum(v * v for v in spec_narrow)
        energy_broad = sum(v * v for v in spec_broad)
        assert energy_narrow < energy_broad


class TestQuerySpectrum:
    def test_basic_query(self):
        spec = query_spectrum("How does authentication work?")
        assert len(spec) == N_BINS
        assert max(spec) > 0.0

    def test_synonym_expansion(self):
        """Synonyms should add harmonic peaks at half amplitude."""
        syn_map = {"auth": ["jwt", "login", "security"]}
        spec_no_syn = query_spectrum("auth", synonym_map=None)
        spec_with_syn = query_spectrum("auth", synonym_map=syn_map)
        # Synonym expansion should increase total spectral energy
        energy_no_syn = sum(v * v for v in spec_no_syn)
        energy_with_syn = sum(v * v for v in spec_with_syn)
        assert energy_with_syn > energy_no_syn

    def test_empty_query(self):
        spec = query_spectrum("")
        assert all(v == 0.0 for v in spec)


class TestGeneSpectrum:
    def test_gene_with_promoter_tags(self):
        gene = make_gene("auth code", domains=["auth", "security"], entities=["JWT"])
        spec = gene_spectrum(gene)
        assert len(spec) == N_BINS
        assert max(spec) > 0.0

    def test_gene_without_tags_uses_codons(self):
        gene = make_gene("some content", domains=[], entities=[])
        # make_gene provides default codons ["chunk_0", "chunk_1", "chunk_2"]
        spec = gene_spectrum(gene)
        assert max(spec) > 0.0

    def test_cached_spectrum(self):
        clear_spectrum_cache()
        gene = make_gene("auth code", domains=["auth"])
        s1 = cached_gene_spectrum(gene)
        s2 = cached_gene_spectrum(gene)
        assert s1 == s2


# ── Resonance Scoring ────────────────────────────────────────────


class TestResonanceScore:
    def test_identical_spectra(self):
        spec = build_spectrum(["auth", "security"])
        score = resonance_score(spec, spec)
        assert score == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_spectra(self):
        """Completely different terms should have low resonance."""
        spec_a = build_spectrum(["auth"], peak_width=0.5)
        spec_b = build_spectrum(["fluid_dynamics"], peak_width=0.5)
        score = resonance_score(spec_a, spec_b)
        # Very narrow peaks at different frequencies → near zero
        assert score < 0.3

    def test_similar_spectra(self):
        """Overlapping terms should have high resonance."""
        spec_a = build_spectrum(["auth", "security", "jwt"])
        spec_b = build_spectrum(["auth", "security", "token"])
        score = resonance_score(spec_a, spec_b)
        assert score > 0.5

    def test_zero_spectrum(self):
        zero = [0.0] * N_BINS
        spec = build_spectrum(["auth"])
        assert resonance_score(zero, spec) == 0.0
        assert resonance_score(spec, zero) == 0.0

    def test_symmetry(self):
        spec_a = build_spectrum(["auth"])
        spec_b = build_spectrum(["database"])
        assert resonance_score(spec_a, spec_b) == pytest.approx(
            resonance_score(spec_b, spec_a), abs=0.001
        )

    def test_range(self):
        spec_a = build_spectrum(["a", "b", "c"])
        spec_b = build_spectrum(["d", "e", "f"])
        score = resonance_score(spec_a, spec_b)
        assert 0.0 <= score <= 1.0


class TestResonanceRank:
    def test_ranks_by_relevance(self):
        auth_gene = make_gene("auth code", domains=["auth", "security"], entities=["JWT"])
        db_gene = make_gene("database code", domains=["database", "sql"], entities=["PostgreSQL"])
        api_gene = make_gene("api code", domains=["api", "auth"], entities=["endpoint"])

        ranked = resonance_rank("authentication security", [auth_gene, db_gene, api_gene], k=2)
        assert len(ranked) == 2
        # Auth gene should rank first (most resonant with "authentication security")
        assert ranked[0].gene_id == auth_gene.gene_id

    def test_returns_all_when_under_k(self):
        gene = make_gene("test", domains=["test"])
        ranked = resonance_rank("test query", [gene], k=5)
        assert len(ranked) == 1

    def test_empty_candidates(self):
        assert resonance_rank("query", [], k=5) == []

    def test_lost_in_the_middle_guard(self):
        """If too few candidates score well, pad with remaining."""
        genes = [make_gene(f"gene_{i}", domains=[f"unrelated_{i}"]) for i in range(10)]
        ranked = resonance_rank("very_specific_query_xyz", genes, k=5)
        # Should still return up to k genes even if resonance is low
        assert len(ranked) >= 1


# ── Interference Splice ──────────────────────────────────────────


class TestInterferenceSplice:
    def test_basic_splice(self):
        gene = make_gene(
            "auth and database code",
            domains=["auth", "database"],
        )
        # Override codons with meaningful labels
        gene.codons = ["authentication_flow", "database_schema", "error_handling"]

        result = interference_splice("authentication security", [gene])
        assert gene.gene_id in result
        assert isinstance(result[gene.gene_id], str)

    def test_empty_genes(self):
        assert interference_splice("query", []) == {}

    def test_fix2_empty_splice_guard(self):
        """If no codons resonate, keep first min_codons_kept."""
        gene = make_gene("unrelated", domains=["zzz_unrelated"])
        gene.codons = ["totally_irrelevant_a", "totally_irrelevant_b", "totally_irrelevant_c"]

        result = interference_splice(
            "authentication security", [gene],
            splice_aggressiveness=0.9,  # Very aggressive → high threshold
            min_codons_kept=2,
        )
        # Should still have content (either kept codons or complement)
        assert gene.gene_id in result
        assert len(result[gene.gene_id]) > 0

    def test_low_aggressiveness_keeps_more(self):
        gene = make_gene("mixed content", domains=["auth"])
        gene.codons = ["auth_check", "logging", "database_query", "error_handling"]

        result_low = interference_splice("auth", [gene], splice_aggressiveness=0.1)
        result_high = interference_splice("auth", [gene], splice_aggressiveness=0.9)

        # Low aggressiveness should keep more codons (lower threshold)
        text_low = result_low[gene.gene_id]
        text_high = result_high[gene.gene_id]
        assert text_low.count("|") >= text_high.count("|") or len(text_low) >= len(text_high)

    def test_gene_without_codons_uses_complement(self):
        gene = make_gene("content", domains=["test"])
        gene.codons = []

        result = interference_splice("test", [gene])
        assert gene.gene_id in result
        # Should fall back to complement
        assert "Summary of:" in result[gene.gene_id]

    def test_all_genes_get_result(self):
        genes = [
            make_gene(f"gene_{i}", domains=[f"domain_{i}"])
            for i in range(5)
        ]
        result = interference_splice("test query", genes)
        for gene in genes:
            assert gene.gene_id in result


# ── Harmonic Co-activation ───────────────────────────────────────


class TestHarmonicWeight:
    def test_identical_genes_max_weight(self):
        gene = make_gene("auth code", domains=["auth", "security"])
        w = harmonic_weight(gene, gene)
        assert w == pytest.approx(1.0, abs=0.001)

    def test_different_genes_lower_weight(self):
        gene_a = make_gene("auth code", domains=["auth", "security"])
        gene_b = make_gene("database code", domains=["database", "sql"])
        w = harmonic_weight(gene_a, gene_b)
        assert w < 1.0

    def test_compute_pairwise(self):
        genes = [
            make_gene("auth", domains=["auth"]),
            make_gene("db", domains=["database"]),
            make_gene("api", domains=["api"]),
        ]
        weights = compute_harmonic_weights(genes)
        assert isinstance(weights, list)
        for gene_a, gene_b, w in weights:
            assert isinstance(w, float)
            assert 0.0 <= w <= 1.0

    def test_single_gene_no_weights(self):
        gene = make_gene("solo", domains=["solo"])
        assert compute_harmonic_weights([gene]) == []


# ── Q-factor Mapping ─────────────────────────────────────────────


class TestQFactorMapping:
    def test_zero_aggressiveness_broad(self):
        width = aggressiveness_to_peak_width(0.0)
        assert width == 2.0  # max width in useful zone

    def test_max_aggressiveness_narrow(self):
        width = aggressiveness_to_peak_width(1.0)
        assert width == 0.5  # sharp peaks, high selectivity

    def test_mid_aggressiveness(self):
        width = aggressiveness_to_peak_width(0.5)
        assert 1.0 <= width <= 1.5  # sweet spot for discrimination

    def test_monotonically_decreasing(self):
        widths = [aggressiveness_to_peak_width(a / 10) for a in range(11)]
        for i in range(len(widths) - 1):
            assert widths[i] >= widths[i + 1]

    def test_never_below_floor(self):
        assert aggressiveness_to_peak_width(1.0) >= 0.5
        assert aggressiveness_to_peak_width(1.5) >= 0.5  # beyond range still safe


# ── Diagnostics ──────────────────────────────────────────────────


class TestCymaticsInfo:
    def test_info_returns_dict(self):
        info = cymatics_info()
        assert isinstance(info, dict)
        assert "math_backend" in info
        assert "n_bins" in info
        assert info["n_bins"] == 256

    def test_math_backend_valid(self):
        assert MATH_BACKEND in ("numpy", "python")

"""Tests for cymatics flux integral — adaptive bin weighting."""

import math
import pytest

from helix_context.cymatics import (
    N_BINS,
    build_weight_vector,
    flux_score,
    resonance_score,
    build_spectrum,
    term_to_frequency,
    resonance_rank,
)
from tests.conftest import make_gene


class TestBuildWeightVector:
    def test_shape(self):
        w = build_weight_vector("what port does helix use")
        assert len(w) == N_BINS

    def test_baseline_when_empty(self):
        w = build_weight_vector("")
        assert all(v == pytest.approx(0.8) for v in w)

    def test_amplified_near_query_terms(self):
        w = build_weight_vector("authentication database")
        freq_auth = term_to_frequency("authentication")
        freq_db = term_to_frequency("database")
        # Bins at query term frequencies should be above baseline
        assert w[freq_auth] > 0.8
        assert w[freq_db] > 0.8

    def test_amplification_is_gaussian(self):
        """Bins near but not at query freq should be boosted, decaying with distance."""
        w = build_weight_vector("helix", peak_width=3.0)
        freq = term_to_frequency("helix")
        # At center: maximum boost
        center_val = w[freq]
        # 1 bin away: still boosted but less
        if freq + 1 < N_BINS:
            assert w[freq + 1] < center_val
            assert w[freq + 1] > 0.8  # still above baseline
        # Far away: baseline
        far = (freq + 100) % N_BINS
        assert w[far] == pytest.approx(0.8, abs=0.05)

    def test_synonym_boost(self):
        syns = {"auth": ["jwt", "login", "security"]}
        w = build_weight_vector("auth", synonym_map=syns)
        freq_jwt = term_to_frequency("jwt")
        # Synonym bin should be above baseline
        assert w[freq_jwt] > 0.8

    def test_synonym_boost_less_than_primary(self):
        syns = {"auth": ["jwt"]}
        w = build_weight_vector("auth", synonym_map=syns)
        freq_auth = term_to_frequency("auth")
        freq_jwt = term_to_frequency("jwt")
        # Primary term boost > synonym boost (1.5 vs 1.2)
        assert w[freq_auth] >= w[freq_jwt]


class TestFluxScore:
    def test_uniform_weights_equals_resonance(self):
        """flux_score with uniform weights should equal resonance_score."""
        a = build_spectrum(["auth", "database"])
        b = build_spectrum(["auth", "cache"])
        uniform = [1.0] * N_BINS
        fs = flux_score(a, b, uniform)
        rs = resonance_score(a, b)
        assert fs == pytest.approx(rs, abs=0.001)

    def test_zero_spectrum(self):
        a = [0.0] * N_BINS
        b = build_spectrum(["test"])
        w = [1.0] * N_BINS
        assert flux_score(a, b, w) == 0.0

    def test_amplified_bins_boost_relevant(self):
        """Genes sharing query-relevant bins should score higher with flux."""
        query_terms = ["authentication"]
        gene_relevant = build_spectrum(["authentication", "security"])
        gene_irrelevant = build_spectrum(["cooking", "recipes"])
        query_spec = build_spectrum(query_terms)
        weights = build_weight_vector("authentication")

        # Both scoring methods
        flat_relevant = resonance_score(query_spec, gene_relevant)
        flat_irrelevant = resonance_score(query_spec, gene_irrelevant)
        flux_relevant = flux_score(query_spec, gene_relevant, weights)
        flux_irrelevant = flux_score(query_spec, gene_irrelevant, weights)

        # Flux should widen the gap between relevant and irrelevant
        flat_gap = flat_relevant - flat_irrelevant
        flux_gap = flux_relevant - flux_irrelevant
        assert flux_gap >= flat_gap

    def test_range(self):
        a = build_spectrum(["auth"])
        b = build_spectrum(["auth"])
        w = build_weight_vector("auth")
        score = flux_score(a, b, w)
        assert 0.0 <= score <= 1.0

    def test_symmetry(self):
        a = build_spectrum(["auth", "db"])
        b = build_spectrum(["cache", "db"])
        w = build_weight_vector("database query")
        assert flux_score(a, b, w) == pytest.approx(flux_score(b, a, w))


class TestResonanceRankFlux:
    def test_flux_default_on(self):
        """resonance_rank uses flux by default (use_flux=True)."""
        genes = [
            make_gene(domains=["authentication", "security"], gene_id="auth1"),
            make_gene(domains=["cooking", "recipes"], gene_id="cook1"),
            make_gene(domains=["database", "auth"], gene_id="db1"),
        ]
        result = resonance_rank("authentication login", genes, k=2)
        # Should return 2 genes, auth-related ones preferred
        assert len(result) == 2

    def test_flux_off_fallback(self):
        """use_flux=False falls back to flat resonance_score."""
        genes = [
            make_gene(domains=["authentication"], gene_id="a1"),
            make_gene(domains=["cooking"], gene_id="c1"),
        ]
        result = resonance_rank("auth", genes, k=1, use_flux=False)
        assert len(result) == 1

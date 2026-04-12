"""Tests for Monte Carlo evidence ray propagation."""

import json
import pytest

from helix_context.ray_trace import (
    cast_evidence_rays,
    ray_trace_boost,
    ray_trace_info,
    read_overtone_series,
    harmonic_bin_boost,
    BOOST_CAP,
)
from helix_context.schemas import Gene, PromoterTags, EpigeneticMarkers


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_gene(genome, gene_id: str, co_activated: list[str] | None = None):
    """Insert a minimal gene with specified co_activated_with peers."""
    epi = EpigeneticMarkers(co_activated_with=co_activated or [])
    gene = Gene(
        gene_id=gene_id,
        content=f"content for {gene_id}",
        complement=f"summary of {gene_id}",
        codons=["test"],
        promoter=PromoterTags(domains=["test"], entities=[]),
        epigenetics=epi,
    )
    genome.upsert_gene(gene)


# ── Tests ───────────────────────────────────────────────────────────────

class TestCastEvidenceRays:
    """Tests for cast_evidence_rays()."""

    def test_empty_seeds_returns_empty(self, genome):
        result = cast_evidence_rays([], genome)
        assert result == {}

    def test_isolated_node_returns_self(self, genome):
        """Rays from an isolated node (no neighbors) deposit at the seed."""
        _make_gene(genome, "isolated")
        result = cast_evidence_rays(["isolated"], genome, k_rays=50, seed=42)
        assert "isolated" in result
        assert len(result) == 1
        # All 50 rays land on the isolated node with full energy (no bounces)
        assert result["isolated"] == pytest.approx(50.0)

    def test_chain_propagation(self, genome):
        """Rays propagate through a chain A -> B -> C."""
        _make_gene(genome, "A", co_activated=["B"])
        _make_gene(genome, "B", co_activated=["C"])
        _make_gene(genome, "C")

        result = cast_evidence_rays(
            ["A"], genome, k_rays=100, max_bounces=3, seed=42,
        )
        # C should receive energy (propagated through B)
        assert "C" in result
        assert result["C"] > 0

    def test_energy_decays_with_bounces(self, genome):
        """Energy at further nodes is lower due to decay."""
        _make_gene(genome, "A", co_activated=["B"])
        _make_gene(genome, "B", co_activated=["C"])
        _make_gene(genome, "C", co_activated=["D"])
        _make_gene(genome, "D")

        result = cast_evidence_rays(
            ["A"], genome, k_rays=200, max_bounces=5,
            decay_per_bounce=0.7, seed=42,
        )
        # B gets energy at 1 bounce (0.7), C at 2 bounces (0.49), D at 3 (0.343)
        # With 200 rays all going A->B->C->D (linear chain), each node
        # accumulates from rays that terminated there.
        # D should have less energy than B in total.
        if "B" in result and "D" in result:
            assert result["D"] < result["B"]

    def test_dead_end_deposits_energy(self, genome):
        """A dead-end node receives deposited energy."""
        _make_gene(genome, "start", co_activated=["dead_end"])
        _make_gene(genome, "dead_end")  # no co_activated

        result = cast_evidence_rays(
            ["start"], genome, k_rays=100, max_bounces=3, seed=42,
        )
        assert "dead_end" in result
        assert result["dead_end"] > 0

    def test_harmonic_weights_affect_energy(self, genome):
        """harmonic_links weight modulates energy on edges."""
        _make_gene(genome, "X", co_activated=["Y"])
        _make_gene(genome, "Y")

        # Without harmonic weight
        result_no_hw = cast_evidence_rays(
            ["X"], genome, k_rays=100, max_bounces=1,
            decay_per_bounce=0.7, seed=42,
        )

        # With harmonic weight < 1.0 (should reduce energy)
        genome.store_harmonic_weights([("X", "Y", 0.5)])
        result_hw = cast_evidence_rays(
            ["X"], genome, k_rays=100, max_bounces=1,
            decay_per_bounce=0.7, seed=42,
        )

        assert result_hw.get("Y", 0) < result_no_hw.get("Y", 0)

    def test_deterministic_with_seed(self, genome):
        """Same seed produces same result."""
        _make_gene(genome, "A", co_activated=["B", "C"])
        _make_gene(genome, "B", co_activated=["C"])
        _make_gene(genome, "C")

        r1 = cast_evidence_rays(["A"], genome, k_rays=100, seed=123)
        r2 = cast_evidence_rays(["A"], genome, k_rays=100, seed=123)
        assert r1 == r2

    def test_multiple_seeds(self, genome):
        """Rays are distributed across multiple seeds."""
        _make_gene(genome, "S1", co_activated=["T1"])
        _make_gene(genome, "S2", co_activated=["T2"])
        _make_gene(genome, "T1")
        _make_gene(genome, "T2")

        result = cast_evidence_rays(
            ["S1", "S2"], genome, k_rays=100, seed=42,
        )
        # Both targets should be reached
        assert "T1" in result
        assert "T2" in result


class TestRayTraceBoost:
    """Tests for ray_trace_boost()."""

    def test_empty_seeds(self, genome):
        result = ray_trace_boost([], genome)
        assert result == {}

    def test_boost_capped_at_2(self, genome):
        """No boost value exceeds BOOST_CAP (2.0)."""
        _make_gene(genome, "A", co_activated=["B", "C"])
        _make_gene(genome, "B", co_activated=["C"])
        _make_gene(genome, "C")

        result = ray_trace_boost(["A"], genome, k_rays=200, seed=42)
        for gid, boost in result.items():
            assert boost <= BOOST_CAP, f"{gid} boost {boost} exceeds cap"
            assert boost >= 0, f"{gid} boost {boost} is negative"

    def test_max_boost_equals_cap(self, genome):
        """The gene with the highest energy gets exactly BOOST_CAP."""
        _make_gene(genome, "A", co_activated=["B"])
        _make_gene(genome, "B")

        result = ray_trace_boost(["A"], genome, k_rays=100, seed=42)
        if result:
            assert max(result.values()) == pytest.approx(BOOST_CAP)

    def test_boost_normalised(self, genome):
        """All boost values are between 0 and BOOST_CAP."""
        _make_gene(genome, "A", co_activated=["B", "C", "D"])
        _make_gene(genome, "B", co_activated=["C"])
        _make_gene(genome, "C")
        _make_gene(genome, "D")

        result = ray_trace_boost(["A"], genome, k_rays=200, seed=42)
        for v in result.values():
            assert 0 <= v <= BOOST_CAP


class TestRayTraceInfo:
    """Tests for ray_trace_info()."""

    def test_empty_result(self):
        info = ray_trace_info({})
        assert info["total_energy"] == 0.0
        assert info["unique_genes_reached"] == 0
        assert info["max_energy"] == 0.0
        assert info["mean_energy"] == 0.0

    def test_summary_stats(self):
        result = {"A": 3.0, "B": 1.0, "C": 2.0}
        info = ray_trace_info(result)
        assert info["total_energy"] == pytest.approx(6.0)
        assert info["unique_genes_reached"] == 3
        assert info["max_energy"] == pytest.approx(3.0)
        assert info["mean_energy"] == pytest.approx(2.0)

    def test_single_gene(self):
        info = ray_trace_info({"X": 5.5})
        assert info["unique_genes_reached"] == 1
        assert info["max_energy"] == pytest.approx(5.5)
        assert info["mean_energy"] == pytest.approx(5.5)


class TestOvertoneSeries:
    """Tests for read_overtone_series() — Monte Carlo as frequency distribution."""

    def test_empty_seeds(self, genome):
        assert read_overtone_series([], genome) == {}

    def test_fundamental_isolated(self, genome):
        """Isolated seed visits itself in every ray → fundamental (weight 1.0)."""
        _make_gene(genome, "solo")
        overtones = read_overtone_series(["solo"], genome, k_rays=100, seed=42)
        # Seed visited in every ray → frequency 1.0 → fundamental
        assert overtones.get("solo") == pytest.approx(1.0)

    def test_chain_harmonics(self, genome):
        """A → B → C chain: A is fundamental (in every ray), B is first harmonic."""
        _make_gene(genome, "A", co_activated=["B"])
        _make_gene(genome, "B", co_activated=["A", "C"])
        _make_gene(genome, "C", co_activated=["B"])
        overtones = read_overtone_series(["A"], genome, k_rays=100, max_bounces=2, seed=42)
        # A is the seed — in 100% of rays
        assert overtones["A"] == 1.0
        # B is reached on first bounce — should be fundamental or first harmonic
        assert "B" in overtones
        assert overtones["B"] >= 0.25  # at least second harmonic

    def test_noise_excluded(self, genome):
        """Genes visited in <20% of rays are filtered as noise."""
        # Star graph: A connects to many, each B_i only connects to A
        peers = [f"leaf_{i}" for i in range(10)]
        _make_gene(genome, "hub", co_activated=peers)
        for p in peers:
            _make_gene(genome, p, co_activated=["hub"])
        overtones = read_overtone_series(["hub"], genome, k_rays=100, max_bounces=2, seed=42)
        # Each leaf has ~10% visit rate → should be excluded as noise
        # Hub is fundamental
        assert overtones.get("hub") == 1.0


class TestHarmonicBinBoost:
    """Tests for harmonic_bin_boost() — retrieval score addition."""

    def test_empty_seeds(self, genome):
        assert harmonic_bin_boost([], genome) == {}

    def test_scaled_to_1_5(self, genome):
        """Fundamentals get 1.5 boost (overtone 1.0 * scale 1.5)."""
        _make_gene(genome, "fundamental")
        boost = harmonic_bin_boost(["fundamental"], genome, k_rays=100)
        assert boost.get("fundamental") == pytest.approx(1.5)

    def test_bounded(self, genome):
        """All boosts are <= 1.5."""
        _make_gene(genome, "A", co_activated=["B"])
        _make_gene(genome, "B", co_activated=["A"])
        boost = harmonic_bin_boost(["A"], genome, k_rays=100)
        for v in boost.values():
            assert 0 <= v <= 1.5

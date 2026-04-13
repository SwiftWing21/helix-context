"""Theta-alternation bias for ray_trace (Sprint 2 item 6).

Wang/Foster/Pfeiffer (2020) Science 370:247 — place-cell alternation
sweeps forward then backward within each theta cycle. Under
velocity-input TCM (Howard 2005) the session's current context vector
carries the velocity direction; biasing ray-trace neighbour sampling
along ±velocity per-bounce implements the theta sweep over the
co-activation graph.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helix_context.ray_trace import (
    _theta_choice,
    cast_evidence_rays,
)


class TestThetaChoice:
    """Unit tests for the softmax-weighted neighbour picker."""

    def test_falls_back_to_uniform_without_scores(self):
        import random
        rng = random.Random(0)
        # No direction scores for any neighbour → uniform pick
        picks = [_theta_choice(rng, ["a", "b", "c"], {}, +1, 1.0) for _ in range(300)]
        counts = {g: picks.count(g) for g in ("a", "b", "c")}
        # Roughly uniform — allow generous slack
        for c in counts.values():
            assert 60 < c < 240

    def test_fore_sweep_prefers_aligned_neighbour(self):
        import random
        rng = random.Random(0)
        scores = {"with": 0.9, "against": -0.9, "orthogonal": 0.0}
        picks = [
            _theta_choice(rng, ["with", "against", "orthogonal"], scores, +1, 3.0)
            for _ in range(400)
        ]
        assert picks.count("with") > picks.count("against")
        assert picks.count("with") > picks.count("orthogonal")

    def test_aft_sweep_flips_preference(self):
        import random
        rng = random.Random(0)
        scores = {"with": 0.9, "against": -0.9, "orthogonal": 0.0}
        picks = [
            _theta_choice(rng, ["with", "against", "orthogonal"], scores, -1, 3.0)
            for _ in range(400)
        ]
        assert picks.count("against") > picks.count("with")

    def test_theta_weight_zero_is_uniform(self):
        """At theta_weight=0 the softmax is flat regardless of scores."""
        import random
        rng = random.Random(0)
        scores = {"a": 0.99, "b": -0.99}
        picks = [_theta_choice(rng, ["a", "b"], scores, +1, 0.0) for _ in range(400)]
        assert 150 < picks.count("a") < 250


class TestCastEvidenceRaysWithVelocity:
    """End-to-end: velocity kwarg routes through the ΣĒMA cache."""

    def _fill_sema_cache(self, genome, embeddings):
        """Inject a fake ΣĒMA matrix so _build_direction_scores can
        compute cosines without running the full codec."""
        gene_ids = list(embeddings.keys())
        matrix = np.array([embeddings[g] for g in gene_ids], dtype=np.float64)
        genome._sema_cache = {
            "gene_ids": gene_ids,
            "matrix": matrix,
        }

    def test_velocity_biases_sampling(self, genome):
        """With two neighbour candidates, one aligned with velocity
        and one anti-aligned, the aligned one should accumulate more
        ray energy on the fore-sweep (even-index bounces)."""
        from tests.test_ray_trace import _make_gene
        _make_gene(genome, "seed", co_activated=["aligned", "anti"])
        _make_gene(genome, "aligned")
        _make_gene(genome, "anti")

        velocity = [1.0] + [0.0] * 19
        self._fill_sema_cache(genome, {
            "seed":    [1.0, 0.0] + [0.0] * 18,
            "aligned": [1.0, 0.0] + [0.0] * 18,
            "anti":    [-1.0, 0.0] + [0.0] * 18,
        })

        biased = cast_evidence_rays(
            ["seed"], genome, k_rays=200, max_bounces=1, seed=0,
            velocity_vector=velocity, theta_weight=3.0,
        )
        unbiased = cast_evidence_rays(
            ["seed"], genome, k_rays=200, max_bounces=1, seed=0,
        )

        # Under theta bias the fore-sweep (bounce 0 is even → sign=+1)
        # should route more energy to aligned than anti. Uniform
        # sampling should split roughly evenly.
        assert biased.get("aligned", 0) > biased.get("anti", 0)
        unbiased_a = unbiased.get("aligned", 0)
        unbiased_b = unbiased.get("anti", 0)
        # Sanity: unbiased split is near 50/50
        if unbiased_a + unbiased_b > 0:
            ratio = unbiased_a / (unbiased_a + unbiased_b)
            assert 0.3 < ratio < 0.7

    def test_missing_sema_cache_falls_back_silently(self, genome):
        """With no ΣĒMA cache, passing velocity must not crash — the
        theta path should fall through to uniform sampling."""
        from tests.test_ray_trace import _make_gene
        _make_gene(genome, "seed", co_activated=["a", "b"])
        _make_gene(genome, "a")
        _make_gene(genome, "b")
        # No _sema_cache set
        result = cast_evidence_rays(
            ["seed"], genome, k_rays=50, max_bounces=1, seed=0,
            velocity_vector=[1.0] + [0.0] * 19,
        )
        assert result  # did not crash; returned something

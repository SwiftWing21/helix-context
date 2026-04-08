"""
Integration tests -- ScoreRift CD probe and resolution pipeline.
No external services needed (uses in-memory genome + mock backend).
"""

import math
import pytest

from helix_context.integrations.scorerift import (
    CDSignal,
    GenomeHealthProbe,
    cd_signal,
    resolution_to_gene,
)


class TestCDSignal:
    def test_aligned(self):
        sig = cd_signal(0.90, 0.88)
        assert sig.status == "aligned"
        assert sig.delta_epsilon == pytest.approx(0.02, abs=0.001)
        assert sig.ellipticity > 0.99

    def test_diverged(self):
        sig = cd_signal(0.90, 0.70)
        assert sig.status == "diverged"
        assert sig.delta_epsilon == pytest.approx(0.20, abs=0.001)
        assert 0.5 < sig.ellipticity < 0.95

    def test_denatured(self):
        sig = cd_signal(0.90, 0.40)
        assert sig.status == "denatured"
        assert sig.delta_epsilon == pytest.approx(0.50, abs=0.001)
        assert sig.ellipticity < 0.5

    def test_unmeasured(self):
        sig = cd_signal(0.90, None)
        assert sig.status == "unmeasured"
        assert sig.delta_epsilon == 0.0
        assert sig.ellipticity == 1.0

    def test_perfect_alignment(self):
        sig = cd_signal(0.85, 0.85)
        assert sig.status == "aligned"
        assert sig.delta_epsilon == 0.0
        assert sig.ellipticity == 1.0

    def test_custom_thresholds(self):
        # With a tight threshold, 0.10 gap should be diverged
        sig = cd_signal(0.90, 0.80, divergence_threshold=0.05)
        assert sig.status == "diverged"

        # With a loose threshold, 0.10 gap should be aligned
        sig = cd_signal(0.90, 0.80, divergence_threshold=0.20)
        assert sig.status == "aligned"

    def test_ellipticity_is_symmetric(self):
        sig1 = cd_signal(0.90, 0.70)
        sig2 = cd_signal(0.70, 0.90)
        assert sig1.ellipticity == pytest.approx(sig2.ellipticity)
        assert sig1.delta_epsilon == pytest.approx(sig2.delta_epsilon)

    def test_ellipticity_monotonically_decreases(self):
        """Larger divergence should always produce lower ellipticity."""
        prev = 1.0
        for gap in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]:
            sig = cd_signal(1.0, 1.0 - gap)
            assert sig.ellipticity <= prev
            prev = sig.ellipticity

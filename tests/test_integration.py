"""
Integration tests -- ScoreRift CD probe and resolution pipeline.
No external services needed (uses in-memory genome + mock backend).
"""

import math
from unittest.mock import MagicMock

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


class TestCheckRelevanceShape:
    """Pin the /context response shape that check_relevance consumes.

    Regression guard for the dict-vs-list bug: /context returns a dict
    with keys name/content/context_health, not a list. If the endpoint
    ever reverts to a list, or if check_relevance re-introduces list
    iteration, these tests break loudly.
    """

    def _probe_with_mock(self, response_json, status_code=200):
        probe = GenomeHealthProbe.__new__(GenomeHealthProbe)
        probe.helix_url = "http://mock"
        probe.timeout = 5.0
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = response_json
        mock_client.post.return_value = mock_resp
        probe._client = mock_client
        return probe

    def test_dict_response_with_rich_content_scores_high(self):
        payload = {
            "name": "Helix Context",
            "content": (
                "The 6-step pipeline: extract, express, re-rank, splice, "
                "assemble, replicate. The genome is stored in SQLite and "
                "encodes genes with promoter tags, ΣĒMA vectors, and "
                "chromatin state. Retrieval uses 12 signals including "
                "path_key_index, promoter tags, FTS5, SPLADE, SEMA cold, "
                "harmonic_links, cymatics resonance, TCM drift, and a "
                "party_id octave gate for multi-tenant scoping."
            ),
            "context_health": {"open_ratio": 0.62, "total_genes": 128},
        }
        probe = self._probe_with_mock(payload)
        score, detail = probe.check_relevance("How does the system work?")
        assert score > 0.0
        assert score == 1.0  # >500 chars → max score
        assert detail["response_length"] == len(payload["content"])
        assert detail["query"] == "How does the system work?"

    def test_dict_response_with_short_content_scores_mid(self):
        payload = {
            "name": "Helix",
            "content": "Short but non-empty content block.",
            "context_health": {},
        }
        probe = self._probe_with_mock(payload)
        score, _ = probe.check_relevance("anything")
        assert 0.0 < score <= 0.4

    def test_no_relevant_context_phrase_returns_low_score(self):
        payload = {
            "name": "Helix",
            "content": "No relevant context found for your query.",
            "context_health": {},
        }
        probe = self._probe_with_mock(payload)
        score, detail = probe.check_relevance("obscure query")
        assert score == pytest.approx(0.2)
        assert detail["reason"] == "no matching genes"

    def test_list_response_is_treated_as_empty(self):
        # Guard the regression: the old code expected a list and silently
        # pulled data[0].get("content") — now we require a dict.
        probe = self._probe_with_mock([{"content": "should not count"}])
        score, detail = probe.check_relevance("q")
        assert score == 0.0
        assert detail.get("reason") == "empty response"

    def test_empty_dict_response_scores_zero(self):
        probe = self._probe_with_mock({})
        score, detail = probe.check_relevance("q")
        assert score == 0.0
        assert detail.get("reason") == "empty response"

    def test_http_error_returns_zero_score(self):
        probe = self._probe_with_mock({}, status_code=500)
        score, detail = probe.check_relevance("q")
        assert score == 0.0
        assert "HTTP 500" in detail.get("error", "")

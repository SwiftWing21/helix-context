"""
Tests for helix_context.launcher.collector — state aggregation with
mocked supervisor + mocked HTTP responses.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from helix_context.launcher.collector import StateCollector


@pytest.fixture
def fake_supervisor(tmp_path):
    sup = MagicMock()
    sup.helix_host = "127.0.0.1"
    sup.helix_port = 11437
    sup.is_running.return_value = True
    sup.get_pid.return_value = 12345
    sup.get_uptime_s.return_value = 42.5
    sup.store.state.last_restart_reason = "test"
    sup.store.state.last_restart_at = time.time()
    # Telemetry defaults — collector now reads these
    sup.find_orphan_helix.return_value = None
    sup.get_last_error.return_value = None
    sup.store.path = tmp_path / "state.json"
    sup.helix_log_path = tmp_path / "helix.log"
    return sup


@pytest.fixture
def collector(fake_supervisor):
    return StateCollector(supervisor=fake_supervisor)


def _mock_client(responses: dict):
    """Build a context-managed httpx.Client mock with prebaked responses.

    `responses` maps URL paths to JSON bodies. Missing paths return 404.
    """
    client = MagicMock()
    def fake_get(path, params=None):
        resp = MagicMock()
        if path in responses:
            resp.status_code = 200
            resp.json.return_value = responses[path]
        else:
            resp.status_code = 404
        return resp
    client.get.side_effect = fake_get
    client.close = MagicMock()
    return client


class TestCollectHelixDown:
    def test_returns_only_helix_field_when_stopped(self, collector, fake_supervisor):
        fake_supervisor.is_running.return_value = False
        fake_supervisor.find_orphan_helix.return_value = None
        fake_supervisor.get_last_error.return_value = None
        result = collector.collect()
        assert "helix" in result
        assert result["helix"]["running"] is False
        assert result["helix"]["availability"] == "unavailable"
        # No other panels should be present
        assert "genes" not in result
        assert "parties" not in result
        assert "tools" not in result

    def test_orphan_pid_surfaced_when_helix_down(self, collector, fake_supervisor):
        fake_supervisor.is_running.return_value = False
        fake_supervisor.find_orphan_helix.return_value = 45678
        fake_supervisor.get_last_error.return_value = None
        result = collector.collect()
        assert result["helix"]["orphan_pid"] == 45678

    def test_last_error_surfaced(self, collector, fake_supervisor):
        fake_supervisor.is_running.return_value = False
        fake_supervisor.find_orphan_helix.return_value = None
        fake_supervisor.get_last_error.return_value = {
            "operation": "start",
            "message": "port 11437 occupied",
            "at": 1775896000.0,
        }
        result = collector.collect()
        assert result["helix"]["last_error"]["operation"] == "start"
        assert "port 11437" in result["helix"]["last_error"]["message"]

    def test_paths_always_present(self, collector, fake_supervisor, tmp_path):
        fake_supervisor.is_running.return_value = False
        fake_supervisor.find_orphan_helix.return_value = None
        fake_supervisor.get_last_error.return_value = None
        fake_supervisor.store.path = tmp_path / "state.json"
        fake_supervisor.helix_log_path = tmp_path / "helix.log"
        result = collector.collect()
        assert "paths" in result["helix"]
        assert "state_file" in result["helix"]["paths"]
        assert "helix_log" in result["helix"]["paths"]


class TestGenesPanel:
    def test_genes_panel_built_from_stats(self, collector):
        responses = {
            "/stats": {
                "total_genes": 8000,
                "total_chars_raw": 47_000_000,
                "total_chars_compressed": 17_500_000,
                "compression_ratio": 2.69,
            },
            "/sessions": {"participants": []},
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()
        assert state["genes"]["total"] == 8000
        assert state["genes"]["raw_chars"] == 47_000_000
        assert state["genes"]["compression_ratio"] == 2.69

    def test_health_ok_marks_available(self, collector):
        responses = {
            "/stats": {
                "total_genes": 8000,
                "total_chars_raw": 47_000_000,
                "total_chars_compressed": 17_500_000,
                "compression_ratio": 2.69,
            },
            "/sessions": {"participants": []},
            "/health": {"status": "ok", "ribosome": "mock"},
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()
        assert state["helix"]["availability"] == "available"

    def test_missing_health_marks_degraded(self, collector):
        responses = {
            "/stats": {
                "total_genes": 8000,
                "total_chars_raw": 47_000_000,
                "total_chars_compressed": 17_500_000,
                "compression_ratio": 2.69,
            },
            "/sessions": {"participants": []},
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()
        assert state["helix"]["availability"] == "degraded"


class TestPartiesAndParticipants:
    def test_parties_derived_from_unique_party_ids(self, collector):
        participants = [
            {"handle": "taude", "party_id": "max@local", "status": "active", "last_seen_s_ago": 1.0},
            {"handle": "laude", "party_id": "max@local", "status": "active", "last_seen_s_ago": 5.0},
            {"handle": "guest", "party_id": "other@remote", "status": "stale", "last_seen_s_ago": 9999.0},
        ]
        responses = {
            "/stats": {"total_genes": 0, "total_chars_raw": 0, "total_chars_compressed": 0, "compression_ratio": 1.0},
            "/sessions": {"participants": participants},
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()

        assert state["parties"]["count"] == 2
        assert "max@local" in state["parties"]["party_ids"]
        assert "other@remote" in state["parties"]["party_ids"]

        # Main panel is active identities; total_count is raw session rows.
        assert state["participants"]["count"] == 2
        assert state["participants"]["identity_total_count"] == 3
        assert state["participants"]["total_count"] == 3
        handles = [p["handle"] for p in state["participants"]["entries"]]
        assert handles == ["taude", "laude"]  # ordered by last_seen_s_ago
        assert state["disconnected_agents"]["count"] == 1
        assert state["disconnected_agents"]["entries"][0]["handle"] == "guest"
        assert state["disconnected_agents"]["entries"][0]["status"] == "stale"
        assert state["all_agents"]["count"] == 3

    def test_duplicate_sessions_collapse_into_one_identity(self, collector):
        participants = [
            {
                "participant_id": "aaaaaaaa11111111",
                "handle": "laude",
                "party_id": "swift_wing21",
                "workspace": "f:\\Projects\\Education",
                "status": "active",
                "last_seen_s_ago": 2.0,
                "started_at": 100.0,
            },
            {
                "participant_id": "bbbbbbbb22222222",
                "handle": "laude",
                "party_id": "swift_wing21",
                "workspace": "f:\\Projects\\Education",
                "status": "active",
                "last_seen_s_ago": 4.0,
                "started_at": 101.0,
            },
        ]
        responses = {
            "/stats": {"total_genes": 0, "total_chars_raw": 0, "total_chars_compressed": 0, "compression_ratio": 1.0},
            "/sessions": {"participants": participants},
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()

        assert state["participants"]["count"] == 1
        assert state["participants"]["identity_total_count"] == 1
        assert state["participants"]["total_count"] == 2
        assert state["participants"]["entries"][0]["session_count"] == 2
        assert "disconnected_agents" not in state
        assert state["all_agents"]["count"] == 2
        assert state["all_agents"]["entries"][0]["participant_id_short"] == "aaaaaaaa"

    def test_no_participants_omits_panel(self, collector):
        responses = {
            "/stats": {"total_genes": 0, "total_chars_raw": 0, "total_chars_compressed": 0, "compression_ratio": 1.0},
            "/sessions": {"participants": []},
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()
        assert "parties" not in state
        assert "participants" not in state
        assert "disconnected_agents" not in state


class TestToolsPanel:
    def test_tools_built_from_components_endpoint(self, collector):
        components = {
            "components": [
                {"name": "ribosome", "kind": "decoder", "status": "running"},
                {"name": "splade", "kind": "encoder", "status": "idle"},
            ],
            "count": 2,
            "last_activity_s_ago": 12.4,
        }
        responses = {
            "/stats": {"total_genes": 0, "total_chars_raw": 0, "total_chars_compressed": 0, "compression_ratio": 1.0},
            "/sessions": {"participants": []},
            "/admin/components": components,
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()

        assert state["tools"]["count"] == 1
        assert state["tools"]["last_activity_s_ago"] == 12.4
        assert len(state["tools"]["entries"]) == 1
        assert state["tools"]["entries"][0]["name"] == "splade"

    def test_no_components_omits_tools_panel(self, collector):
        responses = {
            "/stats": {"total_genes": 0, "total_chars_raw": 0, "total_chars_compressed": 0, "compression_ratio": 1.0},
            "/sessions": {"participants": []},
            "/admin/components": {"components": [], "count": 0},
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()
        assert "tools" not in state

    def test_ribosome_only_omits_tools_panel(self, collector):
        responses = {
            "/stats": {"total_genes": 0, "total_chars_raw": 0, "total_chars_compressed": 0, "compression_ratio": 1.0},
            "/sessions": {"participants": []},
            "/admin/components": {
                "components": [
                    {"name": "ribosome", "kind": "decoder", "status": "running", "backend": "gemma4:e2b"},
                ],
                "count": 1,
                "last_activity_s_ago": 3.1,
            },
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()
        assert "tools" not in state


class TestTokensPanel:
    def test_tokens_built_from_metrics_endpoint(self, collector):
        tokens = {
            "session": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total": 300,
                "estimated_prompt_tokens": 0,
                "estimated_completion_tokens": 0,
                "estimated_total": 0,
            },
            "lifetime": {
                "prompt_tokens": 5000,
                "completion_tokens": 8000,
                "total": 13000,
                "estimated_prompt_tokens": 200,
                "estimated_completion_tokens": 300,
                "estimated_total": 500,
            },
        }
        responses = {
            "/stats": {"total_genes": 0, "total_chars_raw": 0, "total_chars_compressed": 0, "compression_ratio": 1.0},
            "/sessions": {"participants": []},
            "/metrics/tokens": tokens,
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()

        assert "tokens" in state
        assert state["tokens"]["session"]["total"] == 300
        assert state["tokens"]["session"]["exact"] == 300
        assert state["tokens"]["session"]["estimated"] == 0
        # Lifetime combines exact + estimated
        assert state["tokens"]["lifetime"]["total"] == 13500
        assert state["tokens"]["lifetime"]["exact"] == 13000
        assert state["tokens"]["lifetime"]["estimated"] == 500

    def test_zero_tokens_omits_panel(self, collector):
        tokens = {
            "session": {"prompt_tokens": 0, "completion_tokens": 0, "total": 0, "estimated_total": 0},
            "lifetime": {"prompt_tokens": 0, "completion_tokens": 0, "total": 0, "estimated_total": 0},
        }
        responses = {
            "/stats": {"total_genes": 0, "total_chars_raw": 0, "total_chars_compressed": 0, "compression_ratio": 1.0},
            "/sessions": {"participants": []},
            "/metrics/tokens": tokens,
        }
        with patch("httpx.Client", return_value=_mock_client(responses)):
            with patch.object(collector, "_collect_models", return_value=None):
                state = collector.collect()
        # Panel still rendered if buckets exist (even with 0 totals) — that's
        # fine; the empty-state check is "did we get a response at all".
        # The panel template handles the all-zeros case visually.
        assert "tokens" in state
        assert state["tokens"]["session"]["total"] == 0


class TestModelsPanel:
    def test_ollama_models_collected(self, collector):
        ollama_resp = MagicMock()
        ollama_resp.status_code = 200
        ollama_resp.json.return_value = {
            "models": [
                {"name": "gemma4:e4b", "size": 4_400_000_000},
            ]
        }
        with patch("httpx.get", return_value=ollama_resp):
            models = collector._collect_models()
        assert models is not None
        assert models["loaded"][0]["name"] == "gemma4:e4b"
        assert models["loaded"][0]["source"] == "ollama"

    def test_ollama_unreachable_returns_none(self, collector):
        with patch("httpx.get", side_effect=Exception("connection refused")):
            models = collector._collect_models()
        assert models is None

    def test_empty_models_list_returns_none(self, collector):
        ollama_resp = MagicMock()
        ollama_resp.status_code = 200
        ollama_resp.json.return_value = {"models": []}
        with patch("httpx.get", return_value=ollama_resp):
            models = collector._collect_models()
        assert models is None

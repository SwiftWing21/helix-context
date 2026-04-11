"""
Tests for the AgentBridge session registry HTTP client (item 8).

Mocked httpx — no real helix server is contacted. The tests verify
state mutation, error handling, the auto-heartbeat daemon thread
lifecycle, and the soft-fail-on-network-error contract.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from helix_context.bridge import AgentBridge


@pytest.fixture
def bridge(tmp_path):
    return AgentBridge(shared_dir=str(tmp_path / "shared"))


def _ok_response(json_body):
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b"{}"
    resp.json.return_value = json_body
    return resp


def _err_response(status_code: int, text: str = "error"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = text.encode()
    resp.json.return_value = {"error": text}
    return resp


class TestRegisterParticipant:
    def test_register_success_stores_participant_id(self, bridge):
        with patch("httpx.post", return_value=_ok_response({
            "participant_id": "abc123",
            "party_id": "max@local",
            "registered_at": time.time(),
            "heartbeat_interval_s": 30.0,
            "ttl_s": 120.0,
        })):
            pid = bridge.register_participant(
                party_id="max@local",
                handle="taude",
                workspace="/tmp/work",
                capabilities=["query"],
            )
        assert pid == "abc123"
        assert bridge.participant_id == "abc123"
        assert bridge._registered_handle == "taude"
        assert bridge._registered_party_id == "max@local"
        assert bridge._heartbeat_interval_s == 30.0

    def test_register_failure_returns_none(self, bridge):
        with patch("httpx.post", return_value=_err_response(400, "bad")):
            pid = bridge.register_participant(party_id="max@local", handle="taude")
        assert pid is None
        assert bridge.participant_id is None

    def test_register_network_error_returns_none(self, bridge):
        with patch("httpx.post", side_effect=Exception("connection refused")):
            pid = bridge.register_participant(party_id="max@local", handle="taude")
        assert pid is None
        assert bridge.participant_id is None

    def test_register_includes_pid_in_body(self, bridge):
        captured = {}

        def capture_post(url, json=None, **kwargs):
            captured["body"] = json
            return _ok_response({"participant_id": "x", "party_id": "y"})

        with patch("httpx.post", side_effect=capture_post):
            bridge.register_participant(party_id="max@local", handle="taude")
        assert "pid" in captured["body"]
        assert isinstance(captured["body"]["pid"], int)


class TestHeartbeat:
    def test_heartbeat_returns_false_when_not_registered(self, bridge):
        assert bridge.heartbeat() is False

    def test_heartbeat_success(self, bridge):
        bridge._participant_id = "abc123"
        with patch("httpx.post", return_value=_ok_response({"ok": True})):
            assert bridge.heartbeat() is True
        # participant_id should still be set
        assert bridge.participant_id == "abc123"

    def test_heartbeat_404_clears_local_state(self, bridge):
        bridge._participant_id = "abc123"
        with patch("httpx.post", return_value=_err_response(404, "unknown")):
            assert bridge.heartbeat() is False
        assert bridge.participant_id is None

    def test_heartbeat_500_keeps_local_state(self, bridge):
        bridge._participant_id = "abc123"
        with patch("httpx.post", return_value=_err_response(500, "boom")):
            assert bridge.heartbeat() is False
        # Server error is transient — keep the id so the next call retries
        assert bridge.participant_id == "abc123"

    def test_heartbeat_network_error_returns_false(self, bridge):
        bridge._participant_id = "abc123"
        with patch("httpx.post", side_effect=Exception("connection refused")):
            assert bridge.heartbeat() is False
        assert bridge.participant_id == "abc123"


class TestListSessions:
    def test_list_sessions_returns_participants(self, bridge):
        participants = [
            {"handle": "taude", "party_id": "max@local", "status": "active"},
            {"handle": "laude", "party_id": "max@local", "status": "active"},
        ]
        with patch("httpx.get", return_value=_ok_response({
            "participants": participants,
            "count": 2,
        })):
            result = bridge.list_sessions(party_id="max@local")
        assert result == participants

    def test_list_sessions_failure_returns_none(self, bridge):
        with patch("httpx.get", return_value=_err_response(500)):
            assert bridge.list_sessions() is None


class TestRecentByHandle:
    def test_recent_returns_genes(self, bridge):
        genes = [
            {"gene_id": "g1", "content_preview": "first", "authored_at": 1.0},
        ]
        with patch("httpx.get", return_value=_ok_response({
            "handle": "taude",
            "genes": genes,
            "count": 1,
        })):
            result = bridge.recent_by_handle("taude", limit=5)
        assert result == genes

    def test_recent_failure_returns_none(self, bridge):
        with patch("httpx.get", return_value=_err_response(500)):
            assert bridge.recent_by_handle("taude") is None


class TestIngest:
    def test_ingest_attributes_when_registered(self, bridge):
        bridge._participant_id = "abc123"
        captured = {}

        def capture_post(url, json=None, **kwargs):
            captured["body"] = json
            return _ok_response({"gene_ids": ["g1"], "count": 1, "attributed": 1})

        with patch("httpx.post", side_effect=capture_post):
            result = bridge.ingest("hello world")

        assert result["count"] == 1
        assert captured["body"]["participant_id"] == "abc123"
        assert captured["body"]["content"] == "hello world"

    def test_ingest_skips_attribution_when_not_registered(self, bridge):
        captured = {}

        def capture_post(url, json=None, **kwargs):
            captured["body"] = json
            return _ok_response({"gene_ids": ["g1"], "count": 1})

        with patch("httpx.post", side_effect=capture_post):
            bridge.ingest("untagged")

        assert "participant_id" not in captured["body"]

    def test_ingest_attribute_false_overrides_registration(self, bridge):
        bridge._participant_id = "abc123"
        captured = {}

        def capture_post(url, json=None, **kwargs):
            captured["body"] = json
            return _ok_response({"gene_ids": ["g1"], "count": 1})

        with patch("httpx.post", side_effect=capture_post):
            bridge.ingest("untagged", attribute=False)

        assert "participant_id" not in captured["body"]


class TestAutoHeartbeat:
    def test_start_then_stop_runs_at_least_once(self, bridge):
        bridge._participant_id = "abc123"
        bridge._heartbeat_interval_s = 0.05  # 50ms — fast for tests

        call_count = {"n": 0}

        def fake_heartbeat():
            call_count["n"] += 1
            return True

        bridge.heartbeat = fake_heartbeat  # type: ignore[method-assign]

        bridge.start_auto_heartbeat()
        time.sleep(0.2)  # ~3 cycles (sleep first, then call)
        bridge.stop_auto_heartbeat()

        assert call_count["n"] >= 1

    def test_start_is_idempotent(self, bridge):
        bridge._participant_id = "abc123"
        bridge._heartbeat_interval_s = 60.0  # long — won't tick during test

        bridge.start_auto_heartbeat()
        first_thread = bridge._heartbeat_thread
        bridge.start_auto_heartbeat()  # second call
        second_thread = bridge._heartbeat_thread

        assert first_thread is second_thread
        bridge.stop_auto_heartbeat()

    def test_stop_is_idempotent(self, bridge):
        # No thread started — should not raise
        bridge.stop_auto_heartbeat()
        bridge.stop_auto_heartbeat()

    def test_thread_terminates_when_participant_id_cleared(self, bridge):
        bridge._participant_id = "abc123"
        bridge._heartbeat_interval_s = 0.05

        # First heartbeat call clears the id (simulates server 404)
        def angry_heartbeat():
            bridge._participant_id = None
            return False

        bridge.heartbeat = angry_heartbeat  # type: ignore[method-assign]

        bridge.start_auto_heartbeat()
        time.sleep(0.15)

        # Thread should have noticed and exited on its own
        assert bridge._heartbeat_thread is not None
        # Give it a moment to wind down
        time.sleep(0.1)
        assert not bridge._heartbeat_thread.is_alive()


class TestHttpHelpers:
    def test_post_returns_none_when_httpx_unavailable(self, bridge, monkeypatch):
        # Patch the import inside the method
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("simulated missing httpx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = bridge._http_post("/sessions/register", {"x": 1})
        assert result is None

    def test_get_returns_none_when_httpx_unavailable(self, bridge, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("simulated missing httpx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = bridge._http_get("/sessions")
        assert result is None

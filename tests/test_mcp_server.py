"""Tests for helix_context.mcp_server helper behavior."""

from helix_context.mcp_server import _default_ingest_identity, _normalize_health_payload


class TestNormalizeHealthPayload:
    def test_unreachable_becomes_unavailable(self):
        payload = {
            "_error": "helix unreachable",
            "_detail": "connection refused",
        }
        result = _normalize_health_payload(payload)
        assert result["availability"] == "unavailable"
        assert "helix-launcher" in result["next_action"]
        assert result["server"] == payload

    def test_ok_payload_becomes_available(self):
        payload = {
            "status": "ok",
            "genes": 42,
            "ribosome": "mock",
        }
        result = _normalize_health_payload(payload)
        assert result["availability"] == "available"
        assert "helix_context" in result["next_action"]

    def test_empty_genome_changes_next_action(self):
        payload = {
            "status": "ok",
            "genes": 0,
            "ribosome": "mock",
        }
        result = _normalize_health_payload(payload)
        assert result["availability"] == "available"
        assert "genome is empty" in result["next_action"]

    def test_http_error_becomes_degraded(self):
        payload = {
            "_error": "HTTP 500",
            "_detail": "boom",
        }
        result = _normalize_health_payload(payload)
        assert result["availability"] == "degraded"
        assert "restart" in result["next_action"].lower()

    def test_degraded_payload_preserves_server_message(self):
        payload = {
            "status": "degraded",
            "message": "Upstream model server is unreachable.",
            "checks": {"upstream_ready": False},
        }
        result = _normalize_health_payload(payload)
        assert result["availability"] == "degraded"
        assert "upstream model server is unreachable" in result["message"].lower()


class TestDefaultIngestIdentity:
    def test_prefers_explicit_full_identity_env(self, monkeypatch):
        monkeypatch.setenv("HELIX_ORG", "SwiftWing")
        monkeypatch.setenv("HELIX_DEVICE", "Wing-21")
        monkeypatch.setenv("HELIX_USER", "Max")
        monkeypatch.setenv("HELIX_AGENT", "Laude")
        monkeypatch.setenv("HELIX_AGENT_KIND", "Claude-Code")
        monkeypatch.setenv("HELIX_MCP_HANDLE", "ignored-session")
        monkeypatch.setenv("HELIX_MCP_HOST", "ignored-host")

        result = _default_ingest_identity()

        assert result == {
            "org_id": "swiftwing",
            "party_id": "wing-21",
            "participant_handle": "max",
            "agent_handle": "laude",
            "agent_kind": "claude-code",
        }

    def test_falls_back_to_session_presence_env(self, monkeypatch):
        monkeypatch.delenv("HELIX_ORG", raising=False)
        monkeypatch.delenv("HELIX_DEVICE", raising=False)
        monkeypatch.delenv("HELIX_USER", raising=False)
        monkeypatch.delenv("HELIX_AGENT", raising=False)
        monkeypatch.delenv("HELIX_AGENT_KIND", raising=False)
        monkeypatch.setenv("HELIX_PARTY_ID", "swift_wing21")
        monkeypatch.setenv("HELIX_MCP_HANDLE", "laude")
        monkeypatch.setenv("HELIX_MCP_HOST", "claude-code")

        result = _default_ingest_identity()

        assert result == {
            "party_id": "swift_wing21",
            "agent_handle": "laude",
            "agent_kind": "claude-code",
        }

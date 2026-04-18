"""Tests for the helix-status CLI helpers."""

from __future__ import annotations

import json

import helix_status as status_mod


class TestCheckMcpConfig:
    def test_canonical_config_detected(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "helix-context": {
                    "args": ["-m", "helix_context.mcp_server"],
                    "env": {"HELIX_MCP_URL": "http://127.0.0.1:11437"},
                }
            }
        }), encoding="utf-8")
        result = status_mod._check_mcp_config(cfg)
        assert result["status"] == "canonical"
        assert result["env_var"] == "HELIX_MCP_URL"

    def test_legacy_config_detected(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "helix-context": {
                    "args": ["-m", "helix_context.mcp.server"],
                    "env": {"HELIX_URL": "http://127.0.0.1:11437"},
                }
            }
        }), encoding="utf-8")
        result = status_mod._check_mcp_config(cfg)
        assert result["status"] == "legacy"
        assert "mcp_server" in result["next_action"]


class TestCollectStatus:
    def test_collect_status_available(self, tmp_path, monkeypatch):
        skill_dir = tmp_path / "helix-context"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("ok", encoding="utf-8")

        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "helix-context": {
                    "args": ["-m", "helix_context.mcp_server"],
                    "env": {"HELIX_MCP_URL": "http://127.0.0.1:11437"},
                }
            }
        }), encoding="utf-8")

        def fake_get_json(url: str, timeout_s: float = 1.5):
            if url.endswith("/health"):
                return {"status": "ok", "genes": 10}
            if url.endswith("/api/state"):
                return {"helix": {"running": True}}
            raise AssertionError(url)

        monkeypatch.setattr(status_mod, "_get_json", fake_get_json)

        result = status_mod.collect_status(
            mcp_config=cfg,
            skill_dir=skill_dir,
        )
        assert result["availability"] == "available"
        assert result["integration_ready"] is True
        assert result["mcp_config"]["status"] == "canonical"
        assert result["skill"]["status"] == "present"

    def test_collect_status_unavailable_points_to_launcher(self, tmp_path, monkeypatch):
        def fake_get_json(url: str, timeout_s: float = 1.5):
            return {"reachable": False, "error": "unreachable", "detail": "connection refused"}

        monkeypatch.setattr(status_mod, "_get_json", fake_get_json)

        result = status_mod.collect_status(
            mcp_config=tmp_path / ".mcp.json",
            skill_dir=tmp_path / "helix-context",
        )
        assert result["availability"] == "unavailable"
        assert result["integration_ready"] is False
        assert "helix-launcher" in result["next_action"]

    def test_collect_status_available_but_not_integration_ready_without_skill(self, tmp_path, monkeypatch):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "helix-context": {
                    "args": ["-m", "helix_context.mcp_server"],
                    "env": {"HELIX_MCP_URL": "http://127.0.0.1:11437"},
                }
            }
        }), encoding="utf-8")

        def fake_get_json(url: str, timeout_s: float = 1.5):
            if url.endswith("/health"):
                return {"status": "ok", "genes": 10}
            if url.endswith("/api/state"):
                return {"helix": {"running": True}}
            raise AssertionError(url)

        monkeypatch.setattr(status_mod, "_get_json", fake_get_json)

        result = status_mod.collect_status(
            mcp_config=cfg,
            skill_dir=tmp_path / "missing-skill",
        )
        assert result["availability"] == "available"
        assert result["integration_ready"] is False
        assert "shared `helix-context` skill" in result["next_action"]

    def test_find_mcp_config_accepts_string_start_dir(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text("{}", encoding="utf-8")
        found = status_mod._find_mcp_config(start_dir=str(tmp_path))
        assert found == cfg

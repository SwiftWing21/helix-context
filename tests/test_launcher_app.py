"""
Tests for helix_context.launcher.app — FastAPI endpoints with mocked
supervisor + collector. No real helix process is spawned.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from helix_context.config import HeadroomConfig, HelixConfig, ServerConfig
from helix_context.launcher.app import create_app
from helix_context.launcher.supervisor import (
    AlreadyRunning,
    NotRunning,
    ShutdownTimeout,
    StartupTimeout,
)


@pytest.fixture
def fake_store():
    store = MagicMock()
    store.state.helix_pid = None
    store.state.last_restart_reason = None
    store.state.last_restart_at = None
    return store


@pytest.fixture
def fake_supervisor(fake_store, tmp_path):
    sup = MagicMock()
    sup.store = fake_store
    sup.store.path = tmp_path / "state.json"
    sup.helix_host = "127.0.0.1"
    sup.helix_port = 11437
    sup.helix_log_path = tmp_path / "helix.log"
    sup.is_running.return_value = False
    sup.get_pid.return_value = None
    sup.get_uptime_s.return_value = None
    sup.adopt.return_value = False
    sup.find_orphan_helix.return_value = None
    sup.get_last_error.return_value = None
    sup.owns_process.return_value = False
    return sup


@pytest.fixture
def fake_collector():
    collector = MagicMock()
    collector.collect.return_value = {
        "helix": {
            "running": False,
            "host": "127.0.0.1",
            "port": 11437,
        }
    }
    return collector


@pytest.fixture
def client(fake_store, fake_supervisor, fake_collector):
    app = create_app(store=fake_store, supervisor=fake_supervisor, collector=fake_collector)
    with TestClient(app) as c:
        yield c


class TestDashboardHTML:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        # Page must contain the brand and the empty-state message
        assert "Helix Launcher" in resp.text
        assert "Helix is stopped" in resp.text

    def test_root_renders_running_state(self, client, fake_supervisor, fake_collector):
        fake_supervisor.is_running.return_value = True
        fake_supervisor.get_pid.return_value = 12345
        fake_collector.collect.return_value = {
            "helix": {"running": True, "pid": 12345, "port": 11437},
            "genes": {
                "total": 8000,
                "raw_chars": 47_000_000,
                "compressed_chars": 17_500_000,
                "compression_ratio": 2.69,
            },
        }
        resp = client.get("/")
        assert resp.status_code == 200
        assert "8,000" in resp.text or "8000" in resp.text


class TestApiState:
    def test_api_state_returns_collector_payload(self, client, fake_collector):
        fake_collector.collect.return_value = {"helix": {"running": False, "port": 11437}}
        resp = client.get("/api/state")
        assert resp.status_code == 200
        assert resp.json() == {"helix": {"running": False, "port": 11437}}


class TestLauncherOwnership:
    def test_shutdown_does_not_stop_adopted_helix(self, fake_store, fake_supervisor, fake_collector):
        fake_supervisor.adopt.return_value = True
        fake_supervisor.is_running.return_value = True
        fake_supervisor.owns_process.return_value = False

        app = create_app(store=fake_store, supervisor=fake_supervisor, collector=fake_collector)
        with TestClient(app):
            pass

        fake_supervisor.stop.assert_not_called()

    def test_shutdown_stops_owned_helix(self, fake_store, fake_supervisor, fake_collector):
        fake_supervisor.adopt.return_value = False
        fake_supervisor.is_running.return_value = True
        fake_supervisor.owns_process.return_value = True

        app = create_app(store=fake_store, supervisor=fake_supervisor, collector=fake_collector)
        with TestClient(app):
            pass

        fake_supervisor.stop.assert_called_once()


class TestPanelsPartial:
    def test_panels_partial_returns_html(self, client):
        resp = client.get("/api/state/panels")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        # Empty state when helix down
        assert "Helix is stopped" in resp.text

    def test_panels_partial_renders_degraded_message(self, client, fake_collector):
        fake_collector.collect.return_value = {
            "helix": {
                "running": True,
                "availability": "degraded",
                "next_action": "Restart it from the launcher UI.",
                "health_message": "Upstream model server is unreachable.",
                "port": 11437,
            }
        }
        resp = client.get("/api/state/panels")
        assert resp.status_code == 200
        assert "Upstream model server is unreachable." in resp.text

    def test_panels_partial_renders_disconnected_agents(self, client, fake_collector):
        fake_collector.collect.return_value = {
            "helix": {
                "running": True,
                "availability": "available",
                "port": 11437,
            },
            "disconnected_agents": {
                "count": 1,
                "entries": [
                    {
                        "handle": "raude",
                        "participant_id_short": "abc12345",
                        "participant_id": "abc12345-full",
                        "status": "stale",
                        "last_seen_s_ago": 3600,
                        "identifier": "swift_wing21",
                    }
                ],
            },
        }
        resp = client.get("/api/state/panels")
        assert resp.status_code == 200
        assert "Disconnected Agents" in resp.text
        assert "abc12345-full" in resp.text


class TestControlStart:
    def test_start_success(self, client, fake_supervisor):
        fake_supervisor.start.return_value = 99999
        resp = client.post("/api/control/start")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "pid": 99999}

    def test_start_already_running_returns_409(self, client, fake_supervisor):
        fake_supervisor.start.side_effect = AlreadyRunning("already")
        resp = client.post("/api/control/start")
        assert resp.status_code == 409

    def test_start_timeout_returns_500(self, client, fake_supervisor):
        fake_supervisor.start.side_effect = StartupTimeout("did not start")
        resp = client.post("/api/control/start")
        assert resp.status_code == 500


class TestControlStop:
    def test_stop_success(self, client, fake_supervisor):
        fake_supervisor.stop.return_value = None
        resp = client.post("/api/control/stop")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_stop_not_running_returns_409(self, client, fake_supervisor):
        fake_supervisor.stop.side_effect = NotRunning("not running")
        resp = client.post("/api/control/stop")
        assert resp.status_code == 409

    def test_stop_shutdown_timeout_returns_408(self, client, fake_supervisor):
        fake_supervisor.stop.side_effect = ShutdownTimeout("port stuck")
        resp = client.post("/api/control/stop")
        assert resp.status_code == 408


class TestControlRestart:
    def test_restart_success(self, client, fake_supervisor):
        fake_supervisor.restart.return_value = 88888
        resp = client.post("/api/control/restart")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "pid": 88888}


class TestNativeFailFast:
    def test_main_exits_1_when_native_without_pywebview(self, monkeypatch):
        """--native must fail loudly when pywebview isn't available, not silently exit."""
        from helix_context.launcher import app as app_mod

        monkeypatch.setattr(app_mod, "_check_native_available", lambda: False)
        rc = app_mod.main(["--native", "--no-browser", "--no-autostart"])
        assert rc == 1

    def test_check_native_available_returns_bool(self):
        from helix_context.launcher.app import _check_native_available
        assert isinstance(_check_native_available(), bool)


class TestMaybeBuildHeadroom:
    def test_adopts_running_headroom_even_when_disabled(self, fake_store, monkeypatch):
        from helix_context.launcher import app as app_mod
        from helix_context import config as config_mod

        class FakeHeadroomSupervisor:
            def __init__(self, store, host, port, mode):
                self.store = store
                self.host = host
                self.port = port
                self.mode = mode
                self.start_calls = 0

            def adopt(self):
                return True

            def start(self):
                self.start_calls += 1
                return 4242

        cfg = HelixConfig(
            headroom=HeadroomConfig(
                enabled=False,
                autostart=True,
                host="127.0.0.1",
                port=8787,
                mode="token",
                dashboard_path="/dashboard",
            )
        )

        monkeypatch.setattr(app_mod, "is_headroom_installed", lambda: True)
        monkeypatch.setattr(app_mod, "HeadroomSupervisor", FakeHeadroomSupervisor)
        monkeypatch.setattr(config_mod, "load_config", lambda: cfg)

        headroom, dashboard_url = app_mod._maybe_build_headroom(fake_store)

        assert headroom is not None
        assert dashboard_url == "http://127.0.0.1:8787/dashboard"
        assert headroom.start_calls == 0

    def test_disabled_headroom_without_running_proxy_stays_hidden(
        self,
        fake_store,
        monkeypatch,
    ):
        from helix_context.launcher import app as app_mod
        from helix_context import config as config_mod

        instances = []

        class FakeHeadroomSupervisor:
            def __init__(self, store, host, port, mode):
                self.store = store
                self.host = host
                self.port = port
                self.mode = mode
                self.start_calls = 0
                instances.append(self)

            def adopt(self):
                return False

            def start(self):
                self.start_calls += 1
                return 4242

        cfg = HelixConfig(
            headroom=HeadroomConfig(
                enabled=False,
                autostart=True,
                host="127.0.0.1",
                port=8787,
                mode="token",
                dashboard_path="/dashboard",
            )
        )

        monkeypatch.setattr(app_mod, "is_headroom_installed", lambda: True)
        monkeypatch.setattr(app_mod, "HeadroomSupervisor", FakeHeadroomSupervisor)
        monkeypatch.setattr(config_mod, "load_config", lambda: cfg)

        headroom, dashboard_url = app_mod._maybe_build_headroom(
            fake_store,
            autostart_override=True,
        )

        assert headroom is None
        assert dashboard_url is None
        assert len(instances) == 1
        assert instances[0].start_calls == 0


class TestHeadroomAutoRoute:
    def test_remote_upstream_routes_helix_via_headroom(self, monkeypatch):
        from helix_context.launcher import app as app_mod

        cfg = HelixConfig(
            server=ServerConfig(upstream="https://api.openai.com/v1"),
            headroom=HeadroomConfig(host="127.0.0.1", port=8787),
        )

        monkeypatch.delenv("HELIX_SERVER_UPSTREAM", raising=False)
        monkeypatch.delenv("OPENAI_TARGET_API_URL", raising=False)

        routed = app_mod._configure_helix_upstream_routing(cfg, auto_override=True)

        assert routed is True
        assert os.environ["HELIX_SERVER_UPSTREAM"] == "http://127.0.0.1:8787"
        assert os.environ["OPENAI_TARGET_API_URL"] == "https://api.openai.com/v1"

    def test_local_ollama_upstream_stays_direct(self, monkeypatch):
        from helix_context.launcher import app as app_mod

        cfg = HelixConfig(
            server=ServerConfig(upstream="http://localhost:11434"),
            headroom=HeadroomConfig(host="127.0.0.1", port=8787),
        )

        monkeypatch.setenv("HELIX_SERVER_UPSTREAM", "http://127.0.0.1:8787")
        monkeypatch.setenv("OPENAI_TARGET_API_URL", "https://api.openai.com/v1")

        routed = app_mod._configure_helix_upstream_routing(cfg, auto_override=True)

        assert routed is False
        assert "HELIX_SERVER_UPSTREAM" not in os.environ
        assert "OPENAI_TARGET_API_URL" not in os.environ

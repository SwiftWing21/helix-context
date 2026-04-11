"""
Tests for helix_context.launcher.app — FastAPI endpoints with mocked
supervisor + collector. No real helix process is spawned.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

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
def fake_supervisor(fake_store):
    sup = MagicMock()
    sup.store = fake_store
    sup.helix_host = "127.0.0.1"
    sup.helix_port = 11437
    sup.is_running.return_value = False
    sup.get_pid.return_value = None
    sup.get_uptime_s.return_value = None
    sup.adopt.return_value = False
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


class TestPanelsPartial:
    def test_panels_partial_returns_html(self, client):
        resp = client.get("/api/state/panels")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        # Empty state when helix down
        assert "Helix is stopped" in resp.text


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

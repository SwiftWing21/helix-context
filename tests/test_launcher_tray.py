"""
Tests for helix_context.launcher.tray — tray menu action handlers and
CLI integration. pystray is fully mocked so tests run headless.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from helix_context.launcher import tray as tray_mod
from helix_context.launcher.tray import HelixTrayIcon, _build_icon_image, is_tray_available
from helix_context.launcher.supervisor import (
    AlreadyRunning,
    NotRunning,
    SupervisorError,
)


@pytest.fixture
def fake_supervisor():
    sup = MagicMock()
    sup.is_running.return_value = True
    sup.get_pid.return_value = 12345
    sup.start.return_value = 12345
    sup.restart.return_value = 67890
    return sup


@pytest.fixture
def tray_icon(fake_supervisor):
    return HelixTrayIcon(
        supervisor=fake_supervisor,
        dashboard_url="http://127.0.0.1:11438/",
    )


class TestIsTrayAvailable:
    def test_returns_bool(self):
        assert isinstance(is_tray_available(), bool)


class TestBuildIconImage:
    def test_default_size(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            pytest.skip("PIL not installed — build_icon_image requires Pillow")
        img = _build_icon_image()
        assert img.size == (64, 64)

    def test_custom_size(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            pytest.skip("PIL not installed")
        img = _build_icon_image(size=32)
        assert img.size == (32, 32)


class TestMenuActions:
    def test_open_dashboard_calls_webbrowser(self, tray_icon):
        with patch("helix_context.launcher.tray.webbrowser.open") as mock_open:
            tray_icon._open_dashboard(None, None)
            mock_open.assert_called_once_with("http://127.0.0.1:11438/")

    def test_open_dashboard_swallows_errors(self, tray_icon):
        with patch(
            "helix_context.launcher.tray.webbrowser.open",
            side_effect=Exception("no browser"),
        ):
            # Should not raise
            tray_icon._open_dashboard(None, None)

    def test_start_calls_supervisor_start(self, tray_icon, fake_supervisor):
        fake_supervisor.is_running.return_value = False
        tray_icon._start_helix(None, None)
        fake_supervisor.start.assert_called_once()

    def test_start_handles_already_running(self, tray_icon, fake_supervisor):
        fake_supervisor.start.side_effect = AlreadyRunning("already up")
        # Should not raise
        tray_icon._start_helix(None, None)

    def test_start_handles_supervisor_error(self, tray_icon, fake_supervisor):
        fake_supervisor.start.side_effect = SupervisorError("port in use")
        # Should not raise
        tray_icon._start_helix(None, None)

    def test_restart_calls_supervisor_restart(self, tray_icon, fake_supervisor):
        tray_icon._restart_helix(None, None)
        fake_supervisor.restart.assert_called_once()

    def test_stop_calls_supervisor_stop(self, tray_icon, fake_supervisor):
        tray_icon._stop_helix(None, None)
        fake_supervisor.stop.assert_called_once()

    def test_stop_handles_not_running(self, tray_icon, fake_supervisor):
        fake_supervisor.stop.side_effect = NotRunning("not running")
        # Should not raise
        tray_icon._stop_helix(None, None)


class TestQuitAction:
    def test_quit_stops_helix_when_running(self, tray_icon, fake_supervisor):
        fake_supervisor.is_running.return_value = True
        tray_icon._icon = MagicMock()
        with patch("helix_context.launcher.tray.os.kill"):
            tray_icon._quit(None, None)
        fake_supervisor.stop.assert_called_once()
        tray_icon._icon.stop.assert_called_once()
        assert tray_icon._quit_event.is_set()

    def test_quit_skips_helix_stop_when_already_stopped(self, tray_icon, fake_supervisor):
        fake_supervisor.is_running.return_value = False
        tray_icon._icon = MagicMock()
        with patch("helix_context.launcher.tray.os.kill"):
            tray_icon._quit(None, None)
        fake_supervisor.stop.assert_not_called()
        tray_icon._icon.stop.assert_called_once()

    def test_quit_calls_on_quit_extra(self, fake_supervisor):
        on_quit_mock = MagicMock()
        tray = HelixTrayIcon(
            supervisor=fake_supervisor,
            dashboard_url="http://127.0.0.1:11438/",
            on_quit=on_quit_mock,
        )
        tray._icon = MagicMock()
        fake_supervisor.is_running.return_value = False
        with patch("helix_context.launcher.tray.os.kill"):
            tray._quit(None, None)
        on_quit_mock.assert_called_once()

    def test_quit_survives_on_quit_hook_exception(self, fake_supervisor):
        on_quit_mock = MagicMock(side_effect=Exception("boom"))
        tray = HelixTrayIcon(
            supervisor=fake_supervisor,
            dashboard_url="http://127.0.0.1:11438/",
            on_quit=on_quit_mock,
        )
        tray._icon = MagicMock()
        fake_supervisor.is_running.return_value = False
        with patch("helix_context.launcher.tray.os.kill"):
            # Should not raise
            tray._quit(None, None)
        assert tray._quit_event.is_set()


class TestCLIIntegration:
    def test_tray_and_native_rejected_on_non_windows(self, monkeypatch):
        """--tray --native combined returns exit 2 on non-Windows platforms."""
        from helix_context.launcher import app as app_mod
        monkeypatch.setattr(app_mod.sys, "platform", "darwin")
        rc = app_mod.main(["--tray", "--native", "--no-browser", "--no-autostart"])
        assert rc == 2

    def test_tray_and_native_rejected_on_linux(self, monkeypatch):
        from helix_context.launcher import app as app_mod
        monkeypatch.setattr(app_mod.sys, "platform", "linux"),
        rc = app_mod.main(["--tray", "--native", "--no-browser", "--no-autostart"])
        assert rc == 2

    def test_tray_without_extras_fails_fast(self, monkeypatch):
        """--tray with pystray unavailable returns exit code 1, not silent exit."""
        from helix_context.launcher import app as app_mod
        monkeypatch.setattr(app_mod, "_check_tray_available", lambda: False)
        rc = app_mod.main(["--tray", "--no-autostart"])
        assert rc == 1

    def test_check_tray_available_returns_bool(self):
        from helix_context.launcher.app import _check_tray_available
        assert isinstance(_check_tray_available(), bool)

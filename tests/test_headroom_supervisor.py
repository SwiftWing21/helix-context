"""Unit tests for HeadroomSupervisor.

Covers the adoption path, ownership gate on stop, cmdline marker
matching, and the HeadroomNotInstalled guard. Does NOT spawn real
headroom subprocesses — those live in integration tests (future).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from helix_context.launcher.headroom_supervisor import (
    HeadroomSupervisor,
    HeadroomSupervisorError,
    HeadroomNotInstalled,
    HeadroomNotRunning,
    is_headroom_installed,
)
from helix_context.launcher.state import LauncherState, StateStore


@pytest.fixture
def store(tmp_path):
    state_path = tmp_path / "launcher_state.json"
    return StateStore(state_path)


@pytest.fixture
def supervisor(store, tmp_path):
    return HeadroomSupervisor(
        store=store,
        host="127.0.0.1",
        port=18787,  # non-standard to avoid colliding with a real proxy
        mode="token",
        log_path=tmp_path / "headroom.log",
    )


# ── cmdline markers ──────────────────────────────────────────────────


def test_cmdline_markers_match_headroom_proxy(supervisor):
    assert supervisor._cmdline_looks_like_headroom([
        "python", "-m", "headroom", "proxy", "--port", "8787",
    ])
    assert supervisor._cmdline_looks_like_headroom([
        "/usr/bin/python3", "-m", "headroom", "proxy",
    ])


def test_cmdline_markers_reject_other_processes(supervisor):
    assert not supervisor._cmdline_looks_like_headroom([
        "python", "-m", "uvicorn", "helix_context.server:app",
    ])
    # "headroom" alone (no proxy subcommand) → not the proxy
    assert not supervisor._cmdline_looks_like_headroom([
        "python", "-m", "headroom", "memory", "list",
    ])


# ── command + dashboard URL ──────────────────────────────────────────


def test_command_uses_configured_host_port_mode(supervisor):
    cmd = supervisor._command()
    assert "headroom.cli" in cmd
    assert "proxy" in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "18787" in cmd
    assert "--mode" in cmd and "token" in cmd


def test_dashboard_url_uses_configured_host_port(supervisor):
    assert supervisor.dashboard_url() == "http://127.0.0.1:18787/dashboard"
    assert supervisor.dashboard_url("/ui") == "http://127.0.0.1:18787/ui"


def test_base_url_uses_configured_host_port(supervisor):
    assert supervisor.base_url() == "http://127.0.0.1:18787"


# ── is_running cleans stale state ────────────────────────────────────


def test_is_running_returns_false_when_no_pid(supervisor):
    assert supervisor.is_running() is False
    assert supervisor.owns_process() is False


def test_is_running_clears_dead_pid(supervisor, store):
    store.set_headroom(
        pid=999_999,  # presumed dead
        command=["python", "-m", "headroom", "proxy"],
        port=18787,
        owned=True,
    )
    # Mock psutil so pid_exists returns False
    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = False
    supervisor._psutil = fake_psutil

    assert supervisor.is_running() is False
    # State should be cleared
    assert store.state.headroom_pid is None


def test_is_running_rejects_pid_with_wrong_cmdline(supervisor, store):
    store.set_headroom(
        pid=12345,
        command=["python", "-m", "headroom", "proxy"],
        port=18787,
        owned=True,
    )
    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = ["python", "-m", "something_else"]

    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = True
    fake_psutil.Process.return_value = fake_proc
    fake_psutil.NoSuchProcess = Exception
    fake_psutil.AccessDenied = Exception
    supervisor._psutil = fake_psutil

    assert supervisor.is_running() is False
    assert store.state.headroom_pid is None


def test_is_running_restores_ownership_from_state(supervisor, store):
    store.set_headroom(
        pid=12345,
        command=["python", "-m", "headroom", "proxy"],
        port=18787,
        owned=False,  # adopted
    )
    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = ["python", "-m", "headroom", "proxy"]

    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = True
    fake_psutil.Process.return_value = fake_proc
    supervisor._psutil = fake_psutil

    assert supervisor.is_running() is True
    assert supervisor.owns_process() is False


# ── adoption ────────────────────────────────────────────────────────


def test_adopt_uses_stored_pid_first(supervisor, store):
    """If the state file already points at a live headroom, adopt it without
    scanning net_connections."""
    store.set_headroom(
        pid=12345,
        command=["python", "-m", "headroom", "proxy"],
        port=18787,
        owned=False,
    )
    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = ["python", "-m", "headroom", "proxy"]

    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = True
    fake_psutil.Process.return_value = fake_proc
    supervisor._psutil = fake_psutil

    assert supervisor.adopt() is True
    assert supervisor.owns_process() is False


def test_adopt_falls_back_to_orphan_scan(supervisor):
    """No stored state → scan net_connections for a LISTEN on the port."""
    fake_conn = MagicMock()
    fake_conn.status = "LISTEN"
    fake_conn.laddr = MagicMock(port=18787)
    fake_conn.pid = 54321

    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = ["python", "-m", "headroom", "proxy"]
    fake_proc.parent.return_value = None

    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = True
    fake_psutil.net_connections.return_value = [fake_conn]
    fake_psutil.Process.return_value = fake_proc
    supervisor._psutil = fake_psutil

    assert supervisor.adopt() is True
    assert supervisor.owns_process() is False
    assert supervisor.store.state.headroom_pid == 54321
    assert supervisor.store.state.headroom_owned is False


def test_adopt_ignores_non_headroom_on_port(supervisor):
    """Port listener exists but cmdline doesn't match → don't adopt."""
    fake_conn = MagicMock()
    fake_conn.status = "LISTEN"
    fake_conn.laddr = MagicMock(port=18787)
    fake_conn.pid = 54321

    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = ["python", "-m", "uvicorn", "app:app"]
    fake_proc.parent.return_value = None

    fake_psutil = MagicMock()
    fake_psutil.net_connections.return_value = [fake_conn]
    fake_psutil.Process.return_value = fake_proc
    supervisor._psutil = fake_psutil

    assert supervisor.adopt() is False
    assert supervisor.store.state.headroom_pid is None


def test_adopt_returns_false_when_nothing_listening(supervisor):
    fake_psutil = MagicMock()
    fake_psutil.net_connections.return_value = []
    supervisor._psutil = fake_psutil

    assert supervisor.adopt() is False


# ── stop() ownership gate ───────────────────────────────────────────


def test_stop_is_noop_for_adopted_process_without_force(supervisor, store):
    """Critical safety: adopted headroom survives Quit."""
    store.set_headroom(
        pid=12345,
        command=["python", "-m", "headroom", "proxy"],
        port=18787,
        owned=False,
    )
    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = ["python", "-m", "headroom", "proxy"]
    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = True
    fake_psutil.Process.return_value = fake_proc
    supervisor._psutil = fake_psutil

    # stop() without force — should log + return without killing
    with patch.object(supervisor, "_kill_tree") as mock_kill:
        supervisor.stop(force=False)
        mock_kill.assert_not_called()

    # State still holds the adopted pid
    assert store.state.headroom_pid == 12345


def test_stop_with_force_overrides_ownership(supervisor, store):
    """'Stop Headroom' tray click uses force=True to override ownership."""
    store.set_headroom(
        pid=12345,
        command=["python", "-m", "headroom", "proxy"],
        port=18787,
        owned=False,  # adopted
    )
    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = ["python", "-m", "headroom", "proxy"]
    fake_psutil = MagicMock()
    fake_psutil.pid_exists.return_value = True
    fake_psutil.Process.return_value = fake_proc
    supervisor._psutil = fake_psutil

    # Mock port-freeing loop to succeed immediately
    with patch.object(supervisor, "_kill_tree") as mock_kill, \
         patch(
             "helix_context.launcher.headroom_supervisor._port_is_free",
             return_value=True,
         ):
        supervisor.stop(force=True)
        mock_kill.assert_called_once_with(12345)

    assert store.state.headroom_pid is None


def test_stop_raises_when_not_running(supervisor):
    with pytest.raises(HeadroomNotRunning):
        supervisor.stop()


# ── not-installed guard ─────────────────────────────────────────────


def test_start_raises_when_headroom_not_installed(supervisor):
    with patch(
        "helix_context.launcher.headroom_supervisor.is_headroom_installed",
        return_value=False,
    ):
        with pytest.raises(HeadroomNotInstalled):
            supervisor.start()


def test_is_headroom_installed_real_probe():
    """Sanity: if we got this far pytest ran, the probe returns a bool."""
    assert isinstance(is_headroom_installed(), bool)


# ── state round-trip ─────────────────────────────────────────────────


def test_set_and_clear_headroom_state(store):
    assert store.state.headroom_pid is None
    store.set_headroom(
        pid=9999,
        command=["x", "y"],
        port=8787,
        owned=True,
    )
    assert store.state.headroom_pid == 9999
    assert store.state.headroom_port == 8787
    assert store.state.headroom_owned is True
    assert store.state.headroom_command == ["x", "y"]
    assert store.state.headroom_start_time is not None

    store.clear_headroom()
    assert store.state.headroom_pid is None
    assert store.state.headroom_owned is False
    assert store.state.headroom_command == []

"""
Tests for helix_context.launcher.supervisor.

All external side effects (subprocess spawn, psutil, httpx, taskkill) are
mocked — these are pure unit tests. No real helix process is ever started.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from helix_context.launcher.state import StateStore
from helix_context.launcher.supervisor import (
    AlreadyRunning,
    HelixSupervisor,
    NotRunning,
    StartupTimeout,
    SupervisorError,
)


@pytest.fixture
def store(tmp_path):
    return StateStore(path=tmp_path / "state.json")


@pytest.fixture
def supervisor(store, tmp_path):
    return HelixSupervisor(
        store=store,
        helix_host="127.0.0.1",
        helix_port=11999,  # unlikely to be in use
        helix_log_path=tmp_path / "helix.log",
    )


class _FakePsutil:
    """Minimal psutil stand-in. Controls pid_exists + Process cmdline."""

    def __init__(self, alive_pids=None, cmdlines=None):
        self._alive = set(alive_pids or [])
        self._cmdlines = cmdlines or {}
        self.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        self.AccessDenied = type("AccessDenied", (Exception,), {})

    def pid_exists(self, pid):
        return pid in self._alive

    def Process(self, pid):
        if pid not in self._alive:
            raise self.NoSuchProcess(pid)
        mock = MagicMock()
        mock.cmdline.return_value = self._cmdlines.get(pid, [])
        return mock


class TestIsRunning:
    def test_false_when_no_pid_in_state(self, supervisor):
        assert supervisor.is_running() is False

    def test_false_when_pid_dead(self, supervisor, store):
        store.set_helix(pid=12345, command=["python"], port=11999)
        fake = _FakePsutil(alive_pids=set())
        supervisor._psutil = fake
        assert supervisor.is_running() is False
        # State should be cleared
        assert store.state.helix_pid is None

    def test_false_when_cmdline_mismatch(self, supervisor, store):
        store.set_helix(pid=12345, command=["python"], port=11999)
        fake = _FakePsutil(
            alive_pids={12345},
            cmdlines={12345: ["nginx", "-g", "daemon off;"]},
        )
        supervisor._psutil = fake
        assert supervisor.is_running() is False
        assert store.state.helix_pid is None

    def test_true_when_pid_alive_and_matching(self, supervisor, store):
        store.set_helix(pid=12345, command=["python"], port=11999)
        fake = _FakePsutil(
            alive_pids={12345},
            cmdlines={12345: ["python", "-m", "uvicorn", "helix_context.server:app"]},
        )
        supervisor._psutil = fake
        assert supervisor.is_running() is True


class TestStart:
    def test_refuses_if_already_running(self, supervisor, store):
        store.set_helix(pid=12345, command=["python"], port=11999)
        supervisor._psutil = _FakePsutil(
            alive_pids={12345},
            cmdlines={12345: ["python", "-m", "uvicorn", "helix_context.server:app"]},
        )
        with pytest.raises(AlreadyRunning):
            supervisor.start()

    def test_refuses_if_port_in_use_by_unmanaged_process(self, supervisor):
        supervisor._psutil = _FakePsutil(alive_pids=set())
        with patch("helix_context.launcher.supervisor._port_is_free", return_value=False):
            with pytest.raises(SupervisorError, match="already in use"):
                supervisor.start()

    def test_start_spawns_subprocess_and_writes_state(self, supervisor, store):
        supervisor._psutil = _FakePsutil(alive_pids=set())

        fake_popen = MagicMock()
        fake_popen.pid = 54321

        with patch("helix_context.launcher.supervisor._port_is_free", return_value=True):
            with patch("subprocess.Popen", return_value=fake_popen) as popen_mock:
                with patch.object(supervisor, "_wait_for_ready"):
                    pid = supervisor.start()

        assert pid == 54321
        assert store.state.helix_pid == 54321
        assert store.state.helix_port == 11999
        assert popen_mock.called

        # Verify CREATE_NO_WINDOW was passed on Windows, 0 elsewhere
        popen_kwargs = popen_mock.call_args.kwargs
        assert "creationflags" in popen_kwargs
        # It's an int — either CREATE_NO_WINDOW (0x08000000) or 0
        assert isinstance(popen_kwargs["creationflags"], int)

    def test_start_rolls_back_on_startup_timeout(self, supervisor, store):
        supervisor._psutil = _FakePsutil(alive_pids=set())

        fake_popen = MagicMock()
        fake_popen.pid = 54321

        with patch("helix_context.launcher.supervisor._port_is_free", return_value=True):
            with patch("subprocess.Popen", return_value=fake_popen):
                with patch.object(
                    supervisor, "_wait_for_ready",
                    side_effect=StartupTimeout("timeout"),
                ):
                    with patch.object(supervisor, "_kill_tree") as kill_mock:
                        with pytest.raises(StartupTimeout):
                            supervisor.start()
                        kill_mock.assert_called_once_with(54321)

        # State should be cleared after rollback
        assert store.state.helix_pid is None


class TestStop:
    def test_refuses_when_not_running(self, supervisor):
        supervisor._psutil = _FakePsutil(alive_pids=set())
        with pytest.raises(NotRunning):
            supervisor.stop()

    def test_stop_announces_waits_kills_clears(self, supervisor, store):
        store.set_helix(pid=12345, command=["python"], port=11999)
        supervisor._psutil = _FakePsutil(
            alive_pids={12345},
            cmdlines={12345: ["python", "-m", "uvicorn", "helix_context.server:app"]},
        )

        announce_mock = MagicMock()
        kill_mock = MagicMock()
        port_free_mock = MagicMock(return_value=True)

        with patch.object(supervisor, "_announce_restart", announce_mock):
            with patch.object(supervisor, "_kill_tree", kill_mock):
                with patch(
                    "helix_context.launcher.supervisor._port_is_free",
                    port_free_mock,
                ):
                    supervisor.stop(reason="test stop")

        announce_mock.assert_called_once()
        kill_mock.assert_called_once_with(12345)
        assert store.state.helix_pid is None
        assert store.state.last_restart_reason == "test stop"


class TestAdopt:
    def test_adopt_returns_false_when_nothing_to_adopt(self, supervisor):
        supervisor._psutil = _FakePsutil(alive_pids=set())
        assert supervisor.adopt() is False

    def test_adopt_returns_true_when_alive_and_matching(self, supervisor, store):
        store.set_helix(pid=12345, command=["python"], port=11999)
        supervisor._psutil = _FakePsutil(
            alive_pids={12345},
            cmdlines={12345: ["python", "-m", "uvicorn", "helix_context.server:app"]},
        )
        assert supervisor.adopt() is True


class TestCommand:
    def test_command_includes_uvicorn_invocation(self, supervisor):
        cmd = supervisor._command()
        assert "-m" in cmd
        assert "uvicorn" in cmd
        assert "helix_context.server:app" in cmd
        assert "11999" in cmd

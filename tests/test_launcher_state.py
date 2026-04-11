"""
Tests for helix_context.launcher.state — atomic JSON read/write,
default values, mutation helpers, and corrupt-file recovery.
"""

import json
import os

import pytest

from helix_context.launcher.state import LauncherState, StateStore


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "state.json"


class TestStateStoreInit:
    def test_default_state_when_file_missing(self, state_path):
        store = StateStore(path=state_path)
        assert store.state.helix_pid is None
        assert store.state.helix_port == 11437
        assert store.state.helix_command == []

    def test_does_not_write_file_on_init(self, state_path):
        StateStore(path=state_path)
        # Init should create the parent dir but NOT write the state file.
        assert state_path.parent.exists()
        assert not state_path.exists()

    def test_corrupt_file_falls_back_to_default(self, state_path):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("not valid json", encoding="utf-8")
        store = StateStore(path=state_path)
        assert store.state.helix_pid is None  # fell back cleanly


class TestSetHelix:
    def test_set_helix_writes_file(self, state_path):
        store = StateStore(path=state_path)
        store.set_helix(
            pid=12345,
            command=["python", "-m", "uvicorn", "helix_context.server:app"],
            port=11437,
        )
        assert state_path.exists()
        on_disk = json.loads(state_path.read_text(encoding="utf-8"))
        assert on_disk["helix_pid"] == 12345
        assert on_disk["helix_port"] == 11437
        assert on_disk["helix_start_time"] is not None

    def test_set_helix_is_readable_by_fresh_store(self, state_path):
        StateStore(path=state_path).set_helix(
            pid=99, command=["python"], port=11440,
        )
        fresh = StateStore(path=state_path)
        assert fresh.state.helix_pid == 99
        assert fresh.state.helix_port == 11440

    def test_clear_helix_resets_fields(self, state_path):
        store = StateStore(path=state_path)
        store.set_helix(pid=99, command=["x"], port=11437)
        store.clear_helix()
        assert store.state.helix_pid is None
        assert store.state.helix_command == []


class TestRecordRestart:
    def test_record_restart_persists(self, state_path):
        store = StateStore(path=state_path)
        store.record_restart("test reason")
        fresh = StateStore(path=state_path)
        assert fresh.state.last_restart_reason == "test reason"
        assert fresh.state.last_restart_at is not None


class TestAtomicWrite:
    def test_tempfile_cleaned_up_on_success(self, state_path, tmp_path):
        store = StateStore(path=state_path)
        store.set_helix(pid=1, command=["x"], port=11437)
        # No stray tempfiles after atomic write.
        leftover_tmps = [
            f for f in os.listdir(tmp_path)
            if f.startswith("state_") and f.endswith(".tmp")
        ]
        assert leftover_tmps == []

    def test_reload_picks_up_external_change(self, state_path):
        store = StateStore(path=state_path)
        store.set_helix(pid=1, command=["x"], port=11437)

        # Simulate an external writer modifying the file.
        data = json.loads(state_path.read_text(encoding="utf-8"))
        data["helix_pid"] = 9999
        state_path.write_text(json.dumps(data), encoding="utf-8")

        reloaded = store.reload()
        assert reloaded.helix_pid == 9999


class TestLauncherStateDataclass:
    def test_defaults(self):
        s = LauncherState()
        assert s.helix_pid is None
        assert s.helix_port == 11437
        assert s.helix_command == []

    def test_unknown_fields_ignored_on_load(self, state_path):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"helix_pid": 5, "future_field": "ignored"}),
            encoding="utf-8",
        )
        store = StateStore(path=state_path)
        assert store.state.helix_pid == 5

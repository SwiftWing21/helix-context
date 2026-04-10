"""
Tests for the cross-session restart announcement protocol.

See docs/RESTART_PROTOCOL.md for the design.
"""

import json
import time

import pytest

from helix_context.bridge import AgentBridge


def test_announce_restart_writes_signal(tmp_path):
    """announce_restart populates server_state with the expected fields."""
    bridge = AgentBridge(shared_dir=str(tmp_path))
    bridge.announce_restart(
        reason="unit test",
        actor="pytest",
        expected_downtime_s=10,
    )

    signal = bridge.read_signal("server_state")
    assert signal is not None
    assert signal["state"] == "restarting"
    assert signal["actor"] == "pytest"
    assert signal["reason"] == "unit test"
    assert signal["expected_downtime_s"] == 10
    assert signal["phase"] == "shutting_down"
    assert "timestamp" in signal
    assert "timestamp_human" in signal


def test_read_server_state_returns_none_when_missing(tmp_path):
    """Observer gets None when no signal exists yet."""
    bridge = AgentBridge(shared_dir=str(tmp_path))
    assert bridge.read_server_state() is None


def test_read_server_state_fresh_restart(tmp_path):
    """Fresh restart signal is_stale=False, age under 1s."""
    bridge = AgentBridge(shared_dir=str(tmp_path))
    bridge.announce_restart(
        reason="fresh",
        actor="pytest",
        expected_downtime_s=30,
    )

    result = bridge.read_server_state()
    assert result is not None
    signal, is_stale, age_s = result
    assert signal["state"] == "restarting"
    assert is_stale is False
    assert age_s < 1.0


def test_read_server_state_stale_restart(tmp_path):
    """Restart signal older than budget+15s is reported stale."""
    bridge = AgentBridge(shared_dir=str(tmp_path))
    bridge.write_signal("server_state", {
        "state": "restarting",
        "actor": "ghost",
        "reason": "never completed",
        "expected_downtime_s": 5,
    })

    # Manually rewrite timestamp to 100s ago (well past 5+15=20 budget)
    path = bridge.signals / "server_state.json"
    data = json.loads(path.read_text())
    data["timestamp"] = time.time() - 100
    path.write_text(json.dumps(data))

    result = bridge.read_server_state()
    assert result is not None
    signal, is_stale, age_s = result
    assert signal["state"] == "restarting"
    assert is_stale is True
    assert age_s > 90


def test_read_server_state_running_never_stale(tmp_path):
    """state=running is never stale regardless of age."""
    bridge = AgentBridge(shared_dir=str(tmp_path))
    bridge.write_signal("server_state", {
        "state": "running",
        "actor": "lifespan",
        "pid": 12345,
        "expected_downtime_s": 0,
    })

    # Manually age the signal to 1000s ago
    path = bridge.signals / "server_state.json"
    data = json.loads(path.read_text())
    data["timestamp"] = time.time() - 1000
    path.write_text(json.dumps(data))

    result = bridge.read_server_state()
    assert result is not None
    signal, is_stale, age_s = result
    assert signal["state"] == "running"
    assert is_stale is False  # running is never stale


def test_write_signal_atomic_no_temp_leftover(tmp_path):
    """Atomic write should not leave .json.tmp files behind."""
    bridge = AgentBridge(shared_dir=str(tmp_path))
    bridge.write_signal("test", {"key": "value"})

    tmp_files = list(bridge.signals.glob("*.json.tmp"))
    assert len(tmp_files) == 0

    # And the final file exists + is readable
    signal = bridge.read_signal("test")
    assert signal["key"] == "value"

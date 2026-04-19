"""
Launcher state — atomic JSON read/write at ~/.helix/launcher/state.json.

Tracks the supervised helix child process across launcher restarts. The
launcher can crash or be restarted without killing helix; on next start
it reads this file, validates the PID is still alive and matches the
expected command line, and adopts the running process.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("helix.launcher.state")


DEFAULT_STATE_DIR = Path.home() / ".helix" / "launcher"
DEFAULT_STATE_PATH = DEFAULT_STATE_DIR / "state.json"


@dataclass
class LauncherState:
    """Persisted launcher state — read/written atomically."""

    helix_pid: Optional[int] = None
    helix_port: int = 11437
    helix_start_time: Optional[float] = None
    helix_command: List[str] = field(default_factory=list)
    launcher_pid: Optional[int] = None
    launcher_start_time: Optional[float] = None
    last_restart_reason: Optional[str] = None
    last_restart_at: Optional[float] = None
    # Headroom proxy (optional child — see HeadroomSupervisor)
    headroom_pid: Optional[int] = None
    headroom_port: int = 8787
    headroom_start_time: Optional[float] = None
    headroom_command: List[str] = field(default_factory=list)
    # True only if launcher spawned this headroom; False if adopted.
    # Adopted processes survive launcher Quit unless the user explicitly
    # clicked "Stop Headroom" in the tray.
    headroom_owned: bool = False


class StateStore:
    """File-backed, atomically-updated launcher state."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_STATE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: LauncherState = self._load()

    # ── read ────────────────────────────────────────────────────────

    @property
    def state(self) -> LauncherState:
        return self._state

    def _load(self) -> LauncherState:
        if not self.path.exists():
            return LauncherState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return LauncherState(**{k: v for k, v in raw.items() if k in LauncherState.__dataclass_fields__})
        except Exception:
            log.warning("Failed to read launcher state at %s, starting fresh", self.path, exc_info=True)
            return LauncherState()

    # ── write (atomic) ──────────────────────────────────────────────

    def _write(self) -> None:
        """Atomic write via tempfile + os.replace.

        On both POSIX and Windows, os.replace is atomic within a single
        filesystem. Readers never see a partially-written file.
        """
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="state_", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self._state), f, indent=2, sort_keys=True)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── mutation helpers ───────────────────────────────────────────

    def set_helix(
        self,
        pid: int,
        command: List[str],
        port: int,
        start_time: Optional[float] = None,
    ) -> None:
        self._state.helix_pid = pid
        self._state.helix_command = list(command)
        self._state.helix_port = port
        self._state.helix_start_time = start_time if start_time is not None else time.time()
        self._write()

    def clear_helix(self) -> None:
        self._state.helix_pid = None
        self._state.helix_start_time = None
        self._state.helix_command = []
        self._write()

    def set_headroom(
        self,
        pid: int,
        command: List[str],
        port: int,
        owned: bool,
        start_time: Optional[float] = None,
    ) -> None:
        self._state.headroom_pid = pid
        self._state.headroom_command = list(command)
        self._state.headroom_port = port
        self._state.headroom_owned = owned
        self._state.headroom_start_time = (
            start_time if start_time is not None else time.time()
        )
        self._write()

    def clear_headroom(self) -> None:
        self._state.headroom_pid = None
        self._state.headroom_start_time = None
        self._state.headroom_command = []
        self._state.headroom_owned = False
        self._write()

    def set_launcher(self, pid: int, start_time: Optional[float] = None) -> None:
        self._state.launcher_pid = pid
        self._state.launcher_start_time = start_time if start_time is not None else time.time()
        self._write()

    def record_restart(self, reason: str) -> None:
        self._state.last_restart_reason = reason
        self._state.last_restart_at = time.time()
        self._write()

    def reload(self) -> LauncherState:
        """Re-read from disk — useful if another process may have updated."""
        self._state = self._load()
        return self._state

"""
Headroom supervisor — manages the optional Headroom proxy child process.

Headroom (https://github.com/chopratejas/headroom, Apache-2.0) is a
separate CLI process that serves a compression proxy + dashboard at
``http://{host}:{port}/dashboard``. This supervisor mirrors the pattern
in ``supervisor.py`` (HelixSupervisor) with two important behavioural
differences:

1. **Graceful adoption is primary, not fallback.** If a headroom proxy
   is already running on the configured port when the launcher starts
   (common — a developer may have launched it manually), we adopt it
   rather than spawning a duplicate. Duplicate spawn would fail with
   "port busy" anyway; adoption preserves the user's choices.

2. **Ownership gates destructive ops.** Adopted processes stay alive on
   launcher Quit. Only when ``_owned=True`` (we spawned it) does Quit
   also stop headroom. The explicit "Stop Headroom" menu item in the
   tray always attempts a stop, overriding ownership.

Never imports headroom directly — communication is over HTTP at
``http://{host}:{port}`` and process-level signals via psutil.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx

from .state import StateStore

log = logging.getLogger("helix.launcher.headroom")


class HeadroomSupervisorError(Exception):
    """Base class for headroom supervisor failures."""


class HeadroomAlreadyRunning(HeadroomSupervisorError):
    pass


class HeadroomNotRunning(HeadroomSupervisorError):
    pass


class HeadroomStartupTimeout(HeadroomSupervisorError):
    pass


class HeadroomShutdownTimeout(HeadroomSupervisorError):
    pass


class HeadroomNotInstalled(HeadroomSupervisorError):
    """Raised when the ``headroom`` package isn't importable."""


def _is_windows() -> bool:
    return sys.platform == "win32"


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
        except (ConnectionRefusedError, OSError):
            return True
        return False


def is_headroom_installed() -> bool:
    """Cheap probe: is `helix-context[codec]` actually installed?"""
    return importlib.util.find_spec("headroom") is not None


class HeadroomSupervisor:
    """Lifecycle manager for one optional Headroom proxy child.

    Exactly mirrors HelixSupervisor's start/stop/restart surface so the
    tray menu can treat both uniformly.
    """

    # Any argv token that marks a "headroom proxy" process. Used for
    # adoption — we verify an orphan's cmdline contains these markers
    # before treating it as adoptable.
    _CMDLINE_MARKERS = ("headroom", "proxy")

    def __init__(
        self,
        store: StateStore,
        host: str = "127.0.0.1",
        port: int = 8787,
        mode: str = "token",
        python_executable: Optional[str] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.mode = mode
        self.python_executable = python_executable or sys.executable
        self.log_path = log_path or (
            Path.home() / ".helix" / "launcher" / "headroom.log"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._psutil = None
        self._owns_process = False

    def _get_psutil(self):
        if self._psutil is None:
            try:
                import psutil  # type: ignore
                self._psutil = psutil
            except ImportError as e:
                raise HeadroomSupervisorError(
                    "psutil is required. Install with: "
                    "pip install helix-context[launcher]"
                ) from e
        return self._psutil

    # ── command + dashboard URL ────────────────────────────────────

    def _command(self) -> List[str]:
        """`python -m headroom proxy --host ... --port ... --mode ...`."""
        return [
            self.python_executable,
            "-m",
            "headroom",
            "proxy",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--mode",
            self.mode,
        ]

    def dashboard_url(self, path: str = "/dashboard") -> str:
        return f"http://{self.host}:{self.port}{path}"

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ── liveness + adoption ────────────────────────────────────────

    def is_running(self) -> bool:
        """Return True if a tracked headroom process is alive.

        Follows the HelixSupervisor pattern: PID from state → exists →
        cmdline contains the expected markers. Cleans up stale state.
        """
        pid = self.store.state.headroom_pid
        if pid is None:
            self._owns_process = False
            return False
        psutil = self._get_psutil()
        if not psutil.pid_exists(pid):
            log.info("Stored headroom PID %d is dead; clearing state", pid)
            self.store.clear_headroom()
            self._owns_process = False
            return False
        try:
            proc = psutil.Process(pid)
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.store.clear_headroom()
            self._owns_process = False
            return False
        if not self._cmdline_looks_like_headroom(cmdline):
            log.warning(
                "PID %d exists but cmdline doesn't match headroom proxy; "
                "clearing state",
                pid,
            )
            self.store.clear_headroom()
            self._owns_process = False
            return False
        # Ownership survived roundtrip from persisted state.
        self._owns_process = bool(self.store.state.headroom_owned)
        return True

    def _cmdline_looks_like_headroom(self, cmdline: List[str]) -> bool:
        joined = " ".join(cmdline).lower()
        return all(m in joined for m in self._CMDLINE_MARKERS)

    def get_pid(self) -> Optional[int]:
        return self.store.state.headroom_pid if self.is_running() else None

    def get_uptime_s(self) -> Optional[float]:
        if not self.is_running():
            return None
        start = self.store.state.headroom_start_time
        if start is None:
            return None
        return max(0.0, time.time() - start)

    def owns_process(self) -> bool:
        return self._owns_process

    def ping(self, timeout: float = 1.0) -> bool:
        """HTTP probe — does headroom respond on /health?"""
        try:
            resp = httpx.get(
                f"{self.base_url()}/health",
                timeout=timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── orphan detection ───────────────────────────────────────────

    def find_orphan_headroom(self) -> Optional[int]:
        """Find a headroom proxy listening on `self.port` we didn't spawn.

        Same strategy as HelixSupervisor.find_orphan_helix: walk
        psutil.net_connections looking for a LISTEN on our port, then
        verify the process's cmdline matches the markers.

        Returns the PID or None. Never raises.
        """
        try:
            psutil = self._get_psutil()
        except HeadroomSupervisorError:
            return None

        listener_pid: Optional[int] = None
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (
                    conn.status == "LISTEN"
                    and conn.laddr
                    and conn.laddr.port == self.port
                    and conn.pid is not None
                ):
                    listener_pid = conn.pid
                    break
        except (psutil.AccessDenied, PermissionError):
            log.debug("Orphan scan: net_connections denied", exc_info=True)
            return None
        except Exception:
            log.debug("Orphan scan: net_connections failed", exc_info=True)
            return None

        if listener_pid is None:
            return None

        try:
            proc = psutil.Process(listener_pid)
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

        if not self._cmdline_looks_like_headroom(cmdline):
            log.debug(
                "Orphan scan: PID %d listens on %d but is not headroom (%r)",
                listener_pid, self.port, cmdline[:3],
            )
            return None

        # Walk up one level in case the listener is a uvicorn worker and
        # the top-level entrypoint is the parent (headroom proxy uses uvicorn).
        try:
            parent = proc.parent()
            if parent is not None:
                parent_cmdline = parent.cmdline()
                if self._cmdline_looks_like_headroom(parent_cmdline):
                    return parent.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return listener_pid

    def adopt(self) -> bool:
        """Try to adopt an already-running headroom.

        Two-stage:
            1. Stored PID → is_running() verifies + returns
            2. Orphan scan → find any headroom on the configured port
               and adopt it (write PID + owned=False)

        Returns True if a headroom was adopted, False otherwise.
        """
        if self.is_running():
            log.info(
                "Adopted existing headroom via state file (pid=%d, port=%d)",
                self.store.state.headroom_pid, self.store.state.headroom_port,
            )
            # State already flagged ownership; keep it as-is.
            return True

        orphan_pid = self.find_orphan_headroom()
        if orphan_pid is None:
            return False

        try:
            psutil = self._get_psutil()
            proc = psutil.Process(orphan_pid)
            cmdline = proc.cmdline()
        except Exception:
            cmdline = self._command()

        self.store.set_headroom(
            pid=orphan_pid,
            command=cmdline,
            port=self.port,
            owned=False,
        )
        self._owns_process = False
        log.info(
            "Adopted orphan headroom on %s:%d (pid=%d) — "
            "launcher will NOT stop it on Quit",
            self.host, self.port, orphan_pid,
        )
        return True

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self, wait_ready: bool = True, timeout: float = 20.0) -> int:
        """Spawn headroom, or adopt an existing one.

        Raises:
            HeadroomNotInstalled: if the headroom package isn't importable.
            HeadroomAlreadyRunning: a tracked headroom is already running.
            HeadroomSupervisorError: port busy with a non-headroom process.
            HeadroomStartupTimeout: spawn succeeded but /health never responded.
        """
        if not is_headroom_installed():
            raise HeadroomNotInstalled(
                "headroom is not installed. "
                "Install with: pip install 'helix-context[codec]'"
            )

        if self.is_running():
            raise HeadroomAlreadyRunning(
                f"headroom already running (pid={self.get_pid()})"
            )

        if not _port_is_free(self.host, self.port):
            # Port busy — adopt if it's a headroom we can recognize.
            orphan_pid = self.find_orphan_headroom()
            if orphan_pid is not None:
                log.info(
                    "Start: port %d busy with orphan headroom (pid=%d) — adopting",
                    self.port, orphan_pid,
                )
                try:
                    psutil = self._get_psutil()
                    proc = psutil.Process(orphan_pid)
                    cmdline = proc.cmdline()
                except Exception:
                    cmdline = self._command()
                self.store.set_headroom(
                    pid=orphan_pid,
                    command=cmdline,
                    port=self.port,
                    owned=False,
                )
                self._owns_process = False
                return orphan_pid

            raise HeadroomSupervisorError(
                f"Port {self.host}:{self.port} is already in use by a "
                "non-headroom process. Free the port or set a different "
                "[headroom] port in helix.toml."
            )

        cmd = self._command()
        log.info("Starting headroom: %s", " ".join(cmd))

        # Windows-safe: suppress console window flash.
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if _is_windows() else 0
        )
        preexec_fn = os.setsid if not _is_windows() else None
        log_file = open(self.log_path, "ab")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
                close_fds=True,
            )
        except Exception:
            log_file.close()
            raise

        self.store.set_headroom(
            pid=proc.pid,
            command=cmd,
            port=self.port,
            owned=True,
        )
        self._owns_process = True

        if wait_ready:
            try:
                self._wait_for_ready(timeout=timeout)
            except HeadroomStartupTimeout:
                try:
                    self._kill_tree(proc.pid)
                except Exception:
                    pass
                self.store.clear_headroom()
                self._owns_process = False
                raise

        log.info("Headroom started (pid=%d)", proc.pid)
        return proc.pid

    def stop(
        self,
        reason: str = "manual stop from launcher",
        timeout: float = 10.0,
        force: bool = False,
    ) -> None:
        """Stop headroom.

        If ``force=False`` and we don't own the process (it was adopted),
        this is a no-op with a log line. Pass ``force=True`` from the
        "Stop Headroom" tray click to override — user-initiated shutdown
        overrides ownership.
        """
        if not self.is_running():
            raise HeadroomNotRunning("headroom is not running")

        if not self._owns_process and not force:
            log.info(
                "stop(): headroom was adopted, not owned — leaving it alive "
                "(pass force=True to override)"
            )
            return

        pid = self.store.state.headroom_pid
        assert pid is not None

        log.info("Stopping headroom (pid=%d, reason=%s, force=%s)",
                 pid, reason, force)
        self._kill_tree(pid)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _port_is_free(self.host, self.port):
                self.store.clear_headroom()
                self._owns_process = False
                log.info("Headroom stopped (pid=%d)", pid)
                return
            time.sleep(0.2)

        raise HeadroomShutdownTimeout(
            f"Port {self.port} did not free within {timeout}s after kill"
        )

    def restart(self, reason: str = "manual restart from launcher") -> int:
        if self.is_running():
            # Restart always forces — user-initiated.
            self.stop(reason=reason, force=True)
        return self.start()

    # ── internals ──────────────────────────────────────────────────

    def _kill_tree(self, pid: int) -> None:
        psutil = self._get_psutil()
        if _is_windows():
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
            except Exception:
                log.warning("taskkill failed for pid %d", pid, exc_info=True)
            return

        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            log.warning("SIGTERM to pgid failed for pid %d", pid, exc_info=True)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                return
            time.sleep(0.1)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            log.warning("SIGKILL to pgid failed for pid %d", pid, exc_info=True)

    def _wait_for_ready(self, timeout: float = 20.0) -> None:
        url = f"{self.base_url()}/health"
        deadline = time.monotonic() + timeout
        last_error: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
                last_error = f"HTTP {resp.status_code}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise HeadroomStartupTimeout(
            f"headroom did not become ready within {timeout}s "
            f"(last_error: {last_error})"
        )

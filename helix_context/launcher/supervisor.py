"""
Supervisor — manages the helix child process lifecycle.

Responsibilities:
    - Start a new helix uvicorn process with the user's configured port
    - Stop it gracefully via the restart protocol (announce → kill → wait)
    - Restart (stop + start)
    - Adopt an already-running helix from state file (PID + command-line match)
    - Cross-platform process tree kill (taskkill /F /T on Windows, killpg on POSIX)

Never imports helix_context.server directly — all communication with the
supervised helix is over HTTP at http://127.0.0.1:{port}.
"""

from __future__ import annotations

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

log = logging.getLogger("helix.launcher.supervisor")


class SupervisorError(Exception):
    """Base class for supervisor failures."""


class AlreadyRunning(SupervisorError):
    pass


class NotRunning(SupervisorError):
    pass


class StartupTimeout(SupervisorError):
    pass


class ShutdownTimeout(SupervisorError):
    pass


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


class HelixSupervisor:
    """Lifecycle manager for one helix child process."""

    def __init__(
        self,
        store: StateStore,
        helix_host: str = "127.0.0.1",
        helix_port: int = 11437,
        python_executable: Optional[str] = None,
        helix_log_path: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.helix_host = helix_host
        self.helix_port = helix_port
        self.python_executable = python_executable or sys.executable
        self.helix_log_path = helix_log_path or (
            Path.home() / ".helix" / "launcher" / "helix.log"
        )
        self.helix_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Import psutil lazily so the module can be imported even when the
        # [launcher] extra is not installed.
        self._psutil = None
        # Telemetry — stashed by start/stop/restart on failure, cleared on success.
        self._last_error: Optional[str] = None
        self._last_error_at: Optional[float] = None
        self._last_error_operation: Optional[str] = None
        self._owns_helix_process = False
        # Tracked Popen handle — used for POSIX zombie reap via poll().
        # Windows reaps through taskkill; POSIX needs a wait() sibling call.
        self._proc: Optional[subprocess.Popen] = None

    def _get_psutil(self):
        if self._psutil is None:
            try:
                import psutil  # type: ignore
                self._psutil = psutil
            except ImportError as e:
                raise SupervisorError(
                    "psutil is required. Install with: pip install helix-context[launcher]"
                ) from e
        return self._psutil

    # ── command construction ───────────────────────────────────────

    def _command(self) -> List[str]:
        return [
            self.python_executable,
            "-m",
            "uvicorn",
            "helix_context.server:app",
            "--host",
            self.helix_host,
            "--port",
            str(self.helix_port),
        ]

    # ── liveness checks ────────────────────────────────────────────

    def is_running(self) -> bool:
        """Return True if a tracked helix process is alive and responsive."""
        # POSIX zombie reap: non-blocking poll() retires a dead child so it
        # doesn't accumulate as <defunct> in ps. No-op on a live process.
        if self._proc is not None:
            try:
                self._proc.poll()
            except Exception:
                log.warning("Popen.poll() failed", exc_info=True)
        pid = self.store.state.helix_pid
        if pid is None:
            self._owns_helix_process = False
            return False
        psutil = self._get_psutil()
        if not psutil.pid_exists(pid):
            log.info("Stored helix PID %d is dead; clearing state", pid)
            self.store.clear_helix()
            self._owns_helix_process = False
            return False
        # PID exists — verify it's actually our uvicorn process.
        try:
            proc = psutil.Process(pid)
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.store.clear_helix()
            self._owns_helix_process = False
            return False
        expected_marker = "helix_context.server:app"
        if not any(expected_marker in part for part in cmdline):
            log.warning(
                "PID %d exists but command line doesn't match helix uvicorn; clearing state",
                pid,
            )
            self.store.clear_helix()
            self._owns_helix_process = False
            return False
        return True

    def get_pid(self) -> Optional[int]:
        return self.store.state.helix_pid if self.is_running() else None

    def get_uptime_s(self) -> Optional[float]:
        if not self.is_running():
            return None
        start = self.store.state.helix_start_time
        if start is None:
            return None
        return max(0.0, time.time() - start)

    # ── telemetry ──────────────────────────────────────────────────

    def get_last_error(self) -> Optional[dict]:
        """Return the last failed operation + message, or None if no error.

        Cleared on successful start/stop/restart. Structure:
            {"operation": "start" | "stop" | "restart", "message": str, "at": float}
        """
        if self._last_error is None:
            return None
        return {
            "operation": self._last_error_operation,
            "message": self._last_error,
            "at": self._last_error_at,
        }

    def _record_error(self, operation: str, message: str) -> None:
        self._last_error = message
        self._last_error_at = time.time()
        self._last_error_operation = operation

    def _clear_error(self) -> None:
        self._last_error = None
        self._last_error_at = None
        self._last_error_operation = None

    def owns_process(self) -> bool:
        """Return True if this launcher instance spawned the tracked Helix."""
        return self._owns_helix_process

    # ── orphan detection ───────────────────────────────────────────

    def find_orphan_helix(self) -> Optional[int]:
        """Scan for an unmanaged helix uvicorn process on the configured port.

        Uses psutil's process-wide connection table to find the listener
        on helix_host:helix_port, then walks up to the uvicorn parent
        process whose command line matches helix_context.server:app.

        Returns the uvicorn **parent** PID (the one ``subprocess.Popen``
        would hand us if we'd started helix ourselves), or None if no
        matching orphan is found.

        Never raises — returns None on any psutil failure.
        """
        try:
            psutil = self._get_psutil()
        except SupervisorError:
            return None

        # Step 1: find the PID listening on helix_port.
        listener_pid: Optional[int] = None
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (
                    conn.status == "LISTEN"
                    and conn.laddr
                    and conn.laddr.port == self.helix_port
                    and conn.pid is not None
                ):
                    listener_pid = conn.pid
                    break
        except (psutil.AccessDenied, PermissionError):
            log.debug("Orphan scan: net_connections denied (need admin?)", exc_info=True)
            return None
        except Exception:
            log.debug("Orphan scan: net_connections failed", exc_info=True)
            return None

        if listener_pid is None:
            return None

        # Step 2: verify the listener is actually a helix uvicorn process.
        try:
            listener_proc = psutil.Process(listener_pid)
            listener_cmdline = listener_proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

        expected_marker = "helix_context.server:app"
        if not any(expected_marker in part for part in listener_cmdline):
            log.debug(
                "Orphan scan: PID %d listens on %d but is not helix (%r)",
                listener_pid, self.helix_port, listener_cmdline[:3],
            )
            return None

        # Step 3: walk up to the uvicorn parent (matches what subprocess.Popen
        # would hand us for our own spawned helix). On Windows the listener
        # is typically the worker (uvicorn spawns a child process); its
        # parent is the top-level uvicorn entrypoint.
        try:
            parent = listener_proc.parent()
            if parent is not None:
                parent_cmdline = parent.cmdline()
                if any(expected_marker in part for part in parent_cmdline):
                    return parent.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Parent doesn't match (or we couldn't read it) — track the listener itself.
        return listener_pid

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self, wait_ready: bool = True, timeout: float = 30.0) -> int:
        """Spawn a new helix uvicorn subprocess, or adopt an existing one.

        If the port is occupied by another helix uvicorn process (e.g.
        started outside the launcher by a developer's bash shell), the
        start path adopts that process instead of failing. Only when
        the port is occupied by a *non-helix* process does start() raise.
        """
        if self.is_running():
            raise AlreadyRunning(f"helix already running (pid={self.get_pid()})")

        if not _port_is_free(self.helix_host, self.helix_port):
            # Port is busy — check whether it's a helix we can adopt.
            orphan_pid = self.find_orphan_helix()
            if orphan_pid is not None:
                log.info(
                    "Start: port %d busy with orphan helix (pid=%d) — adopting instead",
                    self.helix_port, orphan_pid,
                )
                try:
                    psutil = self._get_psutil()
                    proc = psutil.Process(orphan_pid)
                    cmdline = proc.cmdline()
                except Exception:
                    cmdline = self._command()
                self.store.set_helix(
                    pid=orphan_pid,
                    command=cmdline,
                    port=self.helix_port,
                )
                self._owns_helix_process = False
                self._clear_error()
                return orphan_pid

            msg = (
                f"Port {self.helix_host}:{self.helix_port} is already in use "
                "by a non-helix process. Free the port or change --helix-port."
            )
            self._record_error("start", msg)
            raise SupervisorError(msg)

        cmd = self._command()
        log.info("Starting helix: %s", " ".join(cmd))

        # Per project convention (CLAUDE.md): suppress console window flash on Windows.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if _is_windows() else 0

        # On POSIX, put the child in a new process group so we can signal
        # the whole tree via killpg.
        preexec_fn = os.setsid if not _is_windows() else None

        # Popen dups the fd internally on both POSIX and Windows, so the
        # parent can close its handle immediately after Popen returns.
        with open(self.helix_log_path, "ab") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=self._cwd(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
                close_fds=True,
            )

        self._proc = proc
        self.store.set_helix(pid=proc.pid, command=cmd, port=self.helix_port)
        self._owns_helix_process = True

        if wait_ready:
            try:
                self._wait_for_ready(timeout=timeout)
            except StartupTimeout as exc:
                # Roll back: kill whatever started and clear state.
                try:
                    self._kill_tree(proc.pid)
                except Exception:
                    pass
                self.store.clear_helix()
                self._owns_helix_process = False
                self._record_error("start", str(exc))
                raise

        log.info("Helix started (pid=%d)", proc.pid)
        self._clear_error()
        return proc.pid

    def stop(
        self,
        reason: str = "manual stop from launcher",
        announce: bool = True,
        timeout: float = 10.0,
    ) -> None:
        """Announce, wait, kill, wait for port to free up."""
        if not self.is_running():
            raise NotRunning("helix is not running")

        pid = self.store.state.helix_pid
        assert pid is not None  # narrowed by is_running

        if announce:
            self._announce_restart(reason=reason, expected_downtime_s=int(timeout))
            # Sleep ~750ms so observers see the signal (per restart protocol).
            time.sleep(0.75)

        self.store.record_restart(reason)
        self._kill_tree(pid)

        # Wait for port to free up.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _port_is_free(self.helix_host, self.helix_port):
                self.store.clear_helix()
                self._owns_helix_process = False
                log.info("Helix stopped (pid=%d)", pid)
                self._clear_error()
                return
            time.sleep(0.2)

        msg = (
            f"Port {self.helix_port} did not free up within {timeout}s after kill"
        )
        self._record_error("stop", msg)
        raise ShutdownTimeout(msg)

    def restart(self, reason: str = "manual restart from launcher") -> int:
        try:
            if self.is_running():
                self.stop(reason=reason)
            pid = self.start()
            self._clear_error()
            return pid
        except SupervisorError as exc:
            self._record_error("restart", str(exc))
            raise

    def adopt(self) -> bool:
        """Try to adopt an already-running helix.

        Two-stage:
            1. Stored PID in state file → is_running() verifies and returns
            2. Orphan scan → find any helix on the configured port and
               adopt it by writing its PID to the state file

        Returns True if a helix was adopted by either path, False
        otherwise.
        """
        # Stage 1: stored PID
        if self.is_running():
            log.info(
                "Adopted existing helix via state file (pid=%d, port=%d)",
                self.store.state.helix_pid, self.store.state.helix_port,
            )
            self._owns_helix_process = False
            return True

        # Stage 2: orphan scan
        orphan_pid = self.find_orphan_helix()
        if orphan_pid is None:
            return False

        # Reconstruct the expected command line so is_running() will
        # validate it. We pull the actual cmdline from psutil rather
        # than synthesizing it — the orphan may have been launched
        # with a slightly different argv order than our _command().
        try:
            psutil = self._get_psutil()
            proc = psutil.Process(orphan_pid)
            cmdline = proc.cmdline()
        except Exception:
            cmdline = self._command()

        self.store.set_helix(
            pid=orphan_pid,
            command=cmdline,
            port=self.helix_port,
        )
        self._owns_helix_process = False
        log.info(
            "Adopted orphan helix on %s:%d (pid=%d)",
            self.helix_host, self.helix_port, orphan_pid,
        )
        return True

    # ── internals ──────────────────────────────────────────────────

    def _cwd(self) -> Optional[str]:
        """Where to run helix from — default is the helix-context repo root if
        we're inside it, else None (use inherited cwd)."""
        try:
            here = Path(__file__).resolve()
            # helix_context/launcher/supervisor.py → helix-context root
            candidate = here.parent.parent.parent
            if (candidate / "pyproject.toml").exists():
                return str(candidate)
        except Exception:
            pass
        return None

    def _announce_restart(self, reason: str, expected_downtime_s: int) -> None:
        """Best-effort announce via helix /admin/announce_restart."""
        url = f"http://{self.helix_host}:{self.helix_port}/admin/announce_restart"
        payload = {
            "actor": "launcher",
            "reason": reason,
            "expected_downtime_s": expected_downtime_s,
        }
        try:
            httpx.post(url, json=payload, timeout=2.0)
        except Exception as exc:
            log.warning("Announce restart failed (continuing anyway): %s", exc)

    def _kill_tree(self, pid: int) -> None:
        """Kill the entire process tree rooted at pid, cross-platform."""
        psutil = self._get_psutil()

        if _is_windows():
            # taskkill /F /T is the reliable path on Windows.
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

        # POSIX: send SIGTERM to the process group, then SIGKILL after grace.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            log.warning("SIGTERM to pgid failed for pid %d", pid, exc_info=True)

        # Grace period, then SIGKILL if still alive.
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

    def _wait_for_ready(self, timeout: float = 30.0) -> None:
        """Poll GET /stats until helix responds."""
        url = f"http://{self.helix_host}:{self.helix_port}/stats"
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
        raise StartupTimeout(
            f"helix did not become ready within {timeout}s (last_error: {last_error})"
        )

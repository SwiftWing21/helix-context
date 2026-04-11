"""
Bridge — Shared memory layer between AI assistants.

Creates a file-based protocol that any AI assistant (Claude, Gemini, etc.)
can read and write to share context through the Agentome genome.

Architecture:
    ~/.helix/shared/          — shared memory directory
        inbox/                — files TO ingest (any assistant drops files here)
        outbox/               — genome context snapshots (assistants read from here)
        signals/              — lightweight status signals between assistants
        SHARED_CONTEXT.md     — always-current genome summary for instruction files

    The bridge watches inbox/ and auto-ingests new files into the genome.
    It periodically snapshots the genome health + recent genes into outbox/.
    Signals allow lightweight coordination ("I'm ingesting", "query X").

Usage:
    from helix_context.bridge import AgentBridge

    bridge = AgentBridge()
    bridge.write_signal("ingesting", {"files": 1500, "eta_min": 30})
    bridge.drop_to_inbox("fact: the port is 11437", source="gemini")
    bridge.update_shared_context(genome_stats)

Integration:
    - Claude Code: reads SHARED_CONTEXT.md via /helix skill
    - Gemini Code Assist: reads SHARED_CONTEXT.md via GEMINI.md include
    - Any agent: drops files into inbox/ for genome ingestion
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("helix.bridge")

# Default shared directory
DEFAULT_SHARED_DIR = os.path.expanduser("~/.helix/shared")


class AgentBridge:
    """File-based memory bridge between AI assistants."""

    def __init__(
        self,
        shared_dir: Optional[str] = None,
        helix_base_url: str = "http://127.0.0.1:11437",
        http_timeout: float = 5.0,
    ):
        self.shared_dir = Path(shared_dir or DEFAULT_SHARED_DIR)
        self.inbox = self.shared_dir / "inbox"
        self.outbox = self.shared_dir / "outbox"
        self.signals = self.shared_dir / "signals"

        # Create directory structure
        for d in [self.inbox, self.outbox, self.signals]:
            d.mkdir(parents=True, exist_ok=True)

        # Session registry HTTP client config (item 8 of SESSION_REGISTRY.md).
        # Used by register_participant / heartbeat / list_sessions /
        # recent_by_handle / ingest. Defaults target the standard local
        # helix server but are overridable for testing or remote use.
        self.helix_base_url = helix_base_url.rstrip("/")
        self.http_timeout = http_timeout
        self._participant_id: Optional[str] = None
        self._registered_handle: Optional[str] = None
        self._registered_party_id: Optional[str] = None
        self._heartbeat_interval_s: float = 30.0
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop: threading.Event = threading.Event()

        log.info("AgentBridge initialized at %s", self.shared_dir)

    # ── Inbox: receive content from other assistants ──────────────

    def drop_to_inbox(
        self,
        content: str,
        source: str = "unknown",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Drop content into the inbox for genome ingestion.
        Any assistant can call this to share knowledge.
        """
        if filename is None:
            filename = f"{source}_{int(time.time())}.md"
        path = self.inbox / filename
        path.write_text(content, encoding="utf-8")
        log.info("Inbox: %s dropped %s (%d chars)", source, filename, len(content))
        return path

    def collect_inbox(self) -> List[Dict]:
        """
        Collect all files from inbox for ingestion.
        Returns list of {path, content, source} dicts.
        Files are removed after collection.
        """
        items = []
        for f in sorted(self.inbox.iterdir()):
            if f.is_file() and f.suffix in (".md", ".txt", ".json"):
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                    source = f.stem.split("_")[0]  # Extract source from filename
                    items.append({
                        "path": str(f),
                        "content": content,
                        "source": source,
                    })
                    f.unlink()  # Remove after collection
                except Exception:
                    log.warning("Failed to collect inbox file: %s", f, exc_info=True)
        return items

    # ── Outbox: publish genome state for other assistants ─────────

    def update_shared_context(self, stats: Dict, recent_queries: Optional[List] = None) -> Path:
        """
        Write SHARED_CONTEXT.md — a live summary of the genome state
        that other assistants can read from their instruction files.
        """
        lines = [
            "# Helix Genome — Shared Context",
            f"*Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Genome Stats",
            f"- **Genes:** {stats.get('total_genes', 0)}",
            f"- **Compression:** {stats.get('compression_ratio', 0):.1f}x",
            f"- **Open:** {stats.get('open', 0)} | Euchromatin: {stats.get('euchromatin', 0)} | Heterochromatin: {stats.get('heterochromatin', 0)}",
            "",
        ]

        health = stats.get("health", {})
        if health:
            lines.extend([
                "## Health",
                f"- **Queries logged:** {health.get('total_queries', 0)}",
                f"- **Avg ellipticity:** {health.get('avg_ellipticity', 0):.3f}",
                f"- **Status distribution:** {health.get('status_counts', {})}",
                "",
            ])

        config = stats.get("config", {})
        if config:
            lines.extend([
                "## Config",
                f"- **Decoder mode:** {config.get('decoder_mode', '?')}",
                f"- **Max genes/turn:** {config.get('max_genes_per_turn', '?')}",
                f"- **Expression budget:** {config.get('expression_tokens', '?')} tokens",
                "",
            ])

        lines.extend([
            "## How to Use",
            "- **Query:** POST http://127.0.0.1:11437/context with `{\"query\": \"...\", \"decoder_mode\": \"none\"}`",
            "- **Ingest:** Drop .md/.txt files into `~/.helix/shared/inbox/`",
            "- **Signal:** Write JSON to `~/.helix/shared/signals/<name>.json`",
            "",
            "## Inbox Protocol",
            "Any AI assistant can share knowledge by writing files to:",
            f"`{self.inbox}`",
            "",
            "Files are auto-ingested into the genome on the next cycle.",
            "Filename format: `<source>_<timestamp>.md` (e.g., `gemini_1712700000.md`)",
        ])

        path = self.shared_dir / "SHARED_CONTEXT.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Updated SHARED_CONTEXT.md (%d genes)", stats.get("total_genes", 0))
        return path

    # ── Signals: lightweight coordination ─────────────────────────

    def write_signal(self, name: str, data: Dict) -> Path:
        """
        Write a signal for other assistants to read.

        Atomic via write-to-temp + os.replace — readers never see a
        partially-written file. Works on both POSIX and Windows (NT
        kernel provides atomic rename semantics).
        """
        data["timestamp"] = time.time()
        data["timestamp_human"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = self.signals / f"{name}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic rename
        return path

    def read_signal(self, name: str) -> Optional[Dict]:
        """Read a signal from another assistant."""
        path = self.signals / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def clear_signal(self, name: str) -> None:
        """Clear a signal."""
        path = self.signals / f"{name}.json"
        if path.exists():
            path.unlink()

    def list_signals(self) -> Dict[str, Dict]:
        """List all active signals."""
        result = {}
        for f in self.signals.iterdir():
            if f.suffix == ".json":
                try:
                    result[f.stem] = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return result

    # ── Server restart protocol ───────────────────────────────────

    def announce_restart(
        self,
        reason: str,
        actor: str,
        expected_downtime_s: int = 30,
        pid: Optional[int] = None,
    ) -> Path:
        """
        Announce an intentional server restart to other sessions.

        Writes a 'server_state' signal with state='restarting' so that
        observers polling the signal file can distinguish an intentional
        restart from an unexpected crash. Call this BEFORE killing the
        server process.

        Recommended pattern:
            bridge.announce_restart("swapping ribosome model", actor="laude")
            time.sleep(0.75)  # let filesystem flush + observers see it
            # ... trigger the actual restart ...

        Args:
            reason: short human-readable reason (e.g., "VRAM rescue")
            actor: who initiated (e.g., "laude", "raude", "human")
            expected_downtime_s: observer's TTL budget for this restart
            pid: optional — the dying process's PID (for log correlation)

        Returns:
            Path to the written signal file.
        """
        return self.write_signal("server_state", {
            "state": "restarting",
            "actor": actor,
            "reason": reason,
            "pid": pid,
            "expected_downtime_s": expected_downtime_s,
            "phase": "shutting_down",
        })

    def read_server_state(self) -> Optional[Tuple[Dict, bool, float]]:
        """
        Read the current server_state signal with TTL-aware staleness check.

        Returns None if no signal exists. Otherwise returns a 3-tuple:
            (signal_dict, is_stale, age_s)

        Where:
          - signal_dict: the raw signal as written (unmutated)
          - is_stale: bool — True if the announcement is older than its TTL
          - age_s: float — seconds since the signal was written

        Staleness rules:
          - state='running'    → never stale
          - state='restarting' → stale if age > expected_downtime_s + 15
          - state='stopped'    → same window
          - unknown state      → 5-minute TTL

        Usage:
            result = bridge.read_server_state()
            if result is None:
                handle_crash()  # no signal, legacy server or genuine outage
            else:
                signal, is_stale, age_s = result
                if signal["state"] == "restarting" and not is_stale:
                    print(f"Waiting for {signal['actor']}: {signal['reason']}")
        """
        signal = self.read_signal("server_state")
        if signal is None:
            return None

        state = signal.get("state", "unknown")
        ts = signal.get("timestamp", 0)
        budget = signal.get("expected_downtime_s", 30)
        age_s = time.time() - ts

        if state == "running":
            is_stale = False
        elif state in ("restarting", "stopped"):
            is_stale = age_s > (budget + 15)
        else:
            is_stale = age_s > 300  # unknown state → 5min TTL

        return signal, is_stale, round(age_s, 1)

    # ── Session registry HTTP client (item 8 of SESSION_REGISTRY.md) ──
    #
    # Convenience methods that talk to the helix session registry over
    # HTTP. After register_participant() the bridge remembers the
    # participant_id so subsequent heartbeat() / ingest() calls don't
    # need it as an argument. start_auto_heartbeat() spins up a daemon
    # thread that refreshes liveness every heartbeat_interval_s, which
    # solves the "Taude goes stale unless I manually re-register"
    # problem for long-running interactive sessions.
    #
    # All methods soft-fail with logging — they never raise on network
    # errors. Returns None / False on failure so callers can decide
    # whether to retry or escalate.

    def _http_post(
        self,
        path: str,
        json_body: Optional[Dict] = None,
    ) -> Optional[Dict]:
        try:
            import httpx  # local import — keeps the bridge importable without httpx for file-only use
        except ImportError:
            log.warning("httpx not installed — bridge HTTP methods unavailable")
            return None
        url = f"{self.helix_base_url}{path}"
        try:
            resp = httpx.post(url, json=json_body or {}, timeout=self.http_timeout)
            if resp.status_code >= 400:
                log.warning(
                    "POST %s returned %d: %s",
                    path, resp.status_code, resp.text[:200],
                )
                return None
            return resp.json() if resp.content else {}
        except Exception as exc:
            log.warning("POST %s failed: %s", path, exc)
            return None

    def _http_get(
        self,
        path: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        try:
            import httpx
        except ImportError:
            log.warning("httpx not installed — bridge HTTP methods unavailable")
            return None
        url = f"{self.helix_base_url}{path}"
        try:
            resp = httpx.get(url, params=params, timeout=self.http_timeout)
            if resp.status_code >= 400:
                log.warning(
                    "GET %s returned %d: %s",
                    path, resp.status_code, resp.text[:200],
                )
                return None
            return resp.json() if resp.content else {}
        except Exception as exc:
            log.warning("GET %s failed: %s", path, exc)
            return None

    def register_participant(
        self,
        party_id: str,
        handle: str,
        workspace: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        display_name: Optional[str] = None,
        start_auto_heartbeat: bool = False,
    ) -> Optional[str]:
        """Register a participant with the helix session registry.

        On success, the bridge remembers the participant_id internally
        so heartbeat() and ingest() can use it without re-passing.

        If ``start_auto_heartbeat=True``, also spawns a daemon thread
        that refreshes liveness automatically every
        ``heartbeat_interval_s`` (returned by the server, default 30s).
        Solves the "long-idle session goes stale" problem.

        Returns the participant_id on success, or None on failure.
        """
        body: Dict[str, Any] = {
            "party_id": party_id,
            "handle": handle,
        }
        if workspace is not None:
            body["workspace"] = workspace
        if capabilities is not None:
            body["capabilities"] = capabilities
        if display_name is not None:
            body["display_name"] = display_name
        try:
            body["pid"] = os.getpid()
        except Exception:
            pass

        result = self._http_post("/sessions/register", json_body=body)
        if not result or "participant_id" not in result:
            return None

        self._participant_id = result["participant_id"]
        self._registered_handle = handle
        self._registered_party_id = party_id
        self._heartbeat_interval_s = float(result.get("heartbeat_interval_s", 30.0))
        log.info(
            "Registered as %s (party=%s, participant_id=%s)",
            handle, party_id, self._participant_id,
        )

        if start_auto_heartbeat:
            self.start_auto_heartbeat()
        return self._participant_id

    @property
    def participant_id(self) -> Optional[str]:
        """The currently-registered participant_id, or None if not registered."""
        return self._participant_id

    def heartbeat(self) -> bool:
        """Refresh liveness for the registered participant.

        Returns True on success, False if not registered or the call
        failed. On 404 (participant unknown — server may have lost
        state), clears the local participant_id so the caller can
        re-register.
        """
        if not self._participant_id:
            return False
        try:
            import httpx
        except ImportError:
            return False
        url = f"{self.helix_base_url}/sessions/{self._participant_id}/heartbeat"
        try:
            resp = httpx.post(url, timeout=self.http_timeout)
            if resp.status_code == 404:
                log.warning("Heartbeat: participant_id unknown to server, clearing local state")
                self._participant_id = None
                return False
            if resp.status_code >= 400:
                log.warning("Heartbeat returned %d", resp.status_code)
                return False
            return True
        except Exception as exc:
            log.warning("Heartbeat failed: %s", exc)
            return False

    def list_sessions(
        self,
        party_id: Optional[str] = None,
        status: str = "active",
    ) -> Optional[List[Dict]]:
        """List participants. Returns the participants list, or None on failure."""
        params: Dict[str, str] = {"status": status}
        if party_id:
            params["party_id"] = party_id
        result = self._http_get("/sessions", params=params)
        if result is None:
            return None
        return result.get("participants", [])

    def recent_by_handle(
        self,
        handle: str,
        limit: int = 10,
        party_id: Optional[str] = None,
    ) -> Optional[List[Dict]]:
        """Fetch recent genes authored by a handle, chronologically.

        Uses the BM25-bypass /sessions/{handle}/recent path so short
        broadcasts surface even when the genome holds a much larger
        unrelated corpus. Returns the genes list, or None on failure.
        """
        params: Dict[str, str] = {"limit": str(int(limit))}
        if party_id:
            params["party_id"] = party_id
        result = self._http_get(f"/sessions/{handle}/recent", params=params)
        if result is None:
            return None
        return result.get("genes", [])

    def ingest(
        self,
        content: str,
        content_type: str = "text",
        metadata: Optional[Dict] = None,
        attribute: bool = True,
    ) -> Optional[Dict]:
        """Ingest content into the genome via the helix /ingest endpoint.

        If ``attribute=True`` (default) and a participant has been
        registered, the resulting genes are tagged via the session
        registry attribution path so they can be retrieved later via
        recent_by_handle().

        Returns the response dict (gene_ids, count, attributed) on
        success, None on failure.
        """
        body: Dict[str, Any] = {
            "content": content,
            "content_type": content_type,
        }
        if metadata is not None:
            body["metadata"] = metadata
        if attribute and self._participant_id:
            body["participant_id"] = self._participant_id
        return self._http_post("/ingest", json_body=body)

    # ── Auto-heartbeat daemon thread ──────────────────────────────

    def start_auto_heartbeat(self, interval_s: Optional[float] = None) -> None:
        """Spawn a daemon thread that calls heartbeat() periodically.

        Idempotent — calling twice does not start a second thread. If
        ``interval_s`` is omitted, uses the value returned by the
        server at registration time (default 30s).

        The thread terminates automatically when:
          - stop_auto_heartbeat() is called
          - The participant_id is cleared (e.g., by a 404 from the server)
          - The process exits (it's a daemon thread)
        """
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        if interval_s is not None:
            self._heartbeat_interval_s = float(interval_s)

        self._heartbeat_stop.clear()

        def _worker() -> None:
            log.info(
                "Auto-heartbeat started for participant_id=%s (interval=%.1fs)",
                self._participant_id, self._heartbeat_interval_s,
            )
            while not self._heartbeat_stop.is_set():
                # Sleep first so the thread doesn't double-heartbeat right
                # after register_participant (which already counts as fresh).
                if self._heartbeat_stop.wait(timeout=self._heartbeat_interval_s):
                    break
                if self._participant_id is None:
                    log.info("Auto-heartbeat: participant_id cleared, stopping")
                    break
                ok = self.heartbeat()
                if not ok and self._participant_id is None:
                    # Heartbeat cleared the id (server returned 404) — stop
                    break

        self._heartbeat_thread = threading.Thread(
            target=_worker,
            daemon=True,
            name="bridge-heartbeat",
        )
        self._heartbeat_thread.start()

    def stop_auto_heartbeat(self, timeout: float = 2.0) -> None:
        """Signal the auto-heartbeat thread to stop and join it briefly.

        Idempotent. Safe to call even if no thread was started.
        """
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=timeout)
        self._heartbeat_thread = None

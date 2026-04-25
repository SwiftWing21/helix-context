"""
State collector — builds the `/api/state` payload the dashboard renders.

Aggregates data from:
    - Supervisor (helix process liveness, pid, uptime)
    - Helix HTTP endpoints: /stats, /sessions, /health
    - Ollama: /api/ps (optional, soft-fails if unreachable)

All HTTP calls use short timeouts. Any upstream failure produces a
"data not available" null in the corresponding field rather than
raising — the dashboard is expected to hide panels whose data is empty.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from .supervisor import HelixSupervisor

log = logging.getLogger("helix.launcher.collector")


class StateCollector:
    """Builds the launcher-side state snapshot by polling helix + ollama."""

    def __init__(
        self,
        supervisor: HelixSupervisor,
        ollama_base_url: str = "http://127.0.0.1:11434",
        http_timeout: float = 4.0,
    ) -> None:
        self.supervisor = supervisor
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.http_timeout = http_timeout

    def collect(self) -> Dict[str, Any]:
        """Return the full launcher state dict. Never raises."""
        helix_state = self._collect_helix_process()
        state: Dict[str, Any] = {"helix": helix_state}

        if not helix_state["running"]:
            return state

        base = f"http://{self.supervisor.helix_host}:{self.supervisor.helix_port}"
        client = httpx.Client(base_url=base, timeout=self.http_timeout)
        health_seen = False
        try:
            stats = self._safe_get_json(client, "/stats")
            if stats:
                state["genes"] = self._genes_panel(stats)

            sessions = self._safe_get_json(client, "/sessions", params={"status": "all"})
            if sessions and sessions.get("participants"):
                participants = sessions["participants"]
                state["parties"] = self._parties_panel(participants)
                state["participants"] = self._participants_panel(participants)
                disconnected_agents = self._disconnected_agents_panel(participants)
                if disconnected_agents:
                    state["disconnected_agents"] = disconnected_agents
                state["all_agents"] = self._all_agents_panel(participants)

            health = self._safe_get_json(client, "/health")
            if health:
                health_seen = True
                state["helix"]["ribosome"] = health.get("ribosome")
                checks = health.get("checks", {}) or {}
                state["helix"]["availability"] = (
                    "available" if health.get("status") == "ok" else "degraded"
                )
                if health.get("status") == "ok":
                    state["helix"]["next_action"] = (
                        "Helix is healthy. Query it through MCP or the OpenAI-compatible endpoint."
                    )
                elif checks.get("upstream_ready") is False:
                    state["helix"]["next_action"] = (
                        "Start or fix the upstream model server, then use Restart if Helix stays degraded."
                    )
                elif checks.get("genome_ready") is False:
                    state["helix"]["next_action"] = (
                        "Inspect the local genome database, then use Restart if Helix stays degraded."
                    )
                else:
                    state["helix"]["next_action"] = (
                        "Helix responded unexpectedly. Restart it from the launcher UI."
                    )
                state["helix"]["health_message"] = health.get("message")

            components = self._safe_get_json(client, "/admin/components")
            if components and components.get("components"):
                tools = self._tools_panel(components)
                if tools:
                    state["tools"] = tools

            tokens = self._safe_get_json(client, "/metrics/tokens")
            if tokens and (tokens.get("session") or tokens.get("lifetime")):
                state["tokens"] = self._tokens_panel(tokens)
        finally:
            client.close()

        if not health_seen:
            state["helix"]["availability"] = "degraded"
            state["helix"]["next_action"] = (
                "The Helix process exists but did not answer its health endpoints. "
                "Restart it from the launcher UI."
            )

        models = self._collect_models()
        if models:
            state["models"] = models

        return state

    # ── helix process ──────────────────────────────────────────────

    def _collect_helix_process(self) -> Dict[str, Any]:
        running = self.supervisor.is_running()
        out: Dict[str, Any] = {
            "running": running,
            "port": self.supervisor.helix_port,
            "host": self.supervisor.helix_host,
            "availability": "available" if running else "unavailable",
        }
        if running:
            out["pid"] = self.supervisor.get_pid()
            out["uptime_s"] = round(self.supervisor.get_uptime_s() or 0, 1)
            st = self.supervisor.store.state
            out["last_restart_reason"] = st.last_restart_reason
            out["last_restart_at"] = st.last_restart_at
            out["next_action"] = "Wait for the health probe or use Restart if Helix looks stuck."
        else:
            # When helix is down, surface an orphan warning if one is
            # detected on the configured port — it's almost certainly
            # the user's real problem.
            try:
                orphan_pid = self.supervisor.find_orphan_helix()
                if orphan_pid is not None:
                    out["orphan_pid"] = orphan_pid
            except Exception:
                log.debug("Orphan scan failed", exc_info=True)
            out["next_action"] = "Click Start to launch Helix."

        # Last error — present whether helix is up or down.
        last_error = self.supervisor.get_last_error()
        if last_error is not None:
            out["last_error"] = last_error

        # Paths — static information the user wants visible for debugging.
        try:
            state_path = self.supervisor.store.path
            out["paths"] = {
                "state_file": str(state_path),
                "helix_log": str(self.supervisor.helix_log_path),
            }
        except Exception:
            pass

        return out

    # ── genes ──────────────────────────────────────────────────────

    def _genes_panel(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "total": stats.get("total_genes", 0),
            "raw_chars": stats.get("total_chars_raw", 0),
            "compressed_chars": stats.get("total_chars_compressed", 0),
            "compression_ratio": round(stats.get("compression_ratio", 1.0), 2),
        }

    # ── parties + participants ────────────────────────────────────

    def _parties_panel(self, participants: List[Dict[str, Any]]) -> Dict[str, Any]:
        party_ids = sorted({p["party_id"] for p in participants})
        return {
            "count": len(party_ids),
            "party_ids": party_ids,
        }

    def _participants_panel(self, participants: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Main panel is identity-oriented: collapse duplicate rows that share
        # the same operator-visible identity and show how many live sessions
        # are hiding underneath that identity.
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for participant in participants:
            key = self._identity_key(participant)
            group = grouped.get(key)
            if group is None:
                group = {
                    "handle": participant["handle"],
                    "party_id": participant["party_id"],
                    "workspace": participant.get("workspace"),
                    "identifier": self._identity_label(participant),
                    "status": participant["status"],
                    "last_seen_s_ago": participant["last_seen_s_ago"],
                    "session_count": 0,
                    "active_session_count": 0,
                }
                grouped[key] = group

            group["session_count"] += 1
            if participant.get("status") == "active":
                group["active_session_count"] += 1
            if self._status_rank(participant.get("status")) < self._status_rank(group["status"]):
                group["status"] = participant["status"]
            if participant.get("last_seen_s_ago", 0) < group["last_seen_s_ago"]:
                group["last_seen_s_ago"] = participant["last_seen_s_ago"]

        active_entries = [
            entry for entry in grouped.values()
            if entry.get("status") == "active"
        ]
        entries = sorted(active_entries, key=lambda x: x["last_seen_s_ago"])
        return {
            "count": len(entries),
            "identity_total_count": len(grouped),
            "total_count": len(participants),
            "entries": entries,
        }

    def _all_agents_panel(self, participants: List[Dict[str, Any]]) -> Dict[str, Any]:
        entries = []
        for participant in sorted(
            participants,
            key=lambda item: (
                self._status_rank(item.get("status")),
                item.get("last_seen_s_ago", 0),
            ),
        ):
            participant_id = str(participant.get("participant_id", ""))
            entries.append(
                {
                    "handle": participant.get("handle"),
                    "party_id": participant.get("party_id"),
                    "workspace": participant.get("workspace"),
                    "status": participant.get("status"),
                    "last_seen_s_ago": participant["last_seen_s_ago"],
                    "participant_id": participant_id,
                    "participant_id_short": participant_id[:8],
                    "identifier": self._identity_label(participant),
                }
            )
        return {
            "count": len(entries),
            "active_count": sum(1 for item in entries if item["status"] == "active"),
            "entries": entries,
        }

    def _disconnected_agents_panel(self, participants: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        entries = []
        for participant in sorted(
            participants,
            key=lambda item: (
                self._status_rank(item.get("status")),
                item.get("last_seen_s_ago", 0),
            ),
        ):
            status = str(participant.get("status") or "").strip().lower()
            if status == "active":
                continue
            participant_id = str(participant.get("participant_id", ""))
            entries.append(
                {
                    "handle": participant.get("handle"),
                    "party_id": participant.get("party_id"),
                    "workspace": participant.get("workspace"),
                    "status": status or "unknown",
                    "last_seen_s_ago": participant.get("last_seen_s_ago", 0),
                    "participant_id": participant_id,
                    "participant_id_short": participant_id[:8],
                    "identifier": self._identity_label(participant),
                }
            )
        if not entries:
            return None
        return {
            "count": len(entries),
            "entries": entries,
        }

    def _identity_key(self, participant: Dict[str, Any]) -> tuple:
        return (
            str(participant.get("party_id", "")).strip().lower(),
            str(participant.get("handle", "")).strip().lower(),
            str(participant.get("workspace") or "").strip().lower(),
        )

    def _identity_label(self, participant: Dict[str, Any]) -> str:
        workspace = participant.get("workspace")
        if workspace:
            return f"{participant['party_id']} · {workspace}"
        return str(participant.get("party_id", ""))

    def _status_rank(self, status: Optional[str]) -> int:
        order = {
            "active": 0,
            "idle": 1,
            "stale": 2,
            "gone": 3,
        }
        return order.get(str(status or "").strip().lower(), 99)

    # ── tokens ─────────────────────────────────────────────────────

    def _tokens_panel(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        """Project /metrics/tokens into the launcher panel shape.

        Combines exact + estimated buckets into a single 'total' so the
        panel doesn't need to know about the distinction. Both raw
        buckets are still passed through for callers who care.
        """
        session = tokens.get("session", {}) or {}
        lifetime = tokens.get("lifetime", {}) or {}

        def _combined_total(bucket: Dict[str, Any]) -> int:
            return int(bucket.get("total", 0)) + int(bucket.get("estimated_total", 0))

        return {
            "session": {
                "total": _combined_total(session),
                "exact": int(session.get("total", 0)),
                "estimated": int(session.get("estimated_total", 0)),
            },
            "lifetime": {
                "total": _combined_total(lifetime),
                "exact": int(lifetime.get("total", 0)),
                "estimated": int(lifetime.get("estimated_total", 0)),
            },
        }

    # ── models ─────────────────────────────────────────────────────

    def _tools_panel(self, components: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Project /admin/components into the launcher tools panel.

        The launcher already has dedicated Helix health and model panels.
        Hide the ribosome here so Ollama/model activity does not get
        mistaken for a separate operator-facing tool.
        """
        entries = [
            component
            for component in components.get("components", [])
            if self._is_operator_tool(component)
        ]
        if not entries:
            return None
        return {
            "count": len(entries),
            "entries": entries,
            "last_activity_s_ago": components.get("last_activity_s_ago"),
        }

    def _is_operator_tool(self, component: Dict[str, Any]) -> bool:
        name = str(component.get("name", "")).strip().lower()
        return name != "ribosome"

    def _collect_models(self) -> Optional[Dict[str, Any]]:
        """Pull currently-loaded models from Ollama. Soft-fails."""
        try:
            resp = httpx.get(
                f"{self.ollama_base_url}/api/ps",
                timeout=self.http_timeout,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            loaded = []
            for m in data.get("models", []):
                name = m.get("name", "unknown")
                size = m.get("size", 0)
                loaded.append({
                    "name": name,
                    "size_mb": round(size / (1024 * 1024), 1) if size else None,
                    "source": "ollama",
                })
            if not loaded:
                return None
            return {"loaded": loaded}
        except Exception:
            log.debug("Ollama /api/ps unreachable", exc_info=True)
            return None

    # ── helpers ────────────────────────────────────────────────────

    def _safe_get_json(
        self,
        client: httpx.Client,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            resp = client.get(path, params=params)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            log.debug("GET %s failed", path, exc_info=True)
            return None

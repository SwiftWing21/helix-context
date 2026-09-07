"""
State collector — builds the `/api/state` payload the dashboard renders.

Aggregates data from:
    - Supervisor (cymatix process liveness, pid, uptime)
    - Cymatix HTTP endpoints: /stats, /sessions, /health
    - Ollama: /api/ps (optional, soft-fails if unreachable)

All HTTP calls use short timeouts. Any upstream failure produces a
"data not available" null in the corresponding field rather than
raising — the dashboard is expected to hide panels whose data is empty.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from .graph_summary import GraphSummaryCache
from .supervisor import CymatixSupervisor
from .host_labels import compose_label, host_pretty, vendor_pretty
from .model_labels import model_pretty

log = logging.getLogger("cymatix.launcher.collector")


def _coerce_value(value: Any) -> str:
    """Render a config value as a stable string for the dashboard.

    Booleans become "on"/"off", floats keep three decimals, everything
    else falls back to repr-style stringification.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
    if value is None:
        return "—"
    return str(value)


def _switchboard_summary(settings: List[Dict[str, Any]]) -> str:
    """One-line chip summary for the dashboard: 'rrf · dense:on · rerank:off · ...'."""
    pick = {s["label"]: s["value"] for s in settings}
    chips = [
        pick.get("fusion_mode", "?"),
        f"dense:{pick.get('dense_embedding_enabled', '?')}",
        f"splade:{pick.get('splade_enabled', '?')}",
        f"rerank:{pick.get('rerank_enabled', '?')}",
        f"decoder:{pick.get('decoder_mode', '?')}",
    ]
    return " · ".join(chips)


def _build_tooltip(participant: Dict[str, Any]) -> Dict[str, str]:
    """Compose the tooltip field bundle from a participant dict.

    Each label is either a pretty-mapped value or an explicit placeholder
    that hints at the cause:
      - model_label: "Not announced" when model_id is NULL
      - ide_label:   "Not detected" when ide_detected is NULL
      - agent_kind_label: "Not set"  when agent_kind is NULL
    The ide_detection_via line tells the operator how (or whether) the
    IDE value was obtained: "env:VSCODE_PID", "agent_override",
    "no_match", "explicit:CYMATIX_MCP_HOST".

    ide_label fallback chain (in order):
      1. ide_detected — populated by the adapter at register time, or by
         an agent's ide_override via cymatix_announce. Authoritative when present.
      2. mcp_host — PR #26's column. Sessions registered after PR #26 but
         before this change will have ide_detected=NULL but mcp_host set
         from CYMATIX_MCP_HOST env. Fall back so those sessions' chips
         still render correctly until they re-register.
      3. "Not detected" — explicit placeholder when neither column
         carries a value, so the tooltip surfaces the gap rather than
         hiding the row.

    There is no equivalent fallback chain for model_id (the column is
    new in this change; there is no prior column to fall back to).
    """
    return {
        "model_label": model_pretty(participant.get("model_id")) or "Not announced",
        "ide_label": (
            host_pretty(participant.get("ide_detected"))
            or host_pretty(participant.get("mcp_host"))   # PR #26 backward-compat fallback
            or "Not detected"
        ),
        "agent_kind_label": vendor_pretty(participant.get("agent_kind")) or "Not set",
        "ide_detection_via": participant.get("ide_detection_via") or "no_match",
    }


class StateCollector:
    """Builds the launcher-side state snapshot by polling cymatix + ollama."""

    def __init__(
        self,
        supervisor: CymatixSupervisor,
        ollama_base_url: str = "http://127.0.0.1:11434",
        http_timeout: float = 4.0,
        update_checker: Optional[Any] = None,
        graph_summary_cache: Optional[GraphSummaryCache] = None,
    ) -> None:
        self.supervisor = supervisor
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.http_timeout = http_timeout
        self.update_checker = update_checker
        self.graph_summary_cache = graph_summary_cache or GraphSummaryCache()

    def collect(self) -> Dict[str, Any]:
        """Return the full launcher state dict. Never raises."""
        cymatix_state = self._collect_cymatix_process()
        state: Dict[str, Any] = {"cymatix": cymatix_state}
        if self.update_checker is not None:
            state["update"] = self.update_checker.check().as_dict()

        # The switchboard + database panels are useful even when cymatix is
        # stopped — they describe the *next* run's planned configuration.
        # Pipeline/runs panels need a live cymatix so they stay gated below.
        state["switchboard"] = self._switchboard_panel()
        state["database"] = self._database_panel()

        # The graph summary counts rows in the genome file, so it is
        # meaningful with cymatix stopped and belongs above the early
        # return too.
        summary = self._graph_summary_panel(state["database"])
        if summary is not None:
            state["graph_summary"] = summary

        if not cymatix_state["running"]:
            return state

        base = f"http://{self.supervisor.cymatix_host}:{self.supervisor.cymatix_port}"
        client = httpx.Client(base_url=base, timeout=self.http_timeout)
        health_seen = False
        endpoint_seen = False
        try:
            stats = self._safe_get_json(client, "/stats")
            if stats:
                endpoint_seen = True
                self._copy_cymatix_version(state["cymatix"], stats)
                state["genes"] = self._genes_panel(stats)

            sessions = self._safe_get_json(client, "/sessions", params={"status": "all"})
            if sessions and sessions.get("participants"):
                endpoint_seen = True
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
                endpoint_seen = True
                self._copy_cymatix_version(state["cymatix"], health)
                state["cymatix"]["ribosome"] = health.get("ribosome")
                checks = health.get("checks", {}) or {}
                state["cymatix"]["availability"] = (
                    "available" if health.get("status") == "ok" else "degraded"
                )
                if health.get("status") == "ok":
                    state["cymatix"]["next_action"] = (
                        "Cymatix is healthy. Query it through MCP or the OpenAI-compatible endpoint."
                    )
                elif checks.get("upstream_ready") is False:
                    state["cymatix"]["next_action"] = (
                        "Start or fix the upstream model server, then use Restart if Cymatix stays degraded."
                    )
                elif checks.get("genome_ready") is False:
                    state["cymatix"]["next_action"] = (
                        "Inspect the local genome database, then use Restart if Cymatix stays degraded."
                    )
                else:
                    state["cymatix"]["next_action"] = (
                        "Cymatix responded unexpectedly. Restart it from the launcher UI."
                    )
                state["cymatix"]["health_message"] = health.get("message")

            components = self._safe_get_json(client, "/admin/components")
            if components and components.get("components"):
                endpoint_seen = True
                tools = self._tools_panel(components)
                if tools:
                    state["tools"] = tools

            tokens = self._safe_get_json(client, "/metrics/tokens")
            if tokens and (tokens.get("session") or tokens.get("lifetime")):
                endpoint_seen = True
                state["tokens"] = self._tokens_panel(tokens)

            # Backfill the database panel with live information from the
            # running cymatix (the on-disk-discovery path can drift from the
            # process that's actually open). Adds an `active` block in the
            # panel so the dashboard can show "selected vs running".
            live_genome = self._safe_get_json(client, "/admin/genome")
            if live_genome:
                endpoint_seen = True
                state["database"]["live"] = self._live_genome(live_genome)

            # Pipeline viewer events. The ring is only present on a cymatix
            # that includes the launcher-companion code; an older cymatix
            # answers 404 and we skip the panel quietly.
            pipeline = self._safe_get_json(
                client, "/debug/pipeline/recent", params={"limit": 64},
            )
            if pipeline and pipeline.get("events") is not None:
                endpoint_seen = True
                state["pipeline"] = self._pipeline_panel(pipeline)
        finally:
            client.close()

        if not health_seen and endpoint_seen:
            state["cymatix"]["availability"] = "available"
            state["cymatix"]["next_action"] = (
                "Cymatix is responding; the launcher health endpoint has "
                "not answered yet (cold start, or an older build)."
            )
        elif not health_seen:
            state["cymatix"]["availability"] = "degraded"
            state["cymatix"]["next_action"] = (
                "The Cymatix process exists but did not answer its health endpoints. "
                "Restart it from the launcher UI."
            )

        models = self._collect_models()
        if models:
            state["models"] = models

        return state

    def _copy_cymatix_version(self, cymatix: Dict[str, Any], payload: Dict[str, Any]) -> None:
        version = payload.get("version") or payload.get("cymatix_version")
        if isinstance(version, str) and version.strip():
            cymatix["version"] = version.strip()

    # ── cymatix process ──────────────────────────────────────────────

    def _collect_cymatix_process(self) -> Dict[str, Any]:
        running = self.supervisor.is_running()
        out: Dict[str, Any] = {
            "running": running,
            "port": self.supervisor.cymatix_port,
            "host": self.supervisor.cymatix_host,
            "availability": "available" if running else "unavailable",
        }
        if running:
            out["pid"] = self.supervisor.get_pid()
            out["uptime_s"] = round(self.supervisor.get_uptime_s() or 0, 1)
            st = self.supervisor.store.state
            out["last_restart_reason"] = st.last_restart_reason
            out["last_restart_at"] = st.last_restart_at
            out["next_action"] = "Wait for the health probe or use Restart if Cymatix looks stuck."
        else:
            # When cymatix is down, surface an orphan warning if one is
            # detected on the configured port — it's almost certainly
            # the user's real problem.
            try:
                orphan_pid = self.supervisor.find_orphan_cymatix()
                if orphan_pid is not None:
                    out["orphan_pid"] = orphan_pid
            except Exception:
                log.debug("Orphan scan failed", exc_info=True)
            out["next_action"] = "Click Start to launch Cymatix."

        # Last error — present whether cymatix is up or down.
        last_error = self.supervisor.get_last_error()
        if last_error is not None:
            out["last_error"] = last_error

        # Paths — static information the user wants visible for debugging.
        try:
            state_path = self.supervisor.store.path
            out["paths"] = {
                "state_file": str(state_path),
                "cymatix_log": str(self.supervisor.cymatix_log_path),
            }
        except Exception:
            pass

        return out

    # ── documents ──────────────────────────────────────────────────────

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
                    "host_label": compose_label(
                        participant.get("agent_kind"),
                        participant.get("mcp_host"),
                    ),
                    "tooltip": _build_tooltip(participant),
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
                    "host_label": compose_label(
                        participant.get("agent_kind"),
                        participant.get("mcp_host"),
                    ),
                    "tooltip": _build_tooltip(participant),
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
                    "host_label": compose_label(
                        participant.get("agent_kind"),
                        participant.get("mcp_host"),
                    ),
                    "tooltip": _build_tooltip(participant),
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
        """Project /admin/components into the launcher components panel.

        The launcher already has dedicated Cymatix health and model panels.
        Hide the compressor here so Ollama/model activity does not get
        mistaken for a separate operator-facing tool.
        """
        all_components = components.get("components", [])
        entries = [
            component
            for component in all_components
            if self._is_operator_tool(component)
        ]
        if not entries:
            return None
        source_count = int(components.get("count", len(all_components)) or 0)
        return {
            "count": len(entries),
            "source_count": source_count,
            "hidden_count": max(0, source_count - len(entries)),
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

    # ── switchboard ────────────────────────────────────────────────

    def _switchboard_panel(self) -> Dict[str, Any]:
        """Snapshot the operationally-interesting retrieval/pipeline knobs.

        Reads `cymatix_context.config.load_config()` directly. Both the
        launcher and the supervised cymatix child resolve the same
        `cymatix.toml` from the same working directory, so this snapshot
        matches the live process. On any load failure the panel reports
        the error so the operator sees the gap.
        """
        try:
            from cymatix_context.config import load_config
            cfg = load_config()
        except Exception as exc:
            log.warning("Switchboard: load_config failed (%s)", exc)
            return {"error": str(exc), "settings": []}

        def _entry(label: str, value: Any, group: str, desc: str = "") -> Dict[str, Any]:
            return {
                "label": label,
                "value": _coerce_value(value),
                "group": group,
                "description": desc,
            }

        retrieval = cfg.retrieval
        budget = cfg.budget
        ingestion = cfg.ingestion
        classifier = getattr(cfg, "classifier", None)
        ribosome = cfg.ribosome

        settings: List[Dict[str, Any]] = [
            _entry("fusion_mode", retrieval.fusion_mode, "retrieval",
                   "additive | rrf — how per-tier scores combine"),
            _entry("dense_embedding_enabled", retrieval.dense_embedding_enabled,
                   "retrieval", "BGE-M3 dense recall participates in ranking"),
            _entry("ann_similarity_threshold", retrieval.ann_similarity_threshold,
                   "retrieval", "Minimum cosine for a dense hit to count"),
            _entry("pki_enabled", retrieval.pki_enabled, "retrieval",
                   "Tier-0 path_key_index read gate"),
            _entry("splade_enabled", ingestion.splade_enabled, "ingestion",
                   "SPLADE sparse expansion at index time"),
            # Issue #341 Task 7: was ingestion.rerank_enabled (the dead
            # post-cap blend branch); the live knob is the [retrieval]
            # pre-cap cross-encoder seam landed in Tasks 5-6.
            _entry("rerank_enabled", retrieval.rerank_enabled, "retrieval",
                   "Cross-encoder rerank on candidates"),
            _entry("decoder_mode", budget.decoder_mode, "budget",
                   "full | condensed | minimal | none — decoder prompt size"),
            _entry("expression_tokens", budget.expression_tokens, "budget",
                   "Retrieved-context token cap per turn"),
            _entry("max_genes_per_turn", budget.max_genes_per_turn, "budget",
                   "Hard ceiling on documents returned per query"),
            _entry("session_delivery_enabled", budget.session_delivery_enabled,
                   "budget", "Elide already-delivered docs from session"),
            _entry("classifier_enabled",
                   getattr(classifier, "enabled", True) if classifier else True,
                   "classifier",
                   "Rule-based query classifier picks decoder mode"),
            _entry("ribosome_backend", ribosome.effective_backend, "ribosome",
                   "litellm | deberta | disabled"),
        ]

        return {
            "settings": settings,
            "summary": _switchboard_summary(settings),
        }

    # ── database (genome registry) ─────────────────────────────────

    def _database_panel(self) -> Dict[str, Any]:
        """Inventory of available genomes + which one is currently selected.

        Reads the on-disk registry. The dashboard pairs this with the
        live `/admin/genome` response (added in `collect`) to surface
        any drift between "what's selected" and "what's actually open".
        """
        try:
            from .genome_registry import (
                active_genome_path,
                discover_genomes,
                is_active,
            )
        except Exception as exc:
            log.warning("Database panel: registry import failed (%s)", exc)
            return {"error": str(exc), "entries": []}

        try:
            entries = discover_genomes()
            # Published as resolved, NOT case-folded: this string is opened
            # with sqlite downstream, and on a case-sensitive filesystem a
            # lowercased path names a file that does not exist. `is_active`
            # below still compares case-insensitively, which is what the
            # Windows "F:/Genomes" vs "f:/genomes" case needs.
            active = str(active_genome_path())
        except Exception as exc:
            log.warning("Database panel: discovery failed (%s)", exc, exc_info=True)
            return {"error": str(exc), "entries": []}

        out_entries = []
        for info in entries:
            row = info.as_dict()
            row["is_active"] = is_active(info)
            out_entries.append(row)

        return {
            "active_path": active,
            "entries": out_entries,
            "count": len(out_entries),
        }

    # ── graph summary ──────────────────────────────────────────────

    def _graph_summary_panel(
        self, database_panel: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Layer counts for the active genome, or None to omit the panel.

        None when the registry could not be read, and when the active path
        does not exist yet: `active_genome_path` falls back to a repo-root
        `genome.db` whether or not it is there, so a fresh install would
        otherwise log a warning on every poll.
        """
        active_path = database_panel.get("active_path")
        if not active_path or not os.path.exists(active_path):
            return None
        try:
            return self.graph_summary_cache.get(active_path)
        except Exception as exc:
            log.warning("Graph summary panel failed (%s)", exc, exc_info=True)
            return None

    def _live_genome(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Project /admin/genome into the database panel's `live` block."""
        return {
            "path": payload.get("path"),
            "configured_path": payload.get("configured_path"),
            "env_override": payload.get("env_override"),
            "size_bytes": payload.get("size_bytes"),
            "total_genes": payload.get("total_genes"),
            "error": payload.get("error"),
        }

    # ── pipeline viewer ────────────────────────────────────────────

    # Ring stages that run INSIDE another timed stage (perf slice 1:
    # delivery_lookup/delivery_log are nested in assemble). Their ms is
    # already inside the parent's ms — counting them in total_ms would
    # double-book exactly the region the perf campaign is shrinking.
    _NESTED_STAGES = frozenset({"delivery_lookup", "delivery_log"})

    def _pipeline_panel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Group raw stage events into per-request rows for the panel.

        Each row carries the request id, total ms across observed
        top-level stages (nested sub-stages are shown in the breakdown
        but excluded from the total), and a per-stage timing breakdown.
        Sorted newest first.
        """
        events = payload.get("events", []) or []
        grouped: Dict[str, Dict[str, Any]] = {}
        for ev in events:
            rid = ev.get("request_id") or "?"
            stage = ev.get("stage") or "?"
            ms = float(ev.get("ms") or 0.0)
            ts = float(ev.get("ts") or 0.0)
            row = grouped.setdefault(rid, {
                "request_id": rid,
                "stages": {},
                "ts": ts,
                "total_ms": 0.0,
            })
            row["stages"][stage] = round(row["stages"].get(stage, 0.0) + ms, 3)
            if stage not in StateCollector._NESTED_STAGES:
                row["total_ms"] = round(row["total_ms"] + ms, 3)
            if ts > row["ts"]:
                row["ts"] = ts

        runs = sorted(grouped.values(), key=lambda r: r["ts"], reverse=True)
        return {
            "runs": runs[:20],
            "event_count": len(events),
            "ring_max": int(payload.get("ring_max") or 0),
        }

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

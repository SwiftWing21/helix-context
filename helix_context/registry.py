"""
Registry — Presence and attribution for multi-session Helix usage.

Provides the data-access layer for:
    - Parties (trust identities — humans, tenants, org service identities)
    - Participants (live runtime actors — Claude sessions, sub-agents)
    - Gene attribution (which party/participant authored which gene)

Schema lives in ``genome.py::_ensure_registry_schema``. See
``docs/SESSION_REGISTRY.md`` for the full design rationale.

Concurrency:
    All methods — reads AND writes — use ``genome.conn`` (the master
    connection), not ``genome.read_conn``. This is deliberate: the
    replication manager propagates gene rows to replicas but does NOT
    sync schema, so the registry tables only exist on the master. WAL
    mode means reads on the master don't block writers, so there is no
    perf penalty for bypassing the replica path. Registry tables are
    metadata, not bulk genome data — master is the right source.

    Writes match the existing ``upsert_gene`` pattern in ``Genome``:
    direct cursor + commit, no separate writer queue.

Cascade / FK semantics:
    SQLite foreign keys are NOT enabled on the genome connection by
    default. The FK declarations in the schema are documentation of
    intent. Until the pragma is enabled, dangling attribution rows from
    deleted genes are harmless — they simply never appear in any
    retrieval path because the JOIN against ``genes`` filters them out.
    A future sweep task can clean them up opportunistically.

Trust model:
    This module implements trust-on-first-use — any client can assert
    any ``party_id`` at registration time. The registry is designed for
    a single-user local-network deployment (localhost:11437). Multi-tenant
    or federated use requires an auth layer that does not yet exist. See
    ``docs/SESSION_REGISTRY.md#trust-model-deferred``.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from typing import List, Optional, Tuple

from .accel import json_dumps, json_loads
from .schemas import (
    GeneAttribution,
    HITLEvent,
    HITLPauseType,
    Participant,
    ParticipantInfo,
    Party,
)

log = logging.getLogger(__name__)


# TTL defaults — can be overridden by config later.
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_TTL_S = 120.0                    # active -> idle after this
IDLE_TTL_S = DEFAULT_TTL_S * 2            # idle -> stale after this
STALE_TTL_S = 24 * 3600.0                 # stale -> gone after 24h
HARD_DELETE_AFTER_S = 7 * 24 * 3600.0     # gone participants hard-deleted after 7 days


def _new_participant_id() -> str:
    """Generate a participant_id. uuid4 for now; ULID could be added later for sortability."""
    return uuid.uuid4().hex


def _status_from_last_heartbeat(last_heartbeat: float, now: Optional[float] = None) -> str:
    """Derive status from last_heartbeat age. Pure function — no DB access."""
    now = now if now is not None else time.time()
    age = now - last_heartbeat
    if age <= DEFAULT_TTL_S:
        return "active"
    if age <= IDLE_TTL_S:
        return "idle"
    if age <= STALE_TTL_S:
        return "stale"
    return "gone"


class Registry:
    """Session registry DAL. Holds a reference to a Genome and operates on its conn."""

    def __init__(self, genome) -> None:
        # Avoid import cycle — type hint Genome lazily
        self.genome = genome

    # ── party / participant lifecycle ───────────────────────────────

    def register_participant(
        self,
        party_id: str,
        handle: str,
        workspace: Optional[str] = None,
        pid: Optional[int] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        display_name: Optional[str] = None,
    ) -> Participant:
        """Register a new participant. Creates the party row on first use (trust-on-first-use).

        Returns the full Participant model with server-generated participant_id.
        """
        now = time.time()
        participant_id = _new_participant_id()
        cur = self.genome.conn.cursor()

        # Ensure party row exists (trust-on-first-use).
        cur.execute(
            "INSERT OR IGNORE INTO parties "
            "(party_id, display_name, trust_domain, created_at, metadata) "
            "VALUES (?, ?, 'local', ?, NULL)",
            (party_id, display_name or party_id, now),
        )

        cur.execute(
            "INSERT INTO participants "
            "(participant_id, party_id, handle, workspace, pid, started_at, "
            " last_heartbeat, status, capabilities, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
            (
                participant_id,
                party_id,
                handle,
                workspace,
                pid,
                now,
                now,
                json_dumps(capabilities or []),
                json_dumps(metadata) if metadata else None,
            ),
        )
        self.genome.conn.commit()
        log.info(
            "Registered participant %s (handle=%s, party=%s)",
            participant_id, handle, party_id,
        )

        return Participant(
            participant_id=participant_id,
            party_id=party_id,
            handle=handle,
            workspace=workspace,
            pid=pid,
            started_at=now,
            last_heartbeat=now,
            status="active",
            capabilities=capabilities or [],
            metadata=metadata,
        )

    def local_org(
        self,
        org_handle: str,
        display_name: Optional[str] = None,
    ) -> str:
        """Find-or-create an org row from a handle ('swiftwing', 'local').

        Used by OS-level federation as the top layer of the 4-layer model
        (org / device / user / agent). Idempotent.

        Returns: the org_id (which == org_handle for local-tier; SSO
        would replace this with the OAuth-issued org claim).
        """
        cur = self.genome.conn.cursor()
        # In local-tier the handle IS the id — keeps lookup trivial and
        # the migration to SSO clean (the SSO upgrade just changes the
        # source of org_id, not the schema).
        cur.execute(
            "INSERT OR IGNORE INTO orgs "
            "(org_id, display_name, trust_domain, created_at) "
            "VALUES (?, ?, 'local', ?)",
            (org_handle, display_name or org_handle, time.time()),
        )
        self.genome.conn.commit()
        return org_handle

    def local_agent(
        self,
        handle: str,
        participant_id: str,
        kind: Optional[str] = None,
    ) -> str:
        """Find-or-create an AI agent persona under a participant.

        Maps (participant_id, agent_handle) -> agent_id. The handle is
        the persona name passed via HELIX_AGENT ('laude', 'taude',
        'claude-code', 'manual'). The kind is the optional category
        ('claude-code', 'gemini', 'gpt-4', 'human') — useful for
        cross-org analytics ("how much did Claude-family agents
        contribute?").

        Idempotent: the (participant_id, handle) UNIQUE constraint
        guarantees the same agent across calls. Touches last_seen_at
        on every call so we can detect dormant agents later.

        Returns: the agent_id (server-generated UUID).
        """
        cur = self.genome.conn.cursor()
        row = cur.execute(
            "SELECT agent_id FROM agents "
            "WHERE participant_id = ? AND handle = ? LIMIT 1",
            (participant_id, handle),
        ).fetchone()
        now = time.time()
        if row is not None:
            agent_id = row["agent_id"]
            cur.execute(
                "UPDATE agents SET last_seen_at = ? WHERE agent_id = ?",
                (now, agent_id),
            )
            self.genome.conn.commit()
            return agent_id

        # New agent — generate UUID, insert, return.
        import uuid as _uuid
        agent_id = _uuid.uuid4().hex
        cur.execute(
            "INSERT INTO agents "
            "(agent_id, participant_id, handle, kind, created_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (agent_id, participant_id, handle, kind, now, now),
        )
        self.genome.conn.commit()
        log.info(
            "Registered agent %s (handle=%s, kind=%s, participant=%s)",
            agent_id, handle, kind, participant_id,
        )
        return agent_id

    def local_participant(
        self,
        handle: str,
        party_id: str,
        workspace: Optional[str] = None,
        org_id: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> str:
        """Find-or-create a participant identified by (party_id, handle).

        This is the OS-level federation entry point — it lets ingest paths
        attribute genes to "max@desktop", "laude@desktop", etc. without
        any auth infrastructure. The handle is the OS username (HELIX_USER
        or getpass.getuser()); the party_id is the hostname (HELIX_DEVICE
        or HELIX_PARTY or socket.gethostname()). See FEDERATION_LOCAL.md.

        When ``org_id`` is provided (typically from HELIX_ORG via
        _local_attribution_defaults), the party is linked to that org so
        the 4-layer attribution chain (org -> device -> user -> agent)
        is queryable end-to-end. Defaults to 'local' org otherwise.

        Idempotent: subsequent calls with the same (party_id, handle)
        return the same participant_id without creating duplicates.
        """
        effective_org = org_id or "local"

        # Ensure the org row exists (idempotent).
        cur = self.genome.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO orgs "
            "(org_id, display_name, trust_domain, created_at) "
            "VALUES (?, ?, 'local', ?)",
            (effective_org, effective_org, time.time()),
        )

        # Ensure the party.org_id link is set even if the party row
        # already existed without one (legacy upgrade path).
        cur.execute(
            "UPDATE parties SET org_id = ? "
            "WHERE party_id = ? AND (org_id IS NULL OR org_id = '')",
            (effective_org, party_id),
        )
        # Update the device's home timezone on every call. This is
        # last-write-wins by design — when a laptop crosses tz, the
        # parties.timezone reflects "where this device usually is now"
        # while gene_attribution.authored_tz captures per-write history.
        # Together they distinguish "device's home" from "where each
        # gene was actually authored."
        if timezone:
            cur.execute(
                "UPDATE parties SET timezone = ? WHERE party_id = ?",
                (timezone, party_id),
            )
        self.genome.conn.commit()

        # Look up by (party_id, handle) — this is the natural local key.
        row = cur.execute(
            "SELECT participant_id FROM participants "
            "WHERE party_id = ? AND handle = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (party_id, handle),
        ).fetchone()
        if row is not None:
            pid_existing = row["participant_id"]
            self.touch_heartbeat(pid_existing)
            return pid_existing

        # Not found — register a new one (creates the party row
        # trust-on-first-use, then we backfill the org_id link).
        p = self.register_participant(
            party_id=party_id,
            handle=handle,
            workspace=workspace,
            metadata={"source": "os_federation", "trust_domain": "local"},
        )
        # Re-stamp the org_id on the party row that register_participant
        # just created with NULL org (it doesn't know about the org layer).
        # Also stamp the timezone on first-creation if provided.
        cur.execute(
            "UPDATE parties SET org_id = ? WHERE party_id = ? AND org_id IS NULL",
            (effective_org, party_id),
        )
        if timezone:
            cur.execute(
                "UPDATE parties SET timezone = ? WHERE party_id = ? AND timezone IS NULL",
                (timezone, party_id),
            )
        self.genome.conn.commit()
        return p.participant_id

    def heartbeat(self, participant_id: str) -> Optional[Tuple[float, str]]:
        """Refresh last_heartbeat for a participant.

        Returns (ttl_remaining_s, new_status) on success, or None if the
        participant_id is unknown. Unknown participants should re-register.
        """
        now = time.time()
        cur = self.genome.conn.cursor()
        row = cur.execute(
            "SELECT participant_id FROM participants WHERE participant_id = ?",
            (participant_id,),
        ).fetchone()
        if row is None:
            return None
        cur.execute(
            "UPDATE participants "
            "SET last_heartbeat = ?, status = 'active' "
            "WHERE participant_id = ?",
            (now, participant_id),
        )
        self.genome.conn.commit()
        return (DEFAULT_TTL_S, "active")

    def touch_heartbeat(self, participant_id: str) -> None:
        """Silent heartbeat refresh — used by implicit ingest-as-activity path.

        Does not raise on unknown participant_id; just skips. Does not commit;
        caller is expected to commit as part of the surrounding write.
        """
        now = time.time()
        cur = self.genome.conn.cursor()
        cur.execute(
            "UPDATE participants "
            "SET last_heartbeat = ?, status = 'active' "
            "WHERE participant_id = ?",
            (now, participant_id),
        )

    # ── queries ─────────────────────────────────────────────────────

    def list_participants(
        self,
        party_id: Optional[str] = None,
        status_filter: str = "active",
        workspace_prefix: Optional[str] = None,
    ) -> List[ParticipantInfo]:
        """List participants with live-computed status from last_heartbeat.

        ``status_filter`` is one of ``active``, ``idle``, ``stale``, ``gone``,
        or ``all``. Status is computed on the fly from ``last_heartbeat``;
        the persisted ``status`` column is a cache that the sweep task
        updates but observers should not trust.
        """
        cur = self.genome.conn.cursor()
        sql = (
            "SELECT participant_id, party_id, handle, workspace, "
            "       started_at, last_heartbeat "
            "FROM participants"
        )
        params: list = []
        conditions: list = []
        if party_id is not None:
            conditions.append("party_id = ?")
            params.append(party_id)
        if workspace_prefix is not None:
            conditions.append("workspace LIKE ?")
            params.append(workspace_prefix + "%")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY last_heartbeat DESC"

        now = time.time()
        rows = cur.execute(sql, params).fetchall()
        out: List[ParticipantInfo] = []
        for r in rows:
            live_status = _status_from_last_heartbeat(r["last_heartbeat"], now)
            if status_filter != "all" and live_status != status_filter:
                continue
            out.append(ParticipantInfo(
                participant_id=r["participant_id"],
                party_id=r["party_id"],
                handle=r["handle"],
                workspace=r["workspace"],
                status=live_status,
                last_seen_s_ago=round(now - r["last_heartbeat"], 1),
                started_at=r["started_at"],
            ))
        return out

    def get_participant(self, participant_id: str) -> Optional[Participant]:
        """Fetch a single participant by id. Returns None if unknown."""
        cur = self.genome.conn.cursor()
        r = cur.execute(
            "SELECT participant_id, party_id, handle, workspace, pid, "
            "       started_at, last_heartbeat, status, capabilities, metadata "
            "FROM participants WHERE participant_id = ?",
            (participant_id,),
        ).fetchone()
        if r is None:
            return None
        caps_raw = r["capabilities"] or "[]"
        meta_raw = r["metadata"]
        try:
            caps = json_loads(caps_raw) if caps_raw else []
        except Exception:
            caps = []
        try:
            meta = json_loads(meta_raw) if meta_raw else None
        except Exception:
            meta = None
        return Participant(
            participant_id=r["participant_id"],
            party_id=r["party_id"],
            handle=r["handle"],
            workspace=r["workspace"],
            pid=r["pid"],
            started_at=r["started_at"],
            last_heartbeat=r["last_heartbeat"],
            status=_status_from_last_heartbeat(r["last_heartbeat"]),
            capabilities=caps,
            metadata=meta,
        )

    def get_recent_by_handle(
        self,
        handle: str,
        limit: int = 10,
        party_id: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[dict]:
        """Return genes recently authored by participants with a given handle.

        This is the BM25 bypass — chronological, no scoring. Joins
        ``gene_attribution`` -> ``participants`` (to match handle) ->
        ``genes`` (for content preview). Returns dicts suitable for JSON
        serialization.
        """
        cur = self.genome.conn.cursor()
        sql = (
            "SELECT ga.gene_id, ga.party_id, ga.participant_id, ga.authored_at, "
            "       g.content "
            "FROM gene_attribution ga "
            "JOIN participants p ON p.participant_id = ga.participant_id "
            "JOIN genes g ON g.gene_id = ga.gene_id "
            "WHERE p.handle = ?"
        )
        params: list = [handle]
        if party_id is not None:
            sql += " AND ga.party_id = ?"
            params.append(party_id)
        if since is not None:
            sql += " AND ga.authored_at >= ?"
            params.append(since)
        sql += " ORDER BY ga.authored_at DESC LIMIT ?"
        params.append(int(limit))

        rows = cur.execute(sql, params).fetchall()
        out = []
        for r in rows:
            content = r["content"] or ""
            preview = content[:200] + ("…" if len(content) > 200 else "")
            out.append({
                "gene_id": r["gene_id"],
                "content_preview": preview,
                "authored_at": r["authored_at"],
                "party_id": r["party_id"],
                "participant_id": r["participant_id"],
            })
        return out

    # ── attribution ─────────────────────────────────────────────────

    def attribute_gene(
        self,
        gene_id: str,
        participant_id: Optional[str] = None,
        party_id: Optional[str] = None,
        authored_at: Optional[float] = None,
        org_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        authored_tz: Optional[str] = None,
    ) -> Optional[GeneAttribution]:
        """Write a 4-axis attribution row for a gene.

        Identity layers (any may be NULL except party_id):
            org_id          top-level tenant (org/team)
            party_id        device (PC) inside the org
            participant_id  human (user) on the device
            agent_id        AI persona working on the user's behalf

        If ``participant_id`` is provided and known, ``party_id`` and
        ``org_id`` are auto-resolved by joining participants -> parties
        -> orgs. Explicit values passed in win over resolved values, so
        callers can override (useful for cross-org SaaS scenarios).

        If only ``party_id`` is provided, attribution is written at the
        party level with NULL participant_id and NULL agent_id, with
        org_id resolved from parties.org_id.

        If neither participant nor party is provided, returns None
        without error. ``agent_id`` requires a participant_id (an AI
        agent always acts on behalf of a known human).
        """
        if not participant_id and not party_id:
            return None

        now = authored_at if authored_at is not None else time.time()

        cur = self.genome.conn.cursor()

        # Resolve party_id from participant if not explicitly given.
        resolved_party = party_id
        if participant_id and not resolved_party:
            row = cur.execute(
                "SELECT party_id FROM participants WHERE participant_id = ?",
                (participant_id,),
            ).fetchone()
            if row is None:
                log.warning(
                    "attribute_gene: unknown participant_id=%s — gene %s not attributed",
                    participant_id, gene_id,
                )
                return None
            resolved_party = row["party_id"]

        if not resolved_party:
            return None

        # Resolve org_id from party if not explicitly given. Falls back
        # to 'local' (the seeded default org) when the party has no
        # org_id link (legacy pre-4-layer rows or local-tier defaults).
        resolved_org = org_id
        if not resolved_org:
            row = cur.execute(
                "SELECT org_id FROM parties WHERE party_id = ?",
                (resolved_party,),
            ).fetchone()
            if row is not None:
                resolved_org = row["org_id"]
            if not resolved_org:
                resolved_org = "local"

        cur.execute(
            "INSERT OR REPLACE INTO gene_attribution "
            "(gene_id, party_id, participant_id, authored_at, org_id, agent_id, authored_tz) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (gene_id, resolved_party, participant_id, now, resolved_org, agent_id, authored_tz),
        )
        # Implicit heartbeat — avoid round trips for clients that ingest often
        if participant_id:
            self.touch_heartbeat(participant_id)
        self.genome.conn.commit()

        return GeneAttribution(
            gene_id=gene_id,
            party_id=resolved_party,
            participant_id=participant_id,
            authored_at=now,
        )

    def get_attribution(self, gene_id: str) -> Optional[GeneAttribution]:
        """Look up attribution for a single gene. Returns None if not attributed."""
        cur = self.genome.conn.cursor()
        r = cur.execute(
            "SELECT gene_id, party_id, participant_id, authored_at "
            "FROM gene_attribution WHERE gene_id = ?",
            (gene_id,),
        ).fetchone()
        if r is None:
            return None
        return GeneAttribution(
            gene_id=r["gene_id"],
            party_id=r["party_id"],
            participant_id=r["participant_id"],
            authored_at=r["authored_at"],
        )

    def get_attributions_for_genes(
        self, gene_ids: List[str]
    ) -> dict:
        """Batch lookup — returns ``{gene_id: {party_id, participant_id, handle}}``.

        Used by the /context citation enrichment path. JOINs gene_attribution
        with participants to resolve the participant's CURRENT handle (handles
        are mutable across re-registrations of the same logical persona, so
        we resolve at read time rather than caching at write time).

        Genes without an attribution row are simply absent from the result.
        Empty input returns ``{}`` without hitting the database.
        """
        if not gene_ids:
            return {}
        cur = self.genome.conn.cursor()
        placeholders = ",".join("?" * len(gene_ids))
        rows = cur.execute(
            f"SELECT ga.gene_id, ga.party_id, ga.participant_id, p.handle "
            f"FROM gene_attribution ga "
            f"LEFT JOIN participants p ON p.participant_id = ga.participant_id "
            f"WHERE ga.gene_id IN ({placeholders})",
            gene_ids,
        ).fetchall()
        return {
            r["gene_id"]: {
                "party_id": r["party_id"],
                "participant_id": r["participant_id"],
                "handle": r["handle"],
            }
            for r in rows
        }

    # ── HITL event logging ──────────────────────────────────────────
    #
    # Motivated by laude's 2026-04-11 HITL observation handoff and
    # raude's M1 discriminating test. The M1 test ruled out genome-
    # mediated propagation of the HITL shift — the mechanism lives in
    # the chat channel, not in the genome substrate. So the logger
    # records chat-channel signals in addition to genome-state snapshots.
    # See ~/.helix/shared/handoffs/2026-04-11_hitl_observation.md.
    #
    # None of the methods below raise on bad input — instrumentation
    # must never fail a session because of a logging error.

    def emit_hitl_event(
        self,
        participant_id: Optional[str],
        pause_type: str,
        task_context: Optional[str] = None,
        resolved_without_operator: bool = False,
        chat_signals: Optional[dict] = None,
        genome_snapshot: Optional[dict] = None,
        metadata: Optional[dict] = None,
        party_id: Optional[str] = None,
    ) -> Optional[str]:
        """Record a HITL pause event on the session registry.

        Returns the generated ``event_id`` on success, or ``None`` if the
        event could not be written (unknown participant + no party_id,
        invalid pause_type, etc). Never raises.

        Resolution rules:
          - If ``participant_id`` is provided and known, ``party_id`` is
            resolved automatically from the participants table.
          - If ``participant_id`` is unknown, logs a warning and returns None.
          - If ``participant_id`` is None, the caller must supply ``party_id``
            explicitly (used by server-side emit flows that know the party
            but not a specific participant).

        ``chat_signals`` is an optional dict that may contain any of:
          - ``tone_uncertainty`` (float 0-1)
          - ``risk_keywords`` (list of strings)
          - ``time_since_last_risk`` (seconds, float)
          - ``recoverability`` (str: "recoverable" / "uncertain" / "lost")

        ``genome_snapshot`` is an optional dict that may contain any of:
          - ``total_genes`` (int)
          - ``hetero_count`` (int)
          - ``cold_cache_size`` (int)

        Both dicts are tolerant to missing keys. Unknown keys are ignored.

        As with ``attribute_gene``, this implicitly refreshes the
        participant's heartbeat — HITL events are activity signals.
        """
        # Validate pause_type
        try:
            pause_enum = HITLPauseType(pause_type)
        except ValueError:
            log.warning("emit_hitl_event: unknown pause_type=%r; coercing to 'other'", pause_type)
            pause_enum = HITLPauseType.other

        # Resolve party_id
        resolved_party = party_id
        if participant_id:
            cur = self.genome.conn.cursor()
            row = cur.execute(
                "SELECT party_id FROM participants WHERE participant_id = ?",
                (participant_id,),
            ).fetchone()
            if row is None:
                log.warning(
                    "emit_hitl_event: unknown participant_id=%s; event not written",
                    participant_id,
                )
                return None
            resolved_party = row["party_id"]

        if not resolved_party:
            log.warning("emit_hitl_event: no participant_id and no party_id; event not written")
            return None

        now = time.time()
        event_id = uuid.uuid4().hex

        cs = chat_signals or {}
        gs = genome_snapshot or {}
        risk_keywords_json = json_dumps(cs.get("risk_keywords", [])) if "risk_keywords" in cs else None

        cur = self.genome.conn.cursor()
        cur.execute(
            """INSERT INTO hitl_events (
                event_id, party_id, participant_id, ts,
                pause_type, task_context, resolved_without_operator,
                operator_tone_uncertainty, operator_risk_keywords,
                time_since_last_risk_event, recoverability_signal,
                genome_total_genes, genome_hetero_count, cold_cache_size,
                metadata
            ) VALUES (?, ?, ?, ?,   ?, ?, ?,   ?, ?, ?, ?,   ?, ?, ?,   ?)""",
            (
                event_id,
                resolved_party,
                participant_id,
                now,
                pause_enum.value,
                task_context,
                1 if resolved_without_operator else 0,
                cs.get("tone_uncertainty"),
                risk_keywords_json,
                cs.get("time_since_last_risk"),
                cs.get("recoverability"),
                gs.get("total_genes"),
                gs.get("hetero_count"),
                gs.get("cold_cache_size"),
                json_dumps(metadata) if metadata else None,
            ),
        )

        # Implicit heartbeat — a HITL event is session activity.
        if participant_id:
            self.touch_heartbeat(participant_id)
        self.genome.conn.commit()
        return event_id

    def get_hitl_events(
        self,
        party_id: Optional[str] = None,
        participant_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        pause_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Query HITL events with optional filters.

        Returns a list of dicts (not Pydantic models) ordered by ``ts`` DESC.
        Matches the ``get_recent_by_handle`` return shape — the caller
        decides whether to hydrate into ``HITLEvent`` models.

        All filter arguments are optional; omit them to get the most
        recent ``limit`` events globally. An empty result is a list,
        not None.
        """
        cur = self.genome.conn.cursor()
        sql = (
            "SELECT event_id, party_id, participant_id, ts, pause_type, "
            "       task_context, resolved_without_operator, "
            "       operator_tone_uncertainty, operator_risk_keywords, "
            "       time_since_last_risk_event, recoverability_signal, "
            "       genome_total_genes, genome_hetero_count, cold_cache_size, "
            "       metadata "
            "FROM hitl_events"
        )
        conditions: list = []
        params: list = []
        if party_id is not None:
            conditions.append("party_id = ?")
            params.append(party_id)
        if participant_id is not None:
            conditions.append("participant_id = ?")
            params.append(participant_id)
        if since is not None:
            conditions.append("ts >= ?")
            params.append(since)
        if until is not None:
            conditions.append("ts <= ?")
            params.append(until)
        if pause_type is not None:
            conditions.append("pause_type = ?")
            params.append(pause_type)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))

        rows = cur.execute(sql, params).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                risk_kw = json_loads(r["operator_risk_keywords"]) if r["operator_risk_keywords"] else []
            except Exception:
                risk_kw = []
            try:
                md = json_loads(r["metadata"]) if r["metadata"] else None
            except Exception:
                md = None
            out.append({
                "event_id": r["event_id"],
                "party_id": r["party_id"],
                "participant_id": r["participant_id"],
                "ts": r["ts"],
                "pause_type": r["pause_type"],
                "task_context": r["task_context"],
                "resolved_without_operator": bool(r["resolved_without_operator"]),
                "operator_tone_uncertainty": r["operator_tone_uncertainty"],
                "operator_risk_keywords": risk_kw,
                "time_since_last_risk_event": r["time_since_last_risk_event"],
                "recoverability_signal": r["recoverability_signal"],
                "genome_total_genes": r["genome_total_genes"],
                "genome_hetero_count": r["genome_hetero_count"],
                "cold_cache_size": r["cold_cache_size"],
                "metadata": md,
            })
        return out

    def hitl_rate(
        self,
        participant_id: str,
        window_seconds: float = 3600.0,
    ) -> float:
        """HITL events per second over the last ``window_seconds`` for a
        given participant.

        Mirrors ``EpigeneticMarkers.access_rate`` — same bell-curve /
        sliding-window idea applied to a different table. Returns 0.0
        for unknown participants, empty windows, or non-positive windows.
        """
        if window_seconds <= 0:
            return 0.0
        cutoff = time.time() - window_seconds
        cur = self.genome.conn.cursor()
        row = cur.execute(
            "SELECT COUNT(*) AS n FROM hitl_events "
            "WHERE participant_id = ? AND ts >= ?",
            (participant_id, cutoff),
        ).fetchone()
        if row is None:
            return 0.0
        return float(row["n"]) / window_seconds

    def hitl_stats(
        self,
        party_id: Optional[str] = None,
        since: Optional[float] = None,
    ) -> dict:
        """Aggregate HITL stats for analysis / dashboard consumption.

        Returns a dict with:
          - ``total``: total event count
          - ``by_pause_type``: dict mapping pause_type -> count
          - ``resolved_without_operator``: count of self-resolved events
          - ``mean_gap_s``: mean seconds between consecutive events (None
            if fewer than 2 events in scope)

        ``party_id`` scopes the aggregate; ``since`` is a lower-bound ts
        filter. Both optional.

        NOT a hot-path method — intended for offline analysis and
        dashboards. Safe to call at any frequency the UI needs.
        """
        cur = self.genome.conn.cursor()
        conditions: list = []
        params: list = []
        if party_id is not None:
            conditions.append("party_id = ?")
            params.append(party_id)
        if since is not None:
            conditions.append("ts >= ?")
            params.append(since)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        total_row = cur.execute(
            f"SELECT COUNT(*) AS n FROM hitl_events{where}", params
        ).fetchone()
        total = total_row["n"] if total_row else 0

        # Breakdown by pause_type
        by_type: dict = {}
        for r in cur.execute(
            f"SELECT pause_type, COUNT(*) AS n FROM hitl_events{where} GROUP BY pause_type",
            params,
        ).fetchall():
            by_type[r["pause_type"]] = r["n"]

        resolved_row = cur.execute(
            f"SELECT COUNT(*) AS n FROM hitl_events{where}{' AND' if where else ' WHERE'} resolved_without_operator = 1",
            params,
        ).fetchone()
        resolved = resolved_row["n"] if resolved_row else 0

        # Mean gap between consecutive events (in-scope only)
        gap_rows = cur.execute(
            f"SELECT ts FROM hitl_events{where} ORDER BY ts ASC",
            params,
        ).fetchall()
        if len(gap_rows) >= 2:
            timestamps = [r["ts"] for r in gap_rows]
            gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            mean_gap = sum(gaps) / len(gaps) if gaps else None
        else:
            mean_gap = None

        return {
            "total": total,
            "by_pause_type": by_type,
            "resolved_without_operator": resolved,
            "mean_gap_s": mean_gap,
        }

    # ── maintenance ─────────────────────────────────────────────────

    def sweep(self, now: Optional[float] = None) -> dict:
        """Update persisted status column based on last_heartbeat age.

        This is a cache update — observers that call list_participants
        get live status regardless. The sweep exists so that queries
        filtering by the persisted ``status`` column are consistent with
        reality between calls to list_participants.

        Returns a summary dict with transition counts.
        """
        now = now if now is not None else time.time()
        cur = self.genome.conn.cursor()

        counts = {"active": 0, "idle": 0, "stale": 0, "gone": 0, "hard_deleted": 0}

        rows = cur.execute(
            "SELECT participant_id, last_heartbeat, status FROM participants"
        ).fetchall()
        for r in rows:
            live = _status_from_last_heartbeat(r["last_heartbeat"], now)
            counts[live] += 1
            if live != r["status"]:
                cur.execute(
                    "UPDATE participants SET status = ? WHERE participant_id = ?",
                    (live, r["participant_id"]),
                )

        # Hard-delete participants that have been gone for more than HARD_DELETE_AFTER_S.
        # Their gene_attribution rows keep party_id but have participant_id NULLed
        # manually (FK ON DELETE SET NULL is declared but not enforced by default).
        hard_delete_cutoff = now - HARD_DELETE_AFTER_S
        gone_rows = cur.execute(
            "SELECT participant_id FROM participants "
            "WHERE status = 'gone' AND last_heartbeat < ?",
            (hard_delete_cutoff,),
        ).fetchall()
        for r in gone_rows:
            pid = r["participant_id"]
            cur.execute(
                "UPDATE gene_attribution SET participant_id = NULL "
                "WHERE participant_id = ?",
                (pid,),
            )
            # HITL events survive participant hard-delete with participant_id NULLed,
            # same pattern as gene_attribution. Preserves the historical record for
            # aggregate analysis while honoring the ON DELETE SET NULL contract.
            cur.execute(
                "UPDATE hitl_events SET participant_id = NULL "
                "WHERE participant_id = ?",
                (pid,),
            )
            cur.execute(
                "DELETE FROM participants WHERE participant_id = ?",
                (pid,),
            )
            counts["hard_deleted"] += 1

        self.genome.conn.commit()
        return counts

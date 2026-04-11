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
from .schemas import GeneAttribution, Participant, ParticipantInfo, Party

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
    ) -> Optional[GeneAttribution]:
        """Write an attribution row for a gene.

        If ``participant_id`` is provided and known, ``party_id`` is
        resolved automatically. If ``participant_id`` is unknown, logs a
        warning and returns None (does not raise — ingest must never fail
        because of registry state).

        If only ``party_id`` is provided, attribution is written at the
        party level with NULL participant_id. This is useful for
        server-side ingestion flows that know the party but not the
        specific participant.

        If neither is provided, returns None without error.
        """
        if not participant_id and not party_id:
            return None

        now = authored_at if authored_at is not None else time.time()

        # If participant_id is provided, resolve party_id via the participants table.
        resolved_party = party_id
        if participant_id:
            cur = self.genome.conn.cursor()
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

        cur = self.genome.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO gene_attribution "
            "(gene_id, party_id, participant_id, authored_at) "
            "VALUES (?, ?, ?, ?)",
            (gene_id, resolved_party, participant_id, now),
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
            cur.execute(
                "DELETE FROM participants WHERE participant_id = ?",
                (pid,),
            )
            counts["hard_deleted"] += 1

        self.genome.conn.commit()
        return counts

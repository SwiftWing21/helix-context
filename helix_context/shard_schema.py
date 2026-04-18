"""Main.db schema — routing table + fingerprint index for sharded genomes.

Main.db is the routing layer for genome sharding (see
docs/FUTURE/GENOME_SHARDING.md and docs/specs/2026-04-17-genome-sharding-plan.md).

It holds only what's needed to pick a shard for a query:
    - shards: registry of category shard .db files
    - fingerprint_index: the ~150 tok push payload per gene
    - identity tables: mirror of the 4-layer registry for cross-shard joins

It does NOT hold content, complement, codons, embeddings, or any other
tier-1/2/3 bulk data. Those stay in category shards.

Schema is additive and idempotent — safe to run on every startup.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)


# ── Shard categories ─────────────────────────────────────────────────

SHARD_CATEGORIES = ("participant", "agent", "reference", "org", "cold")


def open_main_db(path: str) -> sqlite3.Connection:
    """Open or create main.db with WAL + busy_timeout, return connection.

    Mirrors Genome's connection setup so both layers behave identically
    under contention.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_main_db(conn: sqlite3.Connection) -> None:
    """Create all main.db tables + indexes. Idempotent."""
    cur = conn.cursor()
    _create_shards_table(cur)
    _create_fingerprint_index(cur)
    _create_source_index(cur)
    _create_identity_tables(cur)
    conn.commit()


def _create_shards_table(cur: sqlite3.Cursor) -> None:
    """shards — registry of category shard .db files.

    One row per physical shard file on disk. Router reads this at
    startup to know which .db files exist and where to open them.
    """
    cur.execute("""
    CREATE TABLE IF NOT EXISTS shards (
        shard_name   TEXT PRIMARY KEY,
        category     TEXT NOT NULL,
        path         TEXT NOT NULL,
        created_at   REAL NOT NULL,
        updated_at   REAL NOT NULL,
        gene_count   INTEGER NOT NULL DEFAULT 0,
        byte_size    INTEGER NOT NULL DEFAULT 0,
        health       TEXT NOT NULL DEFAULT 'ok',
        metadata     TEXT
    )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_shards_category "
        "ON shards(category)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_shards_health "
        "ON shards(health)"
    )


def _create_fingerprint_index(cur: sqlite3.Cursor) -> None:
    """fingerprint_index — ~150 tok push payload per gene.

    Router queries this FIRST to pick which shards have candidate
    matches, then opens those shards for real FTS/scoring.

    Columns mirror what PUSH_PULL_CONTEXT.md specifies as the eager
    push payload: gene_id, shard location, source_id, domains,
    entities, key_values. No content, no codons, no embeddings.
    """
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fingerprint_index (
        gene_id      TEXT PRIMARY KEY,
        shard_name   TEXT NOT NULL REFERENCES shards(shard_name),
        source_id    TEXT,
        domains      TEXT,     -- JSON list[str]
        entities     TEXT,     -- JSON list[str]
        key_values   TEXT,     -- JSON list[str]
        is_parent    INTEGER NOT NULL DEFAULT 0,
        sequence_idx INTEGER,
        updated_at   REAL NOT NULL
    )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_fp_shard "
        "ON fingerprint_index(shard_name)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_fp_source "
        "ON fingerprint_index(source_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_fp_is_parent "
        "ON fingerprint_index(is_parent)"
    )


def _create_source_index(cur: sqlite3.Cursor) -> None:
    """source_index — provenance + freshness metadata per gene.

    Lightweight metadata only. This lets the agent-context layer answer
    freshness and authority questions without reopening every shard or
    loading bulk gene content.
    """
    cur.execute("""
    CREATE TABLE IF NOT EXISTS source_index (
        gene_id           TEXT PRIMARY KEY,
        shard_name        TEXT NOT NULL REFERENCES shards(shard_name),
        source_id         TEXT,
        repo_root         TEXT,
        source_kind       TEXT,
        observed_at       REAL,
        mtime             REAL,
        content_hash      TEXT,
        volatility_class  TEXT NOT NULL DEFAULT 'medium',
        authority_class   TEXT NOT NULL DEFAULT 'primary',
        support_span      TEXT,
        last_verified_at  REAL,
        invalidated_at    REAL,
        updated_at        REAL NOT NULL
    )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_index_shard "
        "ON source_index(shard_name)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_index_source "
        "ON source_index(source_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_index_repo "
        "ON source_index(repo_root)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_index_kind "
        "ON source_index(source_kind)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_index_volatility "
        "ON source_index(volatility_class)"
    )


def _create_identity_tables(cur: sqlite3.Cursor) -> None:
    """Mirror the 4-layer identity schema from genome.py._ensure_registry_schema.

    Keeping identity in main.db lets the router answer "which shard
    does participant X own?" without opening every category shard.
    Writes to this table happen alongside the shard write during
    upsert (transactional coupling is at the router layer, not SQL).
    """
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orgs (
        org_id         TEXT PRIMARY KEY,
        display_name   TEXT NOT NULL,
        trust_domain   TEXT NOT NULL DEFAULT 'local',
        created_at     REAL NOT NULL,
        metadata       TEXT
    )
    """)
    cur.execute(
        "INSERT OR IGNORE INTO orgs "
        "(org_id, display_name, trust_domain, created_at) "
        "VALUES ('local', 'Local Org (default)', 'local', ?)",
        (time.time(),),
    )

    cur.execute("""
    CREATE TABLE IF NOT EXISTS parties (
        party_id      TEXT PRIMARY KEY,
        display_name  TEXT NOT NULL,
        trust_domain  TEXT NOT NULL DEFAULT 'local',
        org_id        TEXT REFERENCES orgs(org_id),
        timezone      TEXT,
        created_at    REAL NOT NULL,
        metadata      TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS participants (
        participant_id   TEXT PRIMARY KEY,
        party_id         TEXT NOT NULL REFERENCES parties(party_id),
        handle           TEXT NOT NULL,
        workspace        TEXT,
        pid              INTEGER,
        started_at       REAL NOT NULL,
        last_heartbeat   REAL NOT NULL,
        status           TEXT NOT NULL DEFAULT 'active',
        capabilities     TEXT,
        metadata         TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        agent_id        TEXT PRIMARY KEY,
        participant_id  TEXT NOT NULL
                        REFERENCES participants(participant_id) ON DELETE CASCADE,
        handle          TEXT NOT NULL,
        kind            TEXT,
        created_at      REAL NOT NULL,
        last_seen_at    REAL,
        metadata        TEXT,
        UNIQUE (participant_id, handle)
    )
    """)


# ── Shard registry helpers ───────────────────────────────────────────


def register_shard(
    conn: sqlite3.Connection,
    shard_name: str,
    category: str,
    path: str,
    gene_count: int = 0,
    byte_size: int = 0,
    metadata: str | None = None,
) -> None:
    """Upsert a shard row. Call after creating a new category .db file."""
    if category not in SHARD_CATEGORIES:
        raise ValueError(
            f"unknown category {category!r}; must be one of {SHARD_CATEGORIES}"
        )
    now = time.time()
    conn.execute(
        "INSERT INTO shards (shard_name, category, path, created_at, "
        "updated_at, gene_count, byte_size, health, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'ok', ?) "
        "ON CONFLICT(shard_name) DO UPDATE SET "
        "path=excluded.path, updated_at=excluded.updated_at, "
        "gene_count=excluded.gene_count, byte_size=excluded.byte_size, "
        "metadata=excluded.metadata",
        (shard_name, category, path, now, now, gene_count, byte_size, metadata),
    )
    conn.commit()
    log.info("registered shard %s (%s) at %s", shard_name, category, path)


def list_shards(
    conn: sqlite3.Connection,
    category: str | None = None,
) -> list[sqlite3.Row]:
    """Return all shard rows, optionally filtered by category."""
    if category is not None:
        cur = conn.execute(
            "SELECT * FROM shards WHERE category = ? AND health = 'ok' "
            "ORDER BY shard_name",
            (category,),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM shards WHERE health = 'ok' ORDER BY category, shard_name"
        )
    return list(cur.fetchall())


def upsert_fingerprint(
    conn: sqlite3.Connection,
    gene_id: str,
    shard_name: str,
    source_id: str | None,
    domains_json: str | None,
    entities_json: str | None,
    key_values_json: str | None,
    is_parent: bool = False,
    sequence_idx: int | None = None,
) -> None:
    """Write or replace a fingerprint_index row. Call during ingest."""
    conn.execute(
        "INSERT OR REPLACE INTO fingerprint_index "
        "(gene_id, shard_name, source_id, domains, entities, key_values, "
        "is_parent, sequence_idx, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            gene_id, shard_name, source_id, domains_json, entities_json,
            key_values_json, 1 if is_parent else 0, sequence_idx, time.time(),
        ),
    )
    conn.commit()


def upsert_source_index(
    conn: sqlite3.Connection,
    gene_id: str,
    shard_name: str,
    source_id: str | None,
    repo_root: str | None = None,
    source_kind: str | None = None,
    observed_at: float | None = None,
    mtime: float | None = None,
    content_hash: str | None = None,
    volatility_class: str | None = None,
    authority_class: str | None = None,
    support_span: str | None = None,
    last_verified_at: float | None = None,
    invalidated_at: float | None = None,
) -> None:
    """Write or replace a source_index row for a gene."""
    conn.execute(
        "INSERT OR REPLACE INTO source_index "
        "(gene_id, shard_name, source_id, repo_root, source_kind, observed_at, "
        "mtime, content_hash, volatility_class, authority_class, support_span, "
        "last_verified_at, invalidated_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            gene_id,
            shard_name,
            source_id,
            repo_root,
            source_kind,
            observed_at,
            mtime,
            content_hash,
            volatility_class or "medium",
            authority_class or "primary",
            support_span,
            last_verified_at,
            invalidated_at,
            time.time(),
        ),
    )
    conn.commit()

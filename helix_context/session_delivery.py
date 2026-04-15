"""
Session working-set register — Sprint 2 of the AI-consumer roadmap.

Every /context call today is stateless: the same LLM querying Helix 20-60
times in one conversation pays full token cost for overlapping gene sets
delivered on nearly every turn. The expected ~40% token reduction comes
from elidng genes the consumer already holds.

This module owns the `session_delivery_log` table — one row per
(session_id, gene_id, delivered_at) triple — plus the thin DAL the
pipeline uses to check "have I already shipped this?" before assembly
and to "I'm shipping this now" after.

Design notes:

* **TTL is per-session, unbounded in time.** A row never expires on a
  clock; it expires only when its `session_id` stops being referenced
  by incoming /context calls. A future sweeper can purge sessions whose
  most-recent delivery is > N days old, but the MVP keeps everything.
* **Scope is per-session, not per-party.** Federation (share deliveries
  across sessions under the same party) is a later concern; the simpler
  model gets the consumer most of the way.
* **Content hash is over the EXPRESSED text**, not the raw gene. Different
  splice modes of the same gene count as distinct deliveries, because the
  downstream LLM genuinely saw different strings.
* **Opt-out exists.** Callers that *want* redundancy (benchmarks running
  controlled fixtures, eval rigs measuring per-call quality) pass
  `ignore_delivered=true` and bypass the whole check path. They still
  emit delivery rows so future non-opt-out calls see an accurate log.

Schema is co-located in `Genome._create_schema` alongside cwola_log so
the tables migrate together. This module exposes a bare `ensure_schema`
for the in-memory test fixture path — production always gets it via
genome init.

See docs/FUTURE/AI_CONSUMER_ROADMAP_2026-04-14.md Sprint 2.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from typing import List, Optional, Tuple

log = logging.getLogger("helix.session_delivery")


# ── Schema ────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_delivery_log (
    delivery_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    gene_id         TEXT NOT NULL,
    retrieval_id    INTEGER,
    delivered_at    REAL NOT NULL,
    content_hash    TEXT,
    mode            TEXT
);
CREATE INDEX IF NOT EXISTS idx_sdl_session_gene
    ON session_delivery_log(session_id, gene_id);
CREATE INDEX IF NOT EXISTS idx_sdl_session_time
    ON session_delivery_log(session_id, delivered_at);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the session_delivery_log table + indexes if absent.

    Idempotent — safe to call on every genome open. Production sqlite
    instances go through `Genome._create_schema` which already calls
    this path; in-memory test fixtures may call it directly.
    """
    conn.executescript(SCHEMA_SQL)
    conn.commit()


# ── Content hashing ───────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """16-char hex prefix of sha256(text) — stable, cheap, short.

    Used to detect re-expression: if the same gene is spliced differently
    on a later call, its content hash changes and the consumer sees a
    *new* delivery rather than a stale elision stub.

    16 chars = 64 bits, collision-resistant enough for per-session use
    while staying compact in the header-line budget.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── DAL: log_delivery / already_delivered / manifest ──────────────────

def log_delivery(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    gene_id: str,
    content_hash: Optional[str] = None,
    mode: Optional[str] = None,
    retrieval_id: Optional[int] = None,
    ts: Optional[float] = None,
) -> int:
    """Insert one delivery row. Returns the new delivery_id.

    `ts` defaults to now. All metadata columns are nullable so callers
    that don't have (e.g.) the cwola retrieval_id can still log.
    """
    if ts is None:
        ts = time.time()
    cur = conn.execute(
        "INSERT INTO session_delivery_log "
        "(session_id, gene_id, retrieval_id, delivered_at, content_hash, mode) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, gene_id, retrieval_id, ts, content_hash, mode),
    )
    conn.commit()
    return int(cur.lastrowid)


def already_delivered(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    gene_id: str,
    since: Optional[float] = None,
) -> Optional[Tuple[float, Optional[str], Optional[str]]]:
    """Returns `(delivered_at, mode, content_hash)` for the most recent
    delivery of `gene_id` in `session_id`, or None if never delivered.

    If `since` is set, only deliveries at-or-after that epoch count.
    Intended use: caller passes `since = session_start_ts` to confine
    the check to the current logical session rather than ever-.
    """
    if since is not None:
        row = conn.execute(
            "SELECT delivered_at, mode, content_hash "
            "FROM session_delivery_log "
            "WHERE session_id = ? AND gene_id = ? AND delivered_at >= ? "
            "ORDER BY delivered_at DESC LIMIT 1",
            (session_id, gene_id, since),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT delivered_at, mode, content_hash "
            "FROM session_delivery_log "
            "WHERE session_id = ? AND gene_id = ? "
            "ORDER BY delivered_at DESC LIMIT 1",
            (session_id, gene_id),
        ).fetchone()
    if row is None:
        return None
    return float(row["delivered_at"]), row["mode"], row["content_hash"]


def count_deliveries_since(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    gene_id: str,
    since: float,
) -> int:
    """Number of deliveries of `gene_id` in `session_id` since `since`."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM session_delivery_log "
        "WHERE session_id = ? AND gene_id = ? AND delivered_at >= ?",
        (session_id, gene_id, since),
    ).fetchone()
    return int(row["n"]) if row else 0


def session_manifest(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    limit: int = 500,
) -> List[dict]:
    """All deliveries for a session, most recent first. Backs the
    `/session/{id}/manifest` introspection endpoint.
    """
    rows = conn.execute(
        "SELECT delivery_id, gene_id, delivered_at, content_hash, mode, retrieval_id "
        "FROM session_delivery_log "
        "WHERE session_id = ? "
        "ORDER BY delivered_at DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def count_queries_in_session_since(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    since: float,
) -> int:
    """How many /context calls landed in this session AFTER `since`?

    Uses `cwola_log` (which logs one row per /context call) rather than
    `session_delivery_log` (which logs one row per delivered gene). The
    distinction matters for the "N queries ago" marker — otherwise a
    single 12-gene response would inflate the age count 12x.

    Returns 0 if cwola_log doesn't exist in the connection (tests, etc.).
    """
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM cwola_log "
            "WHERE session_id = ? AND ts > ?",
            (session_id, since),
        ).fetchone()
        return int(row["n"]) if row else 0
    except sqlite3.OperationalError:
        # cwola_log absent — tests, or a minimal genome without it
        return 0


# ── Consumer-facing elision stub ──────────────────────────────────────

def format_elision_stub(
    *,
    gene_id: str,
    delivered_at: float,
    now: float,
    queries_ago: int,
    id_width: int = 12,
) -> str:
    """One-line stub replacing a gene's spliced text when the same gene
    was already delivered earlier in this session.

    Shape matches Sprint 1's `[gene=abc12345 ...]` header so downstream
    parsers can treat them uniformly:

        [gene=abc12345 ↻ delivered 3 queries ago / 45s — see earlier response]

    The ↻ glyph is chosen to visually distinguish from ◆/◇/⬦ confidence
    markers — it's a "same thing, already shipped" signal, not a quality
    signal.
    """
    short_id = gene_id[:id_width]
    age_s = max(0.0, now - delivered_at)
    if age_s < 60:
        age_str = f"{age_s:.0f}s"
    elif age_s < 3600:
        age_str = f"{age_s / 60:.0f}m"
    else:
        age_str = f"{age_s / 3600:.1f}h"
    qa = f"{queries_ago} queries ago" if queries_ago > 0 else "just now"
    return f"[gene={short_id} ↻ delivered {qa} / {age_str} — see earlier response]"

"""
CWoLa label logger - STATISTICAL_FUSION.md sect C2 (Sprint 1 half).

Captures per-query rows so the Sprint 3 trainer can bucket them into
A (accepted: no re-query within 60s) and B (re-queried within 60s on
the same session) and train a classifier under the Metodiev/Nachman/
Thaler 2017 (arXiv:1708.02949) Classification Without Labels theorem.

This module only logs and assigns buckets lazily. The trainer is a
separate Sprint 3 module that reads this table.

Sprint 1 uses same-session-within-60s as the bucket-B proxy, ignoring
semantic similarity. The Sprint 3 trainer can recompute buckets using
query embeddings (Singh 2020 Context Mover's Distance or cosine) before
fitting.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any, Dict, Optional

log = logging.getLogger("helix.cwola")

BUCKET_WINDOW_S = 60.0  # same-session re-query counts as Bucket B within this


def log_query(
    conn: sqlite3.Connection,
    *,
    session_id: Optional[str],
    party_id: Optional[str],
    query: str,
    tier_totals: Dict[str, float],
    top_gene_id: Optional[str],
    ts: Optional[float] = None,
) -> Optional[int]:
    """Append one CWoLa log row. Returns the row id, or None on failure.

    tier_totals is the per-query sum-of-contributions dict (the same
    object surfaced on /context verbose=true responses). It is stored
    as JSON so the trainer can extract an ordered feature vector later
    without schema migrations when new tiers ship.
    """
    if ts is None:
        ts = time.time()
    try:
        features_json = json.dumps(tier_totals, sort_keys=True)
    except Exception:
        log.debug("tier_totals not JSON-serialisable; storing empty", exc_info=True)
        features_json = "{}"
    try:
        cur = conn.execute(
            """
            INSERT INTO cwola_log
                (ts, session_id, party_id, query, tier_features,
                 top_gene_id, bucket, bucket_assigned_at, requery_delta_s)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            """,
            (ts, session_id, party_id, query, features_json, top_gene_id),
        )
        conn.commit()
        try:
            from .telemetry import cwola_bucket_counter
            cwola_bucket_counter().add(1, {"bucket": "pending"})
        except Exception:
            pass
        return int(cur.lastrowid)
    except Exception as exc:
        log.warning("CWoLa log_query failed: %s", exc, exc_info=True)
        return None


def sweep_buckets(conn: sqlite3.Connection, now: Optional[float] = None) -> int:
    """Assign buckets to pending entries older than BUCKET_WINDOW_S.

    An entry is assigned:
      'B' if there is any row with the same session_id and
          0 < (other.ts - this.ts) <= BUCKET_WINDOW_S
      'A' otherwise.

    Only rows older than BUCKET_WINDOW_S are eligible - anything newer
    could still flip to 'B' if a re-query arrives. Returns count of
    rows updated.
    """
    if now is None:
        now = time.time()
    cutoff = now - BUCKET_WINDOW_S
    try:
        rows = conn.execute(
            "SELECT retrieval_id, session_id, ts FROM cwola_log "
            "WHERE bucket IS NULL AND ts <= ?",
            (cutoff,),
        ).fetchall()
    except Exception:
        log.debug("sweep_buckets read failed", exc_info=True)
        return 0

    updates = 0
    for rid, session_id, ts in rows:
        if not session_id:
            bucket, delta = "A", None
        else:
            next_row = conn.execute(
                "SELECT ts FROM cwola_log "
                "WHERE session_id = ? AND ts > ? AND ts <= ? "
                "ORDER BY ts ASC LIMIT 1",
                (session_id, ts, ts + BUCKET_WINDOW_S),
            ).fetchone()
            if next_row:
                bucket, delta = "B", float(next_row[0]) - float(ts)
            else:
                bucket, delta = "A", None
        try:
            conn.execute(
                "UPDATE cwola_log SET bucket = ?, bucket_assigned_at = ?, "
                "requery_delta_s = ? WHERE retrieval_id = ?",
                (bucket, now, delta, rid),
            )
            updates += 1
        except Exception:
            log.debug("sweep_buckets update failed for %s", rid, exc_info=True)
    if updates:
        conn.commit()
        # Emit one counter tick per newly-assigned bucket + a gauge .set()
        # with the current f_gap_sq divergence (Gauge sets absolute value
        # rather than accumulating deltas).
        try:
            from .telemetry import cwola_bucket_counter, cwola_f_gap_gauge
            # Re-read the just-assigned rows to label them correctly.
            rows = conn.execute(
                "SELECT bucket, COUNT(*) FROM cwola_log "
                "WHERE bucket_assigned_at >= ? GROUP BY bucket",
                (now - 1.0,),
            ).fetchall()
            for bucket, n in rows:
                cwola_bucket_counter().add(int(n), {"bucket": bucket or "unassigned"})
            s = stats(conn)
            if s.get("f_gap_sq") is not None:
                cwola_f_gap_gauge().set(float(s["f_gap_sq"]))
        except Exception:
            log.debug("cwola telemetry emit failed", exc_info=True)
    return updates


def stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Coarse-grained counters for bench / dashboard observability."""
    try:
        row = conn.execute(
            "SELECT "
            "  COUNT(*), "
            "  SUM(CASE WHEN bucket='A' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN bucket='B' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN bucket IS NULL THEN 1 ELSE 0 END) "
            "FROM cwola_log"
        ).fetchone()
    except Exception:
        return {"total": 0, "a": 0, "b": 0, "pending": 0, "f_gap_sq": None}
    total, a, b, pending = (row[0] or 0, row[1] or 0, row[2] or 0, row[3] or 0)
    resolved = a + b
    gap_sq = None
    if resolved >= 2:
        f_a = a / resolved
        gap_sq = (f_a - (1 - f_a)) ** 2
    return {"total": total, "a": a, "b": b, "pending": pending, "f_gap_sq": gap_sq}

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
import math
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("helix.cwola")

BUCKET_WINDOW_S = 60.0  # same-session re-query counts as Bucket B within this

# STATISTICAL_FUSION.md §C2: a re-query within BUCKET_WINDOW_S only counts as
# B if query_t and query_{t+1} are "textually related" — cosine sim of the
# 20-d ΣĒMA embeddings above this threshold. Re-queries below it indicate
# the user moved on to a different topic (accept, not dissatisfaction).
# 0.4 is the spec default; offline evaluation in SPRINT3_TRAINER_2026-04-21.md
# showed 0.7 gives the best classifier AUC. Tune via sweep_buckets' kwarg.
BUCKET_SEMA_COS_THRESHOLD = 0.4


def _cos_from_jsons(a_json: Any, b_json: Any) -> Optional[float]:
    """Cosine of two JSON-encoded embedding vectors. None when unavailable.

    Returns None (caller falls back to the time-only rule) when either
    payload is missing, malformed, dimensionally mismatched, or zero-norm.
    """
    if not a_json or not b_json:
        return None
    try:
        a = json.loads(a_json)
        b = json.loads(b_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(a, list) or not isinstance(b, list):
        return None
    if not a or len(a) != len(b):
        return None
    try:
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a))
        nb = math.sqrt(sum(float(y) * float(y) for y in b))
    except (TypeError, ValueError):
        return None
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)


def _embed_json(vec: Optional[Sequence[float]]) -> Optional[str]:
    """Serialise an embedding vector for cwola_log storage. None-safe."""
    if vec is None:
        return None
    try:
        return json.dumps([float(x) for x in vec])
    except Exception:
        log.debug("embedding not JSON-serialisable; storing NULL", exc_info=True)
        return None


def log_query(
    conn: sqlite3.Connection,
    *,
    session_id: Optional[str],
    party_id: Optional[str],
    query: str,
    tier_totals: Dict[str, float],
    top_gene_id: Optional[str],
    ts: Optional[float] = None,
    query_sema: Optional[Sequence[float]] = None,
    top_candidate_sema: Optional[Sequence[float]] = None,
) -> Optional[int]:
    """Append one CWoLa log row. Returns the row id, or None on failure.

    tier_totals is the per-query sum-of-contributions dict (the same
    object surfaced on /context verbose=true responses). It is stored
    as JSON so the trainer can extract an ordered feature vector later
    without schema migrations when new tiers ship.

    query_sema / top_candidate_sema are optional 20d SEMA vectors captured
    at retrieval time (PWPC Phase 1 enrichment — see
    docs/collab/comms/REPLY_PWPC_FROM_LAUDE.md). Stored as JSON lists;
    NULL for rows logged before this column landed or when the codec
    is unavailable.
    """
    if ts is None:
        ts = time.time()
    try:
        features_json = json.dumps(tier_totals, sort_keys=True)
    except Exception:
        log.debug("tier_totals not JSON-serialisable; storing empty", exc_info=True)
        features_json = "{}"
    query_sema_json = _embed_json(query_sema)
    top_candidate_sema_json = _embed_json(top_candidate_sema)
    try:
        cur = conn.execute(
            """
            INSERT INTO cwola_log
                (ts, session_id, party_id, query, tier_features,
                 top_gene_id, bucket, bucket_assigned_at, requery_delta_s,
                 query_sema, top_candidate_sema)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
            """,
            (
                ts, session_id, party_id, query, features_json, top_gene_id,
                query_sema_json, top_candidate_sema_json,
            ),
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


def sweep_buckets(
    conn: sqlite3.Connection,
    now: Optional[float] = None,
    *,
    cos_threshold: float = BUCKET_SEMA_COS_THRESHOLD,
) -> int:
    """Assign buckets to pending entries older than BUCKET_WINDOW_S.

    An entry is assigned:
      'B' if there is a same-session row with 0 < (other.ts - this.ts) <=
          BUCKET_WINDOW_S AND cos(this.query_sema, other.query_sema) >
          cos_threshold (STATISTICAL_FUSION.md §C2 intent-delta filter).
          Legacy rows lacking one or both sema vectors fall back to the
          time-only rule so behavior is preserved for pre-PWPC-Phase-1 data.
      'A' otherwise — including re-queries that fail the cos filter (user
          changed topic, not dissatisfied).

    Only rows older than BUCKET_WINDOW_S are eligible — anything newer could
    still flip to 'B' if a re-query arrives. Returns count of rows updated.
    """
    if now is None:
        now = time.time()
    cutoff = now - BUCKET_WINDOW_S
    try:
        rows = conn.execute(
            "SELECT retrieval_id, session_id, ts, query_sema FROM cwola_log "
            "WHERE bucket IS NULL AND ts <= ?",
            (cutoff,),
        ).fetchall()
    except Exception:
        log.warning("sweep_buckets read failed", exc_info=True)
        return 0

    updates = 0
    updated_rids: List[Any] = []
    n_filtered = 0
    n_fallback_legacy = 0
    for rid, session_id, ts, this_sema in rows:
        if not session_id:
            bucket, delta = "A", None
        else:
            next_row = conn.execute(
                "SELECT ts, query_sema FROM cwola_log "
                "WHERE session_id = ? AND ts > ? AND ts <= ? "
                "ORDER BY ts ASC LIMIT 1",
                (session_id, ts, ts + BUCKET_WINDOW_S),
            ).fetchone()
            if not next_row:
                bucket, delta = "A", None
            else:
                next_ts, next_sema = next_row
                cos = _cos_from_jsons(this_sema, next_sema)
                if cos is None:
                    # Legacy: one or both sema missing. Preserve the old
                    # time-only rule so historical data still buckets.
                    bucket, delta = "B", float(next_ts) - float(ts)
                    n_fallback_legacy += 1
                elif cos > cos_threshold:
                    bucket, delta = "B", float(next_ts) - float(ts)
                else:
                    # Intent-delta filter: re-query was topically unrelated.
                    # User moved on — treat as accept, not dissatisfaction.
                    bucket, delta = "A", None
                    n_filtered += 1
        try:
            conn.execute(
                "UPDATE cwola_log SET bucket = ?, bucket_assigned_at = ?, "
                "requery_delta_s = ? WHERE retrieval_id = ?",
                (bucket, now, delta, rid),
            )
            updates += 1
            updated_rids.append(rid)
        except Exception:
            log.warning("sweep_buckets update failed for %s", rid, exc_info=True)
    if updates:
        conn.commit()
        # Emit one counter tick per newly-assigned bucket + a gauge .set()
        # with the current f_gap_sq divergence (Gauge sets absolute value
        # rather than accumulating deltas).
        try:
            from .telemetry import cwola_bucket_counter, cwola_f_gap_gauge
            # Re-read the just-assigned rows by retrieval_id rather than a
            # time-window query — a 1s window races with commit latency,
            # skipping rows that committed slightly later than `now-1.0`.
            # Chunk the IN-list to stay under SQLite's 999-parameter cap.
            SQLITE_PARAM_CAP = 900
            bucket_counts: Dict[str, int] = {}
            for i in range(0, len(updated_rids), SQLITE_PARAM_CAP):
                chunk = updated_rids[i:i + SQLITE_PARAM_CAP]
                placeholders = ",".join("?" * len(chunk))
                chunk_rows = conn.execute(
                    f"SELECT bucket, COUNT(*) FROM cwola_log "
                    f"WHERE retrieval_id IN ({placeholders}) GROUP BY bucket",
                    tuple(chunk),
                ).fetchall()
                for bucket, n in chunk_rows:
                    key = bucket or "unassigned"
                    bucket_counts[key] = bucket_counts.get(key, 0) + int(n)
            for bucket, n in bucket_counts.items():
                cwola_bucket_counter().add(int(n), {"bucket": bucket})
            s = stats(conn)
            if s.get("f_gap_sq") is not None:
                cwola_f_gap_gauge().set(float(s["f_gap_sq"]))
        except Exception:
            log.debug("cwola telemetry emit failed", exc_info=True)
    if n_filtered or n_fallback_legacy:
        log.info(
            "sweep_buckets: intent-delta filter reassigned %d row(s) "
            "B->A (cos <= %.2f); %d row(s) used legacy time-only rule "
            "(sema missing)", n_filtered, cos_threshold, n_fallback_legacy,
        )
    return updates


# ---------------------------------------------------------------------------
# Sliding-window correlation-matrix feature extractor
#
# Per docs/collab/comms/LOCKSTEP_MATRIX_FINDINGS_2026-04-14.md: the scalar
# lockstep gate failed at |r|>=0.2, but the population correlation matrices
# between A and B buckets diverged sharply (Frobenius 1.29, every top-delta
# entry involving sema_boost). This extractor surfaces that population-level
# structure as a per-retrieval feature by correlating tier scores over a
# rolling window of the same session's recent retrievals.
#
# Consumers: batman's agreement head (PWPC manifold) + the Sprint 3 PLR
# trainer. Feature shape: dict of 36 unique off-diagonal correlation entries
# keyed "tier_i__tier_j" where (i, j) index in the canonical tier order below.
# ---------------------------------------------------------------------------

TIER_ORDER: List[str] = [
    "fts5", "splade", "sema_boost", "lex_anchor",
    "tag_exact", "tag_prefix", "pki", "harmonic", "sr",
]

# Minimum rows needed for a stable window correlation. Below this we emit
# degenerate=True and callers should skip the feature.
MIN_WINDOW_ROWS = 2


def sliding_window_features(
    conn: sqlite3.Connection,
    *,
    session_id: Optional[str],
    before_ts: float,
    window_size: int = 50,
    tier_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Correlation-matrix feature vector over the last `window_size` retrievals.

    Returns a dict:
        {
          "n_rows": int,                 # actual number of rows used
          "degenerate": bool,            # True if extraction not meaningful
          "features": {"tier_i__tier_j": float, ...},  # 36 entries when ok
          "reason": Optional[str],       # populated when degenerate
        }

    Degenerate cases (features returned empty or zero, no-op for consumers):
      - No rows for this session before `before_ts`.
      - Only 1 row (can't correlate a single observation).
      - All rows have identical tier scores (zero variance → undefined corr).
      - numpy not importable (falls back with explicit reason).

    The window is strictly pre-`before_ts` — feature computed from history,
    never leaks current-row info forward into itself.

    Tier scores that never fired in a column are left at 0. Columns with
    zero variance across the window produce 0 correlations (not NaN).
    """
    tiers: List[str] = list(tier_order) if tier_order else list(TIER_ORDER)
    n_tiers = len(tiers)
    empty: Dict[str, Any] = {"n_rows": 0, "degenerate": True, "features": {}, "reason": None}

    if not session_id:
        empty["reason"] = "no session_id"
        return empty

    try:
        rows = conn.execute(
            "SELECT tier_features FROM cwola_log "
            "WHERE session_id = ? AND ts < ? "
            "ORDER BY ts DESC LIMIT ?",
            (session_id, before_ts, window_size),
        ).fetchall()
    except Exception:
        log.debug("sliding_window_features read failed", exc_info=True)
        empty["reason"] = "db read failed"
        return empty

    n = len(rows)
    if n < MIN_WINDOW_ROWS:
        empty["n_rows"] = n
        empty["reason"] = f"n_rows={n} < MIN_WINDOW_ROWS={MIN_WINDOW_ROWS}"
        return empty

    try:
        import numpy as np
    except ImportError:
        empty["n_rows"] = n
        empty["reason"] = "numpy not available"
        return empty

    # Build X matrix (n × n_tiers). Missing tiers left at 0.
    X = np.zeros((n, n_tiers), dtype=float)
    for i, (feat_json,) in enumerate(rows):
        if not feat_json:
            continue
        try:
            feat = json.loads(feat_json)
        except Exception:
            continue
        if not isinstance(feat, dict):
            continue
        for j, tier in enumerate(tiers):
            v = feat.get(tier)
            if v is None:
                continue
            try:
                X[i, j] = float(v)
            except (TypeError, ValueError):
                continue

    # Zero-variance columns produce NaN rows/cols in corrcoef — we mask those
    # to 0 in the output so consumers can treat them as "no signal" uniformly.
    col_std = X.std(axis=0, ddof=0)
    zero_var_cols = col_std < 1e-12
    if zero_var_cols.all():
        return {
            "n_rows": n,
            "degenerate": True,
            "features": {},
            "reason": "all tier columns have zero variance",
        }

    # np.corrcoef emits a RuntimeWarning for zero-variance columns ("invalid
    # value encountered in divide"). Suppress it; we handle the NaNs below.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract 36 unique off-diagonal entries (i < j) in canonical tier order.
    features: Dict[str, float] = {}
    for i in range(n_tiers):
        for j in range(i + 1, n_tiers):
            features[f"{tiers[i]}__{tiers[j]}"] = float(C[i, j])

    return {
        "n_rows": n,
        "degenerate": False,
        "features": features,
        "reason": None,
    }


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

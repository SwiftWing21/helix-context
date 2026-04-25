"""Tests for CWoLa label logger (STATISTICAL_FUSION sect C2, Sprint 1)."""

import json
import sqlite3
import time

import pytest

from helix_context import cwola


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript("""
    CREATE TABLE cwola_log (
        retrieval_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        ts                 REAL NOT NULL,
        session_id         TEXT,
        party_id           TEXT,
        query              TEXT,
        tier_features      TEXT,
        top_gene_id        TEXT,
        bucket             TEXT,
        bucket_assigned_at REAL,
        requery_delta_s    REAL,
        query_sema         TEXT,
        top_candidate_sema TEXT
    );
    CREATE INDEX idx_cwola_session_time ON cwola_log(session_id, ts);
    CREATE INDEX idx_cwola_bucket ON cwola_log(bucket);
    """)
    yield c
    c.close()


def test_log_query_writes_row(conn):
    rid = cwola.log_query(
        conn,
        session_id="s1",
        party_id="alice",
        query="what port does helix use",
        tier_totals={"pki": 12.0, "fts5": 3.5},
        top_gene_id="g_001",
    )
    assert rid is not None
    row = conn.execute(
        "SELECT session_id, party_id, query, tier_features, top_gene_id, bucket "
        "FROM cwola_log WHERE retrieval_id=?", (rid,)
    ).fetchone()
    assert row[0] == "s1"
    assert row[1] == "alice"
    assert row[2] == "what port does helix use"
    assert json.loads(row[3]) == {"pki": 12.0, "fts5": 3.5}
    assert row[4] == "g_001"
    assert row[5] is None  # bucket pending


def test_sweep_assigns_A_to_lone_query(conn):
    t = 1000.0
    rid = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q",
        tier_totals={}, top_gene_id=None, ts=t,
    )
    # Sweep 120s later — nothing else in the session, should be A
    updated = cwola.sweep_buckets(conn, now=t + 120)
    assert updated == 1
    bucket, delta = conn.execute(
        "SELECT bucket, requery_delta_s FROM cwola_log WHERE retrieval_id=?",
        (rid,),
    ).fetchone()
    assert bucket == "A"
    assert delta is None


def test_sweep_assigns_B_on_requery_within_60s(conn):
    t = 1000.0
    rid1 = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q1",
        tier_totals={}, top_gene_id=None, ts=t,
    )
    cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q2",
        tier_totals={}, top_gene_id=None, ts=t + 30,
    )
    # Sweep far in the future so both rows are eligible
    cwola.sweep_buckets(conn, now=t + 600)
    row1 = conn.execute(
        "SELECT bucket, requery_delta_s FROM cwola_log WHERE retrieval_id=?",
        (rid1,),
    ).fetchone()
    assert row1[0] == "B"
    assert row1[1] == pytest.approx(30.0)


def test_sweep_assigns_A_when_requery_outside_window(conn):
    t = 1000.0
    rid1 = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q1",
        tier_totals={}, top_gene_id=None, ts=t,
    )
    cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q2",
        tier_totals={}, top_gene_id=None, ts=t + 120,  # outside 60s window
    )
    cwola.sweep_buckets(conn, now=t + 600)
    bucket = conn.execute(
        "SELECT bucket FROM cwola_log WHERE retrieval_id=?", (rid1,),
    ).fetchone()[0]
    assert bucket == "A"


def test_sweep_does_not_assign_within_window(conn):
    """Rows younger than BUCKET_WINDOW_S must remain pending — a
    re-query could still arrive and flip them to B."""
    t = time.time()
    rid = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q",
        tier_totals={}, top_gene_id=None, ts=t,
    )
    cwola.sweep_buckets(conn, now=t + 10)  # only 10s elapsed
    bucket = conn.execute(
        "SELECT bucket FROM cwola_log WHERE retrieval_id=?", (rid,),
    ).fetchone()[0]
    assert bucket is None


def test_sweep_isolates_sessions(conn):
    """Two different sessions with near-simultaneous queries must not
    cross-assign buckets."""
    t = 1000.0
    rid_s1 = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q",
        tier_totals={}, top_gene_id=None, ts=t,
    )
    cwola.log_query(
        conn, session_id="s2", party_id="bob", query="q",
        tier_totals={}, top_gene_id=None, ts=t + 5,
    )
    cwola.sweep_buckets(conn, now=t + 600)
    bucket = conn.execute(
        "SELECT bucket FROM cwola_log WHERE retrieval_id=?", (rid_s1,),
    ).fetchone()[0]
    assert bucket == "A"  # different sessions do not count as re-query


def test_log_query_stores_sema_vectors(conn):
    """PWPC Phase 1: query_sema + top_candidate_sema passthrough."""
    q_vec = [0.1 * i for i in range(20)]
    c_vec = [-0.05 * i for i in range(20)]
    rid = cwola.log_query(
        conn,
        session_id="s1", party_id="alice", query="q",
        tier_totals={"fts5": 1.0}, top_gene_id="g_001",
        query_sema=q_vec, top_candidate_sema=c_vec,
    )
    assert rid is not None
    row = conn.execute(
        "SELECT query_sema, top_candidate_sema "
        "FROM cwola_log WHERE retrieval_id=?", (rid,),
    ).fetchone()
    assert json.loads(row[0]) == pytest.approx(q_vec)
    assert json.loads(row[1]) == pytest.approx(c_vec)


def test_log_query_sema_vectors_optional(conn):
    """Missing embeddings stay NULL rather than failing the insert."""
    rid = cwola.log_query(
        conn,
        session_id="s1", party_id="alice", query="q",
        tier_totals={}, top_gene_id=None,
    )
    assert rid is not None
    row = conn.execute(
        "SELECT query_sema, top_candidate_sema "
        "FROM cwola_log WHERE retrieval_id=?", (rid,),
    ).fetchone()
    assert row[0] is None
    assert row[1] is None


def test_sweep_filters_B_when_next_query_is_unrelated(conn):
    """STATISTICAL_FUSION.md §C2 intent-delta filter — re-query within 60s
    with cos(query_sema, next.query_sema) <= threshold is treated as A
    (user moved on to a different topic, not dissatisfaction)."""
    t = 1000.0
    q_sema = [1.0, 0.0] + [0.0] * 18
    unrelated = [0.0, 1.0] + [0.0] * 18  # orthogonal → cos = 0
    rid1 = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="how do I invert a matrix",
        tier_totals={}, top_gene_id=None, ts=t, query_sema=q_sema,
    )
    cwola.log_query(
        conn, session_id="s1", party_id="alice", query="what time is lunch",
        tier_totals={}, top_gene_id=None, ts=t + 30, query_sema=unrelated,
    )
    cwola.sweep_buckets(conn, now=t + 600)
    bucket, delta = conn.execute(
        "SELECT bucket, requery_delta_s FROM cwola_log WHERE retrieval_id=?",
        (rid1,),
    ).fetchone()
    assert bucket == "A"
    assert delta is None


def test_sweep_keeps_B_when_next_query_is_related(conn):
    """Same topic re-query clears the cos filter and stays labelled B."""
    t = 1000.0
    q_sema = [1.0, 0.0] + [0.0] * 18
    related = [0.9, 0.1] + [0.0] * 18  # cos ≈ 0.994 > 0.4
    rid1 = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="how do I invert a matrix",
        tier_totals={}, top_gene_id=None, ts=t, query_sema=q_sema,
    )
    cwola.log_query(
        conn, session_id="s1", party_id="alice", query="show me matrix inversion",
        tier_totals={}, top_gene_id=None, ts=t + 30, query_sema=related,
    )
    cwola.sweep_buckets(conn, now=t + 600)
    bucket, delta = conn.execute(
        "SELECT bucket, requery_delta_s FROM cwola_log WHERE retrieval_id=?",
        (rid1,),
    ).fetchone()
    assert bucket == "B"
    assert delta == pytest.approx(30.0)


def test_sweep_falls_back_to_time_rule_when_sema_missing(conn):
    """Legacy rows without query_sema preserve pre-PWPC-Phase-1 behavior:
    any same-session re-query within 60s is B (time-only rule)."""
    t = 1000.0
    rid1 = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q1",
        tier_totals={}, top_gene_id=None, ts=t,  # no query_sema
    )
    cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q2",
        tier_totals={}, top_gene_id=None, ts=t + 30,  # no query_sema
    )
    cwola.sweep_buckets(conn, now=t + 600)
    bucket = conn.execute(
        "SELECT bucket FROM cwola_log WHERE retrieval_id=?", (rid1,),
    ).fetchone()[0]
    assert bucket == "B"


def test_sweep_falls_back_when_only_next_sema_missing(conn):
    """Asymmetric case: current row has sema, next row doesn't → legacy
    fallback preserves B assignment rather than discarding the signal."""
    t = 1000.0
    q_sema = [1.0, 0.0] + [0.0] * 18
    rid1 = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q1",
        tier_totals={}, top_gene_id=None, ts=t, query_sema=q_sema,
    )
    cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q2",
        tier_totals={}, top_gene_id=None, ts=t + 30,  # no query_sema
    )
    cwola.sweep_buckets(conn, now=t + 600)
    bucket = conn.execute(
        "SELECT bucket FROM cwola_log WHERE retrieval_id=?", (rid1,),
    ).fetchone()[0]
    assert bucket == "B"


def test_sweep_honors_custom_cos_threshold(conn):
    """Tighter threshold (t07 from SPRINT3_TRAINER): the same related-but-not-
    identical re-query that passes at 0.4 should fail at 0.7."""
    t = 1000.0
    q_sema = [1.0, 0.0] + [0.0] * 18
    # Angle ~60°: cos ≈ 0.5, passes 0.4 but fails 0.7
    moderate = [0.5, 0.866] + [0.0] * 18
    rid1 = cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q1",
        tier_totals={}, top_gene_id=None, ts=t, query_sema=q_sema,
    )
    cwola.log_query(
        conn, session_id="s1", party_id="alice", query="q2",
        tier_totals={}, top_gene_id=None, ts=t + 30, query_sema=moderate,
    )
    cwola.sweep_buckets(conn, now=t + 600, cos_threshold=0.7)
    bucket = conn.execute(
        "SELECT bucket FROM cwola_log WHERE retrieval_id=?", (rid1,),
    ).fetchone()[0]
    assert bucket == "A"  # cos 0.5 <= 0.7 → filtered


def test_stats_reports_f_gap(conn):
    t = 1000.0
    cwola.log_query(conn, session_id="s1", party_id="a", query="q",
                    tier_totals={}, top_gene_id=None, ts=t)
    cwola.log_query(conn, session_id="s1", party_id="a", query="q2",
                    tier_totals={}, top_gene_id=None, ts=t + 5)
    cwola.log_query(conn, session_id="s2", party_id="b", query="q",
                    tier_totals={}, top_gene_id=None, ts=t + 10)
    cwola.sweep_buckets(conn, now=t + 600)
    s = cwola.stats(conn)
    assert s["total"] == 3
    assert s["a"] + s["b"] == 3
    assert s["pending"] == 0
    assert s["f_gap_sq"] is not None

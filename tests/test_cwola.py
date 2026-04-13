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
        requery_delta_s    REAL
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

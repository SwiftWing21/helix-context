"""Tests for sliding-window correlation-matrix feature extractor.

See docs/collab/comms/LOCKSTEP_MATRIX_FINDINGS_2026-04-14.md — the
scalar LOCKSTEP_TEST failed the |r|>=0.2 gate but the population
correlation matrices differed sharply between A and B, with every
top-delta entry involving `sema_boost`. This extractor surfaces
that population-level structure as per-retrieval features by
computing a rolling-window correlation matrix over the last K
retrievals in the same session.
"""

import json
import math
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
    """)
    yield c
    c.close()


def _seed(conn, session_id, ts, tier_features, party_id="p"):
    conn.execute(
        "INSERT INTO cwola_log (ts, session_id, party_id, query, tier_features, top_gene_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, session_id, party_id, "q", json.dumps(tier_features), "g"),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# degenerate cases
# ---------------------------------------------------------------------------


def test_empty_session_is_degenerate(conn):
    out = cwola.sliding_window_features(
        conn, session_id="nonexistent", before_ts=1000.0
    )
    assert out["n_rows"] == 0
    assert out["degenerate"] is True
    assert out["features"] == {}


def test_single_row_session_is_degenerate(conn):
    _seed(conn, "s1", 100.0, {"fts5": 1.0, "splade": 2.0})
    out = cwola.sliding_window_features(conn, session_id="s1", before_ts=200.0)
    # N=1 is below min threshold — can't correlate
    assert out["degenerate"] is True
    assert out["n_rows"] == 1


def test_degenerate_when_all_rows_have_zero_variance(conn):
    # Every row has identical tier_features → correlation undefined
    for i in range(20):
        _seed(conn, "s1", 100.0 + i, {"fts5": 5.0, "splade": 5.0})
    out = cwola.sliding_window_features(conn, session_id="s1", before_ts=200.0)
    assert out["degenerate"] is True


# ---------------------------------------------------------------------------
# session / time isolation
# ---------------------------------------------------------------------------


def test_session_isolation(conn):
    # s1 has varied rows, s2 has constant rows → s2 remains degenerate
    for i in range(20):
        _seed(conn, "s1", 100.0 + i, {"fts5": float(i), "splade": float(i * 2)})
        _seed(conn, "s2", 100.0 + i, {"fts5": 5.0, "splade": 5.0})
    out_s1 = cwola.sliding_window_features(conn, session_id="s1", before_ts=200.0)
    out_s2 = cwola.sliding_window_features(conn, session_id="s2", before_ts=200.0)
    assert out_s1["degenerate"] is False
    assert out_s2["degenerate"] is True


def test_before_ts_filter_excludes_future_rows(conn):
    for i in range(20):
        _seed(conn, "s1", 100.0 + i, {"fts5": float(i), "splade": float(i * 2)})
    # cutoff at 110 → only rows 100..109 (10 rows) included
    out = cwola.sliding_window_features(conn, session_id="s1", before_ts=110.0)
    assert out["n_rows"] == 10


# ---------------------------------------------------------------------------
# signal recovery
# ---------------------------------------------------------------------------


def test_perfectly_correlated_tiers_yield_corr_1(conn):
    # fts5 == splade across 20 rows → corr(fts5, splade) == 1.0
    for i in range(20):
        _seed(conn, "s1", 100.0 + i, {"fts5": float(i), "splade": float(i)})
    out = cwola.sliding_window_features(conn, session_id="s1", before_ts=200.0)
    assert out["degenerate"] is False
    assert math.isclose(out["features"]["fts5__splade"], 1.0, abs_tol=1e-9)


def test_perfectly_anticorrelated_tiers_yield_corr_minus_1(conn):
    for i in range(20):
        _seed(conn, "s1", 100.0 + i, {"fts5": float(i), "splade": float(-i)})
    out = cwola.sliding_window_features(conn, session_id="s1", before_ts=200.0)
    assert math.isclose(out["features"]["fts5__splade"], -1.0, abs_tol=1e-9)


def test_independent_tiers_yield_low_magnitude_corr(conn):
    # fts5 ramps, splade alternates — near-zero correlation
    for i in range(40):
        sp = 1.0 if i % 2 == 0 else -1.0
        _seed(conn, "s1", 100.0 + i, {"fts5": float(i), "splade": sp})
    out = cwola.sliding_window_features(conn, session_id="s1", before_ts=200.0)
    assert abs(out["features"]["fts5__splade"]) < 0.2


# ---------------------------------------------------------------------------
# window sizing
# ---------------------------------------------------------------------------


def test_window_uses_most_recent_rows(conn):
    # First 30 rows: anticorrelated. Last 20 rows: correlated.
    # window_size=20 should capture only the correlated slice → corr ≈ +1
    for i in range(30):
        _seed(conn, "s1", 100.0 + i, {"fts5": float(i), "splade": float(-i)})
    for i in range(20):
        _seed(conn, "s1", 200.0 + i, {"fts5": float(i), "splade": float(i)})
    out = cwola.sliding_window_features(
        conn, session_id="s1", before_ts=300.0, window_size=20
    )
    assert out["n_rows"] == 20
    assert math.isclose(out["features"]["fts5__splade"], 1.0, abs_tol=1e-9)


def test_window_size_larger_than_available_uses_all(conn):
    for i in range(10):
        _seed(conn, "s1", 100.0 + i, {"fts5": float(i), "splade": float(i)})
    out = cwola.sliding_window_features(
        conn, session_id="s1", before_ts=200.0, window_size=100
    )
    assert out["n_rows"] == 10


# ---------------------------------------------------------------------------
# feature-vector shape
# ---------------------------------------------------------------------------


def test_feature_vector_has_36_unique_off_diagonal_entries(conn):
    # All 9 tiers present, varied values → full 36-entry feature vector
    tiers = ["fts5", "splade", "sema_boost", "lex_anchor",
             "tag_exact", "tag_prefix", "pki", "harmonic", "sr"]
    for i in range(20):
        _seed(conn, "s1", 100.0 + i,
              {t: float(i + idx) for idx, t in enumerate(tiers)})
    out = cwola.sliding_window_features(conn, session_id="s1", before_ts=200.0)
    # 9 choose 2 = 36 unique upper-triangle entries
    assert len(out["features"]) == 36
    # Keys should be "tier_i__tier_j" with i < j in the canonical tier order
    for key in out["features"]:
        a, b = key.split("__")
        assert a in tiers
        assert b in tiers
        assert tiers.index(a) < tiers.index(b)


def test_missing_tier_treated_as_zero(conn):
    # Only fts5 and splade appear; other tiers NaN → correlations to them = 0
    for i in range(20):
        _seed(conn, "s1", 100.0 + i, {"fts5": float(i), "splade": float(i * 2)})
    out = cwola.sliding_window_features(conn, session_id="s1", before_ts=200.0)
    # fts5__splade should be ~1
    assert math.isclose(out["features"]["fts5__splade"], 1.0, abs_tol=1e-9)
    # fts5__pki should be 0 (pki never fired → zero-variance column)
    assert out["features"]["fts5__pki"] == 0.0

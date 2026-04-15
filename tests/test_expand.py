"""
Sprint 3 /context/expand — unit + HTTP tests.

Covers:
- fetch_forward_neighbors, fetch_backward_neighbors: SQL against harmonic_links
- fetch_sideways_neighbors: reads gene.epigenetics.co_activated_with
- format_neighbor_compact: small JSON payload per hit
- filter by session delivery log when session_id is given
- HTTP /context/expand endpoint: direction validation, k-cap, empty cases

See docs/FUTURE/AI_CONSUMER_ROADMAP_2026-04-14.md Sprint 3.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from helix_context import expand, session_delivery
from helix_context.config import (
    BudgetConfig,
    GenomeConfig,
    HelixConfig,
    RibosomeConfig,
    ServerConfig,
)
from helix_context.context_manager import HelixContextManager
from helix_context.server import create_app
from tests.conftest import make_gene


# ── Helper fixtures ───────────────────────────────────────────────────

def _make_manager() -> HelixContextManager:
    cfg = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=4),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        synonym_map={},
    )
    return HelixContextManager(cfg)


def _seed_harmonic_link(conn, a, b, weight, source="co_retrieved"):
    import time as _t
    conn.execute(
        "INSERT OR REPLACE INTO harmonic_links "
        "(gene_id_a, gene_id_b, weight, updated_at, source) "
        "VALUES (?, ?, ?, ?, ?)",
        (a, b, weight, _t.time(), source),
    )
    conn.commit()


# ── fetch_forward_neighbors ───────────────────────────────────────────

def test_fetch_forward_neighbors_empty():
    mgr = _make_manager()
    try:
        out = expand.fetch_forward_neighbors(mgr.genome.conn, "nonexistent", k=5)
        assert out == []
    finally:
        mgr.close()


def test_fetch_forward_neighbors_returns_b_side():
    mgr = _make_manager()
    try:
        _seed_harmonic_link(mgr.genome.conn, "a", "b1", 0.9)
        _seed_harmonic_link(mgr.genome.conn, "a", "b2", 0.5)
        _seed_harmonic_link(mgr.genome.conn, "a", "b3", 0.1)
        out = expand.fetch_forward_neighbors(mgr.genome.conn, "a", k=5)
        # Returns (gene_id, score) tuples sorted by weight descending
        assert out == [("b1", 0.9), ("b2", 0.5), ("b3", 0.1)]
    finally:
        mgr.close()


def test_fetch_forward_neighbors_respects_k():
    mgr = _make_manager()
    try:
        for i in range(10):
            _seed_harmonic_link(mgr.genome.conn, "a", f"b{i}", 1.0 - i * 0.1)
        out = expand.fetch_forward_neighbors(mgr.genome.conn, "a", k=3)
        assert len(out) == 3
        assert out[0][0] == "b0"   # highest weight
    finally:
        mgr.close()


def test_fetch_forward_ignores_backward_edges():
    """forward means `gene_id_a = X`; backward edges should not appear."""
    mgr = _make_manager()
    try:
        _seed_harmonic_link(mgr.genome.conn, "other", "a", 0.9)
        out = expand.fetch_forward_neighbors(mgr.genome.conn, "a", k=5)
        assert out == []
    finally:
        mgr.close()


# ── fetch_backward_neighbors ──────────────────────────────────────────

def test_fetch_backward_neighbors_returns_a_side():
    mgr = _make_manager()
    try:
        _seed_harmonic_link(mgr.genome.conn, "a1", "x", 0.8)
        _seed_harmonic_link(mgr.genome.conn, "a2", "x", 0.3)
        out = expand.fetch_backward_neighbors(mgr.genome.conn, "x", k=5)
        assert out == [("a1", 0.8), ("a2", 0.3)]
    finally:
        mgr.close()


# ── fetch_sideways_neighbors ──────────────────────────────────────────

def test_fetch_sideways_empty_when_gene_absent():
    mgr = _make_manager()
    try:
        out = expand.fetch_sideways_neighbors(mgr.genome, "nonexistent", k=5)
        assert out == []
    finally:
        mgr.close()


def test_fetch_sideways_reads_co_activated_with():
    mgr = _make_manager()
    try:
        target = make_gene(
            content="target",
            gene_id="target0000000001",
            co_activated_with=["coact1", "coact2", "coact3"],
        )
        mgr.genome.upsert_gene(target)
        # And the co-activated genes so they can be looked up
        for gid in ("coact1", "coact2", "coact3"):
            mgr.genome.upsert_gene(make_gene(content=gid, gene_id=gid))
        out = expand.fetch_sideways_neighbors(mgr.genome, "target0000000001", k=5)
        returned_ids = [gid for gid, _ in out]
        assert set(returned_ids) == {"coact1", "coact2", "coact3"}
    finally:
        mgr.close()


def test_fetch_sideways_respects_k():
    mgr = _make_manager()
    try:
        coacts = [f"coact{i}" for i in range(10)]
        target = make_gene(
            content="target",
            gene_id="target0000000002",
            co_activated_with=coacts,
        )
        mgr.genome.upsert_gene(target)
        for gid in coacts:
            mgr.genome.upsert_gene(make_gene(content=gid, gene_id=gid))
        out = expand.fetch_sideways_neighbors(mgr.genome, "target0000000002", k=3)
        assert len(out) == 3
    finally:
        mgr.close()


# ── format_neighbor_compact ───────────────────────────────────────────

def test_format_neighbor_compact_includes_fields():
    g = make_gene(
        content="auth middleware",
        gene_id="abc123",
        domains=["auth", "security"],
        entities=["jwt"],
    )
    out = expand.format_neighbor_compact(g, score=0.85)
    assert out["gene_id"] == "abc123"
    assert out["score"] == 0.85
    assert "summary" in out
    assert "auth middleware" in out["summary"] or "auth" in out.get("domains", [])
    assert "auth" in out["domains"]
    assert "jwt" in out["entities"]


def test_format_neighbor_compact_rounds_score():
    g = make_gene(content="x", gene_id="xx")
    out = expand.format_neighbor_compact(g, score=0.123456789)
    # Should be compact — 3 decimals is plenty
    assert out["score"] == 0.123


# ── expand_neighbors (top-level orchestration) ────────────────────────

def test_expand_neighbors_filters_already_delivered():
    """Genes in session_delivery_log for this session should be skipped."""
    mgr = _make_manager()
    try:
        _seed_harmonic_link(mgr.genome.conn, "a", "b1", 0.9)
        _seed_harmonic_link(mgr.genome.conn, "a", "b2", 0.8)
        _seed_harmonic_link(mgr.genome.conn, "a", "b3", 0.7)
        for gid in ("b1", "b2", "b3"):
            mgr.genome.upsert_gene(make_gene(content=gid, gene_id=gid))
        # b2 already delivered in this session
        session_delivery.log_delivery(
            mgr.genome.conn, session_id="sess_X", gene_id="b2",
        )
        result = expand.expand_neighbors(
            mgr.genome, gene_id="a", direction="forward", k=5,
            session_id="sess_X",
        )
        neighbor_ids = [n["gene_id"] for n in result["neighbors"]]
        assert "b2" not in neighbor_ids
        assert "b1" in neighbor_ids
        assert "b3" in neighbor_ids
        assert result["skipped_delivered"] == 1
    finally:
        mgr.close()


def test_expand_neighbors_no_session_shows_all():
    mgr = _make_manager()
    try:
        _seed_harmonic_link(mgr.genome.conn, "a", "b1", 0.9)
        for gid in ("b1",):
            mgr.genome.upsert_gene(make_gene(content=gid, gene_id=gid))
        result = expand.expand_neighbors(
            mgr.genome, gene_id="a", direction="forward", k=5,
            session_id=None,
        )
        assert len(result["neighbors"]) == 1
        assert result["skipped_delivered"] == 0
    finally:
        mgr.close()


def test_expand_neighbors_skips_unknown_gene_rows():
    """harmonic_links may reference genes no longer in the `genes` table;
    don't include them in the response (no summary to show)."""
    mgr = _make_manager()
    try:
        _seed_harmonic_link(mgr.genome.conn, "a", "dangling_ref", 0.9)
        _seed_harmonic_link(mgr.genome.conn, "a", "present", 0.8)
        mgr.genome.upsert_gene(make_gene(content="present", gene_id="present"))
        result = expand.expand_neighbors(
            mgr.genome, gene_id="a", direction="forward", k=5,
        )
        neighbor_ids = [n["gene_id"] for n in result["neighbors"]]
        assert "dangling_ref" not in neighbor_ids
        assert "present" in neighbor_ids
    finally:
        mgr.close()


# ── HTTP endpoint ──────────────────────────────────────────────────

@pytest.fixture
def http_client():
    cfg = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=4),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        server=ServerConfig(upstream="http://localhost:11434"),
    )
    app = create_app(cfg)
    yield TestClient(app)


def test_endpoint_rejects_invalid_direction(http_client):
    resp = http_client.get("/context/expand?gene_id=a&direction=random")
    assert resp.status_code == 400


def test_endpoint_empty_response_for_unknown_gene(http_client):
    resp = http_client.get("/context/expand?gene_id=unknown&direction=forward")
    assert resp.status_code == 200
    data = resp.json()
    assert data["gene_id"] == "unknown"
    assert data["direction"] == "forward"
    assert data["neighbors"] == []
    assert data["count"] == 0


def test_endpoint_returns_forward_neighbors(http_client):
    conn = http_client.app.state.helix.genome.conn
    _seed_harmonic_link(conn, "a", "b1", 0.9)
    _seed_harmonic_link(conn, "a", "b2", 0.5)
    http_client.app.state.helix.genome.upsert_gene(
        make_gene(content="first hit", gene_id="b1"),
    )
    http_client.app.state.helix.genome.upsert_gene(
        make_gene(content="second hit", gene_id="b2"),
    )
    resp = http_client.get("/context/expand?gene_id=a&direction=forward&k=5")
    assert resp.status_code == 200
    data = resp.json()
    ids = [n["gene_id"] for n in data["neighbors"]]
    assert ids == ["b1", "b2"]
    assert data["count"] == 2


def test_endpoint_respects_k_cap(http_client):
    conn = http_client.app.state.helix.genome.conn
    for i in range(10):
        _seed_harmonic_link(conn, "a", f"b{i}", 1.0 - i * 0.05)
        http_client.app.state.helix.genome.upsert_gene(
            make_gene(content=f"b{i}", gene_id=f"b{i}"),
        )
    resp = http_client.get("/context/expand?gene_id=a&direction=forward&k=3")
    data = resp.json()
    assert data["count"] == 3


def test_endpoint_filters_delivered_when_session_given(http_client):
    conn = http_client.app.state.helix.genome.conn
    _seed_harmonic_link(conn, "a", "b1", 0.9)
    _seed_harmonic_link(conn, "a", "b2", 0.5)
    for gid in ("b1", "b2"):
        http_client.app.state.helix.genome.upsert_gene(
            make_gene(content=gid, gene_id=gid),
        )
    session_delivery.log_delivery(conn, session_id="sess_X", gene_id="b1")
    resp = http_client.get(
        "/context/expand?gene_id=a&direction=forward&k=5&session_id=sess_X"
    )
    data = resp.json()
    ids = [n["gene_id"] for n in data["neighbors"]]
    assert ids == ["b2"]
    assert data["skipped_delivered"] == 1

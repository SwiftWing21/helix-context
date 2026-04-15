"""
Sprint 2 session working-set register — DAL unit tests.

Covers:
- ensure_schema: idempotent, creates table + indexes
- content_hash: stable, distinguishes different inputs
- log_delivery: inserts row, returns id
- already_delivered: returns most recent, respects `since` cutoff
- count_deliveries_since: counts inclusive of gene deliveries
- session_manifest: ordered descending by delivery time, respects limit
- session isolation: two sessions don't see each other's deliveries

See docs/FUTURE/AI_CONSUMER_ROADMAP_2026-04-14.md Sprint 2.
"""

from __future__ import annotations

import sqlite3

import pytest

from helix_context import session_delivery


@pytest.fixture
def conn():
    """Fresh in-memory sqlite with session_delivery schema applied."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    session_delivery.ensure_schema(c)
    yield c
    c.close()


# ── ensure_schema ─────────────────────────────────────────────────────

def test_ensure_schema_creates_table(conn):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='session_delivery_log'"
    ).fetchone()
    assert row is not None


def test_ensure_schema_is_idempotent(conn):
    # Calling again shouldn't error
    session_delivery.ensure_schema(conn)
    session_delivery.ensure_schema(conn)


def test_ensure_schema_creates_indexes(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='session_delivery_log'"
    ).fetchall()
    names = {r["name"] for r in rows}
    # Both declared indexes should exist
    assert "idx_sdl_session_gene" in names
    assert "idx_sdl_session_time" in names


# ── content_hash ──────────────────────────────────────────────────────

def test_content_hash_stable():
    h1 = session_delivery.content_hash("spliced content block")
    h2 = session_delivery.content_hash("spliced content block")
    assert h1 == h2


def test_content_hash_differs_on_different_input():
    h1 = session_delivery.content_hash("block one")
    h2 = session_delivery.content_hash("block two")
    assert h1 != h2


def test_content_hash_is_short_prefix():
    # Keep it 16-char so it fits in the header line budget
    h = session_delivery.content_hash("anything")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ── log_delivery ──────────────────────────────────────────────────────

def test_log_delivery_inserts_row(conn):
    did = session_delivery.log_delivery(
        conn,
        session_id="sess_A",
        gene_id="gene_001",
        content_hash="deadbeef12345678",
        mode="full",
        retrieval_id=42,
        ts=1000.0,
    )
    assert did >= 1
    row = conn.execute(
        "SELECT * FROM session_delivery_log WHERE delivery_id = ?", (did,)
    ).fetchone()
    assert row["session_id"] == "sess_A"
    assert row["gene_id"] == "gene_001"
    assert row["content_hash"] == "deadbeef12345678"
    assert row["mode"] == "full"
    assert row["retrieval_id"] == 42
    assert row["delivered_at"] == 1000.0


def test_log_delivery_defaults_timestamp_to_now(conn):
    import time as _t
    before = _t.time()
    did = session_delivery.log_delivery(
        conn, session_id="s", gene_id="g",
    )
    after = _t.time()
    row = conn.execute(
        "SELECT delivered_at FROM session_delivery_log WHERE delivery_id = ?",
        (did,),
    ).fetchone()
    assert before <= row["delivered_at"] <= after


def test_log_delivery_tolerates_optional_fields(conn):
    did = session_delivery.log_delivery(
        conn, session_id="s", gene_id="g", ts=100.0,
    )
    row = conn.execute(
        "SELECT content_hash, mode, retrieval_id FROM session_delivery_log "
        "WHERE delivery_id = ?", (did,),
    ).fetchone()
    assert row["content_hash"] is None
    assert row["mode"] is None
    assert row["retrieval_id"] is None


# ── already_delivered ─────────────────────────────────────────────────

def test_already_delivered_returns_none_when_absent(conn):
    assert session_delivery.already_delivered(
        conn, session_id="sess_A", gene_id="gene_001",
    ) is None


def test_already_delivered_returns_most_recent(conn):
    session_delivery.log_delivery(
        conn, session_id="s", gene_id="g",
        ts=100.0, mode="full", content_hash="h1",
    )
    session_delivery.log_delivery(
        conn, session_id="s", gene_id="g",
        ts=200.0, mode="compressed", content_hash="h2",
    )
    result = session_delivery.already_delivered(
        conn, session_id="s", gene_id="g",
    )
    assert result is not None
    delivered_at, mode, chash = result
    assert delivered_at == 200.0
    assert mode == "compressed"
    assert chash == "h2"


def test_already_delivered_respects_since_cutoff(conn):
    session_delivery.log_delivery(
        conn, session_id="s", gene_id="g", ts=100.0,
    )
    # Oldest delivery ignored by since=150 cutoff
    assert session_delivery.already_delivered(
        conn, session_id="s", gene_id="g", since=150.0,
    ) is None
    # A later delivery becomes visible
    session_delivery.log_delivery(
        conn, session_id="s", gene_id="g", ts=200.0,
    )
    result = session_delivery.already_delivered(
        conn, session_id="s", gene_id="g", since=150.0,
    )
    assert result is not None
    assert result[0] == 200.0


# ── count_deliveries_since ────────────────────────────────────────────

def test_count_deliveries_since_empty(conn):
    assert session_delivery.count_deliveries_since(
        conn, session_id="s", gene_id="g", since=0.0,
    ) == 0


def test_count_deliveries_since_multiple(conn):
    for t in (100.0, 150.0, 200.0, 250.0):
        session_delivery.log_delivery(
            conn, session_id="s", gene_id="g", ts=t,
        )
    # since=120 → 3 deliveries at 150, 200, 250
    assert session_delivery.count_deliveries_since(
        conn, session_id="s", gene_id="g", since=120.0,
    ) == 3


# ── session_manifest ──────────────────────────────────────────────────

def test_session_manifest_empty(conn):
    assert session_delivery.session_manifest(
        conn, session_id="nonexistent",
    ) == []


def test_session_manifest_ordered_descending(conn):
    for t, gid in [(100.0, "g1"), (200.0, "g2"), (300.0, "g3")]:
        session_delivery.log_delivery(
            conn, session_id="s", gene_id=gid, ts=t,
        )
    manifest = session_delivery.session_manifest(conn, session_id="s")
    # Most recent first
    assert [m["gene_id"] for m in manifest] == ["g3", "g2", "g1"]


def test_session_manifest_respects_limit(conn):
    for i in range(10):
        session_delivery.log_delivery(
            conn, session_id="s", gene_id=f"g{i}", ts=float(i),
        )
    manifest = session_delivery.session_manifest(conn, session_id="s", limit=3)
    assert len(manifest) == 3
    # Most recent 3 (g9, g8, g7)
    assert [m["gene_id"] for m in manifest] == ["g9", "g8", "g7"]


def test_session_manifest_includes_delivery_metadata(conn):
    session_delivery.log_delivery(
        conn, session_id="s", gene_id="g",
        ts=100.0, mode="full", content_hash="abc123",
        retrieval_id=7,
    )
    manifest = session_delivery.session_manifest(conn, session_id="s")
    assert len(manifest) == 1
    row = manifest[0]
    assert row["gene_id"] == "g"
    assert row["delivered_at"] == 100.0
    assert row["mode"] == "full"
    assert row["content_hash"] == "abc123"
    assert row["retrieval_id"] == 7


# ── session isolation ────────────────────────────────────────────────

def test_different_sessions_are_isolated(conn):
    session_delivery.log_delivery(
        conn, session_id="sess_A", gene_id="g1", ts=100.0,
    )
    session_delivery.log_delivery(
        conn, session_id="sess_B", gene_id="g1", ts=200.0,
    )
    # sess_A sees only its own delivery
    result_a = session_delivery.already_delivered(
        conn, session_id="sess_A", gene_id="g1",
    )
    assert result_a is not None
    assert result_a[0] == 100.0

    # sess_B sees only its own
    result_b = session_delivery.already_delivered(
        conn, session_id="sess_B", gene_id="g1",
    )
    assert result_b is not None
    assert result_b[0] == 200.0

    # Manifests are isolated too
    manifest_a = session_delivery.session_manifest(conn, session_id="sess_A")
    assert len(manifest_a) == 1
    assert manifest_a[0]["delivered_at"] == 100.0


# ── format_elision_stub (consumer-facing marker) ──────────────────────

def test_format_elision_stub_includes_gene_id():
    stub = session_delivery.format_elision_stub(
        gene_id="abc123456789xyz",
        delivered_at=100.0,
        now=160.0,
        queries_ago=3,
    )
    # Short gene_id present
    assert "gene=abc123456789" in stub
    # Bracketed like the Sprint 1 header
    assert stub.startswith("[")
    assert stub.endswith("]")


def test_format_elision_stub_shows_age():
    stub = session_delivery.format_elision_stub(
        gene_id="g",
        delivered_at=100.0,
        now=160.0,
        queries_ago=3,
    )
    # Should mention either queries-ago or seconds — human-readable is fine
    assert "3 queries ago" in stub or "60s" in stub or "60 s" in stub


def test_format_elision_stub_is_one_line():
    stub = session_delivery.format_elision_stub(
        gene_id="g",
        delivered_at=100.0,
        now=160.0,
        queries_ago=3,
    )
    assert "\n" not in stub


# ── Integration: _assemble uses session working-set register ──────────

from helix_context.config import HelixConfig, BudgetConfig, GenomeConfig, RibosomeConfig
from helix_context.context_manager import HelixContextManager
from tests.conftest import make_gene


def _make_manager(session_delivery_enabled: bool = True) -> HelixContextManager:
    cfg = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(
            max_genes_per_turn=4,
            splice_aggressiveness=0.5,
            legibility_enabled=True,
            session_delivery_enabled=session_delivery_enabled,
        ),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        synonym_map={},
    )
    return HelixContextManager(cfg)


def test_assemble_logs_fresh_delivery_when_session_on():
    mgr = _make_manager(session_delivery_enabled=True)
    try:
        g1 = make_gene(content="alpha content", gene_id="aaaa1111bbbb2222")
        mgr.genome.last_query_scores = {g1.gene_id: 5.0}
        mgr.genome.last_tier_contributions = {g1.gene_id: {"harmonic": 2.0}}
        mgr._assemble(
            query="q",
            candidates=[g1],
            spliced_map={g1.gene_id: "spliced-alpha"},
            session_id="sess_X",
        )
        # Delivery should be logged
        result = session_delivery.already_delivered(
            mgr.genome.conn, session_id="sess_X", gene_id=g1.gene_id,
        )
        assert result is not None
        # content_hash matches what we expect
        assert result[2] == session_delivery.content_hash("spliced-alpha")
    finally:
        mgr.close()


def test_assemble_elides_previously_delivered_gene():
    mgr = _make_manager(session_delivery_enabled=True)
    try:
        g1 = make_gene(content="alpha content", gene_id="aaaa1111bbbb2222")
        # Pre-seed a prior delivery
        session_delivery.log_delivery(
            mgr.genome.conn,
            session_id="sess_X", gene_id=g1.gene_id,
            content_hash="oldhash", mode="full",
            ts=100.0,
        )
        mgr.genome.last_query_scores = {g1.gene_id: 5.0}
        mgr.genome.last_tier_contributions = {g1.gene_id: {"harmonic": 2.0}}
        window = mgr._assemble(
            query="q",
            candidates=[g1],
            spliced_map={g1.gene_id: "spliced-alpha"},
            session_id="sess_X",
        )
        ec = window.expressed_context
        # Should contain the elision stub, NOT the spliced content
        assert "↻" in ec
        assert "delivered" in ec.lower()
        assert "spliced-alpha" not in ec
    finally:
        mgr.close()


def test_assemble_ignore_delivered_bypasses_elision():
    mgr = _make_manager(session_delivery_enabled=True)
    try:
        g1 = make_gene(content="alpha content", gene_id="aaaa1111bbbb2222")
        session_delivery.log_delivery(
            mgr.genome.conn,
            session_id="sess_X", gene_id=g1.gene_id,
            ts=100.0,
        )
        mgr.genome.last_query_scores = {g1.gene_id: 5.0}
        mgr.genome.last_tier_contributions = {g1.gene_id: {"harmonic": 2.0}}
        window = mgr._assemble(
            query="q",
            candidates=[g1],
            spliced_map={g1.gene_id: "spliced-alpha"},
            session_id="sess_X",
            ignore_delivered=True,
        )
        ec = window.expressed_context
        # Should contain the content (NOT a stub) when ignore_delivered=True
        assert "spliced-alpha" in ec
        assert "↻" not in ec
    finally:
        mgr.close()


def test_assemble_flag_off_never_touches_log():
    mgr = _make_manager(session_delivery_enabled=False)
    try:
        g1 = make_gene(content="alpha content", gene_id="aaaa1111bbbb2222")
        mgr.genome.last_query_scores = {g1.gene_id: 5.0}
        mgr.genome.last_tier_contributions = {g1.gene_id: {"harmonic": 2.0}}
        mgr._assemble(
            query="q",
            candidates=[g1],
            spliced_map={g1.gene_id: "spliced-alpha"},
            session_id="sess_X",
        )
        # No delivery should have been logged
        result = session_delivery.already_delivered(
            mgr.genome.conn, session_id="sess_X", gene_id=g1.gene_id,
        )
        assert result is None
    finally:
        mgr.close()


def test_assemble_no_session_id_skips_register():
    """When session_id is None, no lookup and no log — even if flag on."""
    mgr = _make_manager(session_delivery_enabled=True)
    try:
        g1 = make_gene(content="alpha", gene_id="aaaa1111bbbb2222")
        mgr.genome.last_query_scores = {g1.gene_id: 5.0}
        mgr.genome.last_tier_contributions = {g1.gene_id: {"harmonic": 2.0}}
        mgr._assemble(
            query="q",
            candidates=[g1],
            spliced_map={g1.gene_id: "spliced-alpha"},
            session_id=None,
        )
        # No delivery was logged for any session
        row = mgr.genome.conn.execute(
            "SELECT COUNT(*) AS n FROM session_delivery_log"
        ).fetchone()
        assert row["n"] == 0
    finally:
        mgr.close()


def test_assemble_mixed_elision_and_fresh():
    """Some candidates already delivered, some fresh — response has mix."""
    mgr = _make_manager(session_delivery_enabled=True)
    try:
        g_old = make_gene(content="old content", gene_id="aaaa0000bbbb0000")
        g_new = make_gene(content="new content", gene_id="cccc1111dddd1111")
        # Pre-deliver only g_old
        session_delivery.log_delivery(
            mgr.genome.conn,
            session_id="sess_X", gene_id=g_old.gene_id,
            content_hash="oldhash", mode="full",
            ts=100.0,
        )
        mgr.genome.last_query_scores = {g_old.gene_id: 5.0, g_new.gene_id: 3.0}
        mgr.genome.last_tier_contributions = {
            g_old.gene_id: {"harmonic": 2.0},
            g_new.gene_id: {"lex_anchor": 1.5},
        }
        window = mgr._assemble(
            query="q",
            candidates=[g_old, g_new],
            spliced_map={
                g_old.gene_id: "spliced-old",
                g_new.gene_id: "spliced-new",
            },
            session_id="sess_X",
        )
        ec = window.expressed_context
        # Old gene → stub (no content, has ↻)
        assert "spliced-old" not in ec
        assert "↻" in ec
        # New gene → fresh content (has content, has Sprint 1 header)
        assert "spliced-new" in ec
        assert "[gene=cccc1111dddd" in ec
        # Both listed as expressed
        assert g_old.gene_id in window.expressed_gene_ids
        assert g_new.gene_id in window.expressed_gene_ids
    finally:
        mgr.close()


def test_assemble_sessions_dont_cross_contaminate():
    """A delivery in sess_A should not elide for sess_B."""
    mgr = _make_manager(session_delivery_enabled=True)
    try:
        g1 = make_gene(content="shared gene", gene_id="ffffffffffffffff")
        session_delivery.log_delivery(
            mgr.genome.conn,
            session_id="sess_A", gene_id=g1.gene_id,
            ts=100.0,
        )
        mgr.genome.last_query_scores = {g1.gene_id: 5.0}
        mgr.genome.last_tier_contributions = {g1.gene_id: {"harmonic": 2.0}}
        # Query from sess_B — should NOT elide (different session)
        window = mgr._assemble(
            query="q",
            candidates=[g1],
            spliced_map={g1.gene_id: "spliced-shared"},
            session_id="sess_B",
        )
        ec = window.expressed_context
        assert "spliced-shared" in ec
        assert "↻" not in ec
    finally:
        mgr.close()


# ── HTTP endpoint: /session/{id}/manifest ─────────────────────────────

from fastapi.testclient import TestClient
from helix_context.config import ServerConfig
from helix_context.server import create_app


@pytest.fixture
def http_client():
    """TestClient against an app with session_delivery enabled."""
    config = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(
            max_genes_per_turn=4,
            session_delivery_enabled=True,
        ),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        server=ServerConfig(upstream="http://localhost:11434"),
    )
    app = create_app(config)
    # Mock ribosome backend so no Ollama is required
    app.state.helix.ribosome.backend = _MockBackend()
    yield TestClient(app)


class _MockBackend:
    """Minimal mock — returns '{}' for everything. Ribosome stays quiet."""
    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        return "{}"


def test_session_manifest_empty_session_returns_empty_list(http_client):
    resp = http_client.get("/session/nonexistent_session/manifest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "nonexistent_session"
    assert data["deliveries"] == []
    assert data["count"] == 0


def test_session_manifest_returns_logged_deliveries(http_client):
    app = http_client.app
    conn = app.state.helix.genome.conn
    # Seed some deliveries directly
    session_delivery.log_delivery(
        conn, session_id="sess_X", gene_id="g1",
        content_hash="h1", mode="full", ts=100.0,
    )
    session_delivery.log_delivery(
        conn, session_id="sess_X", gene_id="g2",
        content_hash="h2", mode="full", ts=200.0,
    )
    session_delivery.log_delivery(
        conn, session_id="sess_OTHER", gene_id="g3",
        content_hash="h3", mode="full", ts=300.0,
    )

    resp = http_client.get("/session/sess_X/manifest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess_X"
    assert data["count"] == 2
    # Most recent first
    gene_ids = [d["gene_id"] for d in data["deliveries"]]
    assert gene_ids == ["g2", "g1"]


def test_session_manifest_respects_limit_param(http_client):
    app = http_client.app
    conn = app.state.helix.genome.conn
    for i in range(5):
        session_delivery.log_delivery(
            conn, session_id="sess_X", gene_id=f"g{i}", ts=float(i),
        )
    resp = http_client.get("/session/sess_X/manifest?limit=2")
    assert resp.status_code == 200
    assert resp.json()["count"] == 2

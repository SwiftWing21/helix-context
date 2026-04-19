"""Smoke tests for main.db schema (Task 1 of genome sharding).

Verifies:
    - init_main_db creates all expected tables + indexes
    - register_shard upserts cleanly
    - upsert_fingerprint writes and replaces
    - list_shards filters by category
    - Category validation rejects unknown categories
    - Re-running init_main_db is idempotent
    - Default 'local' org seeded
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from helix_context.shard_schema import (
    SHARD_CATEGORIES,
    init_main_db,
    list_shards,
    open_main_db,
    query_claims,
    register_shard,
    upsert_claim,
    upsert_claim_edge,
    upsert_fingerprint,
    upsert_source_index,
)


@pytest.fixture
def main_db():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "main.db")
        conn = open_main_db(path)
        init_main_db(conn)
        yield conn
        conn.close()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def test_init_creates_all_tables(main_db):
    tables = _table_names(main_db)
    assert "shards" in tables
    assert "fingerprint_index" in tables
    assert "source_index" in tables
    assert "claims" in tables
    assert "claim_edges" in tables
    assert "orgs" in tables
    assert "parties" in tables
    assert "participants" in tables
    assert "agents" in tables


def test_init_seeds_local_org(main_db):
    row = main_db.execute(
        "SELECT org_id, display_name FROM orgs WHERE org_id='local'"
    ).fetchone()
    assert row is not None
    assert row["org_id"] == "local"


def test_init_is_idempotent(main_db):
    # Re-run init on the same connection
    init_main_db(main_db)
    init_main_db(main_db)
    # Still exactly one 'local' org
    count = main_db.execute(
        "SELECT COUNT(*) c FROM orgs WHERE org_id='local'"
    ).fetchone()["c"]
    assert count == 1


def test_register_shard_inserts_row(main_db):
    register_shard(
        main_db,
        shard_name="reference_third_party",
        category="reference",
        path="/tmp/reference/third_party.db",
        gene_count=1000,
        byte_size=50_000,
    )
    rows = list_shards(main_db)
    assert len(rows) == 1
    assert rows[0]["shard_name"] == "reference_third_party"
    assert rows[0]["category"] == "reference"
    assert rows[0]["gene_count"] == 1000


def test_register_shard_upserts_on_conflict(main_db):
    register_shard(main_db, "s1", "reference", "/a.db", gene_count=100)
    register_shard(main_db, "s1", "reference", "/a.db", gene_count=500)
    rows = list_shards(main_db)
    assert len(rows) == 1
    assert rows[0]["gene_count"] == 500


def test_register_shard_rejects_unknown_category(main_db):
    with pytest.raises(ValueError, match="unknown category"):
        register_shard(main_db, "s1", "nonsense", "/a.db")


def test_list_shards_filters_by_category(main_db):
    register_shard(main_db, "s_ref", "reference", "/r.db")
    register_shard(main_db, "s_agent", "agent", "/a.db")
    register_shard(main_db, "s_party", "participant", "/p.db")

    refs = list_shards(main_db, category="reference")
    assert len(refs) == 1
    assert refs[0]["shard_name"] == "s_ref"

    all_shards = list_shards(main_db)
    assert len(all_shards) == 3


def test_upsert_fingerprint_writes_and_replaces(main_db):
    register_shard(main_db, "s_ref", "reference", "/r.db")

    upsert_fingerprint(
        main_db,
        gene_id="g1",
        shard_name="s_ref",
        source_id="/docs/intro.md",
        domains_json='["docs"]',
        entities_json='["helix"]',
        key_values_json='["chunk_count=1"]',
        is_parent=False,
    )
    row = main_db.execute(
        "SELECT * FROM fingerprint_index WHERE gene_id='g1'"
    ).fetchone()
    assert row is not None
    assert row["shard_name"] == "s_ref"
    assert row["is_parent"] == 0

    # Replace
    upsert_fingerprint(
        main_db,
        gene_id="g1",
        shard_name="s_ref",
        source_id="/docs/intro.md",
        domains_json='["docs", "design"]',
        entities_json='["helix"]',
        key_values_json='["chunk_count=3", "is_parent=true"]',
        is_parent=True,
    )
    row = main_db.execute(
        "SELECT * FROM fingerprint_index WHERE gene_id='g1'"
    ).fetchone()
    assert row["is_parent"] == 1
    assert '"design"' in row["domains"]


def test_upsert_source_index_writes_and_replaces(main_db):
    register_shard(main_db, "s_ref", "reference", "/r.db")

    upsert_source_index(
        main_db,
        gene_id="g1",
        shard_name="s_ref",
        source_id="/docs/intro.md",
        repo_root="/repo",
        source_kind="doc",
        observed_at=100.0,
        mtime=90.0,
        content_hash="abc123",
        volatility_class="stable",
        authority_class="primary",
        support_span="1:20",
        last_verified_at=101.0,
    )
    row = main_db.execute(
        "SELECT * FROM source_index WHERE gene_id='g1'"
    ).fetchone()
    assert row is not None
    assert row["source_kind"] == "doc"
    assert row["volatility_class"] == "stable"
    assert row["repo_root"] == "/repo"

    upsert_source_index(
        main_db,
        gene_id="g1",
        shard_name="s_ref",
        source_id="/docs/intro.md",
        source_kind="config",
        volatility_class="hot",
        authority_class="derived",
    )
    row = main_db.execute(
        "SELECT * FROM source_index WHERE gene_id='g1'"
    ).fetchone()
    assert row["source_kind"] == "config"
    assert row["volatility_class"] == "hot"
    assert row["authority_class"] == "derived"


def test_shard_categories_constant():
    assert "participant" in SHARD_CATEGORIES
    assert "agent" in SHARD_CATEGORIES
    assert "reference" in SHARD_CATEGORIES
    assert "org" in SHARD_CATEGORIES
    assert "cold" in SHARD_CATEGORIES


# ── Claims layer ────────────────────────────────────────────────────


def _register_shard(main_db, name="s_ref"):
    register_shard(main_db, name, "reference", f"/tmp/{name}.db")


def test_upsert_claim_writes_row(main_db):
    _register_shard(main_db)
    upsert_claim(
        main_db,
        claim_id="c1",
        gene_id="g1",
        shard_name="s_ref",
        claim_type="path_value",
        claim_text="genome.db lives at genomes/main/genome.db",
        entity_key="genomes/main/genome.db",
        extraction_kind="literal",
        specificity=0.9,
        confidence=0.8,
        observed_at=1_776_000_000.0,
    )
    row = main_db.execute("SELECT * FROM claims WHERE claim_id='c1'").fetchone()
    assert row["gene_id"] == "g1"
    assert row["claim_type"] == "path_value"
    assert row["entity_key"] == "genomes/main/genome.db"
    assert row["extraction_kind"] == "literal"
    assert row["specificity"] == 0.9
    assert row["confidence"] == 0.8


def test_upsert_claim_idempotent_on_claim_id(main_db):
    _register_shard(main_db)
    for text in ("first", "second"):
        upsert_claim(
            main_db,
            claim_id="c1",
            gene_id="g1",
            shard_name="s_ref",
            claim_type="config_value",
            claim_text=text,
        )
    rows = main_db.execute("SELECT claim_text FROM claims").fetchall()
    assert len(rows) == 1
    assert rows[0]["claim_text"] == "second"


def test_upsert_claim_rejects_unknown_type(main_db):
    _register_shard(main_db)
    import pytest
    with pytest.raises(ValueError, match="unknown claim_type"):
        upsert_claim(
            main_db, claim_id="c1", gene_id="g1", shard_name="s_ref",
            claim_type="not_a_type", claim_text="...",
        )
    with pytest.raises(ValueError, match="unknown extraction_kind"):
        upsert_claim(
            main_db, claim_id="c1", gene_id="g1", shard_name="s_ref",
            claim_type="path_value", claim_text="...",
            extraction_kind="bogus",
        )


def test_upsert_claim_edge_writes_and_replaces(main_db):
    _register_shard(main_db)
    # Seed two claims
    for cid in ("c1", "c2"):
        upsert_claim(
            main_db, claim_id=cid, gene_id=cid, shard_name="s_ref",
            claim_type="path_value", claim_text=cid,
        )
    upsert_claim_edge(main_db, "c1", "c2", "contradicts", weight=0.8)
    row = main_db.execute(
        "SELECT * FROM claim_edges WHERE src_claim_id='c1' AND dst_claim_id='c2'"
    ).fetchone()
    assert row["edge_type"] == "contradicts"
    assert row["weight"] == 0.8

    # Upsert on same (src,dst,type) replaces weight
    upsert_claim_edge(main_db, "c1", "c2", "contradicts", weight=0.5)
    rows = main_db.execute(
        "SELECT weight FROM claim_edges WHERE src_claim_id='c1'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["weight"] == 0.5


def test_upsert_claim_edge_rejects_unknown_type(main_db):
    import pytest
    with pytest.raises(ValueError, match="unknown edge_type"):
        upsert_claim_edge(main_db, "c1", "c2", "not_a_type")


def test_query_claims_filters(main_db):
    _register_shard(main_db, "s_ref")
    _register_shard(main_db, "s_org")
    upsert_claim(
        main_db, claim_id="c1", gene_id="g1", shard_name="s_ref",
        claim_type="path_value", claim_text="p1",
        entity_key="/etc/foo", observed_at=100.0,
    )
    upsert_claim(
        main_db, claim_id="c2", gene_id="g2", shard_name="s_org",
        claim_type="config_value", claim_text="p2",
        entity_key="/etc/foo", observed_at=200.0,
    )
    upsert_claim(
        main_db, claim_id="c3", gene_id="g3", shard_name="s_ref",
        claim_type="path_value", claim_text="p3",
        entity_key="/etc/bar", observed_at=150.0,
    )

    # Entity-key filter: 2 rows, newest first
    rows = query_claims(main_db, entity_key="/etc/foo")
    assert [r["claim_id"] for r in rows] == ["c2", "c1"]

    # Type filter
    rows = query_claims(main_db, claim_type="path_value")
    assert {r["claim_id"] for r in rows} == {"c1", "c3"}

    # Shard filter
    rows = query_claims(main_db, shard_name="s_ref")
    assert len(rows) == 2

    # Combined filter (entity + type)
    rows = query_claims(main_db, entity_key="/etc/foo", claim_type="path_value")
    assert [r["claim_id"] for r in rows] == ["c1"]

    # No filter = all rows, newest first
    rows = query_claims(main_db)
    assert len(rows) == 3
    assert rows[0]["claim_id"] == "c2"  # observed_at=200 is newest


def test_query_claims_limit_bounds_result(main_db):
    _register_shard(main_db)
    for i in range(5):
        upsert_claim(
            main_db, claim_id=f"c{i}", gene_id=f"g{i}", shard_name="s_ref",
            claim_type="path_value", claim_text=f"p{i}",
        )
    assert len(query_claims(main_db, limit=3)) == 3
    assert len(query_claims(main_db, limit=100)) == 5


def test_init_creates_claim_indexes(main_db):
    names = {
        r["name"]
        for r in main_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_claims_gene" in names
    assert "idx_claims_entity" in names
    assert "idx_claims_type" in names
    assert "idx_claims_supersedes" in names
    assert "idx_claim_edges_dst" in names
    assert "idx_claim_edges_type" in names

"""Tests for Phase 2 claim extraction (helix_context/claims.py)."""

from __future__ import annotations

import pytest

from helix_context.claims import (
    claim_id_for,
    extract_entity_keys,
    extract_literal_claims,
    persist_claims,
)
from helix_context.schemas import Gene
from helix_context.shard_schema import (
    init_main_db,
    open_main_db,
    query_claims,
    register_shard,
)


@pytest.fixture
def main_db(tmp_path):
    db = open_main_db(tmp_path / "main.db")
    init_main_db(db)
    register_shard(db, "s_ref", "reference", "/tmp/s_ref.db")
    yield db
    db.close()


def _gene(
    gene_id: str = "g1",
    content: str = "",
    source_kind: str | None = None,
    source_id: str | None = None,
    key_values: list[str] | None = None,
    observed_at: float | None = None,
) -> Gene:
    return Gene(
        gene_id=gene_id,
        content=content,
        complement="",
        codons=[],
        source_kind=source_kind,
        source_id=source_id,
        key_values=key_values or [],
        observed_at=observed_at,
    )


# ── Entity-key extraction ────────────────────────────────────────────


def test_entity_keys_extracts_file_path():
    keys = extract_entity_keys("the live genome path is genomes/main/genome.db today")
    assert any("genomes/main/genome.db" in k for k in keys)


def test_entity_keys_extracts_port():
    keys = extract_entity_keys("helix listens on port 11437")
    assert "port:11437" in keys


def test_entity_keys_extracts_symbol():
    keys = extract_entity_keys("HELIX_USE_SHARDS is off by default")
    assert "HELIX_USE_SHARDS" in keys


def test_entity_keys_extracts_kv_lhs():
    keys = extract_entity_keys("MODEL = qwen3:4b")
    assert "MODEL" in keys


def test_entity_keys_dedups():
    keys = extract_entity_keys("port 11437 port 11437")
    assert keys.count("port:11437") == 1


# ── Claim ID determinism ─────────────────────────────────────────────


def test_claim_id_deterministic():
    a = claim_id_for("g1", "config_value", "MODEL = qwen3", "MODEL")
    b = claim_id_for("g1", "config_value", "MODEL = qwen3", "MODEL")
    assert a == b


def test_claim_id_changes_with_content():
    a = claim_id_for("g1", "config_value", "MODEL = qwen3", "MODEL")
    b = claim_id_for("g1", "config_value", "MODEL = qwen4", "MODEL")
    assert a != b


# ── key_values fallback ──────────────────────────────────────────────


def test_key_values_always_extracted():
    """Even with source_kind=None, key_values become config_value claims."""
    gene = _gene(key_values=["port=11437", "model=qwen3:4b"])
    claims = extract_literal_claims(gene, shard_name="s_ref")
    assert len(claims) == 2
    texts = {c.claim_text for c in claims}
    assert "port = 11437" in texts
    assert "model = qwen3:4b" in texts
    for c in claims:
        assert c.claim_type == "config_value"
        assert c.extraction_kind == "literal"
        assert c.shard_name == "s_ref"


def test_key_values_with_empty_gene_is_empty():
    assert extract_literal_claims(_gene()) == []


# ── Code extractor ───────────────────────────────────────────────────


def test_code_extractor_def_and_class():
    content = (
        "def build_context_packet(query, task_type='explain'):\n"
        "    return None\n"
        "\n"
        "class ContextPacket:\n"
        "    pass\n"
        "\n"
        "async def refresh_targets():\n"
        "    pass\n"
    )
    gene = _gene(content=content, source_kind="code")
    claims = extract_literal_claims(gene)
    names = {c.entity_key for c in claims if c.claim_type == "api_contract"}
    assert "build_context_packet" in names
    assert "ContextPacket" in names
    assert "refresh_targets" in names


def test_code_extractor_const():
    content = "DEFAULT_PORT = 11437\nMODEL_NAME = 'qwen3'\n"
    gene = _gene(content=content, source_kind="code")
    claims = extract_literal_claims(gene)
    const_texts = {
        c.claim_text for c in claims if c.claim_type == "config_value"
    }
    assert "DEFAULT_PORT = 11437" in const_texts
    assert "MODEL_NAME = 'qwen3'" in const_texts


# ── Config extractor ─────────────────────────────────────────────────


def test_config_extractor_toml_style():
    content = (
        "port = 11437\n"
        "model = \"qwen3:4b\"\n"
        "  nested = skipme\n"        # indented = sub-key, skip
        "# commented_out = 99\n"
        "[section]\n"                # section header, not a kv
    )
    gene = _gene(content=content, source_kind="config")
    claims = extract_literal_claims(gene)
    keys = {c.entity_key for c in claims}
    assert "port" in keys
    assert "model" in keys
    assert "nested" not in keys
    assert "commented_out" not in keys


# ── Doc extractor ────────────────────────────────────────────────────


def test_doc_extractor_headers_and_ports():
    content = (
        "# Helix Design\n"
        "## Retrieval Pipeline\n"
        "### Deep section\n"                # skipped (> H2)
        "\n"
        "The server listens on port 11437.\n"
    )
    gene = _gene(content=content, source_kind="doc")
    claims = extract_literal_claims(gene)
    titles = {c.claim_text for c in claims if c.claim_type == "operational_state"}
    assert "section: Helix Design" in titles
    assert "section: Retrieval Pipeline" in titles
    assert "section: Deep section" not in titles
    ports = {c.entity_key for c in claims if c.claim_type == "path_value"}
    assert "port:11437" in ports


# ── Benchmark extractor ──────────────────────────────────────────────


def test_benchmark_extractor_metric_value_pairs():
    content = (
        "avg_tier: 1.04\n"
        "avg_tokens: 3773.6\n"
        "miss_rate: 27.7%\n"
        "comment: this is prose without digits\n"  # no digit → skipped
    )
    gene = _gene(content=content, source_kind="benchmark")
    claims = extract_literal_claims(gene)
    keys = {c.entity_key for c in claims if c.claim_type == "benchmark_result"}
    assert "avg_tier" in keys
    assert "avg_tokens" in keys
    assert "miss_rate" in keys
    assert "comment" not in keys


# ── Source-kind=None fallback ────────────────────────────────────────


def test_unknown_source_kind_still_harvests_key_values():
    gene = _gene(
        source_kind="unknown_kind",
        key_values=["port=11437"],
        content="ignored because unknown kind has no extractor",
    )
    claims = extract_literal_claims(gene)
    assert len(claims) == 1
    assert claims[0].entity_key == "port"


# ── Dedup ────────────────────────────────────────────────────────────


def test_extract_dedups_duplicate_claims():
    """key_values and config extractor both emit port=11437 → only 1 claim."""
    gene = _gene(
        content="port = 11437\n",
        source_kind="config",
        key_values=["port=11437"],
    )
    claims = extract_literal_claims(gene)
    assert len([c for c in claims if c.entity_key == "port"]) == 1


# ── Persist round-trip ───────────────────────────────────────────────


def test_ingest_hook_auto_populates_claims(tmp_path):
    """Full round-trip: Genome wired with main_conn → upsert_gene triggers
    claim extraction → query_claims returns claims without reading gene content.

    First-milestone verification for Phase 2: "Helix can answer structured
    fact questions without reopening bulk content."
    """
    from helix_context.genome import Genome
    main_db_path = tmp_path / "main.db"
    main_db = open_main_db(main_db_path)
    init_main_db(main_db)
    register_shard(main_db, "primary", "reference", str(tmp_path / "primary.db"))

    genome_path = tmp_path / "primary.db"
    genome = Genome(
        str(genome_path),
        main_conn=main_db,
        shard_name="primary",
    )
    try:
        gene = _gene(
            gene_id="g_fleet_port",
            content="port = 11437\nmodel = \"qwen3:4b\"\n",
            source_kind="config",
            source_id="fleet/fleet.toml",
            observed_at=1_776_000_000.0,
        )
        genome.upsert_gene(gene, apply_gate=False)

        # Prove the first-milestone behavior: we can retrieve the fact
        # without re-opening the shard that holds the gene content.
        rows = query_claims(main_db, gene_id="g_fleet_port")
        assert len(rows) >= 2
        texts = {r["claim_text"] for r in rows}
        assert "port = 11437" in texts
        assert 'model = qwen3:4b' in texts

        port_rows = query_claims(main_db, entity_key="port")
        assert len(port_rows) == 1
        assert port_rows[0]["shard_name"] == "primary"
    finally:
        genome.close()
        main_db.close()


def test_ingest_without_main_conn_is_noop(tmp_path):
    """Default Genome (no main_conn) ingests fine without writing claims."""
    from helix_context.genome import Genome
    genome = Genome(str(tmp_path / "primary.db"))
    try:
        gene = _gene(
            gene_id="g_nohook",
            content="port = 11437\n",
            source_kind="config",
        )
        # Should not raise even though no main.db is wired.
        genome.upsert_gene(gene, apply_gate=False)
    finally:
        genome.close()


def test_persist_and_query(main_db):
    gene = _gene(
        gene_id="g42",
        source_kind="config",
        content="port = 11437\nmodel = \"qwen3\"\n",
        observed_at=1_776_000_000.0,
    )
    claims = extract_literal_claims(gene, shard_name="s_ref")
    n = persist_claims(main_db, claims)
    assert n == len(claims) >= 2

    # Query by gene_id should return all persisted claims
    rows = query_claims(main_db, gene_id="g42")
    assert len(rows) == n

    # Query by entity_key should return the port claim
    port_rows = query_claims(main_db, entity_key="port")
    assert len(port_rows) == 1
    assert port_rows[0]["claim_text"] == "port = 11437"
    assert port_rows[0]["observed_at"] == 1_776_000_000.0

    # Re-persist same claims → idempotent (no duplicate rows)
    persist_claims(main_db, claims)
    rows_again = query_claims(main_db, gene_id="g42")
    assert len(rows_again) == n

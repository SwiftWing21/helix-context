"""Freshness-labeled packet builder tests."""

from __future__ import annotations

from helix_context.context_packet import build_context_packet, get_refresh_targets
from helix_context.genome import Genome
from helix_context.shard_schema import init_main_db, open_main_db, register_shard, upsert_source_index

from tests.conftest import make_gene


def test_build_context_packet_marks_recent_stable_doc_verified():
    now_ts = 10_000.0
    genome = Genome(":memory:")
    try:
        gene = make_gene("Helix design notes for the agent index", domains=["helix", "design"])
        gene.source_id = "/repo/docs/design.md"
        gene.source_kind = "doc"
        gene.volatility_class = "stable"
        gene.authority_class = "primary"
        gene.last_verified_at = now_ts - 120.0
        genome.upsert_gene(gene, apply_gate=False)

        packet = build_context_packet(
            "helix design",
            task_type="explain",
            genome=genome,
            now_ts=now_ts,
        )

        assert len(packet.verified) == 1
        assert packet.verified[0].status == "verified"
        assert packet.verified[0].source_id == "/repo/docs/design.md"
        assert packet.refresh_targets == []
    finally:
        genome.close()


def test_build_context_packet_marks_hot_old_config_for_refresh():
    now_ts = 20_000.0
    genome = Genome(":memory:")
    try:
        gene = make_gene("Auth config sets jwt ttl to fifteen minutes", domains=["auth", "config"])
        gene.source_id = "/repo/config/auth.toml"
        gene.source_kind = "config"
        gene.volatility_class = "hot"
        gene.authority_class = "primary"
        gene.last_verified_at = now_ts - 4_000.0
        genome.upsert_gene(gene, apply_gate=False)

        packet = build_context_packet(
            "auth config",
            task_type="edit",
            genome=genome,
            now_ts=now_ts,
        )

        assert packet.verified == []
        assert len(packet.stale_risk) == 1
        assert packet.stale_risk[0].status == "needs_refresh"
        assert packet.refresh_targets[0].source_id == "/repo/config/auth.toml"
    finally:
        genome.close()


def test_source_index_metadata_overrides_gene_metadata():
    now_ts = 30_000.0
    genome = Genome(":memory:")
    main_conn = open_main_db(":memory:")
    init_main_db(main_conn)
    register_shard(main_conn, "main_ref", "reference", ":memory:")

    try:
        gene = make_gene("JWT configuration lives here", domains=["jwt", "config"])
        gene.source_id = "/repo/docs/auth.md"
        gene.source_kind = "doc"
        gene.volatility_class = "stable"
        gene.authority_class = "primary"
        gene.last_verified_at = now_ts - 60.0
        gene_id = genome.upsert_gene(gene, apply_gate=False)

        upsert_source_index(
            main_conn,
            gene_id=gene_id,
            shard_name="main_ref",
            source_id="/repo/config/auth.toml",
            source_kind="config",
            volatility_class="hot",
            authority_class="derived",
            last_verified_at=now_ts - 4_000.0,
            invalidated_at=now_ts - 10.0,
        )

        packet = build_context_packet(
            "jwt config",
            task_type="edit",
            genome=genome,
            main_conn=main_conn,
            now_ts=now_ts,
        )

        assert len(packet.stale_risk) == 1
        item = packet.stale_risk[0]
        assert item.source_id == "/repo/config/auth.toml"
        assert item.source_kind == "config"
        assert item.authority_class == "derived"
        assert item.status == "needs_refresh"
    finally:
        main_conn.close()
        genome.close()


def test_get_refresh_targets_returns_only_refreshable_sources():
    now_ts = 40_000.0
    genome = Genome(":memory:")
    try:
        fresh = make_gene("Stable architecture notes", domains=["architecture"])
        fresh.source_id = "/repo/docs/architecture.md"
        fresh.source_kind = "doc"
        fresh.volatility_class = "stable"
        fresh.last_verified_at = now_ts - 60.0
        genome.upsert_gene(fresh, apply_gate=False)

        stale = make_gene("Runtime port is 11437", domains=["runtime", "port"])
        stale.source_id = "/repo/config/runtime.toml"
        stale.source_kind = "config"
        stale.volatility_class = "hot"
        stale.last_verified_at = now_ts - 5_000.0
        genome.upsert_gene(stale, apply_gate=False)

        targets = get_refresh_targets(
            "runtime port",
            task_type="ops",
            genome=genome,
            now_ts=now_ts,
        )

        assert len(targets) == 1
        assert targets[0].source_id == "/repo/config/runtime.toml"
    finally:
        genome.close()

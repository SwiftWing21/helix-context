"""Tests for the claims-graph DAG walker."""

from __future__ import annotations

import pytest

from helix_context.claims_graph import (
    contradiction_clusters,
    latest_in_chain,
    resolve,
    resolve_from_packet,
    supersedes_chain,
    topologically_sorted,
)
from helix_context.shard_schema import (
    init_main_db,
    open_main_db,
    register_shard,
    upsert_claim,
    upsert_claim_edge,
)


@pytest.fixture
def main_db(tmp_path):
    db = open_main_db(tmp_path / "main.db")
    init_main_db(db)
    register_shard(db, "s", "reference", "/tmp/s.db")
    yield db
    db.close()


def _seed_claim(db, claim_id: str, *, claim_type: str = "path_value",
                gene_id: str | None = None, observed_at: float = 100.0,
                supersedes: str | None = None,
                claim_text: str | None = None):
    upsert_claim(
        db,
        claim_id=claim_id,
        gene_id=gene_id or f"g_{claim_id}",
        shard_name="s",
        claim_type=claim_type,
        claim_text=claim_text or f"text-{claim_id}",
        observed_at=observed_at,
        supersedes_claim_id=supersedes,
    )


# ── supersedes_chain + latest_in_chain ───────────────────────────────


def test_supersedes_chain_walks_forward(main_db):
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2", supersedes="c1")
    _seed_claim(main_db, "c3", supersedes="c2")

    chain = supersedes_chain(main_db, "c1")
    assert chain == ["c1", "c2", "c3"]


def test_supersedes_chain_terminates_at_head(main_db):
    _seed_claim(main_db, "c1")
    assert supersedes_chain(main_db, "c1") == ["c1"]


def test_supersedes_chain_breaks_on_cycle(main_db):
    # Shouldn't happen, but defensive.
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2", supersedes="c1")
    # Manually create a cycle (supersedes c1 → c2 → c3 → c1 would violate schema)
    # Instead we test max_depth kicks in by building a long chain.
    for i in range(50):
        _seed_claim(main_db, f"c{i+3}", supersedes=f"c{i+2}")
    chain = supersedes_chain(main_db, "c1", max_depth=5)
    assert len(chain) <= 6  # c1 + 5 steps


def test_latest_in_chain(main_db):
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2", supersedes="c1")
    _seed_claim(main_db, "c3", supersedes="c2")
    assert latest_in_chain(main_db, "c1") == "c3"
    assert latest_in_chain(main_db, "c2") == "c3"
    assert latest_in_chain(main_db, "c3") == "c3"


# ── contradiction_clusters ───────────────────────────────────────────


def test_clusters_singletons_for_non_contradicting(main_db):
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2")
    _seed_claim(main_db, "c3")
    clusters = contradiction_clusters(main_db, ["c1", "c2", "c3"])
    assert sorted(clusters, key=len) == [["c1"], ["c2"], ["c3"]]


def test_clusters_pair_contradicting_claims(main_db):
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2")
    upsert_claim_edge(main_db, "c1", "c2", "contradicts")

    clusters = contradiction_clusters(main_db, ["c1", "c2"])
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ["c1", "c2"]


def test_clusters_transitive_via_contradicts(main_db):
    """A contradicts B, B contradicts C → all three cluster."""
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2")
    _seed_claim(main_db, "c3")
    upsert_claim_edge(main_db, "c1", "c2", "contradicts")
    upsert_claim_edge(main_db, "c2", "c3", "contradicts")

    clusters = contradiction_clusters(main_db, ["c1", "c2", "c3"])
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ["c1", "c2", "c3"]


def test_clusters_duplicates_merge_like_contradicts(main_db):
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2")
    upsert_claim_edge(main_db, "c1", "c2", "duplicates")
    clusters = contradiction_clusters(main_db, ["c1", "c2"])
    assert len(clusters) == 1


def test_clusters_supports_edge_does_not_merge(main_db):
    """'supports' is affirmation, not conflict — shouldn't cluster."""
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2")
    upsert_claim_edge(main_db, "c1", "c2", "supports")
    clusters = contradiction_clusters(main_db, ["c1", "c2"])
    assert len(clusters) == 2


# ── topologically_sorted ─────────────────────────────────────────────


def test_topo_orders_predecessors_first(main_db):
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2", supersedes="c1")
    _seed_claim(main_db, "c3", supersedes="c2")

    order = topologically_sorted(main_db, ["c3", "c1", "c2"])
    assert order.index("c1") < order.index("c2") < order.index("c3")


def test_topo_handles_disconnected(main_db):
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2")
    _seed_claim(main_db, "c3", supersedes="c1")
    order = topologically_sorted(main_db, ["c1", "c2", "c3"])
    assert order.index("c1") < order.index("c3")
    assert "c2" in order  # non-related claim still returned


def test_topo_empty_input_returns_empty(main_db):
    assert topologically_sorted(main_db, []) == []


# ── resolve ──────────────────────────────────────────────────────────


def test_resolve_latest_then_authority_picks_head(main_db):
    _seed_claim(main_db, "c1", observed_at=100.0)
    _seed_claim(main_db, "c2", supersedes="c1", observed_at=200.0)

    result = resolve(main_db, ["c1", "c2"], policy="latest_then_authority")
    accepted_ids = {c["claim_id"] for c in result["accepted"]}
    rejected_ids = {c["claim_id"] for c in result["rejected"]}
    assert "c2" in accepted_ids
    assert "c1" in rejected_ids
    # Rejection reason should name the superseder
    c1_reject = [r for r in result["rejected"] if r["claim_id"] == "c1"][0]
    assert "superseded_by" in c1_reject["rejected_reason"]


def test_resolve_keep_all_with_flags_preserves_all(main_db):
    _seed_claim(main_db, "c1")
    _seed_claim(main_db, "c2", supersedes="c1")
    upsert_claim_edge(main_db, "c1", "c2", "contradicts")

    result = resolve(main_db, ["c1", "c2"], policy="keep_all_with_flags")
    accepted_ids = {c["claim_id"] for c in result["accepted"]}
    assert accepted_ids == {"c1", "c2"}
    # c1 should have a superseded_by pointer
    c1 = [c for c in result["accepted"] if c["claim_id"] == "c1"][0]
    assert c1["superseded_by"] == "c2"


def test_resolve_contradiction_picks_one_winner(main_db):
    _seed_claim(main_db, "c1", observed_at=200.0)
    _seed_claim(main_db, "c2", observed_at=100.0)
    upsert_claim_edge(main_db, "c1", "c2", "contradicts")

    result = resolve(main_db, ["c1", "c2"])
    assert len(result["accepted"]) == 1
    assert len(result["rejected"]) == 1
    # Newer observed_at should win (c1)
    assert result["accepted"][0]["claim_id"] == "c1"
    assert "contradicts_winner" in result["rejected"][0]["rejected_reason"]


def test_resolve_empty_input(main_db):
    result = resolve(main_db, [])
    assert result == {"accepted": [], "rejected": [], "clusters": []}


# ── resolve_from_packet ──────────────────────────────────────────────


def test_resolve_from_packet_pulls_claims_via_gene_ids(main_db):
    _seed_claim(main_db, "c1", gene_id="g_foo")
    _seed_claim(main_db, "c2", gene_id="g_bar")
    packet = {
        "verified": [{"gene_id": "g_foo", "source_id": "/foo"}],
        "stale_risk": [{"gene_id": "g_bar", "source_id": "/bar"}],
        "contradictions": [],
        "refresh_targets": [],
    }
    result = resolve_from_packet(main_db, packet)
    ids = {c["claim_id"] for c in result["accepted"]}
    assert "c1" in ids
    assert "c2" in ids


def test_resolve_from_packet_empty_packet(main_db):
    result = resolve_from_packet(main_db, {"verified": [], "stale_risk": [],
                                           "contradictions": [],
                                           "refresh_targets": []})
    assert result["accepted"] == []

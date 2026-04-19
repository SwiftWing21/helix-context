"""Tests for the claim-edge detector (helix_context/claims_analyze.py)."""

from __future__ import annotations

import pytest

from helix_context.claims_analyze import (
    _NOISE_ENTITY_KEYS,
    _jaccard,
    _tokenize,
    detect_and_persist_edges,
    detect_edges_for_group,
)
from helix_context.shard_schema import (
    init_main_db,
    open_main_db,
    register_shard,
    upsert_claim,
)


@pytest.fixture
def main_db(tmp_path):
    db = open_main_db(tmp_path / "main.db")
    init_main_db(db)
    register_shard(db, "s", "reference", "/tmp/s.db")
    yield db
    db.close()


def _seed(db, claim_id: str, *, gene_id: str, entity_key: str,
          claim_text: str, observed_at: float | None = None,
          claim_type: str = "config_value"):
    upsert_claim(
        db,
        claim_id=claim_id,
        gene_id=gene_id,
        shard_name="s",
        claim_type=claim_type,
        claim_text=claim_text,
        entity_key=entity_key,
        observed_at=observed_at,
    )


# ── Token + Jaccard helpers ─────────────────────────────────────────


def test_tokenize_lowercases_and_drops_shortwords():
    assert _tokenize("Hello, a World!") == {"hello", "world"}


def test_tokenize_keeps_alphanumeric():
    assert _tokenize("port = 11437") == {"port", "11437"}


def test_jaccard_basic():
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


# ── detect_edges_for_group — contradicts ─────────────────────────────


def test_contradicts_fires_when_values_disagree():
    claims = [
        {"claim_id": "c1", "gene_id": "g1", "claim_type": "config_value",
         "entity_key": "port:11437", "claim_text": "port = 11437",
         "observed_at": 100.0},
        {"claim_id": "c2", "gene_id": "g2", "claim_type": "config_value",
         "entity_key": "port:11437", "claim_text": "launches on 8787 now",
         "observed_at": 100.0},
    ]
    edges = detect_edges_for_group(claims)
    types = [e[2] for e in edges]
    assert "contradicts" in types


def test_contradicts_emits_single_canonical_edge():
    claims = [
        {"claim_id": "c2", "gene_id": "g1",
         "claim_type": "config_value", "entity_key": "x",
         "claim_text": "apples oranges bananas", "observed_at": 1.0},
        {"claim_id": "c1", "gene_id": "g2",
         "claim_type": "config_value", "entity_key": "x",
         "claim_text": "cats dogs mice rabbits", "observed_at": 1.0},
    ]
    edges = detect_edges_for_group(claims)
    contradicts = [e for e in edges if e[2] == "contradicts"]
    assert len(contradicts) == 1
    # Canonical ordering: lower claim_id first
    assert contradicts[0][0] < contradicts[0][1]


# ── detect_edges_for_group — duplicates ─────────────────────────────


def test_duplicates_fires_when_text_nearly_identical():
    claims = [
        {"claim_id": "c1", "gene_id": "g1", "claim_type": "config_value",
         "entity_key": "port", "claim_text": "port configured to 11437",
         "observed_at": 100.0},
        {"claim_id": "c2", "gene_id": "g2", "claim_type": "config_value",
         "entity_key": "port", "claim_text": "port configured to 11437",
         "observed_at": 100.0},
    ]
    edges = detect_edges_for_group(claims)
    types = [e[2] for e in edges]
    assert "duplicates" in types


# ── detect_edges_for_group — supersedes ─────────────────────────────


def test_supersedes_fires_when_near_dup_and_observed_at_differs():
    claims = [
        {"claim_id": "c_old", "gene_id": "g1", "claim_type": "config_value",
         "entity_key": "p", "claim_text": "version release notes updated",
         "observed_at": 100.0},
        {"claim_id": "c_new", "gene_id": "g2", "claim_type": "config_value",
         "entity_key": "p", "claim_text": "version release notes updated",
         "observed_at": 200.0},
    ]
    edges = detect_edges_for_group(claims)
    supersedes = [e for e in edges if e[2] == "supersedes"]
    assert len(supersedes) == 1
    # Edge points from older → newer
    assert supersedes[0][0] == "c_old"
    assert supersedes[0][1] == "c_new"


def test_supersedes_not_fired_without_observed_at_difference():
    """Equal observed_at + high similarity → duplicates, not supersedes."""
    claims = [
        {"claim_id": "a", "gene_id": "g1", "claim_type": "config_value",
         "entity_key": "k", "claim_text": "the same text",
         "observed_at": 100.0},
        {"claim_id": "b", "gene_id": "g2", "claim_type": "config_value",
         "entity_key": "k", "claim_text": "the same text",
         "observed_at": 100.0},
    ]
    edges = detect_edges_for_group(claims)
    types = [e[2] for e in edges]
    assert "duplicates" in types
    assert "supersedes" not in types


# ── Skip rules ───────────────────────────────────────────────────────


def test_skip_same_gene_pairs():
    """Claims on the same gene can't contradict themselves."""
    claims = [
        {"claim_id": "c1", "gene_id": "g_SAME", "claim_type": "config_value",
         "entity_key": "x", "claim_text": "one claim text",
         "observed_at": 1.0},
        {"claim_id": "c2", "gene_id": "g_SAME", "claim_type": "config_value",
         "entity_key": "x", "claim_text": "totally different content here",
         "observed_at": 1.0},
    ]
    assert detect_edges_for_group(claims) == []


def test_silent_middle_no_edge():
    """Jaccard between thresholds (too similar to contradict, too
    different to dup) → no edge."""
    claims = [
        {"claim_id": "c1", "gene_id": "g1", "claim_type": "config_value",
         "entity_key": "x", "claim_text": "apple orange banana grape",
         "observed_at": 1.0},
        {"claim_id": "c2", "gene_id": "g2", "claim_type": "config_value",
         "entity_key": "x", "claim_text": "apple orange peach cherry",
         "observed_at": 1.0},
    ]
    edges = detect_edges_for_group(claims)
    # Jaccard = 2/6 ≈ 0.33 — below dup threshold (0.8) but above
    # contradict threshold (0.3). Should emit no edge.
    assert edges == []


# ── Noise-key filtering ─────────────────────────────────────────────


def test_noise_key_not_in_groups(main_db):
    """Entity keys in _NOISE_ENTITY_KEYS should be filtered out."""
    _seed(main_db, "c1", gene_id="g1", entity_key="error", claim_text="a")
    _seed(main_db, "c2", gene_id="g2", entity_key="error", claim_text="b")
    _seed(main_db, "c3", gene_id="g1", entity_key="port:11437",
          claim_text="port = 11437")
    _seed(main_db, "c4", gene_id="g2", entity_key="port:11437",
          claim_text="totally different about 8787")

    summary = detect_and_persist_edges(main_db)
    # "error" is noise — no edges. port:11437 isn't — should see contradicts.
    assert summary.get("contradicts", 0) >= 1
    rows = main_db.execute(
        "SELECT src_claim_id, dst_claim_id FROM claim_edges").fetchall()
    # No edge involving c1 or c2 (both are "error" key)
    for src, dst in rows:
        assert src not in ("c1", "c2")
        assert dst not in ("c1", "c2")


# ── Integration: detect_and_persist_edges ──────────────────────────


def test_persist_writes_to_claim_edges_table(main_db):
    _seed(main_db, "c1", gene_id="g1", entity_key="config_foo",
          claim_text="one two three four five")
    _seed(main_db, "c2", gene_id="g2", entity_key="config_foo",
          claim_text="one two three four five")
    summary = detect_and_persist_edges(main_db)
    rows = main_db.execute(
        "SELECT edge_type, weight FROM claim_edges").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "duplicates"
    assert summary["duplicates"] == 1


def test_idempotent_rerun(main_db):
    """Re-running detection after it fires should produce no new edges."""
    _seed(main_db, "c1", gene_id="g1", entity_key="foo_bar",
          claim_text="one two three four")
    _seed(main_db, "c2", gene_id="g2", entity_key="foo_bar",
          claim_text="one two three four")
    detect_and_persist_edges(main_db)
    n_first = main_db.execute(
        "SELECT COUNT(*) FROM claim_edges").fetchone()[0]
    detect_and_persist_edges(main_db)
    n_second = main_db.execute(
        "SELECT COUNT(*) FROM claim_edges").fetchone()[0]
    assert n_first == n_second


def test_entity_keys_filter_limits_scope(main_db):
    _seed(main_db, "c1", gene_id="g1", entity_key="group_a",
          claim_text="text one two three")
    _seed(main_db, "c2", gene_id="g2", entity_key="group_a",
          claim_text="text one two three")
    _seed(main_db, "c3", gene_id="g1", entity_key="group_b",
          claim_text="text one two three")
    _seed(main_db, "c4", gene_id="g2", entity_key="group_b",
          claim_text="text one two three")

    summary = detect_and_persist_edges(main_db, entity_keys=["group_a"])
    assert summary["duplicates"] == 1
    # Only group_a's pair (c1, c2) — not group_b's
    rows = main_db.execute("SELECT src_claim_id FROM claim_edges").fetchall()
    edge_claim_ids = {r[0] for r in rows}
    assert edge_claim_ids == {"c1"}


def test_max_group_size_skips_oversize(main_db):
    """Groups larger than max_group_size are skipped to avoid O(N²)."""
    # Seed 10 identical claims with the same entity_key
    for i in range(10):
        _seed(main_db, f"c{i}", gene_id=f"g{i}", entity_key="big_group",
              claim_text="identical content here")
    # With max_group_size=5, the group should be skipped entirely
    summary = detect_and_persist_edges(main_db, max_group_size=5)
    assert summary.get("duplicates", 0) == 0
    # With max_group_size=20, it runs — 10 choose 2 = 45 dup pairs
    main_db.execute("DELETE FROM claim_edges")
    main_db.commit()
    summary2 = detect_and_persist_edges(main_db, max_group_size=20)
    assert summary2.get("duplicates", 0) == 45

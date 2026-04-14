"""Tests for the three new retrieval dimensions wired into query_genes().

1. Harmonic co-activation boost (Tier 5)
2. Access-rate tiebreaker
3. Gene attribution party_id filter + bonus
"""

import time

import pytest

from helix_context.genome import Genome
from helix_context.schemas import Gene, ChromatinState, PromoterTags, EpigeneticMarkers
from helix_context.exceptions import PromoterMismatch
from helix_context.accel import parse_epigenetics

from tests.conftest import make_gene


# ── Helpers ────────────────────────────────────────────────────────────


def _insert_party(genome: Genome, party_id: str) -> None:
    """Insert a party row. Tables already exist from Genome.__init__."""
    genome.conn.execute(
        "INSERT OR IGNORE INTO parties "
        "(party_id, display_name, created_at) "
        "VALUES (?, ?, ?)",
        (party_id, party_id, time.time()),
    )
    genome.conn.commit()


def _attribute_gene(genome: Genome, gene_id: str, party_id: str) -> None:
    genome.conn.execute(
        "INSERT OR REPLACE INTO gene_attribution "
        "(gene_id, party_id, participant_id, authored_at) "
        "VALUES (?, ?, NULL, ?)",
        (gene_id, party_id, time.time()),
    )
    genome.conn.commit()


# ── Tier 5: Harmonic co-activation boost ──────────────────────────────


class TestHarmonicBoost:
    """Genes linked via harmonic_links get a score bonus when both are
    candidates."""

    def test_harmonic_boost_increases_score(self, genome):
        """Two candidates linked harmonically should both get boosted."""
        g1 = make_gene("auth login handler code", domains=["auth"])
        g2 = make_gene("auth token validation logic", domains=["auth"])
        g3 = make_gene("auth session management util", domains=["auth"])
        id1 = genome.upsert_gene(g1)
        id2 = genome.upsert_gene(g2)
        id3 = genome.upsert_gene(g3)

        # Query WITHOUT harmonic links
        genome.query_genes(["auth"], [])
        scores_before = dict(genome.last_query_scores)

        # Now add harmonic links between g1 and g2
        genome.store_harmonic_weights([(id1, id2, 0.9)])

        genome.query_genes(["auth"], [])
        scores_after = dict(genome.last_query_scores)

        # Both linked genes should have higher scores
        assert scores_after[id1] > scores_before[id1]
        assert scores_after[id2] > scores_before[id2]
        # Unlinked gene should be unchanged
        assert scores_after.get(id3, 0) == scores_before.get(id3, 0)

    def test_harmonic_boost_capped_at_3(self, genome):
        """Harmonic bonus per gene is capped at 3.0 even with many links."""
        genes = []
        ids = []
        for i in range(6):
            g = make_gene(f"network module part {i} details", domains=["network"])
            gid = genome.upsert_gene(g)
            genes.append(g)
            ids.append(gid)

        # Query without harmonic links — baseline score for gene 0
        genome.query_genes(["network"], [])
        baseline = genome.last_query_scores[ids[0]]

        # Link gene 0 to all others — 5 links
        weights = [(ids[0], ids[j], 0.8) for j in range(1, 6)]
        genome.store_harmonic_weights(weights)

        genome.query_genes(["network"], [])
        boosted = genome.last_query_scores[ids[0]]

        # Harmonic bonus is capped at 3.0 per gene
        harmonic_delta = boosted - baseline
        assert harmonic_delta <= 3.0 + 0.01  # tiny float tolerance
        assert harmonic_delta > 0  # should have some boost

    def test_no_harmonic_table_is_safe(self, genome):
        """query_genes works fine when harmonic_links table doesn't exist."""
        g = make_gene("server startup code", domains=["server"])
        genome.upsert_gene(g)
        # harmonic_links was never created — should not error
        results = genome.query_genes(["server"], [])
        assert len(results) >= 1

    def test_harmonic_only_between_candidates(self, genome):
        """Harmonic links only boost when BOTH ends are candidates."""
        g1 = make_gene("api endpoint handler code", domains=["api"])
        g2 = make_gene("database schema migration tool", domains=["database"])
        id1 = genome.upsert_gene(g1)
        id2 = genome.upsert_gene(g2)

        # Query for "api" baseline
        genome.query_genes(["api"], [])
        score_before = genome.last_query_scores[id1]

        genome.store_harmonic_weights([(id1, id2, 0.9)])

        # Query only for "api" — g2 is not a candidate
        genome.query_genes(["api"], [])
        score_after = genome.last_query_scores[id1]

        # g1 score should not change — g2 is not in candidates
        assert score_after == score_before


# ── Access-rate tiebreaker ────────────────────────────────────────────


class TestAccessRateTiebreaker:
    """Genes with higher recent access rate get a small score bonus."""

    def test_hot_gene_wins_tie(self, genome):
        """Between two equally-scored genes, the hotter one ranks higher."""
        g1 = make_gene("cache invalidation strategy alpha", domains=["cache"])
        g2 = make_gene("cache invalidation strategy beta", domains=["cache"])
        id1 = genome.upsert_gene(g1)
        id2 = genome.upsert_gene(g2)

        # Touch g1 many times to build access rate
        now = time.time()
        cur = genome.conn.cursor()
        row = cur.execute(
            "SELECT epigenetics FROM genes WHERE gene_id = ?", (id1,)
        ).fetchone()
        epi = parse_epigenetics(row["epigenetics"], use_cache=False)
        # Simulate 10 accesses in the last hour
        epi.recent_accesses = [now - i * 60 for i in range(10)]
        cur.execute(
            "UPDATE genes SET epigenetics = ? WHERE gene_id = ?",
            (epi.model_dump_json(), id1),
        )
        genome.conn.commit()

        genome.query_genes(["cache"], [])
        scores = genome.last_query_scores

        # g1 should have a small bonus over g2
        assert scores[id1] > scores[id2]

    def test_access_rate_bonus_bounded(self, genome):
        """Access-rate bonus is small (max 0.25) relative to base score."""
        g = make_gene("config parser module code", domains=["config"])
        gid = genome.upsert_gene(g)

        # Get baseline score without access rate
        genome.query_genes(["config"], [])
        baseline = genome.last_query_scores[gid]

        # Set extremely high access rate
        now = time.time()
        cur = genome.conn.cursor()
        row = cur.execute(
            "SELECT epigenetics FROM genes WHERE gene_id = ?", (gid,)
        ).fetchone()
        epi = parse_epigenetics(row["epigenetics"], use_cache=False)
        # 100 accesses all in last second = very high rate
        epi.recent_accesses = [now - 0.01 * i for i in range(100)]
        cur.execute(
            "UPDATE genes SET epigenetics = ? WHERE gene_id = ?",
            (epi.model_dump_json(), gid),
        )
        genome.conn.commit()

        genome.query_genes(["config"], [])
        boosted = genome.last_query_scores[gid]

        # Access-rate contribution: 0.05 * min(rate * 3600, 5) = 0.05 * 5 = 0.25 max
        delta = boosted - baseline
        assert 0 < delta <= 0.25 + 0.01

    def test_no_recent_accesses_no_bonus(self, genome):
        """Gene with empty recent_accesses gets no tiebreaker bonus."""
        g1 = make_gene("router middleware handler code", domains=["router"])
        g2 = make_gene("router path matching engine code", domains=["router"])
        id1 = genome.upsert_gene(g1)
        id2 = genome.upsert_gene(g2)

        # Neither gene has been touched — both have empty recent_accesses
        genome.query_genes(["router"], [])
        scores = genome.last_query_scores

        # Both should have equal tier-1 scores (no tiebreaker applies)
        # Allow tiny float difference from IDF
        assert abs(scores[id1] - scores[id2]) < 0.01


# ── Party ID filter + attribution bonus ───────────────────────────────


class TestPartyIdFilter:
    """party_id filters Tiers 1-3 and adds a +0.5 attribution bonus."""

    def test_party_filter_returns_only_attributed_genes(self, genome):
        """With party_id set, only genes attributed to that party appear."""
        _insert_party(genome, "alice")
        _insert_party(genome, "bob")

        g1 = make_gene("deploy pipeline script code", domains=["deploy"])
        g2 = make_gene("deploy rollback procedure code", domains=["deploy"])
        id1 = genome.upsert_gene(g1)
        id2 = genome.upsert_gene(g2)

        _attribute_gene(genome, id1, "alice")
        _attribute_gene(genome, id2, "bob")

        results = genome.query_genes(["deploy"], [], party_id="alice")
        result_ids = {g.gene_id for g in results}

        assert id1 in result_ids
        assert id2 not in result_ids

    def test_no_party_filter_returns_all(self, genome):
        """Without party_id, all genes are returned (backward compat)."""
        _insert_party(genome, "alice")

        g1 = make_gene("logging framework util code", domains=["logging"])
        g2 = make_gene("logging rotation config code", domains=["logging"])
        id1 = genome.upsert_gene(g1)
        id2 = genome.upsert_gene(g2)

        _attribute_gene(genome, id1, "alice")
        # g2 has no attribution

        results = genome.query_genes(["logging"], [])
        result_ids = {g.gene_id for g in results}

        assert id1 in result_ids
        assert id2 in result_ids

    def test_party_attribution_bonus(self, genome):
        """Attributed genes get a +0.5 score bonus."""
        _insert_party(genome, "alice")

        g1 = make_gene("testing utility helper code", domains=["testing"])
        g2 = make_gene("testing mock framework code", domains=["testing"])
        id1 = genome.upsert_gene(g1)
        id2 = genome.upsert_gene(g2)

        _attribute_gene(genome, id1, "alice")
        _attribute_gene(genome, id2, "alice")

        # Query without party_id — no bonus
        genome.query_genes(["testing"], [])
        scores_no_party = dict(genome.last_query_scores)

        # Query with party_id — should add +0.5
        genome.query_genes(["testing"], [], party_id="alice")
        scores_with_party = dict(genome.last_query_scores)

        assert scores_with_party[id1] > scores_no_party[id1]
        assert scores_with_party[id1] - scores_no_party[id1] == pytest.approx(0.5, abs=0.01)

    def test_party_filter_legacy_unattributed_still_retrievable(self, genome):
        """Legacy (unattributed) genes remain retrievable under party_id filter.

        Phase 2a semantics (2026-04-14): the party filter excludes genes
        attributed to OTHER parties, but INCLUDES genes with no
        gene_attribution row at all. Without this fallback, retrieval on
        the predominantly-unattributed production genome would collapse to
        ~0 hits the moment any caller passed party_id.
        """
        g = make_gene("metrics dashboard widget code", domains=["metrics"])
        genome.upsert_gene(g)

        _insert_party(genome, "alice")
        # Gene has no attribution row → included regardless of party
        results = genome.query_genes(["metrics"], [], party_id="alice")
        assert len(results) == 1
        assert results[0].content == "metrics dashboard widget code"

    def test_party_filter_empty_party_returns_nothing(self, genome):
        """party_id with no attributed genes raises PromoterMismatch."""
        _insert_party(genome, "alice")
        _insert_party(genome, "ghost")

        g = make_gene("scheduler task queue code", domains=["scheduler"])
        gid = genome.upsert_gene(g)
        _attribute_gene(genome, gid, "alice")

        with pytest.raises(PromoterMismatch):
            genome.query_genes(["scheduler"], [], party_id="ghost")


# ── Integration: all three dimensions together ────────────────────────


class TestIntegration:
    """Verify the three dimensions compose correctly."""

    def test_all_dimensions_compose(self, genome):
        """Harmonic + access-rate + party all apply in one query."""
        _insert_party(genome, "team_alpha")

        g1 = make_gene("worker pool manager alpha code", domains=["worker"])
        g2 = make_gene("worker thread scheduler beta code", domains=["worker"])
        g3 = make_gene("worker health check gamma code", domains=["worker"])
        id1 = genome.upsert_gene(g1)
        id2 = genome.upsert_gene(g2)
        id3 = genome.upsert_gene(g3)

        # Attribute all to team_alpha
        for gid in [id1, id2, id3]:
            _attribute_gene(genome, gid, "team_alpha")

        # Add harmonic link between g1 and g2
        genome.store_harmonic_weights([(id1, id2, 0.8)])

        # Make g1 hot
        now = time.time()
        cur = genome.conn.cursor()
        row = cur.execute(
            "SELECT epigenetics FROM genes WHERE gene_id = ?", (id1,)
        ).fetchone()
        epi = parse_epigenetics(row["epigenetics"], use_cache=False)
        epi.recent_accesses = [now - i * 30 for i in range(20)]
        cur.execute(
            "UPDATE genes SET epigenetics = ? WHERE gene_id = ?",
            (epi.model_dump_json(), id1),
        )
        genome.conn.commit()

        results = genome.query_genes(["worker"], [], party_id="team_alpha")
        scores = genome.last_query_scores

        # g1 has: tier-1 + harmonic + access-rate + party bonus
        # g2 has: tier-1 + harmonic + party bonus
        # g3 has: tier-1 + party bonus only
        assert scores[id1] > scores[id2]
        assert scores[id2] > scores[id3]

        # All three should be returned (all attributed to team_alpha)
        result_ids = [g.gene_id for g in results]
        assert id1 in result_ids
        assert id2 in result_ids
        assert id3 in result_ids

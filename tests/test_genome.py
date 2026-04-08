"""Gate 1 — Genome storage tests (no model calls, in-memory SQLite)."""

import pytest
import time

from helix_context.genome import Genome
from helix_context.schemas import Gene, ChromatinState, PromoterTags, EpigeneticMarkers
from helix_context.exceptions import PromoterMismatch

from tests.conftest import make_gene


# ── Gene ID ─────────────────────────────────────────────────────────


class TestGeneId:
    def test_content_addressable(self):
        """Same content always produces the same gene_id."""
        assert Genome.make_gene_id("hello") == Genome.make_gene_id("hello")

    def test_different_content_different_id(self):
        assert Genome.make_gene_id("hello") != Genome.make_gene_id("world")

    def test_id_length(self):
        assert len(Genome.make_gene_id("anything")) == 16


# ── Basic CRUD ──────────────────────────────────────────────────────


class TestCrud:
    def test_insert_and_retrieve(self, genome):
        gene = make_gene("auth middleware code", domains=["auth"])
        gene_id = genome.upsert_gene(gene)

        retrieved = genome.get_gene(gene_id)
        assert retrieved is not None
        assert retrieved.content == "auth middleware code"
        assert retrieved.gene_id == gene_id

    def test_upsert_is_idempotent(self, genome):
        gene = make_gene("same content twice", domains=["test"])
        id1 = genome.upsert_gene(gene)
        id2 = genome.upsert_gene(gene)

        assert id1 == id2
        assert genome.stats()["total_genes"] == 1

    def test_upsert_updates_fields(self, genome):
        gene = make_gene("evolving content", domains=["v1"])
        gene_id = genome.upsert_gene(gene)

        # Update the gene's promoter
        gene.promoter.domains = ["v2"]
        genome.upsert_gene(gene)

        retrieved = genome.get_gene(gene_id)
        assert "v2" in retrieved.promoter.domains

    def test_get_nonexistent_returns_none(self, genome):
        assert genome.get_gene("doesnotexist") is None


# ── Promoter Retrieval ──────────────────────────────────────────────


class TestPromoterRetrieval:
    def test_domain_match(self, genome):
        genome.upsert_gene(make_gene("auth code", domains=["auth"]))
        genome.upsert_gene(make_gene("db code", domains=["database"]))
        genome.upsert_gene(make_gene("ui code", domains=["ui"]))

        results = genome.query_genes(domains=["auth"], entities=[])
        assert len(results) == 1
        assert results[0].content == "auth code"

    def test_entity_match(self, genome):
        genome.upsert_gene(make_gene("jwt handler", entities=["jwt"]))
        genome.upsert_gene(make_gene("css styles", entities=["tailwind"]))

        results = genome.query_genes(domains=[], entities=["jwt"])
        assert len(results) == 1
        assert results[0].content == "jwt handler"

    def test_multi_domain_ranking(self, genome):
        """Gene matching more tags should rank higher."""
        genome.upsert_gene(make_gene("auth only", domains=["auth"]))
        genome.upsert_gene(make_gene(
            "auth + security combo",
            domains=["auth", "security"],
        ))

        results = genome.query_genes(domains=["auth", "security"], entities=[])
        assert len(results) == 2
        # The gene matching both tags should come first
        assert results[0].content == "auth + security combo"

    def test_no_match_raises_promoter_mismatch(self, genome):
        genome.upsert_gene(make_gene("only auth", domains=["auth"]))

        with pytest.raises(PromoterMismatch):
            genome.query_genes(domains=["quantum_physics"], entities=[])

    def test_empty_terms_raises_promoter_mismatch(self, genome):
        with pytest.raises(PromoterMismatch):
            genome.query_genes(domains=[], entities=[])

    def test_chromatin_filter(self, genome):
        """HETEROCHROMATIN genes should be excluded from queries."""
        genome.upsert_gene(make_gene("active gene", domains=["auth"]))
        genome.upsert_gene(make_gene(
            "stale gene",
            domains=["auth"],
            chromatin=ChromatinState.HETEROCHROMATIN,
        ))

        results = genome.query_genes(domains=["auth"], entities=[])
        assert len(results) == 1
        assert results[0].content == "active gene"


# ── Fix 1: Synonym Expansion ───────────────────────────────────────


class TestSynonymExpansion:
    def test_synonym_expands_query(self, genome):
        """Querying 'slow' should also match genes tagged with 'performance'."""
        genome.upsert_gene(make_gene(
            "latency optimization guide",
            domains=["performance"],
        ))

        # 'slow' isn't a direct tag, but synonym_map maps it to 'performance'
        results = genome.query_genes(domains=["slow"], entities=[])
        assert len(results) == 1
        assert results[0].content == "latency optimization guide"

    def test_synonym_expands_multiple(self, genome):
        """'auth' should expand to match 'jwt', 'login', 'security', 'token'."""
        genome.upsert_gene(make_gene("jwt handler", domains=["jwt"]))
        genome.upsert_gene(make_gene("login page", domains=["login"]))
        genome.upsert_gene(make_gene("unrelated css", domains=["ui"]))

        results = genome.query_genes(domains=["auth"], entities=[])
        assert len(results) == 2
        contents = {r.content for r in results}
        assert "jwt handler" in contents
        assert "login page" in contents

    def test_direct_match_still_works(self, genome):
        """Synonym expansion should not break direct matches."""
        genome.upsert_gene(make_gene("auth module", domains=["auth"]))

        results = genome.query_genes(domains=["auth"], entities=[])
        assert len(results) == 1


# ── Fix 1: Co-Activation Pull-Forward ──────────────────────────────


class TestCoActivation:
    def test_coactivated_genes_pulled_in(self, genome):
        """Querying for Gene A should also return Gene B if A co-activates with B."""
        gene_b = make_gene("background service", domains=["background"], gene_id="gene_b_id")
        gene_a = make_gene(
            "auth middleware",
            domains=["auth"],
            co_activated_with=["gene_b_id"],
            gene_id="gene_a_id",
        )

        genome.upsert_gene(gene_b)
        genome.upsert_gene(gene_a)

        results = genome.query_genes(domains=["auth"], entities=[])

        result_ids = {r.gene_id for r in results}
        assert "gene_a_id" in result_ids
        assert "gene_b_id" in result_ids  # Pulled in via co-activation

    def test_coactivation_respects_chromatin(self, genome):
        """Co-activated genes in HETEROCHROMATIN should NOT be pulled in."""
        gene_b = make_gene(
            "stale service",
            domains=["background"],
            chromatin=ChromatinState.HETEROCHROMATIN,
            gene_id="stale_b",
        )
        gene_a = make_gene(
            "auth code",
            domains=["auth"],
            co_activated_with=["stale_b"],
            gene_id="active_a",
        )

        genome.upsert_gene(gene_b)
        genome.upsert_gene(gene_a)

        results = genome.query_genes(domains=["auth"], entities=[])
        result_ids = {r.gene_id for r in results}
        assert "active_a" in result_ids
        assert "stale_b" not in result_ids  # Filtered by chromatin


# ── Touch + Co-Activation Links ────────────────────────────────────


class TestEpigenetics:
    def test_touch_updates_access(self, genome):
        gene = make_gene("touchable", domains=["test"])
        gid = genome.upsert_gene(gene)

        before = genome.get_gene(gid)
        initial_count = before.epigenetics.access_count

        genome.touch_genes([gid])
        after = genome.get_gene(gid)

        assert after.epigenetics.access_count == initial_count + 1
        assert after.epigenetics.last_accessed >= before.epigenetics.last_accessed
        assert after.chromatin == ChromatinState.OPEN

    def test_link_coactivated_creates_mutual_links(self, genome):
        g1 = make_gene("gene one", domains=["test"], gene_id="g1")
        g2 = make_gene("gene two", domains=["test"], gene_id="g2")
        genome.upsert_gene(g1)
        genome.upsert_gene(g2)

        genome.link_coactivated(["g1", "g2"])

        r1 = genome.get_gene("g1")
        r2 = genome.get_gene("g2")
        assert "g2" in r1.epigenetics.co_activated_with
        assert "g1" in r2.epigenetics.co_activated_with


# ── Compaction ──────────────────────────────────────────────────────


class TestCompaction:
    def test_stale_genes_get_compacted(self, genome):
        gene = make_gene("old gene", domains=["test"])
        # Backdate the access time
        gene.epigenetics.last_accessed = time.time() - 7200  # 2 hours ago
        gene.epigenetics.decay_score = 0.25  # Just above the threshold
        genome.upsert_gene(gene)

        # First compaction: decay 0.25 * 0.95 = 0.2375 → below 0.3 → HETEROCHROMATIN
        compacted = genome.compact()
        assert compacted == 1

        retrieved = genome.get_gene(gene.gene_id)
        assert retrieved.chromatin == ChromatinState.HETEROCHROMATIN

    def test_fresh_genes_not_compacted(self, genome):
        gene = make_gene("fresh gene", domains=["test"])
        # last_accessed is now (default), well within stale_threshold
        genome.upsert_gene(gene)

        compacted = genome.compact()
        assert compacted == 0

        retrieved = genome.get_gene(gene.gene_id)
        assert retrieved.chromatin == ChromatinState.OPEN


# ── Stats ───────────────────────────────────────────────────────────


class TestStats:
    def test_empty_genome_stats(self, genome):
        stats = genome.stats()
        assert stats["total_genes"] == 0
        assert stats["compression_ratio"] == 0  # 0 / max(0, 1) = 0

    def test_stats_after_insert(self, genome):
        genome.upsert_gene(make_gene("some content", domains=["test"]))
        stats = genome.stats()
        assert stats["total_genes"] == 1
        assert stats["open"] == 1
        assert stats["total_chars_raw"] > 0

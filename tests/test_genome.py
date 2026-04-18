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

    def test_provenance_metadata_round_trips(self, genome):
        gene = make_gene("context packet notes", domains=["docs"])
        gene.source_id = "/repo/docs/notes.md"
        gene.repo_root = "/repo"
        gene.source_kind = "doc"
        gene.observed_at = 123.0
        gene.mtime = 120.0
        gene.content_hash = "deadbeef"
        gene.volatility_class = "stable"
        gene.authority_class = "primary"
        gene.support_span = "12:30"
        gene.last_verified_at = 124.0

        gene_id = genome.upsert_gene(gene, apply_gate=False)
        retrieved = genome.get_gene(gene_id)

        assert retrieved is not None
        assert retrieved.repo_root == "/repo"
        assert retrieved.source_kind == "doc"
        assert retrieved.observed_at == 123.0
        assert retrieved.mtime == 120.0
        assert retrieved.content_hash == "deadbeef"
        assert retrieved.volatility_class == "stable"
        assert retrieved.authority_class == "primary"
        assert retrieved.support_span == "12:30"
        assert retrieved.last_verified_at == 124.0

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


# ── Phase 2a: party_id filter semantics ────────────────────────────


class TestPartyScoping:
    """query_genes(party_id=...) implements 3-way partition:
       attributed-to-me + unattributed (included), other-party (excluded).
    """

    def _register_party(self, genome, party_id: str) -> None:
        """Minimal party row so FK-constrained gene_attribution writes succeed."""
        genome.conn.execute(
            "INSERT OR IGNORE INTO parties "
            "(party_id, display_name, trust_domain, created_at) "
            "VALUES (?, ?, 'local', ?)",
            (party_id, party_id, time.time()),
        )
        genome.conn.commit()

    def _attribute(self, genome, gene_id: str, party_id: str) -> None:
        genome.conn.execute(
            "INSERT OR REPLACE INTO gene_attribution "
            "(gene_id, party_id, participant_id, authored_at) "
            "VALUES (?, ?, NULL, ?)",
            (gene_id, party_id, time.time()),
        )
        genome.conn.commit()

    def test_legacy_unattributed_genes_still_retrievable(self, genome):
        """Genes with NO gene_attribution row MUST remain retrievable
        when party_id is set — a strict IN(...) would break retrieval
        on the predominantly-unattributed production genome."""
        self._register_party(genome, "alice")
        # Three genes, NONE of them attributed.
        for content in ("legacy auth 1", "legacy auth 2", "legacy auth 3"):
            genome.upsert_gene(make_gene(content, domains=["auth"]))

        results = genome.query_genes(domains=["auth"], entities=[], party_id="alice")
        assert len(results) == 3

    def test_other_party_genes_excluded(self, genome):
        """Cross-party leakage prevention: genes attributed to party B
        MUST NOT surface when querying as party A."""
        self._register_party(genome, "alice")
        self._register_party(genome, "bob")
        bob_gene = make_gene("bob's secret auth note", domains=["auth"])
        bob_id = genome.upsert_gene(bob_gene)
        self._attribute(genome, bob_id, "bob")
        # And one legacy gene alice SHOULD see.
        genome.upsert_gene(make_gene("public auth doc", domains=["auth"]))

        results = genome.query_genes(domains=["auth"], entities=[], party_id="alice")
        contents = {g.content for g in results}
        assert "bob's secret auth note" not in contents
        assert "public auth doc" in contents

    def test_own_party_genes_included(self, genome):
        """Genes attributed to the querying party show up."""
        self._register_party(genome, "alice")
        my_gene = make_gene("alice's auth note", domains=["auth"])
        my_id = genome.upsert_gene(my_gene)
        self._attribute(genome, my_id, "alice")

        results = genome.query_genes(domains=["auth"], entities=[], party_id="alice")
        assert any(g.content == "alice's auth note" for g in results)

    def test_no_party_id_preserves_existing_behavior(self, genome):
        """party_id=None (the default) means no filtering — all matching
        genes surface regardless of attribution."""
        self._register_party(genome, "bob")
        bob_gene = make_gene("bob's note", domains=["auth"])
        bob_id = genome.upsert_gene(bob_gene)
        self._attribute(genome, bob_id, "bob")
        genome.upsert_gene(make_gene("public note", domains=["auth"]))

        results = genome.query_genes(domains=["auth"], entities=[])  # no party_id
        contents = {g.content for g in results}
        assert "bob's note" in contents
        assert "public note" in contents


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
    def test_source_changed_gene_gets_marked(self, genome, tmp_path):
        """Gene with source_id pointing to a modified file gets EUCHROMATIN."""
        # Create a temp file and ingest a gene from it
        src_file = tmp_path / "test_source.py"
        src_file.write_text("def hello(): return 'world'")

        gene = make_gene("def hello(): return 'world'", domains=["test"])
        gene.source_id = str(src_file)
        gene.epigenetics.last_accessed = time.time() - 10  # Accessed 10s ago
        genome.upsert_gene(gene)

        # Modify the source file (mtime will be newer than last_accessed)
        time.sleep(0.1)
        src_file.write_text("def hello(): return 'changed!'")

        # Compaction should detect the change
        changed = genome.compact()
        assert changed == 1

        retrieved = genome.get_gene(gene.gene_id)
        assert retrieved.chromatin == ChromatinState.EUCHROMATIN
        assert retrieved.epigenetics.decay_score == 0.5

    def test_unchanged_source_not_affected(self, genome, tmp_path):
        """Gene with source_id pointing to unchanged file stays OPEN."""
        src_file = tmp_path / "stable.py"
        src_file.write_text("CONSTANT = 42")

        gene = make_gene("CONSTANT = 42", domains=["test"])
        gene.source_id = str(src_file)
        # last_accessed is AFTER file mtime
        gene.epigenetics.last_accessed = time.time() + 10
        genome.upsert_gene(gene)

        changed = genome.compact()
        assert changed == 0

        retrieved = genome.get_gene(gene.gene_id)
        assert retrieved.chromatin == ChromatinState.OPEN
        assert retrieved.epigenetics.decay_score == 1.0

    def test_gene_without_source_not_affected(self, genome):
        """Gene with no source_id (conversation, manual) never decays."""
        gene = make_gene("conversation exchange", domains=["test"])
        gene.epigenetics.last_accessed = time.time() - 86400 * 365  # 1 year old
        genome.upsert_gene(gene)

        changed = genome.compact()
        assert changed == 0

        retrieved = genome.get_gene(gene.gene_id)
        assert retrieved.chromatin == ChromatinState.OPEN
        assert retrieved.epigenetics.decay_score == 1.0

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

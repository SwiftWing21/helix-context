"""Sprint 4 - seeded edges, Hebbian decay, dense-rank miss weighting."""

from __future__ import annotations

import pytest

from helix_context.seeded_edges import (
    SOURCE_SEEDED,
    SOURCE_CO_RETRIEVED,
    SOURCE_CWOLA,
    SOURCE_WEIGHT_MULTIPLIER,
    CO_PROMOTE_MIN_COUNT,
    CO_PROMOTE_MIN_RATIO,
    PRUNE_FLOOR,
    _laplace_ratio,
    effective_weight,
    dense_rank,
    miss_weight,
    seed_edges,
    multi_signal_overlap,
    update_edge_evidence,
)


class TestLaplaceRatio:
    def test_neutral_at_no_evidence(self):
        assert _laplace_ratio(0, 0.0) == pytest.approx(0.5)

    def test_rises_with_co_count(self):
        assert _laplace_ratio(10, 0.0) > _laplace_ratio(1, 0.0) > 0.5

    def test_falls_with_miss_count(self):
        assert _laplace_ratio(0, 10.0) < _laplace_ratio(0, 1.0) < 0.5

    def test_one_outlier_does_not_flip(self):
        # seed at neutrality, one miss moves only slightly
        shifted = _laplace_ratio(0, 1.0)
        assert 0.3 < shifted < 0.5


class TestEffectiveWeight:
    def test_source_multiplier_applied(self):
        w = effective_weight(1.0, SOURCE_SEEDED, 0, 0.0)
        assert w == pytest.approx(0.15)  # 1.0 * 0.3 * 0.5
        w = effective_weight(1.0, SOURCE_CO_RETRIEVED, 0, 0.0)
        assert w == pytest.approx(0.35)
        w = effective_weight(1.0, SOURCE_CWOLA, 0, 0.0)
        assert w == pytest.approx(0.5)

    def test_proven_edge_reaches_full_source_weight(self):
        # Lots of co-evidence, no misses -> ratio approaches 1.0
        w = effective_weight(1.0, SOURCE_CWOLA, 100, 0.0)
        assert 0.99 < w / SOURCE_WEIGHT_MULTIPLIER[SOURCE_CWOLA] <= 1.0


class TestDenseRank:
    def test_unique_scores_give_unique_ranks(self):
        ranks = dense_rank([5.0, 3.0, 2.0, 1.0])
        assert ranks == {0: 1, 1: 2, 2: 3, 3: 4}

    def test_ties_share_rank(self):
        ranks = dense_rank([5.0, 3.0, 3.0, 1.0])
        assert ranks == {0: 1, 1: 2, 2: 2, 3: 3}

    def test_all_tied(self):
        ranks = dense_rank([5.0, 5.0, 5.0])
        assert ranks == {0: 1, 1: 1, 2: 1}


class TestMissWeight:
    def test_in_cut_is_zero(self):
        assert miss_weight(1, 8) == 0.0
        assert miss_weight(8, 8) == 0.0

    def test_just_below_cut_is_heavy(self):
        assert miss_weight(9, 8) == pytest.approx(8 / 9)  # ~0.889

    def test_deep_in_pool_is_light(self):
        assert miss_weight(100, 8) == pytest.approx(0.08)

    def test_capped_at_one(self):
        # Not physically reachable (rank <= max_genes is in_cut), but
        # defensive behaviour when callers hand in weird input.
        assert miss_weight(4, 8) == 0.0  # in cut
        # Guard against negative rank if somehow passed
        assert miss_weight(-1, 8) == 0.0


class TestMultiSignalOverlap:
    """Overlap gate uses promoter + key_values + chromatin state.
    Plain FakeGenome stub mirrors read_conn.cursor() → fetchall shape."""

    class _FakeConn:
        def __init__(self, rows):
            self._rows = rows

        def cursor(self):
            return self

        def execute(self, sql, params):
            ids = {params[0], params[1]}
            self._result = [r for r in self._rows if r["gene_id"] in ids]
            return self

        def fetchall(self):
            return self._result

    class _FakeGenome:
        def __init__(self, rows):
            self.read_conn = TestMultiSignalOverlap._FakeConn(rows)

    def _row(self, gid, domains=None, entities=None, kv=None, chromatin=0):
        import json
        class Row(dict):
            def __getitem__(self, k): return super().__getitem__(k)
        return Row({
            "gene_id": gid,
            "promoter": json.dumps({"domains": domains or [], "entities": entities or []}),
            "key_values": json.dumps(kv or []),
            "chromatin": chromatin,
        })

    def test_no_overlap_rejects(self):
        g = self._FakeGenome([
            self._row("a", domains=["db"], entities=["pg"]),
            self._row("b", domains=["ui"], entities=["react"]),
        ])
        assert multi_signal_overlap(g, "a", "b") is False

    def test_domain_plus_open_is_two_signals(self):
        g = self._FakeGenome([
            self._row("a", domains=["auth"], chromatin=0),
            self._row("b", domains=["auth"], chromatin=0),
        ])
        assert multi_signal_overlap(g, "a", "b") is True

    def test_single_signal_rejects(self):
        # Shared domain but one gene is EUCHROMATIN - only 1 signal.
        g = self._FakeGenome([
            self._row("a", domains=["auth"], chromatin=0),
            self._row("b", domains=["auth"], chromatin=1),
        ])
        assert multi_signal_overlap(g, "a", "b") is False

    def test_kv_plus_entity_passes(self):
        g = self._FakeGenome([
            self._row("a", entities=["User"], kv=["port=5432"], chromatin=1),
            self._row("b", entities=["User"], kv=["port=8080"], chromatin=1),
        ])
        assert multi_signal_overlap(g, "a", "b") is True


class TestUpdateEdgeEvidence:
    """update_edge_evidence reads harmonic_links, increments co / miss,
    promotes, and prunes. Uses an in-memory sqlite to keep the test
    hermetic."""

    def _setup(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE harmonic_links (
                gene_id_a  TEXT NOT NULL,
                gene_id_b  TEXT NOT NULL,
                weight     REAL NOT NULL,
                updated_at REAL NOT NULL,
                source     TEXT NOT NULL DEFAULT 'co_retrieved',
                co_count   INTEGER NOT NULL DEFAULT 0,
                miss_count REAL NOT NULL DEFAULT 0.0,
                created_at REAL,
                PRIMARY KEY (gene_id_a, gene_id_b)
            )
        """)

        class _Genome:
            pass
        g = _Genome()
        g.conn = conn
        return g, conn

    def _insert(self, conn, a, b, source="seeded", co=0, miss=0.0):
        conn.execute(
            """INSERT INTO harmonic_links
               (gene_id_a, gene_id_b, weight, updated_at, source, co_count, miss_count, created_at)
               VALUES (?, ?, 1.0, 0.0, ?, ?, ?, 0.0)""",
            (a, b, source, co, miss),
        )
        conn.commit()

    def test_co_expression_increments_co_count(self):
        g, conn = self._setup()
        self._insert(conn, "a", "b", source=SOURCE_SEEDED)
        gene_scores = {"a": 5.0, "b": 4.0, "c": 1.0}
        n = update_edge_evidence(g, gene_scores, ["a", "b"], max_genes=2)
        assert n == 1
        row = conn.execute(
            "SELECT co_count, miss_count, source FROM harmonic_links"
        ).fetchone()
        assert row["co_count"] == 1
        assert row["miss_count"] == 0.0

    def test_miss_counts_only_when_neighbour_is_candidate(self):
        g, conn = self._setup()
        self._insert(conn, "a", "b", source=SOURCE_SEEDED)
        # b not in gene_scores at all - candidacy gate rejects
        gene_scores = {"a": 5.0, "c": 1.0}
        update_edge_evidence(g, gene_scores, ["a"], max_genes=1)
        row = conn.execute("SELECT co_count, miss_count FROM harmonic_links").fetchone()
        assert row["co_count"] == 0
        assert row["miss_count"] == 0.0

    def test_miss_weighted_by_rank(self):
        g, conn = self._setup()
        self._insert(conn, "a", "b", source=SOURCE_SEEDED)
        # b is in candidate pool at rank 3 with max_genes=2 -> miss_weight=2/3
        gene_scores = {"a": 5.0, "c": 4.0, "b": 3.0}
        update_edge_evidence(g, gene_scores, ["a"], max_genes=2)
        row = conn.execute("SELECT miss_count FROM harmonic_links").fetchone()
        assert row["miss_count"] == pytest.approx(2 / 3)

    def test_promotion_after_sustained_evidence(self):
        g, conn = self._setup()
        self._insert(conn, "a", "b", source=SOURCE_SEEDED, co=2, miss=0.0)
        # One more co-expression -> co=3, ratio = 4/5 = 0.8 -> promote
        gene_scores = {"a": 5.0, "b": 4.0}
        update_edge_evidence(g, gene_scores, ["a", "b"], max_genes=2)
        row = conn.execute("SELECT source, co_count FROM harmonic_links").fetchone()
        assert row["co_count"] == 3
        assert row["source"] == SOURCE_CO_RETRIEVED

    def test_pruning_deletes_low_weight_edges(self):
        g, conn = self._setup()
        # Seed with lots of misses -> effective weight < prune floor
        self._insert(conn, "a", "b", source=SOURCE_SEEDED, co=0, miss=30.0)
        # Another miss - edge should be deleted
        gene_scores = {"a": 5.0, "b": 0.01}  # b in pool, ranked behind a
        update_edge_evidence(g, gene_scores, ["a"], max_genes=1)
        remaining = conn.execute(
            "SELECT COUNT(*) FROM harmonic_links"
        ).fetchone()[0]
        assert remaining == 0


class TestSeedEdges:
    """seed_edges integration - writes rows only when overlap_fn passes."""

    def _mk_genome(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE harmonic_links (
                gene_id_a TEXT NOT NULL,
                gene_id_b TEXT NOT NULL,
                weight REAL NOT NULL,
                updated_at REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'seeded',
                co_count INTEGER NOT NULL DEFAULT 0,
                miss_count REAL NOT NULL DEFAULT 0.0,
                created_at REAL,
                PRIMARY KEY (gene_id_a, gene_id_b)
            )
        """)
        conn.commit()

        class _G:
            pass
        g = _G()
        g.conn = conn
        return g, conn

    def test_empty_or_single_id_writes_nothing(self):
        g, conn = self._mk_genome()
        assert seed_edges(g, []) == 0
        assert seed_edges(g, ["only"]) == 0

    def test_overlap_fn_gate(self):
        g, conn = self._mk_genome()
        written = seed_edges(g, ["a", "b", "c"], overlap_fn=lambda *_: True)
        assert written == 3  # all pairs written
        written = seed_edges(g, ["x", "y"], overlap_fn=lambda *_: False)
        assert written == 0

    def test_idempotent_on_existing_pair(self):
        g, conn = self._mk_genome()
        seed_edges(g, ["a", "b"], overlap_fn=lambda *_: True)
        # Second call with same pair returns 0 - ON CONFLICT DO NOTHING
        written = seed_edges(g, ["a", "b"], overlap_fn=lambda *_: True)
        assert written == 0


class TestQueryGenesHebbianIntegration:
    """End-to-end: with seeded_edges_enabled, query_genes fires the
    evidence updater after building gene_scores. Uses a real Genome
    against :memory: + hand-inserted seeded edge."""

    def test_query_genes_increments_co_count(self):
        from helix_context.genome import Genome
        from helix_context.schemas import (
            Gene, PromoterTags, EpigeneticMarkers, ChromatinState,
        )

        g = Genome(":memory:", seeded_edges_enabled=True)

        def _gene(gid, domain):
            return Gene(
                gene_id=gid,
                content=f"content for {gid}",
                complement=f"summary for {gid}",
                codons=["codon1"],
                promoter=PromoterTags(domains=[domain], entities=["Entity"]),
                epigenetics=EpigeneticMarkers(),
                chromatin=ChromatinState.OPEN,
            )

        g.upsert_gene(_gene("a", "auth"), apply_gate=False)
        g.upsert_gene(_gene("b", "auth"), apply_gate=False)
        # Seed an edge between them
        g.conn.execute(
            """INSERT INTO harmonic_links
               (gene_id_a, gene_id_b, weight, updated_at, source, co_count, miss_count, created_at)
               VALUES ('a', 'b', 1.0, 0.0, 'seeded', 0, 0.0, 0.0)"""
        )
        g.conn.commit()

        # Query that should retrieve both
        g.query_genes(domains=["auth"], entities=["Entity"], max_genes=4)
        row = g.conn.execute(
            "SELECT co_count FROM harmonic_links WHERE gene_id_a='a' AND gene_id_b='b'"
        ).fetchone()
        assert row[0] == 1, "Hebbian hook should have incremented co_count"
        g.close()

    def test_query_genes_read_only_skips_hebbian_writeback(self):
        from helix_context.genome import Genome
        from helix_context.schemas import (
            Gene, PromoterTags, EpigeneticMarkers, ChromatinState,
        )

        g = Genome(":memory:", seeded_edges_enabled=True)

        def _gene(gid, domain):
            return Gene(
                gene_id=gid,
                content=f"content for {gid}",
                complement=f"summary for {gid}",
                codons=["codon1"],
                promoter=PromoterTags(domains=[domain], entities=["Entity"]),
                epigenetics=EpigeneticMarkers(),
                chromatin=ChromatinState.OPEN,
            )

        g.upsert_gene(_gene("a", "auth"), apply_gate=False)
        g.upsert_gene(_gene("b", "auth"), apply_gate=False)
        g.conn.execute(
            """INSERT INTO harmonic_links
               (gene_id_a, gene_id_b, weight, updated_at, source, co_count, miss_count, created_at)
               VALUES ('a', 'b', 1.0, 0.0, 'seeded', 0, 0.0, 0.0)"""
        )
        g.conn.commit()

        g.query_genes(
            domains=["auth"],
            entities=["Entity"],
            max_genes=4,
            read_only=True,
        )
        row = g.conn.execute(
            "SELECT co_count FROM harmonic_links WHERE gene_id_a='a' AND gene_id_b='b'"
        ).fetchone()
        assert row[0] == 0, "read_only queries must not mutate harmonic_links"
        g.close()

"""Bugbash BUG-1 / BUG-3: request-scoped retrieval scores + post-trim health.

BUG-1 (score/result atomicity): ``_build_context_impl`` used to read
``genome.last_query_scores`` at six-plus points spread across the turn
(tiering, slate ordering, legibility, budget trim, health). The map is
instance-global mutable state republished by EVERY retrieval, so a
concurrent request could associate query A's candidates with query B's
scores — silently altering tiering, ordering, trimming, and abstention.
The fix snapshots the map once (under the store lock) right after this
request's retrieval and threads the request-scoped dict through the
rest of the pipeline (``query_scores`` parameter on the refiners,
``_assemble``, and ``_compute_health``).

BUG-3 (pre-trim health): ``_assemble`` used to compute health over the
PRE-trim candidate list, so ``genes_expressed`` / coverage / freshness
described evidence that the budget trim had already dropped from the
returned context. Health must reflect the post-trim, actually-returned
set.
"""
from __future__ import annotations

import pytest
import threading
from types import SimpleNamespace

from cymatix_context.config import (
    BudgetConfig,
    ClassifierConfig,
    GenomeConfig,
    CymatixConfig,
    RibosomeConfig,
)
from cymatix_context.context_manager import CymatixContextManager
from cymatix_context.schemas import Gene, PromoterTags

from tests.conftest import MockCompressorBackend, make_gene, make_cymatix_config


def _make_manager():
    cfg = CymatixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=12, min_delivered_docs=0),  # legacy evict-path tests ([w24-floor-flip])
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        classifier=ClassifierConfig(enabled=False),
    )
    return CymatixContextManager(cfg)


def _make_genes(gene_ids, contents=None):
    return [
        Gene(
            gene_id=gid,
            content=(contents or {}).get(gid, f"{gid} content payload " * 20),
            complement=f"complement-{gid}",
            codons=["c"],
            promoter=PromoterTags(sequence_index=i),
        )
        for i, gid in enumerate(gene_ids)
    ]


def _spliced_map(gene_ids):
    return {
        gid: f"content for {gid}: " + ("payload " * 60)
        for gid in gene_ids
    }


SCORES = [10.0, 1.0, 8.0, 2.0, 9.0]
GENE_IDS = [f"gene_{i:02d}" for i in range(5)]


def _run_publication_blend(monkeypatch, genome, refiner, query_scores=None):
    """Exercise publication with deterministic signals, preserving blend math."""
    from cymatix_context.scoring import cymatics, ray_trace
    from cymatix_context.scoring.blend import apply_candidate_refiners

    monkeypatch.setattr(cymatics, "query_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(cymatics, "build_weight_vector", lambda *a, **k: None)
    monkeypatch.setattr(cymatics, "cached_doc_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(cymatics, "flux_score_dispatch", lambda *a, **k: 0.5)
    monkeypatch.setattr(
        ray_trace, "harmonic_bin_boost", lambda *a, **k: {"a": 0.25, "b": 0.25, "c": 0.25},
    )
    return apply_candidate_refiners(
        "query", _make_genes(["a", "b", "c"]), 3, genome=genome,
        use_cymatics=refiner == "cymatics", use_harmonic_bin=refiner == "harmonic",
        use_tcm=False, blend_mode="legacy", query_scores=query_scores,
    )


@pytest.mark.parametrize("refiner", ["cymatics", "harmonic"])
class TestPublishedScoreMapIsolation:
    def test_blend_publishes_copy_not_alias_of_request_dict(self, monkeypatch, refiner):
        request = {"a": 10.0, "b": 5.0, "c": 1.0}
        genome = SimpleNamespace(last_query_scores={"foreign": 999.0})
        candidates, _ = _run_publication_blend(monkeypatch, genome, refiner, request)
        assert [g.gene_id for g in candidates] == ["a", "b", "c"]
        assert request == {"a": 10.25, "b": 5.25, "c": 1.25}
        assert genome.last_query_scores == request
        assert genome.last_query_scores is not request
        request["a"] = -100.0
        assert genome.last_query_scores["a"] == 10.25

    def test_blend_direct_caller_does_not_mutate_live_map(self, monkeypatch, refiner):
        published = {"a": 10.0, "b": 5.0, "c": 1.0}
        genome = SimpleNamespace(last_query_scores=published)
        _run_publication_blend(monkeypatch, genome, refiner)
        assert published == {"a": 10.0, "b": 5.0, "c": 1.0}
        assert genome.last_query_scores == {"a": 10.25, "b": 5.25, "c": 1.25}

    @pytest.mark.parametrize("request_scoped", [False, True])
    def test_blend_snapshots_and_publishes_under_lock(self, monkeypatch, refiner, request_scoped):
        class ObservedGenome:
            def __init__(self):
                self._last_query_scores_lock = threading.Lock()
                self._scores = {"a": 10.0, "b": 5.0, "c": 1.0}
                self.accesses = []

            @property
            def last_query_scores(self):
                self.accesses.append(("read", self._last_query_scores_lock.locked()))
                return self._scores

            @last_query_scores.setter
            def last_query_scores(self, value):
                self.accesses.append(("write", self._last_query_scores_lock.locked()))
                self._scores = value

        genome = ObservedGenome()
        request = dict(genome._scores) if request_scoped else None
        _run_publication_blend(monkeypatch, genome, refiner, request)
        assert genome._scores == {"a": 10.25, "b": 5.25, "c": 1.25}
        assert ("write", True) in genome.accesses
        assert all(held for _, held in genome.accesses), genome.accesses
        if not request_scoped:
            assert ("read", True) in genome.accesses


def test_cold_tier_scores_are_written_under_publication_lock():
    """Cold scores preserve the hot scores while respecting snapshot readers."""
    import numpy as np
    from cymatix_context.knowledge_store import KnowledgeStore

    lock = threading.Lock()
    writes = []

    class ObservedScores(dict):
        def __setitem__(self, key, value):
            writes.append(lock.locked())
            super().__setitem__(key, value)

    genes = {g.gene_id: g for g in _make_genes(["cold-a", "cold-b"])}
    genome = SimpleNamespace(
        _sema_codec=SimpleNamespace(encode=lambda query: [1.0, 0.0]),
        _cold_sema_cache={"gene_ids": ["cold-a", "cold-b"], "matrix": np.array([[1.0, 0.0], [0.5, 0.5]])},
        _last_query_scores_lock=lock,
        last_query_scores=ObservedScores({"hot": 10.0}),
        get_doc=genes.get,
    )
    result = KnowledgeStore.query_cold_tier(genome, "query")
    assert [g.gene_id for g in result] == ["cold-a", "cold-b"]
    assert genome.last_query_scores == {"hot": 10.0, "cold-a": 1.0, "cold-b": 0.5}
    assert writes == [True, True]


# ─────────────────────────────────────────────────────────────────────
# BUG-1: request-scoped scores
# ─────────────────────────────────────────────────────────────────────

class TestRequestScopedScores:
    def test_assemble_trim_uses_request_scoped_scores(self):
        """``_assemble(query_scores=...)`` must trim by THIS request's
        scores even when ``genome.last_query_scores`` has since been
        republished by a concurrent request with an inverted ranking."""
        mgr = _make_manager()
        try:
            genes = _make_genes(GENE_IDS)
            request_scores = {
                gid: SCORES[i] for i, gid in enumerate(GENE_IDS)
            }
            # Simulate a concurrent request B publishing its own map
            # between this request's retrieval and assembly. Ranking is
            # inverted so a genome-state read drops the WRONG gene.
            mgr.genome.last_query_scores = {
                gid: 100.0 - SCORES[i] for i, gid in enumerate(GENE_IDS)
            }
            mgr.config.budget.expression_tokens = 300
            mgr.config.budget.ribosome_tokens = 100

            window = mgr._assemble(
                query="q",
                candidates=genes,
                spliced_map=_spliced_map(GENE_IDS),
                relation_graph={},
                answer_slate=None,
                query_scores=request_scores,
            )

            # gene_00 (request score 10) must survive; gene_01 (request
            # score 1) must be the first dropped. Under the clobbered
            # genome map the ranking is exactly inverted.
            assert "gene_00" in window.expressed_context
            assert "gene_01" not in window.expressed_context
        finally:
            mgr.close()

    def test_build_context_health_immune_to_concurrent_score_clobber(self):
        """End-to-end: after the refiner stage, a concurrent request
        republishes ``genome.last_query_scores`` with a foreign map.
        Downstream stages (tiering, trim, health) must keep using this
        request's own scores."""
        config = make_cymatix_config(
            synonym_map={"auth": ["jwt", "login", "security"]},
        )
        mgr = CymatixContextManager(config)
        mgr.ribosome.backend = MockCompressorBackend()
        try:
            seed = [
                make_gene(
                    "JWT authentication middleware",
                    domains=["auth", "security"], entities=["jwt"],
                    gene_id="auth_gene_001",
                ),
                make_gene(
                    "Database connection pooling",
                    domains=["database", "performance"], entities=["postgres"],
                    gene_id="db_gene_0001",
                ),
                make_gene(
                    "React component state management",
                    domains=["frontend", "react"], entities=["useState"],
                    gene_id="react_gene_01",
                ),
            ]
            for g in seed:
                mgr.genome.upsert_gene(g)

            real_scores: dict = {}
            original_refiners = mgr._apply_candidate_refiners

            def clobbering_refiners(*args, **kwargs):
                result = original_refiners(*args, **kwargs)
                # Snapshot what THIS request's scores actually are, then
                # simulate a concurrent request republishing the shared map.
                real_scores.update(dict(mgr.genome.last_query_scores or {}))
                mgr.genome.last_query_scores = {
                    gid: 10_000.0 + i
                    for i, gid in enumerate(real_scores)
                }
                return result

            mgr._apply_candidate_refiners = clobbering_refiners

            window = mgr.build_context("How does JWT auth work?")

            health = window.context_health
            assert health.genes_expressed >= 1, (
                "Pipeline mis-tiered/abstained under a foreign score map — "
                f"status={health.status!r}, metadata={window.metadata!r}"
            )
            # top_score_raw is computed off the score map; the request's
            # own scores are O(1-50), the foreign clobber is O(10_000).
            assert health.top_score_raw < 1_000.0, (
                "Health was computed from a concurrent request's score "
                f"map (top_score_raw={health.top_score_raw})."
            )
        finally:
            mgr.close()


# ─────────────────────────────────────────────────────────────────────
# BUG-3: health must reflect the post-trim, returned set
# ─────────────────────────────────────────────────────────────────────

class TestPostTrimHealth:
    def test_health_genes_expressed_matches_returned_set(self):
        """After a budget trim fires, ``health.genes_expressed`` must
        equal the number of documents actually in the returned context,
        not the pre-trim candidate count."""
        mgr = _make_manager()
        try:
            genes = _make_genes(GENE_IDS)
            mgr.genome.last_query_scores = {
                gid: SCORES[i] for i, gid in enumerate(GENE_IDS)
            }
            mgr.config.budget.expression_tokens = 300
            mgr.config.budget.ribosome_tokens = 100

            window = mgr._assemble(
                query="q",
                candidates=genes,
                spliced_map=_spliced_map(GENE_IDS),
                relation_graph={},
                answer_slate=None,
            )

            assert len(window.expressed_gene_ids) < len(genes), (
                "Budget did not fire a drop — lower expression_tokens to "
                "force a trim."
            )
            health = window.context_health
            assert health.genes_expressed == len(window.expressed_gene_ids)
            assert health.genes_expressed == window.metadata["genes_expressed"]
        finally:
            mgr.close()

    def test_health_coverage_excludes_trimmed_evidence(self):
        """A query term that appears ONLY in a trimmed-away document must
        not count toward coverage — the returned context does not contain
        that evidence."""
        mgr = _make_manager()
        try:
            contents = {
                "gene_01": "zzquniqueterm only lives here " * 20,
            }
            genes = _make_genes(GENE_IDS, contents=contents)
            mgr.genome.last_query_scores = {
                gid: SCORES[i] for i, gid in enumerate(GENE_IDS)
            }
            mgr.config.budget.expression_tokens = 300
            mgr.config.budget.ribosome_tokens = 100

            window = mgr._assemble(
                query="q",
                candidates=genes,
                spliced_map=_spliced_map(GENE_IDS),
                relation_graph={},
                answer_slate=None,
                query_signals=(["zzquniqueterm"], []),
            )

            # gene_01 (score=1, the global minimum) is trimmed first.
            assert "gene_01" not in window.expressed_context, (
                "Trim did not drop gene_01 — tighten the budget."
            )
            assert window.context_health.coverage == 0.0, (
                "Coverage counted a term whose only evidence was trimmed "
                "out of the returned context."
            )
        finally:
            mgr.close()


# ─────────────────────────────────────────────────────────────────────
# W1.6b (issue #350): the know/miss builder reads the WINDOW snapshot
# ─────────────────────────────────────────────────────────────────────

class TestKnowMissWindowSnapshot:
    """Issue #350: ``_compute_know_or_miss_block`` must compute reported
    confidence from the request-scoped snapshot ``build_context`` attaches
    to the window — the genome-global ``last_query_scores`` may already
    hold a CONCURRENT request's publication by the time the route reads it
    (W1.6a made that read atomic; this makes it correctly attributed)."""

    SCORES_A = {"g-A1": 10.0, "g-A2": 2.0}
    SCORES_B = {"g-B1": 0.42}

    def _env(self):
        from types import SimpleNamespace

        from cymatix_context.genome import Genome

        genome = Genome(path=":memory:")
        mgr = SimpleNamespace(
            genome=genome,
            config=None,  # -> load_calibration_from_toml() fallback
            _mtime_cache={},
            _cold_tier_peek=lambda q, k=3, min_cosine=0.4: [],
            _last_cold_peek_targets=[],
        )
        window = SimpleNamespace(
            expressed_gene_ids=["g-A1", "g-A2"],
            context_health=SimpleNamespace(
                freshness_min=None, genes_expressed=3, status="healthy",
            ),
            retrieval_scores=None,
            tier_contributions=None,
        )
        return genome, mgr, window

    def _spy_decide(self, monkeypatch):
        import cymatix_context.server.helpers as helpers

        captured = {}
        real = helpers.decide_know_or_miss

        def spy(window=None, **kw):
            captured.update(
                top_score=kw.get("top_score"), score_gap=kw.get("score_gap"),
            )
            return real(window=window, **kw)

        monkeypatch.setattr(helpers, "decide_know_or_miss", spy)
        return captured

    @staticmethod
    def _publish(genome, scores):
        # Byte-identical to the store's publication seam.
        with genome._last_query_scores_lock:
            genome.last_query_scores = dict(scores)

    def test_window_snapshot_wins_over_concurrent_publication(self, monkeypatch):
        """B publishes between A's retrieval and A's know/miss build; A's
        window carries A's snapshot -> A's confidence comes from A."""
        import cymatix_context.server.helpers as helpers

        genome, mgr, window = self._env()
        captured = self._spy_decide(monkeypatch)
        try:
            self._publish(genome, self.SCORES_A)   # A's retrieval
            window.retrieval_scores = dict(self.SCORES_A)
            window.tier_contributions = {}
            self._publish(genome, self.SCORES_B)   # B publishes mid-A
            helpers._compute_know_or_miss_block(
                cymatix=mgr, window=window, query="query A",
            )
            assert captured["top_score"] == 10.0
            assert captured["score_gap"] == 8.0
        finally:
            genome.close()

    def test_legacy_window_falls_back_to_genome_global(self, monkeypatch):
        """A window without a snapshot (packet-path callers) keeps the
        pre-#350 genome-global read."""
        import cymatix_context.server.helpers as helpers

        genome, mgr, window = self._env()
        captured = self._spy_decide(monkeypatch)
        try:
            self._publish(genome, self.SCORES_B)
            helpers._compute_know_or_miss_block(
                cymatix=mgr, window=window, query="query A",
            )
            assert captured["top_score"] == 0.42
        finally:
            genome.close()

    def test_build_context_attaches_excluded_snapshot(self):
        """build_context attaches the request-scoped snapshot to the window,
        and the fields never serialize into responses (exclude=True)."""
        mgr = _make_manager()
        mgr.ribosome.backend = MockCompressorBackend()
        try:
            genes = _make_genes(GENE_IDS)
            scores = dict(zip(GENE_IDS, SCORES))

            def fake_retrieve(domains, entities, max_genes, **_kwargs):
                mgr.genome.last_query_scores = dict(scores)
                return list(genes)

            mgr._retrieve = fake_retrieve
            mgr._express = fake_retrieve
            mgr._apply_candidate_refiners = (
                lambda q, cands, mg, **_kw: (list(cands), {})
            )
            win = mgr.build_context("anything")
            assert win.retrieval_scores == scores
            assert win.tier_contributions == {}
            dumped = win.model_dump()
            assert "retrieval_scores" not in dumped
            assert "tier_contributions" not in dumped
        finally:
            mgr.close()


# ─────────────────────────────────────────────────────────────────────
# W1.6b remainder (issue #350): /fingerprint snapshots at its own
# retrieval step
# ─────────────────────────────────────────────────────────────────────

class TestFingerprintSnapshot:
    """Issue #350: ``/fingerprint`` has no build_context window — it must
    capture (scores, tier_contributions) atomically under the store lock
    IMMEDIATELY after its own ``_retrieve`` and thread the map through the
    refiners, so a concurrent request republishing the genome-global maps
    during the refiner stage cannot contaminate the response."""

    SCORES_A = {"g-A1": 10.0, "g-A2": 2.0}
    TIERS_A = {"g-A1": {"fts5": 10.0}, "g-A2": {"fts5": 2.0}}
    SCORES_B = {"g-B1": 0.42}
    TIERS_B = {"g-B1": {"fts5": 0.42}}

    def _client_with_contaminating_refiners(self, monkeypatch):
        from tests.conftest import make_client

        from cymatix_context.schemas import Gene, PromoterTags

        client = make_client()
        cymatix = client.app.state.cymatix

        genes = [
            Gene(
                gene_id=gid,
                content=f"content-{gid}",
                complement=f"summary-{gid}",
                codons=[],
                promoter=PromoterTags(
                    domains=[], entities=[], intent="", summary="",
                ),
                source_id=f"src/{gid}.py",
            )
            for gid in self.SCORES_A
        ]

        def fake_retrieve(domains, entities, max_results, **_kw):
            # Publication seam: THIS request's retrieval publishes A.
            with cymatix.genome._last_query_scores_lock:
                cymatix.genome.last_query_scores = dict(self.SCORES_A)
                cymatix.genome.last_tier_contributions = {
                    k: dict(v) for k, v in self.TIERS_A.items()
                }
            return list(genes)

        def contaminating_refiners(query, candidates, max_results, **_kw):
            # Concurrent request B republishes the genome-global maps
            # while THIS request is still in its refiner stage — the
            # exact window the pre-#350 post-refiner read was exposed to.
            with cymatix.genome._last_query_scores_lock:
                cymatix.genome.last_query_scores = dict(self.SCORES_B)
                cymatix.genome.last_tier_contributions = {
                    k: dict(v) for k, v in self.TIERS_B.items()
                }
            return list(candidates), {g.gene_id: {} for g in candidates}

        monkeypatch.setattr(cymatix, "_retrieve", fake_retrieve)
        monkeypatch.setattr(cymatix, "_express", fake_retrieve)
        monkeypatch.setattr(
            cymatix, "_apply_candidate_refiners", contaminating_refiners,
        )
        return client

    def test_fingerprint_scores_immune_to_concurrent_publication(
        self, monkeypatch,
    ):
        client = self._client_with_contaminating_refiners(monkeypatch)
        resp = client.post("/fingerprint", json={
            "query": "anything", "max_results": 10, "profile": "fast",
        })
        assert resp.status_code == 200
        data = resp.json()
        by_id = {row["gene_id"]: row for row in data["fingerprints"]}
        assert set(by_id) == set(self.SCORES_A), (
            "Fingerprint rows do not match this request's candidates."
        )
        for gid, expected in self.SCORES_A.items():
            assert by_id[gid]["score"] == pytest.approx(expected), (
                "Fingerprint score came from a concurrent request's map "
                f"(gene {gid}: {by_id[gid]['score']})."
            )
            assert by_id[gid]["tier_contributions"] == {
                "fts5": pytest.approx(self.TIERS_A[gid]["fts5"])
            }


# ─────────────────────────────────────────────────────────────────────
# W1.6b remainder (issue #350): the packet path snapshots at its own
# retrieval step (build_context_packet never runs build_context)
# ─────────────────────────────────────────────────────────────────────

class TestPacketRetrievalSnapshot:
    """Issue #350: ``build_context_packet``'s top_score / score_gap /
    lexical_dense_agree inputs and its attached snapshot must come from
    the (scores, tier_contributions) pair captured atomically under the
    store lock right after ITS query_docs() call — not from the
    genome-global maps a concurrent request may republish mid-build."""

    SCORES_A = {"g-A1": 10.0, "g-A2": 2.0}
    TIERS_A = {
        "g-A1": {"fts5": 0.9, "splade": 0.8},
        "g-A2": {"fts5": 0.2},
    }
    SCORES_B = {"g-B1": 0.42}
    TIERS_B = {"g-B1": {"fts5": 0.42}}

    def _genome_publishing_a(self):
        from tests.conftest import make_gene

        from cymatix_context.genome import Genome

        genome = Genome(path=":memory:")
        genes = [
            make_gene(f"packet content for {gid}", gene_id=gid)
            for gid in self.SCORES_A
        ]

        def fake_query_docs(*, domains, entities, max_genes, read_only=False):
            # Publication seam: THIS request's retrieval publishes A.
            with genome._last_query_scores_lock:
                genome.last_query_scores = dict(self.SCORES_A)
                genome.last_tier_contributions = {
                    k: dict(v) for k, v in self.TIERS_A.items()
                }
            return list(genes)

        genome.query_docs = fake_query_docs
        return genome

    def _clobber_with_b(self, genome):
        with genome._last_query_scores_lock:
            # In-place mutation first (the mode a concurrent request's
            # refiners produce when they run with query_scores=None —
            # they mutate the live dict the pre-#350 packet reader still
            # aliased), then republication by assignment. The packet's
            # snapshot must be immune to both contamination modes.
            genome.last_query_scores.clear()
            genome.last_query_scores.update(self.SCORES_B)
            genome.last_tier_contributions.clear()
            genome.last_tier_contributions.update(self.TIERS_B)
            genome.last_query_scores = dict(self.SCORES_B)
            genome.last_tier_contributions = {
                k: dict(v) for k, v in self.TIERS_B.items()
            }

    def test_packet_know_computed_from_retrieval_step_snapshot(
        self, monkeypatch,
    ):
        """B publishes between the packet's retrieval and its know/miss
        build; the know inputs must come from A's snapshot."""
        import cymatix_context.context_packet as cp
        from cymatix_context.scoring import know_decision

        genome = self._genome_publishing_a()
        captured = {}
        real_decide = know_decision.decide_know_or_miss

        def spy(window=None, **kw):
            captured.update(
                top_score=kw.get("top_score"),
                score_gap=kw.get("score_gap"),
                lexical_dense_agree=kw.get("lexical_dense_agree"),
            )
            return real_decide(window=window, **kw)

        monkeypatch.setattr(know_decision, "decide_know_or_miss", spy)

        real_signals = cp._coordinate_signals

        def clobbering_signals(query, genes):
            # Simulate concurrent request B publishing mid-packet-build
            # (after retrieval, before the know/miss computation).
            self._clobber_with_b(genome)
            return real_signals(query, genes)

        monkeypatch.setattr(cp, "_coordinate_signals", clobbering_signals)
        try:
            cp.build_context_packet(
                "query A",
                task_type="explain",
                genome=genome,
                now_ts=10_000.0,
            )
            assert captured["top_score"] == 10.0, (
                "Packet know/miss inputs came from a concurrent request's "
                f"score map (top_score={captured.get('top_score')})."
            )
            assert captured["score_gap"] == 8.0
            # TIERS_A has lexical+dense agreeing on g-A1; B's map (fts5
            # only) has no dense signal. A contaminated tier read flips
            # this to False.
            assert captured["lexical_dense_agree"] is True, (
                "Packet lexical_dense_agree was computed from a concurrent "
                "request's tier map."
            )
        finally:
            genome.close()

    def test_packet_attaches_excluded_snapshot(self):
        """build_context_packet attaches the request-scoped snapshot to
        the packet for the server-layer PLR reader, and the fields never
        serialize into responses (exclude=True)."""
        from cymatix_context.context_packet import build_context_packet

        genome = self._genome_publishing_a()
        try:
            packet = build_context_packet(
                "query A", genome=genome, now_ts=10_000.0,
            )
            # Contaminate AFTER the build: the attached snapshot must be
            # a copy, not an alias of the genome-global maps.
            self._clobber_with_b(genome)
            assert packet.retrieval_scores == self.SCORES_A
            assert packet.tier_contributions == self.TIERS_A
            dumped = packet.model_dump()
            assert "retrieval_scores" not in dumped
            assert "tier_contributions" not in dumped
        finally:
            genome.close()


# ─────────────────────────────────────────────────────────────────────
# W1.6b remainder (issue #350): the packet-path PLR reader prefers the
# packet's snapshot
# ─────────────────────────────────────────────────────────────────────

class TestPlrSnapshot:
    """Issue #350: ``_compute_plr_confidence`` runs AFTER
    ``build_context_packet`` returns — the genome-global maps may already
    hold a concurrent request's publication, so it must prefer the
    snapshot the route threads through from ``packet.retrieval_scores`` /
    ``packet.tier_contributions`` (genome-global stays as the legacy
    fallback for callers that pass nothing)."""

    SCORES_A = {"g-A1": 10.0, "g-A2": 2.0}
    TIERS_A = {
        "g-A1": {"fts5": 0.9, "splade": 0.7},
        "g-A2": {"fts5": 0.3},
    }
    SCORES_B = {"g-B1": 0.42}
    TIERS_B = {"g-B1": {"fts5": 0.42}}

    def _env(self, monkeypatch):
        from types import SimpleNamespace

        import cymatix_context.retrieval.fusion_plr as fusion_plr

        captured = {}

        class _FakeFuser:
            meta = {"label_set": ["clean", "artifact"]}

            def query_confidence(self, tier_totals, window_features, cos_qc):
                captured["tier_totals"] = dict(tier_totals)
                return {"prob_B": 0.25, "logit": -1.1, "score_A": 0.75}

        monkeypatch.setattr(
            fusion_plr, "get_fuser", lambda _path, **_kw: _FakeFuser()
        )

        class _Codec:
            def encode(self, text):
                return [1.0, 0.0]

        def _get_doc(gid):
            captured["top_gene_id"] = gid
            return None

        genome = SimpleNamespace(
            last_query_scores=dict(self.SCORES_B),
            last_tier_contributions={
                k: dict(v) for k, v in self.TIERS_B.items()
            },
            get_doc=_get_doc,
        )
        cymatix = SimpleNamespace(genome=genome, _sema_codec=_Codec())
        config = SimpleNamespace(
            plr=SimpleNamespace(model_path="unused", high_risk_threshold=0.5),
        )
        return cymatix, config, captured

    def test_plr_prefers_packet_snapshot(self, monkeypatch):
        """The genome-global maps hold B (a concurrent publication); the
        threaded snapshot holds A — PLR features must come from A."""
        import cymatix_context.server.helpers as helpers

        cymatix, config, captured = self._env(monkeypatch)
        out = helpers._compute_plr_confidence(
            cymatix, config, "query A",
            retrieval_scores=dict(self.SCORES_A),
            tier_contributions={k: dict(v) for k, v in self.TIERS_A.items()},
        )
        assert out is not None
        assert captured["tier_totals"] == {
            "fts5": pytest.approx(1.2), "splade": pytest.approx(0.7),
        }, "PLR tier_totals came from the genome-global (contaminated) map."
        assert captured["top_gene_id"] == "g-A1", (
            "PLR top-gene lookup used the genome-global (contaminated) map."
        )

    def test_plr_falls_back_to_genome_global(self, monkeypatch):
        """Legacy direct callers that pass no snapshot keep the pre-#350
        genome-global read."""
        import cymatix_context.server.helpers as helpers

        cymatix, config, captured = self._env(monkeypatch)
        out = helpers._compute_plr_confidence(cymatix, config, "query A")
        assert out is not None
        assert captured["tier_totals"] == {"fts5": pytest.approx(0.42)}
        assert captured["top_gene_id"] == "g-B1"

"""Admission evidence must come from the stage that admitted a document."""

import sqlite3

import pytest

from cymatix_context.exceptions import PromoterMismatch
from cymatix_context.schemas import ChromatinState, Gene, PromoterTags


@pytest.fixture
def lexical_store(genome):
    genome._fusion_mode = "rrf"
    genome._bm25_prefilter_enabled = False
    genome._bm25_shortlist_enabled = True
    genome._bm25_shortlist_size = 1
    genome._dense_embedding_enabled = False
    genome._splade_enabled = False
    for gid, content in (
        ("winner", "quartz " * 20),
        ("gold", "quartz " + "padding " * 80),
        ("tail", "quartz " + "padding " * 160),
    ):
        genome.upsert_gene(Gene(
            gene_id=gid, content=content, complement="", codons=[],
            promoter=PromoterTags(domains=["unrelated"]),
        ))
    return genome


def _query(store):
    return store.query_docs(["quartz"], [], max_genes=2, read_only=True)


def test_identical_final_maps_do_not_imply_identical_fts_admission(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    store = lexical_store
    store._fts5_candidate_depth = 1
    with capture_stages({"gold"}) as shallow:
        shallow_docs = _query(store)
    shallow_scores = dict(store.last_query_scores)
    store._fts5_candidate_depth = 3
    with capture_stages({"gold"}) as deep:
        deep_docs = _query(store)
    assert store.last_query_scores == shallow_scores
    assert [d.gene_id for d in shallow_docs] == [d.gene_id for d in deep_docs]
    shallow_stages = shallow.report()["retrievals"][0]["stages"]
    stages = deep.report()["retrievals"][0]["stages"]
    assert shallow_stages["fts_raw"]["count"] == 1
    assert shallow_stages["fts_raw"]["gold_ids"] == []
    assert stages["fts_raw"]["count"] == 3
    assert stages["fts_raw"]["gold_ids"] == ["gold"]
    assert stages["pre_shortlist"]["gold_ids"] == ["gold"]
    assert stages["post_shortlist"]["count"] == 1
    assert stages["post_shortlist"]["gold_ids"] == []
    assert stages["post_shortlist"]["filter_status"] == "applied"
    assert stages["final_scoring"]["gold_ids"] == []


def test_disabled_capture_preserves_results_scores_and_has_no_observations(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    store = lexical_store
    with capture_stages({"gold"}, enabled=False) as off:
        off_docs = _query(store)
    off_scores = dict(store.last_query_scores)
    with capture_stages({"gold"}) as on:
        on_docs = _query(store)
    assert [d.gene_id for d in on_docs] == [d.gene_id for d in off_docs]
    assert store.last_query_scores == off_scores
    assert off.report()["status"] == "not_captured"
    assert off.report()["retrievals"] == []
    assert on.report()["status"] == "complete"


def test_failure_and_unexecuted_capture_cannot_reuse_previous_absence(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    with capture_stages({"missing"}) as previous:
        _query(lexical_store)
    assert previous.report()["retrievals"][0]["stages"]["fts_raw"]["gold_ids"] == []
    with pytest.raises(PromoterMismatch):
        with capture_stages({"gold"}) as failed:
            lexical_store.query_docs([], [], read_only=True)
    report = failed.report()
    assert report["status"] == "failed"
    assert report["retrievals"][0]["status"] == "failed"
    assert report["retrievals"][0]["stages"]["fts_raw"]["gold_ids"] is None
    assert report["retrievals"][0]["stages"]["fts_raw"]["count"] is None
    with capture_stages({"gold"}) as unused:
        pass
    assert unused.report()["status"] == "not_captured"


def test_swallowed_fts_error_is_failed_not_an_empty_fetch(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    # Deliberately leave availability true: SQLite raises at the actual lane.
    lexical_store.conn.execute("DROP TABLE genes_fts")
    with capture_stages({"gold"}) as capture:
        with pytest.raises(PromoterMismatch):
            _query(lexical_store)
    stages = capture.report()["retrievals"][0]["stages"]
    assert stages["fts_raw"]["status"] == "failed"
    assert stages["fts_raw"]["count"] is None
    assert stages["fts_raw"]["gold_ids"] is None
    assert capture.report()["status"] == "failed"


def test_separate_calls_and_nested_scopes_do_not_merge_admission(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    with capture_stages({"gold"}) as outer:
        lexical_store._fts5_candidate_depth = 1
        _query(lexical_store)
        with capture_stages({"winner"}) as inner:
            _query(lexical_store)
        lexical_store._fts5_candidate_depth = 3
        _query(lexical_store)
    calls = outer.report()["retrievals"]
    assert len(calls) == 2
    assert calls[0]["stages"]["fts_raw"]["gold_ids"] == []
    assert calls[1]["stages"]["fts_raw"]["gold_ids"] == ["gold"]
    assert inner.report()["retrievals"][0]["stages"]["fts_raw"]["gold_ids"] == ["winner"]


def test_return_expansion_and_real_blend_reintroduce_shortlist_loss(lexical_store, monkeypatch):
    from cymatix_context.context_manager import CymatixContextManager
    from cymatix_context.retrieval.measurement import capture_stages
    from cymatix_context.scoring import cymatics
    from tests.conftest import MockCompressorBackend, make_cymatix_config

    winner = lexical_store.get_doc("winner")
    winner.epigenetics.co_activated_with = ["gold"]
    lexical_store.upsert_gene(winner)
    lexical_store._fts5_candidate_depth = 3
    # Keep the real blend and graph expansion; control only its spectral input.
    monkeypatch.setattr(cymatics, "query_spectrum", lambda *a, **kw: None)
    monkeypatch.setattr(cymatics, "build_weight_vector", lambda *a, **kw: None)
    monkeypatch.setattr(cymatics, "cached_doc_spectrum", lambda doc, **kw: doc.gene_id)
    monkeypatch.setattr(cymatics, "flux_score_dispatch", lambda q, gid, w, metric: 1.0 if gid == "gold" else 0.0)
    manager = CymatixContextManager(make_cymatix_config())
    original_store = manager.genome
    manager.genome = lexical_store
    manager.ribosome.backend = MockCompressorBackend()
    manager._use_cymatics = True
    manager._blend_mode = "legacy"
    try:
        with capture_stages({"gold"}, enabled=False):
            off = manager.build_context("quartz", max_genes=2, read_only=True, ignore_delivered=True)
        with capture_stages({"gold"}) as capture:
            on = manager.build_context("quartz", max_genes=2, read_only=True, ignore_delivered=True)
        report = capture.report()
        stages = report["retrievals"][0]["stages"]
        assert stages["pre_shortlist"]["gold_ids"] == ["gold"]
        assert stages["post_shortlist"]["gold_ids"] == []
        assert stages["final_scoring"]["gold_ids"] == []
        assert stages["retrieval_returned"]["gold_ids"] == ["gold"]
        assert report["stages"]["post_blend_scores"]["gold_ids"] == ["gold"]
        assert report["stages"]["post_blend_candidates"]["gold_ids"] == ["gold"]
        assert on.retrieval_scores["gold"] > on.retrieval_scores["winner"]
        assert on.retrieval_scores == off.retrieval_scores
        assert on.expressed_context == off.expressed_context
        assert on.expressed_gene_ids == off.expressed_gene_ids
        assert on.model_dump().keys() == off.model_dump().keys()
        assert on.metadata.keys() == off.metadata.keys()
        # Report is an immutable snapshot, even if a later consumer edits it.
        report["stages"]["post_blend_scores"]["gold_ids"].clear()
        assert capture.report()["stages"]["post_blend_scores"]["gold_ids"] == ["gold"]
    finally:
        manager.genome = original_store
        manager.close()


def test_ann_cannot_claim_complete_lexical_admission_measurement(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    with capture_stages({"gold"}) as capture:
        lexical_store.query_docs_ann("quartz", domains=["quartz"], entities=[], max_genes=2)
    assert capture.report()["status"] == "not_captured"
    assert capture.report()["unsupported"]


def test_additive_capture_uses_eligible_scoring_map_not_legacy_publication(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    lexical_store._fusion_mode = "additive"
    with capture_stages({"gold"}) as capture:
        _query(lexical_store)
    stages = capture.report()["retrievals"][0]["stages"]
    assert stages["pre_shortlist"]["gold_ids"] == ["gold"]
    assert stages["final_scoring"]["gold_ids"] == []
    # Preserve the old additive publication behavior while naming it honestly.
    assert "gold" in lexical_store.last_query_scores


def test_prefiltered_raw_fetch_states_its_scope(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    lexical_store._bm25_prefilter_enabled = True
    lexical_store._bm25_prefilter_size = 1
    with capture_stages({"gold"}) as capture:
        _query(lexical_store)
    stages = capture.report()["retrievals"][0]["stages"]
    assert stages["fts_raw"]["prefilter_applied"] is True
    assert stages["fts_raw"]["count"] == 1
    assert stages["fts_raw"]["gold_ids"] == []
    assert stages["post_shortlist"]["filter_status"] == "not_applied"


@pytest.mark.parametrize("exclusion", ["lifecycle", "party"])
def test_fts_eligible_measures_membership_after_filters(lexical_store, exclusion):
    from cymatix_context.retrieval.measurement import capture_stages

    store = lexical_store
    party_id = None
    if exclusion == "lifecycle":
        gold = store.get_doc("gold")
        gold.chromatin = ChromatinState.HETEROCHROMATIN
        store.upsert_gene(gold)
    else:
        store.conn.execute(
            "INSERT INTO parties (party_id, display_name, trust_domain, created_at) "
            "VALUES ('bob', 'Bob', 'local', 0)"
        )
        store.conn.execute(
            "INSERT INTO gene_attribution (gene_id, party_id, authored_at) "
            "VALUES ('gold', 'bob', 0)"
        )
        store.conn.commit()
        party_id = "alice"

    with capture_stages({"gold"}) as capture:
        docs = store.query_docs(
            ["quartz"], [], max_genes=2, party_id=party_id, read_only=True,
        )
    report = capture.report()
    stages = report["retrievals"][0]["stages"]
    assert report["status"] == "complete"
    assert stages["fts_raw"]["count"] == 3
    assert stages["fts_raw"]["gold_ids"] == ["gold"]
    assert stages["fts_eligible"] == {
        "status": "captured", "count": 2, "gold_ids": [],
    }
    assert stages["pre_shortlist"]["gold_ids"] == []
    assert "gold" not in {doc.gene_id for doc in docs}


def test_fts_eligibility_failure_preserves_raw_fetch_evidence(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    # Let the real raw FTS query finish, then have SQLite reject the lifecycle
    # lookup. This distinguishes a failed eligible set from a failed raw fetch.
    state = {"raw_started": False, "denied": False}

    def trace(sql):
        if sql.lstrip().startswith("SELECT gene_id, rank") and "FROM genes_fts" in sql:
            state["raw_started"] = True

    def authorize(action, table, column, database, source):
        if (
            state["raw_started"] and not state["denied"]
            and action == sqlite3.SQLITE_READ
            and table == "genes" and column == "chromatin"
        ):
            state["denied"] = True
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    lexical_store.conn.set_trace_callback(trace)
    lexical_store.conn.set_authorizer(authorize)
    try:
        with capture_stages({"gold"}) as capture:
            with pytest.raises(PromoterMismatch):
                _query(lexical_store)
    finally:
        lexical_store.conn.set_trace_callback(None)
        lexical_store.conn.set_authorizer(None)

    report = capture.report()
    stages = report["retrievals"][0]["stages"]
    assert state["denied"] is True
    assert report["status"] == "failed"
    assert stages["fts_raw"]["status"] == "captured"
    assert stages["fts_raw"]["count"] == 3
    assert stages["fts_raw"]["gold_ids"] == ["gold"]
    assert stages["fts_eligible"] == {
        "status": "failed", "count": None, "gold_ids": None, "reason": "DatabaseError",
    }


def test_empty_shortlist_retains_tag_candidates_and_records_empty_fts(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    # Prefix tags match "unrelated", while the exact FTS token "unrel" is absent.
    with capture_stages({"gold"}) as capture:
        docs = lexical_store.query_docs(["unrel"], [], max_genes=2, read_only=True)
    report = capture.report()
    stages = report["retrievals"][0]["stages"]
    assert report["status"] == "complete"
    for name in ("fts_raw", "fts_eligible"):
        assert stages[name]["status"] == "captured"
        assert stages[name]["count"] == 0
        assert stages[name]["gold_ids"] == []
    assert stages["pre_shortlist"]["count"] == 3
    assert stages["pre_shortlist"]["gold_ids"] == ["gold"]
    assert stages["post_shortlist"] == {
        "status": "captured", "count": 3, "gold_ids": ["gold"],
        "filter_status": "empty_fallback",
    }
    assert "gold" in {doc.gene_id for doc in docs}


def test_shortlist_sql_failure_invalidates_capture_even_when_retrieval_succeeds(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    # A Python int outside SQLite's signed 64-bit range fails at the real
    # shortlist LIMIT binding; earlier FTS retrieval keeps its normal depth.
    lexical_store._bm25_shortlist_size = 2**63
    with capture_stages({"gold"}) as capture:
        docs = _query(lexical_store)
    report = capture.report()
    call = report["retrievals"][0]
    stages = call["stages"]
    assert report["status"] == "failed"
    assert call["status"] == "complete"
    assert stages["fts_raw"]["status"] == "captured"
    assert stages["fts_eligible"]["gold_ids"] == ["gold"]
    assert stages["pre_shortlist"]["count"] == 3
    assert stages["post_shortlist"] == {
        "status": "captured", "count": 3, "gold_ids": ["gold"],
        "filter_status": "failed",
    }
    assert stages["retrieval_returned"]["gold_ids"] == ["gold"]
    assert "gold" in {doc.gene_id for doc in docs}


def test_failure_after_capture_retains_evidence_without_claiming_query_success(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages

    with pytest.raises(RuntimeError, match="assembly failed"):
        with capture_stages({"missing"}) as capture:
            _query(lexical_store)
            raise RuntimeError("assembly failed")
    report = capture.report()
    assert report["retrievals"][0]["stages"]["fts_raw"]["gold_ids"] == []
    assert report["status"] == "failed"
    assert report["error"] == "RuntimeError: assembly failed"


def test_disabled_nested_scope_cannot_write_into_enclosing_retrieval(lexical_store):
    from cymatix_context.retrieval.measurement import capture_stages, trace_retrieval

    @trace_retrieval
    def run_without_measurement():
        with capture_stages({"gold"}, enabled=False) as disabled:
            _query(lexical_store)
        return disabled

    with capture_stages({"gold"}) as outer:
        disabled = run_without_measurement()
    assert disabled.report()["retrievals"] == []
    stages = outer.report()["retrievals"][0]["stages"]
    assert stages["fts_raw"]["status"] == "not_executed"
    assert stages["fts_raw"]["gold_ids"] is None

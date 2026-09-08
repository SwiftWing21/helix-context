"""Stage-aware pool-depth decisions, without a corpus or encoder."""

import importlib.util
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_probe():
    path = Path(__file__).resolve().parents[1] / "benchmarks/dogfood/erb/probe_pool_depth.py"
    spec = importlib.util.spec_from_file_location("_pool_depth_provenance", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe(monkeypatch):
    # The command-line script adjusts stdout encoding; pytest's capture does not.
    monkeypatch.setattr(sys.stdout, "reconfigure", lambda **kwargs: None, raising=False)
    return _load_probe()


def _record(*gold_by_call, status="complete", stage_status="captured"):
    return {
        "needle": "n0", "status": status, "rank_of_first_gold": None,
        "stage_provenance": {
            "version": 1, "status": status, "error": None,
            "retrievals": [{
                "status": "complete", "error": None,
                "stages": {"post_shortlist": {
                    "status": stage_status,
                    "count": 1 if stage_status == "captured" else None,
                    "gold_ids": list(gold) if stage_status == "captured" else None,
                }},
            } for gold in gold_by_call],
            "stages": {},
        },
    }


def test_admission_uses_post_shortlist_membership_not_final_score_map(probe):
    present = _record(["gold"])
    absent = _record([])
    absent["rank_of_first_gold"] = 1
    assert probe.stage_presence(present, "post_shortlist") == "present"
    assert probe.stage_presence(absent, "post_shortlist") == "absent"


@pytest.mark.parametrize("record, expected", [
    ({"rank_of_first_gold": None}, "not_captured"),
    ({"rank_of_first_gold": 1, "error": "query failed"}, "failed"),
    (_record([], status="failed"), "failed"),
    (_record([], stage_status="not_executed"), "not_executed"),
    (_record(), "not_executed"),
])
def test_missing_or_failed_capture_is_never_absence(probe, record, expected):
    assert probe.stage_presence(record, "post_shortlist") == expected


def test_multi_retrieval_is_not_collapsed_into_one_admission_pool(probe):
    rec = _record([], ["gold"])
    assert probe.stage_presence(rec, "post_shortlist") == "not_captured"
    rec["stage_provenance"]["retrievals"][1]["stages"] = {}
    assert probe.stage_presence(rec, "post_shortlist") == "not_captured"


def test_admission_verdict_requires_the_full_measured_cohort(probe):
    result = probe.admission_summary([_record([]), _record([])], expected_n=2, threshold=1)
    assert result["verdict"] == "KILL"
    assert result["absent"] == 2
    assert probe.admission_summary([_record(["g"])], expected_n=2, threshold=1)["verdict"] == "INCONCLUSIVE"
    result = probe.admission_summary([_record(["g"]), _record([], status="failed")],
                                     expected_n=2, threshold=1)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["absent"] == 0
    assert result["failed"] == 1
    assert probe.admission_summary([_record(["g"]), _record([])],
                                    expected_n=2, threshold=1)["verdict"] == "PASS"


def test_legacy_checkpoint_cannot_satisfy_requested_measurement(probe, tmp_path):
    path = tmp_path / "deep1000.jsonl"
    path.write_text(json.dumps({"needle": "n0", "rank_of_first_gold": None}) + "\n")
    with pytest.raises(ValueError, match="stage provenance"):
        probe.require_stage_provenance([json.loads(path.read_text())], path, ["n0"])
    # Captured failures are valid receipts of failure, not reasons to rerun a query.
    probe.require_stage_provenance([_record([], status="failed")], path, ["n0"])


def test_explicit_unmeasured_checkpoint_resumes_without_becoming_absent(probe):
    rec = _record([], status="not_captured")
    probe.require_stage_provenance([rec], Path("checkpoint.jsonl"), ["n0"])
    assert probe.stage_presence(rec, "post_shortlist") == "not_captured"


def test_histogram_separates_query_failures_from_measured_score_map_absence(probe):
    result = probe.histogram([None, None, None, 10],
                             statuses=["failed", "not_captured", "complete", "complete"])
    assert result["absent_at_1000"] == 1
    assert result["failed"] == 1
    assert result["not_captured"] == 1
    assert result["le_48"] == 1


@pytest.fixture
def manager_stub(monkeypatch):
    from cymatix_context import config, context_manager
    from cymatix_context.retrieval.measurement import (
        current_capture, current_retrieval, trace_retrieval,
    )

    @trace_retrieval
    def retrieve():
        capture = current_retrieval()
        if capture is not None:
            for stage in ("fts_raw", "pre_shortlist", "post_shortlist",
                          "final_scoring", "retrieval_returned"):
                capture.record(stage, ["gold"])

    class Manager:
        def __init__(self, cfg):
            self.config = cfg
            self.genome = SimpleNamespace(
                last_query_scores={}, last_signal_timings={},
                last_tier_contributions={}, synonym_map={},
                _expand_terms=lambda terms: list(terms), close=lambda: None,
            )

        def build_context(self, query, **kwargs):
            if query == "fail":
                raise ValueError("query failed before retrieval")
            if query != "empty":
                retrieve()
                self.genome.last_query_scores = {"gold": 1.0}
                self.genome.last_ranked_ids = ["gold"]
                capture = current_capture()
                if capture is not None:
                    capture.record("post_blend_scores", ["gold"])
                    capture.record("post_blend_candidates", ["gold"])
            return SimpleNamespace(metadata={}, expressed_gene_ids=["gold"])

        def close(self):
            pass

    monkeypatch.setattr(config, "load_config", lambda *args: config.CymatixConfig())
    monkeypatch.setattr(context_manager, "CymatixContextManager", Manager)
    return Manager


@pytest.mark.parametrize("enabled", [False, True])
def test_ladder_per_query_capture_excludes_warmup_and_previous_queries(manager_stub, enabled):
    path = Path(__file__).resolve().parents[1] / "benchmarks/dogfood/erb/ablation_ladder.py"
    spec = importlib.util.spec_from_file_location("_ladder_provenance", path)
    ladder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ladder
    spec.loader.exec_module(ladder)
    needles = [{"name": q, "query": q} for q in ("ok", "fail", "empty")]
    result = ladder.run_arm(
        ladder.Arm(name="test", knobs={}, gate="test", validation=ladder.VALIDATION_CONTROL),
        genome_path="unused.db", config_path=None, needles=needles,
        gold_by_needle={q["name"]: {"gold"} for q in needles}, k=12,
        per_query=enabled,
    )
    if not enabled:
        assert "per_query" not in result
        return
    ok, failed, empty = result["per_query"]
    assert len(ok["stage_provenance"]["retrievals"]) == 1
    assert ok["stage_provenance"]["stages"]["post_blend_scores"]["gold_ids"] == ["gold"]
    assert failed["stage_provenance"]["status"] == "failed"
    assert failed["stage_provenance"]["retrievals"] == []
    assert empty["stage_provenance"]["retrievals"] == []


@pytest.mark.parametrize("enabled", [False, True])
def test_probe_query_records_distinguish_failure_and_gate_capture(probe, manager_stub, enabled):
    needles = {q: {"query": q} for q in ("ok", "fail", "empty")}
    result = probe.run_arm("test", {}, list(needles), needles,
                           {q: ["gold"] for q in needles}, False,
                           stage_provenance=enabled)
    ok, failed, empty = result["per_query"]
    assert failed["status"] == "failed"
    assert "ValueError" in failed["error"]
    assert probe.stage_presence(failed, "post_shortlist") == "failed"
    assert ok["status"] == "complete"
    if enabled:
        assert len(ok["stage_provenance"]["retrievals"]) == 1
        assert failed["stage_provenance"]["status"] == "failed"
        assert failed["stage_provenance"]["retrievals"] == []
        assert empty["stage_provenance"]["retrievals"] == []
    else:
        assert all("stage_provenance" not in r for r in result["per_query"])


def test_probe_resume_rejects_old_needle_cache_before_manager_open(probe, monkeypatch, tmp_path):
    from cymatix_context import config, context_manager

    monkeypatch.setattr(config, "load_config", lambda *args: config.CymatixConfig())

    def manager_must_not_open(cfg):
        pytest.fail("legacy cache must be rejected before opening a store")

    monkeypatch.setattr(context_manager, "CymatixContextManager", manager_must_not_open)
    checkpoint = tmp_path / "test.jsonl"
    checkpoint.write_text(json.dumps({"needle": "ok", "rank_of_first_gold": None}) + "\n")
    with pytest.raises(ValueError, match="stage provenance"):
        probe.run_arm("test", {}, ["ok"], {"ok": {"query": "ok"}},
                      {"ok": ["gold"]}, False, ckpt_dir=tmp_path,
                      stage_provenance=True)


def test_bm25_proxy_marks_empty_success_missing_terms_and_fts_failure(probe, monkeypatch, tmp_path):
    path = tmp_path / "proxy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE VIRTUAL TABLE genes_fts USING fts5(gene_id UNINDEXED, content)")
        conn.execute("INSERT INTO genes_fts VALUES ('gold', 'apples')")
    monkeypatch.setattr(probe, "BED", str(path))
    result = probe.bm25_proxy({"hit": ["apples"], "miss": ["pears"], "missing": [],
                               "failed": ['broken"syntax']}, {"hit": ["gold"]}, 100)
    records = result["per_needle"]
    assert records["hit"]["status"] == "complete"
    assert records["miss"]["status"] == "complete"
    assert records["miss"]["pool"] == 0
    assert records["missing"]["status"] == "not_captured"
    assert records["failed"]["status"] == "failed"


def test_bm25_attribution_never_calls_failed_proxy_absent(probe):
    result = probe.bm25_window_attribution([
        {"bm25_proxy_rank": None, "bm25_proxy_status": "failed"},
        {"bm25_proxy_rank": None, "bm25_proxy_status": "not_captured"},
        {"bm25_proxy_rank": None, "bm25_proxy_status": "complete"},
        {"bm25_proxy_rank": 60, "bm25_proxy_status": "complete"},
    ])
    assert result["gold_bm25_absent_20000"] == 1
    assert result["failed"] == 1
    assert result["not_captured"] == 1
    assert result["gold_bm25_51_1000"] == 1


def _main_inputs(probe, monkeypatch, tmp_path):
    bed = tmp_path / "unused.db"
    bed.touch()
    monkeypatch.setattr(probe, "BED", str(bed))
    monkeypatch.setattr(probe, "load_inputs", lambda: (
        {"n0": {"query": "query", "question_type": "semantic"}},
        {"n0": ["gold"]}, {"n0": "pool_absent"}, ["n0"],
    ))
    monkeypatch.setattr(probe, "pick_controls", lambda *args: [])
    monkeypatch.setattr(probe, "source_map", lambda *args: ({}, 0))
    monkeypatch.setattr(probe, "bm25_proxy", lambda *args: {
        "per_needle": {"n0": {"bm25_rank": None, "pool": None, "status": "failed"}},
        "wall_ms_p50": None,
    })


def test_cli_emits_provenance_and_inconclusive_for_unmeasured_stages(probe, monkeypatch, tmp_path):
    _main_inputs(probe, monkeypatch, tmp_path)

    def arm(arm_name, knobs, targets, needles, gold, capture, **kwargs):
        assert kwargs["stage_provenance"] is True
        rec = _record([], status="failed")
        rec.update(map_size=0, gold_ranks=[], delivered_gold=0,
                   delivered_gold_rank=None, delivered_count=0, wall_ms=1, query_terms=None)
        return {"arm": arm_name, "wall_ms_p50": 1, "wall_ms_p95": 1,
                "map_size_median": 0, "per_query": [rec]}

    monkeypatch.setattr(probe, "run_arm", arm)
    out = tmp_path / "receipt.json"
    assert probe.main(["--stage-provenance", "--out", str(out),
                       "--capture", str(tmp_path / "capture.db")]) == 0
    receipt = json.loads(out.read_text())
    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["admission_measurement"]["failed"] == 1
    assert receipt["by_bucket"]["pool_absent"]["deep_hist"]["absent_at_1000"] == 0
    assert receipt["by_bucket"]["pool_absent"]["bm25_window_attribution"]["gold_bm25_absent_20000"] == 0
    assert receipt["rows"][0]["stage_provenance"]["deep1000"]["status"] == "failed"


def test_cli_rejects_old_arm_cache_in_stage_mode(probe, monkeypatch, tmp_path):
    _main_inputs(probe, monkeypatch, tmp_path)
    key = hashlib.sha256(b"n0").hexdigest()[:12]
    cache = tmp_path / f"arm_deep1000_{key}.json"
    cache.write_text(json.dumps({"per_query": [{"needle": "n0"}]}))
    with pytest.raises(ValueError, match="stage provenance"):
        probe.main(["--stage-provenance", "--cache", str(tmp_path),
                    "--capture", str(tmp_path / "capture.db")])

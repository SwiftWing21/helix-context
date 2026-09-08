"""Exercise the ERB CLI through real config, SQLite retrieval, and JSON export."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cymatix_context.knowledge_store import KnowledgeStore
from cymatix_context.schemas import Document, DocumentTags


ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "benchmarks/dogfood/erb/ablation_ladder.py"


@pytest.mark.parametrize("depth,shortlist,per_query", [
    (1, 1, True),
    (3, 1, True),
    (3, 3, True),
    (3, 3, False),
])
def test_ladder_cli_captures_real_configured_boundaries(tmp_path, depth, shortlist, per_query):
    """The child has no pytest fixtures or patched manager/retrieval methods.

    Non-default depth and shortlist values must reach the real store. Gold
    appears in the deeper FTS fetch even when the shortlist removes it; final
    score-map ranks cannot substitute for these observed stage memberships.
    """
    bed = tmp_path / "lexical.db"
    store = KnowledgeStore(path=str(bed))
    try:
        for doc_id, content in (
            ("winner", "quartz " * 20),
            ("gold", "quartz " + "padding " * 80),
            ("tail", "quartz " + "padding " * 160),
        ):
            store.upsert_doc(Document(
                gene_id=doc_id, content=content, complement="", codons=[],
                promoter=DocumentTags(domains=["unrelated"]),
            ), apply_gate=False)
    finally:
        store.close()

    config = tmp_path / "lexical.toml"
    config.write_text(f"""
[ribosome]
enabled = false
query_expansion_enabled = false
query_decomposition_enabled = false
[ingestion]
sema_embed_on_ingest = false
dense_embed_on_ingest = false
splade_enabled = false
[retrieval]
dense_embedding_enabled = false
rerank_enabled = false
fusion_mode = "rrf"
fts5_candidate_depth = {depth}
bm25_prefilter_enabled = false
bm25_shortlist_enabled = true
bm25_shortlist_size = {shortlist}
[cymatics]
enabled = false
[classifier]
enabled = false
[context]
cold_tier_enabled = false
[budget]
abstain_enabled = false
""", encoding="utf-8")
    needles = tmp_path / "needles.json"
    needles.write_text(json.dumps({"needles": [
        {"name": "gold_query", "query": "quartz"},
        {"name": "winner_query", "query": "quartz"},
        {"name": "missing_query", "query": "unobtainium"},
        {"name": "failed_query", "query": None},
    ]}), encoding="utf-8")
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({
        "gold_query": ["gold"], "winner_query": ["winner"],
        "missing_query": ["gold"], "failed_query": ["gold"],
    }), encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    # Remove parent test/service overrides. Start outside the checkout so the
    # CLI's own import prologue, rather than pytest's sys.path, selects its code.
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith("CYMATIX_") and key.upper() != "PYTHONPATH"}
    env.update(CYMATIX_DISABLE_LEARN="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, str(LADDER), "--genome", str(bed),
               "--config", str(config), "--resolved", str(needles),
               "--gold", str(gold), "--arms", "baseline", "--limit", "0",
               "--k", "2", "--out", str(receipt_path)]
    if per_query:
        command.append("--per-query")
    child = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True,
        timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert child.returncode == 0, child.stdout + "\n" + child.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert Path(receipt["bed"]) == bed
    assert Path(receipt["config"]) == config
    assert receipt["needle_count"] == 4
    arm, = receipt["arms"]
    assert len(arm["errors"]) == 1
    assert arm["errors"][0].startswith("failed_query: ")
    assert arm["warmup_ran"] is True
    assert arm["arm_valid"] is True
    if not per_query:
        assert "per_query" not in receipt
        assert "per_query" not in arm
        return

    records = arm["per_query"]
    assert [record["needle"] for record in records] == [
        "gold_query", "winner_query", "missing_query", "failed_query",
    ]
    # With at most three candidates the real tier logic selects BROAD. The
    # no-match path never reaches tier selection; a raised request has no
    # returned metadata. Neither may inherit a preceding request's tier.
    assert [record["budget_tier"] for record in records] == ["broad", "broad", None, None]
    assert records[3]["error"]
    first, second, missing, failed = [record["stage_provenance"] for record in records]
    for report in (first, second):
        assert report["status"] == "complete"
        assert report["unsupported"] == []
        # Exactly one call excludes the warmup and prior timed requests.
        assert len(report["retrievals"]) == 1
        assert report["retrievals"][0]["status"] == "complete"

    stages = first["retrievals"][0]["stages"]
    assert stages["fts_raw"]["count"] == depth
    assert stages["fts_raw"]["gold_ids"] == (["gold"] if depth == 3 else [])
    assert stages["pre_shortlist"]["gold_ids"] == (["gold"] if depth == 3 else [])
    assert stages["post_shortlist"]["count"] == shortlist
    assert stages["post_shortlist"]["filter_status"] == "applied"
    retained = ["gold"] if shortlist == 3 else []
    assert stages["post_shortlist"]["gold_ids"] == retained
    assert stages["final_scoring"]["gold_ids"] == retained
    assert first["stages"]["post_blend_scores"]["gold_ids"] == retained
    assert second["retrievals"][0]["stages"]["post_shortlist"]["gold_ids"] == ["winner"]
    # The manager catches an empty retrieval's PromoterMismatch and returns a
    # no-match window. Export must retain that failed call and its partial
    # FTS evidence, without borrowing the preceding query's shortlist/blend.
    assert missing["status"] == "failed"
    assert len(missing["retrievals"]) == 1
    assert "PromoterMismatch" in missing["retrievals"][0]["error"]
    assert missing["retrievals"][0]["stages"]["fts_raw"]["count"] == 0
    assert missing["retrievals"][0]["stages"]["post_shortlist"]["status"] == "not_executed"
    assert missing["retrievals"][0]["stages"]["post_shortlist"]["gold_ids"] is None
    assert missing["stages"]["post_blend_scores"]["gold_ids"] is None
    assert failed["status"] == "failed"
    assert failed["retrievals"] == []
    assert failed["error"] == records[3]["error"]

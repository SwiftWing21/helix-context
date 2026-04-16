"""Tests for the SNOW LLM cascade consumer."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure benchmarks package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.snow.cascade import llm_cascade


class MockModel:
    """Fake model that returns pre-programmed responses."""

    def __init__(self, responses):
        self.responses = responses
        self.call_idx = 0
        self.calls = []

    def chat(self, messages):
        resp = self.responses[min(self.call_idx, len(self.responses) - 1)]
        self.call_idx += 1
        self.calls.append(messages)
        return {"message": {"content": resp},
                "eval_count": 10, "prompt_eval_count": 50}


def _make_fp(gene_id="abc123def456full", **kw):
    base = {"gene_id": gene_id, "source": "test.md", "score": 1.0,
            "tiers": {}, "domains": [], "entities": []}
    base.update(kw)
    return base


def test_cascade_answers_at_t0():
    model = MockModel(["ANSWER: 11437"])
    fps = [_make_fp()]
    result = llm_cascade("What port?", fps, model, {})
    assert result["tier"] == 0
    assert result["hops"] == 0
    assert result["answer"] == "11437"
    assert result["miss"] is False
    assert len(model.calls) == 1  # only triage call


def test_cascade_reads_then_answers():
    model = MockModel(["READ: abc123def456", "ANSWER: Decimal"])
    fps = [_make_fp()]
    gene_fields = {"abc123def456full": {"key_values": "port=11437 type=Decimal"}}
    result = llm_cascade("What type?", fps, model, gene_fields)
    assert result["tier"] == 1
    assert result["hops"] == 1
    assert result["answer"] == "Decimal"
    assert len(model.calls) == 2


def test_cascade_escalates_through_tiers():
    model = MockModel(["READ: abc", "ESCALATE", "ESCALATE", "ANSWER: found"])
    fps = [_make_fp()]
    gene_fields = {"abc123def456full": {
        "key_values": "k=v",
        "complement": "some complement data",
        "content": "full content here",
    }}
    result = llm_cascade("Find it", fps, model, gene_fields)
    assert result["tier"] == 3
    assert result["hops"] == 3
    assert result["answer"] == "found"
    assert len(model.calls) == 4  # triage + 3 tier reads


def test_cascade_returns_miss():
    model = MockModel(["MISS"])
    fps = [_make_fp()]
    result = llm_cascade("Unknown?", fps, model, {})
    assert result["tier"] == -1
    assert result["miss"] is True
    assert result["answer"] is None
    assert len(model.calls) == 1


def test_cascade_handles_think_tags():
    model = MockModel(["<think>reasoning here</think>\nANSWER: 42"])
    fps = [_make_fp()]
    result = llm_cascade("What is the answer?", fps, model, {})
    assert result["tier"] == 0
    assert result["answer"] == "42"
    assert result["miss"] is False

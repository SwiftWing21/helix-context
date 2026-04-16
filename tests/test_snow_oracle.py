"""Tests for the SNOW oracle consumer — string matching per data tier."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure benchmarks package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.snow.oracle import oracle_cascade


def _make_fp(**kw):
    """Build a fingerprint dict with defaults."""
    return {
        "entities": kw.get("entities", []),
        "key_values": kw.get("key_values", "{}"),
        "complement": kw.get("complement", ""),
        "content": kw.get("content", ""),
    }


# ── T0: entities ────────────────────────────────────────────────────

def test_oracle_finds_answer_in_entities():
    fps = {"g1": _make_fp(entities=["port", "11437", "helix"])}
    result = oracle_cascade(
        expected_answer="11437",
        accept=["11437"],
        gene_ids=["g1"],
        fingerprints=fps,
        neighbors={},
    )
    assert result["tier"] == 0
    assert result["gene_id"] == "g1"
    assert result["tokens"] > 0


# ── T1: key_values ──────────────────────────────────────────────────

def test_oracle_finds_answer_in_key_values():
    fps = {"g1": _make_fp(
        entities=["port", "server"],
        key_values='{"port": "11437"}',
    )}
    result = oracle_cascade(
        expected_answer="11437",
        accept=["11437"],
        gene_ids=["g1"],
        fingerprints=fps,
        neighbors={},
    )
    assert result["tier"] == 1
    assert result["gene_id"] == "g1"
    assert result["tokens"] > 0


# ── T2: complement ──────────────────────────────────────────────────

def test_oracle_finds_answer_in_complement():
    fps = {"g1": _make_fp(
        complement="Use Decimal type for monetary values",
    )}
    result = oracle_cascade(
        expected_answer="Decimal",
        accept=["decimal", "Decimal"],
        gene_ids=["g1"],
        fingerprints=fps,
        neighbors={},
    )
    assert result["tier"] == 2
    assert result["gene_id"] == "g1"
    assert result["tokens"] > 0


# ── T3: content ─────────────────────────────────────────────────────

def test_oracle_finds_answer_in_content():
    fps = {"g1": _make_fp(
        content="creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)",
    )}
    result = oracle_cascade(
        expected_answer="CREATE_NO_WINDOW",
        accept=["CREATE_NO_WINDOW"],
        gene_ids=["g1"],
        fingerprints=fps,
        neighbors={},
    )
    assert result["tier"] == 3
    assert result["gene_id"] == "g1"
    assert result["tokens"] > 0


# ── T4: neighbor walk ───────────────────────────────────────────────

def test_oracle_finds_answer_in_neighbor():
    fps = {
        "g1": _make_fp(),  # empty — forces walk
        "nb1": _make_fp(content='timeout is set to 30 seconds'),
    }
    result = oracle_cascade(
        expected_answer="30",
        accept=["30"],
        gene_ids=["g1"],
        fingerprints=fps,
        neighbors={"g1": [("nb1", 0.9)]},
    )
    assert result["tier"] == 4
    assert result["gene_id"] == "nb1"
    assert result["tokens"] > 0


# ── MISS ────────────────────────────────────────────────────────────

def test_oracle_returns_miss():
    fps = {"g1": _make_fp(
        entities=["alpha"],
        key_values='{"k": "v"}',
        complement="some text",
        content="more text",
    )}
    result = oracle_cascade(
        expected_answer="nonexistent_value",
        accept=["nonexistent_value"],
        gene_ids=["g1"],
        fingerprints=fps,
        neighbors={},
    )
    assert result["tier"] == -1
    assert result["gene_id"] is None
    assert result["tokens"] > 0

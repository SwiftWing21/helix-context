"""Tests for the SNOW benchmark harness helpers."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Ensure benchmarks package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.snow.bench_snow import aggregate_scorecard, build_fingerprints, run_t0


class MockFingerprintClient:
    """Tiny fake for the /fingerprint client contract."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def fingerprint(self, query: str, max_results: int, profile: str):
        self.calls.append({
            "query": query,
            "max_results": max_results,
            "profile": profile,
        })
        return self.payload


def test_run_t0_normalizes_fingerprint_payload():
    client = MockFingerprintClient({
        "fingerprints": [
            {
                "gene_id": "g1",
                "score": 3.5,
                "tier_contributions": {"fts5": 2.0, "tag_exact": 1.5},
            },
            {
                "gene_id": "g2",
                "score": 1.25,
                "tier_contributions": {"splade": 1.25},
            },
        ]
    })

    payload, gene_ids, scores, tier_contribs = run_t0(
        client,
        query="jwt refresh token",
        profile="quality",
        max_results=12,
    )

    assert payload["fingerprints"][0]["gene_id"] == "g1"
    assert gene_ids == ["g1", "g2"]
    assert scores == {"g1": 3.5, "g2": 1.25}
    assert tier_contribs == {
        "g1": {"fts5": 2.0, "tag_exact": 1.5},
        "g2": {"splade": 1.25},
    }
    assert client.calls == [{
        "query": "jwt refresh token",
        "max_results": 12,
        "profile": "quality",
    }]


def test_build_fingerprints_uses_endpoint_metadata_with_db_fallback():
    retrieval_fps = [
        {
            "gene_id": "g1",
            "source": "src/auth.py",
            "domains": ["auth"],
            "entities": ["JWT"],
        },
        {
            "gene_id": "g2",
            "path": "docs/notes.md",
        },
    ]
    gene_fields = {
        "g1": {"domains": ["fallback-auth"], "entities": ["fallback-jwt"]},
        "g2": {"domains": ["docs"], "entities": ["note"]},
    }

    fingerprints = build_fingerprints(
        retrieval_fps=retrieval_fps,
        gene_ids=["g1", "g2"],
        scores={"g1": 4.0, "g2": 2.5},
        tier_contribs={"g1": {"fts5": 4.0}, "g2": {"tag_exact": 2.5}},
        gene_fields=gene_fields,
    )

    assert fingerprints == [
        {
            "gene_id": "g1",
            "source": "src/auth.py",
            "score": 4.0,
            "tiers": {"fts5": 4.0},
            "domains": ["auth"],
            "entities": ["JWT"],
        },
        {
            "gene_id": "g2",
            "source": "docs/notes.md",
            "score": 2.5,
            "tiers": {"tag_exact": 2.5},
            "domains": ["docs"],
            "entities": ["note"],
        },
    ]


def test_aggregate_scorecard_carries_profile_metadata(tmp_path):
    genome_path = tmp_path / "genome.db"
    conn = sqlite3.connect(genome_path)
    conn.execute("CREATE TABLE genes (gene_id TEXT PRIMARY KEY)")
    conn.executemany("INSERT INTO genes(gene_id) VALUES (?)", [("g1",), ("g2",)])
    conn.commit()
    conn.close()

    scorecard = aggregate_scorecard(
        query_results=[
            {
                "oracle_result": {"tier": 1, "tokens": 42, "latency_s": 0.01, "gene_id": "g1"},
                "llm_result": None,
            }
        ],
        model_name="oracle-only",
        genome_path=str(genome_path),
        fingerprint_profile="fast",
        helix_url="http://127.0.0.1:11437",
    )

    assert scorecard["meta"]["fingerprint_profile"] == "fast"
    assert scorecard["meta"]["helix_url"] == "http://127.0.0.1:11437"
    assert scorecard["meta"]["gene_count"] == 2

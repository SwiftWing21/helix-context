"""Regression tests for /ingest metadata handling and binary-content guards.

Covers the ingest-path behaviors that drive know-vs-go packet labels:

- metadata["source_id"] aliases through to gene.source_id
- provenance knobs like observed_at / authority_class / support_span
  reach the real gene columns the packet builder reads
- content_type can carry a more precise source kind than the file
  extension alone
- chunked rich-media transcripts get derived authority + chunk spans

Also keeps the 2026-04-19 binary-content rejection guards in place.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from helix_context.config import (
    BudgetConfig,
    GenomeConfig,
    HelixConfig,
    RibosomeConfig,
    ServerConfig,
)
from helix_context.server import create_app


class _MockBackend:
    """Minimal ribosome mock that returns plausible JSON for any prompt."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        if "compression engine" in system:
            return json.dumps({
                "codons": [{"meaning": "test_codon", "weight": 0.8, "is_exon": True}],
                "complement": "compressed.",
                "promoter": {
                    "domains": ["test"],
                    "entities": ["TestEntity"],
                    "intent": "test",
                    "summary": "summary",
                },
            })
        return "{}"


def _all_genes(client):
    """Return every gene currently in the in-memory genome."""
    genome = client.app.state.helix.genome
    rows = genome.conn.execute(
        "SELECT gene_id, source_id, source_kind, volatility_class, authority_class, "
        "support_span, observed_at, last_verified_at, repo_root, mtime, content_hash, content "
        "FROM genes"
    ).fetchall()

    import types

    out = []
    for r in rows:
        out.append(types.SimpleNamespace(
            gene_id=r["gene_id"],
            source_id=r["source_id"],
            source_kind=r["source_kind"],
            volatility_class=r["volatility_class"],
            authority_class=r["authority_class"],
            support_span=r["support_span"],
            observed_at=r["observed_at"],
            last_verified_at=r["last_verified_at"],
            repo_root=r["repo_root"],
            mtime=r["mtime"],
            content_hash=r["content_hash"],
            content=r["content"],
        ))
    return out


def _genes_by_ids(client, gene_ids):
    all_genes = {g.gene_id: g for g in _all_genes(client)}
    return [all_genes[gid] for gid in gene_ids if gid in all_genes]


@pytest.fixture
def client():
    config = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=4),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        server=ServerConfig(upstream="http://localhost:11434"),
    )
    app = create_app(config)
    app.state.helix.ribosome.backend = _MockBackend()
    return TestClient(app)


class TestMetadataAlias:
    """metadata["source_id"] must work the same as metadata["path"]."""

    def test_metadata_path_populates_source_id(self, client):
        resp = client.post("/ingest", json={
            "content": "content with marker MARKER_PATH_KEY",
            "content_type": "text",
            "metadata": {"path": "/probe/path_key.txt"},
        })
        assert resp.status_code == 200

        genes = _all_genes(client)
        hits = [g for g in genes if g.source_id == "/probe/path_key.txt"]
        assert len(hits) >= 1

    def test_metadata_source_id_alias_populates_source_id(self, client):
        resp = client.post("/ingest", json={
            "content": "content with marker MARKER_SID_KEY",
            "content_type": "text",
            "metadata": {"source_id": "/probe/sid_key.txt"},
        })
        assert resp.status_code == 200

        genes = _all_genes(client)
        hits = [g for g in genes if g.source_id == "/probe/sid_key.txt"]
        assert len(hits) >= 1

    def test_metadata_path_wins_when_both_provided(self, client):
        resp = client.post("/ingest", json={
            "content": "dual-key content MARKER_DUAL",
            "content_type": "text",
            "metadata": {"path": "/probe/winner.txt", "source_id": "/probe/loser.txt"},
        })
        assert resp.status_code == 200

        genes = _all_genes(client)
        source_ids = {g.source_id for g in genes if g.source_id}
        assert "/probe/winner.txt" in source_ids
        assert "/probe/loser.txt" not in source_ids


class TestKnowVsGoProvenance:
    def test_metadata_fields_reach_packet_facing_gene_columns(self, client):
        resp = client.post("/ingest", json={
            "content": "port = 11437\nmodel = qwen3:8b\n",
            "content_type": "config",
            "metadata": {
                "source_id": "/repo/helix/helix.toml",
                "repo_root": "/repo/helix",
                "observed_at": 1234.5,
                "last_verified_at": 1235.5,
                "mtime": 1200.0,
                "content_hash": "deadbeef",
                "authority_class": "derived",
                "volatility_class": "hot",
                "support_span": "lines:1-2",
            },
        })
        assert resp.status_code == 200

        genes = _genes_by_ids(client, resp.json()["gene_ids"])
        assert len(genes) == 1
        gene = genes[0]
        assert gene.source_id == "/repo/helix/helix.toml"
        assert gene.repo_root == "/repo/helix"
        assert gene.observed_at == 1234.5
        assert gene.last_verified_at == 1235.5
        assert gene.mtime == 1200.0
        assert gene.content_hash == "deadbeef"
        assert gene.authority_class == "derived"
        assert gene.volatility_class == "hot"
        assert gene.support_span == "lines:1-2"

    def test_content_type_hint_can_override_extension_for_source_kind(self, client):
        resp = client.post("/ingest", json={
            "content": '{"ans_full": 5, "latency_ms": 887}',
            "content_type": "benchmark",
            "metadata": {"source_id": "/repo/results/run.json"},
        })
        assert resp.status_code == 200

        genes = _genes_by_ids(client, resp.json()["gene_ids"])
        assert len(genes) == 1
        gene = genes[0]
        assert gene.source_kind == "benchmark"
        assert gene.volatility_class == "medium"
        assert gene.authority_class == "primary"

    def test_media_transcript_chunks_get_derived_authority_and_chunk_spans(self, client):
        para_a = "Speaker A: we shipped the audio pipeline and need time-coded evidence. " * 70
        para_b = "Speaker B: every chunk should say what span it covers before an agent acts. " * 70
        transcript = para_a + "\n\n" + para_b

        resp = client.post("/ingest", json={
            "content": transcript,
            "content_type": "transcript",
            "metadata": {"source_id": "/media/demo.mp4"},
        })
        assert resp.status_code == 200

        gene_ids = resp.json()["gene_ids"]
        assert len(gene_ids) >= 2

        genes = _genes_by_ids(client, gene_ids)
        assert len(genes) >= 2
        assert all(g.source_kind == "transcript" for g in genes)
        assert all(g.authority_class == "derived" for g in genes)
        assert all((g.support_span or "").startswith("chunk:") for g in genes)


class TestBinaryContentRejection:
    def test_whitespace_only_rejected(self, client):
        resp = client.post("/ingest", json={
            "content": "   \n\t  ",
            "content_type": "text",
        })
        assert resp.status_code == 400
        assert "content" in resp.json().get("error", "").lower()

    def test_null_byte_in_text_rejected(self, client):
        resp = client.post("/ingest", json={
            "content": "before\x00after",
            "content_type": "text",
        })
        assert resp.status_code == 400
        assert "NULL" in resp.json().get("error", "") or "null" in resp.json().get("error", "").lower()

    def test_base64_content_accepted(self, client):
        import base64

        raw_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        resp = client.post("/ingest", json={
            "content": b64,
            "content_type": "text",
            "metadata": {"path": "/fixture.png"},
        })
        assert resp.status_code == 200


def test_existing_empty_content_still_rejected(client):
    resp = client.post("/ingest", json={"content": ""})
    assert resp.status_code == 400

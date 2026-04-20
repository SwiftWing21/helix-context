"""Regression tests for /ingest metadata handling and binary-content guards.

Covers two bugs caught by `tests/diagnostics/test_file_type_ingest.py`
on 2026-04-19:

    Bug #1 — ``metadata["source_id"]`` was silently dropped; only
             ``metadata["path"]`` mapped through to ``gene.source_id``.
             Without a populated source_id, provenance inference (kind,
             volatility_class) was skipped and HTTP-ingested content was
             effectively invisible to retrieval.
    Bug #4 — Binary content declared as ``content_type="text"`` caused
             SQLite TEXT column to truncate at the first NULL byte,
             producing ghost genes with empty / near-empty content.
             Whitespace-only content was also accepted.

See `~/.helix/shared/handoffs/2026-04-19_file_type_ingest_bugs.md` for
the full bug catalog + root causes.
"""

from __future__ import annotations

import json
from unittest.mock import patch

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
    """Minimal ribosome mock — returns plausible JSON for any prompt."""

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
    """Return every gene currently in the in-memory genome.

    Genome doesn't expose a bulk-list method, so we read directly from
    the sqlite connection it owns. This is test-only — production
    callers go through `query_genes()` with a proper query.
    """
    genome = client.app.state.helix.genome
    rows = genome.conn.execute(
        "SELECT gene_id, source_id, source_kind, volatility_class, content "
        "FROM genes"
    ).fetchall()
    # rows are sqlite3.Row; expose the fields we test as simple attrs
    import types
    out = []
    for r in rows:
        g = types.SimpleNamespace(
            gene_id=r["gene_id"],
            source_id=r["source_id"],
            source_kind=r["source_kind"],
            volatility_class=r["volatility_class"],
            content=r["content"],
        )
        out.append(g)
    return out


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


# ── Bug #1: metadata key alias ────────────────────────────────────────


class TestMetadataAlias:
    """`metadata["source_id"]` must work the same as `metadata["path"]`."""

    def test_metadata_path_populates_source_id(self, client):
        resp = client.post("/ingest", json={
            "content": "content with marker MARKER_PATH_KEY",
            "content_type": "text",
            "metadata": {"path": "/probe/path_key.txt"},
        })
        assert resp.status_code == 200

        genes = _all_genes(client)
        hits = [g for g in genes if g.source_id == "/probe/path_key.txt"]
        assert len(hits) >= 1, "metadata.path did not propagate to gene.source_id"

    def test_metadata_source_id_alias_populates_source_id(self, client):
        """Callers using metadata.source_id get the same result as .path."""
        resp = client.post("/ingest", json={
            "content": "content with marker MARKER_SID_KEY",
            "content_type": "text",
            "metadata": {"source_id": "/probe/sid_key.txt"},
        })
        assert resp.status_code == 200

        genes = _all_genes(client)
        hits = [g for g in genes if g.source_id == "/probe/sid_key.txt"]
        assert len(hits) >= 1, (
            "metadata.source_id alias did not propagate — regression of bug #1 "
            "from 2026-04-19 file-type diagnostic"
        )

    def test_metadata_path_wins_when_both_provided(self, client):
        """If both keys are provided, `path` takes precedence (stability)."""
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


# ── Bug #4: binary / empty content rejection ─────────────────────────


class TestBinaryContentRejection:
    def test_whitespace_only_rejected(self, client):
        resp = client.post("/ingest", json={
            "content": "   \n\t  ",
            "content_type": "text",
        })
        assert resp.status_code == 400
        assert "content" in resp.json().get("error", "").lower()

    def test_null_byte_in_text_rejected(self, client):
        """Binary payload declared as text-content must be rejected, not
        silently truncated at the first NULL byte."""
        resp = client.post("/ingest", json={
            "content": "before\x00after",
            "content_type": "text",
        })
        assert resp.status_code == 400, (
            "NULL-byte content was accepted — regression of bug #4 "
            "(SQLite TEXT column truncates at first NUL, creating ghost genes)"
        )
        assert "NULL" in resp.json().get("error", "") or "null" in resp.json().get("error", "").lower()

    def test_base64_content_accepted(self, client):
        """Base64-encoded binary (no NULLs) passes through fine."""
        import base64
        raw_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        resp = client.post("/ingest", json={
            "content": b64,
            "content_type": "text",
            "metadata": {"path": "/fixture.png"},
        })
        assert resp.status_code == 200, (
            "Base64-wrapped binary should pass NULL check and be stored"
        )


# ── Sanity: the existing empty-content path still rejects ─────────────


def test_existing_empty_content_still_rejected(client):
    resp = client.post("/ingest", json={"content": ""})
    assert resp.status_code == 400

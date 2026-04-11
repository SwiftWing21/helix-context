"""
Gate 4 -- HTTP sidecar + proxy tests.

Tests the FastAPI endpoints using TestClient (no real Ollama or upstream needed).
The HelixContextManager is initialized with a mock ribosome backend.
"""

import json
import pytest

from fastapi.testclient import TestClient

from helix_context.config import HelixConfig, BudgetConfig, GenomeConfig, RibosomeConfig, ServerConfig
from helix_context.server import create_app


# -- Helpers -----------------------------------------------------------

class ServerMockBackend:
    """Returns plausible JSON for all ribosome operations."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        if "compression engine" in system:
            return json.dumps({
                "codons": [{"meaning": "test_codon", "weight": 0.8, "is_exon": True}],
                "complement": "Compressed test content.",
                "promoter": {
                    "domains": ["test"],
                    "entities": ["TestEntity"],
                    "intent": "test",
                    "summary": "Test content for server tests",
                },
            })
        elif "expression scorer" in system:
            return json.dumps({})
        elif "context splicer" in system:
            return json.dumps({})
        elif "replication engine" in system:
            return json.dumps({
                "codons": [{"meaning": "exchange", "weight": 1.0, "is_exon": True}],
                "complement": "Test exchange.",
                "promoter": {"domains": ["test"], "entities": [], "intent": "test", "summary": "test"},
            })
        return "{}"


@pytest.fixture
def client():
    config = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=4),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        server=ServerConfig(upstream="http://localhost:11434"),
    )
    app = create_app(config)

    # Inject mock backend into the HelixContextManager
    app.state.helix.ribosome.backend = ServerMockBackend()

    test_client = TestClient(app)
    yield test_client


# -- Endpoint shape tests (no upstream needed) -------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ribosome" in data
        assert "genes" in data
        assert "upstream" in data


class TestStatsEndpoint:
    def test_stats_returns_genome_info(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_genes" in data
        assert "config" in data
        assert "pending_replications" in data


class TestMetricsTokensEndpoint:
    def test_tokens_starts_at_zero(self, client):
        resp = client.get("/metrics/tokens")
        assert resp.status_code == 200
        data = resp.json()
        assert "session" in data
        assert "lifetime" in data
        assert data["session"]["total"] == 0
        assert data["session"]["estimated_total"] == 0

    def test_tokens_session_started_at_present(self, client):
        resp = client.get("/metrics/tokens")
        data = resp.json()
        assert "started_at" in data["session"]
        assert isinstance(data["session"]["started_at"], (int, float))


class TestAdminComponentsEndpoint:
    def test_components_lists_ribosome(self, client):
        resp = client.get("/admin/components")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "count" in data
        assert "last_activity_s_ago" in data
        assert "idle_threshold_s" in data

        names = [c["name"] for c in data["components"]]
        # Ribosome is always loaded.
        assert "ribosome" in names
        # Every component must have name/kind/status fields.
        for c in data["components"]:
            assert "name" in c
            assert "kind" in c
            assert c["kind"] in ("encoder", "decoder")
            assert "status" in c
            assert c["status"] in ("running", "idle")

    def test_components_status_running_after_recent_activity(self, client):
        # /stats does NOT bump activity, but /context does.
        # Trigger activity via /context with a trivial query.
        client.post("/context", json={"query": "test"})
        resp = client.get("/admin/components")
        data = resp.json()
        assert data["last_activity_s_ago"] < 5.0
        # At least ribosome should be 'running' right after activity.
        ribosome = next(c for c in data["components"] if c["name"] == "ribosome")
        assert ribosome["status"] == "running"

    def test_components_count_matches_entries(self, client):
        resp = client.get("/admin/components")
        data = resp.json()
        assert data["count"] == len(data["components"])


class TestIngestEndpoint:
    def test_ingest_text(self, client):
        resp = client.post("/ingest", json={
            "content": "This is test content about authentication and JWT tokens.",
            "content_type": "text",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "gene_ids" in data
        assert data["count"] >= 1

    def test_ingest_code(self, client):
        resp = client.post("/ingest", json={
            "content": "def hello():\n    return 'world'",
            "content_type": "code",
        })
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_ingest_empty_content_rejected(self, client):
        resp = client.post("/ingest", json={"content": ""})
        assert resp.status_code == 400

    def test_ingest_with_metadata(self, client):
        resp = client.post("/ingest", json={
            "content": "File content here.",
            "content_type": "text",
            "metadata": {"path": "src/auth.py"},
        })
        assert resp.status_code == 200

    def test_stats_reflects_ingest(self, client):
        # Ingest something first
        client.post("/ingest", json={
            "content": "Content for stats check.",
            "content_type": "text",
        })

        resp = client.get("/stats")
        data = resp.json()
        assert data["total_genes"] >= 1


class TestContextEndpoint:
    def test_context_returns_continue_format(self, client):
        # Ingest first so there's something to find
        client.post("/ingest", json={
            "content": "Authentication uses JWT tokens for session management.",
            "content_type": "text",
        })

        resp = client.post("/context", json={"query": "auth jwt"})
        assert resp.status_code == 200
        data = resp.json()

        # Should return Continue HTTP context provider format: list of objects
        assert isinstance(data, list)
        if data:
            assert "name" in data[0]
            assert "description" in data[0]
            assert "content" in data[0]

    def test_context_empty_query_rejected(self, client):
        resp = client.post("/context", json={"query": ""})
        assert resp.status_code == 400


class TestProxyEndpoint:
    def test_proxy_no_messages_rejected(self, client):
        resp = client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 400

    def test_proxy_no_user_message_attempts_passthrough(self, client):
        """If no user message, proxy should attempt to forward raw.
        Result depends on whether upstream is running."""
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "system", "content": "test"}],
        })
        # If upstream is running, we get 200 (passthrough); if not, 500
        assert resp.status_code in (200, 500, 502, 503)

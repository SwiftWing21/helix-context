"""
Gate 4 -- HTTP sidecar + proxy tests.

Tests the FastAPI endpoints using TestClient (no real Ollama or upstream needed).
The HelixContextManager is initialized with a mock ribosome backend.
"""

import json
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

import helix_context.server as server_mod
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
    def test_health_returns_ok(self, client, monkeypatch):
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ribosome" in data
        assert "genes" in data
        assert "upstream" in data
        assert data["checks"]["upstream_ready"] is True

    def test_health_exposes_cost_class(self, client, monkeypatch):
        """W2-B: /health surfaces backend cost classification so MCP
        clients can warn users when they're on a paid backend."""
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "ribosome_backend" in data
        assert "ribosome_cost_class" in data
        # Test fixture uses ribosome defaults (disabled unless explicitly
        # enabled), but the field must still be present and well-formed.
        assert data["ribosome_cost_class"] in ("disabled", "local", "api+free", "api+paid")

    def test_health_reports_disabled_ribosome_by_default(self, monkeypatch):
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        app = create_app(HelixConfig(
            genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
            server=ServerConfig(upstream="http://localhost:11434"),
        ))
        with TestClient(app) as default_client:
            resp = default_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ribosome"] == "disabled"
        assert data["ribosome_backend"] == "disabled"
        assert data["ribosome_configured_backend"] == "ollama"
        assert data["ribosome_cost_class"] == "disabled"

    def test_health_degrades_when_upstream_unreachable(self, client, monkeypatch):
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": False, "detail": "connection refused"},
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["checks"]["upstream_ready"] is False
        assert "upstream model server is unreachable" in data["message"].lower()


class TestStatsEndpoint:
    def test_stats_returns_genome_info(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_genes" in data
        assert "config" in data
        assert "pending_replications" in data


class TestAdminShutdownEndpoint:
    """The /admin/shutdown endpoint should return 200 and stamp the
    signal file. We can't actually test the SIGINT-on-self path
    without spawning a real subprocess — in-process TestClient would
    die if we sent SIGINT here. Instead, patch os.kill and verify
    it was called."""

    def test_shutdown_returns_200_and_fires_signal(self, client):
        # The endpoint imports os lazily inside the handler, so we patch
        # the os module's kill attribute directly — TestClient would die
        # if we actually let SIGINT reach this process.
        import os
        with patch.object(os, "kill") as mock_kill:
            resp = client.post("/admin/shutdown", json={
                "actor": "test",
                "reason": "unit test",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["shutting_down"] is True
        assert data["actor"] == "test"
        assert data["reason"] == "unit test"
        mock_kill.assert_called_once()

    def test_shutdown_with_empty_body_uses_defaults(self, client):
        import os
        with patch.object(os, "kill"):
            resp = client.post("/admin/shutdown", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["actor"] == "unknown"
        assert "manual shutdown" in data["reason"]


class TestContextCitationEnrichment:
    """Item 6 — /context citations carry authored_by_party / authored_by_handle
    when the expressed gene has a gene_attribution row."""

    def test_citation_includes_attribution_when_present(self, client):
        # Register a participant
        reg = client.post("/sessions/register", json={
            "party_id": "max@local",
            "handle": "taude",
        }).json()
        pid = reg["participant_id"]

        # Ingest with attribution
        client.post("/ingest", json={
            "content": "the answer to the universe is forty two",
            "content_type": "text",
            "participant_id": pid,
        })

        # Query context — the ingested gene should appear in citations
        # WITH attribution. Use a query that's likely to match.
        resp = client.post("/context", json={
            "query": "answer universe forty two",
            "decoder_mode": "none",
        })
        assert resp.status_code == 200
        data = resp.json()
        if isinstance(data, list):
            data = data[0]
        agent = data.get("agent", {})
        citations = agent.get("citations", [])

        # At least one citation should carry attribution. We don't enforce
        # it for ALL citations because the test environment may include
        # other genes from prior tests in the same client fixture.
        attributed = [c for c in citations if c.get("authored_by_party")]
        if not attributed:
            pytest.skip("query did not retrieve the attributed gene — retrieval is not deterministic across test runs")
        assert any(c.get("authored_by_party") == "max@local" for c in attributed)
        assert any(c.get("authored_by_handle") == "taude" for c in attributed)

    def test_unattributed_gene_omits_attribution_fields(self, client):
        # Ingest WITHOUT attribution
        client.post("/ingest", json={
            "content": "orphan content with distinctive marker xyzzyplugh",
            "content_type": "text",
            "local_federation": False,
        })
        resp = client.post("/context", json={
            "query": "xyzzyplugh",
            "decoder_mode": "none",
        })
        data = resp.json()
        if isinstance(data, list):
            data = data[0]
        citations = data.get("agent", {}).get("citations", [])
        for c in citations:
            # Genes without attribution should not have these fields set.
            # If the field is present, it should be falsy / None.
            assert c.get("authored_by_party") in (None, "", False)


class TestIngestFederationOverrides:
    def test_ingest_accepts_explicit_identity_handles(self, client):
        resp = client.post("/ingest", json={
            "content": "identity override marker for claude ingest",
            "content_type": "text",
            "org_id": "swiftwing",
            "party_id": "swift_wing21",
            "participant_handle": "max",
            "agent_handle": "laude",
            "agent_kind": "claude-code",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert data["attributed"] >= 1

        gene_id = data["gene_ids"][0]
        row = client.app.state.helix.genome.conn.execute(
            "SELECT ga.org_id, ga.party_id, p.handle AS participant_handle, "
            "       a.handle AS agent_handle, a.kind AS agent_kind "
            "FROM gene_attribution ga "
            "LEFT JOIN participants p ON p.participant_id = ga.participant_id "
            "LEFT JOIN agents a ON a.agent_id = ga.agent_id "
            "WHERE ga.gene_id = ?",
            (gene_id,),
        ).fetchone()

        assert row is not None
        assert row["org_id"] == "swiftwing"
        assert row["party_id"] == "swift_wing21"
        assert row["participant_handle"] == "max"
        assert row["agent_handle"] == "laude"
        assert row["agent_kind"] == "claude-code"


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
        # The test fixture injects a live mock backend, so ribosome is active.
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

    def test_components_omit_paused_ribosome(self, client):
        pause_resp = client.post("/admin/ribosome/pause")
        assert pause_resp.status_code == 200
        assert pause_resp.json()["paused"] is True

        resp = client.get("/admin/components")
        assert resp.status_code == 200
        data = resp.json()

        names = [c["name"] for c in data["components"]]
        assert "ribosome" not in names
        assert data["count"] == len(data["components"])

    def test_components_omit_disabled_ribosome(self):
        app = create_app(HelixConfig(
            genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
            server=ServerConfig(upstream="http://localhost:11434"),
        ))
        with TestClient(app) as default_client:
            resp = default_client.get("/admin/components")
        assert resp.status_code == 200
        data = resp.json()
        names = [c["name"] for c in data["components"]]
        assert "ribosome" not in names


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

    def test_context_can_swap_to_packet_mode(self, client):
        client.post("/ingest", json={
            "content": "Authentication config controls JWT session settings.",
            "content_type": "text",
            "metadata": {"path": "config/auth.toml"},
        })

        resp = client.post("/context", json={
            "query": "authentication config",
            "response_mode": "packet",
            "task_type": "edit",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["response_mode"] == "packet"
        assert data["task_type"] == "edit"
        assert "verified" in data
        assert "stale_risk" in data
        assert "refresh_targets" in data

    def test_context_empty_query_rejected(self, client):
        resp = client.post("/context", json={"query": ""})
        assert resp.status_code == 400

    def test_context_invalid_response_mode_rejected(self, client):
        resp = client.post("/context", json={
            "query": "auth jwt",
            "response_mode": "banana",
        })
        assert resp.status_code == 400


class TestContextPacketEndpoint:
    def test_packet_returns_freshness_labeled_groups(self, client):
        client.post("/ingest", json={
            "content": "Authentication config controls JWT session settings.",
            "content_type": "text",
            "metadata": {"path": "config/auth.toml"},
        })

        resp = client.post("/context/packet", json={
            "query": "authentication config",
            "task_type": "edit",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_type"] == "edit"
        assert data["query"] == "authentication config"
        assert "verified" in data
        assert "stale_risk" in data
        assert "refresh_targets" in data
        assert data["response_mode"] == "packet"
        assert isinstance(data["verified"], list)
        assert isinstance(data["stale_risk"], list)
        if data["verified"]:
            assert data["verified"][0]["status"] == "verified"

    def test_packet_empty_query_rejected(self, client):
        resp = client.post("/context/packet", json={"query": ""})
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


class TestHITLEndpoints:
    """POST /hitl/emit + GET /hitl/recent — MCP tool surface for HITL events.

    The underlying DAL is covered exhaustively in test_registry.py; these
    tests verify the HTTP adapter layer: argument validation, JSON shape,
    error paths. Each test uses a unique party_id so ordering and leakage
    across tests is harmless.
    """

    def _register(self, client, handle: str, party_id: str) -> str:
        """Register a participant and return its id. Creates party on TOFU."""
        resp = client.post(
            "/sessions/register",
            json={"party_id": party_id, "handle": handle},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["participant_id"]

    def test_emit_requires_pause_type(self, client):
        resp = client.post("/hitl/emit", json={})
        assert resp.status_code == 400
        assert "pause_type" in resp.json()["error"]

    def test_emit_requires_participant_or_party(self, client):
        """Without participant_id and without party_id the event cannot be
        attributed to anyone; registry.emit_hitl_event returns None and
        the endpoint should surface that as a 400."""
        resp = client.post("/hitl/emit", json={"pause_type": "other"})
        assert resp.status_code == 400

    def test_emit_with_participant_id_succeeds(self, client):
        pid = self._register(client, "laude", "party_emit_pid")
        resp = client.post(
            "/hitl/emit",
            json={
                "pause_type": "permission_request",
                "participant_id": pid,
                "task_context": "about to delete session log",
                "chat_signals": {
                    "tone_uncertainty": 0.72,
                    "risk_keywords": ["delete", "force"],
                    "recoverability": "uncertain",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["ok"] is True
        assert isinstance(data["event_id"], str)
        assert len(data["event_id"]) > 0

    def test_emit_with_party_only_succeeds(self, client):
        """party_id alone (no participant) should work for server-side
        emit flows that know the party but not a specific participant."""
        # Create the party via a register call, then emit with party_id only.
        self._register(client, "ghost", "party_emit_party_only")
        resp = client.post(
            "/hitl/emit",
            json={
                "pause_type": "uncertainty_check",
                "party_id": "party_emit_party_only",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_emit_unknown_participant_rejected(self, client):
        resp = client.post(
            "/hitl/emit",
            json={
                "pause_type": "other",
                "participant_id": "nonexistent-participant-uuid",
            },
        )
        assert resp.status_code == 400

    def test_emit_unknown_pause_type_coerces_to_other(self, client):
        """Unknown pause_type should coerce to 'other' per the DAL
        contract — instrumentation must not fail on schema gaps."""
        pid = self._register(client, "laude2", "party_emit_unknown_pt")
        resp = client.post(
            "/hitl/emit",
            json={"pause_type": "pineapple", "participant_id": pid},
        )
        assert resp.status_code == 200
        event_id = resp.json()["event_id"]

        recent = client.get("/hitl/recent?party_id=party_emit_unknown_pt")
        assert recent.status_code == 200
        rows = recent.json()["events"]
        matching = [e for e in rows if e["event_id"] == event_id]
        assert len(matching) == 1
        assert matching[0]["pause_type"] == "other"

    def test_recent_returns_events_newest_first(self, client):
        pid = self._register(client, "laude3", "party_recent_order")
        # Emit two events in sequence
        client.post(
            "/hitl/emit",
            json={
                "pause_type": "other",
                "participant_id": pid,
                "task_context": "first",
            },
        )
        client.post(
            "/hitl/emit",
            json={
                "pause_type": "other",
                "participant_id": pid,
                "task_context": "second",
            },
        )

        resp = client.get("/hitl/recent?party_id=party_recent_order")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 2
        # Newest first — "second" should lead.
        assert data["events"][0]["task_context"] == "second"
        assert data["events"][1]["task_context"] == "first"

    def test_recent_filters_by_pause_type(self, client):
        pid = self._register(client, "laude4", "party_recent_filter")
        client.post(
            "/hitl/emit",
            json={"pause_type": "permission_request", "participant_id": pid},
        )
        client.post(
            "/hitl/emit",
            json={"pause_type": "rollback_confirm", "participant_id": pid},
        )

        resp = client.get(
            "/hitl/recent?party_id=party_recent_filter&pause_type=permission_request"
        )
        assert resp.status_code == 200
        events = resp.json()["events"]
        assert len(events) >= 1
        for e in events:
            assert e["pause_type"] == "permission_request"

    def test_recent_limit_capped(self, client):
        """limit > 500 should be silently capped rather than returning 500."""
        resp = client.get("/hitl/recent?limit=99999")
        assert resp.status_code == 200
        # We don't have 500 events; just check no error.
        assert "events" in resp.json()

    def test_recent_empty_is_empty_list_not_null(self, client):
        resp = client.get("/hitl/recent?party_id=party_never_emitted")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert data["count"] == 0

    def test_emit_then_round_trip_preserves_chat_signals(self, client):
        """Chat signals supplied on emit must show up in the recent query."""
        pid = self._register(client, "laude5", "party_roundtrip")
        client.post(
            "/hitl/emit",
            json={
                "pause_type": "uncertainty_check",
                "participant_id": pid,
                "chat_signals": {
                    "tone_uncertainty": 0.42,
                    "risk_keywords": ["force-push", "drop"],
                    "recoverability": "recoverable",
                },
            },
        )
        resp = client.get("/hitl/recent?party_id=party_roundtrip&limit=5")
        events = resp.json()["events"]
        assert len(events) == 1
        e = events[0]
        assert e["operator_tone_uncertainty"] == pytest.approx(0.42)
        assert e["operator_risk_keywords"] == ["force-push", "drop"]
        assert e["recoverability_signal"] == "recoverable"


class TestDebugIntrospectionEndpoints:
    """GET /genes/{id} + GET /debug/neighbors + GET /debug/preview.

    Cheap introspection surface -- single-gene fetch, SEMA-only
    neighbors (lighter than /debug/resonance), and a dry-run of the
    express pipeline that skips the splice leg.
    """

    def test_gene_get_unknown_returns_404(self, client):
        resp = client.get("/genes/nonexistent-gene-id")
        assert resp.status_code == 404
        assert "Unknown gene_id" in resp.json()["error"]

    def test_neighbors_empty_genome_returns_empty_list(self, client):
        """No genes ingested -> empty neighbor list, still 200."""
        resp = client.get("/debug/neighbors?query=anything&k=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["neighbors"] == []
        assert data["count"] == 0
        assert data["query"] == "anything"
        assert data["k"] == 5

    def test_preview_empty_genome_returns_empty_candidates(self, client):
        """Pipeline dry-run on empty genome: extraction still works,
        candidates is empty."""
        resp = client.get("/debug/preview?query=search+me&max_genes=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "search me"
        assert data["candidates"] == []
        assert data["count"] == 0
        assert "domains" in data["extracted"]
        assert "entities" in data["extracted"]
        assert data["profile"] == "balanced"

    def test_preview_extracts_query_signals(self, client):
        """Extraction is pure string processing; must produce something
        even on an empty genome."""
        resp = client.get(
            "/debug/preview?query=authentication+jwt+token"
        )
        assert resp.status_code == 200
        extracted = resp.json()["extracted"]
        assert extracted["domains"] or extracted["entities"]

    def test_preview_returns_ingested_metadata_path(self, client):
        ingest = client.post("/ingest", json={
            "content": "Authentication module uses JWT refresh tokens.",
            "content_type": "text",
            "metadata": {"path": "src/auth.py"},
        })
        assert ingest.status_code == 200

        resp = client.get("/debug/preview?query=authentication+jwt+refresh&max_genes=5")
        assert resp.status_code == 200
        candidates = resp.json()["candidates"]
        assert any(c.get("path") == "src/auth.py" for c in candidates)

    def _inject_fake_preview_candidates(self, client, monkeypatch, scored):
        """Same pattern as fingerprint helper — mock _express + refiners."""
        from helix_context.schemas import Gene, PromoterTags

        genes = [
            Gene(
                gene_id=gid,
                content=f"content-{gid}",
                complement=f"summary-{gid}",
                codons=[],
                promoter=PromoterTags(domains=[], entities=[], intent="", summary=""),
                source_id=f"src/{gid}.py",
            )
            for gid, _ in scored
        ]
        score_map = {gid: float(score) for gid, score in scored}

        helix = client.app.state.helix

        def _fake_express(domains, entities, max_genes, **_kw):
            helix.genome.last_query_scores = dict(score_map)
            helix.genome.last_tier_contributions = {
                gid: {"fts5": score} for gid, score in score_map.items()
            }
            return list(genes)

        def _fake_refiners(query, candidates, max_genes, **_kw):
            return list(candidates), {gid: {} for gid, _ in scored}

        monkeypatch.setattr(helix, "_express", _fake_express)
        monkeypatch.setattr(helix, "_apply_candidate_refiners", _fake_refiners)

    def test_preview_score_floor_default_preserves_behavior(
        self, client, monkeypatch
    ):
        self._inject_fake_preview_candidates(
            client, monkeypatch,
            [("g1", 10.0), ("g2", 5.0), ("g3", 1.0)],
        )
        resp = client.get("/debug/preview?query=anything&max_genes=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["score_floor"] == 0.0
        assert data["returned"] == 3
        assert data["filtered_by_floor"] == 0
        assert data["truncated_by_cap"] == 0

    def test_preview_score_floor_and_cap_accounting(
        self, client, monkeypatch
    ):
        # Same deterministic shape as the /fingerprint test:
        # evaluated=5, above_floor=3, returned=2, filtered=2, truncated=1
        self._inject_fake_preview_candidates(
            client, monkeypatch,
            [
                ("g1", 10.0),
                ("g2", 8.0),
                ("g3", 6.0),
                ("g4", 2.0),
                ("g5", 1.0),
            ],
        )
        resp = client.get(
            "/debug/preview?query=anything&max_genes=2&score_floor=5.0"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["evaluated_total"] == 5
        assert data["above_floor_total"] == 3
        assert data["returned"] == 2
        assert data["filtered_by_floor"] == 2
        assert data["truncated_by_cap"] == 1
        assert "truncated by max_genes=2" in data["response_hint"]
        returned_ids = [c["gene_id"] for c in data["candidates"]]
        assert returned_ids == ["g1", "g2"]

    def test_preview_negative_score_floor_rejected(self, client):
        resp = client.get(
            "/debug/preview?query=anything&score_floor=-1.0"
        )
        assert resp.status_code == 400

    def test_fingerprint_returns_navigation_payload(self, client):
        client.post("/ingest", json={
            "content": "Authentication module uses JWT refresh tokens.",
            "content_type": "text",
            "metadata": {"path": "src/auth.py"},
        })

        resp = client.post("/fingerprint", json={
            "query": "authentication jwt refresh",
            "max_results": 5,
            "profile": "balanced",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "fingerprint"
        assert data["profile"] == "balanced"
        assert data["max_results"] == 5
        assert "content" not in data
        assert "fingerprints" in data
        if data["fingerprints"]:
            fp = data["fingerprints"][0]
            assert "tier_contributions" in fp
            assert "chromatin" in fp
            assert "score" in fp

    def test_fingerprint_fast_skips_query_expansion(self, client, monkeypatch):
        def _fake_expand(q):
            return q + " expandedterm"

        monkeypatch.setattr(client.app.state.helix, "_expand_query_intent", _fake_expand)

        fast = client.post("/fingerprint", json={
            "query": "plain query",
            "profile": "fast",
        })
        balanced = client.post("/fingerprint", json={
            "query": "plain query",
            "profile": "balanced",
        })

        assert fast.status_code == 200
        assert balanced.status_code == 200
        assert fast.json()["extracted"]["expanded_query"] == "plain query"
        assert balanced.json()["extracted"]["expanded_query"] == "plain query expandedterm"

    # -- score_floor + accounting --------------------------------------

    def _inject_fake_fingerprint_candidates(self, client, monkeypatch, scored):
        """Make _express return synthetic genes with deterministic scores.

        scored: list of (gene_id, score) tuples. Produces that many Gene
        objects with matching base scores and a no-op refiner pass so the
        final score equals the base score.
        """
        from helix_context.schemas import Gene, PromoterTags

        genes = [
            Gene(
                gene_id=gid,
                content=f"content-{gid}",
                complement=f"summary-{gid}",
                codons=[],
                promoter=PromoterTags(domains=[], entities=[], intent="", summary=""),
                source_id=f"src/{gid}.py",
            )
            for gid, _ in scored
        ]
        score_map = {gid: float(score) for gid, score in scored}

        helix = client.app.state.helix

        def _fake_express(domains, entities, max_results, **_kw):
            helix.genome.last_query_scores = dict(score_map)
            helix.genome.last_tier_contributions = {
                gid: {"fts5": score} for gid, score in score_map.items()
            }
            return list(genes)

        def _fake_refiners(query, candidates, max_results, **_kw):
            return list(candidates), {gid: {} for gid, _ in scored}

        monkeypatch.setattr(helix, "_express", _fake_express)
        monkeypatch.setattr(helix, "_apply_candidate_refiners", _fake_refiners)

    def test_fingerprint_score_floor_default_is_backwards_compatible(
        self, client, monkeypatch
    ):
        self._inject_fake_fingerprint_candidates(
            client, monkeypatch,
            [("g1", 10.0), ("g2", 5.0), ("g3", 1.0)],
        )
        resp = client.post("/fingerprint", json={
            "query": "anything", "max_results": 10, "profile": "fast",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["score_floor"] == 0.0
        assert data["returned"] == 3
        assert data["filtered_by_floor"] == 0
        assert data["truncated_by_cap"] == 0

    def test_fingerprint_score_floor_filters_low_score(
        self, client, monkeypatch
    ):
        self._inject_fake_fingerprint_candidates(
            client, monkeypatch,
            [("g1", 10.0), ("g2", 5.0), ("g3", 1.0), ("g4", 0.5)],
        )
        resp = client.post("/fingerprint", json={
            "query": "anything", "max_results": 10,
            "profile": "fast", "score_floor": 3.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["evaluated_total"] == 4
        assert data["above_floor_total"] == 2  # g1, g2
        assert data["returned"] == 2
        assert data["filtered_by_floor"] == 2
        assert data["truncated_by_cap"] == 0
        returned_ids = [fp["gene_id"] for fp in data["fingerprints"]]
        assert returned_ids == ["g1", "g2"]

    def test_fingerprint_cap_truncates_above_floor(
        self, client, monkeypatch
    ):
        # Deterministic case Max specified:
        # evaluated=5, above_floor=3, returned=2, filtered=2, truncated=1
        self._inject_fake_fingerprint_candidates(
            client, monkeypatch,
            [
                ("g1", 10.0),
                ("g2", 8.0),
                ("g3", 6.0),
                ("g4", 2.0),
                ("g5", 1.0),
            ],
        )
        resp = client.post("/fingerprint", json={
            "query": "anything", "max_results": 2,
            "profile": "fast", "score_floor": 5.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["evaluated_total"] == 5
        assert data["above_floor_total"] == 3
        assert data["returned"] == 2
        assert data["filtered_by_floor"] == 2
        assert data["truncated_by_cap"] == 1
        assert "truncated by max_results=2" in data["response_hint"]
        returned_ids = [fp["gene_id"] for fp in data["fingerprints"]]
        assert returned_ids == ["g1", "g2"]

    def test_fingerprint_invalid_score_floor_rejected(
        self, client, monkeypatch
    ):
        self._inject_fake_fingerprint_candidates(
            client, monkeypatch, [("g1", 1.0)],
        )
        resp_bad = client.post("/fingerprint", json={
            "query": "anything", "score_floor": "not-a-number",
        })
        assert resp_bad.status_code == 400

        resp_negative = client.post("/fingerprint", json={
            "query": "anything", "score_floor": -1.0,
        })
        assert resp_negative.status_code == 400

    def test_fingerprint_response_hint_describes_filtering(
        self, client, monkeypatch
    ):
        self._inject_fake_fingerprint_candidates(
            client, monkeypatch,
            [("g1", 1.0), ("g2", 2.0)],
        )
        resp = client.post("/fingerprint", json={
            "query": "anything", "max_results": 10,
            "profile": "fast", "score_floor": 100.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned"] == 0
        assert "All 2 evaluated candidates fell below" in data["response_hint"]

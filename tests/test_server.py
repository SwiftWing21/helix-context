"""
Gate 4 -- HTTP sidecar + proxy tests.

Tests the FastAPI endpoints using TestClient (no real Ollama or upstream needed).
The CymatixContextManager is initialized with a mock ribosome backend.
"""

import json
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

import cymatix_context.server as server_mod
from cymatix_context.config import CymatixConfig, GenomeConfig, KnowConfig, RibosomeConfig, ServerConfig
from cymatix_context.server import create_app

from tests.conftest import make_client, make_cymatix_config, make_gene, requires_spacy_model


# -- Helpers -----------------------------------------------------------
#
# The mock ribosome backend and standard test config previously defined
# here (``ServerMockBackend`` / inline ``CymatixConfig(...)``) are now the
# canonical ``MockCompressorBackend`` / ``make_cymatix_config`` in
# tests/conftest.py — this file's variant was the one the canonical
# version was derived from. The few tests below that build their own
# ``CymatixConfig(...)`` with REAL (non-mock) ribosome/budget defaults are
# intentionally left local — they exercise the disabled/real-backend
# path, not the shared mock shape.


@pytest.fixture
def client():
    return make_client()


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
        app = create_app(CymatixConfig(
            genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
            server=ServerConfig(upstream="http://localhost:11434"),
        ))
        with TestClient(app) as default_client:
            resp = default_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ribosome"] == "disabled"
        assert data["ribosome_backend"] == "disabled"
        # Honest reporting: while [ribosome].enabled=false, the placeholder
        # backend string is not dispatched, so we report None instead of
        # echoing the TOML default (which was a recurring source of "is
        # this LLM-backed?" confusion). When enabled flips to true, this
        # field reflects the configured backend string verbatim — see
        # ``test_health_reports_configured_backend_when_ribosome_enabled``.
        assert data["ribosome_configured_backend"] is None
        assert data["ribosome_cost_class"] == "disabled"

    def test_health_reports_configured_backend_when_ribosome_enabled(self, monkeypatch):
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        app = create_app(CymatixConfig(
            genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
            server=ServerConfig(upstream="http://localhost:11434"),
            ribosome=RibosomeConfig(enabled=True, backend="litellm"),
        ))
        with TestClient(app) as default_client:
            resp = default_client.get("/health")
        data = resp.json()
        assert data["ribosome_configured_backend"] == "litellm"
        assert data["ribosome_backend"] == "litellm"

    def test_health_degrades_when_upstream_unreachable(self, client, monkeypatch):
        """Active ribosome + unreachable upstream — the ONLY case that
        degrades on the upstream axis (three-state model, 2026-07-30).
        ``client`` has a mock compressor injected, so the runtime ribosome
        is active and the upstream is a real dependency."""
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": False, "detail": "connection refused"},
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["upstream_state"] == "unreachable"
        assert data["checks"]["upstream_ready"] is False
        assert "upstream model server is unreachable" in data["message"].lower()

    def test_health_ok_and_upstream_state_ok_when_active_and_reachable(self, client, monkeypatch):
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["upstream_state"] == "ok"
        assert data["checks"]["upstream_ready"] is True

    def test_health_inactive_ribosome_is_not_degraded_and_skips_probe(self, monkeypatch):
        """Disabled ribosome → upstream is not a dependency: status stays
        ok, upstream_state reports "inactive", and the probe is never
        called (a down Ollama must not add probe-timeout latency to every
        tray health poll on an LLM-free profile)."""
        def _probe_must_not_run(_url, timeout_s=1.0):
            raise AssertionError("_probe_upstream must not be called while ribosome is disabled")
        monkeypatch.setattr(server_mod, "_probe_upstream", _probe_must_not_run)
        app = create_app(CymatixConfig(
            genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
            server=ServerConfig(upstream="http://localhost:11434"),
        ))
        with TestClient(app) as default_client:
            resp = default_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["upstream_state"] == "inactive"
        assert data["upstream_reachable"] is None
        assert data["checks"]["upstream_ready"] is None
        assert "inactive" in data["message"].lower()

    def test_health_genome_failure_still_degrades_when_ribosome_inactive(self, monkeypatch):
        """The genome axis is unconditional: inactive upstream must not
        mask a broken knowledge store."""
        def _probe_must_not_run(_url, timeout_s=1.0):
            raise AssertionError("_probe_upstream must not be called while ribosome is disabled")
        monkeypatch.setattr(server_mod, "_probe_upstream", _probe_must_not_run)
        app = create_app(CymatixConfig(
            genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
            server=ServerConfig(upstream="http://localhost:11434"),
        ))
        with TestClient(app) as default_client:
            def _boom():
                raise RuntimeError("stats unavailable")
            default_client.app.state.cymatix.genome.stats = _boom
            resp = default_client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["upstream_state"] == "inactive"
        assert "genome stats failed" in data["message"].lower()

    def test_health_endpoint_includes_hardware_block(self, client, monkeypatch):
        """Task 12: /health surfaces the detected hardware + fallback state
        so operators can see at a glance whether they ended up on CPU
        because cuda probe failed, etc."""
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        from cymatix_context import hardware
        hardware.reset_for_test()
        fake = hardware.HardwareInfo(
            device="cpu", device_type="cpu", device_name="AMD Ryzen 9 7900X",
            vram_total_gb=None, vram_free_gb=None,
            cpu_arch="x86_64", cpu_brand="AMD Ryzen 9 7900X",
            system_ram_gb=64.0, requested_device="cuda",
            fallback_reason="cuda probe failed: RuntimeError: no driver",
            batch_size_overrides={},
        )
        monkeypatch.setattr(hardware, "_detect", lambda: fake)

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "hardware" in body
        hw = body["hardware"]
        assert hw["device"] == "cpu"
        assert hw["device_name"] == "AMD Ryzen 9 7900X"
        assert hw["requested_device"] == "cuda"
        assert hw["fallback_active"] is True
        assert "cuda probe failed" in hw["fallback_reason"]
        assert hw["vram_total_gb"] is None
        assert hw["system_ram_gb"] == 64.0
        assert hw["low_vram_warning"] is False

    # -- Issue #239: know/miss calibration staleness surfacing ---------

    def test_health_know_calibration_uncalibrated_is_stale(self, monkeypatch):
        """A fresh install (calibrated_at never set) must read as stale —
        not merely 'age unknown' — so agents don't silently trust the
        ship-time default betas as if they were bench-validated."""
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        client = make_client(make_cymatix_config(know=KnowConfig()))
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "know_calibration" in body
        know = body["know_calibration"]
        assert know["calibrated_at"] is None
        assert know["calibrated_on_n"] is None
        assert know["stale"] is True

    def test_health_know_calibration_fresh_is_not_stale(self, monkeypatch):
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        import datetime as _dt
        now_iso = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
        client = make_client(make_cymatix_config(
            know=KnowConfig(calibrated_at=now_iso, calibrated_on_n=842, stale_after_days=30),
        ))
        resp = client.get("/health")
        assert resp.status_code == 200
        know = resp.json()["know_calibration"]
        assert know["calibrated_at"] == now_iso
        assert know["calibrated_on_n"] == 842
        assert know["stale"] is False

    def test_health_know_calibration_old_calibration_is_stale(self, monkeypatch):
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        import datetime as _dt
        old_iso = (
            _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=60)
        ).replace(microsecond=0).isoformat()
        client = make_client(make_cymatix_config(
            know=KnowConfig(calibrated_at=old_iso, calibrated_on_n=500, stale_after_days=30),
        ))
        resp = client.get("/health")
        assert resp.status_code == 200
        know = resp.json()["know_calibration"]
        assert know["calibrated_at"] == old_iso
        assert know["stale"] is True

    def test_health_know_calibration_respects_custom_stale_after_days(self, monkeypatch):
        """A calibration 10 days old is fresh under stale_after_days=30
        but stale under stale_after_days=5 — the threshold is read from
        config, not hardcoded."""
        monkeypatch.setattr(
            server_mod,
            "_probe_upstream",
            lambda _url, timeout_s=1.0: {"reachable": True, "probe": "/api/tags", "status_code": 200},
        )
        import datetime as _dt
        ten_days_ago = (
            _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=10)
        ).replace(microsecond=0).isoformat()
        client = make_client(make_cymatix_config(
            know=KnowConfig(calibrated_at=ten_days_ago, calibrated_on_n=100, stale_after_days=5),
        ))
        resp = client.get("/health")
        know = resp.json()["know_calibration"]
        assert know["stale"] is True


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

    @requires_spacy_model
    def test_citation_includes_attribution_when_present(self, client):
        # Register a participant
        reg = client.post("/sessions/register", json={
            "party_id": "max@local",
            "handle": "taude",
        }).json()
        pid = reg["participant_id"]

        # Explicit tags keep this short fixture eligible for hot retrieval.
        ingest = client.post("/ingest", json={
            "content": "the answer to the universe is forty two",
            "content_type": "text",
            "participant_id": pid,
            "metadata": {"domains": ["astronomy"]},
        })
        assert ingest.status_code == 200
        ingested = ingest.json()
        assert ingested["count"] == 1
        assert ingested["attributed"] == 1
        gene_id, = ingested["gene_ids"]

        # Query as the owning party so the real party filter admits the document.
        resp = client.post("/context", json={
            "query": "answer universe forty two",
            "decoder_mode": "none",
            "party_id": "max@local",
        })
        assert resp.status_code == 200
        data = resp.json()
        if isinstance(data, list):
            data = data[0]
        agent = data.get("agent", {})
        citations = agent.get("citations", [])

        matching = [c for c in citations if c["gene_id"] == gene_id]
        assert matching, "The ingested document must reach citation enrichment"
        citation, = matching
        assert citation["authored_by_party"] == "max@local"
        assert citation["authored_by_handle"] == "taude"

    @pytest.mark.parametrize("attributed", [False, True], ids=["unattributed", "attributed"])
    def test_stored_document_citation_attribution(self, client, attributed):
        # Seed the real store and registry so citation coverage also runs on
        # fresh installs without the optional spaCy ingestion pipeline.
        gene = make_gene(
            content="the answer to the universe is forty two",
            domains=["astronomy"],
        )
        client.app.state.cymatix.genome.upsert_gene(gene, apply_gate=False)
        registry = client.app.state.registry
        if attributed:
            reg = client.post("/sessions/register", json={
                "party_id": "max@local", "handle": "taude",
            })
            assert reg.status_code == 200
            result = registry.attribute_gene(
                gene.gene_id, participant_id=reg.json()["participant_id"],
            )
            assert result is not None
        else:
            assert registry.get_attribution(gene.gene_id) is None

        resp = client.post("/context", json={
            "query": "answer universe forty two",
            "decoder_mode": "none",
            "party_id": "max@local",
        })
        assert resp.status_code == 200
        data = resp.json()
        if isinstance(data, list):
            data = data[0]
        citations = data.get("agent", {}).get("citations", [])
        matching = [c for c in citations if c["gene_id"] == gene.gene_id]
        assert matching, "The stored document must reach citation enrichment"
        citation, = matching
        if attributed:
            assert citation["authored_by_party"] == "max@local"
            assert citation["authored_by_handle"] == "taude"
        else:
            assert not citation.get("authored_by_party")
            assert not citation.get("authored_by_handle")


# Ingest-driving tests need the en_core_web_sm pipeline (#313).
@requires_spacy_model
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
        row = client.app.state.cymatix.genome.conn.execute(
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
            # #219 slice 2: lazily-armed encoders report "idle (not loaded)"
            # until their first use; "loaded" carries the explicit boolean.
            assert c["status"] in ("running", "idle", "idle (not loaded)")
            assert "loaded" in c

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
        app = create_app(CymatixConfig(
            genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
            server=ServerConfig(upstream="http://localhost:11434"),
        ))
        with TestClient(app) as default_client:
            resp = default_client.get("/admin/components")
        assert resp.status_code == 200
        data = resp.json()
        names = [c["name"] for c in data["components"]]
        assert "ribosome" not in names


@requires_spacy_model
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

    @requires_spacy_model
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


class TestResolveCallerAgent:
    """The _resolve_caller_agent helper resolves the `agent` telemetry
    label per /context request. Cardinality-bounded via an allowlist that
    collapses unknown handles to "other" so Prometheus label cardinality
    can't explode when callers ship freeform handles.
    """

    def _make_request(self, headers=None):
        from types import SimpleNamespace
        return SimpleNamespace(headers=headers or {})

    def test_body_agent_takes_precedence(self, monkeypatch):
        from cymatix_context.server import _resolve_caller_agent
        monkeypatch.setenv("CYMATIX_AGENT", "raude")
        req = self._make_request({"x-cymatix-agent": "taude"})
        assert _resolve_caller_agent(req, {"agent": "laude"}) == "laude"

    def test_header_when_body_missing(self, monkeypatch):
        from cymatix_context.server import _resolve_caller_agent
        monkeypatch.setenv("CYMATIX_AGENT", "raude")
        req = self._make_request({"x-cymatix-agent": "taude"})
        assert _resolve_caller_agent(req, {}) == "taude"

    def test_env_when_body_and_header_missing(self, monkeypatch):
        from cymatix_context.server import _resolve_caller_agent
        monkeypatch.setenv("CYMATIX_AGENT", "gemini")
        assert _resolve_caller_agent(self._make_request(), {}) == "gemini"

    def test_unknown_when_no_source_supplies(self, monkeypatch):
        from cymatix_context.server import _resolve_caller_agent
        monkeypatch.delenv("CYMATIX_AGENT", raising=False)
        assert _resolve_caller_agent(self._make_request(), {}) == "unknown"

    def test_off_allowlist_collapses_to_other(self, monkeypatch):
        """Cardinality cap: any handle not in the allowlist (or
        CYMATIX_AGENT_ALLOW extension) becomes "other", protecting
        Prometheus from per-pid / per-tab handles."""
        from cymatix_context.server import _resolve_caller_agent
        monkeypatch.delenv("CYMATIX_AGENT_ALLOW", raising=False)
        req = self._make_request()
        assert _resolve_caller_agent(req, {"agent": "weird-tab-37"}) == "other"

    def test_cymatix_agent_allow_extends_allowlist(self, monkeypatch):
        from cymatix_context.server import _resolve_caller_agent
        monkeypatch.setenv("CYMATIX_AGENT_ALLOW", "qaude, vaude")
        req = self._make_request()
        assert _resolve_caller_agent(req, {"agent": "qaude"}) == "qaude"
        assert _resolve_caller_agent(req, {"agent": "vaude"}) == "vaude"
        # Still collapses things not in either set.
        assert _resolve_caller_agent(req, {"agent": "rogue"}) == "other"

    def test_normalizes_case_and_whitespace(self, monkeypatch):
        from cymatix_context.server import _resolve_caller_agent
        monkeypatch.delenv("CYMATIX_AGENT", raising=False)
        req = self._make_request()
        assert _resolve_caller_agent(req, {"agent": "  Laude  "}) == "laude"


class TestProxyEndpoint:
    def test_proxy_no_messages_rejected(self, client):
        resp = client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 400

    def test_proxy_no_user_message_attempts_passthrough(self, client):
        """If no user message, proxy should attempt to forward raw.
        Outcome depends on the upstream:
          - 200: upstream accepted the forwarded request
          - 400/404: upstream is up and rejected (model not pulled, bad shape)
          - 502/503: upstream is unreachable
          - 500: forwarded request raised on upstream
        Any of these prove the proxy didn't short-circuit on its own."""
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3",
            "messages": [{"role": "system", "content": "test"}],
        })
        assert resp.status_code in (200, 400, 404, 500, 502, 503)


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

    @requires_spacy_model
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
        from cymatix_context.schemas import Gene, PromoterTags

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

        cymatix = client.app.state.cymatix

        def _fake_express(domains, entities, max_genes, **_kw):
            cymatix.genome.last_query_scores = dict(score_map)
            cymatix.genome.last_tier_contributions = {
                gid: {"fts5": score} for gid, score in score_map.items()
            }
            return list(genes)

        def _fake_refiners(query, candidates, max_genes, **_kw):
            return list(candidates), {gid: {} for gid, _ in scored}

        # R3 Stage C renamed _express -> _retrieve (with _express alias).
        # Internal callers use the canonical name, so patch that.
        monkeypatch.setattr(cymatix, "_retrieve", _fake_express)
        monkeypatch.setattr(cymatix, "_express", _fake_express)
        monkeypatch.setattr(cymatix, "_apply_candidate_refiners", _fake_refiners)

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

        monkeypatch.setattr(client.app.state.cymatix, "_expand_query_intent", _fake_expand)

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
        from cymatix_context.schemas import Gene, PromoterTags

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

        cymatix = client.app.state.cymatix

        def _fake_express(domains, entities, max_results, **_kw):
            cymatix.genome.last_query_scores = dict(score_map)
            cymatix.genome.last_tier_contributions = {
                gid: {"fts5": score} for gid, score in score_map.items()
            }
            return list(genes)

        def _fake_refiners(query, candidates, max_results, **_kw):
            return list(candidates), {gid: {} for gid, _ in scored}

        # R3 Stage C renamed _express -> _retrieve (with _express alias).
        # Internal callers use the canonical name, so patch that.
        monkeypatch.setattr(cymatix, "_retrieve", _fake_express)
        monkeypatch.setattr(cymatix, "_express", _fake_express)
        monkeypatch.setattr(cymatix, "_apply_candidate_refiners", _fake_refiners)

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


@requires_spacy_model
class TestContextPacketEndpoint:
    """POST /context/packet — agent-safe context bundle with freshness labels.

    Shape contract: the response must always carry verified /
    stale_risk / refresh_targets keys (plus task_type + query). Empty
    query returns 400. Oversize queries must not crash the server.
    """

    def _seed_one_gene(self, client):
        """Ingest a single gene so the packet has something to classify."""
        resp = client.post("/ingest", json={
            "content": (
                "Cymatix Context exposes /context/packet for agent-safe "
                "actions. The packet carries verified, stale_risk, and "
                "refresh_targets lists so the caller can decide which "
                "sources to reread before edits."
            ),
            "content_type": "text",
            "metadata": {"path": "docs/packet.md"},
        })
        assert resp.status_code == 200

    def test_packet_returns_required_shape(self, client):
        self._seed_one_gene(client)
        resp = client.post("/context/packet", json={
            "query": "how does /context/packet label freshness?",
            "task_type": "explain",
        })
        assert resp.status_code == 200
        data = resp.json()

        # Core shape — these keys must always be present, even when
        # empty, so callers can unconditionally iterate them.
        assert "verified" in data
        assert "stale_risk" in data
        assert "refresh_targets" in data
        assert isinstance(data["verified"], list)
        assert isinstance(data["stale_risk"], list)
        assert isinstance(data["refresh_targets"], list)

        # Echo fields from the request.
        assert data.get("task_type") == "explain"
        assert data.get("query") == "how does /context/packet label freshness?"
        assert data.get("response_mode") == "packet"

    def test_packet_empty_query_returns_400(self, client):
        resp = client.post("/context/packet", json={"query": ""})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_packet_whitespace_only_query_returns_400(self, client):
        resp = client.post("/context/packet", json={"query": "   \n\t  "})
        assert resp.status_code == 400

    def test_packet_large_query_does_not_crash(self, client):
        self._seed_one_gene(client)
        # 50KB query — unusual but must not raise.
        big = "how does cymatix compress context? " * 1500
        resp = client.post("/context/packet", json={
            "query": big,
            "task_type": "explain",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "verified" in data
        assert "stale_risk" in data
        assert "refresh_targets" in data

    def test_packet_max_genes_is_clamped(self, client):
        """Out-of-range max_genes must be clamped, not rejected."""
        self._seed_one_gene(client)
        # max_genes=999 should clamp to 32 silently.
        resp = client.post("/context/packet", json={
            "query": "packet freshness",
            "max_genes": 999,
        })
        assert resp.status_code == 200
        # Non-int max_genes should fall back to the default (no 500).
        resp2 = client.post("/context/packet", json={
            "query": "packet freshness",
            "max_genes": "not-an-int",
        })
        assert resp2.status_code == 200

    def test_packet_verified_items_carry_verified_status(self, client):
        """Items inside the `verified` list carry ``status == "verified"``
        — the shape tests above don't check the per-item freshness label.
        (Merged from the former TestContextPacketEndpointFreshness class.)
        """
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
        if data["verified"]:
            assert data["verified"][0]["status"] == "verified"


class TestSessionsRegister:
    """Tests for /sessions/register endpoint — optional vendor fields."""

    def test_sessions_register_accepts_vendor_host(self, client):
        """Register a participant with agent_kind and mcp_host fields."""
        resp = client.post(
            "/sessions/register",
            json={
                "party_id": "party_vh",
                "handle": "laude",
                "agent_kind": "claude-code",
                "mcp_host": "vscode",
            },
        )
        assert resp.status_code == 200, resp.text
        pid = resp.json()["participant_id"]

        # Verify it landed in the registry projection
        listing = client.get("/sessions", params={"party_id": "party_vh"}).json()
        rows = listing.get("participants", [])
        matching = [r for r in rows if r.get("participant_id") == pid]
        assert len(matching) == 1
        assert matching[0]["agent_kind"] == "claude-code"
        assert matching[0]["mcp_host"] == "vscode"

    def test_sessions_register_omitting_vendor_host_still_works(self, client):
        """Pre-existing clients (no agent_kind / mcp_host in body) keep working."""
        resp = client.post(
            "/sessions/register",
            json={"party_id": "party_legacy", "handle": "taude"},
        )
        assert resp.status_code == 200, resp.text
        pid = resp.json()["participant_id"]

        # Verify it registered without agent_kind/mcp_host (should be None)
        listing = client.get("/sessions", params={"party_id": "party_legacy"}).json()
        rows = listing.get("participants", [])
        matching = [r for r in rows if r.get("participant_id") == pid]
        assert len(matching) == 1
        assert matching[0].get("agent_kind") is None
        assert matching[0].get("mcp_host") is None


# -- Announce endpoint tests -------------------------------------------


def test_sessions_register_accepts_announce_fields(client):
    resp = client.post(
        "/sessions/register",
        json={
            "party_id": "party_register_announce",
            "handle": "laude",
            "ide_detected": "vscode",
            "ide_detection_via": "env:VSCODE_PID",
        },
    )
    assert resp.status_code == 200, resp.text
    pid = resp.json()["participant_id"]

    listing = client.get("/sessions", params={"party_id": "party_register_announce"}).json()
    rows = listing if isinstance(listing, list) else listing.get("participants", [])
    matching = [r for r in rows if r.get("participant_id") == pid]
    assert len(matching) == 1
    assert matching[0]["ide_detected"] == "vscode"
    assert matching[0]["ide_detection_via"] == "env:VSCODE_PID"
    assert matching[0].get("model_id") is None


def test_announce_endpoint_sets_model_id(client):
    """POST /sessions/{participant_id}/announce updates model_id.

    Registers with IDE + vendor fields so the same round trip also
    verifies announce (without ide_override) leaves them untouched at
    the HTTP layer — folded from the retired
    test_cymatix_announce_plumbing.py end-to-end test.
    """
    reg = client.post(
        "/sessions/register",
        json={
            "party_id": "party_announce",
            "handle": "laude",
            "ide_detected": "vscode",
            "ide_detection_via": "env:VSCODE_PID",
            "agent_kind": "claude-code",
        },
    )
    pid = reg.json()["participant_id"]

    resp = client.post(
        f"/sessions/{pid}/announce",
        json={"model_id": "claude-opus-4-7"},
    )
    assert resp.status_code == 200, resp.text

    listing = client.get("/sessions", params={"party_id": "party_announce"}).json()
    rows = listing if isinstance(listing, list) else listing.get("participants", [])
    matching = [r for r in rows if r.get("participant_id") == pid]
    assert matching[0]["model_id"] == "claude-opus-4-7"
    # Announce without ide_override must not clobber register-time fields.
    assert matching[0]["ide_detected"] == "vscode"
    assert matching[0]["ide_detection_via"] == "env:VSCODE_PID"
    assert matching[0]["agent_kind"] == "claude-code"


def test_announce_endpoint_with_ide_override_sets_agent_override_via(client):
    reg = client.post(
        "/sessions/register",
        json={
            "party_id": "party_override",
            "handle": "laude",
            "ide_detected": "vscode",
            "ide_detection_via": "env:VSCODE_PID",
        },
    )
    pid = reg.json()["participant_id"]

    resp = client.post(
        f"/sessions/{pid}/announce",
        json={"model_id": "gpt-5", "ide_override": "cursor"},
    )
    assert resp.status_code == 200, resp.text

    listing = client.get("/sessions", params={"party_id": "party_override"}).json()
    rows = listing if isinstance(listing, list) else listing.get("participants", [])
    matching = [r for r in rows if r.get("participant_id") == pid]
    assert matching[0]["ide_detected"] == "cursor"
    assert matching[0]["ide_detection_via"] == "agent_override"
    assert matching[0]["model_id"] == "gpt-5"


def test_announce_endpoint_unknown_participant_returns_200(client):
    """Silent no-op on unknown participant_id matches registry semantics
    and avoids leaking participant existence via 404 vs 200 distinction."""
    resp = client.post(
        "/sessions/does-not-exist/announce",
        json={"model_id": "claude-opus-4-7"},
    )
    assert resp.status_code == 200, resp.text

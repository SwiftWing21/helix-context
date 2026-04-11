"""
Session registry tests — DAL + FastAPI endpoints.

Covers the first slice of the session registry (see docs/SESSION_REGISTRY.md):
    - Schema migration runs on genome init
    - Registry DAL: register, heartbeat, list, get, attribute, recent, sweep
    - FastAPI endpoints: /sessions/register, /sessions/{id}/heartbeat,
      /sessions, /sessions/{handle}/recent
    - /ingest extension: participant_id -> automatic attribution

All tests run against in-memory SQLite — no touching of the live genome.db
at F:\\Projects\\helix-context\\genome.db. Safe to run while the real server
is live.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from helix_context.config import (
    BudgetConfig,
    GenomeConfig,
    HelixConfig,
    RibosomeConfig,
    ServerConfig,
)
from helix_context.registry import (
    DEFAULT_TTL_S,
    IDLE_TTL_S,
    STALE_TTL_S,
    Registry,
    _status_from_last_heartbeat,
)
from helix_context.server import create_app

from tests.conftest import make_gene


# ═══ DAL unit tests ═══════════════════════════════════════════════════


@pytest.fixture
def registry(genome):
    """Registry bound to the in-memory genome fixture from conftest."""
    return Registry(genome)


class TestSchemaMigration:
    def test_registry_tables_created_on_genome_init(self, genome):
        cur = genome.conn.cursor()
        tables = {
            row[0] for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "parties" in tables
        assert "participants" in tables
        assert "gene_attribution" in tables

    def test_registry_indexes_created(self, genome):
        cur = genome.conn.cursor()
        indexes = {
            row[0] for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_participants_handle" in indexes
        assert "idx_attribution_party_time" in indexes
        assert "idx_attribution_participant_time" in indexes

    def test_migration_is_idempotent(self, genome):
        """Running the migration twice should not raise."""
        cur = genome.conn.cursor()
        genome._ensure_registry_schema(cur)
        genome._ensure_registry_schema(cur)
        genome.conn.commit()


class TestRegisterParticipant:
    def test_register_creates_party_on_first_use(self, registry, genome):
        p = registry.register_participant(
            party_id="max@local",
            handle="taude",
            workspace="/f/Projects/Education",
        )
        assert p.participant_id
        assert p.party_id == "max@local"
        assert p.handle == "taude"
        assert p.status == "active"

        row = genome.conn.execute(
            "SELECT party_id, display_name, trust_domain FROM parties WHERE party_id = ?",
            ("max@local",),
        ).fetchone()
        assert row is not None
        assert row["trust_domain"] == "local"

    def test_second_participant_reuses_existing_party(self, registry, genome):
        registry.register_participant(party_id="max@local", handle="taude")
        registry.register_participant(party_id="max@local", handle="laude")

        party_count = genome.conn.execute(
            "SELECT COUNT(*) FROM parties WHERE party_id = ?",
            ("max@local",),
        ).fetchone()[0]
        assert party_count == 1

        participant_count = genome.conn.execute(
            "SELECT COUNT(*) FROM participants WHERE party_id = ?",
            ("max@local",),
        ).fetchone()[0]
        assert participant_count == 2

    def test_capabilities_round_trip(self, registry):
        p = registry.register_participant(
            party_id="max@local",
            handle="taude",
            capabilities=["ingest", "query"],
        )
        got = registry.get_participant(p.participant_id)
        assert got is not None
        assert got.capabilities == ["ingest", "query"]


class TestHeartbeat:
    def test_heartbeat_refreshes_last_seen(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")
        # Rewind last_heartbeat so the refresh is visible.
        genome.conn.execute(
            "UPDATE participants SET last_heartbeat = ? WHERE participant_id = ?",
            (time.time() - 60, p.participant_id),
        )
        genome.conn.commit()

        result = registry.heartbeat(p.participant_id)
        assert result is not None
        ttl, status = result
        assert ttl == DEFAULT_TTL_S
        assert status == "active"

        row = genome.conn.execute(
            "SELECT last_heartbeat FROM participants WHERE participant_id = ?",
            (p.participant_id,),
        ).fetchone()
        assert row["last_heartbeat"] > time.time() - 5

    def test_heartbeat_unknown_returns_none(self, registry):
        assert registry.heartbeat("nonexistent-id") is None


class TestListParticipants:
    def test_filter_by_party(self, registry):
        registry.register_participant(party_id="max@local", handle="taude")
        registry.register_participant(party_id="max@local", handle="laude")
        registry.register_participant(party_id="other@remote", handle="guest")

        max_participants = registry.list_participants(party_id="max@local")
        assert len(max_participants) == 2
        assert {p.handle for p in max_participants} == {"taude", "laude"}

        other = registry.list_participants(party_id="other@remote")
        assert len(other) == 1
        assert other[0].handle == "guest"

    def test_status_filter_all_returns_everyone(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")
        # Age one participant into "stale".
        genome.conn.execute(
            "UPDATE participants SET last_heartbeat = ? WHERE participant_id = ?",
            (time.time() - IDLE_TTL_S - 10, p.participant_id),
        )
        genome.conn.commit()

        active_only = registry.list_participants(party_id="max@local", status_filter="active")
        assert len(active_only) == 0

        all_statuses = registry.list_participants(party_id="max@local", status_filter="all")
        assert len(all_statuses) == 1
        assert all_statuses[0].status == "stale"

    def test_workspace_prefix_filter(self, registry):
        registry.register_participant(
            party_id="max@local", handle="taude",
            workspace="/f/Projects/Education",
        )
        registry.register_participant(
            party_id="max@local", handle="other",
            workspace="/f/Projects/Unrelated",
        )
        result = registry.list_participants(
            party_id="max@local",
            workspace_prefix="/f/Projects/Education",
        )
        assert len(result) == 1
        assert result[0].handle == "taude"


class TestStatusFromHeartbeat:
    def test_fresh_is_active(self):
        now = time.time()
        assert _status_from_last_heartbeat(now, now) == "active"

    def test_within_ttl_is_active(self):
        now = time.time()
        assert _status_from_last_heartbeat(now - DEFAULT_TTL_S + 1, now) == "active"

    def test_past_ttl_is_idle(self):
        now = time.time()
        assert _status_from_last_heartbeat(now - DEFAULT_TTL_S - 1, now) == "idle"

    def test_past_idle_is_stale(self):
        now = time.time()
        assert _status_from_last_heartbeat(now - IDLE_TTL_S - 1, now) == "stale"

    def test_past_stale_is_gone(self):
        now = time.time()
        assert _status_from_last_heartbeat(now - STALE_TTL_S - 1, now) == "gone"


class TestAttribution:
    def test_attribute_gene_writes_row(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")
        gene = make_gene(content="VS Code 1.115 shipped with agents app")
        genome.upsert_gene(gene)

        result = registry.attribute_gene(
            gene_id=gene.gene_id,
            participant_id=p.participant_id,
        )
        assert result is not None
        assert result.gene_id == gene.gene_id
        assert result.party_id == "max@local"
        assert result.participant_id == p.participant_id

        got = registry.get_attribution(gene.gene_id)
        assert got is not None
        assert got.party_id == "max@local"

    def test_attribute_unknown_participant_returns_none(self, registry, genome):
        gene = make_gene(content="orphan gene")
        genome.upsert_gene(gene)
        result = registry.attribute_gene(
            gene_id=gene.gene_id,
            participant_id="bogus-id",
        )
        assert result is None
        assert registry.get_attribution(gene.gene_id) is None

    def test_attribute_by_party_only(self, registry, genome):
        """Server-side ingests may know the party but not a specific participant."""
        # Create the party manually (no participant registered).
        genome.conn.execute(
            "INSERT INTO parties (party_id, display_name, trust_domain, created_at) "
            "VALUES ('server@local', 'server', 'local', ?)",
            (time.time(),),
        )
        genome.conn.commit()

        gene = make_gene(content="server-authored note")
        genome.upsert_gene(gene)

        result = registry.attribute_gene(
            gene_id=gene.gene_id,
            party_id="server@local",
        )
        assert result is not None
        assert result.party_id == "server@local"
        assert result.participant_id is None

    def test_attribute_implicit_heartbeat(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")
        # Rewind heartbeat.
        genome.conn.execute(
            "UPDATE participants SET last_heartbeat = ? WHERE participant_id = ?",
            (time.time() - 60, p.participant_id),
        )
        genome.conn.commit()

        gene = make_gene(content="something")
        genome.upsert_gene(gene)
        registry.attribute_gene(gene_id=gene.gene_id, participant_id=p.participant_id)

        row = genome.conn.execute(
            "SELECT last_heartbeat FROM participants WHERE participant_id = ?",
            (p.participant_id,),
        ).fetchone()
        assert row["last_heartbeat"] > time.time() - 5


class TestGetRecentByHandle:
    def test_returns_chronological_order(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")

        gene_a = make_gene(content="oldest note")
        gene_b = make_gene(content="middle note")
        gene_c = make_gene(content="newest note")
        genome.upsert_gene(gene_a)
        genome.upsert_gene(gene_b)
        genome.upsert_gene(gene_c)

        t0 = time.time()
        registry.attribute_gene(gene_id=gene_a.gene_id, participant_id=p.participant_id, authored_at=t0 - 100)
        registry.attribute_gene(gene_id=gene_b.gene_id, participant_id=p.participant_id, authored_at=t0 - 50)
        registry.attribute_gene(gene_id=gene_c.gene_id, participant_id=p.participant_id, authored_at=t0)

        recent = registry.get_recent_by_handle("taude", limit=5)
        assert len(recent) == 3
        assert recent[0]["gene_id"] == gene_c.gene_id
        assert recent[1]["gene_id"] == gene_b.gene_id
        assert recent[2]["gene_id"] == gene_a.gene_id

    def test_limit_honored(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")
        for i in range(5):
            g = make_gene(content=f"note {i}")
            genome.upsert_gene(g)
            registry.attribute_gene(gene_id=g.gene_id, participant_id=p.participant_id)

        recent = registry.get_recent_by_handle("taude", limit=2)
        assert len(recent) == 2

    def test_filters_by_handle(self, registry, genome):
        p_taude = registry.register_participant(party_id="max@local", handle="taude")
        p_laude = registry.register_participant(party_id="max@local", handle="laude")

        g1 = make_gene(content="taude note")
        g2 = make_gene(content="laude note")
        genome.upsert_gene(g1)
        genome.upsert_gene(g2)

        registry.attribute_gene(gene_id=g1.gene_id, participant_id=p_taude.participant_id)
        registry.attribute_gene(gene_id=g2.gene_id, participant_id=p_laude.participant_id)

        taude_genes = registry.get_recent_by_handle("taude")
        assert len(taude_genes) == 1
        assert "taude note" in taude_genes[0]["content_preview"]

        laude_genes = registry.get_recent_by_handle("laude")
        assert len(laude_genes) == 1
        assert "laude note" in laude_genes[0]["content_preview"]

    def test_bm25_bypass_short_text_surfaces(self, registry, genome):
        """Regression for the VS Code 1.115 broadcast failure: short notes
        must surface via the recent endpoint even when the genome has no
        other content, proving this is not a retrieval-quality path."""
        p = registry.register_participant(party_id="max@local", handle="taude")
        short_note = make_gene(content="VS Code 1.115 shipped 2026-04-08.")
        genome.upsert_gene(short_note)
        registry.attribute_gene(gene_id=short_note.gene_id, participant_id=p.participant_id)

        recent = registry.get_recent_by_handle("taude")
        assert len(recent) == 1
        assert "VS Code 1.115" in recent[0]["content_preview"]


class TestSweep:
    def test_sweep_updates_status_column(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")
        genome.conn.execute(
            "UPDATE participants SET last_heartbeat = ? WHERE participant_id = ?",
            (time.time() - IDLE_TTL_S - 10, p.participant_id),
        )
        genome.conn.commit()

        counts = registry.sweep()
        assert counts["stale"] >= 1

        row = genome.conn.execute(
            "SELECT status FROM participants WHERE participant_id = ?",
            (p.participant_id,),
        ).fetchone()
        assert row["status"] == "stale"


class TestBackgroundSweepTask:
    """Item 7 — _background_registry_sweep async helper.

    The lifespan integration is hard to unit-test in isolation, but
    the loop body itself is just `sweep() + log + sleep`. We verify
    the function exists, calls sweep(), and survives sweep() raising.
    """

    @pytest.mark.asyncio
    async def test_sweep_called_at_least_once_within_interval(self, registry, monkeypatch):
        import asyncio
        from helix_context import server as server_mod

        # Shrink the interval so the test runs in <1s
        monkeypatch.setattr(server_mod, "_REGISTRY_SWEEP_INTERVAL", 0.05)

        call_count = {"n": 0}
        original_sweep = registry.sweep

        def counting_sweep(*args, **kwargs):
            call_count["n"] += 1
            return original_sweep(*args, **kwargs)

        registry.sweep = counting_sweep  # type: ignore[method-assign]

        task = asyncio.create_task(server_mod._background_registry_sweep(registry))
        try:
            await asyncio.sleep(0.2)  # ~4 sweep cycles
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert call_count["n"] >= 1

    @pytest.mark.asyncio
    async def test_sweep_loop_survives_sweep_exception(self, registry, monkeypatch):
        import asyncio
        from helix_context import server as server_mod

        monkeypatch.setattr(server_mod, "_REGISTRY_SWEEP_INTERVAL", 0.05)

        call_count = {"n": 0}

        def angry_sweep(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("simulated sweep failure")

        registry.sweep = angry_sweep  # type: ignore[method-assign]

        task = asyncio.create_task(server_mod._background_registry_sweep(registry))
        try:
            await asyncio.sleep(0.2)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Loop should have called sweep multiple times despite each raising
        assert call_count["n"] >= 2


class TestGetAttributionsForGenes:
    """Item 6 — batch attribution lookup for /context citation enrichment."""

    def test_empty_input_returns_empty_dict(self, registry):
        assert registry.get_attributions_for_genes([]) == {}

    def test_returns_party_participant_handle_for_attributed(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")
        gene = make_gene(content="cite me")
        genome.upsert_gene(gene)
        registry.attribute_gene(gene_id=gene.gene_id, participant_id=p.participant_id)

        out = registry.get_attributions_for_genes([gene.gene_id])
        assert gene.gene_id in out
        assert out[gene.gene_id]["party_id"] == "max@local"
        assert out[gene.gene_id]["participant_id"] == p.participant_id
        assert out[gene.gene_id]["handle"] == "taude"

    def test_unattributed_genes_omitted(self, registry, genome):
        gene = make_gene(content="orphan")
        genome.upsert_gene(gene)
        out = registry.get_attributions_for_genes([gene.gene_id, "nonexistent-id"])
        assert gene.gene_id not in out
        assert "nonexistent-id" not in out
        assert out == {}

    def test_party_only_attribution_returns_null_handle(self, registry, genome):
        # Server-side ingest with party_id but no participant
        genome.conn.execute(
            "INSERT INTO parties (party_id, display_name, trust_domain, created_at) "
            "VALUES ('server@local', 'server', 'local', ?)",
            (time.time(),),
        )
        genome.conn.commit()
        gene = make_gene(content="server-authored")
        genome.upsert_gene(gene)
        registry.attribute_gene(gene_id=gene.gene_id, party_id="server@local")

        out = registry.get_attributions_for_genes([gene.gene_id])
        assert out[gene.gene_id]["party_id"] == "server@local"
        assert out[gene.gene_id]["participant_id"] is None
        assert out[gene.gene_id]["handle"] is None  # LEFT JOIN, no participant row

    def test_batch_lookup_handles_mixed(self, registry, genome):
        p = registry.register_participant(party_id="max@local", handle="taude")
        attributed = make_gene(content="attributed gene")
        orphan = make_gene(content="orphan gene")
        genome.upsert_gene(attributed)
        genome.upsert_gene(orphan)
        registry.attribute_gene(gene_id=attributed.gene_id, participant_id=p.participant_id)

        out = registry.get_attributions_for_genes([attributed.gene_id, orphan.gene_id])
        assert attributed.gene_id in out
        assert orphan.gene_id not in out


# ═══ Endpoint integration tests ═══════════════════════════════════════


class _ServerMockBackend:
    """Minimal ribosome mock matching the test_server.py pattern."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        if "compression engine" in system:
            return json.dumps({
                "codons": [{"meaning": "test_codon", "weight": 0.8, "is_exon": True}],
                "complement": "Compressed test content.",
                "promoter": {
                    "domains": ["test"],
                    "entities": ["TestEntity"],
                    "intent": "test",
                    "summary": "Test content for registry tests",
                },
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
    app.state.helix.ribosome.backend = _ServerMockBackend()
    with TestClient(app) as c:
        yield c


class TestRegisterEndpoint:
    def test_register_happy_path(self, client):
        resp = client.post("/sessions/register", json={
            "party_id": "max@local",
            "handle": "taude",
            "workspace": "/f/Projects/Education",
            "capabilities": ["ingest", "query"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["party_id"] == "max@local"
        assert data["participant_id"]
        assert data["heartbeat_interval_s"] > 0
        assert data["ttl_s"] > data["heartbeat_interval_s"]

    def test_register_missing_fields_returns_400(self, client):
        resp = client.post("/sessions/register", json={"party_id": "max@local"})
        assert resp.status_code == 400


class TestHeartbeatEndpoint:
    def test_heartbeat_happy_path(self, client):
        reg = client.post("/sessions/register", json={
            "party_id": "max@local",
            "handle": "taude",
        }).json()
        pid = reg["participant_id"]

        resp = client.post(f"/sessions/{pid}/heartbeat")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["status"] == "active"

    def test_heartbeat_unknown_returns_404(self, client):
        resp = client.post("/sessions/bogus-id/heartbeat")
        assert resp.status_code == 404


class TestListEndpoint:
    def test_list_sees_registered_participant(self, client):
        client.post("/sessions/register", json={
            "party_id": "max@local",
            "handle": "taude",
        })
        client.post("/sessions/register", json={
            "party_id": "max@local",
            "handle": "laude",
        })

        resp = client.get("/sessions", params={"party_id": "max@local"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        handles = {p["handle"] for p in data["participants"]}
        assert handles == {"taude", "laude"}


class TestIngestAttribution:
    def test_ingest_with_participant_id_writes_attribution(self, client):
        reg = client.post("/sessions/register", json={
            "party_id": "max@local",
            "handle": "taude",
        }).json()
        pid = reg["participant_id"]

        resp = client.post("/ingest", json={
            "content": "VS Code 1.115 shipped with Agents companion app",
            "content_type": "text",
            "participant_id": pid,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert data.get("attributed", 0) >= 1

    def test_ingest_without_participant_id_skips_attribution(self, client):
        resp = client.post("/ingest", json={
            "content": "untagged note",
            "content_type": "text",
        })
        assert resp.status_code == 200
        data = resp.json()
        # No `attributed` field in response when no id was provided.
        assert "attributed" not in data


class TestRecentEndpoint:
    def test_recent_returns_tagged_genes(self, client):
        reg = client.post("/sessions/register", json={
            "party_id": "max@local",
            "handle": "taude",
        }).json()
        pid = reg["participant_id"]

        client.post("/ingest", json={
            "content": "VS Code 1.115 released 2026-04-08 with Agents companion app",
            "content_type": "text",
            "participant_id": pid,
        })

        resp = client.get("/sessions/taude/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["handle"] == "taude"
        assert data["count"] >= 1
        assert any("VS Code 1.115" in g["content_preview"] for g in data["genes"])

    def test_recent_unknown_handle_returns_empty(self, client):
        resp = client.get("/sessions/nobody/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

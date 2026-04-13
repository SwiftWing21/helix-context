"""
Helix Context Server -- The cell membrane.

A FastAPI HTTP sidecar that acts as an OpenAI-compatible proxy.
Clients point their model endpoint at this server instead of Ollama directly.
Context compression happens transparently in the proxy layer.

Endpoints:
    POST /v1/chat/completions  -- proxy (primary integration)
    POST /ingest               -- manual content ingestion
    POST /context              -- Continue HTTP context provider format
    GET  /stats                -- genome and compression metrics
    GET  /health               -- ribosome model and gene count
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
import socket
from contextlib import asynccontextmanager
from typing import Dict, Optional

# Module-level stash for paused ribosome backends. Maps id(backend) →
# original complete() method. Not persisted — lost on server restart,
# which is fine because restart defaults to un-paused.
_paused_ribosomes: Dict[int, object] = {}

from .accel import json_loads

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import HelixConfig, load_config
from .context_manager import HelixContextManager
from .registry import DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_TTL_S, Registry

log = logging.getLogger("helix.server")

_CHECKPOINT_INTERVAL = 60  # seconds between background WAL checkpoints
_REGISTRY_SWEEP_INTERVAL = 60  # seconds between session registry status sweeps


def _local_attribution_defaults() -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolve OS-level 4-layer identity for trust-on-first-use attribution.

    Returns (user_handle, device, org, agent_handle):

        org           HELIX_ORG env       || 'local'
        device        HELIX_DEVICE env    || HELIX_PARTY env (legacy)
                                          || socket.gethostname()
        user_handle   HELIX_USER env      || HELIX_AGENT env (legacy fallback
                                             — when only HELIX_AGENT is set,
                                             treat it as the human handle so
                                             pre-4-layer setups don't break)
                                          || getpass.getuser()
        agent_handle  HELIX_AGENT env     || None (no AI agent — manual ingest)

    The legacy/back-compat note: pre-4-layer code overloaded HELIX_AGENT
    as the "handle" of whoever was acting (could be human or AI). When
    HELIX_USER is now also set, we honor the new split. When only
    HELIX_AGENT is set without HELIX_USER, we keep treating it as the
    handle (preserves the prior commit's behaviour) AND also surface it
    as agent_handle so the agents table picks it up.

    Any field may be None — /ingest tolerates None on every axis and
    falls through to writing whichever subset resolved cleanly.

    See docs/FEDERATION_LOCAL.md for the full design.
    """
    # Org (top layer)
    org = os.environ.get("HELIX_ORG") or "local"

    # Device (PC) — accept HELIX_DEVICE preferentially, fall back to
    # legacy HELIX_PARTY, then hostname.
    try:
        device = (
            os.environ.get("HELIX_DEVICE")
            or os.environ.get("HELIX_PARTY")
            or socket.gethostname()
        )
    except Exception:
        device = None

    # Agent (AI persona) — explicit only. None means "manual / no agent".
    agent_handle = os.environ.get("HELIX_AGENT") or None

    # User (human) — HELIX_USER wins; otherwise we have to pick one of
    # HELIX_AGENT (legacy back-compat) or OS user. Logic: if HELIX_USER
    # is set, use it. Else if HELIX_AGENT is set AND HELIX_USER is not,
    # the user must be the OS account that started the process (we can't
    # tell from env alone). Use OS user.
    try:
        user_handle = os.environ.get("HELIX_USER") or getpass.getuser()
    except Exception:
        user_handle = None

    # Normalize: lowercase, strip whitespace, sanity-cap length
    def _norm(v):
        if not v:
            return None
        s = str(v).strip().lower()[:64]
        return s or None

    return _norm(user_handle), _norm(device), _norm(org), _norm(agent_handle)


async def _background_checkpoint(helix: HelixContextManager) -> None:
    """Periodically flush WAL to main database file."""
    while True:
        await asyncio.sleep(_CHECKPOINT_INTERVAL)
        try:
            helix.genome.checkpoint("PASSIVE")
        except Exception:
            log.warning("Background WAL checkpoint failed", exc_info=True)


async def _background_registry_sweep(registry_obj) -> None:
    """Periodically sweep session registry status.

    Updates the persisted ``status`` column on participants based on
    ``last_heartbeat`` age, transitioning active -> idle -> stale -> gone
    on schedule. Hard-deletes participants whose ``gone`` state has aged
    past 7 days, NULLing their gene_attribution.participant_id while
    preserving party_id.

    The sweep is non-destructive for live data — observers can call
    list_participants() at any time and get correct status regardless
    of when the sweep last ran (live status is recomputed from
    last_heartbeat). The sweep exists so the persisted column stays
    consistent for any code that filters by it directly.
    """
    while True:
        await asyncio.sleep(_REGISTRY_SWEEP_INTERVAL)
        try:
            counts = registry_obj.sweep()
            # Only log when something interesting happened
            if counts.get("hard_deleted", 0) > 0 or counts.get("gone", 0) > 0:
                log.info("Registry sweep: %s", counts)
        except Exception:
            log.warning("Background registry sweep failed", exc_info=True)


def create_app(config: Optional[HelixConfig] = None) -> FastAPI:
    """Factory -- creates the FastAPI app with a HelixContextManager."""
    import os  # for getpid() in lifespan stamps
    from .bridge import AgentBridge

    if config is None:
        config = load_config()

    helix = HelixContextManager(config)

    # Bridge instantiated up here so the lifespan closure can capture it.
    # The /bridge/* endpoints below close over this same instance.
    bridge = AgentBridge()

    # Session registry — presence + attribution. See docs/SESSION_REGISTRY.md.
    # Reuses helix.genome.conn; the DAL operates on the same SQLite file.
    registry = Registry(helix.genome)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Startup: stamp server_state=running so observer sessions know
            a restart completed (or this is the first launch).
        Shutdown: WAL checkpoint + stamp server_state=stopped as a
            fallback for clean shutdowns (Ctrl+C, OS shutdown).
            Does NOT run under kill -9 — agents should call
            bridge.announce_restart BEFORE killing the process.
        """
        # Stamp "running" so observer sessions know a restart completed.
        try:
            bridge.write_signal("server_state", {
                "state": "running",
                "actor": "lifespan",
                "reason": None,
                "pid": os.getpid(),
                "expected_downtime_s": 0,
                "phase": "up",
            })
            log.info("Startup: server_state=running stamped (pid=%d)", os.getpid())
        except Exception:
            log.warning("Startup: failed to stamp server_state signal", exc_info=True)

        task = asyncio.create_task(_background_checkpoint(helix))
        sweep_task = asyncio.create_task(_background_registry_sweep(registry))
        yield
        task.cancel()
        sweep_task.cancel()
        for _t in (task, sweep_task):
            try:
                await _t
            except asyncio.CancelledError:
                pass
        helix.genome.checkpoint("TRUNCATE")

        # Flush token counter so lifetime totals persist across restart.
        try:
            helix.token_counter.flush()
        except Exception:
            log.warning("Token counter flush failed during shutdown", exc_info=True)

        # Belt-and-suspenders: stamp "stopped" on clean shutdown.
        try:
            bridge.write_signal("server_state", {
                "state": "stopped",
                "actor": "lifespan",
                "reason": "clean shutdown",
                "pid": os.getpid(),
                "expected_downtime_s": 0,
                "phase": "shutting_down",
            })
        except Exception:
            log.warning("Shutdown: failed to stamp server_state signal", exc_info=True)

        log.info("Shutdown: final WAL checkpoint completed")

    app = FastAPI(title="Helix Context Proxy", version="0.1.0", lifespan=lifespan)
    app.state.helix = helix  # Expose for testing
    app.state.bridge = bridge  # Expose for testing
    app.state.registry = registry  # Expose for testing

    # -- Proxy endpoint (primary integration) --------------------------

    @app.post("/v1/chat/completions")
    async def chat_proxy(request: Request, background_tasks: BackgroundTasks):
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            return JSONResponse({"error": "No messages provided"}, status_code=400)

        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            # No user message -- pass through unmodified
            return await _forward_raw(body, config, helix)

        # Step 1-5: Expression pipeline
        downstream_model = body.get("model")
        context_window = await helix.build_context_async(user_query, downstream_model=downstream_model)

        # Delta-epsilon health signal
        health = context_window.context_health
        log.info(
            "Context health: status=%s ellipticity=%.3f coverage=%.2f "
            "density=%.2f freshness=%.2f genes=%d/%d",
            health.status, health.ellipticity, health.coverage,
            health.density, health.freshness,
            health.genes_expressed, health.genes_available,
        )

        # Munge messages: inject context, apply history stripping
        body["messages"] = _munge_messages(
            messages=messages,
            expressed_context=context_window.expressed_context,
            ribosome_prompt=context_window.ribosome_prompt,
            total_genes=helix.genome.stats()["total_genes"],
            cold_start_threshold=config.genome.cold_start_threshold,
        )

        # Suppress think mode for small models — their reasoning loops
        # consume the entire output budget without producing answers.
        # Extends to qwen3:4b for extraction-heavy workloads (benchmarks,
        # agent tool-calls) where think tokens add cost without accuracy.
        downstream_model_name = body.get("model", "").lower()
        suppress_think = (
            context_window.metadata.get("moe_mode")
            or downstream_model_name.startswith("qwen3:4b")
            or downstream_model_name.startswith("qwen3:1.7b")
            or downstream_model_name.startswith("qwen3:0.6b")
        )
        if suppress_think:
            body["temperature"] = 0
            # Inject /no_think into user message for Qwen3 think suppression
            for msg in reversed(body["messages"]):
                if msg.get("role") == "user":
                    if not msg["content"].startswith("/no_think"):
                        msg["content"] = "/no_think " + msg["content"]
                    break

        if body.get("stream", False):
            return StreamingResponse(
                _stream_and_tee(body, config, helix, user_query, background_tasks),
                media_type="text/event-stream",
            )
        else:
            return await _forward_and_replicate(body, config, helix, user_query, background_tasks)

    # -- Ingest endpoint -----------------------------------------------

    @app.post("/ingest")
    async def ingest_endpoint(request: Request):
        import time as _time
        helix._last_activity_ts = _time.time()

        data = await request.json()
        content = data.get("content", "")
        content_type = data.get("content_type", "text")
        metadata = data.get("metadata")
        participant_id = data.get("participant_id")
        party_id = data.get("party_id")
        org_id = data.get("org_id")
        agent_id = data.get("agent_id")
        agent_kind = data.get("agent_kind")  # e.g. "claude-code", "gemini"
        # Trust-on-first-use OS-level 4-layer federation: when the caller
        # doesn't supply explicit IDs, derive (org, device, user, agent)
        # from env vars (HELIX_ORG / HELIX_DEVICE|HELIX_PARTY /
        # HELIX_USER / HELIX_AGENT) with safe fallbacks to socket and
        # getpass. Every gene auto-attributes across all four layers
        # without any auth infrastructure. See docs/FEDERATION_LOCAL.md.
        # Caller can disable by passing ``"local_federation": false``.
        local_federation = data.get("local_federation", True)
        if local_federation and not participant_id:
            user_handle, default_device, default_org, agent_handle = (
                _local_attribution_defaults()
            )
            effective_party = party_id or default_device
            effective_org = org_id or default_org
            try:
                # 4-layer find-or-create chain. Ordering matters because
                # each layer FK-references the one above:
                #   org → party (device) → participant (user) → agent
                if effective_org:
                    org_id = registry.local_org(effective_org)
                if user_handle and effective_party:
                    participant_id = registry.local_participant(
                        handle=user_handle,
                        party_id=effective_party,
                        org_id=org_id,
                    )
                    if not party_id:
                        party_id = effective_party
                # Agent layer is optional — only created when
                # HELIX_AGENT is set OR the caller passed agent_id
                # explicitly. NULL agent_id at attribution time means
                # "manual ingest, no AI persona involved."
                if agent_handle and participant_id and not agent_id:
                    agent_id = registry.local_agent(
                        handle=agent_handle,
                        participant_id=participant_id,
                        kind=agent_kind,
                    )
            except Exception:
                log.warning(
                    "OS-level federation failed (user=%s device=%s org=%s agent=%s)",
                    user_handle, effective_party, effective_org, agent_handle,
                    exc_info=True,
                )

        if not content:
            return JSONResponse({"error": "No content provided"}, status_code=400)

        try:
            gene_ids = await helix.ingest_async(content, content_type, metadata)
        except Exception as exc:
            log.warning("Ingest failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Ingest failed: {exc}", "gene_ids": [], "count": 0},
                status_code=422,
            )

        # Attribution — additive, never fails the ingest.
        # See docs/SESSION_REGISTRY.md + docs/FEDERATION_LOCAL.md.
        # All 4 layers (org, party, participant, agent) plumb through
        # if resolved; missing layers are written as NULL, which is the
        # natural representation of "this attribution dimension is unknown".
        attributed = 0
        if participant_id or party_id:
            for gid in gene_ids:
                try:
                    result = registry.attribute_gene(
                        gene_id=gid,
                        participant_id=participant_id,
                        party_id=party_id,
                        org_id=org_id,
                        agent_id=agent_id,
                    )
                    if result is not None:
                        attributed += 1
                except Exception:
                    log.warning(
                        "Attribution write failed for gene %s",
                        gid, exc_info=True,
                    )

        response = {"gene_ids": gene_ids, "count": len(gene_ids)}
        if participant_id or party_id:
            response["attributed"] = attributed
        return response

    # -- Context endpoint (Continue HTTP context provider format) -------

    @app.post("/context")
    async def context_endpoint(request: Request):
        import time as _time
        t0 = _time.time()
        helix._last_activity_ts = t0

        data = await request.json()
        query = data.get("query", "")
        decoder_override = data.get("decoder_mode")
        verbose = data.get("verbose", False)  # Agent-mode: include gene citations
        # Per-request cold-tier override (C.2 of B->C, 2026-04-10)
        # None  = honor [context] cold_tier_enabled config flag
        # True  = force cold-tier ON for this request
        # False = force cold-tier OFF for this request
        include_cold = data.get("include_cold")
        if include_cold is not None:
            include_cold = bool(include_cold)

        # session_context: optional dict carrying the caller's working
        # context (active_project, active_files). Plumbed through to the
        # path_key_index tier so PKI can fire on (project, key) pairs even
        # when the user's natural query doesn't restate the project name.
        # Shape:
        #   {"active_project": "helix-context",
        #    "active_files": ["helix_context/genome.py", "helix.toml"],
        #    "active_projects": ["helix-context", "cosmictasha"]}
        # All keys are optional. Unknown keys are ignored.
        session_context = data.get("session_context")
        if session_context is not None and not isinstance(session_context, dict):
            session_context = None  # ignore malformed input

        if not query:
            return JSONResponse({"error": "No query provided"}, status_code=400)

        # Per-request decoder mode override
        if decoder_override and decoder_override in ("full", "condensed", "minimal", "none"):
            from .context_manager import DECODER_MODES
            original_mode = helix._decoder_mode
            original_prompt = helix._decoder_prompt
            helix._decoder_mode = decoder_override
            helix._decoder_prompt = DECODER_MODES[decoder_override]

        window = await helix.build_context_async(
            query,
            include_cold=include_cold,
            session_context=session_context,
        )

        # Restore original mode after request
        if decoder_override and decoder_override in ("full", "condensed", "minimal", "none"):
            helix._decoder_mode = original_mode
            helix._decoder_prompt = original_prompt

        health = window.context_health
        latency_ms = round((_time.time() - t0) * 1000, 1)

        # Build base response (Continue-compatible)
        response = {
            "name": "Helix Genome Context",
            "description": (
                f"{health.genes_expressed} genes expressed, "
                f"{window.compression_ratio:.1f}x compression, "
                f"health={health.status} (Δε={health.ellipticity:.2f})"
            ),
            "content": window.expressed_context,
            "context_health": health.model_dump(),
        }

        # Agent-mode fields: structured metadata for programmatic use
        # Always included (low cost, high value for agents)
        try:
            scores = helix.genome.last_query_scores or {}
            # Fetch source_id for expressed genes for citation
            gene_ids = window.expressed_gene_ids or []
            citations = []
            if gene_ids:
                cur = helix.genome.read_conn.cursor()
                placeholders = ",".join("?" * len(gene_ids))
                rows = cur.execute(
                    f"SELECT gene_id, source_id, promoter FROM genes "
                    f"WHERE gene_id IN ({placeholders})",
                    gene_ids,
                ).fetchall()
                row_map = {r["gene_id"]: r for r in rows}

                # Session registry citation enrichment (item 6 of SESSION_REGISTRY.md):
                # batch-resolve attribution for the expressed genes so each
                # citation can carry authored_by_party / authored_by_handle
                # when available. Soft-fails — citations still render without
                # attribution if the registry is unreachable.
                attribution_map: dict = {}
                try:
                    attribution_map = registry.get_attributions_for_genes(gene_ids)
                except Exception:
                    log.debug("Citation attribution lookup failed", exc_info=True)

                for gid in gene_ids:
                    r = row_map.get(gid)
                    if r is None:
                        continue
                    citation = {
                        "gene_id": gid,
                        "source": r["source_id"] or "",
                        "score": round(scores.get(gid, 0.0), 3),
                    }
                    attribution = attribution_map.get(gid)
                    if attribution:
                        citation["authored_by_party"] = attribution.get("party_id")
                        if attribution.get("handle"):
                            citation["authored_by_handle"] = attribution["handle"]
                    if verbose:
                        # Include promoter tags for deeper inspection
                        try:
                            from .accel import parse_promoter
                            prom = parse_promoter(r["promoter"]) if r["promoter"] else None
                            if prom:
                                citation["domains"] = prom.domains[:5]
                                citation["entities"] = prom.entities[:5]
                        except Exception:
                            pass
                    citations.append(citation)

            # Actionable recommendation for the agent based on health
            if health.status == "aligned":
                recommendation = "trust"
                hint = "Context is well-grounded. Use directly."
            elif health.status == "sparse":
                recommendation = "verify"
                hint = "Context has gaps. Verify specific values before acting on them."
            elif health.status == "stale":
                recommendation = "refresh"
                hint = "Expressed genes are outdated. Re-ingest source files or verify from disk."
            else:  # denatured
                recommendation = "reread_raw"
                hint = "Context is unreliable. Read raw files instead of trusting the genome."

            response["agent"] = {
                "recommendation": recommendation,
                "hint": hint,
                "citations": citations,
                "latency_ms": latency_ms,
                "total_tokens_est": window.total_estimated_tokens,
                "compression_ratio": round(window.compression_ratio, 2),
                "moe_mode": window.metadata.get("moe_mode", False),
                "budget_tier": window.metadata.get("budget_tier", "broad"),
                "budget_tokens_est": window.metadata.get("budget_tokens_est", 15000),
                # C.2 of B->C: cold-tier retrieval markers
                "cold_tier_used": getattr(helix, "_last_cold_tier_used", False),
                "cold_tier_count": getattr(helix, "_last_cold_tier_count", 0),
            }
        except Exception:
            log.debug("Agent metadata enrichment failed", exc_info=True)

        return [response]

    # -- Stats endpoint ------------------------------------------------

    @app.get("/stats")
    async def stats_endpoint():
        return helix.stats()

    # -- Health history endpoint ----------------------------------------

    @app.get("/health/history")
    async def health_history_endpoint(limit: int = 50):
        return helix.genome.health_history(limit=limit)

    # -- Token metrics endpoint -----------------------------------------

    @app.get("/metrics/tokens")
    async def metrics_tokens_endpoint():
        """Session + lifetime token counters.

        Counts come from upstream `usage` fields when available, falling
        back to char-count estimation. Both exact and estimated buckets
        are reported separately. See helix_context/metrics.py.
        """
        try:
            return helix.token_counter.snapshot()
        except Exception as exc:
            log.warning("Token metrics snapshot failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Token snapshot failed: {exc}"},
                status_code=500,
            )

    # -- Session registry endpoints (see docs/SESSION_REGISTRY.md) -----

    @app.post("/sessions/register")
    async def session_register_endpoint(request: Request):
        """Register a participant under a party. Trust-on-first-use for party_id.

        Required body fields: party_id, handle.
        Optional: workspace, pid, capabilities (list), metadata (dict), display_name.
        """
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        party_id = data.get("party_id")
        handle = data.get("handle")
        if not party_id or not handle:
            return JSONResponse(
                {"error": "party_id and handle are required"},
                status_code=400,
            )

        try:
            participant = registry.register_participant(
                party_id=party_id,
                handle=handle,
                workspace=data.get("workspace"),
                pid=data.get("pid"),
                capabilities=data.get("capabilities"),
                metadata=data.get("metadata"),
                display_name=data.get("display_name"),
            )
        except Exception as exc:
            log.warning("Session register failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Registration failed: {exc}"},
                status_code=500,
            )

        return {
            "participant_id": participant.participant_id,
            "party_id": participant.party_id,
            "registered_at": participant.started_at,
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "ttl_s": DEFAULT_TTL_S,
        }

    @app.post("/sessions/{participant_id}/heartbeat")
    async def session_heartbeat_endpoint(participant_id: str):
        """Refresh last_heartbeat for a participant. Returns 404 if unknown."""
        result = registry.heartbeat(participant_id)
        if result is None:
            return JSONResponse(
                {"error": "Unknown participant_id — please re-register"},
                status_code=404,
            )
        ttl_remaining_s, status = result
        return {
            "ok": True,
            "ttl_remaining_s": ttl_remaining_s,
            "status": status,
        }

    @app.get("/sessions")
    async def session_list_endpoint(
        party_id: Optional[str] = None,
        status: str = "active",
        workspace: Optional[str] = None,
    ):
        """List participants. Filters: party_id, status, workspace prefix."""
        try:
            infos = registry.list_participants(
                party_id=party_id,
                status_filter=status,
                workspace_prefix=workspace,
            )
        except Exception as exc:
            log.warning("Session list failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"List failed: {exc}"},
                status_code=500,
            )
        return {
            "participants": [info.model_dump() for info in infos],
            "count": len(infos),
        }

    @app.get("/sessions/{handle}/recent")
    async def session_recent_endpoint(
        handle: str,
        limit: int = 10,
        party_id: Optional[str] = None,
        since: Optional[float] = None,
    ):
        """Return recent genes authored by a handle, chronologically (no BM25).

        This is the reliable broadcast channel — short notes surface here
        regardless of how much code/spec material lives in the genome.
        """
        try:
            genes = registry.get_recent_by_handle(
                handle=handle,
                limit=limit,
                party_id=party_id,
                since=since,
            )
        except Exception as exc:
            log.warning("Session recent failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Recent lookup failed: {exc}"},
                status_code=500,
            )
        return {
            "handle": handle,
            "genes": genes,
            "count": len(genes),
        }

    # -- Consolidate endpoint (session memory) ----------------------------

    @app.post("/consolidate")
    async def consolidate_endpoint():
        """Trigger session memory consolidation.

        Distills the session buffer into consolidated knowledge genes,
        extracting only new facts, decisions, and discoveries.
        """
        try:
            gene_ids = await helix.consolidate_session_async()
            return {
                "facts_extracted": len(gene_ids),
                "gene_ids": gene_ids,
            }
        except Exception as exc:
            log.warning("Consolidation endpoint failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Consolidation failed: {exc}", "facts_extracted": 0, "gene_ids": []},
                status_code=500,
            )

    # -- Health endpoint -----------------------------------------------

    @app.get("/health")
    async def health_endpoint():
        ribosome_model = "unknown"
        if hasattr(helix.ribosome, "backend") and hasattr(helix.ribosome.backend, "model"):
            ribosome_model = helix.ribosome.backend.model
        elif hasattr(helix.ribosome, "ollama_ribosome"):
            ribosome_model = f"deberta+{helix.ribosome.ollama_ribosome.backend.model}"

        return {
            "status": "ok",
            "ribosome": ribosome_model,
            "genes": helix.genome.stats()["total_genes"],
            "upstream": config.server.upstream,
        }

    @app.get("/replicas")
    async def replicas_endpoint():
        if helix._replication_mgr is None:
            return {"enabled": False, "replicas": []}
        return {"enabled": True, **helix._replication_mgr.status()}

    @app.post("/replicas/sync")
    async def replicas_sync_endpoint():
        if helix._replication_mgr is None:
            return {"synced": 0, "error": "replication not configured"}
        synced = helix._replication_mgr.sync_now()
        return {"synced": synced}

    # ── Admin: genome management ────────────────────────────────

    @app.post("/admin/refresh")
    async def admin_refresh():
        """Reopen genome connection to see external changes (deletions, thinning)."""
        helix.genome.refresh()
        new_count = helix.genome.stats()["total_genes"]
        return {"refreshed": True, "genes": new_count}

    @app.post("/admin/vacuum")
    async def admin_vacuum():
        """Reclaim free pages from the genome database.

        Runs VACUUM to compact the SQLite file after thinning, compaction,
        or large-scale deletions. Blocks all writers during the operation —
        run during maintenance windows. Returns before/after sizes.
        """
        try:
            result = helix.genome.vacuum()
            return {"ok": True, **result}
        except Exception as exc:
            log.warning("VACUUM failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"ok": False, "error": str(exc)},
                status_code=500,
            )

    @app.post("/admin/kv-backfill")
    async def admin_kv_backfill():
        """Run CPU regex KV extraction on genes missing key_values."""
        import re as _re
        from .accel import json_dumps, json_loads
        cur = helix.genome.conn.cursor()
        rows = cur.execute(
            "SELECT gene_id, content FROM genes "
            "WHERE key_values IS NULL OR key_values = '[]' OR key_values = 'null'"
        ).fetchall()
        if not rows:
            return {"backfilled": 0, "total": helix.genome.stats()["total_genes"]}

        patterns = [
            _re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*["\']([^"\'\n]{1,100})["\']', _re.MULTILINE),
            _re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(\d+(?:\.\d+)?)\s*$', _re.MULTILINE),
            _re.compile(r'"([a-z_]\w*)":\s*["\']?([^,}"\'\n]{1,80})["\']?'),
            _re.compile(r'(?:\*\*|[-*])\s*([A-Za-z ]{2,30})(?:\*\*)?:\s*(.{1,80})'),
        ]
        updated = 0
        for row in rows:
            content = row["content"][:3000]
            kvs = set()
            for pat in patterns:
                for match in pat.finditer(content):
                    g = match.groups()
                    if len(g) == 2 and g[0] and g[1]:
                        kvs.add(f"{g[0].strip()[:40]}={g[1].strip()[:80]}")
            cur.execute(
                "UPDATE genes SET key_values = ? WHERE gene_id = ?",
                (json_dumps(sorted(kvs)[:15]), row["gene_id"]),
            )
            updated += 1
        helix.genome.conn.commit()
        return {"backfilled": updated, "total": helix.genome.stats()["total_genes"]}

    @app.post("/admin/compact")
    async def admin_compact(dry_run: bool = False, density_threshold: float = 0.3, access_threshold: int = 5):
        """Run compaction sweep: demote low-density genes to compressed tiers."""
        result = helix.genome.compact_genome(
            density_threshold=density_threshold,
            access_threshold=access_threshold,
            dry_run=dry_run,
        )
        return result

    @app.post("/admin/checkpoint")
    async def admin_checkpoint(mode: str = "PASSIVE"):
        """Force a WAL checkpoint."""
        helix.genome.checkpoint(mode)
        return {"checkpointed": True, "mode": mode}

    @app.post("/admin/ribosome/pause")
    async def admin_ribosome_pause():
        """
        Disable the ribosome's LLM calls without unloading or restarting anything.

        Monkey-patches ``backend.complete()`` on the live Ribosome instance to
        raise a RuntimeError. The existing fallback paths in ``replicate()``
        and ``pack()`` already catch this and produce minimal genes from the
        raw exchange, so ``learn()`` stays fully functional — it just skips
        the LLM pass.

        Use case: when something else (e.g., a concurrent benchmark) needs
        GPU VRAM and you want to unload the ribosome model from Ollama
        without Helix re-triggering a load on the next ``learn()`` call.

        Pair with:
            curl -X POST localhost:11434/api/generate \
                 -d '{"model": "<ribosome-model>", "keep_alive": 0}'

        Resume with ``POST /admin/ribosome/resume``.
        """
        backend = helix.ribosome.backend
        backend_id = id(backend)
        if backend_id in _paused_ribosomes:
            return {
                "paused": True,
                "already": True,
                "model": getattr(backend, "model", "unknown"),
            }

        _paused_ribosomes[backend_id] = backend.complete

        def _raise_paused(*args, **kwargs):
            raise RuntimeError(
                "Ribosome paused by /admin/ribosome/pause — "
                "learn() fallback path engaged"
            )

        backend.complete = _raise_paused
        log.info(
            "Ribosome backend paused (model=%s). LLM calls will raise.",
            getattr(backend, "model", "unknown"),
        )
        return {
            "paused": True,
            "model": getattr(backend, "model", "unknown"),
            "hint": (
                "LLM calls will raise. learn() builds minimal genes from "
                "raw exchange. Resume with POST /admin/ribosome/resume."
            ),
        }

    @app.post("/admin/ribosome/resume")
    async def admin_ribosome_resume():
        """Restore the ribosome backend after /admin/ribosome/pause."""
        backend = helix.ribosome.backend
        backend_id = id(backend)
        if backend_id not in _paused_ribosomes:
            return {"resumed": False, "reason": "not paused"}

        backend.complete = _paused_ribosomes.pop(backend_id)
        log.info(
            "Ribosome backend resumed (model=%s)",
            getattr(backend, "model", "unknown"),
        )
        return {
            "resumed": True,
            "model": getattr(backend, "model", "unknown"),
        }

    @app.get("/admin/ribosome/status")
    async def admin_ribosome_status():
        """Check whether the ribosome is currently paused."""
        backend = helix.ribosome.backend
        return {
            "paused": id(backend) in _paused_ribosomes,
            "model": getattr(backend, "model", "unknown"),
            "backend_type": type(backend).__name__,
        }

    @app.post("/admin/shutdown")
    async def admin_shutdown(request: Request):
        """Graceful shutdown — stamps the signal file and raises SIGINT.

        Complements /admin/announce_restart for the case where the
        caller wants helix to go DOWN (not restart). After this returns,
        the server begins its lifespan shutdown sequence (WAL checkpoint,
        bridge state stamp, token metrics flush) and then exits.

        Body fields (all optional):
            actor    — who is asking the server to stop (e.g. "launcher", "taude")
            reason   — short human string for the signal + log

        Returns 200 immediately after firing SIGINT on the current PID.
        The actual shutdown happens asynchronously as uvicorn processes
        the signal. Callers that need to wait for the port to free up
        should poll GET /stats until connection refused.
        """
        import os as _os
        import signal as _signal

        try:
            data = await request.json()
        except Exception:
            data = {}
        actor = data.get("actor") or "unknown"
        reason = data.get("reason") or "manual shutdown"

        # Stamp the signal file so observers see the clean shutdown before
        # the lifespan hook fires.
        try:
            bridge.write_signal("server_state", {
                "state": "stopped",
                "actor": actor,
                "reason": reason,
                "pid": _os.getpid(),
                "expected_downtime_s": 0,
                "phase": "shutting_down",
            })
        except Exception:
            log.warning("Shutdown: failed to stamp signal", exc_info=True)

        log.info("Shutdown requested by %s: %s", actor, reason)

        # Fire SIGINT on self so uvicorn runs its graceful-shutdown path,
        # which invokes the lifespan cleanup (WAL checkpoint, token flush).
        try:
            _os.kill(_os.getpid(), _signal.SIGINT)
        except Exception:
            log.warning("SIGINT on self failed", exc_info=True)

        return {
            "shutting_down": True,
            "actor": actor,
            "reason": reason,
            "hint": "Poll GET /stats — connection refused means shutdown complete.",
        }

    @app.post("/admin/announce_restart")
    async def admin_announce_restart(request: Request):
        """
        Announce an intentional server restart to other sessions.

        Body:
            {
                "reason": "swapping ribosome model for benchmark",
                "actor": "laude",
                "expected_downtime_s": 30  (optional, default 30)
            }

        Writes a 'server_state=restarting' signal that other sessions
        polling ~/.helix/shared/signals/server_state.json can see.

        RECOMMENDED WORKFLOW (from the restarting agent):
          1. POST /admin/announce_restart with reason + actor
          2. Sleep ~750ms (let filesystem flush + observers see it)
          3. Kill the server process and restart it
          4. New server's lifespan hook stamps 'server_state=running'

        Observer sessions should read ~/.helix/shared/signals/server_state.json
        directly (no HTTP needed — the server may be down) whenever they get
        a ConnectionRefused from Helix, and interpret 'restarting' as expected.

        See docs/RESTART_PROTOCOL.md for the full protocol.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON body"},
                status_code=400,
            )

        reason = body.get("reason")
        actor = body.get("actor")
        if not reason or not actor:
            return JSONResponse(
                {"error": "Both 'reason' and 'actor' are required"},
                status_code=400,
            )

        expected_downtime_s = int(body.get("expected_downtime_s", 30))

        try:
            import os as _os
            bridge.announce_restart(
                reason=reason,
                actor=actor,
                expected_downtime_s=expected_downtime_s,
                pid=_os.getpid(),
            )
        except Exception as exc:
            log.warning("announce_restart failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Announce failed: {exc}"},
                status_code=500,
            )

        log.info(
            "Restart announced by %s: %s (expected_downtime=%ds)",
            actor, reason, expected_downtime_s,
        )
        return {
            "announced": True,
            "actor": actor,
            "reason": reason,
            "expected_downtime_s": expected_downtime_s,
            "hint": "Sleep ~750ms before killing the server to let observers see the signal.",
        }

    @app.get("/admin/components")
    async def admin_components():
        """Return the list of active subsystems with running/idle status.

        Feeds the launcher's tools panel (docs/LAUNCHER.md). A component
        is 'running' if helix has processed a /ingest or /context call
        in the last 60 seconds, else 'idle'. Components that are not
        configured / not loaded are omitted from the list entirely,
        matching the 'only active/online' display rule.
        """
        import time as _time
        idle_threshold_s = 60.0
        age = _time.time() - getattr(helix, "_last_activity_ts", 0.0)
        active_status = "running" if age < idle_threshold_s else "idle"

        components = []

        # Ribosome — always loaded (required). Decoder: pack/splice/replicate.
        ribosome_backend = "unknown"
        if hasattr(helix.ribosome, "backend") and hasattr(helix.ribosome.backend, "model"):
            ribosome_backend = helix.ribosome.backend.model
        components.append({
            "name": "ribosome",
            "kind": "decoder",
            "status": active_status,
            "backend": ribosome_backend,
        })

        # ΣĒMA codec — encoder, optional (loaded if sentence-transformers available).
        if getattr(helix, "_sema_codec", None) is not None:
            components.append({
                "name": "sema",
                "kind": "encoder",
                "status": active_status,
            })

        # CPU tagger — encoder, optional (spaCy-based, config-gated).
        if getattr(helix, "_cpu_tagger", None) is not None:
            components.append({
                "name": "cpu_tagger",
                "kind": "encoder",
                "status": active_status,
            })

        # SPLADE inverted index — encoder, optional (config flag).
        if getattr(helix.genome, "_splade_enabled", False):
            components.append({
                "name": "splade",
                "kind": "encoder",
                "status": active_status,
            })

        # Entity graph — encoder, optional (config flag).
        if getattr(helix.genome, "_entity_graph_enabled", False):
            components.append({
                "name": "entity_graph",
                "kind": "encoder",
                "status": active_status,
            })

        # Headroom bridge — decoder, optional (loaded if [codec] extra installed).
        try:
            from .headroom_bridge import is_headroom_available
            if is_headroom_available():
                components.append({
                    "name": "headroom",
                    "kind": "decoder",
                    "status": active_status,
                })
        except Exception:
            pass

        return {
            "components": components,
            "count": len(components),
            "last_activity_s_ago": round(age, 1),
            "idle_threshold_s": idle_threshold_s,
        }

    @app.post("/admin/sema/rebuild")
    async def admin_sema_rebuild():
        """Force-rebuild the ΣĒMA vector cache from the current genome state.

        Useful after bulk ingest or external DB changes when the cache
        would otherwise stay stale until the next upsert invalidates it.
        """
        helix.genome.invalidate_sema_cache()
        helix.genome._build_sema_cache()
        cache = helix.genome._sema_cache
        return {
            "rebuilt": True,
            "vectors": len(cache["gene_ids"]) if cache else 0,
            "memory_kb": (cache["matrix"].nbytes // 1024) if cache else 0,
        }

    @app.post("/admin/reload")
    async def admin_reload():
        """
        Hot-reload server runtime state without killing the process.

        What this refreshes:
          - helix.toml config (ports, thresholds, budget, model routing)
          - Genome WAL snapshot (see external writes)
          - ΣĒMA vector cache (rebuild from current genome)
          - last_query_scores (clear stale per-query state)

        What this does NOT do:
          - Reload Python code (needs process restart)
          - Reconnect the write DB connection (read conn refresh only)
          - Rebuild the ribosome backend (model stays loaded)

        Use /admin/reload for config/data changes; restart the process
        for code changes.
        """
        changes = {}

        # 1. Reload config from helix.toml
        try:
            from .config import load_config
            new_config = load_config()
            old_budget = helix.config.budget.max_genes_per_turn
            new_budget = new_config.budget.max_genes_per_turn
            helix.config = new_config
            if old_budget != new_budget:
                changes["max_genes_per_turn"] = {"old": old_budget, "new": new_budget}
            else:
                changes["config"] = "reloaded (no visible changes)"
        except Exception as exc:
            changes["config_error"] = str(exc)[:200]

        # 2. Refresh genome snapshot (see external WAL state)
        try:
            helix.genome.refresh()
            total = helix.genome.stats().get("total_genes", 0)
            changes["genome_genes"] = total
        except Exception as exc:
            changes["genome_error"] = str(exc)[:200]

        # 3. Rebuild ΣĒMA vector cache
        try:
            helix.genome.invalidate_sema_cache()
            helix.genome._build_sema_cache()
            cache = helix.genome._sema_cache
            if cache:
                changes["sema_vectors"] = len(cache["gene_ids"])
        except Exception as exc:
            changes["sema_error"] = str(exc)[:200]

        # 4. Clear last_query_scores (stale per-query state)
        helix.genome.last_query_scores = {}

        log.info("Admin reload complete: %s", changes)
        return {"reloaded": True, "changes": changes}

    # ── Bridge: shared memory between AI assistants ────────────
    # (AgentBridge instance created at top of create_app for lifespan capture)

    @app.get("/bridge/status")
    async def bridge_status():
        signals = bridge.list_signals()
        inbox_count = len(list(bridge.inbox.iterdir())) if bridge.inbox.exists() else 0
        return {
            "shared_dir": str(bridge.shared_dir),
            "inbox_pending": inbox_count,
            "signals": signals,
        }

    @app.post("/bridge/collect")
    async def bridge_collect():
        """Collect inbox files and ingest into genome."""
        items = bridge.collect_inbox()
        gene_ids = []
        for item in items:
            try:
                ids = helix.ingest(
                    item["content"],
                    content_type="text",
                    metadata={"path": f"__bridge_{item['source']}__"},
                )
                gene_ids.extend(ids)
            except Exception:
                log.warning("Bridge ingest failed for %s", item["path"], exc_info=True)

        # Update shared context
        bridge.update_shared_context(helix.stats())
        return {"collected": len(items), "genes_created": len(gene_ids)}

    @app.post("/bridge/signal")
    async def bridge_signal(request: Request):
        body = await request.json()
        name = body.get("name", "unnamed")
        data = body.get("data", {})
        bridge.write_signal(name, data)
        return {"ok": True, "signal": name}

    return app


# -- Message munging ---------------------------------------------------

def _munge_messages(
    messages: list[dict],
    expressed_context: str,
    ribosome_prompt: str,
    total_genes: int,
    cold_start_threshold: int,
) -> list[dict]:
    """
    Inject expressed context into the system message.
    Apply history stripping based on genome maturity (Fix 3).

    Fix 3 (cold-start bootstrap):
        If total_genes < threshold, retain the last 2 conversation turns
        alongside the current turn. Once the genome matures, strip all
        prior turns -- the genome covers them.
    """
    if not messages:
        return messages

    # Find or create system message
    system_msg = None
    other_messages = []
    for msg in messages:
        if msg.get("role") == "system" and system_msg is None:
            system_msg = dict(msg)  # copy
        else:
            other_messages.append(msg)

    if system_msg is None:
        system_msg = {"role": "system", "content": ""}

    # Append context to system message (never overwrite user's custom prompts)
    system_msg["content"] = (
        f"{system_msg['content']}\n\n{ribosome_prompt}\n\n{expressed_context}"
        if system_msg["content"].strip()
        else f"{ribosome_prompt}\n\n{expressed_context}"
    )

    result = [system_msg]

    # Current user message (always keep)
    current_turn = other_messages[-1] if other_messages else None

    if total_genes < cold_start_threshold:
        # Cold start: keep last 2 turns + current for continuity
        history_window = other_messages[-3:-1] if len(other_messages) > 2 else other_messages[:-1]
        result.extend(history_window)
    # else: strip all history -- genome covers it

    if current_turn:
        result.append(current_turn)

    return result


# -- Streaming proxy with SSE tee -------------------------------------

async def _stream_and_tee(
    body: dict,
    config: HelixConfig,
    helix: HelixContextManager,
    user_query: str,
    background_tasks: BackgroundTasks,
):
    """
    Stream chunks from upstream to client while accumulating the
    full response for background replication.
    """
    accumulated: list[str] = []
    captured_usage: Optional[dict] = None

    async with httpx.AsyncClient(timeout=config.server.upstream_timeout) as client:
        async with client.stream(
            "POST",
            f"{config.server.upstream}/v1/chat/completions",
            json=body,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    yield "\n"
                    continue

                # Forward to client immediately
                yield f"{line}\n"

                # Parse SSE data for accumulation (orjson when available)
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        chunk = json_loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated.append(content)
                        # Capture usage from any chunk that includes it
                        # (modern Ollama / OpenAI with stream_options.include_usage=true).
                        chunk_usage = chunk.get("usage")
                        if isinstance(chunk_usage, dict):
                            captured_usage = chunk_usage
                    except (ValueError, TypeError):
                        pass

    # Stream is complete -- fire background replication
    full_response = "".join(accumulated)
    if full_response:
        background_tasks.add_task(helix.learn, user_query, full_response)

    # Token accounting — prefer authoritative usage if upstream provided it,
    # else estimate from the user query + accumulated response.
    try:
        if not helix.token_counter.add_from_usage(captured_usage):
            from .metrics import estimate_tokens
            helix.token_counter.add(
                prompt_tokens=estimate_tokens(user_query),
                completion_tokens=estimate_tokens(full_response),
                estimated=True,
            )
    except Exception:
        log.debug("Token counter update failed (stream)", exc_info=True)


# -- Non-streaming forward --------------------------------------------

async def _forward_and_replicate(
    body: dict,
    config: HelixConfig,
    helix: HelixContextManager,
    user_query: str,
    background_tasks: BackgroundTasks,
):
    """Forward non-streaming request, then replicate."""
    async with httpx.AsyncClient(timeout=config.server.upstream_timeout) as client:
        resp = await client.post(
            f"{config.server.upstream}/v1/chat/completions",
            json=body,
        )
        data = resp.json()

    choices = data.get("choices", [])
    content = ""
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if content:
            background_tasks.add_task(helix.learn, user_query, content)

    # Token accounting — exact if usage was provided, else estimated.
    try:
        if not helix.token_counter.add_from_usage(data.get("usage")):
            from .metrics import estimate_tokens
            helix.token_counter.add(
                prompt_tokens=estimate_tokens(user_query),
                completion_tokens=estimate_tokens(content),
                estimated=True,
            )
    except Exception:
        log.debug("Token counter update failed (non-stream)", exc_info=True)

    return JSONResponse(data)


# -- Raw passthrough (no user message found) ---------------------------

async def _forward_raw(body: dict, config: HelixConfig, helix: Optional[HelixContextManager] = None):
    """Pass request through to upstream without context injection."""
    async with httpx.AsyncClient(timeout=config.server.upstream_timeout) as client:
        resp = await client.post(
            f"{config.server.upstream}/v1/chat/completions",
            json=body,
        )
        data = resp.json()

    # Token accounting if helix is wired in.
    if helix is not None:
        try:
            helix.token_counter.add_from_usage(data.get("usage"))
        except Exception:
            log.debug("Token counter update failed (raw)", exc_info=True)

    return JSONResponse(data)


# -- Entry point -------------------------------------------------------

def main():
    config = load_config()
    app = create_app(config)
    log.info("Helix Context Proxy starting on %s:%d", config.server.host, config.server.port)
    log.info("Upstream: %s", config.server.upstream)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


# Module-level app for `uvicorn helix_context.server:app --reload`
# Only create when imported by uvicorn (not when run via `python -m`)
if __name__ != "__main__":
    app = create_app()

if __name__ == "__main__":
    main()

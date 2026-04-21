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
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import HelixConfig, load_config
from .context_packet import build_context_packet, get_refresh_targets
from .context_manager import HelixContextManager
from .registry import DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_TTL_S, Registry

log = logging.getLogger("helix.server")

_CHECKPOINT_INTERVAL = 60  # seconds between background WAL checkpoints
_REGISTRY_SWEEP_INTERVAL = 60  # seconds between session registry status sweeps


def _local_timezone() -> Optional[str]:
    """Resolve the local IANA timezone name for attribution.

    Order of precedence:
      1. ``HELIX_TZ`` env var (e.g., 'America/Los_Angeles', 'Europe/Berlin')
         — the only path that guarantees an IANA name on Windows, where
         the OS exposes display names like 'Pacific Standard Time'.
      2. ``tzlocal.get_localzone_name()`` if the package is installed.
         Cross-platform, always returns IANA. Soft-import so we don't
         add a hard dependency.
      3. Stdlib ``datetime.now().astimezone().tzname()`` — returns
         abbreviation on Linux/Mac (e.g., 'PDT') and display name on
         Windows ('Pacific Daylight Time'). Not IANA, but better than
         nothing for forensic value.
      4. ``time.tzname[time.daylight]`` — last-ditch from the time module.
      5. ``'UTC'`` — final fallback so attribution writes don't fail.

    Returns the resolved name as a string. Caller can normalize at
    query time if needed (Windows display names map cleanly to IANA via
    a small lookup table).

    The IANA name is a label for a DST rule set, NOT a location. It
    tells us the longitude band + DST policy the device is on, not the
    user's city or actual coordinates. See docs/FEDERATION_LOCAL.md
    "What timezone capture actually tells us" for the honest framing.
    """
    if tz := os.environ.get("HELIX_TZ"):
        return tz.strip()[:64] or None
    try:
        from tzlocal import get_localzone_name  # type: ignore
        name = get_localzone_name()
        if name:
            return str(name)[:64]
    except Exception:
        pass
    try:
        import datetime as _dt
        name = _dt.datetime.now().astimezone().tzname()
        if name:
            return name[:64]
    except Exception:
        pass
    try:
        import time as _time
        name = _time.tzname[1 if _time.daylight else 0]
        if name:
            return name[:64]
    except Exception:
        pass
    return "UTC"


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


def _normalize_identity_token(value: Optional[str]) -> Optional[str]:
    """Normalize request-supplied identity tokens to local-tier shape."""
    if not value:
        return None
    normalized = str(value).strip().lower()[:64]
    return normalized or None


def _merge_tier_contributions(base: dict, extra: dict) -> dict:
    """Merge per-gene tier contributions without mutating inputs."""
    merged = {gid: dict(contribs) for gid, contribs in (base or {}).items()}
    for gid, contribs in (extra or {}).items():
        row = merged.setdefault(gid, {})
        for tier, score in contribs.items():
            row[tier] = row.get(tier, 0.0) + score
    return merged


def _probe_upstream(upstream_url: str, timeout_s: float = 1.0) -> Dict[str, object]:
    """Best-effort readiness probe for the configured upstream model server.

    Helix most often fronts Ollama, but we tolerate any OpenAI-compatible
    upstream by probing a few common endpoints and treating any non-5xx
    response as "reachable". This avoids false greens when the model server
    is entirely down while still handling auth-gated upstreams honestly.
    """
    base_url = (upstream_url or "").rstrip("/")
    if not base_url:
        return {
            "reachable": False,
            "detail": "No upstream URL configured.",
        }

    probes = ("/api/tags", "/v1/models", "/health")
    last_error: Optional[str] = None

    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            for probe in probes:
                try:
                    resp = client.get(f"{base_url}{probe}")
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    continue

                if resp.status_code < 500:
                    return {
                        "reachable": True,
                        "probe": probe,
                        "status_code": resp.status_code,
                    }
                last_error = f"HTTP {resp.status_code} on {probe}"
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"

    return {
        "reachable": False,
        "detail": last_error or "No upstream probe succeeded.",
    }


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

    # ── Cost-visibility startup warning (W2-B) ─────────────────────
    # Surface paid-API ribosome backends loudly so operators are
    # never surprised by metered cost. Local-first is the helix
    # promise; making a paid backend silent broke that promise
    # for hours of bench work in the 2026-04-14 session.
    _cost_class = config.ribosome.cost_class
    if _cost_class == "api+paid":
        log.warning(
            "RIBOSOME PAID-API ACTIVE: backend=%s model=%s. Every "
            "ingest/replicate/rerank call hits a metered API. Flip "
            "[ribosome] enabled=false (or switch to backend=deberta) "
            "for non-metered operation.",
            config.ribosome.effective_backend,
            config.ribosome.active_model,
        )
    elif _cost_class == "disabled":
        log.info(
            "Ribosome disabled: configured enabled=%s backend=%s",
            config.ribosome.enabled,
            config.ribosome.backend,
        )
    else:
        log.info(
            "Ribosome cost_class=%s backend=%s model=%s",
            _cost_class, config.ribosome.effective_backend, config.ribosome.active_model,
        )

    helix = HelixContextManager(config)

    # W2-B: emit the ribosome info-metric for dashboard visibility.
    # No-op if OTel is disabled.
    try:
        from .telemetry import ribosome_info_gauge
        ribosome_info_gauge().set(
            1,
            attributes={
                "backend": config.ribosome.effective_backend,
                "model": config.ribosome.active_model,
                "cost_class": _cost_class,
            },
        )
    except Exception:  # pragma: no cover - telemetry must not break startup
        pass

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

    # OpenTelemetry init (disabled unless HELIX_OTEL_ENABLED=1). Wraps
    # every FastAPI route in a span + exposes shared tracer/meter globals
    # for the tier-instrumentation emit points in genome.py / cwola.py.
    try:
        from .telemetry import setup_telemetry
        setup_telemetry(app, service_name="helix-context")
    except Exception:
        log.debug("OTel setup failed", exc_info=True)

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
        # Budget-zone signal: sum all message content tokens so the
        # pipeline can see how full the caller's window already is.
        # Computed here (not in context_manager) because messages[] is
        # a proxy-layer concept. No-op unless HELIX_BUDGET_ZONE=1.
        try:
            from .metrics import estimate_tokens as _est_tokens
            _prompt_tokens = sum(
                _est_tokens(m.get("content", "") or "") for m in messages
            )
        except Exception:
            _prompt_tokens = None
        context_window = await helix.build_context_async(
            user_query,
            downstream_model=downstream_model,
            prompt_tokens_hint=_prompt_tokens,
        )

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
        participant_handle = _normalize_identity_token(data.get("participant_handle"))
        agent_handle_override = _normalize_identity_token(data.get("agent_handle"))
        agent_kind = data.get("agent_kind")  # e.g. "claude-code", "gemini"
        # Trust-on-first-use OS-level 4-layer federation: when the caller
        # doesn't supply explicit IDs, derive (org, device, user, agent)
        # from env vars (HELIX_ORG / HELIX_DEVICE|HELIX_PARTY /
        # HELIX_USER / HELIX_AGENT) with safe fallbacks to socket and
        # getpass. Every gene auto-attributes across all four layers
        # without any auth infrastructure. See docs/FEDERATION_LOCAL.md.
        # Caller can disable by passing ``"local_federation": false``.
        local_federation = data.get("local_federation", True)
        # Per-write tz capture (5th forensic axis). Caller can override
        # by passing "authored_tz" in the body (e.g., a remote ingest
        # client may know its own tz better than the server does).
        authored_tz = data.get("authored_tz") or _local_timezone()

        # Validate content BEFORE federation writes. Previously the
        # federation block ran first, which left orphan org/party rows
        # in the registry for every empty-content request that hit the
        # endpoint. Validating here short-circuits the whole path.
        if not content or not content.strip():
            return JSONResponse(
                {"error": "No content provided"},
                status_code=400,
            )

        # Reject binary content declared as text. SQLite's TEXT column
        # silently truncates at the first NULL byte, so a caller that
        # utf-8-decodes raw bytes with errors="replace" and POSTs the
        # result produces ghost genes with zero or near-zero stored
        # content. Force callers to base64-encode binary payloads (no
        # NULLs) or use a content-type that maps to BLOB storage later.
        # See tests/diagnostics/test_file_type_ingest.py.
        if "\x00" in content:
            return JSONResponse(
                {
                    "error": (
                        "content contains NULL bytes (binary payload declared as "
                        "text). Base64-encode binary content before POSTing."
                    ),
                },
                status_code=400,
            )

        if local_federation and not participant_id:
            user_handle, default_device, default_org, agent_handle = (
                _local_attribution_defaults()
            )
            effective_user = participant_handle or user_handle
            effective_party = _normalize_identity_token(party_id) or default_device
            effective_org = _normalize_identity_token(org_id) or default_org
            effective_agent = agent_handle_override or agent_handle
            try:
                # 4-layer find-or-create chain. Ordering matters because
                # each layer FK-references the one above:
                #   org → party (device) → participant (user) → agent
                if effective_org:
                    org_id = registry.local_org(effective_org)
                if effective_user and effective_party:
                    participant_id = registry.local_participant(
                        handle=effective_user,
                        party_id=effective_party,
                        org_id=org_id,
                        timezone=authored_tz,  # device home tz (last-write-wins)
                    )
                    if not party_id:
                        party_id = effective_party
                # Agent layer is optional — only created when
                # an explicit agent handle resolves or the caller passed
                # agent_id explicitly. NULL agent_id at attribution time means
                # "manual ingest, no AI persona involved."
                if effective_agent and participant_id and not agent_id:
                    agent_id = registry.local_agent(
                        handle=effective_agent,
                        participant_id=participant_id,
                        kind=agent_kind,
                    )
            except Exception:
                log.warning(
                    "OS-level federation failed (user=%s device=%s org=%s agent=%s)",
                    effective_user, effective_party, effective_org, effective_agent,
                    exc_info=True,
                )

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
                        authored_tz=authored_tz,
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

    def _request_read_only(data: dict) -> bool:
        explicit = data.get("read_only")
        if explicit is not None:
            return bool(explicit)
        # Synthetic benches already use clean=true to mean "isolate this
        # query from prior benchmark state". Treating that as read-only by
        # default also prevents query-time graph writeback from polluting
        # later rows in the same run.
        return bool(data.get("clean", False))

    @app.post("/context")
    async def context_endpoint(request: Request):
        import time as _time
        t0 = _time.time()
        helix._last_activity_ts = t0

        data = await request.json()
        query = data.get("query", "")
        response_mode = str(
            data.get("response_mode", data.get("format", "continue"))
        ).strip().lower()
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

        # CWoLa label-logger identifiers. See cwola.py and
        # STATISTICAL_FUSION.md sect C2.
        #
        # Prior to 2026-04-13 this block passed NULL to log_query when the
        # request omitted these fields, which caused sweep_buckets to treat
        # every row as Bucket A (no re-query detectable without a session).
        # The [session] config now provides a deterministic fallback: a
        # synthetic session_id from sha1(client_ip + time_window_bucket) so
        # close-in-time same-operator requests group into coherent sessions,
        # and a default party_id for attribution. Set
        # `synthetic_session_enabled = false` in helix.toml to restore the
        # prior behavior.
        cwola_session_id = data.get("session_id")
        cwola_party_id = data.get("party_id")
        if cwola_session_id is None and config.session.synthetic_session_enabled:
            import hashlib as _hashlib
            client_ip = request.client.host if request.client else "unknown"
            window_s = max(1, config.session.synthetic_session_window_s)
            bucket_ts = int(t0 // window_s) * window_s
            cwola_session_id = "syn_" + _hashlib.sha1(
                f"{client_ip}:{bucket_ts}".encode("utf-8")
            ).hexdigest()[:12]
        if cwola_party_id is None:
            cwola_party_id = config.session.default_party_id

        # clean=true: reset per-session caches (TCM drift, intent LRU,
        # shadow pool) before this request runs. Intended for synthetic
        # benches where every query is independent of the previous one —
        # letting state accumulate across unrelated queries pollutes
        # signals like TCM that are designed for related-query coherence.
        # Real users / live sessions should leave this false (default).
        if data.get("clean", False):
            try:
                helix.reset_session_state()
            except Exception:
                log.debug("reset_session_state failed", exc_info=True)
        read_only = _request_read_only(data)

        if not query:
            return JSONResponse({"error": "No query provided"}, status_code=400)
        if response_mode not in {"continue", "packet"}:
            return JSONResponse(
                {"error": "Invalid response_mode", "allowed": ["continue", "packet"]},
                status_code=400,
            )

        if response_mode == "packet":
            try:
                max_genes = int(data.get("max_genes", config.budget.max_genes_per_turn))
            except (TypeError, ValueError):
                max_genes = config.budget.max_genes_per_turn
            max_genes = max(1, min(max_genes, 32))
            packet = build_context_packet(
                str(query),
                task_type=str(data.get("task_type", "explain") or "explain"),
                genome=helix.genome,
                max_genes=max_genes,
                now_ts=t0,
            )
            payload = packet.model_dump()
            payload["response_mode"] = "packet"
            return payload

        # Per-request decoder mode override — plumbed as a parameter to
        # avoid mutating shared singleton state and racing across
        # concurrent /context calls.
        _decoder_override = (
            decoder_override
            if decoder_override in ("full", "condensed", "minimal", "none")
            else None
        )

        # Budget-zone signal: the /context endpoint has no messages[] so
        # callers must supply prompt_tokens explicitly if they want the
        # zone cap to kick in. Missing => treated as clean/no-cap.
        _prompt_tokens_hint = data.get("prompt_tokens")
        if _prompt_tokens_hint is not None:
            try:
                _prompt_tokens_hint = int(_prompt_tokens_hint)
            except (TypeError, ValueError):
                _prompt_tokens_hint = None

        # Sprint 2 session working-set: thread session_id into the
        # pipeline so _assemble can elide already-delivered genes.
        # ignore_delivered=true bypasses elision (benches, smoke tests).
        _ignore_delivered = bool(data.get("ignore_delivered", False))

        window = await helix.build_context_async(
            query,
            include_cold=include_cold,
            session_context=session_context,
            party_id=cwola_party_id,
            prompt_tokens_hint=_prompt_tokens_hint,
            session_id=cwola_session_id,
            ignore_delivered=_ignore_delivered,
            read_only=read_only,
            decoder_override=_decoder_override,
        )

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

            # Activation profile: per-tier score breakdown for the
            # genes that made the top-k cut. Used by the skill activation
            # profiler bench to visualize WHICH retrieval signals fired
            # for which query shapes. Only populated when verbose=true to
            # avoid bloating responses for non-debug callers.
            if verbose:
                try:
                    tier_contrib = getattr(helix.genome, "last_tier_contributions", {}) or {}
                    expressed_ids = set(window.expressed_gene_ids or [])
                    activation = {
                        gid: contribs
                        for gid, contribs in tier_contrib.items()
                        if gid in expressed_ids
                    }
                    # Aggregate: sum of contributions per tier across
                    # expressed genes — gives the "what fired and how much"
                    # heatmap row for this query.
                    tier_totals: dict = {}
                    for contribs in activation.values():
                        for tier, score in contribs.items():
                            tier_totals[tier] = tier_totals.get(tier, 0.0) + score
                    response["agent"]["tier_contributions"] = activation
                    response["agent"]["tier_totals"] = {
                        k: round(v, 3) for k, v in tier_totals.items()
                    }
                except Exception:
                    log.debug("Tier contribution surfacing failed", exc_info=True)
        except Exception:
            log.debug("Agent metadata enrichment failed", exc_info=True)

        # ── CWoLa label logger (STATISTICAL_FUSION sect C2) ──────────
        # Writes one row per /context call regardless of verbose flag.
        # Sweeps pending buckets lazily so no background thread is
        # needed. Always soft-fails — the retrieval result must not be
        # affected by logger hiccups.
        try:
            from . import cwola
            tier_contrib_all = getattr(helix.genome, "last_tier_contributions", {}) or {}
            cwola_tier_totals: dict = {}
            for contribs in tier_contrib_all.values():
                for tier, score in contribs.items():
                    cwola_tier_totals[tier] = cwola_tier_totals.get(tier, 0.0) + score
            expressed = window.expressed_gene_ids or []
            top_gene = expressed[0] if expressed else None

            # PWPC Phase 1 enrichment — capture 20d SEMA vectors so the
            # downstream retrieval-manifold trainer has semantic signal, not
            # just normalized tier features. See docs/collab/comms/
            # REPLY_PWPC_FROM_LAUDE.md. Soft-fails: missing codec or missing
            # gene embedding → NULL columns, which the trainer already handles.
            query_sema_vec = None
            top_candidate_sema_vec = None
            try:
                codec = getattr(helix, "_sema_codec", None)
                if codec is not None:
                    query_sema_vec = codec.encode(query)
                if top_gene:
                    gene = helix.genome.get_gene(top_gene)
                    if gene is not None and gene.embedding:
                        top_candidate_sema_vec = gene.embedding
            except Exception:
                log.debug("CWoLa sema enrichment failed", exc_info=True)

            cwola.log_query(
                helix.genome.conn,
                session_id=cwola_session_id,
                party_id=cwola_party_id,
                query=query,
                tier_totals=cwola_tier_totals,
                top_gene_id=top_gene,
                ts=t0,
                query_sema=query_sema_vec,
                top_candidate_sema=top_candidate_sema_vec,
            )
            cwola.sweep_buckets(helix.genome.conn, now=_time.time())
        except Exception:
            log.debug("CWoLa log_query/sweep failed", exc_info=True)

        # OTel latency histogram — measured at the /context boundary so
        # it captures decoder mode + cold-tier + cymatics + everything
        # downstream of the retrieval. Labelled by health status so the
        # aligned/sparse/denatured/stale split is visible in Grafana.
        try:
            from .telemetry import context_latency_histogram, redact_query
            context_latency_histogram().record(
                _time.time() - t0,
                {
                    "health": health.status,
                    "budget_tier": window.metadata.get("budget_tier", "broad"),
                    "cold_tier_used": str(getattr(helix, "_last_cold_tier_used", False)),
                },
            )
        except Exception:
            # Promoted to warning so silent histogram failures surface.
            # /context latency is the primary user-visible health metric;
            # if it disappears the dashboard goes blind.
            log.warning("OTel /context latency emit failed", exc_info=True)

        return [response]

    @app.post("/context/packet")
    async def context_packet_endpoint(request: Request):
        """Freshness-labeled evidence packet for agent-safe actions."""
        import time as _time

        t0 = _time.time()
        helix._last_activity_ts = t0

        data = await request.json()
        query = data.get("query", "")
        task_type = data.get("task_type", "explain")
        max_genes = data.get("max_genes", 8)
        if data.get("clean", False):
            try:
                helix.reset_session_state()
            except Exception:
                log.debug("reset_session_state failed", exc_info=True)
        read_only = _request_read_only(data)

        if not query or not str(query).strip():
            return JSONResponse({"error": "No query provided"}, status_code=400)

        try:
            max_genes = int(max_genes)
        except (TypeError, ValueError):
            max_genes = 8
        max_genes = max(1, min(max_genes, 32))

        packet = build_context_packet(
            str(query),
            task_type=str(task_type or "explain"),
            genome=helix.genome,
            max_genes=max_genes,
            now_ts=t0,
            read_only=read_only,
        )
        payload = packet.model_dump()
        payload["response_mode"] = "packet"
        return payload

    @app.post("/context/refresh-plan")
    async def context_refresh_plan_endpoint(request: Request):
        """Just the refresh-before-action plan for an agent-safe task.

        Thin convenience over ``/context/packet``: returns only the
        ``refresh_targets`` list without the full evidence items. Useful
        when the caller already has the content cached and only needs
        to decide which sources to reread.
        """
        import time as _time

        t0 = _time.time()
        helix._last_activity_ts = t0

        data = await request.json()
        query = data.get("query", "")
        task_type = data.get("task_type", "edit")
        max_genes = data.get("max_genes", 8)
        if data.get("clean", False):
            try:
                helix.reset_session_state()
            except Exception:
                log.debug("reset_session_state failed", exc_info=True)
        read_only = _request_read_only(data)

        if not query or not str(query).strip():
            return JSONResponse({"error": "No query provided"}, status_code=400)

        try:
            max_genes = int(max_genes)
        except (TypeError, ValueError):
            max_genes = 8
        max_genes = max(1, min(max_genes, 32))

        targets = get_refresh_targets(
            str(query),
            task_type=str(task_type or "edit"),
            genome=helix.genome,
            max_genes=max_genes,
            now_ts=t0,
            read_only=read_only,
        )
        return {
            "query": str(query),
            "task_type": str(task_type or "edit"),
            "refresh_targets": [t.model_dump() for t in targets],
            "response_mode": "refresh_plan",
        }

    @app.post("/fingerprint")
    async def fingerprint_endpoint(request: Request):
        """Navigation-first retrieval payload with tier scores, not content."""
        import time as _time

        t0 = _time.time()
        helix._last_activity_ts = t0

        data = await request.json()
        query = data.get("query", "")
        if not query:
            return JSONResponse({"error": "No query provided"}, status_code=400)

        profile = str(
            data.get("profile") or config.context.fingerprint_mode_profile
        ).strip().lower()
        if profile not in {"fast", "balanced", "quality"}:
            return JSONResponse(
                {"error": "Invalid profile", "allowed": ["fast", "balanced", "quality"]},
                status_code=400,
            )

        include_cold = data.get("include_cold")
        if include_cold is not None:
            include_cold = bool(include_cold)

        session_context = data.get("session_context")
        if session_context is not None and not isinstance(session_context, dict):
            session_context = None

        if data.get("clean", False):
            try:
                helix.reset_session_state()
            except Exception:
                log.debug("reset_session_state failed", exc_info=True)

        try:
            max_results = int(
                data.get("max_results", config.budget.max_fingerprints_per_turn)
            )
        except (TypeError, ValueError):
            max_results = config.budget.max_fingerprints_per_turn
        max_results = max(1, min(max_results, 200))

        # Optional score_floor — drops candidates whose post-refiner score falls
        # below the threshold. Operates on final scores (base + refiner), not
        # raw retrieval scores, so TCM/cymatics/etc. bumps count. Omitted or 0.0
        # preserves backwards-compatible behavior (no filtering).
        score_floor_raw = data.get("score_floor")
        if score_floor_raw is None:
            score_floor = 0.0
        else:
            try:
                score_floor = float(score_floor_raw)
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "score_floor must be a number"},
                    status_code=400,
                )
            if score_floor < 0:
                return JSONResponse(
                    {"error": "score_floor must be >= 0"},
                    status_code=400,
                )

        # Evaluation budget: when a score_floor is in play we need to consider
        # more candidates than max_results so truncated_by_cap can be meaningful
        # (otherwise _express would already cap at max_results and cap is a
        # no-op after floor filtering).
        if score_floor > 0:
            eval_budget = min(max(max_results * 3, 50), 200)
        else:
            eval_budget = max_results

        party_id = data.get("party_id")
        if party_id is None:
            party_id = config.session.default_party_id

        expand_query = profile in {"balanced", "quality"}
        use_harmonic = profile == "quality"
        use_sr = profile == "quality"
        use_cymatics = profile == "quality"
        use_harmonic_bin = profile == "quality"
        use_tcm = True

        try:
            expanded_query, domains, entities = helix._prepare_query_signals(
                query,
                session_context=session_context,
                expand_query=expand_query,
            )
            candidates = helix._express(
                domains,
                entities,
                eval_budget,
                query_text=query,
                include_cold=include_cold,
                party_id=party_id,
                use_harmonic=use_harmonic,
                use_sr=use_sr,
            )
            candidates, refiner_contrib = helix._apply_candidate_refiners(
                query,
                candidates,
                eval_budget,
                use_cymatics=use_cymatics,
                use_harmonic_bin=use_harmonic_bin,
                use_tcm=use_tcm,
                allow_rerank=(profile == "quality"),
            )
        except Exception as exc:
            log.warning("/fingerprint failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

        base_scores = dict(helix.genome.last_query_scores or {})
        merged_tiers = _merge_tier_contributions(
            getattr(helix.genome, "last_tier_contributions", {}) or {},
            refiner_contrib,
        )

        # Final score = base retrieval + refiner bumps (TCM etc.). Score_floor
        # applies against this, not the raw base score.
        def _final_score(gene_id: str) -> float:
            tcm_bonus = refiner_contrib.get(gene_id, {}).get("tcm", 0.0)
            return float(base_scores.get(gene_id, 0.0) + tcm_bonus)

        evaluated_total = len(candidates)
        above_floor = [g for g in candidates if _final_score(g.gene_id) >= score_floor]
        above_floor_total = len(above_floor)
        truncated = above_floor[:max_results]
        returned = len(truncated)
        filtered_by_floor = evaluated_total - above_floor_total
        truncated_by_cap = above_floor_total - returned

        attribution_map: dict = {}
        if truncated:
            try:
                attribution_map = registry.get_attributions_for_genes(
                    [g.gene_id for g in truncated]
                )
            except Exception:
                log.debug("Fingerprint attribution lookup failed", exc_info=True)

        fingerprints = []
        for rank, g in enumerate(truncated):
            path = None
            if g.promoter and g.promoter.metadata:
                path = g.promoter.metadata.get("path")
            tiers = merged_tiers.get(g.gene_id, {})
            tcm_bonus = refiner_contrib.get(g.gene_id, {}).get("tcm", 0.0)
            row = {
                "rank": rank,
                "gene_id": g.gene_id,
                "score": round(float(base_scores.get(g.gene_id, 0.0) + tcm_bonus), 4),
                "preview": (g.content or "")[:160],
                "path": path,
                "source": g.source_id or "",
                "domains": list(g.promoter.domains) if g.promoter else [],
                "entities": list(g.promoter.entities) if g.promoter else [],
                "chromatin": int(getattr(g, "chromatin", 0) or 0),
                "tier_contributions": {
                    k: round(float(v), 4) for k, v in sorted(tiers.items())
                },
            }
            attribution = attribution_map.get(g.gene_id)
            if attribution:
                row["authored_by_party"] = attribution.get("party_id")
                if attribution.get("handle"):
                    row["authored_by_handle"] = attribution["handle"]
            fingerprints.append(row)

        tier_totals: dict = {}
        for row in fingerprints:
            for tier, score in row["tier_contributions"].items():
                tier_totals[tier] = tier_totals.get(tier, 0.0) + score

        # response_hint — tell the caller when filtering/truncation mattered.
        # Accounting counts are defined over the evaluated set (not the whole
        # corpus), so the hint only speaks to what this call actually saw.
        if returned == 0 and evaluated_total > 0 and score_floor > 0:
            response_hint = (
                f"All {evaluated_total} evaluated candidates fell below "
                f"score_floor={score_floor}; consider lowering it or "
                f"refining the query."
            )
        elif truncated_by_cap > 0:
            response_hint = (
                f"{truncated_by_cap} additional candidates cleared the floor "
                f"but were truncated by max_results={max_results}; raise "
                f"max_results to see more."
            )
        elif filtered_by_floor > 0:
            response_hint = (
                f"{filtered_by_floor} evaluated candidates fell below "
                f"score_floor={score_floor}."
            )
        else:
            response_hint = "No filtering or truncation applied."

        latency_ms = round((_time.time() - t0) * 1000, 1)
        return {
            "mode": "fingerprint",
            "profile": profile,
            "query": query,
            "extracted": {
                "expanded_query": expanded_query,
                "domains": list(domains),
                "entities": list(entities),
            },
            "fingerprints": fingerprints,
            "count": len(fingerprints),
            "max_results": max_results,
            "score_floor": score_floor,
            "evaluated_total": evaluated_total,
            "above_floor_total": above_floor_total,
            "returned": returned,
            "filtered_by_floor": filtered_by_floor,
            "truncated_by_cap": truncated_by_cap,
            "response_hint": response_hint,
            "agent": {
                "recommendation": "triage",
                "hint": "Use tier fingerprints to decide which genes to fetch in full.",
                "latency_ms": latency_ms,
                "cold_tier_used": getattr(helix, "_last_cold_tier_used", False),
                "cold_tier_count": getattr(helix, "_last_cold_tier_count", 0),
                "tier_totals": {k: round(v, 4) for k, v in sorted(tier_totals.items())},
            },
        }

    # -- Stats endpoint ------------------------------------------------

    @app.get("/stats")
    async def stats_endpoint():
        # Refresh OTel gauges that represent absolute state rather than
        # event-stream metrics — cheap DB query, runs on each /stats hit.
        try:
            from .telemetry import emit_gauges_snapshot
            emit_gauges_snapshot(helix.genome)
        except Exception:
            log.debug("telemetry gauges snapshot failed", exc_info=True)
        return helix.stats()

    # -- Resonance introspection endpoint -------------------------------
    # Joint view over the four retrieval primitives: ΣĒMA prime vector,
    # cymatic spectrum, harmonic_links edges. Pure read-only — no
    # retrieval-path side effects, safe to call any time. Returns JSON
    # suitable for a notebook / dashboard to plot as a two-panel
    # resonance chart (spectrum left, ΣĒMA neighborhood right).
    @app.get("/debug/resonance")
    async def resonance_endpoint(query: str, k: int = 10, downsample: int = 64):
        import json as _json

        try:
            genome = helix.genome
            codec = getattr(helix, "_sema_codec", None)

            from .cymatics import query_spectrum, cached_gene_spectrum, resonance_score
            q_spec = query_spectrum(query)
            q_sema = codec.encode(query) if codec is not None else None

            neighbors: list = []
            if q_sema is not None:
                rows = genome.read_conn.execute(
                    "SELECT gene_id, embedding FROM genes "
                    "WHERE embedding IS NOT NULL AND chromatin < 2 "
                    "LIMIT 20000"
                ).fetchall()
                scored = []
                for r in rows:
                    try:
                        vec = _json.loads(r["embedding"])
                    except Exception:
                        continue
                    sim = codec.similarity(q_sema, vec)
                    scored.append((sim, r["gene_id"]))
                scored.sort(key=lambda x: x[0], reverse=True)
                top = scored[:k]

                for sim, gid in top:
                    g = genome.get_gene(gid)
                    if g is None:
                        continue
                    try:
                        g_spec = cached_gene_spectrum(g)
                        cym_sim = resonance_score(q_spec, g_spec)
                    except Exception:
                        cym_sim = 0.0
                    path = None
                    if g.promoter and g.promoter.metadata:
                        path = g.promoter.metadata.get("path")
                    chrom = 0
                    if g.epigenetics:
                        chrom = getattr(g.epigenetics, "chromatin", 0)
                    neighbors.append({
                        "gene_id": gid,
                        "sema_cos_sim": round(float(sim), 4),
                        "cymatic_cos_sim": round(float(cym_sim), 4),
                        "path": path,
                        "preview": (g.content or "")[:120],
                        "chromatin": chrom,
                    })

            edges: list = []
            if neighbors:
                ids = [n["gene_id"] for n in neighbors]
                placeholders = ",".join("?" * len(ids))
                edge_rows = genome.read_conn.execute(
                    f"SELECT gene_id_a, gene_id_b, weight, source FROM harmonic_links "
                    f"WHERE gene_id_a IN ({placeholders}) AND gene_id_b IN ({placeholders})",
                    (*ids, *ids),
                ).fetchall()
                for r in edge_rows:
                    edges.append({
                        "from": r[0], "to": r[1],
                        "weight": round(float(r[2]), 4),
                        "source": r[3],
                    })

            def _downsample(spec, n):
                if len(spec) <= n:
                    return [round(float(x), 4) for x in spec]
                step = len(spec) / n
                out = []
                for i in range(n):
                    lo = int(i * step)
                    hi = int((i + 1) * step)
                    chunk = spec[lo:hi] or [spec[lo]]
                    out.append(round(sum(chunk) / len(chunk), 4))
                return out

            return {
                "query": query,
                "query_sema": [round(float(x), 4) for x in q_sema] if q_sema is not None else None,
                "query_spectrum": _downsample(q_spec, downsample),
                "spectrum_bins": downsample,
                "spectrum_bins_raw": len(q_spec),
                "neighbors": neighbors,
                "edges": edges,
                "edge_count": len(edges),
                "k": k,
                "sema_available": codec is not None,
            }
        except Exception:
            # Log the full traceback server-side only; never include
            # exception text in the response body (leaks internal paths
            # and sqlite/httpx internals to callers).
            log.error(
                "/debug/resonance failed for query=%r", query, exc_info=True
            )
            raise HTTPException(status_code=500, detail="Internal error")

    # -- Debug: single-gene fetch ---------------------------------------

    @app.get("/genes/{gene_id}")
    async def gene_get_endpoint(gene_id: str):
        """Fetch a single gene by ID.

        Returns the full gene model as JSON (content, promoter tags,
        epigenetics, codons, chromatin state, embedding). 404 if unknown.

        Intended for debugging retrieval: "why did gene X rank where it
        did? what were its promoter tags?"
        """
        try:
            gene = helix.genome.get_gene(gene_id)
        except Exception as exc:
            log.warning("/genes/%s failed: %s", gene_id, exc, exc_info=True)
            return JSONResponse(
                {"error": f"Gene lookup failed: {exc}"}, status_code=500,
            )
        if gene is None:
            return JSONResponse(
                {"error": f"Unknown gene_id: {gene_id}"}, status_code=404,
            )
        return gene.model_dump()

    # -- Debug: lightweight SEMA neighbors ------------------------------

    @app.get("/debug/neighbors")
    async def neighbors_endpoint(query: str, k: int = 10):
        """Top-k SEMA neighbors for ``query`` -- lighter than /debug/resonance.

        Returns just neighbors (gene_id, sema_cos_sim, preview, path)
        without the cymatic spectrum, harmonic edges, or query SEMA
        vector. Cheapest introspection path when the caller just wants
        "which genes are closest to this query in SEMA space?".

        Read-only; safe to call anytime.
        """
        import json as _json

        try:
            rows = helix.genome.read_conn.execute(
                "SELECT gene_id, embedding FROM genes "
                "WHERE embedding IS NOT NULL AND chromatin < 2 "
                "LIMIT 20000"
            ).fetchall()
            if not rows:
                return {
                    "query": query,
                    "k": k,
                    "count": 0,
                    "neighbors": [],
                }
            codec = getattr(helix, "_sema_codec", None)
            if codec is None:
                return JSONResponse(
                    {
                        "error": "SEMA codec not available",
                        "hint": "Ingest must have populated embeddings first.",
                    },
                    status_code=503,
                )
            q_sema = codec.encode(query)
            scored: list = []
            for r in rows:
                try:
                    vec = _json.loads(r["embedding"])
                except Exception:
                    continue
                sim = codec.similarity(q_sema, vec)
                scored.append((sim, r["gene_id"]))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:k]

            neighbors: list = []
            for sim, gid in top:
                g = helix.genome.get_gene(gid)
                if g is None:
                    continue
                path = None
                if g.promoter and g.promoter.metadata:
                    path = g.promoter.metadata.get("path")
                neighbors.append({
                    "gene_id": gid,
                    "sema_cos_sim": round(float(sim), 4),
                    "preview": (g.content or "")[:160],
                    "path": path,
                })
            return {
                "query": query,
                "k": k,
                "neighbors": neighbors,
                "count": len(neighbors),
            }
        except Exception:
            # Full traceback goes to the log only; never into the HTTP
            # response body (no str(exc), no type name leak).
            log.error("/debug/neighbors failed", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    # -- Debug: context-pipeline dry run (no splice) --------------------

    @app.get("/debug/preview")
    async def preview_endpoint(
        query: str,
        max_genes: int = 12,
        profile: str = "balanced",
        score_floor: float = 0.0,
    ):
        """Dry-run the retrieval pipeline up to candidate selection.

        Runs the cheap part of the /context pipeline -- query extraction
        + multi-tier express -- but SKIPS the splice step (which is the
        expensive ribosome-heavy leg). Returns the candidate genes that
        WOULD be fed to splice, their scores, and the extracted query
        signals.

        Useful for debugging "why isn't my query surfacing gene X?"
        without paying the full /context cost. Much cheaper than
        /context itself; no ribosome calls.

        ``score_floor`` mirrors the /fingerprint knob: candidates whose
        post-refiner score falls below the threshold are filtered out.
        Accounting fields describe what was filtered vs truncated,
        defined only over the evaluated candidate set.
        """
        profile = str(profile or "balanced").strip().lower()
        if profile not in {"fast", "balanced", "quality"}:
            return JSONResponse(
                {"error": "Invalid profile", "allowed": ["fast", "balanced", "quality"]},
                status_code=400,
            )

        if score_floor < 0:
            return JSONResponse(
                {"error": "score_floor must be >= 0"},
                status_code=400,
            )

        # Same eval-budget logic as /fingerprint: expand the retrieval
        # window when a floor is in play so truncated_by_cap is
        # meaningful. Keep current behavior when floor is 0.
        if score_floor > 0:
            eval_budget = min(max(max_genes * 3, 50), 200)
        else:
            eval_budget = max_genes

        expand_query = profile in {"balanced", "quality"}
        use_harmonic = profile == "quality"
        use_sr = profile == "quality"
        use_cymatics = profile == "quality"
        use_harmonic_bin = profile == "quality"
        use_tcm = True

        try:
            expanded_query, domains, entities = helix._prepare_query_signals(
                query,
                session_context=None,
                expand_query=expand_query,
            )
            candidates = helix._express(
                domains=domains,
                entities=entities,
                max_genes=eval_budget,
                query_text=query,
                use_harmonic=use_harmonic,
                use_sr=use_sr,
            )
            candidates, refiner_contrib = helix._apply_candidate_refiners(
                query,
                candidates,
                eval_budget,
                use_cymatics=use_cymatics,
                use_harmonic_bin=use_harmonic_bin,
                use_tcm=use_tcm,
                allow_rerank=(profile == "quality"),
            )
        except Exception:
            # Full traceback goes to the log only; never into the HTTP
            # response body (no str(exc), no type name leak).
            log.error("/debug/preview failed", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

        scores = dict(helix.genome.last_query_scores or {})
        merged_tiers = _merge_tier_contributions(
            getattr(helix.genome, "last_tier_contributions", {}) or {},
            refiner_contrib,
        )

        def _final_score(gene_id: str) -> float:
            tcm_bonus = refiner_contrib.get(gene_id, {}).get("tcm", 0.0)
            return float(scores.get(gene_id, 0.0) + tcm_bonus)

        evaluated_total = len(candidates)
        above_floor = [g for g in candidates if _final_score(g.gene_id) >= score_floor]
        above_floor_total = len(above_floor)
        truncated = above_floor[:max_genes]
        returned = len(truncated)
        filtered_by_floor = evaluated_total - above_floor_total
        truncated_by_cap = above_floor_total - returned

        result = []
        for rank, g in enumerate(truncated):
            path = None
            if g.promoter and g.promoter.metadata:
                path = g.promoter.metadata.get("path")
            result.append({
                "rank": rank,
                "gene_id": g.gene_id,
                "score": round(_final_score(g.gene_id), 4),
                "preview": (g.content or "")[:160],
                "path": path,
                "domains": list(g.promoter.domains) if g.promoter else [],
                "entities": list(g.promoter.entities) if g.promoter else [],
                "chromatin": int(getattr(g, "chromatin", 0) or 0),
                "tier_contributions": {
                    k: round(float(v), 4)
                    for k, v in sorted(merged_tiers.get(g.gene_id, {}).items())
                },
            })

        if returned == 0 and evaluated_total > 0 and score_floor > 0:
            response_hint = (
                f"All {evaluated_total} evaluated candidates fell below "
                f"score_floor={score_floor}; consider lowering it or "
                f"refining the query."
            )
        elif truncated_by_cap > 0:
            response_hint = (
                f"{truncated_by_cap} additional candidates cleared the floor "
                f"but were truncated by max_genes={max_genes}; raise "
                f"max_genes to see more."
            )
        elif filtered_by_floor > 0:
            response_hint = (
                f"{filtered_by_floor} evaluated candidates fell below "
                f"score_floor={score_floor}."
            )
        else:
            response_hint = "No filtering or truncation applied."

        return {
            "query": query,
            "profile": profile,
            "extracted": {
                "expanded_query": expanded_query,
                "domains": list(domains),
                "entities": list(entities),
            },
            "candidates": result,
            "fingerprints": result,
            "count": len(result),
            "max_genes": max_genes,
            "score_floor": score_floor,
            "evaluated_total": evaluated_total,
            "above_floor_total": above_floor_total,
            "returned": returned,
            "filtered_by_floor": filtered_by_floor,
            "truncated_by_cap": truncated_by_cap,
            "response_hint": response_hint,
            "note": "Splice step skipped; these are pre-splice candidates.",
        }

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
        # Validate identity tokens before touching the registry. Mirrors
        # the /ingest NULL-byte guard (see ~line 527) and adds shape
        # checks so we don't materialize ghost participant rows for
        # whitespace-only or near-empty handles.
        if party_id is None or handle is None:
            raise HTTPException(
                status_code=400,
                detail="party_id and handle are required",
            )
        if not isinstance(party_id, str) or not isinstance(handle, str):
            raise HTTPException(
                status_code=400,
                detail="party_id and handle must be strings",
            )
        if "\x00" in party_id or "\x00" in handle:
            raise HTTPException(
                status_code=400,
                detail=(
                    "party_id/handle contain NULL bytes (binary payload "
                    "declared as text). Base64-encode binary content "
                    "before POSTing."
                ),
            )
        party_id_stripped = party_id.strip()
        handle_stripped = handle.strip()
        if not party_id_stripped or not handle_stripped:
            raise HTTPException(
                status_code=400,
                detail="party_id and handle must be non-empty",
            )
        if len(party_id_stripped) < 3 or len(handle_stripped) < 3:
            raise HTTPException(
                status_code=400,
                detail="party_id and handle must be at least 3 characters",
            )
        party_id = party_id_stripped
        handle = handle_stripped

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
    async def session_heartbeat_endpoint(
        participant_id: str,
        request: Request,
    ):
        """Refresh last_heartbeat for a participant. Returns 404 if unknown.

        Optional body (all fields optional) publishes a retrievable presence
        gene so other participants' /context queries can surface this
        participant's state:

            {
              "handle":           "laude",
              "party_id":         "swift_wing21",
              "current_focus":    "PWPC Phase 1 follow-up",
              "blocked_on":       ["batman access"],
              "in_flight":        ["v2 drilldown", "heartbeat endpoint"],
              "last_commit_hash": "aeb1f45",
              "notes":            "optional free-form markdown"
            }

        Empty body keeps the original behavior — registry heartbeat only,
        no presence gene emitted. See docs/TEAM_PRESENCE.md (when written).
        """
        result = registry.heartbeat(participant_id)
        if result is None:
            return JSONResponse(
                {"error": "Unknown participant_id — please re-register"},
                status_code=404,
            )
        ttl_remaining_s, status = result

        # Optional presence-gene emit. Body is optional + additive; we must
        # not fail the heartbeat if the gene emit chokes.
        presence_gene_id: Optional[str] = None
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict) and body:
            try:
                presence_gene_id = registry.upsert_presence_gene(
                    participant_id,
                    handle=body.get("handle"),
                    party_id=body.get("party_id"),
                    current_focus=body.get("current_focus"),
                    blocked_on=body.get("blocked_on"),
                    in_flight=body.get("in_flight"),
                    last_commit_hash=body.get("last_commit_hash"),
                    extra_notes=body.get("notes"),
                )
            except Exception:
                log.warning(
                    "presence gene upsert failed for %s",
                    participant_id, exc_info=True,
                )

        return {
            "ok": True,
            "ttl_remaining_s": ttl_remaining_s,
            "status": status,
            "presence_gene_id": presence_gene_id,
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

    # -- AI-Consumer Sprint 3: 1-hop neighborhood expand --------------
    #
    # /context/expand?gene_id=X&direction=forward|backward|sideways&k=5
    # Lets the consumer follow a thread from a known gene without a
    # full /context round-trip. Uses the existing harmonic_links graph
    # (or gene.epigenetics.co_activated_with for sideways). Filters
    # already-delivered genes when session_id is supplied.

    @app.get("/context/expand")
    async def context_expand_endpoint(
        gene_id: str,
        direction: str = "forward",
        k: int = 5,
        session_id: Optional[str] = None,
    ):
        """1-hop expand from `gene_id`. See helix_context/expand.py."""
        try:
            from . import expand as _expand
            result = _expand.expand_neighbors(
                helix.genome,
                gene_id=gene_id,
                direction=direction,
                k=max(1, min(100, int(k))),
                session_id=session_id,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            log.warning("context_expand failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Expand failed: {exc}"},
                status_code=500,
            )
        return result

    # -- AI-Consumer Sprint 2: session working-set introspection -------
    #
    # /session/{id}/manifest — returns everything the genome has shipped
    # to a given session, most-recent first. Lets a client introspect
    # what's "held in suspension" for their session so they can reason
    # about redundancy without a round-trip through /context.

    @app.get("/session/{session_id}/manifest")
    async def session_manifest_endpoint(session_id: str, limit: int = 500):
        """List gene deliveries recorded for a session.

        Response shape: {"session_id": "...", "deliveries": [...], "count": N}
        Each delivery row includes: delivery_id, gene_id, delivered_at,
        content_hash, mode, retrieval_id. Returns an empty list for
        unknown sessions — never 404s (a valid client may be asking
        about a fresh session).
        """
        try:
            from . import session_delivery as _sd
            rows = _sd.session_manifest(
                helix.genome.conn,
                session_id=session_id,
                limit=max(1, min(5000, int(limit))),
            )
        except Exception as exc:
            log.warning("session_manifest failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Session manifest lookup failed: {exc}"},
                status_code=500,
            )
        return {
            "session_id": session_id,
            "deliveries": rows,
            "count": len(rows),
        }

    # -- HITL event endpoints ------------------------------------------
    #
    # Human-In-The-Loop pause events. Storage + DAL + pydantic models
    # landed earlier (hitl_events table, registry.emit_hitl_event, etc).
    # These HTTP endpoints expose them to remote callers (in particular
    # the MCP tool surface -- helix_hitl_emit / helix_hitl_recent -- so
    # MCP hosts like Claude Code / Desktop / Antigravity can record
    # operator signals without HTTP boilerplate of their own).
    #
    # Consumer-before-producer rationale: wire the surface first. Scorer
    # modules (tone-uncertainty classifier, risk-keyword vocab) land
    # later and populate the optional chat_signals fields. Clients that
    # detect HITL events manually can emit today.

    @app.post("/hitl/emit")
    async def hitl_emit_endpoint(request: Request):
        """Record a HITL pause event.

        Required body:
          - ``pause_type``: one of ``permission_request``, ``uncertainty_check``,
            ``rollback_confirm``, ``other``

        Participant resolution (at least one required):
          - ``participant_id``: explicit participant UUID
          - ``party_id``: explicit party (used when caller knows the party
            but not a specific participant)

        Optional:
          - ``task_context`` (str): free-form task description at pause time
          - ``resolved_without_operator`` (bool): true if session self-resolved
          - ``chat_signals`` (dict): any of ``tone_uncertainty`` (0-1 float),
            ``risk_keywords`` (list[str]), ``time_since_last_risk`` (float),
            ``recoverability`` (``recoverable`` | ``uncertain`` | ``lost``)
          - ``genome_snapshot`` (dict): any of ``total_genes``,
            ``hetero_count``, ``cold_cache_size`` (all int)
          - ``metadata`` (dict): free-form JSON, stamped onto the event

        Returns ``{event_id}`` on success, ``{error}`` on failure.
        Never mutates genome state; only writes to the ``hitl_events`` table.
        """
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        pause_type = data.get("pause_type")
        if not pause_type:
            return JSONResponse(
                {"error": "pause_type is required"}, status_code=400,
            )

        try:
            event_id = registry.emit_hitl_event(
                participant_id=data.get("participant_id"),
                pause_type=pause_type,
                task_context=data.get("task_context"),
                resolved_without_operator=bool(
                    data.get("resolved_without_operator", False)
                ),
                chat_signals=data.get("chat_signals"),
                genome_snapshot=data.get("genome_snapshot"),
                metadata=data.get("metadata"),
                party_id=data.get("party_id"),
            )
        except Exception as exc:
            log.warning("HITL emit failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Emit failed: {exc}"}, status_code=500,
            )

        if event_id is None:
            return JSONResponse(
                {
                    "error": (
                        "Event not written -- unknown participant_id and no "
                        "party_id provided, or participant_id not registered."
                    )
                },
                status_code=400,
            )
        return {"event_id": event_id, "ok": True}

    @app.get("/hitl/recent")
    async def hitl_recent_endpoint(
        party_id: Optional[str] = None,
        participant_id: Optional[str] = None,
        pause_type: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 50,
    ):
        """List recent HITL events, newest first.

        All filters are optional. With none, returns the most recent
        ``limit`` events globally (default 50, capped at 500 server-side).
        """
        # Sanity cap on limit — protects against accidental "return
        # everything" calls that could return millions of rows.
        safe_limit = max(1, min(int(limit), 500))
        try:
            events = registry.get_hitl_events(
                party_id=party_id,
                participant_id=participant_id,
                pause_type=pause_type,
                since=since,
                until=until,
                limit=safe_limit,
            )
        except Exception as exc:
            log.warning("HITL recent failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Recent lookup failed: {exc}"}, status_code=500,
            )
        return {"events": events, "count": len(events)}

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
        ribosome_disabled = getattr(helix.ribosome.backend, "is_disabled_backend", False)
        ribosome_model = "disabled" if ribosome_disabled else "unknown"
        if not ribosome_disabled and hasattr(helix.ribosome, "backend") and hasattr(helix.ribosome.backend, "model"):
            ribosome_model = helix.ribosome.backend.model
        elif not ribosome_disabled and hasattr(helix.ribosome, "ollama_ribosome"):
            ribosome_model = f"deberta+{helix.ribosome.ollama_ribosome.backend.model}"

        genome_ready = True
        total_genes = 0
        try:
            total_genes = helix.genome.stats()["total_genes"]
        except Exception:
            genome_ready = False
            # Keep the detail in server-side logs only; do not leak to callers.
            log.warning("/health genome stats failed", exc_info=True)

        # Offload the sync httpx probe off the event loop so the /health
        # handler (an async def) doesn't block on network I/O.
        upstream_probe = await asyncio.to_thread(
            _probe_upstream, config.server.upstream
        )
        upstream_reachable = bool(upstream_probe.get("reachable"))
        status = "ok" if genome_ready and upstream_reachable else "degraded"

        if status == "ok":
            message = "Helix and its upstream model server answered readiness checks."
        elif not genome_ready and not upstream_reachable:
            message = "Genome stats failed and the upstream model server is unreachable."
        elif not genome_ready:
            message = "Genome stats failed; inspect the local knowledge store."
        else:
            message = "Upstream model server is unreachable; final chat proxy calls will fail."

        # Intentionally minimized response — documented contract in
        # CLAUDE.md is "ribosome model, gene count, upstream URL". We keep
        # those and omit raw error strings, cost class, and the full
        # probe dict to avoid leaking internal paths/configuration.
        return {
            "status": status,
            "message": message,
            "ribosome": ribosome_model,
            "ribosome_backend": config.ribosome.effective_backend,
            "genes": total_genes,
            "upstream": config.server.upstream,
            "upstream_reachable": upstream_reachable,
            "checks": {
                "genome_ready": genome_ready,
                "upstream_ready": upstream_reachable,
            },
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
        ribosome_paused = id(helix.ribosome.backend) in _paused_ribosomes
        ribosome_disabled = getattr(helix.ribosome.backend, "is_disabled_backend", False)

        components = []

        # Ribosome — hide it when paused so the launcher only shows
        # active/online tools, matching the panel contract.
        if not ribosome_paused and not ribosome_disabled:
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
        # Sync I/O (iterdir, file reads) — offload to avoid blocking
        # the event loop when the shared dir lives on a slow disk.
        def _collect_status() -> Dict[str, object]:
            signals = bridge.list_signals()
            inbox_count = (
                len(list(bridge.inbox.iterdir())) if bridge.inbox.exists() else 0
            )
            return {
                "shared_dir": str(bridge.shared_dir),
                "inbox_pending": inbox_count,
                "signals": signals,
            }

        return await asyncio.to_thread(_collect_status)

    @app.post("/bridge/collect")
    async def bridge_collect():
        """Collect inbox files and ingest into genome."""
        # collect_inbox() is sync file I/O — offload it.
        items = await asyncio.to_thread(bridge.collect_inbox)
        gene_ids: list = []
        for item in items:
            try:
                ids = await helix.ingest_async(
                    item["content"],
                    content_type="text",
                    metadata={"path": f"__bridge_{item['source']}__"},
                )
                gene_ids.extend(ids)
            except Exception:
                log.warning("Bridge ingest failed for %s", item["path"], exc_info=True)

        # Update shared context — sync I/O, offload.
        try:
            stats_snapshot = helix.stats()
            await asyncio.to_thread(bridge.update_shared_context, stats_snapshot)
        except Exception:
            log.warning("Bridge shared-context update failed", exc_info=True)
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

    Inspects the upstream HTTP status before the first yield; non-2xx
    responses raise HTTPException so FastAPI propagates the real status
    code + body instead of forwarding an error payload to the client
    as if it were a successful stream.
    """
    accumulated: list[str] = []
    captured_usage: Optional[dict] = None

    async with httpx.AsyncClient(timeout=config.server.upstream_timeout) as client:
        async with client.stream(
            "POST",
            f"{config.server.upstream}/v1/chat/completions",
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                # Drain the body so we can forward a readable error
                # detail upstream. aread() materializes the stream.
                try:
                    err_bytes = await resp.aread()
                    err_detail = err_bytes.decode("utf-8", errors="replace")[:2000]
                except Exception:
                    log.warning(
                        "Upstream stream error body read failed",
                        exc_info=True,
                    )
                    err_detail = f"Upstream returned HTTP {resp.status_code}"
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=err_detail,
                )
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
        if resp.status_code >= 400:
            # Propagate upstream status + body instead of forwarding
            # the error payload as HTTP 200 to the caller.
            try:
                err_detail = resp.text[:2000]
            except Exception:
                log.warning(
                    "Upstream error body decode failed",
                    exc_info=True,
                )
                err_detail = f"Upstream returned HTTP {resp.status_code}"
            raise HTTPException(
                status_code=resp.status_code,
                detail=err_detail,
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
        if resp.status_code >= 400:
            # Mirror the guard in _forward_and_replicate: surface upstream
            # error status/body instead of forwarding an error JSON as HTTP
            # 200, and avoid raising on .json() when the body is non-JSON.
            try:
                err_detail = resp.text[:2000]
            except Exception:
                log.warning(
                    "Upstream error body decode failed (raw)",
                    exc_info=True,
                )
                err_detail = f"Upstream returned HTTP {resp.status_code}"
            log.warning(
                "Upstream /v1/chat/completions returned %d: %s",
                resp.status_code,
                err_detail,
            )
            raise HTTPException(
                status_code=resp.status_code,
                detail=err_detail,
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

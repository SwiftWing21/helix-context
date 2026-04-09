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

import logging
from typing import Optional

from .accel import json_loads

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import HelixConfig, load_config
from .context_manager import HelixContextManager

log = logging.getLogger("helix.server")


def create_app(config: Optional[HelixConfig] = None) -> FastAPI:
    """Factory -- creates the FastAPI app with a HelixContextManager."""

    if config is None:
        config = load_config()

    helix = HelixContextManager(config)

    app = FastAPI(title="Helix Context Proxy", version="0.1.0")
    app.state.helix = helix  # Expose for testing

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
            return await _forward_raw(body, config)

        # Step 1-5: Expression pipeline
        context_window = await helix.build_context_async(user_query)

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
        data = await request.json()
        content = data.get("content", "")
        content_type = data.get("content_type", "text")
        metadata = data.get("metadata")

        if not content:
            return JSONResponse({"error": "No content provided"}, status_code=400)

        try:
            gene_ids = await helix.ingest_async(content, content_type, metadata)
            return {"gene_ids": gene_ids, "count": len(gene_ids)}
        except Exception as exc:
            log.warning("Ingest failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Ingest failed: {exc}", "gene_ids": [], "count": 0},
                status_code=422,
            )

    # -- Context endpoint (Continue HTTP context provider format) -------

    @app.post("/context")
    async def context_endpoint(request: Request):
        data = await request.json()
        query = data.get("query", "")
        decoder_override = data.get("decoder_mode")

        if not query:
            return JSONResponse({"error": "No query provided"}, status_code=400)

        # Per-request decoder mode override
        if decoder_override and decoder_override in ("full", "condensed", "minimal", "none"):
            from .context_manager import DECODER_MODES
            original_mode = helix._decoder_mode
            original_prompt = helix._decoder_prompt
            helix._decoder_mode = decoder_override
            helix._decoder_prompt = DECODER_MODES[decoder_override]

        window = await helix.build_context_async(query)

        # Restore original mode after request
        if decoder_override and decoder_override in ("full", "condensed", "minimal", "none"):
            helix._decoder_mode = original_mode
            helix._decoder_prompt = original_prompt
        health = window.context_health
        return [
            {
                "name": "Helix Genome Context",
                "description": (
                    f"{health.genes_expressed} genes expressed, "
                    f"{window.compression_ratio:.1f}x compression, "
                    f"health={health.status} (Δε={health.ellipticity:.2f})"
                ),
                "content": window.expressed_context,
                "context_health": health.model_dump(),
            }
        ]

    # -- Stats endpoint ------------------------------------------------

    @app.get("/stats")
    async def stats_endpoint():
        return helix.stats()

    # -- Health history endpoint ----------------------------------------

    @app.get("/health/history")
    async def health_history_endpoint(limit: int = 50):
        return helix.genome.health_history(limit=limit)

    # -- Health endpoint -----------------------------------------------

    @app.get("/health")
    async def health_endpoint():
        ribosome_model = "unknown"
        if hasattr(helix.ribosome.backend, "model"):
            ribosome_model = helix.ribosome.backend.model

        return {
            "status": "ok",
            "ribosome": ribosome_model,
            "genes": helix.genome.stats()["total_genes"],
            "upstream": config.server.upstream,
        }

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
                    except (ValueError, TypeError):
                        pass

    # Stream is complete -- fire background replication
    full_response = "".join(accumulated)
    if full_response:
        background_tasks.add_task(helix.learn, user_query, full_response)


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
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if content:
            background_tasks.add_task(helix.learn, user_query, content)

    return JSONResponse(data)


# -- Raw passthrough (no user message found) ---------------------------

async def _forward_raw(body: dict, config: HelixConfig):
    """Pass request through to upstream without context injection."""
    async with httpx.AsyncClient(timeout=config.server.upstream_timeout) as client:
        resp = await client.post(
            f"{config.server.upstream}/v1/chat/completions",
            json=body,
        )
        return JSONResponse(resp.json())


# -- Entry point -------------------------------------------------------

def main():
    config = load_config()
    app = create_app(config)
    log.info("Helix Context Proxy starting on %s:%d", config.server.host, config.server.port)
    log.info("Upstream: %s", config.server.upstream)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


# Module-level app for `uvicorn helix_context.server:app --reload`
app = create_app()

if __name__ == "__main__":
    main()

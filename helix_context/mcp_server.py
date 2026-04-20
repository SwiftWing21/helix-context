"""
MCP server for helix — exposes helix as a first-class tool inside MCP hosts.

Thin adapter: stdio JSON-RPC server that declares a handful of tools and
proxies each call to helix's HTTP API. Lets Claude Code / Claude Desktop
/ Cursor consume helix without any HTTP client boilerplate in the host.

Tools exposed:
    Retrieval / genome:
      helix_context         — main retrieval (the big one)
      helix_context_packet  — agent-safe packet with freshness labels +
                               refresh plan (per agent-context-index
                               build spec, 2026-04-17)
      helix_refresh_targets — just the reread plan for an edit/ops task
      helix_stats           — genome health + size
      helix_ingest          — add content to the genome
      helix_resonance       — four-primitive introspection chart (ΣĒMA +
                               cymatic + harmonic + neighbor set) — new in
                               2026-04-14, see server.py:/debug/resonance
      helix_consolidate     — distill the session buffer into
                               consolidated knowledge genes

    Session registry:
      helix_sessions_list   — list active participants (filter by party,
                               status, workspace)
      helix_session_recent  — genes authored by a handle, chronological

    HITL events:
      helix_hitl_emit       — record a Human-In-The-Loop pause event
      helix_hitl_recent     — query recent HITL events

    Operational:
      helix_health          — ribosome / genes / upstream readiness probe
      helix_metrics_tokens  — session + lifetime token counters
      helix_bridge_status   — federation/bridge inbox + signal state

    Introspection / debugging:
      helix_gene_get        — fetch a single gene by ID
      helix_neighbors       — top-k SEMA neighbors for a query (light)
      helix_splice_preview  — dry-run retrieval pipeline (skip splice)

    Software-vocabulary aliases (per docs/ROSETTA.md):
      helix_document_get     — alias for helix_gene_get
      helix_document_query   — alias for helix_context
      helix_document_preview — alias for helix_splice_preview
      All three are thin pass-throughs; callers should prefer the
      ``document_*`` names in new code. Legacy names remain valid.

Run (stdio transport — what MCP hosts spawn):
    python -m helix_context.mcp_server

Configure in Claude Code .mcp.json:
    {
      "mcpServers": {
        "helix-context": {
          "command": "python",
          "args": ["-m", "helix_context.mcp_server"],
          "env": {
            "HELIX_MCP_URL": "http://127.0.0.1:11437"
          }
        }
      }
    }

Env:
    HELIX_MCP_URL        - helix HTTP base URL (default http://127.0.0.1:11437)
    HELIX_MCP_TIMEOUT    - per-request timeout in seconds (default 30)
    HELIX_MCP_HANDLE     - live MCP session handle for registry presence
    HELIX_PARTY_ID       - party/device id for session presence; also the
                           ingest default when HELIX_DEVICE is unset
    HELIX_MCP_HOST       - MCP host/tool family for presence tags
    HELIX_ORG            - ingest attribution org id
    HELIX_DEVICE         - ingest attribution device/party id override
    HELIX_USER           - ingest attribution human participant handle
    HELIX_AGENT          - ingest attribution AI agent handle
    HELIX_AGENT_KIND     - optional ingest attribution agent kind
                           (defaults to HELIX_MCP_HOST when omitted)

Composition hook: Headroom already ships `codebase-memory-mcp` (manual
install, off-by-default as of 2026-04-14 per Tejas on Discord). Its
scope is the call graph — `trace_call_path(function_name, direction)`
etc — NOT compression. Composition story:

  1. User-facing: helix-mcp spawns codebase-memory-mcp as a child and
     re-exports its tools (`helix_trace_calls` etc) — user's .mcp.json
     stays one entry.
  2. Internal: helix's /context retrieval becomes a CLIENT of
     codebase-memory-mcp — call-path relevance gets added as a retrieval
     tier, invisible to the MCP host.

See note at bottom of this file for the sketch.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("helix.mcp")

HELIX_URL = os.environ.get("HELIX_MCP_URL", "http://127.0.0.1:11437").rstrip("/")
TIMEOUT_S = float(os.environ.get("HELIX_MCP_TIMEOUT", "30"))

# Stable session_id for this MCP subprocess lifetime. Used to attribute
# every `helix_context` call from this host to the same row in
# session_delivery_log, so already-delivered genes can be elided with a
# pointer stub on subsequent calls within the same MCP session. Prefer
# HELIX_MCP_HANDLE when set (hosts commonly set "laude", "raude", etc);
# otherwise fall back to "mcp-<pid>" which is still stable for one
# subprocess lifetime. Matches the _register_with_registry() handle
# scheme, so the session_id here aligns with the registry participant.
MCP_SESSION_ID = os.environ.get("HELIX_MCP_HANDLE", f"mcp-{os.getpid()}")

mcp = FastMCP("helix")


# ── HTTP helper ──────────────────────────────────────────────────────
# Keep it tiny: json-in / json-out, explicit timeout, structured errors
# that the MCP host can render instead of a crashed tool call.

def _http(method: str, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
    url = f"{HELIX_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"_raw": raw}
    except urllib.error.HTTPError as exc:
        return {
            "_error": f"HTTP {exc.code}",
            "_detail": exc.read().decode("utf-8", errors="replace")[:500],
        }
    except urllib.error.URLError as exc:
        return {
            "_error": "helix unreachable",
            "_detail": f"{exc.reason} at {url}",
            "_hint": (
                "Start `helix-launcher` (recommended) or run `helix` manually. "
                "If Helix is already running elsewhere, check HELIX_MCP_URL."
            ),
        }
    except Exception as exc:
        return {"_error": str(exc)}


def _normalize_health_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Project raw /health or transport errors into a stable status shape."""
    normalized: Dict[str, Any] = {"server": payload}

    if payload.get("_error") == "helix unreachable":
        normalized.update({
            "availability": "unavailable",
            "next_action": (
                "Run `helix-launcher` to start the canonical supervisor. "
                "If you intentionally run Helix elsewhere, update HELIX_MCP_URL."
            ),
            "message": payload.get("_detail", "Helix is unreachable."),
        })
        return normalized

    if payload.get("_error"):
        normalized.update({
            "availability": "degraded",
            "next_action": (
                "Inspect the server error details, then restart Helix with "
                "`helix-launcher` if the issue persists."
            ),
            "message": payload.get("_detail") or payload.get("_error"),
        })
        return normalized

    if payload.get("status") == "ok":
        genes = int(payload.get("genes", 0) or 0)
        next_action = "Use `helix_context` for repo questions."
        if genes == 0:
            next_action = (
                "Helix is up but the genome is empty. Ingest project content "
                "or point Helix at the intended database before relying on retrieval."
            )
        normalized.update({
            "availability": "available",
            "next_action": next_action,
            "message": "Helix answered /health successfully.",
        })
        return normalized

    if payload.get("status") in {"degraded", "unavailable"}:
        normalized.update({
            "availability": payload.get("status"),
            "next_action": (
                payload.get("message")
                or "Inspect the Helix health payload, then restart with `helix-launcher` if needed."
            ),
            "message": payload.get("message", "Helix reported a degraded health state."),
        })
        return normalized

    normalized.update({
        "availability": "degraded",
        "next_action": (
            "Helix responded, but the health payload was unexpected. "
            "Check `/health`, then restart with `helix-launcher` if needed."
        ),
        "message": "Unexpected /health payload.",
    })
    return normalized


def _normalize_identity_token(value: Optional[str]) -> Optional[str]:
    """Normalize env-driven identity tokens to Helix's local-tier shape."""
    if not value:
        return None
    normalized = str(value).strip().lower()[:64]
    return normalized or None


def _default_ingest_identity() -> Dict[str, Any]:
    """Resolve explicit ingest attribution forwarded from the MCP process.

    Unlike registry presence, Helix's HTTP /ingest route runs in another
    process and cannot see this MCP process's env vars. We therefore ship
    the resolved identity over the wire so ingests can be attributed to the
    intended org/device/user/agent chain even when Helix was launched from
    a different shell.
    """
    org_id = _normalize_identity_token(os.environ.get("HELIX_ORG"))
    party_id = _normalize_identity_token(
        os.environ.get("HELIX_DEVICE")
        or os.environ.get("HELIX_PARTY_ID")
        or os.environ.get("HELIX_PARTY")
    )
    participant_handle = _normalize_identity_token(os.environ.get("HELIX_USER"))
    agent_handle = _normalize_identity_token(
        os.environ.get("HELIX_AGENT") or os.environ.get("HELIX_MCP_HANDLE")
    )
    agent_kind = _normalize_identity_token(
        os.environ.get("HELIX_AGENT_KIND") or os.environ.get("HELIX_MCP_HOST")
    )

    payload: Dict[str, Any] = {}
    if org_id:
        payload["org_id"] = org_id
    if party_id:
        payload["party_id"] = party_id
    if participant_handle:
        payload["participant_handle"] = participant_handle
    if agent_handle:
        payload["agent_handle"] = agent_handle
    if agent_kind:
        payload["agent_kind"] = agent_kind
    return payload


# ── Tool: helix_context ──────────────────────────────────────────────
# The main retrieval path — MCP hosts call this to get a compressed
# context window for a query. Returns the same shape as /context,
# minus streaming (MCP tools are one-shot).

@mcp.tool()
def helix_context(
    query: str,
    decoder_mode: Optional[str] = None,
    downstream_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compressed context window for `query` from the helix genome.

    decoder_mode: "condensed" (default), "broad", or "dense". Controls
        how genes are unfolded into tokens. "broad" → more genes, less
        per-gene detail. "condensed" → fewer genes, more detail each.
    downstream_model: hint string so helix can size the budget for the
        target model (e.g. "claude-opus-4-6", "gpt-4").
    session_id: explicit session id for the working-set register. When
        omitted, defaults to MCP_SESSION_ID (HELIX_MCP_HANDLE or
        mcp-<pid>) so every call from this MCP subprocess is attributed
        to the same session. Pass a fresh value to isolate benches.
    """
    body: Dict[str, Any] = {"query": query}
    if decoder_mode:
        body["decoder_mode"] = decoder_mode
    if downstream_model:
        body["downstream_model"] = downstream_model
    body["session_id"] = session_id or MCP_SESSION_ID
    return _http("POST", "/context", body)


# ── Tool: helix_context_packet ───────────────────────────────────────
# Agent-safe retrieval per docs/specs/2026-04-17-agent-context-index-
# build-spec.md. Returns evidence labeled verified / stale_risk /
# needs_refresh plus explicit refresh_targets, instead of raw content.
# Use this when the agent needs to decide whether to act on retrieved
# evidence OR reread the source first.

@mcp.tool()
def helix_context_packet(
    query: str,
    task_type: str = "explain",
    max_genes: int = 8,
) -> Dict[str, Any]:
    """Freshness-labeled evidence packet for agent-safe actions.

    Returns a packet with three evidence buckets and a refresh plan:
        verified         — fresh, authoritative, coordinate-aligned
        stale_risk       — relevant but aging or weakly grounded
        refresh_targets  — concrete sources to reread before action

    task_type: "plan" | "explain" | "review" | "edit" | "debug" | "ops"
        | "quote". Higher-risk types are stricter on freshness and
        coordinate confidence — "edit" and "ops" will flag marginal
        evidence that "explain" would accept.
    max_genes: retrieval top-K (1-32). Default 8.

    Composition: freshness_score × authority × specificity gives
    live_truth; coordinate_confidence gates for "did we resolve to the
    right place." Notes include coordinate_confidence warnings when
    the retrieval may be off-target.
    """
    body: Dict[str, Any] = {
        "query": query,
        "task_type": task_type,
        "max_genes": max_genes,
    }
    return _http("POST", "/context/packet", body)


# ── Tool: helix_refresh_targets ──────────────────────────────────────

@mcp.tool()
def helix_refresh_targets(
    query: str,
    task_type: str = "edit",
    max_genes: int = 8,
) -> Dict[str, Any]:
    """Just the reread plan for a high-risk action.

    Returns refresh_targets only — skips the evidence buckets. Use this
    when the caller already has content cached and only needs to know
    which sources are stale enough that rereading is required before
    the action completes.

    Defaults to task_type="edit" because that's the usual caller.
    """
    body: Dict[str, Any] = {
        "query": query,
        "task_type": task_type,
        "max_genes": max_genes,
    }
    return _http("POST", "/context/refresh-plan", body)


# ── Tool: helix_stats ────────────────────────────────────────────────

@mcp.tool()
def helix_stats() -> Dict[str, Any]:
    """Return genome health + size stats.

    Gives gene counts, chromatin distribution, session info, current
    ribosome model. Useful as a readiness probe or to confirm the
    genome looks healthy before heavy retrieval work.
    """
    return _http("GET", "/stats")


# ── Tool: helix_ingest ───────────────────────────────────────────────

@mcp.tool()
def helix_ingest(
    content: str,
    content_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ingest raw text into the genome.

    content_type: "text" | "markdown" | "python" | "rust" | ... — see
        helix's tree_chunker for the full list. Affects how content is
        split into genes.
    metadata: optional dict stamped onto every created gene. Include
        "source_id" to make re-ingests idempotent.

    Attribution defaults come from this MCP process's env vars and are
    forwarded explicitly to Helix's 4-layer federation:
        HELIX_ORG -> org_id
        HELIX_DEVICE or HELIX_PARTY_ID -> party_id
        HELIX_USER -> participant_handle
        HELIX_AGENT or HELIX_MCP_HANDLE -> agent_handle
        HELIX_AGENT_KIND or HELIX_MCP_HOST -> agent_kind

    This keeps ingests correctly attributed even when the Helix HTTP
    server was launched from a different shell. See docs/FEDERATION_LOCAL.md.
    """
    body: Dict[str, Any] = {"content": content, "content_type": content_type}
    if metadata:
        body["metadata"] = metadata
    body.update(_default_ingest_identity())
    return _http("POST", "/ingest", body)


# ── Tool: helix_resonance ────────────────────────────────────────────

@mcp.tool()
def helix_resonance(query: str, k: int = 10, downsample: int = 64) -> Dict[str, Any]:
    """Four-primitive introspection view for `query`.

    Returns SEMA prime vector, cymatic spectrum (256 -> `downsample` bins),
    top-k SEMA neighbors with per-neighbor cymatic similarity, and the
    harmonic_links edges among those neighbors. Read-only; safe to call
    anytime without affecting retrieval state.

    Use this when you want to debug *why* a query is retrieving what it
    does, or to visualize the genome's local structure around a concept.
    """
    path = f"/debug/resonance?query={urllib.request.quote(query)}&k={k}&downsample={downsample}"
    return _http("GET", path)


# ── Tool: helix_hitl_emit ────────────────────────────────────────────
# Record a Human-In-The-Loop pause event from an MCP host. Storage and
# DAL shipped earlier (hitl_events table + registry.emit_hitl_event);
# this surface lets Claude Code / Desktop / Antigravity emit events
# without HTTP client boilerplate on their side.

@mcp.tool()
def helix_hitl_emit(
    pause_type: str,
    task_context: Optional[str] = None,
    resolved_without_operator: bool = False,
    tone_uncertainty: Optional[float] = None,
    risk_keywords: Optional[List[str]] = None,
    recoverability: Optional[str] = None,
    participant_id: Optional[str] = None,
    party_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a HITL (Human-In-The-Loop) pause event in the session registry.

    pause_type: one of "permission_request", "uncertainty_check",
        "rollback_confirm", "other". Unknown values coerce to "other".

    Chat-channel signals (optional — populate when the client's scorer
    infrastructure can compute them):
        tone_uncertainty: 0-1 proxy score of operator tone
        risk_keywords: list of trigger keywords spotted in the session
        recoverability: "recoverable" | "uncertain" | "lost"

    Participant resolution (pick the most specific you have):
        participant_id: explicit participant UUID (from /sessions/register)
        party_id: explicit party (if no participant)
        If neither is given, HELIX_PARTY_ID env var is used, or the
        default "swift_wing21" — this ensures events always land
        somewhere rather than dropping silently.

    Returns {event_id, ok: true} on success, {error: str} on failure.
    Does not mutate genome state; only writes to hitl_events.
    """
    body: Dict[str, Any] = {"pause_type": pause_type}

    if task_context:
        body["task_context"] = task_context
    if resolved_without_operator:
        body["resolved_without_operator"] = True

    chat_signals: Dict[str, Any] = {}
    if tone_uncertainty is not None:
        chat_signals["tone_uncertainty"] = tone_uncertainty
    if risk_keywords:
        chat_signals["risk_keywords"] = list(risk_keywords)
    if recoverability:
        chat_signals["recoverability"] = recoverability
    if chat_signals:
        body["chat_signals"] = chat_signals

    if participant_id:
        body["participant_id"] = participant_id

    # Default party from env so events don't drop when no participant
    # registration happened (e.g., MCP host didn't run _register_with_registry
    # or the registration failed silently).
    if not participant_id and not party_id:
        party_id = os.environ.get("HELIX_PARTY_ID", "swift_wing21")
    if party_id:
        body["party_id"] = party_id

    return _http("POST", "/hitl/emit", body)


# ── Tool: helix_hitl_recent ──────────────────────────────────────────
# Query recent HITL events -- the inverse of helix_hitl_emit. Lets
# clients ask "has this operator been flagging events recently?"
# without a separate HTTP client.

@mcp.tool()
def helix_hitl_recent(
    party_id: Optional[str] = None,
    pause_type: Optional[str] = None,
    since_ts: Optional[float] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """List recent HITL pause events, newest first.

    party_id: defaults to the HELIX_PARTY_ID env var (typically
        "swift_wing21") so calls without args scope to this session's
        party. Pass an explicit party_id to override.
    pause_type: filter to one of "permission_request", "uncertainty_check",
        "rollback_confirm", "other".
    since_ts: Unix timestamp lower-bound filter.
    limit: max events to return (server caps at 500).

    Returns {events: [...], count: int}.
    """
    if party_id is None:
        party_id = os.environ.get("HELIX_PARTY_ID", "swift_wing21")

    qs_parts = [f"party_id={urllib.request.quote(party_id)}"]
    if pause_type:
        qs_parts.append(f"pause_type={urllib.request.quote(pause_type)}")
    if since_ts is not None:
        qs_parts.append(f"since={since_ts}")
    qs_parts.append(f"limit={int(limit)}")

    return _http("GET", f"/hitl/recent?{'&'.join(qs_parts)}")


# ── Tool: helix_sessions_list ────────────────────────────────────────
# List active participants in the session registry. Lets MCP clients
# see peers -- "who else is working under this party right now?"

@mcp.tool()
def helix_sessions_list(
    party_id: Optional[str] = None,
    status: str = "active",
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """List participants from the session registry.

    party_id: scope to one party (default: all parties)
    status: "active" (default) | "stale" | "all" -- filters by last_heartbeat
    workspace: prefix match on participant workspace path

    Returns {participants: [...], count: int}. Useful for discovering
    other sibling sessions under the same party (laude, raude, gemini,
    batman, etc) without the caller needing the /sessions HTTP path.
    """
    qs_parts = []
    if party_id:
        qs_parts.append(f"party_id={urllib.request.quote(party_id)}")
    if status:
        qs_parts.append(f"status={urllib.request.quote(status)}")
    if workspace:
        qs_parts.append(f"workspace={urllib.request.quote(workspace)}")
    path = "/sessions" + (f"?{'&'.join(qs_parts)}" if qs_parts else "")
    return _http("GET", path)


# ── Tool: helix_session_recent ───────────────────────────────────────
# Genes authored by a specific handle, chronological. This is the
# reliable broadcast channel -- short notes surface here regardless of
# how much code/spec material lives in the genome.

@mcp.tool()
def helix_session_recent(
    handle: str,
    limit: int = 10,
    party_id: Optional[str] = None,
    since_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Recent genes authored by `handle`, newest first. No BM25 scoring.

    handle: session handle (e.g. "laude", "raude", "gemini", "batman")
    limit: max genes to return (default 10)
    party_id: optional scope -- narrows to a single party if a handle
        is reused across parties (uncommon).
    since_ts: optional Unix timestamp lower bound.

    Ideal for "what did raude just check in?" style peer awareness.
    """
    qs_parts = [f"limit={int(limit)}"]
    if party_id:
        qs_parts.append(f"party_id={urllib.request.quote(party_id)}")
    if since_ts is not None:
        qs_parts.append(f"since={since_ts}")
    path = f"/sessions/{urllib.request.quote(handle)}/recent?{'&'.join(qs_parts)}"
    return _http("GET", path)


# ── Tool: helix_consolidate ──────────────────────────────────────────
# Trigger session memory consolidation. Distills the session buffer
# into consolidated knowledge genes.

@mcp.tool()
def helix_consolidate() -> Dict[str, Any]:
    """Consolidate the current session buffer into long-term knowledge genes.

    Extracts only new facts, decisions, and discoveries from the
    buffered exchange stream, packing them as genes in the genome.
    Cheap but non-idempotent -- call at natural checkpoints (end of
    task, before handoff) not on every turn.

    Returns {facts_extracted: int, gene_ids: [...]}.
    """
    return _http("POST", "/consolidate")


# ── Tool: helix_health ───────────────────────────────────────────────
# Lightweight readiness probe. Separate from helix_stats (which is
# heavier) -- useful for "is the server reachable / ribosome configured?"
# checks without pulling full genome aggregates.

@mcp.tool()
def helix_health() -> Dict[str, Any]:
    """Ribosome model, gene count, upstream URL, and overall status.

    Cheaper than helix_stats -- returns just the readiness signals
    (status, ribosome backend, total genes, upstream). Use this for
    connectivity probes; use helix_stats for detailed genome health.
    """
    return _normalize_health_payload(_http("GET", "/health"))


# ── Tool: helix_metrics_tokens ───────────────────────────────────────
# Session + lifetime token counters, exact-from-upstream when possible,
# char-estimate fallback. Surfaces helix's cost/savings story.

@mcp.tool()
def helix_metrics_tokens() -> Dict[str, Any]:
    """Token counters for the current session and lifetime.

    Returns exact counts from upstream `usage` fields when available
    and char-count estimates otherwise, split into exact vs estimated
    buckets. Useful for answering "how much budget am I burning
    through /v1/chat/completions right now?".
    """
    return _http("GET", "/metrics/tokens")


# ── Tool: helix_bridge_status ────────────────────────────────────────
# Federation bridge state -- shared-dir location, signal list, inbox
# count. Pairs with the /bridge/collect + /bridge/signal endpoints
# which remain server-side only (writes are better done via helix_ingest
# or direct HTTP from a privileged client).

@mcp.tool()
def helix_bridge_status() -> Dict[str, Any]:
    """Federation bridge status: shared_dir, inbox count, signal list.

    The bridge is helix's multi-instance handoff channel (laude ↔ raude
    ↔ batman etc). This tool is read-only; use it to check whether
    inbox items are waiting to be collected, or which signals are in
    flight between instances.
    """
    return _http("GET", "/bridge/status")


# ── Tool: helix_gene_get ─────────────────────────────────────────────
# Fetch a single gene by ID. Indispensable for debugging retrieval
# results -- "what were this gene's promoter tags? what's the content?"

@mcp.tool()
def helix_gene_get(gene_id: str) -> Dict[str, Any]:
    """Fetch a single gene by ID.

    Returns the full gene model as JSON -- content, promoter tags
    (domains, entities, intent, summary), epigenetics (access_rate,
    co_activated_with), codons, chromatin state, embedding vector.

    Use when investigating a specific retrieval result: "gene X ranked
    #3, let me see what its tags were."

    Returns {error: str} if the gene_id is unknown.
    """
    return _http("GET", f"/genes/{urllib.request.quote(gene_id)}")


# ── Tool: helix_neighbors ────────────────────────────────────────────
# Lightweight top-k SEMA neighbors. Cheaper than helix_resonance when
# the caller only wants "what's near this query in SEMA space?" without
# the cymatic spectrum / harmonic edges.

@mcp.tool()
def helix_neighbors(query: str, k: int = 10) -> Dict[str, Any]:
    """Top-k SEMA neighbors for a query (light version of helix_resonance).

    Returns {query, k, neighbors: [{gene_id, sema_cos_sim, preview,
    path}], count}. No cymatic spectrum, no harmonic edges, no query
    SEMA vector -- just the neighbor list.

    Use this when debugging "which genes are semantically closest to X?"
    and you don't need the full four-primitive introspection of
    helix_resonance.
    """
    path = f"/debug/neighbors?query={urllib.request.quote(query)}&k={int(k)}"
    return _http("GET", path)


# ── Tool: helix_splice_preview ───────────────────────────────────────
# Dry-run the retrieval pipeline: extract -> express -> candidates,
# SKIPS the expensive splice step. Answers "what WOULD be in the context
# window?" without paying full /context cost (no ribosome calls).

@mcp.tool()
def helix_splice_preview(query: str, max_genes: int = 12) -> Dict[str, Any]:
    """Preview which genes WOULD be selected for a query's context window.

    Runs the cheap half of the /context pipeline: query keyword
    extraction + multi-tier express (promoter tags, FTS, SEMA,
    harmonic boost, TCM tiebreaker, access-rate tiebreaker), then
    STOPS before the splice step.

    Returns {query, extracted: {domains, entities}, candidates:
    [{rank, gene_id, score, preview, path, domains, entities,
    chromatin}], count}.

    Much cheaper than a full /context call -- no ribosome calls
    at all. Use for "why isn't query X surfacing gene Y?" debugging
    without burning model quota on splice.
    """
    path = (
        f"/debug/preview?query={urllib.request.quote(query)}"
        f"&max_genes={int(max_genes)}"
    )
    return _http("GET", path)


@mcp.tool()
def helix_fingerprint(
    query: str,
    max_results: Optional[int] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Return navigation-first retrieval fingerprints, not assembled content."""
    body: Dict[str, Any] = {"query": query}
    if max_results is not None:
        body["max_results"] = int(max_results)
    if profile:
        body["profile"] = profile
    return _http("POST", "/fingerprint", body)


# ── Software-vocabulary aliases (per docs/ROSETTA.md) ────────────────
# Pass-through tools registered under canonical software names. Body
# delegates to the legacy implementation -- same network call, same
# response shape, same behavior. Adding aliases instead of renaming
# keeps existing MCP clients working unchanged.
#
# Lexicon: see docs/ROSETTA.md for the full biology<->software map.

@mcp.tool()
def helix_document_get(document_id: str) -> Dict[str, Any]:
    """Fetch a single document by ID. Canonical alias for ``helix_gene_get``.

    Returns the full document model (content, tags, signals, fragments,
    lifecycle tier, embedding). 404 if unknown.

    Identical behavior to ``helix_gene_get`` -- prefer this name in
    new code.
    """
    return _http("GET", f"/genes/{urllib.request.quote(document_id)}")


@mcp.tool()
def helix_document_query(
    query: str,
    decoder_mode: Optional[str] = None,
    downstream_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compressed context window for ``query``. Canonical alias
    for ``helix_context``.

    Identical behavior. Prefer this name in new code.

    session_id: explicit session id for the working-set register. When
        omitted, defaults to MCP_SESSION_ID so repeated calls within
        this MCP subprocess elide already-delivered genes.
    """
    body: Dict[str, Any] = {"query": query}
    if decoder_mode:
        body["decoder_mode"] = decoder_mode
    if downstream_model:
        body["downstream_model"] = downstream_model
    body["session_id"] = session_id or MCP_SESSION_ID
    return _http("POST", "/context", body)


@mcp.tool()
def helix_document_preview(query: str, max_genes: int = 12) -> Dict[str, Any]:
    """Preview which documents WOULD be selected for a query. Canonical
    alias for ``helix_splice_preview``.

    Runs the retrieval pipeline through candidate selection, skips the
    final compression step. Cheap; no compressor calls. Prefer this
    name in new code.
    """
    path = (
        f"/debug/preview?query={urllib.request.quote(query)}"
        f"&max_genes={int(max_genes)}"
    )
    return _http("GET", path)


@mcp.tool()
def helix_document_fingerprint(
    query: str,
    max_results: Optional[int] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Canonical alias for ``helix_fingerprint``."""
    body: Dict[str, Any] = {"query": query}
    if max_results is not None:
        body["max_results"] = int(max_results)
    if profile:
        body["profile"] = profile
    return _http("POST", "/fingerprint", body)


# ── Future: codebase-memory-mcp composition ──────────────────────────
# Headroom's `codebase-memory-mcp` is a call-graph MCP (exists today,
# manual install only, still in testing per Tejas 2026-04-14). Two
# composition patterns:
#
#  A. Re-export (user sees 1 entry, gets both toolsets):
#     HELIX_EMBED_CODEGRAPH=1  → helix-mcp spawns codebase-memory-mcp
#                                as a stdio child, relays its tools
#                                under `helix_trace_calls` etc
#     HELIX_CODEGRAPH_PATH=...  → override binary path if not on PATH
#
#  B. Retrieval enrichment (invisible to host):
#     helix's /context internally queries codebase-memory-mcp for
#     call-path distance from query target, adds as a scoring tier.
#     No new MCP tools exposed — just smarter retrieval.
#
# Pattern A is the "reduce MCP count" story — useful once the user
# actually installs codebase-memory-mcp. Pattern B is the bigger long-
# term win: helix gene scoring gains structural signal. Both deferred
# until codebase-memory-mcp stabilizes (currently off-by-default). Hook
# points: this file for A, helix_context/context_manager.py for B.


def _register_with_registry() -> None:
    """Register this MCP process as a participant in the session registry.

    Closes the gap where MCP-host sessions (Claude Code, Claude Desktop,
    Antigravity, Cursor) did not appear in ``GET /sessions`` alongside
    laude/raude/taude. Each host spawns its own mcp_server process, so
    each gets its own participant_id under the configured party.

    Env vars (all optional — sensible defaults):
        HELIX_MCP_HANDLE   Handle for this session (default: mcp-<pid>).
                           Hosts SHOULD set this: "laude", "gemini", etc.
        HELIX_PARTY_ID     Party this participant belongs to
                           (default: "swift_wing21").
        HELIX_MCP_HOST     MCP host name — used as a capability tag so
                           ``GET /sessions`` can tell which IDE spawned
                           this process. E.g. "claude-code",
                           "antigravity", "cursor". Default: "unknown".

    Registration failure is non-fatal — tool calls still proxy to the
    HTTP API. Logged as a warning so the user can diagnose.
    """
    try:
        from helix_context.bridge import AgentBridge
    except Exception as exc:
        log.warning("Registry bridge import failed, skipping registration: %s", exc)
        return

    handle = os.environ.get("HELIX_MCP_HANDLE", f"mcp-{os.getpid()}")
    party_id = os.environ.get("HELIX_PARTY_ID", "swift_wing21")
    mcp_host = os.environ.get("HELIX_MCP_HOST", "unknown")
    workspace: Optional[str]
    try:
        workspace = os.getcwd()
    except Exception:
        workspace = None

    # Capability tags let GET /sessions consumers filter by host/role.
    capabilities = ["mcp_tools", f"host:{mcp_host}"]

    bridge = AgentBridge(helix_base_url=HELIX_URL)
    participant_id = bridge.register_participant(
        party_id=party_id,
        handle=handle,
        workspace=workspace,
        capabilities=capabilities,
        start_auto_heartbeat=True,
    )
    if participant_id:
        log.info(
            "Registered as %s (party=%s, host=%s, pid=%d)",
            handle, party_id, mcp_host, os.getpid(),
        )
    else:
        log.warning(
            "Session registration failed (is helix running at %s?) "
            "— tool calls will still work",
            HELIX_URL,
        )


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("HELIX_MCP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("helix-mcp starting — proxying to %s (timeout=%.1fs)",
             HELIX_URL, TIMEOUT_S)

    _register_with_registry()

    mcp.run()


if __name__ == "__main__":
    main()

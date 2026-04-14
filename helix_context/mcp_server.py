"""
MCP server for helix — exposes helix as a first-class tool inside MCP hosts.

Thin adapter: stdio JSON-RPC server that declares a handful of tools and
proxies each call to helix's HTTP API. Lets Claude Code / Claude Desktop
/ Cursor consume helix without any HTTP client boilerplate in the host.

Tools exposed:
    helix_context    — main retrieval (the big one)
    helix_stats      — genome health + size
    helix_ingest     — add content to the genome
    helix_resonance  — four-primitive introspection chart (ΣĒMA +
                        cymatic + harmonic + neighbor set) — new in
                        2026-04-14, see server.py:/debug/resonance

Run (stdio transport — what MCP hosts spawn):
    python -m helix_context.mcp_server

Configure in Claude Code .mcp.json:
    {
      "mcpServers": {
        "helix": {
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
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("helix.mcp")

HELIX_URL = os.environ.get("HELIX_MCP_URL", "http://127.0.0.1:11437").rstrip("/")
TIMEOUT_S = float(os.environ.get("HELIX_MCP_TIMEOUT", "30"))

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
            "_hint": "Is helix running? Check HELIX_MCP_URL env var.",
        }
    except Exception as exc:
        return {"_error": str(exc)}


# ── Tool: helix_context ──────────────────────────────────────────────
# The main retrieval path — MCP hosts call this to get a compressed
# context window for a query. Returns the same shape as /context,
# minus streaming (MCP tools are one-shot).

@mcp.tool()
def helix_context(
    query: str,
    decoder_mode: Optional[str] = None,
    downstream_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compressed context window for `query` from the helix genome.

    decoder_mode: "condensed" (default), "broad", or "dense". Controls
        how genes are unfolded into tokens. "broad" → more genes, less
        per-gene detail. "condensed" → fewer genes, more detail each.
    downstream_model: hint string so helix can size the budget for the
        target model (e.g. "claude-opus-4-6", "gpt-4").
    """
    body: Dict[str, Any] = {"query": query}
    if decoder_mode:
        body["decoder_mode"] = decoder_mode
    if downstream_model:
        body["downstream_model"] = downstream_model
    return _http("POST", "/context", body)


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

    Persona/user attribution comes from HELIX_AGENT / HELIX_USER env
    vars on the helix *server* (not on this MCP process), via helix's
    4-layer federation. See docs/FEDERATION_LOCAL.md.
    """
    body: Dict[str, Any] = {"content": content, "content_type": content_type}
    if metadata:
        body["metadata"] = metadata
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

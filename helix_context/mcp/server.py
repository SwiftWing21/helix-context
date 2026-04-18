"""
Helix Context MCP Server — exposes genome tools to Claude Code.

LEGACY SERVER: kept for rollback compatibility only.
Prefer `python -m helix_context.mcp_server` with `HELIX_MCP_URL`.

Three tools:
    helix_context    — query compressed context for a topic (the money saver)
    helix_ingest     — ingest content into the genome
    helix_stats      — genome health metrics + delta-epsilon history

Claude auto-discovers these tools and calls them when relevant,
replacing raw file reads with 7x compressed genome context.

Usage:
    Register in .mcp.json:
    {
        "mcpServers": {
            "helix-context": {
                "command": "python",
                "args": ["F:/Projects/helix-context/helix_context/mcp/server.py"]
            }
        }
    }

    Or with cmd /c wrapper for Claude Code on Windows:
    {
        "mcpServers": {
            "helix-context": {
                "command": "cmd",
                "args": ["/c", "python", "F:/Projects/helix-context/helix_context/mcp/server.py"]
            }
        }
    }
"""

from __future__ import annotations

import json
import os
import sys

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
TIMEOUT = float(os.environ.get("HELIX_TIMEOUT", "30"))

_LEGACY_WARNING = (
    "WARNING: helix_context.mcp.server is the legacy 3-tool MCP server. "
    "Switch to `python -m helix_context.mcp_server` and `HELIX_MCP_URL` "
    "for the canonical 18-tool surface."
)


# ── MCP stdio protocol (minimal, no SDK dependency) ─────────────────
#
# The MCP stdio protocol is JSON-RPC 2.0 over stdin/stdout.
# We implement just enough to register tools and handle calls.
# This avoids requiring the `mcp` SDK which has compatibility issues.


_stdin = None
_stdout = None


def _init_io() -> None:
    """Set up binary-safe stdin/stdout for MCP Content-Length framing.

    Works on both Windows and Linux. On Windows, stdin/stdout default to
    text mode which mangles \\r\\n. We switch to raw binary and wrap with
    a buffered reader/writer.
    """
    global _stdin, _stdout

    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    _stdin = sys.stdin.buffer
    _stdout = sys.stdout.buffer


def _read_message() -> dict | None:
    """Read a JSON-RPC message from stdin (Content-Length framing)."""
    # Read headers line by line until empty line
    content_length = 0
    while True:
        line = _stdin.readline()
        if not line:
            return None  # EOF
        line_str = line.decode("utf-8").strip()
        if line_str == "":
            break  # End of headers
        if line_str.lower().startswith("content-length:"):
            content_length = int(line_str.split(":", 1)[1].strip())

    if content_length == 0:
        return None

    body = _stdin.read(content_length)
    return json.loads(body.decode("utf-8"))


def _send_message(msg: dict) -> None:
    """Write a JSON-RPC message to stdout (Content-Length framing)."""
    body = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    _stdout.write(header + body)
    _stdout.flush()


def _result(id: int | str, result: dict) -> None:
    _send_message({"jsonrpc": "2.0", "id": id, "result": result})


def _error(id: int | str, code: int, message: str) -> None:
    _send_message({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})


# ── Tool definitions ────────────────────────────────────────────────

TOOLS = [
    {
        "name": "helix_context",
        "description": (
            "Query compressed project context from the Helix genome. "
            "Returns 7x compressed codebase knowledge instead of raw files. "
            "Use this BEFORE reading source files to save tokens. "
            "The genome contains ingested code, docs, and conversation history."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up (e.g., 'auth system', 'database schema', 'how does the API work')",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "helix_ingest",
        "description": (
            "Ingest new content into the Helix genome. "
            "Use after creating or modifying files so the genome stays current. "
            "The content is compressed and stored for future context queries."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to ingest (code, docs, architecture notes)",
                },
                "content_type": {
                    "type": "string",
                    "enum": ["text", "code"],
                    "description": "Type of content (default: text)",
                    "default": "text",
                },
                "path": {
                    "type": "string",
                    "description": "Source file path for traceability (optional)",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "helix_stats",
        "description": (
            "Get Helix genome health metrics: gene count, compression ratio, "
            "delta-epsilon health signals, and recent query history."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_history": {
                    "type": "boolean",
                    "description": "Include recent query health history (default: false)",
                    "default": False,
                },
            },
        },
    },
]


# ── Tool handlers ───────────────────────────────────────────────────

def _handle_helix_context(args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "Error: query is required"

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{HELIX_URL}/context", json={"query": query})
            if resp.status_code != 200:
                return f"Helix error: HTTP {resp.status_code}"

            data = resp.json()
            if not data or not isinstance(data, list) or not data[0]:
                return "No context found in genome."

            entry = data[0]
            desc = entry.get("description", "")
            content = entry.get("content", "")
            health = entry.get("context_health", {})

            result = f"[Helix Context] {desc}\n"
            if health:
                status = health.get("status", "unknown")
                ell = health.get("ellipticity", 0)
                result += f"[Health: {status}, ellipticity={ell:.2f}]\n\n"
            result += content
            return result

    except httpx.ConnectError:
        return f"Cannot connect to Helix at {HELIX_URL}. Is the server running?"
    except Exception as e:
        return f"Error: {e}"


def _handle_helix_ingest(args: dict) -> str:
    content = args.get("content", "")
    if not content:
        return "Error: content is required"

    content_type = args.get("content_type", "text")
    metadata = {}
    if args.get("path"):
        metadata["path"] = args["path"]

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{HELIX_URL}/ingest", json={
                "content": content,
                "content_type": content_type,
                "metadata": metadata if metadata else None,
            })
            if resp.status_code == 200:
                data = resp.json()
                return f"Ingested: {data.get('count', 0)} genes created"
            elif resp.status_code == 422:
                return f"Ingest failed (ribosome error): {resp.json().get('error', 'unknown')}"
            else:
                return f"Ingest error: HTTP {resp.status_code}"

    except httpx.ConnectError:
        return f"Cannot connect to Helix at {HELIX_URL}. Is the server running?"
    except Exception as e:
        return f"Error: {e}"


def _handle_helix_stats(args: dict) -> str:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(f"{HELIX_URL}/stats")
            if resp.status_code != 200:
                return f"Stats error: HTTP {resp.status_code}"

            stats = resp.json()
            lines = [
                "Helix Genome Stats",
                f"  Genes: {stats.get('total_genes', 0)}",
                f"  Compression: {stats.get('compression_ratio', 0):.1f}x",
                f"  Open: {stats.get('open', 0)}, Euchromatin: {stats.get('euchromatin', 0)}, "
                f"Heterochromatin: {stats.get('heterochromatin', 0)}",
                f"  Raw chars: {stats.get('total_chars_raw', 0):,}",
                f"  Compressed: {stats.get('total_chars_compressed', 0):,}",
            ]

            health = stats.get("health", {})
            if health and health.get("total_queries", 0) > 0:
                lines.append(f"\n  Health Summary ({health['total_queries']} queries):")
                lines.append(f"    Avg ellipticity: {health.get('avg_ellipticity', 0):.3f}")
                lines.append(f"    Avg coverage: {health.get('avg_coverage', 0):.3f}")
                lines.append(f"    Avg density: {health.get('avg_density', 0):.3f}")
                lines.append(f"    Status counts: {health.get('status_counts', {})}")

            if args.get("include_history"):
                hist_resp = client.get(f"{HELIX_URL}/health/history?limit=10")
                if hist_resp.status_code == 200:
                    history = hist_resp.json()
                    if history:
                        lines.append(f"\n  Recent Queries:")
                        for h in history[:10]:
                            q = h.get("query", "")[:40]
                            lines.append(
                                f"    {h['status']:12s} e={h['ellipticity']:.2f} "
                                f"g={h['genes_expressed']}/{h['genes_available']} | {q}"
                            )

            return "\n".join(lines)

    except httpx.ConnectError:
        return f"Cannot connect to Helix at {HELIX_URL}. Is the server running?"
    except Exception as e:
        return f"Error: {e}"


HANDLERS = {
    "helix_context": _handle_helix_context,
    "helix_ingest": _handle_helix_ingest,
    "helix_stats": _handle_helix_stats,
}


# ── Main loop ───────────────────────────────────────────────────────

def main():
    """Run the MCP stdio server."""
    print(_LEGACY_WARNING, file=sys.stderr)
    _init_io()

    while True:
        msg = _read_message()
        if msg is None:
            break

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "initialize":
            _result(msg_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "helix-context",
                    "version": "0.1.0",
                },
            })

        elif method == "notifications/initialized":
            pass  # No response needed for notifications

        elif method == "tools/list":
            _result(msg_id, {"tools": TOOLS})

        elif method == "tools/call":
            params = msg.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            handler = HANDLERS.get(tool_name)
            if handler is None:
                _error(msg_id, -32601, f"Unknown tool: {tool_name}")
            else:
                text = handler(tool_args)
                _result(msg_id, {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                })

        elif method == "ping":
            _result(msg_id, {})

        elif msg_id is not None:
            _error(msg_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()

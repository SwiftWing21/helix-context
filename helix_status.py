"""
Helix status CLI — quick operator check for the canonical Helix setup.

Entry point: ``helix-status``.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_SERVER_URL = os.environ.get("HELIX_STATUS_URL", "http://127.0.0.1:11437").rstrip("/")
DEFAULT_LAUNCHER_URL = os.environ.get("HELIX_LAUNCHER_URL", "http://127.0.0.1:11438").rstrip("/")


def _get_json(url: str, timeout_s: float = 1.5) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "reachable": False,
            "error": f"HTTP {exc.code}",
            "detail": exc.read().decode("utf-8", errors="replace")[:500],
        }
    except urllib.error.URLError as exc:
        return {
            "reachable": False,
            "error": "unreachable",
            "detail": str(exc.reason),
        }
    except Exception as exc:
        return {
            "reachable": False,
            "error": type(exc).__name__,
            "detail": str(exc),
        }


def _find_mcp_config(
    explicit_path: Optional[Path] = None,
    start_dir: Optional[Path] = None,
) -> Optional[Path]:
    if explicit_path is not None:
        return Path(explicit_path)
    cur = Path(start_dir) if start_dir is not None else Path.cwd()
    cur = cur.resolve()
    for candidate in [cur, *cur.parents]:
        path = candidate / ".mcp.json"
        if path.exists():
            return path
    return None


def _check_mcp_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {
            "status": "missing",
            "next_action": "Create a `.mcp.json` that points at `helix_context.mcp_server`.",
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "detail": str(exc),
            "next_action": "Fix the JSON syntax in the MCP config file.",
        }

    servers = data.get("mcpServers") or {}
    server_name = None
    server = None
    for candidate in ("helix-context", "helix"):
        if isinstance(servers.get(candidate), dict):
            server_name = candidate
            server = servers[candidate]
            break

    if not isinstance(server, dict):
        return {
            "status": "missing",
            "path": str(path),
            "next_action": "Add a `helix-context` MCP entry that points at the canonical server.",
        }

    args = server.get("args") or []
    env = server.get("env") or {}
    module = args[1] if len(args) >= 2 and args[0] == "-m" else None

    if (
        module == "helix_context.mcp_server"
        and "HELIX_MCP_URL" in env
        and server_name == "helix-context"
    ):
        return {
            "status": "canonical",
            "path": str(path),
            "server_name": server_name,
            "module": module,
            "env_var": "HELIX_MCP_URL",
        }

    if module == "helix_context.mcp_server" and "HELIX_MCP_URL" in env:
        return {
            "status": "noncanonical",
            "path": str(path),
            "server_name": server_name,
            "module": module,
            "env_var": "HELIX_MCP_URL",
            "next_action": "Rename the MCP server entry to `helix-context` to match the shared docs and status checks.",
        }

    if module == "helix_context.mcp.server":
        return {
            "status": "legacy",
            "path": str(path),
            "server_name": server_name,
            "module": module,
            "env_var": "HELIX_URL" if "HELIX_URL" in env else None,
            "next_action": "Switch to `helix_context.mcp_server` and `HELIX_MCP_URL`.",
        }

    return {
        "status": "noncanonical",
        "path": str(path),
        "server_name": server_name,
        "module": module,
        "env_keys": sorted(env.keys()),
        "next_action": "Point the Helix MCP entry at `helix_context.mcp_server` with `HELIX_MCP_URL`.",
    }


def _check_skill(skill_path: Path) -> Dict[str, Any]:
    skill_file = skill_path / "SKILL.md"
    if skill_file.exists():
        return {
            "status": "present",
            "path": str(skill_file),
        }
    return {
        "status": "missing",
        "path": str(skill_file),
        "next_action": "Install or restore the shared `helix-context` skill.",
    }


def collect_status(
    *,
    server_url: str = DEFAULT_SERVER_URL,
    launcher_url: str = DEFAULT_LAUNCHER_URL,
    mcp_config: Optional[Path] = None,
    skill_dir: Optional[Path] = None,
    start_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    server = _get_json(f"{server_url}/health")
    launcher = _get_json(f"{launcher_url}/api/state")
    mcp_status = _check_mcp_config(_find_mcp_config(mcp_config, start_dir=start_dir))
    skill_status = _check_skill(
        skill_dir or (Path.home() / ".claude" / "skills" / "helix-context")
    )

    launcher_reachable = "error" not in launcher
    server_reachable = "error" not in server and server.get("status") == "ok"
    integration_ready = (
        server_reachable
        and mcp_status["status"] == "canonical"
        and skill_status["status"] == "present"
    )

    if not server_reachable:
        if launcher_reachable:
            availability = "degraded"
            next_action = "Open the launcher UI and click Start or Restart to bring Helix up."
        else:
            availability = "unavailable"
            next_action = "Run `helix-launcher` to start the canonical supervisor."
    else:
        availability = "available"
        if mcp_status.get("next_action"):
            next_action = mcp_status["next_action"]
        elif skill_status.get("next_action"):
            next_action = skill_status["next_action"]
        else:
            next_action = "Use `helix_context` for repo questions."

    return {
        "availability": availability,
        "integration_ready": integration_ready,
        "next_action": next_action,
        "server": {
            "url": server_url,
            "reachable": server_reachable,
            "payload": server,
        },
        "launcher": {
            "url": launcher_url,
            "reachable": launcher_reachable,
            "payload": launcher,
        },
        "mcp_config": mcp_status,
        "skill": skill_status,
    }


def _render_text(status: Dict[str, Any]) -> str:
    lines = [
        f"Helix availability: {status['availability']}",
        f"Next action: {status['next_action']}",
        "",
        f"Server: {'up' if status['server']['reachable'] else 'down'} ({status['server']['url']})",
        f"Launcher: {'up' if status['launcher']['reachable'] else 'down'} ({status['launcher']['url']})",
        f"Integration ready: {'yes' if status['integration_ready'] else 'no'}",
        f"MCP config: {status['mcp_config']['status']}",
        f"Skill: {status['skill']['status']}",
    ]

    if status["mcp_config"].get("path"):
        lines.append(f"MCP path: {status['mcp_config']['path']}")
    if status["skill"].get("path"):
        lines.append(f"Skill path: {status['skill']['path']}")
    if status["mcp_config"].get("next_action"):
        lines.append(f"MCP fix: {status['mcp_config']['next_action']}")
    if status["skill"].get("next_action"):
        lines.append(f"Skill fix: {status['skill']['next_action']}")

    return "\n".join(lines)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="helix-status",
        description="Check the canonical Helix server, launcher, MCP config, and shared skill.",
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--launcher-url", default=DEFAULT_LAUNCHER_URL)
    parser.add_argument("--mcp-config", type=Path, default=None)
    parser.add_argument("--skill-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    status = collect_status(
        server_url=args.server_url.rstrip("/"),
        launcher_url=args.launcher_url.rstrip("/"),
        mcp_config=args.mcp_config,
        skill_dir=args.skill_dir,
    )
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(_render_text(status))
    return 0 if status["integration_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

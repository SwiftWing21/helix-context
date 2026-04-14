"""
Daemon entry: auto-memory → helix sync.

Reads config from helix.toml `[mem_sync]` section, honors env-var
overrides, starts the poll loop.

Usage:
    python scripts/run_mem_sync.py

Env-var overrides (take precedence over toml):
    HELIX_MEM_SYNC_URL        - helix server URL (default http://127.0.0.1:11437)
    HELIX_MEM_SYNC_INTERVAL   - poll interval in seconds (default 60)
    HELIX_MEM_SYNC_DIRS       - colon-separated dirs (overrides toml list)

Persona/agent attribution is automatic via the syncer process's env:
    HELIX_AGENT=raude         - which persona is doing the writes
    HELIX_USER=max            - the human principal
    HELIX_DEVICE=<hostname>   - auto-detected if unset
    HELIX_ORG=<org>           - optional

Set these once in your shell profile and every gene ingested from this
syncer carries the attribution automatically.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to path so `helix_context.mem_sync` resolves when
# running as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_context.mem_sync import run_daemon  # noqa: E402


def _load_toml_config() -> dict:
    """Read [mem_sync] from helix.toml. Returns {} if section missing."""
    try:
        import tomllib  # py3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return {}
    toml_path = Path(__file__).resolve().parent.parent / "helix.toml"
    if not toml_path.exists():
        return {}
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("mem_sync", {})
    except Exception as exc:
        print(f"[mem_sync] failed to read helix.toml: {exc}", file=sys.stderr)
        return {}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_toml_config()
    if not cfg.get("enabled", False):
        print("[mem_sync] disabled in helix.toml — set [mem_sync].enabled=true",
              file=sys.stderr)
        return 1

    helix_url = (
        os.environ.get("HELIX_MEM_SYNC_URL")
        or cfg.get("helix_url", "http://127.0.0.1:11437")
    )
    interval = int(
        os.environ.get("HELIX_MEM_SYNC_INTERVAL")
        or cfg.get("sync_interval_s", 60)
    )
    env_dirs = os.environ.get("HELIX_MEM_SYNC_DIRS")
    if env_dirs:
        watch_dirs = [d.strip() for d in env_dirs.split(os.pathsep) if d.strip()]
    else:
        watch_dirs = list(cfg.get("watch_dirs", []))

    if not watch_dirs:
        print("[mem_sync] no watch_dirs configured — set [mem_sync].watch_dirs "
              "in helix.toml or export HELIX_MEM_SYNC_DIRS", file=sys.stderr)
        return 1

    # Expand ~ in paths — toml doesn't.
    watch_dirs = [os.path.expanduser(d) for d in watch_dirs]

    agent_kind = cfg.get("agent_kind") or os.environ.get("HELIX_AGENT_KIND")

    run_daemon(
        watch_dirs=watch_dirs,
        helix_url=helix_url,
        sync_interval_s=interval,
        agent_kind=agent_kind,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

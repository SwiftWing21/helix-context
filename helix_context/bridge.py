"""
Bridge — Shared memory layer between AI assistants.

Creates a file-based protocol that any AI assistant (Claude, Gemini, etc.)
can read and write to share context through the Agentome genome.

Architecture:
    ~/.helix/shared/          — shared memory directory
        inbox/                — files TO ingest (any assistant drops files here)
        outbox/               — genome context snapshots (assistants read from here)
        signals/              — lightweight status signals between assistants
        SHARED_CONTEXT.md     — always-current genome summary for instruction files

    The bridge watches inbox/ and auto-ingests new files into the genome.
    It periodically snapshots the genome health + recent genes into outbox/.
    Signals allow lightweight coordination ("I'm ingesting", "query X").

Usage:
    from helix_context.bridge import AgentBridge

    bridge = AgentBridge()
    bridge.write_signal("ingesting", {"files": 1500, "eta_min": 30})
    bridge.drop_to_inbox("fact: the port is 11437", source="gemini")
    bridge.update_shared_context(genome_stats)

Integration:
    - Claude Code: reads SHARED_CONTEXT.md via /helix skill
    - Gemini Code Assist: reads SHARED_CONTEXT.md via GEMINI.md include
    - Any agent: drops files into inbox/ for genome ingestion
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("helix.bridge")

# Default shared directory
DEFAULT_SHARED_DIR = os.path.expanduser("~/.helix/shared")


class AgentBridge:
    """File-based memory bridge between AI assistants."""

    def __init__(self, shared_dir: Optional[str] = None):
        self.shared_dir = Path(shared_dir or DEFAULT_SHARED_DIR)
        self.inbox = self.shared_dir / "inbox"
        self.outbox = self.shared_dir / "outbox"
        self.signals = self.shared_dir / "signals"

        # Create directory structure
        for d in [self.inbox, self.outbox, self.signals]:
            d.mkdir(parents=True, exist_ok=True)

        log.info("AgentBridge initialized at %s", self.shared_dir)

    # ── Inbox: receive content from other assistants ──────────────

    def drop_to_inbox(
        self,
        content: str,
        source: str = "unknown",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Drop content into the inbox for genome ingestion.
        Any assistant can call this to share knowledge.
        """
        if filename is None:
            filename = f"{source}_{int(time.time())}.md"
        path = self.inbox / filename
        path.write_text(content, encoding="utf-8")
        log.info("Inbox: %s dropped %s (%d chars)", source, filename, len(content))
        return path

    def collect_inbox(self) -> List[Dict]:
        """
        Collect all files from inbox for ingestion.
        Returns list of {path, content, source} dicts.
        Files are removed after collection.
        """
        items = []
        for f in sorted(self.inbox.iterdir()):
            if f.is_file() and f.suffix in (".md", ".txt", ".json"):
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                    source = f.stem.split("_")[0]  # Extract source from filename
                    items.append({
                        "path": str(f),
                        "content": content,
                        "source": source,
                    })
                    f.unlink()  # Remove after collection
                except Exception:
                    log.warning("Failed to collect inbox file: %s", f, exc_info=True)
        return items

    # ── Outbox: publish genome state for other assistants ─────────

    def update_shared_context(self, stats: Dict, recent_queries: Optional[List] = None) -> Path:
        """
        Write SHARED_CONTEXT.md — a live summary of the genome state
        that other assistants can read from their instruction files.
        """
        lines = [
            "# Helix Genome — Shared Context",
            f"*Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Genome Stats",
            f"- **Genes:** {stats.get('total_genes', 0)}",
            f"- **Compression:** {stats.get('compression_ratio', 0):.1f}x",
            f"- **Open:** {stats.get('open', 0)} | Euchromatin: {stats.get('euchromatin', 0)} | Heterochromatin: {stats.get('heterochromatin', 0)}",
            "",
        ]

        health = stats.get("health", {})
        if health:
            lines.extend([
                "## Health",
                f"- **Queries logged:** {health.get('total_queries', 0)}",
                f"- **Avg ellipticity:** {health.get('avg_ellipticity', 0):.3f}",
                f"- **Status distribution:** {health.get('status_counts', {})}",
                "",
            ])

        config = stats.get("config", {})
        if config:
            lines.extend([
                "## Config",
                f"- **Decoder mode:** {config.get('decoder_mode', '?')}",
                f"- **Max genes/turn:** {config.get('max_genes_per_turn', '?')}",
                f"- **Expression budget:** {config.get('expression_tokens', '?')} tokens",
                "",
            ])

        lines.extend([
            "## How to Use",
            "- **Query:** POST http://127.0.0.1:11437/context with `{\"query\": \"...\", \"decoder_mode\": \"none\"}`",
            "- **Ingest:** Drop .md/.txt files into `~/.helix/shared/inbox/`",
            "- **Signal:** Write JSON to `~/.helix/shared/signals/<name>.json`",
            "",
            "## Inbox Protocol",
            "Any AI assistant can share knowledge by writing files to:",
            f"`{self.inbox}`",
            "",
            "Files are auto-ingested into the genome on the next cycle.",
            "Filename format: `<source>_<timestamp>.md` (e.g., `gemini_1712700000.md`)",
        ])

        path = self.shared_dir / "SHARED_CONTEXT.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Updated SHARED_CONTEXT.md (%d genes)", stats.get("total_genes", 0))
        return path

    # ── Signals: lightweight coordination ─────────────────────────

    def write_signal(self, name: str, data: Dict) -> Path:
        """Write a signal for other assistants to read."""
        data["timestamp"] = time.time()
        data["timestamp_human"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = self.signals / f"{name}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def read_signal(self, name: str) -> Optional[Dict]:
        """Read a signal from another assistant."""
        path = self.signals / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def clear_signal(self, name: str) -> None:
        """Clear a signal."""
        path = self.signals / f"{name}.json"
        if path.exists():
            path.unlink()

    def list_signals(self) -> Dict[str, Dict]:
        """List all active signals."""
        result = {}
        for f in self.signals.iterdir():
            if f.suffix == ".json":
                try:
                    result[f.stem] = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return result

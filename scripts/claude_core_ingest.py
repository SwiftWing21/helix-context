"""
Claude API core-folder ingest — synchronous.

Walks a small set of high-priority folders and re-ingests each file through
the LLM ribosome with ClaudeBackend. Uses the existing Ribosome.pack() path
directly (NOT CpuTagger) so the gene gets full Claude-quality complement,
codons, and KV extraction.

This is the CHECK-IN pass: a few hundred genes, ~$5-7 on Haiku. Run this
first to validate the pipe end-to-end, inspect a few complements for quality,
then decide whether to run the bulk pass via claude_batch_ingest.py.

Usage:
    python scripts/claude_core_ingest.py
    python scripts/claude_core_ingest.py --roots F:/Projects/helix-context/helix_context
    python scripts/claude_core_ingest.py --model claude-sonnet-4-6
    python scripts/claude_core_ingest.py --dry-run   # count files, no API calls

Requires:
    - helix.toml: ribosome.backend = "claude"
    - ANTHROPIC_API_KEY in env (routed through Headroom via claude_base_url)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helix_context.codons import CodonChunker, CodonEncoder
from helix_context.config import load_config
from helix_context.genome import Genome
from helix_context.ribosome import ClaudeBackend, LiteLLMBackend, Ribosome

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest.claude_core")

# Default core roots — high-value, moderate size
DEFAULT_ROOTS = [
    "F:/Projects/helix-context/helix_context",
    "F:/Projects/Education/fleet",
    "F:/Projects/Education/autoresearch",
]

TEXT_EXTS = {".txt", ".md", ".cfg", ".ini", ".conf", ".toml"}
CODE_EXTS = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml"}
INGEST_EXTS = TEXT_EXTS | CODE_EXTS

SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build",
    ".pytest_cache", "target", ".claude", "knowledge",  # fleet/knowledge is ~2000 files, not core
}

MAX_FILE_SIZE = 100_000   # 100KB — core files are usually well under this
MIN_FILE_SIZE = 100


def walk_core(roots: list[str]) -> list[tuple[str, str]]:
    """Return [(path, ext)] of files to ingest."""
    out: list[tuple[str, str]] = []
    for root in roots:
        if not os.path.isdir(root):
            log.warning("Root not found: %s", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in INGEST_EXTS:
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    continue
                if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
                    continue
                out.append((fpath, ext))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="genome.db")
    parser.add_argument("--config", default="helix.toml")
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--model", default=None, help="Override helix.toml claude_model")
    parser.add_argument("--dry-run", action="store_true", help="Count files, no API calls")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N files (0=all)")
    args = parser.parse_args()

    files = walk_core(args.roots)
    log.info("Found %d candidate files across %d root(s)", len(files), len(args.roots))
    if args.limit:
        files = files[: args.limit]
        log.info("Limiting to first %d files", len(files))

    if args.dry_run:
        total_bytes = sum(os.path.getsize(p) for p, _ in files)
        log.info("Dry-run total size: %.1f KB", total_bytes / 1024)
        return 0

    cfg = load_config(args.config)
    if cfg.ribosome.backend not in ("claude", "litellm"):
        log.error("helix.toml ribosome.backend must be 'claude' or 'litellm' (got %r). Aborting.", cfg.ribosome.backend)
        return 2

    if cfg.ribosome.backend == "litellm":
        model = args.model or cfg.ribosome.litellm_model
        log.info("Using LiteLLM model: %s (proxy: %s)", model, cfg.ribosome.claude_base_url or "direct")
        backend = LiteLLMBackend(
            model=model,
            base_url=cfg.ribosome.claude_base_url,
            max_tokens=cfg.budget.ribosome_tokens,
            timeout=cfg.ribosome.timeout,
        )
    else:
        model = args.model or cfg.ribosome.claude_model
        log.info("Using Claude model: %s (proxy: %s)", model, cfg.ribosome.claude_base_url or "direct")
        backend = ClaudeBackend(
            model=model,
            base_url=cfg.ribosome.claude_base_url,
            max_tokens=cfg.budget.ribosome_tokens,
            timeout=cfg.ribosome.timeout,
        )
    encoder = CodonEncoder()
    ribosome = Ribosome(
        backend=backend,
        encoder=encoder,
        splice_aggressiveness=cfg.budget.splice_aggressiveness,
    )
    chunker = CodonChunker(max_chars_per_strand=4000)
    genome = Genome(path=args.db, synonym_map={}, splade_enabled=True, entity_graph=True)

    t0 = time.perf_counter()
    stats = {"files": 0, "genes": 0, "errors": 0, "skipped": 0}

    for fpath, ext in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            stats["errors"] += 1
            continue

        ct = "code" if ext in CODE_EXTS else "text"
        strands = chunker.chunk(content, content_type=ct)
        for i, strand in enumerate(strands):
            try:
                gene = ribosome.pack(strand.content, content_type=ct)
            except Exception as exc:
                log.warning("pack failed on %s strand %d: %s", fpath, i, exc)
                stats["errors"] += 1
                continue
            # Patch source tracking that Ribosome.pack() doesn't set
            gene.source_id = fpath
            gene.is_fragment = strand.is_fragment
            genome.upsert_gene(gene)
            stats["genes"] += 1

        stats["files"] += 1
        if stats["files"] % 10 == 0:
            elapsed = time.perf_counter() - t0
            log.info(
                "[%d/%d files | %d genes | %.1f files/s] %s",
                stats["files"], len(files), stats["genes"],
                stats["files"] / max(elapsed, 0.01),
                os.path.basename(fpath),
            )

    # Final checkpoint
    try:
        genome.checkpoint("TRUNCATE") if hasattr(genome, "checkpoint") else None
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    log.info("=" * 60)
    log.info("Claude core ingest complete in %.0fs", elapsed)
    log.info("  Files: %d (%d errors)", stats["files"], stats["errors"])
    log.info("  Genes: %d (%.1f/s)", stats["genes"], stats["genes"] / max(elapsed, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

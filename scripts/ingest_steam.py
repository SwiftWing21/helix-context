"""
Ingest SteamLibrary — full content for text/code, manifest genes for skipped files.

Every game gets:
  - Full gene encoding for .lua, .py, .json, .cfg, .ini, .txt, .md, etc.
  - A manifest gene summarizing skipped files (binaries, textures, models)
    with path/name/size/type metadata — so the genome knows what's installed.

Usage:
    python scripts/ingest_steam.py                           # default F:\SteamLibrary
    python scripts/ingest_steam.py --root "D:\SteamLibrary"  # custom path
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helix_context.config import load_config
from helix_context.genome import Genome
from helix_context.tagger import CpuTagger
from helix_context.codons import CodonChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest.steam")

# Full content ingest
TEXT_EXTS = {".txt", ".md", ".cfg", ".ini", ".conf", ".properties", ".vdf", ".acf"}
CODE_EXTS = {
    ".lua", ".py", ".cs", ".js", ".json", ".yaml", ".yml", ".toml",
    ".bat", ".sh", ".html", ".xml", ".csv",
}
INGEST_EXTS = TEXT_EXTS | CODE_EXTS

# Directories with no knowledge value
SKIP_DIRS = {
    "shadercache", "temp", "downloading", "depotcache", "__pycache__",
    ".git", "node_modules", "Mono", "MonoBleedingEdge",
}

MAX_FILE_SIZE = 500_000   # 500KB for full content ingest
MIN_FILE_SIZE = 50        # Skip tiny files


def flush_manifest(game_name, batch, genome, tagger):
    """Create a single manifest gene summarizing skipped files for a game."""
    if not batch:
        return 0

    by_ext: dict[str, list] = {}
    total_size = 0
    for path, size in batch:
        ext = os.path.splitext(path)[1].lower() or "(no ext)"
        by_ext.setdefault(ext, []).append((path, size))
        total_size += size

    lines = [
        f"Game: {game_name}",
        f"Total skipped files: {len(batch)}, Total size: {total_size / 1024 / 1024:.1f} MB",
        "",
    ]
    for ext in sorted(by_ext, key=lambda e: -len(by_ext[e])):
        items = by_ext[ext]
        total_ext_size = sum(s for _, s in items)
        lines.append(f"{ext}: {len(items)} files ({total_ext_size / 1024 / 1024:.1f} MB)")
        for path, size in items[:10]:
            short = path.replace("\\", "/").split("/")
            short = "/".join(short[-3:])
            lines.append(f"  {short} ({size / 1024:.0f} KB)")
        if len(items) > 10:
            lines.append(f"  ... and {len(items) - 10} more")

    content = "\n".join(lines)
    source = f"F:/SteamLibrary/steamapps/common/{game_name}/"
    gene = tagger.pack(content, content_type="text", source_id=source)
    genome.upsert_gene(gene)
    return 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="F:/SteamLibrary")
    args = parser.parse_args()

    root = args.root
    config = load_config()

    genome = Genome(
        path=config.genome.path,
        synonym_map=config.synonym_map,
        splade_enabled=config.ingestion.splade_enabled,
        entity_graph=config.ingestion.entity_graph,
    )
    tagger = CpuTagger(synonym_map=config.synonym_map)
    chunker = CodonChunker()

    files_ingested = 0
    manifest_genes = 0
    genes_created = 0
    errors = 0
    t_start = time.perf_counter()

    current_game = None
    current_game_skipped: list[tuple[str, int]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        # Detect game boundary
        parts = dirpath.replace("\\", "/").split("/")
        game = None
        try:
            ci = parts.index("common")
            if ci + 1 < len(parts):
                game = parts[ci + 1]
        except ValueError:
            pass

        # Flush previous game manifest on game change
        if game != current_game and current_game is not None:
            n = flush_manifest(current_game, current_game_skipped, genome, tagger)
            manifest_genes += n
            if n:
                genes_created += 1
            current_game_skipped = []
        current_game = game

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()

            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue

            # Full content ingest?
            if ext in INGEST_EXTS and MIN_FILE_SIZE <= size <= MAX_FILE_SIZE:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception:
                    errors += 1
                    continue

                content_type = "code" if ext in CODE_EXTS else "text"
                strands = chunker.chunk(content, content_type=content_type)
                for i, strand in enumerate(strands):
                    gene = tagger.pack(
                        strand.content,
                        content_type=content_type,
                        source_id=fpath,
                        sequence_index=i,
                    )
                    gene.is_fragment = strand.is_fragment
                    genome.upsert_gene(gene)
                    genes_created += 1
                files_ingested += 1
            else:
                # Skipped — collect for manifest
                current_game_skipped.append((fpath, size))

            if genes_created > 0 and genes_created % 100 == 0:
                elapsed = time.perf_counter() - t_start
                log.info(
                    "[%d files, %d genes, %d manifests] %.1f genes/s | %s",
                    files_ingested, genes_created, manifest_genes,
                    genes_created / elapsed, game or "?",
                )

    # Flush final game
    if current_game_skipped:
        n = flush_manifest(current_game, current_game_skipped, genome, tagger)
        manifest_genes += n
        genes_created += n

    elapsed = time.perf_counter() - t_start
    stats = genome.stats()
    log.info("=" * 60)
    log.info("SteamLibrary ingest complete")
    log.info("  Files ingested (full): %d", files_ingested)
    log.info("  Manifest genes (skipped file index): %d", manifest_genes)
    log.info("  Genes created: %d in %.1fs (%.1f genes/s)", genes_created, elapsed, genes_created / elapsed)
    log.info("  Errors: %d", errors)
    log.info("  Genome total: %d genes", stats["total_genes"])


if __name__ == "__main__":
    main()

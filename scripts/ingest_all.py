"""
Ingest all sources into the genome in one pass.

Sources:
  - F:\Projects (code, docs, configs)
  - F:\SteamLibrary (game Lua/JSON/configs + manifests)
  - F:\OpenModels (GGUF model headers)
  - E:\ (games, programs)

Skips binaries, limits file size to 200KB to avoid stalls on
massive JSON/XML blobs. Commits every 100 genes.
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helix_context.tagger import CpuTagger
from helix_context.genome import Genome
from helix_context.codons import CodonChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest.all")

TEXT_EXTS = {".txt", ".md", ".cfg", ".ini", ".conf", ".properties", ".vdf", ".acf"}
CODE_EXTS = {
    ".lua", ".py", ".cs", ".js", ".json", ".yaml", ".yml", ".toml",
    ".bat", ".sh", ".html", ".rs", ".go", ".java", ".c", ".cpp", ".h",
    ".rb", ".ts", ".tsx", ".jsx", ".sql", ".r", ".ps1",
}
INGEST_EXTS = TEXT_EXTS | CODE_EXTS

SKIP_DIRS = {
    "shadercache", "temp", "downloading", "depotcache", "__pycache__",
    ".git", "node_modules", "Mono", "MonoBleedingEdge", ".venv", "venv",
    "dist", "build", ".pytest_cache", "target", ".claude",
    "$RECYCLE.BIN", "System Volume Information", "WpSystem",
    "WUDownloadCache", "WindowsApps",
    # Keep benchmark prompts, docs, and result artifacts out of the
    # live working genome so they cannot be retrieved as evidence.
    "benchmarks",
}

MAX_FILE_SIZE = 200_000   # 200KB — avoids stalls on giant JSON/XML
MIN_FILE_SIZE = 50


def ingest_tree(root, genome, tagger, chunker, stats):
    """Walk a directory tree and ingest text/code files."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in INGEST_EXTS:
                stats["skipped"] += 1
                continue

            fpath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue

            if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
                stats["skipped"] += 1
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                stats["errors"] += 1
                continue

            ct = "code" if ext in CODE_EXTS else "text"
            strands = chunker.chunk(content, content_type=ct)
            for i, strand in enumerate(strands):
                gene = tagger.pack(
                    strand.content,
                    content_type=ct,
                    source_id=fpath,
                    sequence_index=i,
                )
                gene.is_fragment = strand.is_fragment
                genome.upsert_gene(gene)
                stats["genes"] += 1

            stats["files"] += 1

            if stats["genes"] % 200 == 0 and stats["genes"] > 0:
                elapsed = time.perf_counter() - stats["t0"]
                log.info(
                    "[%d files, %d genes] %.1f genes/s | %s",
                    stats["files"], stats["genes"],
                    stats["genes"] / elapsed,
                    os.path.basename(dirpath),
                )


def ingest_models(root, genome, tagger, stats):
    """Ingest Ollama model headers (GGUF metadata)."""
    # Reuse the model ingester logic inline
    try:
        from scripts.ingest_models import read_ollama_manifests, model_to_gene_content
    except ImportError:
        # Direct import if run from scripts dir
        sys.path.insert(0, os.path.dirname(__file__))
        from ingest_models import read_ollama_manifests, model_to_gene_content

    models = read_ollama_manifests(root)
    for model in models:
        content = model_to_gene_content(model)
        gene = tagger.pack(content, content_type="text", source_id=model["manifest_path"])
        genome.upsert_gene(gene)
        stats["genes"] += 1
        stats["files"] += 1
        log.info("  Model: %s (%.1f GB, %d tensors)",
                 model["name"], model["size_gb"],
                 model["gguf"]["tensor_count"] if model.get("gguf") else 0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="genome.db")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    genome = Genome(path=args.db, synonym_map={}, splade_enabled=True, entity_graph=True)
    tagger = CpuTagger()
    chunker = CodonChunker()

    stats = {"files": 0, "genes": 0, "skipped": 0, "errors": 0, "t0": time.perf_counter()}

    sources = [
        ("F:/Projects", "projects"),
        ("F:/SteamLibrary", "steam-f"),
        ("F:/OpenModels", "models"),
        ("E:/SteamLibrary", "steam-e"),
        ("E:/Program Files", "programs-e"),
        ("E:/NetMose", "netmose"),
    ]

    for root, label in sources:
        if not os.path.isdir(root):
            log.info("Skipping %s (not found)", root)
            continue

        if label == "models" and not args.skip_models:
            log.info("=== Ingesting models from %s ===", root)
            ingest_models(root, genome, tagger, stats)
        else:
            log.info("=== Ingesting %s (%s) ===", root, label)
            ingest_tree(root, genome, tagger, chunker, stats)

    elapsed = time.perf_counter() - stats["t0"]
    genome_stats = genome.stats()
    log.info("=" * 60)
    log.info("Full ingest complete")
    log.info("  Files: %d ingested, %d skipped, %d errors", stats["files"], stats["skipped"], stats["errors"])
    log.info("  Genes: %d in %.0fs (%.1f genes/s)", stats["genes"], elapsed, stats["genes"] / max(elapsed, 1))
    log.info("  Genome: %d total genes", genome_stats["total_genes"])


if __name__ == "__main__":
    main()

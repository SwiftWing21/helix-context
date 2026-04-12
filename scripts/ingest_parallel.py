"""
Parallel ingest — worker-queue architecture for 6-7x speedup.

Architecture:
  N worker processes (each with own spaCy instance, ~100ms/gene)
      ↓ Gene dicts via multiprocessing.Queue
  1 writer process (SQLite writes, <10ms/gene — never bottlenecks)

The spaCy NER pipeline is the bottleneck at ~100-150ms per doc.
SQLite WAL writes are <10ms. So N workers producing into a single
writer gives near-linear scaling up to ~6 workers before the GIL
and I/O contention kick in.

Default: min(cpu_count - 2, 6) workers. Scales down to 1 worker
on small machines. The writer is always a dedicated process.

Usage:
    python scripts/ingest_parallel.py                    # full ingest, auto workers
    python scripts/ingest_parallel.py --workers 4        # explicit worker count
    python scripts/ingest_parallel.py --dry-run           # count files only
    python scripts/ingest_parallel.py --db genome.db      # target DB

Sources are the same as ingest_all.py (F:/Projects, F:/SteamLibrary, etc).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest.parallel")

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
}

MAX_FILE_SIZE = 200_000
MIN_FILE_SIZE = 50

SOURCES = [
    ("F:/Projects", "projects"),
    ("F:/SteamLibrary", "steam-f"),
    ("F:/OpenModels", "models"),
    ("E:/SteamLibrary", "steam-e"),
    ("E:/Program Files", "programs-e"),
    ("E:/NetMose", "netmose"),
]


# ── File discovery (runs in main process) ─────────────────────────

def discover_files(sources: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Walk all source directories, return [(fpath, ext)] of ingestable files."""
    files: list[tuple[str, str]] = []
    for root, label in sources:
        if not os.path.isdir(root):
            log.info("Skipping %s (not found)", root)
            continue
        log.info("Discovering files in %s (%s)...", root, label)
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
                if MIN_FILE_SIZE <= size <= MAX_FILE_SIZE:
                    files.append((fpath, ext))
    return files


# ── Worker process (spaCy + tagger, CPU-bound) ────────────────────

_worker_tagger = None
_worker_chunker = None


def _init_worker():
    """Called once per worker process — loads spaCy model."""
    global _worker_tagger, _worker_chunker
    from helix_context.tagger import CpuTagger
    from helix_context.codons import CodonChunker
    _worker_tagger = CpuTagger()
    _worker_chunker = CodonChunker()


def _process_file(args: tuple[str, str]) -> list[dict[str, Any]]:
    """Process a single file → list of Gene dicts. Runs in worker process."""
    fpath, ext = args
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []

    ct = "code" if ext in CODE_EXTS else "text"
    strands = _worker_chunker.chunk(content, content_type=ct)
    genes: list[dict[str, Any]] = []

    for i, strand in enumerate(strands):
        try:
            gene = _worker_tagger.pack(
                strand.content,
                content_type=ct,
                source_id=fpath,
                sequence_index=i,
            )
            gene.is_fragment = strand.is_fragment
            genes.append(gene.model_dump())
        except Exception:
            pass

    return genes


# ── Main: discover → pool → write ─────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="F:/Projects/helix-context/genome.db")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker count (0=auto: cpu_count - 2, max 6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover files only, no ingest")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--chunksize", type=int, default=4,
                        help="Files per worker batch (higher = less IPC overhead)")
    args = parser.parse_args()

    # Discover files
    t0 = time.perf_counter()
    files = discover_files(SOURCES)
    discover_time = time.perf_counter() - t0
    log.info("Discovered %d files in %.1fs", len(files), discover_time)

    if args.dry_run:
        total_bytes = sum(os.path.getsize(p) for p, _ in files)
        log.info("Dry-run: %d files, %.1f MB total", len(files), total_bytes / 1_048_576)
        return 0

    if not files:
        log.error("No files found — nothing to ingest")
        return 2

    # Worker count
    n_workers = args.workers or min(max(1, os.cpu_count() - 2), 6)
    log.info("Starting %d workers (chunksize=%d)", n_workers, args.chunksize)

    # Writer setup (main process)
    from helix_context.genome import Genome
    from helix_context.schemas import Gene
    genome = Genome(path=args.db, synonym_map={}, splade_enabled=True, entity_graph=True)

    stats = {"files": 0, "genes": 0, "errors": 0}
    t0 = time.perf_counter()

    # Process files in parallel, write sequentially
    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        for gene_dicts in pool.imap_unordered(_process_file, files, chunksize=args.chunksize):
            if not gene_dicts:
                stats["errors"] += 1
                continue

            for gd in gene_dicts:
                try:
                    gene = Gene(**gd)
                    genome.upsert_gene(gene)
                    stats["genes"] += 1
                except Exception:
                    stats["errors"] += 1

            stats["files"] += 1

            if stats["genes"] % 500 == 0 and stats["genes"] > 0:
                elapsed = time.perf_counter() - t0
                log.info(
                    "[%d files | %d genes | %.1f genes/s | %d errors]",
                    stats["files"], stats["genes"],
                    stats["genes"] / max(elapsed, 0.01),
                    stats["errors"],
                )

    # Final checkpoint
    try:
        genome.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    log.info("=" * 60)
    log.info("Parallel ingest complete")
    log.info("  Workers:  %d", n_workers)
    log.info("  Files:    %d (%d errors)", stats["files"], stats["errors"])
    log.info("  Genes:    %d in %.0fs (%.1f genes/s)", stats["genes"], elapsed,
             stats["genes"] / max(elapsed, 1))
    log.info("  Speedup:  %.1fx vs single-threaded (est 6 genes/s baseline)",
             (stats["genes"] / max(elapsed, 1)) / 6.0)
    return 0


if __name__ == "__main__":
    mp.freeze_support()  # Required on Windows
    sys.exit(main())

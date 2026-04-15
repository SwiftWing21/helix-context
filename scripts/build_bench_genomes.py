"""
Build reproducible bench-target genomes for cross-comparison.

Produces two genome files distinct from the working genome.db so that:
  * bench results stop drifting as the working genome accumulates
    organic conversational state and bench-output pollution
  * multiple bench genomes can be compared side-by-side to separate
    pipeline improvements from corpus changes

Targets built (configurable via --target):
  helix   -> F:/Projects/helix-context/ only (tight, ~1-3k genes, <10 min)
  organic -> F:/Projects/ minus every `benchmarks/` dir (wide, 1-2 hrs)
  both    → helix then organic (default)

Output sits next to the working genome.db as:
  genome_bench_helix.db
  genome_bench_organic.db

Usage:
  python scripts/build_bench_genomes.py --target helix
  python scripts/build_bench_genomes.py --target organic
  python scripts/build_bench_genomes.py              # builds both

Notes on the SKIP set (on top of ingest_all.py's list):
  * Adds `benchmarks` — these dirs hold JSON bench output that creates
    the "User queried for X; assistant reported unknown" needle-fixture
    co-activation clusters seen on 2026-04-14.
  * Adds `cwola_export` — same reason (analysis outputs, not source).
  * Adds `genome*.db`, `*.jsonl` — artifact files that slip through the
    extension filter on older layouts.
"""

from __future__ import annotations

import argparse
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
log = logging.getLogger("bench.build")


TEXT_EXTS = {".txt", ".md", ".cfg", ".ini", ".conf", ".properties", ".vdf", ".acf"}
CODE_EXTS = {
    ".lua", ".py", ".cs", ".js", ".json", ".yaml", ".yml", ".toml",
    ".bat", ".sh", ".html", ".rs", ".go", ".java", ".c", ".cpp", ".h",
    ".rb", ".ts", ".tsx", ".jsx", ".sql", ".r", ".ps1",
}
INGEST_EXTS = TEXT_EXTS | CODE_EXTS

# Bench-genome-specific SKIP set. Superset of ingest_all's production
# list PLUS anything that tainted the 2026-04-14 harmonic graph.
SKIP_DIRS = {
    # ingest_all.py's original list
    "shadercache", "temp", "downloading", "depotcache", "__pycache__",
    ".git", "node_modules", "Mono", "MonoBleedingEdge", ".venv", "venv",
    "dist", "build", ".pytest_cache", "target", ".claude",
    "$RECYCLE.BIN", "System Volume Information", "WpSystem",
    "WUDownloadCache", "WindowsApps",
    # bench-genome additions — the whole point of this target
    "benchmarks",          # needle-fixture JSON outputs
    "cwola_export",        # collab analysis outputs
    "logs",                # transient diagnostic dumps
    "training",            # model weights, not content
}

MAX_FILE_SIZE = 200_000
MIN_FILE_SIZE = 50


def ingest_tree(root: str, genome: Genome, tagger, chunker, stats: dict) -> None:
    """Walk `root` and ingest matching files, respecting SKIP_DIRS."""
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
                try:
                    gene = tagger.pack(
                        strand.content,
                        content_type=ct,
                        source_id=fpath,
                        sequence_index=i,
                    )
                    gene.is_fragment = strand.is_fragment
                    genome.upsert_gene(gene)
                    stats["genes"] += 1
                except Exception:
                    stats["errors"] += 1

            stats["files"] += 1

            if stats["genes"] % 200 == 0 and stats["genes"] > 0:
                elapsed = time.perf_counter() - stats["t0"]
                log.info(
                    "[%d files, %d genes] %.1f genes/s | %s",
                    stats["files"], stats["genes"],
                    stats["genes"] / max(elapsed, 0.001),
                    os.path.basename(dirpath)[:60],
                )


def build(db_path: str, roots: list) -> dict:
    """Build a fresh genome at `db_path` from the given source roots."""
    out_dir = os.path.dirname(os.path.abspath(db_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(db_path):
        log.info("Removing existing %s to start fresh", db_path)
        os.remove(db_path)

    log.info("Opening fresh genome at %s", db_path)
    genome = Genome(
        path=db_path, synonym_map={},
        splade_enabled=True, entity_graph=True,
    )
    tagger = CpuTagger()
    chunker = CodonChunker()

    stats = {
        "files": 0, "genes": 0, "skipped": 0, "errors": 0,
        "t0": time.perf_counter(),
    }

    for root in roots:
        if not os.path.exists(root):
            log.warning("Root %s does not exist, skipping", root)
            continue
        log.info("=== Ingesting %s ===", root)
        ingest_tree(root, genome, tagger, chunker, stats)

    elapsed = time.perf_counter() - stats["t0"]
    stats["elapsed_s"] = elapsed

    # Report
    genome_stats = genome.stats()
    # harmonic_links count isn't in stats(); query directly
    try:
        hl_row = genome.conn.execute(
            "SELECT COUNT(*) AS n FROM harmonic_links"
        ).fetchone()
        hl_count = int(hl_row["n"]) if hl_row else 0
    except Exception:
        hl_count = 0

    log.info("=" * 60)
    log.info("DONE building %s in %.1fs", db_path, elapsed)
    log.info("  Files: %d ingested, %d skipped, %d errors",
             stats["files"], stats["skipped"], stats["errors"])
    log.info("  Genes: %d ingested (final genome has %d total)",
             stats["genes"], genome_stats.get("total_genes", -1))
    log.info("  Harmonic links: %d", hl_count)

    genome.close()
    stats["total_genes"] = genome_stats.get("total_genes", 0)
    stats["harmonic_links"] = hl_count
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["helix", "organic", "both"],
        default="both",
        help="Which bench genome(s) to build",
    )
    parser.add_argument(
        "--out-dir", default=".",
        help="Directory to write the bench genome files into "
             "(default: cwd, sits next to genome.db)",
    )
    args = parser.parse_args()

    results = {}

    if args.target in ("helix", "both"):
        log.info("### Target 1/2: helix-context self (tight bench) ###")
        results["helix"] = build(
            db_path=os.path.join(args.out_dir, "genome_bench_helix.db"),
            roots=[
                os.path.join("F:\\", "Projects", "helix-context"),
            ],
        )

    if args.target in ("organic", "both"):
        log.info("### Target 2/2: F:\\Projects minus benchmarks (wide bench) ###")
        results["organic"] = build(
            db_path=os.path.join(args.out_dir, "genome_bench_organic.db"),
            roots=[
                os.path.join("F:\\", "Projects"),
            ],
        )

    log.info("=" * 60)
    log.info("SUMMARY")
    for name, s in results.items():
        log.info(
            "  %s: %d genes, %d links, %.1fs (files=%d skipped=%d errors=%d)",
            name, s["total_genes"], s["harmonic_links"], s["elapsed_s"],
            s["files"], s["skipped"], s["errors"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

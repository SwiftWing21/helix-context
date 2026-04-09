"""
Resequence — Re-encode all genes through the CPU pipeline.

Reads every gene from the existing genome.db, re-packs each through
CpuTagger + SPLADE + entity graph, and writes to a new genome.

Usage:
    python scripts/resequence_cpu.py                    # default: genome.db -> genome_cpu.db
    python scripts/resequence_cpu.py --in-place         # overwrite genome.db (backs up first)
    python scripts/resequence_cpu.py --output fresh.db  # write to custom path
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helix_context.config import load_config
from helix_context.tagger import CpuTagger
from helix_context.genome import Genome
from helix_context.codons import CodonChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("resequence")


def main():
    parser = argparse.ArgumentParser(description="Resequence genome through CPU pipeline")
    parser.add_argument("--input", default="genome.db", help="Source genome DB")
    parser.add_argument("--output", default="genome_cpu.db", help="Destination genome DB")
    parser.add_argument("--in-place", action="store_true", help="Overwrite source (backs up first)")
    parser.add_argument("--batch-size", type=int, default=50, help="Commit every N genes")
    parser.add_argument("--skip-splade", action="store_true", help="Skip SPLADE encoding (faster)")
    args = parser.parse_args()

    src_path = args.input
    if args.in_place:
        backup = f"{src_path}.bak.{int(time.time())}"
        log.info("Backing up %s -> %s", src_path, backup)
        shutil.copy2(src_path, backup)
        dst_path = src_path
    else:
        dst_path = args.output
        if os.path.exists(dst_path):
            log.warning("Output %s exists, will be overwritten", dst_path)
            os.unlink(dst_path)

    # Load config
    config = load_config()
    skip_splade = args.skip_splade

    # Open source genome (read-only)
    src = sqlite3.connect(src_path)
    src.row_factory = sqlite3.Row
    total = src.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
    log.info("Source genome: %d genes from %s", total, src_path)

    # Create destination genome with all features enabled
    dst = Genome(
        path=dst_path if not args.in_place else f"{dst_path}.tmp",
        synonym_map=config.synonym_map,
        splade_enabled=config.ingestion.splade_enabled and not skip_splade,
        entity_graph=config.ingestion.entity_graph,
    )

    # Initialize tagger
    tagger = CpuTagger(synonym_map=config.synonym_map)

    # Read all genes
    rows = src.execute(
        "SELECT gene_id, content, source_id, is_fragment, chromatin, "
        "embedding, version, supersedes FROM genes ORDER BY ROWID"
    ).fetchall()
    src.close()

    t_start = time.perf_counter()
    success = 0
    errors = 0
    splade_time = 0.0
    tagger_time = 0.0

    for i, row in enumerate(rows):
        content = row["content"]
        source_id = row["source_id"] or ""
        is_fragment = bool(row["is_fragment"])

        # Detect content type from source
        if source_id:
            ext = os.path.splitext(source_id)[1].lower()
            if ext in (".py", ".rs", ".js", ".ts", ".go", ".java", ".c", ".cpp", ".h",
                        ".rb", ".lua", ".sh", ".bat", ".ps1", ".toml", ".yaml", ".yml",
                        ".json", ".html", ".css", ".sql"):
                content_type = "code"
            else:
                content_type = "text"
        else:
            content_type = "text"

        try:
            # CPU pack
            t0 = time.perf_counter()
            gene = tagger.pack(
                content,
                content_type=content_type,
                source_id=source_id,
                sequence_index=i,
            )
            tagger_time += time.perf_counter() - t0

            gene.is_fragment = is_fragment

            # Preserve embedding if present
            emb_raw = row["embedding"]
            if emb_raw:
                try:
                    from helix_context.accel import json_loads
                    gene.embedding = json_loads(emb_raw)
                except Exception:
                    pass

            # Upsert (SPLADE encoding happens inside genome.upsert_gene if enabled)
            t_sp = time.perf_counter()
            dst.upsert_gene(gene)
            splade_time += time.perf_counter() - t_sp

            success += 1
        except Exception as exc:
            errors += 1
            if errors <= 5:
                log.warning("Gene %d failed: %s", i, exc)

        # Progress
        if (i + 1) % args.batch_size == 0 or i == len(rows) - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] %.0f genes/s | tagger=%.1fs splade+db=%.1fs | ETA %.0fs | ok=%d err=%d",
                i + 1, total, rate, tagger_time, splade_time, eta, success, errors,
            )

    elapsed = time.perf_counter() - t_start

    # If in-place, swap tmp over original
    if args.in_place:
        tmp_path = f"{dst_path}.tmp"
        dst.conn.close()
        if os.path.exists(dst_path):
            os.unlink(dst_path)
        os.rename(tmp_path, dst_path)
        # Clean up WAL/SHM from tmp
        for suffix in ("-wal", "-shm"):
            tmp_extra = tmp_path + suffix
            if os.path.exists(tmp_extra):
                os.unlink(tmp_extra)
        log.info("In-place swap complete: %s", dst_path)

    # Summary
    stats = dst.stats() if not args.in_place else {}
    log.info("=" * 60)
    log.info("Resequence complete")
    log.info("  Genes: %d ok, %d errors (of %d)", success, errors, total)
    log.info("  Time: %.1fs (%.1f genes/s)", elapsed, total / elapsed)
    log.info("  Tagger: %.1fs (%.1f ms/gene)", tagger_time, tagger_time / max(success, 1) * 1000)
    log.info("  SPLADE+DB: %.1fs (%.1f ms/gene)", splade_time, splade_time / max(success, 1) * 1000)
    log.info("  Output: %s", dst_path if not args.in_place else src_path)
    if stats:
        log.info("  Genome stats: %s", stats)


if __name__ == "__main__":
    main()

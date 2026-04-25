"""
One-shot backfill: inject filename-derived domain tokens into promoter_index.

Since 2026-04-24, CpuTagger._extract_filename_domains() adds filename stem
and parent-dir tokens to domains at ingest time.  Genes already in the genome
need a one-shot patch so their promoter_index rows gain these tokens.

The backfill is ADDITIVE and IDEMPOTENT:
  - Reads every gene that has a source_id.
  - Derives the same tokens that CpuTagger._extract_filename_domains() now
    produces (shares the same logic via the tagger helper).
  - INSERT OR IGNOREs into promoter_index with tag_type = 'domain'.
  - Rows already present are untouched.

Usage:
    python scripts/backfill_filename_domains.py [--db PATH] [--dry-run]

Default DB: helix.toml genome.path (via config loader).
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_context.config import load_config  # noqa: E402
from helix_context.tagger import CpuTagger  # noqa: E402


def _get_filename_tokens(tagger: CpuTagger, source_id: str) -> list[str]:
    """Return the filename-derived domain tokens for a given source_id."""
    return tagger._extract_filename_domains(source_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="genome DB path (default: from helix.toml)")
    ap.add_argument("--batch-size", type=int, default=2000, help="commit interval")
    ap.add_argument("--dry-run", action="store_true", help="scan and count without writing")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("backfill_filename_domains")

    db = args.db
    if db is None:
        cfg = load_config()
        db = cfg.genome.path
    log.info("backfilling promoter_index (filename domains) on %s", db)
    if args.dry_run:
        log.info("DRY-RUN mode — no writes")

    conn = sqlite3.connect(db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL mode for concurrent-safe writes (matches genome.py)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    rows = conn.execute(
        "SELECT gene_id, source_id FROM genes "
        "WHERE source_id IS NOT NULL AND source_id != ''"
    ).fetchall()
    total = len(rows)
    log.info("scanning %d genes with source_id", total)

    tagger = CpuTagger()
    t0 = time.time()
    stamped = 0
    skipped = 0

    for i, row in enumerate(rows):
        gene_id: str = row["gene_id"]
        source_id: str = row["source_id"]

        tokens = _get_filename_tokens(tagger, source_id)
        if not tokens:
            skipped += 1
            continue

        if not args.dry_run:
            # promoter_index has no UNIQUE constraint; guard idempotency by
            # fetching existing domain tags for this gene before inserting.
            existing = {
                r[0]
                for r in conn.execute(
                    "SELECT tag_value FROM promoter_index "
                    "WHERE gene_id=? AND tag_type='domain'",
                    (gene_id,),
                ).fetchall()
            }
            for tok in tokens:
                if tok not in existing:
                    conn.execute(
                        "INSERT INTO promoter_index (gene_id, tag_type, tag_value) "
                        "VALUES (?, 'domain', ?)",
                        (gene_id, tok),
                    )

        stamped += 1

        if not args.dry_run and (i + 1) % args.batch_size == 0:
            conn.commit()
            elapsed = time.time() - t0
            log.info("  %d / %d genes processed (%.1fs)", i + 1, total, elapsed)

    if not args.dry_run:
        conn.commit()
    conn.close()

    elapsed = time.time() - t0
    log.info(
        "done: %d genes patched, %d skipped (no filename tokens), %.1fs",
        stamped,
        skipped,
        elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

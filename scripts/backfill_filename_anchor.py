"""
One-shot backfill: populate filename_index from existing genes.

New ingests auto-index via helix_context.genome.upsert_gene, but genes
already in the genome at the time filename_anchor shipped have empty
filename_index rows. Run this once per genome DB.

Usage:
    python scripts/backfill_filename_anchor.py [--db PATH]

Default DB: helix.toml genome.path (via config loader).
Safe to re-run — uses INSERT OR IGNORE.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_context import filename_anchor  # noqa: E402
from helix_context.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="genome DB path (default: from helix.toml)")
    ap.add_argument("--batch-size", type=int, default=5000, help="commit interval")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("backfill_filename_anchor")

    db = args.db
    if db is None:
        cfg = load_config()
        db = cfg.genome.path
    log.info("backfilling filename_index on %s", db)

    conn = sqlite3.connect(db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    filename_anchor.ensure_schema(conn)

    rows = conn.execute(
        "SELECT gene_id, source_id FROM genes "
        "WHERE source_id IS NOT NULL AND source_id != ''"
    ).fetchall()
    total = len(rows)
    log.info("scanning %d genes", total)

    t0 = time.time()
    stamped = 0
    skipped = 0
    for i, r in enumerate(rows, 1):
        stem = filename_anchor.filename_stem(r["source_id"])
        if stem is None:
            skipped += 1
            continue
        conn.execute(
            "INSERT OR IGNORE INTO filename_index (filename_stem, gene_id) VALUES (?, ?)",
            (stem, r["gene_id"]),
        )
        stamped += 1
        if i % args.batch_size == 0:
            conn.commit()
            log.info("  %d/%d genes processed (stamped=%d skipped=%d)",
                     i, total, stamped, skipped)

    conn.commit()
    # Final verification
    n_filename = conn.execute("SELECT COUNT(*) FROM filename_index").fetchone()[0]
    n_distinct = conn.execute(
        "SELECT COUNT(DISTINCT filename_stem) FROM filename_index"
    ).fetchone()[0]
    conn.close()

    dt = time.time() - t0
    log.info("DONE in %.1fs: stamped=%d skipped=%d rows_in_index=%d distinct_stems=%d",
             dt, stamped, skipped, n_filename, n_distinct)
    return 0


if __name__ == "__main__":
    sys.exit(main())

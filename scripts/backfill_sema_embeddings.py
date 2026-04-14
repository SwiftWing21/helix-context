"""
Backfill ΣĒMA embeddings for genes that are missing them.

Runs independently of any LLM ingest path. ΣĒMA is a deterministic CPU
encoder — safe to run whenever the `genes.embedding` column is null.

Idempotent: only touches rows where embedding IS NULL. Safe to re-run.

Expected runtime: ~1-2 ms per gene on CPU; ~30s for 17K genes.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_context.sema import SemaCodec  # noqa: E402

DB_PATH = "F:/Projects/helix-context/genome.db"
BATCH_SIZE = 200  # commit every N rows for WAL friendliness
REPORT_EVERY = 1000


def main() -> int:
    codec = SemaCodec()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()

    # Count work
    cur.execute("SELECT COUNT(*) FROM genes WHERE embedding IS NULL")
    (total_missing,) = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM genes WHERE embedding IS NOT NULL")
    (total_have,) = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM genes")
    (total,) = cur.fetchone()

    print(f"Genome: {total} total genes")
    print(f"  have embedding:     {total_have}")
    print(f"  missing embedding:  {total_missing}")
    if total_missing == 0:
        print("Nothing to backfill.")
        return 0

    # Stream candidates
    reader = conn.cursor()
    reader.execute(
        "SELECT gene_id, content FROM genes WHERE embedding IS NULL AND content IS NOT NULL"
    )

    t0 = time.perf_counter()
    processed = 0
    updated = 0
    errors = 0
    since_commit = 0

    for gene_id, content in reader:
        processed += 1
        if not content:
            continue
        try:
            vec = codec.encode(content)
            emb_json = json.dumps(vec if isinstance(vec, list) else vec.tolist())
            cur.execute(
                "UPDATE genes SET embedding = ? WHERE gene_id = ?",
                (emb_json, gene_id),
            )
            updated += 1
            since_commit += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if errors <= 5:
                print(f"  error on gene {gene_id[:12]}: {exc}")

        if since_commit >= BATCH_SIZE:
            conn.commit()
            since_commit = 0

        if processed % REPORT_EVERY == 0:
            elapsed = time.perf_counter() - t0
            rate = processed / elapsed
            remaining = (total_missing - processed) / rate if rate > 0 else 0
            print(
                f"  [{processed}/{total_missing}] updated={updated} errors={errors} "
                f"| {rate:.0f} genes/s | ~{remaining:.0f}s remaining"
            )

    # Final commit
    conn.commit()
    elapsed = time.perf_counter() - t0

    # Verify
    cur.execute("SELECT COUNT(*) FROM genes WHERE embedding IS NULL")
    (still_missing,) = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM genes WHERE embedding IS NOT NULL")
    (now_have,) = cur.fetchone()

    print()
    print("=" * 50)
    print(f"Complete in {elapsed:.1f}s")
    print(f"  processed:        {processed}")
    print(f"  updated:          {updated}")
    print(f"  errors:           {errors}")
    print(f"  still missing:    {still_missing}")
    print(f"  have embedding:   {now_have} ({100*now_have/total:.1f}%)")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())

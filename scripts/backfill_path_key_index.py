"""
Backfill the path_key_index table for an existing genome.

The path_key_index maps (path_token, kv_key) → gene_id for fast compound
lookup on template queries. It's normally populated at ingest time by
upsert_gene, but existing genes (pre-this-commit) have no rows in the
index yet. This script fills it in one pass.

Zero LLM calls. Pure regex + SQL.

Usage:
    # Dry run (counts what would be inserted):
    python scripts/backfill_path_key_index.py --db F:/Projects/helix-context/genome.db

    # Apply (writes rows):
    python scripts/backfill_path_key_index.py --db F:/Projects/helix-context/genome.db --apply

    # Backup before applying:
    python scripts/backfill_path_key_index.py --db F:/Projects/helix-context/genome.db --apply --backup

Coordination:
  - WAL mode allows concurrent reads. Writes queue behind the server's
    writer lock. For the 17K-gene genome, the backfill takes ~20s and
    inserts ~200-500K rows. Running against a live server is safe but
    slower — you'll see brief query latency bumps during INSERT batches.
  - Stop the server first if you want a clean-slate operation:
      # Kill whatever holds :11437, run backfill, restart.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path


def _ensure_root_on_path():
    """Make helix_context importable when run as a script."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create path_key_index if it doesn't exist.

    The Genome __init__ normally handles this, but a caller may point at
    a DB that hasn't been opened by the current code yet.
    """
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS path_key_index (
        path_token TEXT NOT NULL,
        kv_key     TEXT NOT NULL,
        gene_id    TEXT NOT NULL,
        PRIMARY KEY (path_token, kv_key, gene_id)
    )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pki_lookup "
        "ON path_key_index(path_token, kv_key)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pki_gene "
        "ON path_key_index(gene_id)"
    )
    conn.commit()


def backfill(
    db_path: str,
    dry_run: bool = True,
    backup: bool = False,
    batch_size: int = 2000,
) -> dict:
    """Populate path_key_index for every gene with source_id + key_values.

    Returns stats dict with counts. On dry_run=True, no writes happen.
    """
    _ensure_root_on_path()
    from helix_context.genome import path_tokens, _kv_keys_from_list

    if backup and not dry_run:
        bak = db_path + ".pre-pki-backfill.bak"
        print(f"Backing up {db_path} -> {bak}")
        shutil.copy2(db_path, bak)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    _ensure_table(conn)

    cur = conn.cursor()

    # Count starting state
    total_genes = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
    pre_rows = cur.execute("SELECT COUNT(*) FROM path_key_index").fetchone()[0]
    eligible = cur.execute(
        "SELECT COUNT(*) FROM genes "
        "WHERE source_id IS NOT NULL AND source_id != '' "
        "AND key_values IS NOT NULL AND key_values != '' AND key_values != '{}'"
    ).fetchone()[0]

    stats = {
        "total_genes": total_genes,
        "eligible_genes": eligible,
        "pre_rows": pre_rows,
        "rows_inserted": 0,
        "genes_processed": 0,
        "genes_with_no_tokens": 0,
        "genes_with_bad_kv": 0,
        "elapsed_s": 0.0,
    }

    print(f"Genome: {total_genes} total, {eligible} eligible "
          f"(source_id + key_values), {pre_rows} existing index rows")
    if dry_run:
        print("Mode: DRY RUN (no writes)")
    else:
        print("Mode: APPLY")

    start = time.time()

    # Stream in batches to keep memory bounded
    offset = 0
    pending: list = []

    while True:
        rows = cur.execute(
            "SELECT gene_id, source_id, key_values FROM genes "
            "WHERE source_id IS NOT NULL AND source_id != '' "
            "AND key_values IS NOT NULL AND key_values != '' "
            "AND key_values != '{}' "
            "LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()

        if not rows:
            break

        for r in rows:
            gid = r["gene_id"]
            src = r["source_id"]
            kv_raw = r["key_values"]

            try:
                kv = json.loads(kv_raw) if isinstance(kv_raw, str) else kv_raw
                if not kv:
                    stats["genes_with_bad_kv"] += 1
                    continue
            except Exception:
                stats["genes_with_bad_kv"] += 1
                continue

            p_tokens = path_tokens(src)
            if not p_tokens:
                stats["genes_with_no_tokens"] += 1
                continue

            kv_keys = _kv_keys_from_list(kv)
            if not kv_keys:
                stats["genes_with_bad_kv"] += 1
                continue
            for pt in p_tokens:
                for kk in kv_keys:
                    pending.append((pt, kk, gid))

            stats["genes_processed"] += 1

        if not dry_run and pending:
            # Batched insert
            cur.executemany(
                "INSERT OR IGNORE INTO path_key_index "
                "(path_token, kv_key, gene_id) VALUES (?, ?, ?)",
                pending,
            )
            conn.commit()
            stats["rows_inserted"] += len(pending)
            pending.clear()
        elif dry_run and pending:
            stats["rows_inserted"] += len(pending)
            pending.clear()

        offset += batch_size
        if stats["genes_processed"] and stats["genes_processed"] % 5000 == 0:
            print(f"  ... {stats['genes_processed']} genes processed, "
                  f"{stats['rows_inserted']} rows "
                  f"({'would insert' if dry_run else 'inserted'})")

    stats["elapsed_s"] = round(time.time() - start, 2)

    post_rows = cur.execute("SELECT COUNT(*) FROM path_key_index").fetchone()[0]
    stats["post_rows"] = post_rows

    conn.close()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="Path to genome.db")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write rows (default: dry run)")
    ap.add_argument("--backup", action="store_true",
                    help="Copy DB to .pre-pki-backfill.bak before writing")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--output", help="Write stats JSON here")
    args = ap.parse_args()

    stats = backfill(
        db_path=args.db,
        dry_run=not args.apply,
        backup=args.backup,
        batch_size=args.batch_size,
    )

    print()
    print("=" * 60)
    print(f"{'Backfill (DRY RUN)' if not args.apply else 'Backfill applied'}")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")
    print()

    if args.output:
        Path(args.output).write_text(json.dumps(stats, indent=2))
        print(f"Stats written to {args.output}")

    if not args.apply:
        print("DRY RUN — no changes written. Pass --apply to execute.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Backfill synthetic session_id + party_id on existing cwola_log rows.

Problem (discovered 2026-04-13):
    The /context endpoint was passing NULL to cwola.log_query when clients
    didn't explicitly send `session_id` / `party_id` in the request body.
    Since sweep_buckets can't detect re-queries without a session, EVERY
    row defaulted to Bucket A. Result: 791 rows, 100% A, 0% B, 100% NULL
    session_id — unusable for CWoLa training.

Server-side fix (commit pending):
    server.py now falls back to a synthetic session_id from
    sha1(client_ip + time_window_bucket) and a default party_id from
    helix.toml `[session]`. See the cwola_session_id block in
    helix_context/server.py (~line 500).

This script:
    Applies the same synthetic pattern retroactively to rows with NULL
    session_id, then re-runs cwola.sweep_buckets so buckets / requery_delta_s
    get reassigned based on the now-visible sessions. Some A-buckets will
    flip to B when a re-query within 60s becomes visible.

Usage:
    python scripts/backfill_cwola_sessions.py --dry-run    # see what would change
    python scripts/backfill_cwola_sessions.py              # apply backfill
    python scripts/backfill_cwola_sessions.py --revert     # undo (sets those rows back to NULL)

Safety:
    - Uses a transaction — either all backfill applies or none does.
    - Only touches rows where session_id IS NULL (idempotent).
    - Caller must stop the helix server first to avoid write contention.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("backfill_cwola_sessions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", type=Path, default=Path("genome.db"),
                   help="Path to genome.db (default: ./genome.db)")
    p.add_argument("--window-s", type=int, default=300,
                   help="Session grouping window in seconds (default: 300 = 5 min); must match helix.toml")
    p.add_argument("--party-id", type=str, default="swift_wing21",
                   help="Default party_id to assign (default: swift_wing21)")
    p.add_argument("--client-ip", type=str, default="historical",
                   help="Synthetic client IP for hash input (default: 'historical'; same-IP=same-session)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would change without writing")
    p.add_argument("--revert", action="store_true",
                   help="Revert previous backfill — set party_id and synthetic session_ids back to NULL")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def synth_session_id(client_ip: str, bucket_ts: int) -> str:
    """Same formula as server.py's fallback."""
    digest = hashlib.sha1(f"{client_ip}:{bucket_ts}".encode("utf-8")).hexdigest()[:12]
    return f"syn_{digest}"


def current_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT "
        "  COUNT(*), "
        "  SUM(CASE WHEN bucket='A' THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN bucket='B' THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN bucket IS NULL THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN session_id IS NULL THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN session_id LIKE 'syn_%' THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN party_id IS NULL THEN 1 ELSE 0 END) "
        "FROM cwola_log"
    ).fetchone()
    return {
        "total": row[0] or 0,
        "a": row[1] or 0,
        "b": row[2] or 0,
        "bucket_null": row[3] or 0,
        "session_null": row[4] or 0,
        "session_synth": row[5] or 0,
        "party_null": row[6] or 0,
    }


def print_stats(label: str, stats: dict) -> None:
    print(f"  {label}: total={stats['total']}, A={stats['a']}, B={stats['b']}, "
          f"bucket_null={stats['bucket_null']}, session_null={stats['session_null']}, "
          f"session_synth={stats['session_synth']}, party_null={stats['party_null']}")


def do_backfill(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    """Return number of rows touched."""
    rows = conn.execute(
        "SELECT retrieval_id, ts FROM cwola_log WHERE session_id IS NULL ORDER BY ts"
    ).fetchall()

    if not rows:
        log.info("no rows with NULL session_id — nothing to do")
        return 0

    window = max(1, int(args.window_s))
    buckets: dict[int, list[int]] = defaultdict(list)
    for rid, ts in rows:
        b = int(ts // window) * window
        buckets[b].append(rid)

    log.info("found %d rows to backfill across %d session-windows",
             len(rows), len(buckets))

    updates = []
    for bucket_ts, rids in buckets.items():
        sid = synth_session_id(args.client_ip, bucket_ts)
        log.debug("  window ts=%s -> %d rows -> session_id=%s",
                  bucket_ts, len(rids), sid)
        for rid in rids:
            updates.append((sid, args.party_id, rid))

    if args.dry_run:
        print(f"\n  DRY RUN: would update {len(updates)} rows across {len(buckets)} sessions")
        # Show first 3 window sizes for sanity
        sizes = sorted([len(rs) for rs in buckets.values()], reverse=True)
        print(f"  session sizes: top 5 = {sizes[:5]}; median = {sizes[len(sizes) // 2]}; min = {min(sizes)}")
        return 0

    try:
        conn.execute("BEGIN")
        conn.executemany(
            "UPDATE cwola_log SET session_id = ?, party_id = ? "
            "WHERE retrieval_id = ? AND session_id IS NULL",
            updates,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        log.exception("backfill failed, rolled back")
        raise

    return len(updates)


def do_revert(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    """Undo backfill — only touches rows whose session_id starts with 'syn_'.

    Does NOT touch rows with real (non-synthetic) session_ids.
    """
    # Count first for messaging
    row = conn.execute(
        "SELECT COUNT(*) FROM cwola_log WHERE session_id LIKE 'syn_%'"
    ).fetchone()
    n = row[0] if row else 0
    log.info("will revert %d synthetic sessions", n)

    if args.dry_run:
        print(f"\n  DRY RUN: would revert {n} rows (session_id LIKE 'syn_%')")
        return 0

    try:
        conn.execute("BEGIN")
        # Reset synthetic sessions and their downstream bucket assignments.
        cur = conn.execute(
            "UPDATE cwola_log SET session_id = NULL, party_id = NULL, "
            "bucket = NULL, bucket_assigned_at = NULL, requery_delta_s = NULL "
            "WHERE session_id LIKE 'syn_%'"
        )
        conn.commit()
        return cur.rowcount
    except Exception:
        conn.rollback()
        log.exception("revert failed, rolled back")
        raise


def do_sweep(conn: sqlite3.Connection) -> int:
    """Re-run cwola.sweep_buckets on the now-populated sessions."""
    # Reset ALL non-NULL buckets that were assigned under the "session=NULL
    # -> default to A" rule, so sweep_buckets can reassign them with real
    # session context. This is only safe because we know all pre-fix rows
    # had session=NULL. Rows with real (non-synthetic) session_ids that
    # were assigned buckets correctly should NOT be reset.
    cur = conn.execute(
        "UPDATE cwola_log SET bucket = NULL, bucket_assigned_at = NULL, "
        "requery_delta_s = NULL "
        "WHERE session_id LIKE 'syn_%' AND bucket IS NOT NULL"
    )
    reset_count = cur.rowcount
    conn.commit()
    log.info("reset %d stale bucket assignments on synthetic sessions", reset_count)

    # Now import and run sweep_buckets
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from helix_context import cwola
    updates = cwola.sweep_buckets(conn)
    log.info("sweep_buckets assigned %d new bucket labels", updates)
    return updates


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.db.exists():
        log.error("genome.db not found: %s", args.db.resolve())
        return 1

    conn = sqlite3.connect(str(args.db.resolve()), timeout=30)
    try:
        before = current_stats(conn)
        print("\n=== Before ===")
        print_stats("stats", before)

        if args.revert:
            n = do_revert(conn, args)
            print(f"\n  reverted {n} rows")
        else:
            n = do_backfill(conn, args)
            if not args.dry_run and n > 0:
                print(f"\n  backfilled {n} rows — now running sweep_buckets...")
                # Sleep briefly to ensure ts > cutoff for all rows
                # (sweep_buckets requires ts <= now - BUCKET_WINDOW_S)
                assigned = do_sweep(conn)
                print(f"  sweep_buckets assigned {assigned} bucket labels")

        after = current_stats(conn)
        print("\n=== After ===")
        print_stats("stats", after)

        # Delta summary
        print("\n=== Delta ===")
        print(f"  A: {before['a']} -> {after['a']} ({after['a'] - before['a']:+d})")
        print(f"  B: {before['b']} -> {after['b']} ({after['b'] - before['b']:+d})")
        print(f"  session_null: {before['session_null']} -> {after['session_null']} "
              f"({after['session_null'] - before['session_null']:+d})")
        print(f"  session_synth: {before['session_synth']} -> {after['session_synth']} "
              f"({after['session_synth'] - before['session_synth']:+d})")
        print()

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

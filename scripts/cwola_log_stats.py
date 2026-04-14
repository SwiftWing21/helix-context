"""Diagnostic stats over the full cwola_log table.

Companion to export_cwola_log.py — answers "do we have the data volume
and bucket separation to run Track A of the joint experiment?"

Usage:
    python scripts/cwola_log_stats.py
    python scripts/cwola_log_stats.py --db path/to/genome.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=Path("genome.db"))
    args = p.parse_args()

    if not args.db.exists():
        print(f"error: {args.db.resolve()} not found", file=sys.stderr)
        return 1

    uri = f"file:{args.db.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM cwola_log").fetchone()[0]
    print(f"total rows:  {total}")
    print()

    print("bucket distribution (whole table):")
    for row in cur.execute(
        "SELECT COALESCE(bucket,'NULL'), COUNT(*) "
        "FROM cwola_log GROUP BY bucket ORDER BY 2 DESC"
    ):
        print(f"  {row[0]:8s}: {row[1]}")
    print()

    print("party distribution (top 10):")
    for row in cur.execute(
        "SELECT COALESCE(party_id,'NULL'), COUNT(*) "
        "FROM cwola_log GROUP BY party_id ORDER BY 2 DESC LIMIT 10"
    ):
        print(f"  {row[0]:30s}: {row[1]}")
    print()

    print("temporal distribution:")
    row = cur.execute(
        "SELECT MIN(ts), MAX(ts), (MAX(ts)-MIN(ts))/3600.0 FROM cwola_log"
    ).fetchone()
    if row and row[0]:
        mn, mx, hrs = row
        print(f"  earliest: {datetime.fromtimestamp(mn, timezone.utc).isoformat()}")
        print(f"  latest:   {datetime.fromtimestamp(mx, timezone.utc).isoformat()}")
        print(f"  span:     {hrs:.1f} hours ({hrs / 24:.1f} days)")
    print()

    print("requery_delta_s histogram (rows with delta assigned):")
    buckets = [
        (0, 5), (5, 10), (10, 30), (30, 60),
        (60, 120), (120, 600), (600, 3600), (3600, 86400),
    ]
    any_printed = False
    for lo, hi in buckets:
        n = cur.execute(
            "SELECT COUNT(*) FROM cwola_log "
            "WHERE requery_delta_s >= ? AND requery_delta_s < ?",
            (lo, hi),
        ).fetchone()[0]
        if n > 0:
            any_printed = True
            print(f"  {lo:5d}-{hi:5d}s:  {n}")
    if not any_printed:
        print("  (no rows have requery_delta_s set — all either A with no next query or NULL)")
    n_null = cur.execute(
        "SELECT COUNT(*) FROM cwola_log WHERE requery_delta_s IS NULL"
    ).fetchone()[0]
    print(f"  NULL       :  {n_null}")
    print()

    # Session analysis
    print("session counts:")
    n_sessions = cur.execute(
        "SELECT COUNT(DISTINCT session_id) FROM cwola_log "
        "WHERE session_id IS NOT NULL"
    ).fetchone()[0]
    n_null_session = cur.execute(
        "SELECT COUNT(*) FROM cwola_log WHERE session_id IS NULL"
    ).fetchone()[0]
    print(f"  distinct session_ids: {n_sessions}")
    print(f"  rows with NULL session_id: {n_null_session}")
    print()

    # Histogram of retrievals per session
    if n_sessions > 0:
        print("retrievals-per-session distribution:")
        for row in cur.execute(
            "SELECT cnt, COUNT(*) FROM ("
            "  SELECT COUNT(*) AS cnt FROM cwola_log "
            "  WHERE session_id IS NOT NULL GROUP BY session_id"
            ") GROUP BY cnt ORDER BY cnt"
        ).fetchmany(20):
            cnt, n_sess = row
            print(f"  {cnt:4d} retrievals per session: {n_sess} session(s)")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

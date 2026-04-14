"""Export cwola_log rows for the Celestia × Helix joint experiment.

Reads genome.db (SQLite, read-only), filters cwola_log rows with assigned
A/B buckets, parses tier_features JSON, writes two artifacts:

    cwola_export_YYYYMMDD.json  - actual row data
    cwola_meta.json             - schema, stats, column dictionary

Usage:
    python scripts/export_cwola_log.py                    # full export, default paths
    python scripts/export_cwola_log.py --sample 100       # first-pass scan before governance review
    python scripts/export_cwola_log.py --redact           # drop raw query text (governance scan)
    python scripts/export_cwola_log.py --since 2026-04-01 # filter by date

Author/owner: helix side of the Celestia × Helix joint experiment
See: docs/collab/CELESTIA_JOINT_EXPERIMENT.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("export_cwola_log")

DEFAULT_DB = Path("genome.db")
DEFAULT_OUT = Path("cwola_export")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", type=Path, default=DEFAULT_DB,
                   help="Path to genome.db (default: ./genome.db)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help="Output directory (default: ./cwola_export)")
    p.add_argument("--since", type=str, default=None,
                   help="Filter: ts >= this date (YYYY-MM-DD)")
    p.add_argument("--until", type=str, default=None,
                   help="Filter: ts < this date (YYYY-MM-DD)")
    p.add_argument("--sample", type=int, default=None,
                   help="Take only the first N rows after filtering (for governance scans)")
    p.add_argument("--redact", action="store_true",
                   help="Replace query text with <redacted len=N> for privacy review before full upload")
    p.add_argument("--min-bucket-count", type=int, default=100,
                   help="Refuse to export if fewer than this many A+B rows exist (default: 100)")
    p.add_argument("--include-pending", action="store_true",
                   help="Include rows with bucket=NULL (not yet assigned). Default: exclude")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def date_to_epoch(date_str: str) -> float:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def build_where(args: argparse.Namespace) -> tuple[str, list]:
    clauses = []
    params: list = []
    if not args.include_pending:
        clauses.append("bucket IS NOT NULL")
    if args.since:
        clauses.append("ts >= ?")
        params.append(date_to_epoch(args.since))
    if args.until:
        clauses.append("ts < ?")
        params.append(date_to_epoch(args.until))
    where = " AND ".join(clauses) if clauses else "1=1"
    return where, params


def redact_query(query: str | None) -> str | None:
    if query is None:
        return None
    return f"<redacted len={len(query)}>"


def parse_tier_features(raw: str | None) -> dict:
    """Parse the tier_features JSON column; return empty dict on failure."""
    if not raw:
        return {}
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        log.warning("failed to parse tier_features: %r", raw[:80] if raw else None)
        return {}


def row_to_dict(row: sqlite3.Row, redact: bool) -> dict:
    d = {
        "retrieval_id": row["retrieval_id"],
        "ts": row["ts"],
        "session_id": row["session_id"],
        "party_id": row["party_id"],
        "query": redact_query(row["query"]) if redact else row["query"],
        "tier_features": parse_tier_features(row["tier_features"]),
        "top_gene_id": row["top_gene_id"],
        "bucket": row["bucket"],
        "bucket_assigned_at": row["bucket_assigned_at"],
        "requery_delta_s": row["requery_delta_s"],
    }
    return d


def compute_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"row_count": 0}

    bucket_counts = Counter(r["bucket"] for r in rows)
    party_counts = Counter(r["party_id"] or "<null>" for r in rows)
    tier_feature_keys: Counter = Counter()
    for r in rows:
        tier_feature_keys.update(r["tier_features"].keys())

    timestamps = [r["ts"] for r in rows if r["ts"] is not None]
    ts_range = {
        "min": min(timestamps) if timestamps else None,
        "max": max(timestamps) if timestamps else None,
        "min_iso": datetime.fromtimestamp(min(timestamps), tz=timezone.utc).isoformat() if timestamps else None,
        "max_iso": datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat() if timestamps else None,
    }

    requery_deltas = [r["requery_delta_s"] for r in rows
                      if r["requery_delta_s"] is not None]
    requery_stats = None
    if requery_deltas:
        requery_deltas.sort()
        n = len(requery_deltas)
        requery_stats = {
            "n_with_delta": n,
            "median_s": requery_deltas[n // 2],
            "p10_s": requery_deltas[max(0, n // 10)],
            "p90_s": requery_deltas[min(n - 1, (n * 9) // 10)],
        }

    return {
        "row_count": len(rows),
        "bucket_distribution": dict(bucket_counts),
        "party_distribution": dict(party_counts),
        "tier_feature_keys": dict(tier_feature_keys),
        "timestamp_range": ts_range,
        "requery_delta_stats": requery_stats,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.db.exists():
        log.error("genome.db not found: %s", args.db.resolve())
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    where, params = build_where(args)
    query = f"""
        SELECT retrieval_id, ts, session_id, party_id, query,
               tier_features, top_gene_id, bucket, bucket_assigned_at,
               requery_delta_s
        FROM cwola_log
        WHERE {where}
        ORDER BY ts DESC
    """
    if args.sample:
        query += f" LIMIT {int(args.sample)}"

    # Read-only connection via URI.
    uri = f"file:{args.db.resolve()}?mode=ro"
    log.info("opening %s", uri)
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.execute(query, params)
        rows = [row_to_dict(r, redact=args.redact) for r in cur]
    except sqlite3.Error:
        log.exception("query failed")
        return 2
    finally:
        conn.close()

    log.info("fetched %d rows", len(rows))

    stats = compute_stats(rows)

    n_assigned = sum(c for b, c in stats.get("bucket_distribution", {}).items()
                     if b in ("A", "B"))
    if n_assigned < args.min_bucket_count and not args.sample:
        log.error("only %d rows with bucket IN ('A', 'B'); need >= %d",
                  n_assigned, args.min_bucket_count)
        log.error("pass --min-bucket-count %d to override, or --sample N for a governance scan",
                  n_assigned)
        return 3

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = f"_sample{args.sample}" if args.sample else ""
    redact_suffix = "_redacted" if args.redact else ""
    data_path = args.out_dir / f"cwola_export_{today}{suffix}{redact_suffix}.json"
    meta_path = args.out_dir / "cwola_meta.json"

    with data_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    log.info("wrote %s (%d rows, %d KB)",
             data_path, len(rows), data_path.stat().st_size // 1024)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(args.db.resolve()),
        "filters": {
            "since": args.since,
            "until": args.until,
            "sample": args.sample,
            "include_pending": args.include_pending,
            "redacted": args.redact,
        },
        "schema": {
            "retrieval_id": "INTEGER PRIMARY KEY",
            "ts": "REAL — epoch seconds UTC of the retrieval",
            "session_id": "TEXT — groups retrievals within one user session",
            "party_id": "TEXT — identity / project / tenant; see helix federation docs",
            "query": "TEXT — raw query text (redacted if --redact was set)",
            "tier_features": "JSON object {tier_name: raw_score} for the 9 retrieval dimensions",
            "top_gene_id": "TEXT — the winning gene for this retrieval",
            "bucket": "'A' (accepted — no re-query within 60s) | 'B' (re-queried within 60s) | NULL (pending)",
            "bucket_assigned_at": "REAL — epoch when bucket was assigned",
            "requery_delta_s": "REAL — seconds to next same-session query (NULL if none within 60s)",
        },
        "bucket_semantics": {
            "A": "No same-session re-query within 60s of this retrieval. Implicit accept signal.",
            "B": "Same-session re-query within 60s. Implicit reject signal (user wasn't satisfied).",
            "NULL": "Not yet assigned — either pending (<60s since retrieval) or session hasn't ended.",
        },
        "stats": stats,
        "companion_doc": "docs/collab/CELESTIA_JOINT_EXPERIMENT.md",
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("wrote %s", meta_path)

    print("\n=== Export summary ===")
    print(f"  rows:            {stats['row_count']}")
    print(f"  bucket A / B:    {stats.get('bucket_distribution', {}).get('A', 0)} / "
          f"{stats.get('bucket_distribution', {}).get('B', 0)}")
    print(f"  parties:         {len(stats.get('party_distribution', {}))}")
    if stats.get("timestamp_range", {}).get("min_iso"):
        print(f"  date range:      {stats['timestamp_range']['min_iso']} -> "
              f"{stats['timestamp_range']['max_iso']}")
    print(f"  data file:       {data_path}")
    print(f"  meta file:       {meta_path}")
    if args.redact:
        print("  [redacted mode] — safe to inspect before unredacted export")
    elif not args.sample:
        print("\n  Next step: scan a sample of queries for sensitive content before uploading.")
        print("             python scripts/export_cwola_log.py --sample 100 --redact")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

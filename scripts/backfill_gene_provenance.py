"""Backfill source_kind, volatility_class, and last_verified_at on genes.

Phase 1 of GT's agent-context-index build spec (see
``docs/specs/2026-04-17-agent-context-index-build-spec.md``) requires
ingest to populate provenance metadata so the packet builder can
answer freshness questions. Until ingest is updated, existing genes
have all 9 provenance fields NULL, which forces
``/context/packet`` to degrade every result to ``stale_risk`` or
``needs_refresh`` regardless of actual freshness.

This script walks every gene with a non-null ``source_id``, infers
``source_kind`` from the file extension, derives
``volatility_class`` from the kind, and sets ``last_verified_at`` to
the most informative timestamp available (observed_at > mtime >
epigenetics.created_at > now).

It does NOT touch the source_index table in main.db — the packet
builder falls back to gene-local metadata when main.db is absent.
That's the right default for single-shard deployments.

Usage::

    python scripts/backfill_gene_provenance.py --dry-run
    python scripts/backfill_gene_provenance.py
    python scripts/backfill_gene_provenance.py --genome path/to/other.db
    python scripts/backfill_gene_provenance.py --force  # overwrite non-NULL

Safety:
- Default is backfill-only: rows with existing (non-NULL) provenance
  are skipped unless --force is set.
- --dry-run reports what would change without mutating.
- No backup taken; ``cp genome.db genome.db.pre-backfill.bak`` first
  if you care. The writes are idempotent so re-running is safe.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import PurePosixPath

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_GENOME = "F:/Projects/helix-context/genomes/main/genome.db"

# Extension → source_kind. Mapping is conservative — unknown extensions
# fall through to "doc" (safest default for retrieval behavior since
# doc has a stable half-life).
_EXT_TO_KIND: dict[str, str] = {
    # code
    ".py": "code", ".pyi": "code", ".rs": "code", ".ts": "code",
    ".tsx": "code", ".js": "code", ".jsx": "code", ".mjs": "code",
    ".go": "code", ".java": "code", ".rb": "code", ".cpp": "code",
    ".cc": "code", ".c": "code", ".h": "code", ".hpp": "code",
    ".swift": "code", ".kt": "code", ".scala": "code", ".sh": "code",
    ".bash": "code", ".zsh": "code", ".ps1": "code", ".bat": "code",
    ".sql": "code", ".lua": "code",
    # config
    ".toml": "config", ".yaml": "config", ".yml": "config",
    ".ini": "config", ".cfg": "config", ".env": "config",
    ".conf": "config", ".properties": "config", ".config": "config",
    # data-ish config that's often hand-edited
    ".json": "config",
    # doc
    ".md": "doc", ".mdx": "doc", ".rst": "doc", ".adoc": "doc",
    ".txt": "doc", ".tex": "doc",
    # notebook (document-ish)
    ".ipynb": "doc",
    # log
    ".log": "log", ".out": "log",
    # db
    ".db": "db", ".sqlite": "db", ".sqlite3": "db",
    # tabular data
    ".csv": "db", ".tsv": "db", ".parquet": "db", ".arrow": "db",
}

# source_kind → volatility_class. Matches GT's spec half-lives:
# stable=7d, medium=12h, hot=15min.
_KIND_TO_VOLATILITY: dict[str, str] = {
    "code": "stable",
    "config": "hot",       # configs are the ops-sensitive ones
    "doc": "stable",
    "log": "hot",
    "db": "medium",
    "benchmark": "medium",
    "tool_output": "hot",
    "session_note": "medium",
    "user_assertion": "medium",
}


def infer_source_kind(source_id: str | None) -> str | None:
    """Return the inferred source_kind or None if source_id is empty."""
    if not source_id:
        return None
    # Strip query strings, fragments, line-range suffixes
    path = str(source_id).split("?", 1)[0].split("#", 1)[0]
    # Windows-safe extension split
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    return _EXT_TO_KIND.get(suffix, "doc")


def infer_volatility(source_kind: str | None) -> str:
    if not source_kind:
        return "medium"
    return _KIND_TO_VOLATILITY.get(source_kind, "medium")


def pick_last_verified_at(
    observed_at: float | None,
    mtime: float | None,
    epigenetics_created_at: float | None,
    now_ts: float,
) -> float:
    """Most informative available timestamp, clamped to [epoch, now_ts]."""
    for candidate in (observed_at, mtime, epigenetics_created_at):
        if candidate is not None:
            try:
                v = float(candidate)
                if 0 < v <= now_ts + 1:
                    return v
            except (TypeError, ValueError):
                continue
    return now_ts


def _epigenetics_created_at(blob: str | None) -> float | None:
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    v = data.get("created_at")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def backfill(
    conn: sqlite3.Connection,
    *,
    dry_run: bool,
    force: bool,
    now_ts: float,
) -> dict:
    """Return counts dict {examined, updated, skipped_existing, skipped_no_source}."""
    # Pull everything in one pass; the genome is small enough.
    cur = conn.execute(
        "SELECT gene_id, source_id, source_kind, volatility_class, "
        "last_verified_at, observed_at, mtime, epigenetics "
        "FROM genes"
    )
    examined = 0
    updated = 0
    skipped_existing = 0
    skipped_no_source = 0
    kind_histogram: dict[str, int] = {}

    rows_to_write: list[tuple[str, str, str, float]] = []

    for row in cur:
        examined += 1
        (gene_id, source_id, cur_kind, cur_vol,
         cur_last_verified, observed_at, mtime, epig_blob) = row

        if not source_id:
            skipped_no_source += 1
            continue

        has_any = any(v is not None for v in (cur_kind, cur_vol, cur_last_verified))
        if has_any and not force:
            skipped_existing += 1
            continue

        inferred_kind = infer_source_kind(source_id) or "doc"
        inferred_vol = infer_volatility(inferred_kind)
        epig_ts = _epigenetics_created_at(epig_blob)
        last_verified = pick_last_verified_at(observed_at, mtime, epig_ts, now_ts)

        rows_to_write.append((gene_id, inferred_kind, inferred_vol, last_verified))
        kind_histogram[inferred_kind] = kind_histogram.get(inferred_kind, 0) + 1
        updated += 1

    print(f"[backfill] examined:           {examined:>6}")
    print(f"[backfill] would-update:       {updated:>6}")
    print(f"[backfill] skipped (existing): {skipped_existing:>6}")
    print(f"[backfill] skipped (no src):   {skipped_no_source:>6}")
    print(f"[backfill] by source_kind:")
    for k, v in sorted(kind_histogram.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<12} {v:>6}")

    if dry_run:
        print("\n[backfill] dry-run: no writes performed.")
        return {
            "examined": examined, "updated": 0,
            "skipped_existing": skipped_existing,
            "skipped_no_source": skipped_no_source,
        }

    # Chunked write so the UPDATE lock doesn't block readers for long.
    for i in range(0, len(rows_to_write), 500):
        batch = rows_to_write[i : i + 500]
        conn.executemany(
            "UPDATE genes SET source_kind = ?, volatility_class = ?, "
            "last_verified_at = ? WHERE gene_id = ?",
            [(k, v, t, gid) for (gid, k, v, t) in batch],
        )
        conn.commit()
        print(f"[backfill] wrote batch {i + len(batch):>6} / {len(rows_to_write)}")

    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass

    return {
        "examined": examined, "updated": updated,
        "skipped_existing": skipped_existing,
        "skipped_no_source": skipped_no_source,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default=DEFAULT_GENOME)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report changes without writing.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing non-NULL provenance fields.")
    args = ap.parse_args()

    print(f"[backfill] genome: {args.genome}")
    if not os.path.exists(args.genome):
        print("[backfill] ERROR: genome not found")
        return 2

    conn = sqlite3.connect(args.genome)
    try:
        backfill(conn, dry_run=args.dry_run, force=args.force, now_ts=time.time())
    finally:
        conn.close()

    print("[backfill] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Backfill: re-run CpuTagger._extract_key_values over existing genes to
purge Python-type-annotation leaks introduced under the pre-f4c91e3
regex.

Context:
  Commit f4c91e3 fixed a bug in CpuTagger where annotated Python
  assignments (`port: int = 8080`) produced type-name KV entries
  (`port=int`, `int=8080`) in gene.key_values. The fix prevents future
  occurrences, but the 8000+ live genes were ingested under the buggy
  code and still carry the leaked entries in their JSON key_values.

  This script walks every gene, re-runs _extract_key_values over
  gene.content with the fixed tagger, and updates the row if the
  result differs. It's non-destructive (makes a timestamped backup
  of genome.db before any writes) and idempotent (safe to re-run).

Usage:
    # Dry-run: show what would change, no writes
    python scripts/backfill_kv_tagger_fix.py --dry-run

    # Apply to live genome.db with backup
    python scripts/backfill_kv_tagger_fix.py --apply

    # Target a different DB (e.g. a benchmark snapshot)
    python scripts/backfill_kv_tagger_fix.py --apply \\
        --db F:/Projects/helix-context/genome.db

Output:
    - Counts: total genes, unchanged, changed, type-leaks removed
    - Sample of 10 before/after pairs for manual inspection
    - Path to backup file (if --apply)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helix_context.tagger import CpuTagger, _KV_TYPE_ANNOTATION_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill.kv")


def count_type_leaks(kvs: list[str]) -> int:
    """Count entries whose key OR value is a bare Python type name."""
    n = 0
    for kv in kvs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k.strip().lower() in _KV_TYPE_ANNOTATION_NAMES:
            n += 1
        elif v.strip().lower() in _KV_TYPE_ANNOTATION_NAMES:
            n += 1
    return n


def backfill(db_path: str, apply: bool, sample_limit: int = 10) -> dict:
    """Walk all genes, recompute key_values, report changes (and optionally apply)."""
    tagger = CpuTagger()
    log.info("Tagger loaded")

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT gene_id, content, key_values FROM genes"
    ).fetchall()
    log.info("Loaded %d genes from %s", len(rows), db_path)

    stats = {
        "total": len(rows),
        "parse_errors": 0,
        "empty_content": 0,
        "unchanged": 0,
        "changed": 0,
        "type_leaks_removed": 0,
        "type_leaks_remaining": 0,
    }
    samples: list[dict] = []
    updates: list[tuple[str, str]] = []  # (json_kvs, gene_id)

    for gene_id, content, kv_raw in rows:
        if not content:
            stats["empty_content"] += 1
            continue
        try:
            old_kvs = json.loads(kv_raw) if kv_raw else []
        except Exception:
            stats["parse_errors"] += 1
            continue

        new_kvs = tagger._extract_key_values(content)

        old_leaks = count_type_leaks(old_kvs)
        new_leaks = count_type_leaks(new_kvs)
        stats["type_leaks_remaining"] += new_leaks

        if old_kvs == new_kvs:
            stats["unchanged"] += 1
            continue

        stats["changed"] += 1
        stats["type_leaks_removed"] += max(0, old_leaks - new_leaks)

        if len(samples) < sample_limit:
            samples.append({
                "gene_id": gene_id[:12],
                "old": old_kvs[:6],
                "new": new_kvs[:6],
                "old_leaks": old_leaks,
                "new_leaks": new_leaks,
            })

        updates.append((json.dumps(new_kvs), gene_id))

    log.info("Walked %d genes", stats["total"])
    log.info("  unchanged:           %d", stats["unchanged"])
    log.info("  changed:             %d", stats["changed"])
    log.info("  type_leaks_removed:  %d", stats["type_leaks_removed"])
    log.info("  type_leaks_remaining:%d (genuine leftover — content has no cleaner interpretation)", stats["type_leaks_remaining"])
    log.info("  parse_errors:        %d", stats["parse_errors"])
    log.info("  empty_content:       %d", stats["empty_content"])

    log.info("Sample of up to %d changed genes:", sample_limit)
    for s in samples:
        log.info("  %s: old=%s leaks=%d  →  new=%s leaks=%d",
                 s["gene_id"], s["old"], s["old_leaks"], s["new"], s["new_leaks"])

    if not apply:
        log.info("Dry-run — no writes. Re-run with --apply to commit.")
        conn.close()
        return stats

    if not updates:
        log.info("No changes to apply.")
        conn.close()
        return stats

    # Backup before writing
    backup_path = f"{db_path}.pre-kv-backfill.{int(time.time())}.bak"
    log.info("Backing up %s → %s", db_path, backup_path)
    # Close connection so we can safely copy the DB file
    conn.close()
    shutil.copy2(db_path, backup_path)
    log.info("Backup complete")

    # Apply updates in a single transaction
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN")
        conn.executemany(
            "UPDATE genes SET key_values = ? WHERE gene_id = ?",
            updates,
        )
        conn.commit()
        log.info("Applied %d updates to %s", len(updates), db_path)
    except Exception:
        conn.rollback()
        log.exception("Update failed — rolled back")
        raise
    finally:
        conn.close()

    stats["backup_path"] = backup_path
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="F:/Projects/helix-context/genome.db",
                        help="Path to genome.db (default: live helix genome)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually apply changes (default: dry-run)")
    parser.add_argument("--sample-limit", type=int, default=10,
                        help="Number of sample before/after pairs to log")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        log.error("DB not found: %s", args.db)
        return 2

    stats = backfill(args.db, apply=args.apply, sample_limit=args.sample_limit)
    log.info("Done: %s", json.dumps({k: v for k, v in stats.items() if k != "backup_path"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

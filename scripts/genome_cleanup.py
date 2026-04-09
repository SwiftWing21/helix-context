r"""
Genome Cleanup — targeted metadata fixes without full re-sequencing.

Fixes promoter tag quality, source_id coverage, and orphan genes
using pure SQL operations. No ribosome calls needed.

Usage:
    python scripts/genome_cleanup.py                    # run cleanup
    python scripts/genome_cleanup.py --dry-run          # preview only
    python scripts/genome_cleanup.py --genome path.db   # custom path
"""

import argparse
import hashlib
import os
import sqlite3
import sys


TAG_NORMALIZATIONS = {
    "next.js": "nextjs",
    "js": "javascript",
    "ts": "typescript",
    "software_development": "software_engineering",
    "rest": "api",
    "db": "database",
    "sql": "sqlite",
    "soc 2": "soc2",
    "machine learning": "machine_learning",
    "code review": "code_review",
    "deep sea": "deep_sea",
}

FILE_EXTENSIONS = [".py", ".ts", ".js", ".md", ".toml", ".rs", ".json", ".yaml", ".yml"]

PROGRESS_FILE = os.path.join(os.path.dirname(__file__), ".ingest_progress")


def run_cleanup(genome_path, dry_run=False):
    conn = sqlite3.connect(genome_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
    print(f"Genome: {total} genes")
    print()

    # 1. Normalize fragmented tags
    fix1 = 0
    for old, new in TAG_NORMALIZATIONS.items():
        cnt = cur.execute(
            "SELECT COUNT(*) FROM promoter_index WHERE tag_value = ?", (old,)
        ).fetchone()[0]
        if cnt > 0:
            if not dry_run:
                cur.execute("UPDATE promoter_index SET tag_value = ? WHERE tag_value = ?", (new, old))
            fix1 += cnt
            print(f"  Normalize: {old}({cnt}) -> {new}")
    print(f"  Total normalized: {fix1}")
    print()

    # 2. Strip file extensions from tags
    fix2 = 0
    for ext in FILE_EXTENSIONS:
        rows = cur.execute(
            "SELECT rowid, tag_value FROM promoter_index WHERE tag_value LIKE ?",
            (f"%{ext}",),
        ).fetchall()
        for row in rows:
            clean = row["tag_value"].replace(ext, "").replace(".", "_").strip("_")
            if clean and len(clean) > 2:
                if not dry_run:
                    cur.execute("UPDATE promoter_index SET tag_value = ? WHERE rowid = ?", (clean, row["rowid"]))
            else:
                if not dry_run:
                    cur.execute("DELETE FROM promoter_index WHERE rowid = ?", (row["rowid"],))
            fix2 += 1
    print(f"  File extensions stripped: {fix2}")
    print()

    # 3. Delete orphan conversation genes (no tags, content starts with "User query:")
    orphan_count = cur.execute("""
        SELECT COUNT(*) FROM genes WHERE gene_id IN (
            SELECT g.gene_id FROM genes g
            LEFT JOIN promoter_index pi ON g.gene_id = pi.gene_id
            WHERE pi.gene_id IS NULL AND g.content LIKE 'User query:%'
        )
    """).fetchone()[0]
    if not dry_run and orphan_count > 0:
        cur.execute("""
            DELETE FROM genes WHERE gene_id IN (
                SELECT g.gene_id FROM genes g
                LEFT JOIN promoter_index pi ON g.gene_id = pi.gene_id
                WHERE pi.gene_id IS NULL AND g.content LIKE 'User query:%'
            )
        """)
    print(f"  Orphan conversation genes removed: {orphan_count}")
    print()

    # 4. Backfill source_id via content fingerprinting
    fix4 = 0
    if os.path.exists(PROGRESS_FILE):
        paths = open(PROGRESS_FILE, encoding="utf-8").read().splitlines()
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                content = open(path, encoding="utf-8", errors="replace").read()
                fingerprint = content[:80]
                if len(fingerprint) < 50:
                    continue
                if not dry_run:
                    r = cur.execute(
                        "UPDATE genes SET source_id = ? WHERE source_id IS NULL AND content LIKE ?",
                        (path, fingerprint + "%"),
                    )
                    fix4 += r.rowcount
                else:
                    cnt = cur.execute(
                        "SELECT COUNT(*) FROM genes WHERE source_id IS NULL AND content LIKE ?",
                        (fingerprint + "%",),
                    ).fetchone()[0]
                    fix4 += cnt
            except Exception:
                continue
    print(f"  Source_id backfilled: {fix4}")
    print()

    # 5. Deduplicate tags
    if not dry_run:
        fix5 = cur.execute("""
            DELETE FROM promoter_index WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM promoter_index GROUP BY gene_id, tag_type, tag_value
            )
        """).rowcount
    else:
        fix5 = cur.execute("""
            SELECT COUNT(*) FROM promoter_index WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM promoter_index GROUP BY gene_id, tag_type, tag_value
            )
        """).fetchone()[0]
    print(f"  Duplicate tags removed: {fix5}")

    if not dry_run:
        conn.commit()
        print("\nCommitted.")
    else:
        print("\nDry run — no changes made.")

    # Final stats
    total = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
    with_source = cur.execute("SELECT COUNT(*) FROM genes WHERE source_id IS NOT NULL").fetchone()[0]
    unique_tags = cur.execute("SELECT COUNT(DISTINCT tag_value) FROM promoter_index").fetchone()[0]
    print(f"\nFinal: {total} genes, {with_source} with source_id ({with_source/total*100:.0f}%), {unique_tags} unique tags")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genome metadata cleanup")
    parser.add_argument("--genome", default="genome.db", help="Path to genome.db")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    run_cleanup(args.genome, args.dry_run)

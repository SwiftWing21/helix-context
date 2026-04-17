"""Delete orphan parent genes whose source file no longer exists on disk.

Parent gene IDs are derived from ``sha256(source_path + "::parent")[:16]``
(see ``HelixContextManager._make_parent_gene_id`` — shipped 2026-04-16
with the layered-fingerprints work). That means when a source file is
moved or deleted, the old parent gene is NOT overwritten on re-ingest —
it becomes an orphan with dangling CHUNK_OF edges pointing at it.

This script finds those orphans (parents whose ``source_id`` no longer
resolves on the filesystem) and deletes them along with their
CHUNK_OF edges. Run it after any bulk file move (e.g. a ``docs/``
reorg) or periodic sweep.

Default: parents only. ``--include-children`` also sweeps child genes
whose source files are missing — useful when a file is deleted outright
rather than moved. Conservative default because a moved file's child
genes are still content-correct; only their ``source_id`` is stale.

Safety:
- Takes a write lock. Run while helix is stopped OR accept that the
  live server may momentarily see a shrinking genome.
- ``--dry-run`` prints what would be deleted without touching the DB.
- Takes no backup of its own. Make one first if you care:
  ``cp genome.db genome.db.pre-tombstone.bak``

Usage::

    python scripts/tombstone_orphan_parents.py --dry-run
    python scripts/tombstone_orphan_parents.py
    python scripts/tombstone_orphan_parents.py --include-children
    python scripts/tombstone_orphan_parents.py --genome path/to/other.db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_GENOME = "F:/Projects/helix-context/genomes/main/genome.db"
CHUNK_OF_RELATION = 100  # StructuralRelation.CHUNK_OF


def _is_parent(key_values_raw: str | None) -> bool:
    if not key_values_raw:
        return False
    try:
        kv = json.loads(key_values_raw)
    except Exception:
        return False
    return any(
        isinstance(k, str) and k.lower() == "is_parent=true"
        for k in (kv or [])
    )


def _source_exists(source_id: str | None) -> bool:
    """True if source_id points to an existing file on disk.

    Empty/None/non-path source_ids are treated as "exists" (nothing to
    check against). Only paths that clearly look like filesystem paths
    and fail to resolve are considered orphans.
    """
    if not source_id:
        return True  # nothing to verify
    # Heuristic: treat as a path if it contains a slash or backslash.
    # Free-form source_ids like "agent:laude" or "manual_note" shouldn't
    # be tombstoned just because they don't resolve.
    if "/" not in source_id and "\\" not in source_id:
        return True
    return os.path.exists(source_id)


def find_orphans(
    conn: sqlite3.Connection,
    include_children: bool,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (orphan_parents, orphan_children) as [(gene_id, source_id), ...]."""
    orphan_parents: list[tuple[str, str]] = []
    orphan_children: list[tuple[str, str]] = []

    cur = conn.execute(
        "SELECT gene_id, source_id, key_values FROM genes "
        "WHERE source_id IS NOT NULL AND source_id != ''"
    )
    for row in cur:
        gid, sid, kv = row[0], row[1], row[2]
        if _source_exists(sid):
            continue
        if _is_parent(kv):
            orphan_parents.append((gid, sid))
        elif include_children:
            orphan_children.append((gid, sid))

    return orphan_parents, orphan_children


def count_chunk_of_edges(
    conn: sqlite3.Connection,
    parent_ids: list[str],
) -> int:
    if not parent_ids:
        return 0
    placeholders = ",".join("?" * len(parent_ids))
    row = conn.execute(
        f"SELECT COUNT(*) FROM gene_relations "
        f"WHERE gene_id_b IN ({placeholders}) AND relation = ?",
        parent_ids + [CHUNK_OF_RELATION],
    ).fetchone()
    return int(row[0])


def delete_genes_and_edges(
    conn: sqlite3.Connection,
    gene_ids: list[str],
    include_chunk_of_edges: bool,
) -> tuple[int, int]:
    """Delete the given gene rows + (optionally) their CHUNK_OF edges.

    Returns (n_genes_deleted, n_edges_deleted).
    """
    if not gene_ids:
        return 0, 0
    placeholders = ",".join("?" * len(gene_ids))

    n_edges = 0
    if include_chunk_of_edges:
        edge_res = conn.execute(
            f"DELETE FROM gene_relations "
            f"WHERE gene_id_b IN ({placeholders}) AND relation = ?",
            gene_ids + [CHUNK_OF_RELATION],
        )
        n_edges = edge_res.rowcount or 0

    # Also clean any incoming edges FROM these genes (the orphans might
    # have been a child in some other relationship). Keep this broad —
    # any edge touching the dead gene becomes an orphan edge.
    incoming_res = conn.execute(
        f"DELETE FROM gene_relations "
        f"WHERE gene_id_a IN ({placeholders}) OR gene_id_b IN ({placeholders})",
        gene_ids + gene_ids,
    )
    n_edges += incoming_res.rowcount or 0

    gene_res = conn.execute(
        f"DELETE FROM genes WHERE gene_id IN ({placeholders})",
        gene_ids,
    )
    return int(gene_res.rowcount or 0), n_edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default=DEFAULT_GENOME)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report orphans without deleting.")
    ap.add_argument("--include-children", action="store_true",
                    help="Also delete non-parent genes whose source file "
                         "is missing (files that were deleted outright).")
    args = ap.parse_args()

    print(f"[tombstone] genome: {args.genome}")
    if not os.path.exists(args.genome):
        print(f"[tombstone] ERROR: genome not found")
        return 2

    conn = sqlite3.connect(args.genome)
    conn.row_factory = sqlite3.Row

    orphan_parents, orphan_children = find_orphans(
        conn, include_children=args.include_children
    )

    parent_ids = [gid for gid, _ in orphan_parents]
    child_ids = [gid for gid, _ in orphan_children]

    edge_count = count_chunk_of_edges(conn, parent_ids)

    print(f"[tombstone] orphan parents:  {len(orphan_parents):>6}")
    print(f"[tombstone] CHUNK_OF edges:  {edge_count:>6} (pointing at orphan parents)")
    if args.include_children:
        print(f"[tombstone] orphan children: {len(orphan_children):>6}")

    # Show a few examples so the operator can sanity-check
    if orphan_parents:
        print("\n  sample orphan parents (first 5):")
        for gid, sid in orphan_parents[:5]:
            print(f"    {gid[:12]}  src={sid}")
    if args.include_children and orphan_children:
        print("\n  sample orphan children (first 5):")
        for gid, sid in orphan_children[:5]:
            print(f"    {gid[:12]}  src={sid}")

    if args.dry_run:
        print("\n[tombstone] dry-run: no deletions performed.")
        conn.close()
        return 0

    if not parent_ids and not child_ids:
        print("\n[tombstone] nothing to do.")
        conn.close()
        return 0

    # Parents first: cleans the specific problem that motivated this.
    n_pg, n_pe = delete_genes_and_edges(
        conn, parent_ids, include_chunk_of_edges=True,
    )
    print(f"\n[tombstone] deleted {n_pg} parent genes + {n_pe} related edges")

    if child_ids:
        n_cg, n_ce = delete_genes_and_edges(
            conn, child_ids, include_chunk_of_edges=False,
        )
        print(f"[tombstone] deleted {n_cg} child genes + {n_ce} related edges")

    conn.commit()

    # Checkpoint so the WAL doesn't sit on the deletions.
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass

    conn.close()
    print("[tombstone] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

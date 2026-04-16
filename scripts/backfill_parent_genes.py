"""Backfill parent genes + CHUNK_OF edges for existing multi-chunk files.

Idempotent. Safe to re-run. Reads the current genome, groups existing
genes by ``source_id``, and for any source with ≥ 2 chunks creates the
deterministic parent gene (UPSERT) and CHUNK_OF edges.

WARNING: run with helix stopped and AFTER WAL checkpoint / cleanup.
Grabs a write lock on genome.db. Also run with a backup in hand
(E:\\Helix-backup is the operator's current off-disk blob).

Usage:
    python scripts/backfill_parent_genes.py \
        --genome C:/helix-cache/genome.db \
        [--dry-run] [--limit N]

--dry-run  Report how many parents would be created, no writes.
--limit N  Only process the first N multi-chunk sources.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make helix_context importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_context.context_manager import HelixContextManager  # noqa: E402
from helix_context.schemas import StructuralRelation  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows CP1252 guard
except Exception:
    pass


def group_by_source(conn: sqlite3.Connection) -> dict[str, list[tuple[str, int]]]:
    """Return {source_id: [(gene_id, sequence_index), ...]} for all non-parent
    genes with a source_id.

    Sequence index is read from promoter.sequence_index; missing → 0.
    Excludes rows already marked as parents (is_parent=true in key_values).
    """
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    cur = conn.execute(
        "SELECT gene_id, source_id, promoter, key_values FROM genes "
        "WHERE source_id IS NOT NULL AND source_id != ''"
    )
    skipped_parents = 0
    for row in cur:
        kv_raw = row["key_values"] or "[]"
        try:
            kv = json.loads(kv_raw) if isinstance(kv_raw, str) else kv_raw
        except Exception:
            kv = []
        if any(isinstance(k, str) and k.lower() == "is_parent=true" for k in kv):
            skipped_parents += 1
            continue

        promoter_raw = row["promoter"] or "{}"
        try:
            promoter = json.loads(promoter_raw) if isinstance(promoter_raw, str) else promoter_raw
        except Exception:
            promoter = {}
        seq = promoter.get("sequence_index") if isinstance(promoter, dict) else None
        if not isinstance(seq, int):
            seq = 0
        groups[row["source_id"]].append((row["gene_id"], seq))

    if skipped_parents:
        print(f"  (skipped {skipped_parents} existing parent genes)")
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default="C:/helix-cache/genome.db")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(args.genome)
    conn.row_factory = sqlite3.Row

    print(f"[backfill] scanning {args.genome}")
    groups = group_by_source(conn)

    multi = {sid: chunks for sid, chunks in groups.items() if len(chunks) >= 2}
    print(f"[backfill] found {len(groups):,} unique source_ids")
    print(f"[backfill] {len(multi):,} are multi-chunk (eligible for parent)")

    if args.limit:
        multi = dict(list(multi.items())[: args.limit])
        print(f"[backfill] --limit applied: {len(multi)} sources will be processed")

    if args.dry_run:
        print("[backfill] dry-run: no writes")
        sample = list(multi.items())[:5]
        for sid, chunks in sample:
            print(f"  would-parent: {sid[-70:]}  ({len(chunks)} chunks)")
        return 0

    now = time.time()
    n_parents = 0
    n_edges = 0
    for source_id, chunks in multi.items():
        chunks.sort(key=lambda t: t[1])  # order by sequence_index
        child_ids = [gid for gid, _ in chunks]

        parent_gid = HelixContextManager._make_parent_gene_id(source_id)

        # Minimal parent gene row — mirrors _upsert_parent_gene shape.
        # Skip content for backfill (would require re-reading files from disk);
        # use a short placeholder. Reassembly via codons still works.
        total_bytes = sum(1 for _ in chunks) * 4000  # rough estimate
        parent_row = (
            parent_gid,
            f"[backfilled parent] {source_id}",          # content
            f"Parent aggregating {len(chunks)} chunks",  # complement
            json.dumps(list(child_ids)),                 # codons
            json.dumps({"sequence_index": -1}),          # promoter
            json.dumps({}),                              # epigenetics
            0,                                           # chromatin (OPEN)
            0,                                           # is_fragment
            None,                                        # embedding
            source_id,                                   # source_id
            1,                                           # version
            None,                                        # supersedes
            json.dumps([
                f"chunk_count={len(chunks)}",
                f"total_size_bytes={total_bytes}",
                "is_parent=true",
                "backfilled=true",
            ]),                                          # key_values
        )
        conn.execute(
            "INSERT OR REPLACE INTO genes (gene_id, content, complement, codons, "
            "promoter, epigenetics, chromatin, is_fragment, embedding, source_id, "
            "version, supersedes, key_values) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            parent_row,
        )
        n_parents += 1

        # Batch-insert CHUNK_OF edges.
        edges = [
            (cid, parent_gid, int(StructuralRelation.CHUNK_OF), 1.0, now)
            for cid in child_ids
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO gene_relations "
            "(gene_id_a, gene_id_b, relation, confidence, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            edges,
        )
        n_edges += len(edges)

        if n_parents % 100 == 0:
            conn.commit()
            print(f"  ...created {n_parents:,} parents, {n_edges:,} edges")

    conn.commit()
    conn.close()

    print(f"[backfill] DONE: {n_parents:,} parent genes, {n_edges:,} CHUNK_OF edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())

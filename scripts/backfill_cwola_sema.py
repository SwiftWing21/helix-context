"""Backfill query_sema and top_candidate_sema for existing cwola_log rows.

Part of PWPC Phase 1 enrichment (see docs/collab/comms/REPLY_PWPC_FROM_LAUDE.md).

For each row where query_sema is NULL, encode the stored query via the
SEMA codec. For each row where top_candidate_sema is NULL and top_gene_id
references a gene that still exists with a stored embedding, copy that
embedding. Rows whose top_gene has been deleted or never had an embedding
stay NULL, which is correct — the trainer treats NULL as missing, not as zero.

Usage:
    python scripts/backfill_cwola_sema.py --db genome.db [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _import_helix():
    """Import helix_context modules — ensures SQLite schema has new columns."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helix_context import genome as _genome  # noqa: F401
    from helix_context.sema import SemaCodec
    from helix_context.genome import Genome
    return Genome, SemaCodec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("genome.db"),
                    help="Path to genome.db (default: ./genome.db)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count what would be updated, don't write")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N rows (useful for spot-checking)")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"ERROR: {args.db} not found", file=sys.stderr)
        return 1

    Genome, SemaCodec = _import_helix()
    # Genome() init applies schema migrations — ensures the new columns exist
    # before we try to SELECT/UPDATE them.
    g = Genome(str(args.db))

    codec = SemaCodec()
    _sample = codec.encode("hello")
    print(f"SemaCodec ready. intermediate embed_dim={codec.embed_dim}, output sema_dim={len(_sample)}")

    cur = g.conn.cursor()
    rows = cur.execute(
        "SELECT retrieval_id, query, top_gene_id, query_sema, top_candidate_sema "
        "FROM cwola_log "
        "WHERE query_sema IS NULL OR top_candidate_sema IS NULL "
        "ORDER BY retrieval_id"
    ).fetchall()
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Found {len(rows)} rows needing backfill.")
    if args.dry_run:
        missing_q = sum(1 for r in rows if r[3] is None)
        missing_c = sum(1 for r in rows if r[4] is None)
        print(f"  query_sema NULL:          {missing_q}")
        print(f"  top_candidate_sema NULL:  {missing_c}")
        print("  (dry-run — no writes)")
        return 0

    encoded_q = 0
    filled_c = 0
    skipped_c_no_gene = 0
    skipped_c_no_embed = 0
    for retrieval_id, query, top_gene_id, qs, cs in rows:
        updates: dict[str, str] = {}
        if qs is None and query:
            try:
                vec = codec.encode(query)
                updates["query_sema"] = json.dumps([float(x) for x in vec])
                encoded_q += 1
            except Exception as exc:
                print(f"  rid={retrieval_id}: encode failed: {exc}")
        if cs is None and top_gene_id:
            gene = g.get_gene(top_gene_id)
            if gene is None:
                skipped_c_no_gene += 1
            elif not gene.embedding:
                skipped_c_no_embed += 1
            else:
                updates["top_candidate_sema"] = json.dumps([float(x) for x in gene.embedding])
                filled_c += 1

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            cur.execute(
                f"UPDATE cwola_log SET {set_clause} WHERE retrieval_id = ?",
                (*updates.values(), retrieval_id),
            )
    g.conn.commit()

    print("Done.")
    print(f"  query_sema encoded:           {encoded_q}")
    print(f"  top_candidate_sema filled:    {filled_c}")
    print(f"  candidates skipped (no gene): {skipped_c_no_gene}")
    print(f"  candidates skipped (no emb):  {skipped_c_no_embed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

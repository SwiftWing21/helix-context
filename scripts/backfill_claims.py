"""Backfill Phase 2 claims from an existing genome into main.db.

Usage:
    python scripts/backfill_claims.py \
        --genome genomes/main/genome.db \
        --main-db genomes/main.db \
        --shard-name primary

Hybrid-mode recovery (per build-spec §634): new genes get claims at
ingest via the Genome(main_conn=...) hook; old genes need this
backfill. Idempotent — re-running upserts the same claim_ids.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helix_context.claims import extract_literal_claims, persist_claims  # noqa: E402
from helix_context.claims_analyze import detect_and_persist_edges  # noqa: E402
from helix_context.genome import Genome  # noqa: E402
from helix_context.shard_schema import (  # noqa: E402
    init_main_db, open_main_db, register_shard, SHARD_CATEGORIES,
)


def backfill(
    genome_path: Path,
    main_db_path: Path,
    shard_name: str = "primary",
    shard_category: str = "reference",
    batch_size: int = 500,
    progress_every: int = 1000,
    detect_edges: bool = True,
) -> dict:
    print(f"Opening genome: {genome_path}")
    print(f"Opening main.db: {main_db_path}")

    main_db = open_main_db(main_db_path)
    init_main_db(main_db)

    # Register the shard if absent (FK constraint on claims.shard_name)
    existing = main_db.execute(
        "SELECT shard_name FROM shards WHERE shard_name = ?",
        (shard_name,),
    ).fetchone()
    if not existing:
        if shard_category not in SHARD_CATEGORIES:
            raise ValueError(f"unknown category {shard_category!r}")
        register_shard(main_db, shard_name, shard_category, str(genome_path))

    genome = Genome(str(genome_path))
    try:
        # Stream via raw SQL — Genome doesn't expose iter_genes universally
        rows = genome.conn.execute(
            "SELECT gene_id FROM genes ORDER BY rowid"
        ).fetchall()
        total = len(rows)
        print(f"Backfilling claims for {total} genes (batch={batch_size})")

        t0 = time.time()
        n_claims = 0
        n_processed = 0
        batch: list = []

        for (gene_id,) in rows:
            # Load the full gene via the existing row_to_gene path
            row = genome.conn.execute(
                "SELECT * FROM genes WHERE gene_id = ?", (gene_id,),
            ).fetchone()
            if row is None:
                continue
            try:
                gene = genome._row_to_gene(row)
            except Exception:
                # Row schema mismatch — skip
                continue
            claims = extract_literal_claims(gene, shard_name=shard_name)
            batch.extend(claims)
            n_processed += 1

            if len(batch) >= batch_size:
                n_claims += persist_claims(main_db, batch)
                batch = []

            if n_processed % progress_every == 0:
                elapsed = time.time() - t0
                rate = n_processed / max(elapsed, 0.01)
                print(f"  {n_processed:>6}/{total} genes, "
                      f"{n_claims:>6} claims written, "
                      f"{rate:>5.0f} genes/s")

        if batch:
            n_claims += persist_claims(main_db, batch)

        elapsed = time.time() - t0
    finally:
        genome.close()

    claim_count = main_db.execute(
        "SELECT COUNT(*) FROM claims"
    ).fetchone()[0]

    edge_summary: dict = {}
    if detect_edges:
        print("\nDetecting claim edges (contradicts / duplicates / supersedes)…")
        t_edge0 = time.time()
        edge_summary = detect_and_persist_edges(main_db)
        t_edge = time.time() - t_edge0
        print(f"  {edge_summary.get('n_groups', 0)} entity_key groups scanned "
              f"in {t_edge:.1f}s")
        print(f"  contradicts={edge_summary.get('contradicts', 0)}  "
              f"duplicates={edge_summary.get('duplicates', 0)}  "
              f"supersedes={edge_summary.get('supersedes', 0)}")

    main_db.close()

    print(f"\nBackfill complete in {elapsed:.1f}s")
    print(f"  genes processed: {n_processed}")
    print(f"  claims inserted this run: {n_claims}")
    print(f"  claims total in main.db: {claim_count}")

    return {
        "elapsed_s": elapsed,
        "genes_processed": n_processed,
        "claims_inserted": n_claims,
        "claims_total": claim_count,
        "edges": edge_summary,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--genome", default="genomes/main/genome.db")
    p.add_argument("--main-db", default="genomes/main.db")
    p.add_argument("--shard-name", default="primary")
    p.add_argument("--shard-category", default="reference",
                   choices=list(SHARD_CATEGORIES))
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--no-detect-edges", action="store_true",
                   help="Skip the claim-edge detection pass")
    args = p.parse_args()

    backfill(
        genome_path=Path(args.genome),
        main_db_path=Path(args.main_db),
        shard_name=args.shard_name,
        shard_category=args.shard_category,
        batch_size=args.batch_size,
        detect_edges=not args.no_detect_edges,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

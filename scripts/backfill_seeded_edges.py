"""
One-shot backfill of `source = 'seeded'` harmonic_links edges.

Fixes the Sprint 2/4 cold-start diagnostic: on the live genome 85.8%
of OPEN genes have empty `co_activated_with`, so Successor
Representation Tier 5.5 has nothing to traverse from most seed
genes. This script populates seeded-edges via the Sprint 4
`multi_signal_overlap` admission gate, with raw_weight modulated
by ΣĒMA cosine to preserve semantic nuance.

Design (see conversation 2026-04-13):
  - Admission: ≥2 of {shared domain, shared entity, shared KV key,
    both-OPEN chromatin} — unchanged from seeded_edges.multi_signal_overlap.
  - Raw weight: weight = 0.7 + 0.3 * max(0, cos(sema_a, sema_b)).
    Floor at 0.7 keeps effective_weight > PRUNE_FLOOR even on
    first-query misses; upper 0.3 band differentiates strong semantic
    matches.
  - Cross-source pairs (different source_id directories) require
    cos > 0.5 — respects functional locality.
  - Index-driven candidate selection (promoter domains/entities +
    path_key_index KV keys) — avoids O(N²) over 18K genes.
  - Top-K = 20 per gene — prevents hub-gene explosion in SR
    propagation. Cap is applied AFTER weight computation so only the
    20 highest-weighted neighbours survive.
  - Batched inserts of 5K rows per commit.

Usage:
  python scripts/backfill_seeded_edges.py
  python scripts/backfill_seeded_edges.py --dry-run
  python scripts/backfill_seeded_edges.py --k 30 --min-cross-source-cos 0.4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from helix_context.config import load_config
from helix_context.genome import Genome

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill")


RAW_WEIGHT_FLOOR = 0.7
RAW_WEIGHT_CEIL = 1.0


def _weight(cos: float) -> float:
    """Map cosine [-1, 1] into raw weight [0.7, 1.0]."""
    c = max(0.0, min(1.0, cos))
    return RAW_WEIGHT_FLOOR + (RAW_WEIGHT_CEIL - RAW_WEIGHT_FLOOR) * c


def load_open_genes(genome: Genome) -> list[dict]:
    """Fetch all OPEN genes with the fields needed for seeding."""
    cur = genome.read_conn.cursor()
    rows = cur.execute(
        "SELECT gene_id, source_id, promoter, key_values "
        "FROM genes WHERE chromatin = 0"
    ).fetchall()
    genes = []
    for r in rows:
        try:
            promoter = json.loads(r["promoter"]) if r["promoter"] else {}
        except Exception:
            promoter = {}
        try:
            kv_list = json.loads(r["key_values"]) if r["key_values"] else []
        except Exception:
            kv_list = []
        genes.append({
            "gene_id": r["gene_id"],
            "source_id": r["source_id"] or "",
            "domains": set((promoter.get("domains") or [])),
            "entities": set((promoter.get("entities") or [])),
            "kv_keys": {
                pair.split("=", 1)[0] for pair in kv_list if "=" in pair
            },
        })
    return genes


def build_inverted_indexes(genes: list[dict]) -> tuple[dict, dict, dict, dict]:
    """domain → {gene_id}, entity → {gene_id}, kv_key → {gene_id}, source_id → {gene_id}."""
    domain_idx: dict = defaultdict(set)
    entity_idx: dict = defaultdict(set)
    kv_idx: dict = defaultdict(set)
    source_idx: dict = defaultdict(set)
    for g in genes:
        gid = g["gene_id"]
        for d in g["domains"]:
            domain_idx[d].add(gid)
        for e in g["entities"]:
            entity_idx[e].add(gid)
        for k in g["kv_keys"]:
            kv_idx[k].add(gid)
        if g["source_id"]:
            source_idx[g["source_id"]].add(gid)
    return domain_idx, entity_idx, kv_idx, source_idx


def signal_count(a: dict, b: dict) -> int:
    """Count matching signals between two gene dicts (max 4)."""
    n = 0
    if a["domains"] & b["domains"]:
        n += 1
    if a["entities"] & b["entities"]:
        n += 1
    if a["kv_keys"] & b["kv_keys"]:
        n += 1
    # Both genes are from the load_open_genes query, so both OPEN — always +1
    n += 1
    return n


def build_sema_lookup(genome: Genome) -> tuple[dict, object]:
    """Return (gid → matrix-row-index, numpy matrix or None).
    Forces the _sema_cache build which normally happens on first query."""
    try:
        genome._build_sema_cache()
    except Exception:
        log.warning("ΣĒMA cache failed to build; all weights will be 0.7", exc_info=True)
        return {}, None
    cache = getattr(genome, "_sema_cache", None)
    if not cache or cache.get("matrix") is None:
        return {}, None
    gids = cache["gene_ids"]
    return {g: i for i, g in enumerate(gids)}, cache["matrix"]


def sema_cos(matrix, ia: int, ib: int) -> float:
    import numpy as np
    va = matrix[ia]
    vb = matrix[ib]
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute but do not insert")
    ap.add_argument("--k", type=int, default=20,
                    help="Top-K seeded edges per gene (default 20)")
    ap.add_argument("--min-cross-source-cos", type=float, default=0.5,
                    help="Minimum ΣĒMA cosine for cross-source pairs (default 0.5)")
    ap.add_argument("--batch", type=int, default=5000,
                    help="Commit every N inserts (default 5000)")
    ap.add_argument("--inbound-cap", type=int, default=500,
                    help="Max inbound seeded edges per gene — anti-hub guard (default 500)")
    args = ap.parse_args()

    cfg = load_config()
    log.info("Opening genome at %s", cfg.genome.path)
    genome = Genome(path=cfg.genome.path, synonym_map=cfg.synonym_map)

    t0 = time.time()
    log.info("Loading OPEN genes ...")
    genes = load_open_genes(genome)
    log.info("  %d OPEN genes loaded in %.1fs", len(genes), time.time() - t0)

    t0 = time.time()
    log.info("Building inverted indexes (domain / entity / kv / source) ...")
    domain_idx, entity_idx, kv_idx, source_idx = build_inverted_indexes(genes)
    log.info("  %d domains, %d entities, %d kv keys, %d sources in %.1fs",
             len(domain_idx), len(entity_idx), len(kv_idx), len(source_idx),
             time.time() - t0)

    t0 = time.time()
    log.info("Building ΣĒMA cache ...")
    gid_to_semaidx, sema_matrix = build_sema_lookup(genome)
    log.info("  %d vectors in %.1fs", len(gid_to_semaidx), time.time() - t0)

    # Map gene_id -> gene dict for fast lookup
    gid_to_gene = {g["gene_id"]: g for g in genes}

    log.info("Scoring candidate pairs (top-K per gene = %d) ...", args.k)
    t0 = time.time()
    # Per-gene candidate pool -> list of (weight, other_gid)
    per_gene_candidates: dict[str, list[tuple[float, str]]] = {}
    scored_pairs = 0
    admitted_pairs = 0

    for i, g in enumerate(genes):
        if i % 2000 == 0 and i > 0:
            log.info("  scored %d/%d genes (admitted=%d, scored=%d)",
                     i, len(genes), admitted_pairs, scored_pairs)
        gid = g["gene_id"]
        # Union of candidates by inverted-index hits. Exclude self.
        cand: set = set()
        for d in g["domains"]:
            cand.update(domain_idx.get(d, ()))
        for e in g["entities"]:
            cand.update(entity_idx.get(e, ()))
        for k in g["kv_keys"]:
            cand.update(kv_idx.get(k, ()))
        cand.discard(gid)

        ranked: list[tuple[float, str]] = []
        for other in cand:
            scored_pairs += 1
            b = gid_to_gene.get(other)
            if b is None:
                continue
            sig = signal_count(g, b)
            if sig < 2:
                continue
            cos = 0.0
            if sema_matrix is not None:
                ia = gid_to_semaidx.get(gid)
                ib = gid_to_semaidx.get(other)
                if ia is not None and ib is not None:
                    cos = sema_cos(sema_matrix, ia, ib)
            # Cross-source pairs require stronger semantic signal
            if g["source_id"] != b["source_id"] and cos < args.min_cross_source_cos:
                continue
            w = _weight(cos)
            ranked.append((w, other))
            admitted_pairs += 1

        # Top-K per gene by descending weight
        ranked.sort(reverse=True)
        per_gene_candidates[gid] = ranked[: args.k]

    log.info("  scored_pairs=%d admitted_pairs=%d in %.1fs",
             scored_pairs, admitted_pairs, time.time() - t0)

    # Deduplicate edges (A,B) vs (B,A) — canonicalise by gene_id ordering
    edges: dict[tuple[str, str], float] = {}
    for gid, ranked in per_gene_candidates.items():
        for w, other in ranked:
            a, b = (gid, other) if gid < other else (other, gid)
            if (a, b) in edges:
                edges[(a, b)] = max(edges[(a, b)], w)
            else:
                edges[(a, b)] = w
    log.info("Canonical edge count (after dedup): %d", len(edges))

    # Inbound-degree cap — prevents "topic bridge" hub genes from being
    # the anchor of too many edges. Biological parallel: synaptic
    # wiring density cap per neuron; stops a single cell becoming an
    # absorbing state in Markov propagation (the Agentome "computational
    # seizure" failure mode).
    INBOUND_CAP = args.inbound_cap
    inbound: dict[str, list[tuple[float, tuple[str, str]]]] = defaultdict(list)
    for (a, b), w in edges.items():
        inbound[a].append((w, (a, b)))
        inbound[b].append((w, (a, b)))
    trimmed = 0
    for gid, inlist in inbound.items():
        if len(inlist) > INBOUND_CAP:
            inlist.sort(reverse=True)
            for _, edge_key in inlist[INBOUND_CAP:]:
                if edge_key in edges:
                    del edges[edge_key]
                    trimmed += 1
    if trimmed:
        log.info("Trimmed %d edges from %d hub genes (inbound cap=%d)",
                 trimmed,
                 sum(1 for gid, il in inbound.items() if len(il) > INBOUND_CAP),
                 INBOUND_CAP)
    log.info("Edge count after inbound-degree cap: %d", len(edges))

    if args.dry_run:
        log.info("Dry run — no inserts performed.")
        # Small distribution report
        if edges:
            import statistics
            weights = list(edges.values())
            log.info("weight stats: min=%.3f median=%.3f max=%.3f mean=%.3f",
                     min(weights), statistics.median(weights), max(weights),
                     sum(weights) / len(weights))
        return 0

    log.info("Inserting %d edges as source='seeded' in batches of %d ...",
             len(edges), args.batch)
    t0 = time.time()
    now = time.time()
    cur = genome.conn.cursor()
    sql = (
        "INSERT INTO harmonic_links "
        "(gene_id_a, gene_id_b, weight, updated_at, source, co_count, miss_count, created_at) "
        "VALUES (?, ?, ?, ?, 'seeded', 0, 0.0, ?) "
        "ON CONFLICT(gene_id_a, gene_id_b) DO NOTHING"
    )
    inserted = 0
    skipped = 0
    batch: list = []
    for (a, b), w in edges.items():
        batch.append((a, b, w, now, now))
        if len(batch) >= args.batch:
            before = cur.rowcount
            cur.executemany(sql, batch)
            genome.conn.commit()
            batch.clear()
    if batch:
        cur.executemany(sql, batch)
        genome.conn.commit()

    # Post-insert distribution
    total = cur.execute(
        "SELECT COUNT(*) FROM harmonic_links WHERE source='seeded'"
    ).fetchone()[0]
    co_tot = cur.execute(
        "SELECT COUNT(*) FROM harmonic_links WHERE source='co_retrieved'"
    ).fetchone()[0]
    per_gene = cur.execute(
        "SELECT gene_id_a AS g, COUNT(*) AS c FROM harmonic_links "
        "WHERE source='seeded' GROUP BY gene_id_a "
        "ORDER BY c DESC LIMIT 5"
    ).fetchall()
    log.info("Post-insert: seeded=%d co_retrieved=%d (time=%.1fs)",
             total, co_tot, time.time() - t0)
    log.info("Top-5 hub genes by seeded out-edges:")
    for r in per_gene:
        log.info("  %s: %d seeded edges", r["g"], r["c"])

    genome.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

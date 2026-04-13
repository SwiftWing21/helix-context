"""
Sprint 4 - Seeded co-activation edges with Hebbian evidence decay.

Solves the "cold-start graph-topology" gap flagged in Sprint 2:
brand-new OPEN genes have no co_activated_with history, so the Tier 5
harmonic boost and Tier 5.5 SR propagation cannot surface them on the
first few retrievals after ingest. Seeding edges at ingest time - when
multi-signal agreement suggests two genes are semantically linked -
gives new genes a "social standing" in the graph before the first
retrieval lands.

Design
------
Three edge classes, provenance-tagged on harmonic_links.source:
  seeded            weight_mul 0.3   - admitted on multi-signal agreement
  co_retrieved      weight_mul 0.7   - observed co-expression in retrieval
  cwola_validated   weight_mul 1.0   - classifier-confirmed (Sprint 3)

Each edge carries Laplace-smoothed evidence counters:
  co_count          - # co-retrievals (both endpoints in expressed set)
  miss_count        - sum of dense-rank weighted miss events

Effective weight during retrieval scoring:
  ratio = (co_count + 1) / (co_count + miss_count + 2)
  weight = raw_weight * source_mul * ratio

This gives new seeds Optimistic Neutrality (ratio=0.5 at no evidence),
promoting with sustained co-expression and attriting under contradiction.

Promotion thresholds:
  seeded -> co_retrieved         co_count >= 3 AND ratio >= 0.4
  co_retrieved -> cwola_validated (deferred to Sprint 3 trainer)

Pruning: effective weight < PRUNE_FLOOR drops the edge entirely.

Miss counting uses dense-rank so tied genes at the cutoff share the
same miss_weight, preventing stable-sort artifacts from creating
"persistent bad-luck" seeds.

See STATISTICAL_FUSION.md + SUCCESSOR_REPRESENTATION.md for context.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .genome import Genome

log = logging.getLogger("helix.seeded_edges")


SOURCE_SEEDED = "seeded"
SOURCE_CO_RETRIEVED = "co_retrieved"
SOURCE_CWOLA = "cwola_validated"

SOURCE_WEIGHT_MULTIPLIER = {
    SOURCE_SEEDED: 0.3,
    SOURCE_CO_RETRIEVED: 0.7,
    SOURCE_CWOLA: 1.0,
}

CO_PROMOTE_MIN_COUNT = 3      # co_count floor for seeded -> co_retrieved
CO_PROMOTE_MIN_RATIO = 0.4    # Laplace ratio floor for seeded -> co_retrieved
PRUNE_FLOOR = 0.05            # effective-weight floor below which edge is deleted


def _laplace_ratio(co: int, miss: float) -> float:
    """(co + 1) / (co + miss + 2). No evidence -> 0.5; strong co -> ~1.0;
    strong miss -> ~0.0. Prevents one-off events from flipping the edge."""
    return (co + 1) / (co + miss + 2)


def effective_weight(raw_weight: float, source: str, co: int, miss: float) -> float:
    """Retrieval-time weight factoring in provenance + Hebbian evidence."""
    mul = SOURCE_WEIGHT_MULTIPLIER.get(source, 1.0)
    return raw_weight * mul * _laplace_ratio(co, miss)


def dense_rank(sorted_scores: List[float]) -> Dict[int, int]:
    """Return {index_in_sorted_list: dense_rank}, rank 1 for the top.

    Tied scores share a rank. sorted_scores must already be sorted
    descending. Used for miss-weight computation so genes tied at the
    cutoff share the same penalty rather than having it arbitrarily
    pinned to whoever stable-sort happened to put first.
    """
    ranks: Dict[int, int] = {}
    current_rank = 0
    last_score: Optional[float] = None
    for i, s in enumerate(sorted_scores):
        if last_score is None or s != last_score:
            current_rank += 1
            last_score = s
        ranks[i] = current_rank
    return ranks


def miss_weight(rank: int, max_genes: int) -> float:
    """Dense-rank miss weight. Y ranked just below the cut -> near 1.0;
    Y ranked deep in the candidate pool -> near 0.0."""
    if rank <= max_genes or rank <= 0:
        return 0.0
    return min(1.0, max_genes / rank)


def seed_edges(
    genome: "Genome",
    gene_ids: Iterable[str],
    *,
    overlap_fn=None,
    weight: float = 1.0,
) -> int:
    """Ingest-batch seeding pass over OPEN genes only.

    For each unordered pair in gene_ids, check if overlap_fn(genome, a, b)
    returns True (multi-signal agreement gate); if so insert or
    refresh a 'seeded' edge. Returns the number of edges written.

    overlap_fn defaults to multi_signal_overlap below. Caller can pass
    a stub in tests.
    """
    if overlap_fn is None:
        overlap_fn = multi_signal_overlap
    gene_ids = list(gene_ids)
    if len(gene_ids) < 2:
        return 0
    import time as _time

    cur = genome.conn.cursor()
    now = _time.time()
    written = 0
    for i in range(len(gene_ids)):
        for j in range(i + 1, len(gene_ids)):
            a, b = gene_ids[i], gene_ids[j]
            if not overlap_fn(genome, a, b):
                continue
            try:
                cur.execute(
                    """INSERT INTO harmonic_links
                       (gene_id_a, gene_id_b, weight, updated_at,
                        source, co_count, miss_count, created_at)
                       VALUES (?, ?, ?, ?, 'seeded', 0, 0.0, ?)
                       ON CONFLICT(gene_id_a, gene_id_b) DO NOTHING""",
                    (a, b, weight, now, now),
                )
                if cur.rowcount:
                    written += 1
            except Exception:
                log.debug("seed_edges insert failed for (%s,%s)", a, b, exc_info=True)
    if written:
        genome.conn.commit()
    return written


def multi_signal_overlap(genome: "Genome", gene_id_a: str, gene_id_b: str) -> bool:
    """Two-of-N signal agreement gate for ingest-batch seeding.

    Requires at least two of:
      - shared domain tag
      - shared entity tag
      - shared KV key (ignores value)
      - both OPEN chromatin state
    Same-directory proximity alone is NOT enough - the gate's whole
    purpose is to avoid file-system coincidence edges.
    """
    cur = genome.read_conn.cursor()
    rows = cur.execute(
        "SELECT gene_id, promoter, key_values, chromatin "
        "FROM genes WHERE gene_id IN (?, ?)", (gene_id_a, gene_id_b),
    ).fetchall()
    if len(rows) != 2:
        return False

    import json
    signals = 0
    proms = []
    kvs = []
    chroms = []
    for r in rows:
        try:
            p = json.loads(r["promoter"]) if r["promoter"] else {}
        except Exception:
            p = {}
        try:
            kv = json.loads(r["key_values"]) if r["key_values"] else []
        except Exception:
            kv = []
        proms.append(p)
        kvs.append(kv)
        chroms.append(r["chromatin"])

    d_a = set((proms[0].get("domains") or []))
    d_b = set((proms[1].get("domains") or []))
    if d_a & d_b:
        signals += 1

    e_a = set((proms[0].get("entities") or []))
    e_b = set((proms[1].get("entities") or []))
    if e_a & e_b:
        signals += 1

    k_a = {pair.split("=", 1)[0] for pair in kvs[0] if "=" in pair}
    k_b = {pair.split("=", 1)[0] for pair in kvs[1] if "=" in pair}
    if k_a & k_b:
        signals += 1

    if chroms[0] == 0 and chroms[1] == 0:  # both OPEN
        signals += 1

    return signals >= 2


def update_edge_evidence(
    genome: "Genome",
    gene_scores: Dict[str, float],
    expressed_ids: List[str],
    *,
    max_genes: int,
) -> int:
    """Walk seeded / co_retrieved edges anchored at expressed genes,
    increment co_count or miss_count per dense-rank miss weighting.

    Returns count of edges updated. Intended to be called once per
    retrieval after candidates are finalised. Caller owns the commit
    cadence (we commit here; override via transaction-control if that
    becomes a contention bottleneck on very high QPS).
    """
    if not expressed_ids or not gene_scores:
        return 0

    sorted_ids = sorted(gene_scores.keys(), key=lambda g: gene_scores[g], reverse=True)
    sorted_scores = [gene_scores[g] for g in sorted_ids]
    ranks = dense_rank(sorted_scores)
    id_to_rank = {g: ranks[i] for i, g in enumerate(sorted_ids)}
    expressed_set = set(expressed_ids)

    cur = genome.conn.cursor()
    # Dedupe: an edge (a,b) must update exactly once even if both a and b
    # are in expressed_ids. Collect unique edges first, then walk.
    placeholders = ",".join("?" * len(expressed_ids))
    try:
        edges = cur.execute(
            f"""SELECT gene_id_a, gene_id_b, co_count, miss_count, source
                FROM harmonic_links
                WHERE (gene_id_a IN ({placeholders}) OR gene_id_b IN ({placeholders}))
                  AND source IN ('seeded', 'co_retrieved')""",
            (*expressed_ids, *expressed_ids),
        ).fetchall()
    except Exception:
        log.debug("update_edge_evidence read failed", exc_info=True)
        return 0

    updates = 0
    for row in edges:
        a, b = row["gene_id_a"], row["gene_id_b"]
        if a == b:
            continue
        # Decide which endpoint anchors this event. If both are
        # expressed it's a co_count event (order doesn't matter). If
        # one is expressed, the other is either a candidate miss or
        # outside the pool (candidacy gate via id_to_rank).
        if a in expressed_set and b in expressed_set:
            new_co = row["co_count"] + 1
            new_miss = row["miss_count"]
            new_source = row["source"]
            if (
                new_source == SOURCE_SEEDED
                and new_co >= CO_PROMOTE_MIN_COUNT
                and _laplace_ratio(new_co, new_miss) >= CO_PROMOTE_MIN_RATIO
            ):
                new_source = SOURCE_CO_RETRIEVED
            cur.execute(
                """UPDATE harmonic_links SET co_count=?, source=?, updated_at=strftime('%s','now')
                   WHERE gene_id_a=? AND gene_id_b=?""",
                (new_co, new_source, a, b),
            )
            updates += 1
            continue

        # Exactly one endpoint is in expressed_set - the other is the
        # candidate to weigh for miss. Candidacy gate: must also appear
        # in gene_scores (i.e. scored nonzero on some tier).
        other = b if a in expressed_set else a
        rank = id_to_rank.get(other)
        if rank is None:
            continue
        mw = miss_weight(rank, max_genes)
        if mw <= 0:
            continue
        new_miss = row["miss_count"] + mw
        new_co = row["co_count"]
        eff = effective_weight(1.0, row["source"], new_co, new_miss)
        if eff < PRUNE_FLOOR:
            cur.execute(
                "DELETE FROM harmonic_links WHERE gene_id_a=? AND gene_id_b=?",
                (a, b),
            )
        else:
            cur.execute(
                """UPDATE harmonic_links SET miss_count=?, updated_at=strftime('%s','now')
                   WHERE gene_id_a=? AND gene_id_b=?""",
                (new_miss, a, b),
            )
        updates += 1
    if updates:
        genome.conn.commit()
    return updates

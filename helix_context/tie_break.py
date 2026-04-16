"""
Walking tie-break — associative-graph-informed ordering for tied top-k scores.

When helix's 12-tier fusion produces two or more genes with bitwise-identical
scores, the current behaviour is to fall through to dict insertion order —
effectively arbitrary. This module replaces that arbitrary choice with a
deterministic ladder of signals drawn from data helix already computes.

Empirical basis for the ladder (see docs/FUTURE/tie_break_walking.md):

    Pass 3b (2026-04-15) measured signal availability on 74 real tied pairs:
        - neighborhood_size asymmetric:    94.6% of ties
        - direct harmonic edge present:    45.9%
        - NLI relation between:            32.4%
        - ANY signal present:              94.6% (100% for rank 0-3 head ties)
        - graph-invisible (isolates):       5.4% (tail only)

The ladder below uses those signals in the order their asymmetry is
most interpretable. Each rule is a pairwise comparator that either
returns a decision or abstains and lets the next rule try.

Opt-in via HELIX_WALKING_TIEBREAK=1 for now. No config-schema change
until the approach is validated on real workloads.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from functools import cmp_to_key
from typing import Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_STRONG_EDGE_THRESHOLD = 0.7   # edges above this mean "graph-validated twins"


def is_enabled() -> bool:
    """True iff HELIX_WALKING_TIEBREAK is set to a truthy value."""
    return os.environ.get("HELIX_WALKING_TIEBREAK", "").lower() in _TRUTHY


# ── Per-gene attribute lookup (batched for efficiency) ──────────────

def _fetch_gene_attrs(
    conn: sqlite3.Connection, gene_ids: Sequence[str]
) -> Dict[str, Dict]:
    """Fetch neighbor counts + freshness for each gene in one round trip.

    Returns: gene_id -> {"neighbors": int, "freshness": int}
    Genes absent from genes table or with no harmonic_links still get an
    entry with sensible defaults (neighbors=0, freshness=0). The ladder
    treats "no data" as a decidable value, not as missing.
    """
    if not gene_ids:
        return {}
    placeholders = ",".join("?" * len(gene_ids))

    # Neighbor counts via both edge directions
    neighbors: Dict[str, int] = {gid: 0 for gid in gene_ids}
    for gid, n in conn.execute(
        f"SELECT gene_id_a, COUNT(*) FROM harmonic_links "
        f"WHERE gene_id_a IN ({placeholders}) GROUP BY gene_id_a",
        tuple(gene_ids),
    ):
        neighbors[gid] = neighbors.get(gid, 0) + n
    for gid, n in conn.execute(
        f"SELECT gene_id_b, COUNT(*) FROM harmonic_links "
        f"WHERE gene_id_b IN ({placeholders}) GROUP BY gene_id_b",
        tuple(gene_ids),
    ):
        neighbors[gid] = neighbors.get(gid, 0) + n

    # Freshness proxy — SQLite rowid is monotonically increasing with
    # INSERT order, so "larger rowid = later ingest." Not perfect (a
    # rewritten gene keeps its rowid), but deterministic and available
    # without a dedicated created_at column. Missing rows → 0 (treated
    # as oldest possible, so known-ingested genes beat unknown ones).
    freshness: Dict[str, int] = {gid: 0 for gid in gene_ids}
    for gid, rid in conn.execute(
        f"SELECT gene_id, rowid FROM genes WHERE gene_id IN ({placeholders})",
        tuple(gene_ids),
    ):
        freshness[gid] = int(rid or 0)

    return {
        gid: {"neighbors": neighbors.get(gid, 0), "freshness": freshness.get(gid, 0)}
        for gid in gene_ids
    }


def _fetch_pairwise_edges(
    conn: sqlite3.Connection, gene_ids: Sequence[str]
) -> Dict[Tuple[str, str], float]:
    """Fetch harmonic_link weights for all ordered pairs in the group.

    Key is a normalized (a, b) tuple where a < b lexically, so each edge
    appears once regardless of storage direction.
    """
    if len(gene_ids) < 2:
        return {}
    placeholders = ",".join("?" * len(gene_ids))
    edges: Dict[Tuple[str, str], float] = {}
    for a, b, w in conn.execute(
        f"SELECT gene_id_a, gene_id_b, weight FROM harmonic_links "
        f"WHERE gene_id_a IN ({placeholders}) AND gene_id_b IN ({placeholders})",
        tuple(gene_ids) * 2,
    ):
        key = (a, b) if a < b else (b, a)
        # If both directions exist, keep the max weight (union semantics)
        edges[key] = max(edges.get(key, 0.0), float(w or 0.0))
    return edges


def _fetch_nli(
    conn: sqlite3.Connection, gene_ids: Sequence[str]
) -> Dict[Tuple[str, str], Tuple[int, float]]:
    """Fetch gene_relations rows for ordered pairs.

    Returns: (a, b) -> (relation, confidence) where (a, b) is stored-
    direction so the sign of the relation is preserved (a entails b,
    or a contradicts b — these are directional).
    """
    if len(gene_ids) < 2:
        return {}
    placeholders = ",".join("?" * len(gene_ids))
    rels: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for a, b, rel, conf in conn.execute(
        f"SELECT gene_id_a, gene_id_b, relation, confidence FROM gene_relations "
        f"WHERE gene_id_a IN ({placeholders}) AND gene_id_b IN ({placeholders})",
        tuple(gene_ids) * 2,
    ):
        rels[(a, b)] = (int(rel), float(conf or 0.0))
    return rels


# ── The ladder ──────────────────────────────────────────────────────

# Relation constants mirror schemas.py::NLRelation
_NLI_ENTAILS = 1       # a entails b (a strictly more informative)
_NLI_CONTRADICTS = 2   # a contradicts b (surface both, don't pick)


def _rule_strong_edge_freshness(
    a: str, b: str, edges: Dict, attrs: Dict
) -> Optional[int]:
    """If a strong harmonic edge exists between a and b, they're graph-
    validated twins. Prefer the fresher one; freshness is a walk-direction
    signal in its own right (recent retrievals carry recency context).
    """
    key = (a, b) if a < b else (b, a)
    edge = edges.get(key)
    if edge is None or edge < _STRONG_EDGE_THRESHOLD:
        return None
    ta = attrs[a]["freshness"]
    tb = attrs[b]["freshness"]
    if ta == tb:
        return None  # abstain, try next rule
    return -1 if ta > tb else 1  # -1 means "a comes first"


def _rule_neighborhood_size(
    a: str, b: str, edges: Dict, attrs: Dict
) -> Optional[int]:
    """Bigger harmonic-neighborhood = more walkable in general. Centrality
    beats peripherality. Decided by neighbor count; abstains on exact tie.
    """
    na = attrs[a]["neighbors"]
    nb = attrs[b]["neighbors"]
    if na == nb:
        return None
    return -1 if na > nb else 1


def _rule_nli_entailment(
    a: str, b: str, edges: Dict, attrs: Dict, nli: Dict
) -> Optional[int]:
    """If an NLI edge says a entails b, prefer a (strictly more informative).
    Symmetric contradiction abstains — contradictions should probably be
    surfaced explicitly rather than resolved silently.
    """
    row_ab = nli.get((a, b))
    row_ba = nli.get((b, a))
    if row_ab and row_ab[0] == _NLI_ENTAILS and row_ab[1] >= 0.5:
        return -1
    if row_ba and row_ba[0] == _NLI_ENTAILS and row_ba[1] >= 0.5:
        return 1
    return None


def _rule_freshness_fallback(
    a: str, b: str, edges: Dict, attrs: Dict
) -> Optional[int]:
    """Final non-arbitrary fallback: prefer the fresher gene. Recency is
    a real signal (new genes more likely to match current context) and
    is deterministic."""
    ta = attrs[a]["freshness"]
    tb = attrs[b]["freshness"]
    if ta == tb:
        return None
    return -1 if ta > tb else 1


def _rule_lexical_gene_id(
    a: str, b: str, edges: Dict, attrs: Dict
) -> int:
    """Deterministic lexical fallback. Only reached when every other
    signal has abstained; guarantees a stable total order."""
    return -1 if a < b else (1 if a > b else 0)


# ── Public entry point ─────────────────────────────────────────────

def walking_reorder(
    conn: sqlite3.Connection,
    gene_ids_sorted: List[str],
    scores: Dict[str, float],
) -> List[str]:
    """Re-order tied-score runs within an already-sorted gene_ids list.

    Groups consecutive gene_ids with identical scores, then applies the
    walking ladder to each multi-gene group. Single-gene groups (no tie)
    pass through unchanged.

    The overall score ordering is preserved — only within-tie ordering
    changes. This is an additive, non-breaking refinement.
    """
    if len(gene_ids_sorted) < 2:
        return list(gene_ids_sorted)

    # Group adjacent tied genes
    groups: List[List[str]] = []
    i = 0
    while i < len(gene_ids_sorted):
        j = i + 1
        si = scores.get(gene_ids_sorted[i], 0.0)
        while j < len(gene_ids_sorted) and scores.get(gene_ids_sorted[j], 0.0) == si:
            j += 1
        groups.append(gene_ids_sorted[i:j])
        i = j

    # Fast path: no ties at all
    if not any(len(g) > 1 for g in groups):
        return list(gene_ids_sorted)

    # Batched attribute fetch for every gene in a tied group
    tied_gene_ids = [gid for g in groups if len(g) > 1 for gid in g]
    attrs = _fetch_gene_attrs(conn, tied_gene_ids)
    edges = _fetch_pairwise_edges(conn, tied_gene_ids)
    nli = _fetch_nli(conn, tied_gene_ids)

    def compare(a: str, b: str) -> int:
        for rule in (_rule_strong_edge_freshness, _rule_neighborhood_size):
            verdict = rule(a, b, edges, attrs)
            if verdict is not None:
                return verdict
        verdict = _rule_nli_entailment(a, b, edges, attrs, nli)
        if verdict is not None:
            return verdict
        for rule in (_rule_freshness_fallback,):
            verdict = rule(a, b, edges, attrs)
            if verdict is not None:
                return verdict
        return _rule_lexical_gene_id(a, b, edges, attrs)

    cmp_key = cmp_to_key(compare)

    reordered: List[str] = []
    for g in groups:
        if len(g) == 1:
            reordered.extend(g)
        else:
            reordered.extend(sorted(g, key=cmp_key))
    return reordered


def explain_pair(
    conn: sqlite3.Connection, a: str, b: str
) -> Dict:
    """Return a per-rule verdict trace for a single tied pair.

    Used by the A/B benchmark to render why the walking tie-break chose
    one gene over another. Shape: {rule_name: "a" | "b" | "abstain"}.
    """
    attrs = _fetch_gene_attrs(conn, [a, b])
    edges = _fetch_pairwise_edges(conn, [a, b])
    nli = _fetch_nli(conn, [a, b])

    def render(verdict: Optional[int]) -> str:
        if verdict is None:
            return "abstain"
        if verdict < 0:
            return "a"
        if verdict > 0:
            return "b"
        return "tie"

    trace = {
        "strong_edge_freshness": render(_rule_strong_edge_freshness(a, b, edges, attrs)),
        "neighborhood_size": render(_rule_neighborhood_size(a, b, edges, attrs)),
        "nli_entailment": render(_rule_nli_entailment(a, b, edges, attrs, nli)),
        "freshness_fallback": render(_rule_freshness_fallback(a, b, edges, attrs)),
        "lexical_gene_id": render(_rule_lexical_gene_id(a, b, edges, attrs)),
        "attrs_a": attrs.get(a),
        "attrs_b": attrs.get(b),
        "edge_weight": edges.get((a, b) if a < b else (b, a)),
        "nli_ab": nli.get((a, b)),
        "nli_ba": nli.get((b, a)),
    }
    return trace

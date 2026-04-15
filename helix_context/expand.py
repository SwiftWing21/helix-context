"""
`/context/expand` — Sprint 3 of the AI-consumer roadmap.

A dedicated endpoint for 1-hop graph traversal from a known gene_id,
letting the consumer follow a thread without spinning up a fresh
`/context` query.

Today the LLM consuming /context has to reverse-engineer the retrieval
to expand on something it just saw:

    "I got gene=abc12345 fired=harmonic:2.3 — what's connected to it?"
    → invent a synthetic follow-up query → /context → run all 6 pipeline
    steps → pay 2-8k tokens, most of it redundant

This module short-circuits that. Given a gene_id and a direction, walk
the existing co-activation graph (or the gene's stored
`co_activated_with` list) and return a compact summary of the 1-hop
neighborhood — sub-100 tokens typical, no ribosome splice, no rerank.

Three directions (per roadmap):
    forward:  harmonic_links.gene_id_a = X   (things X points to)
    backward: harmonic_links.gene_id_b = X   (things that point at X)
    sideways: gene.epigenetics.co_activated_with (the gene's own
              memory of what it's been co-retrieved with)

When `session_id` is provided, the response filters out genes the
consumer already has (via session_delivery_log), so expand responses
stay non-redundant with the conversation's running context.

See docs/FUTURE/AI_CONSUMER_ROADMAP_2026-04-14.md Sprint 3.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from . import session_delivery as _session_delivery
from .schemas import Gene

log = logging.getLogger("helix.expand")


DIRECTIONS = ("forward", "backward", "sideways")


# ── Direction-specific edge queries ──────────────────────────────────

def fetch_forward_neighbors(
    conn: sqlite3.Connection,
    gene_id: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """harmonic_links where gene_id_a = X → list of (gene_id_b, weight).

    Sorted by weight descending, capped at k.
    """
    rows = conn.execute(
        "SELECT gene_id_b AS gid, weight FROM harmonic_links "
        "WHERE gene_id_a = ? ORDER BY weight DESC LIMIT ?",
        (gene_id, k),
    ).fetchall()
    return [(r["gid"], float(r["weight"])) for r in rows]


def fetch_backward_neighbors(
    conn: sqlite3.Connection,
    gene_id: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """harmonic_links where gene_id_b = X → list of (gene_id_a, weight).

    Uses the idx_harmonic_b index added for SR bulk neighbor queries.
    """
    rows = conn.execute(
        "SELECT gene_id_a AS gid, weight FROM harmonic_links "
        "WHERE gene_id_b = ? ORDER BY weight DESC LIMIT ?",
        (gene_id, k),
    ).fetchall()
    return [(r["gid"], float(r["weight"])) for r in rows]


def fetch_sideways_neighbors(
    genome,
    gene_id: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """gene.epigenetics.co_activated_with — order-preserving, capped at k.

    No explicit weight is stored on co_activated_with, so a synthetic
    score is derived from position: first entry gets 1.0, last gets
    ~1.0 - (k-1)*step. This keeps the response shape uniform with the
    forward/backward paths without promising a real edge weight.

    Returns an empty list if the gene is missing from the genome.
    """
    try:
        g = genome.get_gene(gene_id)
    except Exception:
        log.debug("get_gene(%s) failed", gene_id, exc_info=True)
        return []
    if g is None:
        return []
    coacts = list(g.epigenetics.co_activated_with or [])[:k]
    if not coacts:
        return []
    step = 1.0 / max(len(coacts), 1)
    return [(gid, round(1.0 - i * step, 3)) for i, gid in enumerate(coacts)]


# ── Per-neighbor compact formatter ───────────────────────────────────

def format_neighbor_compact(
    gene: Gene,
    score: float,
    *,
    summary_max_chars: int = 80,
) -> Dict[str, Any]:
    """Serialize a Gene to a small JSON blob for the expand response.

    Targeting ~20-30 tokens per neighbor: gene_id, rounded score,
    trimmed summary, promoter tag lists. No content, no codons — the
    consumer can re-query via /genes/{gene_id} if they want more.
    """
    summary = (gene.promoter.summary or "")[:summary_max_chars]
    return {
        "gene_id": gene.gene_id,
        "score": round(float(score), 3),
        "summary": summary,
        "domains": list(gene.promoter.domains or []),
        "entities": list(gene.promoter.entities or []),
    }


# ── Top-level orchestration ──────────────────────────────────────────

def expand_neighbors(
    genome,
    *,
    gene_id: str,
    direction: str,
    k: int = 5,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """1-hop neighborhood from `gene_id` in the given direction.

    When `session_id` is provided, filters out genes already delivered
    in that session (via session_delivery_log) so the response is
    non-redundant with whatever the consumer is currently holding.

    Shape:
        {
          "gene_id": "...",
          "direction": "forward|backward|sideways",
          "neighbors": [{"gene_id": "...", "score": 0.85,
                         "summary": "...", "domains": [...],
                         "entities": [...]}, ...],
          "count": N,            # len(neighbors)
          "skipped_delivered": M  # 0 when session_id is None
        }

    Invalid direction raises ValueError — the HTTP layer converts this
    to a 400.
    """
    if direction not in DIRECTIONS:
        raise ValueError(
            f"direction must be one of {DIRECTIONS}, got {direction!r}"
        )
    if k <= 0:
        k = 5

    conn = genome.conn
    # Fetch extra candidates so we can still hit k after filtering out
    # already-delivered ones. The overhead is tiny and bounded.
    raw_k = max(k * 3, k + 5)
    if direction == "forward":
        raw = fetch_forward_neighbors(conn, gene_id, k=raw_k)
    elif direction == "backward":
        raw = fetch_backward_neighbors(conn, gene_id, k=raw_k)
    else:
        raw = fetch_sideways_neighbors(genome, gene_id, k=raw_k)

    skipped = 0
    neighbors: List[Dict[str, Any]] = []
    for gid, score in raw:
        if session_id is not None:
            try:
                prior = _session_delivery.already_delivered(
                    conn, session_id=session_id, gene_id=gid,
                )
                if prior is not None:
                    skipped += 1
                    continue
            except Exception:
                log.debug("already_delivered lookup failed", exc_info=True)
                # Fall through: don't drop the neighbor just because the
                # log check errored — consumer can still decide to skip.
        # Resolve the gene row so we can emit summary + promoter tags.
        try:
            gene = genome.get_gene(gid)
        except Exception:
            log.debug("get_gene(%s) failed during expand", gid, exc_info=True)
            gene = None
        if gene is None:
            # Dangling reference in harmonic_links — skip silently.
            continue
        neighbors.append(format_neighbor_compact(gene, score=score))
        if len(neighbors) >= k:
            break

    return {
        "gene_id": gene_id,
        "direction": direction,
        "neighbors": neighbors,
        "count": len(neighbors),
        "skipped_delivered": skipped,
    }

"""Claims-graph traversal — the DAG layer that consumes Phase 2 claim_edges.

Helix emits ``claims`` + ``claim_edges`` (see shard_schema.py + claims.py);
this module turns those into agent-actionable signals:

    - ``supersedes_chain(claim_id)`` — walk supersedes to the head (latest)
    - ``contradiction_clusters(claim_ids)`` — union-find on contradicts edges
    - ``resolve(claim_ids, policy)`` — apply a resolution policy
      (latest-supersedes / highest-authority / keep-all-with-flags)
    - ``topologically_sorted(claim_ids)`` — supersedes-chain topo order

Helix is the router above the stack (emits signals); this is the DAG
layer of the stack (executes traversal). Kept narrow by design —
contradiction *detection* lives in claim_edges generation; this module
only walks what's already been written.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from typing import Iterable, Optional

log = logging.getLogger("helix.claims_graph")


# ── Supersedes chain walking ───────────────────────────────────────


def supersedes_chain(
    conn: sqlite3.Connection,
    claim_id: str,
    max_depth: int = 32,
) -> list[str]:
    """Walk the ``supersedes_claim_id`` chain forward from ``claim_id``.

    Returns [claim_id, next_in_chain, ..., head] where the last element
    is the claim that NOTHING else supersedes. Cycles are broken at
    ``max_depth`` (supersedes should be acyclic, but defensive).
    """
    chain = [claim_id]
    seen = {claim_id}
    current = claim_id

    for _ in range(max_depth):
        # Find claims that supersede the current one (current was replaced by ...)
        row = conn.execute(
            "SELECT claim_id FROM claims WHERE supersedes_claim_id = ? LIMIT 1",
            (current,),
        ).fetchone()
        if row is None:
            break
        nxt = row[0]
        if nxt in seen:
            log.warning("supersedes cycle detected at %s → %s", current, nxt)
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    return chain


def latest_in_chain(
    conn: sqlite3.Connection,
    claim_id: str,
) -> str:
    """Return the head of the supersedes chain (what replaced this claim).

    If ``claim_id`` isn't superseded, returns it unchanged.
    """
    chain = supersedes_chain(conn, claim_id)
    return chain[-1]


# ── Contradiction clustering ───────────────────────────────────────


def _neighbors(
    conn: sqlite3.Connection,
    claim_id: str,
    edge_types: tuple[str, ...],
) -> set[str]:
    """Undirected neighbor lookup across `edge_types`."""
    rows = conn.execute(
        f"""SELECT src_claim_id, dst_claim_id FROM claim_edges
            WHERE edge_type IN ({','.join('?' * len(edge_types))})
            AND (src_claim_id = ? OR dst_claim_id = ?)""",
        (*edge_types, claim_id, claim_id),
    ).fetchall()
    out: set[str] = set()
    for src, dst in rows:
        if src == claim_id:
            out.add(dst)
        else:
            out.add(src)
    return out


def contradiction_clusters(
    conn: sqlite3.Connection,
    claim_ids: Iterable[str],
) -> list[list[str]]:
    """Union-find on ``contradicts`` + ``duplicates`` edges.

    Returns clusters (lists of claim_ids). Claims in the same cluster
    are mutually inconsistent — the agent must pick one or surface
    the conflict. Singletons (claims with no contradictions) get their
    own 1-element cluster for uniform downstream handling.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    claim_list = list(claim_ids)
    for cid in claim_list:
        parent.setdefault(cid, cid)

    for cid in claim_list:
        for neighbor in _neighbors(conn, cid, ("contradicts", "duplicates")):
            parent.setdefault(neighbor, neighbor)
            union(cid, neighbor)

    # Keep every claim we were asked about (even if no neighbors)
    clusters: dict[str, list[str]] = defaultdict(list)
    for cid in claim_list:
        clusters[find(cid)].append(cid)

    return [sorted(members) for members in clusters.values()]


# ── Topological order over supersedes ──────────────────────────────


def topologically_sorted(
    conn: sqlite3.Connection,
    claim_ids: Iterable[str],
) -> list[str]:
    """Return ``claim_ids`` in topological order: supersedes-predecessor first.

    If claim A is superseded by claim B, A appears before B in output.
    Cycles (shouldn't exist in a supersedes chain but guard anyway) are
    broken deterministically by claim_id lex order.
    """
    claim_set = set(claim_ids)
    if not claim_set:
        return []

    # Build adjacency: (superseded) → (replaced_by) for every edge
    # where BOTH endpoints are in our set.
    incoming: dict[str, set[str]] = {c: set() for c in claim_set}
    outgoing: dict[str, set[str]] = {c: set() for c in claim_set}

    rows = conn.execute(
        f"""SELECT claim_id, supersedes_claim_id FROM claims
            WHERE claim_id IN ({','.join('?' * len(claim_set))})
            AND supersedes_claim_id IS NOT NULL""",
        tuple(claim_set),
    ).fetchall()
    for replacer, superseded in rows:
        if superseded in claim_set:
            incoming[replacer].add(superseded)
            outgoing[superseded].add(replacer)

    # Kahn — roots are claims nothing in the set supersedes.
    ready = sorted([c for c in claim_set if not incoming[c]])
    ordered: list[str] = []
    seen: set[str] = set()
    while ready:
        cid = ready.pop(0)
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(cid)
        for nxt in sorted(outgoing[cid]):
            incoming[nxt].discard(cid)
            if not incoming[nxt] and nxt not in seen:
                ready.append(nxt)
        ready.sort()

    # Claims left out (caught in a cycle) — append deterministically
    leftover = sorted(claim_set - seen)
    ordered.extend(leftover)
    return ordered


# ── Resolution policies ────────────────────────────────────────────


_AUTHORITY_RANK = {"primary": 0, "derived": 1, "inferred": 2}


def resolve(
    conn: sqlite3.Connection,
    claim_ids: Iterable[str],
    policy: str = "latest_then_authority",
) -> dict:
    """Apply a resolution policy across a set of candidate claims.

    Policies:
        - ``latest_then_authority`` — follow supersedes chains to the head,
          then within each contradiction cluster pick highest authority
          (primary > derived > inferred). Ties broken by observed_at desc,
          then claim_id.
        - ``keep_all_with_flags`` — return every claim plus flags
          (``superseded_by`` / ``contradicts_ids``).

    Always returns a dict:
        {
            "accepted":  list[dict],   # claims the agent should act on
            "rejected":  list[dict],   # claims filtered out + reason
            "clusters":  list[list[str]],  # contradiction clusters (diag)
        }
    """
    claim_list = list(claim_ids)
    if not claim_list:
        return {"accepted": [], "rejected": [], "clusters": []}

    # Fetch full rows for the input set
    rows = conn.execute(
        f"""SELECT claim_id, gene_id, shard_name, claim_type, entity_key,
                   claim_text, extraction_kind, specificity, confidence,
                   observed_at, supersedes_claim_id, updated_at
            FROM claims
            WHERE claim_id IN ({','.join('?' * len(claim_list))})""",
        tuple(claim_list),
    ).fetchall()
    claims_by_id = {r[0]: dict(
        claim_id=r[0], gene_id=r[1], shard_name=r[2], claim_type=r[3],
        entity_key=r[4], claim_text=r[5], extraction_kind=r[6],
        specificity=r[7], confidence=r[8], observed_at=r[9],
        supersedes_claim_id=r[10], updated_at=r[11],
    ) for r in rows}

    if policy == "keep_all_with_flags":
        accepted = []
        for cid in claim_list:
            c = claims_by_id.get(cid)
            if not c:
                continue
            c = dict(c)
            c["contradicts_ids"] = sorted(
                _neighbors(conn, cid, ("contradicts",))
            )
            c["superseded_by"] = latest_in_chain(conn, cid) \
                if latest_in_chain(conn, cid) != cid else None
            accepted.append(c)
        return {
            "accepted": accepted,
            "rejected": [],
            "clusters": contradiction_clusters(conn, claim_list),
        }

    # Default: latest_then_authority
    # 1. Map every claim to its supersedes-chain head
    head_map = {cid: latest_in_chain(conn, cid) for cid in claim_list}
    heads = set(head_map.values())

    rejected: list[dict] = []
    for cid, head in head_map.items():
        if head != cid:
            c = claims_by_id.get(cid)
            if c:
                c = dict(c)
                c["rejected_reason"] = f"superseded_by {head}"
                rejected.append(c)

    # 2. Cluster heads by contradiction and pick winner per cluster
    clusters = contradiction_clusters(conn, list(heads))
    accepted: list[dict] = []
    for cluster in clusters:
        ranked = sorted(
            (claims_by_id[c] for c in cluster if c in claims_by_id),
            key=lambda c: (
                _AUTHORITY_RANK.get(
                    (c.get("claim_type") or "").lower(), 99
                ),
                -(c.get("observed_at") or 0),
                c["claim_id"],
            ),
        )
        if not ranked:
            continue
        winner = ranked[0]
        accepted.append(winner)
        for loser in ranked[1:]:
            loser = dict(loser)
            loser["rejected_reason"] = (
                f"contradicts_winner {winner['claim_id']}"
            )
            rejected.append(loser)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "clusters": clusters,
    }


# ── Packet-driven entry point ──────────────────────────────────────


def resolve_from_packet(
    conn: sqlite3.Connection,
    packet: dict,
    policy: str = "latest_then_authority",
) -> dict:
    """Convenience: pull claim_ids out of a Helix packet and resolve.

    The packet's ContextItems may carry ``claim_id`` pointers; we also
    look up claims for every ``gene_id`` mentioned. Returns the
    ``resolve()`` dict.
    """
    gene_ids: list[str] = []
    for bucket in ("verified", "stale_risk", "contradictions"):
        for item in packet.get(bucket, []) or []:
            if item.get("gene_id"):
                gene_ids.append(item["gene_id"])

    if not gene_ids:
        return {"accepted": [], "rejected": [], "clusters": []}

    rows = conn.execute(
        f"""SELECT claim_id FROM claims WHERE gene_id IN
            ({','.join('?' * len(gene_ids))})""",
        tuple(gene_ids),
    ).fetchall()
    claim_ids = [r[0] for r in rows]
    return resolve(conn, claim_ids, policy=policy)

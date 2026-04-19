"""Claim-edge detection — post-extraction pass that populates claim_edges.

Extraction (claims.py) produces literal claims from gene content. This
module groups those claims by ``entity_key`` and emits structural
edges the DAG walker (claims_graph.py) consumes:

- ``contradicts``: same entity_key, conflicting text from different
  genes. Fires when text similarity is LOW — two genes disagree on
  what the entity should be.
- ``duplicates``: same entity_key, near-identical text from different
  genes. The same fact recorded in multiple places.
- ``supersedes``: same entity_key + comparable text, one claim has a
  strictly newer ``observed_at`` than the other. Picks the newer claim
  as the canonical one.

Design choices:

- **Pairwise per entity_key group** with a safety cap. Groups bigger
  than ``max_group_size`` (default 50) are skipped to avoid O(N²)
  blowup on common keys like ``error`` or ``log``.
- **Token-set Jaccard** for similarity — cheap and good enough for
  the "is this the same fact?" question.
- **Asymmetric for supersedes**: edge points from old → new. For
  contradicts/duplicates we emit a single canonical ordering
  (lower claim_id → higher) to avoid duplicate rows.
- **Idempotent**: all writes go through ``upsert_claim_edge`` which
  replaces on conflict. Safe to re-run after new claims land.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from typing import Iterable, Optional

from .shard_schema import upsert_claim_edge

log = logging.getLogger("helix.claims_analyze")


# Entity keys too generic to meaningfully compare. Tune as we see
# false positives in production. Starts conservative.
_NOISE_ENTITY_KEYS = {
    "error", "log", "warning", "info", "debug", "trace", "data",
    "value", "item", "name", "type", "status", "config", "path",
    "file", "id", "key", "version", "port",  # "port" alone — port:8787 etc are ok
}


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> set[str]:
    """Lowercased token-set for Jaccard."""
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """Canonical ordering so (a,b) and (b,a) hash the same."""
    return (a, b) if a < b else (b, a)


def _fetch_claim_groups(
    conn: sqlite3.Connection,
    entity_keys: Optional[Iterable[str]] = None,
    max_group_size: int = 50,
) -> dict[str, list[dict]]:
    """Group claims by entity_key. Skip noise keys + oversize groups."""
    if entity_keys is not None:
        keys = [k for k in entity_keys
                if k and k.lower() not in _NOISE_ENTITY_KEYS]
        if not keys:
            return {}
        rows = conn.execute(
            f"""SELECT claim_id, gene_id, claim_type, entity_key,
                       claim_text, observed_at
                FROM claims
                WHERE entity_key IN ({','.join('?' * len(keys))})""",
            tuple(keys),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT claim_id, gene_id, claim_type, entity_key,
                      claim_text, observed_at
               FROM claims
               WHERE entity_key IS NOT NULL"""
        ).fetchall()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        ek = (row[3] or "").strip()
        if not ek or ek.lower() in _NOISE_ENTITY_KEYS:
            continue
        groups[ek].append(dict(
            claim_id=row[0],
            gene_id=row[1],
            claim_type=row[2],
            entity_key=row[3],
            claim_text=row[4] or "",
            observed_at=row[5],
        ))
    # Drop oversize groups
    return {k: v for k, v in groups.items() if len(v) <= max_group_size}


def detect_edges_for_group(
    claims: list[dict],
    duplicate_threshold: float = 0.8,
    contradict_threshold: float = 0.3,
) -> list[tuple[str, str, str, float]]:
    """Emit ``(src_claim_id, dst_claim_id, edge_type, weight)`` tuples
    for all pairs in this entity_key group.

    Rules:
        - Skip self-comparisons (same claim_id).
        - Skip pairs where both claims come from the *same gene* —
          same gene can't contradict itself.
        - Jaccard ≥ ``duplicate_threshold``:
            if both have ``observed_at`` and one is strictly newer
                → ``supersedes`` (older → newer)
            else
                → ``duplicates``
        - Jaccard ≤ ``contradict_threshold``:
            → ``contradicts``
        - In between (``contradict_threshold`` < J < ``duplicate_threshold``)
          we don't emit an edge — too similar to be contradicting,
          too different to call duplicates. Silent middle.
    """
    out: list[tuple[str, str, str, float]] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    # Precompute tokens once per claim
    tokens = [_tokenize(c["claim_text"]) for c in claims]

    for i, a in enumerate(claims):
        for j in range(i + 1, len(claims)):
            b = claims[j]
            if a["claim_id"] == b["claim_id"]:
                continue
            if a["gene_id"] == b["gene_id"]:
                continue
            sim = _jaccard(tokens[i], tokens[j])

            if sim >= duplicate_threshold:
                # Supersedes if we have observed_at and they differ
                oa_a = a.get("observed_at")
                oa_b = b.get("observed_at")
                if oa_a is not None and oa_b is not None and oa_a != oa_b:
                    older, newer = (a, b) if oa_a < oa_b else (b, a)
                    pair = (older["claim_id"], newer["claim_id"], "supersedes")
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        out.append((*pair, round(sim, 3)))
                else:
                    lo, hi = _pair_key(a["claim_id"], b["claim_id"])
                    pair = (lo, hi, "duplicates")
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        out.append((*pair, round(sim, 3)))
            elif sim <= contradict_threshold:
                lo, hi = _pair_key(a["claim_id"], b["claim_id"])
                pair = (lo, hi, "contradicts")
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    # Weight = 1 - sim, so lower-similarity contradictions
                    # score higher (more confident conflict).
                    out.append((*pair, round(1.0 - sim, 3)))
            # else: silent middle — no edge
    return out


def detect_and_persist_edges(
    conn: sqlite3.Connection,
    entity_keys: Optional[Iterable[str]] = None,
    max_group_size: int = 50,
    duplicate_threshold: float = 0.8,
    contradict_threshold: float = 0.3,
) -> dict:
    """Scan claims, emit edges to claim_edges. Returns a summary.

    Typical usage:
        - After bulk backfill (`scripts/backfill_claims.py`), call
          this once over all groups: ``detect_and_persist_edges(main_db)``.
        - After per-gene ingest that added claims, call with the
          affected entity_keys only:
          ``detect_and_persist_edges(main_db, entity_keys=set(keys))``.
    """
    groups = _fetch_claim_groups(conn, entity_keys, max_group_size)
    log.info("claim-edge scan: %d entity_key groups", len(groups))

    counts = defaultdict(int)
    for entity_key, claims in groups.items():
        edges = detect_edges_for_group(
            claims,
            duplicate_threshold=duplicate_threshold,
            contradict_threshold=contradict_threshold,
        )
        for src, dst, edge_type, weight in edges:
            upsert_claim_edge(
                conn, src_claim_id=src, dst_claim_id=dst,
                edge_type=edge_type, weight=weight,
            )
            counts[edge_type] += 1
    counts["n_groups"] = len(groups)
    counts["n_total_edges"] = sum(v for k, v in counts.items()
                                  if k in ("contradicts", "duplicates",
                                           "supersedes", "supports"))
    return dict(counts)

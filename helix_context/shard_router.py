"""ShardRouter — federated query across category shard .db files.

Task 2 of phase-2 sharding (docs/specs/2026-04-17-genome-sharding-plan.md).

Owns main.db (routing + fingerprint_index) and a lazy cache of Genome
instances for each category shard. On query, picks candidate shards from
fingerprint_index, fans out the query to each, merges results by score,
returns the top-K.

V1 design decisions:
- **Router fusion**: each shard runs its full query_genes (existing
  fusion math untouched), router merges by raw score. FTS BM25 is
  corpus-local, which means cross-shard scores are not perfectly
  calibrated — accepted for V1, revisit if round-trip validation
  shows regression. (See spec §"Open decisions".)
- **FTS5 placement**: unchanged. Each shard keeps its own FTS; main.db
  has only the fingerprint_index for shard selection.
- **Shard selection**: LIKE scan over JSON-encoded domains/entities in
  fingerprint_index. V2 can replace with a bloom prefilter or inverted
  index once shard count justifies it.

Not in this task:
- Ingest-time routing (Task 6)
- HelixContextManager integration behind HELIX_USE_SHARDS flag (Task 7)
- Cross-shard FTS bloom prefilter (deferred)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from .genome import Genome
from .schemas import Gene
from .shard_schema import list_shards, open_main_db

log = logging.getLogger(__name__)


class ShardRouter:
    """Routes queries across category shard .db files.

    Owns:
        - main.db connection (fingerprint_index + shards + identity)
        - lazy dict of Genome instances keyed by shard_name

    Exposes a subset of Genome's API (query_genes for now; ingest
    lands in Task 6).
    """

    def __init__(self, main_path: str, **genome_kwargs):
        """Open main.db. Shard Genomes open lazily on first access.

        genome_kwargs are forwarded to each Genome on lazy-open — keep
        them identical to what HelixContextManager passes when
        constructing a solo Genome, so sharded + non-sharded return
        identical tiers.
        """
        self.main_path = main_path
        self.main_conn = open_main_db(main_path)
        self._genome_kwargs = genome_kwargs
        self._shards: Dict[str, Genome] = {}

        # Retrieval introspection — mirrors Genome's interface so
        # callers can use either interchangeably.
        self.last_query_scores: Dict[str, float] = {}
        self.last_tier_contributions: Dict[str, Dict[str, float]] = {}

    # ── Shard lifecycle ─────────────────────────────────────────────

    def _open_shard(self, shard_name: str) -> Genome:
        """Lazy-open a Genome against the shard .db. Cached."""
        if shard_name not in self._shards:
            row = self.main_conn.execute(
                "SELECT path FROM shards WHERE shard_name = ? AND health = 'ok'",
                (shard_name,),
            ).fetchone()
            if row is None:
                raise ValueError(f"shard not registered: {shard_name}")
            path = row["path"]
            log.info("opening shard %s at %s", shard_name, path)
            self._shards[shard_name] = Genome(path, **self._genome_kwargs)
        return self._shards[shard_name]

    def known_shards(self, category: Optional[str] = None) -> List[str]:
        """Shard names registered in main.db, optionally filtered."""
        return [r["shard_name"] for r in list_shards(self.main_conn, category)]

    # ── Routing ─────────────────────────────────────────────────────

    def route(self, domains: List[str], entities: List[str]) -> List[str]:
        """Return shard names likely to contain matches for the query.

        V1: LIKE scan against fingerprint_index.domains/entities.
        Match is loose — if any query term appears in any shard's
        fingerprint domains or entities, that shard is in. Ordered
        by match count (most matches first).

        Empty query → return every healthy shard (preserves the
        fallback path callers rely on for "return something").
        """
        terms = [t.lower() for t in (domains + entities) if t]
        if not terms:
            return self.known_shards()

        # Match either domains or entities JSON containing the term
        # quoted exactly: ["auth"] or ["JWT"]. Simple substring match
        # on JSON-encoded lists.
        like_clauses = []
        params: list = []
        for t in terms:
            like_clauses.append("(LOWER(domains) LIKE ? OR LOWER(entities) LIKE ?)")
            needle = f'%"{t}"%'
            params.extend([needle, needle])

        sql = (
            "SELECT shard_name, COUNT(*) AS hits "
            "FROM fingerprint_index "
            f"WHERE {' OR '.join(like_clauses)} "
            "GROUP BY shard_name "
            "ORDER BY hits DESC"
        )
        rows = self.main_conn.execute(sql, params).fetchall()
        return [r["shard_name"] for r in rows]

    # ── Query fan-out ───────────────────────────────────────────────

    def query_genes(
        self,
        domains: List[str],
        entities: List[str],
        max_genes: int = 8,
        party_id: Optional[str] = None,
    ) -> List[Gene]:
        """Fan query across routed shards, merge by score, top-K.

        Mirrors Genome.query_genes signature so callers can swap
        without branching.
        """
        shard_names = self.route(domains, entities)
        if not shard_names:
            self.last_query_scores = {}
            self.last_tier_contributions = {}
            return []

        merged: Dict[str, Gene] = {}
        merged_scores: Dict[str, float] = {}
        merged_tier: Dict[str, Dict[str, float]] = {}

        for shard_name in shard_names:
            try:
                shard = self._open_shard(shard_name)
            except Exception:
                log.warning("shard %s failed to open; skipping", shard_name, exc_info=True)
                continue

            try:
                genes = shard.query_genes(
                    domains=domains,
                    entities=entities,
                    max_genes=max_genes,
                    party_id=party_id,
                )
            except Exception:
                log.warning("shard %s query failed; skipping", shard_name, exc_info=True)
                continue

            # Dedupe by gene_id — content-hashed ids should only collide
            # across shards if the same content was extracted twice
            # (legitimate in copy-extract phase). Keep the highest-score
            # copy.
            for gene in genes:
                score = shard.last_query_scores.get(gene.gene_id, 0.0)
                if gene.gene_id not in merged or score > merged_scores.get(gene.gene_id, 0.0):
                    merged[gene.gene_id] = gene
                    merged_scores[gene.gene_id] = score
                    merged_tier[gene.gene_id] = dict(
                        shard.last_tier_contributions.get(gene.gene_id, {})
                    )

        # Rank merged candidates by score, take top-K.
        ranked_ids = sorted(
            merged.keys(),
            key=lambda gid: merged_scores.get(gid, 0.0),
            reverse=True,
        )[:max_genes]

        self.last_query_scores = {gid: merged_scores[gid] for gid in ranked_ids}
        self.last_tier_contributions = {gid: merged_tier[gid] for gid in ranked_ids}
        return [merged[gid] for gid in ranked_ids]

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        """Close all lazy-opened shard genomes + main.db."""
        for shard_name, genome in self._shards.items():
            try:
                genome.conn.close()
            except Exception:
                log.warning("failed to close shard %s", shard_name, exc_info=True)
            try:
                if getattr(genome, "_reader", None):
                    genome._reader.close()
            except Exception:
                pass
        self._shards.clear()
        try:
            self.main_conn.close()
        except Exception:
            pass


# ── Feature-flag helper ──────────────────────────────────────────────


def use_shards_enabled() -> bool:
    """Read HELIX_USE_SHARDS env flag. Default OFF until Task 8 cutover."""
    return os.environ.get("HELIX_USE_SHARDS", "").strip() in ("1", "true", "yes", "on")

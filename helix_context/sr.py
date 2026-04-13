"""
Successor Representation - Tier 5.5 retrieval boost.

Stachenfeld, Botvinick & Gershman (2017) "The hippocampus as a
predictive map" (Nature Neuroscience 20:1643). The SR matrix

    M = (I - gamma * P)^-1  =  sum_{k=0..inf} gamma^k * P^k

gives, for each state s, the gamma-discounted expected number of
future visits to every other state. Tier 5's harmonic boost is
effectively the k=1 slice of this; SR generalises to multi-hop
futures without densifying the whole matrix.

For helix's 18K-gene genome, dense M is 18K x 18K float32 = 1.3 GB.
We never build that. Instead, per query, we compute M[seed, :] via a
truncated sparse power series over the co-activation graph - one row
per seed, k_steps sparse matvecs.

Per-row cost at k=4 and branching ~10 is ~10^4 ops, sub-millisecond.

Integration: slots between Tier 5 (harmonic, 1-hop co-activation) and
the access-rate tiebreaker in query_genes(). Contributes a "sr" bonus
per gene to the tier_contributions dict alongside the existing tiers.

See SUCCESSOR_REPRESENTATION.md for design + validation notes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .genome import Genome

log = logging.getLogger("helix.sr")

DEFAULT_GAMMA = 0.85
DEFAULT_K_STEPS = 4
DEFAULT_WEIGHT = 1.5
DEFAULT_CAP = 3.0
FRONTIER_CAP = 2000       # Max frontier size per hop — prevents pathological
                          # BFS expansion on a dense graph. Keeps latency
                          # bounded at O(FRONTIER_CAP × k_steps) SQL work.


def sr_boost(
    genome: "Genome",
    seed_ids: List[str],
    gamma: float = DEFAULT_GAMMA,
    k_steps: int = DEFAULT_K_STEPS,
    weight: float = DEFAULT_WEIGHT,
    cap: float = DEFAULT_CAP,
    co_activation_cache: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """Discounted future-occupancy boost over the co-activation graph.

    Walks k_steps from the seed set, spreading mass to co-activated
    neighbours at each step with discount factor gamma. Returns a
    {gene_id: bonus} dict for all genes reached, excluding seeds (they
    already scored on Tiers 0-5).

    Parameters mirror Stachenfeld 2017 sect "Modeling SR as RL value
    function":
      gamma    — discount factor (0.5: ~1.5-hop, 0.85: ~6-hop, 0.99: ~70-hop)
      k_steps  — truncation depth of sum_k gamma^k P^k
      weight   — per-gene contribution multiplier
      cap      — max per-gene boost; prevents a single runaway
                 propagation chain from saturating the score.

    co_activation_cache lets the caller hand in a prefetched adjacency
    dict (e.g. already built by ray_trace) to skip repeated DB hits.
    """
    if not seed_ids:
        return {}

    # Lazy import avoids a genome.py <-> sr.py circular dep.
    from .ray_trace import _load_co_activated

    # Build the neighbour lookup cache ONCE up-front. Without this, each
    # hop issued one SQL query per frontier gene — k_steps=4 over a
    # ~10K-gene frontier on the 191K-edge harmonic_links table was
    # ~30s per /context call (discovered via staged dim_lock A/B on
    # 2026-04-13). The lookup is now O(1) per frontier node after a
    # single-pass seed scan + BFS expansion.
    neighbours_cache: Dict[str, List[str]] = {}
    if co_activation_cache is not None:
        neighbours_cache = dict(co_activation_cache)

    # Hoist the harmonic_links lookup to a single bulk query per hop.
    def _fill_cache(gids: List[str]) -> None:
        """Populate neighbours_cache for any gids missing from it."""
        missing = [g for g in gids if g not in neighbours_cache]
        if not missing:
            return
        # Legacy source: gene.epigenetics.co_activated_with JSON field.
        # _load_co_activated is one row per call — fine at this scope
        # since `missing` is bounded by the BFS frontier size.
        legacy_lookup: Dict[str, List[str]] = {}
        for g in missing:
            legacy_lookup[g] = _load_co_activated(genome, g)
        # Sprint 4 source: harmonic_links table. One SQL round-trip for
        # the entire missing batch — IN (...) with a reasonable cap.
        link_lookup: Dict[str, List[str]] = {g: [] for g in missing}
        try:
            cur = genome.read_conn.cursor()
            placeholders = ",".join("?" * len(missing))
            rows = cur.execute(
                f"SELECT gene_id_a, gene_id_b FROM harmonic_links "
                f"WHERE gene_id_a IN ({placeholders}) OR gene_id_b IN ({placeholders})",
                (*missing, *missing),
            ).fetchall()
            for a, b in rows:
                if a in link_lookup:
                    link_lookup[a].append(b)
                if b in link_lookup:
                    link_lookup[b].append(a)
        except Exception:
            log.debug("harmonic_links bulk read failed", exc_info=True)
        # Union + dedupe per gene, store in cache.
        for g in missing:
            seen = set()
            out: List[str] = []
            for n in list(legacy_lookup.get(g, [])) + link_lookup.get(g, []):
                if n and n != g and n not in seen:
                    seen.add(n)
                    out.append(n)
            neighbours_cache[g] = out

    def neighbours(gid: str) -> List[str]:
        if gid not in neighbours_cache:
            _fill_cache([gid])
        return neighbours_cache[gid]

    # Uniform seed mass. Accumulator holds the discounted occupancy
    # measure; `mass` is the current wavefront that gets propagated.
    seed_mass = 1.0 / len(seed_ids)
    mass: Dict[str, float] = {gid: seed_mass for gid in seed_ids}
    accumulated: Dict[str, float] = dict(mass)

    # Pre-fill neighbours for the seeds so the first hop is a pure
    # in-memory scan. Subsequent hops fill their frontier in bulk too.
    _fill_cache(list(mass.keys()))

    for _step in range(k_steps):
        # Frontier cap: if the wavefront has exploded past FRONTIER_CAP,
        # keep only the top-N by current mass. Truncating the tail is
        # information-preserving — those nodes contribute negligibly to
        # SR at their mass level anyway.
        if len(mass) > FRONTIER_CAP:
            top = sorted(mass.items(), key=lambda x: -x[1])[:FRONTIER_CAP]
            mass = dict(top)
        # Bulk-fill any frontier nodes whose neighbours aren't cached yet.
        _fill_cache(list(mass.keys()))
        next_mass: Dict[str, float] = {}
        for gid, m in mass.items():
            ns = neighbours(gid)
            if not ns:
                continue
            share = (gamma * m) / len(ns)
            for n in ns:
                next_mass[n] = next_mass.get(n, 0.0) + share
        if not next_mass:
            break
        for n, m in next_mass.items():
            accumulated[n] = accumulated.get(n, 0.0) + m
        mass = next_mass

    seed_set = set(seed_ids)
    return {
        gid: min(weight * v, cap)
        for gid, v in accumulated.items()
        if gid not in seed_set
    }

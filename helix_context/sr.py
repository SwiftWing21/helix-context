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

    def neighbours(gid: str) -> List[str]:
        if co_activation_cache is not None and gid in co_activation_cache:
            return co_activation_cache[gid]
        return _load_co_activated(genome, gid)

    # Uniform seed mass. Accumulator holds the discounted occupancy
    # measure; `mass` is the current wavefront that gets propagated.
    seed_mass = 1.0 / len(seed_ids)
    mass: Dict[str, float] = {gid: seed_mass for gid in seed_ids}
    accumulated: Dict[str, float] = dict(mass)

    for _step in range(k_steps):
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

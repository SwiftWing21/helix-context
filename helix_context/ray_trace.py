"""
Monte Carlo evidence propagation over the gene co-activation graph.

Ports the ScoreRift ray-trace pattern into a retrieval dimension for
helix-context.  Casts random rays from seed genes through co-activation
edges (``co_activated_with`` + ``harmonic_links``), accumulating energy
at terminal nodes.  High-energy terminals are "supported by evidence"
from the seed set and receive a retrieval boost.

Origin story (2026-04-11):
  Max asked "why are my RT cores sitting at 0%?" while staring at the
  GPU performance panel during a helix bench run.  The chain that
  followed:

    1. RT cores do hardware-accelerated ray-trace against BVH structures
    2. Could we repurpose that for something other than triangles?
    3. Monte Carlo ray-tracing is the CPU fallback for when you don't
       have RTX hardware
    4. Monte Carlo over a graph IS evidence propagation
    5. Which is what ScoreRift's ``cast_ray`` already does on compliance
       dimensions
    6. Which maps onto the gene co-activation graph...
    7. → this module exists

  The current implementation is pure Python on CPU.  The RT cores are
  still sitting at 0%.  The real version would encode the co-activation
  graph as a BVH and cast OptiX rays through it — ~1000x throughput,
  hardware-accelerated nearest-neighbour in high-dim SEMA space, and
  actual physical justification for the ray-trace framing.  That's a
  rabbit hole for another night.

Design decisions:
  - Adjacency built from 2-hop neighbourhood of seeds (keeps graph local)
  - harmonic_links weight multiplied when available; else neutral (1.0)
  - Boost normalised to [0, 2.0] for safe addition to query_genes() scores
  - Reproducible via ``random.Random(seed)``
"""

from __future__ import annotations

import json
import logging
import random
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .genome import Genome

__all__ = [
    "cast_evidence_rays",
    "ray_trace_boost",
    "ray_trace_info",
    "read_overtone_series",
    "harmonic_bin_boost",
]

log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

ABSORPTION_THRESHOLD = 0.01
DEFAULT_K_RAYS = 200
DEFAULT_MAX_BOUNCES = 3
DEFAULT_DECAY = 0.7
BOOST_CAP = 2.0


# ── Helpers ─────────────────────────────────────────────────────────────

def _load_co_activated(genome: "Genome", gene_id: str) -> List[str]:
    """Read co_activated_with list for a single gene from the DB."""
    cur = genome.conn.cursor()
    row = cur.execute(
        "SELECT epigenetics FROM genes WHERE gene_id = ?", (gene_id,)
    ).fetchone()
    if row is None:
        return []
    try:
        epi = json.loads(row["epigenetics"])
        return epi.get("co_activated_with", [])
    except Exception:
        log.warning("Failed to parse epigenetics for %s", gene_id)
        return []


def _build_adjacency(
    genome: "Genome",
    seed_gene_ids: List[str],
) -> Dict[str, List[str]]:
    """Build adjacency dict from co_activated_with, 2 hops from seeds."""
    adjacency: Dict[str, List[str]] = {}
    visited: set = set()

    # Hop 0: seeds themselves
    frontier = list(seed_gene_ids)

    for _hop in range(2):
        next_frontier: List[str] = []
        for gid in frontier:
            if gid in visited:
                continue
            visited.add(gid)
            neighbors = _load_co_activated(genome, gid)
            adjacency[gid] = neighbors
            next_frontier.extend(neighbors)
        frontier = next_frontier

    return adjacency


def _load_harmonic_weights(
    genome: "Genome",
    gene_ids: set,
) -> Dict[tuple, float]:
    """Load harmonic_links weights for all gene pairs in the neighbourhood."""
    weights: Dict[tuple, float] = {}
    cur = genome.conn.cursor()

    # Check if harmonic_links table exists
    has_table = cur.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type='table' AND name='harmonic_links'"
    ).fetchone()[0]
    if not has_table:
        return weights

    # Load all relevant edges in one query
    if not gene_ids:
        return weights
    placeholders = ",".join("?" * len(gene_ids))
    rows = cur.execute(
        f"SELECT gene_id_a, gene_id_b, weight FROM harmonic_links "
        f"WHERE gene_id_a IN ({placeholders}) AND gene_id_b IN ({placeholders})",
        (*gene_ids, *gene_ids),
    ).fetchall()
    for r in rows:
        weights[(r["gene_id_a"], r["gene_id_b"])] = r["weight"]
    return weights


# ── Core Algorithm ──────────────────────────────────────────────────────

def cast_evidence_rays(
    seed_gene_ids: List[str],
    genome: "Genome",
    k_rays: int = DEFAULT_K_RAYS,
    max_bounces: int = DEFAULT_MAX_BOUNCES,
    decay_per_bounce: float = DEFAULT_DECAY,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    """
    Cast Monte Carlo rays from seed genes through co-activation graph.

    Returns {gene_id: accumulated_energy} for all genes rays landed on.
    Higher energy = more evidence support from the seed set.

    Args:
        seed_gene_ids: Starting gene IDs to cast rays from.
        genome: Genome instance (uses genome.conn for DB reads).
        k_rays: Total number of rays to cast (distributed across seeds).
        max_bounces: Maximum hops per ray before forced deposit.
        decay_per_bounce: Energy multiplier at each bounce.
        seed: RNG seed for reproducibility (None for stochastic).

    Returns:
        Dict mapping gene_id to accumulated energy.
    """
    if not seed_gene_ids:
        return {}

    rng = random.Random(seed)

    # Build local graph (2 hops from seeds)
    adjacency = _build_adjacency(genome, seed_gene_ids)

    # Collect all gene_ids in the neighbourhood for harmonic weight lookup
    all_gene_ids: set = set()
    for gid, neighbors in adjacency.items():
        all_gene_ids.add(gid)
        all_gene_ids.update(neighbors)

    harmonic = _load_harmonic_weights(genome, all_gene_ids)

    # Accumulator
    energy_acc: Dict[str, float] = {}

    # Distribute rays across seeds
    for ray_idx in range(k_rays):
        # Pick a random seed to start from
        start = seed_gene_ids[ray_idx % len(seed_gene_ids)]
        energy = 1.0
        current = start

        for _bounce in range(max_bounces):
            neighbors = adjacency.get(current, [])
            if not neighbors:
                break  # dead-end — deposit at current

            next_gene = rng.choice(neighbors)

            # Apply harmonic weight if available
            hw = harmonic.get((current, next_gene))
            if hw is None:
                hw = harmonic.get((next_gene, current))
            if hw is not None:
                energy *= hw

            # Decay
            energy *= decay_per_bounce

            if energy < ABSORPTION_THRESHOLD:
                current = next_gene
                break

            current = next_gene

        # Deposit remaining energy at terminal node
        energy_acc[current] = energy_acc.get(current, 0.0) + energy

    return energy_acc


def ray_trace_boost(
    seed_gene_ids: List[str],
    genome: "Genome",
    k_rays: int = DEFAULT_K_RAYS,
    max_bounces: int = DEFAULT_MAX_BOUNCES,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    """
    Compute retrieval boost for genes connected to seeds via evidence rays.

    Returns {gene_id: boost} where boost is normalised to [0, 2.0].
    Intended as a Tier 6 addition to query_genes() scoring.

    Args:
        seed_gene_ids: Gene IDs from which to propagate evidence.
        genome: Genome instance.
        k_rays: Total number of rays.
        max_bounces: Max hops per ray.
        seed: RNG seed for reproducibility.

    Returns:
        Dict mapping gene_id to capped boost value in [0, 2.0].
    """
    raw = cast_evidence_rays(
        seed_gene_ids, genome,
        k_rays=k_rays, max_bounces=max_bounces, seed=seed,
    )
    if not raw:
        return {}

    max_energy = max(raw.values())
    if max_energy <= 0:
        return {}

    # Normalise to [0, BOOST_CAP]
    return {
        gid: min(BOOST_CAP, (energy / max_energy) * BOOST_CAP)
        for gid, energy in raw.items()
    }


# ── Harmonic bin reading (overtone series interpretation) ───────────────

def read_overtone_series(
    seed_gene_ids: List[str],
    genome: "Genome",
    k_rays: int = DEFAULT_K_RAYS,
    max_bounces: int = DEFAULT_MAX_BOUNCES,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    """
    Read Monte Carlo rays as a FREQUENCY distribution, not an energy rank.

    Cymatics insight: the Chladni plate doesn't run a tournament — it
    applies one frequency and the sand finds nodes simultaneously at
    every scale. A gene appearing in k rays' paths IS the antinode.

    Returns {gene_id: overtone_weight} where:
      - Fundamental (gene appears in >= 70% of rays' paths): weight 1.0
      - First harmonic (appears in 40-70%): weight 0.5
      - Second harmonic (appears in 20-40%): weight 0.25
      - Noise (< 20%): weight 0.0 (excluded)

    The fundamental IS the expression candidate. Harmonics are candidate
    support. This reframes ranked cutoffs as resonance reading.
    """
    if not seed_gene_ids:
        return {}

    rng = random.Random(seed)
    adjacency = _build_adjacency(genome, seed_gene_ids)

    # Track: for each ray, which genes it visited
    visit_count: Dict[str, int] = {}

    for ray_idx in range(k_rays):
        start = seed_gene_ids[ray_idx % len(seed_gene_ids)]
        current = start
        visited: set = {current}

        for _bounce in range(max_bounces):
            neighbors = adjacency.get(current, [])
            if not neighbors:
                break
            current = rng.choice(neighbors)
            visited.add(current)

        # Count unique gene visits for this ray
        for gid in visited:
            visit_count[gid] = visit_count.get(gid, 0) + 1

    # Convert to overtone weights via harmonic bins
    overtones: Dict[str, float] = {}
    for gid, count in visit_count.items():
        frequency = count / k_rays
        if frequency >= 0.70:
            overtones[gid] = 1.0       # Fundamental
        elif frequency >= 0.40:
            overtones[gid] = 0.5       # First harmonic
        elif frequency >= 0.20:
            overtones[gid] = 0.25      # Second harmonic
        # else: noise, excluded

    return overtones


def harmonic_bin_boost(
    seed_gene_ids: List[str],
    genome: "Genome",
    k_rays: int = DEFAULT_K_RAYS,
    max_bounces: int = DEFAULT_MAX_BOUNCES,
) -> Dict[str, float]:
    """
    Return harmonic bin boost for retrieval scoring.

    Reads the overtone series and returns normalized [0, 1.5] weights
    for use as a retrieval score addition. Fundamentals get the full
    boost; harmonics get proportional amounts.
    """
    overtones = read_overtone_series(seed_gene_ids, genome, k_rays, max_bounces)
    # Scale weights to [0, 1.5] (fundamental=1.5, 1st harmonic=0.75, etc.)
    return {gid: w * 1.5 for gid, w in overtones.items()}


# ── Diagnostics ─────────────────────────────────────────────────────────

def ray_trace_info(result: Dict[str, float]) -> Dict:
    """Summary stats: total energy, unique genes reached, max/mean energy."""
    if not result:
        return {
            "total_energy": 0.0,
            "unique_genes_reached": 0,
            "max_energy": 0.0,
            "mean_energy": 0.0,
        }

    energies = list(result.values())
    return {
        "total_energy": sum(energies),
        "unique_genes_reached": len(energies),
        "max_energy": max(energies),
        "mean_energy": sum(energies) / len(energies),
    }

"""
TCM -- Temporal Context Model (Howard & Kahana, 2002).

Session-level drift vector for retrieval scoring.  The key equation:

    t_i = rho_i * t_{i-1} + beta * t^IN_i

Where:
    t_i     -- context vector after item i
    rho_i   -- normalization constant ensuring ||t_i|| = 1
    beta    -- context integration rate (default 0.5)
    t^IN_i  -- input representation of the accessed gene

Key property: **forward-recall asymmetry** -- queries about early items
preferentially surface later items from the same session, because the
context vector has drifted closer to recent items.

This module is a *tiebreaker*, not a primary relevance signal.

Numpy is optional -- pure-Python fallback is always available (follows
the cymatics.py pattern).
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Dict, List, Optional, Tuple

from .schemas import Gene

log = logging.getLogger("helix.tcm")

# -- Numpy detection (mirrors cymatics.py) ---------------------

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

MATH_BACKEND = "numpy" if _HAS_NUMPY else "python"

# -- Constants --------------------------------------------------

N_DIMS = 20          # Match SEMA dimensionality
DEFAULT_BETA = 0.5   # Context integration rate
BONUS_WEIGHT = 0.3   # TCM bonus scaling -- tiebreaker only

_HASH_SEED = b"helix-tcm"  # Deterministic seed for tag->dimension hashing


# -- Vector math helpers ----------------------------------------

def _norm(v: List[float]) -> float:
    """Euclidean norm of a vector."""
    if _HAS_NUMPY:
        return float(np.linalg.norm(v))
    return math.sqrt(sum(x * x for x in v))


def _scale(v: List[float], s: float) -> List[float]:
    """Scale vector by scalar."""
    if _HAS_NUMPY:
        return (np.array(v, dtype=np.float64) * s).tolist()
    return [x * s for x in v]


def _add(a: List[float], b: List[float]) -> List[float]:
    """Element-wise addition."""
    if _HAS_NUMPY:
        return (np.array(a, dtype=np.float64) + np.array(b, dtype=np.float64)).tolist()
    return [x + y for x, y in zip(a, b)]


def _normalize(v: List[float]) -> List[float]:
    """Return unit vector, or zero vector if norm is ~0."""
    n = _norm(v)
    if n < 1e-12:
        return [0.0] * len(v)
    return _scale(v, 1.0 / n)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    if _HAS_NUMPY:
        va = np.array(a, dtype=np.float64)
        vb = np.array(b, dtype=np.float64)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


# -- Gene -> 20D input vector ----------------------------------

def gene_input_vector(gene: Gene) -> List[float]:
    """Derive a 20D input representation for a gene.

    Strategy:
        1. If gene.embedding exists and is 20D, use it directly (SEMA).
        2. Otherwise, build a deterministic 20D vector from promoter tags:
           hash each tag to a dimension index (mod 20), accumulate weights.
           Then normalize to unit length.
    """
    # Fast path: use SEMA embedding if available and correct dimension
    if gene.embedding is not None and len(gene.embedding) == N_DIMS:
        return _normalize(list(gene.embedding))

    # Fallback: hash promoter tags into dimensions
    tags = list(gene.promoter.domains) + list(gene.promoter.entities)
    if not tags:
        # Last resort: use codons
        tags = list(gene.codons[:5])

    if not tags:
        return [0.0] * N_DIMS

    vec = [0.0] * N_DIMS
    for tag in tags:
        h = hashlib.md5(_HASH_SEED + tag.lower().encode("utf-8")).digest()
        dim = int.from_bytes(h[:2], "little") % N_DIMS
        # Use bytes 2-4 to get a weight in [0.5, 1.5]
        weight = 0.5 + (int.from_bytes(h[2:4], "little") % 1000) / 1000.0
        vec[dim] += weight

    return _normalize(vec)


# -- SessionContext ---------------------------------------------

class SessionContext:
    """Temporal Context Model state for a single session.

    Tracks the evolving context vector as genes are accessed during a
    session.  The context drifts toward recently accessed items, producing
    the forward-recall asymmetry described by Howard & Kahana (2002).
    """

    def __init__(self, n_dims: int = N_DIMS, beta: float = DEFAULT_BETA):
        if not (0.0 < beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        self.n_dims = n_dims
        self.beta = beta
        self.context_vector: List[float] = [0.0] * n_dims
        self.item_history: List[Tuple[str, List[float]]] = []
        # Howard 2005 Eq. 16 velocity input: the next update subtracts
        # this from the new raw input to produce t^IN. None on first
        # call in a session -> fall back to raw input (no predecessor).
        self.prev_raw_input: Optional[List[float]] = None

    def update(self, gene_id: str, input_vector: List[float]) -> None:
        """Integrate a new item into the context vector (TCM evolution).

        t_i = rho_i * t_{i-1} + beta * t^IN_i

        where t^IN_i = gene_input_vector(g_i) - gene_input_vector(g_{i-1})
        per Howard 2005 Eq. 16 (velocity input). On the first call in a
        session there is no predecessor -- t^IN falls back to the raw
        input. t^IN is then Gram-Schmidt projected onto the subspace
        orthogonal to t_{i-1}, restoring the orthogonality assumption
        behind the closed-form rho (Howard & Kahana 2002 sect 3).

        Pre-2026-04-13 behaviour treated raw input as t^IN and silently
        absorbed the orthogonality violation via a final _normalize()
        safety net. That masked both the velocity signal and the rho
        divergence. Both are fixed here.
        """
        if len(input_vector) != self.n_dims:
            raise ValueError(
                f"input_vector has {len(input_vector)} dims, expected {self.n_dims}"
            )

        raw_input = list(input_vector)

        # Howard 2005 Eq. 16: t^IN is the velocity delta from the
        # previous raw input, not the raw input itself.
        if self.prev_raw_input is not None:
            delta = [a - b for a, b in zip(raw_input, self.prev_raw_input)]
            t_in = _normalize(delta)
        else:
            t_in = _normalize(raw_input)
        t_in_norm = _norm(t_in)

        # First item: context becomes the input directly.
        if not self.item_history:
            self.context_vector = list(t_in)
            self.item_history.append((gene_id, list(t_in)))
            self.prev_raw_input = raw_input
            return

        t_prev = self.context_vector
        t_prev_norm = _norm(t_prev)

        # Edge: previous context is zero -> seed with t^IN.
        if t_prev_norm < 1e-12:
            self.context_vector = list(t_in)
            self.item_history.append((gene_id, list(t_in)))
            self.prev_raw_input = raw_input
            return

        # Edge: identical successive inputs -> zero velocity, no update.
        if t_in_norm < 1e-12:
            self.item_history.append((gene_id, list(t_in)))
            self.prev_raw_input = raw_input
            return

        # Gram-Schmidt: project t^IN onto the subspace orthogonal to
        # t_{i-1}. Matches Howard 2005's perpendicular-projection step
        # and restores the orthogonality baked into the closed-form rho.
        dot_tp = sum(a * b for a, b in zip(t_in, t_prev))
        proj = dot_tp / (t_prev_norm * t_prev_norm)
        t_in_perp = [a - proj * b for a, b in zip(t_in, t_prev)]
        t_in = _normalize(t_in_perp)
        t_in_norm = _norm(t_in)

        # Edge: velocity was parallel to current context -> no new
        # direction to integrate, context is unchanged.
        if t_in_norm < 1e-12:
            self.item_history.append((gene_id, [0.0] * self.n_dims))
            self.prev_raw_input = raw_input
            return

        # rho = sqrt(1 - beta^2 * ||t^IN||^2) / ||t_{i-1}||  (orthogonal case)
        beta_sq_tin_sq = self.beta * self.beta * t_in_norm * t_in_norm
        rho_arg = max(0.0, 1.0 - beta_sq_tin_sq)
        rho = math.sqrt(rho_arg) / t_prev_norm

        new_ctx = _add(_scale(t_prev, rho), _scale(t_in, self.beta))

        # After orthogonal update + Gram-Schmidt, ||new_ctx|| should be
        # ~1 without further rescaling. Log a warning if it drifts -
        # surfaces numerical issues instead of masking them.
        new_norm = _norm(new_ctx)
        if abs(new_norm - 1.0) > 1e-6:
            log.warning("TCM ctx norm drift after orthogonal update: %.9f", new_norm)
        self.context_vector = _normalize(new_ctx)
        self.item_history.append((gene_id, list(t_in)))
        self.prev_raw_input = raw_input

    def update_from_gene(self, gene: Gene) -> None:
        """Convenience: derive input vector from a Gene and update."""
        vec = gene_input_vector(gene)
        self.update(gene.gene_id, vec)

    def context_similarity(self, candidate_vector: List[float]) -> float:
        """Cosine similarity between current context and a candidate vector."""
        if len(candidate_vector) != self.n_dims:
            return 0.0
        return _cosine_similarity(self.context_vector, candidate_vector)

    def reset(self) -> None:
        """Reset context state (new session)."""
        self.context_vector = [0.0] * self.n_dims
        self.item_history.clear()
        self.prev_raw_input = None

    @property
    def depth(self) -> int:
        """Number of items integrated into context so far."""
        return len(self.item_history)


# -- Retrieval bonus --------------------------------------------

def tcm_bonus(
    session: SessionContext,
    candidates: List[Gene],
    weight: float = BONUS_WEIGHT,
) -> Dict[str, float]:
    """Return {gene_id: bonus} for candidates based on TCM context similarity.

    Bonus = weight * context_similarity(gene_vector).
    This is a TIEBREAKER, not a primary signal.  Default weight is 0.3.

    Returns 0.0 for genes with no computable input vector, and for
    sessions with no history (empty context).
    """
    if session.depth == 0:
        return {g.gene_id: 0.0 for g in candidates}

    bonuses: Dict[str, float] = {}
    for gene in candidates:
        vec = gene_input_vector(gene)
        sim = session.context_similarity(vec)
        # Clamp to [0, weight] -- negative similarity means anti-correlation,
        # which should not penalize (leave at 0).
        bonuses[gene.gene_id] = max(0.0, weight * sim)

    return bonuses


# -- Diagnostics ------------------------------------------------

def tcm_info(session: SessionContext) -> Dict:
    """Report TCM state for debugging."""
    ctx_norm = _norm(session.context_vector)
    return {
        "math_backend": MATH_BACKEND,
        "n_dims": session.n_dims,
        "beta": session.beta,
        "depth": session.depth,
        "context_norm": round(ctx_norm, 6),
        "item_ids": [gid for gid, _ in session.item_history],
        "bonus_weight": BONUS_WEIGHT,
    }

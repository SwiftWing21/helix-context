"""Tests for helix_context.tcm -- Temporal Context Model."""

import math
import pytest

from helix_context.schemas import Gene, PromoterTags
from helix_context.tcm import (
    SessionContext,
    gene_input_vector,
    tcm_bonus,
    tcm_info,
    _norm,
    _normalize,
    _cosine_similarity,
    N_DIMS,
)


# -- Helpers ----------------------------------------------------

def _make_gene(gene_id: str, domains=None, entities=None, codons=None, embedding=None):
    """Create a minimal Gene for testing."""
    return Gene(
        gene_id=gene_id,
        content=f"content for {gene_id}",
        complement=f"complement for {gene_id}",
        codons=codons or ["codon1", "codon2"],
        promoter=PromoterTags(
            domains=domains or [],
            entities=entities or [],
        ),
        embedding=embedding,
    )


def _orthogonal_vectors(n_dims: int, count: int):
    """Generate `count` orthogonal unit vectors in n_dims space.

    Uses standard basis vectors (e_0, e_1, ...).
    """
    vecs = []
    for i in range(count):
        v = [0.0] * n_dims
        v[i % n_dims] = 1.0
        vecs.append(v)
    return vecs


# -- Vector math tests -----------------------------------------

class TestVectorMath:
    def test_norm_unit_vector(self):
        v = [0.0] * N_DIMS
        v[0] = 1.0
        assert abs(_norm(v) - 1.0) < 1e-10

    def test_norm_zero_vector(self):
        v = [0.0] * N_DIMS
        assert _norm(v) == 0.0

    def test_normalize_preserves_direction(self):
        v = [3.0, 4.0] + [0.0] * (N_DIMS - 2)
        nv = _normalize(v)
        assert abs(_norm(nv) - 1.0) < 1e-10
        # Direction preserved: nv[0]/nv[1] == 3/4
        assert abs(nv[0] / nv[1] - 0.75) < 1e-10

    def test_normalize_zero_vector(self):
        v = [0.0] * N_DIMS
        nv = _normalize(v)
        assert all(x == 0.0 for x in nv)

    def test_cosine_similarity_identical(self):
        v = [1.0, 2.0, 3.0] + [0.0] * (N_DIMS - 3)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal(self):
        a = [1.0] + [0.0] * (N_DIMS - 1)
        b = [0.0, 1.0] + [0.0] * (N_DIMS - 2)
        assert abs(_cosine_similarity(a, b)) < 1e-10

    def test_cosine_similarity_zero_vector(self):
        v = [1.0] + [0.0] * (N_DIMS - 1)
        z = [0.0] * N_DIMS
        assert _cosine_similarity(v, z) == 0.0
        assert _cosine_similarity(z, v) == 0.0


# -- gene_input_vector tests -----------------------------------

class TestGeneInputVector:
    def test_uses_embedding_when_available(self):
        emb = [0.0] * N_DIMS
        emb[0] = 1.0
        gene = _make_gene("g1", embedding=emb)
        vec = gene_input_vector(gene)
        assert len(vec) == N_DIMS
        assert abs(_norm(vec) - 1.0) < 1e-10
        # Should be the same direction as the embedding
        assert abs(vec[0] - 1.0) < 1e-10

    def test_falls_back_to_promoter_tags(self):
        gene = _make_gene("g1", domains=["python", "flask"], entities=["API"])
        vec = gene_input_vector(gene)
        assert len(vec) == N_DIMS
        assert abs(_norm(vec) - 1.0) < 1e-10

    def test_falls_back_to_codons(self):
        gene = _make_gene("g1", codons=["function", "class", "module"])
        vec = gene_input_vector(gene)
        assert len(vec) == N_DIMS
        assert abs(_norm(vec) - 1.0) < 1e-10

    def test_empty_gene_returns_zero(self):
        gene = Gene(
            gene_id="g1",
            content="empty",
            complement="empty",
            codons=[],
            promoter=PromoterTags(),
            embedding=None,
        )
        vec = gene_input_vector(gene)
        assert len(vec) == N_DIMS
        assert all(x == 0.0 for x in vec)

    def test_deterministic(self):
        gene = _make_gene("g1", domains=["python", "flask"])
        v1 = gene_input_vector(gene)
        v2 = gene_input_vector(gene)
        assert v1 == v2

    def test_different_tags_different_vectors(self):
        g1 = _make_gene("g1", domains=["python", "flask"])
        g2 = _make_gene("g2", domains=["rust", "tokio"])
        v1 = gene_input_vector(g1)
        v2 = gene_input_vector(g2)
        # Not identical (could be similar by hash collision, but very unlikely)
        assert v1 != v2


# -- SessionContext tests ---------------------------------------

class TestSessionContext:
    def test_init_defaults(self):
        ctx = SessionContext()
        assert ctx.n_dims == N_DIMS
        assert ctx.beta == 0.5
        assert ctx.depth == 0
        assert all(x == 0.0 for x in ctx.context_vector)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            SessionContext(beta=0.0)
        with pytest.raises(ValueError):
            SessionContext(beta=-0.5)
        with pytest.raises(ValueError):
            SessionContext(beta=1.5)

    def test_first_update_sets_context(self):
        ctx = SessionContext()
        v = [0.0] * N_DIMS
        v[0] = 1.0
        ctx.update("g1", v)
        assert ctx.depth == 1
        assert abs(ctx.context_vector[0] - 1.0) < 1e-10

    def test_context_norm_stays_near_one(self):
        """After multiple updates, ||context|| should remain ~1.0."""
        ctx = SessionContext(beta=0.5)
        vecs = _orthogonal_vectors(N_DIMS, 5)
        for i, v in enumerate(vecs):
            ctx.update(f"g{i}", v)
            norm = _norm(ctx.context_vector)
            assert abs(norm - 1.0) < 1e-6, f"norm={norm} after update {i}"

    def test_context_norm_with_non_orthogonal_vectors(self):
        """Norm stays ~1.0 even with correlated inputs."""
        ctx = SessionContext(beta=0.5)
        # All vectors point mostly in dim 0 with slight variations
        for i in range(10):
            v = [0.0] * N_DIMS
            v[0] = 1.0
            v[i % N_DIMS] += 0.3
            ctx.update(f"g{i}", v)
            norm = _norm(ctx.context_vector)
            assert abs(norm - 1.0) < 1e-6, f"norm={norm} after update {i}"

    def test_forward_recall_asymmetry(self):
        """After updating with A, B, C, D, E in order, context is more similar
        to E than to A.

        This is the core TCM property: the context vector drifts toward
        the most recently accessed items.  With orthogonal inputs and
        beta=0.8, the drift is strong enough to see the asymmetry within
        5 items.
        """
        ctx = SessionContext(beta=0.8)
        vecs = _orthogonal_vectors(N_DIMS, 5)

        for i, v in enumerate(vecs):
            ctx.update(f"g{i}", v)

        sim_first = ctx.context_similarity(vecs[0])
        sim_last = ctx.context_similarity(vecs[4])

        # Last item should be more similar to context than first
        assert sim_last > sim_first, (
            f"sim_last={sim_last} should be > sim_first={sim_first}"
        )

    def test_forward_recall_with_many_items(self):
        """Even with 10 items, last item has higher similarity than first."""
        ctx = SessionContext(beta=0.5)
        vecs = _orthogonal_vectors(N_DIMS, 10)

        for i, v in enumerate(vecs):
            ctx.update(f"g{i}", v)

        sim_first = ctx.context_similarity(vecs[0])
        sim_last = ctx.context_similarity(vecs[9])

        assert sim_last > sim_first

    def test_dimension_mismatch_raises(self):
        ctx = SessionContext(n_dims=20)
        with pytest.raises(ValueError):
            ctx.update("g1", [1.0] * 10)

    def test_zero_vector_input_no_crash(self):
        """Updating with a zero vector should not crash or change context."""
        ctx = SessionContext()
        v = [0.0] * N_DIMS
        v[0] = 1.0
        ctx.update("g1", v)

        old_ctx = list(ctx.context_vector)
        ctx.update("g2", [0.0] * N_DIMS)

        assert ctx.depth == 2
        # Context should not have changed
        for a, b in zip(ctx.context_vector, old_ctx):
            assert abs(a - b) < 1e-10

    def test_reset(self):
        ctx = SessionContext()
        v = [0.0] * N_DIMS
        v[0] = 1.0
        ctx.update("g1", v)
        assert ctx.depth == 1

        ctx.reset()
        assert ctx.depth == 0
        assert all(x == 0.0 for x in ctx.context_vector)
        assert ctx.item_history == []

    def test_update_from_gene(self):
        ctx = SessionContext()
        gene = _make_gene("g1", domains=["python", "flask"], entities=["API"])
        ctx.update_from_gene(gene)
        assert ctx.depth == 1
        assert abs(_norm(ctx.context_vector) - 1.0) < 1e-6

    def test_context_similarity_wrong_dims(self):
        ctx = SessionContext(n_dims=20)
        v = [0.0] * 20
        v[0] = 1.0
        ctx.update("g1", v)
        # Wrong dimension should return 0.0
        assert ctx.context_similarity([1.0, 2.0]) == 0.0

    def test_beta_one_replaces_context(self):
        """With beta=1.0, new input should dominate (rho approaches 0)."""
        ctx = SessionContext(beta=1.0)
        v1 = [0.0] * N_DIMS
        v1[0] = 1.0
        v2 = [0.0] * N_DIMS
        v2[1] = 1.0

        ctx.update("g1", v1)
        ctx.update("g2", v2)

        # With beta=1, context should be very close to v2
        sim_v2 = ctx.context_similarity(v2)
        assert sim_v2 > 0.95, f"sim_v2={sim_v2}, expected near 1.0"


# -- tcm_bonus tests -------------------------------------------

class TestTcmBonus:
    def test_empty_session_returns_zeros(self):
        ctx = SessionContext()
        genes = [_make_gene("g1", domains=["python"]), _make_gene("g2", domains=["rust"])]
        bonuses = tcm_bonus(ctx, genes)
        assert bonuses == {"g1": 0.0, "g2": 0.0}

    def test_returns_dict_with_gene_ids(self):
        ctx = SessionContext()
        v = [0.0] * N_DIMS
        v[0] = 1.0
        ctx.update("seed", v)

        genes = [_make_gene("g1", domains=["python"]), _make_gene("g2", domains=["rust"])]
        bonuses = tcm_bonus(ctx, genes)

        assert isinstance(bonuses, dict)
        assert "g1" in bonuses
        assert "g2" in bonuses
        assert all(isinstance(v, float) for v in bonuses.values())

    def test_bonus_values_in_range(self):
        ctx = SessionContext()
        v = [0.0] * N_DIMS
        v[0] = 1.0
        ctx.update("seed", v)

        genes = [_make_gene(f"g{i}", domains=[f"domain{i}"]) for i in range(5)]
        bonuses = tcm_bonus(ctx, genes)

        for gene_id, bonus in bonuses.items():
            assert 0.0 <= bonus <= 0.3, f"bonus for {gene_id} = {bonus}"

    def test_recently_accessed_gets_higher_bonus(self):
        """Gene similar to recently accessed items should get higher bonus."""
        ctx = SessionContext()

        # Create genes with known embeddings
        emb_a = [0.0] * N_DIMS
        emb_a[0] = 1.0
        emb_b = [0.0] * N_DIMS
        emb_b[1] = 1.0

        gene_a = _make_gene("a", embedding=emb_a)
        gene_b = _make_gene("b", embedding=emb_b)

        # Update context with gene_b's vector
        ctx.update("seed", emb_b)

        bonuses = tcm_bonus(ctx, [gene_a, gene_b])
        # gene_b should get higher bonus since its vector matches context
        assert bonuses["b"] > bonuses["a"]

    def test_custom_weight(self):
        ctx = SessionContext()
        v = [0.0] * N_DIMS
        v[0] = 1.0
        ctx.update("seed", v)

        gene = _make_gene("g1", embedding=list(v))
        bonuses_default = tcm_bonus(ctx, [gene])
        bonuses_half = tcm_bonus(ctx, [gene], weight=0.15)

        assert abs(bonuses_half["g1"] - bonuses_default["g1"] / 2.0) < 1e-6

    def test_empty_candidates(self):
        ctx = SessionContext()
        v = [0.0] * N_DIMS
        v[0] = 1.0
        ctx.update("seed", v)
        bonuses = tcm_bonus(ctx, [])
        assert bonuses == {}


# -- tcm_info tests ---------------------------------------------

class TestTcmInfo:
    def test_info_empty_session(self):
        ctx = SessionContext()
        info = tcm_info(ctx)
        assert info["depth"] == 0
        assert info["context_norm"] == 0.0
        assert info["item_ids"] == []
        assert info["beta"] == 0.5
        assert info["n_dims"] == N_DIMS

    def test_info_after_updates(self):
        ctx = SessionContext()
        v = [0.0] * N_DIMS
        v[0] = 1.0
        ctx.update("g1", v)
        ctx.update("g2", v)

        info = tcm_info(ctx)
        assert info["depth"] == 2
        assert abs(info["context_norm"] - 1.0) < 1e-6
        assert info["item_ids"] == ["g1", "g2"]

    def test_info_has_math_backend(self):
        ctx = SessionContext()
        info = tcm_info(ctx)
        assert info["math_backend"] in ("numpy", "python")
        assert info["bonus_weight"] == 0.3

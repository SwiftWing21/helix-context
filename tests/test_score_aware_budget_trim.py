"""Tests for the score-aware budget-trimmer (#58).

Under sequence_index ordering (the default for narrative coherence on
factual/multi_hop queries without an answer_slate), the budget-overflow
trim used to drop ``parts[-1]`` — whichever gene happened to land last
in the assembled prompt. If the highest-scored gene was near the end
of the file, that's exactly the gene that got dropped.

This module pins the score-aware behavior with monotonicity assertions
that don't depend on exact token-count tuning: regardless of how many
genes the trim ends up dropping, the dropped set must be a prefix of
the score-ascending order. A higher-scored gene can never be dropped
while a lower-scored gene is still in the assembled context.

Gemini 3.1 Pro flagged this during the 2026-05-08 council session;
patch deferred to a follow-up because it conceptually sits outside
the Stage 5 caller_model_class branch routing. Spec lives in the
body of issue #58.
"""
from __future__ import annotations

import pytest

from cymatix_context.config import (
    BudgetConfig,
    ClassifierConfig,
    GenomeConfig,
    CymatixConfig,
    RibosomeConfig,
)
from cymatix_context.context_manager import CymatixContextManager
from cymatix_context.schemas import Gene, PromoterTags


@pytest.fixture
def manager_five_genes():
    """Five genes ordered 0..4 by sequence_index with scores [10, 1, 8, 2, 9].

    The score distribution is deliberately non-monotonic in
    sequence_index so a position-based ``parts.pop()`` trim drops the
    score=9 gene (highest score, last by sequence_index) while a
    score-aware trim drops the score=1 gene (lowest score, second by
    sequence_index).
    """
    cfg = CymatixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=12),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        classifier=ClassifierConfig(enabled=False),
    )
    mgr = CymatixContextManager(cfg)

    scores = [10.0, 1.0, 8.0, 2.0, 9.0]
    gene_ids = [f"gene_{i:02d}" for i in range(5)]
    genes = [
        Gene(
            gene_id=gid,
            content=f"{gid} content payload " * 20,
            complement=f"complement-{gid}",
            codons=["c"],
            promoter=PromoterTags(sequence_index=i),
        )
        for i, gid in enumerate(gene_ids)
    ]

    mgr.genome.last_query_scores = {gid: scores[i] for i, gid in enumerate(gene_ids)}

    yield mgr, genes, gene_ids, scores
    mgr.close()


def _spliced_map(gene_ids):
    """Each gene's spliced text is large enough that dropping one
    materially reduces the assembled-prompt token count."""
    return {
        gid: f"content for {gid}: " + ("payload " * 60)
        for gid in gene_ids
    }


def _survivors_by_score(window, gene_ids, scores):
    """Return ``(survivors, dropped)`` partitions sorted by score-DESC.

    Reads window.expressed_context (the actual emitted prompt) rather
    than window.expressed_gene_ids so we're testing the user-visible
    surface, not an internal bookkeeping field.
    """
    surviving = [
        (gid, scores[i])
        for i, gid in enumerate(gene_ids)
        if gid in window.expressed_context
    ]
    dropped = [
        (gid, scores[i])
        for i, gid in enumerate(gene_ids)
        if gid not in window.expressed_context
    ]
    return (
        sorted(surviving, key=lambda x: x[1], reverse=True),
        sorted(dropped, key=lambda x: x[1], reverse=True),
    )


class TestScoreAwareBudgetTrim:
    """Default branch (no respect_caller_order, no caller_model_class)
    sorts by sequence_index. Position-based trim would drop the wrong
    gene; score-aware trim must drop the lowest-scored regardless of
    its position."""

    @pytest.mark.parametrize("budget_expression,budget_ribosome", [
        (700, 100),    # very tight — most drops
        (800, 100),    # tight — many drops
        (900, 100),    # mid — a few drops
        (1200, 100),   # looser — few drops
        (1500, 100),   # comfortably above 5-gene baseline — possibly no drops
    ])
    def test_drops_monotone_in_score(
        self, manager_five_genes, budget_expression, budget_ribosome,
    ):
        """For ANY trim count k, the surviving set is the top-(N-k) by score.

        This is the load-bearing invariant of score-aware trimming. It
        doesn't depend on exact token tuning — at every budget level
        the property must hold.
        """
        mgr, genes, gene_ids, scores = manager_five_genes

        mgr.config.budget.expression_tokens = budget_expression
        mgr.config.budget.ribosome_tokens = budget_ribosome

        window = mgr._assemble(
            query="q",
            candidates=genes,
            spliced_map=_spliced_map(gene_ids),
            relation_graph={},
            answer_slate=None,
        )

        survivors, dropped = _survivors_by_score(window, gene_ids, scores)

        if dropped and survivors:
            lowest_survivor_score = survivors[-1][1]
            highest_dropped_score = dropped[0][1]
            assert lowest_survivor_score >= highest_dropped_score, (
                f"Score-aware trim violated: a higher-score gene was "
                f"dropped while a lower-score gene survives. "
                f"Survivors (high→low): {survivors!r}. "
                f"Dropped (high→low): {dropped!r}. "
                f"Lowest survivor score ({lowest_survivor_score}) must "
                f"be >= highest dropped score ({highest_dropped_score})."
            )

    def test_lowest_score_dropped_first_under_overflow(
        self, manager_five_genes,
    ):
        """Spot check: at a budget tight enough to fire at least one
        drop, gene_01 (score=1, the global minimum) must be among the
        dropped, and gene_00 (score=10, the global maximum) must
        survive. Position-based pop() would drop gene_04 (last by
        sequence_index, score=9) — exactly backwards."""
        mgr, genes, gene_ids, scores = manager_five_genes

        # Exercise eviction rather than the delivered-seat truncation floor.
        mgr.config.budget.min_delivered_docs = 0
        mgr.config.budget.expression_tokens = 400
        mgr.config.budget.ribosome_tokens = 100

        window = mgr._assemble(
            query="q",
            candidates=genes,
            spliced_map=_spliced_map(gene_ids),
            relation_graph={},
            answer_slate=None,
        )

        survivors, dropped = _survivors_by_score(window, gene_ids, scores)

        assert dropped, "Fixture must exercise a real budget eviction"
        assert window.metadata["budget_evicted"] > 0
        assert window.metadata["budget_truncated"] == 0

        assert ("gene_01", 1.0) in dropped, (
            f"gene_01 (score=1) must be the first dropped under "
            f"score-aware trim. Got dropped={dropped!r}."
        )
        assert ("gene_00", 10.0) in survivors, (
            f"gene_00 (score=10) must survive while any other gene is "
            f"still in context. Got survivors={survivors!r}, "
            f"dropped={dropped!r}."
        )

    def test_foveated_branch_unchanged_under_respect_caller_order(
        self, manager_five_genes,
    ):
        """Foveated path uses position-based pop(0) by construction.

        Under respect_caller_order=True the caller has arranged the
        candidates in REVERSE-rank order so the top-rank gene lands
        LAST in the prompt. Position-0 == lowest-rank by that
        invariant, so positional pop(0) is correct and is the
        invariant this PR preserves.
        """
        mgr, genes, gene_ids, scores = manager_five_genes

        reverse_ranked = sorted(
            genes,
            key=lambda g: scores[gene_ids.index(g.gene_id)],
        )

        mgr.config.budget.expression_tokens = 800
        mgr.config.budget.ribosome_tokens = 100

        window = mgr._assemble(
            query="q",
            candidates=reverse_ranked,
            spliced_map=_spliced_map(gene_ids),
            relation_graph={},
            answer_slate=None,
            respect_caller_order=True,
        )

        assert "gene_00" in window.expressed_context, (
            "Foveated trim dropped the top-rank gene. Position-0 under "
            "reverse-rank IS the lowest-score gene by construction; "
            "pop(0) should preserve gene_00 (top score)."
        )

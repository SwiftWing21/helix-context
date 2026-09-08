"""Stage 7 — freshness gate tests.

Spec: docs/specs/2026-05-08-stage-7-freshness-gate.md §13.

All tests are mock-only — no Ollama, no real filesystem polling beyond
``tmp_path`` fixtures. Reuses Stage 6 fixture shapes from
``test_know_miss_block.py`` while keeping its own ``conftest`` so test
collection stays orthogonal.

Cases (one per spec bullet in §13):

  1. test_freshness_top1_dominates_padding              — headline
                                                          regression test
  2. test_compute_health_back_compat_freshness_field    — back-compat shim
  3. test_cold_tier_peek_emits_miss_cold                — cold-tier wiring
  4. test_supersession_downgrades_top1                  — Path A
  5. test_revalidate_caches_mtime_60s_ttl               — TTL contract
  6. test_read_only_does_not_write_last_verified_at     — read_only contract
  7. test_unknown_freshness_treated_as_neither_fresh_nor_stale
  8. test_soft_stale_know_block_recommends_refresh
  9. test_refresh_targets_required_for_stale_cold_superseded — validator

Plus integration-shape tests that confirm the public surface
(`MissBlock.to_refresh_targets`, `agent_prompt.CYMATIX_REFRESH_FRAGMENT`)
is wired.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from cymatix_context.agent_prompt import (
    CYMATIX_NO_MATCH_FRAGMENT,
    CYMATIX_REFRESH_FRAGMENT,
    full_fragment,
)
from cymatix_context.context_manager import CymatixContextManager
from cymatix_context.retrieval.freshness import (
    DEFAULT_CACHE_TTL_S,
    check_superseded,
    revalidate_and_mark,
    revalidate_source,
)
from cymatix_context.scoring.know_calibration import (
    KnowCalibration,
    compute_confidence,
)
from cymatix_context.scoring.know_decision import decide_know_or_miss
from cymatix_context.schemas import (
    ContextHealth,
    ContextWindow,
    EpigeneticMarkers,
    Gene,
    KnowBlock,
    MISS_REASONS,
    MissBlock,
    PromoterTags,
    RefreshTarget,
)


@pytest.mark.parametrize("missing", [False, True])
def test_mtime_cache_never_exceeds_cap(monkeypatch, missing):
    """Both successful stats and negative sentinels have bounded lifetime."""
    from cymatix_context.retrieval import freshness

    def stat(path):
        if missing:
            raise FileNotFoundError(path)
        return SimpleNamespace(st_mtime=10.0)

    monkeypatch.setattr(freshness, "os", SimpleNamespace(stat=stat))
    cap = getattr(freshness, "MTIME_CACHE_MAX_ENTRIES", 4096)
    cache = {}
    for i in range(cap + 100):
        gene = SimpleNamespace(source_id=f"/source/{i}", last_verified_at=20.0)
        status = revalidate_source(gene, mtime_cache=cache, now_ts=100.0)
        assert status == ("missing" if missing else "fresh")
        assert len(cache) <= cap


def test_mtime_cache_evicts_ttl_expired_before_clearing(monkeypatch):
    """A still-valid entry survives pressure when expired entries free space."""
    from cymatix_context.retrieval import freshness

    cap = getattr(freshness, "MTIME_CACHE_MAX_ENTRIES", 4096)
    cache = {f"/expired/{i}": (10.0, 0.0) for i in range(cap - 1)}
    cache["/keep"] = (10.0, 99.0)
    monkeypatch.setattr(
        freshness, "os", SimpleNamespace(stat=lambda path: SimpleNamespace(st_mtime=10.0)),
    )
    gene = SimpleNamespace(source_id="/new", last_verified_at=20.0)
    assert revalidate_source(gene, mtime_cache=cache, now_ts=100.0) == "fresh"
    assert cache == {"/keep": (10.0, 99.0), "/new": (10.0, 100.0)}


def test_admin_refresh_clears_mtime_cache_in_place(tmp_path):
    """Refresh makes an externally changed source observable before its TTL."""
    from tests.conftest import make_client

    src = tmp_path / "source.txt"
    src.write_text("before", encoding="utf-8")
    gene = SimpleNamespace(source_id=str(src), last_verified_at=100.0)
    client = make_client()
    manager = client.app.state.cymatix
    cache = manager._mtime_cache
    cache[str(src)] = (50.0, 200.0)
    try:
        assert revalidate_source(gene, mtime_cache=cache, now_ts=201.0) == "fresh"
        response = client.post("/admin/refresh")
        assert response.status_code == 200
        assert response.json()["refreshed"] is True
        assert manager._mtime_cache is cache
        assert cache == {}
        assert revalidate_source(gene, mtime_cache=cache, now_ts=201.0) == "stale"
    finally:
        client.close()
        manager.close()


# ─────────────────────────────────────────────────────────────────────
# Helpers — minimal Gene/Window builders so each test reads top-down
# ─────────────────────────────────────────────────────────────────────

def _make_gene(gene_id: str, *, decay: float, source_id: str | None = None) -> Gene:
    return Gene(
        gene_id=gene_id,
        content=f"content for {gene_id}",
        complement=f"complement for {gene_id}",
        codons=[gene_id],
        promoter=PromoterTags(),
        epigenetics=EpigeneticMarkers(created_at=0.0, decay_score=decay),
        source_id=source_id or f"/synthetic/{gene_id}.py",
    )


def _make_window(
    *,
    status: str,
    genes_expressed: int,
    freshness_min: float | None = None,
    freshness_top1: float | None = None,
    freshness_weighted: float | None = None,
    expressed_ids: list[str] | None = None,
) -> ContextWindow:
    return ContextWindow(
        ribosome_prompt="",
        expressed_context="(genes here)",
        expressed_gene_ids=expressed_ids or [],
        context_health=ContextHealth(
            ellipticity=0.9,
            coverage=0.7,
            density=0.8,
            freshness=freshness_weighted if freshness_weighted is not None else 1.0,
            genes_available=100,
            genes_expressed=genes_expressed,
            status=status,
            freshness_min=freshness_min,
            freshness_top1=freshness_top1,
            freshness_weighted=freshness_weighted,
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# §13 case 1 — headline regression: stale top-1 + 11 fresh padding
# ─────────────────────────────────────────────────────────────────────

def test_freshness_top1_dominates_padding(tmp_path):
    """Spec §13 case 1 (headline): the stale needle scenario.

    Top-1 has decay=0.1 (stale); 11 padding genes have decay=1.0
    (fresh). Under the legacy mean(decay) math, freshness ≈ 0.92
    → status="aligned" — the bug. Under the Stage 7 rewrite,
    freshness_top1 = 0.1 < 0.4 → status="stale".
    """
    # Build a fake context manager with a controlled stats() so
    # _compute_health doesn't depend on a real genome.
    stub = MagicMock(spec=CymatixContextManager)
    stub.config = MagicMock()
    stub.config.budget = MagicMock()
    stub.config.budget.expression_tokens = 6000
    stub.config.budget.max_genes_per_turn = 12
    stub.genome = MagicMock()
    stub.genome.stats = MagicMock(return_value={"total_genes": 100})
    stub.genome.last_query_scores = {
        "needle": 0.95,
        **{f"pad_{i}": 0.1 for i in range(11)},
    }
    stub.genome.last_tier_contributions = {}

    candidates = [_make_gene("needle", decay=0.1)] + [
        _make_gene(f"pad_{i}", decay=1.0) for i in range(11)
    ]

    health = CymatixContextManager._compute_health(
        stub,
        query_terms=["needle"],
        candidates=candidates,
        compressed_chars=2000,
    )

    # The headline assertion: stale top-1 cannot be masked.
    assert health.status == "stale", (
        f"expected status='stale' (top-1 decay=0.1), got "
        f"{health.status!r} with freshness_top1={health.freshness_top1}, "
        f"freshness_weighted={health.freshness_weighted}"
    )
    # Sanity: the three signals are populated and ordered as expected.
    assert health.freshness_top1 == pytest.approx(0.1, abs=1e-3)
    assert health.freshness_min == pytest.approx(0.1, abs=1e-3)
    # Score-weighted sum: needle has ~0.95/(0.95+11*0.1)=0.46 weight,
    # so freshness_weighted ≈ 0.46*0.1 + 0.54*1.0 ≈ 0.586.
    # Either trigger (top1<0.4 OR weighted<0.5) flips status to stale.
    assert health.freshness_weighted < 1.0


# ─────────────────────────────────────────────────────────────────────
# §13 case 2 — back-compat: health.freshness == freshness_weighted
# ─────────────────────────────────────────────────────────────────────

def test_compute_health_back_compat_freshness_field():
    """Spec §13 case 2: ``health.freshness`` becomes ``freshness_weighted``.

    Legacy callers reading ``health.freshness`` continue to get a
    meaningful number — and a stale top-1 pulls it down (vs the old
    mean(decay) where 11 padding genes could float it back up).
    """
    stub = MagicMock(spec=CymatixContextManager)
    stub.config = MagicMock()
    stub.config.budget = MagicMock()
    stub.config.budget.expression_tokens = 6000
    stub.config.budget.max_genes_per_turn = 12
    stub.genome = MagicMock()
    stub.genome.stats = MagicMock(return_value={"total_genes": 100})
    stub.genome.last_query_scores = {
        "needle": 0.95,
        **{f"pad_{i}": 0.05 for i in range(11)},
    }
    stub.genome.last_tier_contributions = {}

    candidates = [_make_gene("needle", decay=0.1)] + [
        _make_gene(f"pad_{i}", decay=1.0) for i in range(11)
    ]
    health = CymatixContextManager._compute_health(
        stub, query_terms=["needle"],
        candidates=candidates, compressed_chars=2000,
    )

    # back-compat shim: health.freshness now mirrors weighted signal.
    assert health.freshness == health.freshness_weighted
    # And it reflects the stale top-1 — well below the 0.5 floor.
    assert health.freshness_weighted < 0.5


# ─────────────────────────────────────────────────────────────────────
# §13 case 3 — cold-tier peek emits MissBlock(reason="cold")
# ─────────────────────────────────────────────────────────────────────

def test_cold_tier_peek_emits_miss_cold():
    """Spec §13 case 3: cold-tier hit translates to MissBlock(cold).

    Confidence below floor + cold-peek non-empty → reason="cold"
    instead of reason="sparse"; refresh_targets carries the cold gene
    source paths.
    """
    window = _make_window(status="sparse", genes_expressed=1)
    # Force confidence below floor by giving the discriminator a
    # weak signal (low score + tiny gap + no agreement).
    out = decide_know_or_miss(
        window=window,
        query="archived knowledge",
        top_score=0.1,
        score_gap=0.01,
        lexical_dense_agree=False,
        coordinate_confidence=0.0,
        top_gene=_make_gene("g1", decay=1.0),
        cold_refresh_targets=["/cold/source.md", "/cold/notes.md"],
    )
    assert isinstance(out, MissBlock)
    assert out.reason == "cold"
    assert out.refresh_targets == ["/cold/source.md", "/cold/notes.md"]
    assert out.escalate_to == []


def test_cold_tier_peek_does_not_fire_on_strong_signal():
    """Cold peek does NOT down-rank a strong KnowBlock — it only fires
    when the discriminator would otherwise emit MissBlock(sparse)."""
    window = _make_window(
        status="aligned",
        genes_expressed=4,
        freshness_min=0.95,
        freshness_top1=0.95,
        freshness_weighted=0.9,
    )
    out = decide_know_or_miss(
        window=window,
        query="strong query",
        top_score=0.95,
        score_gap=0.7,
        lexical_dense_agree=True,
        coordinate_confidence=0.8,
        top_gene=_make_gene("g1", decay=0.95),
        freshness_min=0.95,
        cold_refresh_targets=["/cold/should/not/fire.md"],
    )
    assert isinstance(out, KnowBlock)


# ─────────────────────────────────────────────────────────────────────
# §13 case 4 — supersession demotes top-1
# ─────────────────────────────────────────────────────────────────────

def test_supersession_downgrades_top1():
    """Spec §13 case 4: planted G2 supersedes=G1 demotes top-1 result.

    The decision branch fires before the confidence floor so even a
    high-score retrieval gets demoted when a successor exists.
    """
    window = _make_window(
        status="aligned",
        genes_expressed=2,
        freshness_min=0.95,
        freshness_top1=0.95,
        freshness_weighted=0.9,
    )
    top1 = _make_gene("g1", decay=0.95, source_id="/repo/old.py")
    out = decide_know_or_miss(
        window=window,
        query="some query",
        top_score=0.95,
        score_gap=0.7,
        lexical_dense_agree=True,
        coordinate_confidence=0.8,
        top_gene=top1,
        successor_source_id="/repo/new.py",
        freshness_min=0.95,
    )
    assert isinstance(out, MissBlock)
    assert out.reason == "superseded"
    assert out.refresh_targets == ["/repo/new.py"]


def test_check_superseded_path_a_query():
    """Path A reverse-lookup: SELECT FROM genes WHERE supersedes=?."""
    # MagicMock-backed genome with a row-shaped fetchone result.
    fake_row = MagicMock()
    fake_row.keys = MagicMock(return_value=["gene_id", "source_id"])
    fake_row.__getitem__ = lambda self, k: {
        "gene_id": "g_new",
        "source_id": "/repo/new.py",
    }[k]
    cur = MagicMock()
    cur.execute = MagicMock(return_value=cur)
    cur.fetchone = MagicMock(return_value=fake_row)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur)
    genome = MagicMock()
    genome.read_conn = conn
    genome.conn = conn

    gene = _make_gene("g_old", decay=1.0)
    out = check_superseded(genome, gene)
    assert out == "/repo/new.py"


def test_check_superseded_no_successor():
    """No row → returns None."""
    cur = MagicMock()
    cur.execute = MagicMock(return_value=cur)
    cur.fetchone = MagicMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur)
    genome = MagicMock()
    genome.read_conn = conn
    genome.conn = conn
    gene = _make_gene("g_old", decay=1.0)
    assert check_superseded(genome, gene) is None


# ─────────────────────────────────────────────────────────────────────
# §13 case 5 — TTL contract: cache hits within 60s
# ─────────────────────────────────────────────────────────────────────

def test_revalidate_caches_mtime_60s_ttl(tmp_path, monkeypatch):
    """Spec §13 case 5: two calls within 60s = one os.stat; +61s = two."""
    src = tmp_path / "file.txt"
    src.write_text("hello")
    real_stat = os.stat
    call_count = {"n": 0}

    def counting_stat(path, *args, **kwargs):
        call_count["n"] += 1
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)

    cache: dict[str, tuple[float, float]] = {}
    gene = _make_gene("g1", decay=1.0, source_id=str(src))
    gene.last_verified_at = real_stat(src).st_mtime + 100.0  # ahead of mtime → fresh

    t0 = 1_000_000.0
    s1 = revalidate_source(gene, mtime_cache=cache, now_ts=t0)
    s2 = revalidate_source(gene, mtime_cache=cache, now_ts=t0 + 30.0)
    s3 = revalidate_source(gene, mtime_cache=cache, now_ts=t0 + 61.0)

    assert s1 == "fresh"
    assert s2 == "fresh"
    assert s3 == "fresh"
    # Two stats: first call + the +61s call (cache expired).
    assert call_count["n"] == 2


# ─────────────────────────────────────────────────────────────────────
# §13 case 6 — read_only contract: no DB write
# ─────────────────────────────────────────────────────────────────────

def test_read_only_does_not_write_last_verified_at(tmp_path):
    """Spec §13 case 6: read_only=True passes the mtime check but the
    column is left unchanged (mark_verified is a no-op).
    """
    src = tmp_path / "file.txt"
    src.write_text("hello")

    genome = MagicMock()
    # mark_verified should NOT be called when read_only=True.
    genome.mark_verified = MagicMock()

    cache: dict[str, tuple[float, float]] = {}
    gene = _make_gene("g1", decay=1.0, source_id=str(src))
    gene.last_verified_at = os.stat(src).st_mtime + 100.0  # fresh

    status = revalidate_and_mark(
        genome, gene, mtime_cache=cache,
        now_ts=time.time(), read_only=True,
    )
    assert status == "fresh"
    genome.mark_verified.assert_not_called()

    # And under read_only=False, mark_verified IS called.
    status2 = revalidate_and_mark(
        genome, gene, mtime_cache=cache,
        now_ts=time.time(), read_only=False,
    )
    assert status2 == "fresh"
    genome.mark_verified.assert_called_once()
    args, kwargs = genome.mark_verified.call_args
    assert args[0] == ["g1"]  # gene_ids
    assert kwargs.get("read_only") is False


def test_genome_mark_verified_noop_under_read_only(tmp_path):
    """``Genome.mark_verified(read_only=True)`` is a silent no-op."""
    from cymatix_context.genome import Genome
    g = Genome(str(tmp_path / "test.db"))
    try:
        # No rows to update, but the call must not raise and must
        # return 0 under read_only=True.
        n = g.mark_verified(["nonexistent"], time.time(), read_only=True)
        assert n == 0
        n = g.mark_verified(["nonexistent"], time.time(), read_only=False)
        assert n == 0  # row doesn't exist; UPDATE matches zero rows
    finally:
        g.close()


# ─────────────────────────────────────────────────────────────────────
# §13 case 7 — unknown freshness is neutral
# ─────────────────────────────────────────────────────────────────────

def test_unknown_freshness_treated_as_neither_fresh_nor_stale():
    """Spec §13 case 7: ``last_verified_at IS NULL`` means freshness is
    "unknown". A strong score gap still emits KnowBlock; β5 contributes
    zero (neutral) rather than max-stale.
    """
    cache: dict[str, tuple[float, float]] = {}
    gene = _make_gene("g1", decay=0.9, source_id="/nonexistent/file")
    gene.last_verified_at = None  # legacy row
    # File doesn't exist at the synthetic path — should land on
    # "missing" not "unknown" actually. To get "unknown" we need
    # ``last_verified_at`` to be None AND the file to exist.
    # Use a real tmp file:


def test_unknown_freshness_keeps_know_block_emittable(tmp_path):
    src = tmp_path / "legacy.txt"
    src.write_text("legacy content")
    cache: dict[str, tuple[float, float]] = {}
    gene = _make_gene("g_legacy", decay=0.9, source_id=str(src))
    gene.last_verified_at = None
    status = revalidate_source(gene, mtime_cache=cache, now_ts=time.time())
    assert status == "unknown"

    # And the discriminator should still emit KnowBlock when the
    # signal is otherwise strong (β5 with freshness_min=None contributes 0).
    window = _make_window(
        status="aligned",
        genes_expressed=4,
        freshness_min=None,
        freshness_top1=None,
        freshness_weighted=None,
    )
    out = decide_know_or_miss(
        window=window, query="strong query",
        top_score=0.9, score_gap=0.6,
        lexical_dense_agree=True, coordinate_confidence=0.8,
        top_gene=gene,
        freshness_status="unknown",
        freshness_min=None,
    )
    assert isinstance(out, KnowBlock)


def test_known_fresh_confidence_higher_than_unknown_freshness():
    """β5 contribution: freshness_min=1.0 yields higher confidence
    than freshness_min=None for the same other features."""
    common = dict(
        top_score=0.6, score_gap=0.2,
        lexical_dense_agree=True, coordinate_confidence=0.5,
    )
    fresh = compute_confidence(**common, freshness_min=1.0)
    unknown = compute_confidence(**common, freshness_min=None)
    assert fresh > unknown


# ─────────────────────────────────────────────────────────────────────
# §13 case 8 — soft-stale on KnowBlock
# ─────────────────────────────────────────────────────────────────────

def test_soft_stale_know_block_recommends_refresh():
    """Spec §13 case 8: top-1 fresh, lower ranks stale → KnowBlock with
    soft_stale=True.
    """
    window = _make_window(
        status="aligned",
        genes_expressed=4,
        freshness_min=0.2,        # rank 2..K stale
        freshness_top1=0.95,      # top-1 fresh
        freshness_weighted=0.55,  # body weighted toward top-1
    )
    top1 = _make_gene("g1", decay=0.95, source_id="/repo/file.py")
    out = decide_know_or_miss(
        window=window, query="some query",
        top_score=0.9, score_gap=0.6,
        lexical_dense_agree=True, coordinate_confidence=0.8,
        top_gene=top1,
        freshness_status="fresh",
        freshness_min=0.2,
    )
    assert isinstance(out, KnowBlock)
    assert out.soft_stale is True


def test_strong_know_block_is_not_soft_stale():
    window = _make_window(
        status="aligned",
        genes_expressed=4,
        freshness_min=0.95,
        freshness_top1=0.95,
        freshness_weighted=0.9,
    )
    out = decide_know_or_miss(
        window=window, query="strong query",
        top_score=0.95, score_gap=0.7,
        lexical_dense_agree=True, coordinate_confidence=0.8,
        top_gene=_make_gene("g1", decay=0.95),
        freshness_status="fresh",
        freshness_min=0.95,
    )
    assert isinstance(out, KnowBlock)
    assert out.soft_stale is False


# ─────────────────────────────────────────────────────────────────────
# §13 case 9 — pydantic validator
# ─────────────────────────────────────────────────────────────────────

def test_refresh_targets_required_for_stale_cold_superseded():
    """Spec §13 case 9: pydantic blocks stale|cold|superseded with
    empty refresh_targets.
    """
    for reason in ("stale", "cold", "superseded"):
        with pytest.raises(ValidationError):
            MissBlock(
                reason=reason, top_score=0.5, ratio=1.0,
                escalate_to=[], refresh_targets=[],
            )
        # And construction succeeds when refresh_targets is non-empty.
        mb = MissBlock(
            reason=reason, top_score=0.5, ratio=1.0,
            escalate_to=[], refresh_targets=["/some/path"],
        )
        assert mb.reason == reason


def test_refresh_targets_forbidden_for_escalate_class_reasons():
    """Mutual exclusivity: escalate-class reasons cannot carry
    refresh_targets.
    """
    for reason in ("abstain", "denatured", "sparse", "no_promoter_match"):
        with pytest.raises(ValidationError):
            MissBlock(
                reason=reason, top_score=0.5, ratio=1.0,
                escalate_to=["rag"], refresh_targets=["/should/fail"],
            )


def test_escalate_to_required_for_escalate_class_reasons():
    """Mutual exclusivity: escalate-class reasons must carry at least
    one tool — empty escalate_to is only valid for refresh-class
    reasons.
    """
    for reason in ("abstain", "denatured", "sparse", "no_promoter_match"):
        with pytest.raises(ValidationError):
            MissBlock(
                reason=reason, top_score=0.5, ratio=1.0,
                escalate_to=[],
            )


# ─────────────────────────────────────────────────────────────────────
# Integration shape: public exports + adapter
# ─────────────────────────────────────────────────────────────────────

def test_cymatix_refresh_fragment_constant():
    """The Stage 7 prompt fragment exists and carries the load-bearing
    semantics (refresh != escalate, refresh_targets, do not answer)."""
    assert isinstance(CYMATIX_REFRESH_FRAGMENT, str)
    assert "refresh_targets" in CYMATIX_REFRESH_FRAGMENT
    assert "DO NOT answer from the genome" in CYMATIX_REFRESH_FRAGMENT
    assert "refresh" in CYMATIX_REFRESH_FRAGMENT
    assert "escalate" in CYMATIX_REFRESH_FRAGMENT


def test_full_fragment_concatenates_stage6_and_stage7():
    full = full_fragment()
    assert CYMATIX_NO_MATCH_FRAGMENT in full
    assert CYMATIX_REFRESH_FRAGMENT in full


def test_miss_block_to_refresh_targets_adapter():
    """Spec §11: MissBlock.to_refresh_targets() converts to the
    RefreshTarget wire shape."""
    mb = MissBlock(
        reason="stale", top_score=0.8, ratio=1.4,
        escalate_to=[],
        refresh_targets=["/repo/file.py", "https://api.example.com/v1"],
    )
    rts = mb.to_refresh_targets()
    assert len(rts) == 2
    assert all(isinstance(rt, RefreshTarget) for rt in rts)
    assert rts[0].source_id == "/repo/file.py"
    assert rts[0].target_kind == "file"
    assert rts[0].reason == "stale_mtime"
    assert rts[1].source_id == "https://api.example.com/v1"
    assert rts[1].target_kind == "url"


def test_miss_reasons_extended():
    """Stage 7 MUST add the three new reasons additively."""
    for r in ("abstain", "denatured", "sparse", "no_promoter_match"):
        assert r in MISS_REASONS  # Stage 6 still in the tuple
    for r in ("stale", "cold", "superseded"):
        assert r in MISS_REASONS  # Stage 7 additions


def test_idx_genes_supersedes_exists(tmp_path):
    """Stage 7 (spec §2 + §7): partial index on genes(supersedes)
    must be created on genome init."""
    from cymatix_context.genome import Genome
    g = Genome(str(tmp_path / "test.db"))
    try:
        rows = g.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_genes_supersedes'"
        ).fetchall()
        assert len(rows) == 1
    finally:
        g.close()


# ─────────────────────────────────────────────────────────────────────
# Acceptance §14 — synthetic round trip
# ─────────────────────────────────────────────────────────────────────

def test_synthetic_planted_stale_emits_miss_stale():
    """§14: planted-stale needles produce MissBlock(reason="stale").

    Synthetic version: build 10 cases where freshness_status="stale",
    confirm 100% (>= 95% acceptance) emit reason="stale".
    """
    n = 10
    miss_count = 0
    for i in range(n):
        window = _make_window(
            status="aligned", genes_expressed=4,
            freshness_min=0.1, freshness_top1=0.1, freshness_weighted=0.3,
        )
        top1 = _make_gene(f"g_{i}", decay=0.1, source_id=f"/repo/needle_{i}.py")
        out = decide_know_or_miss(
            window=window, query=f"query_{i}",
            top_score=0.9, score_gap=0.6,
            lexical_dense_agree=True, coordinate_confidence=0.8,
            top_gene=top1,
            freshness_status="stale",
            freshness_min=0.1,
        )
        if isinstance(out, MissBlock) and out.reason == "stale":
            miss_count += 1
    rate = miss_count / n
    assert rate >= 0.95, f"planted-stale demotion rate {rate:.2f} below 0.95"

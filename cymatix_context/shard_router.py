"""ShardRouter — federated query across category shard .db files.

Task 2 of phase-2 sharding (docs/specs/2026-04-17-knowledge store-sharding-plan.md).

Owns main.db (routing + fingerprint_index) and a lazy cache of KnowledgeStore
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
- CymatixContextManager integration behind CYMATIX_USE_SHARDS flag (Task 7)
- Cross-shard FTS bloom prefilter (deferred)
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .genome import Genome
from .hardware import sqlite_memory_budget
from .schemas import Gene
from .shard_schema import list_shards, open_main_db

log = logging.getLogger(__name__)

try:  # optional dependency — pins BLAS threads so N shard workers x M BLAS
    # threads don't oversubscribe cores during the parallel dense matmul.
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover - threadpoolctl absent
    _threadpool_limits = None


def _blas_limit(n: int = 1):
    """Context manager pinning BLAS thread count during the parallel fan-out.

    Each shard's dense recall runs ``matrix @ query_vec`` which dispatches to
    BLAS; with W worker threads each spawning BLAS's default (= core count)
    threads, the machine thrashes. Pin BLAS to ``n`` inside the pool so the
    parallelism comes from the W shards, not nested BLAS pools. No-op if
    threadpoolctl is unavailable (worker count is still capped).
    """
    if _threadpool_limits is None:
        return contextlib.nullcontext()
    return _threadpool_limits(limits=n, user_api="blas")


def shard_fanout_workers(n_shards: Optional[int] = None) -> int:
    """Worker count for concurrent shard fan-out.

    Resolution order (issue #206, 2026-06-12):

    1. ``CYMATIX_SHARD_WORKERS`` set and parseable → that value (explicit env
       always wins; ``CYMATIX_SHARD_WORKERS=1`` forces the serial path). An
       unparseable value keeps the legacy behavior and forces serial.
    2. Env unset/empty and ``n_shards`` (the routed-shard count for this
       query) > 4 → :func:`cymatix_context.parallel.auto_shard_workers`
       sizes the pool from VRAM/CPU headroom. Evidence: the serial default
       measured 5 min/query at 829K genes / 100 shards vs ~55s at 8
       workers (issue #206, 2026-06-12) — an unconfigured large store was
       paying a 5x-plus latency tax for a knob nobody knew existed.
    3. Otherwise ``1`` → serial fan-out, byte-identical to the
       pre-parallel path. Small/monolithic stores (≤4 routed shards) and
       the determinism regression tests keep the serial reference oracle
       by default (``tests/test_shard_router.py`` pins workers explicitly
       via the env var).

    Parallel fan-out only changes *when* each per-shard fetch runs — the
    accumulation/merge stays in deterministic ``shard_names`` order, so
    ranked output is byte-identical at any worker count.
    """
    raw = os.environ.get("CYMATIX_SHARD_WORKERS", "").strip()
    if not raw:
        if n_shards is not None and n_shards > 4:
            try:
                from .parallel import auto_shard_workers
                return max(1, int(auto_shard_workers()))
            except Exception:  # pragma: no cover - sizer probe failure
                log.warning(
                    "auto shard-worker sizing failed; falling back to serial",
                    exc_info=True,
                )
                return 1
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


# ── Cross-shard BM25 IDF normalization (#118) ────────────────────────
#
# BM25 IDF is corpus-local: each shard computes IDF over its own docs
# only, so a term that's rare globally but common in one shard gets a
# small IDF locally — and the BM25 contribution that flows into
# Genome.last_query_scores is artificially compressed for that shard.
# When the router merges across shards, those compressed scores compete
# unfairly with shards where the term is rare locally and IDF is large.
#
# The shard-correction multiplier ``m_shard`` rescales a shard's
# scores by the ratio of cumulative global IDF to cumulative local IDF
# (weighted by global IDF so rare terms dominate, mirroring BM25's
# own weighting). Clipped to a bounded range to keep extreme single-term
# l_idf≈0 cases from blowing up the merge.
#
# Note this is an APPROXIMATION — BM25's per-document score is a sum
# over (IDF · TF-normalized) terms, not strictly proportional to the
# IDF aggregate. The approximation is good enough for cross-shard
# RANKING (we only need monotone shifts), as confirmed empirically on
# the medium-sharded bench (#118).
IDF_EPS = 1e-3                # floor for local IDF to avoid divide-by-zero
IDF_CLIP_LO = 0.5             # clip range for m_shard
IDF_CLIP_HI = 3.0


# ── Cross-shard co-activation expansion (#120) ───────────────────────
#
# Blob mode pulls docs reachable via 1-hop ``harmonic_links`` edges from
# the BM25 candidate set into the result (``KnowledgeStore._expand_coactivated``
# → ``storage.co_activation.expand_coactivated``). That is how a
# README.md / CLAUDE.md doc surfaces even when its direct keyword match
# is weak — it is linked from a more-specific doc that DOES match.
#
# Sharded mode's ``harmonic_links`` table is intra-shard only, and the
# router has no cross-shard expansion pass, so a gold doc reachable only
# via a harmonic link from a doc in a DIFFERENT shard never surfaces.
# The pass below is the sharded equivalent: after the IDF-corrected RRF
# merge produces a top-K candidate set, each candidate's harmonic links
# are read from its owning shard, the linked gene_ids are resolved to
# their shards via ``fingerprint_index``, and the linked docs are merged
# in at a discounted score so they can be re-ranked into the result.
#
# Mirrors blob's semantics: 1-hop only, a small per-source neighbour cap
# (blob uses 5), and linked docs that are already in the candidate set
# are not re-scored down (highest score wins on collision).
COACT_LINK_BOOST = 0.5        # linked-doc score = boost × source-doc score
COACT_MAX_LINKS_PER_DOC = 5   # 1-hop fan-out cap per candidate (blob: [:5])


# ── Intra-shard doc-type ranking boost (#121) ────────────────────────
#
# Sharded retrieval ranks each shard's candidates with that shard's own
# corpus-local statistics. High-level summary docs (README.md,
# CLAUDE.md, INDEX.md) state answers in conceptual terms; within a
# shard they are routinely out-ranked by more-specific files that
# mention the query terms more densely. The IDF correction (above)
# fixes cross-shard *scale* mismatch but not this *intra-shard rank*
# gap — so a README that holds the answer can sit at intra-shard rank
# ~7 and never clear the cross-shard merge truncation to top-K.
#
# ``DOC_TYPE_BOOST`` is a small multiplicative bump applied to the
# IDF-corrected score of candidates whose source path basename matches
# one of ``DOC_TYPE_BOOST_BASENAMES``. It is deliberately small: it
# only re-orders candidates that are already near-tied. A specific
# implementation file that genuinely out-scores a README by a wide
# margin keeps its lead — a 15% bump cannot overtake a 2× score gap —
# so the boost cannot regress queries whose gold IS a deep file (the
# explicit risk called out in #121's approach (a)).
#
# Applied ONLY on the genuine cross-shard merge path (≥2 shards),
# mirroring the IDF correction's gate. The single-shard fast path stays
# byte-identical to a bare ``Genome.query_docs`` call, and blob mode
# never constructs a ShardRouter at all — so blob behavior is unchanged
# by construction.
DOC_TYPE_BOOST = 1.15
DOC_TYPE_BOOST_BASENAMES = frozenset({
    "readme.md",
    "claude.md",
    "index.md",
})

# Valid modes for the #264 doc-type boost knob. See _apply_doc_type_boost.
DOC_TYPE_BOOST_MODES = ("additive", "rank", "off")

# #264 "rank" mode: the doc-type lift becomes an extra rank-domain tier fed
# into the cross-shard Fuser (rather than a magnitude multiply on the
# IDF-corrected score). weight == the shard tiers' unit weight so an eligible
# summary doc contributes exactly one rank hit at its corrected-rank — enough
# to reorder genuine rank near-ties (the RRF regime #121 was meant to help),
# never enough to leapfrog a doc that already dominates the per-shard ranks.
DOC_TYPE_RANK_TIER_WEIGHT = 1.0


def _doc_type_boost_for(source_id: Optional[str]) -> float:
    """Return the doc-type score multiplier for a candidate's source path.

    A path whose basename (case-insensitively) is one of
    ``DOC_TYPE_BOOST_BASENAMES`` gets ``DOC_TYPE_BOOST``; everything
    else gets ``1.0`` (identity — no change). Handles both ``/`` and
    ``\\`` separators so Windows-ingested paths match too. A missing
    or empty ``source_id`` gets the identity multiplier.

    Used by :meth:`ShardRouter.query_genes` on the cross-shard merge
    path only.
    """
    if not source_id:
        return 1.0
    # Basename only — split on both separators so a path ingested on
    # Windows (back-slashes) is treated the same as a POSIX path.
    basename = source_id.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    if basename in DOC_TYPE_BOOST_BASENAMES:
        return DOC_TYPE_BOOST
    return 1.0


def _compute_shard_idf_correction(
    query_terms: List[str],
    shard_n: Dict[str, int],
    shard_dfs: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """Compute a per-shard score-correction multiplier for IDF mismatch.

    For each query term ``t``::

        local_idf(t, shard)  = log((N_shard - df_shard(t) + 0.5) /
                                   (df_shard(t) + 0.5) + 1.0)
        global_idf(t)        = log((Σ N_shard - Σ df_shard(t) + 0.5) /
                                   (Σ df_shard(t) + 0.5) + 1.0)

    The per-shard multiplier is the global-IDF-weighted mean of
    ``global_idf(t) / max(local_idf(t), IDF_EPS)``, clipped to
    ``[IDF_CLIP_LO, IDF_CLIP_HI]``. A shard where query terms are
    over-represented locally (small local IDF → BM25 score artificially
    compressed) gets ``m_shard > 1`` and its candidates are amplified
    in the cross-shard ranking. A shard where terms are rarer locally
    (large local IDF → BM25 score artificially inflated) gets
    ``m_shard < 1`` and its candidates are deflated.

    Used by ``ShardRouter.query_genes`` to renormalize each shard's
    ``last_query_scores`` before the cross-shard merge.

    Args:
        query_terms: list of query terms (after dedup, before per-shard
            synonym expansion — we use the unexpanded set so multipliers
            are consistent across shards).
        shard_n: ``{shard_name: doc_count}``.
        shard_dfs: ``{shard_name: {term: document_frequency_in_shard}}``.

    Returns:
        ``{shard_name: m_shard}``. Shards absent from ``shard_n`` get
        no entry. A shard whose every query term has df=0 (or whose
        terms have no global signal) gets ``m_shard = 1.0`` — falls
        back to identity transform.
    """
    if not query_terms or not shard_n:
        return {sn: 1.0 for sn in shard_n}

    # Aggregate global N and global df per term across all shards.
    total_n = sum(shard_n.values())
    global_df: Dict[str, int] = {t: 0 for t in query_terms}
    for shard_name, dfs in shard_dfs.items():
        for t, df in dfs.items():
            if t in global_df:
                global_df[t] += int(df)

    if total_n <= 0:
        return {sn: 1.0 for sn in shard_n}

    # Compute global IDF per term using BM25's standard "+0.5" smoothing.
    # A term with df=0 globally has no signal across the corpus — give
    # it the maximum smoothed IDF (treat it as a hapax) so it doesn't
    # crash the weighting; but in practice df=0 means no shard returned
    # candidates on it, so it never dominates the average.
    global_idf: Dict[str, float] = {}
    for t in query_terms:
        df_g = global_df[t]
        # Smoothed BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        global_idf[t] = math.log((total_n - df_g + 0.5) / (df_g + 0.5) + 1.0)

    # Per-shard multiplier: weighted mean of global_idf / local_idf,
    # weighted by global_idf so the term with strongest cross-corpus
    # discrimination dominates. Clipped to a bounded range.
    multipliers: Dict[str, float] = {}
    for shard_name, n_shard in shard_n.items():
        dfs = shard_dfs.get(shard_name, {})
        # Only consider terms that actually fire in this shard.
        # A term with df=0 in this shard contributed nothing to BM25
        # scores from this shard, so it shouldn't drive the correction.
        weighted_sum = 0.0
        weight_total = 0.0
        for t in query_terms:
            df = int(dfs.get(t, 0))
            if df <= 0:
                continue
            # BM25-style smoothed local IDF
            local_idf = math.log(
                (n_shard - df + 0.5) / (df + 0.5) + 1.0
            )
            local_idf_safe = max(local_idf, IDF_EPS)
            g_idf = global_idf.get(t, 0.0)
            # Skip terms with no global discrimination signal.
            if g_idf <= 0.0:
                continue
            ratio = g_idf / local_idf_safe
            # Weight rare-global-term ratios more heavily — these
            # are the terms BM25 considers most discriminative.
            weighted_sum += g_idf * ratio
            weight_total += g_idf

        if weight_total <= 0.0:
            # No participating terms; identity transform.
            multipliers[shard_name] = 1.0
            continue

        m = weighted_sum / weight_total
        # Clip extreme values. Without clipping, a single very-common
        # local term (l_idf near zero) can produce m_shard >> 100×,
        # which would let the unrelated co-shard docs steamroll the
        # merge. The clip is a stability knob; tune in cymatix.toml if
        # later bench data wants different bounds.
        multipliers[shard_name] = max(IDF_CLIP_LO, min(IDF_CLIP_HI, m))

    return multipliers


def _global_idf_enabled() -> bool:
    """True iff CYMATIX_SHARD_GLOBAL_IDF is set to a truthy value.

    Truthy (case-insensitive): '1', 'true', 'yes', 'on'. Anything else
    (including unset) is False. Gates the per-doc global-IDF lexical
    re-score in :meth:`ShardRouter.query_genes` (#182). OFF by default —
    when off, the router uses the legacy per-shard scalar ``m_shard``
    correction and behaviour is byte-identical to the pre-#182 build.
    """
    v = os.environ.get("CYMATIX_SHARD_GLOBAL_IDF")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# #265: process-once suppression notice. The global-IDF splice is undefined on
# the additive/BM25 magnitude scale when the per-shard stores score in RRF, so
# the router falls back to the scalar m_shard path even though the flag is set.
# Warn once (not per query) so a mis-set flag is visible without log spam.
_GLOBAL_IDF_RRF_WARNED = False


def _warn_global_idf_rrf_suppressed() -> None:
    """Log once that CYMATIX_SHARD_GLOBAL_IDF is inert under per-shard RRF (#265)."""
    global _GLOBAL_IDF_RRF_WARNED
    if _GLOBAL_IDF_RRF_WARNED:
        return
    _GLOBAL_IDF_RRF_WARNED = True
    log.warning(
        "CYMATIX_SHARD_GLOBAL_IDF is set but the per-shard Genomes score in "
        "RRF; the global-IDF lexical splice (raw - old_local_lex + "
        "new_global_lex) is undefined on the BM25 magnitude scale under RRF "
        "(#265) and is suppressed. The scalar m_shard IDF correction runs "
        "instead. Pin fusion_mode='additive' to use the splice."
    )


def _compute_global_idf_map(
    query_terms: List[str],
    shard_n: Dict[str, int],
    shard_dfs: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """Per-term TRUE GLOBAL IDF from aggregated cross-shard N and df.

    Uses the RAW (unsmoothed) Robertson-Sparck-Jones IDF that SQLite
    FTS5's ``bm25()`` uses internally — NOT the ``+1.0``-smoothed variant
    that :func:`_compute_shard_idf_correction` uses for its scalar
    multiplier::

        global_idf(t) = ln((ΣN - Σdf(t) + 0.5) / (Σdf(t) + 0.5))

    This is deliberately different from the scalar path. The scalar
    ``m_shard`` only ever forms a RATIO of two IDFs, so the constant
    ``+1.0`` smoothing cancels and keeps the multiplier strictly positive
    and well-conditioned. The per-doc lexical re-score, by contrast, must
    reproduce ``bm25()``'s ABSOLUTE scale so its output can be SPLICED
    (``raw − old_local_lex + new_global_lex``) against the genome's
    ``-rank`` lexical sub-score. The genome's ``-rank`` is exactly the raw
    (unsmoothed) FTS5 BM25 sum; using the smoothed always-positive IDF here
    would put the global lexical term on a different (always-larger,
    never-negative) scale, and the splice would corrupt ranking — the
    CYMATIX_SHARD_GLOBAL_IDF recall regression (0.347 → 0.168) traced to
    exactly this scale mismatch (#182).

    Returns ``{}`` when there's no corpus.
    """
    if not query_terms or not shard_n:
        return {}
    total_n = sum(shard_n.values())
    if total_n <= 0:
        return {}
    global_df: Dict[str, int] = {t: 0 for t in query_terms}
    for _shard_name, dfs in shard_dfs.items():
        for t, df in dfs.items():
            if t in global_df:
                global_df[t] += int(df)
    out: Dict[str, float] = {}
    for t in query_terms:
        df_g = global_df[t]
        # RAW FTS5 IDF, with FTS5's exact negative-IDF clamp. SQLite's
        # fts5 bm25() computes idf = ln((N - n + 0.5) / (n + 0.5)) and then
        # forces ``if (idf <= 0.0) idf = 1e-6;`` — a term in > half the
        # corpus does NOT subtract from the score, it contributes ~nothing.
        # We reproduce that clamp here so the injected global IDF is
        # bit-exact with bm25(). Without the clamp a globally-ubiquitous
        # query term would inject a large NEGATIVE lexical sub-score that
        # the engine never produced, re-corrupting the splice in the
        # opposite direction from the original #182 bug.
        idf = math.log((total_n - df_g + 0.5) / (df_g + 0.5))
        out[t] = idf if idf > 0.0 else FTS5_IDF_FLOOR
    return out


# Lexical score cap — mirrors KnowledgeStore.query_docs Tier-3, which
# adds ``min(-rank, 6.0)`` into the fused score. The global-IDF rescore
# replaces that same capped lexical sub-score per doc.
FTS5_LEXICAL_CAP = 6.0

# SQLite FTS5's bm25() clamps a non-positive IDF to this tiny floor
# (``if (idf <= 0.0) idf = 1e-6;`` in fts5_aux.c). Reproduced in the
# global-IDF rescore so a globally-ubiquitous term contributes ~0 (never a
# large negative) — matching the engine exactly. See _compute_global_idf_map.
FTS5_IDF_FLOOR = 1e-6


# ── Per-shard fetch-depth scaling (#222) ─────────────────────────────
#
# The router fetches ``max_genes * multiplier`` candidates from each shard
# BEFORE the cross-shard merge. The flat legacy ``2×`` is provably too
# shallow whenever post-fetch rescaling (``m_shard`` ∈ [0.5, 3.0], doc-type
# boost, global-IDF splice) can promote a doc past shallower entries from
# other shards: promotion happens AFTER the per-shard cut, so a mid-shard
# gold never reaches the merge. Two config knobs (config.retrieval, threaded
# through the router — see ShardRouter.__init__):
#
#   shard_fetch_multiplier        (float, default 2.0)  — flat oversample.
#   shard_fetch_scale_with_shards (bool,  default off)  — when on, amplify
#       the multiplier by sqrt(n_shards) so a many-shard corpus oversamples
#       each shard MORE deeply (a mid-shard gold in a populous shard must
#       clear a deeper per-shard cut to reach the merge), capped so the
#       per-shard query cost stays bounded on high-shard-count corpora.
#
# The scaled path is clamped to SHARD_FETCH_SCALE_CAP × max_genes. sqrt(n)
# growth is gentle, but a 100-shard corpus would still request
# 2·sqrt(100)=20× without a ceiling; 10× keeps per-shard query cost bounded
# while still giving deep-enough oversampling for the merge. The GLOBAL
# OUTPUT cut stays 2×max_genes regardless (query_genes ``limit``) — only the
# per-shard FETCH grows.
SHARD_FETCH_SCALE_CAP = 10   # × max_genes ceiling for the scaled fetch path

# Env overrides (dark-shipped in #235, still honored, env > toml > default):
#   CYMATIX_SHARD_FETCH_FACTOR   (int) overrides shard_fetch_multiplier.
#   CYMATIX_SHARD_COACT_RESERVE  (int) overrides coact_reserved_slots.


def _env_int(name: str) -> Optional[int]:
    """Parse an int env var. Returns None when unset/empty/unparseable.

    Shared by the legacy env-knob helpers and the router's env-over-config
    resolution so the ``CYMATIX_SHARD_*`` var names are parsed one way only.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _shard_fetch_factor() -> int:
    """Legacy per-shard fetch-depth factor (issue #222). Default 2 = legacy.

    Reads ``CYMATIX_SHARD_FETCH_FACTOR`` only; retained for the standalone
    dark-ship contract (env override, clamped to ≥1, unparseable keeps the
    legacy 2). The router now resolves the effective multiplier from
    config with this env as an override — see
    :meth:`ShardRouter._resolved_fetch_multiplier`.
    """
    v = _env_int("CYMATIX_SHARD_FETCH_FACTOR")
    return max(1, v) if v is not None else 2


def _coact_reserve_slots() -> int:
    """Legacy reserved-slot count for co-activation docs (issue #223).

    Reads ``CYMATIX_SHARD_COACT_RESERVE`` only. Default 0 = legacy (no
    reservation: linked docs enter the final sort at the link discount and
    are truncated like any other candidate — exactly how graph-surfaced
    golds get displaced). Retained for the standalone dark-ship contract;
    the router resolves the effective value from config with this env as an
    override — see :meth:`ShardRouter._resolved_coact_reserve`.
    """
    v = _env_int("CYMATIX_SHARD_COACT_RESERVE")
    return max(0, v) if v is not None else 0


def compute_per_shard_fetch(
    max_genes: int,
    multiplier: float,
    scale_with_shards: bool,
    n_shards: int,
    cap_mult: int = SHARD_FETCH_SCALE_CAP,
) -> int:
    """Per-shard fetch depth before the cross-shard merge (issue #222).

    Legacy defaults (``multiplier=2.0``, ``scale_with_shards=False``) return
    ``2 * max_genes`` — byte-identical to the pre-knob flat cut.

    When ``scale_with_shards`` is on and more than one shard is routed, the
    multiplier is amplified by ``sqrt(n_shards)`` and the result is clamped
    to ``cap_mult * max_genes`` (see SHARD_FETCH_SCALE_CAP). The floor keeps
    the fetch at ≥ ``max_genes`` even for sub-1.0 multipliers, so a shard is
    never asked for fewer candidates than the caller's own cut.

    Raises ``ValueError`` on a negative ``multiplier`` (nonsensical).
    """
    if multiplier < 0:
        raise ValueError(f"shard_fetch_multiplier must be >= 0, got {multiplier}")
    mg = max(1, int(max_genes))
    eff = float(multiplier)
    if scale_with_shards and int(n_shards) > 1:
        eff *= math.sqrt(float(n_shards))
        depth = int(round(mg * eff))
        # Ceiling only on the scaled path — an explicit flat multiplier is
        # the operator's own bounded choice and is honored as given.
        depth = min(depth, int(cap_mult) * mg)
    else:
        depth = int(round(mg * eff))
    return max(mg, depth)


def _validate_shard_knobs(
    multiplier: float,
    reserved_slots: int,
    link_boost: float,
) -> None:
    """Guard the #222/#223 router knobs at construction (negative = raise).

    A negative fetch multiplier, reserved-slot count, or link discount has
    no meaning; raise ``ValueError`` rather than silently clamp so a
    fat-fingered ``cymatix.toml`` fails loud when the router is built.
    """
    if multiplier < 0:
        raise ValueError(
            f"shard_fetch_multiplier must be >= 0, got {multiplier}"
        )
    if reserved_slots < 0:
        raise ValueError(
            f"coact_reserved_slots must be >= 0, got {reserved_slots}"
        )
    if link_boost < 0:
        raise ValueError(
            f"coact_link_boost must be >= 0, got {link_boost}"
        )


def _apply_coact_reserve(
    union_ids: List[str],
    promoted: set,
    corrected: Dict[str, float],
    rrf_all: Dict[str, float],
    limit: int,
    reserve: int,
) -> List[str]:
    """Reserve up to ``reserve`` of the top-``limit`` slots for promoted docs.

    Pure function (issue #223). Given the already-sorted ``union_ids``:
    if fewer than ``reserve`` promoted docs survive the ``[:limit]`` cut,
    the weakest NON-promoted entries in the cut are swapped for the best
    promoted docs below it, then the cut is re-sorted by the same
    (corrected desc, rrf desc, gid asc) key so ordering semantics are
    unchanged. ``reserve=0`` returns the plain truncation (legacy,
    byte-identical). Never grows the result past ``limit`` and never
    drops a promoted doc that already made the cut.
    """
    cut = list(union_ids[:limit])
    if reserve <= 0 or not promoted:
        return cut
    in_cut = sum(1 for g in cut if g in promoted)
    missing = reserve - in_cut
    if missing <= 0:
        return cut
    overflow = [g for g in union_ids[limit:] if g in promoted][:missing]
    if not overflow:
        return cut
    if _score_debug_enabled():
        # #223 receipt validity check: this is the ONLY place a reserved
        # slot actually changes the result (reserve > 0, a promoted doc
        # survived the cross-shard co-activation pull but was sitting
        # beyond the plain [:limit] cut, and got swapped in here). Gated
        # behind the existing CYMATIX_SHARD_SCORE_DEBUG introspection flag
        # (issue #181) rather than a new one.
        log.info(
            "coact_reserve FIRED: reserve=%d in_cut=%d/%d promoting %d doc(s) "
            "from beyond limit=%d: %s",
            reserve, in_cut, len(promoted), len(overflow), limit, overflow,
        )
    # Drop the weakest non-promoted tail entries to make room.
    drop_budget = len(overflow)
    keep: List[str] = []
    for g in reversed(cut):
        if drop_budget > 0 and g not in promoted:
            drop_budget -= 1
            continue
        keep.append(g)
    keep.reverse()
    out = keep + overflow
    out.sort(
        key=lambda gid: (
            -corrected.get(gid, 0.0),
            -rrf_all.get(gid, 0.0),
            gid,
        ),
    )
    return out


def _score_debug_enabled() -> bool:
    """True iff CYMATIX_SHARD_SCORE_DEBUG is set to a truthy value.

    Truthy (case-insensitive): '1', 'true', 'yes', 'on'. Anything else
    (including unset) is False. Gates the score-path instrumentation in
    :meth:`ShardRouter.query_genes` (#181) so the hot path is untouched
    when off.
    """
    v = os.environ.get("CYMATIX_SHARD_SCORE_DEBUG")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


class ShardRouter:
    """Routes queries across category shard .db files.

    Owns:
        - main.db connection (fingerprint_index + shards + identity)
        - lazy dict of KnowledgeStore instances keyed by shard_name

    Exposes a subset of KnowledgeStore's API (query_genes for now; ingest
    lands in Task 6).
    """

    def __init__(self, main_path: str, **genome_kwargs):
        """Open main.db. Shard KnowledgeStores open lazily on first access.

        genome_kwargs are forwarded to each KnowledgeStore on lazy-open — keep
        them identical to what CymatixContextManager passes when
        constructing a solo KnowledgeStore, so sharded + non-sharded return
        identical tiers.
        """
        self.main_path = main_path

        # Sharded fetch-depth (#222) + co-activation budget (#223) knobs.
        # Threaded from config.retrieval via the fanned genome_kwargs (mirrors
        # semantic_broaden_routing below, and fusion_mode's threading). These
        # are ROUTER-only — a solo Genome accepts them (so the fan-out doesn't
        # raise) but ignores them. Defaults reproduce the dark-shipped env-knob
        # behaviour byte-for-byte: 2.0× flat fetch, no shard-count scaling,
        # zero reserved co-activation slots, 0.5 link discount. The env vars
        # CYMATIX_SHARD_FETCH_FACTOR / CYMATIX_SHARD_COACT_RESERVE still override
        # these at query time (env > toml > default), resolved per-call.
        # Resolved + validated FIRST, before any resource (the main.db
        # connection) is acquired, so a fat-fingered cymatix.toml fails loud
        # without leaking an open SQLite handle.
        self._shard_fetch_multiplier: float = float(
            genome_kwargs.get("shard_fetch_multiplier", 2.0)
        )
        self._shard_fetch_scale_with_shards: bool = bool(
            genome_kwargs.get("shard_fetch_scale_with_shards", False)
        )
        self._coact_reserved_slots: int = int(
            genome_kwargs.get("coact_reserved_slots", 0)
        )
        self._coact_link_boost: float = float(
            genome_kwargs.get("coact_link_boost", COACT_LINK_BOOST)
        )
        _validate_shard_knobs(
            self._shard_fetch_multiplier,
            self._coact_reserved_slots,
            self._coact_link_boost,
        )

        # #264 doc-type boost mode (default-inert "additive"). Validated here
        # so a direct ShardRouter(..., doc_type_boost_mode=...) fails fast the
        # same way a typo in cymatix.toml does (RetrievalConfig.__post_init__).
        self._doc_type_boost_mode: str = str(
            genome_kwargs.get("doc_type_boost_mode", "additive")
        )
        if self._doc_type_boost_mode not in DOC_TYPE_BOOST_MODES:
            raise ValueError(
                "doc_type_boost_mode must be one of "
                f"{DOC_TYPE_BOOST_MODES}, got {self._doc_type_boost_mode!r}"
            )

        self.main_conn = open_main_db(main_path)
        self._genome_kwargs = genome_kwargs
        self._shards: Dict[str, Genome] = {}

        # RAM-aware SQLite budget (PRD 2026-05-30: dynamic-ram-scaling),
        # resolved ONCE from the registered shard count and threaded into each
        # lazily-opened shard. Snapshotting here (before shards open) keeps the
        # per-shard mmap/cache figure deterministic across the fan-out.
        try:
            _n = self.main_conn.execute(
                "SELECT COUNT(*) FROM shards WHERE health = 'ok'"
            ).fetchone()[0]
        except Exception:
            _n = 1
        self._mem_plan = sqlite_memory_budget(max(1, int(_n or 1)))

        # Semantic-wiring arm (PRD 2026-06-02): broaden routing to all healthy
        # shards for query_type=="semantic" when CYMATIX_SEMANTIC_ARM=1. Pulled
        # from the fanned genome_kwargs (config.retrieval.semantic_broaden_routing).
        self._semantic_broaden_routing: bool = bool(
            genome_kwargs.get("semantic_broaden_routing", True)
        )

        # Retrieval introspection — mirrors KnowledgeStore's interface so
        # callers can use either interchangeably.
        self.last_query_scores: Dict[str, float] = {}
        self.last_tier_contributions: Dict[str, Dict[str, float]] = {}
        # Cross-shard score-path introspection (issue #181 -- gold score
        # depression). Populated ONLY when CYMATIX_SHARD_SCORE_DEBUG is
        # truthy; stays {} otherwise so the hot path has zero overhead
        # and behaviour is byte-identical. See query_genes.
        #   last_score_breakdown: gene_id -> {shard, raw, m_shard,
        #       doc_type_boost, corrected, rrf} for EVERY merged
        #       candidate (not just the returned top-`limit`).
        #   last_shard_multipliers: shard_name -> m_shard.
        self.last_score_breakdown: Dict[str, dict] = {}
        self.last_shard_multipliers: Dict[str, float] = {}
        # Guards last_query_scores / last_tier_contributions writes from
        # racing concurrent /context calls. KnowledgeStore has the same
        # lock on the same attributes (see knowledge_store.py:479) — the
        # router needs to match that contract so the adapter snapshot in
        # ShardedGenomeAdapter.query_docs sees a consistent pair.
        self._last_query_scores_lock = threading.Lock()
        # Guards the self._shards cache dict against a race when concurrent
        # fan-out workers first-open distinct shards simultaneously (each
        # mutates the dict). Held only around the cache check + Genome
        # construction, never around the heavy query_docs call.
        self._shards_lock = threading.Lock()

    # ── Knob resolution (env > toml > default) ──────────────────────

    def _resolved_fetch_multiplier(self) -> float:
        """Effective per-shard fetch multiplier (issue #222).

        ``CYMATIX_SHARD_FETCH_FACTOR`` (int, dark-ship override) wins when set
        and parseable; otherwise the config-threaded
        ``shard_fetch_multiplier`` (default 2.0). Clamped to ≥1 on the env
        path to preserve the legacy env contract.
        """
        v = _env_int("CYMATIX_SHARD_FETCH_FACTOR")
        if v is not None:
            return float(max(1, v))
        return float(self._shard_fetch_multiplier)

    def _resolved_coact_reserve(self) -> int:
        """Effective reserved co-activation slot count (issue #223).

        ``CYMATIX_SHARD_COACT_RESERVE`` (int, dark-ship override) wins when set
        and parseable; otherwise the config-threaded ``coact_reserved_slots``
        (default 0). Clamped to ≥0.
        """
        v = _env_int("CYMATIX_SHARD_COACT_RESERVE")
        if v is not None:
            return max(0, v)
        return max(0, int(self._coact_reserved_slots))

    def _per_shard_fusion_mode(self) -> str:
        """Effective ``fusion_mode`` of every per-shard Genome.

        The router forwards ``genome_kwargs`` verbatim to each lazily-opened
        shard ``Genome`` (see ``_open_shard``), so a single lookup answers
        "what score-scale do the per-shard ``last_query_scores`` live on?".
        Absent an explicit kwarg the per-shard Genome falls back to
        ``KnowledgeStore``'s own default, which is ``"rrf"`` — so production
        sharded serving runs the shards in RRF unless the operator pins
        ``fusion_mode="additive"``. This is the honest signal the #264
        doc-type boost and #265 global-IDF splice both gate on: their
        additive-scale arithmetic is only well-defined when this is
        ``"additive"``.
        """
        return str(self._genome_kwargs.get("fusion_mode", "rrf"))

    # ── Shard lifecycle ─────────────────────────────────────────────

    def _open_shard(self, shard_name: str) -> Genome:
        """Lazy-open a KnowledgeStore against the shard .db. Cached.

        Thread-safe: the cache check + insert is serialized so concurrent
        fan-out workers can't double-open or corrupt ``self._shards``. The
        lock does NOT cover query execution, so heavy per-shard work still
        runs fully concurrently across distinct shards.
        """
        with self._shards_lock:
            if shard_name not in self._shards:
                row = self.main_conn.execute(
                    "SELECT path FROM shards WHERE shard_name = ? AND health = 'ok'",
                    (shard_name,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"shard not registered: {shard_name}")
                path = row["path"]
                log.info("opening shard %s at %s", shard_name, path)
                _kwargs = dict(self._genome_kwargs)
                _kwargs.setdefault("mem_plan", self._mem_plan)
                self._shards[shard_name] = Genome(path, **_kwargs)
            return self._shards[shard_name]

    def known_shards(self, category: Optional[str] = None) -> List[str]:
        """Shard names registered in main.db, optionally filtered."""
        return [r["shard_name"] for r in list_shards(self.main_conn, category)]

    # ── Routing ─────────────────────────────────────────────────────

    def route(
        self,
        domains: List[str],
        entities: List[str],
        query_type: Optional[str] = None,
    ) -> List[str]:
        """Return shard names likely to contain matches for the query.

        V1: LIKE scan against fingerprint_index.domains/entities.
        Match is loose — if any query term appears in any shard's
        fingerprint domains or entities, that shard is in. Ordered
        by match count (most matches first).

        Empty query → return every healthy shard (preserves the
        fallback path callers rely on for "return something").

        Semantic-wiring arm (PRD 2026-06-02): when ``query_type=="semantic"``
        AND env ``CYMATIX_SEMANTIC_ARM=1`` AND ``semantic_broaden_routing``,
        bypass the LIKE gate and fan out to every healthy shard so dense-top
        golds whose shard the literal gate would drop still enter the pool.
        Any other query_type (or arm off) keeps the byte-identical LIKE scan.
        """
        if (
            query_type == "semantic"
            and self._semantic_broaden_routing
            and os.environ.get("CYMATIX_SEMANTIC_ARM") == "1"
        ):
            return self.known_shards()

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
            # shard_name ASC tiebreak: without it, two shards with equal
            # hit counts come back in whatever insertion/index order
            # SQLite happens to pick, which differs across rebuilds and
            # WAL checkpoints. Deterministic fan-out order matters because
            # the merge in query_genes is "first-shard-wins on ties".
            "ORDER BY hits DESC, shard_name ASC"
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
        read_only: bool = False,
        **kwargs,
    ) -> List[Gene]:
        """Fan query across routed shards and merge with cross-shard IDF correction.

        Mirrors Genome.query_genes signature so callers can swap
        without branching. Extra kwargs (``use_harmonic``, ``cwola_weight``,
        etc.) forward verbatim to each shard's ``Genome.query_genes``; any
        kwarg a given shard rejects is silently dropped for that shard.

        Merge strategy (#118 fix):
            1. Fan candidates out via each shard's ``query_docs``.
            2. Probe each shard's FTS5 index for ``N_shard`` and per-term
               ``df_shard(t)`` over the query terms.
            3. Compute a per-shard correction multiplier
               (see :func:`_compute_shard_idf_correction`) that rescales
               each shard's scores by the global-vs-local IDF mismatch.
            4. Sort candidates by IDF-corrected score, breaking ties
               with RRF (rank-fused) so single-shard rank-1 tie chains
               still get a deterministic ordering.

        For a single-shard query (the only-shard fallback path), the
        correction multiplier collapses to identity and the routing
        behaves as if scores came directly from that shard.
        """
        from .retrieval.measurement import current_capture
        _measurement = current_capture()
        if _measurement is not None:
            _measurement.unsupported.append("shard_router: fanout and merge are not captured")

        # Semantic-wiring arm (PRD 2026-06-02): pop query_type OUT of the
        # generic kwargs BEFORE fan-out. It must NOT ride in **kwargs — the
        # per-shard KnowledgeStore.query_docs would TypeError on an unexpected
        # kwarg and the except-fallback below silently drops ALL kwargs
        # (use_harmonic, etc.). Pass it explicitly into route() and each
        # shard.query_docs() instead. None (default / arm off) is inert.
        query_type = kwargs.pop("query_type", None)
        shard_names = self.route(domains, entities, query_type)
        # #209 phase 1: shard-router fan-out + discrimination telemetry.
        # fanout = shards consulted for this query; discrimination =
        # routed / known healthy shards in [0, 1] (1.0 = router consulted
        # every shard — the #165 degeneracy case, zero discrimination).
        # Soft-fails; instruments are no-ops when OTel is off.
        try:
            from .telemetry import (
                shard_discrimination_histogram,
                shard_fanout_histogram,
            )
            shard_fanout_histogram().record(len(shard_names))
            _known = len(self.known_shards())
            if _known > 0:
                shard_discrimination_histogram().record(
                    len(shard_names) / _known
                )
        except Exception:
            pass
        if not shard_names:
            with self._last_query_scores_lock:
                self.last_query_scores = {}
                self.last_tier_contributions = {}
            return []

        from .retrieval.fusion import Fuser, DEFAULT_RRF_K

        # Materialised candidates by gene_id (last writer wins on
        # content-hash collisions across shards — same content, same id).
        merged: Dict[str, Gene] = {}
        # Carry per-tier contributions through the merge so downstream
        # introspection (cwola log, activation profile in /context) still
        # sees them. Keep the contributions from whichever shard ranked
        # the doc best.
        merged_tier: Dict[str, Dict[str, float]] = {}
        merged_tier_source: Dict[str, int] = {}

        # Per-shard data we need for IDF normalization:
        #   - which gene_ids came from each shard, with raw scores
        #   - each shard's N (FTS doc count)
        #   - each shard's per-term df over the query terms
        # Collect these as we fan out, then compute the multiplier after.
        shard_ranked: Dict[str, List[Tuple[str, float]]] = {}
        shard_n: Dict[str, int] = {}
        shard_dfs: Dict[str, Dict[str, int]] = {}

        # Build the set of query terms used for IDF probing. Deduplicate
        # case-insensitively and drop very short terms (mirrors the
        # ``len(t) > 2`` filter used elsewhere for FTS5 queries).
        idf_terms: List[str] = []
        _idf_seen: set[str] = set()
        for t in (domains or []) + (entities or []):
            if not t:
                continue
            k = t.lower()
            if k in _idf_seen or len(k) <= 2:
                continue
            _idf_seen.add(k)
            idf_terms.append(k)

        fuser = Fuser(k=DEFAULT_RRF_K)

        # Per-shard fetch depth: request 2x more from each shard than the
        # caller's ``max_genes``. Genome.query_docs already returns
        # ``max_genes * 2`` candidates internally; doubling the routed
        # value here gives each shard 4x candidate depth (=4× max_genes
        # candidates), so a doc at intra-shard rank ~30 reaches the
        # cross-shard merge after IDF correction. Without this lift,
        # cross-shard re-ranking sees only each shard's top ~24 — and a
        # gold doc that's intra-shard rank 25+ never enters the merge.
        # The price is per-shard query cost; in benchmarks the medium
        # fixture's 6 shards stay well under 1s with this fan-out.
        #
        # Issue #222: the depth is config-tunable (shard_fetch_multiplier /
        # shard_fetch_scale_with_shards, threaded through the router;
        # CYMATIX_SHARD_FETCH_FACTOR still overrides). Defaults (2.0×, scale
        # off) are byte-identical to the legacy flat 2×max_genes cut. The
        # flat 2× is too shallow whenever post-fetch rescaling (m_shard,
        # doc-type boost, global-IDF splice) would promote a mid-shard doc
        # past other shards' entries — the promotion happens AFTER this cut,
        # so the doc never reaches the merge. Scaling by sqrt(n_shards)
        # oversamples populous many-shard corpora more deeply; the global
        # OUTPUT cut below stays 2×max_genes regardless.
        per_shard_fetch = compute_per_shard_fetch(
            max_genes=int(max_genes),
            multiplier=self._resolved_fetch_multiplier(),
            scale_with_shards=self._shard_fetch_scale_with_shards,
            n_shards=len(shard_names),
        )

        def _fetch(shard_name: str):
            """Open + query one shard + probe IDF, snapshotting its per-query
            introspection. Returns a payload dict, or ``None`` to skip (open
            or query failure — same soft-fail contract as the serial loop).

            Safe to run concurrently across distinct shards: each shard is a
            separate ``Genome``/``KnowledgeStore`` with its own SQLite
            connection and per-instance dense-matrix cache. The only shared
            mutation — the ``self._shards`` cache — is lock-guarded inside
            ``_open_shard``. ``last_query_scores`` / ``last_tier_contributions``
            are instance state, so we snapshot them here before returning.
            """
            try:
                shard = self._open_shard(shard_name)
            except Exception:
                log.warning("shard %s failed to open; skipping", shard_name, exc_info=True)
                return None

            try:
                genes = shard.query_docs(
                    domains=domains,
                    entities=entities,
                    max_genes=per_shard_fetch,
                    party_id=party_id,
                    read_only=read_only,
                    query_type=query_type,
                    **kwargs,
                )
            except TypeError:
                # Kwarg mismatch with an older KnowledgeStore schema — fall back to
                # the minimal signature so a stale shard still contributes.
                log.warning(
                    "shard %s rejected kwargs %s; falling back to base signature",
                    shard_name, list(kwargs.keys()),
                )
                try:
                    genes = shard.query_docs(
                        domains=domains,
                        entities=entities,
                        max_genes=per_shard_fetch,
                        party_id=party_id,
                        read_only=read_only,
                    )
                except Exception:
                    log.warning("shard %s query failed; skipping", shard_name, exc_info=True)
                    return None
            except Exception:
                log.warning("shard %s query failed; skipping", shard_name, exc_info=True)
                return None

            # IDF probe: cheap COUNT-over-MATCH queries against this
            # shard's FTS5. Soft-fails to empty dict — that shard then
            # gets m_shard = 1.0 (identity) so its scores pass through
            # uncorrected.
            try:
                s_n = int(shard.fts_doc_count())
                s_dfs = dict(shard.term_doc_frequencies(idf_terms))
            except Exception:
                log.warning(
                    "shard %s IDF probe failed; falling back to identity correction",
                    shard_name, exc_info=True,
                )
                s_n = 0
                s_dfs = {}

            # Snapshot per-query introspection NOW — it is instance state that
            # a later query on the same shard would overwrite.
            scores_snapshot = dict(shard.last_query_scores)
            tiers_src = shard.last_tier_contributions
            tiers_snapshot = {
                doc.gene_id: dict(tiers_src.get(doc.gene_id, {})) for doc in genes
            }
            return {
                "shard_name": shard_name,
                "genes": genes,
                "scores": scores_snapshot,
                "tiers": tiers_snapshot,
                "shard_n": s_n,
                "shard_dfs": s_dfs,
            }

        # Fetch every routed shard. The serial path (workers<=1) is the
        # reference oracle. The parallel path runs the independent per-shard
        # fetches concurrently but PRESERVES shard_names order in the result
        # list (``ThreadPoolExecutor.map`` is order-preserving), so the
        # deterministic accumulation below is byte-identical either way — only
        # *when* each fetch runs changes, not the merge order. BLAS is pinned
        # to 1 thread inside the pool so W shard workers don't oversubscribe
        # cores via nested BLAS pools on the per-shard dense matmul.
        _workers = shard_fanout_workers(len(shard_names))
        if _workers > 1 and len(shard_names) > 1:
            with _blas_limit(1):
                with ThreadPoolExecutor(
                    max_workers=min(_workers, len(shard_names)),
                    thread_name_prefix="cymatix-shard",
                ) as _ex:
                    fetched = list(_ex.map(_fetch, shard_names))
        else:
            fetched = [_fetch(sn) for sn in shard_names]

        # Accumulate in deterministic shard_names order (identical semantics
        # to the original serial loop body).
        for payload in fetched:
            if payload is None:
                continue
            shard_name = payload["shard_name"]
            shard_n[shard_name] = payload["shard_n"]
            shard_dfs[shard_name] = payload["shard_dfs"]
            shard_scores = payload["scores"]
            payload_tiers = payload["tiers"]
            ranked_for_shard: List[Tuple[str, float]] = []
            for doc in payload["genes"]:
                raw_score = float(shard_scores.get(doc.gene_id, 0.0))
                ranked_for_shard.append((doc.gene_id, raw_score))
                if doc.gene_id not in merged:
                    merged[doc.gene_id] = doc
                if (
                    doc.gene_id not in merged_tier_source
                    or raw_score > merged_tier_source[doc.gene_id]
                ):
                    merged_tier[doc.gene_id] = dict(payload_tiers.get(doc.gene_id, {}))
                    merged_tier_source[doc.gene_id] = raw_score

            shard_ranked[shard_name] = ranked_for_shard
            # Still feed the Fuser — RRF acts as the tiebreaker on the
            # corrected-score sort.
            fuser.add_tier(shard_name, ranked_for_shard, weight=1.0)

        # ── Cross-shard IDF correction ──────────────────────────────
        # Skip when only one shard participated — the corpus IS the
        # shard, so its local IDF is already global. This also keeps
        # the single-shard fallback path byte-identical to a bare
        # ``Genome.query_docs`` call composed under a router.
        if len(shard_ranked) <= 1 or not idf_terms:
            multipliers = {sn: 1.0 for sn in shard_ranked}
        else:
            multipliers = _compute_shard_idf_correction(
                idf_terms, shard_n, shard_dfs,
            )

        # ── Per-doc GLOBAL-IDF lexical re-score (#182, flag-gated) ──
        # The scalar ``m_shard`` rescales every doc in a shard equally, so
        # it cannot change the WITHIN-shard order of two docs — that order
        # is set by each shard's corpus-LOCAL BM25 IDF. A gold whose
        # distinguishing term is globally-rare-but-locally-common is buried
        # under a same-shard incumbent, and no scalar can rescue it.
        #
        # When CYMATIX_SHARD_GLOBAL_IDF is truthy AND ≥2 shards participated,
        # we recompute each candidate's BM25 LEXICAL sub-score per-doc with
        # TRUE GLOBAL IDF (from the aggregated global N / df already probed
        # for the scalar path) and SPLICE it back into that doc's fused
        # raw score, replacing only the lexical portion. The dense/tier
        # parts of raw are left untouched. When the flag is off, this whole
        # block is skipped and the legacy scalar path runs unchanged
        # (behaviour-preserving).
        #
        # #265 SCALE GUARD (capability guard, not a knob): the splice
        # ``corrected = raw − old_local_lex + new_global_lex`` mixes lexical
        # sub-scores on the FTS5 bm25() magnitude scale (0–6, FTS5_LEXICAL_CAP)
        # into ``raw``. That is only scale-coherent when ``raw`` is itself on
        # the additive/BM25 magnitude scale. Production per-shard Genomes now
        # score in RRF (``raw`` ≈ Σ 1/(k+rank) ≈ 0.0x), where subtracting a
        # BM25-scale ``old_lex`` and adding a BM25-scale ``new_lex`` swamps the
        # ~100× smaller RRF ``raw`` — the dense/tier signal in ``raw`` is
        # obliterated and the result lives on a nonsensical mixed scale. The
        # global-IDF splice is therefore UNDEFINED under per-shard RRF, so we
        # hard-gate it to additive-mode shards. When the flag is truthy but the
        # shards run RRF the router silently keeps the scalar ``m_shard`` path
        # (honest no-op) and logs once. Define rrf-native global-IDF semantics
        # (global IDF as a rank input to the per-shard fusion) to lift this.
        _flag_on = _global_idf_enabled()
        _per_shard_additive = self._per_shard_fusion_mode() == "additive"
        if _flag_on and not _per_shard_additive and len(shard_ranked) > 1:
            _warn_global_idf_rrf_suppressed()
        use_global_idf = (
            _flag_on
            and _per_shard_additive
            and len(shard_ranked) > 1
            and bool(idf_terms)
        )
        # {shard_name: {gene_id: new_global_lexical_score}} — populated only
        # under the flag. Empty dict ⇒ scalar path (flag off / 1 shard).
        global_lex: Dict[str, Dict[str, float]] = {}
        if use_global_idf:
            global_idf_map = _compute_global_idf_map(
                idf_terms, shard_n, shard_dfs,
            )
            if global_idf_map:
                for shard_name, pairs in shard_ranked.items():
                    cand_ids = [gid for gid, _raw in pairs]
                    if not cand_ids:
                        continue
                    try:
                        shard = self._shards.get(shard_name)
                        if shard is None:
                            continue
                        global_lex[shard_name] = shard.rescore_lexical_global_idf(
                            cand_ids, idf_terms, global_idf_map,
                        )
                    except Exception:
                        log.warning(
                            "shard %s global-idf rescore failed; "
                            "falling back to scalar for this shard",
                            shard_name, exc_info=True,
                        )
                        global_lex[shard_name] = {}
            else:
                # No global signal — nothing to rescore; fall back to scalar.
                use_global_idf = False

        # Apply correction. A doc that appears in multiple shards (rare —
        # would require content-hash collision across shards) keeps its
        # highest corrected score so cross-shard duplication can't penalize.
        #
        # Under the global-IDF flag, ``m_shard`` is REPLACED by the per-doc
        # lexical splice: corrected = (raw − old_local_lexical) +
        # min(new_global_lexical, cap). When the flag is off (or a shard's
        # rescore produced no entry for a doc), we use the legacy scalar
        # ``raw * m_shard`` for that doc.
        corrected: Dict[str, float] = {}
        for shard_name, pairs in shard_ranked.items():
            m = float(multipliers.get(shard_name, 1.0))
            shard_lex = global_lex.get(shard_name, {}) if use_global_idf else {}
            for gid, raw in pairs:
                if use_global_idf and gid in shard_lex:
                    # Per-doc global-IDF splice. The old local lexical
                    # sub-score is the per-doc ``fts5`` tier contribution
                    # the genome recorded (already capped at the cap), and
                    # the new lexical sub-score is the global-IDF BM25
                    # value capped to the same ceiling.
                    old_lex = float(merged_tier.get(gid, {}).get("fts5", 0.0))
                    new_lex = min(float(shard_lex[gid]), FTS5_LEXICAL_CAP)
                    adj = raw - old_lex + new_lex
                else:
                    adj = raw * m
                if gid not in corrected or adj > corrected[gid]:
                    corrected[gid] = adj

        # ── Intra-shard doc-type boost (#121, mode-gated #264) ──────
        # Lift README/CLAUDE/INDEX summary docs so a high-level doc that
        # holds the answer but lost the intra-shard keyword-density race
        # can still clear the merge truncation. Gated to the genuine
        # cross-shard merge (≥2 shards) so the single-shard fast path
        # stays byte-identical; see DOC_TYPE_BOOST commentary above.
        #
        # #264: HOW the lift is applied is now selectable via
        # ``doc_type_boost_mode`` (default-inert "additive"):
        #   "additive" — the shipped fixed ×DOC_TYPE_BOOST magnitude
        #       multiply on the IDF-corrected score. Calibrated on
        #       additive/BM25-scale margins where it only re-orders
        #       near-ties. Under per-shard RRF the margins compress to
        #       ~1.6% so the multiply becomes decisive on nearly every
        #       pair (the #264 defect) — but this is the byte-identical
        #       default, so nothing changes until an operator flips.
        #   "off" — no lift (the RRF-honest escape; the flip case
        #       resolves because the unboosted impl file already leads).
        #   "rank" — feed the eligible docs into the cross-shard Fuser
        #       as an extra rank-domain tier (below, before all_scores)
        #       instead of multiplying ``corrected``; the final sort then
        #       keys primarily on the rank-fused score. Scale-free: a
        #       summary doc can only reorder genuine RANK near-ties, never
        #       leapfrog a doc that dominates the per-shard ranks.
        rank_mode_active = (
            self._doc_type_boost_mode == "rank" and len(shard_ranked) > 1
        )
        if self._doc_type_boost_mode == "additive" and len(shard_ranked) > 1:
            for gid in corrected:
                doc = merged.get(gid)
                if doc is None:
                    continue
                boost = _doc_type_boost_for(getattr(doc, "source_id", None))
                if boost != 1.0:
                    corrected[gid] *= boost
        elif rank_mode_active:
            # Rank-domain lift: register a synthetic "doc_type" tier of the
            # boost-eligible candidates, ranked by their (unboosted) corrected
            # score, into the SAME Fuser that already carries every per-shard
            # tier. Its RRF mass (1/(k+rank)) is bounded by construction, so it
            # can promote a near-rank-tied summary doc but not overturn a
            # decisive per-shard-rank lead. ``corrected`` is left UNtouched so
            # the published additive-scale scores stay honest.
            eligible = [
                (gid, corrected[gid])
                for gid in corrected
                if _doc_type_boost_for(
                    getattr(merged.get(gid), "source_id", None)
                ) != 1.0
            ]
            if eligible:
                fuser.add_tier(
                    "doc_type", eligible, weight=DOC_TYPE_RANK_TIER_WEIGHT,
                )
        # "off" — no lift applied.

        rrf_all = fuser.all_scores()

        # -- Score-path introspection (issue #181) ------------------
        # Behaviour-preserving: ONLY runs when CYMATIX_SHARD_SCORE_DEBUG is
        # truthy. Populates last_score_breakdown for EVERY merged
        # candidate (including golds that fall below the `limit` cut, so
        # the diag harness can see the depression) plus
        # last_shard_multipliers. Reads only -- never touches `corrected`,
        # `ranked_ids`, or the return value.
        if _score_debug_enabled():
            # Re-derive the shard/raw/m that produced each gid's WINNING
            # adjusted (pre-boost) score -- mirrors the max-over-shards
            # rule in the corrected-building loop above so the attributed
            # (shard, raw, m_shard) matches the value that drove the sort.
            best_pre_boost: Dict[str, Tuple[str, float, float]] = {}
            for shard_name, pairs in shard_ranked.items():
                m = float(multipliers.get(shard_name, 1.0))
                for gid, raw in pairs:
                    adj = raw * m
                    prev = best_pre_boost.get(gid)
                    prev_adj = (prev[1] * prev[2]) if prev is not None else None
                    if prev_adj is None or adj > prev_adj:
                        best_pre_boost[gid] = (shard_name, float(raw), m)
            breakdown: Dict[str, dict] = {}
            for gid in corrected:
                shard_name, raw, m = best_pre_boost.get(gid, ("", 0.0, 1.0))
                doc = merged.get(gid)
                # #264: the reported multiplier is only folded into ``corrected``
                # under the default "additive" mode. In "rank" mode the lift is a
                # Fuser tier (reflected in ``rrf`` below, not ``corrected``); in
                # "off" mode there is no lift. Report 1.0 in those modes so the
                # ``corrected == raw * m_shard * doc_type_boost`` identity the
                # #181 diag harness asserts stays exact and honest.
                boost = (
                    _doc_type_boost_for(getattr(doc, "source_id", None))
                    if (
                        doc is not None
                        and len(shard_ranked) > 1
                        and self._doc_type_boost_mode == "additive"
                    )
                    else 1.0
                )
                row = {
                    "shard": shard_name,
                    "raw": raw,
                    "m_shard": m,
                    "doc_type_boost": boost,
                    "corrected": float(corrected.get(gid, 0.0)),
                    "rrf": rrf_all.get(gid),
                }
                # #182: ONLY when the global-IDF flag is active do we add
                # the ``global_lex`` field (the per-doc global-IDF BM25
                # lexical sub-score that REPLACED ``raw * m_shard``). On
                # the scalar (flag-off) path the breakdown shape stays
                # byte-identical to the pre-#182 build, so the
                # ``corrected == raw * m_shard * doc_type_boost`` identity
                # the #181 diag harness asserts still holds exactly.
                if use_global_idf:
                    gl = global_lex.get(shard_name, {})
                    row["global_lex"] = (
                        min(float(gl[gid]), FTS5_LEXICAL_CAP)
                        if gid in gl else None
                    )
                breakdown[gid] = row
            self.last_score_breakdown = breakdown
            self.last_shard_multipliers = {
                sn: float(multipliers.get(sn, 1.0)) for sn in shard_ranked
            }

        # Primary sort key: corrected score desc.
        # Secondary tiebreaker: RRF score desc (stable for ties in
        # the corrected score, e.g., 6 shards' rank-1 hitting the cap).
        # Tertiary: gene_id asc (deterministic).
        #
        # #264 "rank" mode inverts the primary/secondary keys: the rank-fused
        # score (now carrying the doc_type tier mass) leads, with the
        # additive-scale ``corrected`` as the tiebreaker. This is the whole
        # point of rank mode — deciding the merge in rank space so the
        # doc-type lift acts on ranks, not compressed RRF magnitudes. The
        # default ("additive") and "off" modes keep the shipped corrected-led
        # ordering byte-for-byte.
        if rank_mode_active:
            ranked_ids_full = sorted(
                corrected,
                key=lambda gid: (
                    -rrf_all.get(gid, 0.0),
                    -corrected.get(gid, 0.0),
                    gid,
                ),
            )
        else:
            ranked_ids_full = sorted(
                corrected,
                key=lambda gid: (
                    -corrected.get(gid, 0.0),
                    -rrf_all.get(gid, 0.0),
                    gid,
                ),
            )
        # Match Genome.query_docs contract: return up to ``max_genes * 2``
        # candidates so the downstream assembler (splice + co-activation +
        # freshness gate) has the same depth to work with as a blob-mode
        # genome would. Without this, the router's max_genes cap deletes
        # the deeper candidates the assembler relies on for co-activated
        # pull-forward. Mirrors ``Genome.query_docs`` line ~2288 which
        # truncates to ``limit = max_genes * 2``.
        limit = max(1, int(max_genes) * 2)
        ranked_ids = ranked_ids_full[:limit]

        # ── Cross-shard co-activation expansion (#120) ──────────────
        # Pull docs reachable via 1-hop harmonic links from the top-K
        # candidates but living in a DIFFERENT shard. Mutates ``merged``
        # / ``merged_tier`` / ``corrected`` in place and returns a
        # re-sorted, re-truncated id list. Soft-fails to ``ranked_ids``
        # unchanged so a graph hiccup never perturbs the merge result.
        try:
            ranked_ids = self._expand_cross_shard_coactivation(
                ranked_ids=ranked_ids,
                shard_ranked=shard_ranked,
                corrected=corrected,
                rrf_all=rrf_all,
                merged=merged,
                merged_tier=merged_tier,
                limit=limit,
            )
        except Exception:
            log.warning(
                "cross-shard co-activation expansion failed; "
                "falling back to un-expanded merge",
                exc_info=True,
            )

        # Surface BOTH score families. last_query_scores becomes the
        # IDF-corrected score (which is what downstream tier_logic /
        # bench harness watch). last_tier_contributions is unchanged
        # from the source shard's per-tier breakdown.
        with self._last_query_scores_lock:
            self.last_query_scores = {gid: corrected[gid] for gid in ranked_ids}
            self.last_tier_contributions = {
                gid: merged_tier.get(gid, {}) for gid in ranked_ids
            }
        return [merged[gid] for gid in ranked_ids if gid in merged]

    # ── Cross-shard co-activation expansion (#120) ──────────────────

    def _expand_cross_shard_coactivation(
        self,
        *,
        ranked_ids: List[str],
        shard_ranked: Dict[str, List[Tuple[str, float]]],
        corrected: Dict[str, float],
        rrf_all: Dict[str, float],
        merged: Dict[str, Gene],
        merged_tier: Dict[str, Dict[str, float]],
        limit: int,
    ) -> List[str]:
        """Pull 1-hop harmonic-link neighbours from across shards.

        Sharded equivalent of blob's ``_expand_coactivated``: a gold doc
        whose only path into the result is a harmonic link from a doc in
        a *different* shard never surfaces, because each shard's
        ``harmonic_links`` table is intra-shard only and the merge has no
        graph pass. This method closes that gap.

        For each candidate in ``ranked_ids`` (the post-merge top-K):

        1. Read its 1-hop ``harmonic_links`` neighbours from the DB of
           the shard that ranked it best (``fetch_forward_neighbors`` —
           ``gene_id_a = candidate``, mirroring blob's edge direction).
        2. Resolve each linked gene_id to its owning shard via
           ``fingerprint_index`` (lexicographically-min ``shard_name``
           wins on cross-shard content-hash duplicates — same tie-break
           contract as ``get_citation_rows``).
        3. Score the linked doc at ``COACT_LINK_BOOST × source-doc
           corrected score`` and merge it into ``corrected`` (highest
           score wins — a linked doc already in the candidate set is
           never re-scored *down*).
        4. Materialise newly-introduced docs into ``merged`` from their
           owning shard, re-sort the union by the same key
           ``query_genes`` uses, and truncate to ``limit``.

        ``merged`` / ``merged_tier`` / ``corrected`` are mutated in
        place. Returns the re-sorted, re-truncated id list. On any error
        the caller falls back to the un-expanded ``ranked_ids``.

        Blob-mode parity: this method is reached only from the sharded
        ``ShardRouter.query_genes`` path — blob's ``KnowledgeStore``
        retrieval never calls it, so blob behaviour is unchanged.
        """
        if not ranked_ids or not shard_ranked:
            return ranked_ids

        from .retrieval.expand import fetch_forward_neighbors

        # gene_id → owning shard. A doc can appear in several shards on a
        # content-hash collision; pick the lexicographically-min shard so
        # the resolution is deterministic (matches get_citation_rows).
        gid_to_shard: Dict[str, str] = {}
        for shard_name, pairs in shard_ranked.items():
            for gid, _score in pairs:
                cur = gid_to_shard.get(gid)
                if cur is None or shard_name < cur:
                    gid_to_shard[gid] = shard_name

        # Candidate set we expand from — the post-merge top-K only, so
        # the fan-out cost is bounded by ``limit`` 1-hop reads.
        existing: set[str] = set(ranked_ids)

        # linked gene_id → best discounted score proposed for it.
        linked_scores: Dict[str, float] = {}
        for src_gid in ranked_ids:
            src_shard = gid_to_shard.get(src_gid)
            if src_shard is None:
                continue
            try:
                shard = self._open_shard(src_shard)
            except Exception:
                log.debug(
                    "co-activation: shard %s failed to open for %s",
                    src_shard, src_gid, exc_info=True,
                )
                continue
            try:
                neighbors = fetch_forward_neighbors(
                    shard.conn, src_gid, k=COACT_MAX_LINKS_PER_DOC,
                )
            except Exception:
                log.debug(
                    "co-activation: harmonic_links read failed for %s",
                    src_gid, exc_info=True,
                )
                continue
            src_score = float(corrected.get(src_gid, 0.0))
            if src_score <= 0.0:
                continue
            boosted = src_score * self._coact_link_boost
            for linked_gid, _weight in neighbors:
                # Already a candidate — never re-score it down; the
                # existing (higher) merge score stands.
                if linked_gid in existing:
                    continue
                prev = linked_scores.get(linked_gid)
                if prev is None or boosted > prev:
                    linked_scores[linked_gid] = boosted

        if not linked_scores:
            return ranked_ids

        # Resolve linked gene_ids to their owning shards via main.db's
        # fingerprint_index. A linked doc with no fingerprint row (e.g.
        # an unsharded / dangling reference) is skipped.
        linked_ids = list(linked_scores.keys())
        id_ph = ",".join("?" * len(linked_ids))
        try:
            fp_rows = self.main_conn.execute(
                f"SELECT gene_id, shard_name FROM fingerprint_index "
                f"WHERE gene_id IN ({id_ph})",
                linked_ids,
            ).fetchall()
        except Exception:
            log.debug(
                "co-activation: fingerprint_index resolve failed",
                exc_info=True,
            )
            return ranked_ids

        linked_gid_to_shard: Dict[str, str] = {}
        for r in fp_rows:
            gid = r["gene_id"]
            sn = r["shard_name"]
            cur = linked_gid_to_shard.get(gid)
            if cur is None or sn < cur:
                linked_gid_to_shard[gid] = sn

        # Materialise the linked docs from their owning shards. Group by
        # shard so each shard DB is queried once.
        by_shard: Dict[str, List[str]] = {}
        for gid, sn in linked_gid_to_shard.items():
            by_shard.setdefault(sn, []).append(gid)

        promoted: set[str] = set()
        for shard_name, gids in by_shard.items():
            try:
                shard = self._open_shard(shard_name)
            except Exception:
                log.debug(
                    "co-activation: shard %s failed to open for fetch",
                    shard_name, exc_info=True,
                )
                continue
            for gid in gids:
                if gid in merged:
                    # Already materialised by the main fan-out — still
                    # eligible to be re-ranked in via its linked score.
                    promoted.add(gid)
                    continue
                try:
                    doc = shard.get_doc(gid)
                except Exception:
                    log.debug(
                        "co-activation: get_doc(%s) failed", gid,
                        exc_info=True,
                    )
                    doc = None
                if doc is None:
                    continue
                merged[gid] = doc
                merged_tier.setdefault(gid, {})["co_activation"] = (
                    linked_scores[gid]
                )
                promoted.add(gid)

        if not promoted:
            return ranked_ids

        # Fold the discounted scores into ``corrected`` (highest wins),
        # then re-sort the union with the SAME key query_genes uses and
        # re-truncate to ``limit``.
        for gid in promoted:
            adj = linked_scores.get(gid, 0.0)
            if gid not in corrected or adj > corrected[gid]:
                corrected[gid] = adj

        union_ids = list(dict.fromkeys(list(ranked_ids) + sorted(promoted)))
        union_ids.sort(
            key=lambda gid: (
                -corrected.get(gid, 0.0),
                -rrf_all.get(gid, 0.0),
                gid,
            ),
        )
        # Issue #223: optionally reserve slots for promoted linked docs so
        # the link-discounted graph-surfaced candidates aren't displaced by
        # the flat truncation. The reserve count is config-threaded
        # (coact_reserved_slots; CYMATIX_SHARD_COACT_RESERVE overrides).
        # Reserve=0 (default) is byte-identical to the legacy plain [:limit]
        # cut.
        return _apply_coact_reserve(
            union_ids, promoted, corrected, rrf_all,
            limit, self._resolved_coact_reserve(),
        )

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        """Close all lazy-opened shard knowledge stores + main.db."""
        for shard_name, genome in self._shards.items():
            # A5 (RAM): call genome.close() — it runs checkpoint(TRUNCATE) to
            # flush + truncate the WAL before closing both connections. The
            # old path closed conn/_reader directly, skipping the checkpoint,
            # leaving up to 64 MB of un-truncated WAL per shard on disk.
            try:
                genome.close()
            except Exception:
                log.warning("failed to close shard %s", shard_name, exc_info=True)
        self._shards.clear()
        try:
            self.main_conn.close()
        except Exception:
            pass


# ── Feature-flag helper ──────────────────────────────────────────────


def use_shards_enabled() -> bool:
    """Read CYMATIX_USE_SHARDS env flag. Default OFF until Task 8 cutover."""
    return os.environ.get("CYMATIX_USE_SHARDS", "").strip() in ("1", "true", "yes", "on")

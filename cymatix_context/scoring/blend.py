"""
Scoring blend: apply cymatics + harmonic bin + TCM as post-retrieve refiners.

Extracted from ``context_manager.py`` (Sprint refactor, 2026-05). The
``blend_mode="legacy"`` path is byte-identical to the inline
``_apply_candidate_refiners`` method it replaces -- only the calling
convention changed (explicit parameters instead of ``self``).

Blend-mode knob (Issue #255 / audit §4 item 5, 2026-07-10)
----------------------------------------------------------
The blend layer runs three post-fusion refiners over the candidate set:

* **cymatics** — spectral flux score, scaled by an absolute ``0.5``
  (``blend.py`` cymatics block; the ``* 0.5`` literal). Adds to
  ``genome.last_query_scores`` and re-sorts candidates.
* **harmonic_bin** — overtone-series boost, already normalized to ``[0, 1.5]``
  by ``ray_trace.harmonic_bin_boost`` (the ``w * 1.5`` at ``ray_trace.py:485``).
  Adds to ``genome.last_query_scores`` and re-sorts candidates.
* **TCM** — temporal-context bonus, ``max(0, 0.3 * sim)`` (``tcm.BONUS_WEIGHT``
  ``= 0.3``, bound here as ``weight=0.3``). Does NOT mutate
  ``last_query_scores``; it only re-sorts candidates by ``score + tcm_bonus``.

Every one of these is a *post-fusion mutation on an unspecified score scale*
(audit §2d, class d): ``0.5`` is a mild nudge on the legacy additive scale
(O(3-45)) but an overwrite on the RRF scale (O(0.05-0.5)). They also
contaminate the ``[know]`` logistic, which reads ``top_score``/``score_gap``
straight off the mutated ``last_query_scores`` map. This module exposes the
three modes below. GRADUATED 2026-07-13: ``"scale_relative"`` is now the
shipped default (serving-profile receipt,
``docs/research/2026-07-12-blend-serving-receipt.md``); ``"legacy"`` remains
available and byte-identical to the pre-knob inline block for anyone who sets
it explicitly.

``legacy``
    Absolute additive blend. ``score[g] += bonus`` for cymatics/harmonic;
    TCM sorts by ``score[g] + tcm_bonus[g]``. Byte-identical to the pre-knob
    inline block, so untouched configs are bit-for-bit unchanged.

    DEPRECATED-FOR-REMOVAL (2026-07-13, 3-member council CONCUR 3/3 on the
    serving-profile receipt above): explicit selection (TOML or kwarg) now
    logs a one-time deprecation warning at config load
    (``RetrievalConfig.__post_init__``). Removal is condition-gated, not
    calendar-gated: (a) a validated k-seed-hardened [know] re-fit under
    scale_relative merged to master (#239 — shipped betas were fit under
    legacy, so legacy remains the rollback path for the confidence contract
    until then); (b) the last legacy-capable release is tagged for
    historical bench repro (all pre-2026-07 receipts are legacy-blend);
    (c) the IBM baseline is banked on scale_relative with no know/abstain
    anomaly; (d) the removal PR makes explicit legacy selection fail fast
    with a migration hint. Target v(N+2) at earliest.

``scale_relative``
    Convert each absolute additive bonus ``b`` into a **bounded multiplier**
    of the candidate's own score. For a signal with absolute cap ``C``
    (cymatics ``0.5``, harmonic ``1.5``, TCM ``0.3``) and a reference additive
    score ``S_REF`` (``_BLEND_SCALE_REF``)::

        signal_norm = clamp(b / C, 0, 1)          # ∈ [0, 1], scale-free
        w           = C / S_REF                    # preserves the signal's
                                                   #   relative magnitude at a
                                                   #   typical additive score
        score[g]   *= (1 + w * signal_norm)        # == 1 + clamp(b, 0, C)/S_REF

    The ``C`` cancels for in-range ``b``, so the mapping reduces to the clean
    form ``score[g] *= (1 + b / S_REF)``: an absolute bonus ``b`` scales the
    candidate's own score by the fraction ``b / S_REF`` it *would* have
    represented on a typical additive score, instead of adding a fixed amount.
    Because every bonus is independent of the input score magnitudes and the
    map is multiplicative, the emitted order is **invariant under uniform
    rescale of the input scores** (multiplying every score by ``c > 0`` scales
    every product by the same ``c`` — order preserved). That is the invariance
    the additive form breaks: a fixed ``+0.5`` flips rankings as the score
    scale shifts (the very defect audit §2d flags). The relative magnitudes of
    the three signals are preserved: with ``S_REF = 10.0`` the per-signal max
    boosts are cymatics ``×1.05``, harmonic ``×1.15``, TCM ``×1.03`` — the same
    ``0.5 : 1.5 : 0.3`` ratio as the additive constants, now bounded.
    ``S_REF = 10.0`` is a representative mid-band additive score (audit §2d
    cites additive tier scores as O(3-45); 10.0 sits inside that band). It is
    a documented module constant, not a per-signal hand-tuned exchange rate.

``off``
    Skip the blend mutations of ``last_query_scores`` entirely: none of the
    three refiners run their scoring/sort step, so ``last_query_scores`` is
    left exactly as fusion produced it and candidates keep their pure-fused
    order. The refiners' one **non-scoring** side effect — the rerank/model
    truncation of the candidate list to ``max_genes`` — still runs. This
    clears the desk-test off-cell exact-inversion floor
    (``docs/research/2026-07-10-rerank-combinator-desktest.md`` §5): that floor
    is exactly this blend layer mutating order under the lexical probe.

Note on ``[know]`` (audit item 7): the ``[know]`` confidence logistic reads
``top_score``/``score_gap`` off ``last_query_scores`` (``context_packet.py``).
``blend_mode="off"``/``"scale_relative"`` therefore change what ``[know]``
sees; this knob does NOT touch ``[know]`` behavior itself. Graduation landed
2026-07-13 (serving-profile receipt,
``docs/research/2026-07-12-blend-serving-receipt.md``), so the s_ref/g_ref
re-fit that was sequenced BEHIND this knob (audit §4 item 7) is now unblocked
— tracked on #239, and must run under the new ``scale_relative`` default.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..genome import Genome
    from ..scoring.tcm import TCMSession
    from ..schemas import Gene

log = logging.getLogger(__name__)

__all__ = ["apply_candidate_refiners", "VALID_BLEND_MODES"]

# The three valid blend modes. ``CymatixContextManager.__init__`` validates the
# configured value against this same set so a typo in cymatix.toml fails fast at
# construction.
VALID_BLEND_MODES: Tuple[str, ...] = ("legacy", "scale_relative", "off")

# scale_relative reference additive score. A max-cap bonus C becomes a
# multiplier boost of C / _BLEND_SCALE_REF (cymatics ×1.05, harmonic ×1.15,
# TCM ×1.03). 10.0 is a representative mid-band additive tier score (audit §2d
# cites O(3-45)); see the module docstring for the full derivation.
_BLEND_SCALE_REF: float = 10.0

# Per-signal absolute caps — the additive-scale magnitudes audit §2d lists.
# Used only by scale_relative to normalize each bonus into signal_norm ∈ [0,1].
_CYMATICS_ABS: float = 0.5
_HARMONIC_ABS: float = 1.5
_TCM_ABS: float = 0.3


def _scale_relative_multiplier(bonus: float, abs_cap: float) -> float:
    """Bounded multiplier for an absolute additive blend bonus.

    ``signal_norm = clamp(bonus / abs_cap, 0, 1)``; ``w = abs_cap / S_REF``;
    returns ``1 + w * signal_norm`` (== ``1 + clamp(bonus, 0, abs_cap)/S_REF``).
    Non-positive bonuses map to a no-op ``1.0`` multiplier (the rerank signals
    here are non-negative in practice — cymatics flux ≥ 0, harmonic ∈ [0, 1.5],
    TCM ``max(0, ...)``). See the module docstring for the invariance argument.
    """
    if abs_cap <= 0.0:
        return 1.0
    signal_norm = bonus / abs_cap
    if signal_norm <= 0.0:
        return 1.0
    if signal_norm > 1.0:
        signal_norm = 1.0
    return 1.0 + (abs_cap / _BLEND_SCALE_REF) * signal_norm


def apply_candidate_refiners(
    query: str,
    candidates: List[Gene],
    max_genes: int,
    *,
    genome: Genome,
    cymatics_enabled: bool = True,
    cymatics_peak_width: float = 1.55,  # #357: shipped effective width (aggressiveness_to_peak_width(0.3)); the manager always passes its configured/derived value explicitly
    cymatics_distance_metric: str = "cosine",
    synonym_map: Optional[Dict] = None,
    use_cymatics: bool = True,
    use_harmonic_bin: bool = True,
    use_tcm: bool = True,
    tcm_session: Optional[TCMSession] = None,
    ray_trace_theta: bool = False,
    theta_weight: float = 1.0,
    blend_mode: str = "scale_relative",
    query_scores: Optional[Dict[str, float]] = None,
) -> Tuple[List[Gene], Dict[str, Dict[str, float]]]:
    """Apply post-retrieve candidate refiners before assembly or fingerprinting.

    Returns ``(candidates, refiner_contrib)`` where *refiner_contrib* maps
    gene_id -> {refiner_name: bonus}.

    *query_scores* is the caller's request-scoped score map (2026-07-18
    bugbash: score/result atomicity). When provided, the refiners read and
    mutate THIS dict instead of whatever ``genome.last_query_scores``
    currently holds — a concurrent request republishing the shared map
    between retrieval and refinement can no longer cross-wire scores.
    A copy of the mutated map is published under the store's score lock
    for legacy readers. ``None`` (default) reads a locked snapshot of
    the published map for direct callers.

    *blend_mode* selects how the three refiners combine with the fused scores;
    see the module docstring. Default agrees with ``RetrievalConfig.blend_mode``
    (``"scale_relative"`` since the 2026-07-13 graduation; #256 lesson — layer
    defaults must agree). ``"legacy"`` is still available and byte-identical to
    the pre-knob inline block for callers that set it explicitly.

    Raises:
        ValueError: on an unknown *blend_mode* (defensive — the manager
            validates at construction, so this only trips on a direct
            mis-call).
    """
    if blend_mode not in VALID_BLEND_MODES:
        raise ValueError(
            "blend_mode must be one of "
            f"{VALID_BLEND_MODES}; got {blend_mode!r}"
        )
    _off = blend_mode == "off"
    _scale = blend_mode == "scale_relative"

    def _working_scores() -> Dict[str, float]:
        if query_scores is not None:
            return query_scores
        lock = getattr(genome, "_last_query_scores_lock", None)
        if lock is not None:
            with lock:
                return dict(genome.last_query_scores or {})
        return dict(genome.last_query_scores or {})

    def _publish_scores(scores: Dict[str, float]) -> None:
        lock = getattr(genome, "_last_query_scores_lock", None)
        if lock is not None:
            with lock:
                genome.last_query_scores = dict(scores)
        else:
            genome.last_query_scores = dict(scores)

    refiner_contrib: Dict[str, Dict[str, float]] = {}

    if use_cymatics and cymatics_enabled and len(candidates) > 1 and not _off:
        try:
            from .cymatics import (
                query_spectrum, cached_doc_spectrum,
                flux_score_dispatch, build_weight_vector,
            )
            q_spec = query_spectrum(
                query, synonym_map=synonym_map,
                peak_width=cymatics_peak_width,
            )
            weights = build_weight_vector(
                query, synonym_map=synonym_map,
                peak_width=cymatics_peak_width,
            )
            scores = _working_scores()
            for doc in candidates:
                g_spec = cached_doc_spectrum(doc, peak_width=cymatics_peak_width)
                bonus = flux_score_dispatch(q_spec, g_spec, weights, cymatics_distance_metric) * 0.5
                if bonus:
                    refiner_contrib.setdefault(doc.gene_id, {})["cymatics"] = bonus
                if _scale:
                    scores[doc.gene_id] = scores.get(doc.gene_id, 0) * _scale_relative_multiplier(bonus, _CYMATICS_ABS)
                else:
                    scores[doc.gene_id] = scores.get(doc.gene_id, 0) + bonus
            _publish_scores(scores)
            candidates.sort(key=lambda g: scores.get(g.gene_id, 0), reverse=True)
        except Exception:
            log.debug("Cymatics blend failed", exc_info=True)

    if len(candidates) > max_genes:
        # Issue #341 Task 7: the post-cap rerank branch that used to live
        # here (an attribute-presence check on the ribosome object) was
        # dead code -- it never matched DeBERTaRibosome (which only
        # defines re_rank) and misrouted to the LLM Compressor.rerank for
        # ollama/litellm backends. The cross-encoder now runs pre-cap
        # inside the store (the "5b" block in KnowledgeStore.query_docs_ann,
        # mirrored in query_docs); the per-class flag resolution it uses is
        # retrieval/rerank_combinators.resolve_class_flag. This is a plain
        # truncation.
        candidates = candidates[:max_genes]

    if use_harmonic_bin and len(candidates) >= 3 and not _off:
        try:
            from .ray_trace import harmonic_bin_boost
            seed_ids = [g.gene_id for g in candidates[:3]]
            velocity = None
            theta_w = 1.0
            if (
                ray_trace_theta
                and tcm_session is not None
                and tcm_session.depth >= 2
            ):
                velocity = list(tcm_session.context_vector)
                theta_w = theta_weight
            overtones = harmonic_bin_boost(
                seed_ids,
                genome,
                k_rays=100,
                max_bounces=2,
                velocity_vector=velocity,
                theta_weight=theta_w,
            )
            if overtones:
                scores = _working_scores()
                for doc in candidates:
                    if doc.gene_id in overtones:
                        bonus = overtones[doc.gene_id]
                        refiner_contrib.setdefault(doc.gene_id, {})["harmonic_bin"] = bonus
                        if _scale:
                            scores[doc.gene_id] = scores.get(doc.gene_id, 0) * _scale_relative_multiplier(bonus, _HARMONIC_ABS)
                        else:
                            scores[doc.gene_id] = scores.get(doc.gene_id, 0) + bonus
                _publish_scores(scores)
                candidates.sort(key=lambda g: scores.get(g.gene_id, 0), reverse=True)
        except Exception:
            log.debug("Harmonic bin boost failed", exc_info=True)

    if use_tcm and tcm_session is not None and tcm_session.depth > 0 and not _off:
        try:
            from .tcm import tcm_bonus
            bonuses = tcm_bonus(tcm_session, candidates, weight=0.3)
            for gid, bonus in bonuses.items():
                if bonus:
                    refiner_contrib.setdefault(gid, {})["tcm"] = bonus
            scores = _working_scores()
            if _scale:
                candidates.sort(
                    key=lambda g: scores.get(g.gene_id, 0) * _scale_relative_multiplier(bonuses.get(g.gene_id, 0.0), _TCM_ABS),
                    reverse=True,
                )
            else:
                candidates.sort(
                    key=lambda g: scores.get(g.gene_id, 0) + bonuses.get(g.gene_id, 0),
                    reverse=True,
                )
        except Exception:
            pass

    return candidates, refiner_contrib

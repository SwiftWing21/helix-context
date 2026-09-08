"""ablation_ladder.py — in-process per-layer ablation probe for the ERB beds.

One retrieval layer is disabled per arm against a ``baseline`` arm that runs
the shipped config untouched. The point is a cost/impact re-probe: what does
each layer buy in rank quality, and what does it cost in wall time?

Every arm is built from a FRESH ``load_config()`` object mutated for that arm
and a FRESH ``CymatixContextManager`` (config is per-manager, and several
gates are read once at construction into ``KnowledgeStore`` attributes — a
mid-run mutation would not take).

────────────────────────────────────────────────────────────────────────
Verified query-time gates (file:line, this worktree)
────────────────────────────────────────────────────────────────────────

Every knob below was traced from ``config.py`` through manager construction to
the *branch that actually runs at query time*. Arms whose candidate knob had
no such branch are DROPPED, with the evidence recorded in the receipt under
``dropped_arms`` (and repeated below).

KEPT
  no_splade      [ingestion] splade_enabled
                 config.py:271 -> context_manager.py:965 (splade_enabled=)
                 -> knowledge_store.py:667 (self._splade_enabled)
                 -> knowledge_store.py:2783 `if self._splade_enabled:` (Tier 3.5)
                 signal key: 'splade' (knowledge_store.py:2842)

  no_dense       [retrieval] dense_embedding_enabled
                 config.py:478 -> context_manager.py:986
                 -> knowledge_store.py:710 (self._dense_embedding_enabled)
                 -> context_manager.py:2946 (ANN vs plain query_docs branch)
                 -> knowledge_store.py:2972 (Tier 6 dense) and :3757 (dense recall)
                 signal keys: 'dense' (knowledge_store.py:3040),
                              'ann_dense_leg' (knowledge_store.py:3906)

  no_sema        [ingestion] sema_embed_on_ingest  (#227 semantics: this is the
                 GLOBAL sema switch, not merely an ingest-write flag — it gates
                 codec *materialisation*, so the query side dies with it)
                 config.py:291 -> context_manager.py:903-935 (self._sema_codec
                 stays None when false) -> context_manager.py:953 (sema_codec=)
                 -> knowledge_store.py:665 -> knowledge_store.py:2848
                 `if self._sema_codec is not None:` (Tier 4 A/B)
                 signal key: 'sema_boost' (knowledge_store.py:2894)

  no_cymatics    [cymatics] enabled
                 config.py:390 -> context_manager.py:1216 (self._use_cymatics)
                 -> context_manager.py:3119 (cymatics_enabled=)
                 -> scoring/blend.py:212 `if use_cymatics and cymatics_enabled ...`
                 NO signal key (the blend runs in the manager, not the store).
                 Validated by a call-probe on scoring.cymatics.query_spectrum,
                 which blend.py imports at call time (blend.py:214).

  no_synonyms    [synonyms] map emptied (config.synonym_map)
                 config.py:1097 -> context_manager.py:952 (synonym_map=)
                 -> knowledge_store.py:664 (self.synonym_map)
                 -> knowledge_store.py:1773-1784 `_expand_terms`, called first
                 thing in query_genes (knowledge_store.py:2305-2306).
                 NO signal key. Validated by probing ``_expand_terms`` directly
                 with the baseline map's own keys.

  rerank_on /    [retrieval] rerank_enabled   (#341)
  rerank_off     config.py:778 -> context_manager.py:1018 (rerank_enabled=)
                 -> knowledge_store.py:845 (self._rerank_enabled)
                 -> knowledge_store.py:4281 `if self._rerank_effective(
                 rerank_override) and scored:` (ANN path, the one the
                 receipts use) and :3692 `if _rerank_on and ranked_ids:`
                 (lex path).
                 signal key: 'xenc_rerank' (knowledge_store.py:4308 ANN,
                 :3734 lex)
                 These are a PAIR, and only one of them is ever live for a
                 given config: rerank_on is an ENABLE arm (the shipped
                 default is False, so "ablating" it is a no-op) validated by
                 signal PRESENCE, rerank_off is the ordinary ablation
                 validated by signal absence. Whichever one matches the
                 loaded config's current value is identity-dropped at
                 runtime, which is exactly what makes the order-reversed A/B
                 (benchmarks/dogfood/erb/rerank_ab.py) safe to drive from a
                 single arm table.

  no_classifier  [classifier] enabled
                 config.py:886 -> context_manager.py:1801-1807 (classify_query
                 only called when enabled) -> assembly cap at :2101-2113 and
                 decoder hint at :1829.
                 NO signal key. Validated by the absence of
                 ``window.metadata["classifier"]`` (written at
                 context_manager.py:2316 only when classifier_result is not None).

CONDITIONAL — kept in the arm table, dropped at RUNTIME when the effective
config already sits at the ablated value (an ablation that changes nothing
measures nothing, and reporting it as a null result would be a lie):
  no_cold_tier   [context] cold_tier_enabled -> context_manager.py:3013-3018
  no_sr          [retrieval] sr_enabled -> knowledge_store.py:3162
  no_seeded_edges [retrieval] seeded_edges_enabled -> knowledge_store.py:3462
On the shipped cymatix.toml all three are already ``false`` (cymatix.toml:311,
:349, :359), so all three drop. The check is done against the LOADED config,
not against my reading of the toml, so a future flip re-arms them for free.

DROPPED (no query-time gate — see ``dropped_arms`` in the receipt)
  no_plr         [plr] enabled is consumed ONLY by the HTTP layer:
                 server/routes_context.py:635 -> server/helpers.py:444
                 ``_compute_plr_confidence``. retrieval/fusion_plr.py:31 says it
                 "leaves document ranking untouched" — it decorates the
                 /context/packet response with a query-quality log-odds. It is
                 not on the ``build_context`` path at all, so an in-process arm
                 would measure exactly zero by construction.

  no_harmonic    ``use_harmonic`` is a hardcoded ``True`` default on
                 context_manager.py:2895 (``_retrieve``) and
                 knowledge_store.py:2259 (``query_docs``), and NEITHER call site
                 (context_manager.py:1945, :1965) passes it — so it is a dead
                 parameter, never fed from config. The Tier-5 harmonic block
                 (knowledge_store.py:3119) and the co-activation pull-forward
                 (``_expand_coactivated``, knowledge_store.py:3494) are
                 therefore unconditional. ``[cymatics] harmonic_links`` looks
                 like the gate but is not: context_manager.py:2353 shows it
                 gates the ingest/expression-time edge WRITE, inside a
                 ``if not read_only`` block, not the query-time read.
                 No clean off-switch => dropped.

  no_tag_*       Tier 1 (tag_exact, knowledge_store.py:2697) and Tier 2
                 (tag_prefix, knowledge_store.py:2720) run unconditionally.
                 The #327 knobs are DISCIPLINE knobs whose ablated state is
                 already the default — ``tag_idf_enabled`` (config.py:660,
                 default False) and ``tag_df_cap`` (config.py:668, default 0.0)
                 turn discipline ON, so flipping them is an intervention, not
                 an ablation. The only mute available is weight-zeroing
                 (``tag_prefix_weight = 0.0``), which silences the tier's
                 contribution while still paying its SQL cost — that is not a
                 layer ablation and would misreport the cost axis. Dropped.

────────────────────────────────────────────────────────────────────────
Self-validation
────────────────────────────────────────────────────────────────────────

An ablation that silently failed to apply is worse than no measurement: it
reports the layer as worthless. So every arm carries a positive observable and
is checked twice —

  1. The arm's expected observable must be ABSENT in the arm.
  2. The same observable must be PRESENT in the baseline arm. Without this the
     arm is *unfalsifiable* (e.g. 'hebbian' never fires under read_only=True,
     so "no hebbian" proves nothing), and it is recorded arm_valid=false with
     reason ``unfalsifiable`` rather than counted as a clean ablation.

``VALIDATION_SIGNAL_PRESENT`` (#341) is the same contract with both halves
inverted, for arms that ENABLE a layer instead of ablating one: the observable
must be PRESENT in the arm and ABSENT at baseline. An enable arm cannot be
validated by any of the absence kinds — "xenc_rerank never appeared" and "the
knob never applied" are the same observation there, so an enable arm checked
for absence is unfalsifiable by construction.

Baseline therefore always runs, even if ``--arms`` omits it.

────────────────────────────────────────────────────────────────────────
Measurement caveats (on the record)
────────────────────────────────────────────────────────────────────────

* Ranks come from ``manager.genome.last_query_scores`` — the same source
  ``benchmarks/dogfood/run_ladder.py`` uses. That map is POST-blend: the blend
  layer republishes it (scoring/blend.py:233, :281), so cymatics / harmonic-bin
  mutations ARE reflected. It is not, however, the delivered slice, so the
  receipt also records ``delivered_gold`` from ``window.expressed_gene_ids``.
* ``last_query_scores`` is a SCORE map, and a reorder that does not rewrite
  scores is invisible in it — which is exactly what the cross-encoder does
  (#341: it permutes ids, it does not republish fused scores). So the receipt
  also records the FINAL ORDER basis from ``manager.genome.last_ranked_ids``
  (``final_rank_of_first_gold`` / ``final_recall_hit`` per needle,
  ``final_recall_at_k`` per arm). Both bases ship: score-rank is the
  comparable-to-history one, final-rank is the one a rerank arm can move.
  Depth semantics differ by retrieval path — ANN publishes the full union
  pool, the lex path publishes the post-cut list (knowledge_store.py:940-949)
  — and these A/Bs run dense-on, i.e. the ANN semantics.
* ``last_ranked_ids`` / ``last_rerank_diag`` are store attributes with
  process lifetime, so run_arm CLEARS them immediately before each
  ``build_context`` call. A needle whose query never reaches the publication
  point (no-match early return, abstain, exception) then reads ``None``
  instead of inheriting the previous needle's order — the same staleness trap
  the splice ring-mark guards against.
* Ties are broken by ``gene_id`` ascending, not dict insertion order, so ranks
  are reproducible across arms and processes.
* Timings are wall ms on a shared box. Treat them as within-run relatives; no
  absolute-ms claim is made or asserted anywhere.
* The bed is opened read-write by ``KnowledgeStore`` construction (index/table
  DDL is idempotent-on-open), and queries pass ``read_only=True`` so
  ``touch_genes`` / ``link_coactivated`` / the Hebbian edge update never run
  (context_manager.py:2348, knowledge_store.py:3462). ``cwola_log`` is written
  by the HTTP route, not by ``build_context``, so an in-process run appends
  nothing there.
* The encoder daemon is the CALLER's business. This script never starts or
  stops one; it records ``encoder_client.active_url()`` and the dense codec's
  concrete class per arm so the receipt says which side the encode ran on.

Usage::

    python benchmarks/dogfood/erb/ablation_ladder.py --stamp 20260805-1200
    python benchmarks/dogfood/erb/ablation_ladder.py --arms baseline,no_dense
    python benchmarks/dogfood/erb/ablation_ladder.py --per-query   # paired basis
    python benchmarks/dogfood/erb/ablation_ladder.py --arms rerank_on  # #341

For the #341 rerank question specifically, drive this script through
``benchmarks/dogfood/erb/rerank_ab.py`` rather than by hand: it runs the
ladder twice with the base value reversed (so the ON side is not always the
second arm) and proves from the encoder daemon's own access log that the
cross-encoder really ran out-of-process.

``--per-query`` closes an audit gap: per-arm aggregates alone cannot tell you
WHICH needle moved between two arms, so no paired test can be run from the
receipt. With the flag, each arm carries one record per needle (gold ranks,
recall contribution, wall ms, per-signal ms) in the run order every arm shares.
Off by default, and off means the key is absent — not present-and-empty — so
existing receipts and readers are unaffected.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Retrieval-only probe: never let a query mutate the bed's learning state.
# Must be set BEFORE any manager is constructed.
os.environ.setdefault("CYMATIX_DISABLE_LEARN", "1")


# ── Arm table ────────────────────────────────────────────────────────────

VALIDATION_CONTROL = "control"
VALIDATION_SIGNAL = "signal_absent"
# #341: the ENABLE direction — the observable must appear in the arm and be
# absent at baseline. Every other kind here can only prove absence, which
# makes an enable arm unfalsifiable (see the Self-validation section).
VALIDATION_SIGNAL_PRESENT = "signal_present"
VALIDATION_TIER = "tier_contrib_absent"
VALIDATION_METADATA = "classifier_metadata_absent"
VALIDATION_EXPANSION = "synonym_expansion_absent"
VALIDATION_CALL_PROBE = "cymatics_call_probe"
VALIDATION_COLD = "cold_tier_flag_absent"


@dataclass(frozen=True)
class Arm:
    """One ablation arm.

    ``knobs`` maps dotted config paths to the ablated value. ``gate`` is the
    verified file:line of the query-time branch the knob controls — recorded
    in the receipt so a reader can re-check the claim without re-deriving it.
    """

    name: str
    knobs: Mapping[str, Any] = field(default_factory=dict)
    gate: str = ""
    validation: str = VALIDATION_SIGNAL
    # Observables expected to vanish. Interpreted per ``validation`` kind.
    expect_absent: Tuple[str, ...] = ()
    # #341: observables expected to APPEAR (VALIDATION_SIGNAL_PRESENT only).
    # An arm sets one or the other, never both — the two directions are
    # different arms, not two halves of one.
    expect_present: Tuple[str, ...] = ()


ARMS: Tuple[Arm, ...] = (
    Arm(
        name="baseline",
        knobs={},
        gate="—  (shipped config, untouched)",
        validation=VALIDATION_CONTROL,
    ),
    Arm(
        name="no_splade",
        knobs={"ingestion.splade_enabled": False},
        gate="knowledge_store.py:2783 `if self._splade_enabled:`",
        validation=VALIDATION_SIGNAL,
        expect_absent=("splade",),
    ),
    Arm(
        name="no_pki",
        knobs={"retrieval.pki_enabled": False},
        gate=("knowledge_store.py `if q_lower_tokens and self._pki_enabled:` "
              "(Tier 0 path-key compound index)"),
        validation=VALIDATION_SIGNAL,
        expect_absent=("pki",),
    ),
    Arm(
        name="no_splade_no_pki",
        knobs={
            "ingestion.splade_enabled": False,
            "retrieval.pki_enabled": False,
        },
        gate="both gates above (B3 combined arm, issues #333/#334)",
        validation=VALIDATION_SIGNAL,
        expect_absent=("splade", "pki"),
    ),
    Arm(
        name="no_dense",
        knobs={"retrieval.dense_embedding_enabled": False},
        gate=("knowledge_store.py:2972 / :3757 `if self._dense_embedding_enabled` "
              "+ context_manager.py:2946 ANN-branch selector"),
        validation=VALIDATION_SIGNAL,
        expect_absent=("dense", "ann_dense_leg"),
    ),
    Arm(
        name="no_sema",
        knobs={"ingestion.sema_embed_on_ingest": False},
        gate=("context_manager.py:903-935 (codec never materialised) -> "
              "knowledge_store.py:2848 `if self._sema_codec is not None:`"),
        validation=VALIDATION_SIGNAL,
        expect_absent=("sema_boost",),
    ),
    Arm(
        name="no_cold_tier",
        knobs={"context.cold_tier_enabled": False},
        gate="context_manager.py:3013-3018 cold_enabled gate",
        validation=VALIDATION_COLD,
        expect_absent=("cold_tier_used",),
    ),
    Arm(
        name="no_sr",
        knobs={"retrieval.sr_enabled": False},
        gate="knowledge_store.py:3162 `sr_enabled = self._sr_enabled if use_sr is None`",
        validation=VALIDATION_TIER,
        expect_absent=("sr",),
    ),
    Arm(
        name="no_seeded_edges",
        knobs={"retrieval.seeded_edges_enabled": False},
        gate=("knowledge_store.py:3462 `if self._seeded_edges_enabled and "
              "ranked_ids and not read_only:`"),
        validation=VALIDATION_SIGNAL,
        expect_absent=("hebbian",),
    ),
    Arm(
        name="no_cymatics",
        knobs={"cymatics.enabled": False},
        gate=("context_manager.py:1216 -> :3119 -> scoring/blend.py:212 "
              "`if use_cymatics and cymatics_enabled ...`"),
        validation=VALIDATION_CALL_PROBE,
        expect_absent=("scoring.cymatics.query_spectrum",),
    ),
    Arm(
        name="no_synonyms",
        knobs={"synonym_map": {}},
        gate="knowledge_store.py:1773-1784 `_expand_terms` (called at :2305-2306)",
        validation=VALIDATION_EXPANSION,
        expect_absent=("synonym_expansion",),
    ),
    # #341 rerank pair. Exactly one of these survives ``arm_is_identity`` for
    # any given config, so both can sit in the table unconditionally: with the
    # shipped default (rerank_enabled=False) rerank_on is live and rerank_off
    # drops; with a rerank-on config the reverse. rerank_ab.py drives both
    # directions by pointing each run at the matching config.
    Arm(
        name="rerank_on",
        knobs={"retrieval.rerank_enabled": True},
        gate=("knowledge_store.py:4281 `if self._rerank_effective(rerank_override) "
              "and scored:` (ANN) / :3692 (lex) — pre-gate cross-encoder (#341)"),
        validation=VALIDATION_SIGNAL_PRESENT,
        expect_present=("xenc_rerank",),
    ),
    Arm(
        name="rerank_off",
        knobs={"retrieval.rerank_enabled": False},
        gate=("knowledge_store.py:4281 `if self._rerank_effective(rerank_override) "
              "and scored:` (ANN) / :3692 (lex) — pre-gate cross-encoder (#341)"),
        validation=VALIDATION_SIGNAL,
        expect_absent=("xenc_rerank",),
    ),
    # Encoder-isolation enable trio (bench/encoder-isolation-ab). The INVERSE
    # of the no_* ladder: run against the all-off base config
    # (configs/enc_all_off.toml — dense/splade/pki all false, everything else
    # shipped) so each arm measures ONE layer's marginal contribution over the
    # lexical floor, complementing the abundant all-on data. Against the
    # shipped all-on config each arm is an identity transform and drops via
    # ``arm_is_identity``, exactly like the rerank pair above.
    Arm(
        name="dense_on",
        knobs={"retrieval.dense_embedding_enabled": True},
        gate=("knowledge_store.py:2972 / :3757 `if self._dense_embedding_enabled` "
              "+ context_manager.py:2946 ANN-branch selector"),
        validation=VALIDATION_SIGNAL_PRESENT,
        expect_present=("dense", "ann_dense_leg"),
    ),
    Arm(
        name="splade_on",
        knobs={"ingestion.splade_enabled": True},
        gate="knowledge_store.py:2783 `if self._splade_enabled:`",
        validation=VALIDATION_SIGNAL_PRESENT,
        expect_present=("splade",),
    ),
    Arm(
        name="pki_on",
        knobs={"retrieval.pki_enabled": True},
        gate=("knowledge_store.py `if q_lower_tokens and self._pki_enabled:` "
              "(Tier 0 path-key compound index)"),
        validation=VALIDATION_SIGNAL_PRESENT,
        expect_present=("pki",),
    ),
    Arm(
        name="no_classifier",
        knobs={"classifier.enabled": False},
        gate=("context_manager.py:1801-1807 classify_query gate; metadata at "
              ":2316; assembly cap at :2101-2113"),
        validation=VALIDATION_METADATA,
        expect_absent=("classifier",),
    ),
)

# Arms considered and rejected at authoring time, with the evidence. Copied
# into the receipt verbatim so the null-result question ("did you even try
# PLR?") has an answer that does not require reading this file.
DROPPED_ARMS: Tuple[Dict[str, str], ...] = (
    {
        "arm": "no_plr",
        "knob": "[plr] enabled",
        "reason": (
            "Not on the build_context path. [plr] enabled is read only by "
            "server/routes_context.py:635 -> server/helpers.py:444 "
            "_compute_plr_confidence, which decorates the /context/packet "
            "response; retrieval/fusion_plr.py:31 states it 'leaves document "
            "ranking untouched'. An in-process arm would measure zero by "
            "construction, not by evidence."
        ),
    },
    {
        "arm": "no_harmonic",
        "knob": "use_harmonic / [cymatics] harmonic_links",
        "reason": (
            "No query-time off-switch. `use_harmonic` defaults True at "
            "context_manager.py:2895 and knowledge_store.py:2259 and is passed "
            "by NO call site (context_manager.py:1945, :1965), so Tier 5 "
            "(knowledge_store.py:3119) and _expand_coactivated "
            "(knowledge_store.py:3494) are unconditional. [cymatics] "
            "harmonic_links gates only the ingest/expression-time edge WRITE "
            "inside the `if not read_only` block at context_manager.py:2353."
        ),
    },
    {
        "arm": "no_tag_prefix / tag discipline",
        "knob": "[retrieval] tag_idf_enabled, tag_df_cap, tag_prefix_weight",
        "reason": (
            "Tier 1 (knowledge_store.py:2697) and Tier 2 (:2720) are "
            "unconditional. The #327 knobs default to OFF (config.py:660, "
            ":668), so flipping them is an intervention, not an ablation. The "
            "only mute is weight-zeroing (tag_prefix_weight=0.0), which "
            "silences the contribution while still paying the SQL cost — that "
            "would misreport the cost axis, so no arm is offered."
        ),
    },
)


# ── Pure helpers (unit-tested in tests/test_ablation_tools.py) ───────────


def set_dotted(cfg: Any, path: str, value: Any) -> bool:
    """Set ``cfg.<section>.<key> = value`` (or ``cfg.<key>`` for a bare path).

    Returns False when the path does not exist, so a renamed/removed knob is a
    loud SKIP rather than a silently-null arm (the run_ladder.py lesson).
    """
    section, sep, key = path.partition(".")
    if not sep:
        if not hasattr(cfg, section):
            return False
        setattr(cfg, section, value)
        return True
    obj = getattr(cfg, section, None)
    if obj is None or not hasattr(obj, key):
        return False
    setattr(obj, key, value)
    return True


_MISSING = object()


def get_dotted(cfg: Any, path: str) -> Any:
    """Read ``cfg.<section>.<key>``; returns the ``_MISSING`` sentinel if absent."""
    section, sep, key = path.partition(".")
    if not sep:
        return getattr(cfg, section, _MISSING)
    obj = getattr(cfg, section, None)
    if obj is None:
        return _MISSING
    return getattr(obj, key, _MISSING)


def arm_is_identity(cfg: Any, arm: Arm) -> Optional[str]:
    """Return a reason string when *arm* would not change *cfg*, else None.

    An ablation whose knob already sits at the ablated value in the effective
    config is an identity transform: running it produces a "no effect" number
    that says nothing about the layer. Caught here rather than published.
    """
    if not arm.knobs:
        return None
    unchanged: List[str] = []
    for path, value in arm.knobs.items():
        current = get_dotted(cfg, path)
        if current is _MISSING:
            return f"unknown config path: {path}"
        if current == value:
            unchanged.append(f"{path} already {value!r}")
    if len(unchanged) == len(arm.knobs):
        return "identity vs baseline — " + "; ".join(unchanged)
    return None


def rank_gold(
    scores: Mapping[str, float], gold: Sequence[str] | set, depth: Optional[int] = None,
) -> Tuple[Optional[int], List[int]]:
    """Rank the gold ids inside a ``{gene_id: score}`` map.

    Returns ``(rank_of_first_gold, [all gold ranks])``, 1-indexed, sorted by
    score descending with ``gene_id`` ascending as the tiebreak so the ordering
    is reproducible across arms and processes (dict insertion order is not).
    ``depth`` truncates the considered list; None considers the whole map.
    """
    gold_set = set(gold)
    ordered = [gid for gid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    if depth is not None:
        ordered = ordered[:depth]
    ranks = [i for i, gid in enumerate(ordered, 1) if gid in gold_set]
    return (ranks[0] if ranks else None, ranks)


def recall_at_k(first_ranks: Sequence[Optional[int]], k: int) -> float:
    """Fraction of queries whose first gold landed at rank <= k. Empty -> 0.0."""
    if not first_ranks:
        return 0.0
    hit = sum(1 for r in first_ranks if r is not None and r <= k)
    return round(hit / len(first_ranks), 4)


def delivered_gold_fields(expressed_gene_ids: Sequence[str], gold: set) -> Dict[str, Any]:
    """#335: delivery-basis gold metrics. 'Delivered' = expressed_gene_ids,
    the post-budget-trim slice (context_manager.expressed_gene_ids) — a gold
    at pool rank 3 can still fail delivery; rank-based recall cannot see it."""
    rank = None
    for i, gid in enumerate(expressed_gene_ids, start=1):
        if gid in gold:
            rank = i
            break
    return {"delivered_gold": 1 if rank else 0,
            "delivered_count": len(expressed_gene_ids),
            "delivered_gold_rank": rank}


def final_rank_of_first_gold(
    ranked_ids: Sequence[str], gold: Sequence[str] | set,
) -> Optional[int]:
    """#341: 1-based position of the first gold id in the FINAL order.

    ``ranked_ids`` is ``genome.last_ranked_ids`` — the order retrieval
    actually handed downstream, published on both rerank arms
    (knowledge_store.py:3748 lex / :4320 ANN). A cross-encoder rerank
    permutes ids without rewriting fused scores, so it is invisible in
    ``last_query_scores``; this is the basis that can see it.

    ``None`` when no gold appears (or the order is empty) — an
    unmeasurable, deliberately not a sentinel rank.
    """
    gold_set = set(gold)
    if not gold_set:
        return None
    for i, gid in enumerate(ranked_ids, start=1):
        if gid in gold_set:
            return i
    return None


def final_order_fields(
    ranked_ids: Sequence[str],
    gold: Sequence[str] | set,
    k: int,
    diag: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-needle final-order block: rank, hit@k, and the #341 rerank diag.

    ``diag`` is ``genome.last_rerank_diag`` (``{"gate_kept", "floor_appended"}``,
    published by query_docs_ann on BOTH arms). Missing/empty -> the two keys
    are present and ``None``, never absent: the paired basis must have the
    same columns on every arm or a reader cannot line the vectors up.
    """
    rank = final_rank_of_first_gold(ranked_ids, gold)
    d = dict(diag or {})
    return {
        "final_rank_of_first_gold": rank,
        "final_recall_hit": 1 if (rank is not None and rank <= k) else 0,
        "gate_kept": d.get("gate_kept"),
        "floor_appended": d.get("floor_appended"),
    }


def delivered_gold_rate(per_query: Sequence[Mapping[str, Any]]) -> float:
    if not per_query:
        return 0.0
    return sum(r.get("delivered_gold", 0) for r in per_query) / len(per_query)


def newest_splice_count(events: Sequence[Mapping[str, Any]]) -> Optional[int]:
    """#341 splice-interaction receipt: the last splice-stage ring entry's
    ``n_candidates``, or None when the splice stage never rang for *events*
    (splice only runs when splice is active)."""
    splice_events = [e for e in events if e.get("stage") == "splice"]
    if not splice_events:
        return None
    return splice_events[-1].get("n_candidates")


def pipeline_ring_mark(events: Sequence[Mapping[str, Any]]) -> float:
    """A monotonic mark for a ring snapshot taken immediately BEFORE a
    ``build_context`` call: the newest entry's ``ts`` (wall-clock seconds,
    stamped by ``context_manager._stage_timer``), or ``0.0`` when the ring
    was empty.

    ``context_manager._pipeline_events`` is a module-global, PROCESS-
    LIFETIME deque (maxlen=128, ~8 stage events per query) — across the
    many needles in one ``run_arm`` loop (or across arms sharing a
    process), a needle whose own ``build_context`` never rings a splice
    entry (no-match early return, abstain, or a pre-splice exception) must
    not silently inherit a PRIOR needle's (or the untimed warmup's) splice
    count (review finding, #341). A length-based "read everything past
    index N" snapshot breaks once the deque wraps past that index; ``ts``
    keeps correlating correctly across wraparound because it is a property
    of the entry itself, not of its position in the deque.
    """
    if not events:
        return 0.0
    return max(float(e.get("ts", 0.0)) for e in events)


def events_since_mark(
    events: Sequence[Mapping[str, Any]], mark: float,
) -> List[Mapping[str, Any]]:
    """Entries strictly newer than *mark* (see ``pipeline_ring_mark``) —
    i.e. rung by the ``build_context`` call that followed the mark, not by
    whatever query last happened to ring a matching stage."""
    return [e for e in events if float(e.get("ts", 0.0)) > mark]


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile (no interpolation). Empty -> None.

    Nearest-rank keeps p95 of a 30-query arm an actually-observed value rather
    than an interpolated fiction between two samples.
    """
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * len(ordered) + 0.5)) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return round(float(ordered[idx]), 3)


def median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(statistics.median(values)), 3)


def validate_arm(
    arm: Arm,
    observed: Mapping[str, Any],
    baseline: Optional[Mapping[str, Any]],
) -> Tuple[bool, str]:
    """Decide arm_valid for *arm* from its observables and the baseline's.

    *observed* / *baseline* are the per-arm observable bundles built by
    ``_observables``: signal keys, tier-contribution keys, classifier-metadata
    query count, synonym-expansion count, cymatics call count, cold-tier flag.

    Two conditions, both required:
      * the ablated observable is gone in this arm, and
      * it was present in the baseline (otherwise the arm proves nothing and is
        reported as ``unfalsifiable``, not as a clean ablation).
    """
    if arm.validation == VALIDATION_CONTROL:
        return True, "control arm — shipped config, no ablation to validate"

    if arm.validation in (VALIDATION_SIGNAL, VALIDATION_TIER):
        field_name = "signal_keys" if arm.validation == VALIDATION_SIGNAL else "tier_keys"
        arm_keys = set(observed.get(field_name) or ())
        base_keys = set((baseline or {}).get(field_name) or ())
        expected = list(arm.expect_absent)
        fired_at_baseline = [s for s in expected if s in base_keys]
        if not fired_at_baseline:
            return False, (
                f"unfalsifiable: none of {expected} appear in the baseline "
                f"{field_name} — the layer never ran at baseline, so its "
                "absence here is not evidence of an ablation"
            )
        still_present = [s for s in expected if s in arm_keys]
        if still_present:
            return False, (
                f"ablation did not take: {still_present} still present in "
                f"{field_name}"
            )
        return True, (
            f"{field_name}: {sorted(fired_at_baseline)} present at baseline, "
            f"absent in this arm"
        )

    if arm.validation == VALIDATION_SIGNAL_PRESENT:
        # The ENABLE direction (#341). Both halves inverted vs the absence
        # kinds: present here, absent at baseline. A missing baseline bundle
        # is NOT read as "absent at baseline" — with no control there is
        # nothing to contrast against, so a layer that was on all along would
        # otherwise read as a clean enable.
        if baseline is None:
            return False, (
                "unfalsifiable: no baseline observables to contrast against "
                "(the control arm did not run)"
            )
        arm_keys = set(observed.get("signal_keys") or ())
        base_keys = set(baseline.get("signal_keys") or ())
        expected = list(arm.expect_present)
        already_at_baseline = [s for s in expected if s in base_keys]
        if already_at_baseline:
            return False, (
                f"unfalsifiable: {already_at_baseline} already present in the "
                "baseline signal_keys — the layer was on before this arm "
                "touched anything, so its presence here proves nothing"
            )
        missing = [s for s in expected if s not in arm_keys]
        if missing:
            return False, (
                f"enable did not take: {missing} absent from signal_keys "
                "(knob applied, seam never ran)"
            )
        return True, (
            f"signal_keys: {sorted(expected)} absent at baseline, present in "
            "this arm"
        )

    if arm.validation == VALIDATION_METADATA:
        arm_n = int(observed.get("classifier_metadata_queries") or 0)
        base_n = int((baseline or {}).get("classifier_metadata_queries") or 0)
        if base_n == 0:
            return False, (
                "unfalsifiable: window.metadata['classifier'] absent at "
                "baseline too (context_manager.py:2316 never fired)"
            )
        if arm_n:
            return False, (
                f"ablation did not take: window.metadata['classifier'] still "
                f"written on {arm_n} queries"
            )
        return True, (
            f"window.metadata['classifier'] written on {base_n} baseline "
            "queries, 0 here"
        )

    if arm.validation == VALIDATION_EXPANSION:
        arm_n = int(observed.get("synonym_expansions") or 0)
        base_n = int((baseline or {}).get("synonym_expansions") or 0)
        if base_n == 0:
            return False, (
                "unfalsifiable: baseline synonym map expanded 0 probe terms "
                "(empty or non-matching [synonyms] section)"
            )
        if arm_n:
            return False, (
                f"ablation did not take: _expand_terms still expanded {arm_n} "
                "probe terms"
            )
        return True, (
            f"_expand_terms expanded {base_n} probe terms at baseline, 0 here"
        )

    if arm.validation == VALIDATION_CALL_PROBE:
        arm_n = int(observed.get("cymatics_calls") or 0)
        base_n = int((baseline or {}).get("cymatics_calls") or 0)
        if base_n == 0:
            return False, (
                "unfalsifiable: scoring.cymatics.query_spectrum was never "
                "called at baseline (blend.py:212 needs >1 candidate)"
            )
        if arm_n:
            return False, (
                f"ablation did not take: query_spectrum still called {arm_n} times"
            )
        return True, (
            f"scoring.cymatics.query_spectrum called {base_n}x at baseline, 0 here"
        )

    if arm.validation == VALIDATION_COLD:
        arm_n = int(observed.get("cold_tier_queries") or 0)
        base_n = int((baseline or {}).get("cold_tier_queries") or 0)
        if base_n == 0:
            return False, (
                "unfalsifiable: cold tier never fired at baseline "
                "(_last_cold_tier_used False on every query)"
            )
        if arm_n:
            return False, f"ablation did not take: cold tier fired on {arm_n} queries"
        return True, f"cold tier fired on {base_n} baseline queries, 0 here"

    return False, f"validation='{arm.validation}': no observable implemented"


def per_query_record(
    needle: str,
    *,
    gold_ranks: Sequence[int],
    first_rank: Optional[int],
    k: int,
    n_queries: int,
    wall_ms: float,
    signal_ms: Mapping[str, float],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """One auditable row for the ``--per-query`` receipt.

    Per-arm aggregates alone make the PAIRED basis unauditable: you cannot
    reconstruct which needle moved between two arms, so a paired test (McNemar
    on the hit vector, or a per-needle rank delta) cannot be run from the
    receipt. This record is the missing basis.

    ``recall_hit`` is the exact 0/1 vector a paired test consumes:
    ``sum(recall_hit) / n_queries`` reconstructs ``recall_at_k`` exactly.
    ``recall_contribution`` is the same fact pre-divided (``1/n_queries`` on a
    hit, else 0.0) and carries FULL precision — unrounded on purpose, so the
    column still sums to recall@k. Note the aggregate ``recall_at_k`` is itself
    rounded to 4 decimals, so the two agree to within that rounding, not to the
    last bit.
    """
    hit = first_rank is not None and first_rank <= k
    record: Dict[str, Any] = {
        "needle": needle,
        "gold_ranks": list(gold_ranks),
        "rank_of_first_gold": first_rank,
        "recall_hit": 1 if hit else 0,
        "recall_contribution": (1.0 / n_queries if hit and n_queries else 0.0),
        "wall_ms": round(float(wall_ms), 3),
        "signal_ms": {name: round(float(ms), 3) for name, ms in sorted(signal_ms.items())},
    }
    if error is not None:
        record["error"] = error
    return record


def layer_signal_medians(per_query_signals: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Median ms per signal across a query set. Signals missing from a query
    are OMITTED from that signal's sample, not counted as zero — a tier that
    ran on 3 of 30 queries should report its own cost, not a diluted one."""
    buckets: Dict[str, List[float]] = {}
    for sig in per_query_signals:
        for name, ms in (sig or {}).items():
            buckets.setdefault(name, []).append(float(ms))
    return {
        name: round(float(statistics.median(vals)), 3)
        for name, vals in sorted(buckets.items())
    }


# ── Runtime plumbing ────────────────────────────────────────────────────


class _CallProbe:
    """Count calls to ``module.attr`` for the duration of the context.

    Used where a layer has no timing signal of its own. Restores the original
    attribute unconditionally so one arm cannot leak instrumentation into the
    next.
    """

    def __init__(self, module: Any, attr: str) -> None:
        self._module = module
        self._attr = attr
        self._orig: Any = None
        self.count = 0

    def __enter__(self) -> "_CallProbe":
        self._orig = getattr(self._module, self._attr)
        orig = self._orig
        probe = self

        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            probe.count += 1
            return orig(*args, **kwargs)

        setattr(self._module, self._attr, _wrapper)
        return self

    def __exit__(self, *exc: Any) -> bool:
        setattr(self._module, self._attr, self._orig)
        return False


def _git_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _probe_synonym_expansions(genome: Any, probe_terms: Sequence[str]) -> int:
    """How many probe terms ``_expand_terms`` actually widened.

    Direct observable for the synonyms layer: feed each probe term in alone and
    count the ones that came back with more than themselves.
    """
    expanded = 0
    for term in probe_terms:
        try:
            out = genome._expand_terms([term])
        except Exception:
            continue
        if len(set(out)) > 1:
            expanded += 1
    return expanded


def _dense_codec_name(genome: Any) -> Optional[str]:
    codec = getattr(genome, "_dense_codec", None)
    return type(codec).__name__ if codec is not None else None


def _clear_rank_publication(genome: Any) -> None:
    """Blank ``last_ranked_ids`` / ``last_rerank_diag`` before a query.

    Both are store attributes with PROCESS lifetime (knowledge_store.py:950,
    :955), rewritten only when a query reaches its publication point. Reading
    them after a needle that never got there (no-match early return, abstain,
    exception) would return the PREVIOUS needle's order — the same staleness
    trap ``pipeline_ring_mark`` guards for the splice ring. Clearing first
    turns that case into an honest ``None``.

    Only attributes that already exist are touched, so a store wrapper that
    does not publish them (e.g. ShardedGenomeAdapter, whitelisted as
    adapter-only) neither gains a phantom attribute nor raises here; the
    reader degrades to ``None`` fields instead.
    """
    if hasattr(genome, "last_ranked_ids"):
        genome.last_ranked_ids = []
    if hasattr(genome, "last_rerank_diag"):
        genome.last_rerank_diag = {}
    # W2.1: same staleness discipline for the cover-walk diagnostics.
    if hasattr(genome, "last_cover_walk_diag"):
        genome.last_cover_walk_diag = {}


def _read_rank_publication(genome: Any) -> Tuple[List[str], Dict[str, Any]]:
    """Snapshot ``(last_ranked_ids, last_rerank_diag)`` defensively."""
    ranked = list(getattr(genome, "last_ranked_ids", None) or ())
    diag = dict(getattr(genome, "last_rerank_diag", None) or {})
    return ranked, diag


def _read_cover_walk_diag(genome: Any) -> Optional[Dict[str, Any]]:
    """W2.1: snapshot ``last_cover_walk_diag`` — ``None`` when the walk did
    not run this needle (knob off, additive fusion, or a pre-publication
    failure), a dict (mode / frontier / band_boosted / rescues / displaced)
    when it did. Additive receipt field: absent on pre-W2.1 receipts, never
    renames existing keys."""
    d = getattr(genome, "last_cover_walk_diag", None)
    return dict(d) if d else None


def run_arm(
    arm: Arm,
    *,
    genome_path: str,
    config_path: Optional[str],
    needles: Sequence[Mapping[str, str]],
    gold_by_needle: Mapping[str, set],
    k: int,
    per_query: bool = False,
) -> Dict[str, Any]:
    """Build a manager for *arm*, run every needle, return the receipt row.

    ``per_query`` attaches the per-needle basis (see ``per_query_record``).
    Off by default: the row is byte-identical to the pre-flag shape when it is
    not set — the key is not emitted at all, not emitted empty.
    """
    from cymatix_context.backends import encoder_client
    from cymatix_context.config import load_config
    from cymatix_context.context_manager import (
        CymatixContextManager,
        get_pipeline_ring_max,
        get_recent_pipeline_events,
    )
    from cymatix_context.scoring import cymatics as cymatics_mod
    from cymatix_context.retrieval.measurement import capture_stages

    cfg = load_config(config_path) if config_path else load_config()
    cfg.genome.path = genome_path

    unapplied = [p for p, v in arm.knobs.items() if not set_dotted(cfg, p, v)]
    if unapplied:
        return {
            "arm": arm.name,
            "knobs_changed": dict(arm.knobs),
            "arm_valid": False,
            "validation": f"unknown config path(s): {unapplied}",
            "skipped": True,
            "errors": [f"config paths not found: {unapplied}"],
        }

    # Probe terms are the BASELINE synonym keys — taken from the freshly-loaded
    # config before the arm may have blanked the map, so the no_synonyms arm
    # probes the same terms the baseline did.
    probe_terms = sorted(load_config(config_path).synonym_map if config_path
                         else load_config().synonym_map)[:12]

    errors: List[str] = []
    first_ranks: List[Optional[int]] = []
    final_ranks: List[Optional[int]] = []
    all_ranks: List[int] = []
    latencies: List[float] = []
    records: List[Dict[str, Any]] = []
    per_query_signals: List[Dict[str, float]] = []
    signal_keys: set = set()
    tier_keys: set = set()
    classifier_metadata_queries = 0
    cold_tier_queries = 0
    delivered_gold_hits = 0
    n_scored = 0

    # #341 splice-interaction receipt, review fix: sentinel for "the
    # pre-call ring mark itself could not be read" — deliberately distinct
    # from pipeline_ring_mark's 0.0 ("ring was empty"), because falling
    # back to 0.0 here would treat the WHOLE ring as "since this call" and
    # reproduce the exact leak this guards against.
    _RING_MARK_UNAVAILABLE = object()

    def _resolve_splice_n_candidates(mark: Any, name: str) -> Optional[int]:
        """Drain splice_n_candidates for one needle, correlated to *mark*
        (see pipeline_ring_mark/events_since_mark). Degrades to None on any
        ring-read failure rather than raising or silently falling back to
        an unmarked (whole-ring) drain, matching run_arm's established
        guard pattern on foreign calls (encoder_client.active_url() above)."""
        if mark is _RING_MARK_UNAVAILABLE:
            return None
        try:
            ring_events = get_recent_pipeline_events(get_pipeline_ring_max())
            return newest_splice_count(events_since_mark(ring_events, mark))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: splice ring drain failed: {exc}")
            return None

    manager = CymatixContextManager(cfg)
    warmed = False
    try:
        # Untimed warmup (erb receipt convention). [hardware] lazy_encoders is
        # on by default, so the FIRST query of an arm pays the ~2 GB BGE-M3
        # load plus SPLADE/SEMA construction — measured at ~40 s vs ~20 s for
        # a warm query on the 100K carve. Folding that into p50 would make
        # every arm's cost a function of how many needles it ran. The warmup
        # is outside the cymatics call-probe so it cannot inflate that count.
        if needles:
            try:
                manager.build_context(
                    needles[0]["query"], read_only=True,
                    ignore_delivered=True, max_genes=k,
                )
                warmed = True
            except Exception as exc:  # noqa: BLE001
                errors.append(f"warmup: {type(exc).__name__}: {exc}")

        with _CallProbe(cymatics_mod, "query_spectrum") as cymatics_probe:
            for needle in needles:
                name = needle["name"]
                gold = gold_by_needle.get(name, set())
                t0 = time.perf_counter()
                # #341 review fix: mark the ring immediately BEFORE this
                # call. context_manager._pipeline_events is a process-
                # lifetime global (not reset per query/arm), so draining
                # "the newest splice entry" without this mark would leak a
                # PRIOR needle's (or the untimed warmup's) splice count into
                # a needle whose own build_context never reaches splice
                # (no-match early return, abstain, or a pre-splice
                # exception) — see events_since_mark's docstring.
                try:
                    ring_mark = pipeline_ring_mark(
                        get_recent_pipeline_events(get_pipeline_ring_max()))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}: pipeline ring mark failed: {exc}")
                    ring_mark = _RING_MARK_UNAVAILABLE
                # #341: same discipline for the final-order publication —
                # blank it so this needle can only read its OWN order.
                try:
                    _clear_rank_publication(manager.genome)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}: rank publication clear failed: {exc}")
                try:
                    with capture_stages(gold, enabled=per_query) as stage_capture:
                        window = manager.build_context(
                            needle["query"],
                            read_only=True,
                            ignore_delivered=True,
                            max_genes=k,
                        )
                except Exception as exc:  # noqa: BLE001
                    detail = f"{type(exc).__name__}: {exc}"
                    errors.append(f"{name}: {detail}")
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(wall_ms)
                    first_ranks.append(None)
                    # A failed query is still a row in the paired basis — it is
                    # a miss for THIS arm on THIS needle, and dropping it would
                    # silently shorten the vector a paired test lines up. It is
                    # also a delivered-gold miss: nothing was delivered.
                    error_record = per_query_record(
                        name, gold_ranks=[], first_rank=None, k=k,
                        n_queries=len(needles), wall_ms=wall_ms, signal_ms={},
                        error=detail,
                    )
                    error_record.update(delivered_gold_fields([], set(gold)))
                    error_record["budget_tier"] = None
                    # #341 final-order basis: cleared before the call, so this
                    # is either this needle's own published order (a failure
                    # AFTER retrieval — e.g. assemble) or empty -> None. Never
                    # the previous needle's.
                    err_ranked, err_diag = _read_rank_publication(manager.genome)
                    err_final = final_order_fields(err_ranked, set(gold), k, err_diag)
                    final_ranks.append(err_final["final_rank_of_first_gold"])
                    error_record.update(err_final)
                    error_record["cover_walk"] = _read_cover_walk_diag(manager.genome)
                    # #341 splice-interaction receipt: still None on the usual
                    # failure (splice never reached), but a build_context that
                    # raises AFTER splice (e.g. assemble) leaves the entry
                    # ringed — mark-correlated drain, same as the success path.
                    error_record["splice_n_candidates"] = _resolve_splice_n_candidates(
                        ring_mark, name)
                    if per_query:
                        error_record["stage_provenance"] = stage_capture.report()
                    records.append(error_record)
                    continue
                wall_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(wall_ms)

                scores = dict(manager.genome.last_query_scores or {})
                if scores:
                    n_scored += 1
                first, ranks = rank_gold(scores, gold)
                first_ranks.append(first)
                all_ranks.extend(ranks)

                sig = dict(getattr(manager.genome, "last_signal_timings", None) or {})
                per_query_signals.append(sig)
                signal_keys.update(sig.keys())
                # expressed_gene_ids is the ORDERED post-budget-trim delivery
                # slice — keep it a list (not a set) going into
                # delivered_gold_fields, or delivered_gold_rank goes stale.
                expressed_gene_ids = list(getattr(window, "expressed_gene_ids", None) or ())
                dg_fields = delivered_gold_fields(expressed_gene_ids, set(gold))
                if dg_fields["delivered_gold"]:
                    delivered_gold_hits += 1
                record = per_query_record(
                    name, gold_ranks=ranks, first_rank=first, k=k,
                    n_queries=len(needles), wall_ms=wall_ms, signal_ms=sig,
                )
                record.update(dg_fields)
                record["budget_tier"] = (getattr(window, "metadata", None) or {}).get("budget_tier")
                # #341: the FINAL retrieval order (post-rerank when it ran).
                # last_query_scores cannot see a cross-encoder reorder — it
                # permutes ids without rewriting fused scores — so the rerank
                # A/B needs this basis, and both arms publish it.
                ranked_ids, rerank_diag = _read_rank_publication(manager.genome)
                final_fields = final_order_fields(ranked_ids, set(gold), k, rerank_diag)
                final_ranks.append(final_fields["final_rank_of_first_gold"])
                record.update(final_fields)
                # #341 splice-interaction receipt: mark-correlated drain
                # (see ring_mark above) — only entries rung by THIS
                # build_context call count, so a needle that never reaches
                # splice (no-match, abstain, disabled arm) gets None
                # instead of inheriting a prior needle's stale count.
                record["splice_n_candidates"] = _resolve_splice_n_candidates(
                    ring_mark, name)
                # W2.1: cover-walk per-needle diag (None = walk did not run).
                record["cover_walk"] = _read_cover_walk_diag(manager.genome)
                if per_query:
                    record["stage_provenance"] = stage_capture.report()
                records.append(record)

                contribs = getattr(manager.genome, "last_tier_contributions", None) or {}
                for per_gene in contribs.values():
                    tier_keys.update(per_gene.keys())

                meta = getattr(window, "metadata", None) or {}
                if "classifier" in meta:
                    classifier_metadata_queries += 1
                # build_context resets this per turn (context_manager.py:1844),
                # so reading it after the call is a per-query observation.
                if getattr(manager, "_last_cold_tier_used", False):
                    cold_tier_queries += 1

        observables = {
            "signal_keys": sorted(signal_keys),
            "tier_keys": sorted(tier_keys),
            "classifier_metadata_queries": classifier_metadata_queries,
            "synonym_expansions": _probe_synonym_expansions(manager.genome, probe_terms),
            "cymatics_calls": cymatics_probe.count,
            "cold_tier_queries": cold_tier_queries,
        }
        remote_url = ""
        try:
            remote_url = encoder_client.active_url()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"encoder_client.active_url failed: {exc}")
        dense_codec = _dense_codec_name(manager.genome)
    finally:
        # Drop the store's connections so the next arm opens the bed clean.
        try:
            manager.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"manager.close failed: {exc}")

    hit_ranks = [r for r in first_ranks if r is not None]
    recall_val = recall_at_k(first_ranks, k)
    # --per-query controls only whether `records` is EMITTED on the receipt;
    # the list is always built (exactly one records.append per needle, on both
    # the success and the error path above), so the rate never depends on the
    # flag. delivered_gold_rate(records) == delivered_gold_hits/len(needles)
    # by construction, including the empty case (both give 0.0).
    delivered_rate = delivered_gold_rate(records)
    row: Dict[str, Any] = {
        "arm": arm.name,
        "knobs_changed": dict(arm.knobs),
        "gate": arm.gate,
        "arm_valid": None,          # filled by the caller (needs baseline)
        "validation": arm.validation,
        "validation_detail": "",    # filled by the caller
        "recall_at_k": recall_val,
        # #341: same recall on the FINAL order (post-rerank when it ran).
        # Reported next to recall_at_k, never instead of it — the two answer
        # different questions and a rerank arm can move one without the other.
        "final_recall_at_k": recall_at_k(final_ranks, k),
        "mean_rank_of_gold": (round(statistics.mean(hit_ranks), 2) if hit_ranks else None),
        "median_rank_of_gold": median_or_none(hit_ranks),
        "gold_found_queries": len(hit_ranks),
        "gold_ranks_total": len(all_ranks),
        "delivered_gold_queries": delivered_gold_hits,
        "delivered_gold_rate": delivered_rate,
        "pool_delivery_gap": round(recall_val - delivered_rate, 4),
        "queries_with_scores": n_scored,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "signal_ms_median": layer_signal_medians(per_query_signals),
        "n_queries": len(needles),
        "warmup_ran": warmed,
        "encoder_url": remote_url,
        "dense_codec_class": dense_codec,
        "remote_dense": bool(dense_codec and dense_codec.startswith("Remote")),
        "observables": observables,
        "errors": errors,
    }
    if per_query:
        row["per_query"] = records
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="In-process per-layer ablation probe (ERB beds).",
    )
    ap.add_argument("--genome", default="f:/tmp/erb_carve/erb_100k.db")
    ap.add_argument("--resolved", default="benchmarks/dogfood/erb/needles_resolved_carve.json")
    ap.add_argument("--gold", default="benchmarks/dogfood/erb/gold_by_needle_carve.json")
    ap.add_argument("--limit", type=int, default=30, help="needles to run (0 = all)")
    ap.add_argument("--k", type=int, default=12, help="delivery cap + recall depth")
    ap.add_argument("--arms", default="", help="comma list; empty = all")
    ap.add_argument("--config", default="cymatix.toml",
                    help="cymatix.toml to load (repo-relative ok); '' = auto-discover")
    ap.add_argument("--out", default="benchmarks/dogfood/erb/receipts/ablation_ladder.json")
    ap.add_argument("--stamp", default="", help="caller-supplied run timestamp")
    ap.add_argument("--per-query", action="store_true", dest="per_query",
                    help="emit the per-needle basis in each arm (needle, gold "
                         "ranks, recall contribution, wall ms, signal ms) so "
                         "paired arm-vs-arm tests are auditable from the "
                         "receipt alone; off = receipt shape unchanged")
    args = ap.parse_args(argv)

    resolved_path = _resolve(args.resolved)
    gold_path = _resolve(args.gold)
    needles = json.loads(resolved_path.read_text(encoding="utf-8"))["needles"]
    if args.limit:
        needles = needles[: args.limit]
    gold_by_needle = {
        key: set(val)
        for key, val in json.loads(gold_path.read_text(encoding="utf-8")).items()
    }

    genome_path = str(_resolve(args.genome))
    if not Path(genome_path).exists():
        print(f"genome not found: {genome_path}", file=sys.stderr)
        return 2
    os.environ["CYMATIX_GENOME_PATH"] = genome_path

    config_path: Optional[str] = None
    if args.config:
        candidate = _resolve(args.config)
        if not candidate.exists():
            print(f"config not found: {candidate}", file=sys.stderr)
            return 2
        config_path = str(candidate)

    from cymatix_context.config import load_config

    effective_cfg = load_config(config_path) if config_path else load_config()

    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
    selected: List[Arm] = []
    dropped: List[Dict[str, str]] = [dict(d) for d in DROPPED_ARMS]
    for arm in ARMS:
        if wanted and arm.name not in wanted and arm.name != "baseline":
            continue
        reason = arm_is_identity(effective_cfg, arm)
        if reason:
            dropped.append({
                "arm": arm.name,
                "knob": ", ".join(arm.knobs),
                "reason": (
                    f"{reason} (gate exists at {arm.gate}, but the effective "
                    "config already sits at the ablated value, so this arm "
                    "would measure an identity transform)"
                ),
            })
            continue
        selected.append(arm)
    if wanted and "baseline" not in wanted:
        print("note: baseline is always run — it is the positive control every "
              "ablation arm validates against", file=sys.stderr)

    print(f"ablation ladder: {len(selected)} arms x {len(needles)} needles, "
          f"k={args.k}, bed={genome_path}", flush=True)
    for entry in dropped:
        print(f"  DROP {entry['arm']}: {entry['reason'].splitlines()[0]}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    baseline_observables: Optional[Dict[str, Any]] = None
    for arm in selected:
        row = run_arm(
            arm,
            genome_path=genome_path,
            config_path=config_path,
            needles=needles,
            gold_by_needle=gold_by_needle,
            k=args.k,
            per_query=args.per_query,
        )
        if row.get("skipped"):
            rows.append(row)
            print(f"  SKIP {arm.name}: {row['validation']}", file=sys.stderr)
            continue
        if arm.name == "baseline":
            baseline_observables = row["observables"]
        valid, detail = validate_arm(arm, row["observables"], baseline_observables)
        row["arm_valid"] = valid
        row["validation_detail"] = detail
        rows.append(row)

        flag = "ok " if valid else "BAD"
        print(
            # Both recall bases on the live line: a #341 rerank arm reorders
            # ids without touching fused scores, so r@k can sit stone-still
            # while fr@k moves. Printing only r@k would read as "no effect".
            f"  [{flag}] {arm.name:16s} r@{args.k}={row['recall_at_k']:.3f} "
            f"fr@{args.k}={row['final_recall_at_k']:.3f} "
            f"med_rank={row['median_rank_of_gold']} "
            f"p50={row['latency_ms']['p50']}ms p95={row['latency_ms']['p95']}ms "
            f"-- {detail}",
            flush=True,
        )

    payload = {
        "tool": "benchmarks/dogfood/erb/ablation_ladder.py",
        "stamp": args.stamp,
        "bed": genome_path,
        "bed_bytes": Path(genome_path).stat().st_size,
        "config": config_path or "(auto-discovered)",
        "git_sha": _git_sha(REPO_ROOT),
        "needles_file": str(resolved_path),
        "gold_file": str(gold_path),
        "needle_count": len(needles),
        "k": args.k,
        "env": {
            "CYMATIX_ENCODER_URL": os.environ.get("CYMATIX_ENCODER_URL", "") or None,
            "CYMATIX_ENCODER_URL_set": bool(os.environ.get("CYMATIX_ENCODER_URL", "").strip()),
            "CYMATIX_DISABLE_HEADROOM": os.environ.get("CYMATIX_DISABLE_HEADROOM"),
            "CYMATIX_DISABLE_LEARN": os.environ.get("CYMATIX_DISABLE_LEARN"),
            "CYMATIX_GENOME_PATH": os.environ.get("CYMATIX_GENOME_PATH"),
        },
        "note": (
            "Ranks read from genome.last_query_scores (post-blend; blend "
            "republishes the map at scoring/blend.py:233/:281), ties broken by "
            "gene_id ascending. THREE bases ship side by side and answer "
            "different questions: recall_at_k / *_rank_of_gold on the score "
            "map; final_recall_at_k + per-needle final_rank_of_first_gold on "
            "genome.last_ranked_ids, the FINAL order (post-rerank when it ran "
            "— a cross-encoder permutes ids without rewriting fused scores, so "
            "#341 arms are invisible in the score map); and "
            "delivered_gold_queries / delivered_gold_rate / "
            "delivered_gold_rank on window.expressed_gene_ids, the post-budget "
            "delivery slice (#335). pool_delivery_gap is recall_at_k minus "
            "delivered_gold_rate. Per-needle gate_kept / floor_appended come "
            "from genome.last_rerank_diag (published on both rerank arms). "
            "Each arm runs one untimed warmup query first so the lazy encoder "
            "load is not billed to p50. Latencies are wall ms on a shared box "
            "— within-run relatives only."
        ),
        "dropped_arms": dropped,
        "arms": rows,
    }
    if args.per_query:
        # Only emitted when asked for — the default receipt shape is unchanged.
        payload["per_query"] = True
        payload["per_query_note"] = (
            "each arm carries a per_query list, one record per needle in the "
            "run order every arm shares, so arm-vs-arm comparisons can be run "
            "PAIRED from this receipt alone. recall_hit is the exact 0/1 "
            "vector a paired test consumes and sum(recall_hit)/n_queries "
            "reconstructs recall_at_k; recall_contribution is the pre-divided "
            "form and sums to recall_at_k up to that aggregate's 4-decimal "
            "rounding. Failed queries appear as miss records carrying 'error', "
            "so the vectors stay the same length across arms. Each record also "
            "carries delivered_gold / delivered_count / delivered_gold_rank "
            "(#335 delivery basis), final_rank_of_first_gold / "
            "final_recall_hit (#341 final-order basis, from "
            "genome.last_ranked_ids), gate_kept / floor_appended (#341, from "
            "genome.last_rerank_diag; None when the query never reached the "
            "ANN publication point) and splice_n_candidates (ring-correlated; "
            "None when this needle's own query never rang the splice stage). "
            "stage_provenance records candidate counts and watched gold IDs at "
            "each measured retrieval/blend boundary for this request only; "
            "failed, not_executed, and uncaptured stages are not empty pools."
            " budget_tier is copied from the returned window metadata; null "
            "means the request failed or returned no observed tier."
        )

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    invalid = [r["arm"] for r in rows if r.get("arm_valid") is False]
    if invalid:
        print(f"arms failing self-validation: {invalid}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
CymatixContextManager -- The pipeline orchestrator.

Orchestrates the full /context pipeline per turn:
    1. Extract tags signals from query (heuristic, no model)
    2. Retrieve -- find relevant documents via tags matching + co-activation
    3. Re-rank -- score candidates via compressor (CPU, optional)
    4. Splice -- compress each candidate, keep high-value fragments (CPU, batched)
    5. Assemble -- build the 3k compressor prompt + 6k retrieved context window
    6. Persist -- pack query+response exchange into knowledge store (background)

Token budget:
    3k  = compressor decoder prompt (fixed, tells big model how to read fragments)
    6k  = retrieved context (fragment-encoded, spliced)
    600k = knowledge store (cold storage, never fully loaded)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from .accel import extract_query_signals, estimate_tokens
from .codons import CodonChunker, CodonEncoder
from .config import CymatixConfig
from .exceptions import PromoterMismatch
from .genome import Genome
from .budget_zone import is_enabled as _budget_zone_is_enabled, zone_cap as _budget_zone_cap
from .encoding.headroom_bridge import compress_text
from .encoding import legibility
from .identity import session_delivery as _session_delivery
from .ribosome import DisabledBackend, LiteLLMBackend, Ribosome, OllamaBackend
from .identity.provenance import apply_metadata_hints, apply_provenance
from .retrieval.query_classifier import ClassifierResult, classify_query
from .retrieval.measurement import current_capture
from .schemas import (
    ChromatinState,
    ContextHealth,
    ContextWindow,
    EpigeneticMarkers,
    Gene,
    PromoterTags,
    StructuralRelation,
)

log = logging.getLogger("cymatix.context_manager")


# ── Stage-timer context manager ──────────────────────────────────────
# Wraps each named pipeline stage with a monotonic-clock measurement and
# records a single OTel histogram entry on exit. Exceptions in telemetry
# are suppressed so a broken collector never kills the pipeline.
#
# In addition to the OTel histogram, each stage entry is appended to a
# bounded in-process ring (the "pipeline viewer" feed). The ring is
# scoped per build_context() call via a contextvar holding the request_id
# so the dashboard can group events back into per-request rows.

import collections as _collections
import contextlib as _contextlib
import contextvars as _contextvars
import time as _time
import uuid as _uuid
from .telemetry import (
    pipeline_stage_histogram as _pipeline_stage_histogram,
    pipeline_stage_span as _pipeline_stage_span,
    session_tokens_saved_counter as _session_tokens_saved_counter,
    splice_ratio_histogram as _splice_ratio_histogram,
)


# Per-request id propagated through the sync pipeline so each _stage_timer
# event can be attributed back to the originating build_context call. Set
# at the top of build_context(); reads default to "" when no run is active
# (e.g. ad-hoc calls from debug endpoints).
_pipeline_request_id: "_contextvars.ContextVar[str]" = _contextvars.ContextVar(
    "_pipeline_request_id", default=""
)

# Bounded ring of recent stage events. Each entry is
#   {"request_id", "stage", "ms", "ts"}
# (express also carries {"signals", "model_load_ms"}, tail_writes carries
# {"commits"} — perf slice 1) where ts is wall-clock seconds since the
# epoch. Sized to roughly the last ~10 requests' worth of stage events
# (~10 in-request stages per request since the perf-slice-1 sub-stage
# timers landed; the persist stage runs as a background task with no
# request_id, so it never rings) so the launcher dashboard can render a
# recent-runs table without unbounded memory growth on busy servers.
_PIPELINE_RING_MAX = 128
_pipeline_events: "_collections.deque[dict]" = _collections.deque(
    maxlen=_PIPELINE_RING_MAX
)


def _apply_symbol_expansion_cap(genome, cap: int) -> None:
    """Hand the [retrieval] symbol_expansion_cap to the store's co-activation
    expansion (top-K referenced definitions by centrality — WS3).

    Best-effort by design: a read adapter that does not accept ad-hoc
    attributes (e.g. a slotted/frozen sharded adapter) keeps its built-in
    default cap. But never silent (WS3 review P3) — an explicitly configured
    cap that cannot bind would otherwise no-op invisibly, so the failure is
    logged at warning with the store type and the cap that was dropped.
    """
    try:
        genome._symbol_expansion_cap = cap
    except Exception as exc:
        log.warning(
            "symbol_expansion_cap=%s could not be applied to store %s (%s: %s); "
            "co-activation symbol expansion keeps its built-in default cap",
            cap,
            type(genome).__name__,
            type(exc).__name__,
            exc,
        )


def _pipeline_ring_enabled() -> bool:
    """The ring is on by default; disable with CYMATIX_PIPELINE_RING=0."""
    return os.environ.get("CYMATIX_PIPELINE_RING", "1").strip().lower() not in (
        "0", "false", "off", "no",
    )


def get_recent_pipeline_events(limit: int = 32) -> List[dict]:
    """Return the most recent pipeline stage events (newest last).

    Consumed by the launcher dashboard via /debug/pipeline/recent. Returns
    a shallow-copy list so the caller cannot mutate the deque.
    """
    snapshot = list(_pipeline_events)
    if limit > 0:
        snapshot = snapshot[-limit:]
    return snapshot


def get_pipeline_ring_max() -> int:
    """The deque's maxlen — exposed so the HTTP endpoint and launcher
    dashboard share a single source of truth for the buffer size."""
    return _PIPELINE_RING_MAX


def _shorten_source_path(src: str, anchors) -> str:
    """Shorten an absolute ingest path to a source-type-relative citation.

    Strips everything up to and including the LAST occurrence of the first
    matching anchor in ``anchors`` (default ``['sources', 'Projects']``),
    preserving the source-type prefix (``confluence/``, ``github/``, ...) in
    ``<GENE src=...>`` and anchoring on nested
    ``.../sources/.../sources_attached/...`` layouts; falls back to the last
    three path segments (fixes #146's over-truncation). Issue #207 item 2 —
    anchors come from ``[ingestion] citation_path_anchors`` so non-owner /
    air-gap deployments don't leak owner path segments into citations. Returns
    ``""`` for empty or ``_``-prefixed synthetic sources.
    """
    if not src or src.startswith("_"):
        return ""
    parts = src.replace("\\", "/").split("/")
    for anchor in anchors or ():
        idx = -1
        for i, p in enumerate(parts):
            if p == anchor:
                idx = i
        if idx >= 0 and idx + 1 < len(parts):
            return "/".join(parts[idx + 1:])
    return "/".join(parts[-3:]) if len(parts) > 3 else src


class _stage_timer:
    """Context manager that records cymatix_pipeline_stage_seconds on exit.

    ``extra``: optional zero-arg callable evaluated at exit; its dict is
    merged into the ring entry (never into the histogram labels — payloads
    like per-signal sub-maps would mint unbounded label sets). Reserved
    ring keys (request_id/stage/ms/ts) always win over payload keys, and a
    raising ``extra`` drops the payload, never the entry.
    """

    __slots__ = ("stage", "labels", "extra", "_t0")

    def __init__(self, stage: str, labels: Optional[dict] = None,
                 extra: Optional[Callable[[], dict]] = None):
        self.stage = stage
        self.labels = labels or {}
        self.extra = extra

    def __enter__(self):
        self._t0 = _time.monotonic()
        return self

    def __exit__(self, *exc):
        elapsed = _time.monotonic() - self._t0
        try:
            _pipeline_stage_histogram().record(
                elapsed,
                {"stage": self.stage, **self.labels},
            )
        except Exception:
            pass  # never let telemetry break the pipeline
        try:
            if _pipeline_ring_enabled():
                rid = _pipeline_request_id.get()
                if rid:
                    entry = {}
                    if self.extra is not None:
                        try:
                            entry.update(self.extra() or {})
                        except Exception:
                            pass
                    entry.update({
                        "request_id": rid,
                        "stage": self.stage,
                        "ms": round(elapsed * 1000.0, 3),
                        "ts": _time.time(),
                    })
                    _pipeline_events.append(entry)
        except Exception:
            pass

def _record_ingest_stage(
    stage: str, seconds: float, extra: Optional[dict] = None,
) -> None:
    """Ingest counterpart of ``_stage_timer`` (issue #372).

    Same histogram (cymatix_pipeline_stage_seconds), same ring, but records
    an externally-measured duration — the tag/upsert stages are accumulated
    across a per-strand loop, so a with-block cannot capture them as one
    observation. Reserved ring keys win over ``extra`` keys, mirroring
    ``_stage_timer.__exit__``. Never raises: telemetry must not break ingest.
    """
    try:
        _pipeline_stage_histogram().record(seconds, {"stage": stage})
    except Exception:
        pass  # never let telemetry break the pipeline
    try:
        if _pipeline_ring_enabled():
            rid = _pipeline_request_id.get()
            if rid:
                entry = dict(extra or {})
                entry.update({
                    "request_id": rid,
                    "stage": stage,
                    "ms": round(seconds * 1000.0, 3),
                    "ts": _time.time(),
                })
                _pipeline_events.append(entry)
    except Exception:
        pass


def _model_load_ms_snapshot() -> dict:
    """Cumulative encoder-load ms by model name (perf slice 1, W0.2)."""
    try:
        from .telemetry import model_load_ms
        return model_load_ms()
    except Exception:
        return {}


# Thread pool for running sync pipeline/compressor calls from async
# context. W3.2 (gate G1): this pool IS the server's concurrency ceiling —
# qps saturated at 1.46 from c=2..16 with the old hardcoded 2 workers.
# Lazy + env-sized so the c-curve receipt can sweep sizes; the default
# stays 2 until a receipt picks the knee (benchmarks/dogfood/erb docs).
_executor = None
_executor_init_lock = threading.Lock()


def _resolve_executor_workers() -> int:
    raw = os.environ.get("CYMATIX_EXECUTOR_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            log.warning(
                "CYMATIX_EXECUTOR_WORKERS=%r is not an int; using default 2",
                raw,
            )
    return 2


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_init_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=_resolve_executor_workers(),
                    thread_name_prefix="cymatix-ribosome",
                )
    return _executor


# -- Compressor decoder prompt (3k fixed, tells the big model how to read context) --

# -- Adaptive decoder prompts (tiered by model capability) --------
#
# "full"      ~750 tokens — for small local models (e2b, qwen3:1.7b)
# "condensed" ~300 tokens — for medium local models (e4b, 8b)
# "minimal"   ~80 tokens  — for large local models (26b, 31b)
# "none"      0 tokens    — for API models (Claude, GPT) that don't need instructions

DECODER_FULL = """CRITICAL INSTRUCTIONS — READ BEFORE RESPONDING:

You have access to <expressed_context> blocks below. This is your ONLY source of
project-specific knowledge. You MUST use it as your primary source of truth.

MANDATORY BEHAVIOR:
1. ALWAYS read the <expressed_context> block FIRST before forming any response.
2. Base your answer on what the expressed context ACTUALLY SAYS, not on what you
   think a typical project might look like.
3. Use SPECIFIC details from the context: exact names, exact logic, exact structure.
   If the context says a function is called "merge_weekly_score", say that name.
   Do NOT substitute generic descriptions.
4. If the expressed context does not contain enough information to answer the
   question, say "My context does not cover this" — do NOT guess or hallucinate.
5. If the user's message conflicts with the expressed context, the user's message
   takes priority (it is the latest state).

The expressed context is compressed — each segment between --- dividers is one
knowledge unit selected specifically for this query. Filler has been removed.
What remains is the load-bearing information. Treat it as authoritative fact,
not as a suggestion.

DO NOT:
- Speculate about what the project "might" be or "likely" does
- Use words like "hypothesis", "implies", "suggests" when the context states facts
- Generate generic architectural advice that ignores the actual context
- Mention codons, genes, splicing, or DNA unless the user asks about memory internals"""

DECODER_CONDENSED = """The <expressed_context> below contains project data selected for your query.
Each <GENE> block is one knowledge unit with its source file path.

Extract the SPECIFIC value that answers the question. Look for exact numbers, names, and identifiers.
If a Facts: line is present, check it FIRST — it contains pre-extracted key-value pairs.
Answer with the exact value, not a description."""

DECODER_MINIMAL = """Answer using ONLY the <expressed_context> below. Do not guess beyond what it states."""

DECODER_NONE = ""

# MoE-specific decoder: front-loads extracted facts for sliding-window attention.
# Gemma 4's 5:1 SWA means only 1-in-6 layers see the full context.
# By placing a flat answer slate in the first ~200 tokens, every layer
# (including local SWA with 1024-token window) sees the key facts.
DECODER_MOE = """Answer the question using the ANSWER SLATE below.
The slate contains pre-extracted facts from the knowledge base.
Find the key that matches the question and return its EXACT value.

ANSWER SLATE:
{answer_slate}

If no slate key matches, check the <GENE> blocks below for the answer.
Extract and return the LITERAL value. Do NOT reason or speculate."""

# Model families that use MoE / sliding-window attention
MOE_MODEL_FAMILIES = ("gemma4",)

# Models at or below this param count get the same front-loaded treatment
# as MoE models — their limited capacity can't "look back" across 15K tokens
SMALL_MODEL_THRESHOLD_B = 10.0  # billion params — all local models get slate treatment
SMALL_MODEL_PATTERNS = {
    # model prefix → approximate param count in billions
    # All local models benefit from front-loaded KV facts — Kompress
    # compression loses specific values that the slate preserves.
    "qwen3:0.6b": 0.6, "qwen3:1.7b": 1.7, "qwen3:4b": 4.0, "qwen3:8b": 8.2,
    "gemma4:e2b": 2.0, "gemma4:e4b": 8.0,
    "llama3.2:3b": 3.0, "llama3.2:1b": 1.0,
    "phi-3.5:mini": 3.2, "gemma2:2b": 2.0,
}

# Issue #207 item 6: the exact-string table above requires an exact
# "family:tag" match, so it silently misses every model NOT explicitly
# listed — including whole families (mistral, deepseek, granite) that
# follow the same ":NNb" Ollama tag convention. This regex is the generic
# fallback: it parses a trailing ":<digits[.digits]>b" parameter-size tag
# (e.g. "mistral:7b" -> 7.0, "deepseek-r1:8b" -> 8.0, "granite3:2b" -> 2.0,
# "qwen3:0.6b" -> 0.6, "llama3.1:70b" -> 70.0) so any correctly-tagged model
# gets a size-based bucket instead of the pre-#207 always-"large" default.
_MODEL_PARAM_SIZE_RE = re.compile(r":(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def _parse_model_param_size_b(model_name: str) -> Optional[float]:
    """Parse a trailing ``:NNb`` parameter-size tag from a model name.

    Returns the size in billions of parameters, or ``None`` if the model
    name doesn't carry a recognizable ``:NNb`` suffix (Issue #207 item 6).
    """
    m = _MODEL_PARAM_SIZE_RE.search(model_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def resolve_model_capability_class(
    model_name: str,
    overrides: Optional[Dict[str, str]] = None,
) -> str:
    """Classify a compressor/ribosome model into a decoder capability class.

    Returns one of ``"moe"`` (sliding-window attention, needs a front-loaded
    answer slate), ``"small"`` (limited param capacity, same slate
    treatment), or ``"large"`` (no front-loading needed).

    Issue #207 item 6: the pre-existing ``MOE_MODEL_FAMILIES`` /
    ``SMALL_MODEL_PATTERNS`` tables require an exact model-string match and
    silently classified everything else as ``"large"`` — missing whole
    families (mistral, deepseek, granite) that use the same ``:NNb`` Ollama
    tag convention as the models already in the table. Resolution order:

      1. ``overrides`` (``[budget] decoder_mode_overrides``) — operator
         escape hatch, checked first. The first key that is a case-
         insensitive substring of ``model_name`` wins (dict iteration
         order == declaration order). Values are ``"moe"`` / ``"small"`` /
         ``"large"``. Not the same thing as the unrelated per-request
         ``DECODER_MODES`` prompt-template table (``resolve_decoder_mode``
         in ``retrieval/query_classifier.py``) — this overrides only the
         local compressor-model capability class.
      2. ``MOE_MODEL_FAMILIES`` prefix match -> ``"moe"``.
      3. ``SMALL_MODEL_PATTERNS`` exact match -> ``"small"``/``"large"`` by
         the hand-calibrated param count. Kept ahead of the generic parse
         because these values are measured, not inferred.
      4. Generic ``:NNb`` parameter-size parse (``_parse_model_param_size_b``)
         compared against ``SMALL_MODEL_THRESHOLD_B`` -- the new fallback.
      5. Unparseable / unknown -> ``"large"`` (matches the pre-#207 default
         for anything not in the table).

    Defaults (``overrides={}``) reproduce the prior behavior byte-for-byte
    for every model currently listed in the two tables.
    """
    name = (model_name or "").lower()
    for substr, mode in (overrides or {}).items():
        if substr.lower() in name:
            return mode
    if any(name.startswith(fam) for fam in MOE_MODEL_FAMILIES):
        return "moe"
    if name in SMALL_MODEL_PATTERNS:
        return "small" if SMALL_MODEL_PATTERNS[name] <= SMALL_MODEL_THRESHOLD_B else "large"
    parsed = _parse_model_param_size_b(name)
    if parsed is not None and parsed <= SMALL_MODEL_THRESHOLD_B:
        return "small"
    return "large"


# Stage 5 (2026-05-08): two new decoder modes for the small_moe class.
# See docs/specs/2026-05-08-stage-5-caller-model-class.md §6.
#   answer_slate_only — slate is the entire decoder context (~150 tokens),
#                        no <expressed_context> block. Used for arithmetic
#                        and factual classes on small_moe callers.
#   condensed_with_slate — slate FIRST (so attention locks before prose),
#                          then the condensed decoder prompt. Used for
#                          procedural/multi_hop/default on small_moe.
# The {answer_slate} placeholder is filled by _assemble at render time.
DECODER_ANSWER_SLATE_ONLY = """Answer the question using ONLY the ANSWER SLATE below.
The slate contains pre-extracted facts from the knowledge base.
Find the key that matches the question and return its EXACT value.

ANSWER SLATE:
{answer_slate}

If no slate key matches, reply: "(no slate key matched)"
Do NOT reason, speculate, or invent values."""

DECODER_CONDENSED_WITH_SLATE = """ANSWER SLATE (use first if a key matches):
{answer_slate}

The <expressed_context> below contains project data selected for your query.
If a Facts: line is present, check it FIRST — it contains pre-extracted key-value pairs.
Answer with the exact value, not a description."""

DECODER_MODES = {
    "full": DECODER_FULL,
    "condensed": DECODER_CONDENSED,
    "minimal": DECODER_MINIMAL,
    "none": DECODER_NONE,
    "moe": DECODER_MOE,
    # Stage 5 §6 — small_moe class branches.
    "answer_slate_only": DECODER_ANSWER_SLATE_ONLY,
    "condensed_with_slate": DECODER_CONDENSED_WITH_SLATE,
}

# Keep backward compatibility
RIBOSOME_DECODER = DECODER_FULL


# Shared marker injected when build_context has nothing useful to ship —
# either the knowledge store had no candidates ("denatured") or post-refinement
# scores fell below the FOCUSED floor on both axes ("abstain"). Both
# branches ship the same bytes so the small model's prompt-conditioning
# is identical regardless of which short-circuit fired. The semantic
# difference is observable only via context_health.status.
#
# Stage 6 (2026-05-08): the prose marker is replaced with a structured
# `<cymatix:no_match reason="..." do_not_answer="true"/>` token so a
# frontier agent can branch on a tag rather than a string match. The
# four spec-defined reasons (§6) map 1:1 onto the MissBlock.reason
# enum. _ABSTAIN_MARKER is kept for one release as a deprecated alias
# to keep tests/test_abstain_tier.py passing without modification.
_NO_MATCH_TAG = '<cymatix:no_match reason="{reason}" do_not_answer="true"/>'


def _no_match_token(reason: str) -> str:
    """Format the lowercase, fixed-attribute-order, self-closing tag.

    Defined as a function so tests can assert exact bytes via the
    function rather than scraping the string template. The four valid
    reasons are MissBlock.reason values; passing anything else returns
    the abstain form (defensive default — the discriminator never
    feeds an unknown reason here in practice).

    Legacy-compat surface (ADR 2026-05-14, Q3). The four whitelisted
    reasons are the pre-Stage-7 contract. Stage 7 added
    ``stale``/``cold``/``superseded`` to ``MissBlock.reason``; those
    reasons are NOT emitted as inline ``<cymatix:no_match/>`` tags by
    design, because Stage 7 demotions ship non-empty expressed context
    (the data is shown with ``stale_risk`` + ``refresh_targets``),
    which is semantically incompatible with the tag's
    ``do_not_answer="true"`` contract. New clients should branch on
    the structured ``MissBlock.reason`` / ``KnowBlock`` fields rather
    than scraping this tag. See
    ``docs/architecture/adr/2026-05-14-spec-vs-code-design-decisions.md``.
    """
    valid = {"abstain", "denatured", "sparse", "no_promoter_match"}
    if reason not in valid:
        reason = "abstain"
    return _NO_MATCH_TAG.format(reason=reason)


# Deprecated alias (one-release lifetime — Stage 6 §6). Existing tests
# read this; they migrate transparently because the value is the
# abstain form of the new tag.
_ABSTAIN_MARKER = _no_match_token("abstain")


def _env_truthy(name: str) -> bool:
    """Return True iff env var is set to a truthy value.

    Truthy values (case-insensitive): '1', 'true', 'yes', 'on'. Anything
    else (including unset) returns False. This is the 2-state variant of
    cymatix_context.launcher.app._env_truthy — defined locally to avoid a
    context_manager → launcher import edge.
    """
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


#: C1 rank-hash instrumentation (issue #339 Phase 0.4). One env gate for the
#: server capture and the bench harness both; unset or any non-truthy value is
#: OFF, which is the shipped behaviour. Read at call time via ``_env_truthy``
#: so a test can flip it after import.
BENCH_RANK_HASH_ENV = "CYMATIX_BENCH_RANK_HASH"

#: Byte-recipe version for the C1 hashes. Bump if the recipe ever changes.
BENCH_RANK_HASH_RECIPE = "c1-v3"


def _gene_set_sha256(gene_ids) -> str:
    """sha256 over the SORTED UNIQUE gene ids, '\\n'-joined, utf-8, no trailing
    newline, hex lowercase. Order-free set membership, recipe ``c1-v3``."""
    return hashlib.sha256(
        "\n".join(sorted(set(gene_ids))).encode("utf-8")
    ).hexdigest()


def _compute_foveated_caps(
    n: int,
    alpha: float,
    c_min: float,
    c_max: float = 1.0,
) -> list[float]:
    """Power-law per-document compression caps for foveated-splice.

    c_i = max(c_min, c_max · i^(-α))    for i ∈ [1, N]

    Returns a list of N floats in forward-rank order (caps[0] = rank-1 cap,
    caps[N-1] = rank-N cap). Caller reverses to pair with reverse-rank
    candidate placement.

    Spec: docs/specs/2026-05-03-foveated-splice-design.md §4.1
    """
    if n <= 0:
        return []
    return [max(c_min, c_max * ((i + 1) ** -alpha)) for i in range(n)]


# Splice-floor fix (J-space council kill-switch #1, 2026-07-06). The
# estimate_tokens heuristic averages ~4 chars/token on mixed content; the
# 0.9 headroom absorbs the per-document overhead the splice loop cannot
# see (legibility headers, <GENE> wrappers, separators, decoder prompt)
# so assembly's whole-document eviction loop does not fire on overflow.
_SPLICE_CHARS_PER_TOKEN = 4.0
_SPLICE_BUDGET_SAFETY = 0.9
_SPLICE_LEGACY_FLOOR = 1000


def _compute_splice_target(
    override: int,
    expression_tokens: int,
    n_candidates: int,
) -> int:
    """Per-document char target for the Step-4 splice loop.

    ``override > 0`` pins a fixed target (``1000`` == the exact legacy
    floor). ``override == 0`` (the default) distributes the expression
    char budget across the candidate set —
    ``int(expression_tokens · 4 · 0.9) // n`` — floored at the legacy
    1000 so no document ever gets *less* room than the old uniform cap.
    The old cap under-used the budget whenever
    ``n_candidates · 1000 chars`` fell short of ``expression_tokens``
    (12 × 1000 chars ≈ 3000 tokens vs the default 7000) and truncated
    any answer past char 1000 regardless of the query.
    """
    if override and override > 0:
        return int(override)
    budget_chars = int(
        float(expression_tokens) * _SPLICE_CHARS_PER_TOKEN * _SPLICE_BUDGET_SAFETY
    )
    per_doc = budget_chars // max(1, int(n_candidates))
    return max(_SPLICE_LEGACY_FLOOR, per_doc)


# Stage 5 (2026-05-08) §5: small_moe slate render — JSON-shaped, char-bounded,
# greedy-fill ordered by per-KV score (best-first; caller sorts upstream).
# See docs/specs/2026-05-08-stage-5-caller-model-class.md §5.
_SLATE_WRAPPER_OPEN = "<cymatix:slate>"
_SLATE_WRAPPER_CLOSE = "</cymatix:slate>"
_SLATE_MIN_VALUE_CHARS = 8  # Truncation rule per spec §5.


def _render_small_moe_slate(
    unique_slate: List[str],
    char_budget: int,
) -> str:
    """Render the small_moe answer slate as compact JSON within a char budget.

    Spec §5: greedy fill ordered by caller-provided rank, parse each line as
    `key=value`, dedup keys (first-write-wins). When adding a KV would
    exceed the budget, truncate that KV's value to fit (minimum
    ``_SLATE_MIN_VALUE_CHARS`` retained — drop the entry if the value cannot
    fit). Do NOT silently stop iterating: a low-rank short KV can still fit
    after a high-rank long one was truncated.

    The budget counts the rendered string the model actually sees, INCLUDING
    the wrapper tag, the JSON braces/quotes/commas, and per-KV separators.

    Returns the wrapped slate string `<cymatix:slate>{...}</cymatix:slate>` or an
    empty wrapper `<cymatix:slate>{}</cymatix:slate>` if no KV fits.
    """
    import json as _json
    # Reserve room for wrapper + the empty-object braces.
    wrapper_len = len(_SLATE_WRAPPER_OPEN) + len(_SLATE_WRAPPER_CLOSE)
    minimal_len = wrapper_len + 2  # the two braces of an empty object {}
    if char_budget <= minimal_len:
        # Budget too small to fit even an empty object — fail soft.
        return _SLATE_WRAPPER_OPEN + "{}" + _SLATE_WRAPPER_CLOSE

    # First pass: parse + dedup keys (first-write-wins).
    parsed: List[Tuple[str, str]] = []
    seen_keys: set[str] = set()
    for idx, line in enumerate(unique_slate):
        if "=" in line:
            k, _, v = line.partition("=")
            k = k.strip() or f"kv{idx}"
            v = v
        else:
            k = f"kv{idx}"
            v = line
        if k in seen_keys:
            continue
        seen_keys.add(k)
        parsed.append((k, v))

    # Greedy-fill: try each (k, v) in order; if it fits, keep it; if not,
    # truncate v to fit; if v can't fit, drop and continue (don't stop).
    accepted: dict[str, str] = {}
    for k, v in parsed:
        # Try with full value.
        candidate = dict(accepted)
        candidate[k] = v
        rendered = _json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
        if wrapper_len + len(rendered) <= char_budget:
            accepted = candidate
            continue
        # Try with truncated value.
        # Compute headroom: budget - wrapper - non-this-KV serialized cost.
        # Easier: binary-search on the value length until it fits or hits min.
        lo, hi = _SLATE_MIN_VALUE_CHARS, len(v)
        best_v: Optional[str] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            trial = dict(accepted)
            trial[k] = v[:mid]
            trial_rendered = _json.dumps(trial, ensure_ascii=False, separators=(",", ":"))
            if wrapper_len + len(trial_rendered) <= char_budget:
                best_v = v[:mid]
                lo = mid + 1
            else:
                hi = mid - 1
        if best_v is not None:
            accepted[k] = best_v
        # else: this KV can't even fit at min-length; drop and continue.

    rendered = _json.dumps(accepted, ensure_ascii=False, separators=(",", ":"))
    return _SLATE_WRAPPER_OPEN + rendered + _SLATE_WRAPPER_CLOSE


def _merge_subquery_candidates(
    sub_results: list,
    base_scores: dict,
) -> list:
    """Merge document lists from multiple sub-queries.

    Documents appearing in more sub-queries rank higher regardless of base score.
    Within the same hit count, base_score is the tiebreaker.
    Returns a deduplicated list ordered by (hit_count DESC, base_score DESC).
    """
    from collections import Counter
    seen: dict = {}
    hit_counts: Counter = Counter()
    for sub_list in sub_results:
        for doc in sub_list:
            hit_counts[doc.gene_id] += 1
            if doc.gene_id not in seen:
                seen[doc.gene_id] = doc
    return sorted(
        seen.values(),
        key=lambda g: (hit_counts[g.gene_id], base_scores.get(g.gene_id, 0.0)),
        reverse=True,
    )


class LazyRibosome:
    """Deferred-construction proxy for a heavy ribosome backend (#219 slice 2).

    Wraps a zero-arg factory (e.g. the DeBERTa hybrid ribosome: two
    DeBERTa-v3 models) and constructs it on first real use behind a
    double-checked lock. Until then the serving process pays nothing for
    it — part of the fix for the 20.3 GB-RSS-at-boot eager stack, where
    every process (tray backend, bench server, build workers) loaded the
    full encoder stack whether or not the workload touched it (the
    #176/#191 3-CUDA-context incident class).

    Any public attribute access (``re_rank``, ``splice``, ``encode``,
    ``classify_relations``, ``backend``, …) materializes the real object
    and forwards to it, so behavior after first use is identical to the
    eager construction. Private/dunder lookups raise AttributeError
    instead of forcing a load, so stdlib introspection (copy/pickle/repr
    machinery) can never accidentally pull in a multi-GB model.

    If the factory raises, the failure is logged once and the proxy
    permanently falls back to ``fallback`` (the disabled ribosome) — the
    same end state as the old eager try/except at manager init.

    ``loaded`` / ``peek()`` / ``label`` are non-forcing introspection
    hooks for GET /admin/components.
    """

    is_lazy_component = True

    def __init__(self, factory, fallback, label: str = ""):
        self._factory = factory
        self._fallback = fallback
        self._obj = None
        self._lock = threading.Lock()
        self._label = label

    @property
    def loaded(self) -> bool:
        """True once the real ribosome (or its fallback) is resident."""
        return self._obj is not None

    @property
    def label(self) -> str:
        return self._label

    def peek(self):
        """The materialized ribosome, or None — never triggers a load."""
        return self._obj

    def warm(self):
        """Force construction now ([hardware] lazy_encoders = false path)."""
        return self._materialize()

    def _materialize(self):
        obj = self._obj
        if obj is not None:
            return obj
        with self._lock:
            if self._obj is None:
                try:
                    self._obj = self._factory()
                    log.info(
                        "Lazy %s ribosome materialized on first use",
                        self._label or "backend",
                    )
                except Exception:
                    log.warning(
                        "%s backend failed to load, disabling ribosome",
                        self._label or "lazy", exc_info=True,
                    )
                    self._obj = self._fallback
            return self._obj

    def __getattr__(self, name: str):
        # Only reached when normal attribute lookup misses. Refuse
        # private/dunder names so hasattr() probes on internals and
        # stdlib introspection stay load-free.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._materialize(), name)


# Metadata keys the ingest seam recognizes as caller-supplied tag lists.
# Structured sources (OKF frontmatter, future adapters) pass these through
# `ingest(metadata=...)` and they merge additively with tagger output — the
# resulting rows in promoter_index / genes_fts / path_key_index are
# indistinguishable from tagger-produced tags.
_CALLER_TAG_KEYS = ("domains", "entities", "key_values")


def _merge_caller_tags(gene: Gene, metadata: Optional[Dict]) -> None:
    """Merge caller-supplied tag lists from ingest metadata into *gene*.

    Merge, don't bypass: the tagger has already run; caller values are
    prepended (so they survive any downstream cap, mirroring the tagger's
    own filename-domain prepend) and deduplicated against tagger output —
    case-insensitively for domains/entities (promoter_index lowercases on
    insert), exactly for key_values (values may be case-significant).

    Provable no-op when none of ``_CALLER_TAG_KEYS`` is present in
    ``metadata`` — the bench beds and every existing ingest caller are
    untouched.
    """
    if not metadata or not any(metadata.get(k) for k in _CALLER_TAG_KEYS):
        return

    def _prepend(supplied, existing, casefold: bool) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in list(supplied or []) + list(existing):
            text = str(value).strip()
            if not text:
                continue
            key = text.lower() if casefold else text
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged

    if metadata.get("domains"):
        gene.promoter.domains = _prepend(
            metadata["domains"], gene.promoter.domains, casefold=True
        )
    if metadata.get("entities"):
        gene.promoter.entities = _prepend(
            metadata["entities"], gene.promoter.entities, casefold=True
        )
    if metadata.get("key_values"):
        gene.key_values = _prepend(
            metadata["key_values"], gene.key_values, casefold=False
        )


class CymatixContextManager:
    """
    Main orchestrator. Sits between the client and the upstream LLM.

    Usage:
        cymatix = CymatixContextManager(config)
        cymatix.ingest("some long document")

        # Per turn:
        window = cymatix.build_context("user query")
        # Inject window into the LLM request

        # After response:
        cymatix.learn("user query", "assistant response")
    """

    def __init__(self, config: CymatixConfig):
        self.config = config

        # Encoder daemon (Fork 1 slice 1). Backends never import config, so
        # the manager is the one fan-out point that hands them the configured
        # URL; env CYMATIX_ENCODER_URL still wins over it inside the client.
        # Empty (the default) means every encoder seam stays in-process,
        # byte-identical to the pre-daemon path. The getattr is defensive for
        # tests that build partial config objects.
        from .backends import encoder_client
        from .config import EncoderDaemonConfig as _EncoderDaemonConfig
        encoder_client.configure(
            getattr(config, "encoder_daemon", _EncoderDaemonConfig()).url
        )

        # Blend-layer mode (Issue #255 / audit §4 item 5). "legacy" is
        # byte-identical to the shipped additive blend. Validated first so a
        # typo in cymatix.toml fails fast — before any genome construction —
        # mirroring the store's rerank_combinator guard.
        from .scoring.blend import VALID_BLEND_MODES as _VALID_BLEND_MODES
        self._blend_mode = config.retrieval.blend_mode
        if self._blend_mode not in _VALID_BLEND_MODES:
            raise ValueError(
                "[retrieval] blend_mode must be one of "
                f"{_VALID_BLEND_MODES}; got {self._blend_mode!r}"
            )

        # Activity tracking for GET /admin/components.
        # Bumped on every /context and /ingest call by server.py. Used to
        # derive running/idle status for the launcher's tools panel.
        import time as _time
        self._last_activity_ts: float = _time.time()

        # Token counter (session + lifetime). Persisted next to genome.db so
        # the lifetime counter survives restarts. See cymatix_context/metrics.py
        # and the /metrics/tokens endpoint.
        from pathlib import Path as _Path
        from .telemetry.metrics import TokenCounter
        _genome_path = _Path(config.genome.path)
        if str(_genome_path) == ":memory:":
            # In-memory tests: keep metrics in-memory too (write to a tmp path
            # that we won't actually flush; persistence is opt-in via flush()).
            import tempfile as _tempfile
            _metrics_path = _Path(_tempfile.gettempdir()) / "cymatix_metrics_test.json"
        else:
            _metrics_path = _genome_path.parent / "metrics.json"
        self.token_counter: TokenCounter = TokenCounter(persist_path=_metrics_path)

        # ΣĒMA codec (optional — requires sentence-transformers).
        # #219 slice 2: armed lazily by default ([hardware] lazy_encoders).
        # A serving process that never touches a semantic path no longer
        # pays the MiniLM load at boot (part of the 20.3 GB-RSS eager
        # stack); the first encode constructs the model under a
        # double-checked lock. lazy_encoders = false restores the
        # pre-slice eager warmup so first-query latency is paid at boot.
        self._sema_codec = None
        self._lazy_encoders = bool(getattr(config.hardware, "lazy_encoders", True))
        # Issue #227: gate ingest-time SEMA on an explicit knob (default True).
        # When false, the SEMA codec is never constructed, so MiniLM is not
        # loaded (no per-worker OOM in lexical / multi-worker bench runs);
        # gene.embedding stays unset and TCM uses its text-derived fallback.
        self._sema_embed_on_ingest = bool(
            getattr(config.ingestion, "sema_embed_on_ingest", True)
        )
        try:
            from .backends.sema import LazySemaCodec, sema_available
            # Fork 1 slice 1: with a daemon URL active the daemon owns MiniLM,
            # so sentence-transformers need not be installed here; without one
            # the availability probe gates construction exactly as before.
            _encoder_url = encoder_client.active_url()
            if self._sema_embed_on_ingest and (_encoder_url or sema_available()):
                if _encoder_url:
                    self._sema_codec = encoder_client.RemoteSemaCodec(
                        model_name=self.config.ingestion.sema_model,
                        url=_encoder_url,
                    )
                else:
                    self._sema_codec = LazySemaCodec(
                        model_name=self.config.ingestion.sema_model)  # #207 item 1
                if self._lazy_encoders:
                    log.info(
                        "ΣĒMA codec armed (lazy) — model loads on first semantic call"
                    )
                else:
                    self._sema_codec.warm()
                    log.info("ΣĒMA codec loaded — semantic retrieval enabled")
            elif not self._sema_embed_on_ingest:
                log.info("SEMA off — [ingestion] sema_embed_on_ingest=false (#227)")
            else:
                log.info("sentence-transformers not installed — ΣĒMA disabled")
        except ImportError:
            log.info("sentence-transformers not installed — ΣĒMA disabled")
        except Exception:
            self._sema_codec = None
            log.warning("ΣĒMA codec failed to load", exc_info=True)

        # BGE-M3 dense codec (Tier-0 PR-1, 2026-05-16). Lazy-loaded on the
        # first ingest when [ingestion] dense_embed_on_ingest is true, so a
        # manager that never ingests (or runs with the knob off) pays no
        # model-load cost. ingest() batch-encodes all strands of a document
        # through this and passes each vector to genome.upsert_doc.
        self._dense_codec = None  # type: ignore[var-annotated]

        # KnowledgeStore (SQLite storage) — swapped for a ShardedGenomeAdapter when
        # CYMATIX_USE_SHARDS=1 and the configured path is a routing DB. Writes
        # become no-ops in that mode; suitable for read-heavy serving and
        # benchmarks until ingest-time sharding (spec Task 6) lands.
        # Every config-driven constructor knob comes from the shared
        # build_genome_kwargs — the single source of truth this path shares
        # with /admin/swap-db, so a hot-swapped store is constructed exactly
        # like the boot store (the 2026-08-09 receipt-invalidation drift
        # class: knobs threaded here but hand-copied out of sync in the
        # swap path silently reset to constructor defaults after a swap).
        from .sharding import build_genome_kwargs, open_read_source
        self.genome = open_read_source(
            genome_path=config.genome.path,
            **build_genome_kwargs(config, sema_codec=self._sema_codec),
        )

        # WS3: hand the symbol-expansion cap to the genome's co-activation
        # expansion (top-K referenced defs by centrality).
        _apply_symbol_expansion_cap(
            self.genome, config.retrieval.symbol_expansion_cap
        )

        # Persistence manager (distributed knowledge store clones)
        self._replication_mgr = None
        if config.genome.replicas:
            from .replication import ReplicationManager
            self._replication_mgr = ReplicationManager(
                master=config.genome.path,
                replicas=config.genome.replicas,
                sync_interval=config.genome.replica_sync_interval,
            )
            self.genome.set_replication_manager(self._replication_mgr)

        # Chunker (deterministic text splitting). symbol_graph threads the
        # [ingestion] flag down so flag-off ingest skips symbol extraction
        # entirely (WS2 review FIX-3), not just emission.
        self.chunker = CodonChunker(
            max_chars_per_strand=4000,
            symbol_graph=bool(getattr(config.ingestion, "symbol_graph", False)),
        )
        self.encoder = CodonEncoder()

        # Compressor (small model codec) — explicit opt-in only. Legacy/default
        # Ollama compressor config stays in the file for future use but is
        # intentionally ignored unless a supported backend is selected.
        self.ribosome = Ribosome(
            backend=DisabledBackend(),
            encoder=self.encoder,
            splice_aggressiveness=config.budget.splice_aggressiveness,
        )

        effective_backend = config.ribosome.effective_backend

        if effective_backend == "litellm":
            try:
                litellm_backend = LiteLLMBackend(
                    model=config.ribosome.litellm_model,
                    base_url=config.ribosome.claude_base_url,  # reuse proxy URL
                    max_tokens=config.budget.ribosome_tokens,
                    timeout=config.ribosome.timeout,
                )
                self.ribosome = Ribosome(
                    backend=litellm_backend,
                    encoder=self.encoder,
                    splice_aggressiveness=config.budget.splice_aggressiveness,
                )
                log.info("Using LiteLLM ribosome (model=%s, proxy=%s)",
                         config.ribosome.litellm_model,
                         config.ribosome.claude_base_url or "direct")
            except Exception:
                log.warning("LiteLLMBackend failed to load, disabling ribosome", exc_info=True)
        elif effective_backend == "deberta":
            def _build_deberta_ribosome(
                _config: CymatixConfig = config,
                _encoder: CodonEncoder = self.encoder,
            ):
                """Construct the DeBERTa hybrid ribosome (two DeBERTa-v3 loads).

                Construction args are byte-for-byte the old eager ones —
                only the WHEN moves (#219 slice 2).
                """
                from .backends.deberta_backend import DeBERTaRibosome
                ollama_backend = OllamaBackend(
                    model=_config.ribosome.model,
                    base_url=_config.ribosome.base_url,
                    timeout=_config.ribosome.timeout,
                    keep_alive=_config.ribosome.keep_alive,
                    warmup=_config.ribosome.warmup,
                )
                ollama_ribosome = Ribosome(
                    backend=ollama_backend,
                    encoder=_encoder,
                    splice_aggressiveness=_config.budget.splice_aggressiveness,
                )
                return DeBERTaRibosome(
                    rerank_model_path=_config.ribosome.rerank_model_path,
                    splice_model_path=_config.ribosome.splice_model_path,
                    nli_model_path=_config.ribosome.nli_model_path,
                    ollama_ribosome=ollama_ribosome,
                    device=_config.ribosome.device,
                    splice_threshold=_config.ribosome.splice_threshold,
                    nli_splice_bonus=_config.ribosome.nli_splice_bonus,
                    nli_splice_penalty=_config.ribosome.nli_splice_penalty,
                    rerank_pretrained=_config.ingestion.rerank_model,
                )

            if self._lazy_encoders:
                # #219 slice 2: defer the two DeBERTa-v3 model loads to the
                # first re_rank/splice/classify call. On load failure the
                # proxy falls back to the disabled ribosome — same end state
                # as the eager except-branch below.
                self.ribosome = LazyRibosome(
                    factory=_build_deberta_ribosome,
                    fallback=self.ribosome,
                    label="deberta",
                )
                log.info(
                    "DeBERTa hybrid ribosome armed (lazy) — loads on first re_rank/splice"
                )
            else:
                try:
                    self.ribosome = _build_deberta_ribosome()
                    log.info("Using DeBERTa hybrid ribosome (re_rank + splice accelerated)")
                except Exception:
                    log.warning("DeBERTa backend failed to load, disabling ribosome", exc_info=True)
        else:
            log.info(
                "Ribosome disabled — only explicit LiteLLM or DeBERTa backends are honored "
                "(configured enabled=%s backend=%s)",
                config.ribosome.enabled,
                config.ribosome.backend,
            )

        # CPU tagger (Phase 1: spaCy + regex, no LLM calls)
        self._cpu_tagger = None
        if config.ingestion.backend in ("cpu", "hybrid"):
            try:
                from .tagger import CpuTagger
                self._cpu_tagger = CpuTagger(synonym_map=config.synonym_map)
                log.info("CpuTagger loaded — CPU-native ingestion enabled (backend=%s)",
                         config.ingestion.backend)
            except ImportError:
                log.warning("spaCy not installed — CpuTagger disabled, falling back to Ollama")
            except Exception:
                log.warning("CpuTagger failed to load, falling back to Ollama", exc_info=True)

        # Adaptive decoder prompt based on downstream model capability
        self._decoder_mode = config.budget.decoder_mode
        self._decoder_prompt = DECODER_MODES.get(self._decoder_mode, DECODER_FULL)

        # Answer-slate mode: front-loads KV facts for models that struggle
        # with long-range extraction. Applies to:
        #   1. MoE models (gemma4) — sliding-window attention misses distant tokens
        #   2. Sub-4B models — limited capacity can't attend across 15K tokens
        # Issue #207 item 6: classification (incl. the generic ":NNb" parse
        # fallback + [budget] decoder_mode_overrides) lives in
        # resolve_model_capability_class.
        model_name = config.ribosome.active_model.lower()
        _capability_class = resolve_model_capability_class(
            model_name, overrides=config.budget.decoder_mode_overrides,
        )
        is_moe = _capability_class == "moe"
        is_small = _capability_class == "small"
        self._is_moe = is_moe or is_small
        if self._is_moe:
            log.info(
                "Answer-slate mode enabled for %s (%s)",
                config.ribosome.model,
                "MoE/SWA" if is_moe else "sub-4B",
            )

        # Pending persistence buffer -- documents from background persistence
        # that haven't committed to SQLite yet. Checked during Step 2
        # so follow-up queries don't lose context from the previous turn.
        self._pending: List[Gene] = []
        self._pending_lock = threading.Lock()

        # Cymatics (frequency-domain re_rank + splice, replaces LLM calls)
        self._use_cymatics = config.cymatics.enabled
        if self._use_cymatics:
            # #357: [cymatics] peak_width, when explicitly set, overrides the
            # Q-factor derivation (previously it was parsed but never read).
            if config.cymatics.peak_width is not None:
                self._cymatics_peak_width = float(config.cymatics.peak_width)
            else:
                from .scoring.cymatics import aggressiveness_to_peak_width
                self._cymatics_peak_width = aggressiveness_to_peak_width(
                    config.budget.splice_aggressiveness
                )
        else:
            self._cymatics_peak_width = 3.0

        # TCM session context (Howard & Kahana 2002 temporal drift)
        self._tcm_session = None
        try:
            from .scoring.tcm import SessionContext
            self._tcm_session = SessionContext(n_dims=20, beta=0.5)
            log.info("TCM session context initialized (20D, beta=0.5)")
        except Exception:
            log.debug("TCM not available", exc_info=True)

        # Shadow pool (soft elimination — documents cut from top-k keep
        # residual weight, eligible for Lagrange pull-back if the
        # winners' cluster looks like a gravity well rather than merit).
        self._last_shadow_pool: List[Gene] = []
        self._last_shadow_scores: Dict[str, float] = {}

        # Cold-tier retrieval markers (C.2 of B->C, 2026-04-10).
        # Set by _express() when cold-tier fallthrough actually fires
        # so the response builder can report cold_tier_used in the
        # agent metadata. Reset on every build_context call.
        self._last_cold_tier_used: bool = False
        self._last_cold_tier_count: int = 0

        # Stage 7 (2026-05-08, spec §5) — per-process mtime cache for
        # the freshness gate. Keyed on absolute source path; value is
        # ``(mtime, cached_at)``. Lives on the manager (per-batch
        # state) rather than KnowledgeStore so /admin/refresh can clear it
        # without touching the DB. TTL defaults to 60s — see
        # ``cymatix_context.freshness.DEFAULT_CACHE_TTL_S``.
        self._mtime_cache: Dict[str, Tuple[float, float]] = {}

        # Stage 7 (spec §6) — last cold-peek refresh_targets so the
        # /context route can attach them to a MissBlock(reason="cold")
        # without re-running the cold-tier query. Reset per build.
        self._last_cold_peek_targets: List[str] = []

        # Session buffer -- accumulates query+response pairs for consolidation
        self._session_buffer: List[Tuple[str, str]] = []
        self._session_buffer_lock = threading.Lock()
        self._session_learn_count = 0
        self._consolidation_threshold = 10  # auto-consolidate every N learns

        # Compaction timer
        self._last_compact = time.time()

    # -- Dense codec lazy-loader (Tier-0 PR-1, 2026-05-16) ----------------------

    def _get_dense_codec(self):
        """Lazy-load the BGE-M3 dense codec for inline ingest encoding.

        Returns the codec, or ``None`` if dense-on-ingest is disabled or the
        codec cannot be constructed (e.g. sentence-transformers / FlagEmbedding
        not installed). ingest() treats ``None`` as "skip dense encoding" and
        the genome stores rows with a NULL ``embedding_dense_v2`` column.

        Goes through ``get_shared_codec`` like KnowledgeStore does (Fork 1
        slice 1): the private codec this used to construct directly was a
        second, unshared ~2 GB model in the same process, was pinned to the
        ctor's "cpu" default even under a CUDA config, and — being outside the
        shared factory — was the one dense path an encoder daemon could not
        substitute.
        """
        if not self.config.ingestion.dense_embed_on_ingest:
            return None
        if self._dense_codec is None:
            try:
                from .backends.bgem3_codec import (
                    get_shared_codec,
                    shared_dense_codec_enabled,
                )
                from .hardware import resolve_layer_device
                self._dense_codec = get_shared_codec(
                    dim=self.config.retrieval.dense_embedding_dim,
                    model_name=self.config.retrieval.dense_model,  # #207
                    device=resolve_layer_device("dense"),
                    share=shared_dense_codec_enabled(),
                )
                log.info("BGE-M3 dense codec loaded — dense vectors written at ingest")
            except Exception:
                log.warning(
                    "BGE-M3 dense codec failed to load — ingest will store "
                    "NULL embedding_dense_v2 (backfill later)",
                    exc_info=True,
                )
                return None
        return self._dense_codec

    # -- Ingest: add new content to the knowledge store -------------------------

    def ingest(self, content: str, content_type: str = "text", metadata: Optional[Dict] = None) -> List[str]:
        """
        Pack new content and store in the knowledge store.
        Call for documents, files, or conversation history.
        Returns list of gene_ids created.
        """
        # Ingest stage timers (issue #372): tag this call with a request id
        # so the ring attributes the ingest_* stage entries, mirroring
        # build_context. Stages recorded at the bottom: ingest_chunk,
        # ingest_encode (only when an encoder actually ran), ingest_tag,
        # ingest_upsert, ingest_commit — the commit entry carries the
        # per-document commit-count delta (W0.3 trace-hook counter).
        if _pipeline_ring_enabled():
            _pipeline_request_id.set("ing" + _uuid.uuid4().hex[:9])
        _commit_c0 = getattr(self.genome, "commit_count", 0)

        _t0 = _time.monotonic()
        strands = self.chunker.chunk(content, content_type=content_type, metadata=metadata)
        _chunk_s = _time.monotonic() - _t0
        gene_ids = []
        total_strands = len(strands)

        # Accept either metadata["path"] or metadata["source_id"] — the HTTP
        # /ingest contract historically documented both, but only "path" was
        # honored. Alias so callers using "source_id" get proper provenance
        # (source_kind, volatility_class) populated downstream.
        source_path = None
        if metadata:
            source_path = metadata.get("path") or metadata.get("source_id")

        # Batch-encode ΣĒMA vectors if codec available. _encode_s accumulates
        # across BOTH encoder blocks (sema + dense) into the single
        # encode-if-enabled ingest stage (#372).
        _encode_s = 0.0
        sema_vectors = None
        if self._sema_codec is not None:
            _t0 = _time.monotonic()
            try:
                texts = [s.content[:1000] for s in strands]  # Cap for encoder
                sema_vectors = self._sema_codec.encode_batch(texts)
            except Exception:
                log.debug("ΣĒMA batch encoding failed, skipping")
            _encode_s += _time.monotonic() - _t0

        # Batch-encode BGE-M3 dense vectors (Tier-0 PR-1, 2026-05-16). One
        # codec call for every strand of the document — far cheaper than the
        # per-strand encode upsert_doc would otherwise do. Bound each passage
        # to [ingestion] dense_passage_char_cap (#207 — default 2000, byte-
        # identical to the prior PASSAGE_CHAR_CAP literal; must stay identical
        # to the query-side and backfill slices) with task="passage" (the
        # codec contract). Soft-fails: on any error dense_vectors stays None
        # and upsert_doc stores NULL embedding_dense_v2. Disabled entirely
        # when the config knob is off (_get_dense_codec returns None). Write
        # path only — retrieval still gates on [retrieval]
        # dense_embedding_enabled.
        dense_vectors = None
        dense_codec = self._get_dense_codec()
        if dense_codec is not None:
            _t0 = _time.monotonic()
            try:
                cap = self.config.ingestion.dense_passage_char_cap
                dense_texts = [s.content[:cap] for s in strands]
                dense_vectors = dense_codec.encode_batch(
                    dense_texts, task="passage"
                )
            except Exception:
                log.warning(
                    "BGE-M3 dense batch encoding failed — strands stored with "
                    "NULL embedding_dense_v2 (backfill later)",
                    exc_info=True,
                )
                dense_vectors = None
            _encode_s += _time.monotonic() - _t0

        use_cpu = (
            self._cpu_tagger is not None
            and self.config.ingestion.backend in ("cpu", "hybrid")
        )

        # WS2: (gene_id, defs, refs) per code chunk, for symbol-graph emission.
        chunk_syms: List[tuple] = []

        # ── Pass 1 (tag): pack every strand into a Gene BEFORE any write.
        # Issue #372 restructure — tagging (a CPU-model call per strand) must
        # not run inside the document write transaction below, where it would
        # hold the store's write lock; same never-encode-under-the-lock rule
        # as upsert_doc's dense encode.
        _t0 = _time.monotonic()
        prepared: List[tuple] = []  # (strand, gene, dense_vec)
        for i, strand in enumerate(strands):
            if use_cpu:
                gene = self._cpu_tagger.pack(
                    strand.content,
                    content_type=content_type,
                    source_id=source_path,
                    sequence_index=strand.sequence_index,
                )
            else:
                gene = self.ribosome.encode(strand.content, content_type=content_type)
            # Preserve sequence index from chunking
            gene.promoter.sequence_index = strand.sequence_index
            if metadata:
                gene.promoter.metadata.update(dict(metadata))
            gene.is_fragment = strand.is_fragment
            apply_metadata_hints(
                gene,
                metadata,
                content_type=content_type,
                total_strands=total_strands,
            )
            # Structured-source seam: caller-supplied domains/entities/
            # key_values merge additively with tagger output before the
            # document reaches upsert_doc and its index builders.
            _merge_caller_tags(gene, metadata)
            # Store source file path for change-based decay
            if source_path:
                gene.source_id = source_path
                # Phase 1 of agent-context-index spec: populate provenance
                # (source_kind, volatility_class, observed_at,
                # last_verified_at) at ingest so the packet builder has
                # real freshness data without needing a backfill sweep.
                apply_provenance(
                    gene,
                    source_path,
                    observed_at=gene.observed_at,
                    content_type=content_type,
                )
            else:
                apply_provenance(
                    gene,
                    observed_at=gene.observed_at,
                    content_type=content_type,
                )
            # Attach ΣĒMA vector
            if sema_vectors is not None and i < len(sema_vectors):
                gene.embedding = sema_vectors[i]

            # Tier-0 PR-1 (2026-05-16): hand the precomputed BGE-M3 dense
            # vector to upsert_doc so it persists embedding_dense_v2 without
            # re-encoding per strand. None when dense-on-ingest is disabled
            # or the batch encode failed — upsert_doc then stores NULL.
            dense_vec = (
                dense_vectors[i]
                if dense_vectors is not None and i < len(dense_vectors)
                else None
            )
            prepared.append((strand, gene, dense_vec))
        _tag_s = _time.monotonic() - _t0

        # ── Pass 2 (upsert): ONE transaction per document (issue #372).
        # deferred_commits() folds the per-strand upserts, tier demotions,
        # parent doc, and CHUNK_OF/symbol edges into a single durable commit
        # — one fsync per document instead of one per strand. Crash
        # semantics: an exception rolls the WHOLE document back; partial-
        # document rollback is acceptable and preferable to the old
        # partial-strand states (a committed gene with no CHUNK_OF edges).
        # Stores without the seam (e.g. the sharded read adapter) fall back
        # to the historical commit-per-call behavior via nullcontext.
        _txn_factory = getattr(self.genome, "deferred_commits", None)
        _txn = _txn_factory() if callable(_txn_factory) else _contextlib.nullcontext()
        _t0 = _time.monotonic()
        _upsert_s = 0.0
        with _txn:
            for strand, gene, dense_vec in prepared:
                # Density gate now lives in genome.upsert_doc() itself so that
                # bulk ingest scripts (ingest_steam.py, ingest_all.py, etc.)
                # that call upsert_gene directly also respect it. The gate
                # reads the final lifecycle tier back onto the document object
                # and sets compression_tier accordingly during the INSERT.
                # See cymatix_context/genome.py:apply_density_gate for the logic.
                gid = self.genome.upsert_doc(gene, embedding_dense_v2=dense_vec)
                gene_ids.append(gid)

                # WS2: stash this chunk's symbols (set on the strand metadata by the
                # symbol-aware code chunker) keyed by gene id, for emission below.
                _sm = getattr(strand, "metadata", None) or {}
                if _sm.get("defs") or _sm.get("refs"):
                    chunk_syms.append((gid, _sm.get("defs", []), _sm.get("refs", [])))

                # If the gate demoted the document to heterochromatin, the content
                # column is still populated — compress_to_heterochromatin()
                # drops it and strips SPLADE/FTS indices. Run this post-insert
                # for consistency with the historical behavior.
                if gene.chromatin == ChromatinState.HETEROCHROMATIN and gene.embedding is not None:
                    self.genome.compress_to_heterochromatin(gid)
                elif gene.chromatin == ChromatinState.EUCHROMATIN:
                    self.genome.compress_to_euchromatin(gid)

            # Layered fingerprints: create a parent document when a file chunks
            # into N >= 2 strands. Parent aggregates child fingerprints at
            # query time so multi-chunk hits surface the whole file.
            # See docs/FUTURE/LAYERED_FINGERPRINTS.md. Inside the document
            # transaction: a failure rolls back only its own SAVEPOINT write
            # units, so the already-staged strands really are "still
            # ingested" — they commit at scope exit.
            if len(gene_ids) >= 2 and source_path:
                try:
                    parent_gid = self._upsert_parent_doc(
                        source_path=source_path,
                        child_gene_ids=gene_ids,
                        original_content=content,
                    )
                    if parent_gid:
                        log.debug("Created parent gene %s for %d chunks of %s",
                                  parent_gid, len(gene_ids), source_path)
                except Exception:
                    log.warning("Parent gene creation failed for %s — chunks still ingested",
                                source_path, exc_info=True)

            # WS2: build the symbol graph from this file's chunks — index symbol
            # definitions and emit referencing-chunk -> defining-chunk edges
            # (intra-file resolution; high precision). Best-effort: never break ingest.
            if (content_type == "code" and chunk_syms
                    and getattr(self.config.ingestion, "symbol_graph", False)):
                try:
                    self._emit_symbol_graph(chunk_syms)
                except Exception:
                    log.warning(
                        "symbol-graph emission failed for %s — chunks still ingested",
                        source_path, exc_info=True,
                    )
            _upsert_s = _time.monotonic() - _t0
        # The scope exit above pays the document's single durable commit
        # (plus any WAL checkpoint the cadence deferred), so commit time is
        # everything after the body finished. On the nullcontext fallback
        # commits happened per call inside _upsert_s and _commit_s ≈ 0.
        _commit_s = _time.monotonic() - _t0 - _upsert_s
        _commits = getattr(self.genome, "commit_count", 0) - _commit_c0

        _record_ingest_stage("ingest_chunk", _chunk_s)
        if _encode_s > 0.0:
            _record_ingest_stage("ingest_encode", _encode_s)
        _record_ingest_stage("ingest_tag", _tag_s)
        _record_ingest_stage("ingest_upsert", _upsert_s)
        _record_ingest_stage("ingest_commit", _commit_s,
                             extra={"commits": _commits})

        log.info(
            "Ingested %d strands from %s content (%d chars) — chunk %.1fms "
            "encode %.1fms tag %.1fms upsert %.1fms commit %.1fms "
            "(%d commits)",
            len(gene_ids), content_type, len(content),
            _chunk_s * 1000.0, _encode_s * 1000.0, _tag_s * 1000.0,
            _upsert_s * 1000.0, _commit_s * 1000.0, _commits,
        )
        return gene_ids

    @staticmethod
    def _make_parent_doc_id(source_path: str) -> str:
        """Deterministic parent gene_id from source path.

        Uses a distinct hash input (suffix "::parent") so parent IDs
        can't collide with content-hashed child gene_ids.
        """
        return hashlib.sha256(
            (source_path + "::parent").encode("utf-8")
        ).hexdigest()[:16]

    def _upsert_parent_doc(
        self,
        source_path: str,
        child_gene_ids: List[str],
        original_content: str,
    ) -> Optional[str]:
        """Create or refresh a parent document for a multi-chunk file.

        Parent shape:
            gene_id      — deterministic from source_path
            content      — first 1024 chars of original file
            fragments       — ordered list of child gene_ids (reassembly key)
            key_values   — [chunk_count=N, total_size_bytes=B, is_parent=true]
            is_fragment  — False
            sequence_index = -1 (file-level sentinel)

        Also inserts CHUNK_OF edges from each child to the parent.
        """
        parent_gid = self._make_parent_doc_id(source_path)
        n_chunks = len(child_gene_ids)
        total_bytes = len(original_content)

        parent = Gene(
            gene_id=parent_gid,
            content=original_content[:1024],
            complement=f"File-level parent aggregating {n_chunks} chunks of {source_path}",
            codons=list(child_gene_ids),
            key_values=[
                f"chunk_count={n_chunks}",
                f"total_size_bytes={total_bytes}",
                "is_parent=true",
            ],
            promoter=PromoterTags(sequence_index=-1),
            epigenetics=EpigeneticMarkers(),
            chromatin=ChromatinState.OPEN,
            is_fragment=False,
            source_id=source_path,
        )
        apply_provenance(parent, source_path, content_type="text")
        # apply_gate=False: parents are metadata aggregators, not content —
        # they should not be density-gated into heterochromatin.
        self.genome.upsert_doc(parent, apply_gate=False)

        edges = [
            (child_gid, parent_gid, int(StructuralRelation.CHUNK_OF), 1.0)
            for child_gid in child_gene_ids
        ]
        self.genome.store_relations_batch(edges)
        return parent_gid

    def _emit_symbol_graph(self, chunk_syms: List[tuple]) -> None:
        """Index symbol definitions and emit SYMBOL_REF edges for one file (WS2).

        ``chunk_syms`` is a list of ``(gene_id, defs, refs)``. Resolution is
        **intra-file**: a chunk's reference resolves to a sibling chunk in the
        same file that defines that symbol. This keeps edges high-precision (it
        sidesteps the "every ``process()`` links to every ``process()``" blow-up
        of naive cross-file name matching); cross-file resolution is a query-time
        concern handled via the global ``symbol_defs`` index.
        """
        store_defs = getattr(self.genome, "store_symbol_defs", None)
        store_edges = getattr(self.genome, "store_relations_batch", None)
        if not callable(store_defs) or not callable(store_edges):
            return  # e.g. sharded read adapter — writes are no-ops

        # 1) index definitions + build the intra-file symbol -> [gene_id] map
        def_rows: List[tuple] = []
        local: Dict[str, List[str]] = {}
        for gid, defs, _refs in chunk_syms:
            for sym in defs:
                def_rows.append((sym, gid, "def"))
                local.setdefault(sym, []).append(gid)
        store_defs(def_rows)

        # 2) resolve each chunk's references to sibling definers -> SYMBOL_REF edges
        edges: List[tuple] = []
        seen = set()
        for gid, _defs, refs in chunk_syms:
            for sym in refs:
                for dgid in local.get(sym, ()):
                    if dgid == gid:
                        continue  # a chunk referencing its own definition is not an edge
                    key = (gid, dgid)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append((gid, dgid, int(StructuralRelation.SYMBOL_REF), 1.0))
        if edges:
            store_edges(edges)

    async def ingest_async(self, content: str, content_type: str = "text", metadata: Optional[Dict] = None) -> List[str]:
        """Async wrapper for ingest -- runs compressor calls in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_get_executor(), self.ingest, content, content_type, metadata)

    # -- Build context: the main per-turn operation --------------------

    def _should_use_slate(
        self,
        downstream_model: Optional[str] = None,
        caller_model_class: str = "generic",
    ) -> bool:
        """Check if answer-slate mode should activate for this request.

        Stage 5 (2026-05-08): caller_model_class overrides the legacy
        detection axis (spec §4 behavior matrix):
          - small_moe → always ON
          - frontier  → always OFF
          - generic   → preserve legacy detection (regression baseline)

        Legacy generic-branch detection activates for:
          1. Server-level MoE detection (compressor model is gemma4 etc.)
          2. Per-request downstream model detection (sub-4B or MoE family)
        """
        if caller_model_class == "small_moe":
            return True
        if caller_model_class == "frontier":
            return False
        # generic — preserve legacy behavior byte-identical to pre-Stage-5.
        if self._is_moe:
            return True
        if downstream_model:
            dm = downstream_model.lower()
            if any(dm.startswith(fam) for fam in MOE_MODEL_FAMILIES):
                return True
            if SMALL_MODEL_PATTERNS.get(dm, 999) <= SMALL_MODEL_THRESHOLD_B:
                return True
        return False

    # ─── Stage 4: per-classifier floor / alpha lookup ────────────────────

    # Legacy global constants — kept here as a single source of truth so
    # ``mode="global"`` returns byte-identical pre-Stage-4 values from
    # ``_floors_for`` and ``_alpha_for_cls`` and the inline call sites pick
    # them up unchanged.
    _GLOBAL_TIGHT_FLOOR = 5.0
    _GLOBAL_FOCUSED_FLOOR = 2.5
    _GLOBAL_ABSTAIN_FLOOR = 2.5  # mirrors FOCUSED_SCORE_FLOOR_FOR_ABSTAIN

    def _floors_for(self, cls: Optional[str]):
        """Return ``AbstainClassFloors`` for this query's class.

        ``mode="global"`` returns the legacy hard-coded 5.0/2.5/2.5 floors
        regardless of ``cls`` (preserves Stage-3 behavior byte-for-byte).
        ``mode="per_classifier"`` consults ``config.abstain.per_class[cls]``
        with ``default`` fallback.

        Spec: docs/specs/2026-05-08-stage-4-threshold-calibration.md §6 + §7.
        """
        from .config import AbstainClassFloors
        ab = getattr(self.config, "abstain", None)
        if ab is None or ab.mode == "global":
            # Identity floors that match the legacy hard-coded constants.
            return AbstainClassFloors(
                abstain_top=self._GLOBAL_ABSTAIN_FLOOR,
                focused_top=self._GLOBAL_FOCUSED_FLOOR,
                tight_top=self._GLOBAL_TIGHT_FLOOR,
                foveated_alpha=self.config.budget.foveated_alpha,
            )
        return ab.floors_for(cls)

    def _alpha_for_cls(self, cls: Optional[str]) -> float:
        """Return the foveated splice power-law alpha for this query's class.

        ``mode="global"`` returns ``config.budget.foveated_alpha`` (legacy).
        ``mode="per_classifier"`` returns the per-class alpha with ``default``
        fallback.

        Spec §7.
        """
        ab = getattr(self.config, "abstain", None)
        if ab is None or ab.mode == "global":
            return self.config.budget.foveated_alpha
        return ab.floors_for(cls).foveated_alpha

    def build_context(
        self,
        query: str,
        downstream_model: Optional[str] = None,
        include_cold: Optional[bool] = None,
        session_context: Optional[Dict] = None,
        party_id: Optional[str] = None,
        prompt_tokens_hint: Optional[int] = None,
        session_id: Optional[str] = None,
        ignore_delivered: bool = False,
        read_only: bool = False,
        decoder_override: Optional[str] = None,
        caller_model_class: str = "generic",
        query_type: Optional[str] = None,
        max_genes: Optional[int] = None,
    ) -> ContextWindow:
        """
        Build the active context window for a query.
        Runs the 5-step retrieval pipeline (Steps 1-5).

        Args:
            downstream_model: optional model name from the proxy request,
                used for per-request MoE/small-model detection.
            include_cold: per-request override for cold-tier retrieval.
                ``None`` (default) honors the ``[context] cold_tier_enabled``
                config flag in cymatix.toml. ``True`` forces cold-tier on,
                ``False`` forces it off. Plumbed from the /context endpoint's
                ``include_cold`` body parameter.
            session_context: optional dict carrying the caller's working
                context — typically ``{"active_project": "cymatix-context",
                "active_files": ["cymatix_context/genome.py", ...]}``. The
                path-tokens of these are appended to the entity list so
                the path_key_index tier in ``query_genes`` can fire on
                compound (project, key) pairs even when the user's natural
                query doesn't restate the project name. This is the
                "implicit THIS project" signal that real users have but
                synthetic benches lack. None = no session context, which
                preserves the previous behaviour exactly.
            query_type: optional per-call retrieval-intent hint (semantic-wiring
                arm, PRD 2026-06-02). Forwarded to ``_retrieve`` → sharded
                ``query_docs`` so that ``"semantic"`` + CYMATIX_SEMANTIC_ARM=1
                broadens routing and scopes the dense weight. ``None`` (default)
                or any other value with the arm off is inert / byte-identical —
                production /context callers omit it; the bench injects the
                needle's ground-truth type for A/B. Mirrors the /fingerprint
                ``query_type`` thread.
            max_genes: optional per-call cap on assembled documents.
                ``None`` (default) honors the static ``[budget]
                max_genes_per_turn`` config. Plumbed from
                ``CymatixSession.query(k=...)`` / ``cymatix query --k``.
                Clamped to >= 1; downstream caps (budget-zone, classifier,
                caller_model_class) can still lower it further.
        """
        # Root span for the per-turn pipeline: every cymatix.pipeline.<stage>
        # span opened inside the impl nests under
        # cymatix.pipeline.build_context (the Tempo waterfall root promised
        # by docs/architecture/OBSERVABILITY.md). The impl split keeps the
        # span open across every return path (normal, empty-genome,
        # abstain). Span-only — stage durations are recorded exactly once,
        # by _stage_timer.
        with _pipeline_stage_span("build_context"):
            return self._build_context_impl(
                query,
                downstream_model=downstream_model,
                include_cold=include_cold,
                session_context=session_context,
                party_id=party_id,
                prompt_tokens_hint=prompt_tokens_hint,
                session_id=session_id,
                ignore_delivered=ignore_delivered,
                read_only=read_only,
                decoder_override=decoder_override,
                caller_model_class=caller_model_class,
                query_type=query_type,
                max_genes=max_genes,
            )

    def _build_context_impl(
        self,
        query: str,
        downstream_model: Optional[str] = None,
        include_cold: Optional[bool] = None,
        session_context: Optional[Dict] = None,
        party_id: Optional[str] = None,
        prompt_tokens_hint: Optional[int] = None,
        session_id: Optional[str] = None,
        ignore_delivered: bool = False,
        read_only: bool = False,
        decoder_override: Optional[str] = None,
        caller_model_class: str = "generic",
        query_type: Optional[str] = None,
        max_genes: Optional[int] = None,
    ) -> ContextWindow:
        """``build_context`` body — see ``build_context`` for the contract."""
        self._maybe_compact()

        # Pipeline viewer feed (launcher dashboard). Tagging this call with
        # a short request_id lets _stage_timer attribute every recorded
        # stage back to the originating /context invocation. The contextvar
        # is per-thread/per-task so concurrent calls cannot blend events.
        if _pipeline_ring_enabled():
            _pipeline_request_id.set(_uuid.uuid4().hex[:12])

        # Perf slice 1 (W0.3): commit-count snapshot; the tail_writes ring
        # entry carries the per-request delta (assemble's delivery log +
        # the post-retrieval tail writes; route-level CWoLa commits land
        # after this and are visible only in the process-cumulative count).
        _commit_c0 = getattr(self.genome, "commit_count", 0)

        # Per-call locals for foveated-splice state (spec §4-5). Local —
        # not instance state — so concurrent build_context calls cannot
        # race on writes/reads, and a prior call's state cannot leak in
        # if the dynamic-budget block is skipped this call (e.g. small
        # candidate set or all-zero scores). Same threading pattern as
        # decoder_prompt_override at line ~1949. See code review I1/I2.
        foveated_caps: Optional[List[float]] = None
        foveated_active: bool = False

        # Step 0a: Upstream query classifier / injection router.
        # Always runs (cheap, no I/O) — even when decoder_override is set,
        # so the audit trail records what the classifier *would* have picked.
        # See docs/specs/2026-04-29-query-classifier-injection-router-design.md.
        classifier_enabled = getattr(
            getattr(self.config, "classifier", None), "enabled", True,
        )
        classifier_result: Optional[ClassifierResult] = None
        with _pipeline_stage_span("classify"), _stage_timer("classify"):
            if classifier_enabled:
                classifier_result = classify_query(query)

        # Defensive defaults — referenced later by the classifier metadata
        # block; must be defined on every code path that reaches the bottom.
        override_applied = False
        candidate_pool_size = 0

        # Decoder selection: explicit caller override > classifier × class hint > default.
        # Resolved per-request without mutating shared instance state to prevent
        # races on the singleton manager under concurrent /context calls.
        #
        # Stage 5 (2026-05-08): the classifier-hint branch now consults the
        # 15-cell §6 lookup table via resolve_decoder_mode(cls, caller_model_class)
        # rather than the hard-coded ClassifierResult.decoder_mode literal. The
        # `generic` column of the table is byte-identical to the legacy literals
        # (spec §7 — verified by test_generic_branch_byte_identical_to_pre_stage5_output).
        from .retrieval.query_classifier import resolve_decoder_mode as _resolve_decoder_mode
        effective_decoder_mode_name: Optional[str] = None
        if decoder_override and decoder_override in DECODER_MODES:
            effective_decoder_prompt = DECODER_MODES[decoder_override]
            effective_decoder_mode_name = decoder_override
            override_applied = True
        elif classifier_result is not None:
            _resolved = _resolve_decoder_mode(
                classifier_result.cls, caller_model_class,
            )
            if _resolved is not None and _resolved in DECODER_MODES:
                effective_decoder_prompt = DECODER_MODES[_resolved]
                effective_decoder_mode_name = _resolved
            else:
                effective_decoder_prompt = self._decoder_prompt
            override_applied = False
        else:
            effective_decoder_prompt = self._decoder_prompt
            override_applied = False

        # Reset per-call cold-tier markers (set by _express when cold fires)
        self._last_cold_tier_used = False
        self._last_cold_tier_count = 0
        # Stage 7 (spec §6): reset per-build cold-peek state so a
        # previous query's refresh_targets cannot bleed into this one.
        self._last_cold_peek_targets = []

        if max_genes is None:
            max_genes = self.config.budget.max_genes_per_turn
        else:
            # Explicit caller cap (CymatixSession.query(k=...)) overrides the
            # static [budget] max_genes_per_turn for this call only.
            max_genes = max(1, int(max_genes))

        # ABSTAIN gate enable-state: config flag AND no env override.
        # Resolved per-call so CYMATIX_ABSTAIN_DISABLE flips without restart.
        abstain_enabled = (
            self.config.budget.abstain_enabled
            and not _env_truthy("CYMATIX_ABSTAIN_DISABLE")
        )

        # Budget-zone cap (spike) — clamp max_genes down when the caller's
        # incoming prompt already fills a large share of their window.
        # Returns None when the feature flag is off or the signal is
        # absent, so this is a no-op by default. See budget_zone.py.
        _zone_cap = _budget_zone_cap(prompt_tokens_hint)
        if _zone_cap is not None and _zone_cap < max_genes:
            log.debug(
                "Budget-zone cap: max_genes %d -> %d (prompt_tokens=%s)",
                max_genes, _zone_cap, prompt_tokens_hint,
            )
            max_genes = _zone_cap

        # Step 0b: Sub-query decomposition for broad/multi_hop queries.
        _use_decomposition = (
            classifier_result is not None
            and classifier_result.cls in ("multi_hop", "default")
            and getattr(
                getattr(self.config, "ribosome", None),
                "query_decomposition_enabled", False,
            )
        )
        _sub_queries: list = (
            self._decompose_query(query) if _use_decomposition else [query]
        )

        # Step 0: Query intent expansion (LLM-based, cached)
        # Restates the query with expanded keywords BEFORE tags lookup.
        # This sharpens the initial frequency so retrieval falls into the
        # right gravity well instead of optimizing the wrong one.
        with _pipeline_stage_span("extract"), _stage_timer("extract"):
            if len(_sub_queries) == 1:
                expanded_query, domains, entities = self._prepare_query_signals(
                    _sub_queries[0],
                    session_context=session_context,
                    expand_query=True,
                )
            else:
                # Multi-sub-query: prepare signals for the primary query as
                # the canonical expanded_query (used downstream for logging/health).
                expanded_query, domains, entities = self._prepare_query_signals(
                    query,
                    session_context=session_context,
                    expand_query=True,
                )

        # Issue #255 (classifier-gated combinator): resolve a per-query rerank
        # combinator override from the stage-0 classifier class. Empty map (the
        # default) or a disabled classifier (classifier_result is None) => None
        # => the store keeps its global rerank_combinator, so the default path
        # is byte-identical. Resolved once for the whole turn from the primary
        # query's class and threaded to every sub-query's retrieval below.
        from .retrieval.rerank_combinators import (
            resolve_class_combinator, resolve_class_flag,
        )
        _combinator_override = resolve_class_combinator(
            self.config.retrieval.rerank_combinator_by_class,
            classifier_result.cls if classifier_result is not None else None,
        )

        # Issue #341 (Task 6): resolve a per-query cross-encoder rerank
        # override from the same stage-0 classifier class. Empty map (the
        # default) or a disabled classifier (classifier_result is None) =>
        # None => the store keeps its global rerank_enabled, so the default
        # path is byte-identical. Resolved once for the whole turn and
        # threaded to every sub-query's retrieval below, same as the
        # combinator override above.
        _rerank_override = resolve_class_flag(
            self.config.retrieval.rerank_enabled_by_class,
            classifier_result.cls if classifier_result is not None else None,
        )

        _measurement = current_capture()
        if _measurement is not None and len(_sub_queries) != 1:
            _measurement.unsupported.append("parallel_subqueries: executor stages are not captured")

        # Step 2: Retrieve (knowledge store query + pending buffer + optional cold tier)
        #
        # 2026-07-18 bugbash (score/result atomicity): ``genome.
        # last_query_scores`` / ``last_tier_contributions`` are instance-
        # global and republished by EVERY retrieval, so a concurrent
        # request could swap query B's scores under query A's candidates
        # between stages. Snapshot both under the store lock immediately
        # after THIS request's retrieval; ``query_scores`` /
        # ``tier_contribs`` are then threaded through the refiners,
        # tiering, assembly, and health instead of re-reading the shared
        # attribute.
        with _pipeline_stage_span("express"), _stage_timer(
            "express",
            # Perf slice 1 (W0.2): attach the store's per-signal ms map and
            # the process-cumulative model-load ms to the express ring entry
            # so /debug/pipeline/recent exposes sub-stage detail over HTTP.
            extra=lambda: {
                "signals": dict(
                    getattr(self.genome, "last_signal_timings", None) or {}
                ),
                "model_load_ms": _model_load_ms_snapshot(),
            },
        ):
            if len(_sub_queries) == 1:
                candidates = self._retrieve(
                    domains, entities, max_genes,
                    query_text=_sub_queries[0], include_cold=include_cold,
                    party_id=party_id, read_only=read_only,
                    query_type=query_type,
                    rerank_combinator=_combinator_override,
                    rerank_override=_rerank_override,
                )
                with self.genome._last_query_scores_lock:
                    query_scores: Dict[str, float] = dict(
                        self.genome.last_query_scores or {}
                    )
                    tier_contribs: Dict = dict(
                        getattr(self.genome, "last_tier_contributions", None)
                        or {}
                    )
            else:
                import concurrent.futures

                def _run_sub(sq: str):
                    eq, d, e = self._prepare_query_signals(sq, session_context)
                    genes = self._retrieve(
                        d, e, max_genes,
                        query_text=sq, include_cold=include_cold,
                        party_id=party_id, read_only=read_only,
                        query_type=query_type,
                        rerank_combinator=_combinator_override,
                        rerank_override=_rerank_override,
                    )
                    with self.genome._last_query_scores_lock:
                        scores = dict(self.genome.last_query_scores or {})
                        contribs = dict(
                            getattr(
                                self.genome, "last_tier_contributions", None,
                            ) or {}
                        )
                    return genes, scores, contribs

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(_sub_queries)
                ) as pool:
                    triples = list(pool.map(_run_sub, _sub_queries))

                sub_results = [genes for genes, _, _ in triples]
                base_scores: dict = {}
                tier_contribs = {}
                for _, scores, contribs in triples:
                    base_scores.update(scores)
                    tier_contribs.update(contribs)

                candidates = _merge_subquery_candidates(sub_results, base_scores)
                candidates = candidates[: max_genes * 2]
                query_scores = base_scores

        # Stage 4 (2026-05-08): hoist classifier-derived `cls` so per-classifier
        # floor and alpha lookups work regardless of which downstream branch
        # fires. mode='global' (default) ignores this entirely; 'per_classifier'
        # consults config.abstain.per_class[cls] (with 'default' fallback).
        cls_for_floors: Optional[str] = (
            classifier_result.cls if classifier_result is not None else None
        )

        if not candidates:
            total_genes = self.genome.total_genes()
            # Stage 6 (§6): the legacy "denatured if knowledge store non-empty
            # else sparse" status maps onto two distinct MissBlock
            # reasons:
            #   - knowledge store has documents, none matched → "no_promoter_match"
            #   - knowledge store is empty                → "no_promoter_match"
            # Both ship the same expressed_context byte so the LLM-
            # visible prompt is identical; downstream discriminator
            # picks the MissBlock.reason from health.status +
            # genes_expressed.
            empty_status = "denatured" if total_genes > 0 else "sparse"
            no_match_reason = (
                "denatured" if empty_status == "denatured" else "no_promoter_match"
            )
            empty_health = ContextHealth(
                ellipticity=0.0,
                coverage=0.0,
                density=0.0,
                freshness=0.0,
                genes_available=total_genes,
                genes_expressed=0,
                status=empty_status,
            )
            _empty_win = ContextWindow(
                ribosome_prompt=effective_decoder_prompt,
                expressed_context=_no_match_token(no_match_reason),
                total_estimated_tokens=estimate_tokens(effective_decoder_prompt),
                compression_ratio=1.0,
                context_health=empty_health,
                metadata={
                    "query": query,
                    "genes_expressed": 0,
                    "no_match_reason": no_match_reason,
                },
            )
            # #350: request-scoped snapshot for the server-layer readers.
            _empty_win.retrieval_scores = query_scores
            _empty_win.tier_contributions = tier_contribs
            return _empty_win

        # stage name kept for ring continuity; the cross-encoder now runs pre-cap in the store
        with _pipeline_stage_span("rerank"), _stage_timer("rerank"):
            candidates, _ = self._apply_candidate_refiners(
                query,
                candidates,
                max_genes,
                use_cymatics=True,
                use_harmonic_bin=True,
                use_tcm=True,
                query_scores=query_scores,
            )

        if _measurement is not None:
            _measurement.record("post_blend_scores", query_scores)
            _measurement.record("post_blend_candidates", (g.gene_id for g in candidates))

        # Dynamic budget tiers — delegates to pipeline.tier_logic.
        # Issue #207 item 4: the tier/abstain constants thread from
        # [budget] tier_* + [abstain] ratio_threshold* config knobs;
        # defaults reproduce the prior hard-coded literals byte-for-byte.
        from .config import AbstainConfig
        from .pipeline.tier_logic import apply_budget_tiers as _apply_tiers
        _cls_floors = self._floors_for(cls_for_floors)
        _budget_cfg = self.config.budget
        _abstain_cfg = getattr(self.config, "abstain", None) or AbstainConfig()
        _tier = _apply_tiers(
            candidates,
            query_scores,
            _cls_floors,
            abstain_enabled=abstain_enabled,
            fusion_mode=getattr(self.genome, "_fusion_mode", "additive"),
            tight_ratio=_budget_cfg.tier_tight_ratio,
            focused_ratio=_budget_cfg.tier_focused_ratio,
            hard_floor_frac=_budget_cfg.tier_hard_floor_frac,
            lagrange_frac=_budget_cfg.tier_lagrange_frac,
            abstain_ratio_threshold=_abstain_cfg.ratio_threshold,
            abstain_ratio_threshold_rrf_norm=_abstain_cfg.ratio_threshold_rrf_norm,
        )
        if _tier.abstain:
            _abstain_win = self._build_abstain_window(
                query=query,
                effective_decoder_prompt=effective_decoder_prompt,
                top_score=_tier.abstain_top_score,
                ratio=_tier.abstain_ratio,
                reason="score_below_floor",
            )
            # #350: request-scoped snapshot for the server-layer readers.
            _abstain_win.retrieval_scores = query_scores
            _abstain_win.tier_contributions = tier_contribs
            return _abstain_win
        candidates = _tier.candidates
        budget_tier = _tier.budget_tier
        budget_tokens_est = _tier.budget_tokens_est
        self._last_shadow_pool = _tier.shadow_pool
        self._last_shadow_scores = _tier.shadow_scores

        # Step 3.6: Apply classifier assembly cap.
        # Invariant: classifier can only LOWER the assembled document count.
        # It cannot raise it, and it cannot reduce retrieval depth — the
        # score-ratio tier above already saw the full candidate set.
        #
        # Stage 5 (2026-05-08) §4: caller_model_class adjusts the effective cap
        # AFTER the classifier-derived cap.
        #   small_moe → min(classifier_cap, 4)   (small models drown in >4 documents)
        #   frontier  → max(12, classifier_cap*2) (frontier callers have 200k+ contexts)
        #   generic   → unchanged (regression baseline; classifier_cap as-is)
        candidate_pool_size = len(candidates)
        # Phase 0.5 (2026-08-12 refinement campaign): record the assembly
        # cap that actually truncated the candidate list (None when no cap
        # trim fired). Read below when the window's delivery-visibility
        # block decides cap_binding.
        _applied_assembly_cap: Optional[int] = None
        # C1 (issue #339 Phase 0.4), env-gated, default OFF: pool MEMBERSHIP at
        # the admission choke point. Captured HERE, before any assembly cap
        # trims ``candidates`` below, so the hash describes exactly the set that
        # ``pool_size`` counts. Nothing is added to the response unless the gate
        # is on; see the delivery-visibility writer below.
        _candidate_set_sha256: Optional[str] = None
        if _env_truthy(BENCH_RANK_HASH_ENV):
            _candidate_set_sha256 = _gene_set_sha256(g.gene_id for g in candidates)
        _classifier_cap = (
            classifier_result.assembly_max_genes_cap
            if classifier_result is not None and classifier_result.assembly_max_genes_cap is not None
            else None
        )
        # W2.4 (2026-08-28): the delivered-seat floor also lifts the
        # classifier's per-rule assembly cap — the w1c receipts pin the
        # cap (2/5/6/8), not the token budget, as the cat-(b) seat-loser
        # (splice_n_candidates == delivered_count on every such needle;
        # the cut happens HERE, before splice). Floor 0 = legacy cap
        # untouched. The model-class caps below (small_moe=4) are a model-
        # capability concern, deliberately NOT floored. Wave-1's 235/235
        # coupling attribution is what unlocked cap raises (phase plan
        # L1.2); this is the knob-gated form.
        _seat_floor_cap = int(getattr(self.config.budget, "min_delivered_docs", 0) or 0)
        if _seat_floor_cap > 0 and _classifier_cap is not None:
            _classifier_cap = max(_classifier_cap, _seat_floor_cap)
        if _classifier_cap is not None and len(candidates) > _classifier_cap:
            log.debug(
                "Classifier cap: assembled %d -> %d (class=%s)",
                len(candidates),
                _classifier_cap,
                classifier_result.cls if classifier_result is not None else "n/a",
            )
            candidates = candidates[:_classifier_cap]
            _applied_assembly_cap = _classifier_cap
        # Stage 5 §4: per-class assembly cap applied on top of (or instead of)
        # classifier cap. Generic skips this block entirely so the generic
        # branch stays byte-identical to pre-Stage-5.
        if caller_model_class == "small_moe":
            _moe_cap = min(_classifier_cap, 4) if _classifier_cap is not None else 4
            if len(candidates) > _moe_cap:
                candidates = candidates[:_moe_cap]
                _applied_assembly_cap = _moe_cap
        elif caller_model_class == "frontier":
            _frontier_cap = (
                max(12, _classifier_cap * 2) if _classifier_cap is not None else 12
            )
            # Frontier RAISES the cap relative to classifier_cap, so this is a
            # widening — only meaningful if the candidate pool was bigger than
            # the classifier cap. The classifier cap was already applied above
            # (potentially truncating); re-running with a wider cap would not
            # restore lost candidates. So we only honor the wider cap when the
            # classifier cap was None (no truncation happened upstream).
            if _classifier_cap is None and len(candidates) > _frontier_cap:
                candidates = candidates[:_frontier_cap]
                _applied_assembly_cap = _frontier_cap

        # Foveated-splice (spec §4-5): for BROAD only, replace uniform per-document
        # compression with a rank-scaled power-law schedule AND reverse the
        # assembly order so top-rank lands nearest the user query (lost-in-the-
        # middle exploit). Off by default; see docs/specs/2026-05-03-foveated-
        # splice-design.md §6.3.
        #
        # Placed AFTER the classifier cap so reversal operates on the final
        # post-cap candidate list (otherwise classifier truncation would
        # silently drop the top-rank document under reverse-rank — see code
        # review C1).
        #
        # Uses local variables (not self._last_*) so concurrent build_context
        # calls don't race and stale state from a prior call can't leak into
        # the current one (see code review I1/I2; same pattern as
        # decoder_prompt_override threading).
        #
        # Stage 5 (2026-05-08) §8: caller_model_class gates foveated.
        #   frontier  → SKIP entirely (forward rank-1-first order; long-context
        #               attention regresses under reverse-rank).
        #   small_moe → ON regardless of budget_tier (always benefits from
        #               recency on the document that holds the answer).
        #   generic   → unchanged (broad-tier-only, regression baseline).
        if caller_model_class == "frontier":
            _foveated_should_run = False
        elif caller_model_class == "small_moe":
            _foveated_should_run = (
                self.config.budget.foveated_enabled and len(candidates) > 1
            )
        else:
            _foveated_should_run = (
                budget_tier == "broad"
                and self.config.budget.foveated_enabled
                and len(candidates) > 1
            )
        if _foveated_should_run:
            # Stage 4 (2026-05-08): per-classifier foveated alpha. mode='global'
            # returns config.budget.foveated_alpha (legacy); 'per_classifier'
            # returns the per-class value with 'default' fallback.
            _alpha_for_caps = self._alpha_for_cls(cls_for_floors)
            caps = _compute_foveated_caps(
                n=len(candidates),
                alpha=_alpha_for_caps,
                c_min=self.config.budget.foveated_c_min,
                c_max=1.0,
            )
            # Reverse together so caps[i] still pairs with candidates[i].
            candidates = list(reversed(candidates))
            foveated_caps = list(reversed(caps))
            foveated_active = True

        # Step 3.5: NLI classification (optional, DeBERTa backend only)
        relation_graph = {}
        if hasattr(self.ribosome, 'classify_relations'):
            try:
                relation_graph = self.ribosome.classify_relations(candidates)
            except Exception:
                log.warning("NLI classification failed, proceeding without", exc_info=True)

        # Step 4: Dense document retrieval
        # Each document retrieved as: Facts (KV pairs) + Source + Raw content
        # Dense format minimizes prose for small model extraction.
        spliced_map = {}
        answer_slate_lines = []  # MoE answer slate — flat KV pairs

        # Stage 5 (2026-05-08) §5: small_moe slate is best-first by per-KV score
        # (Gemini's `_best_first` sort preserved as the source order). Generic
        # branch keeps the original "collect during candidate loop" approach
        # for byte-identity with pre-Stage-5 output.
        if caller_model_class == "small_moe":
            _slate_scores = query_scores or {}
            _best_first = sorted(
                candidates,
                key=lambda _g: _slate_scores.get(_g.gene_id, 0),
                reverse=True,
            )
            for _g in _best_first:
                if _g.key_values:
                    for _kv in _g.key_values[:5]:
                        answer_slate_lines.append(_kv)

        # foveated_caps / foveated_active are call-local; see top of build_context.
        foveated_base = self.config.budget.foveated_base_chars
        # Splice-floor fix (J-space council kill-switch #1): the uniform
        # per-document target is budget-proportional by default instead of
        # the query-agnostic 1000-char literal, and the truncation keeps
        # query-term lines (see encoding/headroom_bridge._query_aware_trim).
        # domains + entities are the Stage-1 keyword extraction for this
        # query (stopword-filtered, morphology-expanded) — the same signals
        # assembly receives as query_signals.
        _splice_target = _compute_splice_target(
            getattr(self.config.budget, "splice_target_chars", 0),
            self.config.budget.expression_tokens,
            len(candidates),
        )
        _splice_terms = list(dict.fromkeys([*domains, *entities]))
        # Splice stage (covers the compress_text loop over all documents):
        # _stage_timer records the cymatix_pipeline_stage_seconds point
        # (replacing the former manual _splice_t0 record — exactly one
        # duration per stage), the span feeds the Tempo waterfall.
        with _pipeline_stage_span("splice"), _stage_timer("splice", extra=lambda: {
                "n_candidates": len(candidates), "splice_target": _splice_target}):
            for idx, g in enumerate(candidates):
                src = g.source_id or ""
                short = _shorten_source_path(
                    src, self.config.ingestion.citation_path_anchors)
                # Dense XML document format — structured for small model extraction
                kv_attrs = ""
                if g.key_values:
                    # Top 5 KVs as XML attributes for instant scanning
                    kv_pairs = " ".join(g.key_values[:5])
                    kv_attrs = f' facts="{kv_pairs}"'
                    # Collect KVs for MoE answer slate (generic branch only —
                    # small_moe pre-pass above already populated the slate in
                    # best-first order per spec §5).
                    if caller_model_class != "small_moe":
                        for kv in g.key_values[:5]:
                            answer_slate_lines.append(kv)
                src_attr = f' src="{short}"' if short else ""
                # Semantic compression via Headroom (by Tejas Chopra, Apache-2.0).
                # Dispatches by tags domain: log→LogCompressor,
                # diff→DiffCompressor, else→Kompress (ModernBERT).
                # CodeCompressor disabled (40% invalid syntax — see 2f518dc).
                # Falls back to a query-aware trim when headroom is unavailable
                # (the shipped path — headroom_ai is not installed).
                #
                # Foveated path overrides the uniform target with a
                # rank-proportional cap per document. When foveated_caps is None
                # (default / non-BROAD / disabled), the uniform budget-
                # proportional target applies (splice-floor fix; was a
                # query-agnostic 1000-char literal).
                if foveated_caps is not None:
                    target = max(1, int(foveated_caps[idx] * foveated_base))
                else:
                    target = _splice_target
                content = compress_text(
                    g.content,
                    target_chars=target,
                    content_type=g.promoter.domains,
                    query_terms=_splice_terms,
                )
                spliced_map[g.gene_id] = f"<GENE{src_attr}{kv_attrs}>\n{content}\n</GENE>"

        # Step 5: Assemble (MoE/small-model aware)
        # Stage 5 §4: caller_model_class refines slate emission (small_moe
        # always-on, frontier always-off, generic preserves legacy).
        use_slate = self._should_use_slate(downstream_model, caller_model_class)
        with _pipeline_stage_span("assemble"), _stage_timer("assemble"):
            window = self._assemble(
                query, candidates, spliced_map, relation_graph,
                query_signals=(domains, entities),
                answer_slate=answer_slate_lines if use_slate else None,
                session_id=session_id,
                ignore_delivered=ignore_delivered,
                decoder_prompt_override=effective_decoder_prompt,
                respect_caller_order=foveated_active,
                caller_model_class=caller_model_class,
                query_scores=query_scores,
                tier_contributions=tier_contribs,
            )

        # Annotate window with dynamic budget tier (for telemetry/benchmarks)
        if window.metadata is not None:
            window.metadata["budget_tier"] = budget_tier
            window.metadata["budget_tokens_est"] = budget_tokens_est
            if foveated_active and foveated_caps is not None:
                # Spec §8: per-call provenance for post-hoc α-curve attribution.
                # Absent when foveated_enabled=false or tier != broad.
                window.metadata["foveated_caps"] = foveated_caps
                window.metadata["foveated_alpha"] = self.config.budget.foveated_alpha
                # Scalar reductions for Prometheus translation paths that
                # flatten only scalar attributes (the list above can be
                # silently dropped). See code review I3. Under reverse-rank
                # ordering caps[-1] is the c_max applied to the TOP-rank
                # document (closest to the user query) and caps[0] is the
                # c_min applied to the BOTTOM-rank document — the names
                # `_top` / `_min` reflect the rank semantic, not the
                # list-index semantic.
                window.metadata["foveated_cap_top"] = foveated_caps[-1]
                window.metadata["foveated_cap_min"] = foveated_caps[0]
                window.metadata["foveated_cap_n"] = len(foveated_caps)

        # Classifier observability payload (spec section 5.2).
        if classifier_result is not None and window.metadata is not None:
            window.metadata["classifier"] = {
                "class": classifier_result.cls,
                "signals_matched": list(classifier_result.signals_matched),
                "signal_count": classifier_result.signal_count,
                "threshold_required": classifier_result.threshold_required,
                "assembly_max_genes_cap": classifier_result.assembly_max_genes_cap,
                "max_genes_effective": len(candidates),
                "decoder_selected": classifier_result.decoder_mode,
                "override_applied": override_applied,
                "candidate_pool_size": candidate_pool_size,
            }
            if classifier_result.reason:
                window.metadata["classifier"]["reason"] = classifier_result.reason

        # Phase 0.1/0.5 (2026-08-12 refinement campaign): delivery-
        # visibility block. Records the scored candidate pool, the
        # delivered ids in final order (the ablation ladder's
        # delivered-gold basis), and which constraint actually truncated
        # delivery:
        #   * budget eviction trimmed below the cap  -> "token_budget"
        #   * delivered count equals the applied assembly cap and the
        #     pool had more eligible candidates      -> "assembly_cap"
        #   * otherwise (incl. sub-cap limiters like splice drops or
        #     session elision, where neither constraint was binding)
        #                                            -> "none"
        # Served on /context as a top-level ``delivery`` key.
        if window.metadata is not None:
            _delivered_ids = list(window.expressed_gene_ids or [])
            _n_budget_evicted = int(window.metadata.get("budget_evicted", 0) or 0)
            if _n_budget_evicted > 0:
                _cap_binding = "token_budget"
            elif (
                _applied_assembly_cap is not None
                and len(_delivered_ids) == _applied_assembly_cap
                and candidate_pool_size > _applied_assembly_cap
            ):
                _cap_binding = "assembly_cap"
            else:
                _cap_binding = "none"
            window.metadata["delivery"] = {
                "pool_size": int(candidate_pool_size),
                "delivered_count": len(_delivered_ids),
                "delivered_gene_ids": _delivered_ids,
                "cap_binding": _cap_binding,
            }
            if _candidate_set_sha256 is not None:
                window.metadata["delivery"]["candidate_set_sha256"] = _candidate_set_sha256

        # Touch retrieved documents (update signals).
        # Read-only contract (Stage 1): clean=true ⇒ read_only=True ⇒ skip
        # all knowledge store mutations below. Learning/replication is suppressed so
        # synthetic benches and audit-style queries cannot pollute knowledge store
        # state. log_health (further down) is intentionally OUTSIDE the
        # gate — it writes only to `health_log` (observability, not
        # learning) and is the only way to see what a read-only run did.
        expressed_ids = [g.gene_id for g in candidates]
        # Perf slice 1 (W0.2): the post-retrieval writes (touch/link/
        # harmonic/relations/log_health) were the largest untimed region on
        # the response path. One ring stage covers them; its "commits"
        # payload is the whole request's writer-connection commit delta.
        with _stage_timer(
            "tail_writes",
            extra=lambda: {
                "commits": getattr(self.genome, "commit_count", 0) - _commit_c0,
            },
        ):
            if not read_only:
                self.genome.touch_genes(expressed_ids)
                self.genome.link_coactivated(expressed_ids)

                # Compute harmonic weights between retrieved documents (cymatics)
                if self._use_cymatics and self.config.cymatics.harmonic_links:
                    try:
                        from .scoring.cymatics import compute_harmonic_weights
                        weights = compute_harmonic_weights(
                            candidates, peak_width=self._cymatics_peak_width,
                        )
                        if weights:
                            self.genome.store_harmonic_weights(weights)
                    except Exception:
                        # Harmonic links are diagnostic, not critical — non-blocking,
                        # but log so failures don't disappear silently.
                        log.warning("Harmonic link persistence failed", exc_info=True)

                # Store typed relations in knowledge store (if available)
                if relation_graph:
                    batch = []
                    for (gid_a, gid_b), (relation, confidence) in relation_graph.items():
                        if confidence >= 0.6:
                            batch.append((gid_a, gid_b, int(relation), confidence))
                    if batch:
                        self.genome.store_relations_batch(batch)

            # Update TCM session context with retrieved documents (in-memory only,
            # not gated — TCM session is per-process state, not knowledge store state).
            if self._tcm_session is not None:
                try:
                    for doc in candidates:
                        self._tcm_session.update_from_gene(doc)
                except Exception:
                    pass  # TCM is diagnostic, not critical

            # Log health signal for historical tracking
            health = window.context_health
            self.genome.log_health(
                query=query,
                ellipticity=health.ellipticity,
                coverage=health.coverage,
                density=health.density,
                freshness=health.freshness,
                genes_expressed=health.genes_expressed,
                genes_available=health.genes_available,
                status=health.status,
            )

        # Perf slice 1 (W0.2): expose the request id so route-level work
        # (CWoLa log/sweep) can ring stages correlated to this request —
        # build_context runs on an executor thread, so the route's own
        # contextvar never sees the id set at the top of this function.
        try:
            window.metadata["pipeline_request_id"] = _pipeline_request_id.get()
        except Exception:
            pass

        # Issue #350 (W1.6b): attach the request-scoped snapshot captured
        # under the lock right after retrieval, so the server layer's
        # know/miss, citation-score, and tier-total readers report THIS
        # request's confidence — the genome-global last_query_scores may
        # already hold a concurrent request's publication by the time the
        # route reads it.
        window.retrieval_scores = query_scores
        window.tier_contributions = tier_contribs

        return window

    async def build_context_async(
        self,
        query: str,
        downstream_model: Optional[str] = None,
        include_cold: Optional[bool] = None,
        session_context: Optional[Dict] = None,
        party_id: Optional[str] = None,
        prompt_tokens_hint: Optional[int] = None,
        session_id: Optional[str] = None,
        ignore_delivered: bool = False,
        read_only: bool = False,
        decoder_override: Optional[str] = None,
        caller_model_class: str = "generic",
        query_type: Optional[str] = None,
        max_genes: Optional[int] = None,
    ) -> ContextWindow:
        """Async wrapper -- runs the sync pipeline in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _get_executor(),
            self.build_context,
            query,
            downstream_model,
            include_cold,
            session_context,
            party_id,
            prompt_tokens_hint,
            session_id,
            ignore_delivered,
            read_only,
            decoder_override,
            caller_model_class,
            query_type,
            max_genes,
        )

    def reset_session_state(self) -> None:
        """Clear per-session caches and TCM drift between unrelated queries.

        Intended for synthetic benches (N=1000+) where every needle is
        independent of the previous one — letting TCM drift accumulate
        across unrelated queries pollutes the temporal context signal,
        and lets the intent-expansion LRU grow without bound.

        Resets:
          - _intent_cache (LRU of LLM-expanded queries)
          - _tcm_session (re-initializes the temporal context to zero)
          - genome.last_query_scores (per-call but better cleared)
          - _last_shadow_pool / _last_shadow_scores (Lagrange leftovers)

        Does NOT touch:
          - knowledge store content (documents, embeddings, attribution)
          - LRU-cached parse results (those are content-keyed)
          - lifecycle tier tier state (per-document)

        Safe to call between every /context request when running
        in synthetic-bench mode. Typical real-user sessions should NOT
        call this — the TCM drift IS the value-add for related queries.
        """
        try:
            if hasattr(self, "_intent_cache"):
                self._intent_cache.clear()
        except Exception:
            pass
        try:
            if self._tcm_session is not None:
                from .scoring.tcm import SessionContext
                self._tcm_session = SessionContext(n_dims=20, beta=0.5)
        except Exception:
            pass
        try:
            self.genome.last_query_scores = {}
        except Exception:
            pass
        try:
            self._last_shadow_pool = []
            self._last_shadow_scores = {}
        except Exception:
            pass

    # -- Learn: persist exchange back to knowledge store (Step 6) -------------

    def learn(self, query: str, response: str, timeout_s: float = 15.0) -> Optional[str]:
        """
        Buffer a query+response exchange for later consolidation.

        Appends to the session buffer (last 10 exchanges) and triggers
        auto-consolidation every N learns. The exchange is also immediately
        persisted to the knowledge store for pending-buffer retrieval continuity.

        The compressor persist call is wrapped in a thread timeout so a
        slow/overloaded backend can never hang the background task forever.
        On timeout, a minimal document is synthesized from the raw exchange
        (same fallback path used by ``Ribosome.replicate`` on error).

        Returns gene_id or None on failure.
        """
        # Persist stage (Step 6). learn() runs as a background task after
        # the response ships, so its cymatix.pipeline.persist span is NOT a
        # child of the cymatix.pipeline.build_context root span (the root
        # closed with the request); _stage_timer still feeds the persist
        # bucket of cymatix_pipeline_stage_seconds.
        #
        # In the embedded sync flow learn() runs on the caller's thread,
        # where _pipeline_request_id still holds the previous
        # build_context's id — clear it so persist never rings (the
        # launcher pipeline panel charts in-request stages only).
        _pipeline_request_id.set("")
        with _pipeline_stage_span("persist"), _stage_timer("persist"):
            return self._learn_impl(query, response, timeout_s)

    def _learn_impl(
        self, query: str, response: str, timeout_s: float,
    ) -> Optional[str]:
        """``learn`` body — see ``learn`` for the contract."""
        # Buffer the exchange for consolidation
        with self._session_buffer_lock:
            self._session_buffer.append((query, response))
            # Keep only last 10 exchanges
            if len(self._session_buffer) > 10:
                self._session_buffer = self._session_buffer[-10:]
            self._session_learn_count += 1

        try:
            # Wrap persist in a thread timeout so a stuck compressor
            # can't block this background task indefinitely.
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                future = _ex.submit(self.ribosome.replicate, query, response)
                try:
                    gene = future.result(timeout=timeout_s)
                except _cf.TimeoutError:
                    log.warning(
                        "Ribosome replicate timed out after %.1fs — "
                        "building minimal gene from raw exchange",
                        timeout_s,
                    )
                    # Build minimal document without the compressor (same shape
                    # as Ribosome.replicate's own fallback path)
                    from .genome import Genome as _Genome
                    from .schemas import Gene as _Gene, PromoterTags as _PT, EpigeneticMarkers as _EM
                    exchange = f"User query: {query}\n\nAssistant response: {response}"
                    gene = _Gene(
                        gene_id=_Genome.make_gene_id(exchange),
                        content=exchange,
                        complement=f"Q: {query[:200]} A: {response[:300]}",
                        codons=["exchange"],
                        promoter=_PT(summary=query[:100]),
                        epigenetics=_EM(),
                    )

            # Attach ΣĒMA vector to persisted document
            if self._sema_codec is not None:
                try:
                    gene.embedding = self._sema_codec.encode(gene.content[:1000])
                except Exception:
                    pass

            # Add to pending buffer immediately (before SQLite commit)
            with self._pending_lock:
                self._pending.append(gene)

            gid = self.genome.upsert_doc(gene)

            # Remove from pending now that it's committed
            with self._pending_lock:
                self._pending = [g for g in self._pending if g.gene_id != gid]

            log.info("Replicated exchange into gene %s", gid)

            # Auto-consolidation trigger
            if self._session_learn_count >= self._consolidation_threshold:
                try:
                    self.consolidate_session()
                except Exception:
                    log.warning("Auto-consolidation failed (non-fatal)", exc_info=True)

            return gid

        except Exception:
            log.warning("Replication failed (non-fatal)", exc_info=True)
            return None

    async def learn_async(self, query: str, response: str) -> Optional[str]:
        """Async wrapper for learn."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_get_executor(), self.learn, query, response)

    # -- Session consolidation (Synaptic Plasticity) ---------------------

    def consolidate_session(self) -> List[str]:
        """
        Distill the session buffer into consolidated knowledge documents.

        Sends buffered exchanges to the compressor with a "distill facts" prompt.
        Extracts only new knowledge -- facts, config changes, decisions, discoveries.
        Skips greetings, acknowledgments, and trivial exchanges.

        Returns list of gene_ids created from distilled facts.
        """
        with self._session_buffer_lock:
            if not self._session_buffer:
                log.info("Session buffer empty, nothing to consolidate")
                return []
            # Snapshot and clear
            exchanges = list(self._session_buffer)
            self._session_buffer.clear()
            self._session_learn_count = 0

        # Format exchanges for the distillation prompt
        formatted = []
        for i, (q, r) in enumerate(exchanges, 1):
            formatted.append(f"[Exchange {i}]\nUser: {q[:500]}\nAssistant: {r[:800]}")
        conversation_text = "\n\n".join(formatted)

        distill_prompt = (
            "Extract ONLY new facts, decisions, or discoveries from this conversation.\n"
            "Skip greetings, acknowledgments, thinking-out-loud, and trivial exchanges.\n"
            "Output as a JSON list of short fact strings. If nothing is worth keeping, "
            "return an empty list [].\n\n"
            f"Conversation ({len(exchanges)} exchanges):\n\n{conversation_text}"
        )

        distill_system = (
            "You are a knowledge distillation engine. You receive conversation exchanges "
            "and extract ONLY load-bearing facts. Respond with a JSON list of strings. "
            "Each string should be a single, self-contained fact. No markdown fences."
        )

        try:
            raw = self.ribosome.backend.complete(
                distill_prompt, system=distill_system, temperature=0.0
            )
            from .ribosome import _parse_json
            facts = _parse_json(raw)
        except Exception:
            log.warning("Session consolidation distillation failed", exc_info=True)
            return []

        if not isinstance(facts, list):
            log.warning("Consolidation returned non-list: %s", type(facts))
            return []

        # Filter to strings only
        facts = [f for f in facts if isinstance(f, str) and len(f.strip()) > 5]

        if not facts:
            log.info("No facts extracted from session buffer (%d exchanges)", len(exchanges))
            return []

        gene_ids = []
        for fact in facts:
            try:
                gene = self.ribosome.encode(fact, content_type="text")
                gene.source_id = "__session__"
                # Add session_memory and chat_context to domains
                existing_domains = set(gene.promoter.domains)
                existing_domains.update(["session_memory", "chat_context"])
                gene.promoter.domains = list(existing_domains)

                # Attach ΣĒMA vector if available
                if self._sema_codec is not None:
                    try:
                        gene.embedding = self._sema_codec.encode(gene.content[:1000])
                    except Exception:
                        pass

                gid = self.genome.upsert_doc(gene)
                gene_ids.append(gid)
            except Exception:
                log.warning("Failed to create gene from fact: %s", fact[:100], exc_info=True)

        log.info(
            "Session consolidation: %d facts extracted from %d exchanges -> %d genes",
            len(facts), len(exchanges), len(gene_ids),
        )
        return gene_ids

    async def consolidate_session_async(self) -> List[str]:
        """Async wrapper for consolidate_session."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_get_executor(), self.consolidate_session)

    # -- Stats ---------------------------------------------------------

    def stats(self) -> Dict:
        self.genome.refresh()  # See latest document count from external writers
        genome_stats = self.genome.stats()
        health_summary = self.genome.health_summary()
        return {
            **genome_stats,
            "pending_replications": len(self._pending),
            "session_buffer_size": len(self._session_buffer),
            "session_learn_count": self._session_learn_count,
            "health": health_summary,
            "config": {
                "ribosome_budget": self.config.budget.ribosome_tokens,
                "expression_budget": self.config.budget.expression_tokens,
                "max_genes_per_turn": self.config.budget.max_genes_per_turn,
                "splice_aggressiveness": self.config.budget.splice_aggressiveness,
                "decoder_mode": self._decoder_mode,
                "decoder_tokens": len(self._decoder_prompt) // 4,
                # Budget-zone spike: report server-side flag state so
                # benches can verify the env var actually reached this
                # process before running a sweep. Pure boolean, no cost.
                "budget_zone_enabled": _budget_zone_is_enabled(),
            },
        }

    # -- Internal: Step 1 (extract) ------------------------------------

    def _extract_query_signals(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Lightweight keyword extraction from the query for tags matching.
        No model call -- uses pre-built frozenset from accel module.
        """
        return extract_query_signals(query)

    def _expand_query_intent(self, query: str) -> str:
        """
        Step 0: Sharpen the initial query frequency via LLM expansion.

        A small compressor call (~100 tokens out) restates the query with
        expanded keywords BEFORE tags lookup. This changes which 12
        documents get pulled in the first place — upstream of every bracket
        cut in the pipeline.

        The Thinker metaphor: don't optimize the judge; fix the signal.
        Falls back to the raw query on any failure (LRU-cached per query).
        """
        # LRU-cached expansion (query text → expanded text)
        if not hasattr(self, "_intent_cache"):
            self._intent_cache: Dict[str, str] = {}
        if query in self._intent_cache:
            return self._intent_cache[query]

        # Flag gate — strict LLM-free pipeline sets this false.
        # Upstream path (ingest → 12-tone retrieval) has no LLM calls;
        # this Step 0 call is the last residual one and is disabled by
        # setting ribosome.query_expansion_enabled = false.
        if not getattr(self.config.ribosome, "query_expansion_enabled", True):
            self._intent_cache[query] = query
            return query

        # Only expand when we have a real LLM backend (skip for paused/Ollama-warmup)
        if not hasattr(self.ribosome, "backend"):
            self._intent_cache[query] = query
            return query
        if getattr(self.ribosome.backend, "is_disabled_backend", False):
            self._intent_cache[query] = query
            return query

        system = (
            "You are a query intent expander. Given a user's question, "
            "output a single line of SPACE-SEPARATED KEYWORDS that capture "
            "the question's intent plus likely synonyms and domain terms. "
            "Include the original key words. No prose, no punctuation, just "
            "lowercase keywords separated by spaces. Maximum 15 words."
        )
        prompt = f"Query: {query}\n\nKeywords:"

        try:
            raw = self.ribosome.backend.complete(prompt, system=system, temperature=0.0)
            # Clean: lowercase, strip punctuation, keep only word-like tokens
            import re as _re
            expanded = " ".join(
                _re.findall(r"[a-z0-9_]+", raw.lower())
            )[:500]  # hard cap
            if not expanded:
                expanded = query
            else:
                # Always append original query so the intent isn't drift-away
                expanded = f"{query} {expanded}"
        except Exception:
            log.debug("Query intent expansion failed, using raw query", exc_info=True)
            expanded = query

        # Cache + bound (prevent unbounded growth)
        if len(self._intent_cache) > 256:
            self._intent_cache.clear()
        self._intent_cache[query] = expanded
        return expanded

    def _decompose_query(self, query: str) -> list:
        """Decompose a broad query into 2-4 point-fact sub-queries via one LLM call.

        Returns [query] unchanged when disabled, no backend available, or on any failure.
        LRU-cached at 256 entries.
        """
        if not hasattr(self, "_decompose_cache"):
            self._decompose_cache: dict = {}
        if query in self._decompose_cache:
            return self._decompose_cache[query]

        if not getattr(getattr(self.config, "ribosome", None), "query_decomposition_enabled", False):
            # Try LLM-free template routing based on query intent heuristic
            try:
                from .retrieval.intent_router import sub_queries_for
                from .tagger import CpuTagger
                from .schemas import IntentClass
                _tagger = CpuTagger.__new__(CpuTagger)
                guessed_class = _tagger._classify_intent(query)
                if guessed_class != IntentClass.UNKNOWN:
                    sub_qs = sub_queries_for(query, guessed_class)
                    if len(sub_qs) > 1:
                        if len(self._decompose_cache) > 256:
                            self._decompose_cache.clear()
                        self._decompose_cache[query] = sub_qs
                        return sub_qs
            except Exception:
                pass
            self._decompose_cache[query] = [query]
            return [query]

        if not hasattr(self.ribosome, "backend") or getattr(
            self.ribosome.backend, "is_disabled_backend", False
        ):
            self._decompose_cache[query] = [query]
            return [query]

        system = (
            "You are a retrieval query decomposer. Given a broad question, output "
            "2 to 4 SHORT, SPECIFIC sub-questions that together answer it. Each "
            "sub-question must be answerable from a single fact or rule. "
            "Format: one sub-question per line, numbered. No prose, no headings."
        )
        prompt = f"Broad question: {query}\n\nSub-questions:"

        try:
            import re as _re
            raw = self.ribosome.backend.complete(prompt, system=system, temperature=0.0)
            sub_qs = [
                _re.sub(r"^\d+\.\s*", "", line).strip()
                for line in raw.strip().splitlines()
                if _re.match(r"^\d+\.", line.strip()) and len(line.strip()) > 10
            ]
            if not 2 <= len(sub_qs) <= 4:
                sub_qs = [query]
        except Exception:
            log.debug("Query decomposition failed, using raw query", exc_info=True)
            sub_qs = [query]

        if len(self._decompose_cache) > 256:
            self._decompose_cache.clear()
        self._decompose_cache[query] = sub_qs
        return sub_qs

    def _prepare_query_signals(
        self,
        query: str,
        session_context: Optional[dict] = None,
        *,
        expand_query: bool = True,
    ) -> Tuple[str, List[str], List[str]]:
        """Return the query text and derived retrieval signals for a request."""
        expanded_query = self._expand_query_intent(query) if expand_query else query
        domains, entities = self._extract_query_signals(expanded_query)

        if session_context:
            try:
                from .genome import path_tokens
                implicit = set()
                ap = session_context.get("active_project")
                if ap:
                    implicit |= path_tokens(str(ap))
                for f in session_context.get("active_files", []) or []:
                    implicit |= path_tokens(str(f))
                for p in session_context.get("active_projects", []) or []:
                    implicit |= path_tokens(str(p))
                if implicit:
                    existing = {e.lower() for e in entities}
                    entities = entities + [t for t in implicit if t not in existing]
            except Exception:
                log.debug("session_context plumb failed", exc_info=True)

        return expanded_query, domains, entities

    # -- Internal: Step 2 (retrieve) ------------------------------------

    def _retrieve(
        self,
        domains: List[str],
        entities: List[str],
        max_genes: int,
        query_text: Optional[str] = None,
        include_cold: Optional[bool] = None,
        party_id: Optional[str] = None,
        use_harmonic: bool = True,
        use_sr: Optional[bool] = None,
        read_only: bool = False,
        query_type: Optional[str] = None,
        rerank_combinator: Optional[str] = None,
        rerank_override: Optional[bool] = None,
    ) -> List[Gene]:
        """Query knowledge store + pending buffer for matching documents.

        ``rerank_combinator`` (issue #255, classifier-gated combinator): the
        per-query combinator override resolved from the stage-0 query
        classifier's class. ``None`` (classifier disabled, empty map, or a class
        not in the map) leaves the store on its global ``rerank_combinator`` —
        the byte-identical default. Forwarded to BOTH the dense-ANN branch (the
        dense-on default path) and the plain ``query_docs`` branch; on the
        sharded ``query_docs`` path it rides through to each shard's own
        combinator via the router's verbatim ``**kwargs`` fan-out.

        ``rerank_override`` (issue #341, Task 6): the per-query cross-encoder
        rerank on/off override resolved from the same stage-0 classifier
        class via ``resolve_class_flag`` / ``[retrieval] rerank_enabled_by_class``.
        ``None`` leaves the store on its global ``rerank_enabled`` — the
        byte-identical default. Forwarded to the dense-ANN branch as
        ``query_docs_ann(..., rerank_override=...)`` and to the plain
        ``query_docs`` branch as ``query_docs(..., query_text=query_text,
        rerank_override=...)`` — ``query_text`` must accompany it there since
        Task 5's lex-path rerank seam only fires when ``query_text`` is
        supplied.

        ``query_type`` (semantic-wiring arm, PRD 2026-06-02) is forwarded to the
        sharded ``query_docs`` path so the router can broaden routing and the
        per-shard merge can scope the dense weight when it is "semantic" and
        CYMATIX_SEMANTIC_ARM=1. Defaults to None (arm inert). The dense-ANN solo
        branch is intentionally NOT threaded — the arm is sharded-only.

        Parameters
        ----------
        domains, entities, max_genes:
            Standard hot-tier retrieval inputs.
        query_text : str, optional
            Original natural-language query. Required if cold-tier
            fallthrough fires (used by ``Genome.query_cold_tier`` to
            encode the SEMA query vector). Defaults to None — when None,
            cold-tier is skipped even if config-enabled.
        include_cold : bool, optional
            Per-call override of the ``[context] cold_tier_enabled``
            flag in cymatix.toml. ``None`` (default) uses the config flag,
            ``True`` forces cold-tier on, ``False`` forces it off.
            Plumbed from the /context endpoint's ``include_cold`` body
            parameter so callers can opt in/out per request without
            touching the config file.
        party_id : str, optional
            Caller's party identity. When provided, ``query_genes``
            excludes documents attributed to OTHER parties (cross-party
            leakage prevention) and grants a +0.5 score bonus to documents
            attributed to this party. Unattributed legacy documents remain
            retrievable regardless. ``None`` = no filtering, no bonus
            (existing behavior).
        """
        candidates: List[Gene] = []

        # ── Hot-tier retrieval (lifecycle tier < HETEROCHROMATIN) ────────────
        try:
            if self.genome._dense_embedding_enabled and query_text:
                # Step 4: ANN threshold path — uses BGE-M3 dense vectors to
                # dynamically gate the candidate count by similarity threshold.
                candidates = self.genome.query_docs_ann(
                    query=query_text,
                    domains=domains,
                    entities=entities,
                    max_genes=max_genes,
                    party_id=party_id,
                    use_harmonic=use_harmonic,
                    use_sr=use_sr,
                    use_entity_graph=self.genome._entity_graph_retrieval_enabled,
                    read_only=read_only,
                    rerank_combinator=rerank_combinator,
                    rerank_override=rerank_override,
                )
            else:
                candidates = self.genome.query_docs(
                    domains,
                    entities,
                    max_genes=max_genes,
                    party_id=party_id,
                    use_harmonic=use_harmonic,
                    use_sr=use_sr,
                    use_entity_graph=self.genome._entity_graph_retrieval_enabled,
                    read_only=read_only,
                    query_type=query_type,
                    rerank_combinator=rerank_combinator,
                    query_text=query_text,
                    rerank_override=rerank_override,
                )
        except PromoterMismatch:
            pass

        # Check pending buffer for recently persisted documents not yet committed
        with self._pending_lock:
            for doc in self._pending:
                doc_domains = set(d.lower() for d in doc.promoter.domains)
                doc_entities = set(e.lower() for e in doc.promoter.entities)
                query_terms = set(d.lower() for d in domains + entities)

                if doc_domains & query_terms or doc_entities & query_terms:
                    candidates.append(doc)

        # Dedupe
        seen: set[str] = set()
        deduped: List[Gene] = []
        for g in candidates:
            if g.gene_id not in seen:
                seen.add(g.gene_id)
                deduped.append(g)

        # ── Cold-tier fallthrough (C.2 of B→C, opt-in) ──────────────────
        # When cold-tier is enabled, consult heterochromatin documents via
        # SEMA cosine similarity. Cold documents still hold their content
        # thanks to C.1's non-destructive demotion.
        #
        # Trigger semantics:
        #   include_cold=True (explicit override): ALWAYS try cold-tier,
        #     regardless of min_hot_genes. The caller has explicitly asked
        #     for cold-tier results — honor that even when hot returned
        #     some (possibly wrong) candidates. Cold-tier results are
        #     still subject to the SEMA cosine threshold; this just
        #     bypasses the "hot-was-empty" gate.
        #   include_cold=None (config-driven): consult cold-tier only when
        #     hot returns ≤ cold_tier_min_hot_genes. This is the auto-
        #     fallthrough mode for production traffic — fire cold only
        #     when hot is actually thin.
        #   include_cold=False: never fire cold-tier (overrides config).
        ctx_cfg = getattr(self.config, "context", None)
        cold_enabled = (
            include_cold
            if include_cold is not None
            else (bool(ctx_cfg.cold_tier_enabled) if ctx_cfg is not None else False)
        )
        if cold_enabled and query_text:
            explicit_override = include_cold is True
            min_hot = ctx_cfg.cold_tier_min_hot_genes if ctx_cfg is not None else 0
            should_fire = explicit_override or len(deduped) <= min_hot
            if should_fire:
                k = ctx_cfg.cold_tier_k if ctx_cfg is not None else 3
                min_cos = ctx_cfg.cold_tier_min_cosine if ctx_cfg is not None else 0.25
                try:
                    cold_genes = self.genome.query_cold_tier(
                        query_text=query_text,
                        k=k,
                        min_cosine=min_cos,
                    )
                    for cg in cold_genes:
                        if cg.gene_id not in seen:
                            seen.add(cg.gene_id)
                            deduped.append(cg)
                    if cold_genes:
                        # Mark on the manager so the response builder can
                        # report cold_tier_used in the agent metadata
                        self._last_cold_tier_used = True
                        self._last_cold_tier_count = len(cold_genes)
                except Exception:
                    log.warning("cold-tier retrieval failed", exc_info=True)

        return deduped[:max_genes * 2]

    def _build_abstain_window(
        self,
        *,
        query: str,
        effective_decoder_prompt: str,
        top_score: float,
        ratio: float,
        reason: str,
    ) -> ContextWindow:
        """Return the marker-only ContextWindow shipped when the ABSTAIN tier fires.

        See docs/specs/2026-05-02-abstain-tier-design.md §4. Distinct from the
        empty-candidates branch (above, in build_context) only on
        context_health.status — the LLM-visible bytes are identical
        (both ship the abstain form of <cymatix:no_match/>).

        Stage 6 (§6): the bytes are now the structured tag rather than
        the prose marker. ``_ABSTAIN_MARKER`` is preserved as the same
        string (``_no_match_token("abstain")``) so external callers
        comparing against it keep working for one release.
        """
        health = ContextHealth(
            ellipticity=0.0,
            coverage=0.0,
            density=0.0,
            freshness=0.0,
            genes_available=self.genome.total_genes(),
            genes_expressed=0,
            status="abstain",
        )
        return ContextWindow(
            ribosome_prompt=effective_decoder_prompt,
            expressed_context=_no_match_token("abstain"),
            total_estimated_tokens=estimate_tokens(effective_decoder_prompt),
            compression_ratio=1.0,
            context_health=health,
            metadata={
                "query": query,
                "genes_expressed": 0,
                "budget_tier": "abstain",
                "abstain_reason": reason,
                "top_score": float(top_score),
                "ratio": float(ratio),
                "no_match_reason": "abstain",
            },
        )

    def _apply_candidate_refiners(
        self,
        query: str,
        candidates: List[Gene],
        max_genes: int,
        *,
        use_cymatics: bool = True,
        use_harmonic_bin: bool = True,
        use_tcm: bool = True,
        query_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Gene], Dict[str, Dict[str, float]]]:
        """Apply post-retrieve candidate refiners before assembly or fingerprinting.

        Delegates to :func:`scoring.blend.apply_candidate_refiners`.

        ``query_scores`` is the request-scoped score snapshot (2026-07-18
        bugbash, score/result atomicity): when provided, the blend layer
        reads/mutates it instead of the shared ``genome.last_query_scores``
        so a concurrent request cannot cross-wire scores mid-turn.
        """
        from .scoring.blend import apply_candidate_refiners as _blend
        return _blend(
            query,
            candidates,
            max_genes,
            genome=self.genome,
            cymatics_enabled=self._use_cymatics,
            cymatics_peak_width=self._cymatics_peak_width,
            cymatics_distance_metric=self.config.cymatics.distance_metric,
            synonym_map=self.config.synonym_map,
            use_cymatics=use_cymatics,
            use_harmonic_bin=use_harmonic_bin,
            use_tcm=use_tcm,
            tcm_session=self._tcm_session,
            ray_trace_theta=getattr(self.config.retrieval, "ray_trace_theta", False),
            theta_weight=self.config.retrieval.theta_weight,
            blend_mode=self._blend_mode,
            query_scores=query_scores,
        )

    # -- Internal: Step 5 (assemble) -----------------------------------

    def _assemble(
        self, query: str, candidates: List[Gene],
        spliced_map: Dict[str, str],
        relation_graph: Optional[Dict] = None,
        query_signals: Optional[Tuple[List[str], List[str]]] = None,
        answer_slate: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        ignore_delivered: bool = False,
        decoder_prompt_override: Optional[str] = None,
        respect_caller_order: bool = False,
        caller_model_class: str = "generic",
        query_scores: Optional[Dict[str, float]] = None,
        tier_contributions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> ContextWindow:
        """
        Sort spliced parts, join with dividers, wrap in expressed_context tags.

        ``query_scores`` / ``tier_contributions`` (2026-07-18 bugbash,
        score/result atomicity): the caller's request-scoped snapshots of
        ``genome.last_query_scores`` / ``last_tier_contributions``. When
        provided, ordering, legibility, budget trim, and health all read
        these instead of the shared genome attributes, so a concurrent
        request's retrieval cannot swap the maps mid-assembly. ``None``
        (legacy direct callers / tests) falls back to the live genome
        state.

        MoE mode: sorts documents by retrieval score (highest first) instead of
        sequence_index, so the best match lands in position 0 — inside every
        SWA local attention window. Also injects an answer slate into the
        decoder prompt for front-loaded fact extraction.

        Session working-set (Sprint 2): when session_id is provided and
        budget.session_delivery_enabled is on, documents already delivered in
        this session are emitted as elision stubs rather than full content;
        fresh deliveries are logged to session_delivery_log for future
        elision. ignore_delivered=True bypasses the check (still logs).

        Stage 5 §4 candidate_order branches (caller_model_class):
          - generic   → unchanged (regression baseline). respect_caller_order
                        wins; else slate→score DESC; else sequence_index.
          - small_moe → reversed (foveated always-on); respect_caller_order
                        path covers this — same code path as generic.
          - frontier  → forward rank-1-first (narrative coherence). Not
                        sequence-index ordered — frontier wants the strongest
                        evidence at the top of the prompt under long-context
                        attention.
        """
        use_slate = answer_slate is not None
        # Request-scoped score map with legacy fallback (see docstring).
        _req_scores = (
            query_scores
            if query_scores is not None
            else (self.genome.last_query_scores or {})
        )
        if respect_caller_order:
            # Foveated-splice path (spec §5): the caller has already arranged
            # candidates in the desired emission order (e.g., reverse-rank
            # for BROAD). Skip the re-sort so reverse-rank actually reaches
            # the prompt instead of being clobbered back to score-DESC.
            sorted_genes = list(candidates)
        elif caller_model_class == "frontier":
            # Stage 5 §4: frontier callers want forward rank-1-first ordering
            # (long-context attention prefers narrative coherence with the
            # top evidence at the front of the prompt).
            scores = _req_scores
            sorted_genes = sorted(
                candidates,
                key=lambda g: scores.get(g.gene_id, 0),
                reverse=True,
            )
        elif use_slate:
            # MoE/small-model: relevance-first ordering — best document at position 0
            # so it's within every sliding-window attention layer
            scores = _req_scores
            sorted_genes = sorted(
                candidates,
                key=lambda g: scores.get(g.gene_id, 0),
                reverse=True,
            )
        else:
            # Dense: sequence ordering for narrative coherence
            sorted_genes = sorted(candidates, key=lambda g: g.promoter.sequence_index or 0)

        # Sprint 1 legibility pack: compute z-score stats once over the
        # retrieved document set so every header's confidence symbol is
        # calibrated against THIS response (not a knowledge store-wide baseline).
        # See cymatix_context/legibility.py + docs/FUTURE/AI_CONSUMER_ROADMAP_2026-04-14.md.
        #
        # Stage 5 §4: small_moe suppresses legibility headers entirely
        # (~80 tok/gene cost > legibility benefit for 4B-class callers).
        legibility_on = (
            self.config.budget.legibility_enabled
            and caller_model_class != "small_moe"
        )
        if legibility_on:
            _leg_scores = _req_scores
            _leg_tiers = (
                tier_contributions
                if tier_contributions is not None
                else (getattr(self.genome, "last_tier_contributions", None) or {})
            )
            _expressed_scores = {
                g.gene_id: _leg_scores.get(g.gene_id, 0.0) for g in sorted_genes
            }
            _score_stats = legibility.compute_score_stats(_expressed_scores)
        else:
            _leg_scores = {}
            _leg_tiers = {}
            _score_stats = (0.0, 0.0)

        # Sprint 2 session working-set: look up prior deliveries so documents
        # the consumer already holds can be elided with a stub. Bypassed
        # when flag off, session_id missing, or caller opted out.
        session_on = (
            self.config.budget.session_delivery_enabled
            and session_id is not None
            and not ignore_delivered
        )
        _prior_deliveries: Dict[str, Tuple[float, Optional[str], Optional[str]]] = {}
        _now_ts = time.time()
        if session_on:
            # Perf slice 1 (W0.2): this lookup is one SQL round trip per
            # candidate on the write connection — a campaign-flagged blind
            # spot (W1.3 batches it; this timer is its before/after gate).
            with _stage_timer("delivery_lookup"):
                try:
                    for g in sorted_genes:
                        prior = _session_delivery.already_delivered(
                            # Perf slice 2: pure read — per-thread reader
                            # (prior-turn deliveries are committed; the
                            # writer conn deadlocks cross-thread on py3.14).
                            self.genome.read_conn,
                            session_id=session_id,
                            gene_id=g.gene_id,
                        )
                        if prior is not None:
                            _prior_deliveries[g.gene_id] = prior
                except Exception:
                    log.debug("already_delivered lookup failed", exc_info=True)
                    # Treat as no prior deliveries — soft-fail preserves the
                    # retrieval path; at worst consumer re-sees content.
                    _prior_deliveries = {}

        parts: List[str] = []
        total_raw = 0
        # Track per-document log intent so budget-trim can discard entries
        # that didn't actually make it to the consumer. Value is
        # (mode, content_hash) for fresh deliveries, or None for elided.
        _delivery_log_map: Dict[str, Optional[Tuple[str, str]]] = {}

        # Control-tag neutralization (2026-08-08 audit; default-on since
        # 2026-08-19, #351 — receipt-gated flip, see config.py): escape
        # content-sourced "<cymatix:" so a document cannot forge the
        # genuine no-match/slate control tags. The genuine no-match tag
        # is emitted on the parts-empty branch below — never through
        # this loop — so it stays unescaped.
        neutralize_on = self.config.budget.neutralize_control_tags

        for g in sorted_genes:
            # Prefer compressor-spliced text; fall back to complement summary;
            # last resort is Headroom semantic compression (was content[:500]).
            spliced_text = spliced_map.get(g.gene_id) or g.complement or compress_text(
                g.content,
                target_chars=500,
                content_type=g.promoter.domains,
            )
            if neutralize_on:
                spliced_text = spliced_text.replace("<cymatix:", "&lt;cymatix:")
            prior = _prior_deliveries.get(g.gene_id) if session_on else None
            if prior is not None:
                # Document already delivered in this session — elide content
                prior_ts, _prior_mode, _prior_hash = prior
                try:
                    queries_ago = _session_delivery.count_queries_in_session_since(
                        self.genome.read_conn,  # pure read — see delivery_lookup
                        session_id=session_id,  # type: ignore[arg-type]
                        since=prior_ts,
                    )
                except Exception:
                    queries_ago = 0
                stub = _session_delivery.format_elision_stub(
                    gene_id=g.gene_id,
                    delivered_at=prior_ts,
                    now=_now_ts,
                    queries_ago=queries_ago,
                )
                parts.append(stub)
                _delivery_log_map[g.gene_id] = None  # no re-log on elision
                # #209 phase 2: event-count companion of the tokens-saved
                # counter below (docs elided vs tokens saved).
                try:
                    from .telemetry import session_elided_counter
                    session_elided_counter().add(1)
                except Exception:
                    pass
                # #209 phase 1: estimated tokens saved by eliding an
                # already-delivered document — full spliced text minus the
                # stub actually shipped (~4 chars/token; conservative, a
                # fresh delivery would also carry a legibility header).
                # No-op counter when OTel is off.
                try:
                    _saved = estimate_tokens(spliced_text) - estimate_tokens(stub)
                    if _saved > 0:
                        _session_tokens_saved_counter().add(_saved)
                except Exception:
                    pass
            elif legibility_on:
                header = legibility.format_gene_header(
                    gene_id=g.gene_id,
                    raw_chars=len(g.content),
                    compressed_chars=len(spliced_text),
                    combined_score=_leg_scores.get(g.gene_id, 0.0),
                    tier_contrib=_leg_tiers.get(g.gene_id, {}),
                    score_stats=_score_stats,
                )
                parts.append(f"{header}\n{spliced_text}")
                if session_on:
                    _delivery_log_map[g.gene_id] = (
                        "full", _session_delivery.content_hash(spliced_text),
                    )
            else:
                parts.append(spliced_text)
                if session_on:
                    _delivery_log_map[g.gene_id] = (
                        "full", _session_delivery.content_hash(spliced_text),
                    )
            total_raw += len(g.content)

        # Stage 6 (§6): if assembly produced no parts despite having
        # candidates, ship the structured no-match tag rather than the
        # legacy prose so the agent can branch on a tag.
        expressed = (
            "\n---\n".join(parts) if parts else _no_match_token("no_promoter_match")
        )

        # Wrap in tags
        expressed_wrapped = (
            "<expressed_context>\n"
            f"{expressed}\n"
            "</expressed_context>"
        )

        # MoE answer slate: inject pre-extracted KVs into decoder prompt
        # so they land in the first ~200 tokens (inside every SWA window).
        #
        # Stage 5 (2026-05-08) §5: small_moe gets a JSON-shaped, char-bounded
        # slate wrapped in <cymatix:slate>...</cymatix:slate>. Generic preserves
        # the legacy newline-joined 20-entry cap (regression baseline).
        if answer_slate:
            # Dedupe in arrival order (caller ordered them per §5 already).
            seen_kvs: set[str] = set()
            unique_slate: list[str] = []
            for kv in answer_slate:
                if kv not in seen_kvs:
                    seen_kvs.add(kv)
                    unique_slate.append(kv)

            if caller_model_class == "small_moe":
                # Spec §5: char-bounded greedy fill, JSON shape, MoE-friendly.
                slate_budget = int(getattr(
                    self.config.budget, "slate_char_budget", 1500,
                ))
                slate_text = _render_small_moe_slate(unique_slate, slate_budget)
                # Honor decoder_prompt_override if it has the slate placeholder
                # (answer_slate_only / condensed_with_slate); else fall back
                # to DECODER_MOE for compatibility.
                _template = decoder_prompt_override or DECODER_MOE
                if "{answer_slate}" in _template:
                    decoder_prompt = _template.replace("{answer_slate}", slate_text)
                else:
                    decoder_prompt = DECODER_MOE.replace("{answer_slate}", slate_text)
            else:
                # Generic branch — byte-identical to pre-Stage-5: newline-
                # joined, 20-entry cap, DECODER_MOE template.
                slate_text = "\n".join(unique_slate[:20])
                decoder_prompt = DECODER_MOE.replace("{answer_slate}", slate_text)
        else:
            # Honor per-request override (threaded from build_context) to
            # avoid racing on self._decoder_prompt across concurrent calls.
            decoder_prompt = decoder_prompt_override or self._decoder_prompt

        # Budget enforcement: if over token budget, drop lowest-scored documents
        est_tokens = estimate_tokens(decoder_prompt) + estimate_tokens(expressed_wrapped)
        budget = self.config.budget.ribosome_tokens + self.config.budget.expression_tokens

        # Phase 0.5 (2026-08-12): minimal trim-site recording — how many
        # documents the budget-eviction loop dropped. Surfaced via
        # window.metadata["budget_evicted"] for the delivery block.
        # W2.4: _budget_truncated counts seat-preserving part truncations
        # (min_delivered_docs floor) — a separate fact from evictions.
        _budget_evicted = 0
        _budget_truncated = 0
        if est_tokens > budget and len(parts) > 1:
            # Default path: drop the lowest-SCORED document regardless of
            # its position in the assembled prompt. Position-based pop()
            # drops parts[-1], which is the lowest-rank document only when
            # sorted_genes is in score-DESC order. Under sequence_index
            # ordering (the default for narrative coherence on factual /
            # multi_hop queries), parts[-1] is whichever document came last
            # by source coordinate — often the highest-scored document if
            # it sits near the end of a file. Score-aware lookup via the
            # request-scoped score map ensures the actual lowest-ranked
            # candidate is dropped first.
            #
            # Foveated reverse-rank path (respect_caller_order=True, spec §5):
            # the caller emits BROAD candidates in REVERSE-rank order so the
            # top-rank document lands LAST in the prompt (closest to user
            # query under decoder-only attention). Position-0 IS the
            # lowest-rank document by construction of that ordering, so
            # positional pop(0) is correct here and preserves the placement
            # invariant. A score-aware variant would be a strict improvement
            # under invariant violations, but is out of scope for this PR
            # (see the Stage 5 manager's note in the original #58 thread).
            #
            # sorted_genes is kept aligned with parts so the post-trim
            # delivered_ids / expressed_gene_ids slices stay correct under
            # both directions.
            trim_scores = _req_scores
            # W2.4 (2026-08-28): delivered-seat floor. Evict only down to
            # min_delivered_docs parts (clamped to the candidate count);
            # once the floor is reached, shrink the LARGEST remaining part
            # (tail-truncate, header-preserving, marked) instead of
            # evicting a seat — the crater cat-(b) fix (gold at map rank
            # 9-12 losing its seat, delivered_count 2-8; see
            # BudgetConfig.min_delivered_docs). If every part is already
            # at the truncation char floor and the prompt STILL overflows,
            # the floor yields and eviction resumes — the token budget is
            # the hard contract with the consumer and always wins last.
            # Default 0 keeps this loop byte-identical to the legacy trim.
            _seat_floor = min(
                int(getattr(self.config.budget, "min_delivered_docs", 0) or 0),
                len(parts),
            )
            _TRUNC_MIN_CHARS = 280
            _TRUNC_MARK = " ...[budget-trimmed]"
            while est_tokens > budget and len(parts) > 1:
                if _seat_floor >= 2 and len(parts) <= _seat_floor:
                    _big = max(range(len(parts)), key=lambda i: len(parts[i]))
                    if len(parts[_big]) > _TRUNC_MIN_CHARS:
                        _new_len = max(
                            _TRUNC_MIN_CHARS, int(len(parts[_big]) * 0.75),
                        )
                        parts[_big] = (
                            parts[_big][: max(1, _new_len - len(_TRUNC_MARK))]
                            .rstrip() + _TRUNC_MARK
                        )
                        _budget_truncated += 1
                    else:
                        _seat_floor = 0  # floor exhausted — evict below it
                        continue
                elif respect_caller_order:
                    parts.pop(0)
                    sorted_genes.pop(0)
                    # Phase 0.5 (2026-08-12): count budget evictions so the
                    # delivery-visibility block can report cap_binding =
                    # "token_budget" when this loop (not the classifier
                    # cap) is what truncated delivery.
                    _budget_evicted += 1
                else:
                    worst_idx = min(
                        range(len(sorted_genes)),
                        key=lambda i: trim_scores.get(sorted_genes[i].gene_id, 0.0),
                    )
                    parts.pop(worst_idx)
                    sorted_genes.pop(worst_idx)
                    _budget_evicted += 1
                expressed = "\n---\n".join(parts)
                expressed_wrapped = f"<expressed_context>\n{expressed}\n</expressed_context>"
                est_tokens = estimate_tokens(decoder_prompt) + estimate_tokens(expressed_wrapped)

        compressed_chars = len(expressed)

        # Sprint 2 session working-set: persist deliveries that actually
        # made it to the consumer (post-budget-trim). Elided documents (stubs)
        # already had their original delivery logged on the prior turn, so
        # we don't re-log them here. Any exception is swallowed — a log
        # hiccup must not break the retrieval response.
        if session_on and session_id is not None:
            # Perf slice 1 (W0.2): one INSERT+commit per delivered document
            # on the write connection — W1.3 collapses it to one txn; this
            # timer is its before/after gate.
            with _stage_timer("delivery_log"):
                try:
                    delivered_ids = [g.gene_id for g in sorted_genes[:len(parts)]]
                    # W2.3 Phase A: log_delivery is a module fn doing
                    # INSERT+commit on the shared writer — serialize at
                    # this call site via the store's write lock (the DAL
                    # module stays store-agnostic). getattr-guarded for
                    # sharded/stub genomes without a _write_lock.
                    import contextlib as _contextlib
                    _wl = getattr(self.genome, "_write_lock", None)
                    with (_wl if _wl is not None else _contextlib.nullcontext()):
                        for gid in delivered_ids:
                            entry = _delivery_log_map.get(gid)
                            if entry is None:
                                continue  # elided stub — no fresh log
                            mode, chash = entry
                            _session_delivery.log_delivery(
                                self.genome.conn,
                                session_id=session_id,
                                gene_id=gid,
                                content_hash=chash,
                                mode=mode,
                            )
                except Exception:
                    log.warning("session_delivery log_delivery failed", exc_info=True)

        # Delta-epsilon health signal
        # Use extracted domain/entity signals (not raw word splits with stop words)
        if query_signals:
            query_terms = [t.lower() for t in query_signals[0] + query_signals[1]]
        else:
            query_terms = query.lower().split()
        # 2026-07-18 bugbash: health must describe the post-trim,
        # actually-returned set — not the pre-trim candidate pool. Filter
        # ``candidates`` (which arrives in score-desc order; the
        # freshness_top1 contract) down to the documents that survived
        # the budget trim, preserving that order.
        _delivered_ids = {g.gene_id for g in sorted_genes[:len(parts)]}
        if len(_delivered_ids) < len(candidates):
            health_genes = [g for g in candidates if g.gene_id in _delivered_ids]
        else:
            health_genes = candidates
        health = self._compute_health(
            query_terms, health_genes, compressed_chars, relation_graph,
            query_scores=_req_scores,
        )

        # #209 phase 1: observe the splice compression ratio actually
        # shipped (identical to ContextWindow.compression_ratio below).
        # Balancing signal for splice_aggressiveness sweeps. No-op
        # instrument when OTel is off.
        try:
            _splice_ratio_histogram().record(
                total_raw / max(compressed_chars, 1),
                {"caller_model_class": caller_model_class},
            )
        except Exception:
            pass

        return ContextWindow(
            ribosome_prompt=decoder_prompt,
            expressed_context=expressed_wrapped,
            expressed_gene_ids=[g.gene_id for g in sorted_genes[:len(parts)]],
            total_estimated_tokens=est_tokens,
            compression_ratio=total_raw / max(compressed_chars, 1),
            context_health=health,
            metadata={
                "query": query,
                "genes_expressed": len(parts),
                "raw_chars": total_raw,
                "compressed_chars": compressed_chars,
                "moe_mode": bool(answer_slate),
                # Phase 0.5 (2026-08-12): documents dropped by the
                # budget-eviction loop above (0 = budget never bound).
                "budget_evicted": _budget_evicted,
                # W2.4: seat-preserving part truncations performed by the
                # min_delivered_docs floor (0 = floor off or never bound).
                "budget_truncated": _budget_truncated,
            },
        )

    # -- Internal: delta-epsilon health --------------------------------

    def _compute_health(
        self,
        query_terms: List[str],
        candidates: List[Gene],
        compressed_chars: int,
        relation_graph: Optional[Dict] = None,
        query_scores: Optional[Dict[str, float]] = None,
    ) -> ContextHealth:
        """
        Compute the delta-epsilon context health signal.

        ``query_scores`` (2026-07-18 bugbash, score/result atomicity): the
        caller's request-scoped score snapshot. ``None`` (legacy direct
        callers / tests) falls back to the live
        ``genome.last_query_scores``.

        Measures four dimensions:
            coverage  — fraction of query terms that matched knowledge store tags
            density   — fraction of retrieval token budget actually used
            freshness — three signals (Stage 7, 2026-05-08): freshness_min,
                        freshness_top1, freshness_weighted. Replaces the
                        prior mean(decay_score) so a stale top-1 needle is
                        no longer masked by fresh padding documents.
            ellipticity — composite score (geometric mean of the three)

        Status thresholds:
            aligned   — ellipticity >= 0.7 (knowledge store is well-grounded)
            sparse    — ellipticity >= 0.3 (knowledge store has gaps, model may guess)
            stale     — freshness_top1 < 0.4 OR freshness_weighted < 0.5
                        (Stage 7: top-1 stale, OR score-weighted body stale)
            denatured — ellipticity < 0.3 (context is unreliable)
        """
        import math

        # 2026-08-03 ERB fix: stats()'s SUM(LENGTH()) scans cost 11.1s
        # warm per request at 829k genes; only the count is consumed here.
        total_genes = self.genome.total_genes()
        genes_expressed = len(candidates)
        # Request-scoped score map with legacy fallback (see docstring).
        _scores_map = (
            query_scores
            if query_scores is not None
            else (getattr(self.genome, "last_query_scores", None) or {})
        )

        # Coverage: what fraction of query terms were found in the knowledge store?
        # Checks tags, FTS5 content matches, and key-value extracts.
        if query_terms:
            matched = 0
            # Collect all searchable text from retrieved documents
            all_tags: set[str] = set()
            all_content_lower = ""
            for g in candidates:
                all_tags.update(d.lower() for d in g.promoter.domains)
                all_tags.update(e.lower() for e in g.promoter.entities)
                if g.key_values:
                    all_tags.update(kv.lower() for kv in g.key_values)
                # Content presence check (for FTS5/SPLADE-found documents)
                all_content_lower += " " + (g.content[:2000] or "").lower()
            for term in query_terms:
                t = term.lower()
                if t in all_tags or t in all_content_lower:
                    matched += 1
            coverage = matched / len(query_terms)
        else:
            coverage = 0.0

        # Density: how much of the effective retrieval capacity did we use?
        # Scale budget by documents retrieved vs max — a query that correctly
        # retrieves 4 focused documents shouldn't be penalized for not filling 12 slots.
        max_genes = self.config.budget.max_genes_per_turn
        expressed_ratio = genes_expressed / max(max_genes, 1)
        effective_budget = self.config.budget.expression_tokens * 4 * max(expressed_ratio, 0.25)
        density = min(1.0, compressed_chars / max(effective_budget, 1))

        # Freshness — Stage 7 rewrite (2026-05-08, spec §3):
        # three signals computed in one pass over score-desc-ordered
        # candidates. The previous mean(decay) collapsed the entire
        # retrieved set into a single number and let 11 fresh padding
        # documents mask one stale needle (regression test
        # ``test_freshness_top1_dominates_padding``). The replacement:
        #   freshness_min      = min decay (worst document in set)
        #   freshness_top1     = decay of the top-1 by retrieval score
        #   freshness_weighted = score-weighted sum of decays
        #
        # The back-compat ``freshness`` field is aliased to
        # freshness_weighted so legacy consumers keep a meaningful
        # number — when scores are uniform it equals mean(decay), the
        # exact prior behavior, but when there's a strong top-1 the
        # weighting follows it. New code should read freshness_top1 /
        # freshness_min directly.
        if candidates:
            decays = [
                float(getattr(g.epigenetics, "decay_score", 0.0) or 0.0)
                for g in candidates
            ]
            freshness_min = min(decays)
            # candidates are passed in score-desc order; freshness_top1
            # is the head of that list (NOT min, NOT mean).
            freshness_top1 = decays[0]
            scores_for_weight = _scores_map
            raw_scores = [
                max(float(scores_for_weight.get(g.gene_id, 0.0) or 0.0), 0.0)
                for g in candidates
            ]
            s_total = sum(raw_scores)
            if s_total <= 0.0:
                # Equal-weight fallback when score map is empty (e.g.,
                # cold-start retrieval with no last_query_scores) — keeps
                # back-compat numerically equal to mean(decay).
                weights = [1.0 / len(candidates)] * len(candidates)
            else:
                weights = [s / s_total for s in raw_scores]
            freshness_weighted = sum(w * d for w, d in zip(weights, decays))
        else:
            freshness_min = 0.0
            freshness_top1 = 0.0
            freshness_weighted = 0.0
        # Back-compat shim — external callers that read
        # ``health.freshness`` keep working; the value now carries the
        # score-weighted signal so a stale top-1 pulls it down.
        freshness = freshness_weighted

        # Logical coherence (from NLI relation graph, if available)
        logical_coherence = 0.0
        if relation_graph:
            try:
                from .backends.nli_backend import compute_logical_coherence
                logical_coherence = compute_logical_coherence(relation_graph)
            except Exception:
                pass

        # Ellipticity: geometric mean of signals
        # Clamp inputs to avoid log(0)
        c = max(coverage, 0.01)
        d = max(density, 0.01)
        f = max(freshness, 0.01)
        if logical_coherence > 0:
            # 4-factor ellipticity when NLI is available
            lc = max(logical_coherence, 0.01)
            ellipticity = (c * d * f * lc) ** (1.0 / 4.0)
        else:
            # 3-factor ellipticity (backward compat)
            ellipticity = (c * d * f) ** (1.0 / 3.0)

        # Status classification — Stage 7 rule (spec §3):
        #   freshness_top1   < 0.4  → "stale" (the document we'd answer
        #                              from is itself stale)
        #   freshness_weighted < 0.5 → "stale" (score-weighted body
        #                              of retrieved set is stale)
        # Either trigger fires "stale". The previous rule used a single
        # mean(decay) < 0.4, which 11 fresh padding documents around one
        # stale needle could trivially mask.
        if genes_expressed > 0 and (
            freshness_top1 < 0.4 or freshness_weighted < 0.5
        ):
            status = "stale"
        elif ellipticity >= 0.7:
            status = "aligned"
        elif ellipticity >= 0.3:
            status = "sparse"
        else:
            status = "denatured"

        # Weighing surface (Step 1b, 2026-04-17) — pre-delivery coordinate
        # resolution confidence. ellipticity asks "did we deliver good
        # context"; these ask "how confident was the LOCATE step itself."
        # The know-vs-go signal the agent consumes to decide whether the
        # returned pointer is worth acting on.
        #
        # First pass was a null result: crispness × coverage correlates
        # poorly with ground-truth gold_delivered because both measure
        # internal consistency of what we retrieved, not whether the
        # retrieval was the RIGHT coordinate. Shipping the instrument
        # anyway so the bench can keep iterating on better signals.
        # See benchmarks/results/needle_step1b_conf_null_2026-04-17.json.
        crispness = 0.0
        neighborhood = 0.0
        top_score_raw = 0.0
        top_dominance = 0.0
        path_token_coverage = 0.0
        try:
            scores_map = _scores_map
            if candidates and scores_map:
                all_scored = sorted(scores_map.values(), reverse=True)
                ordered = sorted(
                    (scores_map.get(g.gene_id, 0.0) for g in candidates),
                    reverse=True,
                )
                if ordered:
                    top = ordered[0]
                    tail_idx = min(len(ordered) - 1, max_genes - 1)
                    tail = ordered[tail_idx] if tail_idx > 0 else 0.0
                    if top > 1e-9:
                        crispness = max(0.0, (top - tail) / (top + 1e-9))
                        threshold = 0.3 * top
                        n_strong = sum(1 for s in ordered if s >= threshold)
                        neighborhood = n_strong / max(len(ordered), 1)
                    top_score_raw = float(top)
                    pool = all_scored if len(all_scored) > len(ordered) else ordered
                    if pool:
                        mean_score = sum(pool) / len(pool)
                        if mean_score > 1e-9:
                            top_dominance = top / mean_score
        except Exception:
            log.debug("coordinate confidence calc failed", exc_info=True)

        # Path-token coverage (Step 1b-iter2, 2026-04-18): does the
        # delivered top-K actually *live* in the coordinate region the
        # query names? Pathway-layer signal — measures retrieval
        # location, not content overlap. See
        # benchmarks/results/needle_step1b_iter2_pathcov_2026-04-18.json.
        # File-grain companion added 2026-04-18 to catch same-folder-wrong-file.
        file_token_coverage = 0.0
        try:
            if candidates and query_terms:
                from .genome import file_tokens, path_tokens
                q_set = {t.lower() for t in query_terms if t}
                folder_hits = 0
                file_hits = 0
                for g in candidates:
                    sid = getattr(g, "source_id", None)
                    if not sid:
                        continue
                    if path_tokens(sid) & q_set:
                        folder_hits += 1
                    if file_tokens(sid) & q_set:
                        file_hits += 1
                denom = max(len(candidates), 1)
                path_token_coverage = folder_hits / denom
                file_token_coverage = file_hits / denom
        except Exception:
            log.debug("path token coverage calc failed", exc_info=True)

        # Composite v3 (2026-04-18): blend folder-grain + file-grain.
        # File-grain is weighted higher because same-folder-wrong-file is
        # the dominant silent-miss mode on the 10-needle bench. Coverage
        # acts as a floor so empty-context never scores 1.0.
        coverage_floor = max(coverage, 0.05)
        blended_pathcov = 0.4 * path_token_coverage + 0.6 * file_token_coverage
        resolution_conf = blended_pathcov * math.sqrt(coverage_floor)

        # Telemetry: surface per-query health so dashboards can watch
        # the retrieval-quality distribution over time. No-op if OTel
        # is disabled.
        try:
            from .telemetry import (
                context_ellipticity_histogram,
                context_health_status_counter,
            )
            context_ellipticity_histogram().record(
                float(ellipticity), attributes={"status": status},
            )
            context_health_status_counter().add(
                1, attributes={"status": status},
            )
        except Exception:  # pragma: no cover - telemetry must not break retrieval
            pass

        return ContextHealth(
            ellipticity=round(ellipticity, 4),
            coverage=round(coverage, 4),
            density=round(density, 4),
            freshness=round(freshness, 4),
            logical_coherence=round(logical_coherence, 4),
            genes_available=total_genes,
            genes_expressed=genes_expressed,
            status=status,
            coordinate_crispness=round(crispness, 4),
            neighborhood_density=round(neighborhood, 4),
            resolution_confidence=round(resolution_conf, 4),
            top_score_raw=round(top_score_raw, 4),
            top_dominance=round(top_dominance, 4),
            path_token_coverage=round(path_token_coverage, 4),
            file_token_coverage=round(file_token_coverage, 4),
            # Stage 7 (spec §3, §2 surface row 2446-2470): three
            # freshness signals, populated from the per-pass values
            # computed above. Optional[float] so legacy snapshots /
            # tests that build ContextHealth() with no candidates keep
            # working.
            freshness_min=round(freshness_min, 4),
            freshness_top1=round(freshness_top1, 4),
            freshness_weighted=round(freshness_weighted, 4),
        )

    # -- Stage 7: cold-tier peek + freshness pipeline ------------------

    def _cold_tier_peek(
        self,
        query: str,
        *,
        k: int = 3,
        min_cosine: float = 0.4,
    ) -> List[str]:
        """Stage 7 (spec §6) — surface heterochromatin SEMA hits as
        ``refresh_targets`` so the agent can re-ingest archived sources.

        Trigger contract (caller's responsibility): only invoke this
        when the hot tier returned thin results AND the corpus health
        is not "abstain". This helper is idempotent — it does NOT
        mutate ``last_cold_tier_used`` markers or promote any documents.

        Threshold default of 0.4 is tighter than ``query_cold_tier``'s
        own default (0.15) because cold peek competes with
        ``MissBlock(reason="sparse")``: we want it firing only on real
        archived hits, not on every weak SEMA neighbor.

        Returns:
          List of refresh targets — one ``source_id`` per cold document
          returned, deduped while preserving order. Empty list when
          the SEMA codec is unavailable, the cold-tier index is
          empty, or no match clears ``min_cosine``.
        """
        try:
            cold_hits = self.genome.query_cold_tier(
                query, k=k, min_cosine=min_cosine,
            )
        except Exception:
            log.debug("_cold_tier_peek: query_cold_tier failed", exc_info=True)
            return []
        if not cold_hits:
            return []

        seen: set[str] = set()
        targets: List[str] = []
        for g in cold_hits:
            # spec §6: prefer source_path under signals, then fall
            # through to source_id. Path-shaped string is what the
            # agent will re-read.
            src = (
                getattr(getattr(g, "epigenetics", None), "source_path", None)
                or getattr(g, "source_id", None)
            )
            if not src:
                continue
            if src in seen:
                continue
            seen.add(src)
            targets.append(str(src))
        return targets

    # -- Internal: compaction ------------------------------------------

    def _maybe_compact(self) -> None:
        now = time.time()
        if now - self._last_compact > self.config.genome.compact_interval:
            self.genome.refresh()  # See changes from external writers
            self.genome.compact()
            self._last_compact = now

    # -- Cleanup -------------------------------------------------------

    def close(self) -> None:
        self.genome.close()

    # ── Legacy method aliases (R3 Stage C; see docs/ROSETTA.md) ─────
    # Each alias points to the *same function object* as the canonical
    # method. External callers that still call the legacy names (tests
    # that monkey-patch _express, scripts that call manager._express)
    # keep working unchanged at the *class* level. Tests that
    # monkey-patch via instance attribute assignment must patch the
    # canonical name (or both — see tests/conftest.py for the pattern).
    _express              = _retrieve
    _make_parent_gene_id  = _make_parent_doc_id
    _upsert_parent_gene   = _upsert_parent_doc

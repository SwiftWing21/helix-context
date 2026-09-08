"""
KnowledgeStore — SQLite cold storage for the document pool.

Bio analogue (legacy term: genome):
    The genome is the full DNA library. Only ~1% is expressed per cell cycle.
    Our knowledge store stores all context documents in SQLite with a tags index
    for fast retrieval. Chromatin state controls accessibility.

Includes:
    - DDL (documents table + promoter_index join table)
    - Content-addressed document IDs (SHA256[:16])
    - Fix 1: synonym expansion for tags queries
    - Fix 1: co-activation pull-forward (associative memory)
    - Compaction (decay stale documents → COLD lifecycle tier)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from .accel import (
    json_loads,
    json_dumps,
    parse_promoter,
    parse_epigenetics,
    clear_parse_caches,
    batch_update_epigenetics,
)
from .exceptions import PromoterMismatch
from .hardware import SqliteMemPlan, sqlite_memory_budget
from .schemas import ChromatinState, EpigeneticMarkers, Gene, PromoterTags
from .backends.sema_codec import decode_embedding, sema_vec_to_blob
from .retrieval.measurement import current_capture, current_retrieval, trace_retrieval

log = logging.getLogger(__name__)


# ── Struggle 1 fix: source-path deny list ───────────────────────────────
#
# Paths that are structurally noise regardless of content. Any document whose
# source_id matches one of these patterns goes directly to HETEROCHROMATIN
# without computing a density score — it's cheaper and more reliable than
# relying on the scorer for content types we already know are noise.
#
# Categories covered:
#   - Build artifacts (.next, node_modules, __pycache__, dist, build, target)
#   - Lockfiles and minified bundles
#   - Web manifest files (app-paths-manifest.json, reference-manifest.js)
#   - Non-English locale directories (software i18n is high-volume low-signal)
#
# NOT in this list (deliberate):
#   - *.csv — business CSVs (customer data, financial records, invoice exports)
#     are legitimate ingest targets. Generic low-density CSVs will be caught
#     by the score gate instead.
#   - *.json — JSON is everywhere, most of it is config/data with signal
#   - *.md — markdown is primary signal content
#   - Cargo.toml / pyproject.toml — project metadata is signal
#   - Steam / game content (SteamLibrary, steamapps, BeamNG, Hades,
#     Factorio, Dyson Sphere, etc.) — reframed as high-SNR signal on
#     2026-04-10. Game files are content-dense with unambiguous literal
#     values (configs, enums, item IDs, code) and empirically produced
#     86% of correct answers on the N=50 v2 NIAH benchmark before the
#     original gate. Individual low-density game documents still get caught
#     by the score gate; the structural path is no longer a categorical
#     reject. See docs/BENCHMARKS.md and ~/.cymatix/shared/handoffs/ for
#     the full empirical basis.
#
# Patterns are anchored to directory boundaries to avoid false positives
# on legitimate files that happen to contain the substring.
#
# Issue #207 item 5: this is the documented built-in deny list. Deployments
# that need extra structural exclusions add regex fragments via
# ``[ingestion] deny_list_extra`` (config.py IngestionConfig) instead of
# editing this module — they're ORed onto DENY_PATTERNS + LOCALE_DENY_PATTERN
# (below) at KnowledgeStore construction time. This constant itself is
# unchanged by that knob; it stays the canonical "what ships out of the box"
# reference.
DENY_PATTERNS = [
    # Build artifacts
    r"[\\/]\.next[\\/]",
    r"[\\/]node_modules[\\/]",
    r"[\\/]__pycache__[\\/]",
    r"[\\/]dist[\\/]",
    r"[\\/]build[\\/](?!\.(bat|ps1|sh)$)",  # keep build.bat/build.sh etc.
    r"[\\/]target[\\/]debug[\\/]",
    r"[\\/]target[\\/]release[\\/]",
    # Lockfiles / manifests
    r"[\\/]package-lock\.json$",
    r"[\\/]yarn\.lock$",
    r"[\\/]Cargo\.lock$",
    r"[\\/]uv\.lock$",
    r"[\\/]poetry\.lock$",
    r"[\\/]Pipfile\.lock$",
    r"[\\/]composer\.lock$",
    # Minified bundles / source maps
    r"\.min\.(js|css|mjs)$",
    r"\.map$",
    # Next.js / web-framework manifests
    r"app-paths-manifest\.json$",
    r"app-build-manifest\.json$",
    r"_buildManifest\.js$",
    r"_ssgManifest\.js$",
    r"client-reference-manifest\.(js|json)$",
    r"server-reference-manifest\.(js|json)$",
    # Binary / compiled artifacts
    r"\.(pyc|pyo|so|dll|dylib|exe|wasm|bin|pack|idx)$",
]

# Non-English software locale directories (English is kept as the primary
# user base; other locales are high-volume low-signal for typical retrieval
# workloads). Game subtitles are NOT covered by this pattern — they're
# reframed as signal along with the rest of the game content.
#
# Issue #207 item 5: split out of DENY_PATTERNS into its own named constant
# so it can be toggled independently via ``[ingestion]
# locale_demotion_enabled`` (default True — byte-identical to the prior
# always-on behavior) for deployments that DO want non-English locale/
# directories ingested.
LOCALE_DENY_PATTERN = r"[\\/]locale[\\/](?!en[\\/])[a-z]{2,3}[\\/]"

# Back-compat aliases for the pre-#207 private names (this module's own
# code and any external callers that imported the "private" name).
_DENY_PATTERNS = DENY_PATTERNS + [LOCALE_DENY_PATTERN]
_DENY_RE = re.compile("|".join(_DENY_PATTERNS), re.IGNORECASE)


def build_deny_regex(
    extra_patterns: Optional[List[str]] = None,
    locale_demotion_enabled: bool = True,
) -> "re.Pattern[str]":
    """Compile the structural deny-list regex for one KnowledgeStore.

    Issue #207 item 5: ``extra_patterns`` (``[ingestion] deny_list_extra``)
    are regex fragments ORed onto the built-in ``DENY_PATTERNS``, same
    ``re.IGNORECASE`` semantics. ``locale_demotion_enabled=False`` drops
    ``LOCALE_DENY_PATTERN`` so non-English ``locale/`` directories are no
    longer structurally excluded. Defaults reproduce the prior hardwired
    behavior byte-for-byte.
    """
    patterns = list(DENY_PATTERNS)
    if locale_demotion_enabled:
        patterns.append(LOCALE_DENY_PATTERN)
    patterns.extend(extra_patterns or ())
    return re.compile("|".join(patterns), re.IGNORECASE)


def is_denied_source(
    source_id: Optional[str],
    deny_re: "Optional[re.Pattern[str]]" = None,
) -> bool:
    """Return True iff source_id matches the structural noise deny list.

    Exposed as a module-level function so tests and scripts can reuse it
    without constructing a full KnowledgeStore instance. ``deny_re``
    defaults to the module-level ``_DENY_RE`` (built-in patterns, locale
    demotion on) — pass a ``build_deny_regex(...)`` result to honor a
    KnowledgeStore's configured ``deny_list_extra`` /
    ``locale_demotion_enabled`` (Issue #207 item 5).
    """
    if not source_id:
        return False
    return bool((deny_re or _DENY_RE).search(source_id))


# ── Path tokenization for the path_key_index retrieval layer ────────────
# Splits source_id on common path separators + common filename punctuation.
# Each token becomes a retrieval signal paired with the document's key_values
# keys. A query like "what is the value of cymatix_port?" hits the index on
# path_token='cymatix' AND kv_key='port' → direct boost to the document.
#
# No LLM, no manual project list, no re-ingest required — purely derived
# from source_id + CpuTagger-extracted key_values. When a new project
# ingests at /SomeDir/NewProject/..., the token "newproject" becomes a
# retrieval signal automatically.
_PATH_SPLIT_RE = re.compile(r"[\\/\-_.\s:]+")

# Tokens that appear on nearly every path and carry no discriminating
# signal. Keeping this list tiny on purpose — it's the only maintenance
# burden, and overflowing it would be throwing signal away. Subset
# chosen from the actual source_id distribution on the 2026-04-12 knowledge store.
_PATH_NOISE_TOKENS = frozenset({
    "",
    # Drive / filesystem roots
    "f", "c", "d", "e", "g", "h",
    # Ubiquitous container dirs
    "projects", "project", "src", "lib", "app", "apps", "test", "tests",
    "docs", "doc", "main", "master", "common", "shared", "core", "util",
    "utils", "include", "includes", "public", "private", "tmp", "temp",
    "cache", "dist", "build", "bin", "obj", "x64", "x86", "arm64", "debug",
    "release", "data", "file", "files", "assets", "resources", "scripts",
    "config", "configs", "node_modules", "pycache", "__pycache__",
    # Filename-level noise
    "init", "index", "new", "old", "copy", "readme", "license",
    # Extensions that appear as tokens after splitting
    "py", "js", "ts", "tsx", "jsx", "md", "txt", "json", "toml", "yaml",
    "yml", "ini", "cfg", "xml", "html", "css", "scss", "rs", "go",
    "java", "c", "cpp", "h", "hpp", "sh", "bat", "ps1", "log",
})


def _kv_keys_from_list(kv_list) -> list:
    """Extract lowercased key names from a key_values List[str] of 'key=value'.

    Gene.key_values is stored as ["port=8080", "model=llava", ...] — each
    entry is a 'key=value' string. This helper pulls out just the key side,
    lowercased, deduped, skipping empties.

    Works transparently on dict inputs too for forward-compat — if a future
    document schema changes key_values to Dict[str, str], this still returns
    the right keys.
    """
    if not kv_list:
        return []
    # dict path (future-proof)
    if isinstance(kv_list, dict):
        return [k.lower() for k in kv_list.keys() if k]
    # list-of-strings path (current schema)
    out = []
    seen = set()
    for entry in kv_list:
        if not isinstance(entry, str):
            continue
        # Split on the FIRST '=' only — values can contain '='
        if "=" not in entry:
            continue
        key = entry.split("=", 1)[0].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def path_tokens(source_id: Optional[str]) -> set:
    """Extract retrieval-signal tokens from a file path.

    Splits on path separators and common filename punctuation, lowercases
    everything, drops single-char tokens and generic noise (see
    _PATH_NOISE_TOKENS). The result is the set of tokens that meaningfully
    identify this document's provenance for compound-lookup retrieval.

    Examples:
        "F:/Projects/cymatix-context/cymatix_context/config.py"
          → {"cymatix-context", "cymatix_context", "cymatix", "context"}

        "F:/SteamLibrary/steamapps/common/Hades II/content/maps.lua"
          → {"steamlibrary", "steamapps", "hades", "ii", "content", "maps"}

        "F:/Projects/CosmicTasha/src/components/Hero.tsx"
          → {"cosmictasha", "components", "hero"}

    Exposed as a module-level function so tests, backfill scripts, and
    retrieval code can reuse it without constructing a KnowledgeStore.
    """
    if not source_id:
        return set()
    out = set()
    # First pass: raw split on separators + punctuation
    for tok in _PATH_SPLIT_RE.split(source_id):
        t = tok.lower()
        if len(t) <= 1 or t in _PATH_NOISE_TOKENS:
            continue
        out.add(t)
        # Sub-split on interior hyphens/underscores to surface "cymatix"
        # from "cymatix-context" and "cymatix_context". Keeps the compound
        # form too so direct matches still work.
        if "-" in t or "_" in t:
            for sub in re.split(r"[-_]+", t):
                s = sub.lower()
                if len(s) > 1 and s not in _PATH_NOISE_TOKENS:
                    out.add(s)
    return out


def file_tokens(source_id: Optional[str]) -> set:
    """Extract tokens from just the filename (basename), excluding folders.

    Companion to path_tokens() — where that returns folder + file tokens
    mixed together, this returns only the basename's tokens. Motivated by
    the "same-folder-wrong-file" failure mode on the 10-needle bench:
    a query for "cymatix pipeline" matches path_tokens on any document under
    cymatix-context/, but only the documents whose filename itself mentions
    "pipeline" deserve a coordinate-confidence boost.

    Uses the same split + noise rules as path_tokens() but restricts
    input to the basename after the last separator.

    Examples:
        "F:/Projects/cymatix-context/docs/architecture/PIPELINE_LANES.md"
          → {"pipeline", "lanes"}

        "F:/Projects/cymatix-context/cymatix_context/retrieval.py"
          → {"retrieval"}

        "F:/Projects/cymatix-context/cymatix_context/genome.py"
          → {"knowledge store"}
    """
    if not source_id:
        return set()
    # Find the last separator (either / or \) to isolate the basename.
    # PurePath would also work but adds an import; cheap local split is fine.
    for sep in ("/", "\\"):
        if sep in source_id:
            basename = source_id.rsplit(sep, 1)[-1]
            break
    else:
        basename = source_id
    if not basename:
        return set()
    out = set()
    for tok in _PATH_SPLIT_RE.split(basename):
        t = tok.lower()
        if len(t) <= 1 or t in _PATH_NOISE_TOKENS:
            continue
        out.add(t)
        if "-" in t or "_" in t:
            for sub in re.split(r"[-_]+", t):
                s = sub.lower()
                if len(s) > 1 and s not in _PATH_NOISE_TOKENS:
                    out.add(s)
    return out


# Thresholds for the score-based gate. Calibrated against the 2026-04-10
# noise-diluted knowledge store (8,063 documents, ~42% structural noise). See
# scripts/simulate_density_gate_v2.py for the empirical basis.
_DENSITY_HETEROCHROMATIN_THRESHOLD = 0.50
_DENSITY_EUCHROMATIN_THRESHOLD = 1.00
_DENSITY_CONTENT_LENGTH_FLOOR = 100  # chars — prevents tiny-content score explosion
_DENSITY_ACCESS_OVERRIDE = 5         # access_count >= this keeps document OPEN regardless

# Working-set inference (Phase 1 slice 2 of the 8D dimensional roadmap).
# A document with at least _DENSITY_RATE_MIN_HITS accesses in the last
# _DENSITY_RATE_WINDOW seconds is considered "actively used right now"
# and gets the OPEN override regardless of static density score. The
# rate signal is sharper than the monotonic _DENSITY_ACCESS_OVERRIDE
# because it distinguishes "hot last hour" from "hot once a year ago" —
# the monotonic counter conflates them. Documents with empty recent_accesses
# buffers (legacy documents that pre-date Phase 1, or freshly ingested
# documents that haven't been touched yet) fall through to the monotonic
# fallback path, preserving backward compatibility.
#
# Reference: ~/.cymatix/shared/handoffs/2026-04-11_8d_dimensional_roadmap.md
_DENSITY_RATE_WINDOW = 3600.0   # 1-hour window
_DENSITY_RATE_MIN_HITS = 3      # ≥3 accesses in the window → override

# TTL for the memoized corpus-size count used by the IDF-weighted lexical
# anchor tier. Re-queried at most once per window to avoid a COUNT(*) on
# every retrieval call.
_CORPUS_SIZE_TTL = 60.0

# SQLite caps `WHERE col IN (?, ?, ...)` placeholders at
# SQLITE_LIMIT_VARIABLE_NUMBER. The historical default is 999; modern builds
# raise it (2000 on the Python 3.12 / SQLite 3.50 ARM64+x86 wheels we ship to,
# 32766 in some compile defaults). Batch size 500 stays under every cap we
# care about while keeping the loop overhead negligible for typical retrievals
# (gene_scores cardinality is usually well below 500 with SPLADE-on; this only
# loops when callers pass huge candidate sets).
_IN_BATCH_SIZE = 500


def _iter_in_batches(items, batch_size: int = _IN_BATCH_SIZE):
    """Yield (placeholder_string, batch_list) pairs for an IN-clause query.

    Used by call sites that build `WHERE col IN (?, ?, ...)` from a
    caller-determined candidate set. Splits the set into chunks small enough
    that the placeholder count stays under SQLite's
    `SQLITE_LIMIT_VARIABLE_NUMBER` cap on every supported build.

    Example::

        rows = []
        for ph, batch in _iter_in_batches(ids):
            rows.extend(cur.execute(
                f"SELECT * FROM t WHERE id IN ({ph})", batch,
            ).fetchall())
    """
    items = list(items)
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield ",".join("?" * len(batch)), batch


def _dense_matrix_dtype():
    """Resident dtype for the per-shard dense matrix (A2 RAM lever).

    Default ``float32`` (byte-identical to the historical path). Set
    ``CYMATIX_DENSE_MATRIX_DTYPE=float16`` to halve the resident dense-matrix
    footprint at 100-shard scale (~3.3 GB -> ~1.65 GB); numpy promotes to
    fp32 inside the matmul so cosine precision is unaffected -- only storage
    shrinks. OFF by default because it is a real (if tiny) precision change.
    """
    import numpy as _np
    val = os.environ.get("CYMATIX_DENSE_MATRIX_DTYPE", "float32").strip().lower()
    return _np.float16 if val in ("float16", "fp16", "half") else _np.float32


def apply_ann_gate(
    scored: "list[tuple[str, float]]",
    dense_hits: "list[tuple[str, float]]",
    *,
    threshold: float,
    min_genes: int,
    max_genes: int,
    dense_pool_floor_genes: int = 0,
) -> "list[str]":
    """ANN pool cut: threshold + ``min_genes`` floor + ``max_genes`` cap,
    plus the Issue #214 dense pool floor.

    Pure function over ``(gene_id, score)`` pairs so the gate semantics are
    unit-testable without a KnowledgeStore (``tests/test_dense_pool_floor.py``).

    Parameters
    ----------
    scored:
        Full union pool, sorted descending by score. Lexical-only candidates
        carry the ``threshold - 0.01`` pin from ``query_docs_ann``;
        dense-scored candidates carry their real cosine.
    dense_hits:
        The dense leg — ``(gene_id, cosine)`` sorted descending by cosine,
        exactly as ``query_docs_dense_recall`` returns it.
    dense_pool_floor_genes:
        Issue #214. The margin-over-random calibration computes
        ``mu + sigma_mult*sigma`` over RANDOM gene pairs; embedding
        anisotropy can push that bound ABOVE every real query-doc cosine.
        Measured twice, independently:

        * cc-exchange embedding-upgrade L1b: calibrated threshold 0.779 sat
          above the corpus max query-doc cosine ~0.713 (golds 0.46-0.68) —
          0/5000 pool-doc chunks cleared the gate by dense, so the 200-pool
          was built almost entirely from lexically-pinned docs;
        * a 480-question ERB run with 70.0% never-surfaced golds (gold not
          in top-10), corroborating an independent 67.2% measurement.

        Pool membership is strictly upstream of fusion ranking, so no dense
        re-weighting can recover candidates the gate already dropped. A
        threshold that admits ZERO dense candidates is mis-calibration by
        definition; the floor makes the dense leg degrade gracefully instead
        of dying. When fewer than N dense-scored candidates survive the
        legacy cut — but the dense leg HAD scored candidates — the top-N
        dense hits by cosine are appended to the pool. They then compete
        normally in downstream fusion scoring; nothing else about them is
        marked different. ``0`` disables (legacy gate-only behavior).

    Returns the kept ``gene_id`` list: the legacy cut order (score
    descending, truncated at ``max_genes``) with any floor-rescued dense ids
    appended in cosine order. When the gate is healthy (>= N dense-scored
    survivors) the output is byte-identical to the legacy cut.

    Blast-radius note: the floor appends AFTER the ``max_genes`` cap, so the
    pool exceeds ``max_genes`` only while the threshold is starving the
    dense leg, and by at most ``dense_pool_floor_genes`` entries.
    """
    # Legacy cut (pre-#214, byte-for-byte): admit everything at/above the
    # threshold; below it, only the min_genes rescue keeps admitting; stop at
    # the first non-rescued below-threshold candidate; cap at max_genes.
    kept: list[str] = []
    for gid, score in scored:
        if score >= threshold or len(kept) < min_genes:
            kept.append(gid)
        else:
            break
    kept = kept[:max_genes]

    if dense_pool_floor_genes <= 0 or not dense_hits:
        return kept

    dense_ids = {gid for gid, _ in dense_hits}
    admitted = sum(1 for gid in kept if gid in dense_ids)
    if admitted >= dense_pool_floor_genes:
        # Gate is healthy — the floor changes nothing.
        return kept

    kept_set = set(kept)
    for gid, _cosine in dense_hits:  # already sorted descending by cosine
        if admitted >= dense_pool_floor_genes:
            break
        if gid in kept_set:
            continue
        kept.append(gid)
        kept_set.add(gid)
        admitted += 1
    return kept


# Issue #341: chars of a document handed to the cross-encoder. The
# tokenizer re-truncates at 256 tokens anyway (rerank_backend._MAX_LENGTH),
# so this only bounds the string-slicing / tokenizer input cost — it is not
# the semantic truncation point.
_RERANK_TEXT_MAX = 2000


def _rerank_doc_text(doc) -> str:
    """The text the cross-encoder scores for ``doc``.

    Issue #341 sanctioned pair-text variation (829k A/B receipts,
    2026-08-06): the 829k run landed under the recall bar, so the primary
    form mirrors ``DeBERTaRibosome.re_rank``'s pair text exactly --
    ``f"{summary} [{domains}]"`` -- instead of scoring raw ``content``, so
    the pre-cap cross-encoder judges the same shape of text the ribosome's
    own re-rank does.

    Falls back to the OLD content-based form when ``doc.promoter`` is None
    or its summary is empty: the summary+domains form assumes a summary
    exists, and an empty one would otherwise score a garbage " []" pair
    against every other candidate's real text.
    """
    promoter = doc.promoter
    if promoter is not None and promoter.summary:
        domains = ", ".join(promoter.domains)
        text = f"{promoter.summary} [{domains}]" if domains else promoter.summary
        return text[:_RERANK_TEXT_MAX]
    return (doc.content or (promoter.summary if promoter else "") or "")[
        :_RERANK_TEXT_MAX
    ]


class KnowledgeStore:
    """SQLite-backed document storage with tags-tag retrieval."""

    def __init__(
        self,
        path: str,
        synonym_map: Optional[Dict[str, List[str]]] = None,
        sema_codec=None,
        splade_enabled: bool = False,
        splade_model: str = "naver/splade-cocondenser-ensembledistil",  # #207
        splade_content_cap: int = 1000,  # #207: chars SPLADE-encoded at ingest
        # #207 dense fast-follow: BGE-M3 model ID + passage char cap, threaded
        # through to get_shared_codec / _encode_dense_v2_blob. Defaults
        # reproduce the prior hardwired literals byte-for-byte.
        dense_model: str = "BAAI/bge-m3",
        dense_passage_char_cap: int = 2000,
        # Issue #207 item 5: structural deny-list extensibility. Extra regex
        # fragments ORed onto DENY_PATTERNS, and an on/off switch for the
        # LOCALE_DENY_PATTERN non-English demotion. Defaults reproduce the
        # prior hardwired behavior byte-for-byte.
        deny_list_extra: Optional[List[str]] = None,
        locale_demotion_enabled: bool = True,
        entity_graph: bool = False,
        sr_enabled: bool = False,
        sr_gamma: float = 0.85,
        sr_k_steps: int = 4,
        sr_weight: float = 1.5,
        sr_cap: float = 3.0,
        seeded_edges_enabled: bool = False,
        filename_anchor_enabled: bool = False,
        filename_anchor_weight: float = 4.0,
        bm25_shortlist_enabled: bool = False,
        bm25_shortlist_size: int = 50,
        bm25_prefilter_enabled: bool = False,
        bm25_prefilter_size: int = 200,
        fts5_candidate_depth: int = 0,
        entity_graph_retrieval_enabled: bool = False,
        dense_embedding_enabled: bool = False,
        # Stage 2 (2026-05-08): default dim raised from 256 -> 1024 (full
        # BGE-M3 Matryoshka). dim=256 collapsed random-pair cosine.
        dense_embedding_dim: int = 1024,
        # Tier-0 PR-1 (2026-05-16): when true, upsert_doc computes and
        # persists genes.embedding_dense_v2 inline (unless the caller passes
        # a precomputed vector). This is the WRITE path only and is
        # independent of dense_embedding_enabled, which gates RETRIEVAL.
        dense_embed_on_ingest: bool = False,
        # Issue #164: size-aware SPLADE auto-toggle thresholds. Both
        # default 0 (toggle off; ``splade_enabled`` is authoritative on
        # every upsert). When non-zero, ``upsert_doc`` consults the
        # current ``gene_count`` per upsert and routes through
        # ``storage.indexes.resolve_splade_enabled``. See IngestionConfig
        # docstring for the regime curve rationale.
        splade_auto_enable_below_genes: int = 0,
        splade_auto_disable_above_genes: int = 0,
        # Issue #139 (2026-05-18): default recalibrated 0.35 -> 0.58 for the
        # dim-1024 BGE-M3 v2 vectors. Callers normally pass the configured
        # value (config.retrieval.ann_similarity_threshold); this default is
        # the fallback for direct construction.
        ann_similarity_threshold: float = 0.58,
        ann_threshold_min_genes: int = 1,
        ann_threshold_max_genes: int = 12,
        # Stage 4 (2026-05-08): margin-over-random ANN calibration. When
        # ``ann_threshold_mode == "margin_over_random"``, query_genes_ann
        # reads the persisted threshold from ``genome_calibration``, falling
        # back to ``ann_similarity_threshold`` if the row is missing (with
        # one-time WARN). ``"absolute"`` keeps the legacy hand-picked value.
        # See docs/specs/2026-05-08-stage-4-threshold-calibration.md §3.
        ann_threshold_mode: str = "absolute",
        ann_threshold_sigma_multiplier: float = 3.0,
        # Issue #214: minimum dense-leg presence in the query_docs_ann pool
        # cut. When fewer than this many dense-scored candidates survive the
        # ANN threshold (but the dense leg had scored candidates), the top-N
        # dense hits by cosine are admitted anyway. 0 disables (legacy
        # gate-only). Full rationale + evidence: apply_ann_gate docstring.
        dense_pool_floor_genes: int = 8,
        # Stage 2: dense recall pool size, decoupled from ann_threshold_max_genes.
        dense_pool_size: int = 500,
        # Stage 3 (2026-05-08): Reciprocal Rank Fusion (spec
        # docs/specs/2026-05-08-stage-3-rrf-fusion.md). Default "rrf"
        # matches RetrievalConfig.fusion_mode (#256 harmonized the layer
        # defaults; the config default flipped in #247 with SIKE Run-2
        # receipts). Pass "additive" explicitly for the legacy
        # pre-Stage-3 accumulator (scheduled for removal in v(N+2)).
        # Issue #202: the per-tier weights bind in BOTH fusion modes --
        # under "additive" they are the tier coefficients/caps themselves,
        # under "rrf" they are rank post-multipliers.
        fusion_mode: str = "rrf",
        rrf_k: int = 60,
        # Issue #260 (2026-07-12): rank/confidence-gated RRF. Default-inert —
        # rrf_gate_enabled=False makes the per-store Fuser byte-identical to
        # today. When enabled, a recall arm contributes RRF mass only for its
        # top-M ranks (rrf_gate_top_m) and/or entries whose raw score clears
        # rrf_gate_min_score. Fanned to the solo Genome AND every per-shard
        # Genome via open_read_source, same as the other retrieval knobs.
        rrf_gate_enabled: bool = False,
        rrf_gate_top_m: int = 0,
        rrf_gate_min_score: float = 0.0,
        # Issue #255 (PR-2, 2026-07-10): post-fusion rerank combinator. Only
        # consulted under fusion_mode == "rrf" (the additive branch never
        # touches rerank_additive). Default "additive" reproduces the shipped
        # fused + rerank_additive finalization byte-for-byte; the alternatives
        # (fused_tier / eps_band / off) are bench-gated. See
        # docs/research/2026-07-09-scoring-combinator-exploration.md.
        rerank_combinator: str = "additive",
        rerank_band_delta: float = 0.05,   # eps_band relative tie width δ
        rerank_tier_weight: float = 1.0,   # fused_tier per-class rank multiplier
        fts5_weight: float = 3.0,
        splade_weight: float = 3.5,
        tag_exact_weight: float = 3.0,
        tag_prefix_weight: float = 1.5,
        # Issue #327: Tier-1 tag-IDF discipline (default off — flat legacy
        # scoring). See RetrievalConfig.tag_idf_enabled for the receipt.
        tag_idf_enabled: bool = False,
        # #327 iteration 2: exclude tags with df > cap×N from Tier-1
        # entirely (membership relief). 0.0 = off.
        tag_df_cap: float = 0.0,
        # 2026-08-03 ERB fix: same discipline for SPLADE query terms —
        # drop terms with df > cap×N before the postings join (one real
        # query vector touched 5.06M posting rows at 829k genes). 0.0 = off.
        splade_df_cap_fraction: float = 0.0,
        # Issue #202: warm ΣĒMA boost (Tier 4 Mode A) weight. NEW knob --
        # the tier's additive literal was 2.0 and it had no per-tier
        # weight at all (under RRF it is a post-fusion additive, so it
        # never went through fuser.add_tier()). Default == old literal.
        sema_boost_weight: float = 2.0,
        sema_cold_weight: float = 3.0,
        lex_anchor_weight: float = 1.5,
        harmonic_weight: float = 1.0,
        entity_graph_weight: float = 0.5,
        dense_weight: float = 1.0,
        # Tier-0 PR-3 (2026-05-16): additive-mode dense merge weight.
        # Under fusion_mode == "additive", a dense hit's cosine is
        # multiplied by this before it enters gene_scores. BM25-comparable
        # (tag-exact weight is 3.0). Unused under RRF, where dense ranks
        # are fused, not summed.
        dense_additive_weight: float = 4.0,
        # Tier-0 review fix (2026-05-16): noise floor for the additive-mode
        # dense merge. A dense hit whose cosine is below this does not
        # contribute to gene_scores (still kept as a candidate with
        # negligible weight). Consistent with query_cold_tier's 0.15
        # min_cosine. Unused under RRF.
        dense_additive_min_cosine: float = 0.15,
        # Semantic-wiring arm (PRD 2026-06-02). semantic_dense_additive_weight
        # is consulted only when query_type=="semantic" AND CYMATIX_SEMANTIC_ARM=1.
        # semantic_broaden_routing is read by ShardRouter; accepted here so the
        # fanned genome_kwargs don't raise (Genome ignores it).
        semantic_dense_additive_weight: float = 16.0,
        semantic_broaden_routing: bool = True,
        # Sharded-retrieval router knobs (#222/#223): read by ShardRouter;
        # accepted here so the fanned genome_kwargs don't raise (a solo
        # Genome ignores them — they only bind on the cross-shard merge path).
        shard_fetch_multiplier: float = 2.0,
        shard_fetch_scale_with_shards: bool = False,
        coact_reserved_slots: int = 0,
        coact_link_boost: float = 0.5,
        # #264: doc-type boost mode. Read by ShardRouter on the cross-shard
        # merge; accepted here so the fanned genome_kwargs don't raise (a
        # solo / per-shard Genome ignores it — the boost only binds on the
        # cross-shard merge path).
        doc_type_boost_mode: str = "additive",
        authority_path_selectivity: bool = False,
        pki_weight: float = 1.0,
        # Tier-0 read gate. True is byte-identical to the pre-gate store;
        # False skips the path_key_index read entirely (no SQL, no signal
        # timing, no RRF tier) so the ablation ladder can measure the
        # layer the 2026-08-06 ledger flagged as "no off-switch exists".
        pki_enabled: bool = True,
        # ── Issue #341: query-time cross-encoder rerank (default-OFF) ──
        # Pre-cap rerank seam. With rerank_enabled=False (and no per-query
        # override) every retrieval path below is byte-identical to the
        # pre-#341 store — the invariance suites are the receipt.
        # ``rerank_depth`` is the candidate count fed to the cross-encoder
        # BEFORE the max_genes cut; the cut itself is unchanged, so the
        # returned document COUNT never depends on this knob (only which
        # documents fill those slots).
        rerank_enabled: bool = False,
        rerank_depth: int = 50,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        # DI seam: a (query, texts) -> scores callable. None (the default)
        # lazy-imports backends.rerank_backend.score_pairs on first use, so
        # torch/transformers never enter the import graph of a store that
        # isn't reranking. Tests inject a deterministic fake here.
        # Issue #372: writer-connection PRAGMA synchronous. SQLite's default
        # is FULL (every commit fsyncs); under WAL, NORMAL only fsyncs on
        # checkpoint — worst case on OS crash is losing the last commit, not
        # corruption. In-repo precedent: scripts/backfill_filename_domains.py.
        synchronous: str = "NORMAL",
        rerank_scorer: Optional[Callable[[str, List[str]], List[float]]] = None,
        main_conn: Optional[sqlite3.Connection] = None,
        shard_name: str = "main",
        read_only: bool = False,
        # RAM-aware SQLite pragmas (PRD 2026-05-30). None -> resolve a
        # single-DB budget here; ShardRouter passes a shard-count-aware plan.
        mem_plan: Optional[SqliteMemPlan] = None,
    ):
        self.path = path
        self.read_only = read_only
        self._mem_plan = mem_plan
        self.synonym_map = synonym_map or {}
        self._sema_codec = sema_codec  # Optional SemaCodec for Tier 4 retrieval
        self._replication_mgr = None  # Set by set_replication_manager()
        self._splade_enabled = splade_enabled
        self._splade_model = splade_model  # #207 item 1
        self._splade_content_cap = splade_content_cap  # #207 item 3
        self._dense_model = dense_model  # #207 dense fast-follow
        self._dense_passage_char_cap = dense_passage_char_cap  # #207 dense fast-follow
        # #207 item 5: deny-list extensibility. Compiled once at construction
        # (extra patterns / locale toggle don't change over a store's life).
        self._deny_list_extra = list(deny_list_extra or [])
        self._locale_demotion_enabled = locale_demotion_enabled
        self._deny_re = build_deny_regex(
            extra_patterns=self._deny_list_extra,
            locale_demotion_enabled=self._locale_demotion_enabled,
        )
        # Issue #164: per-upsert SPLADE auto-toggle thresholds.
        self._splade_auto_enable_below: int = int(splade_auto_enable_below_genes or 0)
        self._splade_auto_disable_above: int = int(splade_auto_disable_above_genes or 0)
        # Cached gene_count used by the per-upsert auto-toggle resolver.
        # Refreshed lazily from the DB on each upsert that has the toggle
        # opted in; avoids hammering COUNT(*) when the toggle is off.
        self._splade_auto_cached_count: int = 0
        self._entity_graph_enabled = entity_graph
        # Tier 5b: entity graph retrieval boost (Step 3C, 2026-05-08).
        # Separate from _entity_graph_enabled (write-side) — this controls
        # whether entity_graph rows are consulted during query_genes().
        self._entity_graph_retrieval_enabled: bool = bool(entity_graph_retrieval_enabled)
        # Tier 5.5 Successor Representation (Sprint 2, Stachenfeld 2017).
        self._sr_enabled = sr_enabled
        self._sr_gamma = sr_gamma
        self._sr_k_steps = sr_k_steps
        self._sr_weight = sr_weight
        self._sr_cap = sr_cap
        # Sprint 4 — Hebbian seeded-edge evidence accumulation.
        self._seeded_edges_enabled = seeded_edges_enabled
        # Tier 0.5 filename-anchor (2026-04-15 Dewey-pivot spike).
        self._filename_anchor_enabled = filename_anchor_enabled
        self._filename_anchor_weight = filename_anchor_weight
        self._bm25_shortlist_enabled = bool(bm25_shortlist_enabled)
        self._bm25_shortlist_size = int(bm25_shortlist_size) if bm25_shortlist_size else 50
        self._bm25_prefilter_enabled = bool(bm25_prefilter_enabled)
        self._bm25_prefilter_size = int(bm25_prefilter_size)
        # A4 / #205: Tier-3 FTS content fetch depth override (0 = auto = limit*2).
        self._fts5_candidate_depth = int(fts5_candidate_depth) if fts5_candidate_depth else 0
        # #327: scale the source-authority boost by matching-term selectivity.
        self._authority_path_selectivity = bool(authority_path_selectivity)
        # Step 4 — BGE-M3 dense vectors + ANN threshold (2026-05-08).
        self._dense_embedding_enabled: bool = bool(dense_embedding_enabled)
        self._dense_embedding_dim: int = int(dense_embedding_dim)
        # Tier-0 PR-1 (2026-05-16): inline dense-vector write at ingest.
        # Independent of _dense_embedding_enabled (retrieval gate).
        self._dense_embed_on_ingest: bool = bool(dense_embed_on_ingest)
        self._ann_threshold: float = float(ann_similarity_threshold)
        self._ann_min_genes: int = int(ann_threshold_min_genes)
        self._ann_max_genes: int = int(ann_threshold_max_genes)
        # Stage 4: persisted calibration mode + cache.
        # ``_ann_threshold_calibrated`` is the lazily-loaded value from
        # ``genome_calibration``; ``None`` means "not loaded yet" and the
        # next ``_get_effective_ann_threshold`` call will populate it.
        self._ann_threshold_mode: str = str(ann_threshold_mode)
        self._ann_threshold_sigma_multiplier: float = float(ann_threshold_sigma_multiplier)
        self._ann_threshold_calibrated: Optional[float] = None
        self._ann_threshold_calibration_meta: Optional[Dict[str, Any]] = None
        self._ann_threshold_fallback_warned: bool = False
        # Issue #214: dense pool floor (0 = legacy gate-only). Applies inside
        # apply_ann_gate, i.e. per-store — in sharded mode each per-shard
        # KnowledgeStore receives this via the fanned genome_kwargs, so the
        # floor holds per shard and the router-level merge dedups as usual.
        self._dense_pool_floor_genes: int = int(dense_pool_floor_genes)
        # Stage 2 (2026-05-08): pool size for dense + lex recall, decoupled
        # from max_genes (the post-ranking final cut).
        self._dense_pool_size: int = int(dense_pool_size)
        self._dense_codec: "BGEM3Codec | None" = None  # lazy-loaded
        # Stage 2: in-memory hot-tier dense matrix cache. Populated lazily by
        # _ensure_dense_matrix() on first dense recall query; invalidated by
        # _invalidate_dense_matrix() after upsert/delete batches.
        self._dense_matrix: "np.ndarray | None" = None
        self._dense_matrix_ids: "list[str] | None" = None
        self._dense_matrix_lock = threading.Lock()
        # One-time-only flags so we don't spam logs.
        self._dense_v2_partial_warned: bool = False
        # ── Stage 3 (2026-05-08): RRF fusion mode + per-tier weights ──
        # Spec: docs/specs/2026-05-08-stage-3-rrf-fusion.md.
        # When "additive", query_genes() uses the legacy gene_scores +=
        # tier_score accumulator. When "rrf", each tier ALSO ranks its
        # output through the Fuser and the final sort uses fused scores.
        # Issue #202: the per-tier weights bind in BOTH modes -- additive
        # consumes them as tier coefficients/caps (defaults == the old
        # inline literals -> bit-identical when untouched); RRF consumes
        # them as rank post-multipliers.
        # Validate now so a typo in cymatix.toml fails fast at construction
        # instead of producing surprising rankings at query time.
        if fusion_mode not in ("additive", "rrf"):
            raise ValueError(
                f"fusion_mode must be 'additive' or 'rrf', got {fusion_mode!r}"
            )
        self._fusion_mode: str = fusion_mode
        self._rrf_k: int = int(rrf_k)
        # Issue #260: rank/confidence-gated RRF. Validate now so a bad
        # rrf_gate_top_m in cymatix.toml fails fast at construction, mirroring the
        # fusion_mode / rerank_combinator guards. gate_min_score accepts any
        # float (a negative floor is legitimate for negated-bm25 arms), so it is
        # only coerced. enabled=False keeps the store byte-identical regardless
        # of the two thresholds — see _rrf_gate_params().
        self._rrf_gate_enabled: bool = bool(rrf_gate_enabled)
        if int(rrf_gate_top_m) < 0:
            raise ValueError(
                f"rrf_gate_top_m must be >= 0 (0 = ungated), got {rrf_gate_top_m!r}"
            )
        self._rrf_gate_top_m: int = int(rrf_gate_top_m)
        self._rrf_gate_min_score: float = float(rrf_gate_min_score)
        # Issue #255 (PR-2): validate the rerank combinator now so a typo in
        # cymatix.toml fails fast at construction, mirroring the fusion_mode
        # guard above. The four names are the only valid operators (see
        # retrieval/rerank_combinators.py).
        if rerank_combinator not in ("additive", "fused_tier", "eps_band", "off"):
            raise ValueError(
                "rerank_combinator must be one of "
                "'additive'|'fused_tier'|'eps_band'|'off', "
                f"got {rerank_combinator!r}"
            )
        self._rerank_combinator: str = rerank_combinator
        self._rerank_band_delta: float = float(rerank_band_delta)
        self._rerank_tier_weight: float = float(rerank_tier_weight)
        self._fts5_weight: float = float(fts5_weight)
        self._splade_weight: float = float(splade_weight)
        self._tag_exact_weight: float = float(tag_exact_weight)
        self._tag_idf_enabled: bool = bool(tag_idf_enabled)
        self._tag_df_cap: float = float(tag_df_cap)
        self._splade_df_cap_fraction: float = float(splade_df_cap_fraction)
        self._tag_prefix_weight: float = float(tag_prefix_weight)
        self._sema_boost_weight: float = float(sema_boost_weight)
        self._sema_cold_weight: float = float(sema_cold_weight)
        self._lex_anchor_weight: float = float(lex_anchor_weight)
        self._harmonic_weight: float = float(harmonic_weight)
        self._entity_graph_weight: float = float(entity_graph_weight)
        self._dense_weight: float = float(dense_weight)
        # Tier-0 PR-3: additive-mode dense merge weight (see ctor param).
        self._dense_additive_weight: float = float(dense_additive_weight)
        # Tier-0 review fix: additive-mode dense merge noise floor (see ctor param).
        self._dense_additive_min_cosine: float = float(dense_additive_min_cosine)
        # Semantic-wiring arm (PRD 2026-06-02): scoped dense weight, gated at
        # query time by query_type=="semantic" + env CYMATIX_SEMANTIC_ARM=1.
        self._semantic_dense_additive_weight: float = float(semantic_dense_additive_weight)
        self._semantic_broaden_routing: bool = bool(semantic_broaden_routing)
        self._pki_weight: float = float(pki_weight)
        self._pki_enabled: bool = bool(pki_enabled)
        # Issue #341: pre-cap cross-encoder rerank (see ctor params).
        self._rerank_enabled: bool = bool(rerank_enabled)
        self._rerank_depth: int = int(rerank_depth)
        self._rerank_model: str = str(rerank_model)
        self._rerank_scorer: Optional[Callable[[str, List[str]], List[float]]] = (
            rerank_scorer
        )
        self._threshold_dim_warned: bool = False
        # Phase 2 claims layer (2026-04-19). Optional hook — when a main.db
        # connection is supplied, upsert_gene emits literal claims into it
        # after each ingest. None = no auto-hook, preserving legacy behavior.
        self._main_conn = main_conn
        self._shard_name = shard_name

        # First-run UX (v0.7.0 QA): a fresh clone/worktree has no
        # genomes/main/ directory, and sqlite3 cannot create a database
        # file inside a missing directory — it raises "unable to open
        # database file" and the backend dies at import time. Create the
        # parent chain for plain file paths (":memory:" and URI forms
        # excluded). connect() below still surfaces any real error.
        _path_str = str(self.path)
        if _path_str not in ("", ":memory:") and not _path_str.startswith("file:"):
            try:
                os.makedirs(
                    os.path.dirname(os.path.abspath(_path_str)) or ".",
                    exist_ok=True,
                )
            except OSError:
                log.debug("genome parent mkdir failed", exc_info=True)

        # Checkpoint WAL BEFORE opening our long-lived connection
        # so we see the latest state from any external writers.
        # Speculative pre-open optimisation — the subsequent connect() will
        # still succeed if this fails, so we log at debug level but keep the
        # exc_info so diagnostic scrapes can see why it failed.
        try:
            _tmp = sqlite3.connect(self.path)
            _tmp.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _tmp.close()
        except Exception:
            log.debug("WAL pre-open checkpoint failed", exc_info=True)

        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")  # 30s retry on lock
        # Issue #372: default NORMAL — under WAL a per-commit fsync (FULL,
        # SQLite's own default) is not needed for corruption-safety, and
        # ingest currently commits per strand. Invalid values fall back
        # loudly to NORMAL rather than silently to SQLite's FULL.
        _sync = str(synchronous or "NORMAL").upper()
        if _sync not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            log.warning(
                "[genome] synchronous=%r is not a valid SQLite level "
                "(OFF|NORMAL|FULL|EXTRA) — falling back to NORMAL", synchronous
            )
            _sync = "NORMAL"
        self._synchronous = _sync
        self.conn.execute(f"PRAGMA synchronous={_sync}")
        # Cap WAL file size — without this, SQLite resets (zero-fills) the
        # WAL on truncate but keeps the high-water-mark size on disk.
        self.conn.execute("PRAGMA journal_size_limit=67108864")  # 64 MB
        # RAM-aware SQLite budget (PRD 2026-05-30: dynamic-ram-scaling). The
        # plan scales the per-connection page cache + mmap to the host and
        # shard count; ShardRouter threads a shard-count-aware plan, standalone
        # KnowledgeStores resolve a single-DB budget. The `conservative`
        # profile reproduces v0.6.1 exactly (mmap off, 2/4 MB caches) — the
        # fan-out commit guard, now opt-in rather than the unconditional default.
        _plan = self._mem_plan if self._mem_plan is not None else sqlite_memory_budget(1)
        self.conn.execute(f"PRAGMA cache_size={_plan.writer_cache_size}")
        self.conn.execute(f"PRAGMA mmap_size={_plan.mmap_size}")
        self._upsert_count = 0  # WAL checkpoint cadence counter
        # Bug bash 2026-07-18: the write connection is shared across threads
        # (check_same_thread=False). Serialize multi-statement upsert
        # transactions so a concurrent caller's commit can never publish
        # another request's half-written gene. RLock so a locked caller can
        # re-enter upsert paths without deadlocking.
        self._write_lock = threading.RLock()
        # Issue #372: document-scoped transaction state. _defer_depth > 0
        # means a deferred_commits() scope is open on the thread that holds
        # _write_lock — write units then run as SAVEPOINTs and the single
        # durable commit (one fsync) happens at scope exit. All three fields
        # are only mutated with _write_lock held.
        self._defer_depth = 0
        self._savepoint_seq = 0
        self._pending_checkpoint: Optional[str] = None
        self.last_query_scores: Dict[str, float] = {}  # Retrieval scores from last query
        self._last_query_scores_lock = threading.Lock()
        # Issue #255 (PR-2) debug hooks: the eligible-restricted pre-combination
        # maps from the last RRF query, set under _last_query_scores_lock. Lets
        # the ab_rerank_combinator desk-test recover the exact fused-vs-final
        # inversion signature without re-deriving anything. Empty under
        # fusion_mode == "additive" (that branch never populates them).
        self.last_fused_scores: Dict[str, float] = {}
        self.last_rerank_additive: Dict[str, float] = {}
        # Per-tier score breakdown for the last query: {gene_id: {tier_name: score}}.
        # Populated alongside last_query_scores in query_genes(). Lets the bench /
        # profiler see which retrieval signals fired (and how strongly) for each
        # candidate document — turns the lane graph into a measurable activation matrix.
        # See benchmarks/bench_skill_activation.py and docs/PIPELINE_LANES.md.
        self.last_tier_contributions: Dict[str, Dict[str, float]] = {}
        # Perf slice 1 (W0.2): per-signal wall-time (ms) for the last query,
        # published under _last_query_scores_lock alongside last_query_scores
        # (same last-query contract and the same known cross-request-global
        # caveat — W1.6 scoops all of these into a request-scoped outcome).
        self.last_signal_timings: Dict[str, float] = {}
        # Issue #341: the last query's final rank order, published under
        # _last_query_scores_lock on both retrieval paths and in BOTH rerank
        # arms — with rerank off it is the sim/fused order, so an A/B keeps an
        # identical rank basis and the metric can never silently read a stale
        # ordering.
        #
        # KNOWN ASYMMETRY — the two paths publish different DEPTHS, and a
        # consumer computing rank-of-gold has to know which one it is reading:
        #
        #   query_docs_ann  -> the FULL union pool (~pool_size, default 500)
        #                      in final order, i.e. every candidate the ANN
        #                      path considered, both above and below the gate.
        #   query_docs      -> the POST-CUT list only (max_genes * 2). Ids
        #                      that ranked below that cut are simply absent,
        #                      so a gold document outside the cut reads as
        #                      "not retrieved" rather than "retrieved at rank
        #                      N" — an unmeasurable, not a zero.
        #
        # Deliberately not unified: widening the lex publication would mean
        # widening the ranking itself. The ladder's A/B runs dense-on, so the
        # ANN (full-pool) semantics are the ones the receipts actually use.
        self.last_ranked_ids: List[str] = []
        # Issue #341 companion: {"gate_kept": n, "floor_appended": n} from the
        # last query_docs_ann call (both arms). Lets the ablation ladder slice
        # needles that only surfaced through the #214 dense pool floor.
        # Written by query_docs_ann only; {} everywhere else.
        self.last_rerank_diag: Dict[str, int] = {}
        # Perf slice 1 (W0.3): every COMMIT executed on the writer connection,
        # including raw conn.commit() calls from call sites that bypass
        # _write_lock (Registry, session_delivery, cwola). Uses the sqlite3
        # authorizer-free trace hook so no caller has to cooperate. Counter
        # is process-cumulative; per-request deltas are computed by the
        # pipeline ring. Approximate under concurrent writers by design.
        self.commit_count: int = 0
        self._install_commit_trace()
        self._sema_cache: Optional[Dict] = None  # Pre-materialized ΣĒMA vectors (hot tier)
        self._cold_sema_cache: Optional[Dict] = None  # Pre-materialized ΣĒMA vectors (cold tier, C.2)
        # Council fix: sema vectorless auto-gate. When a cache build finds
        # ZERO non-null genes.embedding rows, the relevant flag flips true
        # and short-circuits further work for that tier: hot
        # (_sema_vectorless) gates the query-side Tier 4 codec.encode() call
        # in query_docs (~:2850) — a hot-tier-vectorless store cannot
        # produce a single sema candidate there, so the remote /encode/sema
        # RTT is pure waste; cold (_cold_sema_vectorless) gates the
        # repeated full-table rescan in query_cold_tier. Kept as two flags
        # rather than one shared flag because the hot and cold builds scan
        # disjoint chromatin partitions — a cold-tier-vectorless bed can
        # still have hot-tier vectors (and vice versa), so folding them
        # into one flag could false-positive-gate the *other* tier's
        # perfectly good encode call. Both share one log-once guard, since
        # spec wants a single "sema tier idle" line per store lifetime.
        # Cleared by invalidate_sema_cache() / invalidate_cold_sema_cache()
        # so a bed that later gains vectors (upsert, backfill script)
        # re-checks on its next build.
        self._sema_vectorless: bool = False
        self._cold_sema_vectorless: bool = False
        self._sema_idle_logged: bool = False  # one log.info per store lifetime
        # Memoized corpus size for IDF weighting (refreshed every
        # _CORPUS_SIZE_TTL seconds). Prevents the IDF denominator from
        # collapsing to the scored-candidate count on every query.
        self._corpus_size: int = 0
        self._corpus_size_ts: float = 0.0
        self._init_db()

        # Dedicated read-only connection — WAL allows concurrent readers
        # without blocking the writer. Separate connection = no lock contention.
        # isolation_level=None makes Python's sqlite3 module skip the implicit
        # BEGIN around SELECTs. Without it, the first read pins a WAL snapshot
        # for the lifetime of the process and prevents wal_checkpoint(TRUNCATE)
        # from advancing — which is the dominant cause of WAL bloat.
        if self.path != ":memory:":
            self._reader = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True,
                check_same_thread=False, timeout=10,
                isolation_level=None,
            )
            self._reader.row_factory = sqlite3.Row
            self._reader.execute("PRAGMA busy_timeout=10000")
            # RAM-aware budget (see writer block). The reader carries the
            # FTS5/dense scan load, so under `conservative` it keeps the
            # historical larger 4 MB cap; under auto/aggressive both connections
            # share the scaled cache figure and the same mmap window.
            self._reader.execute(f"PRAGMA cache_size={_plan.reader_cache_size}")
            self._reader.execute(f"PRAGMA mmap_size={_plan.mmap_size}")
        else:
            self._reader = None
        # Perf slice 2: per-thread reader connections (see read_conn).
        # The creating thread keeps self._reader; other threads get their
        # own mode=ro connection, tracked in a registry for close().
        self._thread_readers = threading.local()
        self._reader_owner_thread = threading.get_ident()
        self._thread_reader_registry: list = []
        self._thread_reader_registry_lock = threading.Lock()

        # Create SPLADE inverted index if enabled, or if the size-aware
        # auto-toggle (issue #164) could turn it on later. Without this
        # table the auto-enable arm has nowhere to write and silently
        # no-ops, so any non-zero ``splade_auto_enable_below_genes`` --
        # even paired with ``splade_enabled=False`` -- still warrants
        # the table.
        _need_splade_table = (
            self._splade_enabled or self._splade_auto_enable_below > 0
        )
        if _need_splade_table:
            try:
                from .backends import splade_backend
                splade_backend.create_splade_table(self.conn)
                log.info("SPLADE inverted index ready")
            except ImportError:
                log.warning("SPLADE backend not available (transformers not installed)")
                self._splade_enabled = False
            except Exception:
                log.warning("SPLADE table creation failed", exc_info=True)
                self._splade_enabled = False

    def _rrf_gate_params(self) -> Tuple[int, float]:
        """Effective (gate_top_m, gate_min_score) for the per-store Fuser (#260).

        Returns the ungated sentinels ``(0, 0.0)`` whenever ``rrf_gate_enabled``
        is False, so a disabled gate is byte-identical to the pre-gate path no
        matter what the two thresholds are set to. When enabled, returns the
        configured values verbatim (either or both may still be a 0 sentinel,
        in which case that sub-lever is individually off).
        """
        if not self._rrf_gate_enabled:
            return (0, 0.0)
        return (self._rrf_gate_top_m, self._rrf_gate_min_score)

    def _init_db(self) -> None:
        from .storage.ddl import fts5_is_external_content, init_db
        self._fts_available = init_db(self.conn)
        # Issue #338: which write discipline genes_fts needs. External-content
        # (new stores + migrated stores) has no shadow text of its own, so
        # deletes must re-supply the prior indexed values; legacy contentful
        # stores keep the plain DELETE+INSERT discipline untouched.
        self._fts_external = (
            self._fts_available and fts5_is_external_content(self.conn.cursor())
        )

    def _ensure_registry_schema(self, cur: sqlite3.Cursor) -> None:
        """Delegate to storage.ddl.ensure_registry_schema."""
        from .storage.ddl import ensure_registry_schema
        ensure_registry_schema(cur, self.conn)

    # ── WAL snapshot management ──────────────────────────────────────

    def _refresh_snapshot(self) -> None:
        """Release stale WAL read transaction so next SELECT sees current state.

        In SQLite WAL mode, Python's sqlite3 module starts an implicit
        transaction on SELECT. That transaction pins a snapshot — external
        writers (ingest, thinning scripts) commit to the WAL but this
        connection won't see those changes until the implicit transaction ends.

        Calling commit() ends the implicit transaction. The next SELECT
        will start a new one with the latest WAL state.
        """
        try:
            # W2.3 Phase A: a bare commit on the shared writer can publish
            # ANOTHER thread's half-written transaction (the commit-steal
            # class) — take the write lock so it only ever lands between
            # complete writer transactions.
            with self._write_lock:
                self.conn.commit()
        except Exception:
            pass  # No active transaction — safe to ignore

    # ── ΣĒMA vector cache (pre-materialized for fast Mode B scans) ──

    def _build_sema_cache(self) -> None:
        """
        Load all ΣĒMA vectors into RAM as a numpy matrix for fast
        cosine similarity. Eliminates 7K json_loads() per Mode B query.

        Cache structure:
            gene_ids: list[str] — ordered document IDs
            matrix: np.ndarray (N, 20) — float32 ΣĒMA vectors
        """
        try:
            import numpy as np
        except ImportError:
            log.debug("numpy not available, ΣĒMA cache disabled")
            return

        cur = self.read_conn.cursor()
        # Try the tier-aware query first; fall back to legacy schema
        # when the read path is a replica that hasn't been migrated yet.
        try:
            rows = cur.execute(
                "SELECT gene_id, embedding FROM genes "
                "WHERE embedding IS NOT NULL AND chromatin < ? "
                "AND COALESCE(compression_tier, 0) < 2",
                (int(ChromatinState.HETEROCHROMATIN),),
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "compression_tier" in str(e):
                log.warning(
                    "read_conn lacks compression_tier column — "
                    "falling back to legacy schema (likely a stale replica)"
                )
                rows = cur.execute(
                    "SELECT gene_id, embedding FROM genes "
                    "WHERE embedding IS NOT NULL AND chromatin < ?",
                    (int(ChromatinState.HETEROCHROMATIN),),
                ).fetchall()
            else:
                raise

        if not rows:
            # Council fix: zero non-null genes.embedding rows in the
            # hot-tier scan -> no candidate can ever carry a vector this
            # cache build. Flip the store-level short-circuit so Tier 4's
            # query-side codec.encode() call (and this full-table rescan,
            # which Mode B would otherwise retry every query) stop firing
            # until invalidate_sema_cache() clears the flag — e.g. after an
            # upsert or backfill script adds a vector.
            self._sema_vectorless = True
            if not self._sema_idle_logged:
                log.info("sema tier idle: no stored vectors")
                self._sema_idle_logged = True
            self._sema_cache = None
            return

        gene_ids = []
        vectors = []
        for r in rows:
            try:
                vec = decode_embedding(r["embedding"])
                if isinstance(vec, list) and len(vec) == 20:
                    gene_ids.append(r["gene_id"])
                    vectors.append(vec)
            except Exception:
                continue

        if vectors:
            matrix = np.array(vectors, dtype=np.float32)
            # Normalize rows for cosine similarity via dot product
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            matrix = matrix / norms
            self._sema_cache = {"gene_ids": gene_ids, "matrix": matrix}
            log.info("ΣĒMA cache built: %d vectors (%d KB)",
                     len(gene_ids), matrix.nbytes // 1024)
        else:
            self._sema_cache = None

    def invalidate_sema_cache(self) -> None:
        """Mark hot-tier cache stale — rebuilt on next Mode B query."""
        self._sema_cache = None
        # Council fix: a bed that gains vectors (upsert, backfill) must
        # re-check on the next build instead of staying permanently
        # gated off by a stale vectorless verdict.
        self._sema_vectorless = False

    # ── Cold-tier ΣĒMA retrieval (C.2, 2026-04-10) ─────────────────────
    #
    # The hot-tier retrieval paths all filter `WHERE lifecycle tier <
    # HETEROCHROMATIN`, so heterochromatin documents are invisible to normal
    # /context queries. C.1 made compress_to_heterochromatin non-destructive
    # so the underlying content/complement/codons are preserved. This
    # block adds the opt-in retrieval path that consults cold-tier documents
    # via ΣĒMA cosine similarity and returns them with content restored.
    #
    # Design notes:
    #   - Separate cache from the hot-tier _sema_cache so hot queries have
    #     zero overhead from cold-tier capability.
    #   - Lazy build on first use. Invalidated on any upsert.
    #   - Requires numpy for batched cosine similarity (falls back to
    #     empty result if unavailable, matching hot-tier Mode B behavior).
    #   - Requires a SemaCodec attached to the KnowledgeStore instance (for
    #     encoding the query text). If no codec, returns empty.
    #   - Callers must explicitly request cold-tier retrieval — it is
    #     never invoked implicitly from query_genes(). The wiring into
    #     context_manager is a follow-up (C.2-wire) and is gated behind
    #     a cymatix.toml config flag.

    def _build_cold_sema_cache(self) -> None:
        """Build the heterochromatin ΣĒMA vector cache for fast cosine scans.

        Scans all documents at ``chromatin = HETEROCHROMATIN`` that still have
        an embedding (ΣĒMA vector). Normalizes and stacks them into a
        dense numpy matrix for batched cosine similarity at query time.
        """
        try:
            import numpy as np
        except ImportError:
            log.debug("numpy not available, cold ΣĒMA cache disabled")
            return

        cur = self.read_conn.cursor()
        rows = cur.execute(
            "SELECT gene_id, embedding FROM genes "
            "WHERE embedding IS NOT NULL AND chromatin = ?",
            (int(ChromatinState.HETEROCHROMATIN),),
        ).fetchall()

        if not rows:
            # Council fix: zero non-null genes.embedding rows in the
            # heterochromatin (cold-tier) scan -> flip the cold-only
            # short-circuit so query_cold_tier stops re-running this
            # full-table scan on every call until invalidate_cold_sema_cache()
            # clears the flag (e.g. after a document is archived with a
            # vector, or a backfill runs).
            self._cold_sema_vectorless = True
            if not self._sema_idle_logged:
                log.info("sema tier idle: no stored vectors")
                self._sema_idle_logged = True
            self._cold_sema_cache = None
            return

        gene_ids = []
        vectors = []
        for r in rows:
            try:
                vec = decode_embedding(r["embedding"])
                if isinstance(vec, list) and len(vec) == 20:
                    gene_ids.append(r["gene_id"])
                    vectors.append(vec)
            except Exception:
                continue

        if vectors:
            matrix = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            matrix = matrix / norms
            self._cold_sema_cache = {"gene_ids": gene_ids, "matrix": matrix}
            log.info("Cold ΣĒMA cache built: %d vectors (%d KB)",
                     len(gene_ids), matrix.nbytes // 1024)
        else:
            self._cold_sema_cache = None

    def invalidate_cold_sema_cache(self) -> None:
        """Mark cold-tier cache stale — rebuilt on next query_cold_tier call."""
        self._cold_sema_cache = None
        # Council fix: re-check on next build instead of staying gated.
        self._cold_sema_vectorless = False

    def mark_verified(
        self,
        gene_ids: List[str],
        ts: float,
        *,
        read_only: bool = False,
    ) -> int:
        """Stage 7: bump ``last_verified_at`` for the listed gene_ids.

        Spec: docs/specs/2026-05-08-stage-7-freshness-gate.md §4 + §5.

        Called from ``freshness.revalidate_and_mark`` after a successful
        mtime-fresh check. ``read_only`` gates the write — under
        ``read_only=True`` this is a no-op so the Stage-1 read_only
        contract holds. Cache updates in ``revalidate_source`` happen
        in-memory and are NOT a DB write, so they remain allowed.

        Returns the number of rows actually updated. Zero is a valid
        return (e.g. caller passed unknown gene_ids); the freshness
        gate uses this method as a hint, not as a correctness check.
        """
        if read_only:
            # No-op under the read_only contract. Stage 7 spec §5
            # requires this to be silent — no warning, no exception.
            return 0
        if not gene_ids:
            return 0
        try:
            # W2.3 Phase A: UPDATE + commit on the shared writer must hold
            # the write lock (unlocked-committer class; see log_health).
            with self._write_lock:
                cur = self.conn.cursor()
                placeholders = ",".join("?" * len(gene_ids))
                cur.execute(
                    f"UPDATE genes SET last_verified_at = ? "
                    f"WHERE gene_id IN ({placeholders})",
                    (float(ts), *gene_ids),
                )
                updated = cur.rowcount or 0
                self.conn.commit()
            return int(updated)
        except Exception:
            log.warning(
                "mark_verified: UPDATE failed for %d gene_ids",
                len(gene_ids),
                exc_info=True,
            )
            return 0

    def query_cold_tier(
        self,
        query_text: str,
        k: int = 3,
        min_cosine: float = 0.15,
    ) -> List[Gene]:
        """Search heterochromatin-tier documents by ΣĒMA cosine similarity.

        Cold-tier retrieval is opt-in — normal ``query_genes()`` does not
        consult this path. Use when a query is known to target archived
        knowledge, or as a fallthrough when hot-tier results are empty or
        too sparse to answer confidently.

        Parameters
        ----------
        query_text : str
            Natural-language query to encode via the attached SemaCodec.
        k : int
            Maximum number of cold-tier documents to return. Defaults to 3 —
            cold retrieval should be a precision tool, not a dump.
        min_cosine : float
            Similarity floor in 20-dim ΣĒMA space. Matches below this
            are discarded. Defaults to 0.25 — ΣĒMA's 20-dim projection
            is sparse by design, so typical close-paraphrase pairs score
            around 0.15–0.30 in cosine. The existing hot-tier Mode A/B
            paths use 0.3/0.4 as meaningful thresholds (see
            ``query_genes`` Tier 4). 0.25 here is slightly more
            permissive because the cold path is only reached when hot
            results are already thin — better to surface a weak match
            than nothing.

        Returns
        -------
        list[Document]
            Up to ``k`` heterochromatin documents with full content restored,
            sorted by cosine similarity descending. Each document's
            ``chromatin`` field will still show HETEROCHROMATIN — the
            caller is responsible for deciding whether to promote the
            document back to OPEN based on the retrieval event (e.g., by
            updating access_count and letting a future sweep reconsider it).

        Returns empty list when any precondition fails:
            - No SemaCodec attached (self._sema_codec is None)
            - numpy unavailable
            - No heterochromatin documents with embeddings in the knowledge store
            - No matches clear the min_cosine threshold
        """
        if self._sema_codec is None:
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        if self._cold_sema_cache is None:
            # Council fix: skip the full-table rescan once a prior build
            # already proved the heterochromatin tier has zero non-null
            # embeddings — invalidate_cold_sema_cache() clears this if a
            # later archive/backfill adds a vector.
            if self._cold_sema_vectorless:
                return []
            self._build_cold_sema_cache()
        if self._cold_sema_cache is None:
            return []  # Nothing in the cold tier

        try:
            query_vec = self._sema_codec.encode(query_text)
            q = np.array(query_vec, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-8:
                return []
            q = q / q_norm

            cache = self._cold_sema_cache
            sims = cache["matrix"] @ q  # (N,20) @ (20,) → (N,)

            # Sort descending, filter by threshold, take top-k
            top_idx = np.argsort(sims)[::-1]
            selected_ids: List[str] = []
            selected_sims: Dict[str, float] = {}
            for idx in top_idx:
                sim = float(sims[idx])
                if sim < min_cosine:
                    break  # Sorted descending — once below threshold, stop
                gid = cache["gene_ids"][idx]
                selected_ids.append(gid)
                selected_sims[gid] = sim
                if len(selected_ids) >= k:
                    break

            # #209 phase 1: observe above-floor cold-tier ΣĒMA cosines so
            # the cold arm shares the cymatix_dense_cosine calibration
            # surface. No-op instrument when OTel is off.
            try:
                from .telemetry import dense_cosine_histogram
                _dch = dense_cosine_histogram()
                for _sim in selected_sims.values():
                    _dch.record(float(_sim), {"arm": "cold"})
            except Exception:
                pass

            if not selected_ids:
                return []

            # Fetch full Document objects (content is preserved thanks to C.1)
            genes: List[Gene] = []
            for gid in selected_ids:
                gene = self.get_doc(gid)
                if gene is not None:
                    genes.append(gene)

            # Expose similarity scores via last_query_scores for the
            # caller (context_manager uses this for tier-budget decisions)
            for gid, sim in selected_sims.items():
                self.last_query_scores[gid] = sim

            return genes
        except Exception:
            log.debug("cold-tier ΣĒMA retrieval failed", exc_info=True)
            return []

    # ── Stage 4: calibrated ANN threshold reader ─────────────────────

    def _get_effective_ann_threshold(self) -> float:
        """Return the active ANN cosine cutoff for ``query_genes_ann``.

        Stage 4 (spec §3): when ``ann_threshold_mode == "margin_over_random"``,
        read the persisted ``ann_threshold`` row from ``genome_calibration``
        on first call and cache it. If the row is missing (e.g. operator has
        not yet run ``scripts/calibrate_thresholds.py``), log a one-time WARN
        and fall back to the legacy absolute ``self._ann_threshold``.

        ``"absolute"`` mode returns the legacy value with no DB read —
        ``mode="global"`` callers (the default) get byte-identical pre-Stage-4
        behavior.

        Cache invalidation: ``set_replication_manager`` clears the cache so
        readers connected through a swapped replica re-read on next call.
        """
        if self._ann_threshold_mode != "margin_over_random":
            return self._ann_threshold
        if self._ann_threshold_calibrated is not None:
            return self._ann_threshold_calibrated

        try:
            row = self.read_conn.execute(
                "SELECT value_json FROM genome_calibration WHERE key = 'ann_threshold'"
            ).fetchone()
        except Exception:
            # Table may not exist on a pre-Stage-4 database — treat as missing.
            log.debug("genome_calibration read failed", exc_info=True)
            row = None

        if row is None:
            if not self._ann_threshold_fallback_warned:
                log.warning(
                    "ann_threshold_mode='margin_over_random' but no calibration row in "
                    "genome_calibration — falling back to ann_similarity_threshold=%.3f. "
                    "Run scripts/calibrate_thresholds.py to populate.",
                    self._ann_threshold,
                )
                self._ann_threshold_fallback_warned = True
            self._ann_threshold_calibrated = self._ann_threshold
            return self._ann_threshold

        try:
            meta = json_loads(row["value_json"]) if hasattr(row, "__getitem__") else json_loads(row[0])
            value = float(meta["value"])
        except Exception:
            log.warning(
                "genome_calibration ann_threshold row is malformed; falling back to "
                "ann_similarity_threshold=%.3f",
                self._ann_threshold,
                exc_info=True,
            )
            self._ann_threshold_calibrated = self._ann_threshold
            return self._ann_threshold

        self._ann_threshold_calibrated = value
        self._ann_threshold_calibration_meta = meta
        return value

    def get_calibration_provenance(self) -> Optional[Dict[str, Any]]:
        """Return the cached calibration metadata (for /health and /context).

        Triggers a lazy load via ``_get_effective_ann_threshold`` if the cache
        is empty AND mode is ``margin_over_random``. Returns ``None`` if the
        mode is ``absolute`` OR no calibration row is present.
        """
        if self._ann_threshold_mode != "margin_over_random":
            return None
        # Trigger lazy load.
        self._get_effective_ann_threshold()
        if self._ann_threshold_calibration_meta is None:
            return None
        # Defensive copy — callers may mutate.
        return dict(self._ann_threshold_calibration_meta)

    def upsert_calibration(self, key: str, value: Dict[str, Any]) -> None:
        """UPSERT a row into ``genome_calibration``. Used by the calibration
        script and tests. Idempotent — last write wins.

        SQLite's ``busy_timeout=30000`` and the journal_mode=WAL configuration
        on ``self.conn`` (set in ``__init__``) handle CROSS-PROCESS writer
        serialization; cross-THREAD access to this one shared connection is
        serialized by ``_write_lock`` (W2.3 Phase A — unlocked, a concurrent
        committer steals the implicit transaction or deadlocks on py3.14).
        """
        payload = json_dumps(value)
        now = time.time()
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO genome_calibration (key, value_json, computed_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "  value_json = excluded.value_json, "
                "  computed_at = excluded.computed_at",
                (key, payload, now),
            )
            self.conn.commit()
        # Invalidate cache so the next read sees the new value.
        if key == "ann_threshold":
            self._ann_threshold_calibrated = None
            self._ann_threshold_calibration_meta = None
            self._ann_threshold_fallback_warned = False

    # ── Persistence ──────────────────────────────────────────────────

    def set_replication_manager(self, mgr) -> None:
        """Attach a ReplicationManager for distributed knowledge store clones."""
        self._replication_mgr = mgr
        # Stage 4: invalidate calibration cache so a swapped replica re-reads.
        self._ann_threshold_calibrated = None
        self._ann_threshold_calibration_meta = None
        self._ann_threshold_fallback_warned = False

    def corpus_size(self) -> int:
        """Return the memoized total document count for IDF weighting.

        Refreshed from ``SELECT COUNT(*) FROM genes`` at most once per
        ``_CORPUS_SIZE_TTL`` seconds. Used by the IDF-weighted lexical
        anchor tier so rare-term boosts reflect the *knowledge store* size, not
        the size of the scored-candidate pool.
        """
        now = time.time()
        # Check the timestamp, not the value: a legitimately empty knowledge store
        # (count = 0) would otherwise never get cached and every call
        # would re-hit SQLite. _corpus_size_ts is initialized to 0.0 in
        # __init__, so a stale/never-refreshed cache still falls through.
        if self._corpus_size_ts > 0 and (now - self._corpus_size_ts) < _CORPUS_SIZE_TTL:
            return self._corpus_size
        try:
            total = self.read_conn.execute(
                "SELECT COUNT(*) FROM genes"
            ).fetchone()[0]
            self._corpus_size = int(total) if total else 0
            self._corpus_size_ts = now
        except Exception:
            log.warning("corpus_size refresh failed", exc_info=True)
            # Fall back to whatever we have (may be 0 on first failure).
        return self._corpus_size

    @property
    def read_conn(self) -> sqlite3.Connection:
        """
        Per-thread read-only connection. In WAL mode, readers and writers
        don't block each other — but only if they use separate connections,
        and (perf slice 2) only if concurrent THREADS do too: two threads
        interleaving large scans on one shared connection livelock on the
        shared page cache + per-step connection mutex, wedging the whole
        server at c=2 (faulthandler-confirmed in co_activation expansion).

        Priority: persistence replica > this thread's reader > write
        connection (:memory: stores, where a second connection would be a
        different database). The init-time ``self._reader`` doubles as the
        creating thread's entry.
        """
        if self._replication_mgr is not None:
            try:
                return self._replication_mgr.get_reader()
            except Exception:
                pass
        if self._reader is not None:
            local = self._thread_readers
            conn = getattr(local, "conn", None)
            if conn is None:
                if threading.get_ident() == self._reader_owner_thread:
                    conn = self._reader
                else:
                    conn = self._open_reader_conn()
                    with self._thread_reader_registry_lock:
                        self._thread_reader_registry.append(conn)
                local.conn = conn
            return conn
        return self.conn

    def _open_reader_conn(self) -> sqlite3.Connection:
        """One mode=ro autocommit connection, pragmas mirroring _reader."""
        conn = sqlite3.connect(
            f"file:{self.path}?mode=ro", uri=True,
            check_same_thread=False, timeout=10,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        _plan = self._mem_plan if self._mem_plan is not None else sqlite_memory_budget(1)
        conn.execute(f"PRAGMA cache_size={_plan.reader_cache_size}")
        conn.execute(f"PRAGMA mmap_size={_plan.mmap_size}")
        return conn

    # ── Document ID (content-addressable) ───────────────────────────────

    @staticmethod
    def make_gene_id(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # ── Upsert ──────────────────────────────────────────────────────

    @contextmanager
    def deferred_commits(self):
        """Document-scoped write transaction (issue #372).

        Holds ``_write_lock`` for the whole scope and folds every write
        unit executed inside — strand upserts, tier demotions, the parent
        doc, relation/symbol batches — into ONE durable commit at scope
        exit: one fsync per document instead of one per strand (a 10-chunk
        file used to pay 12-22).

        Crash semantics: on any exception the WHOLE document rolls back.
        Partial-document rollback is acceptable and preferable to the
        partial-strand states per-strand commits leave behind (e.g. a
        committed gene with no CHUNK_OF edges) — the caller retries the
        document, never repairs half of one.

        Inner write units keep their own atomicity via SAVEPOINTs (see
        ``_write_unit``), so a swallowed inner failure — parent-doc
        creation, symbol-graph emission — can neither commit half a unit
        nor destroy the strands already staged.

        Reentrant: nested scopes join the outermost transaction. Holding
        ``_write_lock`` across the per-strand cache invalidations inside
        the scope is deadlock-free because no ``_dense_matrix_lock``
        holder ever acquires ``_write_lock`` (``_ensure_dense_matrix``
        reads without it).
        """
        if self.read_only:
            yield self  # upserts no-op in read_only; nothing to defer
            return
        with self._write_lock:
            outermost = self._defer_depth == 0
            if outermost and not self.conn.in_transaction:
                # Take SQLite's write lock up front (write_queue.GenomeWriter
                # precedent: one BEGIN IMMEDIATE per batch) and pin
                # conn.in_transaction so python-sqlite3 cannot auto-commit
                # between units. If the shared conn is already mid-
                # transaction we join it — the same hazard every commit on
                # this connection has today (bug bash 2026-07-18 notes).
                self.conn.execute("BEGIN IMMEDIATE")
            self._defer_depth += 1
            try:
                yield self
            except BaseException:
                if outermost:
                    self.conn.rollback()
                    self._pending_checkpoint = None
                raise
            else:
                if outermost:
                    self.conn.commit()
                    # Checkpoint cadence deferred by upsert_doc while the
                    # transaction was open — run it now that it can advance.
                    pending, self._pending_checkpoint = (
                        self._pending_checkpoint, None)
                    if pending:
                        self.checkpoint(pending)
            finally:
                self._defer_depth -= 1

    @contextmanager
    def _write_unit(self):
        """One atomic write unit on the shared writer connection.

        Must be entered with ``_write_lock`` held. Outside a
        ``deferred_commits()`` scope this is exactly the historical
        commit-per-call behavior: commit on success, rollback on failure
        (bug bash 2026-07-18 — a partial unit must never ride along with
        the next caller's commit). Inside a scope the unit becomes a
        SAVEPOINT: released on success, rolled back to on failure — unit
        atomicity is preserved while the durable commit (and its fsync)
        waits for scope exit.
        """
        if self._defer_depth > 0:
            self._savepoint_seq += 1
            name = f"cymatix_unit_{self._savepoint_seq}"
            self.conn.execute(f'SAVEPOINT "{name}"')
            try:
                yield
            except Exception:
                self.conn.execute(f'ROLLBACK TO "{name}"')
                self.conn.execute(f'RELEASE "{name}"')
                raise
            else:
                self.conn.execute(f'RELEASE "{name}"')
        else:
            try:
                yield
            except Exception:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()

    def upsert_doc(
        self,
        gene: Gene,
        apply_gate: bool = True,
        splade_sparse: Optional[Dict[str, float]] = None,
        embedding_dense_v2: Optional[List[float]] = None,
    ) -> str:
        """
        Insert or replace a document in the knowledge store.

        If ``apply_gate`` is True (the default), the density gate runs
        before storage and may override the document's lifecycle tier. Callers
        that have a reason to bypass the gate — cross-store import imports, benchmark
        setup scripts, explicit backfill tools, manual `compact_genome`
        re-runs — can pass ``apply_gate=False`` to preserve the incoming
        lifecycle tier as-is.

        ``embedding_dense_v2`` (Tier-0 PR-1, 2026-05-16) is an optional
        precomputed BGE-M3 dense vector (a ``dense_embedding_dim``-length
        float list). A batched caller — e.g. ``context_manager.ingest()`` —
        encodes all strands in one codec call and passes each vector here so
        per-document encoding is not repeated. When it is *not* supplied and
        ``dense_embed_on_ingest`` is true on this store, the vector is
        computed lazily via the store's BGE-M3 codec. When the knob is false
        and no vector is passed, the ``embedding_dense_v2`` column is left
        NULL (callers can backfill later via
        ``scripts/backfill_bgem3_v2.py``). This is purely the *write* path —
        dense recall during ``query_docs`` is gated separately on
        ``dense_embedding_enabled``.

        Returns the gene_id (content-addressed if not pre-populated).
        """
        gene_id = gene.gene_id or self.make_gene_id(gene.content)
        if self.read_only:
            log.debug("read_only: skipping upsert_doc")
            return gene_id

        # Struggle 1 fix: apply density gate at the storage boundary so
        # that bulk ingest scripts (ingest_steam.py, ingest_fdrive.py,
        # ingest_all.py) calling upsert_gene directly also respect the
        # gate. Previously the gate lived in context_manager.ingest() and
        # was bypassed by every bulk ingest path. See:
        #   scripts/simulate_density_gate_v2.py for the empirical basis
        #   (51.6% of the noise-diluted knowledge store demoted, >97% signal retained).
        #
        # Crucially, the gate only acts on documents arriving as OPEN — if the
        # caller has explicitly set EUCHROMATIN or HETEROCHROMATIN, we
        # trust that decision. This means cross-store import imports, test fixtures, and
        # any code that deliberately creates demoted documents retain their
        # intended state. The gate is admission-control, not state-reset.
        if apply_gate and gene.chromatin == ChromatinState.OPEN:
            new_state, reason = self.apply_density_gate(gene)
            if new_state != gene.chromatin:
                log.debug(
                    "Density gate demoted %s: OPEN -> %s (reason=%s)",
                    gene_id, new_state.name, reason,
                )
                gene.chromatin = new_state
        else:
            reason = "gate_bypassed" if not apply_gate else "explicit_chromatin_preserved"

        # Compute compression tier from final lifecycle tier
        tier = 0  # OPEN
        if gene.chromatin == ChromatinState.EUCHROMATIN:
            tier = 1
        elif gene.chromatin == ChromatinState.HETEROCHROMATIN:
            tier = 2

        observed_at = (
            gene.observed_at
            if gene.observed_at is not None
            else getattr(gene.epigenetics, "created_at", None)
        )
        content_hash = gene.content_hash
        if content_hash is None and gene.content:
            content_hash = hashlib.sha256(gene.content.encode("utf-8")).hexdigest()
        # Stage 7 (2026-05-08, spec §4) — wire ``last_verified_at`` on
        # ingest. The column has shipped since Stage 1 (DDL row :471,
        # ALTER row :499); Stage 7 only ensures the writer populates it.
        # Order: document-supplied > observed_at fallback > now(). Setting
        # to now() on legacy bare-inserts means the freshness gate has
        # something to compare mtime against on the very next turn,
        # rather than treating every newly-ingested document as
        # "freshness unknown" forever.
        last_verified_at = (
            gene.last_verified_at
            if gene.last_verified_at is not None
            else (observed_at if observed_at is not None else time.time())
        )

        # Tier-0 PR-1 (2026-05-16): compute the BGE-M3 dense vector inline so
        # every ingest path populates genes.embedding_dense_v2. Uses the
        # caller-supplied vector when present (batched ingest), else encodes
        # lazily iff dense_embed_on_ingest is set; NULL otherwise. Soft-fails.
        # Computed BEFORE the write lock so codec encoding (potentially a
        # model call) never serializes other writers.
        dense_v2_blob = self._encode_dense_v2_blob(
            gene.content, precomputed=embedding_dense_v2
        )

        # Bug bash 2026-07-18: the write connection is shared across threads
        # (check_same_thread=False, one implicit transaction per connection).
        # Hold the write lock for the whole multi-statement upsert so a
        # concurrent caller's commit can never publish this gene's
        # half-written state — and roll back on failure so a partial row is
        # not silently swept into the next caller's commit.
        with self._write_lock:
            cur = self.conn.cursor()
            # Issue #372: commit/rollback discipline lives in _write_unit —
            # commit-per-call standalone (unchanged behavior), SAVEPOINT
            # inside a deferred_commits() document scope.
            with self._write_unit():
                # Probe the existing row once: feeds the FTS5 delete-first
                # guard below (fresh inserts skip the unindexed FTS scan) and
                # makes cross-source content-hash collisions loud. Gene IDs
                # are content-addressed (see make_gene_id), so identical
                # content from two sources dedups to ONE row by design — but
                # the earlier source's provenance (source_id, and with it the
                # PR #288 document_id attribution) is replaced, which must
                # not happen silently.
                prior = cur.execute(
                    "SELECT source_id FROM genes WHERE gene_id = ?",
                    (gene_id,),
                ).fetchone()
                if (
                    prior is not None
                    and prior["source_id"]
                    and gene.source_id
                    and prior["source_id"] != gene.source_id
                ):
                    log.warning(
                        "gene %s: content-hash collision across sources — "
                        "provenance overwrite %r -> %r (identical content "
                        "dedups to one row; earlier source's attribution "
                        "and freshness are replaced)",
                        gene_id, prior["source_id"], gene.source_id,
                    )

                # Issue #338: on an external-content FTS index the delete
                # below must re-supply the PRIOR row's indexed values, and
                # the INSERT OR REPLACE + promoter rebuild are about to
                # replace both the genes rowid and the view text — capture
                # the row now, while the view still reproduces what was
                # indexed.
                prior_fts_row = None
                if self._fts_available and self._fts_external and prior is not None:
                    from .storage.indexes import fetch_fts_source_row
                    prior_fts_row = fetch_fts_source_row(cur, gene_id)

                cur.execute(
                    "INSERT OR REPLACE INTO genes "
                    "(gene_id, content, complement, codons, promoter, epigenetics, "
                    "chromatin, is_fragment, embedding, source_id, repo_root, source_kind, "
                    "observed_at, mtime, content_hash, volatility_class, authority_class, "
                    "support_span, last_verified_at, version, supersedes, key_values, "
                    "compression_tier, last_seen, embedding_dense_v2) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        gene_id,
                        gene.content,
                        gene.complement,
                        json_dumps(gene.codons),
                        gene.promoter.model_dump_json(),
                        gene.epigenetics.model_dump_json(),
                        int(gene.chromatin),
                        int(gene.is_fragment),
                        sema_vec_to_blob(gene.embedding) if gene.embedding else None,
                        gene.source_id,
                        gene.repo_root,
                        gene.source_kind,
                        observed_at,
                        gene.mtime,
                        content_hash,
                        gene.volatility_class,
                        gene.authority_class,
                        gene.support_span,
                        last_verified_at,
                        gene.version,
                        gene.supersedes,
                        json_dumps(gene.key_values) if gene.key_values else None,
                        tier,
                        time.time(),  # last_seen: always stamp current epoch on every upsert
                        # Tier-0 PR-1: raw little-endian fp32 BGE-M3 vector, or NULL.
                        sqlite3.Binary(dense_v2_blob) if dense_v2_blob is not None else None,
                    ),
                )
                # Invalidate parse cache for this document's promoter/epigenetics
                clear_parse_caches()

                # ── Index population (delegated to storage.indexes) ──────────
                from .storage.indexes import (
                    rebuild_promoter_index,
                    sync_fts5,
                    sync_entity_graph,
                    sync_path_key_index,
                    sync_filename_index,
                    sync_splade_index,
                    resolve_splade_enabled,
                )
                rebuild_promoter_index(cur, gene_id, gene)
                sync_fts5(
                    cur, gene_id, gene, self._fts_available,
                    delete_first=prior is not None,
                    fts_external=self._fts_external,
                    prior_fts_row=prior_fts_row,
                )
                sync_entity_graph(cur, gene_id, gene, self._entity_graph_enabled)
                sync_path_key_index(cur, gene_id, gene)
                sync_filename_index(cur, gene_id, gene.source_id)
                # Issue #164: size-aware SPLADE auto-toggle. Refresh the cached
                # gene_count whenever EITHER threshold is opted in; otherwise
                # the resolver short-circuits to the static flag and the count
                # is unused.
                if self._splade_auto_enable_below > 0 or self._splade_auto_disable_above > 0:
                    try:
                        self._splade_auto_cached_count = int(
                            cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
                        )
                    except sqlite3.Error:
                        self._splade_auto_cached_count = 0
                effective_splade = resolve_splade_enabled(
                    splade_enabled=self._splade_enabled,
                    current_gene_count=self._splade_auto_cached_count,
                    auto_enable_below=self._splade_auto_enable_below,
                    auto_disable_above=self._splade_auto_disable_above,
                )
                sync_splade_index(
                    cur, gene_id, gene.content, effective_splade,
                    splade_sparse=splade_sparse,
                    content_cap=self._splade_content_cap,
                    model_name=self._splade_model,
                )
                # _write_unit commits on exit (document + tags + FTS5 +
                # entity graph + SPLADE, single atomic commit) and rolls
                # back on failure so a partial unit cannot ride along with
                # the next caller's commit (bug bash 2026-07-18).

        # Periodic WAL checkpoint to prevent data loss on crash
        # PASSIVE every 50 documents (~non-blocking), TRUNCATE every 500 (resets WAL)
        self._upsert_count += 1
        _ckpt_mode: Optional[str] = None
        if self._upsert_count % 500 == 0:
            _ckpt_mode = "TRUNCATE"
        elif self._upsert_count % 50 == 0:
            _ckpt_mode = "PASSIVE"
        if _ckpt_mode is not None:
            # Re-acquire the (reentrant) write lock: _defer_depth and
            # _pending_checkpoint are only touched with it held. checkpoint()
            # takes the same lock internally, so lock scope is unchanged for
            # the standalone path.
            with self._write_lock:
                if self._defer_depth > 0:
                    # Issue #372: inside a document transaction wal_checkpoint
                    # cannot advance (this connection holds the write txn) —
                    # stash the strongest requested mode and run it right
                    # after the document commits (deferred_commits scope exit).
                    if self._pending_checkpoint != "TRUNCATE":
                        self._pending_checkpoint = _ckpt_mode
                else:
                    self.checkpoint(_ckpt_mode)

        # Invalidate ΣĒMA caches (new document may have embedding, and
        # lifecycle tier changes can reshuffle hot/cold tier membership).
        # Routed through the invalidate_* methods (not a bare None-assign)
        # so the sema-vectorless auto-gate flags reset too — a bed that
        # just gained a vector must re-check on its next cache build.
        self.invalidate_sema_cache()
        self.invalidate_cold_sema_cache()
        # Stage 2: invalidate the in-memory dense matrix. Rebuild is full
        # (~78 MiB / sub-200 ms at 18.9k rows) on first dense recall after
        # this batch. See spec §4 "Invalidation".
        self._invalidate_dense_matrix()

        # Notify persistence manager (if attached)
        if self._replication_mgr is not None:
            self._replication_mgr.notify_write()

        # Phase 2 claims hook: emit literal claims into main.db if wired.
        # Soft-fail — ingest should never break because of claim extraction.
        if self._main_conn is not None:
            try:
                from .identity.claims import extract_literal_claims, persist_claims
                claims = extract_literal_claims(gene, shard_name=self._shard_name)
                if claims:
                    persist_claims(self._main_conn, claims)
                    # Edge detection scoped to the new claims' entity_keys.
                    # Narrow scan keeps per-ingest cost bounded; groups
                    # larger than max_group_size are skipped anyway.
                    from .identity.claims_analyze import detect_and_persist_edges
                    touched_keys = {
                        c.entity_key for c in claims if c.entity_key
                    }
                    if touched_keys:
                        detect_and_persist_edges(
                            self._main_conn,
                            entity_keys=touched_keys,
                        )
            except Exception:
                log.warning("claim extraction failed at ingest", exc_info=True)

        return gene_id

    # ── Fix 1: synonym expansion ────────────────────────────────────

    def _expand_terms(self, terms: List[str]) -> List[str]:
        expanded = set(t.lower() for t in terms)
        for t in terms:
            key = t.lower()
            if key in self.synonym_map:
                expanded.update(self.synonym_map[key])
        # ``sorted`` rather than ``list(set(...))`` so iteration order
        # is independent of PYTHONHASHSEED. Downstream SQL parameter
        # ordering and rank tiebreaks become deterministic across
        # processes (matters when bench replays the same query in a
        # freshly-spawned uvicorn — see #131).
        return sorted(expanded)

    # ── Authority boosts: distinguish "about X" from "mentions X" ──

    def _apply_authority_boosts(
        self,
        cur,
        gene_scores: Dict[str, float],
        query_terms: List[str],
        rerank_additive: Optional[Dict[str, float]] = None,
        tier_contrib: Optional[Dict[str, Dict[str, float]]] = None,
        rerank_by_class: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """
        Post-rank boosts that distinguish authoritative documents from tangential ones.

        Three signals:
          1. Source authority (+2.0): query term in source_id path
             — a file named BENCHMARK_NOTES.md answering "benchmark" is authoritative
          2. Domain primacy (+1.5): query term in top-3 tags domains
             — primary domains = what the document is ABOUT, not mentions
          3. Creation recency (+0.5): document created in last 48 hours
             — bootstraps new concepts before they build co-activation history

        All boosts are additive to existing scores. Low risk — only raises
        the ceiling on already-scored documents, never adds new candidates.
        """
        if not gene_scores:
            return

        import time as _time
        now = _time.time()
        recency_window = 48 * 3600  # 48 hours in seconds

        gene_ids = list(gene_scores.keys())
        lower_terms = [t.lower() for t in query_terms]

        # #327: the source-authority signal below matches query terms against
        # the whole source_id, which includes the ingest root and repository
        # directory. A query naming the project therefore matches EVERY gene in
        # it: measured on the dogfood bed, "cymatix" hits 88.0% of paths while
        # "fusion" hits 0.8%. The 88% case is a constant, not a signal — it
        # saturates the score (4.00 of a ~4.15 total, identical across the top
        # six) and leaves the retrieval tiers to fight over the remainder.
        #
        # `authority_path_selectivity` scales the boost by how selective the
        # matching term actually is, so a term matching most of the corpus
        # contributes ~nothing and a rare one contributes fully. This is
        # self-limiting: it needs no stopword list and also defuses generic
        # path tokens like "test", "src" and "docs".
        #
        # Default OFF — flipping it is a scoring change and stays bench-gated.
        _selectivity_on = bool(getattr(self, "_authority_path_selectivity", False))
        _sel_cache: Dict[str, float] = {}

        def _term_selectivity(term: str) -> float:
            """1.0 when a term is rare across source paths, ->0 when ubiquitous.

            Computed once per term per query against the genes table, so the
            cost is one COUNT per distinct term rather than per candidate.
            """
            if term in _sel_cache:
                return _sel_cache[term]
            weight = 1.0
            try:
                total = self.read_conn.execute(
                    "SELECT COUNT(*) FROM genes"
                ).fetchone()[0] or 0
                if total > 0:
                    hits = self.read_conn.execute(
                        "SELECT COUNT(*) FROM genes WHERE lower(source_id) LIKE ?",
                        (f"%{term}%",),
                    ).fetchone()[0] or 0
                    frac = hits / total
                    # Linear falloff: full weight below 1% of paths, zero at
                    # 50% or above. Deliberately not IDF's log curve — we want
                    # a hard floor on corpus-wide terms, not a long tail.
                    weight = 0.0 if frac >= 0.5 else max(0.0, 1.0 - (frac / 0.5))
            except Exception:
                weight = 1.0
            _sel_cache[term] = weight
            return weight

        # Fetch source_id, tags, signals for all candidates. Batched to stay
        # under SQLite's IN-clause placeholder cap — see _iter_in_batches.
        rows = []
        for _ph, _batch in _iter_in_batches(gene_ids):
            rows.extend(cur.execute(
                f"SELECT gene_id, source_id, promoter, epigenetics "
                f"FROM genes WHERE gene_id IN ({_ph})",
                _batch,
            ).fetchall())

        for r in rows:
            gid = r["gene_id"]
            boost = 0.0

            # 1. Source authority: query term in path
            source = (r["source_id"] or "").lower()
            if source:
                matched = [t for t in lower_terms if t in source]
                if matched:
                    if _selectivity_on:
                        # Credit the most selective term that matched, rather
                        # than summing: a path matching both "cymatix" and
                        # "fusion" is authoritative because of "fusion".
                        boost += 2.0 * max(_term_selectivity(t) for t in matched)
                    else:
                        boost += 2.0

            # 2. Domain primacy: query term in top-3 tags domains
            try:
                prom = parse_promoter(r["promoter"]) if r["promoter"] else None
                if prom and prom.domains:
                    primary_domains = {d.lower() for d in prom.domains[:3]}
                    if any(t in primary_domains for t in lower_terms):
                        boost += 1.5
            except Exception:
                pass

            # 3. Creation recency: document created in last 48h
            try:
                epi = parse_epigenetics(r["epigenetics"]) if r["epigenetics"] else None
                if epi and epi.created_at > 0:
                    age = now - epi.created_at
                    if 0 < age < recency_window:
                        boost += 0.5
            except Exception:
                pass

            if boost > 0:
                gene_scores[gid] += boost
                # Stage 3: dual-write to the post-fusion additives map
                # so RRF mode can apply authority on top of the fused
                # score (per spec §3 / §99: re-rank class).
                if rerank_additive is not None:
                    rerank_additive[gid] = rerank_additive.get(gid, 0.0) + boost
                # Issue #255 (PR-2): mirror into the per-class map under the
                # "authority" key (matches the tier_contrib label below) so
                # the fused_tier combinator can rank authority independently.
                if rerank_by_class is not None:
                    _auth = rerank_by_class.setdefault("authority", {})
                    _auth[gid] = _auth.get(gid, 0.0) + boost
                if tier_contrib is not None:
                    tier_contrib.setdefault(gid, {})["authority"] = (
                        tier_contrib.get(gid, {}).get("authority", 0.0) + boost
                    )

    # ── Core retrieval (Step 2) — hybrid tags + FTS5 ────────────

    def fts_doc_count(self) -> int:
        """Total document count in this knowledge store's FTS5 index.

        Used by ShardRouter for cross-shard BM25 IDF normalization (#118).
        Returns 0 if FTS is unavailable.
        """
        if not self._fts_available:
            return 0
        try:
            row = self.read_conn.execute(
                "SELECT COUNT(*) FROM genes_fts"
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def term_doc_frequencies(self, terms: list[str]) -> Dict[str, int]:
        """Document frequency in this shard's FTS5 index, per term.

        Returns a dict ``{term: df}``. Missing/unknown terms map to 0.

        Used by ShardRouter to compute global IDFs for cross-shard
        BM25 normalization (#118). Implements one ``MATCH "term"`` COUNT
        per requested term — fine at query time since the number of
        query terms is bounded (typically <20) and SQLite's FTS5
        ``MATCH`` is fast on a single-token exact phrase.
        """
        result: Dict[str, int] = {t: 0 for t in terms}
        if not self._fts_available or not terms:
            return result
        cur = self.read_conn.cursor()
        try:
            for term in terms:
                # Skip terms that are too short to be FTS-indexed (mirrors
                # the >2 filter used elsewhere in query_docs).
                if not term or len(term) <= 2:
                    continue
                # Quote the term to disable FTS5 syntax interpretation —
                # treats hyphens, colons, etc. as literal characters.
                # Double any embedded quotes per FTS5 escape rules.
                escaped = term.replace('"', '""')
                try:
                    row = cur.execute(
                        'SELECT COUNT(*) FROM genes_fts WHERE genes_fts MATCH ?',
                        (f'"{escaped}"',),
                    ).fetchone()
                    if row:
                        result[term] = int(row[0])
                except Exception:
                    # Single-term FTS errors (e.g., malformed token) shouldn't
                    # poison the whole batch. df=0 is a safe default.
                    continue
        finally:
            cur.close()
        return result

    def _ensure_fts_vocab(self) -> None:
        """Best-effort create of the ``genes_fts_vocab`` shadow table.

        Created at DB init (storage.ddl) for new stores; this covers
        LEGACY shard .db files that predate the #182 change. Uses the
        WRITABLE connection (the read connection is read-only and a
        ``CREATE VIRTUAL TABLE`` against it raises). Idempotent and
        soft-failing — on any error the rescore probe falls back to the
        scalar path.
        """
        if not self._fts_available:
            return
        try:
            # W2.3 Phase A: DDL + commit on the shared writer — lock it.
            with self._write_lock:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS genes_fts_vocab "
                    "USING fts5vocab(genes_fts, instance)"
                )
                self.conn.commit()
        except Exception:
            # Read-only-DB or fts5vocab-unavailable build — non-fatal.
            log.debug(
                "global-idf rescore: could not ensure genes_fts_vocab",
                exc_info=True,
            )

    def rescore_lexical_global_idf(
        self,
        candidate_ids: list[str],
        query_terms: list[str],
        global_idf_map: Dict[str, float],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> Dict[str, float]:
        """Recompute each candidate's BM25 LEXICAL score with injected global IDF.

        Cross-shard global-IDF re-score (CYMATIX_SHARD_GLOBAL_IDF, #182).

        SQLite FTS5's built-in ``bm25()`` (the ``rank`` column used in
        :meth:`query_docs`'s Tier-3 FTS5 block) computes IDF over THIS
        shard's corpus only. On sharded retrieval that corpus-LOCAL IDF
        sets the *within-shard* order of two docs, and a per-shard scalar
        (``m_shard``) cannot reorder docs inside one shard. This method
        recomputes the Okapi-BM25 lexical score per candidate doc using:

            * per-doc term frequency ``tf(t,d)`` and doc length ``dl(d)``
              read from the ``fts5vocab(..., instance)`` shadow table of
              this shard's ``genes_fts`` (the SAME content this shard's
              ``bm25()`` scored), and
            * the INJECTED ``global_idf_map`` (computed by the router from
              the aggregated global N and global df over all shards) in
              place of the local IDF.

        Formula (BIT-EXACT with SQLite FTS5's ``bm25()``, k1=1.2, b=0.75)::

            dl(d)    = total instance rows for d over ALL FTS columns
            avgdl    = (Σ_d dl(d)) / N_local
            tf(t,d)  = total occurrences of t in d over ALL FTS columns
            score(d) = Σ_t  idf_global(t) ·
                            tf(t,d)·(k1+1) /
                            (tf(t,d) + k1·(1 - b + b·dl(d)/avgdl))

        where ``idf_global(t)`` is the RAW (unsmoothed) FTS5 IDF
        ``ln((N - n + 0.5) / (n + 0.5))`` injected via ``global_idf_map``
        with FTS5's non-positive clamp already applied. dl/tf/avgdl are
        counted over EVERY indexed column (gene_id, content, complement),
        because FTS5's bm25() treats the row as one bag-of-tokens spanning
        all columns with unit weights — the same columns the bare
        ``genes_fts MATCH ?`` in :meth:`query_docs` searches. Empirically
        verified bit-exact (|Δ| ≈ 1e-15) against ``SELECT bm25(genes_fts)``
        on real shards; see ``benchmarks/diag_global_idf_counterfactual.py``.

        The returned magnitude is on the SAME signed "bigger = better"
        scale as the ``-rank`` value the genome feeds into
        ``last_query_scores`` (the genome caps it at 6.0; the router
        applies the cap when it splices this back into the fused score, so
        this raw value is uncapped).

        Returns:
            ``{gene_id: global_idf_bm25_lexical_score}`` for every
            candidate that appears in this shard's FTS index. Candidates
            with no scored-term match get 0.0. Returns ``{}`` if FTS is
            unavailable or no usable terms/candidates were supplied.
        """
        if not self._fts_available or not candidate_ids or not query_terms:
            return {}
        # Score only terms long enough to be FTS-relevant AND that have a
        # global IDF (router only injects IDF for the terms it probed).
        scored_terms = [
            t.lower() for t in query_terms
            if t and len(t) > 2 and t.lower() in global_idf_map
        ]
        if not scored_terms:
            return {}

        cand_set = set(candidate_ids)
        # The fts5vocab "instance" shadow table (``genes_fts_vocab``) is
        # created at DB init alongside ``genes_fts`` (storage.ddl). It is a
        # zero-storage view: "instance" yields one row per
        # (term, doc, col, offset) occurrence — counting rows per
        # (doc, term) gives per-doc term frequency, and counting all
        # rows per doc gives that doc's length in tokens.
        self._ensure_fts_vocab()
        cur = self.read_conn.cursor()
        try:
            # Probe that the vocab table is queryable; if a legacy DB
            # predates it and we couldn't create it (read-only), soft-fail
            # to the scalar path.
            try:
                cur.execute("SELECT term FROM genes_fts_vocab LIMIT 1")
            except Exception:
                log.warning(
                    "global-idf rescore: genes_fts_vocab unavailable; "
                    "falling back to scalar path",
                    exc_info=True,
                )
                return {}

            # Map this shard's FTS rowid -> gene_id for the candidate set.
            # fts5vocab reports the integer 'doc' (= rowid), so we resolve
            # candidate gene_ids to rowids and back.
            id_ph = ",".join("?" * len(cand_set))
            rowid_rows = cur.execute(
                f"SELECT rowid, gene_id FROM genes_fts "
                f"WHERE gene_id IN ({id_ph})",
                tuple(cand_set),
            ).fetchall()
            rowid_to_gid: Dict[int, str] = {
                int(r["rowid"]): r["gene_id"] for r in rowid_rows
            }
            if not rowid_to_gid:
                return {gid: 0.0 for gid in candidate_ids}
            cand_rowids = set(rowid_to_gid)

            # Doc length per candidate doc = count of instance rows for
            # that doc across ALL FTS columns. SQLite FTS5's bm25()
            # measures D (doc length) as the total token count over every
            # indexed column (here: gene_id, content, complement), with a
            # single corpus-wide avgdl = (Σ all-column tokens) / N. Counting
            # only ``col='content'`` understates both D and avgdl and breaks
            # bit-exact parity with the engine's bm25() — see the parity
            # probe in benchmarks/diag_global_idf_counterfactual.py.
            n_local = self.fts_doc_count() or 0
            avgdl = 0.0
            try:
                tot = cur.execute(
                    "SELECT COUNT(*) FROM genes_fts_vocab"
                ).fetchone()
                total_tokens = int(tot[0]) if tot else 0
                if n_local > 0:
                    avgdl = total_tokens / float(n_local)
            except Exception:
                avgdl = 0.0
            if avgdl <= 0.0:
                avgdl = 1.0  # degenerate corpus; avoid divide-by-zero

            # Per-candidate doc length (ALL columns — matches FTS5's D).
            doc_len: Dict[int, int] = {rid: 0 for rid in cand_rowids}
            try:
                dl_rows = cur.execute(
                    "SELECT doc, COUNT(*) AS dl FROM genes_fts_vocab "
                    "GROUP BY doc"
                ).fetchall()
                for r in dl_rows:
                    rid = int(r["doc"])
                    if rid in doc_len:
                        doc_len[rid] = int(r["dl"])
            except Exception:
                log.warning(
                    "global-idf rescore: doc-length probe failed",
                    exc_info=True,
                )
                return {}

            # Per (term, candidate-doc) term frequency: count instance rows
            # across ALL columns (FTS5's tf is the total occurrence count
            # over every indexed column). One small query per scored term
            # (bounded by the query length).
            #
            # NOTE on the injected ``g_idf`` scale: the router computes it
            # with SQLite FTS5's RAW Robertson-Sparck-Jones IDF
            # ``ln((N - n + 0.5) / (n + 0.5))`` AND FTS5's own non-positive
            # clamp (``idf <= 0 -> 1e-6``), so a globally-ubiquitous term
            # contributes ~nothing rather than a large negative — exactly
            # what the engine's bm25() does. We do NOT re-gate on
            # ``g_idf > 0`` here: the value already carries the engine's
            # clamp, and the OLD code's combination of the SMOOTHED
            # always-positive IDF + a local ``g_idf > 0`` gate is precisely
            # what put the lexical sub-score on the wrong scale and drove
            # the #182 recall regression (0.347 -> 0.168).
            scores: Dict[str, float] = {
                rowid_to_gid[rid]: 0.0 for rid in cand_rowids
            }
            for term in scored_terms:
                if term not in global_idf_map:
                    continue
                g_idf = float(global_idf_map[term])
                escaped_doc_ph = ",".join("?" * len(cand_rowids))
                try:
                    tf_rows = cur.execute(
                        f"SELECT doc, COUNT(*) AS tf FROM genes_fts_vocab "
                        f"WHERE term = ? "
                        f"AND doc IN ({escaped_doc_ph}) "
                        f"GROUP BY doc",
                        (term, *cand_rowids),
                    ).fetchall()
                except Exception:
                    continue
                for r in tf_rows:
                    rid = int(r["doc"])
                    tf = float(r["tf"])
                    if tf <= 0.0 or rid not in rowid_to_gid:
                        continue
                    dl = float(doc_len.get(rid, 0)) or 1.0
                    denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                    if denom <= 0.0:
                        continue
                    contrib = g_idf * (tf * (k1 + 1.0)) / denom
                    scores[rowid_to_gid[rid]] += contrib

            # Candidates absent from the FTS index keep an implicit 0.0.
            for gid in candidate_ids:
                scores.setdefault(gid, 0.0)
            return scores
        except Exception:
            log.warning(
                "global-idf rescore failed; router falls back to scalar path",
                exc_info=True,
            )
            return {}
        finally:
            cur.close()

    def _bm25_candidate_set(self, query_terms: list[str], size: int) -> set[str] | None:
        """Return FTS5 BM25 top-N gene_ids, or None if FTS unavailable/empty. Soft-fails to None."""
        if not self._fts_available:
            return None
        bm25_terms = [t for t in query_terms if len(t) > 2]
        if not bm25_terms:
            return None
        cur = self.read_conn.cursor()
        try:
            bm25_match = " OR ".join(f'"{t.replace(chr(34), chr(34)*2)}"' for t in bm25_terms)
            rows = cur.execute(
                "SELECT gene_id FROM genes_fts WHERE genes_fts MATCH ? ORDER BY rank LIMIT ?",
                (bm25_match, size),
            ).fetchall()
            if not rows:
                return None
            return {r["gene_id"] for r in rows}
        except Exception:
            log.warning("BM25 pre-filter failed — falling back to full corpus", exc_info=True)
            return None
        finally:
            cur.close()

    def _tag_prefix_sql(
        self,
        query_terms: List[str],
        party_filter: str = "",
        party_params: tuple = (),
        prefilter_clause: str = "",
        prefilter_params: tuple = (),
    ) -> Tuple[str, tuple]:
        """Build the Tier-2 prefix-tag query (SQL, params).

        2026-08-03 ERB fix: the legacy form (genes JOIN promoter_index
        WHERE OR-of-LIKEs) planned as SCAN genes + one idx_promoter_gene
        probe per gene, evaluating every LIKE against every tag — 35-44s
        at 829k genes, and ANALYZE does NOT flip the plan. This form
        drives one covering-index range scan per prefix (UNION ALL keeps
        the per-tag multiplicity COUNT depends on) and joins genes only
        for the matched ids. Semantics pinned by
        test_tag_prefix_semantics_pinned.
        """
        # Range predicates, NOT LIKE: with the default
        # case_sensitive_like=OFF, LIKE cannot use a BINARY-collation
        # index and each branch degraded to a full covering-index SCAN
        # (6.2s at 100k genes / 2.4M tag rows). tag_value >= lo AND
        # tag_value < hi is a true index range SEARCH. Tags are
        # ingest-normalized to lowercase (verified 0 mixed-case rows on
        # dogfood and every ERB bed), so lowercasing the query term
        # preserves LIKE's case-insensitive matching.
        sub = " UNION ALL ".join(
            "SELECT gene_id, tag_value FROM promoter_index "
            "WHERE tag_value >= ? AND tag_value < ?"
            for _ in query_terms
        )
        # CROSS JOIN is SQLite's documented ordering hint: it pins the
        # aggregated tag matches as the outer loop so genes is one PK
        # lookup per MATCHED id. A plain JOIN let the planner flatten and
        # drive from SCAN genes again (caught by
        # test_tag_prefix_drives_promoter_index).
        sql = (
            f"SELECT g.gene_id, p.match_count AS match_count "
            f"FROM (SELECT gene_id, COUNT(tag_value) AS match_count "
            f"      FROM ({sub}) GROUP BY gene_id) p "
            f"CROSS JOIN genes g ON g.gene_id = p.gene_id "
            f"WHERE g.chromatin < ? {party_filter} {prefilter_clause}"
        )
        bounds: List[str] = []
        for t in query_terms:
            lo = t.lower()
            # "next prefix" upper bound: increment the last char so every
            # suffix (including astral-plane chars) stays inside the range.
            hi = lo[:-1] + chr(ord(lo[-1]) + 1) if lo else "￿"
            bounds.extend((lo, hi))
        params = (
            *bounds,
            int(ChromatinState.HETEROCHROMATIN),
            *party_params,
            *prefilter_params,
        )
        return sql, params

    # ── Issue #341: pre-cap cross-encoder rerank helpers ─────────────────

    def _rerank_effective(self, override: Optional[bool]) -> bool:
        """Resolve the rerank flag for ONE call.

        ``None`` (the default) defers to the store's configured global; an
        explicit bool wins in both directions, so a per-query-class override
        (``[retrieval] rerank_enabled_by_class``, resolved upstream) can turn
        rerank on for a class the global leaves off — and off for a class the
        global leaves on.
        """
        return self._rerank_enabled if override is None else bool(override)

    def _score_rerank(
        self, query_text: str, docs: "List[Gene]"
    ) -> "Optional[List[float]]":
        """Score ``docs`` against ``query_text``. ``None`` = fall back.

        Never raises. A missing model, an OOM, a torch import error, or a
        scorer that returns garbage all degrade to the caller's existing
        gate/fused order — retrieval keeps working, one WARNING gets logged,
        and the count invariant is untouched because the caller only ever
        *reorders* with the returned scores.

        The ``rerank_backend`` import is deliberately inside this method:
        importing ``knowledge_store`` must never pull torch/transformers into
        the process of a store that isn't reranking.
        """
        try:
            texts = [_rerank_doc_text(d) for d in docs]
            scorer = self._rerank_scorer
            if scorer is None:
                from .backends.rerank_backend import score_pairs
                scores = score_pairs(query_text, texts, model_name=self._rerank_model)
            else:
                scores = scorer(query_text, texts)
            if not isinstance(scores, list) or len(scores) != len(docs):
                log.warning(
                    "rerank scorer returned unusable payload "
                    "(type=%s, len=%s, expected %d); falling back",
                    type(scores).__name__,
                    len(scores) if hasattr(scores, "__len__") else "n/a",
                    len(docs),
                )
                return None
            return [float(s) for s in scores]
        except Exception:
            log.warning(
                "rerank scoring failed; falling back to gate order", exc_info=True
            )
            return None

    @trace_retrieval
    def query_docs(
        self,
        domains: List[str],
        entities: List[str],
        max_genes: int = 8,
        party_id: Optional[str] = None,
        use_harmonic: bool = True,
        use_sr: Optional[bool] = None,
        use_entity_graph: Optional[bool] = None,
        read_only: bool = False,
        query_type: Optional[str] = None,
        rerank_combinator: Optional[str] = None,
        query_text: Optional[str] = None,
        rerank_override: Optional[bool] = None,
    ) -> List[Gene]:
        """
        Find documents matching the given tags signals.

        ``query_text`` / ``rerank_override`` (issue #341, pre-cap
        cross-encoder rerank): when rerank is effective for this call AND
        ``query_text`` is supplied, the post-fusion ranking is widened to
        ``max(limit, rerank_depth)``, the head of that widened order is
        re-scored by a cross-encoder against ``query_text``, and the result is
        cut back to ``limit``. The cut — and therefore the returned document
        COUNT — is unchanged; only which documents fill the slots moves.
        ``query_text`` is a separate parameter (not derived from
        ``domains``/``entities``) because the cross-encoder wants the user's
        raw question, not the expanded keyword bag. Omitting it disables the
        seam entirely, which is how ``query_docs_ann`` suppresses a double
        rerank on its inner lex-pool call.

        ``rerank_combinator`` (issue #255, classifier-gated combinator): a
        per-query override selecting the post-fusion rerank combinator for THIS
        call only (fusion_mode=="rrf"). ``None`` (default) falls back to the
        store's configured global ``self._rerank_combinator``, so the default
        path is byte-identical. context_manager derives it from the stage-0
        query classifier's class via the ``[retrieval] rerank_combinator_by_class``
        map; on the sharded path it rides ``**kwargs`` to each shard's
        ``query_docs``. Values are validated at config load.

        ``query_type`` (semantic-wiring arm, PRD 2026-06-02): when "semantic"
        AND env CYMATIX_SEMANTIC_ARM=1, the additive-mode dense term is scaled by
        ``self._semantic_dense_additive_weight`` instead of the default
        ``self._dense_additive_weight``. All other tiers are untouched. Any
        other value (or arm off) is byte-identical to the prior behaviour.

        Multi-tier retrieval:
            1. Exact tag match (highest confidence)
            2. Prefix tag match — "server" matches "serverconfig" (medium)
            3. FTS5 content search — searches document text directly (fallback)
            3.5 SPLADE sparse retrieval
            4. SEMA semantic retrieval + re-ranking
            5. Harmonic co-activation boost (mutual reinforcement)
            Tiebreaker: access-rate bonus for equal-scored documents

        When party_id is provided, Tiers 1-3 exclude documents attributed
        to OTHER parties (cross-party leakage prevention). Documents with
        NO attribution row (legacy ingests, bridge inbox drops without
        a participant_id) remain retrievable — without this fallback,
        retrieval on an unattributed legacy knowledge store would collapse to
        ~0 hits. Attributed-to-this-party documents do NOT get a retrieval
        bonus here — that is a separate concern handled at a higher
        layer (see roadmap Phase 2c).

        Results are merged with weighted scoring, then expanded via
        co-activation pull-forward. Returns up to max_genes * 2 candidates.
        """
        _measurement = current_retrieval()
        domains = self._expand_terms(domains)
        entities = self._expand_terms(entities)

        query_terms = domains + entities
        if not query_terms:
            raise PromoterMismatch("No query terms after expansion")

        # Perf slice 1 (W0.2): per-signal wall-time collector. Every
        # genome-signal histogram observation is mirrored into a per-query
        # {signal: ms} map, published as self.last_signal_timings on return
        # so the pipeline ring / bench receipts can read sub-stage detail
        # over HTTP without an OTel collector. Cleared up front so a query
        # that raises (e.g. zero tier matches) never leaves the previous
        # query's map behind.
        _sig_ms: Dict[str, float] = {}
        with self._last_query_scores_lock:
            self.last_signal_timings = {}

        def _sig(name: str, t0: float) -> None:
            _dt = time.monotonic() - t0
            _sig_ms[name] = _sig_ms.get(name, 0.0) + _dt * 1000.0
            try:
                from .telemetry import genome_signal_histogram
                genome_signal_histogram().record(_dt, {"signal": name})
            except Exception:
                pass

        self._refresh_snapshot()  # See latest WAL state (external thinning, deletes)
        cur = self.read_conn.cursor()  # Read path — avoids WAL lock contention
        limit = max_genes * 2
        # A4 / #205: Tier-3 FTS content-search fetch depth. Override (config
        # [retrieval] fts5_candidate_depth) widens ONLY the raw candidate pool
        # pulled into tier scoring; the returned pool and delivery cap below
        # are untouched, so a deeper fetch cannot inflate gold_delivered on its
        # own — only lets a starved gold document ENTER scoring where the tiers
        # can float it into the delivered top-K.
        _fts_fetch_depth = getattr(self, "_fts5_candidate_depth", 0) or (limit * 2)

        # ── BM25 pre-filter (tier-0, 2026-05-08 upgrade) ──────────────
        _prefilter_set: set[str] | None = None
        if self._bm25_prefilter_enabled:
            _prefilter_set = self._bm25_candidate_set(query_terms, self._bm25_prefilter_size)
            log.debug("bm25 prefilter: size=%d result=%s", self._bm25_prefilter_size,
                      len(_prefilter_set) if _prefilter_set is not None else "fallback")

        _prefilter_aliased_clause = ""   # for Tier 1/2 using g.gene_id alias
        _prefilter_bare_clause = ""      # for Tier 3/3.5 without alias
        _prefilter_params: list = []
        if _prefilter_set is not None:
            _prefilter_ids = list(_prefilter_set)[:998]  # SQLite variable limit is 999
            ph = ",".join("?" * len(_prefilter_ids))
            _prefilter_aliased_clause = f" AND g.gene_id IN ({ph})"
            _prefilter_bare_clause = f" AND gene_id IN ({ph})"
            _prefilter_params = _prefilter_ids

        # Document scores: gene_id → float (accumulated across tiers)
        gene_scores: Dict[str, float] = {}

        # Per-tier contribution tracking (parallel to gene_scores).
        # Each accumulation point also writes the contribution to
        # tier_contrib[gid][tier_name]. Surfaced via last_tier_contributions
        # for the activation profiler bench (bench_skill_activation.py).
        tier_contrib: Dict[str, Dict[str, float]] = {}

        # ── Stage 3: Reciprocal Rank Fusion accumulator ────────────
        # Spec: docs/specs/2026-05-08-stage-3-rrf-fusion.md.
        # In "rrf" mode, every recall/discovery tier ALSO calls
        # fuser.add_tier(...) with its (gid, raw_score) ranked list and
        # the operator-tunable post-multiplier weight. The final sort
        # branches on _fusion_mode at line ~2489.
        #
        # In "additive" mode, the Fuser is built but never queried —
        # the existing gene_scores accumulator runs unchanged so the
        # output is byte-identical to pre-Stage-3 ranking.
        #
        # Re-rank/tiebreaker tiers (sema_boost, authority, party_attr,
        # access_rate) stay ADDITIVE on top of the fused score. Per
        # spec §3 rule of thumb: agreement across recall tiers is the
        # signal RRF retrieves; "is this document authoritative?" is a
        # different question that survives unchanged.
        from .retrieval.fusion import Fuser as _Fuser
        # Issue #260: resolve the effective rank/score gate. rrf_gate_enabled ==
        # False forces the ungated sentinels (0 / 0.0), so the Fuser accumulates
        # bit-for-bit as before — the master switch is what makes this knob
        # default-inert regardless of the two threshold values.
        _gate_top_m, _gate_min_score = self._rrf_gate_params()
        fuser = _Fuser(
            k=self._rrf_k,
            gate_top_m=_gate_top_m,
            gate_min_score=_gate_min_score,
        )

        # ── Stage 3: re-rank-class additive collector ──────────────
        # The post-fusion additives — sema_boost (gate-only re-rank
        # confidence boost), authority_*, party_attr, access_rate —
        # capture the "is this document authoritative?" signal. Per spec
        # §3 these stay ADDITIVE on top of the fused score, NOT
        # rank-fused. We dual-write them to gene_scores (additive mode
        # path) and to rerank_additive (RRF mode path) so the sort can
        # trivially compose fused + rerank without re-deriving anything.
        rerank_additive: Dict[str, float] = {}
        # Issue #255 (PR-2): per-class breakdown of the same additives, keyed
        # by rerank class ("authority" / "sema_boost" / "party_attr" /
        # "access_rate" — matching the tier_contrib labels). rerank_additive
        # stays the flat sum (the "additive" combinator reads it); the
        # per-class map lets the fused_tier combinator rank each class
        # independently. Both are dual-written at every rerank writer site.
        rerank_by_class: Dict[str, Dict[str, float]] = {}

        # ── party_id filter clause (reused across Tiers 1-3) ──────
        # Semantics: when party_id is provided, return documents that are
        # EITHER attributed to this party OR have no attribution at all
        # (legacy documents ingested before the registry shipped). This keeps
        # retrieval useful on the predominantly-unattributed current
        # knowledge store — a strict IN(...) clause would collapse to ~0 hits.
        # Cross-party leakage is still prevented: documents attributed to a
        # DIFFERENT party are excluded via the NOT IN sub-select.
        _party_filter = ""
        _party_params: list = []
        if party_id is not None:
            try:
                _has_attr_table = cur.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type='table' AND name='gene_attribution'"
                ).fetchone()[0]
            except Exception:
                _has_attr_table = False
            if _has_attr_table:
                _party_filter = (
                    " AND ("
                    "g.gene_id IN (SELECT gene_id FROM gene_attribution WHERE party_id = ?)"
                    " OR g.gene_id NOT IN (SELECT gene_id FROM gene_attribution)"
                    ")"
                )
                _party_params = [party_id]

        # ── Tier 0: path-key compound index (IDF-weighted) ─────────
        # Highest-confidence retrieval signal — a hit means the query
        # mentions both a path-token (project/module) AND a kv_key that
        # an indexed document was tagged with at ingest. This catches
        # template queries like "what is the value of cymatix_port?" where
        # FTS5/SPLADE both miss because the query lacks domain context.
        #
        # CRITICAL: bonus is INVERSELY proportional to the (path_token,
        # kv_key) pair cardinality. Rare pairs like (cymatix, port) — only
        # 2-3 documents share — each get a strong boost (~+5). Common pairs
        # like (steamapps, url) — 3000+ documents share — get ~zero boost,
        # so they don't drown the signal. This is the standard IDF
        # idea applied to compound retrieval keys.
        #
        # Without IDF weighting, a query containing common terms like
        # "url" or "value" would dump +8 on thousands of false-positive
        # documents, regressing retrieval (empirically observed 12% -> 6%
        # on the 2026-04-12 KV-harvest bench before this fix).
        q_lower_tokens = [t.lower() for t in query_terms if t]
        if q_lower_tokens and self._pki_enabled:
            _pki_t0 = time.monotonic()
            try:
                # Group hits by (path_token, kv_key) to compute per-pair
                # cardinality. SQLite GROUP BY + COUNT does this in one
                # pass over the index.
                pt_ph = ",".join("?" * len(q_lower_tokens))
                kk_ph = ",".join("?" * len(q_lower_tokens))
                pki_sql = (
                    f"SELECT path_token, kv_key, gene_id FROM path_key_index "
                    f"WHERE path_token IN ({pt_ph}) AND kv_key IN ({kk_ph})"
                )
                pki_params = list(q_lower_tokens) + list(q_lower_tokens)
                if party_id is not None and _party_filter:
                    pki_sql = (
                        f"SELECT path_token, kv_key, gene_id FROM path_key_index "
                        f"WHERE path_token IN ({pt_ph}) "
                        f"AND kv_key IN ({kk_ph}) "
                        f"AND gene_id IN (SELECT gene_id FROM "
                        f"gene_attribution WHERE party_id = ?)"
                    )
                    pki_params = pki_params + [party_id]
                pki_hits = cur.execute(pki_sql, pki_params).fetchall()

                # Bucket: pair_count[(pt, kk)] = number of distinct documents
                # gene_pairs[gene_id] = list of (pt, kk) pairs that hit
                pair_count: Dict[tuple, int] = {}
                gene_pairs: Dict[str, list] = {}
                for r in pki_hits:
                    pt = r["path_token"]
                    kk = r["kv_key"]
                    gid = r["gene_id"]
                    pair_count[(pt, kk)] = pair_count.get((pt, kk), 0) + 1
                    gene_pairs.setdefault(gid, []).append((pt, kk))

                # Score each document by sum of inverse-cardinality boosts
                # over all (pt, kk) pairs it matched on.
                #
                # Boost formula:
                #   per-pair bonus = PKI_BASE / max(pair_card, PKI_FLOOR)
                # where:
                #   PKI_BASE  = 10.0  (so a unique pair lands at +10)
                #   PKI_FLOOR =  2.0  (caps top-end at +5 for 2-document pairs)
                # A pair with 100 documents contributes only +0.1 per document —
                # essentially noise. A pair with 5 documents contributes +2.
                # Canonical constants live in storage.indexes (issue
                # #165) so the scorer and compact_path_key_index cannot
                # drift on the noise cutoff.
                from .storage.indexes import (
                    PKI_BASE, PKI_FLOOR, PKI_NOISE_CUTOFF,
                )
                _pki_ranked: List[Tuple[str, float]] = []  # Stage 3 RRF
                for gid, pairs in gene_pairs.items():
                    bonus = 0.0
                    for pair in pairs:
                        card = pair_count[pair]
                        if card > PKI_NOISE_CUTOFF:
                            continue  # too common, skip
                        bonus += PKI_BASE / max(card, PKI_FLOOR)
                    if bonus > 0:
                        # Cap total bonus to keep one runaway document from
                        # saturating; 12.0 is roughly 3x the strongest
                        # single signal.
                        capped = min(bonus, 12.0)
                        gene_scores[gid] = gene_scores.get(gid, 0) + capped
                        tier_contrib.setdefault(gid, {})["pki"] = capped
                        _pki_ranked.append((gid, capped))
                # #209 phase 2 / roadmap §3b-1: PKI tier visibility —
                # candidate count per query + noise-cutoff skips. Feeds
                # the R@1 collision census (§4-E1). No-op when OTel off.
                try:
                    from .telemetry import (
                        pki_candidates_histogram,
                        pki_pairs_skipped_counter,
                    )
                    pki_candidates_histogram().record(len(gene_pairs))
                    _n_skipped = sum(
                        1 for _card in pair_count.values()
                        if _card > PKI_NOISE_CUTOFF
                    )
                    if _n_skipped:
                        pki_pairs_skipped_counter().add(_n_skipped)
                except Exception:  # pragma: no cover
                    pass
                # Stage 3: feed PKI ranks into the Fuser regardless of
                # fusion_mode — the Fuser is queried only when "rrf",
                # so additive mode pays a tiny add_tier cost (~1µs per
                # 100 documents) but stays byte-identical at the output.
                fuser.add_tier("pki", _pki_ranked, weight=self._pki_weight)
            except Exception as exc:
                log.debug("path_key_index tier skipped: %s", exc)
            finally:
                _sig("pki", _pki_t0)

        # ── Tier 0.5: filename-anchor boost (flag-gated spike) ─────
        # Dewey bench 2026-04-14: filename alone drives retrieval lift;
        # project/module over-constrain once filename pins location.
        # Boosts documents whose filename_stem matches a query term.
        # Flag-off is a no-op. See cymatix_context/filename_anchor.py.
        if getattr(self, "_filename_anchor_enabled", False):
            try:
                from . import filename_anchor as _fa
                # Stage 3: capture which gids the filename_anchor tier
                # contributed to so we can feed the Fuser. The boost
                # helper writes tier_contrib[gid]["filename_anchor"] —
                # we diff before/after to extract the (gid, score) pairs.
                _fa_gids_before = {
                    gid for gid, contribs in tier_contrib.items()
                    if "filename_anchor" in contribs
                }
                _fa.boost_scores(
                    cur.connection,
                    query_terms,
                    gene_scores,
                    tier_contrib,
                    weight=getattr(self, "_filename_anchor_weight", 4.0),
                    party_filter_sql=_party_filter,
                    party_params=tuple(_party_params),
                )
                _fa_ranked: List[Tuple[str, float]] = [
                    (gid, contribs["filename_anchor"])
                    for gid, contribs in tier_contrib.items()
                    if "filename_anchor" in contribs and gid not in _fa_gids_before
                ]
                fuser.add_tier(
                    "filename_anchor",
                    _fa_ranked,
                    weight=getattr(self, "_filename_anchor_weight", 4.0),
                )
            except Exception as exc:
                log.debug("filename_anchor tier skipped: %s", exc)

        # ── Tier 1: exact tag match (weight: tag_exact_weight) ────
        _tag_exact_t0 = time.monotonic()
        placeholders = ",".join("?" * len(query_terms))
        _effective_terms = list(query_terms)
        if self._tag_df_cap > 0.0:
            # #327 iteration 2: membership relief. Tags covering more than
            # cap×N genes are index noise, not signal — they are removed
            # from the tier's term set entirely, so flood-tied genes whose
            # only match was the flooded tag drop out of the tier instead
            # of staying tied at a scaled score (the measured failure of
            # IDF scaling under RRF).
            _cap_df_rows = cur.execute(
                f"""
                SELECT tag_value, COUNT(DISTINCT gene_id) AS df
                FROM promoter_index
                WHERE tag_value IN ({placeholders})
                GROUP BY tag_value
                """,
                tuple(query_terms),
            ).fetchall()
            _cap_n = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
            if _cap_n > 0:
                _flooded = {r["tag_value"] for r in _cap_df_rows
                            if r["df"] / _cap_n > self._tag_df_cap}
                if _flooded:
                    _effective_terms = [t for t in query_terms
                                        if t not in _flooded]
        if not _effective_terms:
            _tag_exact_ranked: List[Tuple[str, float]] = []
        elif self._tag_idf_enabled:
            # Issue #327: per-tag selectivity scaling. A tag on 25%+ of
            # the corpus ('cymatix' on 1,596/6,276 genes — the verified
            # tag-flood mechanism, docs/benchmarks/2026-07-31-tagger-ab-
            # tag-flood.md) contributes ~0 instead of a full weight;
            # df=1 tags contribute the full weight. Scale, don't drop.
            _eff_placeholders = ",".join("?" * len(_effective_terms))
            _df_rows = cur.execute(
                f"""
                SELECT tag_value, COUNT(DISTINCT gene_id) AS df
                FROM promoter_index
                WHERE tag_value IN ({_eff_placeholders})
                GROUP BY tag_value
                """,
                tuple(_effective_terms),
            ).fetchall()
            _n_genes = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
            _log_n = math.log(_n_genes) if _n_genes > 1 else 1.0
            _idf = {
                r["tag_value"]: (
                    math.log(_n_genes / r["df"]) / _log_n
                    if _n_genes > 1 and r["df"] >= 1 else 1.0
                )
                for r in _df_rows
            }
            rows = cur.execute(
                f"""
                SELECT g.gene_id, pi.tag_value
                FROM genes g
                JOIN promoter_index pi ON g.gene_id = pi.gene_id
                WHERE pi.tag_value IN ({_eff_placeholders})
                  AND g.chromatin < ?
                  {_party_filter}
                  {_prefilter_aliased_clause}
                """,
                (*_effective_terms, int(ChromatinState.HETEROCHROMATIN),
                 *_party_params, *_prefilter_params),
            ).fetchall()
            _idf_scores: Dict[str, float] = {}
            for r in rows:
                _idf_scores[r["gene_id"]] = _idf_scores.get(r["gene_id"], 0.0) + \
                    _idf.get(r["tag_value"], 1.0) * self._tag_exact_weight
            _tag_exact_ranked = sorted(
                _idf_scores.items(), key=lambda kv: kv[1], reverse=True
            )
            for gid, tag_score in _tag_exact_ranked:
                gene_scores[gid] = tag_score
                tier_contrib.setdefault(gid, {})["tag_exact"] = tag_score
        else:
            _eff_placeholders = ",".join("?" * len(_effective_terms))
            rows = cur.execute(
                f"""
                SELECT g.gene_id, COUNT(pi.tag_value) AS match_count
                FROM genes g
                JOIN promoter_index pi ON g.gene_id = pi.gene_id
                WHERE pi.tag_value IN ({_eff_placeholders})
                  AND g.chromatin < ?
                  {_party_filter}
                  {_prefilter_aliased_clause}
                GROUP BY g.gene_id
                """,
                (*_effective_terms, int(ChromatinState.HETEROCHROMATIN), *_party_params, *_prefilter_params),
            ).fetchall()

            _tag_exact_ranked = []  # Stage 3 RRF
            for r in rows:
                # #202: tag_exact_weight (default 3.0 == legacy literal).
                tag_score = r["match_count"] * self._tag_exact_weight
                gene_scores[r["gene_id"]] = tag_score
                tier_contrib.setdefault(r["gene_id"], {})["tag_exact"] = tag_score
                _tag_exact_ranked.append((r["gene_id"], tag_score))
        # Stage 3: count tier — rank by raw score (= match_count ×
        # tag_exact_weight)
        # which is monotone in match_count, so the rank order matches
        # the spec's "rank by match_count descending" rule (§4).
        fuser.add_tier("tag_exact", _tag_exact_ranked, weight=self._tag_exact_weight)
        _sig("tag_exact", _tag_exact_t0)

        # ── Tier 2: prefix tag match (weight: tag_prefix_weight) ───
        # "server" matches "serverconfig", "server_api", etc.
        _tag_prefix_t0 = time.monotonic()
        _tp_sql, _tp_params = self._tag_prefix_sql(
            query_terms,
            _party_filter,
            tuple(_party_params),
            _prefilter_aliased_clause,
            tuple(_prefilter_params),
        )
        rows = cur.execute(_tp_sql, _tp_params).fetchall()

        _tag_prefix_ranked: List[Tuple[str, float]] = []  # Stage 3 RRF
        for r in rows:
            gid = r["gene_id"]
            # #202: tag_prefix_weight (default 1.5 == legacy literal).
            prefix_score = r["match_count"] * self._tag_prefix_weight
            gene_scores[gid] = gene_scores.get(gid, 0) + prefix_score
            tier_contrib.setdefault(gid, {})["tag_prefix"] = prefix_score
            _tag_prefix_ranked.append((gid, prefix_score))
        fuser.add_tier("tag_prefix", _tag_prefix_ranked, weight=self._tag_prefix_weight)
        _sig("tag_prefix", _tag_prefix_t0)

        # ── Tier 3: FTS5 content search (cap: 2.0 × fts5_weight) ───
        if self._fts_available:
            # Build FTS5 query: OR-join all terms
            _fts5_t0 = time.monotonic()
            fts_query = " OR ".join(
                f'"{t}"' for t in query_terms if len(t) > 2
            )
            if fts_query:
                try:
                    fts_rows = cur.execute(
                        f"""
                        SELECT gene_id, rank
                        FROM genes_fts
                        WHERE genes_fts MATCH ?
                        {_prefilter_bare_clause}
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, *_prefilter_params, _fts_fetch_depth),
                    ).fetchall()

                    # Filter by lifecycle tier (batch lookup)
                    if _measurement is not None:
                        _measurement.record(
                            "fts_raw", (r["gene_id"] for r in fts_rows),
                            fetch_depth=_fts_fetch_depth,
                            prefilter_applied=_prefilter_set is not None,
                            scope="bounded_fts_sql_before_lifecycle_and_party_filters",
                        )
                    if fts_rows:
                        fts_ids = [r["gene_id"] for r in fts_rows]
                        fts_ranks = {r["gene_id"]: r["rank"] for r in fts_rows}
                        id_ph = ",".join("?" * len(fts_ids))
                        valid = cur.execute(
                            f"SELECT gene_id FROM genes g "
                            f"WHERE g.gene_id IN ({id_ph}) AND g.chromatin < ?"
                            f" {_party_filter}",
                            (*fts_ids, int(ChromatinState.HETEROCHROMATIN), *_party_params),
                        ).fetchall()
                        valid_ids = {r["gene_id"] for r in valid}
                        if _measurement is not None:
                            _measurement.record("fts_eligible", valid_ids)

                        _fts5_ranked: List[Tuple[str, float]] = []  # Stage 3 RRF
                        for gid in fts_ids:
                            if gid not in valid_ids:
                                continue
                            # FTS5 rank is negative (lower = better match)
                            # Normalize: -rank gives positive. The additive
                            # tier has no leading coefficient -- it adds the
                            # raw BM25 magnitude -- so fts5_weight acts as a
                            # cap-only knob here (#202): cap = 2.0 ×
                            # fts5_weight (default 3.0 -> the legacy 6.0 cap;
                            # was 15*3=45 — drowned out tag matches at 3-9).
                            fts_score = min(-fts_ranks[gid], 2.0 * self._fts5_weight)
                            gene_scores[gid] = gene_scores.get(gid, 0) + fts_score
                            tier_contrib.setdefault(gid, {})["fts5"] = fts_score
                            # Stage 3: feed Fuser with the RAW
                            # negative-bm25 magnitude (-fts_ranks[gid])
                            # rather than the capped value, so the rank
                            # order matches FTS5's true ordering even at
                            # the score-cap saturation point.
                            _fts5_ranked.append((gid, -fts_ranks[gid]))
                        fuser.add_tier("fts5", _fts5_ranked, weight=self._fts5_weight)
                    elif _measurement is not None:
                        _measurement.record("fts_eligible", ())
                except Exception as exc:
                    if _measurement is not None:
                        failed_stage = (
                            "fts_eligible" if _measurement.stages["fts_raw"]["status"] == "captured"
                            else "fts_raw"
                        )
                        _measurement.unavailable(
                            failed_stage, status="failed", reason=type(exc).__name__,
                        )
                    log.warning("FTS5 query failed", exc_info=True)
                finally:
                    _sig("fts5", _fts5_t0)

        # ── Tier 3.5: SPLADE sparse retrieval (weight: splade_weight) ─
        if self._splade_enabled:
            _splade_t0 = time.monotonic()
            try:
                from .backends import splade_backend
                # Check if splade_terms table exists
                has_table = cur.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='splade_terms'"
                ).fetchone()[0]
                if has_table:
                    # Issue #341 fix: a LOCAL keyword-bag string for SPLADE's
                    # own sparse encoder — deliberately NOT named ``query_text``,
                    # which is the function PARAMETER the cross-encoder rerank
                    # seam reads (~:3695). Reusing that name here used to
                    # silently overwrite the caller's raw query with
                    # ``" ".join(query_terms)`` whenever SPLADE ran (the
                    # shipped default), defeating the "raw question, not the
                    # expanded keyword bag" contract documented on the
                    # ``query_text`` parameter above.
                    _splade_query_text = " ".join(query_terms)
                    query_sparse = splade_backend.encode(_splade_query_text)
                    splade_hits = splade_backend.query_splade(
                        self.read_conn, query_sparse, limit=limit * 2,
                        max_df_fraction=self._splade_df_cap_fraction,
                        total_docs=(
                            self.total_genes()
                            if self._splade_df_cap_fraction > 0.0 else None
                        ),
                    )
                    if _prefilter_set is not None:
                        splade_hits = [(gid, s) for gid, s in splade_hits if gid in _prefilter_set]
                    # Chromatin filter (batch lookup) — mirror the FTS5 tier so
                    # cold-demoted genes don't leak back into hot results via
                    # SPLADE. compress_to_heterochromatin is now non-destructive
                    # (it keeps SPLADE terms so the cold tier can reactivate a
                    # gene), so demoted genes still match here and must be
                    # dropped by chromatin < HETEROCHROMATIN. Filter the list
                    # ONCE, before the scoring loop, so both the additive
                    # gene_scores accumulator and the fuser.add_tier("splade")
                    # path receive the filtered set.
                    if splade_hits:
                        _splade_ids = [gid for gid, _ in splade_hits]
                        _splade_id_ph = ",".join("?" * len(_splade_ids))
                        _splade_valid = cur.execute(
                            f"SELECT gene_id FROM genes g "
                            f"WHERE g.gene_id IN ({_splade_id_ph}) AND g.chromatin < ?",
                            (*_splade_ids, int(ChromatinState.HETEROCHROMATIN)),
                        ).fetchall()
                        _splade_valid_ids = {r["gene_id"] for r in _splade_valid}
                        splade_hits = [
                            (gid, s) for gid, s in splade_hits if gid in _splade_valid_ids
                        ]
                    _splade_ranked: List[Tuple[str, float]] = []  # Stage 3 RRF
                    for gid, score in splade_hits:
                        # Normalize SPLADE score to be comparable with
                        # other tiers. #202: leading coefficient =
                        # splade_weight (default 3.5 == legacy literal), so
                        # the tier caps at splade_weight.
                        splade_score = min(score, 20.0) * self._splade_weight / 20.0
                        gene_scores[gid] = gene_scores.get(gid, 0) + splade_score
                        tier_contrib.setdefault(gid, {})["splade"] = splade_score
                        # Stage 3: feed Fuser with the RAW SPLADE score
                        # (uncapped) so saturated documents still have a
                        # well-defined rank.
                        _splade_ranked.append((gid, float(score)))
                    fuser.add_tier("splade", _splade_ranked, weight=self._splade_weight)
            except Exception:
                log.warning("SPLADE retrieval failed", exc_info=True)
            finally:
                _sig("splade", _splade_t0)

        # ── Tier 4: ΣĒMA semantic retrieval + re-ranking ───────────────
        # Two modes:
        #   A) Boost existing candidates (when Tiers 1-3.5 have candidates)
        #   B) Add new candidates via vector scan (when pool is too small)
        if self._sema_codec is not None:
            try:
                # Council fix (sema vectorless auto-gate): resolve
                # vectorlessness BEFORE paying the codec.encode() RTT below,
                # but only on the cycle where Mode B's undersized-pool
                # trigger would fire anyway (same condition Mode B checks
                # further down) — this adds no cache-build work on beds
                # where earlier tiers already fill the pool. If a prior
                # build already proved zero non-null genes.embedding rows,
                # or this proactive check just did, _sema_vectorless is now
                # true and the block below skips encode() + Mode A/B
                # entirely: neither mode can surface a candidate without a
                # single stored vector. invalidate_sema_cache() clears the
                # flag on the next write, so a bed that gains vectors
                # re-enables automatically.
                if (
                    len(gene_scores) < limit // 2
                    and self._sema_cache is None
                    and not self._sema_vectorless
                ):
                    self._build_sema_cache()

                if not self._sema_vectorless:
                    # Issue #341 fix: same shadowing hazard as the SPLADE tier
                    # above — a local keyword-bag string for ΣĒMA's own dense
                    # encode, kept out of the ``query_text`` parameter name so
                    # it can't clobber the raw query the rerank seam reads.
                    _sema_query_text = " ".join(query_terms)
                    query_vec = self._sema_codec.encode(_sema_query_text)
                    top_score = max(gene_scores.values()) if gene_scores else 0

                    # Mode A: Boost existing candidates when confidence is weak
                    _sema_boost_t0 = time.monotonic()
                    if gene_scores and top_score < 20.0:
                        existing_ids = list(gene_scores.keys())
                        # Batched IN-clause — see _iter_in_batches.
                        sema_rows = []
                        for _ph, _batch in _iter_in_batches(existing_ids):
                            sema_rows.extend(cur.execute(
                                f"SELECT gene_id, embedding FROM genes "
                                f"WHERE gene_id IN ({_ph}) AND embedding IS NOT NULL",
                                _batch,
                            ).fetchall())

                        if sema_rows:
                            candidates_sema = []
                            for r in sema_rows:
                                try:
                                    vec = decode_embedding(r["embedding"])
                                    if isinstance(vec, list) and len(vec) == 20:
                                        candidates_sema.append((r["gene_id"], vec))
                                except Exception:
                                    continue

                            if candidates_sema:
                                nearest = self._sema_codec.nearest(
                                    query_vec, candidates_sema, k=len(candidates_sema),
                                )
                                for gid, sim in nearest:
                                    if sim > 0.3:
                                        boost_scale = max(0.5, 1.0 - top_score / 40.0)
                                        # #202: sema_boost_weight (default 2.0
                                        # == legacy literal).
                                        sema_boost = sim * self._sema_boost_weight * boost_scale
                                        gene_scores[gid] += sema_boost
                                        tier_contrib.setdefault(gid, {})["sema_boost"] = sema_boost
                                        # Stage 3: post-fusion additive (re-rank class).
                                        rerank_additive[gid] = rerank_additive.get(gid, 0.0) + sema_boost
                                        # Issue #255 (PR-2): mirror into per-class map.
                                        _sema_cls = rerank_by_class.setdefault("sema_boost", {})
                                        _sema_cls[gid] = _sema_cls.get(gid, 0.0) + sema_boost
                    _sig("sema_boost", _sema_boost_t0)

                    # Mode B: Add new candidates when pool is undersized
                    # Uses pre-materialized numpy cache for fast cosine scan
                    # instead of deserializing 7K JSON blobs per query.
                    if len(gene_scores) < limit // 2:
                        # Build cache on first use (lazy init) — usually
                        # already built by the proactive vectorless check
                        # above; this is a no-op then. Kept for the (rare)
                        # case where the pool shrank between that check and
                        # here — it never does today, but stays a safe
                        # belt-and-suspenders lazy init like before this fix.
                        if self._sema_cache is None:
                            self._build_sema_cache()

                        if self._sema_cache is not None:
                            try:
                                import numpy as np
                                cache = self._sema_cache
                                q = np.array(query_vec, dtype=np.float32)
                                q_norm = np.linalg.norm(q)
                                if q_norm > 1e-8:
                                    q = q / q_norm
                                    # Batch cosine similarity: (N,20) @ (20,) → (N,)
                                    sims = cache["matrix"] @ q
                                    # Mask already-scored documents
                                    existing = set(gene_scores.keys())
                                    fill_count = limit - len(gene_scores)
                                    # Get top-k indices
                                    top_idx = np.argsort(sims)[::-1]
                                    added = 0
                                    _sema_cold_ranked: List[Tuple[str, float]] = []
                                    for idx in top_idx:
                                        if added >= fill_count:
                                            break
                                        gid = cache["gene_ids"][idx]
                                        sim = float(sims[idx])
                                        if gid in existing:
                                            continue
                                        if sim > 0.4:
                                            # #202: sema_cold_weight (default
                                            # 3.0 == legacy literal).
                                            sema_new = sim * self._sema_cold_weight
                                            gene_scores[gid] = sema_new
                                            tier_contrib.setdefault(gid, {})["sema_cold"] = sema_new
                                            # Stage 3: rank by RAW cosine
                                            # similarity, not the weight-
                                            # multiplied score — the
                                            # multiplier is the additive
                                            # weight, but rank comes from
                                            # the cosine ordering itself.
                                            _sema_cold_ranked.append((gid, sim))
                                            added += 1
                                    fuser.add_tier(
                                        "sema_cold", _sema_cold_ranked,
                                        weight=self._sema_cold_weight,
                                    )
                            except ImportError:
                                pass  # numpy not available
            except Exception:
                log.debug("ΣĒMA retrieval failed, continuing without")

        # ── Stage 2/3: dense recall — first-class in BOTH modes ─────
        # Stage 2's query_genes_dense_recall returns [(gid, cosine), ...]
        # already sorted descending.
        #
        # Tier-0 PR-3 (2026-05-16): dense recall is decoupled from
        # ``fusion_mode``. The old gate (``and self._fusion_mode ==
        # "rrf"``) meant dense recall never ran under the default
        # ``additive`` mode — so the BGE-M3 vectors PR-1 computes at
        # ingest were dark in the shipped configuration. The gate is now
        # ``self._dense_embedding_enabled`` alone; the *mode* only
        # decides HOW dense hits enter the ranking:
        #   - RRF mode: feed the Fuser as a tier so cosine ordering
        #     rank-fuses with the lex tiers (the rank-1 dense hit gets
        #     the same 1/(k+1) weight as the rank-1 FTS hit).
        #   - additive mode: merge each dense hit into ``gene_scores``
        #     with ``cosine * self._dense_additive_weight``. The weight
        #     (default 4.0, BM25-comparable — tag-exact is 3.0) puts the
        #     dense contribution on the same scale as the lex tiers
        #     instead of the ``cosine·1.0`` epsilon-noise the old
        #     comment correctly flagged. The lexical accumulator is
        #     otherwise byte-identical; dense is purely *added* on top.
        if self._dense_embedding_enabled:
            try:
                _dense_t0 = time.monotonic()
                dense_hits = self.query_docs_dense_recall(
                    " ".join(query_terms),
                    k=min(self._dense_pool_size, limit * 4),
                    party_id=party_id,
                    read_only=read_only,
                )
                # #209 phase 1: observe every raw dense cosine at its
                # computation site (pre-weight, pre-floor; identical list
                # feeds both fusion modes) so live traffic calibrates
                # dense_additive_weight / dense_additive_min_cosine.
                # No-op instrument when OTel is off.
                try:
                    from .telemetry import dense_cosine_histogram
                    _dch = dense_cosine_histogram()
                    for _gid, _cos in dense_hits:
                        _dch.record(float(_cos), {"arm": "hot"})
                except Exception:
                    pass
                if self._fusion_mode == "rrf":
                    # Stage 3: feed Fuser. raw_score = cosine.
                    fuser.add_tier(
                        "dense", dense_hits, weight=self._dense_weight,
                    )
                    for gid, cosine in dense_hits:
                        # Telemetry observes raw cosine, not multiplied
                        # (spec §6 "raw pre-RRF scores").
                        tier_contrib.setdefault(gid, {})["dense"] = float(cosine)
                        # Ensure dense-only documents appear in
                        # gene_scores so they survive the eligible_ids
                        # gate at the sort. A tiny epsilon is enough —
                        # the actual ordering comes from the Fuser.
                        if gid not in gene_scores:
                            gene_scores[gid] = 1e-9
                else:
                    # Additive mode: merge dense hits into the score
                    # accumulator at a BM25-comparable weight so a
                    # semantic-only match can surface alongside lexical
                    # hits. ``tier_contrib`` records the *weighted*
                    # contribution so telemetry reflects what actually
                    # entered the additive sum.
                    #
                    # Tier-0 review fix (2026-05-16): apply a min-cosine
                    # noise floor, consistent with the cosine cutoffs the
                    # sibling recall paths (query_cold_tier, query_docs_ann)
                    # already enforce. A below-floor hit does NOT contribute
                    # to the score, but is still seeded as a candidate with
                    # a 1e-9 epsilon (same set-membership treatment the RRF
                    # branch above gives dense-only genes); its tier_contrib
                    # is recorded as 0.0 so telemetry stays coherent with
                    # what entered the sum.
                    # Semantic-wiring arm (PRD 2026-06-02): scale ONLY the dense
                    # term by the scoped weight for semantic queries when the arm
                    # is enabled, so dense cosine ordering clears the lexical
                    # stack into @10. Computed once; lex/tag/SPLADE are untouched.
                    _dense_w = self._dense_additive_weight
                    if query_type == "semantic" and os.environ.get("CYMATIX_SEMANTIC_ARM") == "1":
                        _dense_w = self._semantic_dense_additive_weight
                    for gid, cosine in dense_hits:
                        if float(cosine) >= self._dense_additive_min_cosine:
                            contribution = float(cosine) * _dense_w
                            gene_scores[gid] = gene_scores.get(gid, 0.0) + contribution
                            tier_contrib.setdefault(gid, {})["dense"] = contribution
                        else:
                            gene_scores.setdefault(gid, 1e-9)
                            tier_contrib.setdefault(gid, {})["dense"] = 0.0
                _sig("dense", _dense_t0)
            except Exception:
                log.debug("dense recall tier skipped", exc_info=True)

        if not gene_scores:
            raise PromoterMismatch("Zero genes matched across all tiers")

        # ── Lexical anchoring: IDF-weighted rare-term boost ────────
        # Weight query terms by inverse document frequency — rare terms
        # are stronger discriminators. A document matching "conductor" (3 documents)
        # is much more likely the answer than one matching "biged" (200+ documents).
        # Use the real (memoized) knowledge store size, NOT len(gene_scores) — the
        # latter is the scored-candidate pool and collapses IDF to ~0 on
        # large knowledge stores, nullifying the boost.
        total_genes_est = max(self.corpus_size(), len(gene_scores), 100)
        _lex_anchor_t0 = time.monotonic()
        import math as _math
        # Stage 3: lex_anchor accumulates over multiple query terms.
        # We capture per-document total contribution after the loop and
        # rank-feed it as one tier (a single tier name covering all
        # IDF-anchored documents — same semantics as the existing
        # tier_contrib["lex_anchor"] aggregation).
        for term in query_terms:
            term_freq = cur.execute(
                "SELECT COUNT(DISTINCT gene_id) FROM promoter_index WHERE tag_value = ?",
                (term,),
            ).fetchone()[0]
            if term_freq == 0:
                continue
            # IDF boost: rare terms get up to the cap, common terms ~0.5.
            # #202: leading coefficient = lex_anchor_weight (default 1.5 ==
            # legacy literal); the cap scales proportionally at 2.0 × weight
            # (default 3.0 == the legacy cap, which had been lowered from
            # 5.0 to reduce tangential rare-term over-boost).
            idf = _math.log(total_genes_est / term_freq) if term_freq > 0 else 0
            boost = min(idf * self._lex_anchor_weight, 2.0 * self._lex_anchor_weight)
            if boost > 1.0:
                anchor_genes = cur.execute(
                    f"SELECT pi.gene_id FROM promoter_index pi "
                    f"JOIN genes g ON pi.gene_id = g.gene_id "
                    f"WHERE pi.tag_value = ? AND g.chromatin < ?"
                    f" {_party_filter}"
                    f" {_prefilter_aliased_clause}",
                    (term, int(ChromatinState.HETEROCHROMATIN), *_party_params, *_prefilter_params),
                ).fetchall()
                for r in anchor_genes:
                    gid = r["gene_id"]
                    gene_scores[gid] = gene_scores.get(gid, 0) + boost
                    # Anchor IDF can fire for multiple terms in same query; sum them.
                    tc = tier_contrib.setdefault(gid, {})
                    tc["lex_anchor"] = tc.get("lex_anchor", 0.0) + boost
        # Stage 3: feed accumulated lex_anchor totals into the Fuser.
        # Aggregated total IS the per-document strength of the IDF signal,
        # so ranking by it is the right thing.
        _lex_anchor_ranked: List[Tuple[str, float]] = [
            (gid, contribs["lex_anchor"])
            for gid, contribs in tier_contrib.items()
            if "lex_anchor" in contribs
        ]
        fuser.add_tier(
            "lex_anchor", _lex_anchor_ranked, weight=self._lex_anchor_weight,
        )
        _sig("lex_anchor", _lex_anchor_t0)

        # ── Authority boosts: distinguish "about X" from "mentions X" ──
        # Stage 3: thread rerank_additive + tier_contrib so RRF mode
        # can apply the authority bonus on top of the fused score.
        self._apply_authority_boosts(
            cur, gene_scores, query_terms,
            rerank_additive=rerank_additive,
            tier_contrib=tier_contrib,
            rerank_by_class=rerank_by_class,
        )

        # ── Tier 5: harmonic co-activation boost ──────────────────
        # For each candidate, add a score bonus from documents that are
        # harmonically linked to OTHER candidates (mutual reinforcement).
        # Weight: harmonic_weight per link (default 1.0), capped at a
        # total bonus of 3.0 × harmonic_weight (default 3.0) — #202.
        if use_harmonic and gene_scores:
            try:
                _has_harmonic = cur.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type='table' AND name='harmonic_links'"
                ).fetchone()[0]
            except Exception:
                _has_harmonic = False
            if _has_harmonic:
                try:
                    candidate_ids = list(gene_scores.keys())
                    cid_ph = ",".join("?" * len(candidate_ids))
                    harmonic_rows = cur.execute(
                        f"SELECT gene_id_a, gene_id_b, weight "
                        f"FROM harmonic_links "
                        f"WHERE gene_id_a IN ({cid_ph}) "
                        f"  AND gene_id_b IN ({cid_ph})",
                        (*candidate_ids, *candidate_ids),
                    ).fetchall()
                    harmonic_bonus: Dict[str, float] = {}
                    for hr in harmonic_rows:
                        for gid in (hr["gene_id_a"], hr["gene_id_b"]):
                            harmonic_bonus[gid] = min(
                                harmonic_bonus.get(gid, 0) + self._harmonic_weight,
                                3.0 * self._harmonic_weight,
                            )
                    _harmonic_ranked: List[Tuple[str, float]] = []  # Stage 3 RRF
                    for gid, bonus in harmonic_bonus.items():
                        gene_scores[gid] = gene_scores.get(gid, 0) + bonus
                        tier_contrib.setdefault(gid, {})["harmonic"] = bonus
                        _harmonic_ranked.append((gid, bonus))
                    fuser.add_tier(
                        "harmonic", _harmonic_ranked, weight=self._harmonic_weight,
                    )
                except Exception:
                    log.debug("Harmonic boost failed", exc_info=True)

        # ── Tier 5.5: Successor Representation boost ──────────────
        # Discounted future-occupancy over the co-activation graph.
        # Generalises Tier 5's 1-hop harmonic pull-forward to a
        # gamma-discounted k-step horizon. See cymatix_context/sr.py and
        # docs/FUTURE/SUCCESSOR_REPRESENTATION.md. Feature-flagged
        # (retrieval.sr_enabled) so it can A/B before promotion.
        sr_enabled = self._sr_enabled if use_sr is None else bool(use_sr)
        if sr_enabled and gene_scores:
            try:
                from .retrieval.sr import sr_boost
                sr_bonus = sr_boost(
                    self,
                    list(gene_scores.keys()),
                    gamma=self._sr_gamma,
                    k_steps=self._sr_k_steps,
                    weight=self._sr_weight,
                    cap=self._sr_cap,
                )
                _sr_ranked: List[Tuple[str, float]] = []  # Stage 3 RRF
                for gid, bonus in sr_bonus.items():
                    gene_scores[gid] = gene_scores.get(gid, 0) + bonus
                    tier_contrib.setdefault(gid, {})["sr"] = bonus
                    _sr_ranked.append((gid, bonus))
                # Reuse existing _sr_weight knob (spec §3).
                fuser.add_tier("sr", _sr_ranked, weight=self._sr_weight)
            except Exception:
                log.debug("SR Tier 5.5 failed", exc_info=True)

        # ── Tier 5b: entity graph co-occurrence boost ─────────────────────
        # Documents sharing entity nodes with query entities get a score boost.
        # Additive on top of Tier 5 harmonic; +entity_graph_weight per
        # row, capped at 4.0 × entity_graph_weight (default 2.0) — #202.
        # entity_graph schema: entity (TEXT), gene_id (TEXT) — no weight col.
        _eg_enabled = self._entity_graph_retrieval_enabled if use_entity_graph is None else bool(use_entity_graph)
        if _eg_enabled and entities and gene_scores:
            try:
                _eg_t0 = time.monotonic()
                eq_ph = ",".join("?" * len(entities))
                eg_rows = cur.execute(
                    f"SELECT gene_id FROM entity_graph "
                    f"WHERE entity IN ({eq_ph})",
                    entities,
                ).fetchall()
                _eg_ranked: List[Tuple[str, float]] = []  # Stage 3 RRF
                for row in eg_rows:
                    gid = row["gene_id"]
                    if gid in gene_scores:
                        # #202: per-row bonus = 1.0 × entity_graph_weight
                        # (default 0.5 == the legacy 1.0*0.5); cap scales at
                        # 4.0 × weight (default 2.0 == legacy cap).
                        bonus = min(
                            1.0 * self._entity_graph_weight,
                            4.0 * self._entity_graph_weight,
                        )
                        gene_scores[gid] += bonus
                        tier_contrib.setdefault(gid, {})["entity_graph"] = (
                            tier_contrib.get(gid, {}).get("entity_graph", 0.0) + bonus
                        )
                        _eg_ranked.append((gid, bonus))
                fuser.add_tier(
                    "entity_graph", _eg_ranked, weight=self._entity_graph_weight,
                )
                log.debug("tier 5b entity_graph: %d hits, %.1fms",
                          len(eg_rows), (time.monotonic() - _eg_t0) * 1000)
            except Exception:
                log.warning("entity_graph tier 5b failed", exc_info=True)

        # ── Party attribution bonus (+0.5) ────────────────────────
        if party_id is not None and _party_filter and gene_scores:
            try:
                # Batched IN-clause — see _iter_in_batches.
                _attr_ids = list(gene_scores.keys())
                attr_rows = []
                for _ph, _batch in _iter_in_batches(_attr_ids):
                    attr_rows.extend(cur.execute(
                        f"SELECT gene_id FROM gene_attribution "
                        f"WHERE party_id = ? AND gene_id IN ({_ph})",
                        (party_id, *_batch),
                    ).fetchall())
                for ar in attr_rows:
                    gid = ar["gene_id"]
                    gene_scores[gid] += 0.5
                    tier_contrib.setdefault(gid, {})["party_attr"] = 0.5
                    # Stage 3: post-fusion additive (re-rank class).
                    rerank_additive[gid] = rerank_additive.get(gid, 0.0) + 0.5
                    # Issue #255 (PR-2): mirror into per-class map.
                    _party_cls = rerank_by_class.setdefault("party_attr", {})
                    _party_cls[gid] = _party_cls.get(gid, 0.0) + 0.5
            except Exception:
                log.debug("Party attribution bonus failed", exc_info=True)

        # ── Access-rate tiebreaker ────────────────────────────────
        # Small bonus: score += 0.05 * min(rate * 3600, 5). Max 0.25.
        # Only a tiebreaker — breaks ties for documents with equal scores.
        if gene_scores:
            try:
                rate_ids = list(gene_scores.keys())
                # Batched IN-clause — see _iter_in_batches.
                epi_rows = []
                for _ph, _batch in _iter_in_batches(rate_ids):
                    epi_rows.extend(cur.execute(
                        f"SELECT gene_id, epigenetics FROM genes "
                        f"WHERE gene_id IN ({_ph}) AND epigenetics IS NOT NULL",
                        _batch,
                    ).fetchall())
                for er in epi_rows:
                    try:
                        epi = parse_epigenetics(er["epigenetics"])
                        rate = epi.access_rate(3600.0)
                        if rate > 0:
                            bonus = 0.05 * min(rate * 3600.0, 5.0)
                            gid = er["gene_id"]
                            gene_scores[gid] += bonus
                            tier_contrib.setdefault(gid, {})["access_rate"] = bonus
                            # Stage 3: post-fusion additive (tiebreaker class).
                            rerank_additive[gid] = rerank_additive.get(gid, 0.0) + bonus
                            # Issue #255 (PR-2): mirror into per-class map.
                            _acc_cls = rerank_by_class.setdefault("access_rate", {})
                            _acc_cls[gid] = _acc_cls.get(gid, 0.0) + bonus
                    except Exception:
                        continue
            except Exception:
                log.debug("Access-rate tiebreaker failed", exc_info=True)

        # Layered fingerprints: inject parent-document aggregate scores when
        # ≥ 2 chunks of the same file surface in candidates. Opt-in via
        # CYMATIX_LAYERED_FINGERPRINTS=1. See docs/FUTURE/LAYERED_FINGERPRINTS.md.
        if os.environ.get("CYMATIX_LAYERED_FINGERPRINTS", "0") == "1":
            try:
                self._aggregate_parent_fingerprints(gene_scores, tier_contrib)
            except Exception:
                log.warning("parent fingerprint aggregation failed", exc_info=True)

        # Expose scores + per-tier breakdown for score-gated retrieval in
        # context_manager + the activation profiler bench. Both writes
        # need to be under the same lock — concurrent /context calls
        # would otherwise read a (scores_from_call_A, tiers_from_call_B)
        # torn pair.
        with self._last_query_scores_lock:
            self.last_query_scores = dict(gene_scores)
            self.last_tier_contributions = tier_contrib

        # Emit per-tier contribution telemetry (OTel — no-op when off).
        # One histogram observation per (tier, document) pair; a single
        # counter tick per tier that fired at all, labelled by tier
        # name. Makes the bench_skill_activation heatmap live-observable
        # instead of a one-shot static file.
        try:
            from .telemetry import tier_contribution_histogram, tier_fired_counter
            hist = tier_contribution_histogram()
            cnt = tier_fired_counter()
            tiers_seen: set = set()
            for contribs in tier_contrib.values():
                for tier, score in contribs.items():
                    hist.record(float(score), {"tier": tier})
                    tiers_seen.add(tier)
            for tier in tiers_seen:
                cnt.add(1, {"tier": tier})
        except Exception:
            # Promoted to warning so silent histogram failures surface.
            # The tier-activation + per-tier-contribution panels go dark
            # when this swallows; matching the cwola/latency/gauges pattern.
            log.warning("tier telemetry emit failed", exc_info=True)

        if _measurement is not None:
            _measurement.record("pre_shortlist", gene_scores)
        _shortlist_status = "not_applied"

        # ── BM25 shortlist post-filter (research review 2026-04-22) ──
        # When enabled, restrict the final ranking to documents that cleared a
        # BM25 top-N pass. All tiers still accumulated scores above; this
        # drops candidates BM25 would never surface before the sort. Tests
        # the hypothesis that tier-based scoring on BM25-invisible documents
        # is pulling wrong answers into the top-k. Post-filter by design —
        # isolates the ranking-set question from candidate-generation
        # latency work. Soft-fails to the unfiltered ranking on any error.
        if (
            getattr(self, "_bm25_shortlist_enabled", False)
            and not self._bm25_prefilter_enabled  # don't double-filter
            and self._fts_available
            and gene_scores
        ):
            try:
                bm25_terms = [t for t in query_terms if len(t) > 2]
                if bm25_terms:
                    bm25_match = " OR ".join(f'"{t}"' for t in bm25_terms)
                    shortlist_rows = cur.execute(
                        "SELECT gene_id FROM genes_fts "
                        "WHERE genes_fts MATCH ? ORDER BY rank LIMIT ?",
                        (bm25_match, self._bm25_shortlist_size),
                    ).fetchall()
                    shortlist = {r["gene_id"] for r in shortlist_rows}
                    # Empty shortlist (BM25 found nothing) → don't filter;
                    # the tier-only ranking is better than nothing.
                    if shortlist:
                        _shortlist_status = "applied"
                        before = len(gene_scores)
                        gene_scores = {
                            g: s for g, s in gene_scores.items() if g in shortlist
                        }
                        tier_contrib = {
                            g: c for g, c in tier_contrib.items() if g in shortlist
                        }
                        log.debug(
                            "bm25 shortlist: scored=%d shortlist=%d kept=%d",
                            before, len(shortlist), len(gene_scores),
                        )
                    else:
                        _shortlist_status = "empty_fallback"
            except Exception:
                _shortlist_status = "failed"
                log.warning(
                    "bm25 shortlist filter failed — falling back to unfiltered ranking",
                    exc_info=True,
                )

        if _measurement is not None:
            _measurement.record(
                "post_shortlist", gene_scores, filter_status=_shortlist_status,
            )

        # ── Stage 3: branch on fusion_mode for the final ranking ──
        # Spec: docs/specs/2026-05-08-stage-3-rrf-fusion.md §5.
        # additive (default for one release): unchanged sort on the
        #   accumulated gene_scores.
        # rrf: take fused scores from the Fuser, add re-rank-class
        #   additives (sema_boost, authority, party_attr, access_rate)
        #   on top, restrict to documents that survived the bm25 shortlist
        #   filter (= present in gene_scores), then sort.
        #
        # Issue #341: when the pre-cap cross-encoder rerank is effective for
        # this call, BOTH branches emit a wider ranking (max(limit,
        # rerank_depth)) so the cross-encoder can see candidates the cut would
        # otherwise have dropped. The widened list is cut back to ``limit``
        # immediately after the seam below — including on the fallback path —
        # so the returned count is identical either way. Default-off leaves
        # _fuse_limit == limit, i.e. byte-identical to the pre-#341 branch.
        _rerank_on = self._rerank_effective(rerank_override) and query_text is not None
        _fuse_limit = max(limit, self._rerank_depth) if _rerank_on else limit
        if self._fusion_mode == "rrf":
            fused_scores = fuser.all_scores()
            # The bm25_shortlist filter mutated gene_scores above; honor
            # that filter under RRF too — only documents still in
            # gene_scores are eligible. Documents dense-recall surfaced but
            # bm25_shortlist dropped should not resurface in the fused
            # output.
            eligible_ids = set(gene_scores.keys())
            # Restrict all three combinator inputs to the eligible set BEFORE
            # combining (issue #255, PR-2). fused / rr are the flat maps;
            # rr_by_class is the per-class breakdown the fused_tier combinator
            # ranks independently.
            fused = {gid: fused_scores.get(gid, 0.0) for gid in eligible_ids}
            rr = {gid: rerank_additive.get(gid, 0.0) for gid in eligible_ids}
            rr_by_class = {
                cls: {g: v for g, v in members.items() if g in eligible_ids}
                for cls, members in rerank_by_class.items()
            }
            # Combine fused + rerank via the selected operator. Default
            # "additive" reproduces the shipped fused + rerank_additive block
            # byte-for-byte; fused_tier / eps_band / off are bench-gated.
            # Issue #255 (classifier-gated combinator): a per-query override
            # (from the stage-0 query classifier, threaded through
            # context_manager) selects the combinator for THIS call; None falls
            # back to the store's configured global. The map is validated at
            # config load, so a non-None override is a VALID_COMBINATORS member
            # (combine_rerank still raises defensively on a direct mis-call).
            _effective_combinator = rerank_combinator or self._rerank_combinator
            from .retrieval.rerank_combinators import combine_rerank
            final_scores, ranked_ids = combine_rerank(
                _effective_combinator,
                fused,
                rr,
                rr_by_class,
                self._rrf_k,
                self._rerank_tier_weight,
                self._rerank_band_delta,
                _fuse_limit,
            )
            # last_query_scores semantics under RRF: the combined final
            # score, NOT the raw additive accumulator. This is what
            # context_manager reads for ratio gates. Also stash the
            # eligible-restricted pre-combination maps (issue #255 PR-2 debug
            # hooks) under the same lock so the desk-test can recover the
            # exact fused-vs-final inversion signature.
            with self._last_query_scores_lock:
                self.last_query_scores = dict(final_scores)
                self.last_fused_scores = dict(fused)
                self.last_rerank_additive = dict(rr)
            # RRF-distribution telemetry (spec §6). Attribute-less by
            # design: a per-gene_id label would mint one Prometheus series
            # set per document (unbounded on large genomes), and the
            # distribution panel aggregates across documents anyway.
            # No-op instrument when OTel is off.
            try:
                from .telemetry import rrf_fused_score_histogram
                hist = rrf_fused_score_histogram()
                for score in final_scores.values():
                    hist.record(float(score))
            except Exception:
                pass
        else:
            # Additive mode — pre-Stage-3 behavior, byte-identical.
            ranked_ids = sorted(gene_scores, key=gene_scores.get, reverse=True)[:_fuse_limit]

        if _measurement is not None:
            _measurement.record(
                "final_scoring",
                final_scores if self._fusion_mode == "rrf" else gene_scores,
                fusion_mode=self._fusion_mode,
                scope="eligible_scoring_map_before_rerank_and_return_expansion",
            )

        # ── Issue #341: pre-cap cross-encoder rerank ─────────────────────
        # Placed AFTER the fusion branch so the legacy additive profile is
        # covered by the same seam. Reorders the head of the (possibly
        # widened) ranking, then cuts back to ``limit`` — the cut is what
        # keeps ON and OFF the same length. On any scorer failure
        # (_score_rerank -> None) the untouched widened order is cut back to
        # exactly the OFF ranking, so a broken model costs latency, never
        # results.
        # Issue #342: snapshot the fusion-order top-``limit`` BEFORE the
        # rerank seam. Hebbian edge evidence must accrue from this order, not
        # the reranked one — the rerank changes both order and membership of
        # the delivered set, so writing evidence from ``ranked_ids`` after
        # the seam would make an ON arm mutate harmonic_links differently
        # from OFF on the identical query (persistent cross-arm
        # contamination on any shared A/B bed, plus a feedback loop between
        # the cross-encoder's opinions and the seeded-edge graph).
        _evidence_ids = list(ranked_ids[:limit])
        if _rerank_on and ranked_ids:
            _xenc_t0 = time.monotonic()
            depth = min(self._rerank_depth, len(ranked_ids))
            head_ids = ranked_ids[:depth]
            try:
                head_docs_map = self._load_genes_by_ids(head_ids)
                head_docs = [head_docs_map[g] for g in head_ids if g in head_docs_map]
            except Exception:
                # Same containment contract as _score_rerank: nothing in this
                # seam may raise out of retrieval. An empty candidate list
                # scores to an empty reorder, i.e. the fused order untouched.
                log.warning(
                    "rerank candidate load failed; falling back to fused order",
                    exc_info=True,
                )
                head_docs_map, head_docs = {}, []
            xenc = self._score_rerank(query_text, head_docs)
            if xenc is not None:
                # gene_id is the secondary key so equal cross-encoder scores
                # break deterministically (same contract as the Fuser).
                order = sorted(
                    range(len(head_docs)), key=lambda i: (-xenc[i], head_docs[i].gene_id)
                )
                reordered = [head_docs[i].gene_id for i in order]
                # Position-preserving permutation: the reranked ids are written
                # back into the LOADABLE slots of the head, leaving unloadable
                # ids (ranked from a tier index whose genes row is gone — stale
                # dense matrix, deleted doc) exactly where the OFF arm has them.
                # Appending the reranked block and pushing unloadable ids to the
                # tail instead would move them out of the [:limit] window, so the
                # ON arm would deliver MORE bodies than OFF — a count-invariant
                # break. In a self-consistent store every slot is loadable and
                # this is byte-identical to a plain concatenation.
                slots = [i for i, g in enumerate(head_ids) if g in head_docs_map]
                new_head = list(head_ids)
                for slot, gid in zip(slots, reordered):
                    new_head[slot] = gid
                ranked_ids = new_head + ranked_ids[depth:]
            ranked_ids = ranked_ids[:limit]
            # Timing goes through the LOCAL per-signal collector: this
            # function publishes _sig_ms wholesale into last_signal_timings on
            # exit, so a direct merge here would be clobbered.
            _sig("xenc_rerank", _xenc_t0)

        # Issue #341: publish the final order unconditionally — both fusion
        # modes, rerank on or off — so an A/B always compares two orderings
        # produced the same way and the metric basis can never go stale.
        # Deliberately BEFORE the walking tie-break, matching the
        # last_query_scores publication point above.
        #
        # Depth caveat (see the __init__ declaration): this is the POST-CUT
        # list (max_genes * 2), NOT the full candidate pool — query_docs_ann
        # publishes the full union pool instead. A consumer computing
        # rank-of-gold off this path reads a gold document below the cut as
        # absent, which is an unmeasurable, not a rank of zero.
        with self._last_query_scores_lock:
            self.last_ranked_ids = list(ranked_ids)

        # Walking tie-break (opt-in via CYMATIX_WALKING_TIEBREAK=1).
        # When adjacent top-k documents have bitwise-identical fused scores,
        # re-order them using associative-graph signals (neighborhood
        # size, direct edge weight, NLI entailment, freshness) instead
        # of dict insertion order. Overall score ordering is preserved —
        # only within-tie ordering changes. Soft-fails: any exception in
        # the tie-break path falls through to the original ranking.
        # See docs/FUTURE/tie_break_walking.md for the empirical basis.
        try:
            from .retrieval import tie_break
            if tie_break.is_enabled():
                ranked_ids = tie_break.walking_reorder(
                    self.conn, ranked_ids, gene_scores,
                )
        except Exception:
            log.warning("walking tie-break failed, using insertion-order default", exc_info=True)

        # ── Sprint 4: Hebbian evidence accumulation on seeded edges ───
        # Fire-and-forget update to harmonic_links so seeded / co_retrieved
        # rows accrue co_count (both endpoints in top-k) or miss_count
        # (one endpoint retrieved, other in candidate pool but below the
        # cut — weighted by dense-rank distance to cutoff). Candidacy
        # gate: documents outside gene_scores are ignored (topical-orthogonal
        # queries should not punish the edge). Soft-fails — logger
        # hiccups never perturb the retrieval result.
        # Issue #342: evidence accrues from the pre-rerank fusion order
        # (_evidence_ids), keeping ON/OFF rerank arms byte-identical in
        # store mutations — the invariant the isolation receipts assume.
        if self._seeded_edges_enabled and _evidence_ids and not read_only:
            _hebbian_t0 = time.monotonic()
            try:
                from .retrieval.seeded_edges import update_edge_evidence
                # W2.3 Phase A: update_edge_evidence executes UPDATE/DELETE
                # + commit on genome.conn — lock at the store boundary (the
                # retrieval module stays store-agnostic). All SQLite work
                # plus trivial counter math; no model calls inside.
                with self._write_lock:
                    update_edge_evidence(
                        self, gene_scores, _evidence_ids, max_genes=max_genes,
                    )
            except Exception:
                log.debug("Hebbian edge update failed", exc_info=True)
            _sig("hebbian", _hebbian_t0)

        # Batch fetch document rows
        _body_fetch_t0 = time.monotonic()
        id_placeholders = ",".join("?" * len(ranked_ids))
        rows = cur.execute(
            f"SELECT * FROM genes WHERE gene_id IN ({id_placeholders})",
            ranked_ids,
        ).fetchall()

        # Preserve ranked order
        row_map = {r["gene_id"]: r for r in rows}
        genes = [self._row_to_gene(row_map[gid]) for gid in ranked_ids if gid in row_map]
        _sig("body_fetch", _body_fetch_t0)

        # Co-activation pull-forward
        _coact_t0 = time.monotonic()
        expanded = self._expand_coactivated(genes, limit=limit)
        _sig("coact_expand", _coact_t0)

        # Dedupe while preserving order
        seen: set[str] = set()
        result: List[Gene] = []
        for g in expanded:
            if g.gene_id not in seen:
                seen.add(g.gene_id)
                result.append(g)

        with self._last_query_scores_lock:
            self.last_signal_timings = dict(_sig_ms)

        if _measurement is not None:
            _measurement.record("retrieval_returned", (g.gene_id for g in result[:limit]))
        return result[:limit]

    # ── BGE-M3 dense retrieval (Step 4, 2026-05-08; Stage 2 first-class) ─────

    def _get_dense_codec(self):
        """Lazy-load the BGE-M3 codec on first use.

        Stage 2 (2026-05-08): KnowledgeStore wires _dense_embedding_dim from config,
        which now defaults to 1024 (full BGE-M3 Matryoshka). The codec itself
        warns if the dim is not a sanctioned breakpoint.
        """
        if self._dense_codec is None:
            from .backends.bgem3_codec import get_shared_codec, shared_dense_codec_enabled
            # A1: share ONE BGE-M3 model process-wide instead of building a
            # ~2 GB instance per shard (the dominant 100-shard RAM driver).
            # Default on; CYMATIX_SHARE_DENSE_CODEC=0 reverts to per-instance.
            # Perf slice 1 (W0.2): time the codec CONSTRUCTION (metadata
            # only, near-zero). The real ~2 GB weight load happens lazily
            # on first encode inside BGEM3Codec._load(), which records
            # itself under "dense" — receipts subtract THAT one.
            _load_t0 = time.monotonic()
            # 2026-08-04: pass the per-layer resolved device. Without it
            # BGEM3Codec fell to its "cpu" ctor default and dense NEVER used
            # the GPU, even under a CUDA torch with [hardware] device="cuda".
            from .hardware import resolve_layer_device
            self._dense_codec = get_shared_codec(
                dim=self._dense_embedding_dim,
                model_name=self._dense_model,  # #207 dense fast-follow
                device=resolve_layer_device("dense"),
                share=shared_dense_codec_enabled(),
            )
            try:
                from .telemetry import record_model_load
                record_model_load("dense_codec", time.monotonic() - _load_t0)
            except Exception:
                pass
            # One-time threshold-staleness warn: ann_similarity_threshold is
            # calibrated for dim=1024 (Issue #139 / Stage 4, 2026-05-18). If
            # this knowledge store runs at a different dense_embedding_dim, the
            # absolute threshold no longer matches the geometry — surface it.
            if (
                self._dense_embedding_enabled
                and not self._threshold_dim_warned
                and self._dense_embedding_dim != 1024
            ):
                try:
                    coverage = self._reader.execute(
                        "SELECT COUNT(*) AS c FROM genes WHERE embedding_dense_v2 IS NOT NULL"
                    ).fetchone() if self._reader is not None else None
                    has_v2 = bool(coverage and coverage["c"] > 0)
                except Exception:
                    has_v2 = False
                if has_v2:
                    log.warning(
                        "ann_similarity_threshold=%.3f is calibrated for dim=1024; "
                        "v2 vectors are dim=%d. The absolute threshold will not "
                        "transfer — re-measure for this dim. Recall pool is "
                        "independent of threshold.",
                        self._ann_threshold, self._dense_embedding_dim,
                    )
                    self._threshold_dim_warned = True
        return self._dense_codec

    def _trace_commit(self, sql: str) -> None:
        """sqlite3 trace hook on the writer connection (W0.3).

        Counts every COMMIT — conn.commit() surfaces here as the literal
        statement "COMMIT" — so unlocked committers that bypass
        _write_lock are counted without any call-site cooperation.
        """
        try:
            if sql.lstrip()[:6].upper() == "COMMIT":
                self.commit_count += 1
        except Exception:
            pass

    def _install_commit_trace(self) -> None:
        """(Re)register the commit trace hook on the CURRENT self.conn.

        Must be called at every site that rebinds self.conn (init, the
        refresh() bad-connection fallback, the vacuum() reopen) — the hook
        lives on the connection object, so a reopen without this call
        silently freezes commit_count for the rest of the process.
        """
        try:
            self.conn.set_trace_callback(self._trace_commit)
        except Exception:
            log.debug("commit trace callback unavailable", exc_info=True)

    def _encode_dense_v2_blob(
        self,
        content: Optional[str],
        precomputed: Optional[List[float]] = None,
    ) -> Optional[bytes]:
        """Return the ``embedding_dense_v2`` BLOB for an ingest row, or None.

        Tier-0 PR-1 (2026-05-16). Resolution order:

        1. If ``precomputed`` is supplied (a batched caller already encoded
           the vector), pack and return it.
        2. Otherwise, only if ``self._dense_embed_on_ingest`` is true,
           lazy-load the BGE-M3 codec and encode ``content`` as a passage.
        3. Otherwise return ``None`` — the column stays NULL and a later
           ``scripts/backfill_bgem3_v2.py`` run can populate it.

        Encoding is bounded to ``self._dense_passage_char_cap`` chars (#207
        dense fast-follow — configurable via ``[ingestion]
        dense_passage_char_cap``, default 2000, byte-identical to the prior
        ``PASSAGE_CHAR_CAP`` literal) and uses ``task="passage"`` — the same
        contract as the offline backfill, so an inline-ingested genome
        satisfies the backfill's ``length(blob) == dim*4`` idempotency
        skip-clause. Packing goes through the shared ``vec_to_blob`` helper so
        the two write paths cannot drift.

        Encode failures are soft — ingest must never break because the dense
        model is unavailable; the row is stored with a NULL v2 column and a
        WARN is logged.
        """
        from .backends.bgem3_codec import vec_to_blob

        dim = self._dense_embedding_dim
        try:
            if precomputed is not None:
                return vec_to_blob(precomputed, dim)
            if not self._dense_embed_on_ingest:
                return None
            text = (content or "").strip()
            if not text:
                return None
            codec = self._get_dense_codec()
            vec = codec.encode(text[:self._dense_passage_char_cap], task="passage")
            return vec_to_blob(vec, dim)
        except Exception:
            log.warning(
                "dense v2 encode failed at ingest — storing row with NULL "
                "embedding_dense_v2 (backfill later)",
                exc_info=True,
            )
            return None

    def _invalidate_dense_matrix(self, force: bool = False) -> None:
        """Drop the in-memory dense matrix so it rebuilds on next query.

        Called from upsert/delete paths after an ingest batch. Rebuild is full
        (not incremental) — at ~78 MiB and 18.9k rows this is sub-200 ms and
        triggers only on the first dense recall after an ingest batch.

        ``force=True`` is a no-op flag retained for API symmetry (spec §4) —
        the rebuild semantics are always full.
        """
        with self._dense_matrix_lock:
            self._dense_matrix = None
            self._dense_matrix_ids = None

    def _ensure_dense_matrix(self):
        """Build/load the hot-tier (lifecycle tier < 2) dense matrix.

        Returns ``(matrix, gene_ids)`` or ``(None, None)`` if the v2 column is
        empty. Hot-tier only — heterochromatin (lifecycle tier=2) is reachable via
        ``query_cold_tier()``. Stage 7 surfaces cold matches as MissBlocks.

        Raw BLOB layout: little-endian fp32, ``dim * 4`` bytes per row,
        zero-copy via ``np.frombuffer``.
        """
        with self._dense_matrix_lock:
            if self._dense_matrix is not None and self._dense_matrix_ids is not None:
                return self._dense_matrix, self._dense_matrix_ids

            try:
                import numpy as np
            except ImportError:
                log.debug("numpy unavailable; dense recall disabled")
                return None, None

            # Perf slice 2 (W0.2): the first build scans every hot-tier
            # blob into RAM — a cold cost that must not be silently booked
            # to the express stage. Recorded under "dense_matrix".
            _build_t0 = time.monotonic()

            # Hot-tier scan. Partial index idx_genes_dense_v2_hot makes this
            # an index range scan rather than a full table scan during
            # partial-rollout / backfill.
            cur = self._reader.cursor() if self._reader is not None else self.conn.cursor()
            rows = cur.execute(
                "SELECT gene_id, embedding_dense_v2 FROM genes "
                "WHERE embedding_dense_v2 IS NOT NULL AND chromatin < ?",
                (int(ChromatinState.HETEROCHROMATIN),),
            ).fetchall()
            if not rows:
                return None, None

            dim = self._dense_embedding_dim
            expected_bytes = dim * 4  # fp32
            ids: list[str] = []
            vecs: list[np.ndarray] = []
            for r in rows:
                blob = r["embedding_dense_v2"]
                if blob is None or len(blob) != expected_bytes:
                    continue
                # Little-endian fp32, zero-copy view -> copy into accum array.
                vec = np.frombuffer(blob, dtype="<f4")
                if vec.shape[0] != dim:
                    continue
                ids.append(r["gene_id"])
                vecs.append(vec)
            if not vecs:
                return None, None
            # A2 (RAM): optionally store the resident dense matrix as fp16,
            # halving its heap footprint (~3.3 GB -> ~1.65 GB across 100
            # shards). Default float32 = byte-identical to the prior path; set
            # CYMATIX_DENSE_MATRIX_DTYPE=float16 to opt in. Cosine ranking is
            # robust to fp16, but it is a real precision change so it is OFF by
            # default. (numpy promotes to fp32 inside the matmul, so the
            # per-query compute is unaffected; only resident storage shrinks.)
            matrix = np.stack(vecs).astype(_dense_matrix_dtype(), copy=False)
            self._dense_matrix = matrix
            self._dense_matrix_ids = ids
            try:
                from .telemetry import record_model_load
                record_model_load("dense_matrix", time.monotonic() - _build_t0)
            except Exception:
                pass
            log.debug(
                "dense matrix loaded: shape=%s dtype=%s ids=%d",
                matrix.shape, matrix.dtype, len(ids),
            )
            return matrix, ids

    def query_docs_dense_recall(
        self,
        query: str,
        *,
        k: int = 500,
        party_id: Optional[str] = None,
        read_only: bool = False,
    ) -> List[tuple[str, float]]:
        """Stage 2 first-class dense recall.

        Returns ``[(gene_id, cosine), ...]`` sorted descending. Hot-tier only.
        Does NOT load document bodies — that's deferred to ``query_genes_ann`` so
        we body-fetch once after the lex+dense union.

        Falls back to ``[]`` (with one-time warn) if v2 coverage is empty —
        callers degrade to lexical-only.

        ``party_id`` and ``read_only`` are accepted for API symmetry with
        ``query_genes_ann``; party-aware sharding is out of scope for Stage 2
        (spec §13 punts it to a later release).
        """
        del party_id, read_only  # reserved; spec §13 explicitly defers sharding
        if not self._dense_embedding_enabled:
            return []
        try:
            import numpy as np
        except ImportError:
            return []

        matrix, ids = self._ensure_dense_matrix()
        if matrix is None or ids is None:
            if not self._dense_v2_partial_warned:
                log.warning(
                    "embedding_dense_v2 coverage is empty; dense recall disabled. "
                    "Run scripts/backfill_bgem3_v2.py against this genome to populate."
                )
                self._dense_v2_partial_warned = True
            return []

        codec = self._get_dense_codec()
        query_vec = np.asarray(codec.encode(query, task="query"), dtype=np.float32)
        if query_vec.shape[0] != matrix.shape[1]:
            log.warning(
                "dense recall dim mismatch: query=%d matrix=%d; returning []",
                query_vec.shape[0], matrix.shape[1],
            )
            return []

        # All vectors are L2-normalized at encode/backfill time (codec
        # contract). matmul == cosine.
        # When the matrix is fp16 (A2), numpy promotes this matmul to fp32
        # internally, so cosine values are computed at full precision and only
        # the *stored* matrix is halved. ``query_vec`` is small (1024,).
        sims = matrix @ query_vec
        n = sims.shape[0]
        k_eff = min(int(k), n)
        if k_eff <= 0:
            return []
        # argpartition is O(n); the ~k tail is then sorted.
        idx_part = np.argpartition(-sims, k_eff - 1)[:k_eff]
        # Sort the partition descending by similarity.
        idx_sorted = idx_part[np.argsort(-sims[idx_part])]
        return [(ids[int(i)], float(sims[int(i)])) for i in idx_sorted]

    def query_docs_ann(
        self,
        query: str,
        threshold: float | None = None,
        max_genes: int | None = None,
        min_genes: int | None = None,
        domains: list[str] | None = None,
        entities: list[str] | None = None,
        party_id: Optional[str] = None,
        use_harmonic: bool = True,
        use_sr: Optional[bool] = None,
        use_entity_graph: Optional[bool] = None,
        read_only: bool = False,
        *,
        pool_size: int | None = None,
        rerank_combinator: Optional[str] = None,
        rerank_override: Optional[bool] = None,
    ) -> List[Gene]:
        """Stage 2 retrieval: parallel lex + dense recall, union, threshold cut.

        Refactor (spec §6):

        1. Resolve ``pool_size = pool_size or self._dense_pool_size``.
        2. Fan out:
           - lex pool via ``query_genes(..., max_genes=pool_size)``
           - dense pool via ``query_genes_dense_recall(query, k=pool_size)``
        3. Union by gene_id, preserving best score per source.
        4. Body-fetch once via ``_load_genes_by_ids``.
        5. Apply threshold + ``min_genes`` floor; cap at ``max_genes``.
           Issue #214: if fewer than ``dense_pool_floor_genes`` dense-scored
           candidates survive, admit the top-N dense hits by cosine anyway
           (see ``apply_ann_gate``) so a mis-calibrated threshold can no
           longer gate the dense leg out of pool construction entirely.

        ``ann_similarity_threshold`` was recalibrated for dim=1024 v2 vectors
        in Issue #139 / Stage 4 (2026-05-18): measured unrelated-pair cosine
        mean ~0.50 over the bench fixtures, so the default was raised 0.35 ->
        0.58 (just above the p90 of unrelated pairs). ``max_genes`` remains the
        hard cap, so blast radius is bounded regardless of the threshold.

        Issue #341 (``rerank_override``): step 6 re-scores the union pool with
        a cross-encoder BEFORE the gate's count is applied — see the seam
        below. ``None`` defers to the store's ``rerank_enabled``.

        Back-compat: callers that pass only positional/keyword args still
        work; ``pool_size`` is keyword-only and optional.
        """
        _measurement = current_capture()
        if _measurement is not None:
            _measurement.unsupported.append("query_docs_ann: union and ANN gate are not captured")

        # Stage 4: when mode='margin_over_random', read the calibrated value
        # from genome_calibration; falls back to self._ann_threshold (legacy
        # absolute) on missing row. Caller-supplied ``threshold`` still wins.
        threshold = threshold if threshold is not None else self._get_effective_ann_threshold()
        max_genes = max_genes if max_genes is not None else self._ann_max_genes
        min_genes = min_genes if min_genes is not None else self._ann_min_genes
        pool_size = pool_size if pool_size is not None else self._dense_pool_size
        domains = domains or []
        entities = entities or []

        # ── 1. Lex recall pool (size = pool_size, NOT max_genes). ─────
        lex_pool = self.query_docs(
            domains,
            entities,
            max_genes=pool_size,
            party_id=party_id,
            use_harmonic=use_harmonic,
            use_sr=use_sr,
            use_entity_graph=use_entity_graph,
            read_only=read_only,
            # Issue #255: forward the per-query combinator override so the
            # dense-ANN path (dense-on default) honors the classifier gate too;
            # the combinator runs inside this internal query_docs call.
            rerank_combinator=rerank_combinator,
            # Issue #341 — double-rerank suppression: deliberately NO
            # query_text and NO rerank_override here. This call only builds the
            # lex recall pool; reranking it would burn a cross-encoder pass on
            # a pool that the union + gate below immediately re-orders, and
            # would then rerank the survivors a second time. The single
            # authoritative rerank for this path is the one at step 5b.
        )

        # Dense disabled -> degrade to legacy lex-only flow, capped at max_genes.
        if not self._dense_embedding_enabled:
            return lex_pool[:max_genes]

        # ── 2. Dense recall pool (id+score, no bodies). ──────────────
        _ann_dense_t0 = time.monotonic()
        dense_hits = self.query_docs_dense_recall(
            query, k=pool_size, party_id=party_id, read_only=read_only,
        )
        _ann_dense_ms = (time.monotonic() - _ann_dense_t0) * 1000.0

        # ── 3. Union by gene_id. Lex documents get sim=threshold-0.01 unless
        # they also appeared in the dense pool. Dense-only ids get loaded
        # via _load_genes_by_ids so we body-fetch once.
        sim_by_id: dict[str, float] = {gid: float(s) for gid, s in dense_hits}
        ordered_ids: list[str] = [gid for gid, _ in dense_hits]
        seen: set[str] = set(ordered_ids)
        for doc in lex_pool:
            if doc.gene_id not in seen:
                seen.add(doc.gene_id)
                ordered_ids.append(doc.gene_id)
                # Lex-only id w/o dense score: pin slightly below threshold so
                # the min_genes floor can still rescue them. Spec §8 retains
                # this behavior; Stage 3 (RRF) replaces the placeholder.
                sim_by_id[doc.gene_id] = threshold - 0.01

        # Body-fetch missing ids in bulk. Lex pool already has bodies; only
        # dense-only ids need loading.
        lex_by_id = {d.gene_id: d for d in lex_pool}
        missing_ids = [gid for gid in ordered_ids if gid not in lex_by_id]
        _ann_body_t0 = time.monotonic()
        loaded = self._load_genes_by_ids(missing_ids) if missing_ids else {}
        _ann_body_ms = (time.monotonic() - _ann_body_t0) * 1000.0

        # Perf slice 1 (W0.2): merge this function's own leg timings into
        # the signal map the inner query_docs just published, so one read
        # of last_signal_timings covers the whole ANN retrieval.
        with self._last_query_scores_lock:
            _merged = dict(getattr(self, "last_signal_timings", None) or {})
            _merged["ann_dense_leg"] = _merged.get("ann_dense_leg", 0.0) + _ann_dense_ms
            _merged["ann_body_fetch"] = _merged.get("ann_body_fetch", 0.0) + _ann_body_ms
            self.last_signal_timings = _merged

        # ── 4. Resolve final Document objects in score order. ────────────
        scored: list[tuple[Gene, float]] = []
        for gid in ordered_ids:
            doc = lex_by_id.get(gid) or loaded.get(gid)
            if doc is None:
                continue
            scored.append((doc, sim_by_id[gid]))
        scored.sort(key=lambda x: x[1], reverse=True)

        # ── 5. Threshold cut + min_genes floor + max_genes cap, plus the
        # Issue #214 dense pool floor. The cut is a module-level pure
        # function (apply_ann_gate) so the gate semantics are unit-testable
        # without a store; with the floor disabled — or whenever >= N
        # dense-scored candidates survive — it reproduces the legacy loop
        # byte-for-byte.
        gate_ids = apply_ann_gate(
            [(doc.gene_id, sim) for doc, sim in scored],
            dense_hits,
            threshold=threshold,
            min_genes=min_genes,
            max_genes=max_genes,
            dense_pool_floor_genes=self._dense_pool_floor_genes,
        )
        by_id = {doc.gene_id: doc for doc, _ in scored}
        kept_ids = gate_ids
        final_order = [doc.gene_id for doc, _ in scored]
        head_id_set = {
            doc.gene_id
            for doc, _ in scored[: min(self._rerank_depth, len(scored))]
        }
        # Gate-kept ids that sit BEYOND the sim head — i.e. what the #214
        # floor rescued (or what a deep max_genes admitted past rerank_depth).
        n_floor = sum(1 for g in gate_ids if g not in head_id_set)
        # The gate's count AS DELIVERED. apply_ann_gate can hand back an id
        # with no loaded body (a #214 floor rescue whose dense-matrix entry
        # outlived its genes row, or a row _row_to_gene rejected); the return
        # filter at the bottom silently drops those on BOTH arms. Pinning the
        # rerank cut to len(gate_ids) instead would re-fill those slots from
        # the pool and make the ON arm longer than the OFF arm — so pin it to
        # what the OFF arm actually delivers.
        n_deliverable = sum(1 for g in gate_ids if g in by_id)

        # ── 5b. Issue #341: pre-cap cross-encoder rerank. ────────────────
        # The gate decides HOW MANY documents survive; with this on, the
        # cross-encoder decides WHICH. len(kept_ids) is pinned to
        # n_deliverable (see the comment above for why that, and not
        # len(gate_ids)), so THIS FUNCTION returns the same number of
        # documents on both arms. That invariance is scoped to this
        # function's return length and nothing further: membership changes
        # propagate to the downstream budget/assembly cuts, which are
        # score-dependent, so the count entering splice can and does differ
        # between arms — see docs/benchmarks/2026-08-07-rerank-wiring-receipts.md
        # Finding 2.
        if self._rerank_effective(rerank_override) and scored:
            _xenc_t0 = time.monotonic()
            depth = min(self._rerank_depth, len(scored))
            # Candidate set = the sim head PLUS every gate-kept id beyond it
            # (#214 floor rescues land there). The cross-encoder therefore
            # JUDGES rescued documents rather than blindly evicting them, and
            # cand_ids is always a superset of gate_ids — which is what makes
            # the [:n_deliverable] cut below exact.
            head_ids = [doc.gene_id for doc, _ in scored[:depth]]
            cand_ids = head_ids + [g for g in gate_ids if g not in head_id_set]
            cand_docs = [by_id[g] for g in cand_ids if g in by_id]
            # Bodies for the whole pool were fetched in step 3 — no extra I/O.
            xenc = self._score_rerank(query, cand_docs)
            if xenc is not None:
                order = sorted(
                    range(len(cand_docs)),
                    key=lambda i: (-xenc[i], cand_docs[i].gene_id),
                )
                ranked_cand = [cand_docs[i].gene_id for i in order]
                # Count from the gate; order + membership from the cross-encoder.
                kept_ids = ranked_cand[:n_deliverable]
                _in_cand = set(ranked_cand)
                final_order = ranked_cand + [
                    d.gene_id for d, _ in scored if d.gene_id not in _in_cand
                ]
            with self._last_query_scores_lock:
                _m = dict(getattr(self, "last_signal_timings", None) or {})
                _m["xenc_rerank"] = (
                    _m.get("xenc_rerank", 0.0) + (time.monotonic() - _xenc_t0) * 1000.0
                )
                self.last_signal_timings = _m

        with self._last_query_scores_lock:
            # Overwrites the inner query_docs publication with the ANN-final
            # order: the FULL union pool (~pool_size), post-rerank when it
            # ran, spanning candidates both above and below the gate. This is
            # the deeper of the two publication semantics — see the
            # last_ranked_ids declaration in __init__ for why the lex path's
            # post-cut list is deliberately not unified with it.
            self.last_ranked_ids = final_order
            # Published on BOTH arms so the verdict can slice floor-fired needles.
            self.last_rerank_diag = {
                "gate_kept": len(gate_ids), "floor_appended": n_floor,
            }
        return [by_id[gid] for gid in kept_ids if gid in by_id]

    def _load_genes_by_ids(self, gene_ids: list[str]) -> dict[str, Gene]:
        """Bulk-load Document objects by id. Used by query_genes_ann to fetch
        bodies for dense-only hits exactly once.
        """
        if not gene_ids:
            return {}
        ph = ",".join("?" * len(gene_ids))
        rows = self.read_conn.cursor().execute(
            f"SELECT * FROM genes WHERE gene_id IN ({ph})",
            gene_ids,
        ).fetchall()
        out: dict[str, Gene] = {}
        for r in rows:
            try:
                out[r["gene_id"]] = self._row_to_gene(r)
            except Exception:
                log.debug("row_to_gene failed for %s", r["gene_id"], exc_info=True)
        return out

    # ── Entity graph: auto-link documents sharing entities ───────────────

    def _auto_link_by_entity(self, gene_id: str, entities: List[str], cur) -> None:
        """Delegate to storage.co_activation.auto_link_by_entity."""
        from .storage.co_activation import auto_link_by_entity
        auto_link_by_entity(gene_id, entities, cur)

    def _expand_by_entity_graph(
        self, gene_ids: List[str], limit: int, cur
    ) -> List[str]:
        """Delegate to storage.co_activation.expand_by_entity_graph."""
        from .storage.co_activation import expand_by_entity_graph
        return expand_by_entity_graph(gene_ids, limit, cur)

    # ── Co-activation expansion ─────────────────────────────────────

    def _expand_coactivated(self, genes: List[Gene], limit: int) -> List[Gene]:
        from .storage.co_activation import expand_coactivated
        # Perf slice 2: pure-read expansion belongs on the (per-thread)
        # reader, not the shared writer — two threads interleaving these
        # scans on one connection was the c=2 server wedge.
        # Council fix (flag-leak): the entity_graph sub-scan inside this
        # post-ranking expansion is a QUERY-time read and must be gated by
        # the retrieval-side flag (_entity_graph_retrieval_enabled — same
        # flag as Tier 5b at query_docs' entity-graph scoring block), not
        # the write/ingest-side _entity_graph_enabled ([ingestion]
        # entity_graph). Previously the ingest flag (default True) silently
        # authorized this scan on every query even when the retrieval flag
        # (default False) had Tier 5b's scoring disabled.
        return expand_coactivated(
            genes, limit, self.read_conn, self._entity_graph_retrieval_enabled,
            symbol_expansion_cap=getattr(self, "_symbol_expansion_cap", 8),
        )

    # ── Row → Document ──────────────────────────────────────────────────

    def _row_to_gene(self, row: sqlite3.Row) -> Gene:
        def _opt(key: str, default=None):
            try:
                return row[key]
            except (IndexError, KeyError):
                return default

        # Guard against NULL/corrupt metadata fields
        try:
            promoter = parse_promoter(row["promoter"]) if row["promoter"] else PromoterTags()
        except Exception:
            promoter = PromoterTags()
        try:
            epigenetics = parse_epigenetics(row["epigenetics"]) if row["epigenetics"] else EpigeneticMarkers()
        except Exception:
            epigenetics = EpigeneticMarkers()
        try:
            chromatin = ChromatinState(row["chromatin"]) if row["chromatin"] is not None else ChromatinState.OPEN
        except (ValueError, TypeError):
            chromatin = ChromatinState.OPEN

        # key_values may not exist in older databases before ALTER TABLE runs
        try:
            kv_raw = row["key_values"]
            key_values = json_loads(kv_raw) if kv_raw else []
        except (IndexError, KeyError):
            key_values = []

        return Gene(
            gene_id=row["gene_id"],
            content=row["content"] or "",
            # Heterochromatin-compressed documents have complement=NULL after
            # compress_to_heterochromatin(). Fall back to "" for Pydantic.
            complement=row["complement"] or "",
            codons=json_loads(row["codons"]) if row["codons"] else [],
            promoter=promoter,
            epigenetics=epigenetics,
            chromatin=chromatin,
            is_fragment=bool(row["is_fragment"]) if row["is_fragment"] is not None else False,
            embedding=decode_embedding(row["embedding"]),
            source_id=row["source_id"],
            repo_root=_opt("repo_root"),
            source_kind=_opt("source_kind"),
            observed_at=_opt("observed_at"),
            mtime=_opt("mtime"),
            content_hash=_opt("content_hash"),
            volatility_class=_opt("volatility_class"),
            authority_class=_opt("authority_class"),
            support_span=_opt("support_span"),
            last_verified_at=_opt("last_verified_at"),
            version=row["version"] if row["version"] is not None else 1,
            supersedes=row["supersedes"],
            key_values=key_values,
        )

    # ── Touch (update signals on access) ────────────────────────

    def touch_genes(self, gene_ids: List[str]) -> None:
        if self.read_only:
            log.debug("read_only: skipping touch_genes")
            return
        if not gene_ids:
            return

        # W2.3 Phase A: multi-statement read-modify-write transaction on the
        # shared writer — unlocked, a concurrent committer steals the
        # implicit transaction or hard-deadlocks inside an UPDATE on py3.14
        # sqlite3 (same class as log_health, Perf slice 2). Hold the write
        # lock for the whole SELECT→UPDATE*→commit section; the JSON
        # parse/dump inside is pure CPU (mirrors upsert_doc).
        with self._write_lock:
            cur = self.conn.cursor()
            now = time.time()

            # Batch fetch all signals in one query
            placeholders = ",".join("?" * len(gene_ids))
            rows = cur.execute(
                f"SELECT gene_id, epigenetics FROM genes WHERE gene_id IN ({placeholders})",
                gene_ids,
            ).fetchall()

            # Individual UPDATEs — safe against column-swap corruption
            # (CASE WHEN batch was causing signals JSON to land in lifecycle tier)
            for row in rows:
                if not row["epigenetics"]:
                    continue
                epi = parse_epigenetics(row["epigenetics"], use_cache=False)
                epi.last_accessed = now
                epi.access_count += 1
                epi.decay_score = min(1.0, epi.decay_score + 0.1)
                # Phase 1 slice 2 — populate the windowed access-rate buffer.
                # Bounded ring buffer of last 100 timestamps (~800 bytes).
                # The slice 1 schema added the field; this is where it gets
                # written. apply_density_gate consumes it via the rate signal.
                epi.recent_accesses.append(now)
                if len(epi.recent_accesses) > 100:
                    epi.recent_accesses = epi.recent_accesses[-100:]
                cur.execute(
                    "UPDATE genes SET epigenetics = ?, chromatin = ? WHERE gene_id = ?",
                    (epi.model_dump_json(), int(ChromatinState.OPEN), row["gene_id"]),
                )

            self.conn.commit()
        clear_parse_caches()

    # ── Update co-activation links (mutual) ─────────────────────────

    def link_coactivated(self, gene_ids: List[str]) -> None:
        """Create mutual co-activation links between all retrieved documents."""
        if self.read_only:
            log.debug("read_only: skipping link_coactivated")
            return
        if len(gene_ids) < 2:
            return

        # W2.3 Phase A: same shared-writer transaction shape as touch_genes
        # (SELECT→UPDATE*→commit) — hold the write lock for the whole
        # section so a concurrent committer can't steal the implicit
        # transaction or wedge the UPDATE (py3.14 sqlite3 deadlock class).
        with self._write_lock:
            cur = self.conn.cursor()

            # Batch fetch all signals in one query
            placeholders = ",".join("?" * len(gene_ids))
            rows = cur.execute(
                f"SELECT gene_id, epigenetics FROM genes WHERE gene_id IN ({placeholders})",
                gene_ids,
            ).fetchall()

            # Build individual updates (signals only, preserve lifecycle tier)
            for row in rows:
                if not row["epigenetics"]:
                    continue
                epi = parse_epigenetics(row["epigenetics"], use_cache=False)
                gid = row["gene_id"]
                peers = [other for other in gene_ids if other != gid]

                existing = set(epi.co_activated_with)
                existing.update(peers)
                epi.co_activated_with = list(existing)[:10]

                cur.execute(
                    "UPDATE genes SET epigenetics = ? WHERE gene_id = ?",
                    (epi.model_dump_json(), gid),
                )

            self.conn.commit()
        clear_parse_caches()

    # ── Harmonic weights (cymatics) ──────────────────────────────────

    def store_harmonic_weights(self, weights: List[Tuple[str, str, float]]) -> None:
        """Delegate to storage.co_activation.store_harmonic_weights."""
        if self.read_only:
            log.debug("read_only: skipping store_harmonic_weights")
            return
        from .storage.co_activation import store_harmonic_weights
        # W2.3 Phase A: the delegate executes + commits on the shared
        # writer connection — serialize at the store boundary (the storage
        # module stays store-agnostic and must not know about the lock).
        with self._write_lock:
            store_harmonic_weights(self.conn, weights)

    # ── Typed document relations (NLI) ───────────────────────────────────

    def store_relation(
        self, gene_id_a: str, gene_id_b: str,
        relation: int, confidence: float,
    ) -> None:
        """Delegate to storage.co_activation.store_relation."""
        from .storage.co_activation import store_relation
        # W2.3 Phase A: delegate executes + commits on the shared writer —
        # same store-boundary lock rule as store_harmonic_weights.
        with self._write_lock:
            store_relation(self.conn, gene_id_a, gene_id_b, relation, confidence)

    def store_relations_batch(
        self, relations: list,
    ) -> None:
        """Delegate to storage.co_activation.store_relations_batch.

        WS2 review FIX-1: the delegate executes on the shared write
        connection. Hold the write lock for the whole operation (mirrors
        upsert_doc, bug bash 2026-07-18) so this commit can never publish
        another thread's half-written upsert. Issue #372: commit/rollback
        moved up into _write_unit (delegate called with commit=False) so
        the batch folds into a deferred_commits() document scope — the
        CHUNK_OF edges land in the SAME commit as the strands they join.
        """
        from .storage.co_activation import store_relations_batch
        with self._write_lock:
            with self._write_unit():
                store_relations_batch(self.conn, relations, commit=False)

    def get_relations(self, gene_id: str) -> list:
        """Delegate to storage.co_activation.get_relations."""
        from .storage.co_activation import get_relations
        return get_relations(self.conn, gene_id)

    # ── Symbol graph (WS2) ───────────────────────────────────────────

    def store_symbol_defs(self, rows: list) -> None:
        """Index symbol definitions. Each item: (symbol, gene_id, kind).

        WS2 review FIX-1: executes + commits on the shared write connection,
        so it must hold the write lock for the whole operation (mirrors
        upsert_doc, bug bash 2026-07-18) and roll back on failure — both
        via _write_unit since #372, so the rows fold into a
        deferred_commits() document scope when one is open.
        """
        if not rows:
            return
        with self._write_lock:
            with self._write_unit():
                self.conn.executemany(
                    "INSERT OR IGNORE INTO symbol_defs (symbol, gene_id, kind) VALUES (?, ?, ?)",
                    rows,
                )

    def resolve_symbol(self, symbol: str) -> list:
        """gene_ids of chunks that define ``symbol`` (for symbol-aware expansion)."""
        return [
            r[0] for r in self.conn.execute(
                "SELECT gene_id FROM symbol_defs WHERE symbol = ?", (symbol,)
            ).fetchall()
        ]

    # ── Layered fingerprints: query-time parent aggregation ──────────

    def _aggregate_parent_fingerprints(
        self,
        gene_scores: Dict[str, float],
        tier_contrib: Dict[str, Dict[str, float]],
    ) -> None:
        """Inject parent-document aggregate scores into gene_scores + tier_contrib
        when ≥ 2 chunks of the same file hit current candidates.

        Mutates the two dicts in place. Called only when
        CYMATIX_LAYERED_FINGERPRINTS=1. Errors are swallowed by the caller.

        Rules:
          - Candidates with a CHUNK_OF edge to parent P that has N≥2
            distinct children in gene_scores: parent P gets an
            aggregated score = sum(child_scores) * (1 + 0.1 * log1p(N)).
          - Parent tier_contributions = per-tier sum of child contributions.
          - If parent already in gene_scores (because it fired via its
            own content matching), add the aggregated bonus on top
            instead of replacing.
        """
        from math import log1p
        from .schemas import StructuralRelation as _SR

        if not gene_scores:
            return

        cand_ids = list(gene_scores.keys())
        # Batched edge lookup. SQLite IN clause capped at ~999 params;
        # chunk candidates if larger. Normal top-k is < 100 so a single
        # query is fine in practice.
        placeholders = ",".join("?" * len(cand_ids))
        rows = self.conn.execute(
            f"SELECT gene_id_a AS child, gene_id_b AS parent "
            f"FROM gene_relations "
            f"WHERE gene_id_a IN ({placeholders}) AND relation = ?",
            (*cand_ids, int(_SR.CHUNK_OF)),
        ).fetchall()
        if not rows:
            return

        # parent -> set of children hit
        from collections import defaultdict
        parent_children: Dict[str, set] = defaultdict(set)
        for r in rows:
            parent_children[r["parent"]].add(r["child"])

        for parent_gid, children in parent_children.items():
            n_hits = len(children)
            if n_hits < 2:
                continue  # co-activation requires ≥ 2 children

            child_score_sum = sum(gene_scores.get(c, 0.0) for c in children)
            co_activation_bonus = 1.0 + 0.1 * log1p(n_hits)
            aggregated = child_score_sum * co_activation_bonus

            if parent_gid in gene_scores:
                # Parent fired on its own; add aggregated bonus instead of replacing.
                gene_scores[parent_gid] += aggregated
            else:
                gene_scores[parent_gid] = aggregated

            # Aggregate per-tier contributions from children.
            agg_tiers: Dict[str, float] = {}
            for child_gid in children:
                for tier, val in tier_contrib.get(child_gid, {}).items():
                    agg_tiers[tier] = agg_tiers.get(tier, 0.0) + val
            agg_tiers["parent_coactivation"] = round(aggregated, 3)
            agg_tiers["chunks_hit"] = n_hits

            existing = tier_contrib.get(parent_gid, {})
            for tier, val in agg_tiers.items():
                existing[tier] = existing.get(tier, 0.0) + val if tier not in ("chunks_hit",) else val
            tier_contrib[parent_gid] = existing

    # ── Layered fingerprints: parent-pull reassembly ──────────────────

    def reassemble(self, parent_gene_id: str, separator: str = "\n\n") -> dict:
        """Reassemble a file-level parent document into its full content.

        Reads the parent's ``codons`` field (ordered list of child gene_ids),
        fetches each child's content from the documents table, sorts by
        ``promoter.sequence_index`` for deterministic ordering, and joins
        with the given separator.

        See docs/FUTURE/LAYERED_FINGERPRINTS.md for the design.

        Returns:
            {
                "content":          <stitched text>,
                "source_id":        <original file path>,
                "chunk_count":      <N>,
                "reassembled_from": [<child gene_ids in sequence order>],
                "missing_children": [<child gene_ids that no longer exist>],
            }

        Raises:
            ValueError: gene_id does not exist, or is not a parent document.
        """
        from .schemas import StructuralRelation as _SR  # local import, cycle-safe

        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT gene_id, content, codons, source_id, key_values "
            "FROM genes WHERE gene_id = ?",
            (parent_gene_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"gene_id {parent_gene_id!r} not found")

        # Parent detection: key_values contains "is_parent=true"
        kv_raw = row["key_values"] or "[]"
        try:
            kv = json_loads(kv_raw) if isinstance(kv_raw, str) else kv_raw
        except Exception:
            kv = []
        is_parent = any(
            isinstance(k, str) and k.lower() == "is_parent=true" for k in kv
        )
        if not is_parent:
            raise ValueError(
                f"gene_id {parent_gene_id!r} is not a parent gene "
                f"(missing is_parent=true marker in key_values)"
            )

        codons_raw = row["codons"] or "[]"
        try:
            child_ids = json_loads(codons_raw) if isinstance(codons_raw, str) else codons_raw
        except Exception:
            child_ids = []
        if not child_ids:
            raise ValueError(f"parent {parent_gene_id!r} has empty codons list")

        # Batched fetch of children.
        placeholders = ",".join("?" * len(child_ids))
        child_rows = cur.execute(
            f"SELECT gene_id, content, promoter FROM genes "
            f"WHERE gene_id IN ({placeholders})",
            tuple(child_ids),
        ).fetchall()

        found = {r["gene_id"]: r for r in child_rows}
        missing = [cid for cid in child_ids if cid not in found]
        if missing:
            log.warning(
                "reassemble(%s): %d missing children — stitched output will skip them",
                parent_gene_id, len(missing),
            )

        def _seq_index(r) -> int:
            try:
                p = json_loads(r["promoter"]) if r["promoter"] else {}
            except Exception:
                p = {}
            idx = p.get("sequence_index") if isinstance(p, dict) else None
            return idx if isinstance(idx, int) else 0

        ordered = sorted(
            (found[cid] for cid in child_ids if cid in found),
            key=_seq_index,
        )

        stitched = separator.join(r["content"] or "" for r in ordered)
        return {
            "content": stitched,
            "source_id": row["source_id"],
            "chunk_count": len(child_ids),
            "reassembled_from": [r["gene_id"] for r in ordered],
            "missing_children": missing,
        }

    # ── Compaction (decay stale documents) ──────────────────────────────

    def compact(self) -> int:
        """
        Check documents for source file changes. No time-based decay.

        Documents are NEVER removed by time alone. Knowledge doesn't expire.
        Only two things change a document's state:

        1. SOURCE CHANGED: if gene.source_id points to a file whose mtime
           is newer than last_accessed, decay_score drops to 0.5 (AGING)
           and lifecycle tier moves to EUCHROMATIN. The document is still queryable
           but the system knows it's outdated. Re-ingesting resets it.

        2. EXPLICIT SPLICE: the compressor's splice operation cuts introns
           per-query (irrelevant fragments). This is the RNA splicing analog —
           relevance filtering happens at retrieval time, not storage time.

        Time since last access is used ONLY for retrieval priority
        (recently accessed documents rank higher in query results), never
        for deletion or decay.

        Returns the number of documents marked as source-changed.
        """
        cur = self.conn.cursor()
        change_detected = 0

        # W2.3 Phase A: the mtime probes in the loop are filesystem I/O, so
        # the write lock is NOT held across the whole sweep — each UPDATE
        # is locked individually (write order and semantics unchanged; a
        # concurrent committer may publish a prefix of the sweep, which is
        # exactly as safe as the pre-lock behavior since every UPDATE here
        # is final-intent with no rollback path).
        with self._write_lock:
            rows = cur.execute(
                "SELECT gene_id, epigenetics, chromatin, source_id FROM genes"
            ).fetchall()

        for row in rows:
            source_id = row["source_id"]
            if not source_id or not os.path.exists(source_id):
                continue

            try:
                file_mtime = os.path.getmtime(source_id)
            except OSError:
                continue

            epi = parse_epigenetics(row["epigenetics"], use_cache=False)

            if file_mtime > epi.last_accessed:
                # Source changed — document is outdated but NOT removed
                epi.decay_score = min(epi.decay_score, 0.5)
                new_chromatin = int(ChromatinState.EUCHROMATIN)
                change_detected += 1

                with self._write_lock:
                    cur.execute(
                        "UPDATE genes SET epigenetics = ?, chromatin = ? WHERE gene_id = ?",
                        (epi.model_dump_json(), new_chromatin, row["gene_id"]),
                    )

        # WS2 review FIX-2: orphaned symbol rows (genes removed out-of-band)
        # are swept as part of the same compaction commit.
        with self._write_lock:
            self._sweep_symbol_orphans()
            self.conn.commit()
        if change_detected:
            log.info("Compaction: %d source changes detected (genes marked EUCHROMATIN)",
                     change_detected)
        return change_detected

    # ── Stats ───────────────────────────────────────────────────────

    def total_genes(self) -> int:
        """Bare gene count for the request path.

        2026-08-03 ERB fix: every hot-path consumer of ``stats()``
        (_compute_health, _build_abstain_window, the proxy munge,
        /admin/swap-db) only ever read ``total_genes`` — but paid the
        SUM(LENGTH(content))/SUM(LENGTH(complement)) full scans too:
        11.1s warm / 72.7s cold per call on the 829k-gene / 49GB bed.
        COUNT(*) alone measured 29ms on the same bed.
        """
        self._refresh_snapshot()
        return int(
            self.read_conn.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
        )

    def stats(self) -> Dict:
        self._refresh_snapshot()  # See latest WAL state
        # Perf slice 2: pure-read aggregates on the (per-thread) reader.
        # These full-table scans ran on the shared writer ("authoritative")
        # and were the second c=2 livelock site after the co-activation
        # expansion (py-spy: both threads spinning here via
        # _compute_health). The WAL reader sees every committed row, and
        # _refresh_snapshot() above already advanced the snapshot — the
        # authority argument bought nothing but the collision.
        cur = self.read_conn.cursor()

        total = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
        by_chromatin = cur.execute(
            "SELECT chromatin, COUNT(*) FROM genes GROUP BY chromatin"
        ).fetchall()

        chromatin_counts = {}
        for r in by_chromatin:
            try:
                key = ChromatinState(int(r[0])).name if r[0] is not None else "UNKNOWN"
            except (ValueError, TypeError):
                key = "UNKNOWN"
            chromatin_counts[key] = chromatin_counts.get(key, 0) + r[1]

        total_raw = cur.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM genes"
        ).fetchone()[0]
        total_compressed = cur.execute(
            "SELECT COALESCE(SUM(LENGTH(complement)), 0) FROM genes"
        ).fetchone()[0]

        # Compression tier distribution
        tier_counts = {0: 0, 1: 0, 2: 0}
        try:
            by_tier = cur.execute(
                "SELECT COALESCE(compression_tier, 0), COUNT(*) FROM genes "
                "GROUP BY COALESCE(compression_tier, 0)"
            ).fetchall()
            for r in by_tier:
                tier_counts[int(r[0])] = r[1]
        except Exception:
            pass  # Column may not exist yet on old schemas

        return {
            "total_genes": total,
            "open": chromatin_counts.get("OPEN", 0),
            "euchromatin": chromatin_counts.get("EUCHROMATIN", 0),
            "heterochromatin": chromatin_counts.get("HETEROCHROMATIN", 0),
            "total_chars_raw": total_raw,
            "total_chars_compressed": total_compressed,
            "compression_ratio": total_raw / max(total_compressed, 1),
            "compression_tiers": {
                "open_full": tier_counts[0],
                "euchromatin_summary": tier_counts[1],
                "heterochromatin_cold": tier_counts[2],
            },
        }

    # ── Get single document ─────────────────────────────────────────────

    def get_doc(self, gene_id: str) -> Optional[Gene]:
        row = self.conn.execute(
            "SELECT * FROM genes WHERE gene_id = ?", (gene_id,)
        ).fetchone()
        return self._row_to_gene(row) if row else None

    # ── Citation lookup (polymorphic with ShardedGenomeAdapter) ─────────

    def get_citation_rows(self, gene_ids: List[str]) -> Dict[str, Dict]:
        """Return source_id + parsed promoter tags for a batch of gene_ids.

        Used by /context and packet-builder citation construction. Mirrors
        ``ShardedGenomeAdapter.get_citation_rows`` so callers don't need to
        branch on adapter type. Missing ids are simply absent from the
        return map.

        Return shape (per gene_id):
            ``{"source_id": str, "domains": list[str], "entities": list[str]}``
        """
        if not gene_ids:
            return {}
        ph = ",".join("?" * len(gene_ids))
        rows = self.read_conn.cursor().execute(
            f"SELECT gene_id, source_id, promoter FROM genes "
            f"WHERE gene_id IN ({ph})",
            gene_ids,
        ).fetchall()
        out: Dict[str, Dict] = {}
        for r in rows:
            domains: List[str] = []
            entities: List[str] = []
            if r["promoter"]:
                try:
                    prom = parse_promoter(r["promoter"])
                    if prom is not None:
                        domains = list(prom.domains)
                        entities = list(prom.entities)
                except Exception:
                    pass
            out[r["gene_id"]] = {
                "source_id": r["source_id"] or "",
                "domains": domains,
                "entities": entities,
            }
        return out

    # ── Health logging ─────────────────────────────────────────────

    def log_health(
        self,
        query: str,
        ellipticity: float,
        coverage: float,
        density: float,
        freshness: float,
        genes_expressed: int,
        genes_available: int,
        status: str,
    ) -> None:
        """Record a health signal for historical tracking."""
        if self.read_only:
            log.debug("read_only: skipping log_health")
            return
        # Perf slice 2: log_health runs on EVERY build (intentionally
        # outside the read_only gate) — the first always-on unlocked
        # committer to be locked. Unlocked, two concurrent builds steal
        # each other's implicit transaction ("cannot commit - no
        # transaction is active") or hard-deadlock inside the INSERT on
        # py3.14 sqlite3. Remaining unlocked committers: W2.3 Phase A.
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO health_log (timestamp, query, ellipticity, coverage, "
                "density, freshness, genes_expressed, genes_available, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), query, ellipticity, coverage, density, freshness,
                 genes_expressed, genes_available, status),
            )
            self.conn.commit()

    def health_history(self, limit: int = 50) -> List[Dict]:
        """Return recent health signals, newest first."""
        rows = self.conn.execute(
            "SELECT timestamp, query, ellipticity, coverage, density, freshness, "
            "genes_expressed, genes_available, status "
            "FROM health_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "timestamp": r[0],
                "query": r[1],
                "ellipticity": r[2],
                "coverage": r[3],
                "density": r[4],
                "freshness": r[5],
                "genes_expressed": r[6],
                "genes_available": r[7],
                "status": r[8],
            }
            for r in rows
        ]

    def health_summary(self) -> Dict:
        """Aggregate health stats across all logged queries."""
        cur = self.conn.cursor()
        total = cur.execute("SELECT COUNT(*) FROM health_log").fetchone()[0]
        if total == 0:
            return {"total_queries": 0, "avg_ellipticity": 0, "status_counts": {}}

        avg = cur.execute(
            "SELECT AVG(ellipticity), AVG(coverage), AVG(density), AVG(freshness) "
            "FROM health_log"
        ).fetchone()

        status_counts = {}
        for row in cur.execute(
            "SELECT status, COUNT(*) FROM health_log GROUP BY status"
        ).fetchall():
            status_counts[row[0]] = row[1]

        return {
            "total_queries": total,
            "avg_ellipticity": round(avg[0], 4),
            "avg_coverage": round(avg[1], 4),
            "avg_density": round(avg[2], 4),
            "avg_freshness": round(avg[3], 4),
            "status_counts": status_counts,
        }

    # ── FTS5 index rebuild ────────────────────────────────────────────

    def rebuild_fts(self) -> int:
        """Rebuild the FTS5 index from all documents. Returns count indexed.

        Includes source_id + tags in the searchable content so
        tag-based knowledge survives rebuilds. At 100K+ documents this takes
        several seconds — prefer incremental sync for normal operation.
        """
        if not self._fts_available:
            log.warning("FTS5 not available — cannot rebuild")
            return 0

        import time as _time
        t0 = _time.time()
        cur = self.conn.cursor()

        # W2.3 Phase A: DELETE+INSERT+commit is one writer transaction —
        # hold the write lock for the whole section (all SQLite work).
        with self._write_lock:
            if self._fts_external:
                # Issue #338: external-content form — the special 'rebuild'
                # command repopulates the index from the backing view in one
                # pass (plain DELETE is not legal on this form).
                cur.execute("INSERT INTO genes_fts(genes_fts) VALUES ('rebuild')")
            else:
                # Clear and repopulate with enriched content
                cur.execute("DELETE FROM genes_fts")
                cur.execute(
                    "INSERT INTO genes_fts(gene_id, content, complement) "
                    "SELECT g.gene_id, "
                    "  COALESCE(g.source_id,'') || ' ' || "
                    "  COALESCE((SELECT GROUP_CONCAT(pi.tag_value, ' ') "
                    "    FROM promoter_index pi WHERE pi.gene_id = g.gene_id), '') "
                    "  || ' ' || g.content, "
                    "  COALESCE(g.complement, '') "
                    "FROM genes g"
                )
            self.conn.commit()
        count = cur.execute("SELECT COUNT(*) FROM genes_fts").fetchone()[0]
        elapsed = _time.time() - t0
        log.info("FTS5 index rebuilt: %d genes indexed in %.1fs", count, elapsed)
        return count

    # ── WAL refresh ────────────────────────────────────────────────

    def refresh(self) -> None:
        """Refresh WAL snapshot to see changes from external writers.

        Primary mechanism: commit() releases the implicit read transaction,
        forcing the next SELECT to start a new snapshot. This is the
        lightweight Tier 1 fix — no connection churn.

        Fallback: if the connection is in a bad state, close and reopen.
        """
        try:
            self._refresh_snapshot()
            # Verify the connection is healthy
            self.conn.execute("SELECT 1").fetchone()
        except Exception:
            log.warning("Snapshot refresh failed, reopening connection", exc_info=True)
            # W2.3 Phase A: hold the write lock across the close→reopen so
            # no writer holds a cursor on a dying connection (same rule as
            # vacuum()).
            with self._write_lock:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
                self.conn.row_factory = sqlite3.Row
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA busy_timeout=30000")
                self.conn.execute("PRAGMA journal_size_limit=67108864")  # 64 MB
                self._install_commit_trace()

    # ── Close ───────────────────────────────────────────────────────

    def checkpoint(self, mode: str = "PASSIVE") -> None:
        """
        Force a WAL checkpoint to flush data from WAL to main database.

        Modes:
            PASSIVE  — non-blocking, skips frames held by readers (~5ms)
            FULL     — blocks until all frames are checkpointed
            TRUNCATE — like FULL, then truncates WAL file to zero bytes

        Call periodically during bulk ingest to prevent data loss on crash.
        Recommended cadence: PASSIVE every 50 documents, TRUNCATE every 500.
        """
        mode = mode.upper()
        if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            mode = "PASSIVE"
        try:
            # Best-effort: release the reader's WAL snapshot before checkpointing
            # so a TRUNCATE can actually advance. Safe no-op when the reader is
            # already in autocommit mode (isolation_level=None).
            if self._reader is not None:
                try:
                    self._reader.commit()
                except Exception:
                    pass
            # W2.3 Phase A: wal_checkpoint writes on the shared writer
            # connection — unlocked it interleaves with an in-flight
            # writer's statements (py3.14 deadlock class). Lock exactly
            # the execute; the telemetry/log tail stays outside.
            with self._write_lock:
                row = self.conn.execute(
                    f"PRAGMA wal_checkpoint({mode})"
                ).fetchone()
            # Returns (busy, log_pages, checkpointed_pages). busy=1 means a
            # reader was holding a snapshot and the checkpoint was incomplete.
            if row and row[0]:
                log.warning(
                    "WAL checkpoint (%s) blocked by reader: %d/%d pages flushed",
                    mode, row[2] or 0, row[1] or 0,
                )
                try:
                    from .telemetry import genome_checkpoint_blocked_counter
                    genome_checkpoint_blocked_counter().add(
                        1, {"mode": mode}
                    )
                except Exception:
                    pass
            else:
                log.debug(
                    "WAL checkpoint (%s) completed: %d pages flushed",
                    mode, (row[2] if row else 0) or 0,
                )
        except Exception:
            log.warning("WAL checkpoint (%s) failed", mode, exc_info=True)

    def emit_wal_health_gauges(self) -> None:
        """Emit cymatix_genome_wal_size_bytes gauge for the current knowledge store WAL.

        Intended to be called from a background task every ~30 s. No-op for
        in-memory knowledge stores and when OTel is disabled (noop instruments drop the
        call silently). Never raises — failures are logged at DEBUG level.
        """
        if self.path == ":memory:":
            return
        try:
            wal_path = self.path + "-wal"
            wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
            from .telemetry import genome_wal_size_gauge
            genome_wal_size_gauge().set(wal_size)
        except Exception:
            log.debug("emit_wal_health_gauges failed", exc_info=True)

    def vacuum(self) -> Dict[str, int]:
        """
        Reclaim free pages from the knowledge store database.

        After large-scale operations (thinning, compaction, source-change
        repair) SQLite holds deleted pages until a VACUUM releases them.
        For a heavily-thinned knowledge store this can be 30-50% of the file size.

        This method:
          1. Checkpoints the WAL so all data is in the main DB file
          2. Closes the long-lived connection (VACUUM needs exclusive access)
          3. Runs VACUUM via a dedicated connection
          4. Reopens the long-lived connection

        Returns:
            dict with before/after sizes in bytes, and bytes reclaimed.

        Warning: VACUUM is a full-file rewrite — it temporarily doubles disk
        usage and blocks all writers. Run during maintenance windows.
        """
        import os
        path = self.path
        before = os.path.getsize(path) if os.path.exists(path) else 0

        # W2.3 Phase A: hold the write lock across the entire flush→close→
        # VACUUM→reopen sequence so no thread can be mid-statement on a
        # connection that is about to be closed underneath it. Everything
        # inside is SQLite work (VACUUM included) — the size probes stay
        # outside the lock.
        with self._write_lock:
            # Flush WAL and close the main connection
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.conn.commit()
            except Exception:
                log.warning("Pre-VACUUM WAL checkpoint failed", exc_info=True)
            try:
                self.conn.close()
            except Exception:
                pass

            # Run VACUUM on a fresh, autocommit connection
            try:
                vac_conn = sqlite3.connect(path)
                vac_conn.isolation_level = None  # autocommit — VACUUM requires it
                vac_conn.execute("VACUUM")
                vac_conn.close()
                log.info("VACUUM completed on %s", path)
            except Exception:
                log.warning("VACUUM failed", exc_info=True)

            # Reopen the long-lived connection
            self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            self.conn.execute("PRAGMA journal_size_limit=67108864")  # 64 MB
            self._install_commit_trace()

        after = os.path.getsize(path) if os.path.exists(path) else 0
        reclaimed = before - after

        return {
            "before_bytes": before,
            "after_bytes": after,
            "reclaimed_bytes": reclaimed,
            "reclaimed_pct": round(reclaimed / max(before, 1) * 100, 1),
        }

    # ── Cold-storage compression (ΣĒMA-based lifecycle tier tiers) ──────

    TIER_OPEN = 0           # Full fidelity — hot retrieval pool
    TIER_EUCHROMATIN = 1    # Summary + ΣĒMA — warm, reduced storage
    TIER_HETEROCHROMATIN = 2  # ΣĒMA + metadata only — cold, ~90% smaller

    def compute_density_score(self, gene: Gene) -> float:
        """
        Information density score for a document. Higher = more valuable.

        Combines:
          - Entity/domain tag count (tags richness)
          - Key-value extraction count (factual density)
          - Content length efficiency (short + rich > long + sparse)
          - Access count (usage signal from signals)

        Uses a content-length floor (100 chars) in the denominator to
        prevent tiny-content documents from producing nonsensical tag-density
        scores of 20+. See _DENSITY_CONTENT_LENGTH_FLOOR above.
        """
        tag_count = len(gene.promoter.domains) + len(gene.promoter.entities)
        kv_count = len(gene.key_values) if gene.key_values else 0
        # Floor the content length so a 30-char document with 5 tags doesn't
        # produce tag_density = 166 and break all downstream thresholds
        effective_len = max(len(gene.content), _DENSITY_CONTENT_LENGTH_FLOOR)

        # Normalize: tags per KB of (effective) content
        tag_density = tag_count / (effective_len / 1000.0)
        kv_density = kv_count / (effective_len / 1000.0)
        access = gene.epigenetics.access_count

        # Weighted combination (tag density dominates)
        score = (
            tag_density * 0.4
            + kv_density * 0.3
            + min(access / 10.0, 1.0) * 0.2
            + (1.0 if gene.complement and len(gene.complement) > 50 else 0.0) * 0.1
        )
        return score

    def apply_density_gate(self, gene: Gene) -> tuple[ChromatinState, str]:
        """
        Decide the lifecycle tier for a document at ingest time.

        Returns (chromatin_state, reason) — reason is one of:
            "deny_list"           : source path matches structural deny list
            "low_score_hetero"    : score below heterochromatin threshold
            "low_score_euchro"    : score below euchromatin threshold
            "access_rate_override": active in the windowed-rate sense
                                    (Phase 1 slice 2 — preferred when the
                                    document's recent_accesses buffer has data)
            "access_override"     : accessed >= _DENSITY_ACCESS_OVERRIDE
                                    times monotonically (legacy fallback for
                                    documents whose rate buffer is empty)
            "open"                : high score or unknown source, keep OPEN

        Never raises. Never touches the database. Pure decision function.

        The gate has three stages:
          1. Path deny list (fast-reject for structural noise)
          2. Access override (never demote frequently-used documents)
             2a. Windowed access-rate (preferred — sharper signal)
             2b. Monotonic access-count (legacy fallback for empty buffers)
          3. Score-based demotion (tag + KV density with recalibrated thresholds)

        Stages 2a and 2b run BEFORE the score check so that a document
        that's been touched multiple times can't be killed by a batch
        compact_genome sweep just because its static content is sparse.

        Stage 2a is a strict improvement on Stage 2b: it distinguishes
        "actively used right now" from "popular at some point in the
        past." Stage 2b remains as the fallback because all documents that
        pre-date the slice 1 schema change have empty recent_accesses
        buffers and need the monotonic counter to remain a valid signal
        until they're touched again under the slice 2 wiring.
        """
        # Stage 1: structural deny list (#207 item 5: honors this store's
        # configured deny_list_extra / locale_demotion_enabled)
        if is_denied_source(gene.source_id, deny_re=self._deny_re):
            return ChromatinState.HETEROCHROMATIN, "deny_list"

        # Stage 2a: windowed access-rate override (preferred)
        # Empty buffer → access_rate returns 0.0 → falls through to 2b cleanly.
        if gene.epigenetics:
            rate_threshold = _DENSITY_RATE_MIN_HITS / _DENSITY_RATE_WINDOW
            if gene.epigenetics.access_rate(_DENSITY_RATE_WINDOW) >= rate_threshold:
                return ChromatinState.OPEN, "access_rate_override"

        # Stage 2b: monotonic access-count override (legacy fallback)
        access = gene.epigenetics.access_count if gene.epigenetics else 0
        if access >= _DENSITY_ACCESS_OVERRIDE:
            return ChromatinState.OPEN, "access_override"

        # Stage 3: score-based demotion
        score = self.compute_density_score(gene)
        if score < _DENSITY_HETEROCHROMATIN_THRESHOLD:
            return ChromatinState.HETEROCHROMATIN, "low_score_hetero"
        if score < _DENSITY_EUCHROMATIN_THRESHOLD:
            return ChromatinState.EUCHROMATIN, "low_score_euchro"
        return ChromatinState.OPEN, "open"

    def compress_to_euchromatin(self, gene_id: str) -> bool:
        """
        Compress a document to EUCHROMATIN tier: drop raw content, keep summary.

        Keeps: complement, fragments, tags, signals, embedding, key_values
        Drops: content (replaced with pointer to source_id for unwinding)
        """
        # W2.3 Phase A: check-then-write + commit on the shared writer —
        # hold the write lock for the whole section (see log_health).
        with self._write_lock:
            cur = self.conn.cursor()
            row = cur.execute(
                "SELECT source_id, complement FROM genes WHERE gene_id = ?",
                (gene_id,),
            ).fetchone()
            if not row or not row["complement"]:
                return False

            # Issue #372: _write_unit = commit here standalone, SAVEPOINT
            # inside a deferred_commits() document scope (per-strand tier
            # demotion folds into the document's single commit).
            with self._write_unit():
                cur.execute(
                    "UPDATE genes SET content = ?, compression_tier = 1 WHERE gene_id = ?",
                    (f"[COMPRESSED:euchromatin] source={row['source_id'] or 'unknown'}", gene_id),
                )
        log.debug("Compressed gene %s to EUCHROMATIN", gene_id)
        return True

    def compress_to_heterochromatin(self, gene_id: str) -> bool:
        """
        Move a document to HETEROCHROMATIN tier (cold storage).

        As of 2026-04-10 (C.1 of B→C), this is **non-destructive**. The
        function only flips the ``chromatin`` and ``compression_tier``
        flags — it does NOT drop ``content``, ``complement``, ``codons``,
        SPLADE terms, or FTS5 index entries.

        Rationale: the lifecycle tier flag + ``WHERE chromatin < HETEROCHROMATIN``
        filter on all hot-tier retrieval paths already excludes demoted
        documents from normal ``/context`` queries. Destroying the underlying
        content eliminated any possibility of cold-tier retrieval (via
        ΣĒMA cosine similarity on the retained embedding) actually
        returning useful data — you'd match the embedding but have
        nothing to show.

        With content preserved, the cold-tier retrieval path added in C.2
        can reactivate demoted documents on-demand for queries that specifically
        need them. The storage cost is modest — SPLADE terms and FTS5
        index entries are small per-document — and the optional nature of
        cold-tier retrieval means hot queries are unaffected.

        Callers who explicitly want to reclaim disk space on a known-dead
        document can call ``delete_gene()`` instead. Heterochromatin is now
        strictly a **tier flag**, not a destructive compression.

        Keeps: everything (content, complement, fragments, SPLADE, FTS5)
        Flips: ``chromatin = 2``, ``compression_tier = 2``
        """
        # W2.3 Phase A: check-then-write + commit on the shared writer —
        # hold the write lock for the whole section. Cache/dense-matrix
        # invalidation stays OUTSIDE the lock (lock-ordering rule: never
        # take _dense_matrix_lock while holding _write_lock — mirrors
        # upsert_doc / delete_gene).
        with self._write_lock:
            cur = self.conn.cursor()
            row = cur.execute(
                "SELECT source_id FROM genes WHERE gene_id = ?",
                (gene_id,),
            ).fetchone()
            if not row:
                return False

            # Issue #372: _write_unit = commit here standalone, SAVEPOINT
            # inside a deferred_commits() document scope.
            with self._write_unit():
                cur.execute(
                    "UPDATE genes SET chromatin = 2, compression_tier = 2 "
                    "WHERE gene_id = ?",
                    (gene_id,),
                )
        # Document moved hot → cold — both tier caches (and the
        # sema-vectorless auto-gate flags) are now stale.
        self.invalidate_sema_cache()
        self.invalidate_cold_sema_cache()
        # Stage 2: hot/cold transitions change the dense matrix membership too.
        self._invalidate_dense_matrix()
        log.debug("Moved gene %s to HETEROCHROMATIN (non-destructive)", gene_id)
        return True

    def delete_gene(self, gene_id: str) -> bool:
        """Hard-delete a document and every row that references it.

        The reclaim-disk-space counterpart to the non-destructive
        ``compress_to_heterochromatin()`` (whose docstring has promised this
        API since 2026-04-10). Removes, in one locked transaction:

          - the ``genes`` row itself
          - per-gene index rows: FTS5, promoter_index, entity_graph,
            path_key_index, filename_index, splade_terms
          - ``symbol_defs`` rows (WS2 review FIX-2 — symbol rows must not
            outlive their gene)
          - ``gene_relations`` edges touching the gene in EITHER direction
            (SYMBOL_REF, CHUNK_OF, and typed relations alike — any edge
            touching a dead gene is an orphan edge)
          - ``harmonic_links`` edges in either direction

        Returns True if a genes row was deleted, False if the id was unknown
        (index cleanup still runs, so calling this on an already-deleted
        gene sweeps any leftover rows).
        """
        # Optional tables (old schemas / disabled features) — guarded so a
        # missing table degrades to a no-op instead of aborting the delete.
        optional_tables = (
            "promoter_index", "entity_graph", "path_key_index",
            "filename_index", "splade_terms", "symbol_defs",
        )
        with self._write_lock:
            cur = self.conn.cursor()
            try:
                if self._fts_available:
                    if self._fts_external:
                        # Issue #338: external-content delete discipline —
                        # hand the index the prior indexed values via the
                        # special 'delete' insert, read from the backing
                        # view BEFORE the genes/promoter rows go away
                        # (afterwards the token stream is unrecoverable).
                        from .storage.indexes import (
                            fetch_fts_source_row,
                            fts_delete_row,
                        )
                        prior_fts_row = fetch_fts_source_row(cur, gene_id)
                        if prior_fts_row is not None:
                            fts_delete_row(cur, prior_fts_row)
                    else:
                        cur.execute(
                            "DELETE FROM genes_fts WHERE gene_id = ?", (gene_id,)
                        )
                for table in optional_tables:
                    try:
                        cur.execute(
                            f"DELETE FROM {table} WHERE gene_id = ?", (gene_id,)
                        )
                    except sqlite3.OperationalError:
                        pass  # table absent on this schema
                cur.execute(
                    "DELETE FROM gene_relations "
                    "WHERE gene_id_a = ? OR gene_id_b = ?",
                    (gene_id, gene_id),
                )
                try:
                    cur.execute(
                        "DELETE FROM harmonic_links "
                        "WHERE gene_id_a = ? OR gene_id_b = ?",
                        (gene_id, gene_id),
                    )
                except sqlite3.OperationalError:
                    pass
                res = cur.execute(
                    "DELETE FROM genes WHERE gene_id = ?", (gene_id,)
                )
                deleted = bool(res.rowcount)
                self.conn.commit()
            except Exception:
                # Mirror upsert_doc: discard the partial transaction so it
                # cannot ride along with the next caller's commit.
                self.conn.rollback()
                raise

        if deleted:
            # Tier caches (and the sema-vectorless auto-gate flags) indexed
            # the dead gene.
            self.invalidate_sema_cache()
            self.invalidate_cold_sema_cache()
            self._invalidate_dense_matrix()
            log.debug("Deleted gene %s (+ index, symbol, and edge rows)", gene_id)
        return deleted

    def _sweep_symbol_orphans(self) -> int:
        """Delete symbol rows whose gene no longer exists (WS2 review FIX-2).

        Genes removed out-of-band (external cleanup scripts, raw SQL) leave
        ``symbol_defs`` rows and SYMBOL_REF ``gene_relations`` edges behind;
        this sweep runs from both compaction passes (``compact()`` /
        ``compact_genome()``, i.e. the /admin/compact path). Caller owns the
        commit. Returns rows deleted.
        """
        from .schemas import StructuralRelation as _SR

        cur = self.conn.cursor()
        swept = 0
        try:
            swept += cur.execute(
                "DELETE FROM symbol_defs "
                "WHERE gene_id NOT IN (SELECT gene_id FROM genes)"
            ).rowcount or 0
        except sqlite3.OperationalError:
            pass  # symbol_defs absent on this schema
        swept += cur.execute(
            "DELETE FROM gene_relations WHERE relation = ? AND ("
            "  gene_id_a NOT IN (SELECT gene_id FROM genes)"
            "  OR gene_id_b NOT IN (SELECT gene_id FROM genes))",
            (int(_SR.SYMBOL_REF),),
        ).rowcount or 0
        if swept:
            log.info("Compaction: swept %d orphaned symbol rows", swept)
        return swept

    def compact_genome(self, dry_run: bool = False) -> Dict:
        """
        Run a compaction sweep: apply the density gate to every currently-OPEN
        document and demote those that fail it.

        Shares gate logic with ingest-time `apply_density_gate()`, so a document
        that would be demoted by a fresh ingest will also be demoted by a
        retroactive sweep. The three stages are the same:
          1. Structural deny list (Steam, build artifacts, lockfiles, etc.)
          2. Access-count override (access_count >= 5 keeps document OPEN)
          3. Score-based thresholds (< 0.5 hetero, < 1.0 euchro, else open)

        Only operates on documents currently at compression_tier = 0 (OPEN).
        Already-demoted documents are left alone.

        When ``dry_run=True``, returns the same stats without writing to
        the DB. Useful for previewing the impact before running the sweep
        against a live knowledge store.

        Returns a dict with:
            scanned               : number of OPEN documents examined
            to_heterochromatin    : count that would go to HETEROCHROMATIN
            to_euchromatin        : count that would go to EUCHROMATIN
            kept_open             : count that would stay OPEN
            skipped_no_embedding  : count skipped because no ΣĒMA vector exists
            by_reason             : dict of reason counts (deny_list, low_score_*, etc.)
        """
        cur = self.conn.cursor()
        # W2.3 Phase A: the scan is one execute on the shared writer — lock
        # it. The gate loop below is NOT held under the lock (pure-CPU row
        # parsing); the per-row demotions go through compress_to_* which
        # each take the lock around their own execute+commit.
        with self._write_lock:
            rows = cur.execute(
                "SELECT gene_id, content, complement, codons, promoter, "
                "epigenetics, chromatin, embedding, source_id, key_values, "
                "compression_tier "
                "FROM genes WHERE compression_tier = 0"
            ).fetchall()

        stats = {
            "scanned": len(rows),
            "to_euchromatin": 0,
            "to_heterochromatin": 0,
            "kept_open": 0,
            "skipped_no_embedding": 0,
            "by_reason": {},
        }

        for r in rows:
            gene = self._compact_row_to_gene(r)
            if gene is None:
                continue

            new_state, reason = self.apply_density_gate(gene)
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1

            if new_state == ChromatinState.OPEN:
                stats["kept_open"] += 1
                continue

            # Deny-listed documents ALWAYS demote — with or without embedding.
            # The whole point of the deny list is "this is structural
            # noise we never want to retrieve again." compress_to_heterochromatin
            # only needs source_id (it strips content, complement, fragments,
            # SPLADE, and FTS). The pre-existing no-embedding guard below
            # was a safety for score-based demotions, where a document might
            # later turn out to be useful and we want the ΣĒMA vector
            # available for reactivation. That reasoning doesn't apply
            # to structural deny-list hits. Previously this guard cost
            # us ~40% of expected demotions on the 2026-04-10 knowledge store
            # (3358 documents with no embeddings, mostly from pre-ΣĒMA
            # bulk ingests like ingest_steam.py).
            if reason == "deny_list":
                if not dry_run:
                    self.compress_to_heterochromatin(gene.gene_id)
                stats["to_heterochromatin"] += 1
                continue

            # Score-based demotions keep the embedding guard — these
            # documents might turn out to be useful later, so we want the
            # ΣĒMA vector available for reactivation via cosine similarity.
            has_embedding = r["embedding"] is not None
            if not has_embedding:
                stats["skipped_no_embedding"] += 1
                stats["kept_open"] += 1
                continue

            if new_state == ChromatinState.EUCHROMATIN:
                if gene.complement and len(gene.complement) > 30:
                    if not dry_run:
                        self.compress_to_euchromatin(gene.gene_id)
                    stats["to_euchromatin"] += 1
                else:
                    # No summary available to preserve; fall through to
                    # heterochromatin which doesn't need one
                    if not dry_run:
                        self.compress_to_heterochromatin(gene.gene_id)
                    stats["to_heterochromatin"] += 1
            elif new_state == ChromatinState.HETEROCHROMATIN:
                if not dry_run:
                    self.compress_to_heterochromatin(gene.gene_id)
                stats["to_heterochromatin"] += 1

        if not dry_run:
            # WS2 review FIX-2: sweep orphaned symbol rows (genes removed
            # out-of-band) whenever the /admin/compact sweep runs for real.
            # W2.3 Phase A: orphan sweep + commit is one writer transaction.
            with self._write_lock:
                self._sweep_symbol_orphans()
                self.conn.commit()
            self.checkpoint("PASSIVE")

        log.info(
            "Compaction sweep (%s): scanned=%d open=%d euchromatin=%d heterochromatin=%d skipped_no_emb=%d",
            "dry-run" if dry_run else "applied",
            stats["scanned"], stats["kept_open"],
            stats["to_euchromatin"], stats["to_heterochromatin"],
            stats["skipped_no_embedding"],
        )
        return stats

    def _compact_row_to_gene(self, row) -> Optional[Gene]:
        """Convert a compact database row to a Document object. Returns None on error.

        Pre-existing bug fix (2026-04-10): the original version passed
        key_values=None when the DB column was NULL, but Gene.key_values
        is declared as list[str] and Pydantic rejects None. Any document
        without extracted KVs (~35% of the 2026-04-10 knowledge store sample)
        silently failed parsing, causing compact_genome to skip them.
        Now we pass [] as the empty-list fallback, matching how other
        list fields (fragments) are handled.
        """
        try:
            def _opt(key: str, default=None):
                try:
                    return row[key]
                except (IndexError, KeyError):
                    return default

            return Gene(
                gene_id=row["gene_id"],
                content=row["content"] or "",
                complement=row["complement"],
                codons=json_loads(row["codons"]) if row["codons"] else [],
                promoter=parse_promoter(row["promoter"]),
                epigenetics=parse_epigenetics(row["epigenetics"]),
                chromatin=ChromatinState(row["chromatin"]),
                embedding=decode_embedding(row["embedding"]),
                source_id=row["source_id"],
                repo_root=_opt("repo_root"),
                source_kind=_opt("source_kind"),
                observed_at=_opt("observed_at"),
                mtime=_opt("mtime"),
                content_hash=_opt("content_hash"),
                volatility_class=_opt("volatility_class"),
                authority_class=_opt("authority_class"),
                support_span=_opt("support_span"),
                last_verified_at=_opt("last_verified_at"),
                key_values=json_loads(row["key_values"]) if row["key_values"] else [],
            )
        except Exception:
            log.debug("Failed to parse gene row %s", row["gene_id"], exc_info=True)
            return None

    def close(self) -> None:
        self.checkpoint("TRUNCATE")  # Flush all WAL data before closing
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
        # Perf slice 2: close every per-thread reader minted by read_conn.
        for _c in getattr(self, "_thread_reader_registry", []) or []:
            try:
                _c.close()
            except Exception:
                pass
        self.conn.close()

    # ── Legacy method aliases (R3 Stage C; see docs/ROSETTA.md) ─────
    # Each alias points to the *same function object* as the canonical
    # method ("upsert_gene is upsert_doc" is True). External callers
    # that still call .upsert_doc() / .query_docs() / .get_doc()
    # keep working unchanged (notably api.py::gene_get and the
    # cross_store_import / sharding paths).
    #
    # SQL column names (gene_id, etc.) are unchanged — only the Python
    # method-name surface moved.
    upsert_gene              = upsert_doc
    query_genes              = query_docs
    query_genes_ann          = query_docs_ann
    query_genes_dense_recall = query_docs_dense_recall
    get_gene                 = get_doc


# R3 legacy alias — pre-R3 callers still import Genome. Identity preserved:
# Genome is KnowledgeStore. SQL table/column names (genes, gene_id, etc) are
# the on-disk contract and remain untouched. See docs/ROSETTA.md.
Genome = KnowledgeStore

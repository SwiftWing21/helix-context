"""
HelixContextManager -- The cell nucleus.

Orchestrates the full DNA context pipeline per turn:
    1. Extract promoter signals from query (heuristic, no model)
    2. Express -- find relevant genes via promoter matching + co-activation
    3. Re-rank -- score candidates via ribosome (CPU, optional)
    4. Splice -- trim introns, keep exons (CPU, batched)
    5. Assemble -- build the 3k ribosome prompt + 6k expressed context window
    6. Replicate -- pack query+response exchange into genome (background)

Token budget:
    3k  = ribosome decoder prompt (fixed, tells big model how to read codons)
    6k  = expressed context (codon-encoded, spliced)
    600k = genome (cold storage, never fully loaded)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .accel import extract_query_signals, estimate_tokens
from .codons import CodonChunker, CodonEncoder
from .config import HelixConfig
from .exceptions import PromoterMismatch
from .genome import Genome
from .budget_zone import is_enabled as _budget_zone_is_enabled, zone_cap as _budget_zone_cap
from .headroom_bridge import compress_text
from . import legibility
from . import session_delivery as _session_delivery
from .ribosome import DisabledBackend, LiteLLMBackend, Ribosome, OllamaBackend
from .provenance import apply_metadata_hints, apply_provenance
from .schemas import (
    ChromatinState,
    ContextHealth,
    ContextWindow,
    EpigeneticMarkers,
    Gene,
    PromoterTags,
    StructuralRelation,
)

log = logging.getLogger("helix.context_manager")

# Thread pool for running sync ribosome calls from async context
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="helix-ribosome")


# -- Ribosome decoder prompt (3k fixed, tells the big model how to read context) --

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

DECODER_MODES = {
    "full": DECODER_FULL,
    "condensed": DECODER_CONDENSED,
    "minimal": DECODER_MINIMAL,
    "none": DECODER_NONE,
    "moe": DECODER_MOE,
}

# Keep backward compatibility
RIBOSOME_DECODER = DECODER_FULL


class HelixContextManager:
    """
    Main orchestrator. Sits between the client and the upstream LLM.

    Usage:
        helix = HelixContextManager(config)
        helix.ingest("some long document")

        # Per turn:
        window = helix.build_context("user query")
        # Inject window into the LLM request

        # After response:
        helix.learn("user query", "assistant response")
    """

    def __init__(self, config: HelixConfig):
        self.config = config

        # Activity tracking for GET /admin/components.
        # Bumped on every /context and /ingest call by server.py. Used to
        # derive running/idle status for the launcher's tools panel.
        import time as _time
        self._last_activity_ts: float = _time.time()

        # Token counter (session + lifetime). Persisted next to genome.db so
        # the lifetime counter survives restarts. See helix_context/metrics.py
        # and the /metrics/tokens endpoint.
        from pathlib import Path as _Path
        from .metrics import TokenCounter
        _genome_path = _Path(config.genome.path)
        if str(_genome_path) == ":memory:":
            # In-memory tests: keep metrics in-memory too (write to a tmp path
            # that we won't actually flush; persistence is opt-in via flush()).
            import tempfile as _tempfile
            _metrics_path = _Path(_tempfile.gettempdir()) / "helix_metrics_test.json"
        else:
            _metrics_path = _genome_path.parent / "metrics.json"
        self.token_counter: TokenCounter = TokenCounter(persist_path=_metrics_path)

        # ΣĒMA codec (optional — loaded if sentence-transformers available)
        self._sema_codec = None
        try:
            from .sema import SemaCodec
            self._sema_codec = SemaCodec()
            log.info("ΣĒMA codec loaded — semantic retrieval enabled")
        except ImportError:
            log.info("sentence-transformers not installed — ΣĒMA disabled")
        except Exception:
            log.warning("ΣĒMA codec failed to load", exc_info=True)

        # Genome (SQLite storage) — swapped for a ShardedGenomeAdapter when
        # HELIX_USE_SHARDS=1 and the configured path is a routing DB. Writes
        # become no-ops in that mode; suitable for read-heavy serving and
        # benchmarks until ingest-time sharding (spec Task 6) lands.
        from .sharding import open_read_source
        self.genome = open_read_source(
            genome_path=config.genome.path,
            synonym_map=config.synonym_map,
            sema_codec=self._sema_codec,
            splade_enabled=config.ingestion.splade_enabled,
            entity_graph=config.ingestion.entity_graph,
            sr_enabled=config.retrieval.sr_enabled,
            sr_gamma=config.retrieval.sr_gamma,
            sr_k_steps=config.retrieval.sr_k_steps,
            sr_weight=config.retrieval.sr_weight,
            sr_cap=config.retrieval.sr_cap,
            seeded_edges_enabled=config.retrieval.seeded_edges_enabled,
            filename_anchor_enabled=(
                config.retrieval.filename_anchor_enabled
                or os.environ.get("HELIX_FILENAME_ANCHOR_ENABLED", "").lower()
                in {"1", "true", "yes", "on"}
            ),
            filename_anchor_weight=config.retrieval.filename_anchor_weight,
        )

        # Replication manager (distributed genome clones)
        self._replication_mgr = None
        if config.genome.replicas:
            from .replication import ReplicationManager
            self._replication_mgr = ReplicationManager(
                master=config.genome.path,
                replicas=config.genome.replicas,
                sync_interval=config.genome.replica_sync_interval,
            )
            self.genome.set_replication_manager(self._replication_mgr)

        # Chunker (deterministic text splitting)
        self.chunker = CodonChunker(max_chars_per_strand=4000)
        self.encoder = CodonEncoder()

        # Ribosome (small model codec) — explicit opt-in only. Legacy/default
        # Ollama ribosome config stays in the file for future use but is
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
            try:
                from .deberta_backend import DeBERTaRibosome
                ollama_backend = OllamaBackend(
                    model=config.ribosome.model,
                    base_url=config.ribosome.base_url,
                    timeout=config.ribosome.timeout,
                    keep_alive=config.ribosome.keep_alive,
                    warmup=config.ribosome.warmup,
                )
                ollama_ribosome = Ribosome(
                    backend=ollama_backend,
                    encoder=self.encoder,
                    splice_aggressiveness=config.budget.splice_aggressiveness,
                )
                self.ribosome = DeBERTaRibosome(
                    rerank_model_path=config.ribosome.rerank_model_path,
                    splice_model_path=config.ribosome.splice_model_path,
                    nli_model_path=config.ribosome.nli_model_path,
                    ollama_ribosome=ollama_ribosome,
                    device=config.ribosome.device,
                    splice_threshold=config.ribosome.splice_threshold,
                    nli_splice_bonus=config.ribosome.nli_splice_bonus,
                    nli_splice_penalty=config.ribosome.nli_splice_penalty,
                    rerank_pretrained=config.ingestion.rerank_model,
                )
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
        model_name = config.ribosome.active_model.lower()
        is_moe = any(model_name.startswith(fam) for fam in MOE_MODEL_FAMILIES)
        is_small = SMALL_MODEL_PATTERNS.get(model_name, 999) <= SMALL_MODEL_THRESHOLD_B
        self._is_moe = is_moe or is_small
        if self._is_moe:
            log.info(
                "Answer-slate mode enabled for %s (%s)",
                config.ribosome.model,
                "MoE/SWA" if is_moe else "sub-4B",
            )

        # Pending replication buffer -- genes from background replication
        # that haven't committed to SQLite yet. Checked during Step 2
        # so follow-up queries don't lose context from the previous turn.
        self._pending: List[Gene] = []
        self._pending_lock = threading.Lock()

        # Cymatics (frequency-domain re_rank + splice, replaces LLM calls)
        self._use_cymatics = config.cymatics.enabled
        if self._use_cymatics:
            from .cymatics import aggressiveness_to_peak_width
            self._cymatics_peak_width = aggressiveness_to_peak_width(
                config.budget.splice_aggressiveness
            )
        else:
            self._cymatics_peak_width = 3.0

        # TCM session context (Howard & Kahana 2002 temporal drift)
        self._tcm_session = None
        try:
            from .tcm import SessionContext
            self._tcm_session = SessionContext(n_dims=20, beta=0.5)
            log.info("TCM session context initialized (20D, beta=0.5)")
        except Exception:
            log.debug("TCM not available", exc_info=True)

        # Shadow pool (soft elimination — genes cut from top-k keep
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

        # Session buffer -- accumulates query+response pairs for consolidation
        self._session_buffer: List[Tuple[str, str]] = []
        self._session_buffer_lock = threading.Lock()
        self._session_learn_count = 0
        self._consolidation_threshold = 10  # auto-consolidate every N learns

        # Compaction timer
        self._last_compact = time.time()

    # -- Ingest: add new content to the genome -------------------------

    def ingest(self, content: str, content_type: str = "text", metadata: Optional[Dict] = None) -> List[str]:
        """
        Pack new content and store in the genome.
        Call for documents, files, or conversation history.
        Returns list of gene_ids created.
        """
        strands = self.chunker.chunk(content, content_type=content_type, metadata=metadata)
        gene_ids = []
        total_strands = len(strands)

        # Accept either metadata["path"] or metadata["source_id"] — the HTTP
        # /ingest contract historically documented both, but only "path" was
        # honored. Alias so callers using "source_id" get proper provenance
        # (source_kind, volatility_class) populated downstream.
        source_path = None
        if metadata:
            source_path = metadata.get("path") or metadata.get("source_id")

        # Batch-encode ΣĒMA vectors if codec available
        sema_vectors = None
        if self._sema_codec is not None:
            try:
                texts = [s.content[:1000] for s in strands]  # Cap for encoder
                sema_vectors = self._sema_codec.encode_batch(texts)
            except Exception:
                log.debug("ΣĒMA batch encoding failed, skipping")

        use_cpu = (
            self._cpu_tagger is not None
            and self.config.ingestion.backend in ("cpu", "hybrid")
        )

        for i, strand in enumerate(strands):
            if use_cpu:
                gene = self._cpu_tagger.pack(
                    strand.content,
                    content_type=content_type,
                    source_id=source_path,
                    sequence_index=strand.sequence_index,
                )
            else:
                gene = self.ribosome.pack(strand.content, content_type=content_type)
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

            # Density gate now lives in genome.upsert_gene() itself so that
            # bulk ingest scripts (ingest_steam.py, ingest_all.py, etc.)
            # that call upsert_gene directly also respect it. The gate
            # reads the final chromatin state back onto the gene object
            # and sets compression_tier accordingly during the INSERT.
            # See helix_context/genome.py:apply_density_gate for the logic.
            gid = self.genome.upsert_gene(gene)
            gene_ids.append(gid)

            # If the gate demoted the gene to heterochromatin, the content
            # column is still populated — compress_to_heterochromatin()
            # drops it and strips SPLADE/FTS indices. Run this post-insert
            # for consistency with the historical behavior.
            if gene.chromatin == ChromatinState.HETEROCHROMATIN and gene.embedding is not None:
                self.genome.compress_to_heterochromatin(gid)
            elif gene.chromatin == ChromatinState.EUCHROMATIN:
                self.genome.compress_to_euchromatin(gid)

        # Layered fingerprints: create a parent gene when a file chunks
        # into N >= 2 strands. Parent aggregates child fingerprints at
        # query time so multi-chunk hits surface the whole file.
        # See docs/FUTURE/LAYERED_FINGERPRINTS.md.
        if len(gene_ids) >= 2 and source_path:
            try:
                parent_gid = self._upsert_parent_gene(
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

        log.info("Ingested %d strands from %s content (%d chars)",
                 len(gene_ids), content_type, len(content))
        return gene_ids

    @staticmethod
    def _make_parent_gene_id(source_path: str) -> str:
        """Deterministic parent gene_id from source path.

        Uses a distinct hash input (suffix "::parent") so parent IDs
        can't collide with content-hashed child gene_ids.
        """
        return hashlib.sha256(
            (source_path + "::parent").encode("utf-8")
        ).hexdigest()[:16]

    def _upsert_parent_gene(
        self,
        source_path: str,
        child_gene_ids: List[str],
        original_content: str,
    ) -> Optional[str]:
        """Create or refresh a parent gene for a multi-chunk file.

        Parent shape:
            gene_id      — deterministic from source_path
            content      — first 1024 chars of original file
            codons       — ordered list of child gene_ids (reassembly key)
            key_values   — [chunk_count=N, total_size_bytes=B, is_parent=true]
            is_fragment  — False
            sequence_index = -1 (file-level sentinel)

        Also inserts CHUNK_OF edges from each child to the parent.
        """
        parent_gid = self._make_parent_gene_id(source_path)
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
        self.genome.upsert_gene(parent, apply_gate=False)

        edges = [
            (child_gid, parent_gid, int(StructuralRelation.CHUNK_OF), 1.0)
            for child_gid in child_gene_ids
        ]
        self.genome.store_relations_batch(edges)
        return parent_gid

    async def ingest_async(self, content: str, content_type: str = "text", metadata: Optional[Dict] = None) -> List[str]:
        """Async wrapper for ingest -- runs ribosome calls in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.ingest, content, content_type, metadata)

    # -- Build context: the main per-turn operation --------------------

    def _should_use_slate(self, downstream_model: Optional[str] = None) -> bool:
        """Check if answer-slate mode should activate for this request.

        Activates for:
          1. Server-level MoE detection (ribosome model is gemma4 etc.)
          2. Per-request downstream model detection (sub-4B or MoE family)
        """
        if self._is_moe:
            return True
        if downstream_model:
            dm = downstream_model.lower()
            if any(dm.startswith(fam) for fam in MOE_MODEL_FAMILIES):
                return True
            if SMALL_MODEL_PATTERNS.get(dm, 999) <= SMALL_MODEL_THRESHOLD_B:
                return True
        return False

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
    ) -> ContextWindow:
        """
        Build the active context window for a query.
        Runs the 5-step expression pipeline (Steps 1-5).

        Args:
            downstream_model: optional model name from the proxy request,
                used for per-request MoE/small-model detection.
            include_cold: per-request override for cold-tier retrieval.
                ``None`` (default) honors the ``[context] cold_tier_enabled``
                config flag in helix.toml. ``True`` forces cold-tier on,
                ``False`` forces it off. Plumbed from the /context endpoint's
                ``include_cold`` body parameter.
            session_context: optional dict carrying the caller's working
                context — typically ``{"active_project": "helix-context",
                "active_files": ["helix_context/genome.py", ...]}``. The
                path-tokens of these are appended to the entity list so
                the path_key_index tier in ``query_genes`` can fire on
                compound (project, key) pairs even when the user's natural
                query doesn't restate the project name. This is the
                "implicit THIS project" signal that real users have but
                synthetic benches lack. None = no session context, which
                preserves the previous behaviour exactly.
        """
        self._maybe_compact()

        # Resolve per-request decoder override without mutating shared
        # instance state (prevents races on the singleton manager under
        # concurrent /context calls). None falls back to self._decoder_prompt.
        if decoder_override and decoder_override in DECODER_MODES:
            effective_decoder_prompt = DECODER_MODES[decoder_override]
        else:
            effective_decoder_prompt = self._decoder_prompt

        # Reset per-call cold-tier markers (set by _express when cold fires)
        self._last_cold_tier_used = False
        self._last_cold_tier_count = 0

        max_genes = self.config.budget.max_genes_per_turn

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

        # Step 0: Query intent expansion (LLM-based, cached)
        # Restates the query with expanded keywords BEFORE promoter lookup.
        # This sharpens the initial frequency so retrieval falls into the
        # right gravity well instead of optimizing the wrong one.
        expanded_query, domains, entities = self._prepare_query_signals(
            query,
            session_context=session_context,
            expand_query=True,
        )

        # Step 2: Express (genome query + pending buffer + optional cold tier)
        candidates = self._express(
            domains, entities, max_genes,
            query_text=query, include_cold=include_cold, party_id=party_id,
            read_only=read_only,
        )

        if not candidates:
            empty_health = ContextHealth(
                ellipticity=0.0,
                coverage=0.0,
                density=0.0,
                freshness=0.0,
                genes_available=self.genome.stats().get("total_genes", 0),
                genes_expressed=0,
                status="denatured" if self.genome.stats().get("total_genes", 0) > 0 else "sparse",
            )
            return ContextWindow(
                ribosome_prompt=effective_decoder_prompt,
                expressed_context="(no relevant context found in genome)",
                total_estimated_tokens=estimate_tokens(effective_decoder_prompt),
                compression_ratio=1.0,
                context_health=empty_health,
                metadata={"query": query, "genes_expressed": 0},
            )

        candidates, _ = self._apply_candidate_refiners(
            query,
            candidates,
            max_genes,
            use_cymatics=True,
            use_harmonic_bin=True,
            use_tcm=True,
            allow_rerank=True,
        )

        # Dynamic budget tiers — size the expression window based on
        # retrieval confidence instead of always sending max_genes.
        #
        # The insight: on a CURATED query ("what port does helix use?") the
        # top gene will score 5-10x higher than #12. Sending 12 genes for a
        # query with an obvious winner wastes 91% of the budget on padding
        # and dilutes the small model's attention.
        #
        # Tiers (confidence = top_score / mean_score ratio):
        #   - TIGHT   (ratio >= 3.0): top 3 genes   — ~6K total tokens
        #   - FOCUSED (ratio 1.8-3.0): top 6 genes  — ~9K total tokens
        #   - BROAD   (ratio < 1.8):  top max_genes — ~15K total tokens
        #
        # Score-gate floor: always drop genes scoring < 15% of top score.
        budget_tier = "broad"  # default
        budget_tokens_est = 15000
        if len(candidates) > 3:
            all_scores = self.genome.last_query_scores
            # Compute ratio over CANDIDATES only, not all scored genes
            # (all_scores includes genes that didn't make top-N cut,
            # dragging down mean and inflating ratio → always "tight")
            candidate_ids = {g.gene_id for g in candidates}
            scores = {gid: s for gid, s in (all_scores or {}).items() if gid in candidate_ids}
            if scores and any(scores.values()):
                top_score = max(scores.values())
                mean_score = sum(scores.values()) / len(scores) if scores else 1.0
                ratio = top_score / max(mean_score, 0.01)

                # Hard floor: drop anything below 15% of top
                # Shadow scores: preserve cut genes' scores with 0.5x weight
                # so Lagrange check and harmonic binning can pull them back
                # if the landscape changes downstream.
                floor = top_score * 0.15
                gated = [g for g in candidates if scores.get(g.gene_id, 0) >= floor]
                shadow_pool: List[Gene] = [g for g in candidates if scores.get(g.gene_id, 0) < floor]
                if len(gated) >= 3:
                    candidates = gated

                # Confidence tiering (with shadow pool tracking)
                #
                # Absolute floors prevent the ratio from triggering TIGHT/FOCUSED
                # when ALL candidates are weak. Before the floor, a query with
                # top_score=1.2, mean=0.4 (ratio=3.0) got the same "tight" treatment
                # as top=8.5, mean=2.8 — even though the first is "retrieval is
                # uncertain, widen the net" and the second is "we found it, send 3."
                # Empirically: on N=50 KV-harvest bench (2026-04-12), 45/50 failed
                # queries landed in tight mode with top_score < 3.0. Adding the
                # absolute floor keeps weak-signal queries in BROAD mode where
                # the larger candidate set gives them a recall chance.
                TIGHT_SCORE_FLOOR = 5.0
                FOCUSED_SCORE_FLOOR = 2.5
                if ratio >= 3.0 and top_score >= TIGHT_SCORE_FLOOR and len(candidates) >= 3:
                    # High confidence — top gene dominates AND is strong, send 3
                    shadow_pool = shadow_pool + candidates[3:]
                    candidates = candidates[:3]
                    budget_tier = "tight"
                    budget_tokens_est = 6000
                elif ratio >= 1.8 and top_score >= FOCUSED_SCORE_FLOOR and len(candidates) >= 6:
                    # Moderate confidence — narrow to 6
                    shadow_pool = shadow_pool + candidates[6:]
                    candidates = candidates[:6]
                    budget_tier = "focused"
                    budget_tokens_est = 9000
                # else: broad — keep current up-to-max_genes set
                #   (weak absolute scores or weak ratio → widen the net)

                # Stash shadow pool for Lagrange check (#3)
                self._last_shadow_pool = shadow_pool
                self._last_shadow_scores = {
                    g.gene_id: scores.get(g.gene_id, 0) * 0.5
                    for g in shadow_pool
                }

                log.debug(
                    "Dynamic budget: tier=%s ratio=%.2f top=%.1f mean=%.1f genes=%d shadow=%d",
                    budget_tier, ratio, top_score, mean_score, len(candidates), len(shadow_pool),
                )

                # Telemetry: budget-tier distribution over queries.
                try:
                    from .telemetry import budget_tier_counter
                    budget_tier_counter().add(
                        1, attributes={"tier": budget_tier},
                    )
                except Exception:  # pragma: no cover
                    pass

                # Lagrange point check: a gene in the shadow pool with HIGH
                # standalone score but LOW co-activation with the winners is
                # being deflected by cluster gravity, not rejected on merit.
                # Pull it back if its standalone > 70% of winners' floor AND
                # its co-activation overlap with winners is < 20%.
                if shadow_pool and len(candidates) >= 3 and budget_tier != "broad":
                    try:
                        winner_ids = {g.gene_id for g in candidates}
                        winner_coact: set[str] = set()
                        for g in candidates:
                            winner_coact.update(g.epigenetics.co_activated_with or [])
                        winner_floor = min(scores.get(g.gene_id, 0) for g in candidates)
                        lagrange_threshold = winner_floor * 0.7

                        # Rank shadow pool by standalone score
                        shadow_ranked = sorted(
                            shadow_pool,
                            key=lambda g: self._last_shadow_scores.get(g.gene_id, 0),
                            reverse=True,
                        )
                        for g in shadow_ranked[:3]:  # check top 3 shadow candidates
                            shadow_score = scores.get(g.gene_id, 0)
                            if shadow_score < lagrange_threshold:
                                break  # standalone too weak
                            # Co-activation overlap with winners
                            g_coact = set(g.epigenetics.co_activated_with or [])
                            overlap = len(g_coact & (winner_ids | winner_coact))
                            overlap_ratio = overlap / max(len(g_coact), 1) if g_coact else 1.0
                            if overlap_ratio < 0.2:
                                # Low co-activation with winners → being deflected
                                log.debug(
                                    "Lagrange pull-back: gene %s (score=%.2f, overlap=%.1f%%)",
                                    g.gene_id[:12], shadow_score, overlap_ratio * 100,
                                )
                                # Replace the weakest winner with this gene
                                candidates[-1] = g
                                break
                    except Exception:
                        # Lagrange check is a bonus, never blocks — but log
                        # so failures don't silently disable the tier.
                        log.warning("Lagrange pull-back failed", exc_info=True)

        # Step 3.5: NLI classification (optional, DeBERTa backend only)
        relation_graph = {}
        if hasattr(self.ribosome, 'classify_relations'):
            try:
                relation_graph = self.ribosome.classify_relations(candidates)
            except Exception:
                log.warning("NLI classification failed, proceeding without", exc_info=True)

        # Step 4: Dense gene expression
        # Each gene expressed as: Facts (KV pairs) + Source + Raw content
        # Dense format minimizes prose for small model extraction.
        spliced_map = {}
        answer_slate_lines = []  # MoE answer slate — flat KV pairs
        for g in candidates:
            src = g.source_id or ""
            short = ""
            if src and not src.startswith("_"):
                parts = src.replace("\\", "/").split("/")
                try:
                    idx = parts.index("Projects")
                    short = "/".join(parts[idx + 1:])
                except ValueError:
                    short = "/".join(parts[-3:]) if len(parts) > 3 else src
            # Dense XML gene format — structured for small model extraction
            kv_attrs = ""
            if g.key_values:
                # Top 5 KVs as XML attributes for instant scanning
                kv_pairs = " ".join(g.key_values[:5])
                kv_attrs = f' facts="{kv_pairs}"'
                # Collect KVs for MoE answer slate
                for kv in g.key_values[:5]:
                    answer_slate_lines.append(kv)
            src_attr = f' src="{short}"' if short else ""
            # Semantic compression via Headroom (by Tejas Chopra, Apache-2.0).
            # Dispatches by promoter domain: log→LogCompressor,
            # diff→DiffCompressor, else→Kompress (ModernBERT).
            # CodeCompressor disabled (40% invalid syntax — see 2f518dc).
            # Falls back to content[:1000].strip() when headroom is unavailable.
            content = compress_text(
                g.content,
                target_chars=1000,
                content_type=g.promoter.domains,
            )
            spliced_map[g.gene_id] = f"<GENE{src_attr}{kv_attrs}>\n{content}\n</GENE>"

        # Step 5: Assemble (MoE/small-model aware)
        use_slate = self._should_use_slate(downstream_model)
        window = self._assemble(
            query, candidates, spliced_map, relation_graph,
            query_signals=(domains, entities),
            answer_slate=answer_slate_lines if use_slate else None,
            session_id=session_id,
            ignore_delivered=ignore_delivered,
            decoder_prompt_override=effective_decoder_prompt,
        )

        # Annotate window with dynamic budget tier (for telemetry/benchmarks)
        if window.metadata is not None:
            window.metadata["budget_tier"] = budget_tier
            window.metadata["budget_tokens_est"] = budget_tokens_est

        # Touch expressed genes (update epigenetics)
        expressed_ids = [g.gene_id for g in candidates]
        self.genome.touch_genes(expressed_ids)
        self.genome.link_coactivated(expressed_ids)

        # Compute harmonic weights between expressed genes (cymatics)
        if self._use_cymatics and self.config.cymatics.harmonic_links:
            try:
                from .cymatics import compute_harmonic_weights
                weights = compute_harmonic_weights(
                    candidates, peak_width=self._cymatics_peak_width,
                )
                if weights:
                    self.genome.store_harmonic_weights(weights)
            except Exception:
                # Harmonic links are diagnostic, not critical — non-blocking,
                # but log so failures don't disappear silently.
                log.warning("Harmonic link persistence failed", exc_info=True)

        # Update TCM session context with expressed genes
        if self._tcm_session is not None:
            try:
                for gene in candidates:
                    self._tcm_session.update_from_gene(gene)
            except Exception:
                pass  # TCM is diagnostic, not critical

        # Store typed relations in genome (if available)
        if relation_graph:
            batch = []
            for (gid_a, gid_b), (relation, confidence) in relation_graph.items():
                if confidence >= 0.6:
                    batch.append((gid_a, gid_b, int(relation), confidence))
            if batch:
                self.genome.store_relations_batch(batch)

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
    ) -> ContextWindow:
        """Async wrapper -- runs the sync pipeline in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
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
          - genome content (genes, embeddings, attribution)
          - LRU-cached parse results (those are content-keyed)
          - chromatin tier state (per-gene)

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
                from .tcm import SessionContext
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

    # -- Learn: replicate exchange back to genome (Step 6) -------------

    def learn(self, query: str, response: str, timeout_s: float = 15.0) -> Optional[str]:
        """
        Buffer a query+response exchange for later consolidation.

        Appends to the session buffer (last 10 exchanges) and triggers
        auto-consolidation every N learns. The exchange is also immediately
        replicated to the genome for pending-buffer retrieval continuity.

        The ribosome replicate call is wrapped in a thread timeout so a
        slow/overloaded backend can never hang the background task forever.
        On timeout, a minimal gene is synthesized from the raw exchange
        (same fallback path used by ``Ribosome.replicate`` on error).

        Returns gene_id or None on failure.
        """
        # Buffer the exchange for consolidation
        with self._session_buffer_lock:
            self._session_buffer.append((query, response))
            # Keep only last 10 exchanges
            if len(self._session_buffer) > 10:
                self._session_buffer = self._session_buffer[-10:]
            self._session_learn_count += 1

        try:
            # Wrap replicate in a thread timeout so a stuck ribosome
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
                    # Build minimal gene without the ribosome (same shape
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

            # Attach ΣĒMA vector to replicated gene
            if self._sema_codec is not None:
                try:
                    gene.embedding = self._sema_codec.encode(gene.content[:1000])
                except Exception:
                    pass

            # Add to pending buffer immediately (before SQLite commit)
            with self._pending_lock:
                self._pending.append(gene)

            gid = self.genome.upsert_gene(gene)

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
        return await loop.run_in_executor(_executor, self.learn, query, response)

    # -- Session consolidation (Synaptic Plasticity) ---------------------

    def consolidate_session(self) -> List[str]:
        """
        Distill the session buffer into consolidated knowledge genes.

        Sends buffered exchanges to the ribosome with a "distill facts" prompt.
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
                gene = self.ribosome.pack(fact, content_type="text")
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

                gid = self.genome.upsert_gene(gene)
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
        return await loop.run_in_executor(_executor, self.consolidate_session)

    # -- Stats ---------------------------------------------------------

    def stats(self) -> Dict:
        self.genome.refresh()  # See latest gene count from external writers
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
        Lightweight keyword extraction from the query for promoter matching.
        No model call -- uses pre-built frozenset from accel module.
        """
        return extract_query_signals(query)

    def _expand_query_intent(self, query: str) -> str:
        """
        Step 0: Sharpen the initial query frequency via LLM expansion.

        A small ribosome call (~100 tokens out) restates the query with
        expanded keywords BEFORE promoter lookup. This changes which 12
        genes get pulled in the first place — upstream of every bracket
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

    # -- Internal: Step 2 (express) ------------------------------------

    def _express(
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
    ) -> List[Gene]:
        """Query genome + pending buffer for matching genes.

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
            flag in helix.toml. ``None`` (default) uses the config flag,
            ``True`` forces cold-tier on, ``False`` forces it off.
            Plumbed from the /context endpoint's ``include_cold`` body
            parameter so callers can opt in/out per request without
            touching the config file.
        party_id : str, optional
            Caller's party identity. When provided, ``query_genes``
            excludes genes attributed to OTHER parties (cross-party
            leakage prevention) and grants a +0.5 score bonus to genes
            attributed to this party. Unattributed legacy genes remain
            retrievable regardless. ``None`` = no filtering, no bonus
            (existing behavior).
        """
        candidates: List[Gene] = []

        # ── Hot-tier retrieval (chromatin < HETEROCHROMATIN) ────────────
        try:
            candidates = self.genome.query_genes(
                domains,
                entities,
                max_genes=max_genes,
                party_id=party_id,
                use_harmonic=use_harmonic,
                use_sr=use_sr,
                read_only=read_only,
            )
        except PromoterMismatch:
            pass

        # Check pending buffer for recently replicated genes not yet committed
        with self._pending_lock:
            for gene in self._pending:
                gene_domains = set(d.lower() for d in gene.promoter.domains)
                gene_entities = set(e.lower() for e in gene.promoter.entities)
                query_terms = set(d.lower() for d in domains + entities)

                if gene_domains & query_terms or gene_entities & query_terms:
                    candidates.append(gene)

        # Dedupe
        seen: set[str] = set()
        deduped: List[Gene] = []
        for g in candidates:
            if g.gene_id not in seen:
                seen.add(g.gene_id)
                deduped.append(g)

        # ── Cold-tier fallthrough (C.2 of B→C, opt-in) ──────────────────
        # When cold-tier is enabled, consult heterochromatin genes via
        # SEMA cosine similarity. Cold genes still hold their content
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

    def _apply_candidate_refiners(
        self,
        query: str,
        candidates: List[Gene],
        max_genes: int,
        *,
        use_cymatics: bool = True,
        use_harmonic_bin: bool = True,
        use_tcm: bool = True,
        allow_rerank: bool = True,
    ) -> Tuple[List[Gene], Dict[str, Dict[str, float]]]:
        """Apply post-express candidate refiners before assembly or fingerprinting."""
        refiner_contrib: Dict[str, Dict[str, float]] = {}

        if use_cymatics and self._use_cymatics and len(candidates) > 1:
            try:
                from .cymatics import (
                    query_spectrum, cached_gene_spectrum,
                    flux_score_dispatch, build_weight_vector,
                )
                metric = self.config.cymatics.distance_metric
                q_spec = query_spectrum(
                    query, synonym_map=self.config.synonym_map,
                    peak_width=self._cymatics_peak_width,
                )
                weights = build_weight_vector(
                    query, synonym_map=self.config.synonym_map,
                    peak_width=self._cymatics_peak_width,
                )
                scores = self.genome.last_query_scores or {}
                for gene in candidates:
                    g_spec = cached_gene_spectrum(gene, peak_width=self._cymatics_peak_width)
                    bonus = flux_score_dispatch(q_spec, g_spec, weights, metric) * 0.5
                    if bonus:
                        refiner_contrib.setdefault(gene.gene_id, {})["cymatics"] = bonus
                    scores[gene.gene_id] = scores.get(gene.gene_id, 0) + bonus
                self.genome.last_query_scores = scores
                candidates.sort(key=lambda g: scores.get(g.gene_id, 0), reverse=True)
            except Exception:
                log.debug("Cymatics blend failed", exc_info=True)

        if len(candidates) > max_genes:
            if (
                allow_rerank
                and self.config.ingestion.rerank_enabled
                and hasattr(self.ribosome, "re_rank")
            ):
                try:
                    candidates = self.ribosome.re_rank(query, candidates, k=max_genes)
                except Exception:
                    log.warning("Re-rank failed, falling back to retrieval order", exc_info=True)
                    candidates = candidates[:max_genes]
            else:
                candidates = candidates[:max_genes]

        if use_harmonic_bin and len(candidates) >= 3:
            try:
                from .ray_trace import harmonic_bin_boost
                seed_ids = [g.gene_id for g in candidates[:3]]
                velocity = None
                theta_w = 1.0
                if (
                    getattr(self.config.retrieval, "ray_trace_theta", False)
                    and self._tcm_session is not None
                    and self._tcm_session.depth >= 2
                ):
                    velocity = list(self._tcm_session.context_vector)
                    theta_w = self.config.retrieval.theta_weight
                overtones = harmonic_bin_boost(
                    seed_ids,
                    self.genome,
                    k_rays=100,
                    max_bounces=2,
                    velocity_vector=velocity,
                    theta_weight=theta_w,
                )
                if overtones:
                    scores = self.genome.last_query_scores or {}
                    for gene in candidates:
                        if gene.gene_id in overtones:
                            bonus = overtones[gene.gene_id]
                            refiner_contrib.setdefault(gene.gene_id, {})["harmonic_bin"] = bonus
                            scores[gene.gene_id] = scores.get(gene.gene_id, 0) + bonus
                    self.genome.last_query_scores = scores
                    candidates.sort(key=lambda g: scores.get(g.gene_id, 0), reverse=True)
            except Exception:
                log.debug("Harmonic bin boost failed", exc_info=True)

        if use_tcm and self._tcm_session is not None and self._tcm_session.depth > 0:
            try:
                from .tcm import tcm_bonus
                bonuses = tcm_bonus(self._tcm_session, candidates, weight=0.3)
                for gid, bonus in bonuses.items():
                    if bonus:
                        refiner_contrib.setdefault(gid, {})["tcm"] = bonus
                scores = self.genome.last_query_scores or {}
                candidates.sort(
                    key=lambda g: scores.get(g.gene_id, 0) + bonuses.get(g.gene_id, 0),
                    reverse=True,
                )
            except Exception:
                pass

        return candidates, refiner_contrib

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
    ) -> ContextWindow:
        """
        Sort spliced parts, join with dividers, wrap in expressed_context tags.

        MoE mode: sorts genes by retrieval score (highest first) instead of
        sequence_index, so the best match lands in position 0 — inside every
        SWA local attention window. Also injects an answer slate into the
        decoder prompt for front-loaded fact extraction.

        Session working-set (Sprint 2): when session_id is provided and
        budget.session_delivery_enabled is on, genes already delivered in
        this session are emitted as elision stubs rather than full content;
        fresh deliveries are logged to session_delivery_log for future
        elision. ignore_delivered=True bypasses the check (still logs).
        """
        use_slate = answer_slate is not None
        if use_slate:
            # MoE/small-model: relevance-first ordering — best gene at position 0
            # so it's within every sliding-window attention layer
            scores = self.genome.last_query_scores or {}
            sorted_genes = sorted(
                candidates,
                key=lambda g: scores.get(g.gene_id, 0),
                reverse=True,
            )
        else:
            # Dense: sequence ordering for narrative coherence
            sorted_genes = sorted(candidates, key=lambda g: g.promoter.sequence_index or 0)

        # Sprint 1 legibility pack: compute z-score stats once over the
        # expressed gene set so every header's confidence symbol is
        # calibrated against THIS response (not a genome-wide baseline).
        # See helix_context/legibility.py + docs/FUTURE/AI_CONSUMER_ROADMAP_2026-04-14.md.
        legibility_on = self.config.budget.legibility_enabled
        if legibility_on:
            _leg_scores = self.genome.last_query_scores or {}
            _leg_tiers = getattr(self.genome, "last_tier_contributions", None) or {}
            _expressed_scores = {
                g.gene_id: _leg_scores.get(g.gene_id, 0.0) for g in sorted_genes
            }
            _score_stats = legibility.compute_score_stats(_expressed_scores)
        else:
            _leg_scores = {}
            _leg_tiers = {}
            _score_stats = (0.0, 0.0)

        # Sprint 2 session working-set: look up prior deliveries so genes
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
            try:
                for g in sorted_genes:
                    prior = _session_delivery.already_delivered(
                        self.genome.conn,
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
        # Track per-gene log intent so budget-trim can discard entries
        # that didn't actually make it to the consumer. Value is
        # (mode, content_hash) for fresh deliveries, or None for elided.
        _delivery_log_map: Dict[str, Optional[Tuple[str, str]]] = {}

        for g in sorted_genes:
            # Prefer ribosome-spliced text; fall back to complement summary;
            # last resort is Headroom semantic compression (was content[:500]).
            spliced_text = spliced_map.get(g.gene_id) or g.complement or compress_text(
                g.content,
                target_chars=500,
                content_type=g.promoter.domains,
            )
            prior = _prior_deliveries.get(g.gene_id) if session_on else None
            if prior is not None:
                # Gene already delivered in this session — elide content
                prior_ts, _prior_mode, _prior_hash = prior
                try:
                    queries_ago = _session_delivery.count_queries_in_session_since(
                        self.genome.conn,
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

        expressed = "\n---\n".join(parts) if parts else "(no relevant context found)"

        # Wrap in tags
        expressed_wrapped = (
            "<expressed_context>\n"
            f"{expressed}\n"
            "</expressed_context>"
        )

        # MoE answer slate: inject pre-extracted KVs into decoder prompt
        # so they land in the first ~200 tokens (inside every SWA window)
        if answer_slate:
            # Dedupe and limit slate to 20 entries
            seen_kvs: set[str] = set()
            unique_slate: list[str] = []
            for kv in answer_slate:
                if kv not in seen_kvs:
                    seen_kvs.add(kv)
                    unique_slate.append(kv)
            slate_text = "\n".join(unique_slate[:20])
            decoder_prompt = DECODER_MOE.replace("{answer_slate}", slate_text)
        else:
            # Honor per-request override (threaded from build_context) to
            # avoid racing on self._decoder_prompt across concurrent calls.
            decoder_prompt = decoder_prompt_override or self._decoder_prompt

        # Budget enforcement: if over token budget, drop lowest-scored genes
        est_tokens = estimate_tokens(decoder_prompt) + estimate_tokens(expressed_wrapped)
        budget = self.config.budget.ribosome_tokens + self.config.budget.expression_tokens

        if est_tokens > budget and len(parts) > 1:
            # Drop from the end (lowest-ranked after re-rank)
            while est_tokens > budget and len(parts) > 1:
                parts.pop()
                expressed = "\n---\n".join(parts)
                expressed_wrapped = f"<expressed_context>\n{expressed}\n</expressed_context>"
                est_tokens = estimate_tokens(decoder_prompt) + estimate_tokens(expressed_wrapped)

        compressed_chars = len(expressed)

        # Sprint 2 session working-set: persist deliveries that actually
        # made it to the consumer (post-budget-trim). Elided genes (stubs)
        # already had their original delivery logged on the prior turn, so
        # we don't re-log them here. Any exception is swallowed — a log
        # hiccup must not break the retrieval response.
        if session_on and session_id is not None:
            try:
                delivered_ids = [g.gene_id for g in sorted_genes[:len(parts)]]
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
        health = self._compute_health(query_terms, candidates, compressed_chars, relation_graph)

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
            },
        )

    # -- Internal: delta-epsilon health --------------------------------

    def _compute_health(
        self,
        query_terms: List[str],
        candidates: List[Gene],
        compressed_chars: int,
        relation_graph: Optional[Dict] = None,
    ) -> ContextHealth:
        """
        Compute the delta-epsilon context health signal.

        Measures four dimensions:
            coverage  — fraction of query terms that matched genome tags
            density   — fraction of expression token budget actually used
            freshness — average decay score of expressed genes (1=fresh, 0=stale)
            ellipticity — composite score (geometric mean of the three)

        Status thresholds:
            aligned   — ellipticity >= 0.7 (genome is well-grounded)
            sparse    — ellipticity >= 0.3 (genome has gaps, model may guess)
            stale     — freshness < 0.4 (expressed genes are outdated)
            denatured — ellipticity < 0.3 (context is unreliable)
        """
        import math

        genome_stats = self.genome.stats()
        total_genes = genome_stats.get("total_genes", 0)
        genes_expressed = len(candidates)

        # Coverage: what fraction of query terms were found in the genome?
        # Checks promoter tags, FTS5 content matches, and key-value extracts.
        if query_terms:
            matched = 0
            # Collect all searchable text from expressed genes
            all_tags: set[str] = set()
            all_content_lower = ""
            for g in candidates:
                all_tags.update(d.lower() for d in g.promoter.domains)
                all_tags.update(e.lower() for e in g.promoter.entities)
                if g.key_values:
                    all_tags.update(kv.lower() for kv in g.key_values)
                # Content presence check (for FTS5/SPLADE-found genes)
                all_content_lower += " " + (g.content[:2000] or "").lower()
            for term in query_terms:
                t = term.lower()
                if t in all_tags or t in all_content_lower:
                    matched += 1
            coverage = matched / len(query_terms)
        else:
            coverage = 0.0

        # Density: how much of the effective expression capacity did we use?
        # Scale budget by genes expressed vs max — a query that correctly
        # expresses 4 focused genes shouldn't be penalized for not filling 12 slots.
        max_genes = self.config.budget.max_genes_per_turn
        expressed_ratio = genes_expressed / max(max_genes, 1)
        effective_budget = self.config.budget.expression_tokens * 4 * max(expressed_ratio, 0.25)
        density = min(1.0, compressed_chars / max(effective_budget, 1))

        # Freshness: average decay score of expressed genes
        if candidates:
            freshness = sum(g.epigenetics.decay_score for g in candidates) / len(candidates)
        else:
            freshness = 0.0

        # Logical coherence (from NLI relation graph, if available)
        logical_coherence = 0.0
        if relation_graph:
            try:
                from .nli_backend import compute_logical_coherence
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

        # Status classification
        if freshness < 0.4 and genes_expressed > 0:
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
            scores_map = getattr(self.genome, "last_query_scores", None) or {}
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
        )

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

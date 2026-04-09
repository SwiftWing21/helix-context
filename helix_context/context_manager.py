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
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .accel import extract_query_signals, estimate_tokens
from .codons import CodonChunker, CodonEncoder
from .config import HelixConfig
from .exceptions import PromoterMismatch
from .genome import Genome
from .ribosome import Ribosome, OllamaBackend
from .schemas import ContextHealth, ContextWindow, Gene

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

DECODER_CONDENSED = """The <expressed_context> below contains project knowledge selected for your query.
Each section between --- dividers is one knowledge unit. Each starts with [Source: path] showing where it came from.

To answer, follow these steps:
1. LOCATE: Find the section most relevant to the question. Look for specific values, names, and numbers.
2. QUOTE: Identify the exact line or value that answers the question.
3. ANSWER: State the answer using the specific value from the context.

If the context truly does not contain the answer, say so. But LOOK CAREFULLY first — the answer
is usually a specific number, name, or value buried in one of the sections."""

DECODER_MINIMAL = """Answer using ONLY the <expressed_context> below. Do not guess beyond what it states."""

DECODER_NONE = ""

DECODER_MODES = {
    "full": DECODER_FULL,
    "condensed": DECODER_CONDENSED,
    "minimal": DECODER_MINIMAL,
    "none": DECODER_NONE,
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

        # Genome (SQLite storage)
        self.genome = Genome(
            path=config.genome.path,
            synonym_map=config.synonym_map,
            sema_codec=self._sema_codec,
        )

        # Chunker (deterministic text splitting)
        self.chunker = CodonChunker(max_chars_per_strand=4000)
        self.encoder = CodonEncoder()

        # Ribosome (small model codec)
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

        if config.ribosome.backend == "deberta":
            try:
                from .deberta_backend import DeBERTaRibosome
                self.ribosome = DeBERTaRibosome(
                    rerank_model_path=config.ribosome.rerank_model_path,
                    splice_model_path=config.ribosome.splice_model_path,
                    nli_model_path=config.ribosome.nli_model_path,
                    ollama_ribosome=ollama_ribosome,
                    device=config.ribosome.device,
                    splice_threshold=config.ribosome.splice_threshold,
                    nli_splice_bonus=config.ribosome.nli_splice_bonus,
                    nli_splice_penalty=config.ribosome.nli_splice_penalty,
                )
                log.info("Using DeBERTa hybrid ribosome (re_rank + splice accelerated)")
            except Exception:
                log.warning("DeBERTa backend failed to load, falling back to Ollama", exc_info=True)
                self.ribosome = ollama_ribosome
        else:
            self.ribosome = ollama_ribosome

        # Adaptive decoder prompt based on downstream model capability
        self._decoder_mode = config.budget.decoder_mode
        self._decoder_prompt = DECODER_MODES.get(self._decoder_mode, DECODER_FULL)

        # Pending replication buffer -- genes from background replication
        # that haven't committed to SQLite yet. Checked during Step 2
        # so follow-up queries don't lose context from the previous turn.
        self._pending: List[Gene] = []
        self._pending_lock = threading.Lock()

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

        source_path = metadata.get("path") if metadata else None

        # Batch-encode ΣĒMA vectors if codec available
        sema_vectors = None
        if self._sema_codec is not None:
            try:
                texts = [s.content[:1000] for s in strands]  # Cap for encoder
                sema_vectors = self._sema_codec.encode_batch(texts)
            except Exception:
                log.debug("ΣĒMA batch encoding failed, skipping")

        for i, strand in enumerate(strands):
            gene = self.ribosome.pack(strand.content, content_type=content_type)
            # Preserve sequence index from chunking
            gene.promoter.sequence_index = strand.sequence_index
            gene.is_fragment = strand.is_fragment
            # Store source file path for change-based decay
            if source_path:
                gene.source_id = source_path
            # Attach ΣĒMA vector
            if sema_vectors is not None and i < len(sema_vectors):
                gene.embedding = sema_vectors[i]

            gid = self.genome.upsert_gene(gene)
            gene_ids.append(gid)

        log.info("Ingested %d strands from %s content (%d chars)",
                 len(gene_ids), content_type, len(content))
        return gene_ids

    async def ingest_async(self, content: str, content_type: str = "text", metadata: Optional[Dict] = None) -> List[str]:
        """Async wrapper for ingest -- runs ribosome calls in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.ingest, content, content_type, metadata)

    # -- Build context: the main per-turn operation --------------------

    def build_context(self, query: str) -> ContextWindow:
        """
        Build the active context window for a query.
        Runs the 5-step expression pipeline (Steps 1-5).
        """
        self._maybe_compact()

        max_genes = self.config.budget.max_genes_per_turn

        # Step 1: Extract promoter signals (heuristic, no model)
        domains, entities = self._extract_query_signals(query)

        # Step 2: Express (genome query + pending buffer)
        candidates = self._express(domains, entities, max_genes)

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
                ribosome_prompt=self._decoder_prompt,
                expressed_context="(no relevant context found in genome)",
                total_estimated_tokens=estimate_tokens(self._decoder_prompt),
                compression_ratio=1.0,
                context_health=empty_health,
                metadata={"query": query, "genes_expressed": 0},
            )

        # Step 3: Trim to budget (FTS5 hybrid scoring already ranks well)
        # DeBERTa re-rank is disabled — it was trained on limited data and
        # drops FTS5-matched needle genes. The hybrid retrieval score
        # (promoter tag + prefix + FTS5 content) is a better signal.
        if len(candidates) > max_genes:
            candidates = candidates[:max_genes]

        # Step 3.5: NLI classification (optional, DeBERTa backend only)
        relation_graph = {}
        if hasattr(self.ribosome, 'classify_relations'):
            try:
                relation_graph = self.ribosome.classify_relations(candidates)
            except Exception:
                log.warning("NLI classification failed, proceeding without", exc_info=True)

        # Step 4: Contextual splice — raw content with source context prefix.
        # Prepending the source file path gives the downstream model a
        # "situational anchor" — it knows WHERE each gene came from,
        # which dramatically improves extraction accuracy for specific values.
        spliced_map = {}
        for g in candidates:
            # Build source context prefix
            src = g.source_id or ""
            if src and not src.startswith("_"):
                # Extract readable path: "BigEd/config.toml" from full path
                parts = src.replace("\\", "/").split("/")
                # Find the project name (after "Projects" or use last 3 segments)
                try:
                    idx = parts.index("Projects")
                    short = "/".join(parts[idx + 1:])
                except ValueError:
                    short = "/".join(parts[-3:]) if len(parts) > 3 else src
                prefix = f"[Source: {short}]\n"
            else:
                prefix = ""
            # Prepend pre-extracted key-value facts if available
            kv_line = ""
            if g.key_values:
                kv_line = "Facts: " + ", ".join(g.key_values[:10]) + "\n"
            spliced_map[g.gene_id] = kv_line + prefix + g.content[:1400]

        # Step 5: Assemble
        window = self._assemble(query, candidates, spliced_map, relation_graph)

        # Touch expressed genes (update epigenetics)
        expressed_ids = [g.gene_id for g in candidates]
        self.genome.touch_genes(expressed_ids)
        self.genome.link_coactivated(expressed_ids)

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

    async def build_context_async(self, query: str) -> ContextWindow:
        """Async wrapper -- runs the sync pipeline in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.build_context, query)

    # -- Learn: replicate exchange back to genome (Step 6) -------------

    def learn(self, query: str, response: str) -> Optional[str]:
        """
        Buffer a query+response exchange for later consolidation.

        Appends to the session buffer (last 10 exchanges) and triggers
        auto-consolidation every N learns. The exchange is also immediately
        replicated to the genome for pending-buffer retrieval continuity.

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
            gene = self.ribosome.replicate(query, response)

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
            },
        }

    # -- Internal: Step 1 (extract) ------------------------------------

    def _extract_query_signals(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Lightweight keyword extraction from the query for promoter matching.
        No model call -- uses pre-built frozenset from accel module.
        """
        return extract_query_signals(query)

    # -- Internal: Step 2 (express) ------------------------------------

    def _express(self, domains: List[str], entities: List[str], max_genes: int) -> List[Gene]:
        """Query genome + pending buffer for matching genes."""
        candidates: List[Gene] = []

        # Query committed genes from SQLite
        try:
            candidates = self.genome.query_genes(domains, entities, max_genes=max_genes)
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

        return deduped[:max_genes * 2]

    # -- Internal: Step 5 (assemble) -----------------------------------

    def _assemble(self, query: str, candidates: List[Gene], spliced_map: Dict[str, str], relation_graph: Optional[Dict] = None) -> ContextWindow:
        """
        Sort spliced parts by sequence_index, join with dividers,
        wrap in expressed_context tags, prepend decoder prompt.
        """
        # Sort by sequence_index if set
        sorted_genes = sorted(candidates, key=lambda g: g.promoter.sequence_index or 0)

        parts: List[str] = []
        total_raw = 0

        for g in sorted_genes:
            spliced_text = spliced_map.get(g.gene_id, g.complement or g.content[:500])
            parts.append(spliced_text)
            total_raw += len(g.content)

        expressed = "\n---\n".join(parts) if parts else "(no relevant context found)"

        # Wrap in tags
        expressed_wrapped = (
            "<expressed_context>\n"
            f"{expressed}\n"
            "</expressed_context>"
        )

        # Budget enforcement: if over token budget, drop lowest-scored genes
        est_tokens = estimate_tokens(self._decoder_prompt) + estimate_tokens(expressed_wrapped)
        budget = self.config.budget.ribosome_tokens + self.config.budget.expression_tokens

        if est_tokens > budget and len(parts) > 1:
            # Drop from the end (lowest-ranked after re-rank)
            while est_tokens > budget and len(parts) > 1:
                parts.pop()
                expressed = "\n---\n".join(parts)
                expressed_wrapped = f"<expressed_context>\n{expressed}\n</expressed_context>"
                est_tokens = estimate_tokens(self._decoder_prompt) + estimate_tokens(expressed_wrapped)

        compressed_chars = len(expressed)

        # Delta-epsilon health signal
        query_terms = query.lower().split()
        health = self._compute_health(query_terms, candidates, compressed_chars, relation_graph)

        return ContextWindow(
            ribosome_prompt=self._decoder_prompt,
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

        # Coverage: what fraction of query terms hit the promoter index?
        if query_terms:
            matched = 0
            all_tags: set[str] = set()
            for g in candidates:
                all_tags.update(d.lower() for d in g.promoter.domains)
                all_tags.update(e.lower() for e in g.promoter.entities)
            for term in query_terms:
                if term.lower() in all_tags:
                    matched += 1
            coverage = matched / len(query_terms)
        else:
            coverage = 0.0

        # Density: how much of the expression budget did we use?
        budget_chars = self.config.budget.expression_tokens * 4  # ~4 chars/token
        density = min(1.0, compressed_chars / max(budget_chars, 1))

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

        return ContextHealth(
            ellipticity=round(ellipticity, 4),
            coverage=round(coverage, 4),
            density=round(density, 4),
            freshness=round(freshness, 4),
            logical_coherence=round(logical_coherence, 4),
            genes_available=total_genes,
            genes_expressed=genes_expressed,
            status=status,
        )

    # -- Internal: compaction ------------------------------------------

    def _maybe_compact(self) -> None:
        now = time.time()
        if now - self._last_compact > self.config.genome.compact_interval:
            self.genome.compact()
            self._last_compact = now

    # -- Cleanup -------------------------------------------------------

    def close(self) -> None:
        self.genome.close()

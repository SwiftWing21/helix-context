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

RIBOSOME_DECODER = """CRITICAL INSTRUCTIONS — READ BEFORE RESPONDING:

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

        # Genome (SQLite storage)
        self.genome = Genome(
            path=config.genome.path,
            synonym_map=config.synonym_map,
            decay_rate=config.genome.decay_rate,
            heterochromatin_threshold=config.genome.heterochromatin_threshold,
            stale_threshold=config.genome.stale_threshold,
        )

        # Chunker (deterministic text splitting)
        self.chunker = CodonChunker(max_chars_per_strand=4000)
        self.encoder = CodonEncoder()

        # Ribosome (small model codec)
        backend = OllamaBackend(
            model=config.ribosome.model,
            base_url=config.ribosome.base_url,
            timeout=config.ribosome.timeout,
            keep_alive=config.ribosome.keep_alive,
            warmup=config.ribosome.warmup,
        )
        self.ribosome = Ribosome(
            backend=backend,
            encoder=self.encoder,
            splice_aggressiveness=config.budget.splice_aggressiveness,
        )

        # Pending replication buffer -- genes from background replication
        # that haven't committed to SQLite yet. Checked during Step 2
        # so follow-up queries don't lose context from the previous turn.
        self._pending: List[Gene] = []
        self._pending_lock = threading.Lock()

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

        for strand in strands:
            gene = self.ribosome.pack(strand.content, content_type=content_type)
            # Preserve sequence index from chunking
            gene.promoter.sequence_index = strand.sequence_index
            gene.is_fragment = strand.is_fragment

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
                ribosome_prompt=RIBOSOME_DECODER,
                expressed_context="(no relevant context found in genome)",
                total_estimated_tokens=len(RIBOSOME_DECODER) // 4,
                compression_ratio=1.0,
                context_health=empty_health,
                metadata={"query": query, "genes_expressed": 0},
            )

        # Step 3: Re-rank (ribosome, optional -- only if over budget)
        if len(candidates) > max_genes:
            candidates = self.ribosome.re_rank(query, candidates, k=max_genes)

        # Step 4: Splice (ribosome, batched single call)
        spliced_map = self.ribosome.splice(query, candidates)

        # Step 5: Assemble
        window = self._assemble(query, candidates, spliced_map)

        # Touch expressed genes (update epigenetics)
        expressed_ids = [g.gene_id for g in candidates]
        self.genome.touch_genes(expressed_ids)
        self.genome.link_coactivated(expressed_ids)

        return window

    async def build_context_async(self, query: str) -> ContextWindow:
        """Async wrapper -- runs the sync pipeline in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.build_context, query)

    # -- Learn: replicate exchange back to genome (Step 6) -------------

    def learn(self, query: str, response: str) -> Optional[str]:
        """
        Pack a query+response exchange and store in genome.
        Called as a background task after the stream completes.
        Returns gene_id or None on failure.
        """
        try:
            gene = self.ribosome.replicate(query, response)

            # Add to pending buffer immediately (before SQLite commit)
            with self._pending_lock:
                self._pending.append(gene)

            gid = self.genome.upsert_gene(gene)

            # Remove from pending now that it's committed
            with self._pending_lock:
                self._pending = [g for g in self._pending if g.gene_id != gid]

            log.info("Replicated exchange into gene %s", gid)
            return gid

        except Exception:
            log.warning("Replication failed (non-fatal)", exc_info=True)
            return None

    async def learn_async(self, query: str, response: str) -> Optional[str]:
        """Async wrapper for learn."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.learn, query, response)

    # -- Stats ---------------------------------------------------------

    def stats(self) -> Dict:
        genome_stats = self.genome.stats()
        return {
            **genome_stats,
            "pending_replications": len(self._pending),
            "config": {
                "ribosome_budget": self.config.budget.ribosome_tokens,
                "expression_budget": self.config.budget.expression_tokens,
                "max_genes_per_turn": self.config.budget.max_genes_per_turn,
                "splice_aggressiveness": self.config.budget.splice_aggressiveness,
            },
        }

    # -- Internal: Step 1 (extract) ------------------------------------

    def _extract_query_signals(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Lightweight keyword extraction from the query for promoter matching.
        No model call -- just stop-word removal and heuristics.
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to",
            "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "about", "like", "after", "before", "between",
            "and", "or", "but", "not", "no", "if", "then", "than",
            "what", "which", "who", "whom", "this", "that", "these",
            "how", "when", "where", "why", "all", "each", "every",
            "it", "its", "i", "me", "my", "we", "our", "you", "your",
            "he", "she", "they", "them", "his", "her", "their",
        }

        words = query.lower().split()
        keywords = [w.strip("?.,!;:'\"()[]{}") for w in words
                     if w.lower() not in stop_words and len(w) > 2]

        # Heuristic: longer or capitalized words are more likely entities
        entities = []
        for w in keywords:
            if len(w) > 4:
                entities.append(w)
            elif w and w[0].isupper():
                entities.append(w)

        domains = keywords[:5]
        return domains, entities

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

    def _assemble(self, query: str, candidates: List[Gene], spliced_map: Dict[str, str]) -> ContextWindow:
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
        est_tokens = (len(RIBOSOME_DECODER) + len(expressed_wrapped)) // 4
        budget = self.config.budget.ribosome_tokens + self.config.budget.expression_tokens

        if est_tokens > budget and len(parts) > 1:
            # Drop from the end (lowest-ranked after re-rank)
            while est_tokens > budget and len(parts) > 1:
                parts.pop()
                expressed = "\n---\n".join(parts)
                expressed_wrapped = f"<expressed_context>\n{expressed}\n</expressed_context>"
                est_tokens = (len(RIBOSOME_DECODER) + len(expressed_wrapped)) // 4

        compressed_chars = len(expressed)

        # Delta-epsilon health signal
        query_terms = query.lower().split()
        health = self._compute_health(query_terms, candidates, compressed_chars)

        return ContextWindow(
            ribosome_prompt=RIBOSOME_DECODER,
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

        # Ellipticity: geometric mean of the three signals
        # Clamp inputs to avoid log(0)
        c = max(coverage, 0.01)
        d = max(density, 0.01)
        f = max(freshness, 0.01)
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

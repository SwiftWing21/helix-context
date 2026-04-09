"""
DeBERTa Ribosome Backend — Drop-in replacement for OllamaBackend.

Replaces the two most expensive ribosome operations:
    re_rank  — cross-encoder scoring (query, gene_summary) -> relevance
    splice   — binary classification (query, codon) -> keep/drop

PACK and REPLICATE still use the Ollama backend (they need generation).

This backend loads two fine-tuned DeBERTa-v3-small models:
    training/models/rerank/   — cross-encoder for re-ranking
    training/models/splice/   — binary classifier for splice decisions

Usage:
    from helix_context.deberta_backend import DeBERTaRibosome

    ribosome = DeBERTaRibosome(
        rerank_model_path="training/models/rerank",
        splice_model_path="training/models/splice",
        ollama_fallback=OllamaBackend(),  # for pack/replicate
    )
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import torch

from .schemas import Gene, NLRelation

log = logging.getLogger("helix.ribosome.deberta")


class DeBERTaRibosome:
    """
    Hybrid ribosome: DeBERTa for re_rank + splice, Ollama for pack + replicate.

    Drop-in compatible with helix_context.ribosome.Ribosome — same method
    signatures for re_rank() and splice(). Pack/replicate delegate to the
    Ollama-backed Ribosome passed at init.
    """

    def __init__(
        self,
        rerank_model_path: str = "training/models/rerank",
        splice_model_path: str = "training/models/splice",
        nli_model_path: str = "training/models/nli",
        ollama_ribosome=None,
        device: str = "auto",
        splice_threshold: float = 0.5,
        nli_splice_bonus: float = 0.15,
        nli_splice_penalty: float = 0.15,
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        log.info("Loading DeBERTa rerank model from %s", rerank_model_path)
        self._rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
        self._rerank_model = AutoModelForSequenceClassification.from_pretrained(
            rerank_model_path
        ).to(self._device)
        self._rerank_model.train(False)

        log.info("Loading DeBERTa splice model from %s", splice_model_path)
        self._splice_tokenizer = AutoTokenizer.from_pretrained(splice_model_path)
        self._splice_model = AutoModelForSequenceClassification.from_pretrained(
            splice_model_path
        ).to(self._device)
        self._splice_model.train(False)

        self.splice_threshold = splice_threshold
        self.nli_splice_bonus = nli_splice_bonus
        self.nli_splice_penalty = nli_splice_penalty
        self._ollama = ollama_ribosome

        # NLI model — lazy-loaded (optional, may not exist yet)
        self._nli = None
        self._nli_model_path = nli_model_path

        log.info("DeBERTa ribosome ready on %s", self._device)

    # ── Re-rank ────────────────────────────────────────────────────────

    def re_rank(self, query: str, candidates: List[Gene], k: int = 5) -> List[Gene]:
        """Score candidate genes by relevance using the cross-encoder."""
        if not candidates:
            return []
        if len(candidates) <= k:
            return candidates

        t0 = time.perf_counter()

        # Build text pairs
        texts_a = []
        texts_b = []
        for g in candidates:
            texts_a.append(query)
            summary = g.promoter.summary
            domains = ", ".join(g.promoter.domains)
            texts_b.append(f"{summary} [{domains}]" if domains else summary)

        # Tokenize all pairs at once
        encodings = self._rerank_tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Score
        with torch.no_grad():
            outputs = self._rerank_model(**encodings)
            scores = outputs.logits.squeeze(-1)
            # Clamp to 0-1 range
            scores = torch.clamp(scores, 0.0, 1.0).cpu().tolist()

        # If single candidate, scores is a scalar
        if isinstance(scores, float):
            scores = [scores]

        # Sort by score
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "DeBERTa re_rank: %d candidates → top %d in %.1fms",
            len(candidates), k, elapsed_ms,
        )

        return [g for _, g in scored[:k]]

    # ── Splice ─────────────────────────────────────────────────────────

    def splice(
        self,
        query: str,
        genes: List[Gene],
        min_codons_kept: int = 2,
    ) -> Dict[str, str]:
        """Classify each codon as keep/drop using the binary classifier."""
        if not genes:
            return {}

        t0 = time.perf_counter()
        result: Dict[str, str] = {}

        # Batch all (query, codon) pairs across all genes
        all_pairs_a = []
        all_pairs_b = []
        pair_index = []  # (gene_idx, codon_idx) for reconstruction

        for gi, g in enumerate(genes):
            for ci, codon in enumerate(g.codons):
                all_pairs_a.append(query)
                all_pairs_b.append(codon)
                pair_index.append((gi, ci))

        if not all_pairs_a:
            return {g.gene_id: g.complement or g.content[:500] for g in genes}

        # Tokenize all at once
        encodings = self._splice_tokenizer(
            all_pairs_a,
            all_pairs_b,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Predict
        with torch.no_grad():
            outputs = self._splice_model(**encodings)
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]

        # Reconstruct per-gene decisions
        gene_keep: Dict[int, List[int]] = {i: [] for i in range(len(genes))}
        for (gi, ci), prob in zip(pair_index, probs):
            if prob >= self.splice_threshold:
                gene_keep[gi].append(ci)

        # Build spliced text
        for gi, g in enumerate(genes):
            kept_indices = gene_keep[gi]

            # Empty splice guard
            if not kept_indices and g.codons:
                kept_indices = list(range(min(min_codons_kept, len(g.codons))))
                log.info("Empty splice for gene %s, keeping first %d codons", g.gene_id, len(kept_indices))

            if kept_indices:
                kept = [g.codons[i] for i in kept_indices if i < len(g.codons)]
                result[g.gene_id] = " | ".join(kept) if kept else (g.complement or g.content[:500])
            else:
                result[g.gene_id] = g.complement or g.content[:500]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "DeBERTa splice: %d genes, %d codons in %.1fms",
            len(genes), len(all_pairs_a), elapsed_ms,
        )

        return result

    # ── NLI Classification ───────────────────────────────────────────────

    def _load_nli(self):
        """Lazy-load the NLI classifier on first use."""
        if self._nli is not None:
            return self._nli
        try:
            from .nli_backend import NLIClassifier
            self._nli = NLIClassifier(
                model_path=self._nli_model_path,
                device=str(self._device),
            )
        except Exception:
            log.warning("NLI model not available at %s", self._nli_model_path, exc_info=True)
            self._nli = None
        return self._nli

    def classify_relations(self, genes: List[Gene]) -> Dict:
        """Classify NLI relations between expressed genes. Returns relation graph."""
        nli = self._load_nli()
        if nli is None:
            return {}
        return nli.build_relation_graph(genes)

    # ── Delegated to Ollama ────────────────────────────────────────────

    def pack(self, content: str, content_type: str = "text") -> Gene:
        """Delegate to Ollama ribosome (needs generation)."""
        if self._ollama is None:
            raise RuntimeError("DeBERTa ribosome requires an Ollama fallback for pack()")
        return self._ollama.pack(content, content_type)

    def replicate(self, query: str, response: str) -> Gene:
        """Delegate to Ollama ribosome (needs generation)."""
        if self._ollama is None:
            raise RuntimeError("DeBERTa ribosome requires an Ollama fallback for replicate()")
        return self._ollama.replicate(query, response)

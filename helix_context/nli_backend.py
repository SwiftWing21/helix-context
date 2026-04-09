"""
NLI Classifier — MacCartney-Manning natural logic relation detection.

Uses a fine-tuned DeBERTa-v3-small to classify the logical relation
between gene pairs or codon pairs into one of 7 classes:

    entailment, reverse_entailment, equivalence,
    alternation, negation, cover, independence

Integration points:
    - Step 3.5 in context_manager: classify relations between expressed genes
    - Splice post-processing: bias keep/drop based on entailment/alternation
    - Co-activation typing: store typed links instead of bare gene_id lists
    - Health signal: logical coherence metric
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import torch

from .schemas import Gene, NLRelation

log = logging.getLogger("helix.nli")

# Coherence weights for each relation type
_COHERENCE_WEIGHTS: Dict[NLRelation, float] = {
    NLRelation.ENTAILMENT: 1.0,
    NLRelation.REVERSE_ENTAILMENT: 0.8,
    NLRelation.EQUIVALENCE: 1.0,
    NLRelation.ALTERNATION: -0.3,
    NLRelation.NEGATION: -0.5,
    NLRelation.COVER: 0.5,
    NLRelation.INDEPENDENCE: 0.0,
}


class NLIClassifier:
    """DeBERTa-based 7-class natural logic relation classifier."""

    def __init__(
        self,
        model_path: str = "training/models/nli",
        device: str = "auto",
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        log.info("Loading NLI model from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(self._device)
        self._model.train(False)
        log.info("NLI classifier ready on %s", self._device)

    def classify_pair(
        self, text_a: str, text_b: str
    ) -> Tuple[NLRelation, float]:
        """Classify the relation between two texts."""
        encoding = self._tokenizer(
            text_a, text_b,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()

        return NLRelation(pred_class), confidence

    def classify_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[NLRelation, float]]:
        """Classify relations for a batch of text pairs."""
        if not pairs:
            return []

        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]

        encodings = self._tokenizer(
            texts_a, texts_b,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_classes = probs.argmax(dim=-1).cpu().tolist()
            confidences = probs.max(dim=-1).values.cpu().tolist()

        return [
            (NLRelation(cls), conf)
            for cls, conf in zip(pred_classes, confidences)
        ]

    def build_relation_graph(
        self, genes: List[Gene]
    ) -> Dict[Tuple[str, str], Tuple[NLRelation, float]]:
        """
        Classify relations between all pairs of expressed genes.

        For N genes, produces N*(N-1)/2 classifications.
        8 genes = 28 pairs, ~5-10ms on GPU.
        """
        if len(genes) < 2:
            return {}

        t0 = time.perf_counter()

        pairs = []
        pair_keys = []
        for i, ga in enumerate(genes):
            for gb in genes[i + 1:]:
                text_a = _gene_summary_text(ga)
                text_b = _gene_summary_text(gb)
                pairs.append((text_a, text_b))
                pair_keys.append((ga.gene_id, gb.gene_id))

        results = self.classify_batch(pairs)

        graph = {}
        for key, (relation, confidence) in zip(pair_keys, results):
            graph[key] = (relation, confidence)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "NLI graph: %d genes, %d pairs classified in %.1fms",
            len(genes), len(pairs), elapsed_ms,
        )

        return graph


def compute_logical_coherence(
    relation_graph: Dict[Tuple[str, str], Tuple[NLRelation, float]],
) -> float:
    """
    Compute logical coherence from a relation graph.

    Returns a score in [0, 1] where:
      1.0 = all expressed genes are entailment/equivalence-linked
      0.5 = mixed (some entailment, some independence)
      0.0 = contradictory (alternation/negation dominate)
    """
    if not relation_graph:
        return 0.0

    total_weight = 0.0
    for (relation, confidence) in relation_graph.values():
        w = _COHERENCE_WEIGHTS.get(relation, 0.0)
        total_weight += w * confidence

    raw = total_weight / len(relation_graph)
    # Normalize from [-0.5, 1.0] to [0, 1]
    return max(0.0, min(1.0, (raw + 0.5) / 1.5))


def _gene_summary_text(g: Gene) -> str:
    """Build representative text for a gene."""
    parts = []
    if g.promoter.summary:
        parts.append(g.promoter.summary)
    if g.promoter.domains:
        parts.append(f"[{', '.join(g.promoter.domains)}]")
    if g.promoter.entities:
        parts.append(f"({', '.join(g.promoter.entities[:5])})")
    return " ".join(parts) if parts else g.complement[:200]

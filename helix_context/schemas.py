"""
Schemas — Pydantic models for the Helix genome data layer.

These are the stable internal contracts. All models are JSON-serializable
for SQLite storage (via model_dump_json / model_validate_json).
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import List, Optional

from pydantic import BaseModel, Field


class NLRelation(IntEnum):
    """MacCartney-Manning natural logic relations (7-class)."""
    ENTAILMENT = 0          # A ⊂ B (A implies B)
    REVERSE_ENTAILMENT = 1  # A ⊃ B (B implies A)
    EQUIVALENCE = 2         # A = B
    ALTERNATION = 3         # A ∩ B = ∅, A ∪ B ≠ D (mutually exclusive)
    NEGATION = 4            # A ∩ B = ∅, A ∪ B = D (exhaustive opposites)
    COVER = 5               # A ∩ B ≠ ∅, A ∪ B = D (overlap + exhaust)
    INDEPENDENCE = 6        # no reliable relation


class ChromatinState(IntEnum):
    """Gene accessibility state — mirrors biological chromatin compaction."""
    OPEN = 0            # Recently accessed, hot
    EUCHROMATIN = 1     # Accessible, normal state
    HETEROCHROMATIN = 2 # Compacted, stale — excluded from queries


class PromoterTags(BaseModel):
    """Retrieval metadata — how the genome finds this gene."""
    domains: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    intent: str = ""
    summary: str = ""
    sequence_index: Optional[int] = None


class TypedCoActivation(BaseModel):
    """A co-activation link with a typed logical relation."""
    gene_id: str
    relation: NLRelation = NLRelation.INDEPENDENCE
    confidence: float = 0.0


class EpigeneticMarkers(BaseModel):
    """Usage and association metadata — how the gene evolves over time."""
    created_at: float = Field(default_factory=lambda: time.time())
    last_accessed: float = Field(default_factory=lambda: time.time())
    access_count: int = 0
    co_activated_with: List[str] = Field(default_factory=list)
    typed_co_activated: List[TypedCoActivation] = Field(default_factory=list)
    decay_score: float = 1.0


class Gene(BaseModel):
    """The fundamental storage unit in the genome."""
    gene_id: str
    content: str
    complement: str                     # Dense summary (fallback for splice failures)
    codons: List[str]                   # Semantic meaning labels

    promoter: PromoterTags = Field(default_factory=PromoterTags)
    epigenetics: EpigeneticMarkers = Field(default_factory=EpigeneticMarkers)

    chromatin: ChromatinState = ChromatinState.OPEN
    is_fragment: bool = False

    embedding: Optional[List[float]] = None

    # Versioning (future — can ignore for MVP)
    source_id: Optional[str] = None
    version: Optional[int] = None
    supersedes: Optional[str] = None


class ContextHealth(BaseModel):
    """Delta-epsilon context health signal — the 'Check Engine Light.'"""
    ellipticity: float = 1.0            # 0=denatured, 1=perfectly grounded
    coverage: float = 0.0               # Fraction of query terms that matched genes
    density: float = 0.0                # Fraction of expression budget used
    freshness: float = 1.0              # Average decay score of expressed genes
    logical_coherence: float = 0.0      # Pairwise relation coherence of expressed genes
    genes_available: int = 0            # Total genes in genome
    genes_expressed: int = 0            # Genes expressed for this query
    status: str = "unmeasured"          # aligned | sparse | stale | denatured


class ContextWindow(BaseModel):
    """The assembled context ready for the big model."""
    ribosome_prompt: str                # 3k fixed decoder layer
    expressed_context: str              # 6k codon-encoded active context
    expressed_gene_ids: List[str] = Field(default_factory=list)
    total_estimated_tokens: int = 0
    compression_ratio: float = 0.0
    context_health: ContextHealth = Field(default_factory=ContextHealth)
    metadata: dict = Field(default_factory=dict)

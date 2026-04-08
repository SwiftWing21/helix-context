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


class EpigeneticMarkers(BaseModel):
    """Usage and association metadata — how the gene evolves over time."""
    created_at: float = Field(default_factory=lambda: time.time())
    last_accessed: float = Field(default_factory=lambda: time.time())
    access_count: int = 0
    co_activated_with: List[str] = Field(default_factory=list)
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


class ContextWindow(BaseModel):
    """The assembled context ready for the big model."""
    ribosome_prompt: str                # 3k fixed decoder layer
    expressed_context: str              # 6k codon-encoded active context
    expressed_gene_ids: List[str] = Field(default_factory=list)
    total_estimated_tokens: int = 0
    compression_ratio: float = 0.0
    metadata: dict = Field(default_factory=dict)

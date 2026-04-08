"""
Helix Context — Genome-based context compression for local LLMs.

Makes 9k tokens of context window feel like 600k by treating
context like a genome instead of a flat text buffer.
"""

from .config import HelixConfig, load_config
from .schemas import Gene, ContextWindow, ContextHealth, ChromatinState, PromoterTags, EpigeneticMarkers
from .genome import Genome
from .ribosome import Ribosome, OllamaBackend
from .codons import CodonChunker, CodonEncoder, RawStrand, Codon
from .context_manager import HelixContextManager
from .server import create_app
from .exceptions import (
    HelixError,
    CodonAlignmentError,
    PromoterMismatch,
    FoldingError,
    TranscriptionError,
    GenomeFullError,
)

__all__ = [
    "HelixConfig",
    "load_config",
    "Gene",
    "ContextWindow",
    "ContextHealth",
    "ChromatinState",
    "PromoterTags",
    "EpigeneticMarkers",
    "Genome",
    "Ribosome",
    "OllamaBackend",
    "CodonChunker",
    "CodonEncoder",
    "RawStrand",
    "Codon",
    "HelixContextManager",
    "create_app",
    "HelixError",
    "CodonAlignmentError",
    "PromoterMismatch",
    "FoldingError",
    "TranscriptionError",
    "GenomeFullError",
]

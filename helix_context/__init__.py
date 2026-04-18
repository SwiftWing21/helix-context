"""
Helix Context — Genome-based context compression for local LLMs.

Makes 9k tokens of context window feel like 600k by treating
context like a genome instead of a flat text buffer.
"""

from .accel import accel_info, JSON_BACKEND
from .config import HelixConfig, load_config
from .schemas import Gene, ContextWindow, ContextHealth, ChromatinState, PromoterTags, EpigeneticMarkers
from .genome import Genome
from .ribosome import Ribosome, OllamaBackend
from .codons import CodonChunker, CodonEncoder, RawStrand, Codon
from .context_manager import HelixContextManager
from .exceptions import (
    HelixError,
    CodonAlignmentError,
    PromoterMismatch,
    FoldingError,
    TranscriptionError,
    GenomeFullError,
)

# CpuTagger is optional (requires spacy)
try:
    from .tagger import CpuTagger
except ImportError:
    CpuTagger = None

from .replication import ReplicationManager

# ΣĒMA is optional (requires sentence-transformers)
try:
    from .sema import SemaCodec, SemaPrime, PRIMES, PRIME_COUNT
except ImportError:
    SemaCodec = None
    SemaPrime = None
    PRIMES = None
    PRIME_COUNT = None


def create_app(*args, **kwargs):
    """Lazy server import so package import has no HTTP/app side effects."""
    from .server import create_app as _create_app

    return _create_app(*args, **kwargs)

__all__ = [
    "accel_info",
    "JSON_BACKEND",
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
    "SemaCodec",
    "SemaPrime",
    "PRIMES",
    "PRIME_COUNT",
    "CpuTagger",
]

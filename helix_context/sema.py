"""
ΣĒMA — Post-linguistic semantic encoding.

Sigma-Epsilon-Mu-Alpha: a 20-dimensional universal semantic coordinate
system for context genes. Each dimension (a "prime") captures a
fundamental axis of meaning that holds across all domains — code,
prose, conversation, data.

Biology analogy:
    If promoter tags are restriction enzyme binding sites (discrete),
    ΣĒMA vectors are the 3D protein fold coordinates (continuous).
    Tags find the neighborhood; ΣĒMA finds the exact position.

Architecture:
    1. Anchor sentences define each prime's semantic pole
    2. A sentence-transformer encodes anchors into 384D space
    3. The anchor matrix A (20 × 384) becomes a fixed projection basis
    4. For any text: embed(text) @ A.T → 20D ΣĒMA vector
    5. Cosine similarity in 20D = semantic relatedness

Usage:
    from helix_context.sema import SemaCodec

    codec = SemaCodec()                  # loads model + builds projection
    vec = codec.encode("some text")      # → list[float] len=20
    sim = codec.similarity(vec_a, vec_b) # → float [-1, 1]
    nearest = codec.nearest(query_vec, candidates, k=5)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("helix.sema")


# ── The 20 Universal Primes ──────────────────────────────────────────
#
# Each prime is a semantic axis defined by an anchor sentence that
# captures the "positive pole" of that dimension. The anchor's
# embedding in transformer space becomes the projection direction.
#
# Selection criteria:
#   - Domain-agnostic (works for code, prose, config, data)
#   - Maximally orthogonal (each captures a distinct meaning axis)
#   - Retrieval-useful (genes close in prime-space are semantically related)

@dataclass(frozen=True)
class SemaPrime:
    """A single semantic dimension — one axis of meaning."""
    index: int
    name: str
    symbol: str        # Short glyph for compact display
    anchor: str        # Anchor sentence defining this prime's direction
    description: str


PRIMES: List[SemaPrime] = [
    SemaPrime(0,  "agency",       "Ag", "An agent performs actions and makes decisions autonomously",
              "Who acts — agents, users, systems, tools, actors"),
    SemaPrime(1,  "process",      "Pr", "A sequential process transforms input through multiple steps into output",
              "What happens — actions, transformations, workflows, pipelines"),
    SemaPrime(2,  "structure",    "St", "Data is organized in a hierarchical tree structure with parent-child relationships",
              "How it's organized — hierarchy, composition, patterns, layout"),
    SemaPrime(3,  "quantity",     "Qt", "The measurement contains specific numeric values, counts, and thresholds",
              "Amounts — numbers, counts, measurements, sizes, limits"),
    SemaPrime(4,  "temporality",  "Tm", "Events occur in chronological sequence with timestamps and durations",
              "Time — sequence, duration, scheduling, freshness, order"),
    SemaPrime(5,  "causality",    "Ca", "This component depends on another and will fail if the dependency is missing",
              "Why — cause/effect, dependencies, triggers, consequences"),
    SemaPrime(6,  "modality",     "Md", "The configuration option might be required or optional depending on the context",
              "Certainty — possibility, necessity, optionality, conditions"),
    SemaPrime(7,  "evaluation",   "Ev", "The benchmark measures performance quality with accuracy and latency metrics",
              "Quality — correctness, performance, scoring, assessment"),
    SemaPrime(8,  "specificity",  "Sp", "The variable holds the concrete value 127.0.0.1 port 8080 on localhost",
              "Concrete vs abstract — specific instances, examples, literals"),
    SemaPrime(9,  "complexity",   "Cx", "The algorithm combines multiple nested data structures and recursive logic",
              "Simple vs compound — nesting, composition, layering"),
    SemaPrime(10, "domain",       "Dm", "This belongs to the field of machine learning and natural language processing",
              "Knowledge field — which discipline, technology, subject area"),
    SemaPrime(11, "relation",     "Rl", "The function calls another module which returns data to the caller",
              "Connections — links between entities, references, associations"),
    SemaPrime(12, "boundary",     "Bd", "The system enforces limits on memory usage, request rate, and file size",
              "Scope — constraints, limits, permissions, access control"),
    SemaPrime(13, "state",        "Sn", "The server is currently running in production mode with debug disabled",
              "Condition — current status, configuration, mode, flags"),
    SemaPrime(14, "identity",     "Id", "The class is named HelixContextManager and is defined in context_manager.py",
              "Naming — definitions, references, identifiers, types"),
    SemaPrime(15, "instruction",  "In", "To install the package run pip install helix-context and configure helix.toml",
              "Directives — how-to, commands, setup steps, recipes"),
    SemaPrime(16, "data",         "Da", "The JSON response contains fields for name, status, count, and timestamp",
              "Information — values, types, formats, schemas, payloads"),
    SemaPrime(17, "interface",    "If", "The REST API endpoint accepts POST requests and returns JSON responses",
              "Input/output — APIs, protocols, contracts, ports, endpoints"),
    SemaPrime(18, "error",        "Er", "The operation failed with a timeout exception and needs retry logic",
              "Failure — exceptions, bugs, recovery, fallbacks, handling"),
    SemaPrime(19, "context",      "Ct", "The surrounding environment includes the operating system, runtime, and configuration",
              "Environment — surrounding information, prerequisites, setup"),
]

PRIME_COUNT = len(PRIMES)
PRIME_BY_NAME: Dict[str, SemaPrime] = {p.name: p for p in PRIMES}


# ── Codec ─────────────────────────────────────────────────────────────

class SemaCodec:
    """
    Encodes text into 20D ΣĒMA space using sentence-transformer projection.

    The projection matrix is built once from anchor sentences and reused
    for all subsequent encodings. No training required — the anchors
    define the coordinate system.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._embed_dim = self._model.get_sentence_embedding_dimension()

        # Build projection matrix from prime anchors
        self._projection = self._build_projection()

        log.info(
            "ΣĒMA codec ready: %dD → %dD via %s",
            self._embed_dim, PRIME_COUNT, model_name,
        )

    def _build_projection(self) -> np.ndarray:
        """
        Build the (20 × embed_dim) projection matrix from anchor sentences.

        Each row is the normalized embedding of a prime's anchor sentence.
        Projecting any text embedding onto this matrix gives its 20D
        ΣĒMA coordinates (cosine similarity with each anchor direction).
        """
        anchors = [p.anchor for p in PRIMES]
        embeddings = self._model.encode(anchors, normalize_embeddings=True)
        # Shape: (20, embed_dim) — each row is a unit vector
        return np.array(embeddings, dtype=np.float32)

    def encode(self, text: str) -> List[float]:
        """
        Encode text into a 20D ΣĒMA vector.

        Returns list of 20 floats, each in [-1, 1], representing the
        text's projection onto each semantic prime axis.
        """
        embedding = self._model.encode(text, normalize_embeddings=True)
        # Project: (embed_dim,) @ (embed_dim, 20) → (20,)
        sema_vec = embedding @ self._projection.T
        return sema_vec.tolist()

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Encode multiple texts into ΣĒMA vectors. Batched for efficiency."""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, batch_size=batch_size,
        )
        # (N, embed_dim) @ (embed_dim, 20) → (N, 20)
        sema_matrix = embeddings @ self._projection.T
        return sema_matrix.tolist()

    @staticmethod
    def similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Cosine similarity between two ΣĒMA vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def nearest(
        query_vec: List[float],
        candidates: List[Tuple[str, List[float]]],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest candidates to query_vec in ΣĒMA space.

        Args:
            query_vec: 20D ΣĒMA vector
            candidates: list of (id, sema_vector) pairs
            k: number of nearest neighbors

        Returns:
            list of (id, similarity) sorted by similarity descending
        """
        if not candidates:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return [(cid, 0.0) for cid, _ in candidates[:k]]

        q = q / q_norm

        scores = []
        for cid, cvec in candidates:
            c = np.array(cvec, dtype=np.float32)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-8:
                scores.append((cid, 0.0))
            else:
                scores.append((cid, float(np.dot(q, c / c_norm))))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def signature(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Human-readable ΣĒMA signature: top-k strongest primes for this text.

        Returns list of (prime_name, score) sorted by absolute magnitude.
        """
        vec = self.encode(text)
        scored = [(PRIMES[i].name, vec[i]) for i in range(PRIME_COUNT)]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]

    def fingerprint(self, text: str) -> str:
        """
        Compact string fingerprint: top-5 primes with 2-char symbols.

        Example: "Pr.82 If.71 Da.65 St.58 Id.44"
        """
        vec = self.encode(text)
        scored = [(PRIMES[i].symbol, vec[i]) for i in range(PRIME_COUNT)]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return " ".join(f"{sym}.{int(abs(score)*100):02d}" for sym, score in scored[:5])

    @property
    def projection_matrix(self) -> np.ndarray:
        """The (20 × embed_dim) projection matrix."""
        return self._projection

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

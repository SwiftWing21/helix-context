"""
Codons — Semantic chunking and codon encoding.

Two distinct roles:
    1. CodonChunker  — restriction enzyme: cuts raw text into RawStrands
       (pre-gene chunks sized for the ribosome to process)
    2. CodonEncoder  — serialization: converts codon meaning labels into
       prompt-ready strings for the big model

Biology:
    DNA codons are nucleotide triplets mapping to amino acids.
    Our codons are semantic groups mapping to meaning labels.
    The ribosome (small model) does the actual encoding;
    this module provides the chunking and serialization primitives.
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ── Pre-gene chunk (output of chunking, input to ribosome) ──────────

@dataclass
class RawStrand:
    """A pre-gene chunk of raw text waiting for Ribosome translation."""
    content: str
    sequence_index: int
    is_fragment: bool
    content_type: str           # "text", "code", "conversation"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Encoded codon (output of ribosome pack) ─────────────────────────

@dataclass
class Codon:
    """A semantic unit — the fundamental piece of compressed context."""
    tokens: List[str]           # Raw tokens that make up this codon
    meaning: str                # Semantic label / compressed representation
    weight: float = 1.0         # 1.0=critical, 0.5=useful, 0.1=filler
    is_exon: bool = True        # True=load-bearing, False=can be spliced out


# ── Chunker ─────────────────────────────────────────────────────────

class CodonChunker:
    """
    Restriction enzyme — cuts raw content into RawStrands.

    Domain-aware: text (paragraphs), code (functions/classes),
    conversation (turn pairs). Each strategy preserves reading order
    via sequence_index and flags forced cuts via is_fragment.
    """

    def __init__(self, max_chars_per_strand: int = 4000):
        # ~4000 chars ≈ ~1000 tokens, safe for a small ribosome model
        self.max_chars = max_chars_per_strand

    def chunk(
        self,
        content: str,
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawStrand]:
        """Route to the appropriate domain-aware chunking strategy."""
        metadata = metadata or {}
        if content_type == "code":
            return self._chunk_code(content, metadata)
        elif content_type == "conversation":
            return self._chunk_conversation(content, metadata)
        return self._chunk_text(content, metadata)

    # ── Text chunking (paragraph-first, sentence fallback) ──────────

    def _chunk_text(self, text: str, metadata: Dict) -> List[RawStrand]:
        paragraphs = re.split(r"\n\s*\n", text)
        strands: List[RawStrand] = []
        current = ""
        seq = 0

        for p in paragraphs:
            if len(current) + len(p) < self.max_chars:
                current += p + "\n\n"
            else:
                if current:
                    strands.append(RawStrand(
                        content=current.strip(),
                        sequence_index=seq,
                        is_fragment=False,
                        content_type="text",
                        metadata=metadata,
                    ))
                    seq += 1

                if len(p) >= self.max_chars:
                    # Hard cut — polyadenylation trigger
                    strands.append(RawStrand(
                        content=p[: self.max_chars],
                        sequence_index=seq,
                        is_fragment=True,
                        content_type="text",
                        metadata=metadata,
                    ))
                    seq += 1
                    current = p[self.max_chars :] + "\n\n"
                else:
                    current = p + "\n\n"

        if current.strip():
            strands.append(RawStrand(
                content=current.strip(),
                sequence_index=seq,
                is_fragment=False,
                content_type="text",
                metadata=metadata,
            ))

        return strands

    # ── Code chunking (function/class boundary splitting) ───────────

    def _chunk_code(self, code: str, metadata: Dict) -> List[RawStrand]:
        # Split on top-level definitions (MVP heuristic — swap for tree-sitter later)
        blocks = re.split(
            r"(^(?:def |class |async def |struct |interface |type |export ))",
            code,
            flags=re.MULTILINE,
        )

        # Re-stitch split delimiters with their content
        stitched: List[str] = []
        if blocks and not re.match(
            r"^(?:def |class |async def |struct |interface |type |export )", blocks[0]
        ):
            stitched.append(blocks[0])
            blocks = blocks[1:]

        for i in range(0, len(blocks), 2):
            if i + 1 < len(blocks):
                stitched.append(blocks[i] + blocks[i + 1])
            elif blocks[i].strip():
                stitched.append(blocks[i])

        strands: List[RawStrand] = []
        current = ""
        seq = 0

        for block in stitched:
            if len(current) + len(block) < self.max_chars:
                current += block
            else:
                if current:
                    strands.append(RawStrand(
                        content=current.strip(),
                        sequence_index=seq,
                        is_fragment=False,
                        content_type="code",
                        metadata=metadata,
                    ))
                    seq += 1

                if len(block) >= self.max_chars:
                    strands.append(RawStrand(
                        content=block[: self.max_chars],
                        sequence_index=seq,
                        is_fragment=True,
                        content_type="code",
                        metadata=metadata,
                    ))
                    seq += 1
                    current = block[self.max_chars :]
                else:
                    current = block

        if current.strip():
            strands.append(RawStrand(
                content=current.strip(),
                sequence_index=seq,
                is_fragment=False,
                content_type="code",
                metadata=metadata,
            ))

        return strands

    # ── Conversation chunking (turn pairs) ──────────────────────────

    def _chunk_conversation(self, conversation: str, metadata: Dict) -> List[RawStrand]:
        # MVP: fall back to text chunking — the proxy layer handles
        # conversations as structured JSON, not raw strings.
        return self._chunk_text(conversation, metadata)


# ── Encoder (serialization for prompt injection) ────────────────────

class CodonEncoder:
    """
    Serializes codon meaning labels for injection into the big model's prompt.
    Also provides sentence-level chunking used by the ribosome's pack operation.
    """

    def __init__(self, chunk_target: int = 3, overlap: int = 0):
        self.chunk_target = chunk_target
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[List[str]]:
        """Split text into sentence groups (proto-codons for ribosome pack)."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        groups: List[List[str]] = []
        i = 0
        while i < len(sentences):
            end = min(i + self.chunk_target, len(sentences))
            groups.append(sentences[i:end])
            i = end - self.overlap if self.overlap else end
        return groups

    def chunk_code(self, code: str) -> List[List[str]]:
        """Split code into logical blocks (one block = one codon group)."""
        blocks = self._split_code_blocks(code)
        return [[b] for b in blocks] if blocks else [[code]]

    def chunk_conversation(self, messages: List[Dict]) -> List[List[str]]:
        """Split conversation into turn-pair groups."""
        groups: List[List[str]] = []
        for i in range(0, len(messages), 2):
            pair = messages[i : i + 2]
            group = [f"{m.get('role', '?')}: {m.get('content', '')}" for m in pair]
            groups.append(group)
        return groups

    def codons_to_sequence(self, codons: List[Codon], exon_only: bool = False) -> str:
        """Serialize codons into a compact string representation."""
        filtered = [c for c in codons if c.is_exon] if exon_only else codons
        return " ".join(f"[{c.meaning}|w={c.weight:.1f}]" for c in filtered)

    def sequence_to_prompt(self, expressed: str) -> str:
        """Wrap expressed context for injection into the big model's prompt."""
        return (
            "<expressed_context>\n"
            f"{expressed}\n"
            "</expressed_context>"
        )

    @staticmethod
    def codon_id(tokens: List[str]) -> str:
        raw = "||".join(tokens)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _split_code_blocks(code: str) -> List[str]:
        pattern = r"^(?=(?:def |class |async def |function |const |export ))"
        blocks = re.split(pattern, code, flags=re.MULTILINE)
        return [b.strip() for b in blocks if b.strip()]


# ── Compression metrics ─────────────────────────────────────────────

def compression_ratio(raw_text: str, codons: List[Codon], exon_only: bool = True) -> float:
    """How much we compressed. Higher = more compression."""
    encoder = CodonEncoder()
    compressed = encoder.codons_to_sequence(codons, exon_only=exon_only)
    return len(raw_text) / max(len(compressed), 1)

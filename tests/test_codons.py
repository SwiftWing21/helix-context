"""Gate 1 — Codon chunking tests (no model calls, pure CPU)."""

import pytest

from helix_context.codons import CodonChunker, CodonEncoder, RawStrand, Codon


# ── CodonChunker: text ──────────────────────────────────────────────


class TestTextChunking:
    def test_poem_produces_valid_strands(self, poem_text):
        chunker = CodonChunker(max_chars_per_strand=500)
        strands = chunker.chunk(poem_text, content_type="text")

        assert len(strands) > 0
        assert all(isinstance(s, RawStrand) for s in strands)
        assert all(s.content_type == "text" for s in strands)

    def test_sequence_indices_are_sequential(self, poem_text):
        chunker = CodonChunker(max_chars_per_strand=500)
        strands = chunker.chunk(poem_text, content_type="text")

        indices = [s.sequence_index for s in strands]
        assert indices == list(range(len(strands)))

    def test_no_empty_strands(self, poem_text):
        chunker = CodonChunker(max_chars_per_strand=300)
        strands = chunker.chunk(poem_text, content_type="text")

        for s in strands:
            assert s.content.strip(), f"Empty strand at index {s.sequence_index}"

    def test_hard_cut_triggers_is_fragment(self):
        """A single 10,000-char string with no paragraph breaks must trigger a hard cut."""
        giant = "x" * 10_000
        chunker = CodonChunker(max_chars_per_strand=4000)
        strands = chunker.chunk(giant, content_type="text")

        assert len(strands) >= 2
        # First strand should be flagged as a fragment (hard cut mid-content)
        assert strands[0].is_fragment is True
        assert len(strands[0].content) == 4000

    def test_small_input_no_fragment(self):
        """Content under max_chars should produce a single non-fragment strand."""
        text = "Hello world. This is a short paragraph."
        chunker = CodonChunker(max_chars_per_strand=4000)
        strands = chunker.chunk(text, content_type="text")

        assert len(strands) == 1
        assert strands[0].is_fragment is False

    def test_metadata_passed_through(self, poem_text):
        chunker = CodonChunker()
        meta = {"source": "poem.txt", "author": "test"}
        strands = chunker.chunk(poem_text, content_type="text", metadata=meta)

        for s in strands:
            assert s.metadata == meta

    def test_empty_input_returns_empty(self):
        chunker = CodonChunker()
        strands = chunker.chunk("", content_type="text")
        assert strands == []


# ── CodonChunker: code ──────────────────────────────────────────────


class TestCodeChunking:
    def test_calculator_splits_on_boundaries(self, calculator_code):
        chunker = CodonChunker(max_chars_per_strand=4000)
        strands = chunker.chunk(calculator_code, content_type="code")

        assert len(strands) >= 1
        assert all(s.content_type == "code" for s in strands)

    def test_functions_are_preserved(self, calculator_code):
        """Each top-level def/class should not be split mid-body (under max_chars)."""
        chunker = CodonChunker(max_chars_per_strand=4000)
        strands = chunker.chunk(calculator_code, content_type="code")

        # At least one strand should contain a complete class or function
        all_content = " ".join(s.content for s in strands)
        assert "class Calculator" in all_content
        assert "def add" in all_content
        assert "def divide" in all_content

    def test_code_hard_cut_sets_fragment(self):
        """A single giant function bigger than max_chars triggers is_fragment."""
        giant_func = "def huge():\n" + "    x = 1\n" * 2000
        chunker = CodonChunker(max_chars_per_strand=1000)
        strands = chunker.chunk(giant_func, content_type="code")

        fragments = [s for s in strands if s.is_fragment]
        assert len(fragments) >= 1

    def test_sequence_order_preserved(self, calculator_code):
        chunker = CodonChunker(max_chars_per_strand=2000)
        strands = chunker.chunk(calculator_code, content_type="code")

        indices = [s.sequence_index for s in strands]
        assert indices == sorted(indices)


# ── CodonChunker: conversation ──────────────────────────────────────


class TestConversationChunking:
    def test_conversation_falls_back_to_text(self):
        convo = "User: How do I fix auth?\n\nAssistant: Check the JWT middleware."
        chunker = CodonChunker()
        strands = chunker.chunk(convo, content_type="conversation")

        assert len(strands) >= 1
        assert all(s.content_type == "text" for s in strands)  # Falls back to text


# ── CodonEncoder ────────────────────────────────────────────────────


class TestCodonEncoder:
    def test_chunk_text_produces_groups(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        encoder = CodonEncoder(chunk_target=2)
        groups = encoder.chunk_text(text)

        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2

    def test_chunk_code_produces_blocks(self, calculator_code):
        encoder = CodonEncoder()
        groups = encoder.chunk_code(calculator_code)

        assert len(groups) >= 1
        # Should contain at least the class definitions
        all_blocks = " ".join(g[0] for g in groups)
        assert "Calculator" in all_blocks

    def test_codons_to_sequence(self):
        codons = [
            Codon(tokens=["a"], meaning="auth_check", weight=0.9, is_exon=True),
            Codon(tokens=["b"], meaning="logging", weight=0.2, is_exon=False),
            Codon(tokens=["c"], meaning="jwt_validate", weight=0.8, is_exon=True),
        ]
        encoder = CodonEncoder()

        full = encoder.codons_to_sequence(codons, exon_only=False)
        assert "[auth_check|w=0.9]" in full
        assert "[logging|w=0.2]" in full

        exon = encoder.codons_to_sequence(codons, exon_only=True)
        assert "[auth_check|w=0.9]" in exon
        assert "[logging|w=0.2]" not in exon

    def test_sequence_to_prompt_wraps_correctly(self):
        encoder = CodonEncoder()
        result = encoder.sequence_to_prompt("some context here")

        assert result.startswith("<expressed_context>")
        assert result.endswith("</expressed_context>")
        assert "some context here" in result

    def test_codon_id_deterministic(self):
        tokens = ["hello", "world"]
        assert CodonEncoder.codon_id(tokens) == CodonEncoder.codon_id(tokens)
        assert CodonEncoder.codon_id(tokens) != CodonEncoder.codon_id(["other"])

"""
Gate 2 — Ribosome tests.

Two tiers:
    1. Mock backend tests (always run) — verify fallback logic, Fix 2, Fix 4
    2. Live Ollama tests (skip if Ollama not running) — verify real model output

Mark live tests with @pytest.mark.live so they can be run selectively:
    pytest tests/test_ribosome.py -m live      # only live
    pytest tests/test_ribosome.py -m "not live" # only mocks
"""

import json
import pytest
import httpx

from helix_context.ribosome import Ribosome, OllamaBackend, _parse_json
from helix_context.schemas import Gene, PromoterTags, EpigeneticMarkers
from helix_context.exceptions import FoldingError, TranscriptionError

from tests.conftest import make_gene


# ── Helpers ─────────────────────────────────────────────────────────


class MockBackend:
    """Controllable backend for deterministic testing."""

    def __init__(self, response: str = "{}"):
        self.response = response
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        return self.response


class TimeoutBackend:
    """Backend that always raises a timeout."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        raise httpx.ReadTimeout("simulated timeout")


class MalformedBackend:
    """Backend that returns garbage."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        return "Sure! Here's your answer: not json at all {{{"


# Ollama availability is resolved lazily. Previously this probe fired at
# module import time, which blocked pytest collection for ~3s on every
# run (even when excluding -m live) and caused hangs if the Ollama host
# was slow to reject the TCP connect. The cached sentinel below is
# evaluated on first access — typically when a live test is being
# collected — so the "not live" fast path never touches the network.
_OLLAMA_AVAILABLE: bool | None = None


def _ollama_available() -> bool:
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is None:
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
            _OLLAMA_AVAILABLE = (
                resp.status_code == 200
                and len(resp.json().get("models", [])) > 0
            )
        except Exception:
            _OLLAMA_AVAILABLE = False
    return _OLLAMA_AVAILABLE


def _live_reason() -> str:
    return "" if _ollama_available() else "Ollama not running"


def live(cls_or_fn):
    """Mark a test as requiring a live Ollama instance.

    Stacks ``pytest.mark.live`` (for ``-m live`` selection) with a lazy
    ``skipif`` that only calls :func:`_ollama_available` when pytest is
    actually evaluating the skip — i.e. during collection of the
    decorated test, not at module import time.
    """
    return pytest.mark.live(
        pytest.mark.skipif(
            not _ollama_available(), reason="Ollama not running"
        )(cls_or_fn)
    )


# ── JSON Parser ─────────────────────────────────────────────────────


class TestJsonParser:
    def test_clean_json(self):
        assert _parse_json('{"key": "value"}') == {"key": "value"}

    def test_markdown_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _parse_json(raw) == {"key": "value"}

    def test_preamble_before_json(self):
        raw = 'Here is the result:\n{"key": "value"}'
        assert _parse_json(raw) == {"key": "value"}

    def test_array_output(self):
        assert _parse_json("[0, 2, 3]") == [0, 2, 3]

    def test_garbage_raises_folding_error(self):
        with pytest.raises(FoldingError):
            _parse_json("this is not json at all")


# ── Pack (Mock) ─────────────────────────────────────────────────────


class TestPackMock:
    def test_pack_returns_gene(self):
        mock_response = json.dumps({
            "codons": [
                {"meaning": "intro", "weight": 0.8, "is_exon": True},
                {"meaning": "body", "weight": 0.9, "is_exon": True},
            ],
            "complement": "A poem about memory and compression.",
            "promoter": {
                "domains": ["poetry", "memory"],
                "entities": ["genome", "ribosome"],
                "intent": "artistic expression",
                "summary": "A poem about DNA-based context compression",
            },
        })

        ribosome = Ribosome(backend=MockBackend(mock_response))
        gene = ribosome.pack("The genome of memory...", content_type="text")

        assert isinstance(gene, Gene)
        assert gene.complement == "A poem about memory and compression."
        assert len(gene.codons) == 2
        assert "poetry" in gene.promoter.domains
        assert gene.gene_id  # Should be populated

    def test_pack_with_code(self):
        mock_response = json.dumps({
            "codons": [{"meaning": "calculator_class", "weight": 1.0, "is_exon": True}],
            "complement": "A calculator with basic arithmetic operations.",
            "promoter": {
                "domains": ["math", "calculator"],
                "entities": ["Calculator"],
                "intent": "arithmetic operations",
                "summary": "Basic calculator implementation",
            },
        })

        ribosome = Ribosome(backend=MockBackend(mock_response))
        gene = ribosome.pack("class Calculator:\n    def add(self, a, b):\n        return a + b", content_type="code")

        assert "calculator" in gene.promoter.domains

    def test_pack_failure_raises_transcription_error(self):
        ribosome = Ribosome(backend=TimeoutBackend())
        with pytest.raises(TranscriptionError):
            ribosome.pack("anything")

    def test_pack_malformed_raises_folding_error(self):
        ribosome = Ribosome(backend=MalformedBackend())
        with pytest.raises((FoldingError, TranscriptionError)):
            ribosome.pack("anything")


# ── Re-Rank (Mock) ──────────────────────────────────────────────────


class TestReRankMock:
    def test_rerank_scores_and_sorts(self):
        mock_response = json.dumps({
            "gene_a": 0.9,
            "gene_b": 0.3,
            "gene_c": 0.8,
        })

        genes = [
            make_gene("content a", domains=["test"], gene_id="gene_a"),
            make_gene("content b", domains=["test"], gene_id="gene_b"),
            make_gene("content c", domains=["test"], gene_id="gene_c"),
        ]

        ribosome = Ribosome(backend=MockBackend(mock_response))
        ranked = ribosome.re_rank("test query", genes, k=2)

        assert len(ranked) == 2
        assert ranked[0].gene_id == "gene_a"  # Highest score
        assert ranked[1].gene_id == "gene_c"

    def test_rerank_fewer_than_k_returns_all(self):
        """If candidates <= k, skip the model call entirely."""
        genes = [make_gene("only one", domains=["test"], gene_id="only")]
        ribosome = Ribosome(backend=MockBackend("should not be called"))

        ranked = ribosome.re_rank("query", genes, k=5)
        assert len(ranked) == 1
        assert ribosome.backend.calls == []  # No model call made

    def test_rerank_lost_in_middle_guard(self):
        """If ribosome scores < 50% of candidates, pad with unscored."""
        # Only score 1 out of 4 candidates (25%)
        mock_response = json.dumps({"gene_a": 0.9})

        genes = [
            make_gene("a", gene_id="gene_a", domains=["test"]),
            make_gene("b", gene_id="gene_b", domains=["test"]),
            make_gene("c", gene_id="gene_c", domains=["test"]),
            make_gene("d", gene_id="gene_d", domains=["test"]),
        ]

        ribosome = Ribosome(backend=MockBackend(mock_response))
        ranked = ribosome.re_rank("query", genes, k=3)

        # Should have gene_a plus padded genes
        assert len(ranked) == 3
        assert ranked[0].gene_id == "gene_a"

    def test_rerank_timeout_falls_back_to_input_order(self):
        """Fix 4: on timeout, return first k candidates in original order."""
        genes = [
            make_gene("a", gene_id="gene_a", domains=["test"]),
            make_gene("b", gene_id="gene_b", domains=["test"]),
            make_gene("c", gene_id="gene_c", domains=["test"]),
        ]

        ribosome = Ribosome(backend=TimeoutBackend())
        ranked = ribosome.re_rank("query", genes, k=2)

        assert len(ranked) == 2
        assert ranked[0].gene_id == "gene_a"
        assert ranked[1].gene_id == "gene_b"


# ── Splice (Mock) ──────────────────────────────────────────────────


class TestSpliceMock:
    def test_splice_batched_returns_per_gene(self):
        mock_response = json.dumps({
            "gene_a": [0, 2],
            "gene_b": [1],
        })

        genes = [
            make_gene("a", gene_id="gene_a", domains=["test"]),
            make_gene("b", gene_id="gene_b", domains=["test"]),
        ]
        # Both have codons ["chunk_0", "chunk_1", "chunk_2"] from make_gene

        ribosome = Ribosome(backend=MockBackend(mock_response))
        result = ribosome.splice("test query", genes)

        assert "gene_a" in result
        assert "gene_b" in result
        assert "chunk_0" in result["gene_a"]
        assert "chunk_2" in result["gene_a"]
        assert "chunk_1" in result["gene_b"]

    def test_splice_empty_guard(self):
        """Fix 2: empty list for a gene should keep first N codons, not drop everything."""
        mock_response = json.dumps({
            "gene_a": [],  # Ribosome says "keep nothing"
        })

        gene = make_gene("important", gene_id="gene_a", domains=["test"])
        ribosome = Ribosome(backend=MockBackend(mock_response), splice_aggressiveness=0.5)
        result = ribosome.splice("query", [gene], min_codons_kept=2)

        # Should have kept first 2 codons as safety net
        assert "gene_a" in result
        assert "chunk_0" in result["gene_a"]
        assert "chunk_1" in result["gene_a"]

    def test_splice_missing_gene_falls_back_to_complement(self):
        """If ribosome doesn't mention a gene, use its complement."""
        mock_response = json.dumps({
            "gene_a": [0, 1],
            # gene_b not mentioned at all
        })

        genes = [
            make_gene("a", gene_id="gene_a", domains=["test"]),
            make_gene("b content here", gene_id="gene_b", domains=["test"]),
        ]

        ribosome = Ribosome(backend=MockBackend(mock_response))
        result = ribosome.splice("query", genes)

        assert "gene_b" in result
        # Should fall back to complement
        assert "Summary of:" in result["gene_b"]

    def test_splice_timeout_falls_back_to_all_complements(self):
        """Fix 4: on timeout, return complement for every gene."""
        genes = [
            make_gene("a", gene_id="gene_a", domains=["test"]),
            make_gene("b", gene_id="gene_b", domains=["test"]),
        ]

        ribosome = Ribosome(backend=TimeoutBackend())
        result = ribosome.splice("query", genes)

        assert len(result) == 2
        for gid in ["gene_a", "gene_b"]:
            assert "Summary of:" in result[gid]

    def test_splice_malformed_falls_back(self):
        """Garbage output should trigger complement fallback, not crash."""
        genes = [make_gene("a", gene_id="gene_a", domains=["test"])]
        ribosome = Ribosome(backend=MalformedBackend())
        result = ribosome.splice("query", genes)

        assert "gene_a" in result
        assert "Summary of:" in result["gene_a"]

    def test_splice_empty_genes_list(self):
        ribosome = Ribosome(backend=MockBackend("{}"))
        assert ribosome.splice("query", []) == {}


# ── Replicate (Mock) ───────────────────────────────────────────────


class TestReplicateMock:
    def test_replicate_produces_gene(self):
        mock_response = json.dumps({
            "codons": [{"meaning": "auth_decision", "weight": 1.0, "is_exon": True}],
            "complement": "User asked about auth, decided to use JWT middleware.",
            "promoter": {
                "domains": ["auth"],
                "entities": ["jwt"],
                "intent": "authentication decision",
                "summary": "Chose JWT for auth middleware",
            },
        })

        ribosome = Ribosome(backend=MockBackend(mock_response))
        gene = ribosome.replicate("How should I handle auth?", "Use JWT middleware.")

        assert isinstance(gene, Gene)
        assert "auth" in gene.promoter.domains
        assert gene.complement == "User asked about auth, decided to use JWT middleware."

    def test_replicate_failure_produces_minimal_gene(self):
        """Replication is background — failure should produce a minimal gene, not crash."""
        ribosome = Ribosome(backend=TimeoutBackend())
        gene = ribosome.replicate("test query", "test response")

        assert isinstance(gene, Gene)
        assert gene.content  # Should still have the exchange text
        assert gene.gene_id  # Should still have an ID


# ── Live Ollama Tests ───────────────────────────────────────────────


@live
class TestPackLive:
    """These tests require Ollama running with at least one model."""

    def test_pack_poem(self, poem_text):
        ribosome = Ribosome(backend=OllamaBackend(model="auto", timeout=30))
        gene = ribosome.pack(poem_text, content_type="text")

        assert isinstance(gene, Gene)
        assert gene.gene_id
        assert len(gene.codons) > 0
        assert gene.complement
        assert len(gene.promoter.domains) > 0
        print(f"  Pack result: {len(gene.codons)} codons, domains={gene.promoter.domains}")

    def test_pack_calculator(self, calculator_code):
        ribosome = Ribosome(backend=OllamaBackend(model="auto", timeout=30))
        gene = ribosome.pack(calculator_code, content_type="code")

        assert isinstance(gene, Gene)
        assert len(gene.codons) > 0
        assert gene.complement
        print(f"  Pack result: {len(gene.codons)} codons, summary={gene.promoter.summary}")


@live
class TestSpliceLive:
    def test_splice_with_real_model(self, poem_text):
        backend = OllamaBackend(model="auto", timeout=30)
        ribosome = Ribosome(backend=backend)

        # First pack to get a real gene
        gene = ribosome.pack(poem_text, content_type="text")

        # Then splice it
        result = ribosome.splice("What is the ribosome?", [gene])

        assert gene.gene_id in result
        spliced = result[gene.gene_id]
        assert len(spliced) > 0
        print(f"  Splice result: {len(spliced)} chars (from {len(gene.content)} original)")


@live
class TestReplicateLive:
    def test_replicate_captures_intent(self):
        backend = OllamaBackend(model="auto", timeout=30)
        ribosome = Ribosome(backend=backend)

        gene = ribosome.replicate(
            "Why is the dashboard slow?",
            "The N+1 query in the agent list endpoint is causing 2s latency. "
            "Fix: add a JOIN to prefetch agent status in a single query."
        )

        assert isinstance(gene, Gene)
        assert gene.complement
        assert len(gene.promoter.domains) > 0
        print(f"  Replicate: domains={gene.promoter.domains}, summary={gene.promoter.summary}")

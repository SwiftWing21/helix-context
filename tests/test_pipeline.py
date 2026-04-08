"""
Gate 3 -- Pipeline tests (full stack, no HTTP).

Tests the HelixContextManager orchestrator with mock ribosome backend.
Validates the expression pipeline, pending buffer, history munging,
cold-start bootstrap (Fix 3), and build_context assembly.
"""

import json
import pytest

from helix_context.config import HelixConfig, BudgetConfig, GenomeConfig, RibosomeConfig
from helix_context.context_manager import HelixContextManager, RIBOSOME_DECODER
from helix_context.ribosome import Ribosome
from helix_context.genome import Genome
from helix_context.schemas import Gene, PromoterTags, EpigeneticMarkers
from helix_context.server import _munge_messages

from tests.conftest import make_gene


# -- Helpers -----------------------------------------------------------

class PipelineMockBackend:
    """Mock backend that returns plausible JSON for all ribosome operations."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        # Detect which operation by checking the system prompt
        if "compression engine" in system:
            # Pack
            return json.dumps({
                "codons": [
                    {"meaning": "mock_concept", "weight": 0.9, "is_exon": True},
                    {"meaning": "mock_detail", "weight": 0.5, "is_exon": True},
                ],
                "complement": "Mock compressed summary of the content.",
                "promoter": {
                    "domains": ["testing", "mock"],
                    "entities": ["MockEntity"],
                    "intent": "test content",
                    "summary": "A mock gene for pipeline testing",
                },
            })
        elif "expression scorer" in system:
            # Re-rank: score all genes mentioned
            import re
            gene_ids = re.findall(r"(\w{16}):", prompt)
            scores = {gid: round(0.9 - i * 0.1, 1) for i, gid in enumerate(gene_ids)}
            return json.dumps(scores)
        elif "context splicer" in system:
            # Splice: keep first 2 codons for each gene
            import re
            gene_ids = re.findall(r"Gene (\w+)", prompt)
            result = {gid: [0, 1] for gid in gene_ids}
            return json.dumps(result)
        elif "replication engine" in system:
            # Replicate
            return json.dumps({
                "codons": [{"meaning": "exchange", "weight": 1.0, "is_exon": True}],
                "complement": "Mock replicated exchange.",
                "promoter": {
                    "domains": ["exchange"],
                    "entities": [],
                    "intent": "conversation",
                    "summary": "Mock exchange",
                },
            })
        return "{}"


@pytest.fixture
def pipeline_config():
    return HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=4, splice_aggressiveness=0.5),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        synonym_map={
            "slow": ["performance", "latency"],
            "auth": ["jwt", "login", "security"],
        },
    )


@pytest.fixture
def helix(pipeline_config):
    """HelixContextManager with mock backend and in-memory genome."""
    mgr = HelixContextManager(pipeline_config)
    # Replace the ribosome backend with our mock
    mgr.ribosome.backend = PipelineMockBackend()
    yield mgr
    mgr.close()


@pytest.fixture
def seeded_helix(helix):
    """Helix with pre-loaded genes (bypassing ribosome for speed)."""
    genes = [
        make_gene("Authentication middleware with JWT validation",
                  domains=["auth", "security"], entities=["jwt"],
                  gene_id="auth_gene_00001"),
        make_gene("Database connection pooling and query optimization",
                  domains=["database", "performance"], entities=["postgres"],
                  gene_id="db_gene_000001"),
        make_gene("React component lifecycle and state management",
                  domains=["frontend", "react"], entities=["useState"],
                  gene_id="react_gene_0001"),
        make_gene("Kubernetes deployment configuration and scaling",
                  domains=["devops", "kubernetes"], entities=["helm"],
                  gene_id="k8s_gene_00001"),
        make_gene("REST API rate limiting and throttling patterns",
                  domains=["api", "performance", "security"], entities=["redis"],
                  gene_id="api_gene_00001"),
    ]
    for g in genes:
        helix.genome.upsert_gene(g)
    return helix


# -- Pipeline tests ----------------------------------------------------


class TestBuildContext:
    def test_empty_genome_returns_empty_window(self, helix):
        window = helix.build_context("anything")
        assert "no relevant context" in window.expressed_context.lower()
        assert window.total_estimated_tokens > 0  # decoder prompt still counts

    def test_matching_query_returns_context(self, seeded_helix):
        window = seeded_helix.build_context("How does JWT auth work?")
        assert window.metadata.get("genes_expressed", 0) >= 1
        assert len(window.expressed_gene_ids) >= 1
        assert window.compression_ratio > 0

    def test_synonym_expansion_in_pipeline(self, seeded_helix):
        """'slow' should expand to 'performance'/'latency' and match db gene."""
        window = seeded_helix.build_context("Why is the database slow?")
        assert window.metadata.get("genes_expressed", 0) >= 1

    def test_decoder_prompt_always_present(self, seeded_helix):
        window = seeded_helix.build_context("anything about auth")
        assert "expressed_context" in window.ribosome_prompt.lower() or \
               "codon" in window.ribosome_prompt.lower()

    def test_expressed_context_wrapped_in_tags(self, seeded_helix):
        window = seeded_helix.build_context("auth")
        assert "<expressed_context>" in window.expressed_context
        assert "</expressed_context>" in window.expressed_context

    def test_multiple_genes_joined_with_dividers(self, seeded_helix):
        """Query matching multiple genes should join with --- dividers."""
        # 'security' matches both auth and api genes
        window = seeded_helix.build_context("security performance")
        if window.metadata.get("genes_expressed", 0) > 1:
            assert "---" in window.expressed_context


class TestIngest:
    def test_ingest_creates_genes(self, helix):
        gene_ids = helix.ingest("This is a test document about authentication.", content_type="text")
        assert len(gene_ids) >= 1
        stats = helix.stats()
        assert stats["total_genes"] >= 1

    def test_ingest_code(self, helix):
        code = "def hello():\n    return 'world'\n\ndef goodbye():\n    return 'farewell'"
        gene_ids = helix.ingest(code, content_type="code")
        assert len(gene_ids) >= 1


class TestLearn:
    def test_learn_stores_gene(self, helix):
        gid = helix.learn("Why is auth slow?", "The JWT validation is hitting the DB on every request.")
        assert gid is not None
        gene = helix.genome.get_gene(gid)
        assert gene is not None

    def test_pending_buffer_accessible(self, helix):
        """After learn(), the gene should be in the pending buffer momentarily."""
        # Since learn() commits synchronously in our impl, pending is cleared.
        # But let's verify the flow doesn't crash.
        gid = helix.learn("test query", "test response")
        assert gid is not None
        # Pending should be empty after commit
        assert len(helix._pending) == 0


class TestStats:
    def test_stats_include_config(self, seeded_helix):
        stats = seeded_helix.stats()
        assert "config" in stats
        assert stats["config"]["max_genes_per_turn"] == 4
        assert stats["total_genes"] == 5

    def test_stats_include_pending(self, helix):
        stats = helix.stats()
        assert "pending_replications" in stats


# -- Message munging tests (Fix 3: cold-start bootstrap) ---------------


class TestExtractQuerySignals:
    """Test the heuristic keyword extractor directly."""

    def test_stop_words_removed(self, helix):
        domains, entities = helix._extract_query_signals("What is the best way to do this?")
        assert "what" not in domains
        assert "the" not in domains
        assert "best" in domains

    def test_entities_are_longer_words(self, helix):
        domains, entities = helix._extract_query_signals("How does AlphaFold predict protein structure?")
        assert "alphafold" in entities
        assert "predict" in entities
        assert "protein" in entities

    def test_short_query(self, helix):
        domains, entities = helix._extract_query_signals("auth")
        assert "auth" in domains

    def test_empty_query(self, helix):
        domains, entities = helix._extract_query_signals("")
        assert domains == []
        assert entities == []

    def test_punctuation_stripped(self, helix):
        domains, entities = helix._extract_query_signals("What about caching? And redis!")
        assert "caching" in domains
        assert "redis" in domains


class TestMessageMunging:
    def test_mature_genome_strips_history(self):
        """With enough genes, only system + current turn remain."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ]

        result = _munge_messages(
            messages=messages,
            expressed_context="<expressed_context>test</expressed_context>",
            ribosome_prompt="decoder prompt",
            total_genes=100,  # Mature genome
            cold_start_threshold=10,
        )

        # Should have: system (with context injected) + current user turn
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "new question"

    def test_cold_start_keeps_history(self):
        """With few genes, keep last 2 turns for continuity."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ]

        result = _munge_messages(
            messages=messages,
            expressed_context="<expressed_context>test</expressed_context>",
            ribosome_prompt="decoder prompt",
            total_genes=3,  # Cold start
            cold_start_threshold=10,
        )

        # Should have: system + 2 history turns + current
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "follow up"

    def test_context_appended_not_overwritten(self):
        """User's custom system prompt must be preserved."""
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "ahoy"},
        ]

        result = _munge_messages(
            messages=messages,
            expressed_context="<expressed_context>gold</expressed_context>",
            ribosome_prompt="decoder",
            total_genes=100,
            cold_start_threshold=10,
        )

        system_content = result[0]["content"]
        assert "You are a pirate." in system_content
        assert "gold" in system_content

    def test_no_system_message_creates_one(self):
        """If client sends no system message, we create one for context."""
        messages = [
            {"role": "user", "content": "hello"},
        ]

        result = _munge_messages(
            messages=messages,
            expressed_context="<expressed_context>ctx</expressed_context>",
            ribosome_prompt="decoder",
            total_genes=100,
            cold_start_threshold=10,
        )

        assert result[0]["role"] == "system"
        assert "ctx" in result[0]["content"]

    def test_empty_messages_returns_empty(self):
        result = _munge_messages([], "ctx", "dec", 100, 10)
        assert result == []

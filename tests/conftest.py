"""Shared fixtures for Helix Context tests."""

import pytest
from pathlib import Path

from helix_context.genome import Genome
from helix_context.schemas import Gene, PromoterTags, EpigeneticMarkers, ChromatinState


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def poem_text():
    return (FIXTURES_DIR / "poem.txt").read_text(encoding="utf-8")


@pytest.fixture
def calculator_code():
    return (FIXTURES_DIR / "calculator.py").read_text(encoding="utf-8")


@pytest.fixture
def genome():
    """In-memory genome for fast, stateless tests.

    The density gate is disabled by default at the fixture level so that
    existing query-logic / retrieval / co-activation / HGT tests can
    insert hand-crafted test genes without fighting the ingest-time
    demotion heuristic. Tests that specifically want to exercise the
    gate should either use the ``gated_genome`` fixture below or call
    ``genome.upsert_gene(gene, apply_gate=True)`` explicitly.
    """
    g = Genome(
        path=":memory:",
        synonym_map={
            "slow": ["performance", "latency", "bottleneck"],
            "auth": ["jwt", "login", "security", "token"],
            "db": ["database", "sqlite", "sql", "query"],
        },
    )
    # Monkey-patch upsert_gene so the default is gate-off for tests.
    # Tests that want the gate on can still pass apply_gate=True.
    _original_upsert = g.upsert_gene
    def _ungated_upsert(gene, apply_gate=False):
        return _original_upsert(gene, apply_gate=apply_gate)
    g.upsert_gene = _ungated_upsert

    yield g
    g.close()


@pytest.fixture
def gated_genome():
    """In-memory genome with the density gate enabled by default.

    Use this for tests that specifically verify gate behavior at the
    upsert boundary — the gate runs on every upsert_gene call unless
    the test passes apply_gate=False explicitly.
    """
    g = Genome(
        path=":memory:",
        synonym_map={
            "slow": ["performance", "latency"],
            "auth": ["jwt", "login", "security"],
        },
    )
    yield g
    g.close()


def make_gene(
    content: str = "test content",
    domains: list[str] | None = None,
    entities: list[str] | None = None,
    co_activated_with: list[str] | None = None,
    chromatin: ChromatinState = ChromatinState.OPEN,
    is_fragment: bool = False,
    gene_id: str | None = None,
) -> Gene:
    """Helper to build Gene objects for tests without needing the ribosome."""
    gid = gene_id or Genome.make_gene_id(content)
    return Gene(
        gene_id=gid,
        content=content,
        complement=f"Summary of: {content[:50]}",
        codons=["chunk_0", "chunk_1", "chunk_2"],
        promoter=PromoterTags(
            domains=domains or [],
            entities=entities or [],
            intent="test",
            summary=content[:80],
        ),
        epigenetics=EpigeneticMarkers(
            co_activated_with=co_activated_with or [],
        ),
        chromatin=chromatin,
        is_fragment=is_fragment,
    )

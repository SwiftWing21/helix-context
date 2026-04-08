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
    """In-memory genome for fast, stateless tests."""
    g = Genome(
        path=":memory:",
        synonym_map={
            "slow": ["performance", "latency", "bottleneck"],
            "auth": ["jwt", "login", "security", "token"],
            "db": ["database", "sqlite", "sql", "query"],
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

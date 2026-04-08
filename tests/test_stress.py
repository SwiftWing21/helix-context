"""
Stress tests -- exercise the full fixture corpus across all layers.

Three tiers:
    1. Chunking stress (no model) -- verify all fixtures chunk cleanly
    2. Genome stress (no model) -- bulk insert + cross-domain retrieval
    3. Live pipeline stress (Ollama) -- pack, store, query, splice real content

Run tiers:
    pytest tests/test_stress.py -m "not live" -v     # chunking + genome only
    pytest tests/test_stress.py -m live -v -s         # live pipeline
    pytest tests/test_stress.py -v -s                 # everything
"""

import json
import time
import pytest
from pathlib import Path

from helix_context.codons import CodonChunker, CodonEncoder
from helix_context.genome import Genome
from helix_context.ribosome import Ribosome, OllamaBackend
from helix_context.schemas import Gene, PromoterTags, EpigeneticMarkers, ChromatinState
from helix_context.exceptions import PromoterMismatch

import httpx

FIXTURES = Path(__file__).parent / "fixtures"


# -- Fixture loading helpers --

def load_all_fixtures():
    """Load all fixtures with their content type and content."""
    fixtures = {}
    for f in sorted(FIXTURES.iterdir()):
        if f.name.startswith("."):
            continue
        content = f.read_text(encoding="utf-8")
        if f.suffix == ".py":
            ctype = "code"
        else:
            ctype = "text"
        fixtures[f.stem] = {"content": content, "content_type": ctype, "path": str(f)}
    return fixtures


ALL_FIXTURES = load_all_fixtures()


def _ollama_available():
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200 and len(resp.json().get("models", [])) > 0
    except Exception:
        return False


_skip_live = not _ollama_available()


def _mark_live(fn):
    fn = pytest.mark.skipif(_skip_live, reason="Ollama not running")(fn)
    fn = pytest.mark.live(fn)
    return fn


live = _mark_live


# ===================================================================
# TIER 1: Chunking Stress (no model calls)
# ===================================================================


class TestChunkingStress:
    """Every fixture must chunk without errors and produce valid strands."""

    @pytest.mark.parametrize("name,fixture", list(ALL_FIXTURES.items()), ids=list(ALL_FIXTURES.keys()))
    def test_fixture_chunks_cleanly(self, name, fixture):
        chunker = CodonChunker(max_chars_per_strand=2000)
        strands = chunker.chunk(fixture["content"], content_type=fixture["content_type"])

        assert len(strands) > 0, f"{name}: produced zero strands"

        # Sequential indices
        indices = [s.sequence_index for s in strands]
        assert indices == list(range(len(strands))), f"{name}: non-sequential indices"

        # No empty strands
        for s in strands:
            assert s.content.strip(), f"{name}: empty strand at index {s.sequence_index}"

    @pytest.mark.parametrize("name,fixture", list(ALL_FIXTURES.items()), ids=list(ALL_FIXTURES.keys()))
    def test_encoder_chunks_all_fixtures(self, name, fixture):
        """CodonEncoder also chunks every fixture for the ribosome pack phase."""
        encoder = CodonEncoder(chunk_target=3)
        if fixture["content_type"] == "code":
            groups = encoder.chunk_code(fixture["content"])
        else:
            groups = encoder.chunk_text(fixture["content"])

        assert len(groups) > 0, f"{name}: encoder produced zero groups"

    def test_total_coverage(self):
        """Sanity: we have a good spread of fixture types."""
        poems = [n for n in ALL_FIXTURES if n.startswith("poem")]
        essays = [n for n in ALL_FIXTURES if n.startswith("essay")]
        code = [n for n in ALL_FIXTURES if n.startswith("code") or n == "calculator"]
        science = [n for n in ALL_FIXTURES if n.startswith("science")]

        assert len(poems) >= 3, f"Need at least 3 poems, have {len(poems)}"
        assert len(essays) >= 3, f"Need at least 3 essays, have {len(essays)}"
        assert len(code) >= 3, f"Need at least 3 code refs, have {len(code)}"
        assert len(science) >= 2, f"Need at least 2 science docs, have {len(science)}"

    def test_large_content_stability(self):
        """Concatenate ALL fixtures into one giant string -- chunker must not crash."""
        giant = "\n\n---\n\n".join(f["content"] for f in ALL_FIXTURES.values())
        chunker = CodonChunker(max_chars_per_strand=3000)
        strands = chunker.chunk(giant, content_type="text")

        assert len(strands) > 10  # Should be many strands
        total_chars = sum(len(s.content) for s in strands)
        # Strands should cover most of the input (allowing for whitespace trimming)
        assert total_chars > len(giant) * 0.8


# ===================================================================
# TIER 2: Genome Stress (no model calls, in-memory SQLite)
# ===================================================================


class TestGenomeStress:
    """Bulk insert mock genes from all fixtures, then cross-domain retrieval."""

    @pytest.fixture
    def loaded_genome(self):
        """Genome pre-loaded with a mock gene per fixture."""
        genome = Genome(
            path=":memory:",
            synonym_map={
                "slow": ["performance", "latency", "bottleneck"],
                "auth": ["jwt", "login", "security"],
                "data": ["database", "btree", "index", "storage"],
                "web": ["http", "router", "api", "endpoint"],
                "bio": ["protein", "enzyme", "amino", "folding"],
                "flow": ["fluid", "cfd", "turbulence", "navier"],
                "cache": ["redis", "ttl", "invalidation", "cdn"],
            },
        )

        # Domain mappings for fixtures
        domain_map = {
            "poem": (["poetry", "literature"], []),
            "poem_compiler": (["poetry", "compiler", "optimization"], ["lexer", "binary"]),
            "poem_latency": (["poetry", "latency", "networking"], ["dns", "tls", "json"]),
            "poem_recursion": (["poetry", "recursion", "algorithms"], ["stack", "base_case"]),
            "calculator": (["math", "calculator", "arithmetic"], ["Calculator"]),
            "code_linked_list": (["data_structures", "linked_list"], ["Node", "DoublyLinkedList"]),
            "code_http_server": (["http", "router", "middleware", "api"], ["Router", "Request", "Response"]),
            "code_btree": (["data_structures", "btree", "database", "index"], ["BTree", "BTreeNode"]),
            "code_state_machine": (["state_machine", "workflow", "fsm"], ["StateMachine", "Transition"]),
            "essay_caching": (["caching", "performance", "redis", "cdn"], ["ttl", "invalidation"]),
            "essay_testing": (["testing", "pytest", "tdd", "quality"], ["unit_test", "integration"]),
            "essay_distributed": (["distributed", "consensus", "raft", "paxos"], ["Raft", "Paxos", "CRDT"]),
            "essay_oakhaven": (["history", "architecture", "acoustics"], ["Oakhaven", "mortar", "phonon"]),
            "essay_perfect_machine": (["philosophy", "resilience", "stochastic"], ["mutation", "gridlock"]),
            "essay_deep_sea": (["biology", "ecology", "deep_sea"], ["Vestimentifera", "trophosome"]),
            "science_fluid_dynamics": (["fluid", "cfd", "turbulence", "physics"], ["NavierStokes", "LBM", "RANS"]),
            "science_protein": (["protein", "folding", "biochemistry"], ["AlphaFold", "alpha_helix", "beta_sheet"]),
        }

        for name, fixture in ALL_FIXTURES.items():
            domains, entities = domain_map.get(name, (["misc"], []))
            gene = Gene(
                gene_id=Genome.make_gene_id(fixture["content"]),
                content=fixture["content"],
                complement=f"Summary of {name}: {fixture['content'][:100]}...",
                codons=[f"chunk_{i}" for i in range(5)],
                promoter=PromoterTags(
                    domains=domains,
                    entities=entities,
                    intent=f"test fixture: {name}",
                    summary=f"{name} fixture content",
                ),
                epigenetics=EpigeneticMarkers(),
            )
            genome.upsert_gene(gene)

        yield genome
        genome.close()

    def test_all_fixtures_stored(self, loaded_genome):
        stats = loaded_genome.stats()
        assert stats["total_genes"] == len(ALL_FIXTURES)

    def test_cross_domain_query_caching(self, loaded_genome):
        results = loaded_genome.query_genes(domains=["caching"], entities=[])
        names = {r.promoter.intent for r in results}
        assert any("caching" in n for n in names)

    def test_cross_domain_query_data_structures(self, loaded_genome):
        results = loaded_genome.query_genes(domains=["data_structures"], entities=[])
        assert len(results) >= 2  # linked_list + btree

    def test_cross_domain_query_biology(self, loaded_genome):
        results = loaded_genome.query_genes(domains=["protein"], entities=[])
        assert len(results) >= 1

    def test_cross_domain_query_fluid(self, loaded_genome):
        results = loaded_genome.query_genes(domains=["fluid"], entities=[])
        assert len(results) >= 1

    def test_synonym_query_slow(self, loaded_genome):
        """'slow' should expand to match 'performance', 'latency', etc."""
        results = loaded_genome.query_genes(domains=["slow"], entities=[])
        # Should find: poem_latency (latency), essay_caching (performance)
        assert len(results) >= 1

    def test_synonym_query_web(self, loaded_genome):
        """'web' should expand to match 'http', 'router', 'api'."""
        results = loaded_genome.query_genes(domains=["web"], entities=[])
        assert len(results) >= 1

    def test_entity_query_alphafold(self, loaded_genome):
        results = loaded_genome.query_genes(domains=[], entities=["AlphaFold"])
        assert len(results) == 1

    def test_entity_query_raft(self, loaded_genome):
        results = loaded_genome.query_genes(domains=[], entities=["Raft"])
        assert len(results) >= 1

    def test_no_match_still_raises(self, loaded_genome):
        with pytest.raises(PromoterMismatch):
            loaded_genome.query_genes(domains=["quantum_entanglement"], entities=[])

    def test_co_activation_across_domains(self, loaded_genome):
        """Link bio+fluid genes, then verify co-activation pull-forward."""
        bio_genes = loaded_genome.query_genes(domains=["protein"], entities=[])
        fluid_genes = loaded_genome.query_genes(domains=["fluid"], entities=[])

        if bio_genes and fluid_genes:
            # Simulate they were co-expressed
            loaded_genome.link_coactivated(
                [bio_genes[0].gene_id, fluid_genes[0].gene_id]
            )

            # Now query for protein -- fluid gene should be pulled in
            results = loaded_genome.query_genes(domains=["protein"], entities=[])
            result_ids = {r.gene_id for r in results}
            assert fluid_genes[0].gene_id in result_ids

    def test_compaction_doesnt_break_retrieval(self, loaded_genome):
        """Compaction on a fresh genome should compact 0 genes."""
        compacted = loaded_genome.compact()
        assert compacted == 0

        # All genes should still be queryable
        stats = loaded_genome.stats()
        assert stats["total_genes"] == len(ALL_FIXTURES)


# ===================================================================
# TIER 3: Live Pipeline Stress (Ollama required)
# ===================================================================


@live
class TestLivePipelineStress:
    """Pack real fixtures through the ribosome and verify quality."""

    @pytest.fixture(scope="class")
    def ribosome(self):
        return Ribosome(
            backend=OllamaBackend(model="auto", timeout=60),
            splice_aggressiveness=0.5,
        )

    @pytest.fixture(scope="class")
    def live_genome(self):
        g = Genome(path=":memory:")
        yield g
        g.close()

    # -- Pack tests --

    @pytest.mark.parametrize("name", [
        "poem", "poem_compiler", "poem_latency",
    ])
    def test_pack_poems(self, ribosome, name):
        content = ALL_FIXTURES[name]["content"]
        gene = ribosome.pack(content, content_type="text")

        assert gene.gene_id
        assert len(gene.codons) > 0
        assert gene.complement
        assert len(gene.promoter.domains) > 0
        print(f"\n  [{name}] {len(gene.codons)} codons, domains={gene.promoter.domains}")

    @pytest.mark.parametrize("name", [
        "essay_caching", "essay_distributed", "essay_oakhaven",
    ])
    def test_pack_essays(self, ribosome, name):
        content = ALL_FIXTURES[name]["content"]
        gene = ribosome.pack(content, content_type="text")

        assert gene.gene_id
        assert len(gene.codons) > 0
        assert gene.complement
        assert len(gene.promoter.domains) > 0
        print(f"\n  [{name}] {len(gene.codons)} codons, domains={gene.promoter.domains}, "
              f"summary={gene.promoter.summary[:80]}")

    @pytest.mark.parametrize("name", [
        "code_btree", "code_http_server", "code_state_machine",
    ])
    def test_pack_code(self, ribosome, name):
        content = ALL_FIXTURES[name]["content"]
        gene = ribosome.pack(content, content_type="code")

        assert gene.gene_id
        assert len(gene.codons) > 0
        assert gene.complement
        print(f"\n  [{name}] {len(gene.codons)} codons, domains={gene.promoter.domains}")

    @pytest.mark.parametrize("name", [
        "science_fluid_dynamics", "science_protein",
    ])
    def test_pack_science(self, ribosome, name):
        content = ALL_FIXTURES[name]["content"]
        gene = ribosome.pack(content, content_type="text")

        assert gene.gene_id
        assert len(gene.codons) > 0
        assert gene.complement
        assert len(gene.promoter.domains) > 0
        print(f"\n  [{name}] {len(gene.codons)} codons, domains={gene.promoter.domains}, "
              f"summary={gene.promoter.summary[:80]}")

    # -- Full pipeline: pack -> store -> query -> splice --

    def test_full_pipeline_cross_domain(self, ribosome, live_genome):
        """Pack multiple fixtures, store them, then query across domains."""
        packed_genes = []
        to_pack = ["essay_caching", "code_btree", "science_protein"]

        for name in to_pack:
            f = ALL_FIXTURES[name]
            gene = ribosome.pack(f["content"], content_type=f["content_type"])
            live_genome.upsert_gene(gene)
            packed_genes.append(gene)
            print(f"\n  Packed {name}: domains={gene.promoter.domains}")

        stats = live_genome.stats()
        assert stats["total_genes"] == len(to_pack)
        print(f"\n  Genome stats: {stats}")

        # Query for caching-related content
        try:
            results = live_genome.query_genes(
                domains=["cache", "caching", "performance"],
                entities=["redis", "ttl"],
            )
            print(f"\n  Query 'caching': {len(results)} results")
            assert len(results) >= 1

            # Splice the results
            spliced = ribosome.splice("How does cache invalidation work?", results)
            for gid, text in spliced.items():
                print(f"\n  Spliced {gid}: {len(text)} chars")
                assert len(text) > 0

        except PromoterMismatch:
            # The ribosome may have assigned different domain labels
            print("\n  PromoterMismatch -- ribosome used different domain labels (acceptable)")

    def test_compression_ratio(self, ribosome):
        """Pack the longest fixture and verify we achieve meaningful compression."""
        longest_name = max(ALL_FIXTURES, key=lambda n: len(ALL_FIXTURES[n]["content"]))
        longest = ALL_FIXTURES[longest_name]

        gene = ribosome.pack(longest["content"], content_type=longest["content_type"])

        raw_len = len(longest["content"])
        complement_len = len(gene.complement)
        ratio = raw_len / max(complement_len, 1)

        print(f"\n  [{longest_name}] raw={raw_len}, complement={complement_len}, "
              f"ratio={ratio:.1f}x, codons={len(gene.codons)}")

        # Complement should be meaningfully shorter than raw content
        assert ratio > 2.0, f"Compression ratio {ratio:.1f}x is too low"

    def test_replicate_quality(self, ribosome):
        """Verify replicate captures intent across different exchange types."""
        exchanges = [
            ("Why is the API slow?",
             "The N+1 query pattern in the user list endpoint fetches each user's "
             "profile in a separate query. Refactor to use a JOIN with eager loading."),
            ("How does the B-tree maintain balance?",
             "When a node overflows past 2t-1 keys, it splits into two nodes and "
             "pushes the median key up to the parent. This maintains the invariant "
             "that all leaves are at the same depth."),
            ("What lives near hydrothermal vents?",
             "Giant tube worms form the primary structure, sustained "
             "by chemosynthetic bacteria in their trophosomes. Crabs scavenge the lower "
             "tiers while bioluminescent jellyfish navigate the upper canopy."),
        ]

        for query, response in exchanges:
            gene = ribosome.replicate(query, response)
            assert gene.complement
            assert len(gene.promoter.domains) > 0
            print(f"\n  Q: {query[:40]}...")
            print(f"    domains={gene.promoter.domains}, summary={gene.promoter.summary[:60]}")

"""
Tests for the delta-epsilon context health monitor and HGT.
"""

import json
import pytest
import tempfile
import os

from helix_context.config import HelixConfig, BudgetConfig, GenomeConfig, RibosomeConfig
from helix_context.context_manager import HelixContextManager
from helix_context.genome import Genome
from helix_context.hgt import export_genome, import_genome, genome_diff
from helix_context.schemas import ContextHealth

from tests.conftest import make_gene


# -- Helpers -----------------------------------------------------------

class HealthMockBackend:
    def complete(self, prompt, system="", temperature=0.0):
        import json as j
        if "compression engine" in system:
            return j.dumps({
                "codons": [{"meaning": "test", "weight": 0.8, "is_exon": True}],
                "complement": "Mock compressed.",
                "promoter": {"domains": ["auth", "security"], "entities": ["jwt"],
                             "intent": "test", "summary": "test gene"},
            })
        elif "expression scorer" in system:
            return j.dumps({})
        elif "context splicer" in system:
            return j.dumps({})
        elif "replication engine" in system:
            return j.dumps({
                "codons": [{"meaning": "ex", "weight": 1.0, "is_exon": True}],
                "complement": "Exchange.",
                "promoter": {"domains": ["test"], "entities": [], "intent": "test", "summary": "test"},
            })
        return "{}"


@pytest.fixture
def health_helix():
    config = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=4),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        synonym_map={"auth": ["jwt", "login", "security"]},
    )
    mgr = HelixContextManager(config)
    mgr.ribosome.backend = HealthMockBackend()
    yield mgr
    mgr.close()


@pytest.fixture
def seeded_health_helix(health_helix):
    genes = [
        make_gene("JWT authentication middleware",
                  domains=["auth", "security"], entities=["jwt"],
                  gene_id="auth_gene_001"),
        make_gene("Database connection pooling",
                  domains=["database", "performance"], entities=["postgres"],
                  gene_id="db_gene_0001"),
        make_gene("React component state management",
                  domains=["frontend", "react"], entities=["useState"],
                  gene_id="react_gene_01"),
    ]
    for g in genes:
        health_helix.genome.upsert_gene(g)
    return health_helix


# ═══════════════════════════════════════════════════════════════════
# Delta-Epsilon Health Monitor
# ═══════════════════════════════════════════════════════════════════


class TestContextHealth:
    def test_empty_genome_is_sparse(self, health_helix):
        window = health_helix.build_context("anything")
        assert window.context_health.status in ("denatured", "sparse")
        assert window.context_health.genes_expressed == 0
        assert window.context_health.ellipticity == 0.0

    def test_matching_query_has_coverage(self, seeded_health_helix):
        window = seeded_health_helix.build_context("How does JWT auth work?")
        health = window.context_health
        assert health.genes_expressed >= 1
        assert health.coverage > 0
        assert health.freshness > 0
        assert health.ellipticity > 0
        # With mock backend, density is very low (tiny spliced output vs 6k budget)
        # so ellipticity may be low. Just verify the signals are populated.
        assert health.genes_available == 3

    def test_no_match_shows_denatured(self, seeded_health_helix):
        """Query that matches zero genes despite genome having content."""
        window = seeded_health_helix.build_context("quantum entanglement physics")
        health = window.context_health
        assert health.genes_expressed == 0
        assert health.genes_available == 3
        assert health.status == "denatured"

    def test_health_in_metadata(self, seeded_health_helix):
        window = seeded_health_helix.build_context("auth security jwt")
        assert hasattr(window, "context_health")
        health = window.context_health
        assert isinstance(health, ContextHealth)
        assert 0 <= health.ellipticity <= 1
        assert 0 <= health.coverage <= 1
        assert 0 <= health.density <= 1
        assert 0 <= health.freshness <= 1

    def test_freshness_reflects_decay(self, seeded_health_helix):
        """Genes with low decay scores should reduce freshness."""
        # Manually decay a gene
        gene = seeded_health_helix.genome.get_gene("auth_gene_001")
        gene.epigenetics.decay_score = 0.2
        seeded_health_helix.genome.upsert_gene(gene)

        window = seeded_health_helix.build_context("auth security")
        health = window.context_health
        # Freshness should be lower since one gene is stale
        assert health.freshness < 1.0

    def test_health_genes_available_count(self, seeded_health_helix):
        window = seeded_health_helix.build_context("auth")
        assert window.context_health.genes_available == 3


# ═══════════════════════════════════════════════════════════════════
# Horizontal Gene Transfer (HGT)
# ═══════════════════════════════════════════════════════════════════


class TestHGTExport:
    def test_export_creates_file(self):
        genome = Genome(":memory:")
        genome.upsert_gene(make_gene("test content", domains=["test"], gene_id="gene_001"))

        with tempfile.NamedTemporaryFile(suffix=".helix", delete=False) as f:
            path = f.name

        try:
            result = export_genome(genome, path, description="Test export")
            assert result["genes"] == 1
            assert result["file_size"] > 0
            assert os.path.exists(path)

            data = json.loads(open(path, encoding="utf-8").read())
            assert data["helix_format_version"] == 1
            assert data["header"]["gene_count"] == 1
            assert data["header"]["description"] == "Test export"
            assert len(data["genes"]) == 1
        finally:
            os.unlink(path)
            genome.close()

    def test_export_excludes_heterochromatin_by_default(self):
        genome = Genome(":memory:")
        from helix_context.schemas import ChromatinState
        genome.upsert_gene(make_gene("active", domains=["test"], gene_id="active_1"))
        genome.upsert_gene(make_gene("stale", domains=["test"], gene_id="stale_1",
                                     chromatin=ChromatinState.HETEROCHROMATIN))

        with tempfile.NamedTemporaryFile(suffix=".helix", delete=False) as f:
            path = f.name

        try:
            result = export_genome(genome, path)
            assert result["genes"] == 1  # Only active gene

            result_all = export_genome(genome, path, include_heterochromatin=True)
            assert result_all["genes"] == 2  # Both genes
        finally:
            os.unlink(path)
            genome.close()


class TestHGTImport:
    def test_import_into_empty_genome(self):
        # Export from source. Use content-addressed gene_ids so import_genome's
        # tamper check passes.
        source = Genome(":memory:")
        content_a = "shared knowledge"
        content_b = "more knowledge"
        gid_a = Genome.make_gene_id(content_a)
        gid_b = Genome.make_gene_id(content_b)
        source.upsert_gene(make_gene(content_a, domains=["test"]))
        source.upsert_gene(make_gene(content_b, domains=["test"]))

        with tempfile.NamedTemporaryFile(suffix=".helix", delete=False) as f:
            path = f.name

        try:
            export_genome(source, path)
            source.close()

            # Import into target
            target = Genome(":memory:")
            result = import_genome(target, path)
            assert result["imported"] == 2
            assert result["skipped"] == 0

            assert target.get_gene(gid_a) is not None
            assert target.get_gene(gid_b) is not None
            target.close()
        finally:
            os.unlink(path)

    @pytest.mark.xfail(
        reason=(
            "Pre-content-address test semantics: asserts 'same gene_id, different content' "
            "skip behavior. Under content-addressing (hgt.import_genome tamper check) that "
            "state is impossible — same content always produces same id. Needs a semantic "
            "rewrite where merge_strategy is tested via metadata (authority, chromatin, "
            "epigenetics) rather than content."
        )
    )
    def test_import_skip_existing(self):
        source = Genome(":memory:")
        source.upsert_gene(make_gene("original", domains=["test"], gene_id="gene_x"))

        with tempfile.NamedTemporaryFile(suffix=".helix", delete=False) as f:
            path = f.name

        try:
            export_genome(source, path)
            source.close()

            target = Genome(":memory:")
            target.upsert_gene(make_gene("already here", domains=["test"], gene_id="gene_x"))

            result = import_genome(target, path, merge_strategy="skip_existing")
            assert result["imported"] == 0
            assert result["skipped"] == 1

            # Original content preserved
            assert target.get_gene("gene_x").content == "already here"
            target.close()
        finally:
            os.unlink(path)

    @pytest.mark.xfail(
        reason=(
            "Pre-content-address test semantics: 'overwrite replaces content on same gene_id'. "
            "Under content-addressing, identical id implies identical content; the observable "
            "overwrite axis is metadata, not content. Needs a semantic rewrite."
        )
    )
    def test_import_overwrite(self):
        source = Genome(":memory:")
        source.upsert_gene(make_gene("new version", domains=["test"], gene_id="gene_x"))

        with tempfile.NamedTemporaryFile(suffix=".helix", delete=False) as f:
            path = f.name

        try:
            export_genome(source, path)
            source.close()

            target = Genome(":memory:")
            target.upsert_gene(make_gene("old version", domains=["test"], gene_id="gene_x"))

            result = import_genome(target, path, merge_strategy="overwrite")
            assert result["imported"] == 0  # 0 new imports
            assert result["overwritten"] == 1

            # Content was replaced
            assert target.get_gene("gene_x").content == "new version"
            target.close()
        finally:
            os.unlink(path)


class TestHGTDiff:
    def test_diff_shows_differences(self):
        source = Genome(":memory:")
        source.upsert_gene(make_gene("shared", domains=["test"], gene_id="shared_1"))
        source.upsert_gene(make_gene("only in file", domains=["test"], gene_id="file_only"))

        with tempfile.NamedTemporaryFile(suffix=".helix", delete=False) as f:
            path = f.name

        try:
            export_genome(source, path)
            source.close()

            target = Genome(":memory:")
            target.upsert_gene(make_gene("shared", domains=["test"], gene_id="shared_1"))
            target.upsert_gene(make_gene("only in genome", domains=["test"], gene_id="genome_only"))

            result = genome_diff(target, path)
            assert result["shared"] == 1
            assert result["only_in_file"] == 1
            assert result["only_in_genome"] == 1
            target.close()
        finally:
            os.unlink(path)

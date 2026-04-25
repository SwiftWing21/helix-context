"""
Tests for CpuTagger filename-derived domain extraction.

Root cause diagnosed 2026-04-24: claims.py had promoter tags
[gene, config, agent, build, genome] but not 'claims', because
_extract_domains() only scans file *content* against _TECH_TERMS
and never consults the filename.  The tagger fix injects filename
stem tokens into domains at pack() time.
"""

from __future__ import annotations

import pytest

from helix_context.tagger import CpuTagger

_TAGGER = None


def _tagger() -> CpuTagger:
    global _TAGGER
    if _TAGGER is None:
        _TAGGER = CpuTagger()
    return _TAGGER


# Minimal content that produces NO tech-term domains on its own,
# so any domains we observe come purely from the filename path.
_BLANK_CONTENT = "pass"


class TestFilenameDomainsInjected:
    """Filename stem tokens must appear in gene.promoter.domains."""

    def test_simple_stem_added_to_domains(self):
        """claims.py → 'claims' must appear in domains."""
        gene = _tagger().pack(
            _BLANK_CONTENT,
            content_type="code",
            source_id="/repo/helix_context/claims.py",
        )
        assert "claims" in gene.promoter.domains, (
            f"Expected 'claims' in domains; got {gene.promoter.domains}"
        )

    def test_compound_stem_tokenized_into_parts(self):
        """claim_types_handler.py → 'claim', 'types', 'handler' all in domains."""
        gene = _tagger().pack(
            _BLANK_CONTENT,
            content_type="code",
            source_id="/repo/helix_context/claim_types_handler.py",
        )
        domains = gene.promoter.domains
        assert "claim" in domains, f"'claim' missing from {domains}"
        assert "types" in domains, f"'types' missing from {domains}"
        assert "handler" in domains, f"'handler' missing from {domains}"

    def test_full_compound_stem_also_present(self):
        """The full unsplit stem 'claim_types_handler' must also appear."""
        gene = _tagger().pack(
            _BLANK_CONTENT,
            content_type="code",
            source_id="/repo/helix_context/claim_types_handler.py",
        )
        assert "claim_types_handler" in gene.promoter.domains, (
            f"Full stem missing from {gene.promoter.domains}"
        )

    def test_noise_stem_not_injected(self):
        """__init__.py is a noise stem and must NOT pollute domains."""
        gene = _tagger().pack(
            _BLANK_CONTENT,
            content_type="code",
            source_id="/repo/helix_context/__init__.py",
        )
        assert "__init__" not in gene.promoter.domains
        assert "init" not in gene.promoter.domains

    def test_parent_dir_name_injected(self):
        """Direct parent directory name should be added as a domain token."""
        gene = _tagger().pack(
            _BLANK_CONTENT,
            content_type="code",
            source_id="/repo/helix_context/claims.py",
        )
        assert "helix_context" in gene.promoter.domains, (
            f"Parent dir 'helix_context' missing from {gene.promoter.domains}"
        )

    def test_no_source_id_does_not_crash(self):
        """pack() without source_id must still work (no filename tokens)."""
        gene = _tagger().pack(_BLANK_CONTENT, content_type="code")
        assert gene.promoter.domains is not None

    def test_domains_still_contain_content_terms(self):
        """Content-derived tech terms must survive alongside filename tokens."""
        gene = _tagger().pack(
            "import redis\nclient = redis.Redis()",
            content_type="code",
            source_id="/repo/helix_context/claims.py",
        )
        assert "redis" in gene.promoter.domains, (
            f"Content-derived 'redis' missing from {gene.promoter.domains}"
        )
        assert "claims" in gene.promoter.domains, (
            f"Filename 'claims' missing from {gene.promoter.domains}"
        )

    def test_short_tokens_not_injected(self):
        """Tokens <= 2 chars from filename must be filtered out."""
        gene = _tagger().pack(
            _BLANK_CONTENT,
            content_type="code",
            source_id="/repo/a_b_c.py",
        )
        domains = gene.promoter.domains
        for tok in domains:
            assert len(tok) > 2 or "_" in tok, (
                f"Short token '{tok}' leaked into domains from filename"
            )

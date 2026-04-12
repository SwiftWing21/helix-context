"""
Tests for CpuTagger._extract_key_values — the regex-based KV path that
populates gene.key_values during ingest_all.py runs.

Motivated by the 2026-04-11 KV-quality audit: the live genome's key_values
was found to contain Python type annotations (`port=int`, `int=8080`)
instead of the actual literal values those annotations decorate. Root cause
is in tagger._KV_PAIR_PATTERN and _KV_PATTERNS:

  - `[=:]` matches both assignments and Python type hints
  - The value-side skip list filtered type names only as *keys*, not as *values*
  - Named patterns like `model` matched on `model: str` and emitted `model=str`

These tests pin the fix:

  - RED (pre-fix): `port: int = 8080` produces `port=int` or `int=8080`
  - GREEN (post-fix): same input produces `port=8080` and no `int=*` entry
"""

from __future__ import annotations

import pytest

from helix_context.tagger import CpuTagger

# Module-level fixture avoids re-loading spaCy per test (expensive)
_TAGGER = None


def _tagger() -> CpuTagger:
    global _TAGGER
    if _TAGGER is None:
        _TAGGER = CpuTagger()
    return _TAGGER


# ── Type-annotation leak tests ──────────────────────────────────────

class TestTypeAnnotationLeak:
    """The bug: Python type hints slip into key_values as bare type names."""

    def test_annotated_assignment_extracts_value_not_type(self):
        """`port: int = 8080` should emit port=8080, not port=int."""
        content = "port: int = 8080\n"
        kvs = _tagger()._extract_key_values(content)
        assert "port=int" not in kvs, f"type annotation leaked: {kvs}"
        # NOTE: the named port pattern requires `port[=:]\d{2,5}` directly —
        # `port: int = 8080` has `int` between. So the named pattern misses,
        # and we fall through to the generic pair matcher. After the fix,
        # the pair matcher should NOT emit `port=int` either.

    def test_no_reverse_type_value_entry(self):
        """`port: int = 8080` should not produce `int=8080` either.

        When the pair matcher scans `port: int = 8080`, its greedy `finditer`
        sees TWO overlapping matches: `port: int` and `int = 8080`. The
        latter produces a `int=8080` entry where the key is a Python type
        name. This must be skipped.
        """
        content = "port: int = 8080\n"
        kvs = _tagger()._extract_key_values(content)
        assert "int=8080" not in kvs, f"type-name-as-key leaked: {kvs}"

    def test_model_annotation_not_extracted_as_value(self):
        """`model: str = \"qwen3\"` should emit model=qwen3, not model=str."""
        content = 'model: str = "qwen3"\n'
        kvs = _tagger()._extract_key_values(content)
        assert "model=str" not in kvs, f"type annotation leaked: {kvs}"

    def test_multiple_type_annotations_in_dataclass(self):
        content = (
            "@dataclass\n"
            "class Config:\n"
            "    host: str = 'localhost'\n"
            "    port: int = 11437\n"
            "    timeout: float = 30.0\n"
            "    model: str = 'qwen3:8b'\n"
        )
        kvs = _tagger()._extract_key_values(content)
        # No type-name values should appear
        for bad in ("host=str", "port=int", "timeout=float", "model=str"):
            assert bad not in kvs, f"type leak: {bad} in {kvs}"
        # And no type-as-key entries should appear either
        type_key_entries = [kv for kv in kvs if kv.startswith(("str=", "int=", "float=", "bool="))]
        assert not type_key_entries, f"type-as-key entries: {type_key_entries}"


# ── Regression tests: make sure legitimate KVs still work ────────────

class TestLegitimateKVs:
    """The fix must not regress existing extraction on real config content."""

    def test_plain_assignment_still_works(self):
        content = 'port = 11437\nmodel = "qwen3:8b"\n'
        kvs = _tagger()._extract_key_values(content)
        # At least one of the named patterns should fire
        assert any("port" in kv and "11437" in kv for kv in kvs), kvs
        assert any("model" in kv and "qwen3" in kv for kv in kvs), kvs

    def test_yaml_colon_syntax_still_works(self):
        content = "port: 8080\nhost: localhost\ntimeout: 30\n"
        kvs = _tagger()._extract_key_values(content)
        assert any("port" in kv and "8080" in kv for kv in kvs), kvs
        assert any("host" in kv and "localhost" in kv for kv in kvs), kvs

    def test_toml_syntax_still_works(self):
        content = '[server]\nport = 11437\nmodel = "gemma4:e4b"\n'
        kvs = _tagger()._extract_key_values(content)
        assert any("port" in kv and "11437" in kv for kv in kvs), kvs
        assert any("model" in kv and "gemma4" in kv for kv in kvs), kvs

    def test_url_still_extracted(self):
        content = "base_url = http://localhost:11434/api\n"
        kvs = _tagger()._extract_key_values(content)
        assert any("http" in kv for kv in kvs), kvs

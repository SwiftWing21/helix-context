"""Unit tests for bench_needle_1000.py harvest filters (harness v2).

These pin the v2 fixes against raude's three forensic phantom cases from
the 2026-04-10 N=20 Headroom integration benchmark:

    include_heterochromatin=Include   (docstring sentence starter)
    queries_path=os.path.join         (raw Python attribute chain)
    note=Kept / note=This             (generic prose key + TitleCase phantom)

All three were harvested by the v1 logic, produced benchmark "failures" that
were actually correct model answers, and undercounted our retrieval+answer
rates by ~15%.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make benchmarks/ importable without packaging it
BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import bench_needle_1000 as bench  # noqa: E402


# ─── Phantom cases from raude's 2026-04-10 forensic ─────────────────────

def test_include_heterochromatin_docstring_phantom_rejected():
    # v2: "Include" is a plain TitleCase English word → rejected as phantom
    assert bench.is_quality_kv("include_heterochromatin", "Include") is False


def test_queries_path_dotted_chain_rejected():
    # v2: os.path.join is a dotted Python identifier chain, not a literal
    assert bench.is_quality_kv("queries_path", "os.path.join") is False


def test_note_generic_prose_key_rejected():
    # v2: "note" is now in _PROSE_KEYS — rejected regardless of value shape
    assert bench.is_quality_kv("note", "Kept") is False
    assert bench.is_quality_kv("note", "This") is False
    assert bench.is_quality_kv("note", "existing") is False


def test_function_call_shape_rejected():
    # Guard against "compress_text(content)" style expressions
    assert bench.is_quality_kv("transform", "compress_text(content)") is False
    assert bench.is_quality_kv("handler", "callback(foo)") is False


# ─── Positive cases — legitimate KVs must still pass ────────────────────

def test_legit_numeric_values_pass():
    assert bench.is_quality_kv("port", "11437") is True
    assert bench.is_quality_kv("timeout", "30.0") is True
    assert bench.is_quality_kv("max_tokens", "12000") is True


def test_legit_identifier_values_pass():
    # camelCase and with digits — not phantoms
    assert bench.is_quality_kv("model", "qwen3:8b") is True
    assert bench.is_quality_kv("backend", "ollama") is False  # lowercase single word
    # "ollama" gets rejected by the stricter v2 filter — acceptable tradeoff,
    # the user can opt into legacy via BENCH_LEGACY_HARVEST=1.
    assert bench.is_quality_kv("ribosome_model", "gemma4:e4b") is True


def test_legit_path_values_pass():
    assert bench.is_quality_kv("genome_path", "F:/Projects/helix-context/genome.db") is True
    assert bench.is_quality_kv("output_dir", "benchmarks/results") is True


def test_legit_url_values_pass():
    assert bench.is_quality_kv("base_url", "http://localhost:11434") is True


# ─── _looks_like_literal_value unit tests ──────────────────────────────

def test_looks_like_literal_value_positive():
    # Has digits / punctuation / underscores → literal
    assert bench._looks_like_literal_value("11437") is True
    assert bench._looks_like_literal_value("qwen3") is True
    assert bench._looks_like_literal_value("gemma4:e4b") is True
    assert bench._looks_like_literal_value("my_var") is True
    # camelCase identifier
    assert bench._looks_like_literal_value("httpClient") is True
    # Long ALL_CAPS constant
    assert bench._looks_like_literal_value("PENDING") is True
    assert bench._looks_like_literal_value("RUNNING") is True


def test_looks_like_literal_value_rejects_english_words():
    assert bench._looks_like_literal_value("Include") is False
    assert bench._looks_like_literal_value("existing") is False
    assert bench._looks_like_literal_value("Kept") is False
    assert bench._looks_like_literal_value("This") is False
    # Short acronyms are ambiguous but usually prose fragments
    assert bench._looks_like_literal_value("API") is False
    assert bench._looks_like_literal_value("URL") is False


# ─── _word_boundary_match unit tests ───────────────────────────────────

def test_word_boundary_match_alphanumeric_rejects_substring():
    # "api" should NOT match "apis" or "rapid"
    assert bench._word_boundary_match("the apis are here", "api") is False
    assert bench._word_boundary_match("rapid response", "api") is False
    # But should match standalone
    assert bench._word_boundary_match("use the api endpoint", "api") is True


def test_word_boundary_match_numeric_standalone():
    assert bench._word_boundary_match("port 11437 is open", "11437") is True
    assert bench._word_boundary_match("port 114370 is wrong", "11437") is False


def test_word_boundary_match_punctuation_values_substring_ok():
    # Values with punctuation (paths, URLs) fall back to plain substring
    assert bench._word_boundary_match(
        "config.path is /etc/foo.conf always", "/etc/foo.conf"
    ) is True
    assert bench._word_boundary_match(
        "base_url = http://localhost:11434", "http://localhost:11434"
    ) is True


def test_word_boundary_match_empty_inputs():
    assert bench._word_boundary_match("", "foo") is False
    assert bench._word_boundary_match("foo", "") is False


# ─── _value_in_assignment_context unit tests ───────────────────────────

def test_assignment_context_accepts_real_assignment():
    content = 'port = 11437\nribosome_model = "gemma4:e4b"'
    assert bench._value_in_assignment_context(content, "port", "11437") is True
    assert bench._value_in_assignment_context(content, "ribosome_model", "gemma4:e4b") is True


def test_assignment_context_accepts_type_annotated_assignment():
    content = "def f(include_heterochromatin: bool = False) -> None: pass"
    assert bench._value_in_assignment_context(
        content, "include_heterochromatin", "False"
    ) is True


def test_assignment_context_rejects_docstring_phantom():
    # "Include" appears in docstring, NOT in an assignment window from the key
    content = (
        '"""Include heterochromatin genes when exporting.\n\nReturns...\n"""\n'
        "\n\n# ... 500 lines later ...\n\n"
        "def f(include_heterochromatin: bool = False) -> None: pass"
    )
    # The real value is "False", not "Include" — context check should reject
    assert bench._value_in_assignment_context(
        content, "include_heterochromatin", "Include"
    ) is False


def test_assignment_context_rejects_distant_unrelated_match():
    content = "key_a = 'real'\n\n" + ("x\n" * 100) + "\n# somewhere far: phantom"
    assert bench._value_in_assignment_context(content, "key_a", "phantom") is False

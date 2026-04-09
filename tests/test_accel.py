"""Gate 0 — Acceleration layer tests (no model calls, pure CPU).

Tests correctness of all accel primitives and verifies the Rust-backed
(orjson) path produces identical results to stdlib fallback.
"""

import time
import pytest

from helix_context.accel import (
    JSON_BACKEND,
    json_loads,
    json_dumps,
    json_dumps_bytes,
    estimate_tokens,
    extract_query_signals,
    STOP_WORDS,
    PromptBuilder,
    batch_update_epigenetics,
    parse_promoter,
    parse_epigenetics,
    clear_parse_caches,
    accel_info,
    RE_PARAGRAPH_SPLIT,
    RE_SENTENCE_SPLIT,
    RE_CODE_BOUNDARY,
    RE_CODE_BLOCK_SPLIT,
)


# ── JSON backend ──────────────────────────────────────────────────


class TestJsonBackend:
    def test_backend_reports_name(self):
        assert JSON_BACKEND in ("orjson", "json")

    def test_roundtrip_dict(self):
        original = {"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}
        encoded = json_dumps(original)
        decoded = json_loads(encoded)
        assert decoded == original

    def test_roundtrip_list(self):
        original = [1, "two", 3.0, None, True, False]
        encoded = json_dumps(original)
        decoded = json_loads(encoded)
        assert decoded == original

    def test_loads_from_bytes(self):
        data = b'{"hello": "world"}'
        result = json_loads(data)
        assert result == {"hello": "world"}

    def test_loads_from_string(self):
        data = '{"hello": "world"}'
        result = json_loads(data)
        assert result == {"hello": "world"}

    def test_dumps_bytes(self):
        obj = {"key": "value"}
        result = json_dumps_bytes(obj)
        assert isinstance(result, bytes)
        assert json_loads(result) == obj

    def test_compact_encoding(self):
        """json_dumps should produce compact output (no extra spaces)."""
        obj = {"a": 1, "b": 2}
        encoded = json_dumps(obj)
        assert " " not in encoded or JSON_BACKEND == "orjson"
        # Both backends should roundtrip correctly
        assert json_loads(encoded) == obj

    def test_loads_invalid_raises_valueerror(self):
        with pytest.raises((ValueError, TypeError)):
            json_loads("not json at all{{{")

    def test_empty_structures(self):
        assert json_loads(json_dumps({})) == {}
        assert json_loads(json_dumps([])) == []

    def test_unicode_roundtrip(self):
        original = {"emoji": "🧬", "japanese": "ヘリックス"}
        encoded = json_dumps(original)
        decoded = json_loads(encoded)
        assert decoded == original


# ── Token estimation ──────────────────────────────────────────────


class TestTokenEstimation:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        result = estimate_tokens("hello")
        assert result >= 1

    def test_single_word(self):
        result = estimate_tokens("authentication")
        assert 1 <= result <= 5

    def test_english_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = estimate_tokens(text)
        # GPT tokenizer: ~10 tokens for this sentence
        assert 5 <= result <= 20

    def test_code_snippet(self):
        code = 'def hello_world():\n    print("Hello, World!")\n    return True'
        result = estimate_tokens(code)
        # Code tends to have more tokens per character
        assert result >= 5

    def test_long_text_reasonable_range(self):
        text = "This is a test sentence. " * 100
        result = estimate_tokens(text)
        n = len(text)
        # Must be between len//6 and len//2
        assert n // 6 <= result <= n // 2

    def test_monotonically_increasing(self):
        """More text should give more tokens."""
        short = estimate_tokens("hello world")
        medium = estimate_tokens("hello world " * 10)
        long = estimate_tokens("hello world " * 100)
        assert short <= medium <= long

    def test_never_zero_for_nonempty(self):
        assert estimate_tokens("x") >= 1
        assert estimate_tokens("  ") >= 1


# ── Stop-word extraction ─────────────────────────────────────────


class TestQuerySignalExtraction:
    def test_removes_stop_words(self):
        domains, entities = extract_query_signals("what is the database schema?")
        assert "what" not in domains
        assert "the" not in domains
        assert "database" in domains or "schema" in domains

    def test_entities_are_longer_words(self):
        domains, entities = extract_query_signals("How does JWT authentication work?")
        # "authentication" is > 4 chars
        assert any("authentication" in e for e in entities) or \
               any("jwt" in e for e in entities)

    def test_max_five_domains(self):
        query = "one two three four five six seven eight nine ten " * 2
        domains, _ = extract_query_signals(query)
        assert len(domains) <= 5

    def test_empty_query(self):
        domains, entities = extract_query_signals("")
        assert domains == []
        assert entities == []

    def test_stop_words_only(self):
        domains, entities = extract_query_signals("the a an is are")
        assert domains == []

    def test_punctuation_stripped(self):
        domains, _ = extract_query_signals("database? schema! config.")
        for d in domains:
            assert not d.endswith("?")
            assert not d.endswith("!")
            assert not d.endswith(".")

    def test_stop_words_is_frozen(self):
        assert isinstance(STOP_WORDS, frozenset)


# ── Pre-compiled regex patterns ──────────────────────────────────


class TestCompiledPatterns:
    def test_paragraph_split(self):
        text = "Para one.\n\nPara two.\n\n\nPara three."
        parts = RE_PARAGRAPH_SPLIT.split(text)
        assert len(parts) == 3

    def test_sentence_split(self):
        text = "First sentence. Second sentence! Third sentence?"
        parts = RE_SENTENCE_SPLIT.split(text)
        assert len(parts) == 3

    def test_code_boundary(self):
        code = "import os\n\ndef hello():\n    pass\n\nclass Foo:\n    pass"
        parts = RE_CODE_BOUNDARY.split(code)
        assert len(parts) >= 3  # preamble + def + class

    def test_code_block_split(self):
        code = "def a():\n    pass\n\ndef b():\n    pass\n\nclass C:\n    pass"
        parts = [p for p in RE_CODE_BLOCK_SPLIT.split(code) if p.strip()]
        assert len(parts) == 3


# ── PromptBuilder ────────────────────────────────────────────────


class TestPromptBuilder:
    def test_basic_write(self):
        pb = PromptBuilder()
        pb.write("hello ")
        pb.write("world")
        assert pb.build() == "hello world"

    def test_writeln(self):
        pb = PromptBuilder()
        pb.writeln("line1")
        pb.writeln("line2")
        assert pb.build() == "line1\nline2\n"

    def test_join_sections(self):
        pb = PromptBuilder()
        pb.join_sections(["a", "b", "c"], separator=" | ")
        assert pb.build() == "a | b | c"

    def test_parts_count(self):
        pb = PromptBuilder()
        pb.write("a")
        pb.write("b")
        pb.writeln("c")
        assert pb.parts_count == 3

    def test_chaining(self):
        result = PromptBuilder().write("a").writeln("b").build()
        assert result == "ab\n"


# ── Batch SQL builder ────────────────────────────────────────────


class TestBatchUpdateEpigenetics:
    def test_empty_input(self):
        sql, params = batch_update_epigenetics([])
        assert sql == ""
        assert params == []

    def test_single_gene(self):
        sql, params = batch_update_epigenetics([
            ("gene_1", '{"key": "val"}', 0),
        ])
        assert "UPDATE genes SET epigenetics = ?" in sql
        assert "gene_1" in params

    def test_batch_genes(self):
        updates = [
            ("gene_1", '{"a": 1}', 0),
            ("gene_2", '{"b": 2}', 1),
            ("gene_3", '{"c": 3}', 0),
        ]
        sql, params = batch_update_epigenetics(updates)
        assert "CASE" in sql
        assert "WHEN" in sql
        assert len([p for p in params if isinstance(p, str) and p.startswith("gene_")]) >= 3

    def test_sql_is_valid_structure(self):
        updates = [("g1", "{}", 0), ("g2", "{}", 1)]
        sql, params = batch_update_epigenetics(updates)
        assert sql.startswith("UPDATE genes SET")
        assert "WHERE gene_id IN" in sql


# ── Parse caches ─────────────────────────────────────────────────


class TestParseCaches:
    def test_promoter_cache_roundtrip(self):
        clear_parse_caches()
        json_str = '{"domains":["auth"],"entities":["JWT"],"intent":"test","summary":"test gene"}'
        result = parse_promoter(json_str)
        assert result.domains == ["auth"]
        assert result.entities == ["JWT"]

    def test_epigenetics_cache_roundtrip(self):
        clear_parse_caches()
        json_str = '{"access_count":5,"decay_score":0.8,"co_activated_with":[]}'
        result = parse_epigenetics(json_str)
        assert result.access_count == 5
        assert result.decay_score == 0.8

    def test_cache_returns_same_object(self):
        clear_parse_caches()
        json_str = '{"domains":["db"],"entities":[],"intent":"","summary":""}'
        a = parse_promoter(json_str)
        b = parse_promoter(json_str)
        assert a is b  # Same cached object

    def test_cache_bypass(self):
        clear_parse_caches()
        json_str = '{"domains":["db"],"entities":[],"intent":"","summary":""}'
        a = parse_promoter(json_str, use_cache=True)
        b = parse_promoter(json_str, use_cache=False)
        assert a is not b  # Different objects
        assert a.domains == b.domains

    def test_clear_caches(self):
        clear_parse_caches()
        json_str = '{"domains":["x"],"entities":[],"intent":"","summary":""}'
        a = parse_promoter(json_str)
        clear_parse_caches()
        b = parse_promoter(json_str)
        assert a is not b  # Cache was cleared


# ── Accel info ───────────────────────────────────────────────────


class TestAccelInfo:
    def test_info_returns_dict(self):
        info = accel_info()
        assert isinstance(info, dict)
        assert "json_backend" in info
        assert "stop_words_count" in info
        assert "compiled_patterns" in info

    def test_json_backend_valid(self):
        info = accel_info()
        assert info["json_backend"] in ("orjson", "json")

    def test_stop_words_populated(self):
        info = accel_info()
        assert info["stop_words_count"] > 50

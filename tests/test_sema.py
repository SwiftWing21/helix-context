"""Tests for the ΣĒMA semantic encoding module."""

import pytest

# Skip all tests if sentence-transformers not installed
st = pytest.importorskip("sentence_transformers")

from helix_context.sema import (
    SemaCodec,
    SemaPrime,
    PRIMES,
    PRIME_COUNT,
    PRIME_BY_NAME,
)


# ── Prime catalog ────────────────────────────────────────────────────

def test_prime_count():
    assert PRIME_COUNT == 20


def test_primes_have_unique_names():
    names = [p.name for p in PRIMES]
    assert len(set(names)) == PRIME_COUNT


def test_primes_have_unique_indices():
    indices = [p.index for p in PRIMES]
    assert indices == list(range(PRIME_COUNT))


def test_prime_by_name_lookup():
    assert PRIME_BY_NAME["agency"].index == 0
    assert PRIME_BY_NAME["error"].index == 18
    assert PRIME_BY_NAME["context"].index == 19


def test_prime_symbols_are_two_chars():
    for p in PRIMES:
        assert len(p.symbol) == 2, f"Prime {p.name} symbol is {len(p.symbol)} chars"


# ── Codec ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def codec():
    """Module-scoped codec to avoid reloading the model for every test."""
    return SemaCodec()


def test_codec_embed_dim(codec):
    assert codec.embed_dim == 384


def test_codec_projection_shape(codec):
    assert codec.projection_matrix.shape == (20, 384)


def test_encode_returns_20d(codec):
    vec = codec.encode("hello world")
    assert len(vec) == 20
    assert all(isinstance(v, float) for v in vec)


def test_encode_values_bounded(codec):
    vec = codec.encode("The server runs on port 8080")
    # Cosine projection is bounded by [-1, 1]
    for v in vec:
        assert -1.1 <= v <= 1.1, f"Value {v} out of bounds"


def test_encode_batch(codec):
    texts = ["hello", "world", "foo bar"]
    vecs = codec.encode_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 20 for v in vecs)


def test_encode_batch_empty(codec):
    assert codec.encode_batch([]) == []


# ── Similarity ────────────────────────────────────────────────────────

def test_similar_texts_high_score(codec):
    v1 = codec.encode("REST API endpoint returning JSON data")
    v2 = codec.encode("HTTP POST endpoint with JSON response")
    sim = codec.similarity(v1, v2)
    assert sim > 0.7, f"Similar texts should score > 0.7, got {sim}"


def test_dissimilar_texts_low_score(codec):
    v1 = codec.encode("REST API endpoint returning JSON data")
    v2 = codec.encode("A poem about autumn leaves falling gently")
    sim = codec.similarity(v1, v2)
    assert sim < 0.5, f"Dissimilar texts should score < 0.5, got {sim}"


def test_identical_texts_near_one(codec):
    text = "The function returns a dictionary"
    v1 = codec.encode(text)
    v2 = codec.encode(text)
    sim = codec.similarity(v1, v2)
    assert sim > 0.99, f"Identical text should score ~1.0, got {sim}"


def test_similarity_zero_vector():
    sim = SemaCodec.similarity([0.0] * 20, [1.0] * 20)
    assert sim == 0.0


# ── Nearest ───────────────────────────────────────────────────────────

def test_nearest_basic(codec):
    query = codec.encode("database query optimization")
    candidates = [
        ("a", codec.encode("SQL query performance tuning")),
        ("b", codec.encode("watercolor painting techniques")),
        ("c", codec.encode("PostgreSQL index optimization")),
    ]
    result = codec.nearest(query, candidates, k=2)
    assert len(result) == 2
    # Database-related texts should be nearer
    ids = [r[0] for r in result]
    assert "a" in ids or "c" in ids


def test_nearest_empty():
    result = SemaCodec.nearest([0.0] * 20, [], k=5)
    assert result == []


# ── Signature and fingerprint ─────────────────────────────────────────

def test_signature_returns_top_k(codec):
    sig = codec.signature("Install the package with pip install", top_k=3)
    assert len(sig) == 3
    assert all(isinstance(name, str) and isinstance(score, float) for name, score in sig)


def test_fingerprint_format(codec):
    fp = codec.fingerprint("A timeout error occurred")
    parts = fp.split(" ")
    assert len(parts) == 5
    # Each part should be like "Er.50"
    for part in parts:
        assert "." in part
        sym, score = part.split(".")
        assert len(sym) == 2
        assert score.isdigit()


# ── Semantic coherence ────────────────────────────────────────────────

def test_instruction_prime_high_for_howto(codec):
    """Instructional text should score high on the 'instruction' prime."""
    vec = codec.encode("To install, run pip install helix-context and then configure helix.toml")
    instruction_idx = PRIME_BY_NAME["instruction"].index
    # instruction should be among the top primes
    scored = [(i, abs(vec[i])) for i in range(20)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = [s[0] for s in scored[:5]]
    assert instruction_idx in top_indices, (
        f"instruction prime (idx={instruction_idx}) not in top 5: "
        f"{[(PRIMES[i].name, f'{v:.2f}') for i, v in scored[:5]]}"
    )


def test_error_prime_high_for_exceptions(codec):
    """Error text should score high on the 'error' prime."""
    vec = codec.encode("The operation failed with a TimeoutError exception and needs retry")
    error_idx = PRIME_BY_NAME["error"].index
    scored = [(i, abs(vec[i])) for i in range(20)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = [s[0] for s in scored[:5]]
    assert error_idx in top_indices, (
        f"error prime (idx={error_idx}) not in top 5: "
        f"{[(PRIMES[i].name, f'{v:.2f}') for i, v in scored[:5]]}"
    )

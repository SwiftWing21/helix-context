"""Tests for CachedDAL (helix_context.adapters.cache)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from helix_context.adapters.cache import (
    DEFAULT_TTLS_S,
    CachedDAL,
    fetch_packet_sources_cached,
)
from helix_context.adapters.dal import DAL, FetchResult


# ── Basic fetch / hit / miss ─────────────────────────────────────────


def test_first_fetch_is_miss(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    cache = CachedDAL(DAL())
    result = cache.fetch(str(p))
    assert result.ok
    assert result.meta["cache_hit"] is False
    assert cache.stats()["misses"] == 1


def test_second_fetch_is_hit(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    cache = CachedDAL(DAL())
    cache.fetch(str(p))  # miss
    result = cache.fetch(str(p))  # hit
    assert result.ok
    assert result.meta["cache_hit"] is True
    assert cache.stats()["hits"] == 1


def test_hit_returns_same_text(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    cache = CachedDAL(DAL())
    r1 = cache.fetch(str(p))
    r2 = cache.fetch(str(p))
    assert r1.text == r2.text == "hello world"


def test_bypass_forces_miss_without_poisoning(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("first", encoding="utf-8")
    cache = CachedDAL(DAL())
    cache.fetch(str(p))  # populate
    p.write_text("second", encoding="utf-8")

    # Bypass: force fresh fetch
    result = cache.fetch(str(p), bypass_cache=True)
    assert result.text == "second"

    # Subsequent cached fetch should see the new content (bypass wrote through)
    result2 = cache.fetch(str(p))
    assert result2.text == "second"


# ── TTL + volatility ─────────────────────────────────────────────────


def test_volatility_class_maps_to_ttl(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    cache = CachedDAL(DAL())
    cache.fetch(str(p), volatility_class="hot")
    # TTL is stored on the entry
    entry = cache._store[str(p)]
    assert entry.ttl_s == DEFAULT_TTLS_S["hot"]


def test_explicit_ttl_overrides_volatility(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    cache = CachedDAL(DAL())
    cache.fetch(str(p), volatility_class="stable", ttl_s=5.0)
    entry = cache._store[str(p)]
    assert entry.ttl_s == 5.0


def test_expired_entry_refetched(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("first", encoding="utf-8")
    cache = CachedDAL(DAL())
    cache.fetch(str(p), ttl_s=0.05)
    time.sleep(0.1)
    p.write_text("second", encoding="utf-8")

    result = cache.fetch(str(p), ttl_s=0.05)
    assert result.text == "second"
    assert result.meta["cache_hit"] is False


# ── LRU eviction ─────────────────────────────────────────────────────


def test_lru_evicts_oldest_when_full(tmp_path):
    cache = CachedDAL(DAL(), max_entries=2)
    for name in ("a", "b", "c"):
        p = tmp_path / f"{name}.txt"
        p.write_text(name, encoding="utf-8")
        cache.fetch(str(p))
    assert len(cache._store) == 2
    assert cache.stats()["evictions"] == 1
    # "a" should have been evicted (oldest)
    assert str(tmp_path / "a.txt") not in cache._store


def test_lru_touches_on_hit(tmp_path):
    cache = CachedDAL(DAL(), max_entries=2)
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for p in (a, b, c):
        p.write_text(p.name, encoding="utf-8")

    cache.fetch(str(a))
    cache.fetch(str(b))
    cache.fetch(str(a))  # touch a — makes b the oldest
    cache.fetch(str(c))  # should evict b, not a

    assert str(a) in cache._store
    assert str(b) not in cache._store
    assert str(c) in cache._store


# ── Invalidation ─────────────────────────────────────────────────────


def test_invalidate_single_key(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    cache = CachedDAL(DAL())
    cache.fetch(str(p))
    assert cache.invalidate(str(p)) is True
    assert cache.invalidate(str(p)) is False  # second call no-op
    assert cache.stats()["entries"] == 0


def test_invalidate_all(tmp_path):
    cache = CachedDAL(DAL())
    for name in ("a", "b", "c"):
        p = tmp_path / f"{name}.txt"
        p.write_text(name, encoding="utf-8")
        cache.fetch(str(p))
    n = cache.invalidate_all()
    assert n == 3
    assert cache.stats()["entries"] == 0


def test_invalidate_by_prefix(tmp_path):
    """Use-case: ingest hook says /docs/ changed, evict everything under it."""
    cache = CachedDAL(DAL())
    docs_a = tmp_path / "docs_a.txt"
    docs_b = tmp_path / "docs_b.txt"
    code_c = tmp_path / "code_c.txt"
    for p in (docs_a, docs_b, code_c):
        p.write_text(p.name, encoding="utf-8")
        cache.fetch(str(p))

    # Dropping by the "docs_" prefix keeps code_c
    prefix = str(tmp_path / "docs_")
    n = cache.invalidate_by_prefix(prefix)
    assert n == 2
    assert str(code_c) in cache._store


# ── Error handling ───────────────────────────────────────────────────


def test_failed_fetch_is_not_cached():
    """Soft-failures shouldn't poison the cache."""
    fake_dal = MagicMock()
    fake_dal.fetch.return_value = FetchResult(None, {"error": "nope"})
    cache = CachedDAL(fake_dal)
    cache.fetch("/nonexistent")
    assert "/nonexistent" not in cache._store


# ── fetch_packet_sources_cached ─────────────────────────────────────


def test_packet_cached_fetch_honors_verified_bucket(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello", encoding="utf-8")
    packet = {
        "verified": [{"source_id": str(a), "volatility_class": "stable"}],
        "stale_risk": [],
        "contradictions": [],
        "refresh_targets": [],
    }
    cache = CachedDAL(DAL())
    results = fetch_packet_sources_cached(packet, cache)
    assert len(results) == 1
    assert results[0][1].ok
    # Second call: should be a hit
    results2 = fetch_packet_sources_cached(packet, cache)
    assert results2[0][1].meta["cache_hit"] is True


def test_packet_cached_fetch_bypasses_stale_risk(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello", encoding="utf-8")
    packet = {
        "verified": [],
        "stale_risk": [{"source_id": str(a), "volatility_class": "hot"}],
        "contradictions": [],
        "refresh_targets": [],
    }
    cache = CachedDAL(DAL())
    results = fetch_packet_sources_cached(packet, cache)
    # First fetch — stale_risk ALWAYS bypasses cache, so miss
    assert results[0][1].meta["cache_hit"] is False
    # Re-run — stale_risk bucket still forces bypass
    results2 = fetch_packet_sources_cached(packet, cache)
    assert results2[0][1].meta["cache_hit"] is False


def test_packet_cached_fetch_bypasses_refresh_targets(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello", encoding="utf-8")
    packet = {
        "verified": [],
        "stale_risk": [],
        "contradictions": [],
        "refresh_targets": [{"source_id": str(a)}],
    }
    cache = CachedDAL(DAL())
    results = fetch_packet_sources_cached(packet, cache)
    assert results[0][1].meta["cache_hit"] is False


# ── Stats ────────────────────────────────────────────────────────────


def test_reset_stats(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    cache = CachedDAL(DAL())
    cache.fetch(str(p))
    cache.fetch(str(p))
    s = cache.stats()
    assert s["hits"] > 0
    cache.reset_stats()
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0


def test_hit_rate_computed(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    cache = CachedDAL(DAL())
    cache.fetch(str(p))  # miss
    cache.fetch(str(p))  # hit
    cache.fetch(str(p))  # hit
    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3, rel=1e-3)

"""
Tests for helix_context.metrics — TokenCounter atomicity, persistence,
session vs lifetime semantics, exact vs estimated buckets, and the
estimate_tokens helper.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from helix_context.metrics import (
    CHARS_PER_TOKEN_ESTIMATE,
    TokenCounter,
    estimate_tokens,
)


@pytest.fixture
def counter_path(tmp_path):
    return tmp_path / "metrics.json"


@pytest.fixture
def counter(counter_path):
    return TokenCounter(persist_path=counter_path, persist_interval_s=0.0)


class TestEstimateTokens:
    def test_empty_returns_zero(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0  # type: ignore[arg-type]

    def test_short_string_returns_at_least_one(self):
        assert estimate_tokens("hi") >= 1

    def test_proportional_to_length(self):
        short = "x" * 4
        long = "x" * 400
        s = estimate_tokens(short)
        l = estimate_tokens(long)
        assert l > s
        # Roughly chars / CHARS_PER_TOKEN_ESTIMATE
        assert l == 400 // CHARS_PER_TOKEN_ESTIMATE


class TestAddBasic:
    def test_session_starts_at_zero(self, counter):
        snap = counter.snapshot()
        assert snap["session"]["prompt_tokens"] == 0
        assert snap["session"]["completion_tokens"] == 0
        assert snap["session"]["total"] == 0

    def test_lifetime_starts_at_zero_when_no_file(self, counter):
        snap = counter.snapshot()
        assert snap["lifetime"]["prompt_tokens"] == 0
        assert snap["lifetime"]["total"] == 0

    def test_add_bumps_session_and_lifetime(self, counter):
        counter.add(10, 20)
        snap = counter.snapshot()
        assert snap["session"]["prompt_tokens"] == 10
        assert snap["session"]["completion_tokens"] == 20
        assert snap["session"]["total"] == 30
        assert snap["lifetime"]["prompt_tokens"] == 10
        assert snap["lifetime"]["total"] == 30

    def test_negative_values_ignored(self, counter):
        counter.add(-5, -10)
        snap = counter.snapshot()
        assert snap["session"]["total"] == 0

    def test_estimated_bucket_separate(self, counter):
        counter.add(100, 200, estimated=False)
        counter.add(50, 75, estimated=True)
        snap = counter.snapshot()
        assert snap["session"]["prompt_tokens"] == 100
        assert snap["session"]["completion_tokens"] == 200
        assert snap["session"]["estimated_prompt_tokens"] == 50
        assert snap["session"]["estimated_completion_tokens"] == 75
        assert snap["session"]["total"] == 300
        assert snap["session"]["estimated_total"] == 125


class TestAddFromUsage:
    def test_returns_true_on_valid_usage(self, counter):
        ok = counter.add_from_usage({"prompt_tokens": 30, "completion_tokens": 70})
        assert ok is True
        snap = counter.snapshot()
        assert snap["session"]["total"] == 100

    def test_returns_false_on_none(self, counter):
        ok = counter.add_from_usage(None)
        assert ok is False

    def test_returns_false_on_zero_zero(self, counter):
        ok = counter.add_from_usage({"prompt_tokens": 0, "completion_tokens": 0})
        assert ok is False

    def test_returns_false_on_malformed(self, counter):
        ok = counter.add_from_usage({"prompt_tokens": "not-a-number"})
        assert ok is False

    def test_returns_false_on_non_dict(self, counter):
        assert counter.add_from_usage("string") is False  # type: ignore[arg-type]
        assert counter.add_from_usage(42) is False  # type: ignore[arg-type]


class TestPersistence:
    def test_flush_writes_file(self, counter, counter_path):
        counter.add(100, 200)
        counter.flush()
        assert counter_path.exists()
        on_disk = json.loads(counter_path.read_text(encoding="utf-8"))
        assert on_disk["prompt_tokens"] == 100
        assert on_disk["completion_tokens"] == 200

    def test_lifetime_loads_on_init(self, counter_path):
        first = TokenCounter(persist_path=counter_path)
        first.add(100, 200)
        first.flush()

        second = TokenCounter(persist_path=counter_path)
        snap = second.snapshot()
        assert snap["lifetime"]["prompt_tokens"] == 100
        assert snap["lifetime"]["total"] == 300
        # But session resets
        assert snap["session"]["total"] == 0

    def test_session_resets_lifetime_persists(self, counter_path):
        TokenCounter(persist_path=counter_path).add(50, 50)
        # First counter goes out of scope without flushing — interval is default 30s.
        # We need flush() to actually write.
        c = TokenCounter(persist_path=counter_path)
        c.add(50, 50)
        c.flush()

        c2 = TokenCounter(persist_path=counter_path)
        snap = c2.snapshot()
        # session is fresh, lifetime carries the persisted value
        assert snap["session"]["total"] == 0
        assert snap["lifetime"]["total"] == 100  # only the flushed call

    def test_corrupt_file_falls_back_to_zero(self, counter_path):
        counter_path.parent.mkdir(parents=True, exist_ok=True)
        counter_path.write_text("not valid json", encoding="utf-8")
        counter = TokenCounter(persist_path=counter_path)
        snap = counter.snapshot()
        assert snap["lifetime"]["total"] == 0


class TestResetLifetime:
    def test_reset_zeroes_lifetime_and_writes(self, counter, counter_path):
        counter.add(500, 500)
        counter.flush()
        counter.reset_lifetime()
        snap = counter.snapshot()
        assert snap["lifetime"]["total"] == 0
        # Session is unchanged
        assert snap["session"]["total"] == 1000
        # File reflects the zero
        on_disk = json.loads(counter_path.read_text(encoding="utf-8"))
        assert on_disk["prompt_tokens"] == 0
        assert on_disk["completion_tokens"] == 0


class TestThreadSafety:
    def test_concurrent_adds_are_atomic(self, counter):
        # 10 threads, each adding (1, 1) ten thousand times.
        # Final session total should be exactly 200000.
        N_THREADS = 10
        N_ADDS = 10_000

        def worker():
            for _ in range(N_ADDS):
                counter.add(1, 1)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = counter.snapshot()
        assert snap["session"]["prompt_tokens"] == N_THREADS * N_ADDS
        assert snap["session"]["completion_tokens"] == N_THREADS * N_ADDS
        assert snap["session"]["total"] == 2 * N_THREADS * N_ADDS

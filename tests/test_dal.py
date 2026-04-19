"""Tests for the DAL reference adapter (helix_context.adapters.dal)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from helix_context.adapters.dal import (
    DAL,
    FetchResult,
    _detect_scheme,
    fetch_packet_sources,
)


# ── Scheme detection ────────────────────────────────────────────────


def test_detect_scheme_explicit():
    assert _detect_scheme("https://example.com/foo") == "https"
    assert _detect_scheme("s3://bucket/key") == "s3"
    assert _detect_scheme("file:///abs/path") == "file"


def test_detect_scheme_bare_path():
    assert _detect_scheme("/repo/foo.py") == "file"
    assert _detect_scheme("relative/path.md") == "file"


def test_detect_scheme_windows_drive():
    assert _detect_scheme("C:\\Users\\foo.txt") == "file"
    assert _detect_scheme("D:/repo/bar.py") == "file"


# ── FetchResult ────────────────────────────────────────────────────


def test_fetch_result_ok_property():
    assert FetchResult("hello", {}).ok is True
    assert FetchResult(None, {"error": "x"}).ok is False


# ── file fetcher ────────────────────────────────────────────────────


def test_file_fetcher_reads_existing_file(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hello world", encoding="utf-8")

    dal = DAL()
    result = dal.fetch(str(p))
    assert result.ok
    assert result.text == "hello world"
    assert result.meta["scheme"] == "file"
    assert result.meta["bytes_read"] == len("hello world")
    assert result.meta["truncated"] is False


def test_file_fetcher_honors_max_bytes(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("x" * 1000, encoding="utf-8")

    dal = DAL(max_bytes=100)
    result = dal.fetch(str(p))
    assert result.ok
    assert len(result.text) == 100
    assert result.meta["truncated"] is True


def test_file_fetcher_missing_file_soft_fails(tmp_path):
    dal = DAL()
    result = dal.fetch(str(tmp_path / "nope.txt"))
    assert not result.ok
    assert "not found" in result.meta["error"]


def test_file_fetcher_with_file_scheme_prefix(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("x", encoding="utf-8")
    dal = DAL()
    result = dal.fetch(f"file://{p}")
    assert result.ok


# ── http fetcher ────────────────────────────────────────────────────


def test_http_fetcher_success():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.text = "hello http"
    fake_resp.headers = {"content-type": "text/plain"}

    dal = DAL()
    with patch("httpx.get", return_value=fake_resp) as mock_get:
        result = dal.fetch("https://example.com/foo")
    assert result.ok
    assert result.text == "hello http"
    assert result.meta["status_code"] == 200
    assert result.meta["content_type"] == "text/plain"
    mock_get.assert_called_once()


def test_http_fetcher_4xx_returns_none_text_with_error():
    fake_resp = MagicMock()
    fake_resp.status_code = 404
    fake_resp.text = "not found"
    fake_resp.headers = {}

    dal = DAL()
    with patch("httpx.get", return_value=fake_resp):
        result = dal.fetch("https://example.com/missing")
    assert result.text is None
    assert result.meta["status_code"] == 404
    assert "404" in result.meta["error"]


def test_http_fetcher_exception_soft_fails():
    dal = DAL()
    with patch("httpx.get", side_effect=Exception("network down")):
        result = dal.fetch("https://example.com/foo")
    assert not result.ok
    assert "network down" in result.meta["error"]


# ── Scheme registration ─────────────────────────────────────────────


def test_register_custom_scheme():
    def custom_fetcher(source_id, **kwargs):
        return FetchResult(f"got {source_id}", {"scheme": "my"})

    dal = DAL()
    dal.register("my", custom_fetcher)
    result = dal.fetch("my://something")
    assert result.ok
    assert result.text == "got my://something"


def test_register_overrides_default():
    """User can override a default scheme (e.g., a custom file backend)."""
    def noop(source_id, **kwargs):
        return FetchResult("stub", {"scheme": "file"})
    dal = DAL()
    dal.register("file", noop)
    # Any path routes to the stub now
    result = dal.fetch("/nonexistent/path")
    assert result.text == "stub"


def test_unknown_scheme_returns_error():
    dal = DAL()
    result = dal.fetch("git://example.com/repo.git")
    assert not result.ok
    assert "no fetcher registered" in result.meta["error"]


def test_fetcher_returning_wrong_type_is_contained():
    """Custom fetcher returning not-FetchResult doesn't crash the DAL."""
    def bad_fetcher(source_id, **kwargs):
        return "not a FetchResult"

    dal = DAL()
    dal.register("bad", bad_fetcher)
    result = dal.fetch("bad://x")
    assert not result.ok
    assert "FetchResult" in result.meta["error"]


# ── fetch_packet_sources ───────────────────────────────────────────


def test_fetch_packet_sources_iterates_all_buckets(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A", encoding="utf-8")
    b.write_text("B", encoding="utf-8")

    packet = {
        "verified": [{"source_id": str(a)}],
        "stale_risk": [{"source_id": str(b)}],
        "contradictions": [],
        "refresh_targets": [],
    }
    dal = DAL()
    results = fetch_packet_sources(packet, dal)
    assert len(results) == 2
    sids = [sid for sid, _ in results]
    assert str(a) in sids
    assert str(b) in sids
    assert all(r.ok for _, r in results)


def test_fetch_packet_sources_dedupes_source_ids(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("A", encoding="utf-8")

    packet = {
        "verified": [{"source_id": str(a)}],
        "stale_risk": [{"source_id": str(a)}],  # same path
        "contradictions": [],
        "refresh_targets": [{"source_id": str(a)}],  # again
    }
    results = fetch_packet_sources(packet)
    assert len(results) == 1


def test_fetch_packet_sources_caps_at_max(tmp_path):
    items = []
    for i in range(5):
        p = tmp_path / f"f{i}.txt"
        p.write_text(str(i), encoding="utf-8")
        items.append({"source_id": str(p)})
    packet = {"verified": items, "stale_risk": [], "contradictions": [],
              "refresh_targets": []}
    results = fetch_packet_sources(packet, max_sources=3)
    assert len(results) == 3


def test_fetch_packet_sources_skip_refresh_targets_when_disabled(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("A", encoding="utf-8")
    packet = {
        "verified": [],
        "stale_risk": [],
        "contradictions": [],
        "refresh_targets": [{"source_id": str(a)}],
    }
    results = fetch_packet_sources(packet, include_refresh_targets=False)
    assert results == []

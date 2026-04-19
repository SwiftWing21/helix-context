"""Data Access Layer — uniform ``fetch(source_id) -> bytes/str`` across
heterogeneous storage backends.

Helix emits source_ids that can be anything: a local file path, an
``https://`` URL, an ``s3://`` key, a ``git://`` ref, a database URI.
This module provides a scheme-dispatch fetcher so the agent's call
site stays uniform regardless of what store the packet pointed at.

Design principles:

1. **Registry is opt-in.** The default DAL ships with file + HTTP
   fetchers. S3, git, and custom schemes are registered by the user.
2. **Heuristic fallback.** Bare paths (no scheme) are treated as files
   if they exist on disk; otherwise the DAL returns None rather than
   guessing at a scheme.
3. **Soft-fail.** Fetch errors return None + a reason; they don't
   raise. The agent decides what to do with missing bytes.
4. **Cache is out of scope.** Wrap ``DAL.fetch`` in your own cache
   (LRU, redis, whatever). Don't bolt one into the reference adapter.

Basic usage::

    from helix_context.adapters.dal import DAL

    dal = DAL()  # file + http by default
    text, meta = dal.fetch("/repo/config.yaml")
    text, meta = dal.fetch("https://example.com/api/docs.json")

With packet source_ids::

    for item in packet["stale_risk"]:
        sid = item["source_id"]
        text, meta = dal.fetch(sid)
        if text is None:
            log.warning("fetch failed for %s: %s", sid, meta["error"])
            continue
        agent.process(text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

log = logging.getLogger("helix.adapters.dal")


@dataclass
class FetchResult:
    """Normalized fetch outcome.

    ``text`` is None when the fetch failed; ``meta`` always carries
    a ``scheme`` + optional ``error`` / ``status_code`` / ``bytes_read``.
    """
    text: Optional[str]
    meta: dict

    @property
    def ok(self) -> bool:
        return self.text is not None


# Fetcher signature: takes (source_id, **kwargs) → FetchResult
Fetcher = Callable[..., FetchResult]


class DAL:
    """Scheme-dispatching fetcher registry.

    Call ``dal.fetch(source_id)``; the first registered scheme whose
    prefix matches wins. Register new schemes with
    ``dal.register(scheme, fetcher)``.
    """

    def __init__(self, *, http_timeout: float = 15.0,
                 max_bytes: int = 5_000_000) -> None:
        self._fetchers: dict[str, Fetcher] = {}
        self.http_timeout = http_timeout
        self.max_bytes = max_bytes

        # Register defaults. Bare paths dispatch via the "file" scheme
        # after a heuristic in fetch().
        self.register("file", _fetch_file)
        self.register("http", _fetch_http)
        self.register("https", _fetch_http)

    def register(self, scheme: str, fetcher: Fetcher) -> None:
        """Register (or override) a scheme's fetcher."""
        self._fetchers[scheme.lower()] = fetcher

    def fetch(self, source_id: str, **kwargs) -> FetchResult:
        """Resolve a source_id → (text, metadata). Never raises."""
        if not source_id:
            return FetchResult(None, {"error": "empty source_id"})

        scheme = _detect_scheme(source_id)
        fetcher = self._fetchers.get(scheme)
        if fetcher is None:
            return FetchResult(None, {
                "error": f"no fetcher registered for scheme={scheme!r}",
                "scheme": scheme,
            })

        # Merge DAL-level defaults into kwargs
        kwargs.setdefault("timeout", self.http_timeout)
        kwargs.setdefault("max_bytes", self.max_bytes)

        try:
            result = fetcher(source_id, **kwargs)
        except Exception as exc:
            log.warning("DAL fetch failed for %s: %s", source_id, exc)
            return FetchResult(None, {"error": str(exc), "scheme": scheme})
        # Guarantee FetchResult shape even if a custom fetcher mis-returns
        if not isinstance(result, FetchResult):
            return FetchResult(None, {
                "error": f"fetcher returned {type(result).__name__}, "
                         "expected FetchResult",
                "scheme": scheme,
            })
        result.meta.setdefault("scheme", scheme)
        return result


# ── Scheme detection ────────────────────────────────────────────────


def _detect_scheme(source_id: str) -> str:
    """Pick a scheme for ``source_id``.

    Explicit schemes (``s3://``, ``https://``) win via urlparse. Bare
    paths fall through to ``file``. Windows drive letters (``C:\\``)
    also route to ``file``.
    """
    if "://" in source_id:
        parsed = urlparse(source_id)
        if parsed.scheme:
            return parsed.scheme.lower()
    # Windows drive letter? c:\ or C:/
    if len(source_id) >= 2 and source_id[1] == ":":
        return "file"
    return "file"


# ── Built-in fetchers ───────────────────────────────────────────────


def _fetch_file(source_id: str, *, max_bytes: int = 5_000_000,
                **_kw) -> FetchResult:
    """File scheme. Accepts ``file://path`` or bare path."""
    if source_id.startswith("file://"):
        path_str = source_id[len("file://"):]
    else:
        path_str = source_id
    path = Path(path_str)
    try:
        if not path.is_file():
            return FetchResult(None, {
                "error": "file not found",
                "path": str(path),
            })
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_bytes:
            text = text[:max_bytes]
            truncated = True
        else:
            truncated = False
        return FetchResult(text, {
            "scheme": "file",
            "path": str(path),
            "bytes_read": len(text),
            "truncated": truncated,
        })
    except PermissionError as exc:
        return FetchResult(None, {"error": f"permission denied: {exc}"})
    except OSError as exc:
        return FetchResult(None, {"error": f"os error: {exc}"})


def _fetch_http(source_id: str, *, timeout: float = 15.0,
                max_bytes: int = 5_000_000,
                headers: Optional[dict] = None,
                **_kw) -> FetchResult:
    """HTTP/HTTPS scheme via httpx."""
    try:
        import httpx
    except ImportError:
        return FetchResult(None, {
            "error": "httpx not installed (required for http/https fetcher)",
        })
    try:
        resp = httpx.get(
            source_id,
            timeout=timeout,
            headers=headers or {},
            follow_redirects=True,
        )
        body = resp.text
        if len(body) > max_bytes:
            body = body[:max_bytes]
            truncated = True
        else:
            truncated = False
        return FetchResult(
            body if resp.status_code < 400 else None,
            {
                "status_code": resp.status_code,
                "bytes_read": len(body),
                "truncated": truncated,
                "content_type": resp.headers.get("content-type"),
                "error": None if resp.status_code < 400 else
                f"HTTP {resp.status_code}",
            },
        )
    except Exception as exc:
        return FetchResult(None, {"error": f"http error: {exc}"})


# ── Optional S3 fetcher (not registered by default) ─────────────────


def fetch_s3(source_id: str, *, max_bytes: int = 5_000_000,
             **_kw) -> FetchResult:
    """S3 fetcher. Requires ``boto3``; install separately.

    Register via ``dal.register("s3", fetch_s3)`` before calling
    ``fetch("s3://bucket/key")``.
    """
    try:
        import boto3  # type: ignore
    except ImportError:
        return FetchResult(None, {
            "error": "boto3 not installed (required for s3 fetcher). "
                     "pip install boto3",
        })
    if not source_id.startswith("s3://"):
        return FetchResult(None, {"error": f"not an s3 URL: {source_id}"})

    rest = source_id[len("s3://"):]
    if "/" not in rest:
        return FetchResult(None, {"error": "s3 URL missing key component"})
    bucket, key = rest.split("/", 1)
    try:
        s3 = boto3.client("s3")
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read(max_bytes + 1)
        truncated = len(body) > max_bytes
        text = body[:max_bytes].decode("utf-8", errors="replace")
        return FetchResult(text, {
            "scheme": "s3",
            "bucket": bucket,
            "key": key,
            "bytes_read": len(text),
            "truncated": truncated,
            "content_type": resp.get("ContentType"),
        })
    except Exception as exc:
        return FetchResult(None, {"error": f"s3 error: {exc}"})


# ── Packet-driven entry point ──────────────────────────────────────


def fetch_packet_sources(
    packet: dict,
    dal: Optional[DAL] = None,
    *,
    buckets: tuple[str, ...] = ("verified", "stale_risk", "contradictions"),
    include_refresh_targets: bool = True,
    max_sources: int = 12,
) -> list[tuple[str, FetchResult]]:
    """Fetch every source referenced by a Helix packet.

    Returns ``[(source_id, FetchResult), ...]`` preserving packet
    ordering. Deduplicates by source_id. Soft-fails on individual
    fetch errors (FetchResult carries the reason).

    This is the "path 1" integration primitive from the routing doc:
    once Helix has emitted its verdict, pull every byte the agent
    might need in one call.
    """
    dal = dal or DAL()
    seen: set[str] = set()
    ordered: list[str] = []
    for bucket in buckets:
        for item in packet.get(bucket, []) or []:
            sid = item.get("source_id")
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
    if include_refresh_targets:
        for tgt in packet.get("refresh_targets", []) or []:
            sid = tgt.get("source_id")
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)

    results: list[tuple[str, FetchResult]] = []
    for sid in ordered[:max_sources]:
        results.append((sid, dal.fetch(sid)))
    return results

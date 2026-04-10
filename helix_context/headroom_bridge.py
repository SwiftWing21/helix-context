"""
Headroom bridge — CPU-resident semantic compression for gene content.

Replaces the legacy character-level truncation (``g.content[:1000]``) at the
retrieval seam with query-agnostic semantic compression via the Headroom
toolkit (Apache-2.0) by Tejas Chopra.

Upstream: https://github.com/chopratejas/headroom
PyPI:     https://pypi.org/project/headroom-ai/

Headroom is an optional dependency (``helix-context[codec]``). When it is not
installed, ``compress_text`` falls back to the legacy truncation so the rest
of the pipeline keeps working.

Dispatch by ``content_type`` hint (drawn from ``gene.promoter.domains``):

    code / python / rust / js / ts / go  → CodeAwareCompressor
    log / pytest / build / npm / cargo   → LogCompressor
    diff / patch                         → DiffCompressor
    anything else                        → KompressCompressor (ModernBERT ONNX)

All specialists are loaded lazily on first use and cached as module-level
singletons. Headroom itself guards the Kompress model load with an internal
lock, so repeated calls from the async request path are safe.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable, Optional

log = logging.getLogger(__name__)

# ── Availability probe (cheap, no model download) ──────────────────────

_HEADROOM_IMPORT_LOCK = threading.Lock()
_HEADROOM_AVAILABLE: Optional[bool] = None  # tri-state: None = not probed yet


def is_headroom_available() -> bool:
    """Return True iff the headroom-ai package is importable.

    Does NOT load the Kompress model — that only happens on first compress call.
    """
    global _HEADROOM_AVAILABLE
    if _HEADROOM_AVAILABLE is not None:
        return _HEADROOM_AVAILABLE
    with _HEADROOM_IMPORT_LOCK:
        if _HEADROOM_AVAILABLE is not None:
            return _HEADROOM_AVAILABLE
        try:
            import headroom  # noqa: F401
            _HEADROOM_AVAILABLE = True
        except ImportError:
            log.info(
                "headroom-ai not installed — falling back to truncation. "
                "Install with: pip install helix-context[codec]"
            )
            _HEADROOM_AVAILABLE = False
    return _HEADROOM_AVAILABLE


# ── Specialist singletons (lazy) ────────────────────────────────────────

_SPECIALIST_LOCK = threading.Lock()
_KOMPRESS = None
_LOG_COMPRESSOR = None
_DIFF_COMPRESSOR = None
_CODE_COMPRESSOR = None


def _get_kompress():
    global _KOMPRESS
    if _KOMPRESS is not None:
        return _KOMPRESS
    with _SPECIALIST_LOCK:
        if _KOMPRESS is None:
            from headroom.transforms.kompress_compressor import KompressCompressor
            _KOMPRESS = KompressCompressor()
    return _KOMPRESS


def _get_log_compressor():
    global _LOG_COMPRESSOR
    if _LOG_COMPRESSOR is not None:
        return _LOG_COMPRESSOR
    with _SPECIALIST_LOCK:
        if _LOG_COMPRESSOR is None:
            from headroom.transforms.log_compressor import LogCompressor
            _LOG_COMPRESSOR = LogCompressor()
    return _LOG_COMPRESSOR


def _get_diff_compressor():
    global _DIFF_COMPRESSOR
    if _DIFF_COMPRESSOR is not None:
        return _DIFF_COMPRESSOR
    with _SPECIALIST_LOCK:
        if _DIFF_COMPRESSOR is None:
            from headroom.transforms.diff_compressor import DiffCompressor
            _DIFF_COMPRESSOR = DiffCompressor()
    return _DIFF_COMPRESSOR


def _get_code_compressor():
    global _CODE_COMPRESSOR
    if _CODE_COMPRESSOR is not None:
        return _CODE_COMPRESSOR
    with _SPECIALIST_LOCK:
        if _CODE_COMPRESSOR is None:
            from headroom.transforms.code_compressor import CodeAwareCompressor
            _CODE_COMPRESSOR = CodeAwareCompressor()
    return _CODE_COMPRESSOR


# ── Dispatch ────────────────────────────────────────────────────────────

_CODE_DOMAINS = frozenset({
    "code", "python", "rust", "javascript", "js", "typescript", "ts",
    "go", "java", "cpp", "c", "sql", "shell", "bash",
})
_LOG_DOMAINS = frozenset({
    # Strictly log-output-signaling tokens. Tool names like "cargo", "npm",
    # "build" are excluded because they also appear in code genes (e.g. a
    # Rust source file naturally has promoter domain "cargo" from Cargo.toml).
    "log", "logs", "stderr", "stdout", "pytest", "jest", "traceback",
})
_DIFF_DOMAINS = frozenset({"diff", "patch", "git_diff"})

_LANGUAGE_HINTS = {
    "python": "python", "py": "python",
    "rust": "rust", "rs": "rust",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "go": "go", "java": "java", "cpp": "cpp", "c": "c",
}


def _pick_specialist(domains: Iterable[str]) -> str:
    """Return one of: 'code' | 'log' | 'diff' | 'kompress'."""
    domains_set = {d.lower() for d in domains if d}
    if domains_set & _DIFF_DOMAINS:
        return "diff"
    if domains_set & _LOG_DOMAINS:
        return "log"
    if domains_set & _CODE_DOMAINS:
        return "code"
    return "kompress"


def _detect_language(domains: Iterable[str]) -> Optional[str]:
    for d in domains:
        hint = _LANGUAGE_HINTS.get(d.lower())
        if hint:
            return hint
    return None


# ── Public API ──────────────────────────────────────────────────────────

def compress_text(
    content: str,
    target_chars: int = 1000,
    content_type: Optional[Iterable[str]] = None,
) -> str:
    """Compress ``content`` to approximately ``target_chars`` characters.

    Parameters
    ----------
    content : str
        Raw text to compress.
    target_chars : int
        Soft cap on output length. Matches the historical ``content[:1000]``
        truncation so callers can swap 1:1 without budget changes.
    content_type : iterable of str, optional
        Hints about the content's nature, typically ``gene.promoter.domains``
        (e.g. ``["python", "code"]`` or ``["log", "pytest"]``). Determines
        which specialist is invoked. If None, falls through to Kompress.

    Returns
    -------
    str
        Compressed content. Guaranteed to be non-empty if ``content`` was
        non-empty; guaranteed to be shorter than or equal to ``content`` in
        the happy path, and falls back to ``content[:target_chars]`` on any
        error or if headroom is unavailable.
    """
    if not content:
        return ""

    # Fast path: content already under budget
    if len(content) <= target_chars:
        return content

    # Fallback path: headroom not installed
    if not is_headroom_available():
        return content[:target_chars].strip()

    domains = list(content_type) if content_type else []
    specialist = _pick_specialist(domains)

    try:
        if specialist == "code":
            language = _detect_language(domains)
            result = _get_code_compressor().compress(content, language=language)
            compressed = _extract_result_text(result)
        elif specialist == "log":
            result = _get_log_compressor().compress(content)
            compressed = _extract_result_text(result)
        elif specialist == "diff":
            result = _get_diff_compressor().compress(content)
            compressed = _extract_result_text(result)
        else:
            # Kompress aims for target_ratio = target_chars / len(content),
            # clamped so we never ask for 0% or >100%
            ratio = max(0.1, min(1.0, target_chars / max(len(content), 1)))
            result = _get_kompress().compress(content, target_ratio=ratio)
            compressed = _extract_result_text(result)
    except Exception:
        log.warning(
            "Headroom compression failed (specialist=%s), falling back to truncation",
            specialist,
            exc_info=True,
        )
        return content[:target_chars].strip()

    if not compressed:
        return content[:target_chars].strip()

    # Soft-cap: if the specialist returned more than target_chars, truncate
    # the compressed output. Still much better than raw truncation because
    # the first ``target_chars`` of a semantically-compressed string carry
    # more signal than the first ``target_chars`` of raw content.
    if len(compressed) > target_chars * 1.5:
        compressed = compressed[:target_chars].strip()

    return compressed


def _extract_result_text(result) -> str:
    """Pull the compressed string out of any Headroom *Result dataclass.

    Headroom's four specialists all return result objects with a
    ``compressed`` attribute, but we use getattr-with-fallback to stay
    robust against upstream schema drift.
    """
    for attr in ("compressed", "compressed_text", "output", "text"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            return value
    # Last resort — str() the whole result
    return str(result) if result else ""

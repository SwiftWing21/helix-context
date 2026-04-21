"""Provenance inference for the agent-context packet builder.

Phase 1 of ``docs/specs/2026-04-17-agent-context-index-build-spec.md``
requires ingest to populate ``source_kind``, ``volatility_class``, and
``last_verified_at`` on genes so the packet builder can answer freshness
questions. Without these fields, every packet item degrades to
``stale_risk`` / ``needs_refresh`` because ``freshness_known=False``.

This module centralizes the inference rules so that:

- ``scripts/backfill_gene_provenance.py`` (retroactive sweep) and
- ``HelixContextManager.ingest`` (steady-state population)

share the same logic. Changing an extension -> kind mapping here flows
through both paths without drift.

See also:
    - ``scripts/backfill_gene_provenance.py`` - initial sweep
    - ``helix_context/schemas.py::Gene`` - target fields
    - ``helix_context/context_packet.py`` - consumer
"""

from __future__ import annotations

import time
from pathlib import PurePosixPath
from typing import Any, Optional


# Extension -> source_kind. Unknown extensions fall through to "doc"
# because doc has the stable (7d) half-life -- safest retrieval default.
EXT_TO_KIND: dict[str, str] = {
    # code
    ".py": "code", ".pyi": "code", ".rs": "code", ".ts": "code",
    ".tsx": "code", ".js": "code", ".jsx": "code", ".mjs": "code",
    ".go": "code", ".java": "code", ".rb": "code", ".cpp": "code",
    ".cc": "code", ".c": "code", ".h": "code", ".hpp": "code",
    ".swift": "code", ".kt": "code", ".scala": "code", ".sh": "code",
    ".bash": "code", ".zsh": "code", ".ps1": "code", ".bat": "code",
    ".sql": "code", ".lua": "code",
    # config
    ".toml": "config", ".yaml": "config", ".yml": "config",
    ".ini": "config", ".cfg": "config", ".env": "config",
    ".conf": "config", ".properties": "config", ".config": "config",
    ".json": "config",
    # doc / narrative
    ".md": "doc", ".mdx": "doc", ".rst": "doc", ".adoc": "doc",
    ".txt": "doc", ".tex": "doc", ".ipynb": "doc",
    ".pdf": "pdf", ".html": "html", ".htm": "html",
    ".doc": "office", ".docx": "office", ".ppt": "office",
    ".pptx": "office", ".xls": "spreadsheet", ".xlsx": "spreadsheet",
    # log
    ".log": "log", ".out": "log",
    # db / tabular
    ".db": "db", ".sqlite": "db", ".sqlite3": "db",
    ".csv": "db", ".tsv": "db", ".parquet": "db", ".arrow": "db",
    # media / transcript
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".tif": "image", ".tiff": "image",
    ".svg": "image",
    ".wav": "audio", ".mp3": "audio", ".m4a": "audio", ".flac": "audio",
    ".ogg": "audio", ".aac": "audio",
    ".mp4": "video", ".mov": "video", ".avi": "video", ".webm": "video",
    ".mkv": "video",
    ".srt": "transcript", ".vtt": "transcript", ".ass": "transcript",
    ".sbv": "transcript",
}

# Explicit content_type -> source_kind. This is the caller-controlled
# "I know what this content is" hint, and should win over generic file
# extensions when it is more specific (for example benchmark JSON or an
# extracted transcript for an .mp4 source).
CONTENT_TYPE_TO_KIND: dict[str, str] = {
    # generic
    "text": "doc",
    "markdown": "doc",
    "doc": "doc",
    "conversation": "session_note",
    "memo": "session_note",
    "note": "session_note",
    "memory": "session_note",
    "handoff": "session_note",
    "feedback": "user_assertion",
    "assertion": "user_assertion",
    "user_assertion": "user_assertion",
    # code / config
    "code": "code",
    "python": "code", "rust": "code", "javascript": "code",
    "typescript": "code", "tsx": "code", "go": "code",
    "java": "code", "ruby": "code", "lua": "code", "sql": "code",
    "shell": "code", "bash": "code", "powershell": "code",
    "config": "config", "toml": "config", "yaml": "config",
    "yml": "config", "json": "config", "ini": "config",
    "env": "config", "properties": "config",
    # operational / generated
    "log": "log",
    "tool_output": "tool_output",
    "terminal": "tool_output",
    "console": "tool_output",
    "diff": "tool_output",
    "patch": "tool_output",
    "benchmark": "benchmark",
    "bench": "benchmark",
    "metrics": "benchmark",
    # rich / extracted
    "html": "html",
    "pdf": "pdf",
    "office": "office",
    "spreadsheet": "spreadsheet",
    "image": "image",
    "audio": "audio",
    "video": "video",
    "transcript": "transcript",
    "caption": "transcript",
    "subtitle": "transcript",
    "srt": "transcript",
    "vtt": "transcript",
}

# source_kind -> volatility_class. Matches the half-lives in
# ``context_packet.py::_HALF_LIFE_SECONDS`` (stable=7d, medium=12h,
# hot=15min).
KIND_TO_VOLATILITY: dict[str, str] = {
    "code": "stable",
    "config": "hot",       # configs are the ops-sensitive ones
    "doc": "stable",
    "log": "hot",
    "db": "medium",
    "benchmark": "medium",
    "tool_output": "hot",
    "session_note": "medium",
    "user_assertion": "medium",
    "html": "stable",
    "pdf": "stable",
    "office": "stable",
    "spreadsheet": "stable",
    "image": "stable",
    "audio": "stable",
    "video": "stable",
    "transcript": "medium",
}

KIND_TO_AUTHORITY: dict[str, str] = {
    "code": "primary",
    "config": "primary",
    "doc": "primary",
    "log": "primary",
    "db": "primary",
    "benchmark": "primary",
    "session_note": "primary",
    "user_assertion": "primary",
    "tool_output": "derived",
    # These kinds are usually text extracted from richer source media.
    "html": "derived",
    "pdf": "derived",
    "office": "derived",
    "spreadsheet": "derived",
    "image": "derived",
    "audio": "derived",
    "video": "derived",
    "transcript": "derived",
}

_BENCHMARK_PATH_HINTS = ("benchmark", "benchmarks", "metrics", "results")
_LOG_PATH_HINTS = ("logs", "stdout", "stderr", "trace")


def _normalize_kind_hint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    kind = str(value).strip().lower()
    return CONTENT_TYPE_TO_KIND.get(kind, kind or None)


def _kind_from_path_hints(source_id: Optional[str]) -> Optional[str]:
    """Infer a richer kind than extension alone can express."""
    if not source_id:
        return None
    sid = str(source_id).replace("\\", "/").lower()
    if any(hint in sid for hint in _BENCHMARK_PATH_HINTS):
        return "benchmark"
    if any(hint in sid for hint in _LOG_PATH_HINTS):
        return "log"
    return None


def infer_source_kind(
    source_id: Optional[str],
    content_type: Optional[str] = None,
) -> Optional[str]:
    """Return the inferred source_kind or None if source_id is empty.

    Returns "doc" for unrecognized extensions. Returns None only when
    ``source_id`` is falsy OR looks like a non-path identifier (no
    separator, no extension, e.g. ``"__session__"``, ``"agent:laude"``).
    """
    hinted_kind = _normalize_kind_hint(content_type)
    if not source_id:
        return hinted_kind

    sid = str(source_id)
    if "/" not in sid and "\\" not in sid:
        return hinted_kind

    path_kind = _kind_from_path_hints(sid)
    if path_kind:
        return path_kind

    path = sid.split("?", 1)[0].split("#", 1)[0]
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    ext_kind = EXT_TO_KIND.get(suffix)

    if hinted_kind and hinted_kind not in {"doc"}:
        return hinted_kind
    return ext_kind or hinted_kind or "doc"


def infer_volatility(source_kind: Optional[str]) -> str:
    """Return the volatility_class for the given source_kind.

    Defaults to "medium" for unknown kinds so unmapped sources don't
    accidentally inherit the 15-minute hot TTL.
    """
    if not source_kind:
        return "medium"
    return KIND_TO_VOLATILITY.get(source_kind, "medium")


def infer_authority(source_kind: Optional[str]) -> str:
    """Return a conservative authority default for the given source kind."""
    if not source_kind:
        return "primary"
    return KIND_TO_AUTHORITY.get(source_kind, "primary")


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_support_span(
    *,
    metadata: Optional[dict[str, Any]] = None,
    sequence_index: Optional[int] = None,
    total_strands: Optional[int] = None,
    is_fragment: bool = False,
) -> Optional[str]:
    """Infer a human-readable support span for chunk-backed evidence."""
    md = metadata or {}

    explicit = md.get("support_span")
    if explicit:
        return str(explicit)

    line_start = md.get("line_start")
    line_end = md.get("line_end")
    if line_start is not None and line_end is not None:
        return f"lines:{line_start}-{line_end}"

    page = md.get("page")
    if page is not None:
        return f"page:{page}"
    page_start = md.get("page_start")
    page_end = md.get("page_end")
    if page_start is not None and page_end is not None:
        return f"pages:{page_start}-{page_end}"

    char_start = md.get("char_start")
    char_end = md.get("char_end")
    if char_start is not None and char_end is not None:
        return f"chars:{char_start}-{char_end}"

    byte_start = md.get("byte_start")
    byte_end = md.get("byte_end")
    if byte_start is not None and byte_end is not None:
        return f"bytes:{byte_start}-{byte_end}"

    start_ms = md.get("start_ms")
    end_ms = md.get("end_ms")
    if start_ms is not None and end_ms is not None:
        return f"time_ms:{start_ms}-{end_ms}"

    start_s = md.get("start_s")
    end_s = md.get("end_s")
    if start_s is not None and end_s is not None:
        return f"time_s:{start_s}-{end_s}"

    start_ts = md.get("start_ts")
    end_ts = md.get("end_ts")
    if start_ts is not None and end_ts is not None:
        return f"time:{start_ts}-{end_ts}"

    if sequence_index is None or sequence_index < 0:
        return None
    if total_strands is not None and total_strands > 1:
        span = f"chunk:{sequence_index + 1}/{total_strands}"
        if is_fragment:
            span += ":fragment"
        return span
    if is_fragment:
        return f"chunk:{sequence_index + 1}:fragment"
    return None


def apply_metadata_hints(
    gene,
    metadata: Optional[dict[str, Any]] = None,
    *,
    content_type: Optional[str] = None,
    total_strands: Optional[int] = None,
) -> None:
    """Copy ingest metadata into the gene fields the packet builder uses."""
    md = metadata or {}

    if getattr(gene, "repo_root", None) is None and md.get("repo_root"):
        gene.repo_root = str(md["repo_root"])

    if getattr(gene, "mtime", None) is None:
        gene.mtime = _coerce_float(md.get("mtime"))

    if getattr(gene, "content_hash", None) is None and md.get("content_hash"):
        gene.content_hash = str(md["content_hash"])

    if getattr(gene, "observed_at", None) is None:
        gene.observed_at = _coerce_float(md.get("observed_at"))

    if getattr(gene, "last_verified_at", None) is None:
        gene.last_verified_at = _coerce_float(md.get("last_verified_at"))

    if getattr(gene, "source_kind", None) is None:
        hinted_kind = _normalize_kind_hint(md.get("source_kind")) or _normalize_kind_hint(content_type)
        if hinted_kind:
            gene.source_kind = hinted_kind

    if getattr(gene, "volatility_class", None) is None and md.get("volatility_class"):
        gene.volatility_class = str(md["volatility_class"]).strip().lower()

    if getattr(gene, "authority_class", None) is None and md.get("authority_class"):
        gene.authority_class = str(md["authority_class"]).strip().lower()

    if getattr(gene, "support_span", None) is None:
        support_span = infer_support_span(
            metadata=md,
            sequence_index=getattr(getattr(gene, "promoter", None), "sequence_index", None),
            total_strands=total_strands,
            is_fragment=bool(getattr(gene, "is_fragment", False)),
        )
        if support_span:
            gene.support_span = support_span


def apply_provenance(
    gene,
    source_path: Optional[str] = None,
    observed_at: Optional[float] = None,
    content_type: Optional[str] = None,
) -> None:
    """Populate missing provenance fields on a Gene in-place.

    Only writes fields that are currently None - never clobbers
    caller-supplied values. Safe to call unconditionally in the ingest
    path; a no-op for genes without a resolvable source_path.
    """
    sid = source_path or getattr(gene, "source_id", None)
    if not sid and not content_type:
        return

    now_ts = (
        observed_at
        if observed_at is not None
        else getattr(gene, "observed_at", None)
    )
    if now_ts is None:
        now_ts = time.time()

    if getattr(gene, "source_kind", None) is None:
        kind = infer_source_kind(sid, content_type=content_type)
        if kind is not None:
            gene.source_kind = kind

    if getattr(gene, "volatility_class", None) is None:
        gene.volatility_class = infer_volatility(getattr(gene, "source_kind", None))

    if getattr(gene, "authority_class", None) is None:
        gene.authority_class = infer_authority(getattr(gene, "source_kind", None))

    if getattr(gene, "observed_at", None) is None:
        gene.observed_at = now_ts

    if getattr(gene, "last_verified_at", None) is None:
        gene.last_verified_at = now_ts

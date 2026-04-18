"""Provenance inference for the agent-context packet builder.

Phase 1 of ``docs/specs/2026-04-17-agent-context-index-build-spec.md``
requires ingest to populate ``source_kind``, ``volatility_class``, and
``last_verified_at`` on genes so the packet builder can answer freshness
questions. Without these fields, every packet item degrades to
``stale_risk`` / ``needs_refresh`` because ``freshness_known=False``.

This module centralizes the inference rules so that:

- ``scripts/backfill_gene_provenance.py`` (retroactive sweep) and
- ``HelixContextManager.ingest`` (steady-state population)

share the same logic. Changing an extension → kind mapping here flows
through both paths without drift.

See also:
    - ``scripts/backfill_gene_provenance.py`` — initial sweep
    - ``helix_context/schemas.py::Gene`` — target fields
    - ``helix_context/context_packet.py`` — consumer
"""

from __future__ import annotations

import time
from pathlib import PurePosixPath
from typing import Optional


# Extension → source_kind. Unknown extensions fall through to "doc"
# because doc has the stable (7d) half-life — safest retrieval default.
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
    # doc
    ".md": "doc", ".mdx": "doc", ".rst": "doc", ".adoc": "doc",
    ".txt": "doc", ".tex": "doc",
    ".ipynb": "doc",
    # log
    ".log": "log", ".out": "log",
    # db
    ".db": "db", ".sqlite": "db", ".sqlite3": "db",
    # tabular
    ".csv": "db", ".tsv": "db", ".parquet": "db", ".arrow": "db",
}

# source_kind → volatility_class. Matches the half-lives in
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
}


def infer_source_kind(source_id: Optional[str]) -> Optional[str]:
    """Return the inferred source_kind or None if source_id is empty.

    Returns "doc" for unrecognized extensions. Returns None only when
    ``source_id`` is falsy OR looks like a non-path identifier (no
    separator, no extension, e.g. ``"__session__"``, ``"agent:laude"``).
    """
    if not source_id:
        return None
    sid = str(source_id)
    # Non-path identifiers (session, free-form): leave provenance alone.
    # Mirrors the heuristic in scripts/tombstone_orphan_parents.py.
    if "/" not in sid and "\\" not in sid:
        return None
    path = sid.split("?", 1)[0].split("#", 1)[0]
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    return EXT_TO_KIND.get(suffix, "doc")


def infer_volatility(source_kind: Optional[str]) -> str:
    """Return the volatility_class for the given source_kind.

    Defaults to "medium" for unknown kinds so unmapped sources don't
    accidentally inherit the 15-minute hot TTL.
    """
    if not source_kind:
        return "medium"
    return KIND_TO_VOLATILITY.get(source_kind, "medium")


def apply_provenance(
    gene,
    source_path: Optional[str] = None,
    observed_at: Optional[float] = None,
) -> None:
    """Populate missing provenance fields on a Gene in-place.

    Only writes fields that are currently None — never clobbers
    caller-supplied values. Safe to call unconditionally in the ingest
    path; a no-op for genes without a resolvable source_path.

    Parameters
    ----------
    gene : Gene
        The gene to populate. Must have ``source_kind``,
        ``volatility_class``, ``last_verified_at``, ``observed_at`` as
        writable attributes (added in commit f10fc8a).
    source_path : str, optional
        Override the gene's existing ``source_id`` for inference — use
        this when the caller has the path in scope but hasn't yet
        assigned ``gene.source_id``. Defaults to reading
        ``gene.source_id``.
    observed_at : float, optional
        Wall-clock time of ingestion. Defaults to ``time.time()``.
        Flows into ``observed_at`` and ``last_verified_at`` when those
        fields are still None.
    """
    sid = source_path or getattr(gene, "source_id", None)
    if not sid:
        return

    now_ts = observed_at if observed_at is not None else time.time()

    if getattr(gene, "source_kind", None) is None:
        kind = infer_source_kind(sid)
        if kind is not None:
            gene.source_kind = kind

    if getattr(gene, "volatility_class", None) is None:
        gene.volatility_class = infer_volatility(getattr(gene, "source_kind", None))

    if getattr(gene, "observed_at", None) is None:
        gene.observed_at = now_ts

    if getattr(gene, "last_verified_at", None) is None:
        gene.last_verified_at = now_ts

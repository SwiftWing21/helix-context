"""Stage 7 freshness gate — per-document mtime revalidation + supersession.

Spec: docs/specs/2026-05-08-stage-7-freshness-gate.md §5, §7.

Three pure-ish functions plus an in-memory mtime cache. None of them
touch an LLM or the compressor — Stage 7 is LLM-free by design. The
``_revalidate_*`` pair compares on-disk mtime against
``gene.last_verified_at`` to decide whether the underlying source has
moved since the knowledge store last vouched for it; ``check_superseded`` walks
the ``genes.supersedes`` reverse pointer (Path A only — Path B
``claim_edges`` chain walking is explicitly out-of-scope per spec §15).

Read-only contract (Stage 1 boundary, spec §5):
  * ``revalidate_source`` may populate the in-memory mtime cache under
    ``read_only=True`` — the cache is per-process, not a knowledge store write.
  * ``revalidate_and_mark`` calls ``genome.mark_verified`` only when
    ``read_only=False``; under ``read_only=True`` the column is left
    alone and the next non-read-only call re-stats (cache-served if
    within TTL) and writes through.

Cache shape: ``dict[str, tuple[float, float]]`` keyed on absolute
source path, value is ``(mtime, cached_at)``. TTL defaults to 60s; insertion
caps the cache at 4096 entries, reclaiming expired paths first. The
cache lives on ``CymatixContextManager`` (per-batch state, not KnowledgeStore) so
admin /admin/refresh can clear it without touching the DB.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .schemas import Gene

log = logging.getLogger("cymatix.freshness")


# Default cache TTL. Stat is microsecond-cheap warm but we still skip
# it when a recent stat is on hand — keeps tight loops (e.g. the
# bench's planted-stale needles) from re-stat'ing the same path.
DEFAULT_CACHE_TTL_S: float = 60.0
MTIME_CACHE_MAX_ENTRIES: int = 4096
# Serialize eviction + insertion; filesystem calls stay outside this lock.
_mtime_cache_lock = threading.Lock()


def _evict_if_full(
    mtime_cache: dict[str, tuple[float, float]], now_ts: float, cache_ttl_s: float,
) -> None:
    """Prefer expired entries, then clear a full cache of still-valid paths."""
    if len(mtime_cache) < MTIME_CACHE_MAX_ENTRIES:
        return
    # Snapshot so an admin refresh clearing this dict cannot disrupt iteration.
    for path, (_, cached_at) in list(mtime_cache.items()):
        if now_ts - cached_at >= cache_ttl_s:
            mtime_cache.pop(path, None)
    if len(mtime_cache) >= MTIME_CACHE_MAX_ENTRIES:
        mtime_cache.clear()


def _cache_mtime(
    mtime_cache: dict[str, tuple[float, float]], source_path: str,
    mtime: float, now_ts: float, cache_ttl_s: float,
) -> None:
    with _mtime_cache_lock:
        _evict_if_full(mtime_cache, now_ts, cache_ttl_s)
        mtime_cache[source_path] = (mtime, now_ts)


# Status vocabulary returned by ``revalidate_source``. Names are short
# on purpose so callers can match-case against them directly.
FreshnessStatus = Literal["fresh", "stale", "missing", "unknown"]


# ─────────────────────────────────────────────────────────────────────
# URL detection — anything with a `://` scheme delegates to /consolidate
# ─────────────────────────────────────────────────────────────────────

def _looks_like_url(source_path: str) -> bool:
    """Return True if ``source_path`` is a URL (scheme://...).

    The freshness MVP only handles file paths. URLs are routed to
    ``/consolidate`` per spec §5 + §15 — re-implementing HTTP HEAD in
    the read flow would violate Stage 1's read_only contract and is
    out-of-scope.
    """
    if not source_path:
        return False
    # `://` is the cheap discriminator; covers http(s), file, ftp,
    # git+ssh, plus any future scheme without us having to enumerate
    # them. Windows drive letters (``C:\...``) don't trip this.
    return "://" in source_path[:32]


# ─────────────────────────────────────────────────────────────────────
# Per-document revalidation
# ─────────────────────────────────────────────────────────────────────

def revalidate_source(
    gene: "Gene",
    *,
    mtime_cache: dict[str, tuple[float, float]],
    now_ts: float,
    cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
) -> FreshnessStatus:
    """Compare on-disk mtime to ``gene.last_verified_at``.

    Decision matrix (spec §5):
      source_path is None / empty                  -> "unknown"
      source_path is a URL                         -> "unknown"  (delegated to /consolidate)
      file does not exist                          -> "missing"
      gene.last_verified_at is None                -> "unknown"  (legacy row)
      mtime <= last_verified_at                    -> "fresh"
      mtime  > last_verified_at                    -> "stale"

    Cache:
      key   = source_path (the document's file path)
      value = (mtime, cached_at)
      Skip ``os.stat`` when ``now_ts - cached_at < cache_ttl_s``.

    Args:
      document: the Document whose source will be stat'd.
      mtime_cache: shared dict; this function MAY mutate it (writing
        the cache is not a knowledge store mutation, so it is allowed under
        read_only=True per spec §5).
      now_ts: caller-provided "now" — passed in (rather than calling
        ``time.time()`` here) so tests can simulate TTL expiry without
        sleeping.
      cache_ttl_s: skip-stat horizon. Defaults to 60s.

    Returns:
      One of "fresh" | "stale" | "missing" | "unknown".
    """
    source_path = getattr(gene, "source_id", None) or getattr(
        gene, "source_path", None
    )
    if not source_path:
        return "unknown"
    if _looks_like_url(source_path):
        # URL flow delegated to /consolidate — out-of-scope per §15.
        return "unknown"

    # Cache lookup. We trust the cached mtime within the TTL window;
    # outside it, re-stat and write back.
    cached = mtime_cache.get(source_path)
    if cached is not None and (now_ts - cached[1]) < cache_ttl_s:
        mtime = cached[0]
        if mtime < 0:
            # Sentinel: cache previously recorded a missing file.
            return "missing"
    else:
        try:
            st = os.stat(source_path)
            mtime = float(st.st_mtime)
        except FileNotFoundError:
            # Persist the negative sentinel so we don't re-stat the
            # missing file on every revalidation in the same TTL window.
            _cache_mtime(mtime_cache, source_path, -1.0, now_ts, cache_ttl_s)
            return "missing"
        except OSError as exc:
            # Permission error / path-too-long / other transient stat
            # failure. Treat as "unknown" — the freshness gate should
            # not down-rank a document because the OS hiccuped on stat.
            log.warning(
                "revalidate_source: os.stat(%s) failed: %s",
                source_path,
                exc,
            )
            return "unknown"
        _cache_mtime(mtime_cache, source_path, mtime, now_ts, cache_ttl_s)

    last_verified = getattr(gene, "last_verified_at", None)
    if last_verified is None:
        # Legacy row — column predates the writer wiring. Treat as
        # neutral so a strong score gap can still emit KnowBlock; β5
        # in know_calibration applies a confidence haircut.
        return "unknown"

    return "fresh" if mtime <= float(last_verified) else "stale"


def revalidate_and_mark(
    genome,
    gene: "Gene",
    *,
    mtime_cache: dict[str, tuple[float, float]],
    now_ts: float,
    read_only: bool,
    cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
) -> FreshnessStatus:
    """Run ``revalidate_source`` then mark verified when allowed.

    Read-only contract (spec §5):
      * Cache is updated on every call (in-memory, per-process — not a
        knowledge store mutation).
      * ``genome.mark_verified`` is called only when status == "fresh"
        AND ``read_only=False``. Under read_only=True the column is left
        unchanged; the next non-read-only call re-stats (or hits the
        cache) and writes through.

    Returns the same ``FreshnessStatus`` the underlying call produced.
    """
    status = revalidate_source(
        gene,
        mtime_cache=mtime_cache,
        now_ts=now_ts,
        cache_ttl_s=cache_ttl_s,
    )
    if status == "fresh" and not read_only:
        try:
            genome.mark_verified(
                [gene.gene_id],
                now_ts,
                read_only=False,
            )
        except Exception:
            # Verification timestamp is a hint, not a correctness
            # invariant. Don't bubble up writer failures into the
            # retrieval flow.
            log.warning(
                "revalidate_and_mark: mark_verified failed for %s",
                gene.gene_id,
                exc_info=True,
            )
    if status != "fresh":
        # Roadmap §3b-7: demotion-relevant freshness verdicts, visible in
        # Grafana. "fresh" is the common case and is not emitted (volume).
        try:
            from ..telemetry import freshness_demotion_counter
            freshness_demotion_counter().add(1, {"status": str(status)})
        except Exception:  # pragma: no cover
            pass
    return status


# ─────────────────────────────────────────────────────────────────────
# Supersession (Path A — single-pointer reverse lookup)
# ─────────────────────────────────────────────────────────────────────

def check_superseded(genome, gene: "Gene") -> Optional[str]:
    """Return the successor's source_id when ``gene`` has been replaced.

    Path A only (spec §7, §15): single SELECT on the reverse-index of
    ``genes.supersedes``. If a row exists where ``supersedes = gene.gene_id``,
    that row's document is the successor and its ``source_id`` becomes the
    refresh target.

    Path B (claim-chain walking via ``claim_edges``) is explicitly out
    of scope. Stage 7+1 will pick it up if benches surface document rows
    whose ``supersedes IS NULL`` but a claim chain exists.

    Returns:
      The successor document's ``source_id`` (a string), or ``None`` when
      no successor row exists or the successor has no source_id.
    """
    gene_id = getattr(gene, "gene_id", None)
    if not gene_id:
        return None
    try:
        # Use the read connection — supersession lookup must be safe
        # under the read_only contract.
        conn = getattr(genome, "read_conn", None) or genome.conn
        cur = conn.cursor()
        row = cur.execute(
            "SELECT gene_id, source_id FROM genes "
            "WHERE supersedes = ? LIMIT 1",
            (gene_id,),
        ).fetchone()
    except Exception:
        log.warning(
            "check_superseded: SELECT failed for gene_id=%s",
            gene_id,
            exc_info=True,
        )
        return None

    if row is None:
        return None
    # Row may be a sqlite3.Row (mapping-like) or a tuple — handle both.
    try:
        successor_source_id = row["source_id"]
    except (KeyError, IndexError, TypeError):
        successor_source_id = row[1] if len(row) > 1 else None

    if not successor_source_id:
        return None
    try:
        from ..telemetry import freshness_demotion_counter
        freshness_demotion_counter().add(1, {"status": "superseded"})
    except Exception:  # pragma: no cover
        pass
    return str(successor_source_id)


__all__ = [
    "DEFAULT_CACHE_TTL_S",
    "MTIME_CACHE_MAX_ENTRIES",
    "FreshnessStatus",
    "revalidate_source",
    "revalidate_and_mark",
    "check_superseded",
    "_looks_like_url",
]

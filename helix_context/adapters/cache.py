"""DAL cache — TTL-bounded LRU respecting Helix volatility classes.

Wraps a ``DAL`` instance. Same API (``fetch(source_id)``), same
``FetchResult`` shape, plus freshness-aware eviction and an explicit
multi-agent deployment model.

## Multi-agent semantics (important)

The cache is keyed on ``source_id`` — NOT on agent or session. The
bytes at a path are identity-independent: if Laude and Taude both ask
for ``/repo/config.yaml`` from the same machine, they get the same
content, and sharing the cache is correct.

Deployment model:

- **Per party (= per device).** One cache per launcher instance.
  Different machines have different filesystems and should have
  separate caches. The default in-process LRU naturally gives you
  this — no cross-party coordination needed.
- **NOT agent-scoped.** We don't partition by ``HELIX_AGENT``.
  Partitioning would defeat the cache's purpose (Laude fetches, Taude
  refetches the same file 10 seconds later).
- **NOT cross-party.** Don't share caches across devices. Use a
  shared genome + ingest instead — Helix syncs metadata, not bytes.

TTLs come from Helix's ``volatility_class`` (``stable=7d, medium=12h,
hot=15min``) or an explicit ``ttl_s`` argument at fetch time.
Invalidation is explicit via ``invalidate(source_id)`` or
``invalidate_all()``; wire those to your ingest hooks if your corpus
updates faster than the TTL.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from threading import RLock
from typing import Optional

from .dal import DAL, FetchResult

log = logging.getLogger("helix.adapters.cache")


# Default TTLs per volatility_class (seconds). Mirror the values in
# helix_context/context_packet.py's freshness scoring.
DEFAULT_TTLS_S = {
    "stable": 7 * 24 * 3600,   # 7 days
    "medium": 12 * 3600,        # 12 hours
    "hot":    15 * 60,          # 15 minutes
}

# When volatility_class is unknown, fall back to "medium" — err on
# shorter-TTL side rather than serving stale bytes.
_DEFAULT_VOLATILITY = "medium"


class CachedDAL:
    """TTL-bounded LRU cache wrapping a DAL instance.

    Basic usage::

        from helix_context.adapters.dal import DAL
        from helix_context.adapters.cache import CachedDAL

        cache = CachedDAL(DAL(), max_entries=500)
        result = cache.fetch("/repo/config.yaml", volatility_class="hot")

    The wrapper preserves ``FetchResult`` shape. ``result.meta`` gains
    two extra fields:

    - ``cache_hit``: bool
    - ``cached_at``: Unix timestamp of the original fetch
    """

    def __init__(
        self,
        dal: Optional[DAL] = None,
        *,
        max_entries: int = 500,
        default_ttl_s: Optional[float] = None,
    ) -> None:
        self._dal = dal or DAL()
        self._max = max(1, int(max_entries))
        self._default_ttl_s = default_ttl_s

        self._store: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._lock = RLock()

        # Stats — cheap introspection for tests + multi-agent diagnostics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ── Fetch ──────────────────────────────────────────────────────

    def fetch(
        self,
        source_id: str,
        *,
        volatility_class: Optional[str] = None,
        ttl_s: Optional[float] = None,
        bypass_cache: bool = False,
        **kwargs,
    ) -> FetchResult:
        """Fetch through the cache.

        Parameters
        ----------
        source_id: as in DAL.fetch
        volatility_class: Helix's volatility label. When provided (and
            ``ttl_s`` is None), the TTL comes from ``DEFAULT_TTLS_S``.
        ttl_s: explicit TTL override. Wins over ``volatility_class``.
        bypass_cache: force a miss; still writes the result back.
        """
        ttl = self._resolve_ttl(ttl_s, volatility_class)
        now = time.time()

        if not bypass_cache:
            with self._lock:
                entry = self._store.get(source_id)
                if entry is not None:
                    if now - entry.cached_at <= entry.ttl_s:
                        # LRU — move to end
                        self._store.move_to_end(source_id)
                        self._hits += 1
                        meta = dict(entry.result.meta)
                        meta["cache_hit"] = True
                        meta["cached_at"] = entry.cached_at
                        meta["age_s"] = round(now - entry.cached_at, 2)
                        return FetchResult(entry.result.text, meta)
                    else:
                        # Stale — drop and fall through to refetch
                        del self._store[source_id]

        # Miss or bypass — delegate to underlying DAL
        result = self._dal.fetch(source_id, **kwargs)
        if not bypass_cache:
            self._misses += 1
        # Only cache successful fetches; errors shouldn't poison.
        if result.ok:
            self._put(source_id, result, ttl, now)
        meta = dict(result.meta)
        meta["cache_hit"] = False
        meta["cached_at"] = now
        return FetchResult(result.text, meta)

    def _put(self, source_id: str, result: FetchResult,
             ttl_s: float, now: float) -> None:
        with self._lock:
            if source_id in self._store:
                del self._store[source_id]
            self._store[source_id] = _CacheEntry(
                result=result, cached_at=now, ttl_s=ttl_s,
            )
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # LRU evict oldest
                self._evictions += 1

    def _resolve_ttl(
        self,
        ttl_s: Optional[float],
        volatility_class: Optional[str],
    ) -> float:
        if ttl_s is not None:
            return max(0.0, float(ttl_s))
        if volatility_class is not None:
            return float(DEFAULT_TTLS_S.get(
                volatility_class.lower(),
                DEFAULT_TTLS_S[_DEFAULT_VOLATILITY],
            ))
        if self._default_ttl_s is not None:
            return float(self._default_ttl_s)
        return float(DEFAULT_TTLS_S[_DEFAULT_VOLATILITY])

    # ── Invalidation ───────────────────────────────────────────────

    def invalidate(self, source_id: str) -> bool:
        """Drop a single source_id. Returns True if it was cached."""
        with self._lock:
            return self._store.pop(source_id, None) is not None

    def invalidate_all(self) -> int:
        """Drop everything. Returns the number of entries evicted."""
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Drop every source_id starting with ``prefix``.

        Handy for ingest hooks that know a subtree changed (e.g.,
        rebuilt ``/repo/docs/`` → ``invalidate_by_prefix("/repo/docs/")``).
        """
        with self._lock:
            victims = [k for k in self._store if k.startswith(prefix)]
            for k in victims:
                del self._store[k]
            return len(victims)

    # ── Introspection ──────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._store),
                "max_entries": self._max,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses) else 0.0
                ),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0


class _CacheEntry:
    __slots__ = ("result", "cached_at", "ttl_s")

    def __init__(self, result: FetchResult, cached_at: float,
                 ttl_s: float) -> None:
        self.result = result
        self.cached_at = cached_at
        self.ttl_s = ttl_s


# ── Packet-aware batch fetch with cache ─────────────────────────────


def fetch_packet_sources_cached(
    packet: dict,
    cache: Optional[CachedDAL] = None,
    *,
    buckets: tuple[str, ...] = ("verified", "stale_risk", "contradictions"),
    include_refresh_targets: bool = True,
    max_sources: int = 12,
) -> list[tuple[str, FetchResult]]:
    """Like ``fetch_packet_sources`` but the cache honors each item's
    ``volatility_class`` for TTL selection.

    Items in ``stale_risk`` or ``refresh_targets`` bypass the cache
    automatically — Helix already flagged them as needing a fresh
    fetch, and serving stale bytes would defeat the verdict.
    """
    cache = cache or CachedDAL()

    # Build per-source (volatility, bypass) decisions from the packet
    decisions: "OrderedDict[str, tuple[Optional[str], bool]]" = OrderedDict()

    def _add(source_id: Optional[str], volatility: Optional[str],
             bypass: bool) -> None:
        if not source_id:
            return
        if source_id in decisions:
            # First-listed bucket's volatility wins, but bypass sticks
            existing_vol, existing_bypass = decisions[source_id]
            decisions[source_id] = (existing_vol, existing_bypass or bypass)
        else:
            decisions[source_id] = (volatility, bypass)

    # verified first (cacheable by default)
    for item in packet.get("verified", []) or []:
        _add(item.get("source_id"), item.get("volatility_class"), False)
    # stale_risk → bypass cache
    if "stale_risk" in buckets:
        for item in packet.get("stale_risk", []) or []:
            _add(item.get("source_id"), item.get("volatility_class"), True)
    if "contradictions" in buckets:
        for item in packet.get("contradictions", []) or []:
            _add(item.get("source_id"), item.get("volatility_class"), False)
    if include_refresh_targets:
        for tgt in packet.get("refresh_targets", []) or []:
            # Refresh targets always bypass — that's literally what they are
            _add(tgt.get("source_id"), None, True)

    results: list[tuple[str, FetchResult]] = []
    for source_id, (volatility, bypass) in list(decisions.items())[:max_sources]:
        result = cache.fetch(
            source_id,
            volatility_class=volatility,
            bypass_cache=bypass,
        )
        results.append((source_id, result))
    return results

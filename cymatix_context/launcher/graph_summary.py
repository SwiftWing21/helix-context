"""
Graph summary — a read-only inventory of the graph layers in a genome.

The launcher's Knowledge store tab shows how much of each layer the active
knowledge store actually holds: documents, semantic links, document
structure, learned co-activation and entities. It is a ledger, not a
renderer; nothing here draws a node or an edge.

Two things keep it cheap enough to sit behind a two-second poll:

  * every read opens its own short-lived `mode=ro` connection with
    `PRAGMA query_only=ON`, built with `Path.as_uri()` the way
    `cymatix_context/cli/cmd_status.py` builds it, so a running cymatix
    holding the store for writes is never blocked;
  * results are cached by resolved path for `GRAPH_SUMMARY_TTL_S`, so the
    poll costs a dict lookup and the aggregation runs about twice a
    minute.

Distinct undirected COVER pairs are the one expensive count. sqlite serves
that `GROUP BY` with a temp B-tree over every COVER row and no index can
help a grouping on computed expressions, so the query is skipped above
`COVER_LINKS_MAX_ROWS` and the panel reports stored rows instead.

Nothing in this module raises into the collector: an unreadable store
returns an unavailable payload whose reason goes to the log, never to the
page.

The `.modal-backdrop[hidden]` guard the design spec also mentions already
ships in `launcher.css` and is covered by
`tests/test_launcher_dashboard_wiring.py::test_modal_backdrop_has_hidden_display_guard`.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from cymatix_context.schemas import NLRelation, StructuralRelation

log = logging.getLogger("cymatix.launcher.graph_summary")

# The dashboard polls every two seconds; the aggregation runs at most twice
# a minute per genome.
GRAPH_SUMMARY_TTL_S: float = 30.0

# A launcher pointed at a bench directory can walk several genomes in one
# session. Hold a handful and evict the oldest.
GRAPH_SUMMARY_MAX_PATHS: int = 8

# Distinct undirected COVER pairs need `GROUP BY MIN(a,b), MAX(a,b)`, which
# sqlite serves with a temp B-tree over every COVER row; no index can help a
# GROUP BY on computed expressions. Measured on a dev box, warm cache: 14 ms
# at 28,884 COVER rows, 1,689 ms at 800,000, and 10.1 to 11.9 s at
# 7,224,963. Above the ceiling the panel reports stored rows instead of
# distinct pairs.
COVER_LINKS_MAX_ROWS: int = 500_000

# Matches genome_registry's read probe: a genome held open for writes must
# never make the dashboard wait.
_CONNECT_TIMEOUT_S: float = 1.0

# `gene_relations` is a discriminated union — SYMBOL_REF shares the table
# with CHUNK_OF — so every count binds its relation code.
_RELATION_ROWS_SQL = "SELECT COUNT(*) FROM gene_relations WHERE relation = ?"

_COVER_CONNECTED_SQL = (
    "SELECT COUNT(*) FROM ("
    "SELECT gene_id_a AS gene_id FROM gene_relations WHERE relation = ? "
    "UNION "
    "SELECT gene_id_b FROM gene_relations WHERE relation = ?"
    ")"
)

# Reciprocal rows (a->b and b->a) are one link, so each row is normalised
# to an ordered pair before the DISTINCT.
_COVER_LINKS_SQL = (
    "SELECT COUNT(*) FROM ("
    "SELECT DISTINCT "
    "CASE WHEN gene_id_a <= gene_id_b THEN gene_id_a ELSE gene_id_b END AS lo, "
    "CASE WHEN gene_id_a <= gene_id_b THEN gene_id_b ELSE gene_id_a END AS hi "
    "FROM gene_relations WHERE relation = ?"
    ")"
)


def _count(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def _read_graph_summary(path: Path) -> Dict[str, Any]:
    """Count every graph layer in the genome at `path`.

    Raises `sqlite3.Error` if the store cannot be opened or is missing one
    of the graph tables. Callers go through `GraphSummaryCache.get`, which
    turns that into the unavailable payload.
    """
    genome = Path(path)
    cover = int(NLRelation.COVER)
    chunk_of = int(StructuralRelation.CHUNK_OF)

    # `Path.as_uri()` rather than an f-string, the idiom already in
    # `cymatix_context/cli/cmd_status.py`: it percent-encodes, so a genome
    # whose path contains `#`, `?` or `%` cannot re-parse the URI, drop
    # `?mode=ro` and leave sqlite opening some other path read-write.
    conn = sqlite3.connect(
        genome.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=_CONNECT_TIMEOUT_S,
    )
    try:
        conn.execute("PRAGMA query_only=ON")

        total_genes = _count(conn, "SELECT COUNT(*) FROM genes")
        cover_rows = _count(conn, _RELATION_ROWS_SQL, (cover,))
        cover_connected_genes = _count(conn, _COVER_CONNECTED_SQL, (cover, cover))

        cover_links: Optional[int] = None
        if cover_rows <= COVER_LINKS_MAX_ROWS:
            cover_links = _count(conn, _COVER_LINKS_SQL, (cover,))

        chunk_of_rows = _count(conn, _RELATION_ROWS_SQL, (chunk_of,))
        harmonic_links = _count(conn, "SELECT COUNT(*) FROM harmonic_links")
        entities = _count(conn, "SELECT COUNT(DISTINCT entity) FROM entity_graph")
        entity_memberships = _count(conn, "SELECT COUNT(*) FROM entity_graph")
    finally:
        conn.close()

    return {
        "available": True,
        "path": str(genome),
        "total_genes": total_genes,
        "cover_connected_genes": cover_connected_genes,
        "cover_links": cover_links,
        "cover_rows": cover_rows,
        "chunk_of_rows": chunk_of_rows,
        "harmonic_links": harmonic_links,
        "entities": entities,
        "entity_memberships": entity_memberships,
    }


class GraphSummaryCache:
    """Process-local, bounded, TTL cache of graph summaries by path.

    Concurrent misses on the same path share one aggregation. The lock is
    held only for bookkeeping, so one slow genome cannot stall a poll for
    a different one. The TTL starts when the aggregation finishes.
    """

    def __init__(
        self,
        ttl_s: float = GRAPH_SUMMARY_TTL_S,
        max_paths: int = GRAPH_SUMMARY_MAX_PATHS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = ttl_s
        self.max_paths = max_paths
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
        self._inflight: Dict[str, Future] = {}

    def get(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Summary for `path`, from the cache when it is still fresh.

        Returns a copy, so a caller that edits the payload — or a template
        helper that adds a key — cannot poison the next poll.
        """
        genome = Path(path).resolve()
        key = str(genome)
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and self._clock() - hit[0] < self.ttl_s:
                return dict(hit[1])
            pending = self._inflight.get(key)
            refresh = pending is None
            if refresh:
                pending = Future()
                self._inflight[key] = pending

        if not refresh:
            return dict(pending.result())

        try:
            payload = _read_graph_summary(genome)
        except Exception as exc:
            # The reason belongs in the log. The panel says the summary is
            # unavailable and names the file, and stops there.
            log.warning("Graph summary unavailable for %s (%s)", key, exc)
            payload = {"available": False, "path": key, "error": str(exc)}

        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = (self._clock(), payload)
            while len(self._entries) > self.max_paths:
                self._entries.popitem(last=False)
            pending.set_result(payload)
            self._inflight.pop(key)

        return dict(payload)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

"""Path layout, routing helpers, and Genome-shape adapter for the
filesystem-mirroring shard scheme.

The sharded genome layout mirrors the source filesystem so that (a) a shard
filename is self-identifying in backups and (b) a fresh clone can map a
file path back to its owning shard without consulting ``main.genome.db``:

    genomes/
      main.genome.db                          # routing + source_index + registry
      agent/
        laude.genome.db                       # per-handle agent shard
        raude.genome.db
        ...
      <drive>/<mirrored source path>/<label>.genome.db

Categories used in this layout:
  - ``corpus`` — one shard per ingest source root (F:/Projects, F:/Factorio, ...)
  - ``agent``  — one shard per session handle (laude/raude/taude/gemini)

The spec's participant/reference/org categories (see
``docs/specs/2026-04-17-genome-sharding-plan.md``) are orthogonal to this
axis and can be layered in later without moving files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath
from typing import Optional

# Filename characters that Windows + most Unix tools handle cleanly when
# preserved verbatim. Drive letter colons and backslashes are the only
# things that need remapping.
_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/]?")


def drive_prefix(path: str) -> Optional[str]:
    """Extract the drive letter prefix from a Windows path.

    Returns ``"C"``, ``"F"``, etc. for ``"C:/..."`` / ``"F:\\..."`` paths,
    or ``None`` for drive-less / POSIX paths.
    """
    m = _WIN_DRIVE_RE.match(path)
    return m.group(1).upper() if m else None


def strip_drive(path: str) -> str:
    """Return the path with any drive letter prefix removed, using ``/`` separators."""
    no_drive = _WIN_DRIVE_RE.sub("", path)
    return no_drive.replace("\\", "/").lstrip("/")


def corpus_shard_dir(
    source_root: str,
    genomes_root: os.PathLike[str] | str,
) -> Path:
    """Compute the shard directory that mirrors ``source_root`` under ``genomes_root``.

    ``F:/Projects``            -> ``genomes/F/Projects/``
    ``C:/Program Files (x86)/Steam/steamapps/common/Stationeers``
        -> ``genomes/C/Program Files (x86)/Steam/steamapps/common/Stationeers/``
    ``D:/SteamLibrary/steamapps/common/Turing Complete``
        -> ``genomes/D/SteamLibrary/steamapps/common/Turing Complete/``
    ``/home/alice/projects`` (POSIX) -> ``genomes/home/alice/projects/``
    """
    root = Path(genomes_root)
    drive = drive_prefix(source_root)
    rel = strip_drive(source_root)
    if drive is not None:
        return root / drive / rel
    # POSIX or drive-less input: just mirror path segments under genomes_root.
    return root / rel


def corpus_shard_db(
    source_root: str,
    label: str,
    genomes_root: os.PathLike[str] | str,
) -> Path:
    """Return ``<corpus_shard_dir>/<label>.genome.db``."""
    return corpus_shard_dir(source_root, genomes_root) / f"{label}.genome.db"


def agent_shard_db(
    handle: str,
    genomes_root: os.PathLike[str] | str,
) -> Path:
    """Return ``<genomes_root>/agent/<handle>.genome.db``."""
    return Path(genomes_root) / "agent" / f"{handle}.genome.db"


def main_db_path(genomes_root: os.PathLike[str] | str) -> Path:
    """Return ``<genomes_root>/main.genome.db`` — the routing + registry DB."""
    return Path(genomes_root) / "main.genome.db"


class IngestTargetRouter:
    """Maps an individual ingested file to its target shard DB.

    Given a set of registered ``(source_root, shard_db_path)`` pairs, the
    router returns the longest-prefix-matching shard DB for any file path.
    Used by ``scripts/ingest_all.py`` to decide where each gene is written.
    """

    def __init__(self) -> None:
        # List of (normalized_root, shard_db_path) — sorted longest first
        # so that nested sources (e.g., F:/Projects/helix-context inside
        # F:/Projects) would route to the more specific shard.
        self._registered: list[tuple[str, Path]] = []

    @staticmethod
    def _normalize(path: str) -> str:
        return os.path.normpath(path).replace("\\", "/").rstrip("/").lower()

    def register(self, source_root: str, shard_db: Path) -> None:
        norm = self._normalize(source_root)
        # Replace if already present; otherwise insert sorted.
        self._registered = [
            (r, p) for (r, p) in self._registered if r != norm
        ]
        self._registered.append((norm, shard_db))
        self._registered.sort(key=lambda rp: len(rp[0]), reverse=True)

    def resolve(self, file_path: str) -> Optional[Path]:
        """Return the shard DB path that owns ``file_path``, or ``None``.

        Longest matching registered root wins. File path need not exist.
        """
        norm = self._normalize(file_path)
        for root, shard_db in self._registered:
            if norm == root or norm.startswith(root + "/"):
                return shard_db
        return None

    def __len__(self) -> int:
        return len(self._registered)

    def __iter__(self):
        return iter(self._registered)


# ── Read-only Genome-shape adapter ────────────────────────────────────


import logging
from typing import Any, List

log = logging.getLogger(__name__)


class ShardedGenomeAdapter:
    """Present a ``ShardRouter`` with the subset of the ``Genome`` API that
    ``HelixContextManager`` uses on the read path.

    Writes are logged no-ops: the adapter is intended for read-heavy serving
    (benchmarks, agent retrieval) until ingest-time sharding (spec Task 6)
    lands. Enable via the ``HELIX_USE_SHARDS=1`` env flag — the factory
    ``open_read_source`` below wires this in for callers.

    Limitations (V1):
      - ``upsert_gene`` / ``store_*`` / ``touch_*`` etc. are silent no-ops;
        anything that needs to write hot state (replication, session delivery)
        will not persist.
      - ``query_cold_tier`` returns empty; cold-tier fan-out across shards
        is deferred.
      - ``conn`` returns the main-routing connection. Callers that expect a
        full genome schema (session_delivery tables, genes table) will see
        SQLite ``no such table`` errors; guard those paths with
        ``hasattr(genome, '_sharded_adapter')``.
    """

    def __init__(self, main_path: str, **genome_kwargs: Any) -> None:
        from .shard_router import ShardRouter

        self._router = ShardRouter(main_path=main_path, **genome_kwargs)
        self.last_query_scores: dict = {}
        self.last_tier_contributions: dict = {}
        self._sharded_adapter = True  # sentinel for callers to detect

    # ── Reads ─────────────────────────────────────────────────────────

    def query_genes(self, *args: Any, **kwargs: Any) -> List:
        genes = self._router.query_genes(*args, **kwargs)
        self.last_query_scores = dict(self._router.last_query_scores)
        self.last_tier_contributions = dict(self._router.last_tier_contributions)
        return genes

    def query_cold_tier(self, *args: Any, **kwargs: Any) -> list:
        """Cold-tier queries aren't fanned out in V1; return empty."""
        return []

    def stats(self) -> dict:
        rows = self._router.main_conn.execute(
            "SELECT category, shard_name, gene_count, byte_size "
            "FROM shards WHERE health='ok'"
        ).fetchall()
        total = sum((r["gene_count"] or 0) for r in rows)
        by_cat: dict = {}
        for r in rows:
            by_cat.setdefault(r["category"], 0)
            by_cat[r["category"]] += r["gene_count"] or 0
        return {
            "total_genes": total,
            "by_category": by_cat,
            "shards": [
                {"name": r["shard_name"], "category": r["category"],
                 "genes": r["gene_count"], "bytes": r["byte_size"]}
                for r in rows
            ],
            "compression_ratio": None,
        }

    def health_summary(self) -> dict:
        rows = self._router.main_conn.execute(
            "SELECT shard_name, category, gene_count, health FROM shards"
        ).fetchall()
        return {"status": "ok", "shards": [dict(r) for r in rows]}

    @property
    def conn(self):
        """Direct SQL callers see the main routing DB.

        Queries against per-shard tables (``genes``, session-delivery) will
        raise ``sqlite3.OperationalError: no such table``; callers in
        sharded mode should guard or feature-flag those paths.
        """
        return self._router.main_conn

    # ── Write surface (no-ops in V1) ──────────────────────────────────

    def upsert_gene(self, gene, **_kw) -> str:
        log.debug("sharded-adapter: upsert_gene no-op for %s", getattr(gene, "gene_id", "?"))
        return getattr(gene, "gene_id", "")

    def touch_genes(self, *_a, **_kw) -> None: pass
    def link_coactivated(self, *_a, **_kw) -> None: pass
    def store_harmonic_weights(self, *_a, **_kw) -> None: pass
    def store_relations_batch(self, *_a, **_kw) -> None: pass
    def log_health(self, *_a, **_kw) -> None: pass
    def compress_to_heterochromatin(self, *_a, **_kw) -> None: pass
    def compress_to_euchromatin(self, *_a, **_kw) -> None: pass
    def compact(self, *_a, **_kw) -> None: pass
    def refresh(self) -> None: pass
    def set_replication_manager(self, *_a, **_kw) -> None: pass

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        self._router.close()


def open_read_source(genome_path: str, **genome_kwargs: Any):
    """Factory: return a ``Genome`` or a ``ShardedGenomeAdapter``.

    If ``HELIX_USE_SHARDS=1`` and ``genome_path`` ends with
    ``main.genome.db``, open a ``ShardedGenomeAdapter``. Otherwise open
    a regular ``Genome`` at ``genome_path``.
    """
    from .genome import Genome
    from .shard_router import use_shards_enabled

    is_routing_db = os.path.basename(genome_path) == "main.genome.db"
    if use_shards_enabled() and is_routing_db:
        log.info("HELIX_USE_SHARDS=1 and path is routing DB — opening ShardedGenomeAdapter")
        return ShardedGenomeAdapter(main_path=genome_path, **genome_kwargs)
    return Genome(path=genome_path, **genome_kwargs)

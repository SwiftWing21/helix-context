"""Path layout and routing helpers for the filesystem-mirroring shard scheme.

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

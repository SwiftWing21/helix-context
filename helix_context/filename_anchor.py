"""
Filename-anchor retrieval tier — spike.

Dewey bench 2026-04-14 showed `key+filename` → 30% recall@1 vs
`key+project+module+filename` → 10%. Filename is the call-number;
project/module over-constrain and hurt retrieval when all axes fire.

This module adds a new retrieval tier (Tier 0.5) that treats filename
stems as first-class anchors — when a query term matches a gene's
filename stem, the gene gets a dedicated boost larger than any single
path-token match. Project/module tokens continue through the existing
path_key_index tier.

Structure:
    filename_index table:   (filename_stem TEXT, gene_id TEXT)
                            PRIMARY KEY (filename_stem, gene_id)
    Populated at gene upsert from ``filename_stem(source_id)``.
    Retrieval tier reads it like a reverse index.

Noise stems are excluded — generic names like ``__init__`` / ``index`` /
``main`` match too many genes to discriminate. See ``_NOISE_STEMS``.

Flag-gated: ``[retrieval].filename_anchor_enabled`` in helix.toml and
``HELIX_FILENAME_ANCHOR_ENABLED=1`` env var. Default off so the existing
retrieval behavior is untouched until benched.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Dict, List, Optional, Set

log = logging.getLogger("helix.filename_anchor")

# Noise stems — generic filenames that appear across many projects and
# would blanket-boost genes if we treated them as discriminators. This
# list is intentionally conservative; grow it only when a specific stem
# shows up as false-positive in a bench.
_NOISE_STEMS = frozenset({
    "__init__", "__main__", "index", "main", "test", "tests",
    "config", "setup", "readme", "license", "notice",
    "package", "mod", "lib", "util", "utils", "helper", "helpers",
})

# Match a filename with an extension, capture the stem. Accepts things
# like "config.py", "Hero.tsx", "helix-launcher.py.bak" (stem stops at
# the first extension to keep stems meaningful).
_FILE_WITH_EXT = re.compile(r"([A-Za-z0-9_.-]+?)\.([A-Za-z0-9]{1,6})$")


def filename_stem(source_id: Optional[str]) -> Optional[str]:
    """Return the filename stem (basename without extension) or None.

    Examples:
        "F:/Projects/helix-context/helix_context/config.py"  -> "config"
        "src/components/Hero.tsx"                            -> "hero"
        "F:/SteamLibrary/steamapps/common/Hades/maps.lua"    -> "maps"
        "README.md"                                          -> "readme"   # will be filtered
        "docs/overview"                                      -> None       # no extension
        None / ""                                            -> None

    Lowercased for case-insensitive matching against query tokens.
    """
    if not source_id:
        return None
    basename = source_id.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    m = _FILE_WITH_EXT.match(basename)
    if not m:
        return None
    stem = m.group(1).lower()
    # Strip any sub-extension piece like ".bak" left in the stem
    # (e.g. "helix.toml.bak" -> stem candidate was "helix.toml", we
    # want "helix"). Safe to split on '.' since real stems don't contain
    # meaningful dots once the extension is off.
    if "." in stem:
        stem = stem.split(".", 1)[0]
    if len(stem) < 2 or stem in _NOISE_STEMS:
        return None
    return stem


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create filename_index table if missing. Idempotent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS filename_index (
            filename_stem TEXT NOT NULL,
            gene_id       TEXT NOT NULL,
            PRIMARY KEY (filename_stem, gene_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_filename_stem "
        "ON filename_index(filename_stem)"
    )


def index_gene(conn: sqlite3.Connection, gene_id: str, source_id: Optional[str]) -> None:
    """Upsert a gene's filename_stem into filename_index.

    Called from genome.upsert_gene alongside path_key_index population.
    Silently no-ops when the gene has no filename-shaped source_id or
    the stem is a known noise word.
    """
    stem = filename_stem(source_id)
    if stem is None:
        return
    try:
        conn.execute(
            "INSERT OR IGNORE INTO filename_index (filename_stem, gene_id) "
            "VALUES (?, ?)",
            (stem, gene_id),
        )
    except Exception:
        log.warning("filename_anchor index failed for gene=%s", gene_id, exc_info=True)


def remove_gene(conn: sqlite3.Connection, gene_id: str) -> None:
    """Remove a gene from filename_index. Called when a gene is deleted."""
    try:
        conn.execute(
            "DELETE FROM filename_index WHERE gene_id = ?", (gene_id,)
        )
    except Exception:
        log.warning("filename_anchor remove failed for gene=%s", gene_id, exc_info=True)


def is_enabled() -> bool:
    """Check env-var override. Toml flag is read separately by the caller."""
    return os.environ.get("HELIX_FILENAME_ANCHOR_ENABLED", "").lower() in {"1", "true", "yes", "on"}


def boost_scores(
    conn: sqlite3.Connection,
    query_terms: List[str],
    gene_scores: Dict[str, float],
    tier_contrib: Dict[str, Dict[str, float]],
    weight: float = 4.0,
    party_filter_sql: str = "",
    party_params: tuple = (),
) -> int:
    """Add filename-anchor bonus to gene_scores in place.

    For each query term that matches an indexed filename stem, every
    gene with that stem gets +weight added to its score. Multi-term
    matches accumulate (a gene with filename "config" hit by a query
    containing "config" twice gets +2*weight — rare but correct).

    Returns the count of genes that received any boost, for logging.
    """
    if not query_terms:
        return 0
    terms_lower = list({t.lower() for t in query_terms if t and len(t) >= 2})
    if not terms_lower:
        return 0

    # Sanity-cap the IN clause — huge query term bags would bloat the
    # SQL but not materially help retrieval. 64 distinct terms is plenty.
    if len(terms_lower) > 64:
        terms_lower = terms_lower[:64]

    placeholders = ",".join("?" * len(terms_lower))
    sql = (
        f"SELECT fi.filename_stem, fi.gene_id "
        f"FROM filename_index fi "
        f"JOIN genes g ON fi.gene_id = g.gene_id "
        f"WHERE fi.filename_stem IN ({placeholders}) "
        f"AND g.chromatin < 2 "
        f"{party_filter_sql}"
    )
    try:
        rows = conn.execute(sql, (*terms_lower, *party_params)).fetchall()
    except Exception:
        log.warning("filename_anchor query failed", exc_info=True)
        return 0

    boosted: Set[str] = set()
    for r in rows:
        # Row may be a sqlite3.Row (dict-indexable) or tuple depending
        # on the connection's row_factory. Handle both.
        try:
            gid = r["gene_id"]
        except (TypeError, IndexError):
            gid = r[1]
        gene_scores[gid] = gene_scores.get(gid, 0.0) + weight
        tier_contrib.setdefault(gid, {})["filename_anchor"] = (
            tier_contrib.get(gid, {}).get("filename_anchor", 0.0) + weight
        )
        boosted.add(gid)
    if boosted:
        log.debug("filename_anchor boosted %d genes (weight=%.1f)", len(boosted), weight)
    return len(boosted)

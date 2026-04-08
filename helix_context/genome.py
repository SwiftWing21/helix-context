"""
Genome — SQLite cold storage for the gene pool.

Biology:
    The genome is the full DNA library. Only ~1% is expressed per cell cycle.
    Our genome stores all context genes in SQLite with a promoter index
    for fast retrieval. Chromatin state controls accessibility.

Includes:
    - DDL (genes table + promoter_index join table)
    - Content-addressed gene IDs (SHA256[:16])
    - Fix 1: synonym expansion for promoter queries
    - Fix 1: co-activation pull-forward (associative memory)
    - Compaction (decay stale genes → HETEROCHROMATIN)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from typing import Dict, List, Optional

from .exceptions import PromoterMismatch
from .schemas import ChromatinState, EpigeneticMarkers, Gene, PromoterTags

log = logging.getLogger(__name__)


class Genome:
    """SQLite-backed gene storage with promoter-tag retrieval."""

    def __init__(
        self,
        path: str,
        synonym_map: Optional[Dict[str, List[str]]] = None,
        decay_rate: float = 0.95,
        heterochromatin_threshold: float = 0.3,
        stale_threshold: float = 3600.0,
    ):
        self.path = path
        self.synonym_map = synonym_map or {}
        self.decay_rate = decay_rate
        self.heterochromatin_threshold = heterochromatin_threshold
        self.stale_threshold = stale_threshold

        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS genes (
            gene_id      TEXT PRIMARY KEY,
            content      TEXT,
            complement   TEXT,
            codons       TEXT,     -- JSON list[str]
            promoter     TEXT,     -- JSON PromoterTags
            epigenetics  TEXT,     -- JSON EpigeneticMarkers
            chromatin    INTEGER,
            is_fragment  INTEGER,
            embedding    TEXT,     -- JSON list[float] | NULL
            source_id    TEXT,
            version      INTEGER,
            supersedes   TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS promoter_index (
            gene_id   TEXT,
            tag_type  TEXT,   -- 'domain' | 'entity'
            tag_value TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS health_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  REAL,
            query      TEXT,
            ellipticity REAL,
            coverage   REAL,
            density    REAL,
            freshness  REAL,
            genes_expressed INTEGER,
            genes_available INTEGER,
            status     TEXT
        )
        """)

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_promoter_value "
            "ON promoter_index(tag_value)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_promoter_gene "
            "ON promoter_index(gene_id)"
        )

        self.conn.commit()

    # ── Gene ID (content-addressable) ───────────────────────────────

    @staticmethod
    def make_gene_id(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # ── Upsert ──────────────────────────────────────────────────────

    def upsert_gene(self, gene: Gene) -> str:
        gene_id = gene.gene_id or self.make_gene_id(gene.content)
        cur = self.conn.cursor()

        cur.execute(
            "INSERT OR REPLACE INTO genes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                gene_id,
                gene.content,
                gene.complement,
                json.dumps(gene.codons),
                gene.promoter.model_dump_json(),
                gene.epigenetics.model_dump_json(),
                int(gene.chromatin),
                int(gene.is_fragment),
                json.dumps(gene.embedding) if gene.embedding else None,
                gene.source_id,
                gene.version,
                gene.supersedes,
            ),
        )

        # Rebuild promoter index for this gene
        cur.execute("DELETE FROM promoter_index WHERE gene_id = ?", (gene_id,))

        for d in gene.promoter.domains:
            cur.execute(
                "INSERT INTO promoter_index VALUES (?, 'domain', ?)",
                (gene_id, d.lower()),
            )
        for e in gene.promoter.entities:
            cur.execute(
                "INSERT INTO promoter_index VALUES (?, 'entity', ?)",
                (gene_id, e.lower()),
            )

        self.conn.commit()
        return gene_id

    # ── Fix 1: synonym expansion ────────────────────────────────────

    def _expand_terms(self, terms: List[str]) -> List[str]:
        expanded = set(t.lower() for t in terms)
        for t in terms:
            key = t.lower()
            if key in self.synonym_map:
                expanded.update(self.synonym_map[key])
        return list(expanded)

    # ── Core retrieval (Step 2) ─────────────────────────────────────

    def query_genes(
        self,
        domains: List[str],
        entities: List[str],
        max_genes: int = 8,
    ) -> List[Gene]:
        """
        Find genes matching the given promoter signals.

        Applies synonym expansion (Fix 1) and co-activation pull-forward (Fix 1).
        Returns up to max_genes * 2 candidates for downstream re-ranking.
        """
        domains = self._expand_terms(domains)
        entities = self._expand_terms(entities)

        query_terms = domains + entities
        if not query_terms:
            raise PromoterMismatch("No query terms after expansion")

        cur = self.conn.cursor()
        placeholders = ",".join("?" * len(query_terms))

        rows = cur.execute(
            f"""
            SELECT g.*, COUNT(pi.tag_value) AS match_score
            FROM genes g
            JOIN promoter_index pi ON g.gene_id = pi.gene_id
            WHERE pi.tag_value IN ({placeholders})
              AND g.chromatin < ?
            GROUP BY g.gene_id
            ORDER BY match_score DESC
            LIMIT ?
            """,
            (*query_terms, int(ChromatinState.HETEROCHROMATIN), max_genes * 2),
        ).fetchall()

        if not rows:
            raise PromoterMismatch("Zero genes matched query terms")

        genes = [self._row_to_gene(r) for r in rows]

        # Fix 1: co-activation pull-forward
        expanded = self._expand_coactivated(genes, limit=max_genes * 2)

        # Dedupe while preserving order
        seen: set[str] = set()
        result: List[Gene] = []
        for g in expanded:
            if g.gene_id not in seen:
                seen.add(g.gene_id)
                result.append(g)

        return result[: max_genes * 2]

    # ── Co-activation expansion ─────────────────────────────────────

    def _expand_coactivated(self, genes: List[Gene], limit: int) -> List[Gene]:
        cur = self.conn.cursor()

        existing_ids = {g.gene_id for g in genes}
        additional_ids: set[str] = set()

        for g in genes:
            for gid in g.epigenetics.co_activated_with[:3]:
                if gid not in existing_ids:
                    additional_ids.add(gid)

        if not additional_ids:
            return genes

        placeholders = ",".join("?" * len(additional_ids))
        rows = cur.execute(
            f"""
            SELECT * FROM genes
            WHERE gene_id IN ({placeholders})
              AND chromatin < ?
            """,
            (*additional_ids, int(ChromatinState.HETEROCHROMATIN)),
        ).fetchall()

        extra = [self._row_to_gene(r) for r in rows]
        return genes + extra

    # ── Row → Gene ──────────────────────────────────────────────────

    def _row_to_gene(self, row: sqlite3.Row) -> Gene:
        return Gene(
            gene_id=row["gene_id"],
            content=row["content"],
            complement=row["complement"],
            codons=json.loads(row["codons"]),
            promoter=PromoterTags.model_validate_json(row["promoter"]),
            epigenetics=EpigeneticMarkers.model_validate_json(row["epigenetics"]),
            chromatin=ChromatinState(row["chromatin"]),
            is_fragment=bool(row["is_fragment"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            source_id=row["source_id"],
            version=row["version"],
            supersedes=row["supersedes"],
        )

    # ── Touch (update epigenetics on access) ────────────────────────

    def touch_genes(self, gene_ids: List[str]) -> None:
        cur = self.conn.cursor()
        now = time.time()

        for gid in gene_ids:
            row = cur.execute(
                "SELECT epigenetics FROM genes WHERE gene_id = ?", (gid,)
            ).fetchone()
            if not row:
                continue

            epi = EpigeneticMarkers.model_validate_json(row["epigenetics"])
            epi.last_accessed = now
            epi.access_count += 1
            epi.decay_score = min(1.0, epi.decay_score + 0.1)  # Refresh on access

            cur.execute(
                "UPDATE genes SET epigenetics = ?, chromatin = ? WHERE gene_id = ?",
                (epi.model_dump_json(), int(ChromatinState.OPEN), gid),
            )

        self.conn.commit()

    # ── Update co-activation links (mutual) ─────────────────────────

    def link_coactivated(self, gene_ids: List[str]) -> None:
        """Create mutual co-activation links between all expressed genes."""
        if len(gene_ids) < 2:
            return

        cur = self.conn.cursor()
        for gid in gene_ids:
            row = cur.execute(
                "SELECT epigenetics FROM genes WHERE gene_id = ?", (gid,)
            ).fetchone()
            if not row:
                continue

            epi = EpigeneticMarkers.model_validate_json(row["epigenetics"])
            peers = [other for other in gene_ids if other != gid]

            # Merge, keeping most recent 10
            existing = set(epi.co_activated_with)
            existing.update(peers)
            epi.co_activated_with = list(existing)[:10]

            cur.execute(
                "UPDATE genes SET epigenetics = ? WHERE gene_id = ?",
                (epi.model_dump_json(), gid),
            )

        self.conn.commit()

    # ── Compaction (decay stale genes) ──────────────────────────────

    def compact(self) -> int:
        """
        Iterate all genes, decay scores for stale ones,
        promote to HETEROCHROMATIN when decay_score < threshold.
        Returns the number of genes compacted.

        Decay is gentle and access-count-aware:
        - Only genes not accessed in stale_threshold seconds are candidates
        - Decay is applied ONCE per compact cycle (not cumulative per second)
        - Genes with high access_count decay slower (frequently used = important)
        - A gene ingested 15 hours ago but never queried still stays at
          decay_score ~0.7 after a day (not 0.0001 like the old math)
        """
        cur = self.conn.cursor()
        now = time.time()
        compacted = 0

        rows = cur.execute("SELECT gene_id, epigenetics, chromatin FROM genes").fetchall()

        for row in rows:
            epi = EpigeneticMarkers.model_validate_json(row["epigenetics"])
            age = now - epi.last_accessed

            if age < self.stale_threshold:
                continue

            # Access-weighted decay: frequently accessed genes resist decay
            # access_count of 5+ means this gene has proven its value
            access_bonus = min(epi.access_count * 0.005, 0.03)  # Max 3% resistance
            effective_rate = self.decay_rate + access_bonus  # e.g., 0.995 + 0.015 = closer to 1.0

            epi.decay_score *= effective_rate
            new_chromatin = int(row["chromatin"])

            if epi.decay_score < self.heterochromatin_threshold and new_chromatin < int(ChromatinState.HETEROCHROMATIN):
                new_chromatin = int(ChromatinState.HETEROCHROMATIN)
                compacted += 1

            cur.execute(
                "UPDATE genes SET epigenetics = ?, chromatin = ? WHERE gene_id = ?",
                (epi.model_dump_json(), new_chromatin, row["gene_id"]),
            )

        self.conn.commit()
        log.info("Compaction complete: %d genes moved to HETEROCHROMATIN", compacted)
        return compacted

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> Dict:
        cur = self.conn.cursor()

        total = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
        by_chromatin = cur.execute(
            "SELECT chromatin, COUNT(*) FROM genes GROUP BY chromatin"
        ).fetchall()

        chromatin_counts = {ChromatinState(r[0]).name: r[1] for r in by_chromatin}

        total_raw = cur.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM genes"
        ).fetchone()[0]
        total_compressed = cur.execute(
            "SELECT COALESCE(SUM(LENGTH(complement)), 0) FROM genes"
        ).fetchone()[0]

        return {
            "total_genes": total,
            "open": chromatin_counts.get("OPEN", 0),
            "euchromatin": chromatin_counts.get("EUCHROMATIN", 0),
            "heterochromatin": chromatin_counts.get("HETEROCHROMATIN", 0),
            "total_chars_raw": total_raw,
            "total_chars_compressed": total_compressed,
            "compression_ratio": total_raw / max(total_compressed, 1),
        }

    # ── Get single gene ─────────────────────────────────────────────

    def get_gene(self, gene_id: str) -> Optional[Gene]:
        row = self.conn.execute(
            "SELECT * FROM genes WHERE gene_id = ?", (gene_id,)
        ).fetchone()
        return self._row_to_gene(row) if row else None

    # ── Health logging ─────────────────────────────────────────────

    def log_health(
        self,
        query: str,
        ellipticity: float,
        coverage: float,
        density: float,
        freshness: float,
        genes_expressed: int,
        genes_available: int,
        status: str,
    ) -> None:
        """Record a health signal for historical tracking."""
        self.conn.execute(
            "INSERT INTO health_log (timestamp, query, ellipticity, coverage, "
            "density, freshness, genes_expressed, genes_available, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), query, ellipticity, coverage, density, freshness,
             genes_expressed, genes_available, status),
        )
        self.conn.commit()

    def health_history(self, limit: int = 50) -> List[Dict]:
        """Return recent health signals, newest first."""
        rows = self.conn.execute(
            "SELECT timestamp, query, ellipticity, coverage, density, freshness, "
            "genes_expressed, genes_available, status "
            "FROM health_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "timestamp": r[0],
                "query": r[1],
                "ellipticity": r[2],
                "coverage": r[3],
                "density": r[4],
                "freshness": r[5],
                "genes_expressed": r[6],
                "genes_available": r[7],
                "status": r[8],
            }
            for r in rows
        ]

    def health_summary(self) -> Dict:
        """Aggregate health stats across all logged queries."""
        cur = self.conn.cursor()
        total = cur.execute("SELECT COUNT(*) FROM health_log").fetchone()[0]
        if total == 0:
            return {"total_queries": 0, "avg_ellipticity": 0, "status_counts": {}}

        avg = cur.execute(
            "SELECT AVG(ellipticity), AVG(coverage), AVG(density), AVG(freshness) "
            "FROM health_log"
        ).fetchone()

        status_counts = {}
        for row in cur.execute(
            "SELECT status, COUNT(*) FROM health_log GROUP BY status"
        ).fetchall():
            status_counts[row[0]] = row[1]

        return {
            "total_queries": total,
            "avg_ellipticity": round(avg[0], 4),
            "avg_coverage": round(avg[1], 4),
            "avg_density": round(avg[2], 4),
            "avg_freshness": round(avg[3], 4),
            "status_counts": status_counts,
        }

    # ── Close ───────────────────────────────────────────────────────

    def close(self) -> None:
        self.conn.close()

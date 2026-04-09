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
import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional

from .accel import (
    json_loads,
    json_dumps,
    parse_promoter,
    parse_epigenetics,
    clear_parse_caches,
    batch_update_epigenetics,
)
from .exceptions import PromoterMismatch
from .schemas import ChromatinState, EpigeneticMarkers, Gene, PromoterTags

log = logging.getLogger(__name__)


class Genome:
    """SQLite-backed gene storage with promoter-tag retrieval."""

    def __init__(
        self,
        path: str,
        synonym_map: Optional[Dict[str, List[str]]] = None,
        sema_codec=None,
    ):
        self.path = path
        self.synonym_map = synonym_map or {}
        self._sema_codec = sema_codec  # Optional SemaCodec for Tier 4 retrieval

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

        cur.execute("""
        CREATE TABLE IF NOT EXISTS gene_relations (
            gene_id_a  TEXT,
            gene_id_b  TEXT,
            relation   INTEGER,
            confidence REAL,
            updated_at REAL,
            PRIMARY KEY (gene_id_a, gene_id_b)
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

        # ── Auto-repair corrupt data on startup ──────────────────
        repaired = 0
        bad = cur.execute(
            "SELECT COUNT(*) FROM genes WHERE typeof(chromatin) != 'integer' "
            "OR chromatin IS NULL OR chromatin NOT IN (0, 1, 2)"
        ).fetchone()[0]
        if bad:
            cur.execute(
                "UPDATE genes SET chromatin = 0 "
                "WHERE typeof(chromatin) != 'integer' "
                "OR chromatin IS NULL OR chromatin NOT IN (0, 1, 2)"
            )
            repaired += bad
            log.warning("Auto-repaired %d genes with corrupt chromatin", bad)

        null_epi = cur.execute(
            "SELECT COUNT(*) FROM genes WHERE epigenetics IS NULL"
        ).fetchone()[0]
        if null_epi:
            default_epi = '{"created_at":0,"last_accessed":0,"access_count":0,"co_activated_with":[],"typed_co_activated":[],"decay_score":1.0}'
            cur.execute(
                "UPDATE genes SET epigenetics = ? WHERE epigenetics IS NULL",
                (default_epi,),
            )
            repaired += null_epi
            log.warning("Auto-repaired %d genes with NULL epigenetics", null_epi)

        if repaired:
            self.conn.commit()

        # FTS5 full-text index on gene content + complement
        # Standalone table (not content-synced) for simplicity
        try:
            cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS genes_fts USING fts5(
                gene_id,
                content,
                complement
            )
            """)
            self._fts_available = True

            # Auto-sync FTS5 if out of date
            gene_count = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
            fts_count = cur.execute("SELECT COUNT(*) FROM genes_fts").fetchone()[0]
            if abs(gene_count - fts_count) > 0:
                log.info("FTS5 out of sync (%d genes, %d fts) — rebuilding", gene_count, fts_count)
                cur.execute("DELETE FROM genes_fts")
                cur.execute(
                    "INSERT INTO genes_fts(gene_id, content, complement) "
                    "SELECT gene_id, content, COALESCE(complement, '') FROM genes"
                )
                self.conn.commit()
        except Exception:
            log.warning("FTS5 not available — content search disabled")
            self._fts_available = False

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
                json_dumps(gene.codons),
                gene.promoter.model_dump_json(),
                gene.epigenetics.model_dump_json(),
                int(gene.chromatin),
                int(gene.is_fragment),
                json_dumps(gene.embedding) if gene.embedding else None,
                gene.source_id,
                gene.version,
                gene.supersedes,
            ),
        )
        # Invalidate parse cache for this gene's promoter/epigenetics
        clear_parse_caches()

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

        # Sync FTS5 index
        if self._fts_available:
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO genes_fts(gene_id, content, complement) "
                    "VALUES (?, ?, ?)",
                    (gene_id, gene.content, gene.complement or ""),
                )
            except Exception:
                pass  # FTS sync failure is non-fatal

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

    # ── Core retrieval (Step 2) — hybrid promoter + FTS5 ────────────

    def query_genes(
        self,
        domains: List[str],
        entities: List[str],
        max_genes: int = 8,
    ) -> List[Gene]:
        """
        Find genes matching the given promoter signals.

        Three-tier retrieval:
            1. Exact promoter tag match (highest confidence)
            2. Prefix tag match — "server" matches "serverconfig" (medium)
            3. FTS5 content search — searches gene text directly (fallback)

        Results are merged with weighted scoring, then expanded via
        co-activation pull-forward. Returns up to max_genes * 2 candidates.
        """
        domains = self._expand_terms(domains)
        entities = self._expand_terms(entities)

        query_terms = domains + entities
        if not query_terms:
            raise PromoterMismatch("No query terms after expansion")

        cur = self.conn.cursor()
        limit = max_genes * 2

        # Gene scores: gene_id → float (accumulated across tiers)
        gene_scores: Dict[str, float] = {}

        # ── Tier 1: exact promoter tag match (weight 3.0) ──────────
        placeholders = ",".join("?" * len(query_terms))
        rows = cur.execute(
            f"""
            SELECT g.gene_id, COUNT(pi.tag_value) AS match_count
            FROM genes g
            JOIN promoter_index pi ON g.gene_id = pi.gene_id
            WHERE pi.tag_value IN ({placeholders})
              AND g.chromatin < ?
            GROUP BY g.gene_id
            """,
            (*query_terms, int(ChromatinState.HETEROCHROMATIN)),
        ).fetchall()

        for r in rows:
            gene_scores[r["gene_id"]] = r["match_count"] * 3.0

        # ── Tier 2: prefix tag match (weight 1.5) ──────────────────
        # "server" matches "serverconfig", "server_api", etc.
        prefix_conditions = " OR ".join(
            "pi.tag_value LIKE ?" for _ in query_terms
        )
        prefix_params = [f"{t}%" for t in query_terms]
        rows = cur.execute(
            f"""
            SELECT g.gene_id, COUNT(pi.tag_value) AS match_count
            FROM genes g
            JOIN promoter_index pi ON g.gene_id = pi.gene_id
            WHERE ({prefix_conditions})
              AND g.chromatin < ?
            GROUP BY g.gene_id
            """,
            (*prefix_params, int(ChromatinState.HETEROCHROMATIN)),
        ).fetchall()

        for r in rows:
            gid = r["gene_id"]
            prefix_score = r["match_count"] * 1.5
            gene_scores[gid] = gene_scores.get(gid, 0) + prefix_score

        # ── Tier 3: FTS5 content search (weight 2.0) ───────────────
        if self._fts_available:
            # Build FTS5 query: OR-join all terms
            fts_query = " OR ".join(
                f'"{t}"' for t in query_terms if len(t) > 2
            )
            if fts_query:
                try:
                    fts_rows = cur.execute(
                        """
                        SELECT gene_id, rank
                        FROM genes_fts
                        WHERE genes_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, limit * 2),
                    ).fetchall()

                    # Filter by chromatin state (batch lookup)
                    if fts_rows:
                        fts_ids = [r["gene_id"] for r in fts_rows]
                        fts_ranks = {r["gene_id"]: r["rank"] for r in fts_rows}
                        id_ph = ",".join("?" * len(fts_ids))
                        valid = cur.execute(
                            f"SELECT gene_id FROM genes "
                            f"WHERE gene_id IN ({id_ph}) AND chromatin < ?",
                            (*fts_ids, int(ChromatinState.HETEROCHROMATIN)),
                        ).fetchall()
                        valid_ids = {r["gene_id"] for r in valid}

                        for gid in fts_ids:
                            if gid not in valid_ids:
                                continue
                            # FTS5 rank is negative (lower = better match)
                            # Normalize: -rank gives positive, cap at 10
                            fts_score = min(-fts_ranks[gid], 10.0) * 2.0
                            gene_scores[gid] = gene_scores.get(gid, 0) + fts_score
                except Exception:
                    log.warning("FTS5 query failed", exc_info=True)

        # ── Tier 4: ΣĒMA semantic retrieval (Phase 2 — currently disabled)
        # ΣĒMA vectors are stored on every gene (20D projections via
        # sentence-transformer). Retrieval integration needs tuning to
        # avoid displacing exact FTS5 content matches.
        # The codec, vectors, and nearest() are ready — enable when
        # the boost/weight balance is calibrated against a larger needle set.

        if not gene_scores:
            raise PromoterMismatch("Zero genes matched across all tiers")

        # ── Lexical anchoring: IDF-weighted rare-term boost ────────
        # Weight query terms by inverse document frequency — rare terms
        # are stronger discriminators. A gene matching "conductor" (3 genes)
        # is much more likely the answer than one matching "biged" (200+ genes).
        total_genes_est = max(len(gene_scores), 100)
        import math as _math
        for term in query_terms:
            term_freq = cur.execute(
                "SELECT COUNT(DISTINCT gene_id) FROM promoter_index WHERE tag_value = ?",
                (term,),
            ).fetchone()[0]
            if term_freq == 0:
                continue
            # IDF boost: rare terms get up to 5.0, common terms ~0.5
            idf = _math.log(total_genes_est / term_freq) if term_freq > 0 else 0
            boost = min(idf * 1.5, 5.0)
            if boost > 1.0:
                anchor_genes = cur.execute(
                    "SELECT gene_id FROM promoter_index WHERE tag_value = ?",
                    (term,),
                ).fetchall()
                for r in anchor_genes:
                    gene_scores[r["gene_id"]] = gene_scores.get(r["gene_id"], 0) + boost

        # Sort by combined score, fetch top genes
        ranked_ids = sorted(gene_scores, key=gene_scores.get, reverse=True)[:limit]

        # Batch fetch gene rows
        id_placeholders = ",".join("?" * len(ranked_ids))
        rows = cur.execute(
            f"SELECT * FROM genes WHERE gene_id IN ({id_placeholders})",
            ranked_ids,
        ).fetchall()

        # Preserve ranked order
        row_map = {r["gene_id"]: r for r in rows}
        genes = [self._row_to_gene(row_map[gid]) for gid in ranked_ids if gid in row_map]

        # Co-activation pull-forward
        expanded = self._expand_coactivated(genes, limit=limit)

        # Dedupe while preserving order
        seen: set[str] = set()
        result: List[Gene] = []
        for g in expanded:
            if g.gene_id not in seen:
                seen.add(g.gene_id)
                result.append(g)

        return result[:limit]

    # ── Co-activation expansion ─────────────────────────────────────

    def _expand_coactivated(self, genes: List[Gene], limit: int) -> List[Gene]:
        cur = self.conn.cursor()

        existing_ids = {g.gene_id for g in genes}
        additional_ids: set[str] = set()

        for g in genes:
            # Prefer typed relations if available
            if g.epigenetics.typed_co_activated:
                for link in g.epigenetics.typed_co_activated[:5]:
                    if link.gene_id in existing_ids:
                        continue
                    # Entailment/equivalence: always pull forward
                    if link.relation in (0, 1, 2):  # ENTAILMENT, REVERSE_ENTAILMENT, EQUIVALENCE
                        additional_ids.add(link.gene_id)
                    # Alternation: skip (mutually exclusive)
                    elif link.relation == 3:  # ALTERNATION
                        pass
                    # Independence/cover/negation: pull only if high confidence
                    elif link.confidence > 0.7:
                        additional_ids.add(link.gene_id)
            else:
                # Fall back to untyped co-activation
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
        # Guard against NULL/corrupt metadata fields
        try:
            promoter = parse_promoter(row["promoter"]) if row["promoter"] else PromoterTags()
        except Exception:
            promoter = PromoterTags()
        try:
            epigenetics = parse_epigenetics(row["epigenetics"]) if row["epigenetics"] else EpigeneticMarkers()
        except Exception:
            epigenetics = EpigeneticMarkers()
        try:
            chromatin = ChromatinState(row["chromatin"]) if row["chromatin"] is not None else ChromatinState.OPEN
        except (ValueError, TypeError):
            chromatin = ChromatinState.OPEN

        return Gene(
            gene_id=row["gene_id"],
            content=row["content"],
            complement=row["complement"],
            codons=json_loads(row["codons"]) if row["codons"] else [],
            promoter=promoter,
            epigenetics=epigenetics,
            chromatin=chromatin,
            is_fragment=bool(row["is_fragment"]) if row["is_fragment"] is not None else False,
            embedding=json_loads(row["embedding"]) if row["embedding"] else None,
            source_id=row["source_id"],
            version=row["version"] if row["version"] is not None else 1,
            supersedes=row["supersedes"],
        )

    # ── Touch (update epigenetics on access) ────────────────────────

    def touch_genes(self, gene_ids: List[str]) -> None:
        if not gene_ids:
            return

        cur = self.conn.cursor()
        now = time.time()

        # Batch fetch all epigenetics in one query
        placeholders = ",".join("?" * len(gene_ids))
        rows = cur.execute(
            f"SELECT gene_id, epigenetics FROM genes WHERE gene_id IN ({placeholders})",
            gene_ids,
        ).fetchall()

        # Individual UPDATEs — safe against column-swap corruption
        # (CASE WHEN batch was causing epigenetics JSON to land in chromatin)
        for row in rows:
            if not row["epigenetics"]:
                continue
            epi = parse_epigenetics(row["epigenetics"], use_cache=False)
            epi.last_accessed = now
            epi.access_count += 1
            epi.decay_score = min(1.0, epi.decay_score + 0.1)
            cur.execute(
                "UPDATE genes SET epigenetics = ?, chromatin = ? WHERE gene_id = ?",
                (epi.model_dump_json(), int(ChromatinState.OPEN), row["gene_id"]),
            )

        self.conn.commit()
        clear_parse_caches()

    # ── Update co-activation links (mutual) ─────────────────────────

    def link_coactivated(self, gene_ids: List[str]) -> None:
        """Create mutual co-activation links between all expressed genes."""
        if len(gene_ids) < 2:
            return

        cur = self.conn.cursor()

        # Batch fetch all epigenetics in one query
        placeholders = ",".join("?" * len(gene_ids))
        rows = cur.execute(
            f"SELECT gene_id, epigenetics FROM genes WHERE gene_id IN ({placeholders})",
            gene_ids,
        ).fetchall()

        # Build individual updates (epigenetics only, preserve chromatin)
        for row in rows:
            if not row["epigenetics"]:
                continue
            epi = parse_epigenetics(row["epigenetics"], use_cache=False)
            gid = row["gene_id"]
            peers = [other for other in gene_ids if other != gid]

            existing = set(epi.co_activated_with)
            existing.update(peers)
            epi.co_activated_with = list(existing)[:10]

            cur.execute(
                "UPDATE genes SET epigenetics = ? WHERE gene_id = ?",
                (epi.model_dump_json(), gid),
            )

        self.conn.commit()
        clear_parse_caches()

    # ── Typed gene relations (NLI) ───────────────────────────────────

    def store_relation(
        self, gene_id_a: str, gene_id_b: str,
        relation: int, confidence: float,
    ) -> None:
        """Store a typed logical relation between two genes."""
        import time as _time
        self.conn.execute(
            "INSERT OR REPLACE INTO gene_relations "
            "(gene_id_a, gene_id_b, relation, confidence, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (gene_id_a, gene_id_b, relation, confidence, _time.time()),
        )
        self.conn.commit()

    def store_relations_batch(
        self, relations: list,
    ) -> None:
        """Store multiple typed relations. Each item: (id_a, id_b, relation, confidence)."""
        import time as _time
        now = _time.time()
        self.conn.executemany(
            "INSERT OR REPLACE INTO gene_relations "
            "(gene_id_a, gene_id_b, relation, confidence, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(a, b, r, c, now) for a, b, r, c in relations],
        )
        self.conn.commit()

    def get_relations(self, gene_id: str) -> list:
        """Get all typed relations for a gene. Returns [(other_id, relation, confidence)]."""
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT gene_id_b AS other, relation, confidence "
            "FROM gene_relations WHERE gene_id_a = ? "
            "UNION "
            "SELECT gene_id_a AS other, relation, confidence "
            "FROM gene_relations WHERE gene_id_b = ?",
            (gene_id, gene_id),
        ).fetchall()
        return [(r["other"], r["relation"], r["confidence"]) for r in rows]

    # ── Compaction (decay stale genes) ──────────────────────────────

    def compact(self) -> int:
        """
        Check genes for source file changes. No time-based decay.

        Genes are NEVER removed by time alone. Knowledge doesn't expire.
        Only two things change a gene's state:

        1. SOURCE CHANGED: if gene.source_id points to a file whose mtime
           is newer than last_accessed, decay_score drops to 0.5 (AGING)
           and chromatin moves to EUCHROMATIN. The gene is still queryable
           but the system knows it's outdated. Re-ingesting resets it.

        2. EXPLICIT SPLICE: the ribosome's splice operation cuts introns
           per-query (irrelevant codons). This is the RNA splicing analog —
           relevance filtering happens at expression time, not storage time.

        Time since last access is used ONLY for expression priority
        (recently accessed genes rank higher in query results), never
        for deletion or decay.

        Returns the number of genes marked as source-changed.
        """
        cur = self.conn.cursor()
        change_detected = 0

        rows = cur.execute(
            "SELECT gene_id, epigenetics, chromatin, source_id FROM genes"
        ).fetchall()

        for row in rows:
            source_id = row["source_id"]
            if not source_id or not os.path.exists(source_id):
                continue

            try:
                file_mtime = os.path.getmtime(source_id)
            except OSError:
                continue

            epi = parse_epigenetics(row["epigenetics"], use_cache=False)

            if file_mtime > epi.last_accessed:
                # Source changed — gene is outdated but NOT removed
                epi.decay_score = min(epi.decay_score, 0.5)
                new_chromatin = int(ChromatinState.EUCHROMATIN)
                change_detected += 1

                cur.execute(
                    "UPDATE genes SET epigenetics = ?, chromatin = ? WHERE gene_id = ?",
                    (epi.model_dump_json(), new_chromatin, row["gene_id"]),
                )

        self.conn.commit()
        if change_detected:
            log.info("Compaction: %d source changes detected (genes marked EUCHROMATIN)",
                     change_detected)
        return change_detected

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> Dict:
        cur = self.conn.cursor()

        total = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
        by_chromatin = cur.execute(
            "SELECT chromatin, COUNT(*) FROM genes GROUP BY chromatin"
        ).fetchall()

        chromatin_counts = {}
        for r in by_chromatin:
            try:
                key = ChromatinState(int(r[0])).name if r[0] is not None else "UNKNOWN"
            except (ValueError, TypeError):
                key = "UNKNOWN"
            chromatin_counts[key] = chromatin_counts.get(key, 0) + r[1]

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

    # ── FTS5 index rebuild ────────────────────────────────────────────

    def rebuild_fts(self) -> int:
        """Rebuild the FTS5 index from all genes. Returns count indexed."""
        if not self._fts_available:
            log.warning("FTS5 not available — cannot rebuild")
            return 0

        cur = self.conn.cursor()
        # Clear and repopulate
        cur.execute("DELETE FROM genes_fts")
        cur.execute(
            "INSERT INTO genes_fts(gene_id, content, complement) "
            "SELECT gene_id, content, COALESCE(complement, '') FROM genes"
        )
        self.conn.commit()
        count = cur.execute("SELECT COUNT(*) FROM genes_fts").fetchone()[0]
        log.info("FTS5 index rebuilt: %d genes indexed", count)
        return count

    # ── Close ───────────────────────────────────────────────────────

    def close(self) -> None:
        self.conn.close()

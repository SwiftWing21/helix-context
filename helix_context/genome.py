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
import re
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


# ── Struggle 1 fix: source-path deny list ───────────────────────────────
#
# Paths that are structurally noise regardless of content. Any gene whose
# source_id matches one of these patterns goes directly to HETEROCHROMATIN
# without computing a density score — it's cheaper and more reliable than
# relying on the scorer for content types we already know are noise.
#
# Categories covered:
#   - Build artifacts (.next, node_modules, __pycache__, dist, build, target)
#   - Lockfiles and minified bundles
#   - Web manifest files (app-paths-manifest.json, reference-manifest.js)
#   - Non-English locale directories (software i18n is high-volume low-signal)
#
# NOT in this list (deliberate):
#   - *.csv — business CSVs (customer data, financial records, invoice exports)
#     are legitimate ingest targets. Generic low-density CSVs will be caught
#     by the score gate instead.
#   - *.json — JSON is everywhere, most of it is config/data with signal
#   - *.md — markdown is primary signal content
#   - Cargo.toml / pyproject.toml — project metadata is signal
#   - Steam / game content (SteamLibrary, steamapps, BeamNG, Hades,
#     Factorio, Dyson Sphere, etc.) — reframed as high-SNR signal on
#     2026-04-10. Game files are content-dense with unambiguous literal
#     values (configs, enums, item IDs, code) and empirically produced
#     86% of correct answers on the N=50 v2 NIAH benchmark before the
#     original gate. Individual low-density game genes still get caught
#     by the score gate; the structural path is no longer a categorical
#     reject. See docs/BENCHMARKS.md and ~/.helix/shared/handoffs/ for
#     the full empirical basis.
#
# Patterns are anchored to directory boundaries to avoid false positives
# on legitimate files that happen to contain the substring.
_DENY_PATTERNS = [
    # Build artifacts
    r"[\\/]\.next[\\/]",
    r"[\\/]node_modules[\\/]",
    r"[\\/]__pycache__[\\/]",
    r"[\\/]dist[\\/]",
    r"[\\/]build[\\/](?!\.(bat|ps1|sh)$)",  # keep build.bat/build.sh etc.
    r"[\\/]target[\\/]debug[\\/]",
    r"[\\/]target[\\/]release[\\/]",
    # Lockfiles / manifests
    r"[\\/]package-lock\.json$",
    r"[\\/]yarn\.lock$",
    r"[\\/]Cargo\.lock$",
    r"[\\/]uv\.lock$",
    r"[\\/]poetry\.lock$",
    r"[\\/]Pipfile\.lock$",
    r"[\\/]composer\.lock$",
    # Minified bundles / source maps
    r"\.min\.(js|css|mjs)$",
    r"\.map$",
    # Next.js / web-framework manifests
    r"app-paths-manifest\.json$",
    r"app-build-manifest\.json$",
    r"_buildManifest\.js$",
    r"_ssgManifest\.js$",
    r"client-reference-manifest\.(js|json)$",
    r"server-reference-manifest\.(js|json)$",
    # Binary / compiled artifacts
    r"\.(pyc|pyo|so|dll|dylib|exe|wasm|bin|pack|idx)$",
    # Non-English software locale directories (English is kept as the
    # primary user base; other locales are high-volume low-signal for
    # typical retrieval workloads). Game subtitles are NOT in this list —
    # they're reframed as signal along with the rest of the game content.
    r"[\\/]locale[\\/](?!en[\\/])[a-z]{2,3}[\\/]",
]

_DENY_RE = re.compile("|".join(_DENY_PATTERNS), re.IGNORECASE)


def is_denied_source(source_id: Optional[str]) -> bool:
    """Return True iff source_id matches the structural noise deny list.

    Exposed as a module-level function so tests and scripts can reuse it
    without constructing a full Genome instance.
    """
    if not source_id:
        return False
    return bool(_DENY_RE.search(source_id))


# Thresholds for the score-based gate. Calibrated against the 2026-04-10
# noise-diluted genome (8,063 genes, ~42% structural noise). See
# scripts/simulate_density_gate_v2.py for the empirical basis.
_DENSITY_HETEROCHROMATIN_THRESHOLD = 0.50
_DENSITY_EUCHROMATIN_THRESHOLD = 1.00
_DENSITY_CONTENT_LENGTH_FLOOR = 100  # chars — prevents tiny-content score explosion
_DENSITY_ACCESS_OVERRIDE = 5         # access_count >= this keeps gene OPEN regardless


class Genome:
    """SQLite-backed gene storage with promoter-tag retrieval."""

    def __init__(
        self,
        path: str,
        synonym_map: Optional[Dict[str, List[str]]] = None,
        sema_codec=None,
        splade_enabled: bool = False,
        entity_graph: bool = False,
    ):
        self.path = path
        self.synonym_map = synonym_map or {}
        self._sema_codec = sema_codec  # Optional SemaCodec for Tier 4 retrieval
        self._replication_mgr = None  # Set by set_replication_manager()
        self._splade_enabled = splade_enabled
        self._entity_graph_enabled = entity_graph

        # Checkpoint WAL BEFORE opening our long-lived connection
        # so we see the latest state from any external writers
        try:
            _tmp = sqlite3.connect(self.path)
            _tmp.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _tmp.close()
        except Exception:
            pass

        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")  # 30s retry on lock
        self._upsert_count = 0  # WAL checkpoint cadence counter
        self.last_query_scores: Dict[str, float] = {}  # Retrieval scores from last query
        self._sema_cache: Optional[Dict] = None  # Pre-materialized ΣĒMA vectors
        self._init_db()

        # Dedicated read-only connection — WAL allows concurrent readers
        # without blocking the writer. Separate connection = no lock contention.
        if self.path != ":memory:":
            self._reader = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True,
                check_same_thread=False, timeout=10,
            )
            self._reader.row_factory = sqlite3.Row
            self._reader.execute("PRAGMA busy_timeout=10000")
        else:
            self._reader = None

        # Create SPLADE inverted index if enabled
        if self._splade_enabled:
            try:
                from . import splade_backend
                splade_backend.create_splade_table(self.conn)
                log.info("SPLADE inverted index ready")
            except ImportError:
                log.warning("SPLADE backend not available (transformers not installed)")
                self._splade_enabled = False
            except Exception:
                log.warning("SPLADE table creation failed", exc_info=True)
                self._splade_enabled = False

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
            supersedes   TEXT,
            key_values   TEXT     -- JSON list[str] | NULL
        )
        """)

        # Auto-add key_values column if upgrading from older schema
        try:
            cur.execute("SELECT key_values FROM genes LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE genes ADD COLUMN key_values TEXT")
            log.info("Added key_values column to genes table")

        # Auto-add compression_tier column (0=OPEN, 1=EUCHROMATIN, 2=HETEROCHROMATIN)
        try:
            cur.execute("SELECT compression_tier FROM genes LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE genes ADD COLUMN compression_tier INTEGER DEFAULT 0")
            log.info("Added compression_tier column to genes table")

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

        # Entity graph — maps entities to genes for graph-based co-activation
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_graph (
            entity   TEXT NOT NULL,
            gene_id  TEXT NOT NULL,
            PRIMARY KEY (entity, gene_id)
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_graph_entity "
            "ON entity_graph(entity)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_graph_gene "
            "ON entity_graph(gene_id)"
        )

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

            # Incremental FTS5 sync — only add missing genes, don't rebuild
            # Full rebuild is O(N) and blocks startup. At 100K+ genes it takes
            # 30+ seconds. Incremental sync is O(delta) — typically <100 genes.
            gene_count = cur.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
            fts_count = cur.execute("SELECT COUNT(*) FROM genes_fts").fetchone()[0]
            delta = gene_count - fts_count
            if delta > 0:
                # Add only genes missing from FTS5
                cur.execute(
                    "INSERT INTO genes_fts(gene_id, content, complement) "
                    "SELECT g.gene_id, "
                    "  COALESCE(g.source_id,'') || ' ' || "
                    "  COALESCE((SELECT GROUP_CONCAT(pi.tag_value, ' ') "
                    "    FROM promoter_index pi WHERE pi.gene_id = g.gene_id), '') "
                    "  || ' ' || g.content, "
                    "  COALESCE(g.complement, '') "
                    "FROM genes g "
                    "WHERE g.gene_id NOT IN (SELECT gene_id FROM genes_fts)"
                )
                self.conn.commit()
                log.info("FTS5 incremental sync: +%d genes (total: %d)", delta, gene_count)
            elif delta < 0:
                # FTS5 has orphan entries — remove them
                cur.execute(
                    "DELETE FROM genes_fts "
                    "WHERE gene_id NOT IN (SELECT gene_id FROM genes)"
                )
                self.conn.commit()
                log.info("FTS5 cleanup: removed %d orphan entries", -delta)
        except Exception:
            log.warning("FTS5 not available — content search disabled")
            self._fts_available = False

        # ── Session registry tables (see docs/SESSION_REGISTRY.md) ──
        # Purely additive — presence, attribution, and the BM25 bypass for
        # `GET /sessions/{handle}/recent`. Schema creation is idempotent;
        # skipping this block would leave older databases unable to use
        # the registry endpoints but would not break anything else.
        try:
            self._ensure_registry_schema(cur)
        except Exception:
            log.warning("Session registry schema init failed", exc_info=True)

        self.conn.commit()

    def _ensure_registry_schema(self, cur: sqlite3.Cursor) -> None:
        """Create session registry tables + indexes. Idempotent.

        Adds three tables:
            - parties:           atomic trust identities
            - participants:      live runtime actors under a party
            - gene_attribution:  links genes to the party/participant that wrote them

        See docs/SESSION_REGISTRY.md for the full design.
        """
        cur.execute("""
        CREATE TABLE IF NOT EXISTS parties (
            party_id      TEXT PRIMARY KEY,
            display_name  TEXT NOT NULL,
            trust_domain  TEXT NOT NULL DEFAULT 'local',
            created_at    REAL NOT NULL,
            metadata      TEXT
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_parties_trust_domain "
            "ON parties(trust_domain)"
        )

        cur.execute("""
        CREATE TABLE IF NOT EXISTS participants (
            participant_id   TEXT PRIMARY KEY,
            party_id         TEXT NOT NULL REFERENCES parties(party_id),
            handle           TEXT NOT NULL,
            workspace        TEXT,
            pid              INTEGER,
            started_at       REAL NOT NULL,
            last_heartbeat   REAL NOT NULL,
            status           TEXT NOT NULL DEFAULT 'active',
            capabilities     TEXT,
            metadata         TEXT
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_participants_party "
            "ON participants(party_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_participants_heartbeat "
            "ON participants(last_heartbeat)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_participants_handle "
            "ON participants(handle)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_participants_status "
            "ON participants(status)"
        )

        cur.execute("""
        CREATE TABLE IF NOT EXISTS gene_attribution (
            gene_id         TEXT PRIMARY KEY
                            REFERENCES genes(gene_id) ON DELETE CASCADE,
            party_id        TEXT NOT NULL
                            REFERENCES parties(party_id),
            participant_id  TEXT
                            REFERENCES participants(participant_id) ON DELETE SET NULL,
            authored_at     REAL NOT NULL
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_attribution_party_time "
            "ON gene_attribution(party_id, authored_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_attribution_participant_time "
            "ON gene_attribution(participant_id, authored_at DESC)"
        )

    # ── WAL snapshot management ──────────────────────────────────────

    def _refresh_snapshot(self) -> None:
        """Release stale WAL read transaction so next SELECT sees current state.

        In SQLite WAL mode, Python's sqlite3 module starts an implicit
        transaction on SELECT. That transaction pins a snapshot — external
        writers (ingest, thinning scripts) commit to the WAL but this
        connection won't see those changes until the implicit transaction ends.

        Calling commit() ends the implicit transaction. The next SELECT
        will start a new one with the latest WAL state.
        """
        try:
            self.conn.commit()
        except Exception:
            pass  # No active transaction — safe to ignore

    # ── ΣĒMA vector cache (pre-materialized for fast Mode B scans) ──

    def _build_sema_cache(self) -> None:
        """
        Load all ΣĒMA vectors into RAM as a numpy matrix for fast
        cosine similarity. Eliminates 7K json_loads() per Mode B query.

        Cache structure:
            gene_ids: list[str] — ordered gene IDs
            matrix: np.ndarray (N, 20) — float32 ΣĒMA vectors
        """
        try:
            import numpy as np
        except ImportError:
            log.debug("numpy not available, ΣĒMA cache disabled")
            return

        cur = self.read_conn.cursor()
        # Try the tier-aware query first; fall back to legacy schema
        # when the read path is a replica that hasn't been migrated yet.
        try:
            rows = cur.execute(
                "SELECT gene_id, embedding FROM genes "
                "WHERE embedding IS NOT NULL AND chromatin < ? "
                "AND COALESCE(compression_tier, 0) < 2",
                (int(ChromatinState.HETEROCHROMATIN),),
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "compression_tier" in str(e):
                log.warning(
                    "read_conn lacks compression_tier column — "
                    "falling back to legacy schema (likely a stale replica)"
                )
                rows = cur.execute(
                    "SELECT gene_id, embedding FROM genes "
                    "WHERE embedding IS NOT NULL AND chromatin < ?",
                    (int(ChromatinState.HETEROCHROMATIN),),
                ).fetchall()
            else:
                raise

        gene_ids = []
        vectors = []
        for r in rows:
            try:
                vec = json_loads(r["embedding"])
                if isinstance(vec, list) and len(vec) == 20:
                    gene_ids.append(r["gene_id"])
                    vectors.append(vec)
            except Exception:
                continue

        if vectors:
            matrix = np.array(vectors, dtype=np.float32)
            # Normalize rows for cosine similarity via dot product
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            matrix = matrix / norms
            self._sema_cache = {"gene_ids": gene_ids, "matrix": matrix}
            log.info("ΣĒMA cache built: %d vectors (%d KB)",
                     len(gene_ids), matrix.nbytes // 1024)
        else:
            self._sema_cache = None

    def invalidate_sema_cache(self) -> None:
        """Mark cache stale — rebuilt on next Mode B query."""
        self._sema_cache = None

    # ── Replication ──────────────────────────────────────────────────

    def set_replication_manager(self, mgr) -> None:
        """Attach a ReplicationManager for distributed genome clones."""
        self._replication_mgr = mgr

    @property
    def read_conn(self) -> sqlite3.Connection:
        """
        Dedicated read-only connection. In WAL mode, readers and writers
        don't block each other — but only if they use separate connections.

        Priority: replication replica > dedicated reader > write connection.
        """
        if self._replication_mgr is not None:
            try:
                return self._replication_mgr.get_reader()
            except Exception:
                pass
        if self._reader is not None:
            return self._reader
        return self.conn

    # ── Gene ID (content-addressable) ───────────────────────────────

    @staticmethod
    def make_gene_id(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # ── Upsert ──────────────────────────────────────────────────────

    def upsert_gene(self, gene: Gene, apply_gate: bool = True) -> str:
        """
        Insert or replace a gene in the genome.

        If ``apply_gate`` is True (the default), the density gate runs
        before storage and may override the gene's chromatin state. Callers
        that have a reason to bypass the gate — HGT imports, benchmark
        setup scripts, explicit backfill tools, manual `compact_genome`
        re-runs — can pass ``apply_gate=False`` to preserve the incoming
        chromatin state as-is.

        Returns the gene_id (content-addressed if not pre-populated).
        """
        gene_id = gene.gene_id or self.make_gene_id(gene.content)

        # Struggle 1 fix: apply density gate at the storage boundary so
        # that bulk ingest scripts (ingest_steam.py, ingest_fdrive.py,
        # ingest_all.py) calling upsert_gene directly also respect the
        # gate. Previously the gate lived in context_manager.ingest() and
        # was bypassed by every bulk ingest path. See:
        #   scripts/simulate_density_gate_v2.py for the empirical basis
        #   (51.6% of the noise-diluted genome demoted, >97% signal retained).
        #
        # Crucially, the gate only acts on genes arriving as OPEN — if the
        # caller has explicitly set EUCHROMATIN or HETEROCHROMATIN, we
        # trust that decision. This means HGT imports, test fixtures, and
        # any code that deliberately creates demoted genes retain their
        # intended state. The gate is admission-control, not state-reset.
        if apply_gate and gene.chromatin == ChromatinState.OPEN:
            new_state, reason = self.apply_density_gate(gene)
            if new_state != gene.chromatin:
                log.debug(
                    "Density gate demoted %s: OPEN -> %s (reason=%s)",
                    gene_id, new_state.name, reason,
                )
                gene.chromatin = new_state
        else:
            reason = "gate_bypassed" if not apply_gate else "explicit_chromatin_preserved"

        # Compute compression tier from final chromatin state
        tier = 0  # OPEN
        if gene.chromatin == ChromatinState.EUCHROMATIN:
            tier = 1
        elif gene.chromatin == ChromatinState.HETEROCHROMATIN:
            tier = 2

        cur = self.conn.cursor()

        cur.execute(
            "INSERT OR REPLACE INTO genes "
            "(gene_id, content, complement, codons, promoter, epigenetics, "
            "chromatin, is_fragment, embedding, source_id, version, supersedes, "
            "key_values, compression_tier) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                json_dumps(gene.key_values) if gene.key_values else None,
                tier,
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

        # Sync FTS5 index — include source_id + promoter tags in searchable content
        # so tag-based knowledge survives FTS5 rebuilds
        if self._fts_available:
            try:
                tag_text = " ".join(
                    [d.lower() for d in gene.promoter.domains]
                    + [e.lower() for e in gene.promoter.entities]
                )
                fts_content = f"{gene.source_id or ''} {tag_text} {gene.content}"
                cur.execute(
                    "INSERT OR REPLACE INTO genes_fts(gene_id, content, complement) "
                    "VALUES (?, ?, ?)",
                    (gene_id, fts_content, gene.complement or ""),
                )
            except Exception:
                pass  # FTS sync failure is non-fatal

        # Entity graph — index entities for graph-based co-activation
        if self._entity_graph_enabled and gene.promoter.entities:
            cur.execute("DELETE FROM entity_graph WHERE gene_id = ?", (gene_id,))
            for ent in gene.promoter.entities[:15]:
                cur.execute(
                    "INSERT OR IGNORE INTO entity_graph (entity, gene_id) VALUES (?, ?)",
                    (ent.lower(), gene_id),
                )
            # Auto-link: find genes sharing 2+ entities with this gene
            self._auto_link_by_entity(gene_id, gene.promoter.entities, cur)

        # SPLADE sparse index (if enabled, non-blocking)
        if self._splade_enabled:
            try:
                from . import splade_backend
                sparse = splade_backend.encode(gene.content[:1000])
                # Inline the upsert without a separate commit
                cur.execute("DELETE FROM splade_terms WHERE gene_id = ?", (gene_id,))
                if sparse:
                    cur.executemany(
                        "INSERT INTO splade_terms (gene_id, term, weight) VALUES (?, ?, ?)",
                        [(gene_id, term, weight) for term, weight in sparse.items()],
                    )
            except Exception:
                log.debug("SPLADE indexing failed for gene %s", gene_id, exc_info=True)

        # Single atomic commit — gene + promoter + FTS5 + entity graph + SPLADE
        self.conn.commit()

        # Periodic WAL checkpoint to prevent data loss on crash
        # PASSIVE every 50 genes (~non-blocking), TRUNCATE every 500 (resets WAL)
        self._upsert_count += 1
        if self._upsert_count % 500 == 0:
            self.checkpoint("TRUNCATE")
        elif self._upsert_count % 50 == 0:
            self.checkpoint("PASSIVE")

        # Invalidate ΣĒMA cache (new gene may have embedding)
        if self._sema_cache is not None:
            self._sema_cache = None

        # Notify replication manager (if attached)
        if self._replication_mgr is not None:
            self._replication_mgr.notify_write()

        return gene_id

    # ── Fix 1: synonym expansion ────────────────────────────────────

    def _expand_terms(self, terms: List[str]) -> List[str]:
        expanded = set(t.lower() for t in terms)
        for t in terms:
            key = t.lower()
            if key in self.synonym_map:
                expanded.update(self.synonym_map[key])
        return list(expanded)

    # ── Authority boosts: distinguish "about X" from "mentions X" ──

    def _apply_authority_boosts(
        self,
        cur,
        gene_scores: Dict[str, float],
        query_terms: List[str],
    ) -> None:
        """
        Post-rank boosts that distinguish authoritative genes from tangential ones.

        Three signals:
          1. Source authority (+2.0): query term in source_id path
             — a file named BENCHMARK_NOTES.md answering "benchmark" is authoritative
          2. Domain primacy (+1.5): query term in top-3 promoter domains
             — primary domains = what the gene is ABOUT, not mentions
          3. Creation recency (+0.5): gene created in last 48 hours
             — bootstraps new concepts before they build co-activation history

        All boosts are additive to existing scores. Low risk — only raises
        the ceiling on already-scored genes, never adds new candidates.
        """
        if not gene_scores:
            return

        import time as _time
        now = _time.time()
        recency_window = 48 * 3600  # 48 hours in seconds

        gene_ids = list(gene_scores.keys())
        id_ph = ",".join("?" * len(gene_ids))
        lower_terms = [t.lower() for t in query_terms]

        # Fetch source_id, promoter, epigenetics for all candidates in one query
        rows = cur.execute(
            f"SELECT gene_id, source_id, promoter, epigenetics "
            f"FROM genes WHERE gene_id IN ({id_ph})",
            gene_ids,
        ).fetchall()

        for r in rows:
            gid = r["gene_id"]
            boost = 0.0

            # 1. Source authority: query term in path
            source = (r["source_id"] or "").lower()
            if source and any(t in source for t in lower_terms):
                boost += 2.0

            # 2. Domain primacy: query term in top-3 promoter domains
            try:
                prom = parse_promoter(r["promoter"]) if r["promoter"] else None
                if prom and prom.domains:
                    primary_domains = {d.lower() for d in prom.domains[:3]}
                    if any(t in primary_domains for t in lower_terms):
                        boost += 1.5
            except Exception:
                pass

            # 3. Creation recency: gene created in last 48h
            try:
                epi = parse_epigenetics(r["epigenetics"]) if r["epigenetics"] else None
                if epi and epi.created_at > 0:
                    age = now - epi.created_at
                    if 0 < age < recency_window:
                        boost += 0.5
            except Exception:
                pass

            if boost > 0:
                gene_scores[gid] += boost

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

        self._refresh_snapshot()  # See latest WAL state (external thinning, deletes)
        cur = self.read_conn.cursor()  # Read path — avoids WAL lock contention
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

        # ── Tier 3: FTS5 content search (weight 3.0) ───────────────
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
                            # Normalize: -rank gives positive, cap at 15
                            fts_score = min(-fts_ranks[gid], 15.0) * 3.0
                            gene_scores[gid] = gene_scores.get(gid, 0) + fts_score
                except Exception:
                    log.warning("FTS5 query failed", exc_info=True)

        # ── Tier 3.5: SPLADE sparse retrieval (weight 3.5) ─────────
        if self._splade_enabled:
            try:
                from . import splade_backend
                # Check if splade_terms table exists
                has_table = cur.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='splade_terms'"
                ).fetchone()[0]
                if has_table:
                    query_text = " ".join(query_terms)
                    query_sparse = splade_backend.encode(query_text)
                    splade_hits = splade_backend.query_splade(self.read_conn, query_sparse, limit=limit * 2)
                    for gid, score in splade_hits:
                        # Normalize SPLADE score to be comparable with other tiers
                        splade_score = min(score, 20.0) * 3.5 / 20.0  # Cap at 3.5
                        gene_scores[gid] = gene_scores.get(gid, 0) + splade_score
            except Exception:
                log.warning("SPLADE retrieval failed", exc_info=True)

        # ── Tier 4: ΣĒMA semantic retrieval + re-ranking ───────────────
        # Two modes:
        #   A) Boost existing candidates (when Tiers 1-3.5 have candidates)
        #   B) Add new candidates via vector scan (when pool is too small)
        if self._sema_codec is not None:
            try:
                query_text = " ".join(query_terms)
                query_vec = self._sema_codec.encode(query_text)
                top_score = max(gene_scores.values()) if gene_scores else 0

                # Mode A: Boost existing candidates when confidence is weak
                if gene_scores and top_score < 20.0:
                    existing_ids = list(gene_scores.keys())
                    id_ph = ",".join("?" * len(existing_ids))
                    sema_rows = cur.execute(
                        f"SELECT gene_id, embedding FROM genes "
                        f"WHERE gene_id IN ({id_ph}) AND embedding IS NOT NULL",
                        existing_ids,
                    ).fetchall()

                    if sema_rows:
                        candidates_sema = []
                        for r in sema_rows:
                            try:
                                vec = json_loads(r["embedding"])
                                if isinstance(vec, list) and len(vec) == 20:
                                    candidates_sema.append((r["gene_id"], vec))
                            except Exception:
                                continue

                        if candidates_sema:
                            nearest = self._sema_codec.nearest(
                                query_vec, candidates_sema, k=len(candidates_sema),
                            )
                            for gid, sim in nearest:
                                if sim > 0.3:
                                    boost_scale = max(0.5, 1.0 - top_score / 40.0)
                                    gene_scores[gid] += sim * 2.0 * boost_scale

                # Mode B: Add new candidates when pool is undersized
                # Uses pre-materialized numpy cache for fast cosine scan
                # instead of deserializing 7K JSON blobs per query.
                if len(gene_scores) < limit // 2:
                    # Build cache on first use (lazy init)
                    if self._sema_cache is None:
                        self._build_sema_cache()

                    if self._sema_cache is not None:
                        try:
                            import numpy as np
                            cache = self._sema_cache
                            q = np.array(query_vec, dtype=np.float32)
                            q_norm = np.linalg.norm(q)
                            if q_norm > 1e-8:
                                q = q / q_norm
                                # Batch cosine similarity: (N,20) @ (20,) → (N,)
                                sims = cache["matrix"] @ q
                                # Mask already-scored genes
                                existing = set(gene_scores.keys())
                                fill_count = limit - len(gene_scores)
                                # Get top-k indices
                                top_idx = np.argsort(sims)[::-1]
                                added = 0
                                for idx in top_idx:
                                    if added >= fill_count:
                                        break
                                    gid = cache["gene_ids"][idx]
                                    sim = float(sims[idx])
                                    if gid in existing:
                                        continue
                                    if sim > 0.4:
                                        gene_scores[gid] = sim * 3.0
                                        added += 1
                        except ImportError:
                            pass  # numpy not available
            except Exception:
                log.debug("ΣĒMA retrieval failed, continuing without")

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
            # IDF boost: rare terms get up to 3.0, common terms ~0.5
            # Capped at 3.0 (was 5.0) to reduce tangential rare-term over-boost.
            idf = _math.log(total_genes_est / term_freq) if term_freq > 0 else 0
            boost = min(idf * 1.5, 3.0)
            if boost > 1.0:
                anchor_genes = cur.execute(
                    "SELECT pi.gene_id FROM promoter_index pi "
                    "JOIN genes g ON pi.gene_id = g.gene_id "
                    "WHERE pi.tag_value = ? AND g.chromatin < ?",
                    (term, int(ChromatinState.HETEROCHROMATIN)),
                ).fetchall()
                for r in anchor_genes:
                    gene_scores[r["gene_id"]] = gene_scores.get(r["gene_id"], 0) + boost

        # ── Authority boosts: distinguish "about X" from "mentions X" ──
        self._apply_authority_boosts(cur, gene_scores, query_terms)

        # Expose scores for score-gated expression in context_manager
        self.last_query_scores = dict(gene_scores)

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

    # ── Entity graph: auto-link genes sharing entities ───────────────

    def _auto_link_by_entity(self, gene_id: str, entities: List[str], cur) -> None:
        """
        Find genes that share 2+ entities with this gene and create
        co-activation links. This builds the knowledge graph incrementally
        at ingestion time without any LLM calls.
        """
        if len(entities) < 2:
            return

        ent_lower = [e.lower() for e in entities[:15]]
        placeholders = ",".join("?" * len(ent_lower))

        # Find genes sharing entities (excluding self)
        rows = cur.execute(
            f"SELECT gene_id, COUNT(*) as shared "
            f"FROM entity_graph "
            f"WHERE entity IN ({placeholders}) AND gene_id != ? "
            f"GROUP BY gene_id "
            f"HAVING shared >= 2 "
            f"ORDER BY shared DESC "
            f"LIMIT 10",
            ent_lower + [gene_id],
        ).fetchall()

        for r in rows:
            peer_id = r["gene_id"]
            shared_count = r["shared"]
            confidence = min(shared_count / len(ent_lower), 1.0)
            # Store as COVER relation (overlapping topics)
            cur.execute(
                "INSERT OR REPLACE INTO gene_relations "
                "(gene_id_a, gene_id_b, relation, confidence, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (gene_id, peer_id, 5, confidence, time.time()),  # 5 = COVER
            )

    # ── Entity graph: expand retrieval by entity overlap ──────────

    def _expand_by_entity_graph(
        self, gene_ids: List[str], limit: int, cur
    ) -> List[str]:
        """
        Given retrieved gene IDs, find additional genes that share
        entities with them via 1-hop graph traversal.
        """
        if not gene_ids:
            return []

        id_ph = ",".join("?" * len(gene_ids))

        # Get entities of retrieved genes
        rows = cur.execute(
            f"SELECT DISTINCT entity FROM entity_graph WHERE gene_id IN ({id_ph})",
            gene_ids,
        ).fetchall()
        entities = [r["entity"] for r in rows]

        if not entities:
            return []

        ent_ph = ",".join("?" * len(entities))

        # Find genes sharing those entities (1-hop), excluding already retrieved
        neighbor_rows = cur.execute(
            f"SELECT gene_id, COUNT(*) as shared "
            f"FROM entity_graph "
            f"WHERE entity IN ({ent_ph}) AND gene_id NOT IN ({id_ph}) "
            f"GROUP BY gene_id "
            f"HAVING shared >= 2 "
            f"ORDER BY shared DESC "
            f"LIMIT ?",
            entities + gene_ids + [limit],
        ).fetchall()

        return [r["gene_id"] for r in neighbor_rows]

    # ── Co-activation expansion ─────────────────────────────────────

    def _expand_coactivated(self, genes: List[Gene], limit: int) -> List[Gene]:
        cur = self.conn.cursor()  # Always master — replicas may lag

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

        # Entity graph expansion (1-hop neighbor pull)
        if self._entity_graph_enabled:
            try:
                graph_ids = self._expand_by_entity_graph(
                    [g.gene_id for g in genes],
                    limit=5,
                    cur=cur,
                )
                additional_ids.update(gid for gid in graph_ids if gid not in existing_ids)
            except Exception:
                log.debug("Entity graph expansion failed", exc_info=True)

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

        # key_values may not exist in older databases before ALTER TABLE runs
        try:
            kv_raw = row["key_values"]
            key_values = json_loads(kv_raw) if kv_raw else []
        except (IndexError, KeyError):
            key_values = []

        return Gene(
            gene_id=row["gene_id"],
            content=row["content"] or "",
            # Heterochromatin-compressed genes have complement=NULL after
            # compress_to_heterochromatin(). Fall back to "" for Pydantic.
            complement=row["complement"] or "",
            codons=json_loads(row["codons"]) if row["codons"] else [],
            promoter=promoter,
            epigenetics=epigenetics,
            chromatin=chromatin,
            is_fragment=bool(row["is_fragment"]) if row["is_fragment"] is not None else False,
            embedding=json_loads(row["embedding"]) if row["embedding"] else None,
            source_id=row["source_id"],
            version=row["version"] if row["version"] is not None else 1,
            supersedes=row["supersedes"],
            key_values=key_values,
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
        self._refresh_snapshot()  # See latest WAL state
        cur = self.conn.cursor()  # Always master — stats must be authoritative

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

        # Compression tier distribution
        tier_counts = {0: 0, 1: 0, 2: 0}
        try:
            by_tier = cur.execute(
                "SELECT COALESCE(compression_tier, 0), COUNT(*) FROM genes "
                "GROUP BY COALESCE(compression_tier, 0)"
            ).fetchall()
            for r in by_tier:
                tier_counts[int(r[0])] = r[1]
        except Exception:
            pass  # Column may not exist yet on old schemas

        return {
            "total_genes": total,
            "open": chromatin_counts.get("OPEN", 0),
            "euchromatin": chromatin_counts.get("EUCHROMATIN", 0),
            "heterochromatin": chromatin_counts.get("HETEROCHROMATIN", 0),
            "total_chars_raw": total_raw,
            "total_chars_compressed": total_compressed,
            "compression_ratio": total_raw / max(total_compressed, 1),
            "compression_tiers": {
                "open_full": tier_counts[0],
                "euchromatin_summary": tier_counts[1],
                "heterochromatin_cold": tier_counts[2],
            },
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
        """Rebuild the FTS5 index from all genes. Returns count indexed.

        Includes source_id + promoter tags in the searchable content so
        tag-based knowledge survives rebuilds. At 100K+ genes this takes
        several seconds — prefer incremental sync for normal operation.
        """
        if not self._fts_available:
            log.warning("FTS5 not available — cannot rebuild")
            return 0

        import time as _time
        t0 = _time.time()
        cur = self.conn.cursor()

        # Clear and repopulate with enriched content
        cur.execute("DELETE FROM genes_fts")
        cur.execute(
            "INSERT INTO genes_fts(gene_id, content, complement) "
            "SELECT g.gene_id, "
            "  COALESCE(g.source_id,'') || ' ' || "
            "  COALESCE((SELECT GROUP_CONCAT(pi.tag_value, ' ') "
            "    FROM promoter_index pi WHERE pi.gene_id = g.gene_id), '') "
            "  || ' ' || g.content, "
            "  COALESCE(g.complement, '') "
            "FROM genes g"
        )
        self.conn.commit()
        count = cur.execute("SELECT COUNT(*) FROM genes_fts").fetchone()[0]
        elapsed = _time.time() - t0
        log.info("FTS5 index rebuilt: %d genes indexed in %.1fs", count, elapsed)
        return count

    # ── WAL refresh ────────────────────────────────────────────────

    def refresh(self) -> None:
        """Refresh WAL snapshot to see changes from external writers.

        Primary mechanism: commit() releases the implicit read transaction,
        forcing the next SELECT to start a new snapshot. This is the
        lightweight Tier 1 fix — no connection churn.

        Fallback: if the connection is in a bad state, close and reopen.
        """
        try:
            self._refresh_snapshot()
            # Verify the connection is healthy
            self.conn.execute("SELECT 1").fetchone()
        except Exception:
            log.warning("Snapshot refresh failed, reopening connection", exc_info=True)
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")

    # ── Close ───────────────────────────────────────────────────────

    def checkpoint(self, mode: str = "PASSIVE") -> None:
        """
        Force a WAL checkpoint to flush data from WAL to main database.

        Modes:
            PASSIVE  — non-blocking, skips frames held by readers (~5ms)
            FULL     — blocks until all frames are checkpointed
            TRUNCATE — like FULL, then truncates WAL file to zero bytes

        Call periodically during bulk ingest to prevent data loss on crash.
        Recommended cadence: PASSIVE every 50 genes, TRUNCATE every 500.
        """
        mode = mode.upper()
        if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            mode = "PASSIVE"
        try:
            self.conn.execute(f"PRAGMA wal_checkpoint({mode})")
            log.debug("WAL checkpoint (%s) completed", mode)
        except Exception:
            log.warning("WAL checkpoint (%s) failed", mode, exc_info=True)

    def vacuum(self) -> Dict[str, int]:
        """
        Reclaim free pages from the genome database.

        After large-scale operations (thinning, compaction, source-change
        repair) SQLite holds deleted pages until a VACUUM releases them.
        For a heavily-thinned genome this can be 30-50% of the file size.

        This method:
          1. Checkpoints the WAL so all data is in the main DB file
          2. Closes the long-lived connection (VACUUM needs exclusive access)
          3. Runs VACUUM via a dedicated connection
          4. Reopens the long-lived connection

        Returns:
            dict with before/after sizes in bytes, and bytes reclaimed.

        Warning: VACUUM is a full-file rewrite — it temporarily doubles disk
        usage and blocks all writers. Run during maintenance windows.
        """
        import os
        path = self.path
        before = os.path.getsize(path) if os.path.exists(path) else 0

        # Flush WAL and close the main connection
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.conn.commit()
        except Exception:
            log.warning("Pre-VACUUM WAL checkpoint failed", exc_info=True)
        try:
            self.conn.close()
        except Exception:
            pass

        # Run VACUUM on a fresh, autocommit connection
        try:
            vac_conn = sqlite3.connect(path)
            vac_conn.isolation_level = None  # autocommit — VACUUM requires it
            vac_conn.execute("VACUUM")
            vac_conn.close()
            log.info("VACUUM completed on %s", path)
        except Exception:
            log.warning("VACUUM failed", exc_info=True)

        # Reopen the long-lived connection
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")

        after = os.path.getsize(path) if os.path.exists(path) else 0
        reclaimed = before - after

        return {
            "before_bytes": before,
            "after_bytes": after,
            "reclaimed_bytes": reclaimed,
            "reclaimed_pct": round(reclaimed / max(before, 1) * 100, 1),
        }

    # ── Cold-storage compression (ΣĒMA-based chromatin tiers) ──────

    TIER_OPEN = 0           # Full fidelity — hot retrieval pool
    TIER_EUCHROMATIN = 1    # Summary + ΣĒMA — warm, reduced storage
    TIER_HETEROCHROMATIN = 2  # ΣĒMA + metadata only — cold, ~90% smaller

    def compute_density_score(self, gene: Gene) -> float:
        """
        Information density score for a gene. Higher = more valuable.

        Combines:
          - Entity/domain tag count (promoter richness)
          - Key-value extraction count (factual density)
          - Content length efficiency (short + rich > long + sparse)
          - Access count (usage signal from epigenetics)

        Uses a content-length floor (100 chars) in the denominator to
        prevent tiny-content genes from producing nonsensical tag-density
        scores of 20+. See _DENSITY_CONTENT_LENGTH_FLOOR above.
        """
        tag_count = len(gene.promoter.domains) + len(gene.promoter.entities)
        kv_count = len(gene.key_values) if gene.key_values else 0
        # Floor the content length so a 30-char gene with 5 tags doesn't
        # produce tag_density = 166 and break all downstream thresholds
        effective_len = max(len(gene.content), _DENSITY_CONTENT_LENGTH_FLOOR)

        # Normalize: tags per KB of (effective) content
        tag_density = tag_count / (effective_len / 1000.0)
        kv_density = kv_count / (effective_len / 1000.0)
        access = gene.epigenetics.access_count

        # Weighted combination (tag density dominates)
        score = (
            tag_density * 0.4
            + kv_density * 0.3
            + min(access / 10.0, 1.0) * 0.2
            + (1.0 if gene.complement and len(gene.complement) > 50 else 0.0) * 0.1
        )
        return score

    def apply_density_gate(self, gene: Gene) -> tuple[ChromatinState, str]:
        """
        Decide the chromatin state for a gene at ingest time.

        Returns (chromatin_state, reason) — reason is one of:
            "deny_list"        : source path matches structural deny list
            "low_score_hetero" : score below heterochromatin threshold
            "low_score_euchro" : score below euchromatin threshold
            "access_override"  : accessed >=5 times, gate bypassed
            "open"             : high score or unknown source, keep OPEN

        Never raises. Never touches the database. Pure decision function.

        The gate has three stages:
          1. Path deny list (fast-reject for structural noise)
          2. Access-count override (never demote frequently-used genes)
          3. Score-based demotion (tag + KV density with recalibrated thresholds)

        The access override runs BEFORE the score check so that a gene
        that's been touched multiple times can't be killed by a batch
        compact_genome sweep just because its static content is sparse.
        """
        # Stage 1: structural deny list
        if is_denied_source(gene.source_id):
            return ChromatinState.HETEROCHROMATIN, "deny_list"

        # Stage 2: access-count override (meaningful usage history wins)
        access = gene.epigenetics.access_count if gene.epigenetics else 0
        if access >= _DENSITY_ACCESS_OVERRIDE:
            return ChromatinState.OPEN, "access_override"

        # Stage 3: score-based demotion
        score = self.compute_density_score(gene)
        if score < _DENSITY_HETEROCHROMATIN_THRESHOLD:
            return ChromatinState.HETEROCHROMATIN, "low_score_hetero"
        if score < _DENSITY_EUCHROMATIN_THRESHOLD:
            return ChromatinState.EUCHROMATIN, "low_score_euchro"
        return ChromatinState.OPEN, "open"

    def compress_to_euchromatin(self, gene_id: str) -> bool:
        """
        Compress a gene to EUCHROMATIN tier: drop raw content, keep summary.

        Keeps: complement, codons, promoter, epigenetics, embedding, key_values
        Drops: content (replaced with pointer to source_id for unwinding)
        """
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT source_id, complement FROM genes WHERE gene_id = ?",
            (gene_id,),
        ).fetchone()
        if not row or not row["complement"]:
            return False

        cur.execute(
            "UPDATE genes SET content = ?, compression_tier = 1 WHERE gene_id = ?",
            (f"[COMPRESSED:euchromatin] source={row['source_id'] or 'unknown'}", gene_id),
        )
        self.conn.commit()
        log.debug("Compressed gene %s to EUCHROMATIN", gene_id)
        return True

    def compress_to_heterochromatin(self, gene_id: str) -> bool:
        """
        Compress a gene to HETEROCHROMATIN tier: keep only ΣĒMA + metadata.

        Keeps: gene_id, source_id, promoter, embedding, key_values
        Drops: content, complement, codons, SPLADE terms
        """
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT source_id FROM genes WHERE gene_id = ?",
            (gene_id,),
        ).fetchone()
        if not row:
            return False

        cur.execute(
            "UPDATE genes SET "
            "content = ?, complement = NULL, codons = '[]', "
            "chromatin = 2, compression_tier = 2 "
            "WHERE gene_id = ?",
            (f"[COMPRESSED:heterochromatin] source={row['source_id'] or 'unknown'}", gene_id),
        )
        # Remove SPLADE terms (no longer searchable via sparse retrieval)
        if self._splade_enabled:
            cur.execute("DELETE FROM splade_terms WHERE gene_id = ?", (gene_id,))
        # Remove from FTS5 index
        if self._fts_available:
            try:
                cur.execute("DELETE FROM genes_fts WHERE gene_id = ?", (gene_id,))
            except Exception:
                pass
        self.conn.commit()
        log.debug("Compressed gene %s to HETEROCHROMATIN", gene_id)
        return True

    def compact_genome(self, dry_run: bool = False) -> Dict:
        """
        Run a compaction sweep: apply the density gate to every currently-OPEN
        gene and demote those that fail it.

        Shares gate logic with ingest-time `apply_density_gate()`, so a gene
        that would be demoted by a fresh ingest will also be demoted by a
        retroactive sweep. The three stages are the same:
          1. Structural deny list (Steam, build artifacts, lockfiles, etc.)
          2. Access-count override (access_count >= 5 keeps gene OPEN)
          3. Score-based thresholds (< 0.5 hetero, < 1.0 euchro, else open)

        Only operates on genes currently at compression_tier = 0 (OPEN).
        Already-demoted genes are left alone.

        When ``dry_run=True``, returns the same stats without writing to
        the DB. Useful for previewing the impact before running the sweep
        against a live genome.

        Returns a dict with:
            scanned               : number of OPEN genes examined
            to_heterochromatin    : count that would go to HETEROCHROMATIN
            to_euchromatin        : count that would go to EUCHROMATIN
            kept_open             : count that would stay OPEN
            skipped_no_embedding  : count skipped because no ΣĒMA vector exists
            by_reason             : dict of reason counts (deny_list, low_score_*, etc.)
        """
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT gene_id, content, complement, codons, promoter, "
            "epigenetics, chromatin, embedding, source_id, key_values, "
            "compression_tier "
            "FROM genes WHERE compression_tier = 0"
        ).fetchall()

        stats = {
            "scanned": len(rows),
            "to_euchromatin": 0,
            "to_heterochromatin": 0,
            "kept_open": 0,
            "skipped_no_embedding": 0,
            "by_reason": {},
        }

        for r in rows:
            gene = self._compact_row_to_gene(r)
            if gene is None:
                continue

            new_state, reason = self.apply_density_gate(gene)
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1

            if new_state == ChromatinState.OPEN:
                stats["kept_open"] += 1
                continue

            # Deny-listed genes ALWAYS demote — with or without embedding.
            # The whole point of the deny list is "this is structural
            # noise we never want to retrieve again." compress_to_heterochromatin
            # only needs source_id (it strips content, complement, codons,
            # SPLADE, and FTS). The pre-existing no-embedding guard below
            # was a safety for score-based demotions, where a gene might
            # later turn out to be useful and we want the ΣĒMA vector
            # available for reactivation. That reasoning doesn't apply
            # to structural deny-list hits. Previously this guard cost
            # us ~40% of expected demotions on the 2026-04-10 genome
            # (3358 genes with no embeddings, mostly from pre-ΣĒMA
            # bulk ingests like ingest_steam.py).
            if reason == "deny_list":
                if not dry_run:
                    self.compress_to_heterochromatin(gene.gene_id)
                stats["to_heterochromatin"] += 1
                continue

            # Score-based demotions keep the embedding guard — these
            # genes might turn out to be useful later, so we want the
            # ΣĒMA vector available for reactivation via cosine similarity.
            has_embedding = r["embedding"] is not None
            if not has_embedding:
                stats["skipped_no_embedding"] += 1
                stats["kept_open"] += 1
                continue

            if new_state == ChromatinState.EUCHROMATIN:
                if gene.complement and len(gene.complement) > 30:
                    if not dry_run:
                        self.compress_to_euchromatin(gene.gene_id)
                    stats["to_euchromatin"] += 1
                else:
                    # No summary available to preserve; fall through to
                    # heterochromatin which doesn't need one
                    if not dry_run:
                        self.compress_to_heterochromatin(gene.gene_id)
                    stats["to_heterochromatin"] += 1
            elif new_state == ChromatinState.HETEROCHROMATIN:
                if not dry_run:
                    self.compress_to_heterochromatin(gene.gene_id)
                stats["to_heterochromatin"] += 1

        if not dry_run:
            self.checkpoint("PASSIVE")

        log.info(
            "Compaction sweep (%s): scanned=%d open=%d euchromatin=%d heterochromatin=%d skipped_no_emb=%d",
            "dry-run" if dry_run else "applied",
            stats["scanned"], stats["kept_open"],
            stats["to_euchromatin"], stats["to_heterochromatin"],
            stats["skipped_no_embedding"],
        )
        return stats

    def _compact_row_to_gene(self, row) -> Optional[Gene]:
        """Convert a compact database row to a Gene object. Returns None on error.

        Pre-existing bug fix (2026-04-10): the original version passed
        key_values=None when the DB column was NULL, but Gene.key_values
        is declared as list[str] and Pydantic rejects None. Any gene
        without extracted KVs (~35% of the 2026-04-10 genome sample)
        silently failed parsing, causing compact_genome to skip them.
        Now we pass [] as the empty-list fallback, matching how other
        list fields (codons) are handled.
        """
        try:
            return Gene(
                gene_id=row["gene_id"],
                content=row["content"] or "",
                complement=row["complement"],
                codons=json_loads(row["codons"]) if row["codons"] else [],
                promoter=parse_promoter(row["promoter"]),
                epigenetics=parse_epigenetics(row["epigenetics"]),
                chromatin=ChromatinState(row["chromatin"]),
                embedding=json_loads(row["embedding"]) if row["embedding"] else None,
                source_id=row["source_id"],
                key_values=json_loads(row["key_values"]) if row["key_values"] else [],
            )
        except Exception:
            log.debug("Failed to parse gene row %s", row["gene_id"], exc_info=True)
            return None

    def close(self) -> None:
        self.checkpoint("TRUNCATE")  # Flush all WAL data before closing
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
        self.conn.close()

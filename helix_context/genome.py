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


# ── Path tokenization for the path_key_index retrieval layer ────────────
# Splits source_id on common path separators + common filename punctuation.
# Each token becomes a retrieval signal paired with the gene's key_values
# keys. A query like "what is the value of helix_port?" hits the index on
# path_token='helix' AND kv_key='port' → direct boost to the gene.
#
# No LLM, no manual project list, no re-ingest required — purely derived
# from source_id + CpuTagger-extracted key_values. When a new project
# ingests at /SomeDir/NewProject/..., the token "newproject" becomes a
# retrieval signal automatically.
_PATH_SPLIT_RE = re.compile(r"[\\/\-_.\s:]+")

# Tokens that appear on nearly every path and carry no discriminating
# signal. Keeping this list tiny on purpose — it's the only maintenance
# burden, and overflowing it would be throwing signal away. Subset
# chosen from the actual source_id distribution on the 2026-04-12 genome.
_PATH_NOISE_TOKENS = frozenset({
    "",
    # Drive / filesystem roots
    "f", "c", "d", "e", "g", "h",
    # Ubiquitous container dirs
    "projects", "project", "src", "lib", "app", "apps", "test", "tests",
    "docs", "doc", "main", "master", "common", "shared", "core", "util",
    "utils", "include", "includes", "public", "private", "tmp", "temp",
    "cache", "dist", "build", "bin", "obj", "x64", "x86", "arm64", "debug",
    "release", "data", "file", "files", "assets", "resources", "scripts",
    "config", "configs", "node_modules", "pycache", "__pycache__",
    # Filename-level noise
    "init", "index", "new", "old", "copy", "readme", "license",
    # Extensions that appear as tokens after splitting
    "py", "js", "ts", "tsx", "jsx", "md", "txt", "json", "toml", "yaml",
    "yml", "ini", "cfg", "xml", "html", "css", "scss", "rs", "go",
    "java", "c", "cpp", "h", "hpp", "sh", "bat", "ps1", "log",
})


def _kv_keys_from_list(kv_list) -> list:
    """Extract lowercased key names from a key_values List[str] of 'key=value'.

    Gene.key_values is stored as ["port=8080", "model=llava", ...] — each
    entry is a 'key=value' string. This helper pulls out just the key side,
    lowercased, deduped, skipping empties.

    Works transparently on dict inputs too for forward-compat — if a future
    gene schema changes key_values to Dict[str, str], this still returns
    the right keys.
    """
    if not kv_list:
        return []
    # dict path (future-proof)
    if isinstance(kv_list, dict):
        return [k.lower() for k in kv_list.keys() if k]
    # list-of-strings path (current schema)
    out = []
    seen = set()
    for entry in kv_list:
        if not isinstance(entry, str):
            continue
        # Split on the FIRST '=' only — values can contain '='
        if "=" not in entry:
            continue
        key = entry.split("=", 1)[0].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def path_tokens(source_id: Optional[str]) -> set:
    """Extract retrieval-signal tokens from a file path.

    Splits on path separators and common filename punctuation, lowercases
    everything, drops single-char tokens and generic noise (see
    _PATH_NOISE_TOKENS). The result is the set of tokens that meaningfully
    identify this gene's provenance for compound-lookup retrieval.

    Examples:
        "F:/Projects/helix-context/helix_context/config.py"
          → {"helix-context", "helix_context", "helix", "context"}

        "F:/SteamLibrary/steamapps/common/Hades II/content/maps.lua"
          → {"steamlibrary", "steamapps", "hades", "ii", "content", "maps"}

        "F:/Projects/CosmicTasha/src/components/Hero.tsx"
          → {"cosmictasha", "components", "hero"}

    Exposed as a module-level function so tests, backfill scripts, and
    retrieval code can reuse it without constructing a Genome.
    """
    if not source_id:
        return set()
    out = set()
    # First pass: raw split on separators + punctuation
    for tok in _PATH_SPLIT_RE.split(source_id):
        t = tok.lower()
        if len(t) <= 1 or t in _PATH_NOISE_TOKENS:
            continue
        out.add(t)
        # Sub-split on interior hyphens/underscores to surface "helix"
        # from "helix-context" and "helix_context". Keeps the compound
        # form too so direct matches still work.
        if "-" in t or "_" in t:
            for sub in re.split(r"[-_]+", t):
                s = sub.lower()
                if len(s) > 1 and s not in _PATH_NOISE_TOKENS:
                    out.add(s)
    return out


# Thresholds for the score-based gate. Calibrated against the 2026-04-10
# noise-diluted genome (8,063 genes, ~42% structural noise). See
# scripts/simulate_density_gate_v2.py for the empirical basis.
_DENSITY_HETEROCHROMATIN_THRESHOLD = 0.50
_DENSITY_EUCHROMATIN_THRESHOLD = 1.00
_DENSITY_CONTENT_LENGTH_FLOOR = 100  # chars — prevents tiny-content score explosion
_DENSITY_ACCESS_OVERRIDE = 5         # access_count >= this keeps gene OPEN regardless

# Working-set inference (Phase 1 slice 2 of the 8D dimensional roadmap).
# A gene with at least _DENSITY_RATE_MIN_HITS accesses in the last
# _DENSITY_RATE_WINDOW seconds is considered "actively used right now"
# and gets the OPEN override regardless of static density score. The
# rate signal is sharper than the monotonic _DENSITY_ACCESS_OVERRIDE
# because it distinguishes "hot last hour" from "hot once a year ago" —
# the monotonic counter conflates them. Genes with empty recent_accesses
# buffers (legacy genes that pre-date Phase 1, or freshly ingested
# genes that haven't been touched yet) fall through to the monotonic
# fallback path, preserving backward compatibility.
#
# Reference: ~/.helix/shared/handoffs/2026-04-11_8d_dimensional_roadmap.md
_DENSITY_RATE_WINDOW = 3600.0   # 1-hour window
_DENSITY_RATE_MIN_HITS = 3      # ≥3 accesses in the window → override


class Genome:
    """SQLite-backed gene storage with promoter-tag retrieval."""

    def __init__(
        self,
        path: str,
        synonym_map: Optional[Dict[str, List[str]]] = None,
        sema_codec=None,
        splade_enabled: bool = False,
        entity_graph: bool = False,
        sr_enabled: bool = False,
        sr_gamma: float = 0.85,
        sr_k_steps: int = 4,
        sr_weight: float = 1.5,
        sr_cap: float = 3.0,
        seeded_edges_enabled: bool = False,
    ):
        self.path = path
        self.synonym_map = synonym_map or {}
        self._sema_codec = sema_codec  # Optional SemaCodec for Tier 4 retrieval
        self._replication_mgr = None  # Set by set_replication_manager()
        self._splade_enabled = splade_enabled
        self._entity_graph_enabled = entity_graph
        # Tier 5.5 Successor Representation (Sprint 2, Stachenfeld 2017).
        self._sr_enabled = sr_enabled
        self._sr_gamma = sr_gamma
        self._sr_k_steps = sr_k_steps
        self._sr_weight = sr_weight
        self._sr_cap = sr_cap
        # Sprint 4 — Hebbian seeded-edge evidence accumulation.
        self._seeded_edges_enabled = seeded_edges_enabled

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
        # Per-tier score breakdown for the last query: {gene_id: {tier_name: score}}.
        # Populated alongside last_query_scores in query_genes(). Lets the bench /
        # profiler see which retrieval signals fired (and how strongly) for each
        # candidate gene — turns the lane graph into a measurable activation matrix.
        # See benchmarks/bench_skill_activation.py and docs/PIPELINE_LANES.md.
        self.last_tier_contributions: Dict[str, Dict[str, float]] = {}
        self._sema_cache: Optional[Dict] = None  # Pre-materialized ΣĒMA vectors (hot tier)
        self._cold_sema_cache: Optional[Dict] = None  # Pre-materialized ΣĒMA vectors (cold tier, C.2)
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

        # path_key_index: compound (path_token, kv_key) → gene_id lookup for
        # fast retrieval on template queries like "what is the value of
        # helix_port?" where the answer gene has the key "port" and lives
        # under a path containing "helix-context". Populated at ingest from
        # source_id tokenization + key_values.keys(). Auto-applies to every
        # gene that has a source_id + extracted key_values — no LLM, no
        # manual tagging, no project bucket list to maintain.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS path_key_index (
            path_token TEXT NOT NULL,
            kv_key     TEXT NOT NULL,
            gene_id    TEXT NOT NULL,
            PRIMARY KEY (path_token, kv_key, gene_id)
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pki_lookup "
            "ON path_key_index(path_token, kv_key)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pki_gene "
            "ON path_key_index(gene_id)"
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

        Implements the 4-layer federated identity model (see
        docs/FEDERATION_LOCAL.md):
            - orgs:              top-level tenant (org/team)
            - parties:           devices (PCs) belonging to an org
            - participants:      humans (users) on a device
            - agents:            AI personas working on a user's behalf
            - gene_attribution:  4-axis attribution row per gene

        Each layer is independently queryable so we can answer
        "what did Laude on gandalf, on max's behalf, in SwiftWing21, do?"
        with a single composite filter.

        Schema is additive: pre-2026-04-12 databases auto-upgrade via
        IF NOT EXISTS table creates and ALTER ADD COLUMN for new fields
        on existing tables. Existing attribution rows keep their party_id +
        participant_id and acquire NULL org_id/agent_id (interpreted as
        "default org, manual ingest" by the resolver).
        """
        # ── Layer 1: orgs (top-level tenant) ────────────────────────
        # Sits above parties — devices belong to an org. For solo-dev
        # this defaults to "local" so existing single-tenant flows
        # remain semantically valid without explicit org assignment.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orgs (
            org_id         TEXT PRIMARY KEY,
            display_name   TEXT NOT NULL,
            trust_domain   TEXT NOT NULL DEFAULT 'local',
            created_at     REAL NOT NULL,
            metadata       TEXT
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_orgs_trust_domain "
            "ON orgs(trust_domain)"
        )
        # Seed the default 'local' org so trust-on-first-use writes have
        # a foreign-key target without needing an explicit register call.
        cur.execute(
            "INSERT OR IGNORE INTO orgs "
            "(org_id, display_name, trust_domain, created_at) "
            "VALUES ('local', 'Local Org (default)', 'local', ?)",
            (time.time(),),
        )

        cur.execute("""
        CREATE TABLE IF NOT EXISTS parties (
            party_id      TEXT PRIMARY KEY,
            display_name  TEXT NOT NULL,
            trust_domain  TEXT NOT NULL DEFAULT 'local',
            created_at    REAL NOT NULL,
            metadata      TEXT
        )
        """)
        # Add org_id column to existing parties table (4-layer extension).
        # Devices belong to an org. NULL = legacy row without org link;
        # treat as "local" org at query/resolver time.
        try:
            cur.execute(
                "ALTER TABLE parties ADD COLUMN org_id TEXT "
                "REFERENCES orgs(org_id)"
            )
        except sqlite3.OperationalError:
            pass  # column already exists — schema is idempotent

        # parties.timezone — IANA name (e.g., "America/Los_Angeles"),
        # NOT the offset. The IANA database handles DST transitions,
        # historical rule changes, and post-DST policy shifts cleanly.
        # Storing the offset directly would mean a device's identity
        # silently bifurcates twice a year — see docs/FEDERATION_LOCAL.md.
        # NULL = unknown (legacy row pre-2026-04-12); treat as UTC.
        try:
            cur.execute("ALTER TABLE parties ADD COLUMN timezone TEXT")
        except sqlite3.OperationalError:
            pass
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

        # ── Layer 4: agents (AI personas under a participant) ───────
        # An agent is the AI persona doing the work on behalf of a human
        # participant: "laude", "taude", "raude", "claude-code", "gemini",
        # "gpt-4". One human can drive many agents; one agent kind can
        # be invoked by many humans across orgs. The (participant_id,
        # handle) pair is unique — same human, same agent name = same row.
        # NULL agent_id at attribution time means "manual ingest" (no AI
        # involvement), which is its own meaningful signal.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id        TEXT PRIMARY KEY,
            participant_id  TEXT NOT NULL
                            REFERENCES participants(participant_id) ON DELETE CASCADE,
            handle          TEXT NOT NULL,
            kind            TEXT,             -- "claude-code", "gemini", "gpt-4", ...
            created_at      REAL NOT NULL,
            last_seen_at    REAL,
            metadata        TEXT,
            UNIQUE (participant_id, handle)
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_participant "
            "ON agents(participant_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_handle "
            "ON agents(handle)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_kind "
            "ON agents(kind)"
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
        # 4-layer attribution columns added 2026-04-12. Denormalized for
        # fast filter without joins; they should match the resolved
        # (party.org_id, agent_id-via-participant) but we store them on
        # the row so historical queries don't need expensive joins, and
        # so re-parented entities (party moved to a new org) preserve
        # the original write-time identity.
        try:
            cur.execute(
                "ALTER TABLE gene_attribution ADD COLUMN org_id TEXT "
                "REFERENCES orgs(org_id)"
            )
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute(
                "ALTER TABLE gene_attribution ADD COLUMN agent_id TEXT "
                "REFERENCES agents(agent_id) ON DELETE SET NULL"
            )
        except sqlite3.OperationalError:
            pass

        # authored_tz — IANA timezone name at the moment of write.
        # Together with authored_at (Unix epoch UTC) this gives full
        # forensic context: when AND where (regionally) a gene was
        # authored. Detects travel ("max wrote this from Berlin"),
        # DST drift (silent offset shift in the same party's authored_tz
        # over time), and supports cross-jurisdiction compliance queries.
        try:
            cur.execute(
                "ALTER TABLE gene_attribution ADD COLUMN authored_tz TEXT"
            )
        except sqlite3.OperationalError:
            pass

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_attribution_party_time "
            "ON gene_attribution(party_id, authored_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_attribution_participant_time "
            "ON gene_attribution(participant_id, authored_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_attribution_org_time "
            "ON gene_attribution(org_id, authored_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_attribution_agent_time "
            "ON gene_attribution(agent_id, authored_at DESC)"
        )

        # hitl_events — per-session HITL pause log, added 2026-04-11 following
        # laude's HITL observation handoff and raude's M1 discriminating test.
        # The chat-channel signal columns (operator_tone_uncertainty, etc) are
        # deliberately broad because the M1 finding ruled out genome-mediated
        # propagation of the HITL effect — the mechanism lives in the chat
        # channel and must be instrumented there. See handoff
        # ~/.helix/shared/handoffs/2026-04-11_hitl_observation.md
        cur.execute("""
        CREATE TABLE IF NOT EXISTS hitl_events (
            event_id                   TEXT PRIMARY KEY,
            party_id                   TEXT NOT NULL
                                       REFERENCES parties(party_id),
            participant_id             TEXT
                                       REFERENCES participants(participant_id) ON DELETE SET NULL,
            ts                         REAL NOT NULL,

            pause_type                 TEXT NOT NULL,
            task_context               TEXT,
            resolved_without_operator  INTEGER NOT NULL DEFAULT 0,

            operator_tone_uncertainty  REAL,
            operator_risk_keywords     TEXT,
            time_since_last_risk_event REAL,
            recoverability_signal      TEXT,

            genome_total_genes         INTEGER,
            genome_hetero_count        INTEGER,
            cold_cache_size            INTEGER,

            metadata                   TEXT
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_hitl_party_time "
            "ON hitl_events(party_id, ts DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_hitl_participant_time "
            "ON hitl_events(participant_id, ts DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_hitl_pause_type "
            "ON hitl_events(pause_type)"
        )

        # ── CWoLa label log (STATISTICAL_FUSION §C2) ─────────────────
        # Captures (tier_features, session, party, query) per retrieval
        # so the Sprint 3 CWoLa trainer can bucket into A (accepted) vs
        # B (re-queried within 60s) and train a classifier from those
        # two unlabeled mixtures. Metodiev/Nachman/Thaler 2017,
        # arXiv:1708.02949 — the factorised labels are not needed.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cwola_log (
            retrieval_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                 REAL    NOT NULL,
            session_id         TEXT,
            party_id           TEXT,
            query              TEXT,
            tier_features      TEXT,     -- JSON: {tier_name: score}
            top_gene_id        TEXT,
            bucket             TEXT,     -- 'A' (accepted) | 'B' (re-queried) | NULL (pending)
            bucket_assigned_at REAL,
            requery_delta_s    REAL      -- seconds to the next same-session query (NULL if none within 60s)
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_cwola_session_time "
            "ON cwola_log(session_id, ts)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_cwola_bucket "
            "ON cwola_log(bucket)"
        )

        # ── Sprint 4: seeded-edge provenance + Hebbian counters ──────
        # Table was previously created lazily inside
        # store_cymatics_harmonic_links; move to init so seeded_edges.py
        # and the Sprint 4 update hook can assume it exists. Pre-Sprint-4
        # databases get the columns added via the ALTER path in
        # store_cymatics_harmonic_links; new databases get them here.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS harmonic_links (
            gene_id_a  TEXT NOT NULL,
            gene_id_b  TEXT NOT NULL,
            weight     REAL NOT NULL,
            updated_at REAL NOT NULL,
            source     TEXT NOT NULL DEFAULT 'co_retrieved',
            co_count   INTEGER NOT NULL DEFAULT 0,
            miss_count REAL NOT NULL DEFAULT 0.0,
            created_at REAL,
            PRIMARY KEY (gene_id_a, gene_id_b)
        )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_harmonic_source "
            "ON harmonic_links(source)"
        )
        # Bidirectional lookup index — critical for sr_boost's bulk
        # neighbour query (WHERE gene_id_a IN (...) OR gene_id_b IN (...)).
        # Without this, SR on a 191K-edge graph times out at 30s on a
        # multi-hop frontier. Discovered via the 2026-04-13 staged A/B.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_harmonic_b "
            "ON harmonic_links(gene_id_b)"
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
        """Mark hot-tier cache stale — rebuilt on next Mode B query."""
        self._sema_cache = None

    # ── Cold-tier ΣĒMA retrieval (C.2, 2026-04-10) ─────────────────────
    #
    # The hot-tier retrieval paths all filter `WHERE chromatin <
    # HETEROCHROMATIN`, so heterochromatin genes are invisible to normal
    # /context queries. C.1 made compress_to_heterochromatin non-destructive
    # so the underlying content/complement/codons are preserved. This
    # block adds the opt-in retrieval path that consults cold-tier genes
    # via ΣĒMA cosine similarity and returns them with content restored.
    #
    # Design notes:
    #   - Separate cache from the hot-tier _sema_cache so hot queries have
    #     zero overhead from cold-tier capability.
    #   - Lazy build on first use. Invalidated on any upsert.
    #   - Requires numpy for batched cosine similarity (falls back to
    #     empty result if unavailable, matching hot-tier Mode B behavior).
    #   - Requires a SemaCodec attached to the Genome instance (for
    #     encoding the query text). If no codec, returns empty.
    #   - Callers must explicitly request cold-tier retrieval — it is
    #     never invoked implicitly from query_genes(). The wiring into
    #     context_manager is a follow-up (C.2-wire) and is gated behind
    #     a helix.toml config flag.

    def _build_cold_sema_cache(self) -> None:
        """Build the heterochromatin ΣĒMA vector cache for fast cosine scans.

        Scans all genes at ``chromatin = HETEROCHROMATIN`` that still have
        an embedding (ΣĒMA vector). Normalizes and stacks them into a
        dense numpy matrix for batched cosine similarity at query time.
        """
        try:
            import numpy as np
        except ImportError:
            log.debug("numpy not available, cold ΣĒMA cache disabled")
            return

        cur = self.read_conn.cursor()
        rows = cur.execute(
            "SELECT gene_id, embedding FROM genes "
            "WHERE embedding IS NOT NULL AND chromatin = ?",
            (int(ChromatinState.HETEROCHROMATIN),),
        ).fetchall()

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
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            matrix = matrix / norms
            self._cold_sema_cache = {"gene_ids": gene_ids, "matrix": matrix}
            log.info("Cold ΣĒMA cache built: %d vectors (%d KB)",
                     len(gene_ids), matrix.nbytes // 1024)
        else:
            self._cold_sema_cache = None

    def invalidate_cold_sema_cache(self) -> None:
        """Mark cold-tier cache stale — rebuilt on next query_cold_tier call."""
        self._cold_sema_cache = None

    def query_cold_tier(
        self,
        query_text: str,
        k: int = 3,
        min_cosine: float = 0.15,
    ) -> List[Gene]:
        """Search heterochromatin-tier genes by ΣĒMA cosine similarity.

        Cold-tier retrieval is opt-in — normal ``query_genes()`` does not
        consult this path. Use when a query is known to target archived
        knowledge, or as a fallthrough when hot-tier results are empty or
        too sparse to answer confidently.

        Parameters
        ----------
        query_text : str
            Natural-language query to encode via the attached SemaCodec.
        k : int
            Maximum number of cold-tier genes to return. Defaults to 3 —
            cold retrieval should be a precision tool, not a dump.
        min_cosine : float
            Similarity floor in 20-dim ΣĒMA space. Matches below this
            are discarded. Defaults to 0.25 — ΣĒMA's 20-dim projection
            is sparse by design, so typical close-paraphrase pairs score
            around 0.15–0.30 in cosine. The existing hot-tier Mode A/B
            paths use 0.3/0.4 as meaningful thresholds (see
            ``query_genes`` Tier 4). 0.25 here is slightly more
            permissive because the cold path is only reached when hot
            results are already thin — better to surface a weak match
            than nothing.

        Returns
        -------
        list[Gene]
            Up to ``k`` heterochromatin genes with full content restored,
            sorted by cosine similarity descending. Each gene's
            ``chromatin`` field will still show HETEROCHROMATIN — the
            caller is responsible for deciding whether to promote the
            gene back to OPEN based on the retrieval event (e.g., by
            updating access_count and letting a future sweep reconsider it).

        Returns empty list when any precondition fails:
            - No SemaCodec attached (self._sema_codec is None)
            - numpy unavailable
            - No heterochromatin genes with embeddings in the genome
            - No matches clear the min_cosine threshold
        """
        if self._sema_codec is None:
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        if self._cold_sema_cache is None:
            self._build_cold_sema_cache()
        if self._cold_sema_cache is None:
            return []  # Nothing in the cold tier

        try:
            query_vec = self._sema_codec.encode(query_text)
            q = np.array(query_vec, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-8:
                return []
            q = q / q_norm

            cache = self._cold_sema_cache
            sims = cache["matrix"] @ q  # (N,20) @ (20,) → (N,)

            # Sort descending, filter by threshold, take top-k
            top_idx = np.argsort(sims)[::-1]
            selected_ids: List[str] = []
            selected_sims: Dict[str, float] = {}
            for idx in top_idx:
                sim = float(sims[idx])
                if sim < min_cosine:
                    break  # Sorted descending — once below threshold, stop
                gid = cache["gene_ids"][idx]
                selected_ids.append(gid)
                selected_sims[gid] = sim
                if len(selected_ids) >= k:
                    break

            if not selected_ids:
                return []

            # Fetch full Gene objects (content is preserved thanks to C.1)
            genes: List[Gene] = []
            for gid in selected_ids:
                gene = self.get_gene(gid)
                if gene is not None:
                    genes.append(gene)

            # Expose similarity scores via last_query_scores for the
            # caller (context_manager uses this for tier-budget decisions)
            for gid, sim in selected_sims.items():
                self.last_query_scores[gid] = sim

            return genes
        except Exception:
            log.debug("cold-tier ΣĒMA retrieval failed", exc_info=True)
            return []

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

        # path_key_index — compound (path_token, kv_key) → gene_id for
        # fast template-query retrieval ("what is the value of helix_port?"
        # → index hit on (helix, port) → this gene). Auto-derived from
        # source_id + already-extracted key_values; no LLM, no manual list.
        cur.execute("DELETE FROM path_key_index WHERE gene_id = ?", (gene_id,))
        if gene.source_id and gene.key_values:
            p_tokens = path_tokens(gene.source_id)
            kv_keys = _kv_keys_from_list(gene.key_values)
            if p_tokens and kv_keys:
                for pt in p_tokens:
                    for kk in kv_keys:
                        cur.execute(
                            "INSERT OR IGNORE INTO path_key_index "
                            "(path_token, kv_key, gene_id) VALUES (?, ?, ?)",
                            (pt, kk, gene_id),
                        )

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

        # Invalidate ΣĒMA caches (new gene may have embedding, and
        # chromatin state changes can reshuffle hot/cold tier membership)
        if self._sema_cache is not None:
            self._sema_cache = None
        if self._cold_sema_cache is not None:
            self._cold_sema_cache = None

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
        party_id: Optional[str] = None,
    ) -> List[Gene]:
        """
        Find genes matching the given promoter signals.

        Multi-tier retrieval:
            1. Exact promoter tag match (highest confidence)
            2. Prefix tag match — "server" matches "serverconfig" (medium)
            3. FTS5 content search — searches gene text directly (fallback)
            3.5 SPLADE sparse retrieval
            4. SEMA semantic retrieval + re-ranking
            5. Harmonic co-activation boost (mutual reinforcement)
            Tiebreaker: access-rate bonus for equal-scored genes

        When party_id is provided, Tiers 1-3 are filtered to genes
        attributed to that party, and attributed genes get a +0.5 bonus.

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

        # Per-tier contribution tracking (parallel to gene_scores).
        # Each accumulation point also writes the contribution to
        # tier_contrib[gid][tier_name]. Surfaced via last_tier_contributions
        # for the activation profiler bench (bench_skill_activation.py).
        tier_contrib: Dict[str, Dict[str, float]] = {}

        # ── party_id filter clause (reused across Tiers 1-3) ──────
        _party_filter = ""
        _party_params: list = []
        if party_id is not None:
            try:
                _has_attr_table = cur.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type='table' AND name='gene_attribution'"
                ).fetchone()[0]
            except Exception:
                _has_attr_table = False
            if _has_attr_table:
                _party_filter = (
                    " AND g.gene_id IN "
                    "(SELECT gene_id FROM gene_attribution WHERE party_id = ?)"
                )
                _party_params = [party_id]

        # ── Tier 0: path-key compound index (IDF-weighted) ─────────
        # Highest-confidence retrieval signal — a hit means the query
        # mentions both a path-token (project/module) AND a kv_key that
        # an indexed gene was tagged with at ingest. This catches
        # template queries like "what is the value of helix_port?" where
        # FTS5/SPLADE both miss because the query lacks domain context.
        #
        # CRITICAL: bonus is INVERSELY proportional to the (path_token,
        # kv_key) pair cardinality. Rare pairs like (helix, port) — only
        # 2-3 genes share — each get a strong boost (~+5). Common pairs
        # like (steamapps, url) — 3000+ genes share — get ~zero boost,
        # so they don't drown the signal. This is the standard IDF
        # idea applied to compound retrieval keys.
        #
        # Without IDF weighting, a query containing common terms like
        # "url" or "value" would dump +8 on thousands of false-positive
        # genes, regressing retrieval (empirically observed 12% -> 6%
        # on the 2026-04-12 KV-harvest bench before this fix).
        q_lower_tokens = [t.lower() for t in query_terms if t]
        if q_lower_tokens:
            try:
                # Group hits by (path_token, kv_key) to compute per-pair
                # cardinality. SQLite GROUP BY + COUNT does this in one
                # pass over the index.
                pt_ph = ",".join("?" * len(q_lower_tokens))
                kk_ph = ",".join("?" * len(q_lower_tokens))
                pki_sql = (
                    f"SELECT path_token, kv_key, gene_id FROM path_key_index "
                    f"WHERE path_token IN ({pt_ph}) AND kv_key IN ({kk_ph})"
                )
                pki_params = list(q_lower_tokens) + list(q_lower_tokens)
                if party_id is not None and _party_filter:
                    pki_sql = (
                        f"SELECT path_token, kv_key, gene_id FROM path_key_index "
                        f"WHERE path_token IN ({pt_ph}) "
                        f"AND kv_key IN ({kk_ph}) "
                        f"AND gene_id IN (SELECT gene_id FROM "
                        f"gene_attribution WHERE party_id = ?)"
                    )
                    pki_params = pki_params + [party_id]
                pki_hits = cur.execute(pki_sql, pki_params).fetchall()

                # Bucket: pair_count[(pt, kk)] = number of distinct genes
                # gene_pairs[gene_id] = list of (pt, kk) pairs that hit
                pair_count: Dict[tuple, int] = {}
                gene_pairs: Dict[str, list] = {}
                for r in pki_hits:
                    pt = r["path_token"]
                    kk = r["kv_key"]
                    gid = r["gene_id"]
                    pair_count[(pt, kk)] = pair_count.get((pt, kk), 0) + 1
                    gene_pairs.setdefault(gid, []).append((pt, kk))

                # Score each gene by sum of inverse-cardinality boosts
                # over all (pt, kk) pairs it matched on.
                #
                # Boost formula:
                #   per-pair bonus = PKI_BASE / max(pair_card, PKI_FLOOR)
                # where:
                #   PKI_BASE  = 10.0  (so a unique pair lands at +10)
                #   PKI_FLOOR =  2.0  (caps top-end at +5 for 2-gene pairs)
                # A pair with 100 genes contributes only +0.1 per gene —
                # essentially noise. A pair with 5 genes contributes +2.
                PKI_BASE = 10.0
                PKI_FLOOR = 2.0
                # Hard-skip pairs with cardinality > this — they're noise
                PKI_NOISE_CUTOFF = 200
                for gid, pairs in gene_pairs.items():
                    bonus = 0.0
                    for pair in pairs:
                        card = pair_count[pair]
                        if card > PKI_NOISE_CUTOFF:
                            continue  # too common, skip
                        bonus += PKI_BASE / max(card, PKI_FLOOR)
                    if bonus > 0:
                        # Cap total bonus to keep one runaway gene from
                        # saturating; 12.0 is roughly 3x the strongest
                        # single signal.
                        capped = min(bonus, 12.0)
                        gene_scores[gid] = gene_scores.get(gid, 0) + capped
                        tier_contrib.setdefault(gid, {})["pki"] = capped
            except Exception as exc:
                log.debug("path_key_index tier skipped: %s", exc)

        # ── Tier 1: exact promoter tag match (weight 3.0) ──────────
        placeholders = ",".join("?" * len(query_terms))
        rows = cur.execute(
            f"""
            SELECT g.gene_id, COUNT(pi.tag_value) AS match_count
            FROM genes g
            JOIN promoter_index pi ON g.gene_id = pi.gene_id
            WHERE pi.tag_value IN ({placeholders})
              AND g.chromatin < ?
              {_party_filter}
            GROUP BY g.gene_id
            """,
            (*query_terms, int(ChromatinState.HETEROCHROMATIN), *_party_params),
        ).fetchall()

        for r in rows:
            tag_score = r["match_count"] * 3.0
            gene_scores[r["gene_id"]] = tag_score
            tier_contrib.setdefault(r["gene_id"], {})["tag_exact"] = tag_score

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
              {_party_filter}
            GROUP BY g.gene_id
            """,
            (*prefix_params, int(ChromatinState.HETEROCHROMATIN), *_party_params),
        ).fetchall()

        for r in rows:
            gid = r["gene_id"]
            prefix_score = r["match_count"] * 1.5
            gene_scores[gid] = gene_scores.get(gid, 0) + prefix_score
            tier_contrib.setdefault(gid, {})["tag_prefix"] = prefix_score

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
                            f"SELECT gene_id FROM genes g "
                            f"WHERE g.gene_id IN ({id_ph}) AND g.chromatin < ?"
                            f" {_party_filter}",
                            (*fts_ids, int(ChromatinState.HETEROCHROMATIN), *_party_params),
                        ).fetchall()
                        valid_ids = {r["gene_id"] for r in valid}

                        for gid in fts_ids:
                            if gid not in valid_ids:
                                continue
                            # FTS5 rank is negative (lower = better match)
                            # Normalize: -rank gives positive, cap at 6.0
                            # (was 15*3=45 — drowned out tag matches at 3-9)
                            fts_score = min(-fts_ranks[gid], 6.0)
                            gene_scores[gid] = gene_scores.get(gid, 0) + fts_score
                            tier_contrib.setdefault(gid, {})["fts5"] = fts_score
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
                        tier_contrib.setdefault(gid, {})["splade"] = splade_score
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
                                    sema_boost = sim * 2.0 * boost_scale
                                    gene_scores[gid] += sema_boost
                                    tier_contrib.setdefault(gid, {})["sema_boost"] = sema_boost

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
                                        sema_new = sim * 3.0
                                        gene_scores[gid] = sema_new
                                        tier_contrib.setdefault(gid, {})["sema_cold"] = sema_new
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
                    f"SELECT pi.gene_id FROM promoter_index pi "
                    f"JOIN genes g ON pi.gene_id = g.gene_id "
                    f"WHERE pi.tag_value = ? AND g.chromatin < ?"
                    f" {_party_filter}",
                    (term, int(ChromatinState.HETEROCHROMATIN), *_party_params),
                ).fetchall()
                for r in anchor_genes:
                    gid = r["gene_id"]
                    gene_scores[gid] = gene_scores.get(gid, 0) + boost
                    # Anchor IDF can fire for multiple terms in same query; sum them.
                    tc = tier_contrib.setdefault(gid, {})
                    tc["lex_anchor"] = tc.get("lex_anchor", 0.0) + boost

        # ── Authority boosts: distinguish "about X" from "mentions X" ──
        self._apply_authority_boosts(cur, gene_scores, query_terms)

        # ── Tier 5: harmonic co-activation boost ──────────────────
        # For each candidate, add a score bonus from genes that are
        # harmonically linked to OTHER candidates (mutual reinforcement).
        # Weight: 1.0 per link, capped at 3.0 total bonus.
        if gene_scores:
            try:
                _has_harmonic = cur.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type='table' AND name='harmonic_links'"
                ).fetchone()[0]
            except Exception:
                _has_harmonic = False
            if _has_harmonic:
                try:
                    candidate_ids = list(gene_scores.keys())
                    cid_ph = ",".join("?" * len(candidate_ids))
                    harmonic_rows = cur.execute(
                        f"SELECT gene_id_a, gene_id_b, weight "
                        f"FROM harmonic_links "
                        f"WHERE gene_id_a IN ({cid_ph}) "
                        f"  AND gene_id_b IN ({cid_ph})",
                        (*candidate_ids, *candidate_ids),
                    ).fetchall()
                    harmonic_bonus: Dict[str, float] = {}
                    for hr in harmonic_rows:
                        for gid in (hr["gene_id_a"], hr["gene_id_b"]):
                            harmonic_bonus[gid] = min(
                                harmonic_bonus.get(gid, 0) + 1.0, 3.0,
                            )
                    for gid, bonus in harmonic_bonus.items():
                        gene_scores[gid] = gene_scores.get(gid, 0) + bonus
                        tier_contrib.setdefault(gid, {})["harmonic"] = bonus
                except Exception:
                    log.debug("Harmonic boost failed", exc_info=True)

        # ── Tier 5.5: Successor Representation boost ──────────────
        # Discounted future-occupancy over the co-activation graph.
        # Generalises Tier 5's 1-hop harmonic pull-forward to a
        # gamma-discounted k-step horizon. See helix_context/sr.py and
        # docs/FUTURE/SUCCESSOR_REPRESENTATION.md. Feature-flagged
        # (retrieval.sr_enabled) so it can A/B before promotion.
        if self._sr_enabled and gene_scores:
            try:
                from .sr import sr_boost
                sr_bonus = sr_boost(
                    self,
                    list(gene_scores.keys()),
                    gamma=self._sr_gamma,
                    k_steps=self._sr_k_steps,
                    weight=self._sr_weight,
                    cap=self._sr_cap,
                )
                for gid, bonus in sr_bonus.items():
                    gene_scores[gid] = gene_scores.get(gid, 0) + bonus
                    tier_contrib.setdefault(gid, {})["sr"] = bonus
            except Exception:
                log.debug("SR Tier 5.5 failed", exc_info=True)

        # ── Party attribution bonus (+0.5) ────────────────────────
        if party_id is not None and _party_filter and gene_scores:
            try:
                attr_ids_ph = ",".join("?" * len(gene_scores))
                attr_rows = cur.execute(
                    f"SELECT gene_id FROM gene_attribution "
                    f"WHERE party_id = ? AND gene_id IN ({attr_ids_ph})",
                    (party_id, *list(gene_scores.keys())),
                ).fetchall()
                for ar in attr_rows:
                    gid = ar["gene_id"]
                    gene_scores[gid] += 0.5
                    tier_contrib.setdefault(gid, {})["party_attr"] = 0.5
            except Exception:
                log.debug("Party attribution bonus failed", exc_info=True)

        # ── Access-rate tiebreaker ────────────────────────────────
        # Small bonus: score += 0.05 * min(rate * 3600, 5). Max 0.25.
        # Only a tiebreaker — breaks ties for genes with equal scores.
        if gene_scores:
            try:
                rate_ids = list(gene_scores.keys())
                rate_ph = ",".join("?" * len(rate_ids))
                epi_rows = cur.execute(
                    f"SELECT gene_id, epigenetics FROM genes "
                    f"WHERE gene_id IN ({rate_ph}) AND epigenetics IS NOT NULL",
                    rate_ids,
                ).fetchall()
                for er in epi_rows:
                    try:
                        epi = parse_epigenetics(er["epigenetics"])
                        rate = epi.access_rate(3600.0)
                        if rate > 0:
                            bonus = 0.05 * min(rate * 3600.0, 5.0)
                            gid = er["gene_id"]
                            gene_scores[gid] += bonus
                            tier_contrib.setdefault(gid, {})["access_rate"] = bonus
                    except Exception:
                        continue
            except Exception:
                log.debug("Access-rate tiebreaker failed", exc_info=True)

        # Expose scores + per-tier breakdown for score-gated expression in
        # context_manager + the activation profiler bench.
        self.last_query_scores = dict(gene_scores)
        self.last_tier_contributions = tier_contrib

        # Emit per-tier contribution telemetry (OTel — no-op when off).
        # One histogram observation per (tier, gene) pair; a single
        # counter tick per tier that fired at all, labelled by tier
        # name. Makes the bench_skill_activation heatmap live-observable
        # instead of a one-shot static file.
        try:
            from .telemetry import tier_contribution_histogram, tier_fired_counter
            hist = tier_contribution_histogram()
            cnt = tier_fired_counter()
            tiers_seen: set = set()
            for contribs in tier_contrib.values():
                for tier, score in contribs.items():
                    hist.record(float(score), {"tier": tier})
                    tiers_seen.add(tier)
            for tier in tiers_seen:
                cnt.add(1, {"tier": tier})
        except Exception:
            log.debug("tier telemetry emit failed", exc_info=True)

        # Sort by combined score, fetch top genes
        ranked_ids = sorted(gene_scores, key=gene_scores.get, reverse=True)[:limit]

        # ── Sprint 4: Hebbian evidence accumulation on seeded edges ───
        # Fire-and-forget update to harmonic_links so seeded / co_retrieved
        # rows accrue co_count (both endpoints in top-k) or miss_count
        # (one endpoint expressed, other in candidate pool but below the
        # cut — weighted by dense-rank distance to cutoff). Candidacy
        # gate: genes outside gene_scores are ignored (topical-orthogonal
        # queries should not punish the edge). Soft-fails — logger
        # hiccups never perturb the retrieval result.
        if self._seeded_edges_enabled and ranked_ids:
            try:
                from .seeded_edges import update_edge_evidence
                update_edge_evidence(
                    self, gene_scores, ranked_ids, max_genes=max_genes,
                )
            except Exception:
                log.debug("Hebbian edge update failed", exc_info=True)

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
            # Phase 1 slice 2 — populate the windowed access-rate buffer.
            # Bounded ring buffer of last 100 timestamps (~800 bytes).
            # The slice 1 schema added the field; this is where it gets
            # written. apply_density_gate consumes it via the rate signal.
            epi.recent_accesses.append(now)
            if len(epi.recent_accesses) > 100:
                epi.recent_accesses = epi.recent_accesses[-100:]
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

    # ── Harmonic weights (cymatics) ──────────────────────────────────

    def store_harmonic_weights(self, weights: List[Tuple[str, str, float]]) -> None:
        """Store weighted co-activation edges from cymatics spectral overlap.

        As of Sprint 4, edges carry provenance and Hebbian evidence counters:
          source            - 'seeded' | 'co_retrieved' | 'cwola_validated'
          co_count          - # times both endpoints co-expressed in a query
          miss_count        - fractional (dense-rank weighted) miss events
          created_at        - epoch seconds
        co_retrieved and cwola_validated promotion happens in update_edge_evidence().
        """
        if not weights:
            return
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS harmonic_links (
                gene_id_a  TEXT NOT NULL,
                gene_id_b  TEXT NOT NULL,
                weight     REAL NOT NULL,
                updated_at REAL NOT NULL,
                source     TEXT NOT NULL DEFAULT 'co_retrieved',
                co_count   INTEGER NOT NULL DEFAULT 0,
                miss_count REAL NOT NULL DEFAULT 0.0,
                created_at REAL,
                PRIMARY KEY (gene_id_a, gene_id_b)
            )
        """)
        # Best-effort ALTER for pre-Sprint-4 schemas (fails silently if
        # columns already present — SQLite has no IF NOT EXISTS for ADD COLUMN).
        for col, defn in (
            ("source", "TEXT NOT NULL DEFAULT 'co_retrieved'"),
            ("co_count", "INTEGER NOT NULL DEFAULT 0"),
            ("miss_count", "REAL NOT NULL DEFAULT 0.0"),
            ("created_at", "REAL"),
        ):
            try:
                cur.execute(f"ALTER TABLE harmonic_links ADD COLUMN {col} {defn}")
            except sqlite3.OperationalError:
                pass
        now = time.time()
        for a, b, w in weights:
            cur.execute(
                """INSERT INTO harmonic_links
                   (gene_id_a, gene_id_b, weight, updated_at, source, created_at)
                   VALUES (?, ?, ?, ?, 'co_retrieved', ?)
                   ON CONFLICT(gene_id_a, gene_id_b) DO UPDATE SET
                     weight = excluded.weight,
                     updated_at = excluded.updated_at""",
                (a, b, w, now, now),
            )
        self.conn.commit()

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
            "deny_list"           : source path matches structural deny list
            "low_score_hetero"    : score below heterochromatin threshold
            "low_score_euchro"    : score below euchromatin threshold
            "access_rate_override": active in the windowed-rate sense
                                    (Phase 1 slice 2 — preferred when the
                                    gene's recent_accesses buffer has data)
            "access_override"     : accessed >= _DENSITY_ACCESS_OVERRIDE
                                    times monotonically (legacy fallback for
                                    genes whose rate buffer is empty)
            "open"                : high score or unknown source, keep OPEN

        Never raises. Never touches the database. Pure decision function.

        The gate has three stages:
          1. Path deny list (fast-reject for structural noise)
          2. Access override (never demote frequently-used genes)
             2a. Windowed access-rate (preferred — sharper signal)
             2b. Monotonic access-count (legacy fallback for empty buffers)
          3. Score-based demotion (tag + KV density with recalibrated thresholds)

        Stages 2a and 2b run BEFORE the score check so that a gene
        that's been touched multiple times can't be killed by a batch
        compact_genome sweep just because its static content is sparse.

        Stage 2a is a strict improvement on Stage 2b: it distinguishes
        "actively used right now" from "popular at some point in the
        past." Stage 2b remains as the fallback because all genes that
        pre-date the slice 1 schema change have empty recent_accesses
        buffers and need the monotonic counter to remain a valid signal
        until they're touched again under the slice 2 wiring.
        """
        # Stage 1: structural deny list
        if is_denied_source(gene.source_id):
            return ChromatinState.HETEROCHROMATIN, "deny_list"

        # Stage 2a: windowed access-rate override (preferred)
        # Empty buffer → access_rate returns 0.0 → falls through to 2b cleanly.
        if gene.epigenetics:
            rate_threshold = _DENSITY_RATE_MIN_HITS / _DENSITY_RATE_WINDOW
            if gene.epigenetics.access_rate(_DENSITY_RATE_WINDOW) >= rate_threshold:
                return ChromatinState.OPEN, "access_rate_override"

        # Stage 2b: monotonic access-count override (legacy fallback)
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
        Move a gene to HETEROCHROMATIN tier (cold storage).

        As of 2026-04-10 (C.1 of B→C), this is **non-destructive**. The
        function only flips the ``chromatin`` and ``compression_tier``
        flags — it does NOT drop ``content``, ``complement``, ``codons``,
        SPLADE terms, or FTS5 index entries.

        Rationale: the chromatin flag + ``WHERE chromatin < HETEROCHROMATIN``
        filter on all hot-tier retrieval paths already excludes demoted
        genes from normal ``/context`` queries. Destroying the underlying
        content eliminated any possibility of cold-tier retrieval (via
        ΣĒMA cosine similarity on the retained embedding) actually
        returning useful data — you'd match the embedding but have
        nothing to show.

        With content preserved, the cold-tier retrieval path added in C.2
        can reactivate demoted genes on-demand for queries that specifically
        need them. The storage cost is modest — SPLADE terms and FTS5
        index entries are small per-gene — and the optional nature of
        cold-tier retrieval means hot queries are unaffected.

        Callers who explicitly want to reclaim disk space on a known-dead
        gene can call ``delete_gene()`` instead. Heterochromatin is now
        strictly a **tier flag**, not a destructive compression.

        Keeps: everything (content, complement, codons, SPLADE, FTS5)
        Flips: ``chromatin = 2``, ``compression_tier = 2``
        """
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT source_id FROM genes WHERE gene_id = ?",
            (gene_id,),
        ).fetchone()
        if not row:
            return False

        cur.execute(
            "UPDATE genes SET chromatin = 2, compression_tier = 2 "
            "WHERE gene_id = ?",
            (gene_id,),
        )
        self.conn.commit()
        # Gene moved hot → cold — both tier caches are now stale
        if self._sema_cache is not None:
            self._sema_cache = None
        if self._cold_sema_cache is not None:
            self._cold_sema_cache = None
        log.debug("Moved gene %s to HETEROCHROMATIN (non-destructive)", gene_id)
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

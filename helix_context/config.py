"""
Config — Epigenetic environment.

Loads helix.toml and exposes typed configuration for all modules.
Falls back to sensible defaults if no config file exists.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class RibosomeConfig:
    enabled: bool = False              # Master switch; false = ignore ribosome config/runtime
    model: str = "auto"
    base_url: str = "http://localhost:11434"
    timeout: float = 10.0
    keep_alive: str = "30m"     # How long Ollama keeps the ribosome model loaded
    warmup: bool = True         # Pre-load model on server start
    backend: str = "ollama"     # legacy placeholder; only "deberta" or "litellm" are honored when enabled
    claude_model: str = "claude-haiku-4-5-20251001"   # Claude model when backend="claude"
    claude_base_url: str = ""   # Proxy URL (e.g. Headroom at http://127.0.0.1:8787); "" = direct
    litellm_model: str = "gemini/gemini-2.5-flash"    # LiteLLM model string when backend="litellm"
    rerank_model_path: str = "training/models/rerank"
    splice_model_path: str = "training/models/splice"
    splice_threshold: float = 0.5
    nli_model_path: str = "training/models/nli"
    nli_splice_bonus: float = 0.15       # Prob bonus for entailment-linked codons
    nli_splice_penalty: float = 0.15     # Prob penalty for alternation-linked codons
    device: str = "auto"        # "auto", "cpu", "cuda"
    # Step 0 query-intent expansion fires ONE LLM call per novel query
    # (LRU-cached) upstream of the 12-tone retrieval stack. Flip to false
    # for a strictly LLM-free /context pipeline — the 12 tiers below still
    # run on raw query text + synonym map. See context_manager
    # _expand_query_intent.
    query_expansion_enabled: bool = True

    # ── Cost classification (W2-B) ─────────────────────────────────
    # Derived classification of the chosen backend's cost profile. Used
    # by /health and by a server-startup WARNING so operators are never
    # surprised by paid-API ribosome calls. Add new backends to the
    # appropriate tuple.
    _LOCAL_BACKENDS = ("ollama", "deberta")
    _PAID_API_BACKENDS = ("claude",)
    # litellm is paid OR local depending on the underlying model string;
    # classified separately by ``cost_class`` (see below).

    @property
    def normalized_backend(self) -> str:
        return str(self.backend or "").strip().lower()

    @property
    def effective_backend(self) -> str:
        """Return the backend Helix should actually run.

        Ribosome is opt-in. Legacy/default backends like ``ollama`` are kept
        in config for future reference but intentionally ignored unless a
        supported backend is selected explicitly.
        """
        if not self.enabled:
            return "disabled"
        if self.normalized_backend in ("litellm", "deberta"):
            return self.normalized_backend
        return "disabled"

    @property
    def cost_class(self) -> str:
        """Return one of ``disabled`` | ``local`` | ``api+free`` | ``api+paid``.

        - ``disabled``= ribosome is intentionally inactive.
        - ``local``   = runs on the operator's machine (Ollama / DeBERTa).
        - ``api+free``= remote API with no metered cost (none today).
        - ``api+paid``= remote API that bills per call (Claude direct, or
                        LiteLLM routed to a paid model string).

        litellm with a model that starts with ``ollama/`` is treated as
        local (the call goes to the local Ollama). Any other litellm
        model string is treated as paid.
        """
        b = self.effective_backend
        if b == "disabled":
            return "disabled"
        if b in self._LOCAL_BACKENDS:
            return "local"
        if b in self._PAID_API_BACKENDS:
            return "api+paid"
        if b == "litellm":
            return "local" if self.litellm_model.startswith("ollama/") else "api+paid"
        return "api+paid"  # unknown backend defaults to paid for safety

    @property
    def active_model(self) -> str:
        """Return the model string actually in use, given the backend."""
        b = self.effective_backend
        if b == "disabled":
            return "disabled"
        if b == "claude":
            return self.claude_model
        if b == "litellm":
            return self.litellm_model
        return self.model


@dataclass
class BudgetConfig:
    ribosome_tokens: int = 3000
    expression_tokens: int = 6000
    max_genes_per_turn: int = 8
    max_fingerprints_per_turn: int = 40
    splice_aggressiveness: float = 0.5
    decoder_mode: str = "full"  # "full"|"condensed"|"minimal"|"none"
    # Sprint 1 legibility pack (AI-consumer roadmap): emit a one-line
    # metadata header per gene in expressed_context — fired tiers,
    # confidence marker, short gene_id, compression ratio. See
    # helix_context/legibility.py. Default on; flip off to restore the
    # pre-Sprint-1 plain-dividers format (useful for bench A/B).
    legibility_enabled: bool = True
    # Sprint 2 session working-set register: track delivered genes per
    # session, elide repeats with a pointer stub so the consumer doesn't
    # pay full token cost for content it already holds. Dark on first
    # ship — flip to true in helix.toml to A/B. See session_delivery.py.
    session_delivery_enabled: bool = False


@dataclass
class GenomeConfig:
    path: str = "genome.db"
    compact_interval: float = 3600.0    # Seconds between source-change checks
    cold_start_threshold: int = 10      # Fix 3: genes needed before history stripping
    replicas: List[str] = field(default_factory=list)  # Read-only clone paths
    replica_sync_interval: int = 100    # Sync replicas every N inserts


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 11437
    upstream: str = "http://localhost:11434"
    upstream_timeout: float = 120.0     # Timeout for proxied requests to Ollama


@dataclass
class IngestionConfig:
    """Controls which backend encodes raw content into genes."""
    backend: str = "ollama"         # "ollama" | "cpu" | "hybrid"
    splade_enabled: bool = False    # Phase 2: SPLADE sparse expansion at index time
    rerank_model: str = ""          # Phase 3: pretrained cross-encoder HF model ID
    rerank_enabled: bool = False    # Phase 3: enable cross-encoder reranking
    colbert_enabled: bool = False   # Phase 4: ColBERT late interaction (optional)
    entity_graph: bool = False      # Phase 5: entity-based co-activation links


@dataclass
class ContextConfig:
    """Retrieval-time behavior for context_manager.

    Cold-tier knobs were added 2026-04-10 (C.2 of B->C). Cold-tier is the
    opt-in retrieval path that consults heterochromatin genes via SEMA
    cosine similarity, returning their preserved content (only possible
    after C.1 made compress_to_heterochromatin non-destructive).
    """
    cold_tier_enabled: bool = False         # Master opt-in for cold-tier fallthrough
    cold_tier_min_hot_genes: int = 0        # Fall through when hot returns <= this many genes (0 = only on empty)
    cold_tier_k: int = 3                    # Max cold-tier genes to retrieve per query
    cold_tier_min_cosine: float = 0.15      # SEMA cosine floor (sparse 20-dim — see Genome.query_cold_tier)
    fingerprint_mode_profile: str = "balanced"  # "fast" | "balanced" | "quality"


@dataclass
class CymaticsConfig:
    """Frequency-domain re_rank + splice (CPU math replaces LLM calls)."""
    enabled: bool = True                # Master switch
    n_bins: int = 256                   # Spectrum resolution (<2KB per spectrum)
    peak_width: float = 3.0             # Gaussian peak width (overridden by Q-factor)
    splice_threshold_scale: float = 0.7 # Maps splice_aggressiveness to resonance threshold
    use_embeddings: bool = False        # Use Gene.embedding when available
    harmonic_links: bool = True         # Compute weighted co-activation edges
    distance_metric: str = "cosine"     # "cosine" (weighted dot) | "w1" (Werman 1986 circular Wasserstein-1)


@dataclass
class SessionConfig:
    """CWoLa label-logger session/party defaults.

    Without these, clients that don't pass ``session_id`` / ``party_id`` in
    the /context POST body get logged with NULL in cwola_log, which causes
    sweep_buckets to treat every row as Bucket A (no re-query detectable
    without a session). The synthetic fallback generates a deterministic
    session_id from (client_ip, time_window), so bursts of traffic from the
    same operator get grouped into coherent sessions for bucket assignment.

    See docs/future/STATISTICAL_FUSION.md §C2 for the CWoLa framework.
    """
    default_party_id: str = "default"         # Used when the request omits party_id
    synthetic_session_window_s: int = 300     # 5 min — close-in-time same-IP requests become one session
    synthetic_session_enabled: bool = True    # Flip to false to preserve prior "NULL = all A" behavior


@dataclass
class RetrievalConfig:
    """Tier 5.5 SR + theta ray_trace bias (Sprint 2)."""
    # Successor Representation (Stachenfeld 2017) - lazy on-demand SR rows
    # via truncated power series over co-activation graph.
    sr_enabled: bool = False            # Dark ship — flip on for A/B
    sr_gamma: float = 0.85              # Discount factor (5-10 hop horizon at 0.9)
    sr_k_steps: int = 4                 # Power-series truncation depth
    sr_weight: float = 1.5              # Per-gene contribution multiplier
    sr_cap: float = 3.0                 # Max per-gene SR boost (matches harmonic cap)
    # Theta alternation (Wang/Foster/Pfeiffer 2020) — fore/aft ray_trace
    # sampling biased by the current TCM velocity vector.
    ray_trace_theta: bool = False       # Dark ship
    theta_weight: float = 1.0           # Softmax temperature on v·gene dot product
    # Sprint 4 — seeded co-activation edges with Hebbian evidence decay.
    # Three-class edge provenance (seeded / co_retrieved / cwola_validated)
    # with Laplace-smoothed co_count vs miss_count per edge.
    seeded_edges_enabled: bool = False  # Dark ship — flip to start evidence accumulation
    seeded_edge_weight: float = 1.0     # Base weight written on seed insertion
    # Tier 0.5 filename-anchor (2026-04-15 Dewey-pivot spike).
    # Dewey bench showed filename alone outperforms the full
    # project+module+filename bag by 24pp. Boosts genes whose
    # filename_stem matches a query term.
    filename_anchor_enabled: bool = False   # Dark ship — flip for A/B
    filename_anchor_weight: float = 4.0     # Per-match boost (higher than Tier 1's 3.0)


@dataclass
class HeadroomConfig:
    """Headroom proxy lifecycle controls — launcher only.

    Headroom is a separate process (headroom-ai[proxy]) that serves a
    compression proxy + dashboard at `http://{host}:{port}/dashboard`.
    When ``autostart`` is true, the launcher spawns it as a child and
    surfaces it in the tray menu. When it's already running on
    ``port``, the launcher **adopts** the existing process rather than
    spawning a duplicate — the adopted process survives launcher Quit.
    """
    enabled: bool = False               # Master switch; false = do nothing
    autostart: bool = True              # When enabled: adopt if running, spawn if not
    host: str = "127.0.0.1"
    port: int = 8787
    mode: str = "token"                 # "token" | "cache" (passed to --mode)
    dashboard_path: str = "/dashboard"  # Appended to http://{host}:{port}


@dataclass
class HelixConfig:
    ribosome: RibosomeConfig = field(default_factory=RibosomeConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    cymatics: CymaticsConfig = field(default_factory=CymaticsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    headroom: HeadroomConfig = field(default_factory=HeadroomConfig)
    synonym_map: Dict[str, List[str]] = field(default_factory=dict)


def load_config(path: Optional[str] = None) -> HelixConfig:
    """
    Load helix.toml from the given path, or auto-discover from cwd / env.
    Returns defaults if no config file is found.
    """
    if path is None:
        path = os.environ.get("HELIX_CONFIG", "helix.toml")

    config_path = Path(path)
    if not config_path.exists():
        log.info("No config file at %s, using defaults", path)
        return HelixConfig()

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    cfg = HelixConfig()

    # Ribosome
    if "ribosome" in raw:
        r = raw["ribosome"]
        cfg.ribosome = RibosomeConfig(
            enabled=bool(r.get("enabled", cfg.ribosome.enabled)),
            model=r.get("model", cfg.ribosome.model),
            base_url=r.get("base_url", cfg.ribosome.base_url),
            timeout=float(r.get("timeout", cfg.ribosome.timeout)),
            keep_alive=r.get("keep_alive", cfg.ribosome.keep_alive),
            warmup=r.get("warmup", cfg.ribosome.warmup),
            backend=r.get("backend", cfg.ribosome.backend),
            claude_model=r.get("claude_model", cfg.ribosome.claude_model),
            claude_base_url=r.get("claude_base_url", cfg.ribosome.claude_base_url),
            litellm_model=r.get("litellm_model", cfg.ribosome.litellm_model),
            rerank_model_path=r.get("rerank_model_path", cfg.ribosome.rerank_model_path),
            splice_model_path=r.get("splice_model_path", cfg.ribosome.splice_model_path),
            splice_threshold=float(r.get("splice_threshold", cfg.ribosome.splice_threshold)),
            nli_model_path=r.get("nli_model_path", cfg.ribosome.nli_model_path),
            nli_splice_bonus=float(r.get("nli_splice_bonus", cfg.ribosome.nli_splice_bonus)),
            nli_splice_penalty=float(r.get("nli_splice_penalty", cfg.ribosome.nli_splice_penalty)),
            device=r.get("device", cfg.ribosome.device),
            query_expansion_enabled=bool(r.get("query_expansion_enabled", cfg.ribosome.query_expansion_enabled)),
        )

    # Budget
    if "budget" in raw:
        b = raw["budget"]
        cfg.budget = BudgetConfig(
            ribosome_tokens=b.get("ribosome_tokens", cfg.budget.ribosome_tokens),
            expression_tokens=b.get("expression_tokens", cfg.budget.expression_tokens),
            max_genes_per_turn=b.get("max_genes_per_turn", cfg.budget.max_genes_per_turn),
            max_fingerprints_per_turn=b.get("max_fingerprints_per_turn", cfg.budget.max_fingerprints_per_turn),
            splice_aggressiveness=float(b.get("splice_aggressiveness", cfg.budget.splice_aggressiveness)),
            decoder_mode=b.get("decoder_mode", cfg.budget.decoder_mode),
            legibility_enabled=bool(b.get("legibility_enabled", cfg.budget.legibility_enabled)),
            session_delivery_enabled=bool(b.get("session_delivery_enabled", cfg.budget.session_delivery_enabled)),
        )

    # Genome
    if "genome" in raw:
        g = raw["genome"]
        cfg.genome = GenomeConfig(
            path=g.get("path", cfg.genome.path),
            compact_interval=float(g.get("compact_interval", cfg.genome.compact_interval)),
            cold_start_threshold=int(g.get("cold_start_threshold", cfg.genome.cold_start_threshold)),
            replicas=g.get("replicas", cfg.genome.replicas),
            replica_sync_interval=int(g.get("replica_sync_interval", cfg.genome.replica_sync_interval)),
        )

    # Server
    if "server" in raw:
        s = raw["server"]
        cfg.server = ServerConfig(
            host=s.get("host", cfg.server.host),
            port=int(s.get("port", cfg.server.port)),
            upstream=s.get("upstream", cfg.server.upstream),
            upstream_timeout=float(s.get("upstream_timeout", cfg.server.upstream_timeout)),
        )

    # Ingestion
    if "ingestion" in raw:
        i = raw["ingestion"]
        cfg.ingestion = IngestionConfig(
            backend=i.get("backend", cfg.ingestion.backend),
            splade_enabled=i.get("splade_enabled", cfg.ingestion.splade_enabled),
            rerank_model=i.get("rerank_model", cfg.ingestion.rerank_model),
            rerank_enabled=i.get("rerank_enabled", cfg.ingestion.rerank_enabled),
            colbert_enabled=i.get("colbert_enabled", cfg.ingestion.colbert_enabled),
            entity_graph=i.get("entity_graph", cfg.ingestion.entity_graph),
        )

    # Context (cold-tier retrieval knobs — C.2 of B->C, 2026-04-10)
    if "context" in raw:
        c = raw["context"]
        cfg.context = ContextConfig(
            cold_tier_enabled=bool(c.get("cold_tier_enabled", cfg.context.cold_tier_enabled)),
            cold_tier_min_hot_genes=int(c.get("cold_tier_min_hot_genes", cfg.context.cold_tier_min_hot_genes)),
            cold_tier_k=int(c.get("cold_tier_k", cfg.context.cold_tier_k)),
            cold_tier_min_cosine=float(c.get("cold_tier_min_cosine", cfg.context.cold_tier_min_cosine)),
            fingerprint_mode_profile=str(c.get("fingerprint_mode_profile", cfg.context.fingerprint_mode_profile)).lower(),
        )

    # Cymatics
    if "cymatics" in raw:
        cy = raw["cymatics"]
        cfg.cymatics = CymaticsConfig(
            enabled=cy.get("enabled", cfg.cymatics.enabled),
            n_bins=int(cy.get("n_bins", cfg.cymatics.n_bins)),
            peak_width=float(cy.get("peak_width", cfg.cymatics.peak_width)),
            splice_threshold_scale=float(cy.get("splice_threshold_scale", cfg.cymatics.splice_threshold_scale)),
            use_embeddings=cy.get("use_embeddings", cfg.cymatics.use_embeddings),
            harmonic_links=cy.get("harmonic_links", cfg.cymatics.harmonic_links),
            distance_metric=str(cy.get("distance_metric", cfg.cymatics.distance_metric)).lower(),
        )

    # Retrieval (Sprint 2 — SR Tier 5.5 + theta alternation)
    if "retrieval" in raw:
        r = raw["retrieval"]
        cfg.retrieval = RetrievalConfig(
            sr_enabled=bool(r.get("sr_enabled", cfg.retrieval.sr_enabled)),
            sr_gamma=float(r.get("sr_gamma", cfg.retrieval.sr_gamma)),
            sr_k_steps=int(r.get("sr_k_steps", cfg.retrieval.sr_k_steps)),
            sr_weight=float(r.get("sr_weight", cfg.retrieval.sr_weight)),
            sr_cap=float(r.get("sr_cap", cfg.retrieval.sr_cap)),
            ray_trace_theta=bool(r.get("ray_trace_theta", cfg.retrieval.ray_trace_theta)),
            theta_weight=float(r.get("theta_weight", cfg.retrieval.theta_weight)),
            seeded_edges_enabled=bool(r.get("seeded_edges_enabled", cfg.retrieval.seeded_edges_enabled)),
            seeded_edge_weight=float(r.get("seeded_edge_weight", cfg.retrieval.seeded_edge_weight)),
            filename_anchor_enabled=bool(r.get("filename_anchor_enabled", cfg.retrieval.filename_anchor_enabled)),
            filename_anchor_weight=float(r.get("filename_anchor_weight", cfg.retrieval.filename_anchor_weight)),
        )

    # Session (CWoLa session/party fallback — 2026-04-13 fix for always-A bucket bug)
    if "session" in raw:
        s = raw["session"]
        cfg.session = SessionConfig(
            default_party_id=str(s.get("default_party_id", cfg.session.default_party_id)),
            synthetic_session_window_s=int(s.get("synthetic_session_window_s", cfg.session.synthetic_session_window_s)),
            synthetic_session_enabled=bool(s.get("synthetic_session_enabled", cfg.session.synthetic_session_enabled)),
        )

    # Headroom — optional proxy lifecycle controls
    if "headroom" in raw:
        h = raw["headroom"]
        cfg.headroom = HeadroomConfig(
            enabled=bool(h.get("enabled", cfg.headroom.enabled)),
            autostart=bool(h.get("autostart", cfg.headroom.autostart)),
            host=str(h.get("host", cfg.headroom.host)),
            port=int(h.get("port", cfg.headroom.port)),
            mode=str(h.get("mode", cfg.headroom.mode)),
            dashboard_path=str(h.get("dashboard_path", cfg.headroom.dashboard_path)),
        )

    # Fix 1: synonym map
    if "synonyms" in raw:
        cfg.synonym_map = {
            k: list(v) for k, v in raw["synonyms"].items()
        }

    log.info("Config loaded from %s", config_path)
    return cfg

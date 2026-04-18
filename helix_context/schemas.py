"""
Schemas — Pydantic models for the Helix genome data layer.

These are the stable internal contracts. All models are JSON-serializable
for SQLite storage (via model_dump_json / model_validate_json).
"""

from __future__ import annotations

import time
from enum import Enum, IntEnum
from typing import List, Optional

from pydantic import BaseModel, Field


class NLRelation(IntEnum):
    """MacCartney-Manning natural logic relations (7-class)."""
    ENTAILMENT = 0          # A ⊂ B (A implies B)
    REVERSE_ENTAILMENT = 1  # A ⊃ B (B implies A)
    EQUIVALENCE = 2         # A = B
    ALTERNATION = 3         # A ∩ B = ∅, A ∪ B ≠ D (mutually exclusive)
    NEGATION = 4            # A ∩ B = ∅, A ∪ B = D (exhaustive opposites)
    COVER = 5               # A ∩ B ≠ ∅, A ∪ B = D (overlap + exhaust)
    INDEPENDENCE = 6        # no reliable relation


class StructuralRelation(IntEnum):
    """Structural hierarchy relations stored in gene_relations.relation.

    Values ≥ 100 to avoid collision with NLRelation (0-6). The
    gene_relations table is a discriminated union on relation code:
    0-6 are NL semantic links, 100+ are structural hierarchy edges.
    """
    CHUNK_OF = 100          # gene_id_a is a chunk of gene_id_b (parent file gene)


class ChromatinState(IntEnum):
    """Gene accessibility state — mirrors biological chromatin compaction."""
    OPEN = 0            # Recently accessed, hot
    EUCHROMATIN = 1     # Accessible, normal state
    HETEROCHROMATIN = 2 # Compacted, stale — excluded from queries


class PromoterTags(BaseModel):
    """Retrieval metadata — how the genome finds this gene."""
    domains: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    intent: str = ""
    summary: str = ""
    sequence_index: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class TypedCoActivation(BaseModel):
    """A co-activation link with a typed logical relation."""
    gene_id: str
    relation: NLRelation = NLRelation.INDEPENDENCE
    confidence: float = 0.0


class EpigeneticMarkers(BaseModel):
    """Usage and association metadata — how the gene evolves over time."""
    created_at: float = Field(default_factory=lambda: time.time())
    last_accessed: float = Field(default_factory=lambda: time.time())
    access_count: int = 0
    co_activated_with: List[str] = Field(default_factory=list)
    typed_co_activated: List[TypedCoActivation] = Field(default_factory=list)
    decay_score: float = 1.0

    # Working-set inference (Phase 1 of 8D dimensional roadmap, slice 1).
    # Ring buffer of recent access timestamps used to compute a windowed
    # access rate. The monotonic `access_count` above conflates "hot last
    # hour" with "hot once a year ago"; this field lets callers ask "how
    # often has this gene been touched in the last N seconds" via the
    # `access_rate()` helper below.
    #
    # Capped to the most recent 100 timestamps so a single gene's marker
    # blob stays bounded (~800 bytes worst case). Older entries are
    # dropped on append by the touch path; readers can also tolerate any
    # length without breaking.
    #
    # Inert until Slice 2 wires the touch path to populate it. New code
    # should NOT use this field as a primary retrieval signal — see the
    # warning in ~/.helix/shared/handoffs/2026-04-11_8d_dimensional_roadmap.md
    # (Phase 1 section): cost-of-fetch is not a relevance signal.
    recent_accesses: List[float] = Field(default_factory=list)

    def access_rate(self, window_seconds: float = 3600.0) -> float:
        """Accesses per second over the last `window_seconds`.

        Counts entries in `recent_accesses` whose timestamp is newer than
        `now - window_seconds`, divided by the window duration. Returns
        0.0 if the field is empty (e.g. for a gene that hasn't been
        touched since Slice 2 wired the population path, or a freshly
        ingested gene that hasn't been retrieved yet).

        This is the working-set inference primitive from the "n over x
        bell curve" insight: software cannot directly query whether a
        gene is cache-resident, but the access pattern over a sliding
        window is a sufficient Bayesian proxy for tier residency.

        Use as a tiebreaker / prefetch hint, not as a primary relevance
        signal — that direction creates a positive feedback loop where
        the hot tier serves itself regardless of correctness.
        """
        if not self.recent_accesses:
            return 0.0
        if window_seconds <= 0:
            return 0.0
        cutoff = time.time() - window_seconds
        n = sum(1 for t in self.recent_accesses if t > cutoff)
        return n / window_seconds


class Gene(BaseModel):
    """The fundamental storage unit in the genome."""
    gene_id: str
    content: str
    complement: str                     # Dense summary (fallback for splice failures)
    codons: List[str]                   # Semantic meaning labels

    promoter: PromoterTags = Field(default_factory=PromoterTags)
    epigenetics: EpigeneticMarkers = Field(default_factory=EpigeneticMarkers)

    key_values: List[str] = Field(default_factory=list)  # Pre-extracted facts: "port=11437", "model=qwen3"

    chromatin: ChromatinState = ChromatinState.OPEN
    is_fragment: bool = False

    embedding: Optional[List[float]] = None

    # Versioning (future — can ignore for MVP)
    source_id: Optional[str] = None
    repo_root: Optional[str] = None
    source_kind: Optional[str] = None
    observed_at: Optional[float] = None
    mtime: Optional[float] = None
    content_hash: Optional[str] = None
    volatility_class: Optional[str] = None
    authority_class: Optional[str] = None
    support_span: Optional[str] = None
    last_verified_at: Optional[float] = None
    version: Optional[int] = None
    supersedes: Optional[str] = None


class ContextHealth(BaseModel):
    """Delta-epsilon context health signal — the 'Check Engine Light.'"""
    ellipticity: float = 1.0            # 0=denatured, 1=perfectly grounded
    coverage: float = 0.0               # Fraction of query terms that matched genes
    density: float = 0.0                # Fraction of expression budget used
    freshness: float = 1.0              # Average decay score of expressed genes
    logical_coherence: float = 0.0      # Pairwise relation coherence of expressed genes
    genes_available: int = 0            # Total genes in genome
    genes_expressed: int = 0            # Genes expressed for this query
    status: str = "unmeasured"          # aligned | sparse | stale | denatured
    # Step 1b weighing surface (2026-04-17): pre-delivery confidence the
    # consumer can use for know-vs-go decisions. Separate from ellipticity
    # (which is retrospective "did we deliver good context"). These measure
    # "how confident was the coordinate resolution itself."
    coordinate_crispness: float = 0.0   # Top-K score dominance: (s[0]-s[k])/(s[0]+ε)
    neighborhood_density: float = 0.0   # Fraction of top-K with score >= 0.3 * top
    resolution_confidence: float = 0.0  # Composite: sqrt(crispness * coverage)


class ContextWindow(BaseModel):
    """The assembled context ready for the big model."""
    ribosome_prompt: str                # 3k fixed decoder layer
    expressed_context: str              # 6k codon-encoded active context
    expressed_gene_ids: List[str] = Field(default_factory=list)
    total_estimated_tokens: int = 0
    compression_ratio: float = 0.0
    context_health: ContextHealth = Field(default_factory=ContextHealth)
    metadata: dict = Field(default_factory=dict)


class ContextItem(BaseModel):
    """A task-scoped evidence item returned by the packet builder."""
    kind: str = "gene"
    gene_id: Optional[str] = None
    claim_id: Optional[str] = None
    title: str
    content: str
    relevance_score: float = 0.0
    live_truth_score: float = 0.0
    source_id: Optional[str] = None
    source_kind: Optional[str] = None
    volatility_class: Optional[str] = None
    authority_class: Optional[str] = None
    last_verified_at: Optional[float] = None
    status: str = "verified"
    citations: List[str] = Field(default_factory=list)


class RefreshTarget(BaseModel):
    """A concrete reread target before a high-risk action."""
    target_kind: str
    source_id: str
    reason: str
    priority: float = 0.0


class ContextPacket(BaseModel):
    """Agent-safe retrieval packet with freshness labeling."""
    task_type: str
    query: str
    verified: List[ContextItem] = Field(default_factory=list)
    stale_risk: List[ContextItem] = Field(default_factory=list)
    contradictions: List[ContextItem] = Field(default_factory=list)
    refresh_targets: List[RefreshTarget] = Field(default_factory=list)
    working_set_id: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


# ── Session registry (see docs/SESSION_REGISTRY.md) ────────────────────────

class Party(BaseModel):
    """A trust identity — a human principal, tenant, or org service identity.

    Parties hold genes. A party may contain many participants. Parties are
    atomic: they do NOT self-reference. Grouping of parties is handled by
    the future `collectives` layer.
    """
    party_id: str                       # "max@local", "tenant:acme", "peer:swiftwing21"
    display_name: str
    trust_domain: str = "local"         # "local" | "remote" | "tenant:*"
    created_at: float = Field(default_factory=lambda: time.time())
    metadata: Optional[dict] = None


class Participant(BaseModel):
    """A live runtime actor (Claude session, sub-agent, swarm member).

    Participants belong to exactly one party. They are ephemeral — they
    come and go as sessions start and stop. Attribution of genes survives
    participant turnover via the party.
    """
    participant_id: str                 # ULID / uuid4
    party_id: str
    handle: str                         # "taude", "laude", "raude", "subagent-7f3a"
    workspace: Optional[str] = None
    pid: Optional[int] = None
    started_at: float = Field(default_factory=lambda: time.time())
    last_heartbeat: float = Field(default_factory=lambda: time.time())
    status: str = "active"              # "active" | "idle" | "stale" | "gone"
    capabilities: List[str] = Field(default_factory=list)
    metadata: Optional[dict] = None


class ParticipantInfo(BaseModel):
    """Projection used by GET /sessions — what observers see about a sibling."""
    participant_id: str
    party_id: str
    handle: str
    workspace: Optional[str] = None
    status: str
    last_seen_s_ago: float
    started_at: float


class GeneAttribution(BaseModel):
    """Attribution row linking a gene to the party/participant that authored it."""
    gene_id: str
    party_id: str
    participant_id: Optional[str] = None
    authored_at: float


# ── HITL event logging (see ~/.helix/shared/handoffs/2026-04-11_hitl_observation.md) ──


class HITLPauseType(str, Enum):
    """Category of Human-In-The-Loop pause event.

    Stored as string in SQLite for readability. Extend by adding new
    values here and updating the DAL's validation path. `other` is a
    deliberate escape hatch for pauses that don't fit the taxonomy —
    instrumentation should not fail because of a schema gap.
    """
    permission_request = "permission_request"   # session asked operator before acting
    uncertainty_check  = "uncertainty_check"    # session asked "is this right?" mid-task
    rollback_confirm   = "rollback_confirm"     # session asked before a revert/undo
    other              = "other"


class HITLEvent(BaseModel):
    """A single HITL pause event, suitable for storage in hitl_events.

    Motivated by laude's 2026-04-11 HITL observation (handoff off-git)
    and raude's M1 discriminating test which established that the
    mechanism behind the 2026-04-10 HITL shift was NOT genome-mediated.
    This means the logger needs to record chat-channel signals in
    addition to genome-state snapshots — see the optional fields below.

    All signal fields are nullable; the minimal valid event has only
    `party_id`, `ts`, and `pause_type`. Additional fields are populated
    when the caller can compute them.
    """
    event_id: str
    party_id: str
    participant_id: Optional[str] = None
    ts: float

    # Core pause signals (always populated)
    pause_type: HITLPauseType
    task_context: Optional[str] = None
    resolved_without_operator: bool = False

    # Chat-channel signals (added per M1 finding — mechanism was non-genome)
    operator_tone_uncertainty: Optional[float] = None     # 0-1 proxy score
    operator_risk_keywords: List[str] = Field(default_factory=list)
    time_since_last_risk_event: Optional[float] = None    # seconds
    recoverability_signal: Optional[str] = None           # "recoverable" | "uncertain" | "lost"

    # Genome state snapshot (for M3 prospective correlation)
    genome_total_genes: Optional[int] = None
    genome_hetero_count: Optional[int] = None
    cold_cache_size: Optional[int] = None

    metadata: Optional[dict] = None

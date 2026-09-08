"""
Schemas — Pydantic models for the Cymatix knowledge store data layer.

These are the stable internal contracts. All models are JSON-serializable
for SQLite storage (via model_dump_json / model_validate_json).
"""

from __future__ import annotations

import time
from enum import Enum, IntEnum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
    CHUNK_OF = 100          # gene_id_a is a chunk of gene_id_b (parent file document)
    SYMBOL_REF = 101        # gene_id_a (chunk) references a symbol defined in gene_id_b (chunk) [WS2]


class LifecycleTier(IntEnum):
    """Document accessibility state — mirrors biological chromatin compaction (legacy term)."""
    OPEN = 0            # Recently accessed, hot
    EUCHROMATIN = 1     # Accessible, normal state — kept for SQL/string contract
    HETEROCHROMATIN = 2 # Compacted, stale — excluded from queries; SQL/string contract


# R3 legacy alias — pre-R3 callers still import ChromatinState. Identity
# preserved: ChromatinState is LifecycleTier. See docs/ROSETTA.md.
ChromatinState = LifecycleTier


class IntentClass(str, Enum):
    """Structured intent taxonomy for sub-query routing (D8 completion, Step 3B).

    Assigned at ingest by GeneTagExtractor._classify_intent().
    Used by intent_router.py for LLM-free sub-query decomposition.
    """
    UNKNOWN = "unknown"
    MECHANISM = "mechanism"
    CONFIG_KNOB = "config_knob"
    DATA_STRUCTURE = "data_structure"
    PROCESS_STEP = "process_step"
    TRIGGER_CONDITION = "trigger_condition"
    FACT = "fact"
    RELATIONSHIP = "relationship"


class DocumentTags(BaseModel):
    """Retrieval metadata — how the knowledge store finds this document."""
    domains: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    intent: str = ""
    intent_class: IntentClass = IntentClass.UNKNOWN
    summary: str = ""
    sequence_index: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


# R3 legacy alias — pre-R3 callers still import PromoterTags. See docs/ROSETTA.md.
PromoterTags = DocumentTags


class TypedCoActivation(BaseModel):
    """A co-activation link with a typed logical relation."""
    gene_id: str
    relation: NLRelation = NLRelation.INDEPENDENCE
    confidence: float = 0.0


class DocumentSignals(BaseModel):
    """Usage and association metadata — how the document evolves over time."""
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
    # often has this document been touched in the last N seconds" via the
    # `access_rate()` helper below.
    #
    # Capped to the most recent 100 timestamps so a single document's marker
    # blob stays bounded (~800 bytes worst case). Older entries are
    # dropped on append by the touch path; readers can also tolerate any
    # length without breaking.
    #
    # Inert until Slice 2 wires the touch path to populate it. New code
    # should NOT use this field as a primary retrieval signal — see the
    # warning in ~/.cymatix/shared/handoffs/2026-04-11_8d_dimensional_roadmap.md
    # (Phase 1 section): cost-of-fetch is not a relevance signal.
    recent_accesses: List[float] = Field(default_factory=list)

    def access_rate(self, window_seconds: float = 3600.0) -> float:
        """Accesses per second over the last `window_seconds`.

        Counts entries in `recent_accesses` whose timestamp is newer than
        `now - window_seconds`, divided by the window duration. Returns
        0.0 if the field is empty (e.g. for a document that hasn't been
        touched since Slice 2 wired the population path, or a freshly
        ingested document that hasn't been retrieved yet).

        This is the working-set inference primitive from the "n over x
        bell curve" insight: software cannot directly query whether a
        document is cache-resident, but the access pattern over a sliding
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


# R3 legacy alias — pre-R3 callers still import EpigeneticMarkers. See docs/ROSETTA.md.
EpigeneticMarkers = DocumentSignals


class Document(BaseModel):
    """The fundamental storage unit in the knowledge store."""
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


# R3 legacy alias — pre-R3 callers still import Gene. The Pydantic field
# names (gene_id, codons, promoter, epigenetics, chromatin, etc) stay as
# the SQL contract; only the class identity moved to Document. See
# docs/ROSETTA.md.
Gene = Document


class ContextHealth(BaseModel):
    """Delta-epsilon context health signal — the 'Check Engine Light.'"""
    ellipticity: float = 1.0            # 0=denatured, 1=perfectly grounded
    coverage: float = 0.0               # Fraction of query terms that matched documents
    density: float = 0.0                # Fraction of retrieval budget used
    freshness: float = 1.0              # Back-compat: aliases freshness_weighted (Stage 7)
    logical_coherence: float = 0.0      # Pairwise relation coherence of retrieved documents
    genes_available: int = 0            # Total documents in knowledge store
    genes_expressed: int = 0            # Documents retrieved for this query
    status: str = "unmeasured"          # aligned | sparse | stale | denatured

    # Stage 7 (2026-05-08): three freshness signals replace the single
    # mean(decay) value to stop a stale top-1 needle from being masked
    # by fresh padding documents. ``freshness`` (back-compat field above) is
    # set to ``freshness_weighted`` so legacy consumers keep a meaningful
    # number; new consumers should branch on ``freshness_top1`` /
    # ``freshness_min``. None until populated by ``_compute_health``.
    freshness_min: Optional[float] = None       # min decay over candidates
    freshness_top1: Optional[float] = None      # decay of the top-1 (score-desc)
    freshness_weighted: Optional[float] = None  # score-weighted decay sum
    # Step 1b weighing surface (2026-04-17): pre-delivery confidence the
    # consumer can use for know-vs-go decisions. Separate from ellipticity
    # (which is retrospective "did we deliver good context"). These measure
    # "how confident was the coordinate resolution itself."
    coordinate_crispness: float = 0.0   # Top-K score dominance: (s[0]-s[k])/(s[0]+ε)
    neighborhood_density: float = 0.0   # Fraction of top-K with score >= 0.3 * top
    resolution_confidence: float = 0.0  # Composite: pathcov * sqrt(coverage) — iter2
    # Step 1b-iter2 (2026-04-18): three signal candidates measured against
    # gold_delivered on the 10-needle bench. First iteration's crispness
    # and neighborhood were anti-correlated with ground truth — these are
    # the alternates. path_token_coverage is the discriminator (delta +0.48).
    top_score_raw: float = 0.0          # Absolute top-1 score (scale-sensitive)
    top_dominance: float = 0.0          # top / mean(all scored) — how much #1 dominates
    path_token_coverage: float = 0.0    # Fraction of delivered documents whose source_path
                                        # tokens overlap the extracted query signals
    # File-grain coord signal (2026-04-18): path_token_coverage is folder+file
    # tokens mixed; this is basename-only. Catches "same folder, wrong file"
    # silent-miss mode where the delivered set is all in the right project
    # directory but none of the filenames match the queried concept.
    file_token_coverage: float = 0.0    # Fraction of delivered documents whose basename
                                        # tokens overlap the extracted query signals


class ContextWindow(BaseModel):
    """The assembled context ready for the big model."""
    ribosome_prompt: str                # 3k fixed decoder layer
    expressed_context: str              # 6k fragment-encoded active context
    expressed_gene_ids: List[str] = Field(default_factory=list)
    total_estimated_tokens: int = 0
    compression_ratio: float = 0.0
    context_health: ContextHealth = Field(default_factory=ContextHealth)
    metadata: dict = Field(default_factory=dict)
    # Issue #350 (W1.6b): request-scoped retrieval snapshot, captured by
    # build_context under _last_query_scores_lock right after retrieval.
    # Server-layer readers (know/miss, citation scores, tier totals) prefer
    # these over the genome-global last_query_scores / last_tier_contributions,
    # which under concurrent serving may already hold ANOTHER request's
    # publication — cross-request contamination of reported confidence.
    # exclude=True: internal plumbing, never serialized into responses.
    retrieval_scores: Optional[dict] = Field(default=None, exclude=True, repr=False)
    tier_contributions: Optional[dict] = Field(default=None, exclude=True, repr=False)


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
    document_id: Optional[str] = None   # stable portable document identity (source-derived; v1 = raw source_id)
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
    """Agent-safe retrieval packet with freshness labeling.

    Stage 6 (2026-05-08, §5 + §9): adds optional ``know`` and ``miss``
    fields and a first-class ``coordinate_confidence`` so the agent
    can branch on a structured signal instead of scraping ``notes``.
    """
    task_type: str
    query: str
    verified: List[ContextItem] = Field(default_factory=list)
    stale_risk: List[ContextItem] = Field(default_factory=list)
    contradictions: List[ContextItem] = Field(default_factory=list)
    refresh_targets: List[RefreshTarget] = Field(default_factory=list)
    working_set_id: Optional[str] = None
    notes: List[str] = Field(default_factory=list)

    # Stage 6 — top-level know/miss block; exactly one is non-null
    # when this packet was built from a non-empty query. (Pre-existing
    # consumers that ignore unknown keys are unaffected; new consumers
    # should branch on these keys before reading ``verified`` /
    # ``stale_risk``.)
    know: Optional["KnowBlock"] = None  # forward ref — defined below
    miss: Optional["MissBlock"] = None

    # Stage 6 (§9) — promoted from a notes-prose summary to a
    # first-class field so consumers don't need to regex parse it.
    coordinate_confidence: float = 0.0
    file_coverage: float = 0.0

    # Phase 0.1/0.5 (2026-08-12 refinement campaign) — optional
    # delivery-visibility block: delivered ids in final order plus
    # which constraint (assembly cap vs token budget) truncated
    # delivery. Optional/None-safe so pre-existing readers are
    # unaffected; forward ref — defined below next to Know/MissBlock.
    delivery: Optional["DeliveryBlock"] = None

    # Issue #350 (W1.6b): request-scoped retrieval snapshot, captured by
    # build_context_packet under _last_query_scores_lock right after its
    # query_docs() call — the packet-path mirror of ContextWindow.
    # retrieval_scores / tier_contributions above. Server-layer readers
    # (the PLR query-confidence head) prefer these over the genome-global
    # last_query_scores / last_tier_contributions, which under concurrent
    # serving may already hold ANOTHER request's publication.
    # exclude=True: internal plumbing, never serialized into responses.
    retrieval_scores: Optional[dict] = Field(default=None, exclude=True, repr=False)
    tier_contributions: Optional[dict] = Field(default=None, exclude=True, repr=False)


# ── Claims layer (see docs/specs/2026-04-17-agent-context-index-build-spec.md) ──

CLAIM_TYPES = (
    "path_value",
    "config_value",
    "api_contract",
    "entity_membership",
    "benchmark_result",
    "operational_state",
    "version_marker",
    "human_assertion",
)

EXTRACTION_KINDS = ("literal", "derived", "inferred")

CLAIM_EDGE_TYPES = ("contradicts", "supports", "supersedes", "duplicates")


class Claim(BaseModel):
    """A structured fact extracted from a document.

    Agents reason over claims, not bulk document content. A claim carries
    enough provenance (gene_id, shard_name, observed_at) that the
    packet builder can answer freshness questions without opening the
    owning shard's content tier.
    """
    claim_id: str
    gene_id: str
    shard_name: str
    claim_type: str              # one of CLAIM_TYPES
    entity_key: Optional[str] = None
    claim_text: str
    extraction_kind: str = "literal"   # one of EXTRACTION_KINDS
    specificity: float = 0.5
    confidence: float = 0.5
    observed_at: Optional[float] = None
    supersedes_claim_id: Optional[str] = None
    updated_at: float = 0.0


class ClaimEdge(BaseModel):
    """Directed edge between two claims in the contradiction/support graph."""
    src_claim_id: str
    dst_claim_id: str
    edge_type: str               # one of CLAIM_EDGE_TYPES
    weight: float = 1.0
    created_at: float = 0.0


# ── Session registry (see docs/SESSION_REGISTRY.md) ────────────────────────

class Party(BaseModel):
    """A trust identity — a human principal, tenant, or org service identity.

    Parties hold documents. A party may contain many participants. Parties are
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
    come and go as sessions start and stop. Attribution of documents survives
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
    agent_kind: Optional[str] = None    # vendor family — "claude-code", "codex"
    mcp_host: Optional[str] = None      # host tag — "vscode", "cursor"
    ide_detected: Optional[str] = None        # adapter detect at register time
    ide_detection_via: Optional[str] = None   # "env:VSCODE_PID", "agent_override", etc.
    model_id: Optional[str] = None            # agent self-reported via cymatix_announce


class ParticipantInfo(BaseModel):
    """Projection used by GET /sessions — what observers see about a sibling."""
    participant_id: str
    party_id: str
    handle: str
    workspace: Optional[str] = None
    status: str
    last_seen_s_ago: float
    started_at: float
    agent_kind: Optional[str] = None
    mcp_host: Optional[str] = None
    ide_detected: Optional[str] = None
    ide_detection_via: Optional[str] = None
    model_id: Optional[str] = None


class DocumentAttribution(BaseModel):
    """Attribution row linking a document to the party/participant that authored it."""
    gene_id: str
    party_id: str
    participant_id: Optional[str] = None
    authored_at: float


# R3 legacy alias — pre-R3 callers still import GeneAttribution. See docs/ROSETTA.md.
GeneAttribution = DocumentAttribution


# ── HITL event logging (see ~/.cymatix/shared/handoffs/2026-04-11_hitl_observation.md) ──


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
    mechanism behind the 2026-04-10 HITL shift was NOT knowledge store-mediated.
    This means the logger needs to record chat-channel signals in
    addition to knowledge store-state snapshots — see the optional fields below.

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

    # Chat-channel signals (added per M1 finding — mechanism was non-knowledge store)
    operator_tone_uncertainty: Optional[float] = None     # 0-1 proxy score
    operator_risk_keywords: List[str] = Field(default_factory=list)
    time_since_last_risk_event: Optional[float] = None    # seconds
    recoverability_signal: Optional[str] = None           # "recoverable" | "uncertain" | "lost"

    # KnowledgeStore state snapshot (for M3 prospective correlation)
    genome_total_genes: Optional[int] = None
    genome_hetero_count: Optional[int] = None
    cold_cache_size: Optional[int] = None

    metadata: Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — machine-tagged know/miss contract for /context
# Spec: docs/specs/2026-05-08-stage-6-know-miss-blocks.md
#
# EXTENSION POINTS for Stage 7 (freshness gate). This file is laid out
# so each Stage 7 addition is a single-line edit. Search for the
# `# STAGE-7-EXT` markers to find them.
# ─────────────────────────────────────────────────────────────────────

# Stage 6 reason vocabulary. Stage 7 (2026-05-08) appended
# "stale" | "cold" | "superseded" for the freshness-gate demotions.
# Adding a value is a one-line change to this tuple; pydantic accepts
# it everywhere via the runtime check in MissBlock._validate_*. The
# split into "escalate-class" vs "refresh-class" reasons drives the
# new mutual-exclusivity validator below.
MISS_REASONS: tuple[str, ...] = (
    "abstain",
    "denatured",
    "sparse",
    "no_promoter_match",
    # Stage 7 freshness-gate reasons (require refresh_targets)
    "stale",
    "cold",
    "superseded",
)

# Stage 7 (spec §8): reasons that REQUIRE refresh_targets and FORBID
# escalate_to. Mutual-exclusivity is enforced by the model_validator
# on MissBlock.
_REFRESH_REASONS: frozenset[str] = frozenset({"stale", "cold", "superseded"})
_ESCALATE_REASONS: frozenset[str] = frozenset({
    "abstain", "denatured", "sparse", "no_promoter_match",
})

# Tools an agent can escalate to. Cymatix only signals the class; the
# consumer registers the concrete tool. Kept narrow on purpose.
ESCALATE_TARGETS: tuple[str, ...] = (
    "grep",
    "rag",
    "web",
    "ask_human",
)

# Set form for fast membership checks in validators.
_MISS_REASON_SET = frozenset(MISS_REASONS)
_ESCALATE_TARGET_SET = frozenset(ESCALATE_TARGETS)


class KnowBlock(BaseModel):
    """Top-level retrieval-success block emitted at /context.

    Mutually exclusive with MissBlock; envelope validator enforces the
    invariant. Designed for a frontier-model agent: ``found=True`` is the
    machine-tagged equivalent of "the knowledge store did locate this and the
    expressed_context bytes are grounded — you may answer from them."

    Stage 7 extends this additively (NOT a redesign):
      * ``soft_stale: bool`` — top-1 fresh enough to act on, but
        supporting context (rank 2..K) is stale. The agent can answer
        from the knowledge store AND should plan a refresh on its own schedule.
        Legacy parsers ignore unknown fields.
    """

    model_config = {"extra": "forbid"}

    found: Literal[True] = True
    confidence: float = Field(ge=0.0, le=1.0)
    top_score: float
    score_gap: float
    lexical_dense_agree: bool
    gene_id_match: Optional[str] = None
    coordinate_confidence: float = Field(ge=0.0, le=1.0)
    # Stage 7 (spec §9): soft-stale signal — set True when the
    # KnowBlock is still emittable (top-1 is fresh) but
    # freshness_min < 0.5 indicates lower-ranked supporting documents are
    # stale. Drives ``recommendation="refresh"`` at the route layer.
    soft_stale: bool = False


class MissBlock(BaseModel):
    """Top-level retrieval-miss block emitted at /context.

    Carries a discriminator (``reason``) and a concrete next-step list
    (``escalate_to``). ``do_not_answer_from_genome=True`` is a load-bearing
    contract bit: the frontier agent MUST honor it (see
    docs/agent-sdk-fragment.md / CYMATIX_NO_MATCH_FRAGMENT).

    Stage 7 extends this additively (spec §8):
      * ``refresh_targets: list[str]`` — concrete file paths or URLs
        the agent should re-read before retrying. Populated for
        reasons in {stale, cold, superseded}.
      * ``MISS_REASONS`` extended with ``stale | cold | superseded``.
      * ``model_validator`` enforces mutual-exclusivity:
          - refresh-class reason ⇒ refresh_targets non-empty AND
            escalate_to == [].
          - escalate-class reason ⇒ refresh_targets == [] AND
            escalate_to non-empty.
    """

    model_config = {"extra": "forbid"}

    miss: Literal[True] = True
    # Reason is validated against MISS_REASONS at runtime (not via Literal)
    # so Stage 7's one-line tuple extension does not require a pydantic
    # type bump. Trade-off documented; keeps the file extensible.
    reason: str
    top_score: float
    ratio: float
    escalate_to: List[str] = Field(default_factory=list)
    # Stage 7 (spec §8): refresh targets — concrete file paths or
    # URLs the agent should re-read before retrying. Mutually
    # exclusive with ``escalate_to`` per the model_validator below.
    refresh_targets: List[str] = Field(default_factory=list)
    do_not_answer_from_genome: Literal[True] = True

    @model_validator(mode="after")
    def _validate_reason_and_escalate(self) -> "MissBlock":
        if self.reason not in _MISS_REASON_SET:
            raise ValueError(
                f"MissBlock.reason {self.reason!r} not in MISS_REASONS "
                f"({sorted(MISS_REASONS)})"
            )
        for tool in self.escalate_to:
            if tool not in _ESCALATE_TARGET_SET:
                raise ValueError(
                    f"MissBlock.escalate_to entry {tool!r} not in "
                    f"ESCALATE_TARGETS ({sorted(ESCALATE_TARGETS)})"
                )
        # Stage 7 (spec §8) — refresh-vs-escalate mutual exclusivity.
        # The two list fields encode different next-step semantics:
        # ``refresh_targets`` says "the answer is here, just out of
        # date — fetch and retry"; ``escalate_to`` says "the answer
        # isn't here — go ask elsewhere". Conflating them defeats the
        # whole point of the Stage 7 contract.
        if self.reason in _REFRESH_REASONS:
            if not self.refresh_targets:
                raise ValueError(
                    f"MissBlock.reason={self.reason!r} requires "
                    "refresh_targets to be non-empty"
                )
            if self.escalate_to:
                raise ValueError(
                    f"MissBlock.reason={self.reason!r} forbids "
                    f"escalate_to (got {self.escalate_to!r})"
                )
        elif self.reason in _ESCALATE_REASONS:
            if self.refresh_targets:
                raise ValueError(
                    f"MissBlock.reason={self.reason!r} forbids "
                    f"refresh_targets (got {self.refresh_targets!r})"
                )
            # escalate-class reasons must carry at least one tool
            # so the consumer always has a next-step. Empty
            # escalate_to is allowed only on refresh-class reasons.
            if not self.escalate_to:
                raise ValueError(
                    f"MissBlock.reason={self.reason!r} requires "
                    "escalate_to to be non-empty"
                )
        return self

    # Stage 7 (spec §11): adapter to ``RefreshTarget`` so the
    # /context/refresh-plan route can convert a MissBlock-shaped
    # demotion into the existing wire format without callers
    # duplicating the mapping. Returns an empty list when this miss
    # is escalate-class.
    def to_refresh_targets(self) -> List["RefreshTarget"]:
        """Convert ``refresh_targets`` to the RefreshTarget adapter shape.

        Mapping (spec §11):
          MissBlock.refresh_targets[i]              -> source_id
          ("file" if no scheme else "url")          -> target_kind
          stale->stale_mtime, cold->cold_tier,
          superseded->superseded_by_successor       -> reason
          1.0 - freshness_min (or 0.5 default)      -> priority

        ``freshness_min`` is not carried on MissBlock today; callers
        that have it pass it via the wider context. Default priority
        is 0.5 — a neutral mid-band so the refresh plan UI doesn't
        bias against MissBlock-sourced targets relative to other
        sources of refresh demand.
        """
        if not self.refresh_targets:
            return []
        reason_map = {
            "stale": "stale_mtime",
            "cold": "cold_tier",
            "superseded": "superseded_by_successor",
        }
        out: List[RefreshTarget] = []
        for source_id in self.refresh_targets:
            kind = "url" if "://" in str(source_id)[:32] else "file"
            out.append(
                RefreshTarget(
                    target_kind=kind,
                    source_id=str(source_id),
                    reason=reason_map.get(self.reason, self.reason),
                    priority=0.5,
                )
            )
        return out


# Phase 0.1/0.5 (2026-08-12 refinement campaign): which constraint
# actually truncated delivery. "assembly_cap" = the document-count cap
# (the classifier's assembly_max_genes_cap on the pipeline; the
# ``max_genes`` request parameter on the packet path); "token_budget" =
# assembly's budget-eviction loop trimmed below the cap; "none" = every
# pool candidate above the floor was delivered (or the constraint
# cannot be determined — the safe degrade).
CAP_BINDINGS: tuple = ("assembly_cap", "token_budget", "none")

_CAP_BINDING_SET = frozenset(CAP_BINDINGS)


class DeliveryBlock(BaseModel):
    """Delivery-visibility block (refinement campaign Phase 0.1 + 0.5).

    Serve-side counterpart of the ablation ladder's delivered-gold
    math: the ladder computes delivered-gold from
    ``window.expressed_gene_ids``; this block serves the delivered ids
    in final order so any harness can compute delivered-gold /
    pool_delivery_gap exactly by joining ``delivered_gene_ids`` with
    its gold set (gold itself is unknowable at serve time).

    ``cap_binding`` answers the question the 2026-07-30 Gemma
    expression campaign could not: did the assembly cap (not the token
    budget) truncate delivery? Optional on ``ContextPacket`` — old
    readers see ``delivery: null`` and are unaffected.
    """

    model_config = {"extra": "forbid"}

    # Scored candidate pool size for this query (post-floor).
    pool_size: int = Field(default=0, ge=0)
    # Number of documents actually delivered.
    delivered_count: int = Field(default=0, ge=0)
    # Final delivered order — the ladder's join basis.
    delivered_gene_ids: List[str] = Field(default_factory=list)
    # Which constraint truncated delivery; one of CAP_BINDINGS.
    # Validated at runtime (not via Literal) so future one-line tuple
    # extensions don't require a pydantic type bump — same trade-off
    # as MissBlock.reason above.
    cap_binding: str = "none"

    @model_validator(mode="after")
    def _validate_cap_binding(self) -> "DeliveryBlock":
        if self.cap_binding not in _CAP_BINDING_SET:
            raise ValueError(
                f"DeliveryBlock.cap_binding {self.cap_binding!r} not in "
                f"CAP_BINDINGS ({sorted(CAP_BINDINGS)})"
            )
        return self


class ContextResponseEnvelope(BaseModel):
    """Thin wrapper that carries the know/miss exclusivity invariant.

    Used by /context and /context/packet routes to wrap the response
    dict before serialization. The validator raises if both blocks are
    present or both absent — caught at the route boundary; tests in
    test_know_miss_block.py assert it never fires under correct flow.

    Why a wrapper instead of a Union[KnowBlock, MissBlock]: the rest of
    the response dict (citations, agent metadata, expressed_context, …)
    is sibling-keyed at the same level. The envelope holds those plus
    the one mandatory know-or-miss key, and exposes the invariant as a
    pydantic check rather than a route-side `assert`.
    """

    model_config = {"extra": "allow"}

    know: Optional[KnowBlock] = None
    miss: Optional[MissBlock] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "ContextResponseEnvelope":
        if (self.know is None) == (self.miss is None):
            raise ValueError(
                "ContextResponseEnvelope: exactly one of know/miss must "
                "be set (got "
                f"know={'set' if self.know is not None else 'None'}, "
                f"miss={'set' if self.miss is not None else 'None'})"
            )
        return self


# Resolve the ContextPacket forward refs to KnowBlock / MissBlock now
# that both are defined. Without this, ContextPacket(...) blows up the
# first time a route tries to construct it.
ContextPacket.model_rebuild()


# ── Stage 5 (2026-05-08): caller_model_class wire format ────────────────────
#
# See docs/specs/2026-05-08-stage-5-caller-model-class.md §3. Opt-in render-
# branch selector for /context. The /context endpoint accepts an optional
# "caller_model_class" string; unknown values return 400. Default is
# "generic" — tests/test_caller_model_class.py checks current default/generic
# response parity and the archived decoder/render contract. See
# tests/golden/README.md for the historical snapshot's scope.


class CallerModelClass(str, Enum):
    """Caller model class for /context render-branch selection (spec §3)."""
    generic   = "generic"
    small_moe = "small_moe"
    frontier  = "frontier"


CALLER_MODEL_CLASS_DEFAULT = "generic"

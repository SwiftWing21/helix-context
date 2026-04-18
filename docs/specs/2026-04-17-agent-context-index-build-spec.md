# Helix As Agent Context Index — Build Spec

**Status:** Spec, 2026-04-17  
**Scope:** Turn Helix from a high-recall retrieval engine into an agent-safe context index.  
**Mode:** Additive. Preserve the current `/context` and sharded-genome paths while layering provenance, freshness, contradiction, and working-set logic on top.

## Why this exists

Helix already does the hard retrieval work:

- `helix_context/context_manager.py` orchestrates turn-time expression, splice, and assembly.
- `helix_context/shard_schema.py` and `helix_context/shard_router.py` are creating the physical index/routing layer.
- `helix_context/schemas.py` already models `Gene`, `ContextHealth`, session identity, and some version-ish links like `source_id` and `supersedes`.

What an agent still lacks is a way to answer:

1. Is this evidence relevant?
2. Is it still fresh enough to act on?
3. Is it authoritative?
4. What contradicts it?
5. What should be reread before editing, quoting, or making an operational decision?

This spec makes those questions first-class.

## Goal

Given a task, Helix should return a **context packet** that includes:

- the most relevant evidence
- the freshest authoritative evidence
- stale-risk items
- likely contradictions or superseding facts
- a recommended refresh plan before the agent acts

## Non-goals

- Replacing the current retrieval math in `Genome.query_genes`
- Rewriting the biology/software lexicon
- Requiring every ingest to become fully structured on day 1
- Solving cross-machine sync in this phase
- Touching `helix_context/context_manager.py` until the current merge conflict is resolved

## North star behavior

When an agent asks for context before a code edit, Helix should return:

- `verified`: evidence read from stable sources with acceptable freshness
- `stale_risk`: relevant evidence whose source is volatile or old
- `contradictions`: evidence that may supersede or conflict
- `refresh_targets`: exact files/entities that should be reread before acting
- `citations`: enough provenance to let the agent explain where each fact came from

This is a different product surface than “top-K genes.”

## Current repo anchors

This design intentionally builds on the current code:

- `helix_context/schemas.py`
  Existing `Gene`, `ContextHealth`, session, participant, and attribution models.
- `helix_context/shard_schema.py`
  Existing `main.db` routing layer with `shards` and `fingerprint_index`.
- `helix_context/shard_router.py`
  Existing shard fan-out and merge path.
- `helix_context/server.py`
  Existing HTTP entrypoint and future packet endpoints.
- `helix_context/mcp_server.py` and `helix_context/mcp/server.py`
  Existing MCP surfaces where agent-context tools should land.
- `docs/specs/2026-04-17-genome-sharding-plan.md`
  Existing physical-sharding direction that this spec extends, not replaces.

## Product framing

Helix should now be thought of as three layers:

1. **Storage layer**
   Sharded genomes holding content, complements, codons, embeddings, and retrieval indexes.

2. **Index layer**
   `main.db` as the routing, provenance, claims, and freshness authority.

3. **Agent context layer**
   Task-scoped packet building, refresh planning, contradiction surfacing, and working-set maintenance.

The existing system is strong on layers 1 and 2. This spec is mostly about making layer 3 real.

## Design principles

### 1. Keep the genome as the substrate

Do not create a parallel document store. `Gene` remains the canonical content-bearing unit.

### 2. Put routing and freshness authority in `main.db`

The shard plan already gives us the right place to hold lightweight metadata. Avoid forcing every query to open every shard just to answer provenance/freshness questions.

### 3. Index claims, not only text

Agents act on claims like:

- “the live DB path is `genomes/main/genome.db`”
- “`HELIX_USE_SHARDS` is off by default”
- “this benchmark result came from `genome_bench_helix.db`”

Those should be retrievable as structured units with provenance.

### 4. Freshness is task-dependent

An explanation can tolerate older context. An edit, quote, restart instruction, or file-path answer often cannot.

### 5. Contradictions must be surfaced, not hidden

For agents, stale confidence is more dangerous than sparse recall.

## Proposed data model

### A. Extend `Gene` metadata minimally

Add these fields to `helix_context/schemas.py:Gene`:

- `repo_root: Optional[str] = None`
- `source_kind: Optional[str] = None`
- `observed_at: Optional[float] = None`
- `mtime: Optional[float] = None`
- `content_hash: Optional[str] = None`
- `volatility_class: Optional[str] = None`
- `authority_class: Optional[str] = None`
- `support_span: Optional[str] = None`
- `last_verified_at: Optional[float] = None`

Allowed `source_kind` values:

- `code`
- `config`
- `doc`
- `log`
- `db`
- `benchmark`
- `tool_output`
- `session_note`
- `user_assertion`

Allowed `volatility_class` values:

- `stable`
- `medium`
- `hot`

Allowed `authority_class` values:

- `primary`
- `derived`
- `inferred`

These fields are mostly copied into `main.db` and should not require loading bulk gene content to answer freshness questions.

### B. Add `main.db` metadata tables

Add the following tables in `helix_context/shard_schema.py`.

#### `source_index`

One row per gene, lightweight provenance and freshness metadata.

Columns:

- `gene_id TEXT PRIMARY KEY`
- `shard_name TEXT NOT NULL`
- `source_id TEXT`
- `repo_root TEXT`
- `source_kind TEXT`
- `observed_at REAL`
- `mtime REAL`
- `content_hash TEXT`
- `volatility_class TEXT NOT NULL DEFAULT 'medium'`
- `authority_class TEXT NOT NULL DEFAULT 'primary'`
- `support_span TEXT`
- `last_verified_at REAL`
- `invalidated_at REAL`
- `updated_at REAL NOT NULL`

Indexes:

- `(shard_name)`
- `(source_id)`
- `(repo_root)`
- `(source_kind)`
- `(volatility_class)`

#### `claims`

Structured fact layer derived from genes.

Columns:

- `claim_id TEXT PRIMARY KEY`
- `gene_id TEXT NOT NULL`
- `shard_name TEXT NOT NULL`
- `claim_type TEXT NOT NULL`
- `entity_key TEXT`
- `claim_text TEXT NOT NULL`
- `extraction_kind TEXT NOT NULL DEFAULT 'literal'`
- `specificity REAL NOT NULL DEFAULT 0.5`
- `confidence REAL NOT NULL DEFAULT 0.5`
- `observed_at REAL`
- `supersedes_claim_id TEXT`
- `updated_at REAL NOT NULL`

Indexes:

- `(gene_id)`
- `(entity_key)`
- `(claim_type)`
- `(supersedes_claim_id)`

#### `claim_edges`

Graph of contradiction and support links.

Columns:

- `src_claim_id TEXT NOT NULL`
- `dst_claim_id TEXT NOT NULL`
- `edge_type TEXT NOT NULL`
- `weight REAL NOT NULL DEFAULT 1.0`
- `created_at REAL NOT NULL`

Allowed `edge_type` values:

- `contradicts`
- `supports`
- `supersedes`
- `duplicates`

Primary key:

- `(src_claim_id, dst_claim_id, edge_type)`

#### `working_sets`

Task/session-scoped belief registry.

Columns:

- `working_set_id TEXT NOT NULL`
- `session_id TEXT`
- `participant_id TEXT`
- `task_type TEXT NOT NULL`
- `belief_id TEXT NOT NULL`
- `belief_kind TEXT NOT NULL`
- `source_ref TEXT`
- `status TEXT NOT NULL`
- `action_dependency REAL NOT NULL DEFAULT 0.5`
- `added_at REAL NOT NULL`
- `last_checked_at REAL`
- `expires_at REAL`

Allowed `status` values:

- `live`
- `stale_risk`
- `needs_refresh`
- `retired`

Primary key:

- `(working_set_id, belief_id)`

### C. Extend `fingerprint_index`, do not replace it

Keep `fingerprint_index` as the routing prefilter. Add only the smallest extra columns if needed:

- `volatility_class`
- `authority_class`
- `last_verified_at`

Do not bloat this table with long text or claim payloads.

## Claim extraction model

Claims are the unit an agent reasons over.

### Claim types

Start with these:

- `path_value`
- `config_value`
- `api_contract`
- `entity_membership`
- `benchmark_result`
- `operational_state`
- `version_marker`
- `human_assertion`

### Extraction kinds

- `literal`
  Exact path, number, flag, symbol, API route, config value
- `derived`
  Summary produced from content but directly grounded in one gene
- `inferred`
  Higher-risk synthesis across multiple genes

V1 should prioritize `literal` and a little `derived`. Avoid trusting `inferred` claims for edits or ops.

### Entity keys

Every claim should try to emit one or more entity anchors:

- file path
- symbol name
- API route
- config key
- shard name
- model name
- benchmark name

## Freshness and staleness model

Helix should track two separate things:

- `relevance_score`
- `live_truth_score`

### Proposed components

#### `freshness_score`

```
freshness_score = exp(-age_seconds / half_life_seconds(volatility_class))
```

Suggested half-lives:

- `stable`: 7 days
- `medium`: 12 hours
- `hot`: 15 minutes

#### `authority_score`

Default weights:

- `primary`: 1.0
- `derived`: 0.75
- `inferred`: 0.45

#### `specificity_score`

Heuristic scale:

- exact literal path/value/symbol: `1.0`
- line-local statement or table-local state: `0.8`
- summary statement: `0.6`
- abstract inference: `0.3`

#### `contradiction_penalty`

```
contradiction_penalty = min(1.0, strongest_conflicting_edge_weight)
```

#### `live_truth_score`

```
live_truth_score =
    freshness_score
    * authority_score
    * specificity_score
    * (1 - contradiction_penalty)
```

### Action risk

Action risk is not the same as truth.

```
action_risk_score =
    volatility_weight(task_type)
    * (1 - freshness_score)
    + contradiction_penalty
    + exactness_penalty_if_action_requires_literal
```

Where `volatility_weight(task_type)` is higher for `edit`, `ops`, and `quote` than for `plan` or `explain`.

## Task profiles

Helix should rank differently by task type.

### `plan`

- favor semantic coverage
- tolerate older docs/config summaries
- low refresh pressure

### `explain`

- favor relevance first
- surface freshness hints but do not force reread unless contradictions exist

### `edit`

- favor exact literal evidence
- heavily penalize stale-risk code/config claims
- always emit refresh targets for edited entities

### `review`

- retrieve relevant evidence plus nearby contradictions and version markers

### `debug`

- favor newest logs, runtime state, and config values
- stale documents are background only

### `ops`

- almost no tolerance for hot-source staleness
- path, port, process, and config claims should be freshly verified

## Packet model

Add these models to `helix_context/schemas.py`.

### `ContextItem`

- `kind: str`
- `gene_id: Optional[str]`
- `claim_id: Optional[str]`
- `title: str`
- `content: str`
- `relevance_score: float`
- `live_truth_score: float`
- `source_id: Optional[str]`
- `source_kind: Optional[str]`
- `volatility_class: Optional[str]`
- `authority_class: Optional[str]`
- `last_verified_at: Optional[float]`
- `status: str`
- `citations: list[str]`

### `RefreshTarget`

- `target_kind: str`
- `source_id: str`
- `reason: str`
- `priority: float`

### `ContextPacket`

- `task_type: str`
- `query: str`
- `verified: list[ContextItem]`
- `stale_risk: list[ContextItem]`
- `contradictions: list[ContextItem]`
- `refresh_targets: list[RefreshTarget]`
- `working_set_id: Optional[str]`
- `notes: list[str]`

## API additions

### Python API

Add a new service module, ideally `helix_context/context_packet.py`, so we do not force packet logic into the currently conflicted `context_manager.py`.

Proposed functions:

- `build_context_packet(query, task_type="explain", session_id=None, participant_id=None) -> ContextPacket`
- `get_refresh_targets(query, task_type="edit") -> list[RefreshTarget]`
- `query_claims(entity_key=None, claim_type=None, source_kind=None) -> list[Claim]`
- `update_working_set(packet: ContextPacket) -> str`

### HTTP API

Additive endpoints in `helix_context/server.py`:

- `POST /context/packet`
- `POST /context/refresh-plan`
- `GET /claims`
- `GET /working-set/{working_set_id}`

Example `POST /context/packet` body:

```json
{
  "query": "where is the live genome path configured",
  "task_type": "edit",
  "session_id": "abc123",
  "participant_id": "claude-local"
}
```

### MCP surface

Add tools to the canonical MCP server, not the legacy one:

- `helix_context_packet`
- `helix_refresh_targets`
- `helix_claims`

Do not remove existing context tools; this is an additive “agent-safe” surface.

## Integration plan

### Phase 1 — Provenance hardening

Files:

- `helix_context/schemas.py`
- `helix_context/shard_schema.py`
- `helix_context/genome.py`
- ingest scripts that currently default to cwd-relative `genome.db`

Tasks:

- add `Gene` provenance/freshness fields
- add `source_index` table
- write `upsert_source_index(...)`
- normalize `source_id` and `repo_root`
- stop ambiguous cwd-relative DB writes for index-sensitive scripts

Deliverable:
Every new ingest writes enough metadata to answer “what is this and how fresh is it?”

### Phase 2 — Claims

Files:

- new `helix_context/claims.py`
- `helix_context/shard_schema.py`
- optional ingest hooks in `genome.py`

Tasks:

- create `claims` and `claim_edges`
- implement literal-claim extraction for code/config/docs/benchmarks
- add entity-key extraction

Deliverable:
Helix can answer structured fact questions without reopening bulk content.

### Phase 3 — Packet builder

Files:

- new `helix_context/context_packet.py`
- `helix_context/server.py`
- `helix_context/mcp_server.py`

Tasks:

- implement task-profile-aware ranking
- emit `ContextPacket`
- compute refresh targets
- update working set

Deliverable:
Agent-safe packet API available over Python, HTTP, and MCP.

### Phase 4 — Context-manager integration

Only after the merge conflict in `helix_context/context_manager.py` is resolved.

Tasks:

- optionally let `/context` consume packet status when building expressed context
- let packet builder reuse current retrieval/tier contribution introspection

Deliverable:
One integrated path instead of packet builder living beside context manager.

### Phase 5 — Benchmarks

Files:

- `benchmarks/`
- `tests/`

New benchmark families:

- stale path answer
- conflicting config values
- changed file after ingest
- generated log contamination
- duplicate fact across shards with one newer source

Metrics:

- stale fact reuse rate
- contradiction miss rate
- unnecessary refresh rate
- wrong-edit-from-old-context rate
- time-to-safe-packet

## Tests

Add new test files:

- `tests/test_source_index.py`
- `tests/test_claims.py`
- `tests/test_context_packet.py`
- `tests/test_refresh_policy.py`
- `tests/test_contradictions.py`

Critical scenarios:

1. Same entity appears in two shards; newer claim supersedes older.
2. Relevant old doc is returned for `plan` but flagged `stale_risk` for `edit`.
3. Hot-source claim older than threshold yields `needs_refresh`.
4. Packet builder returns exact refresh target for a path-bearing claim.
5. MCP tool returns citations and status labels, not only raw content.

## Migration notes

This should be safe to ship incrementally:

- old `Gene` rows can leave new fields null
- `source_index`, `claims`, `claim_edges`, and `working_sets` are additive
- packet APIs can launch before `/context` integration
- the shard router continues to function unchanged if none of the new tables are queried

## Open decisions

### 1. Where claim extraction runs

Options:

- ingest time only
- lazy on first packet query
- hybrid

Recommendation:
Hybrid. Do literal claims at ingest; allow lazy repair/backfill for old genomes.

### 2. Whether `source_index` is per-gene or per-source

Recommendation:
Per-gene first. It matches current storage and avoids inventing a many-to-many layer too early.

### 3. Whether packet building should reopen shards for exact verification

Recommendation:
Yes, but selectively. Only for:

- high-risk task types
- hot sources
- contradiction candidates
- path/value-bearing claims

## First milestone

The first milestone that changes agent behavior without destabilizing the current stack is:

**“Helix can return a context packet where each item is labeled `verified`, `stale_risk`, or `needs_refresh`, using metadata stored in `main.db`.”**

That is the shortest path from “retrieval engine” to “agent context tool.”

# Helix Session Registry — Parties, Participants, and Authored Genes

A presence and attribution layer for multi-session Helix usage. Lets sibling
Claude/Gemini sessions see each other, tag what they ingest, and retrieve each
other's recent work without relying on BM25 to surface short broadcasts.

**Status:** **SHIPPED** (core + HITL-1 + federation extensions landed
2026-04-11 → 2026-04-14). See "Status — what shipped" below for the
commit-level trail. Remaining work is post-landing refinement, not
core implementation.
**Target version:** `helix-context v0.4.0b1`.
**Depends on:** nothing — purely additive to v0.3.0b5.
**Related:** [`RESTART_PROTOCOL.md`](RESTART_PROTOCOL.md) (complementary — restart
protocol announces outages, session registry announces presence).

## Status — what shipped

| Commit | Date | Slice |
|---|---|---|
| `8f24913` | 2026-04-11 | Session registry core + launcher (supervisor + dashboard) + token metrics |
| `62280c6` | 2026-04-11 | Citation enrichment (`authored_by_party`/`authored_by_handle`), background sweep, `AgentBridge` HTTP client |
| `599fe98` | 2026-04-11 | Launcher tray, install-service, `/admin/shutdown`, orphan adoption |
| `174aea6` | 2026-04-11 | HITL-1: `hitl_events` table + `emit_hitl_event`/`get_hitl_events`/`hitl_rate`/`hitl_stats` + 24 tests |
| `5563763` | ~2026-04-12 | `session_context` + OS-level attribution defaults |
| `8990fb7` | ~2026-04-12 | 4-layer identity model (org / device / user / agent) |
| `a62ca4c` | ~2026-04-12 | Timezone forensics — `parties.timezone` + `authored_tz` |

Known gaps (tracked as follow-up work, not blockers):

- **MCP host participant registration — CLOSED 2026-04-14.**
  `helix_context/mcp_server.py` now calls
  `AgentBridge.register_participant(..., start_auto_heartbeat=True)` on
  startup. Handle reads from `HELIX_MCP_HANDLE` (default `mcp-<pid>`),
  party from `HELIX_PARTY_ID` (default `swift_wing21`), host-tag from
  `HELIX_MCP_HOST` (e.g. `claude-code`, `antigravity`, `cursor`).
  Registration failure is non-fatal — tool calls still proxy. Each
  MCP-host spawn gets its own `participant_id`; stale participants
  TTL out naturally.
- **Phase 2** (GeneAttribution wired into `query_genes` for per-party
  scoping + authorship-class scoring) — design in
  `~/.helix/shared/handoffs/2026-04-11_8d_dimensional_roadmap.md` §
  Phase 2. Unblocked now that the registry is on master.

The rest of this document is the original design spec, preserved for
reference. Any divergence between the spec below and the shipped code
should be resolved by the code.

---

## Motivation

Today, multiple Claude sessions can share one Helix server (via the bridge) but
they cannot:

1. **See each other.** There is no presence/liveness layer. If Laude wants to
   know whether Raude is currently working, there is nothing to query.
2. **Attribute genes.** `/ingest` accepts an opaque `metadata` dict but nothing
   standardizes "who authored this." The informal `source="laude"` convention in
   `AgentBridge.drop_to_inbox` is a hint, not a contract, and doesn't survive
   into the genome.
3. **Broadcast reliably.** Short text notes (e.g., "VS Code 1.115 shipped, check
   out the Agents companion") get drowned in BM25 retrieval by the much larger
   code and spec genes. Empirically verified 2026-04-10: two ingests totalling
   ~400 chars were both invisible to targeted queries after ingestion because
   the retrieval budget preferred older, larger matches.

The session registry solves all three with three additive concepts: a **party**
layer for stable execution-side identity, a **participant** layer for live actors, and an
**authored-genes** view that bypasses BM25 for recency-sorted retrieval.

## The layering

```
participants  runtime actors (Claude sessions, sub-agents, swarm members) — ephemeral
    ↓ belongs to
parties       atomic principals (humans, tenants, org service identities) — stable
    ↓ (future, not in this spec) member of
collectives   recursive groupings with access structures (orgs, communities, consortiums)
```

**Party** = the stable substrate-level principal for session presence. In the
single-user case today, every Claude panel a user runs is a participant under
the **one** party representing that device or host context. In future
federation, party can still represent the execution-side principal even when
the org layer above it carries the broader trust root.

**Participant** = a live actor under a party. A Claude Code panel is a
participant. An `Agent`-tool sub-agent is a participant (under the same party
as its spawner). A future swarm member is a participant.

Terminology note: in the 4-layer authored-ingest model, `participant` refers
to the human principal and `agent` to the software actor working on that
human's behalf. The session registry predates that fuller split, so older
registry prose may describe a live Claude session itself as a participant.
That historical wording remains readable; see [`ROSETTA.md`](ROSETTA.md) and
[`FEDERATION_LOCAL.md`](FEDERATION_LOCAL.md) for the current crosswalk.

**Why this layering?** Shamir Secret Sharing and threshold-cryptography
literature (see Discussion #5) operate across *principals*, not across an
individual's windows. "K of Max's 3 panels must agree" is meaningless — Max
already trusts himself. "K of N collaborating humans must agree" is the real
SSS use case. Putting the party layer at the human/tenant level means the
eventual SSS threshold is already at the right abstraction.

**Hard constraint.** `parties` MUST NOT self-reference. The recursive grouping
layer is deferred to a future `collectives` table (see [Forward compatibility](#forward-compatibility-collectives)).
Resist the temptation to add `parent_party_id` — it commits to the wrong
recursion point and muddies "who holds the share."

## Schema additions

Two new tables in `genome.db`, plus one new column on `genes`.

### `parties`

```sql
CREATE TABLE IF NOT EXISTS parties (
    party_id      TEXT PRIMARY KEY,         -- "max@local", "tenant:acme", "peer:swiftwing21"
    display_name  TEXT NOT NULL,            -- "max", "acme corp", "swiftwing21"
    trust_domain  TEXT NOT NULL DEFAULT 'local',  -- "local" | "remote" | "tenant:*"
    created_at    REAL NOT NULL,            -- unix epoch
    metadata      TEXT                      -- optional JSON blob for future fields
);

CREATE INDEX IF NOT EXISTS idx_parties_trust_domain ON parties(trust_domain);
```

### `participants`

```sql
CREATE TABLE IF NOT EXISTS participants (
    participant_id   TEXT PRIMARY KEY,      -- uuid4 or "taude-<pid>-<nonce>"
    party_id         TEXT NOT NULL REFERENCES parties(party_id),
    handle           TEXT NOT NULL,         -- "taude", "laude", "raude", "subagent-7f3a"
    workspace        TEXT,                  -- absolute path of the workspace, if known
    pid              INTEGER,               -- OS process id of the agent runtime, if known
    started_at       REAL NOT NULL,
    last_heartbeat   REAL NOT NULL,
    status           TEXT NOT NULL DEFAULT 'active',  -- "active" | "idle" | "stale" | "gone"
    capabilities     TEXT,                  -- optional JSON: ["ingest", "query", "admin"]
    metadata         TEXT                   -- optional JSON blob
);

CREATE INDEX IF NOT EXISTS idx_participants_party ON participants(party_id);
CREATE INDEX IF NOT EXISTS idx_participants_heartbeat ON participants(last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_participants_status ON participants(status);
```

### `gene_attribution`

Attribution lives in its own table rather than bolting columns onto `genes`.
Rationale: the existing `genes` table has no top-level `created_at` column
(creation time lives inside the `epigenetics` JSON blob), so the recency
index we need for `GET /sessions/{handle}/recent` would have nothing to sort
on without denormalizing. A dedicated attribution table is cleaner: it keeps
`genes` unchanged, gives us native indexable `authored_at`, and makes
attribution a first-class concern rather than a squatter on the gene row.

```sql
CREATE TABLE IF NOT EXISTS gene_attribution (
    gene_id         TEXT PRIMARY KEY
                    REFERENCES genes(gene_id) ON DELETE CASCADE,
    party_id        TEXT NOT NULL
                    REFERENCES parties(party_id),
    participant_id  TEXT
                    REFERENCES participants(participant_id) ON DELETE SET NULL,
    authored_at     REAL NOT NULL   -- unix epoch, stamped at ingest time
);

CREATE INDEX IF NOT EXISTS idx_attribution_party_time
    ON gene_attribution(party_id, authored_at DESC);
CREATE INDEX IF NOT EXISTS idx_attribution_participant_time
    ON gene_attribution(participant_id, authored_at DESC);
```

Semantics:

- One row per attributed gene. Genes without attribution (legacy ingests,
  bridge inbox drops without a `participant_id`) simply have no row in this
  table — they remain retrievable by the normal BM25 path.
- `gene_id` is the primary key because a gene has exactly one author.
- `ON DELETE CASCADE` on the `gene_id` FK means vacuuming a gene from the
  genome cleans up its attribution automatically.
- `participant_id` is nullable so that party-level attribution (e.g., a
  server-side ingest that knows the party but not the specific participant)
  is expressible.

### Migration

- All schema changes are idempotent (`CREATE TABLE IF NOT EXISTS`,
  `CREATE INDEX IF NOT EXISTS`).
- On server startup, run a one-shot `_ensure_registry_schema()` that creates
  `parties`, `participants`, and `gene_attribution` inside a single
  transaction.
- No data migration needed. Historical genes have no `gene_attribution` row
  and the retrieval paths treat absence as "unknown author, goes through
  normal BM25."

## API endpoints

All endpoints live alongside existing ones at the root of the FastAPI app
(same server, same port `:11437`).

### `POST /sessions/register`

A participant announces itself. Must be called at least once per Claude
session before any other session-aware endpoint is useful.

```bash
curl -X POST http://127.0.0.1:11437/sessions/register \
  -H "Content-Type: application/json" \
  -d '{
    "party_id": "max@local",
    "handle": "taude",
    "workspace": "/f/Projects/Education",
    "pid": 48213,
    "capabilities": ["ingest", "query"]
  }'
```

Request body:

| Field | Type | Required | Notes |
|---|---|---|---|
| `party_id` | string | yes | See [Trust model](#trust-model-deferred). Self-asserted. |
| `handle` | string | yes | Short name — e.g., `"taude"`, `"laude"`, `"raude"`, `"subagent-X"`. Not unique; multiple participants can share a handle across time. |
| `workspace` | string | no | Absolute path of the workspace. Helpful for later filtering. |
| `pid` | int | no | OS process id of the runtime. |
| `capabilities` | list[str] | no | What this participant can do. Free-form for now. |
| `metadata` | object | no | Arbitrary JSON for future extension. |

Response:

```json
{
  "participant_id": "01HXY3ZK8R4P2V6QW9M5T7N0EG",
  "party_id": "max@local",
  "registered_at": 1775980800.123,
  "heartbeat_interval_s": 30,
  "ttl_s": 120
}
```

Behavior:

- If `party_id` does not exist in the `parties` table, the server creates it
  automatically with `trust_domain="local"` and `display_name=party_id`.
  (This is the trust-on-first-use behavior. See [Trust model](#trust-model-deferred).)
- `participant_id` is generated server-side (ULID preferred for sortability).
- The response includes `heartbeat_interval_s` (how often to refresh) and
  `ttl_s` (grace period before the server marks the participant `stale`).

### `POST /sessions/{participant_id}/heartbeat`

Refresh liveness. Should be called roughly every `heartbeat_interval_s`.

```bash
curl -X POST http://127.0.0.1:11437/sessions/01HXY3ZK8R4P2V6QW9M5T7N0EG/heartbeat
```

Response:

```json
{
  "ok": true,
  "ttl_remaining_s": 118,
  "status": "active"
}
```

If the participant was previously `stale` or `gone`, the heartbeat resurrects
it and returns `status: "active"`. If the `participant_id` is unknown, returns
`404` and the client should re-register.

### `GET /sessions`

List currently-known participants with status and last-seen times.

```bash
curl http://127.0.0.1:11437/sessions
```

Query params:

| Param | Default | Notes |
|---|---|---|
| `party_id` | *(all)* | Filter to one party. |
| `status` | `active` | One of `active`, `idle`, `stale`, `gone`, `all`. |
| `workspace` | *(all)* | Filter by workspace path prefix. |

Response:

```json
{
  "participants": [
    {
      "participant_id": "01HXY3ZK8R4P2V6QW9M5T7N0EG",
      "party_id": "max@local",
      "handle": "taude",
      "workspace": "/f/Projects/Education",
      "status": "active",
      "last_seen_s_ago": 12,
      "started_at": 1775980800.123
    },
    {
      "participant_id": "01HXY3ZMY5Z8JXQ6W0E1R4T7VA",
      "party_id": "max@local",
      "handle": "laude",
      "workspace": "/f/Projects/Education",
      "status": "active",
      "last_seen_s_ago": 4,
      "started_at": 1775980755.0
    }
  ],
  "count": 2
}
```

### `GET /sessions/{handle}/recent`

**The BM25 bypass.** Returns the most recent genes authored by any participant
with a given handle, in reverse chronological order, with no retrieval
scoring. This is the reliable broadcast channel.

```bash
curl "http://127.0.0.1:11437/sessions/taude/recent?limit=10"
```

Query params:

| Param | Default | Notes |
|---|---|---|
| `limit` | 10 | Max genes to return. |
| `party_id` | *(all)* | Scope to one party if handles are ambiguous across parties. |
| `since` | *(none)* | Unix timestamp — only return genes created after this. |

Response:

```json
{
  "handle": "taude",
  "genes": [
    {
      "gene_id": "b218ad70cf0ef909",
      "content_preview": "VS Code 1.115 released 2026-04-08 with VS Code Agents...",
      "authored_at": 1775980812.4,
      "party_id": "max@local",
      "participant_id": "01HXY3ZK8R4P2V6QW9M5T7N0EG"
    }
  ],
  "count": 1
}
```

Implementation: the endpoint resolves `handle` to the set of
`participant_id`s ever associated with that handle (via `participants` table),
then selects from `gene_attribution` ordered by `authored_at DESC`, joining
`genes` for the content preview. Scoped by `party_id` when provided to
disambiguate same-handle-different-party cases.

Note the endpoint takes `handle` rather than `participant_id` because handles
are human-meaningful and participants are ephemeral. Two sequential Taude
sessions will have different `participant_id`s but the same `handle`.

### Extension to `POST /ingest`

The existing endpoint gains two optional fields. No existing caller is broken.

```bash
curl -X POST http://127.0.0.1:11437/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "VS Code 1.115 released 2026-04-08 with Agents companion app...",
    "content_type": "text",
    "participant_id": "01HXY3ZK8R4P2V6QW9M5T7N0EG"
  }'
```

New optional fields:

| Field | Type | Notes |
|---|---|---|
| `participant_id` | string | If present, the resulting genes are tagged with both the participant and its party (looked up via the registry). |
| `party_id` | string | Rarely needed — only for server-side ingestion flows that don't have a participant but know the party. Ignored if `participant_id` is also provided. |

Behavior:

- If `participant_id` is unknown, the ingest still succeeds but no
  `gene_attribution` row is written (and a warning is logged). This
  preserves the "registry is additive" invariant — ingest never fails
  because of registry state.
- Tagged ingests write one `gene_attribution` row per resulting gene, with
  `authored_at = time.time()` at ingest time (not the epigenetic
  `created_at`, though in practice they are within milliseconds of each other
  for new genes).
- Tagged ingests also update `last_heartbeat` on the participant row
  (implicit heartbeat on activity — saves a round trip).

### Extension to `POST /context`

No breaking change. The citation objects in the `agent.citations` response
array gain two optional fields when the source gene has a
`gene_attribution` row:

```json
{
  "gene_id": "b218ad70cf0ef909",
  "source": "",
  "score": 42.1,
  "authored_by_party": "max@local",
  "authored_by_handle": "taude"
}
```

Implementation: a LEFT JOIN from the expressed gene_ids to `gene_attribution`
and `participants` resolves the party and current handle in a single query
alongside the existing citation enrichment. Genes without attribution simply
omit both fields.

## Trust model (deferred)

**This spec does NOT implement authentication.** Parties are **self-asserted**
on registration (trust-on-first-use). A malicious client on the local machine
could register as any `party_id` it wants and tag genes accordingly.

For the current use case — single user running multiple Claude panels against
a local Helix server on `localhost:11437` — this is acceptable. The threat
model is "my own panels, on my own machine." No untrusted code is calling the
registry.

**Before any of the following work picks up, auth must be designed:**

- Multi-tenant SaaS Helix
- Federated Helix (cross-instance party identities)
- Any SSS threshold work that relies on party identity being non-forgeable
- Exposing the FastAPI server beyond `localhost`

Candidate directions when it's time:

- Bearer tokens scoped to party_id, issued by an out-of-band registration
  flow (OIDC, OAuth, or a bootstrap CLI).
- mTLS for party-to-party federation transport.
- Signed heartbeat payloads so a long-lived participant proves continuity.

None of this is in this spec. Noted here so it's not forgotten.

## Forward compatibility: collectives

A future `collectives` table will handle org/community/consortium grouping.
It is **intentionally NOT in this spec**, but the current design must not
preclude it.

Expected shape when it arrives:

```sql
CREATE TABLE collectives (
    collective_id         TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    parent_collective_id  TEXT REFERENCES collectives(collective_id),
    access_structure      TEXT,  -- JSON: K-of-N, weighted, hierarchical policies
    created_at            REAL NOT NULL
);

CREATE TABLE collective_members (
    collective_id  TEXT NOT NULL REFERENCES collectives(collective_id),
    member_type    TEXT NOT NULL CHECK(member_type IN ('party', 'collective')),
    member_id      TEXT NOT NULL,
    role           TEXT,
    weight         INTEGER,
    PRIMARY KEY (collective_id, member_type, member_id)
);
```

Collectives recurse; parties do not. A collective may contain parties, other
collectives, or both. Access structures (K-of-N, weighted, hierarchical SSS
policies) live on the collective where the trust decision is actually made.

**The one constraint this spec imposes for collectives to slot in cleanly:**
`parties` must stay atomic (no `parent_party_id` column, ever).

## Python client

A thin helper lives on the existing `AgentBridge` class:

```python
from helix_context.bridge import AgentBridge

bridge = AgentBridge()

# Register once at session start
me = bridge.register_participant(
    party_id="max@local",
    handle="taude",
    workspace="/f/Projects/Education",
    capabilities=["ingest", "query"],
)
# me.participant_id is now available

# Heartbeat periodically (or let AgentBridge do it on a timer)
bridge.heartbeat()

# List siblings
siblings = bridge.list_sessions(party_id="max@local")
# -> [ParticipantInfo(handle="laude", ...), ParticipantInfo(handle="raude", ...)]

# See what laude has been doing recently
recent = bridge.recent_by_handle("laude", limit=5)
# -> [GenePreview(gene_id=..., content_preview=..., created_at=...), ...]

# Ingest with attribution (auto-uses registered participant)
bridge.ingest("VS Code 1.115 shipped today", content_type="text")
```

The existing `drop_to_inbox(source="...")` path is preserved and maps to
`participant_id=None` ingests for backwards compatibility. New code should
prefer `bridge.ingest()`.

## TTL and sweep

Participants transition through states based on `last_heartbeat`:

| Time since heartbeat | Status |
|---|---|
| ≤ `ttl_s` (default 120s) | `active` |
| ≤ `2 * ttl_s` | `idle` |
| ≤ `24h` | `stale` |
| > `24h` | `gone` (soft-deleted after 7 days) |

A periodic sweep task runs every `ttl_s / 2` seconds, updates `status` based
on `last_heartbeat`, and logs transitions at INFO level.

Sweep is NOT destructive — `gone` participants are kept for historical
attribution of their authored genes. Only after 7 days of `gone` status are
they hard-deleted, at which point the `gene_attribution.participant_id`
column is NULLed out for any rows that referenced them. The `party_id`
remains intact — party-level attribution survives participant turnover,
which matches the intent: the party (human/tenant) is the durable trust
identity, the participant is the ephemeral actor.

## Failure modes

| Failure | Behavior |
|---|---|
| Client registers, never heartbeats | Transitions active → idle → stale → gone on schedule. No impact on genes already authored. |
| Client heartbeats unknown participant_id | `404`, client re-registers. |
| Two participants register with the same handle simultaneously | Both get distinct `participant_id`s. They coexist. `GET /sessions/{handle}/recent` returns genes from all of them (ordered by time). |
| Server restarts mid-session | All `participants` rows remain in the DB. On next heartbeat, participants resume active status. If the restart was announced via the restart protocol, observing sessions already know to pause. |
| Ingest with unknown `participant_id` | Ingest succeeds, gene is NOT tagged, server logs warning. |
| Sweep task dies | No correctness impact — `GET /sessions` computes status on the fly from `last_heartbeat`. The persisted `status` column is a cache, not the source of truth. |
| Party row deleted while participants reference it | FK constraint blocks the delete. To remove a party, participants must be removed first, or the delete must cascade explicitly. |
| Gene deleted (vacuum/consolidate) | `ON DELETE CASCADE` on `gene_attribution.gene_id` cleans up automatically. No orphaned attribution rows. |
| Legacy gene re-ingested after registry shipped | A new `gene_attribution` row is written for the re-ingest. If the same `gene_id` already had an attribution row, the PK collision is resolved by updating `party_id` / `participant_id` / `authored_at` to the newer ingest (most recent author wins). |

## What this spec does NOT do

Explicitly out of scope, to keep the first slice shippable:

- **Auth.** See [Trust model](#trust-model-deferred).
- **Inter-instance federation.** Registry is local to one Helix server. Cross-server
  coordination needs a transport layer and federation protocol that do not yet
  exist.
- **Collectives.** See [Forward compatibility](#forward-compatibility-collectives).
- **Push notifications / wake-up signals.** Participants only learn of siblings
  when they query. No pub/sub, no peer-triggered execution. This is
  deliberate — autonomous peer-triggered execution crosses the Usage Policy
  line for agent orchestration.
- **Cryptographic operations.** Shamir split/combine is orthogonal and lives
  in a future `helix_context/sss.py`.
- **Harness integration.** Each client decides when to register and
  heartbeat. A future doc can describe `SessionStart` hook patterns for
  Claude Code, Gemini CLI, etc.

## Implementation checklist

Rough ordering for the first PR. Each item is a small, independently
testable unit.

1. **Schema migration.** `_ensure_registry_schema()` in `genome.py`, idempotent,
   runs on server startup.
2. **Dataclasses.** `Party`, `Participant`, `ParticipantInfo` in `schemas.py`.
3. **Registry DAL.** `registry.py` (new module) — `register_participant`,
   `heartbeat`, `list_participants`, `get_recent_by_handle`, `sweep`.
4. **FastAPI endpoints.** Wire into `server.py` alongside existing endpoints.
5. **Ingest extension.** `ingest_endpoint` accepts `participant_id`, resolves
   `party_id` via registry, writes both columns on the gene row.
6. **Context enrichment.** `context_endpoint` citation objects gain
   `authored_by_party` / `authored_by_handle` when present.
7. **Sweep task.** Add to the lifespan startup block, respecting
   `helix.toml [registry]` config (interval, ttl_s).
8. **Python client.** Methods on `AgentBridge`.
9. **Tests.** Unit tests for DAL + schema migration; integration tests for
   the four endpoints + ingest tagging + context citation enrichment.
10. **Docs.** Update `README.md` Quick Start with a "Multi-session"
    subsection pointing at this doc.

## Related

- [`RESTART_PROTOCOL.md`](RESTART_PROTOCOL.md) — how multiple sessions handle
  server restarts. Complementary: restart protocol is about absence, session
  registry is about presence.
- [Discussion #5 — Shamir Secret Sharing exploration](https://github.com/SwiftWing21/helix-context/discussions/5)
  — where the party layering decision comes from.
- `helix_context/bridge.py` — the existing `AgentBridge` abstraction that this
  spec extends.

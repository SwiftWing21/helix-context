# Federation, Local-First — 4-Layer OS-Level Attribution

> *"The simplest path to federation is to use what the OS already knows."*

This document describes the local federation primitive shipped 2026-04-12.
It captures **4-layer** identity attribution for every ingested gene
with zero auth infrastructure — using OS environment as the source of
truth. It's the on-ramp to the full enterprise federation model in
[`ENTERPRISE.md`](ENTERPRISE.md): same schema, different ID resolver.

---

## The 4-layer model

Real-world identity has at least four independent axes that we want to
query separately:

| Layer | What it represents | Example | Why we need it separately |
|---|---|---|---|
| **org** | tenant / team / company | `swiftwing` | Compliance, billing, cross-device aggregation |
| **device** (party) | physical machine | `gandalf` | Cross-machine vs cross-account separation |
| **user** (participant) | human | `max` | "Who created this?" — the human in the loop |
| **agent** | AI persona | `laude`, `conductor` | "Which AI did the work?" — distinguishes Laude vs Taude vs manual |

Each gene's `gene_attribution` row carries all four. Any may be NULL —
NULL `agent_id` = "manual ingest, no AI involved." NULL `org_id` =
defaults to the seeded `local` org.

This lets us answer questions that would otherwise require human
forensics:
- *"What did SwiftWing produce this quarter?"* — filter by org
- *"What did the gandalf machine create today?"* — filter by device
- *"What did Max work on this week?"* — filter by user
- *"What did Laude specifically build?"* — filter by agent
- *"What did Laude on gandalf, on max's behalf, in SwiftWing, do?"* — composite

## Schema

```sql
-- Layer 1: orgs (top-level tenant)
CREATE TABLE orgs (
    org_id        TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    trust_domain  TEXT NOT NULL DEFAULT 'local',
    created_at    REAL NOT NULL,
    metadata      TEXT
);
-- Seeded with ('local', 'Local Org (default)') so trust-on-first-use
-- writes always have a valid FK target.

-- Layer 2: parties (devices) — extended with org_id link
CREATE TABLE parties (
    party_id      TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    trust_domain  TEXT NOT NULL DEFAULT 'local',
    org_id        TEXT REFERENCES orgs(org_id),  -- added 2026-04-12
    created_at    REAL NOT NULL,
    metadata      TEXT
);

-- Layer 3: participants (humans) — unchanged
CREATE TABLE participants (
    participant_id TEXT PRIMARY KEY,
    party_id       TEXT NOT NULL REFERENCES parties(party_id),
    handle         TEXT NOT NULL,
    -- + workspace, pid, started_at, last_heartbeat, status, capabilities, metadata
);

-- Layer 4: agents (AI personas under a participant) — new
CREATE TABLE agents (
    agent_id        TEXT PRIMARY KEY,
    participant_id  TEXT NOT NULL REFERENCES participants(participant_id),
    handle          TEXT NOT NULL,
    kind            TEXT,                  -- "claude-code", "gemini", ...
    created_at      REAL NOT NULL,
    last_seen_at    REAL,
    metadata        TEXT,
    UNIQUE (participant_id, handle)
);

-- The 4-axis attribution row
CREATE TABLE gene_attribution (
    gene_id         TEXT PRIMARY KEY REFERENCES genes(gene_id),
    org_id          TEXT REFERENCES orgs(org_id),               -- added 2026-04-12
    party_id        TEXT NOT NULL REFERENCES parties(party_id),
    participant_id  TEXT REFERENCES participants(participant_id),
    agent_id        TEXT REFERENCES agents(agent_id),           -- added 2026-04-12
    authored_at     REAL NOT NULL
);
-- Indexes on each axis × authored_at for fast filtered range queries.
```

What changes across federation tiers is **the source of those IDs**:

| Tier | org_id | party_id | participant_id | agent_id |
|---|---|---|---|---|
| **Solo / single-persona (today)** | `'local'` | hostname | OS user UUID | NULL or HELIX_AGENT |
| **Multi-persona dev box** | `HELIX_ORG` env | hostname | OS user UUID | `HELIX_AGENT` per session |
| **Small team** | `HELIX_ORG` env | `HELIX_DEVICE` env | `HELIX_USER` env | `HELIX_AGENT` env |
| **Enterprise SSO** | OAuth org claim | hostname / SaaS gateway | OAuth user claim | request header / agent token |

Because the schema is invariant, every gene attributed at the local tier
remains validly attributed when SSO comes online. The auth edge replaces
the resolver; the rest of the pipeline (`query_genes(party_id=...)`, the
`/context` endpoint, the chromatin tier rules) keeps working unchanged.

## How it resolves IDs today (4-axis)

`helix_context/server.py::_local_attribution_defaults()` returns a
4-tuple `(user_handle, device, org, agent_handle)`:

```python
org           = os.environ.get("HELIX_ORG")    or "local"
device        = os.environ.get("HELIX_DEVICE") or os.environ.get("HELIX_PARTY") or socket.gethostname()
user_handle   = os.environ.get("HELIX_USER")   or getpass.getuser()
agent_handle  = os.environ.get("HELIX_AGENT")  or None   # None = manual ingest
```

Order of precedence, each axis:
1. **Explicit env var** (e.g., `HELIX_AGENT=laude`)
2. **Legacy env var** (e.g., `HELIX_PARTY` for back-compat with the
   2-layer commit)
3. **OS-derived value** (`getpass.getuser()`, `socket.gethostname()`)
4. **Sensible default** (`org='local'`, `agent=None`)

Tokens are normalized: lowercased, whitespace-stripped, length-capped at
64 chars. Conservative because these become primary-key components.

Empirical verification (running server, `HELIX_ORG=swiftwing
HELIX_AGENT=conductor`):

```
gene_attribution:
  gene=08d02ea38de4..  org=swiftwing  dev=gandalf  user=max  agent=conductor
```

## How it wires through `/ingest`

When the request body provides no `participant_id`, the server walks
the 4-layer find-or-create chain (org → device → user → agent) using
`Registry.local_org()`, `local_participant(org_id=…)`, and
`local_agent()`. Each step is **trust-on-first-use**: first call creates
the row, subsequent calls reuse it.

```python
# In /ingest endpoint
if local_federation and not participant_id:
    user_handle, default_device, default_org, agent_handle = _local_attribution_defaults()
    effective_party = party_id or default_device
    effective_org   = org_id  or default_org

    # Layer 1: org
    if effective_org:
        org_id = registry.local_org(effective_org)

    # Layers 2 + 3: device + user (party.org_id link is set inside)
    if user_handle and effective_party:
        participant_id = registry.local_participant(
            handle=user_handle, party_id=effective_party, org_id=org_id,
        )
        if not party_id:
            party_id = effective_party

    # Layer 4: agent (optional — only if HELIX_AGENT is set)
    if agent_handle and participant_id and not agent_id:
        agent_id = registry.local_agent(
            handle=agent_handle, participant_id=participant_id, kind=agent_kind,
        )
```

Then `attribute_gene(org_id=…, party_id=…, participant_id=…, agent_id=…)`
writes the 4-axis row.

Caller can disable the auto-resolution by passing `"local_federation":
false` in the request body (useful when an upstream auth layer provides
IDs explicitly).

## Multi-persona setup (Laude / Taude / Raude / Conductor pattern)

Each persona's IDE plugin / session launches with its own
`HELIX_AGENT` env var. The user (`HELIX_USER` or OS user) and device
(hostname) stay the same — only the agent layer changes per session:

```bash
# Laude's VSCode plugin profile
export HELIX_ORG=swiftwing
export HELIX_AGENT=laude

# Taude's
export HELIX_ORG=swiftwing
export HELIX_AGENT=taude

# Raude's
export HELIX_ORG=swiftwing
export HELIX_AGENT=raude

# Conductor (main session, no specific persona)
export HELIX_ORG=swiftwing
export HELIX_AGENT=conductor
```

All four agents end up as distinct rows in the `agents` table under the
SAME `participant_id` (max), under the SAME `party_id` (gandalf), under
the SAME `org_id` (swiftwing). Each gene each agent creates is
independently queryable:

```sql
-- "Show me everything Laude built this week"
SELECT g.gene_id, ga.authored_at, g.source_id
  FROM gene_attribution ga
  JOIN agents a ON a.agent_id = ga.agent_id
  JOIN genes g ON g.gene_id = ga.gene_id
 WHERE a.handle = 'laude'
   AND ga.authored_at > strftime('%s', 'now', '-7 days');

-- "Show me what the team built today, grouped by agent"
SELECT a.handle, COUNT(*) FROM gene_attribution ga
  JOIN agents a ON a.agent_id = ga.agent_id
 WHERE ga.org_id = 'swiftwing'
   AND ga.authored_at > strftime('%s', 'now', '-1 day')
 GROUP BY a.handle;

-- "Show me org-wide activity per device"
SELECT party_id, COUNT(*) FROM gene_attribution
 WHERE org_id = 'swiftwing'
 GROUP BY party_id;
```

## What this gives you for free, today

1. **Multi-agent attribution** — every gene knows who created it
2. **Cross-machine separation** — `party_id = hostname` distinguishes
   dev box from laptop from server
3. **Cross-account separation** — `getpass.getuser()` distinguishes
   accounts on the same machine
4. **Audit trail** — `gene_attribution.authored_at` is a per-gene
   creation timestamp by definition; Δq queries become possible
5. **Cleanup primitive** — `DELETE FROM gene_attribution WHERE
   participant_id = ?` is the start of a "forget this agent's
   contributions" GDPR-style flow

## What this does NOT give you (yet)

1. **Authentication** — anyone with API access can spoof `HELIX_AGENT`.
   Local trust model assumes the machine itself is trusted (typical for
   solo-dev boxes; not OK for multi-tenant production).
2. **Authorization** — there's no role-based access control over which
   genes a participant can read. `query_genes(party_id=X)` filters, but
   participant-level scoping isn't wired into retrieval yet.
3. **Cross-machine identity** — "max on gandalf" and "max on laptop" are
   two distinct participants today. SSO would unify them.
4. **Audit log of reads** — only writes are attributed. Per-query read
   logging is in `hitl_events` infrastructure but not yet auto-populated.

These gaps are exactly what the enterprise edge layer in
[`ENTERPRISE.md`](ENTERPRISE.md) addresses. The local-first design here
is the runway, not the runway's destination.

## Migration story when SSO arrives

When the OAuth edge layer ships:

1. New auth middleware runs **before** `/ingest` and `/context` endpoints
2. Resolves OAuth token → `(party_id, participant_id, role)`
3. Sets these on the request before forward
4. The ingest path sees `participant_id` already populated, **skips
   the OS fallback**, and writes attribution with the SSO-derived ID
5. Existing genes attributed at the OS tier remain valid — they just
   point to participant_ids that haven't been re-mapped to SSO IDs yet
6. A one-shot migration script can reconcile by joining
   `participants.handle` → SSO email lookup, updating `gene_attribution`
   to use the new participant_id

No data migration required for the genes themselves. Schema invariance
is the gift that keeps giving.

## How this connects to the conductor/librarian pattern

Per the brainstorm in this session: a conductor-orchestrated architecture
where helix is queried as a tool and sub-agents do the heavy lifting
benefits from per-call attribution. When a sub-agent ingests, its
`HELIX_AGENT=researcher-3` env var auto-attributes everything it learned.
The conductor can then ask "show me what researcher-3 found about X"
without any additional plumbing.

In other words: federation isn't just an enterprise compliance feature.
It's the substrate that makes multi-agent introspection cheap.

---

## Implementation footprint (2026-04-12, 4-layer)

- **Schema additions:**
  - `orgs` table (4 columns + 1 index, seeded with 'local' default)
  - `agents` table (8 columns + 3 indexes)
  - `parties.org_id` column (added via ALTER, idempotent)
  - `gene_attribution.org_id` and `agent_id` columns (added via ALTER)
  - 2 new indexes on `gene_attribution` (by org, by agent, both with
    authored_at DESC for time-range filters)
- **registry.py additions:**
  - `Registry.local_org(handle)` — ~20 LOC
  - `Registry.local_agent(handle, participant_id, kind)` — ~40 LOC
  - `local_participant` extended with `org_id` parameter — ~10 LOC delta
  - `attribute_gene` extended with `org_id`, `agent_id` parameters and
    auto-resolution — ~30 LOC delta
- **server.py additions:**
  - `_local_attribution_defaults()` returns 4-tuple — ~30 LOC delta
  - `/ingest` walks the 4-layer find-or-create chain — ~30 LOC delta
  - Accepts `org_id`, `agent_id`, `agent_kind` in request body
- **client opt-out:** `"local_federation": false` in request body
- **dependencies:** none new (`os`, `socket`, `getpass` are stdlib)

Total: ~160 LOC across 3 files + 2 new tables + 2 new columns. Zero
new external dependencies. Zero auth infrastructure. Trust model: the
OS account is who you are; HELIX_ORG / HELIX_AGENT env vars override.

## Migration: pre-4-layer rows

Existing `gene_attribution` rows from before 2026-04-12 had only
`(gene_id, party_id, participant_id, authored_at)`. After the schema
upgrade (`ALTER TABLE … ADD COLUMN`):

- `org_id` is backfilled to the parent party's org (or `'local'` if
  the party has no org link)
- `agent_id` stays NULL — historically pre-agent-layer ingests had no
  way to record this, and NULL is the honest answer

A one-shot backfill script handles the org_id population in batch:

```python
UPDATE parties SET org_id = 'local' WHERE org_id IS NULL;
UPDATE gene_attribution
   SET org_id = COALESCE(
     (SELECT org_id FROM parties WHERE parties.party_id = gene_attribution.party_id),
     'local'
   )
 WHERE org_id IS NULL;
```

Verified on the 2026-04-12 genome: 100% of `gene_attribution.org_id`
populated, with a clean split between historical (`org_id='local',
agent_id=NULL`) and post-upgrade (`org_id='swiftwing',
agent_id=<conductor uuid>`).

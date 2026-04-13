# Federation, Local-First — OS-Level Attribution

> *"The simplest path to federation is to use what the OS already knows."*

This document describes the local federation primitive shipped 2026-04-12.
It captures party / participant attribution for every ingested gene with
zero auth infrastructure — using OS environment as the source of truth.
It's the on-ramp to the full enterprise federation model in
[`ENTERPRISE.md`](ENTERPRISE.md): same schema, different ID resolver.

---

## The split that doesn't change

The `gene_attribution` table is the same at every tier of federation:

```sql
CREATE TABLE gene_attribution (
    gene_id        TEXT NOT NULL,
    party_id       TEXT NOT NULL,    -- the org / tenant / principal
    participant_id TEXT,             -- the user / agent / session
    authored_at    REAL NOT NULL,
    PRIMARY KEY (gene_id)
);
```

What changes across federation tiers is **the source of those IDs**:

| Tier | party_id source | participant_id source |
|---|---|---|
| **Solo / multi-persona (today)** | hostname | `HELIX_AGENT` env → OS username fallback |
| **Small team** | `HELIX_PARTY` env | same as above |
| **Enterprise SSO (future)** | OAuth claim → org_id | OAuth claim → user_id + role |

Because the schema is invariant, every gene attributed at the local tier
remains validly attributed when SSO comes online. The auth edge replaces
the resolver; the rest of the pipeline (`query_genes(party_id=...)`, the
`/context` endpoint, the chromatin tier rules) keeps working unchanged.

## How it resolves IDs today

`helix_context/server.py::_local_attribution_defaults()`:

```python
handle = os.environ.get("HELIX_AGENT") or getpass.getuser()
party  = os.environ.get("HELIX_PARTY") or socket.gethostname()
```

Order of precedence, both axes:
1. **Explicit env var override** (e.g., `HELIX_AGENT=laude`)
2. **OS-derived value** (`getpass.getuser()`, `socket.gethostname()`)
3. **None** — attribution skipped silently (ingest never fails because
   of registry state)

Tokens are normalized: lowercased, whitespace-stripped, length-capped at
64 chars. Conservative because these become primary-key components.

## How it wires through `/ingest`

When the request body provides no `participant_id`, the server resolves
the OS-level default and find-or-creates a participant via the registry's
`local_participant(handle, party_id)` method. This is **trust-on-first-use**
— the first ingest from `max@gandalf` registers a new participant with
that handle; subsequent ingests reuse it without further setup.

```python
# In /ingest endpoint
if not participant_id and local_federation:
    handle, default_party = _local_attribution_defaults()
    if handle:
        effective_party = party_id or default_party
        if effective_party:
            participant_id = registry.local_participant(
                handle=handle, party_id=effective_party,
            )
            if not party_id:
                party_id = effective_party
```

Caller can disable by passing `"local_federation": false` in the request
body (useful when an upstream auth layer is providing IDs).

## Multi-persona setup (Laude / Taude / Raude pattern)

Each persona's IDE plugin / proxy launches with its own `HELIX_AGENT`:

```bash
# Laude's VSCode profile launch
export HELIX_AGENT=laude

# Taude's
export HELIX_AGENT=taude

# Raude's
export HELIX_AGENT=raude
```

Or — if all personas share one helix server — each persona's CLIENT
passes `participant_id="laude"` explicitly in `/ingest` requests. The
OS-level fallback only kicks in when the request is silent on attribution.

Either way, every gene now carries a party + participant trail. Future
queries like "show me everything Laude has touched in helix-context this
week" become trivial:

```sql
SELECT gene_id, authored_at FROM gene_attribution
 WHERE participant_id IN (
   SELECT participant_id FROM participants
    WHERE handle = 'laude' AND party_id = 'gandalf'
 )
   AND authored_at > strftime('%s', 'now', '-7 days');
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

## Implementation footprint (2026-04-12)

- **schema:** unchanged (existing `parties`, `participants`,
  `gene_attribution` from `d2e0219`)
- **registry.py:** added `Registry.local_participant(handle, party_id)` —
  ~40 LOC, find-or-create idempotent
- **server.py:** added `_local_attribution_defaults()` helper + ingest
  fallback path — ~30 LOC
- **client opt-out:** `"local_federation": false` in request body
- **dependencies:** none new (`os`, `socket`, `getpass` are stdlib)

Total: ~70 LOC. Zero new tables. Zero new external dependencies.
Zero auth infrastructure. Trust model: the OS account is who you are.

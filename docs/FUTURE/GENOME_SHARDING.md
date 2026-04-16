# Genome Sharding — Main DB + Category Shards

**Status:** Design direction, 2026-04-16. No code yet. Proposed during a
session where Opus 4.7 (on web chat) pushed back on the growing
`genome.db` with "this is becoming a godfile" and suggested splitting
it. This doc captures the storage-layer decision. See
[WALKER_PATTERNS.md](WALKER_PATTERNS.md) for the execution-layer
companion and [PUSH_PULL_CONTEXT.md](PUSH_PULL_CONTEXT.md) for the API
contract that bridges them.

---

## The problem in one sentence

One `genome.db` holds every gene from every source — the operator's
notes, the operator's code, the agent's session logs, third-party
reference material, multiple tenants' content — and the only thing
telling them apart is a `source_id` prefix convention.

That's fine at 10k genes. It starts to hurt at 100k. At 1M it's a
godfile: no audit boundary, no compliance teeth, no cheap way to
forget one tenant's data, no way to load only the shards a given
query actually needs.

## The proposed split

```
genomes/
  main.db                        — fingerprint meta + routing table
  participant/
    max.db                       — operator's own genes
    anon_9f3a.db                 — other humans' genes (if multi-user)
  agent/
    laude_2026-04-16.db          — this session's genes
    taude_2026-04-14.db          — sister agent's genes
    sub_explore_xyz.db           — sub-agent tool calls
  reference/
    hades_codex.db               — third-party reference material
    factorio_wiki.db             — domain corpora
  org/
    swiftwing21.db               — org-level shared genes
    anthropic_public.db          — account-scoped
  cold/
    archive_2025q4.db            — aged-out, read-only
```

Four canonical category folders (`participant`, `agent`, `reference`,
`org`) mirror the 4-layer identity model already shipped in the genome
(org / party / participant / agent). `cold/` is the aged-out tier,
separate because its read-only nature means it can live on different
storage.

## What lives in main.db

Only what's needed to *route* a query:

- **Fingerprint index** — gene_id, shard_path, source_id, domains,
  entities, tier_contributions — the ~150 tok/gene that's always
  eagerly pushed to consumers anyway (see `PUSH_PULL_CONTEXT.md`).
- **Shard routing table** — which category shards exist, where they
  live on disk, their health, their size, last-modified.
- **Cross-shard indexes** — FTS5 over source_id + entities across all
  shards (small: text only, not content).
- **Identity registry** — the 4-layer org/party/participant/agent
  table.

What does **not** live in main.db:

- `complement`, `content`, `key_values`, `codons` — all the tier-1/2/3
  bulk data. Those stay in the category shards.
- Raw blob fields of any kind.

Main.db stays small enough to hold in memory. It's the index, not the
library.

## Why this matches the brain analogy

User's framing: "what if we do the master meta db and split out the
db to many smaller files — like the brain..."

It's a good frame. The cortex isn't one undifferentiated mass —
visual cortex, motor cortex, hippocampus, amygdala are physically
separate, specialised, and the thalamus routes between them. Our
current `genome.db` is brain-as-blob. Sharded genome is brain-as-regions.

## Concrete wins

**Noise isolation** — a query about "BookKeeper RBAC" should never
have to scan Factorio codex fingerprints. Shard routing skips entire
`.db` files. Current genome scans all of them.

**Compliance teeth** — "delete everything for participant X" becomes
`rm genomes/participant/x.db`. One filesystem call, no partial-delete
risk, auditable. GDPR Article 17 becomes trivial instead of fraught.

**Auditability** — "what did agent Taude ingest last week?" is a
single shard read. Regulators and security reviewers get clean
boundaries instead of `SELECT ... WHERE source_id LIKE 'taude_%'`.

**Backup granularity** — hot shards (current session) back up every
20 min; cold shards back up monthly. Currently we back up 1 GB of
genome.db every 20 min to catch a 2 MB delta.

**Write contention** — each shard has its own WAL. The operator typing
notes doesn't wait on an agent's ingest stream. Currently one writer
blocks all.

**Cold tier migration** — rotate aged shards to `cold/` without
touching the main query path. Retrieval scans main.db fingerprints;
only pulls cold-shard content when actually requested.

## Open questions

### Cross-shard FTS

Do we maintain one FTS5 index in main.db (all source_ids + entities
across shards), or one FTS5 per shard plus a shard-router? Main.db
central index is simpler to query but grows with total gene count.
Per-shard is cleaner but requires merging N result sets per query.

Likely answer: **both**. Main.db gets a lightweight "which shards
have any matches at all" bloom-filter-style prefilter, and per-shard
FTS does the real scoring after shard selection.

### Migration path

Existing `genome.db` files are in the wild on operator machines. A
migration script is straightforward (read each gene, emit to the
correct shard based on source_id pattern), but the *default* behaviour
on first-boot-post-upgrade needs to be decided:

1. Migrate in-place, delete old `genome.db` on success. Risk: partial
   migration on crash.
2. Copy-on-migration — new `genomes/` directory coexists with old
   `genome.db`, old becomes read-only, confidence built over N days,
   old deleted by operator action.

Option 2 is the adult answer. Don't delete the user's data on an
upgrade path.

### Shard granularity knobs

How many agent shards? One per session gets noisy (10k+ shards for a
heavy user). One per week per agent is cleaner. One per agent lifetime
collapses history but loses temporal slicing.

Likely answer: **weekly shards for agents, lifetime shards for
participants, immutable shards for reference**. Cold tier migrates
weekly-agent shards older than N weeks.

### Query planner

With shards, `query_genes(domains=[...])` needs to decide which
shards to scan. Naive: scan all. Better: main.db fingerprint-level
prefilter → shard list → per-shard content fetch. This is where
shard routing lives.

The planner is probably a `ShardRouter` class in `genome.py` that:
1. Reads main.db fingerprint index.
2. Scores candidate shards against the query.
3. Returns an ordered list of shards to actually open.

### Replication / sync across devices

Open question. A sharded layout is actually *easier* to sync
selectively (push only participant/ and org/ to the cloud, keep
agent/ local) but the protocol is unspecified. Punt to a follow-up
doc once this lands.

## What this explicitly doesn't change

- **The genome API.** `query_genes()`, `get_gene()`, `ingest()` all
  stay the same signature. The sharding is below the waterline.
- **Tier math.** Fingerprint scoring, tier fusion, harmonic_links —
  none of that changes.
- **Ingest pipeline.** Compression, entity extraction, codon
  generation — unchanged. Just routes to a shard at the final
  `INSERT` instead of the global `genes` table.

## Related

- [WALKER_PATTERNS.md](WALKER_PATTERNS.md) — the walker navigates
  across these shards. Sharding gives the walker a natural
  parallelism boundary (one librarian per shard).
- [PUSH_PULL_CONTEXT.md](PUSH_PULL_CONTEXT.md) — main.db fingerprints
  are the "push" payload; category shards hold the "pull" content.
  Sharding is how the push/pull split is *actually stored on disk*.
- [../DESIGN_TARGET.md §5](../DESIGN_TARGET.md) — the 4-layer
  identity model whose boundaries the shard folders mirror.

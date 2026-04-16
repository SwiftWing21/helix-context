# Layered Fingerprints — Hierarchical Gene Composition

**Status:** Design direction, 2026-04-16. Implementation plan at
[../specs/2026-04-16-layered-fingerprints-plan.md](../specs/2026-04-16-layered-fingerprints-plan.md).
Fell out of a session examining genome bloat and noticing that files
chunked into many genes produce many independent fingerprints — the
genome knows the chunks are related via `source_id`, but the
retrieval layer treats them as independent candidates.

---

## The idea in one paragraph

Chunks of the same file should have a **parent gene** that aggregates
their fingerprints. When multiple chunks of one file hit a query, the
parent's fingerprint accumulates a co-activation signal proportional
to how many of its children surfaced — so broadly-relevant files
bubble up to file granularity while narrowly-relevant files stay at
chunk granularity. The consumer sees this as a single ranked list
where both parent and child fingerprints can compete, and can pull
either "the whole file reassembled" or "just this chunk" based on
what it actually needs.

## Why this falls out naturally

Today's chunk-level retrieval has three weak spots:

1. **Co-activation is invisible.** Three chunks of File A hitting top-k
   should tell the consumer "File A is *really* relevant." Instead
   they appear as three independent genes of equal weight.
2. **Reassembly is manual.** If the consumer wants the whole file, it
   has to issue N pulls and stitch them. Or read the source from disk.
3. **Top-k gets consumed by single files.** A highly chunked file can
   monopolise top-k with many near-duplicate fingerprints, crowding
   out unrelated-but-relevant chunks from other files.

A parent fingerprint solves all three: co-activation becomes a scored
signal, reassembly becomes one pull call, and "parent surfaces
instead of N chunks" gives top-k breathing room.

## The data model

Helix already has the scaffolding:

- `gene_relations(gene_id_a, gene_id_b, relation, confidence, updated_at)`
  — add a new `relation` code for `CHUNK_OF`
- `promoter.sequence_index` — already preserved per chunk at ingest,
  defines reassembly order
- `is_fragment` — already flags chunks (is_fragment=1)
- `source_id` — shared across all chunks of a file, the natural key
  for parent-child association

**Parent gene shape:**

- `gene_id` = `sha256(source_id + "::parent")[:16]` — stable under
  re-ingest of the same file
- `content` = the first ~1 KB of the file (a "table of contents" or
  header) — gives the parent its own readable surface
- `complement` = aggregated/re-compressed summary of all child
  complements (ribosome can produce this from children at parent
  creation time)
- `codons` = JSON list of child `gene_id`s in sequence order —
  this is the reassembly key
- `source_id` = file path (same as children)
- `is_fragment` = 0 (parent is not a fragment)
- `promoter.sequence_index` = -1 (sentinel for "file-level, not
  chunk-level")
- `key_values` = `{"chunk_count": N, "total_size_bytes": B}`

**Edge shape:**

- For each child chunk: `(child_gene_id, parent_gene_id, CHUNK_OF, 1.0, now)`

## Query-time aggregation

After per-chunk retrieval finishes ranking:

1. For each parent gene in the genome that has chunks hit in top-k,
   compute an aggregated fingerprint:
   - `tier_contributions[parent]` = sum (or avg, or harmonic mean) of
     child `tier_contributions` weighted by hit count
   - `entities[parent]` = union of child entities
   - `domains[parent]` = union of child domains
   - `fused_score[parent]` = some aggregation of child scores + a
     co-activation bonus proportional to hit count
2. Merge parent fingerprints into the top-k candidate list
3. Re-rank
4. Apply a **deduplication rule**: if parent appears in top-k, drop
   all its chunk children below rank K (they're redundant signal).
   Keep children if parent is below top-k (chunks ranked higher than
   their parent is a legitimate "this specific part matters more than
   the whole" signal).

The co-activation bonus is the core mechanic. Three chunks hitting
means the file is relevant in three places; that's stronger evidence
than one chunk hitting three times as hard.

## Reassembly via pull

`pull(parent_gene_id, tier="T3", task="reassemble")`:

1. Load parent gene
2. Read `codons` list — the ordered child `gene_id`s
3. `SELECT content FROM genes WHERE gene_id IN (...) ORDER BY sequence_index`
4. Concatenate with separator (likely `\n\n` for text, or chunk-boundary markers)
5. Return `{answer: <full file content>, source: parent_gene_id, reassembled_from: [child_ids]}`

For libraries: one directive, one return packet, file-level content
in the consumer's context without N round-trips.

## Higher layers are natural (future direction)

Same mechanic, recursive:

- **Directory-level** parent: aggregates the file-level parents in a
  given directory
- **Project-level** parent: aggregates directory-level parents
- **Codebase-level** parent: one fingerprint for the whole repo

Each layer adds co-activation signal at coarser granularity. The
consumer can ask "which project is this about?" before drilling down
to "which file?" before drilling to "which chunk?" — or skip
straight to chunk if the query is narrow.

V1 is just file-level (chunks → file parent). Directory+ are
straightforward extensions but not needed to validate the pattern.

## Open questions

### When is the parent created?

Two options:
1. **At ingest time** — every time a file chunks into N≥2 strands,
   create the parent in the same transaction. Simple, deterministic,
   but pays the cost upfront whether or not the parent ever surfaces.
2. **Lazy at query time** — create parents on-demand when N≥2 chunks
   of the same file appear in the same top-k. Cheaper for cold files;
   requires cache invalidation on re-ingest.

V1 answer: **ingest time**. The cost is one extra row + N edges per
multi-chunk file, which is trivial. Lazy is harder to get right and
the savings don't matter at current genome sizes.

### What about single-chunk files?

Files that fit in one chunk (≤ 4000 chars) don't need a parent —
the single chunk *is* the file. Creating a trivial parent for them
would just double the row count with no benefit. Rule: **parent
only created when a file chunks into N≥2 strands**.

### Re-ingest behaviour

If a file is re-ingested with different content:
1. Old children's `gene_id`s differ (content-hashed), so new children
   get new `gene_id`s.
2. Old parent's `gene_id` is `sha256(source_id + "::parent")[:16]`
   — *stable* across re-ingests of the same path.
3. On re-ingest: UPSERT the parent (replaces `codons` list with new
   child IDs, updates `content`/`complement`), and leave the old
   children as orphans (source-less in the sense that their parent
   no longer references them).
4. Garbage collection: a periodic sweep can delete children whose
   parent's `codons` doesn't include them. Or just accept the bloat;
   it's small.

### Does this interact with `is_fragment=1` in tie-break?

Walking tie-break (`TIE_BREAK_WALKING.md`) uses graph signals to
order ties. `CHUNK_OF` edges become another graph signal — ties
between chunks of the same parent should probably prefer the lower
`sequence_index` (earlier in file), which is already available via
`promoter.sequence_index`.

### Parent in fingerprint push payload

Does the consumer see the parent in the `/context` fingerprint push?
Yes, when it surfaces in top-k. Its fingerprint shape is the same as
a leaf chunk's (`tier_contributions`, `fused_score`, `source_id`,
`domains`, `entities`) plus a `chunk_count` field signalling "I
aggregate N chunks; you can pull me for the whole file." That's the
signal the librarian (walker) uses to decide parent-fetch vs
chunk-fetch.

## What this explicitly doesn't change

- **The genome API.** `query_genes()`, `upsert_gene()`, `get_gene()`
  all keep their signatures. Parent genes are just genes with a
  specific shape.
- **The chunking logic.** `CodonChunker.chunk()` keeps producing
  strands the same way. Parent creation happens *around* the loop,
  not inside the chunker.
- **The fingerprint schema.** Parent fingerprints have the same
  fields as leaf fingerprints; only the `chunk_count` field is new.

## Related

- [WALKER_PATTERNS.md](WALKER_PATTERNS.md) — parent-pull is a walker
  operation; the librarian dispatch pattern handles `task:reassemble`
  by reading the parent's `codons` and fanning out to child reads.
- [PUSH_PULL_CONTEXT.md](PUSH_PULL_CONTEXT.md) — parent fingerprints
  are part of the push payload; parent reassembly is part of the pull
  contract. Layered fingerprints is the *composition rule* for the
  push side.
- [GENOME_SHARDING.md](GENOME_SHARDING.md) — sharding is orthogonal;
  parents and their children live in the same shard (same
  `source_id` scope).
- [TIE_BREAK_WALKING.md](TIE_BREAK_WALKING.md) — `CHUNK_OF` edges are
  another graph signal the walking tie-break can consult for
  same-parent chunks.

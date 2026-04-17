# Genome Sharding — Phase 2 Implementation Plan

**Status:** Spec, 2026-04-17.
**Scope:** Full 4-folder split per [GENOME_SHARDING.md](../FUTURE/GENOME_SHARDING.md).
**Mode:** Copy-extract (source intact until cutover). Phase-1 fresh rebuild
stays as the source-of-truth throughout; shards are built alongside it
and validated before any row is deleted from main.

## Goals

1. Prove **selective extraction** — a subset of genes can leave the
   parent `.db` and still function as a valid standalone genome.
2. Prove **federated query** — a `ShardRouter` can read N `.db` files
   and return merged results equivalent to the pre-shard query.
3. Land the routing table + shard schema that future compliance work
   (GDPR delete, per-tenant audit) plugs into.
4. Do not break the live genome. At every checkpoint the operator can
   revert to `genomes/main/genome.db` and the system works.

## Non-goals (this phase)

- **Deleting extracted rows from main.** Punt to a follow-up after N
  days of confidence. This plan only copies.
- **Cross-shard FTS bloom prefilter.** Ship the router first; optimise
  the prefilter once we have real shard-count data.
- **Weekly agent-shard rotation.** Ship lifetime-per-agent first; add
  rotation when shard sizes justify it.
- **Network sync.** Out of scope; local filesystem only.

## Layout at end of phase

```
genomes/
  main/
    genome.db              — original fresh rebuild (unchanged)
  main.db                  — NEW: routing + fingerprint index
  participant/
    max.db                 — operator's notes, prompts, journal sources
  agent/
    laude.db               — laude-authored genes (lifetime)
    taude.db               — taude-authored genes
  reference/
    third_party.db         — external corpora, design assets, wikis
  org/
    swiftwing21.db         — org-level shared genes
```

Note: `genomes/main/genome.db` (the rebuild) and `genomes/main.db`
(the new routing index) are different files. The rebuild stays as a
safety net until cutover task 9.

## Task breakdown

### Task 1 — `main.db` schema + migration

Create `genomes/main.db` with:
- `shards` — shard_name, category, path, created_at, gene_count,
  byte_size, health
- `fingerprint_index` — gene_id, shard_name, source_id, domains,
  entities, key_values (the ~150-tok push payload; no content, no
  complement, no codons)
- `identity` — org / party / participant / agent rows (copy from
  existing registry)

Deliverable: `helix_context/shard_schema.py` + init script.

### Task 2 — `ShardRouter` class

In `helix_context/shard_router.py`:
- `__init__(main_path)` opens main.db, reads `shards` table
- `_open_shard(name)` lazy-opens a `Genome` against the shard path
- `route(query)` → ordered list of shard names (naive V1: all shards
  with matching domains/entities in fingerprint_index; later: score)
- `query_genes(...)` → fans out to selected shards, merges with
  existing fusion math, returns same shape as `Genome.query_genes`

Integration: `HelixContextManager` gets a `use_shards=True` flag that
swaps its `self.genome` for a `ShardRouter`.

Deliverable: router class + feature-flag integration.

### Task 3 — Extraction script (reference first)

`scripts/extract_shard.py --category reference --pattern ...`:
- Read genes from source `.db` matching pattern rules
- INSERT OR REPLACE into destination shard `.db`
- Write parent genes' CHUNK_OF edges to the same shard
- Record shard in main.db `shards` table
- Copy fingerprint rows into main.db `fingerprint_index`

Reference patterns: source_id starts with known external paths
(BookKeeper knowledge, CosmicTasha docs, etc). Start here because
it's the least entangled with agent/participant identity.

Deliverable: working script + `reference/third_party.db` populated.

### Task 4 — Extract participant / agent / org

Same script, different `--category` + pattern rules:
- **participant**: source_id matches operator note paths + party_id
- **agent**: rows with `key_values` containing `authored_by=<agent>`
- **org**: everything else in the current org scope

Deliverable: all 4 category shards populated from `genomes/main/genome.db`.

### Task 5 — Round-trip validation

Script `scripts/validate_shard_roundtrip.py`:
- Run 5 canonical benchmark queries against `genomes/main/genome.db`
  (pre-shard baseline)
- Run same 5 queries against `ShardRouter(main.db)` (sharded)
- Assert: same gene_ids returned, same top-10 order ±2 positions,
  same answer correctness when piped through the needle bench

Deliverable: pass-fail report. Block cutover if fail.

### Task 6 — Ingest-time routing

`HelixContextManager.ingest(..., shard_hint=)`:
- Hint can come from metadata (`authored_by`, `party_id`) or default
  rules (agent > participant > org > reference)
- `ShardRouter.upsert_gene(gene, shard)` writes to the right `.db`
- Parent gene + CHUNK_OF edges go to the same shard as children

Deliverable: new genes land in correct shard; main/genome.db no
longer grows after this task.

### Task 7 — Cross-shard query integration

Replace `Genome.query_genes` with `ShardRouter.query_genes` behind
the `HELIX_USE_SHARDS=1` flag. Keep flag OFF by default until Task 8.

Deliverable: `HELIX_USE_SHARDS=1 python -m helix_context.mcp_server`
returns results identical to flag OFF on the benchmark.

### Task 8 — Cutover + helix.toml switch

- Update `helix.toml` to `path = "F:/Projects/helix-context/genomes/main.db"`
  (the routing db, not the fresh-rebuild)
- Default `HELIX_USE_SHARDS=1`
- Restart supervisor
- Run benchmark one more time

Deliverable: live helix serves from sharded layout. Old
`genomes/main/genome.db` kept read-only as safety net.

### Task 9 — Archive the old genome

After N days of sharded-helix stability (operator judgment):
- Move `genomes/main/genome.db` to `E:\Helix-backup\pre-shard-2026-04-17.db`
- Drop `shards.source=main/genome.db` from main.db

Deliverable: one unambiguous serving layout, old file archived.

## Risk + rollback

Every task is reversible:
- Tasks 1-4: `rm -rf genomes/main.db genomes/{participant,agent,reference,org}/`
  and main/genome.db is untouched.
- Task 7: toggle `HELIX_USE_SHARDS=0`.
- Task 8: revert `helix.toml` path.
- Task 9 is the only irreversible step; gated on operator confirmation.

## Open decisions (blocking Task 2)

- **Router fusion strategy**: each shard returns top-K; merge by
  normalised score across shards, or re-rank all candidates centrally?
  V1 proposal: each shard returns top-K raw tier_contributions, router
  re-fuses centrally using existing fusion math. Preserves "one fusion,
  one score" semantics.
- **FTS5 placement**: per-shard FTS (keep existing) with central bloom
  in main.db, or central FTS in main.db spanning all content?
  V1 proposal: per-shard FTS unchanged; main.db has only the
  fingerprint index (domains/entities/source_id). Router queries
  fingerprint_index first to pick shards, then runs FTS inside each
  selected shard.

## Estimated scope

5-10 sessions. Task 1-2 is one session (schema + router skeleton).
Task 3 is one session (first extraction + validation). Tasks 4-5 is
one session. Task 6-7 is one session. Task 8-9 is one session. Plus
buffer for integration bugs discovered during round-trip validation.

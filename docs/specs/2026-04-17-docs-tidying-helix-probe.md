# Docs/ Tidy-Up + Helix Path-Sensitivity Probe — 2026-04-17

**Type:** Organizational cleanup + opportunistic data point.
**Scope:** Move ~18 files under `docs/` into topical subdirectories; measure
how retrieval scoring responds to the path change.

## Motivation

Two goals, one commit:

1. **Tidy.** `docs/` has 23 markdown files at root; a reader on GitHub
   has to eyeball the whole list to find what they want. Subdirs by
   topic (architecture, research, papers, ops, benchmarks, positioning)
   make the tree scannable.
2. **Data point.** Moving files changes `source_id` from e.g.
   `docs/MISSION.md` → `docs/positioning/MISSION.md`. That's a
   clean natural experiment on how much helix retrieval depends on
   directory structure vs filename vs content. Relevant to the
   planned project-level layer in layered-fingerprints.

Not a benchmark. One-shot probe, ~15 queries, pre/post diff.

## Proposed layout

```
docs/
  MISSION.md                # entry-point reading — keep at root
  ROSETTA.md
  DESIGN_TARGET.md
  architecture/
    DIMENSIONS.md
    FEDERATION_LOCAL.md
    KNOWLEDGE_GRAPH.md
    PIPELINE_LANES.md
    SESSION_REGISTRY.md
    LAUNCHER.md
    OBSERVABILITY.md
  research/
    MUSIC_OF_RETRIEVAL.md
    RESEARCH.md
    RESEARCH_VELOCITY.md
  papers/
    AGENTOME_PART_II_DRAFT.md
    PAPER_FIGURE_SPECS.md
    PAPER_THREE_CONSTRAINTS.md
    SIKE_POST_DRAFT.md
  positioning/
    ECONOMICS.md
    ENTERPRISE.md
  ops/
    RESTART_PROTOCOL.md
    SKILLS_BUNDLE.md
  benchmarks/
    BENCHMARKS.md
    BENCHMARK_RATIONALE.md
  FUTURE/                   # unchanged
  specs/                    # unchanged
  plans/                    # unchanged (1 file; collapse with specs/ later?)
  collab/                   # unchanged
```

## Probe design

**Target queries (N≈20):** one per moved file, phrased naturally
(not as exact-filename match) so we stress content tiers too.

| File | Query |
|---|---|
| RESTART_PROTOCOL.md | "restart protocol" |
| MUSIC_OF_RETRIEVAL.md | "music of retrieval" |
| DIMENSIONS.md | "9 dimensions helix" |
| FEDERATION_LOCAL.md | "4-layer federation identity" |
| KNOWLEDGE_GRAPH.md | "knowledge graph entity links" |
| PIPELINE_LANES.md | "pipeline lanes" |
| SESSION_REGISTRY.md | "session registry org party participant" |
| LAUNCHER.md | "launcher tray setup" |
| OBSERVABILITY.md | "observability prometheus otel" |
| RESEARCH_VELOCITY.md | "research velocity" |
| AGENTOME_PART_II_DRAFT.md | "agentome part two" |
| PAPER_FIGURE_SPECS.md | "paper figure specs" |
| PAPER_THREE_CONSTRAINTS.md | "three constraints paper" |
| SIKE_POST_DRAFT.md | "sike post draft" |
| ECONOMICS.md | "economics helix" |
| ENTERPRISE.md | "enterprise compliance" |
| SKILLS_BUNDLE.md | "skills bundle" |
| BENCHMARKS.md | "benchmarks" |
| BENCHMARK_RATIONALE.md | "benchmark rationale" |

**Per query we log:**
- top-10 candidates with `gene_id`, `score`, `source_id`
- whether the target doc's gene appears in top-10
- rank + score of target gene
- which genes appear in both pre and post vs only one

**Data captured:**
- `benchmarks/docs_tidy_pre.json` — pre-move probe output
- `benchmarks/docs_tidy_post.json` — post-move probe output
- `benchmarks/docs_tidy_diff.md` — human-readable summary

## Steps

1. Run pre-move probe → save JSON.
2. `git mv` files one commit (no content edits).
3. Trigger re-ingest via `/ingest` or wait for mem_sync watcher to catch.
4. Verify new `source_id` values land (spot-check 2-3 genes).
5. Run post-move probe → save JSON.
6. Write diff summary.

## What we expect to see

Hypotheses (to be confirmed/falsified by the probe):

- **H1 (most likely):** Scores barely move. Content didn't change; FTS5,
  SPLADE, ΣĒMA are content-driven. Filename-anchor (Tier 0.5) keys off
  filename stem which is unchanged. Path changes only affect
  `path_key_index` (Tier 0), which boosts by directory keywords — may
  nudge a file up when queried for its new parent directory name
  (e.g. "benchmarks" query now hits `docs/benchmarks/BENCHMARKS.md`
  with a small boost).
- **H2:** Some genes may be re-ingested as new (different gene_id)
  depending on how `gene_id` is derived — if it's `sha256(source_id + ...)`,
  every moved file gets a new gene_id. If content-derived, gene_id is
  stable. Worth noting which helix does.
- **H3:** If H2 holds (gene_id = f(source_id)), post-move probe may show
  old gene IDs *still present* as heterochromatin (orphan, no
  source-file backing), creating duplicate retrievals. Clean-up
  question: does mem_sync/ingest detect the old path's disappearance
  and tombstone?

## Success criteria (as a data point)

- Probe runs cleanly pre + post.
- Diff is reproducible and written to `docs_tidy_diff.md`.
- Whatever result we get, we learn something about path-sensitivity
  of current retrieval. No "hit or miss" — just a reading.

## Out of scope

- Not a benchmark run. Not comparing to a baseline genome.
- No content edits during move. Pure filesystem rename.
- Not touching FUTURE/, specs/, plans/, collab/.
- Not attempting to fix any retrieval issue discovered. If the diff
  surfaces something worth fixing, that's a follow-up doc, not this one.

---

## Results (2026-04-17)

Full diff at [../../benchmarks/docs_tidy_diff.md](../../benchmarks/docs_tidy_diff.md).

**Headline:** of 19 probe queries:
- 12 stable rank, 2 improved, 2 regressed, 3 miss-both
- **Average score gain on stable queries: +10.8 pts.** Bigger than expected.
- 0 files became unreachable. All previously-retrievable docs still retrievable.
- Genome grew +37 genes (each moved file added a small number of fresh chunks).

**H1 (scores barely move):** *Falsified.* Scores jumped by +8-12 points on
most queries. Likely drivers:
- Fresh epigenetic state (access_count=0, freshness=1.0) on re-ingested genes
- Subdir name (`architecture`, `research`, etc.) added a new path token
  that matches some queries
- Filename-anchor firing at fresh weight

**H2 (gene_id stability):** *Confirmed as content-derived.* Same content →
same gene_id. Re-ingest with correct metadata updates the source_id
in-place. **Important footgun:** metadata key is `path`, not `source_id`
(see [helix_context/context_manager.py:397](../../helix_context/context_manager.py#L397)).
My first re-ingest run used the wrong key and silently wiped every
source_id to `None` — gene was still retrievable (content) but had no
path attribution. Script fixed, but this is a rough edge worth
documenting in the `/ingest` API surface.

**H3 (orphan retention):** *Confirmed, and bigger than expected.* 13 of
the 19 probes returned at least one top-10 candidate with the **old**
(pre-move) source_id still pointing at `docs/X.md`. There's no
filesystem-watch tombstone path for docs/. mem_sync handles this for its
watched dir but docs/ isn't watched. Follow-up: either add docs/ to
mem_sync watch config, or ship a one-shot tombstone-by-missing-file
pass after bulk moves.

**Regressions worth noting:**
- `PIPELINE_LANES` rank 3→18. Pre-move top was `Education/docs/swimlanes/README.md`
  (unrelated). Post-move, ranking shifted to favor `docs/benchmarks/BENCHMARKS.md`
  on "pipeline lanes" — suggests the query is generic enough that
  content-match drifts easily when scores rebalance.
- `FEDERATION_LOCAL` rank 1→5 even though score went up +2.76. Other docs
  (including the 3 FEDERATION_LOCAL orphans) also boosted; the target
  kept rising but was outpaced.

**Takeaways for helix proper:**
1. **Document `metadata["path"]` clearly** in /ingest endpoint docstring.
   `metadata["source_id"]` silently no-ops and overwrites.
2. **Tombstone-on-missing-file** should be a first-class API, not a
   mem_sync-only feature. Bulk renames are common.
3. **Fresh-ingest score bias is real.** A re-ingest without content
   change gives genes a significant epigenetic reset. Worth measuring
   half-life — how long until the fresh bump decays to baseline.
4. **Path tokens contribute to retrieval.** Subdir name enters
   `path_key_index` and gives a measurable boost. Supports the planned
   project-level layer in layered-fingerprints.

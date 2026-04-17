# Hierarchical Math — One Signal, One Level

**Status:** Design hypothesis, 2026-04-17. Not a commitment. Captured
from a conversation with operator after a SNOW oracle-only ablation
sweep (3×N=65, commit `9e6b718` baseline) showed several retrieval
dimensions at or below the noise floor at gene level.

---

## The hypothesis in one sentence

Each retrieval math in Helix has a natural level of operation where
its signal-to-noise is highest; running them all at the gene level
(chunks of ~4KB) puts several of them below the detection floor and
causes destructive interference between the ones that do fire.

## The current state

Every math in [DIMENSIONS.md](../DIMENSIONS.md) operates at the gene
level:

| Math | Current operating level | Signal profile |
|---|---|---|
| D1 FTS5/SPLADE/ΣĒMA | per-gene tokens + embedding | Strong (dominant signal) |
| D6 cymatics | per-gene spectrum (256 bins) | Below N=65 noise floor |
| D8 harmonic co-activation | per-gene edge graph (227k edges) | Faintly destructive (2.1σ) |
| D9 TCM drift | per-gene trajectory | Tiebreaker at Step 3.25 only |

Gene level ≈ 4KB chunks. A chunk is small enough that:
- Its cymatic spectrum is short and high-variance (one paragraph can
  dominate the 256-bin profile).
- Its co-activation edges are dense but mostly coincidental (any two
  chunks in the same file co-fire when that file is retrieved).
- Its temporal drift is ghostly (TCM updates once per session, not
  once per gene-retrieval).

The result: D6, D8, D9 all operate in regimes where their inherent
signal is drowned by same-level FTS/SEMA/SPLADE noise.

## Evidence from tonight's ablation

SNOW oracle-only × 3 repeats × 4 configs at commit `9e6b718`
(`benchmarks/snow/results/ablation_sweep_2026-04-17_r3.json`):

```
variant               avg_tier (mean±sd)   miss_rate (mean±sd)   >2σ?
baseline              0.970 ± 0.037        0.287 ± 0.007         —
d6_cymatics_off       1.007 ± 0.009        0.282 ± 0.007         no
d8_harmonic_off       1.020 ± 0.016        0.272 ± 0.007         yes (miss)
d5_cold_tier_on       0.977 ± 0.037        0.282 ± 0.015         no
```

- **D6**: both deltas inside 1σ. No measurable retrieval-layer
  contribution at gene level.
- **D8 harmonic**: Δmiss at 2.1σ — turning harmonic OFF *improves*
  recall by 1.5pp. Faintly destructive.
- **D5 cold tier**: zero effect in the current hot-heavy regime
  (expected — cold only fires when hot returns 0).

**Honest read:** at gene level, D6 doesn't contribute and D8 actively
hurts. If the hypothesis here is right, that's a wrong-level symptom,
not a wrong-math symptom.

## Proposed level assignment

| Level | Unit | Proposed math | Why this fits |
|---|---|---|---|
| Folder / source cluster | ~50 folders, ~500 co-activation edges | D8 harmonics | Sparse graph, stable over time, folder-folder co-firing is structural (topic coherence) not coincidental. |
| File | ~1000 files, aggregate spectrum | D6 cymatics | Files span a coherent topic → stable resonance signature across chunks. Per-file spectrum aggregated from child chunks has much better signal than per-chunk. |
| Content (chunk/passage) | ~7800 genes, FTS/SEMA/SPLADE | D1 semantic | Already dominant at this level. Adding D9 TCM at sub-gene passage granularity gives fine-grained trajectory for the actual decision boundary. |

**What each math becomes:**

- **Folder-harmonic**: `source_folder × source_folder → co-activation
  edge` graph. Built once at ingest, updated on query hit. The shard
  router (phase-2 sharding) is the natural consumer — folder-harmonic
  picks which shards to scan.
- **File-cymatic**: `file → 256-bin aggregate spectrum` computed as
  weighted sum of child gene spectra. File-level `flux_score_dispatch`
  picks which files within a shard to drill into.
- **Content-TCM**: drift vector updated per retrieval hit (not per
  session). Sub-gene passage representations if passage extraction
  improves. Gene-level semantic retrieval unchanged.

The retrieval pipeline becomes coarse-to-fine routing:

```
Query
  ├─ Folder-harmonic → pick N folders (D8 at its natural level)
  ├─ File-cymatic    → pick M files within those folders (D6 at its natural level)
  └─ Content-retrieval → pick K genes within those files (D1+D9 at their natural level)
```

## Connection to existing work

**Phase-2 sharding** ([GENOME_SHARDING.md](GENOME_SHARDING.md)) is
*approximately* the right decomposition for the top level. Shards
(`reference/`, `agent/`, `participant/`, `org/`) are coarser than
folders, but the routing principle is the same — pick-from-meta,
then-scan-inside. If folder-harmonic lives in `main.db`, the shard
router and this hypothesis collapse into one design:

```
main.db
  ├─ folder_harmonics table    ← D8 at folder level
  ├─ fingerprint_index         ← what's already planned
  └─ file_spectra table        ← D6 at file level
shards/<category>/*.db
  └─ gene-level retrieval      ← D1+D9 at content level
```

**Layered fingerprints** ([LAYERED_FINGERPRINTS.md](LAYERED_FINGERPRINTS.md))
already does *half* of the file-cymatic step. The parent gene
aggregates child `tier_contributions` into a co-activation boost. What
it doesn't do yet is aggregate child cymatic spectra into a parent
file spectrum. The extension is small: `parent.spectrum =
weighted_sum(child.spectrum)`, normalised.

**Walker patterns** ([WALKER_PATTERNS.md](WALKER_PATTERNS.md)) — the
librarian dispatch model naturally maps onto the three levels. The
top-level librarian routes on folder-harmonic, dispatches sub-agents
per file, each sub-agent retrieves content. Three-tier walk.

## What would have to be true

- Folder-level co-activation has meaningfully higher signal-to-noise
  than gene-level. **Testable**: build `folder_harmonics` alongside
  the current graph, compare edge-weight distributions. If folder
  edges are uniformly weak or uniformly strong, the hypothesis falls
  apart.
- File-level cymatic spectra are more discriminative than the mean
  of their child chunks. **Testable**: compute both, do pairwise
  cosine against a held-out query set, compare precision@k.
- Sub-gene TCM is actually trackable given the chunk sizes we have.
  **Unclear** — chunks are small enough that sub-gene passage
  extraction may be pointless. TCM might need to stay at session
  level but update *per-hit* (it currently updates per-`_express`
  call).

## Cost to test

- **Folder-harmonic prototype**: ~1 session. Group existing
  `harmonic_links` by source_id prefix (folder), produce a collapsed
  graph, swap in as D8's retrieval feed, re-ablate.
- **File-cymatic prototype**: ~1 session. Extend layered-fingerprints
  parent gene to carry aggregated spectrum. Route file-level
  flux_score_dispatch before gene-level.
- **Per-hit TCM**: ~0.5 session. Move `_tcm_session.update()` call
  from `_express()` end to `query_genes()` per-hit callback.
- **Re-ablation**: N=200 SNOW oracle-only × 4-6 configs, ~30 min
  compute.

Total: ~3 sessions + 30min compute to tell us whether the hypothesis
holds. Worth it before we either kill D6/D8 (decision gate per
DIMENSIONS.md §Decision Gates) or invest more in them as-is.

## What this explicitly does NOT claim

- It does not claim D6/D8 will definitely work at the proposed
  levels. They might still be zero-contribution. The claim is only
  that gene-level measurements can't tell us.
- It does not claim gene-level retrieval is *wrong* — FTS/SEMA/SPLADE
  clearly work at that level. The claim is specifically about
  **resonance maths and graph maths** needing a coarser unit of
  analysis.
- It does not claim this maps cleanly onto every future math. D4
  access rate and D7 attribution are gene-level concerns by
  definition and don't have a natural higher level.

## Related

- [GENOME_SHARDING.md](GENOME_SHARDING.md) — top-level category
  routing; this doc extends that to 3 levels instead of 2.
- [LAYERED_FINGERPRINTS.md](LAYERED_FINGERPRINTS.md) — file-level
  parent genes; the substrate for file-cymatics.
- [WALKER_PATTERNS.md](WALKER_PATTERNS.md) — coarse-to-fine
  dispatch; this doc gives the walker its three levels.
- `benchmarks/snow/results/ablation_sweep_2026-04-17_r3.json` —
  the data this hypothesis is trying to explain.
- `benchmarks/results/waude_diagnostic_2026-04-17.md` — the
  D-category diagnostic that set up tonight's ablation.

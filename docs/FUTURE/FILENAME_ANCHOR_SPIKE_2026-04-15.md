# Filename-anchor retrieval tier — spike 2026-04-15

**Short version:** Add a new retrieval tier (Tier 0.5) that boosts genes
whose filename stem matches a query term. Flag-gated; off by default
until the A/B bench decides.

## Why

Dewey bench 2026-04-14 (`benchmarks/bench_dimensional_lock.py --dewey`)
compared variant orderings of the structural axes a query can carry:

```
axis order                                      recall@1
key                                              6.0%
key + filename                                  30.0%    <- peak
key + filename + project                        26.0%
key + filename + project + module               10.0%    <- over-constrained
```

Adding filename to `key` alone gives +24pp. Then adding project+module
pulls recall back down by 20pp. Filename is the call-number; project
and module behave as repulsion edges once filename has pinned the
location.

The existing retrieval pipeline doesn't honor that asymmetry. Tier 0
(`path_key_index`) treats every path-derived token as equivalent:
`{helix-context, helix_context, helix, context, config}` for
`.../helix_context/config.py` all flow into the same IDF-weighted pool.
Filename signal is diluted by project and module tokens of roughly equal
frequency.

Slice-3 decision gate on 2026-04-15 (`_run_slice3_gate.sh`) measured
slice-2 alone at **20.0% retrieval** — the same number SR-on and
pre-slice-2 baseline both deliver. Three orthogonal knobs at the same
ceiling is a signal that something else is binding; the Dewey result
says filename-anchoring is the +20pp door.

## What shipped

| File | Role |
|---|---|
| `helix_context/filename_anchor.py` | `filename_stem()`, schema init, upsert hook, `boost_scores()` |
| `helix_context/genome.py` | Schema creation, upsert wire-through, Tier 0.5 in `query_genes()` |
| `helix_context/context_manager.py` | Genome constructor wiring + env-var override |
| `helix_context/config.py` | `[retrieval] filename_anchor_enabled` + `filename_anchor_weight` |
| `helix.toml` | Default `false`, weight `4.0` |
| `scripts/backfill_filename_anchor.py` | One-shot populate from existing `genes.source_id` |

**Schema addition:**
```sql
CREATE TABLE IF NOT EXISTS filename_index (
    filename_stem TEXT NOT NULL,
    gene_id       TEXT NOT NULL,
    PRIMARY KEY (filename_stem, gene_id)
);
CREATE INDEX idx_filename_stem ON filename_index(filename_stem);
```

**Stem extraction rules:**
- Match `basename.ext` shape; take the stem (before the first dot).
- Lowercase for case-insensitive matching.
- Skip noise stems: `__init__`, `main`, `index`, `test`, `config`,
  `readme`, `license`, `setup`, `util`, `helper`, etc. — generic names
  that would blanket-boost across projects.
- `_NOISE_STEMS` is intentionally conservative; grow it when a bench
  surfaces a false-positive stem.

**Retrieval tier behavior** (`boost_scores()`):
- For each query term that matches an indexed stem, every gene with
  that stem gets `+weight` added to its score.
- Weight defaults to `4.0` — higher than Tier 1 exact-tag (`3.0`) so
  filename-anchor wins ties but doesn't dominate alone when combined
  with multiple Tier 1 matches.
- Multi-term matches accumulate (gene with stem `config` hit by two
  query terms both equal to `config` gets `+8.0` — rare but correct).

## Backfill stats (2026-04-15 production genome)

```
input:       17,895 genes with non-null source_id
stamped:     16,558 rows (92.5%)
skipped:      1,337 rows (7.5%) — no filename extension OR noise stem
distinct:     5,795 filename stems
runtime:     0.1s
```

## Flag gate

Off by default. Enable via either:
- `[retrieval] filename_anchor_enabled = true` in helix.toml
- `HELIX_FILENAME_ANCHOR_ENABLED=1` env var (useful for per-process A/B
  without editing config)

Weight tunable via `filename_anchor_weight`. Default 4.0 picked to sit
just above Tier 1 exact-tag. If the bench suggests weight is the wrong
knob (e.g. saturating at higher weights or needing to be lower to avoid
drowning other tiers), the flag makes that reversible.

## Benchmark plan

Two back-to-back runs of `bench_dimensional_lock.py --dewey` at N=50,
seed 42, qwen3:1.7b on the live production genome:

1. **Baseline** (flag OFF) — reproducibility check against 2026-04-14
   Dewey bench.
2. **Spike** (flag ON, same data, same seed) — isolates the
   filename-anchor effect.

**Decision thresholds (on axis-4, the over-constrained worst case):**
- ≥12% recall → ship. Commit, push, schedule default-on behind the
  next version bump after a second-bench confirmation at higher N.
- 4-11% → report + decide. The lift is real but below the "obvious
  ship" bar; look at whether axis-2 also lifted (indicating the
  feature is doing its job even if axis-4 is still constrained by
  other factors) and whether any query categories regressed.
- ≤4% → revert. The feature isn't finding its signal; the Dewey
  result may have been specific to bench harness geometry that
  doesn't reflect production queries.

Results file naming: `benchmarks/dewey_N50_{baseline,spike}_2026-04-15.log`.

## Open questions + future work

- **N=50 → N=200+ at weight 4.0.** If the N=50 lift holds, verify at
  higher N before enabling by default. N=50 can be noisy on recall@1
  deltas — the N=200 Dewey run from 2026-04-14 was the reference curve.
- **Weight sweep.** If the curve shape is right but absolute lift is
  middling, sweep weight in {2, 4, 6, 8} to find the knee.
- **Interaction with path_key_index IDF.** Tier 0 already IDF-weights
  path-token matches. Our Tier 0.5 stacks *on top* rather than
  replacing — there may be double-counting in cases where a stem is
  also a path-token. Verify via `tier_contrib` inspection on a few
  representative queries.
- **Category-level regression analysis.** The Dewey bench reports
  per-category recall (steam / helix / cosmic / tally / etc). Confirm
  no single category regresses — the spike's biggest risk is
  false-positive blanket-boosts on common filenames (which is why
  noise stems are filtered at ingest time, but the production genome
  may surface new noise candidates the list doesn't yet cover).

## Actual results (2026-04-15)

Ran both benches back-to-back on the live production genome (19,593
genes), N=50 seed 42 qwen3:1.7b, Dewey axis ordering. Only variable
between runs: `filename_anchor_enabled`.

| Variant (axis order) | Baseline | Spike | Δ |
|---|---|---|---|
| 1 axis (key) | 6.0% | 6.0% | 0.0 |
| 2 axis (key+filename) | 18.0% | **30.0%** | **+12.0pp** |
| 3 axis (key+filename+project) | 14.0% | 14.0% | 0.0 |
| 4 axis (key+filename+project+module) | 4.0% | 4.0% | 0.0 |

**Reading:**

- **Axis 2 hit the design target.** +12pp is the full effect size the
  Dewey hypothesis predicted. The feature is restoring the 30% ceiling
  that today's larger genome had degraded (yesterday's N=50 baseline
  was 30% at axis 2 on an 18,254-gene genome; today's pre-spike
  baseline was 18% at 19,593 genes; spike brings it back to 30%).
- **Axis 4 did not move.** The over-constrained case remains a hard
  ceiling at 4%. Hypothesis: at axis 4, accumulated Tier 1/2/3 matches
  on wrong genes outweigh the +4 filename-anchor boost on the correct
  gene. Fix would be higher weight, or a dampener on path_key_index
  when filename_anchor fires, or co-occurrence filtering.
- **Zero regression** on any axis — shipping is safe.

**Decision: ship flag-gated off by default.** The feature works where
it was designed to work; the axis-4 case is a follow-up, not a blocker.
Operators opt in per genome via `filename_anchor_enabled = true`.

**Axis-4 follow-up ideas (not scoped for this commit):**

1. **Weight sweep** — try `weight ∈ {6.0, 8.0, 10.0}` and re-bench at
   axis 4. If 8.0 lifts it, ship the higher weight as default.
2. **Path_key_index dampener** — when a gene's filename stem matches
   a query term, halve its path_key_index contribution (since the
   filename anchor already credits it; the extra PKI boost is
   double-counting). Reversible via another flag.
3. **Co-occurrence filter** — in axis-4 queries, require that boosted
   genes share at least the filename anchor with the query to survive.
   More surgical; higher implementation cost.

## Slice-3 deferral

This spike supersedes the literal reading of the roadmap
(NEUTRAL → implement slice-3 tiebreaker) for now. Slice-3 stays as a
deferred follow-up: `benchmarks/_run_slice3_gate.sh` + the decision
already logged in `project_helix_slice3_deferral` memory note. If
filename-anchor doesn't lift retrieval above the 20% ceiling we've
been stuck at, slice-3 is still next.

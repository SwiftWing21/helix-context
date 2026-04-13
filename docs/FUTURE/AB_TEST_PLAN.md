# A/B Test Plan — LLM Ingest (A) vs Math-Only Ingest (B)

Status: **pre-A, A in flight.** Gemini Flash re-ingest running in
powershell as of 2026-04-12. This doc locks in predictions BEFORE results
come back, so we can honestly check whether the predictions aligned with
reality instead of backfitting a story.

---

## Hypotheses

**A (LLM ingest — current pipeline)**: LLM transcribes each gene into
rich metadata (`complement`, `codons`, `promoter.*`, `key_values`). The
downstream model reads expressed_context with curated labels.

**B (math-only ingest — future, per LANGUAGE_AT_THE_EDGES.md)**: Zero
LLM calls at ingest. CpuTagger (regex + spaCy) produces tags. Lightning
strike (OptiX or CPU Monte Carlo) records innate co-activation. Downstream
LLM still reads expressed_context but with math-derived metadata only.

**Null hypothesis**: B scores within 10% of A on retrieval quality. If
true, math-only ingest wins on cost, speed, and determinism. If false,
LLM ingest is carrying more weight than we think.

## Bench methodology

Identical for both configs:

1. Fresh genome snapshot at end of each ingest (`genome-bench-A.db`,
   `genome-bench-B.db`).
2. Both benches target the SAME genome for queries (live server points at
   one; swap DB files between runs).
3. Fixed RNG seed (42) and query model (qwen3:8b via Ollama).
4. **Layer 1 — SIKE (curated N=10)**: `python benchmarks/bench_needle.py`
5. **Layer 2 — KV-harvest (N=50 v2)**: `python benchmarks/bench_needle_1000.py`
   with `N=50 SEED=42 HELIX_MODEL=qwen3:8b`.
6. **Compression**: `python benchmarks/bench_compression.py --queries 16`
7. Capture: retrieval rate, answer accuracy, extraction_miss count,
   retrieval_miss count, errors, latency p50/p95.
8. Also record: genes in DB, ingest wall-clock time, ingest API cost.

## Predictions — lock them in BEFORE results

### Pre-A prediction (made 2026-04-12, before Gemini finishes)

| Metric | A (LLM ingest) | B (math-only) | Gap |
|---|---|---|---|
| **SIKE N=10 retrieval** | 10/10 | 9/10 | -1 |
| **SIKE N=10 answer (qwen3:8b)** | 9/10 | 7/10 | -2 |
| **KV-harvest N=50 retrieval** | 40% | 35% | -5pp |
| **KV-harvest N=50 answer** | 35% | 30% | -5pp |
| **Ingest wall-clock (522 core files)** | 3-4 hours | ~15 minutes | 12x faster |
| **Ingest cost** | ~$0.30 (Flash) | $0 | -$0.30 |
| **Compression ratio (genes → tokens)** | 9x | 8x | -11% |

**Rationale (Pre-A):**
- SIKE is curated — queries use clean terms that match tags directly.
  Both configs should nail it. LLM-tagged gives +1 retrieval from
  better entity coverage.
- KV-harvest queries are template-generated ("value of X?") which don't
  benefit much from LLM tagging since the signal is the key itself
  (regex finds it).
- Answer accuracy on A leads because `complement` gives the model a
  second readable artifact alongside raw content. B has raw content
  only (and extractive summaries).
- Math-only ingest is throughput-bound by GPU embedding, ~200 genes/sec.
- LLM ingest at Flash is 30 genes/sec (2 API calls per gene).
- Compression comparable because Kompress handles both the same way.

### Post-A measurement (2026-04-12, genome-bench-A.db, 17,685 genes)

| Metric | Pre-A predicted | A measured (raw) | A measured (score-floor fix) | Pre-A miss |
|---|---|---|---|---|
| SIKE N=10 retrieval | 10/10 | 8/10 | **10/10** | 0 (after fix) |
| SIKE N=10 answer (qwen3:8b) | 9/10 | 5/10 | **7/10** | -2 |
| KV-harvest N=50 retrieval | 40% | 8% | **12%** | **-28pp** |
| KV-harvest N=50 answer | 35% | 8% | **10%** | **-25pp** |
| Ingest wall-clock | 3-4 hours | ~3 hours | — | ~0 |
| Ingest cost | ~$0.30 | ~$0.30 (gemma4 via LiteLLM) | — | ~0 |

**Score-floor fix (commit pending):** raw A had a bug where the budget
tier used ratio-only (top_score/mean) which is scale-invariant. A query
with top=1.2, mean=0.4 (ratio=3.0) got the same "tight mode, send top 3"
treatment as top=8.5, mean=2.8 — even though the former means "retrieval
is weak, widen the net" and the latter means "we found it." Adding an
absolute score floor (TIGHT requires top_score ≥ 5.0, FOCUSED ≥ 2.5)
keeps weak-signal queries in BROAD mode with 12 candidates. This
recovered SIKE 10/10 and bumped KV-harvest 8→12%.

**What Post-A tells us:**

The KV-harvest collapse (40% predicted → 8% actual) is the headline
finding. SIKE retrieval held at 80% — curated tag-driven queries still
work. But template-generated "value of X?" queries across a 17K-gene
genome fail 46/50 times on retrieval.

This is the **Struggle 1 dilution effect** from the original plan, now
empirically measured. When the genome doubled from 7K → 17K with broad
ingest (Steam, GGUF, BeamNG, education_public), low-density genes
crowded the retrieval pool. The template-query signal (key name, value
pattern) is weaker than the curated-query signal (domain vocabulary,
named entities), so it drowns in noise first.

**Category breakdown — raw A vs score-floor fix:**

| Category | Raw A retr | Fixed A retr | Delta |
|---|---|---|---|
| education_public | 13.3% | **26.7%** | +13.4pp |
| scorerift | 0% | **50%** | +50pp (1/2) |
| steam | 7.1% | 7.1% | flat |
| tally | 25% | 0% | -25pp (regressed) |
| **helix** | **0%** | **0%** | zero (7/7 still missed) |
| **cosmic** | **0%** | **0%** | zero (6/6 still missed) |

The score-floor fix is a structural improvement but only solved the
easy half. Helix and cosmic stayed at 0% — meaning the target gene is
ranked so far outside top-12 that widening the net doesn't help. This
is a retrieval-*ordering* problem, not a budget-tier problem.

After the fix, curated SIKE hits 100% and template KV-harvest still
fails on helix/cosmic. That's diagnostic:
- Curated queries use domain vocabulary that tag-based retrieval
  scores correctly.
- Template queries ("What is the value of X?") give retrieval no
  domain signal, and the 11K Steam/education genes overwhelm the
  ~500 helix genes by volume.

**Revised B prediction (math-only ingest):**

Given A's retrieval floor is this low, B's prediction depends on whether
the problem is:
- **(a) LLM tagging quality** — B would be worse (no `complement`, no
  `intent`, weaker promoter tags). Gap: B retrieval 3-5% vs A's 8%.
- **(b) Genome dilution from broad ingest** — B similar (same content,
  same noise floor). Gap: B retrieval ~7-10% vs A's 8%.
- **(c) Retrieval pipeline calibration** — B could be equal or better
  if the lightning-strike co-activation bonus catches genes that
  tag-based retrieval misses. Gap: B retrieval could exceed A.

The honest answer: **B probably can't be worse than 8% retrieval**, which
means math-only ingest is a viable next experiment regardless of gap.
The LLM ingest isn't providing enough lift to matter on KV-harvest — so
if we can get equivalent quality for free, the math-only path wins.

| Metric | A measured | Revised B prediction | Forecast gap |
|---|---|---|---|
| SIKE N=10 retrieval | 8/10 | 7-9/10 | ±1 |
| SIKE N=10 answer (qwen3:8b) | 5/10 | 3-6/10 | -1 to -2 |
| KV-harvest N=50 retrieval | 8% | 6-12% | ±4pp |
| KV-harvest N=50 answer | 8% | 5-10% | ±3pp |

### Post-B prediction (fill in after both A and B are benched)

Date: `_____________`

| Metric | A measured | B measured | Gap actual |
|---|---|---|---|
| SIKE N=10 retrieval | `____` | `____` | `____` |
| SIKE N=10 answer (qwen3:8b) | `____` | `____` | `____` |
| KV-harvest N=50 retrieval | `____` | `____` | `____` |
| KV-harvest N=50 answer | `____` | `____` | `____` |
| Ingest wall-clock | `____` | `____` | `____` |
| Ingest cost | `____` | `____` | `____` |

### Retrospective — did pre-A predictions align?

| Metric | Pre-A predicted | Actual | Verdict |
|---|---|---|---|
| SIKE retrieval | 10/10 | 8/10 | ❌ off by 20% |
| SIKE answer | 9/10 | 5/10 | ❌ off by 44% |
| KV retrieval | 40% | 8% | ❌ off by 5x |
| KV answer | 35% | 8% | ❌ off by 4x |
| Ingest speed | 3-4 hrs | ~3 hrs | ✅ correct |
| Ingest cost | ~$0.30 | ~$0.30 | ✅ correct |

**What I was most wrong about:**

KV-harvest retrieval. Predicted 40%, got 8%. A 5x miss.

The mental model assumed that since KV-harvest queries the same genome
that produced the needles, retrieval should be near-trivial — the exact
key-value pair exists in some gene's content. The model underweighted
three things:

1. **Genome dilution is superlinear.** 17K genes is >2x the retrieval
   difficulty of 7K genes because the noise floor doesn't just double —
   the probability of a confusing near-miss gene winning the top-k
   grows with genome size AND gene-class diversity. Steam manifests,
   BeamNG configs, and GGUF metadata all competed for the same top-k
   slots that helix/scorerift/cosmic genes needed to occupy.

2. **Template queries are fragile.** "What is the value of X in project
   Y?" doesn't hit tag-based retrieval well. Tags were optimized for
   natural language questions, not parametric value lookup. The
   retrieval pipeline under-serves synthetic queries.

3. **Ingestion-time LLM tagging didn't save us.** Even with full
   LLM-generated `complement`, `intent`, `codons`, and `key_values`, the
   retrieval pipeline failed to recall the needle genes. This is
   evidence for LANGUAGE_AT_THE_EDGES.md's thesis: LLM tagging at
   ingest isn't load-bearing — the tags exist but aren't converting to
   retrieval wins.

**What I was most right about:**

Ingest speed and cost. The operational envelope was predictable because
those are throughput-bound (API rate × gene count). That part of the
mental model is calibrated.

Also SIKE retrieval pattern was roughly right: curated queries worked
(80%), confirming that tag+content retrieval functions on well-formed
queries. The failure isn't retrieval-as-a-concept — it's retrieval
across a diluted genome with template-style queries.

**Calibration signal for next time:**

When predicting retrieval rates, halve the number if the genome exceeds
10K genes and the test queries are template-generated rather than
natural-language. The 40% prediction should have been 15-20% given
those two factors, and that would have been within the 10% error
target.

## Decision tree

After both runs complete:

```
B retrieval / A retrieval = ratio

ratio ≥ 0.95  → Math-only is the winner. Adopt B as default.
                Use lazy LLM annotation for hot genes only (per
                LANGUAGE_AT_THE_EDGES.md §"The lazy annotation principle").

ratio 0.85-0.95 → Math-only is viable. Adopt B as the default for
                ingest, keep A as opt-in flag for projects where
                interpretability > speed.

ratio 0.70-0.85 → Math-only has a real gap. Investigate which
                dimension carries the gap (likely promoter tags on
                domain-specific vocab). Improve CpuTagger with
                project EntityRuler patterns, re-test B.

ratio < 0.70  → LLM ingest is load-bearing. Keep current architecture.
                Lightning strike becomes additive (ingest-time
                co-activation ONLY, LLM still does tagging). Update
                LANGUAGE_AT_THE_EDGES.md with this finding.
```

## Risks & caveats

1. **qwen3:8b may not be the right answer model for the A/B.** It's
   what the current bench uses, but answer accuracy may be bounded by
   extraction capability, not retrieval quality. Consider re-running
   with gemma4:e4b as a secondary model.

2. **Fresh needles from fresh genome.** The KV-harvest bench harvests
   from the genome it queries. If A and B have different gene_ids
   (different content from different pipelines), needle sets will
   differ. Mitigation: for A/B comparison, harvest needles from A's
   genome and query against both A and B. Needles that don't exist in
   B's genome count as retrieval misses (which is the right behavior).

3. **TCM warmup asymmetry.** TCM dimension needs session state to
   contribute. Both A and B benches start cold. Not a problem unless
   the benches run wildly different lengths.

4. **Lightning-strike requires implementation work.** The B ingest
   can't run until `helix_context/lightning.py` exists. Pre-A
   predictions assume it will be implemented by the time B is benched.

## Cost estimate

- A ingest: ~$0.30 Gemini Flash, ~3-4 hours wall-clock, already running
- B ingest: $0, ~15 minutes wall-clock, TBD when implemented
- Bench runs: ~30 minutes total per config (SIKE ~2 min + N=50 ~12 min
  + compression ~2 min, both configs)

Total experiment cost: ~$0.30 + ~1 hour of benching + implementation
time for lightning-strike (~1 day of work per LANGUAGE_AT_THE_EDGES.md
§"Implementation path").

## Follow-up work triggered by A's results

**Headroom `compress_batch` PR** ([issue
#151](https://github.com/chopratejas/headroom/issues/151)) is staged
behind this bench. Once A's SIKE + KV-harvest numbers are in hand:

1. If compression remains the dominant latency contributor (expected
   per issue #151 — ~3,500ms for 12-gene expression), ping Tejas on
   the issue with fresh benchmark data confirming the bottleneck.
2. Once he signals PR welcome, implement `KompressCompressor.compress_batch()`
   (~80 LOC, additive, zero-impact on non-helix users).
3. Update `helix_context/headroom_bridge.py` with
   `compress_text_batch()` helper + feature-detect for older headroom
   versions.
4. Modify `context_manager.py` Step 4 (gene expression) to use batched
   path when available.

Expected impact on helix after the PR lands upstream:
- 12-gene expression: 3,500ms → ~200ms (17x faster)
- 3-gene expression: 800ms → ~100ms (8x faster)
- Proxy p95 latency drops 2-4s per query

This work depends on A's numbers giving us empirical leverage ("we
confirmed on N=50 fresh-genome bench that compression was the
bottleneck"). Without that data, the PR is speculative; with it, the
PR is grounded.

## Companion docs

- [`MISSION.md`](../MISSION.md) — the why
- [`FUTURE/LANGUAGE_AT_THE_EDGES.md`](LANGUAGE_AT_THE_EDGES.md) — the
  design direction this experiment will validate or refute
- [`BENCHMARKS.md`](../BENCHMARKS.md) — methodology for the bench harness

---

## Appendix — prediction philosophy

The point of writing predictions down BEFORE results is not to be right
— it's to check which parts of the mental model are calibrated. A
prediction off by 10% on retrieval but right on ingest speed tells you
the model of scoring is good but the model of ingest capacity was
wrong. Backfitting a story after results come in throws away that
signal.

> *"The difference between research and fuckin around is whether you
>  wrote down the notes."*
> — `docs/RESEARCH_VELOCITY.md`

Predictions are the notes.

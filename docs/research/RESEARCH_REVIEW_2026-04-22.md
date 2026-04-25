# Research review — helix-context, 2026-04-22

**Goal:** suggestions to raise accuracy/determinism over speed in helix-context.
Pareto wins (accuracy AND speed) preferred.

**Method:** 4 parallel research agents + 1 synthesis lead, each given a narrow
remit over the live codebase and recent benchmark evidence (2026-04-22
`helix_rag_composition_2026-04-22.json`, where pure BM25 went **8/8 content_full
at 151 ms** while helix_rag hit **4/8 at 1793 ms**).

- **Agent 1 —** determinism audit (stochastic / drift / cache leaks)
- **Agent 2 —** deterministic-LM training state and accuracy bottleneck
- **Agent 3 —** matrix / tensor architecture (9×9 vs 6×6×3 vs 3×3×3)
- **Agent 4 —** Pareto-frontier architectural moves

---

## TL;DR

| Research question | Answer |
|---|---|
| Do the deterministic LMs need more training? | **Partial yes, necessary not sufficient.** Rerank artifact is stale (trained on ~3.5K-gene era, genome is now 17K) and gated **off**; splice trained on only 179 labels; NLI never trained. Training alone will NOT close the BM25 gap — the dominant failure is structural. |
| Does 6×6×3 raise accuracy + speed? | **No on both.** Natural grouping is 4+2+3 (lexical+semantic+graph), not 6+6. 6×6×3 = 108 features vs current 46 (+33% memory, +50% compute) with no data-backed axis split. Keep the 46-vector. |
| Variable matrix view (per-query-type conditioning)? | **No, under current labels.** [PHASE0_V2_FOLLOWUPS_2026-04-21.md:96](../collab/comms/PHASE0_V2_FOLLOWUPS_2026-04-21.md#L96) showed per-segment max|r| sits below its own permutation p95 in every segment — conditioning has no recoverable signal at current label noise. Revisit only after Option B (per-(q,g) CWoLa refactor) + ≥3 weeks re-accumulation. |

**The real lever is architectural, not training or tensor shape.** At 17K
genes, the test-file corpus has ~10× the keyword density of config files,
so tag/fts5 tiers drown the right answers in noise. BM25's IDF trivially
finds the needles Helix misses. The fix is to flip the pipeline: BM25
generates a tight shortlist, Helix's tiers re-rank inside it.

---

## Combined-regression diagnosis (Agent 4, confirmed by Agent 1)

Three stacked failures explain why Helix is both slower and less accurate than
BM25 on the current genome:

1. **Population dilution at 17K genes.** For "what ports do helix and headroom
   listen on," Helix delivered 12 test/doc/script files and zero config files.
   Test files mention `port` and `helix` hundreds of times in fixtures and
   assertions; `helix.toml` mentions them once each. The tag-exact,
   tag-prefix, FTS5, and source-authority tiers all accumulate on test files
   until no config file can compete. BM25 wins because `11437` has trivially
   low document frequency — pure IDF dominates. (Agent 1, [genome.py:1730-1812](../../helix_context/genome.py#L1730))

2. **PKI tier is structurally broken on this genome.** `packet_notes` says
   `"source_index unavailable; using gene-local metadata only"` on 6/8
   needles, so the coordinate-confidence path collapses and the 0.30 floor
   sends everything to `refresh_targets` instead of `verified`. Additionally,
   PKI uses exact equality: query term `ports` (plural) never matches
   `kv_key=port` (singular from CpuTagger). This is the highest-discriminating
   tier failing silently. (Agent 1, [genome.py:1730](../../helix_context/genome.py#L1730);
   Agent 4, [context_packet.py](../../helix_context/context_packet.py))

3. **helix_only delivers exactly 4555 chars on every needle** regardless of
   what it retrieved. This is the ribosome splice budget hitting before any
   useful content escapes the pipeline — a **content-delivery bug**, not a
   retrieval miss. `helix_rag` reads the pointed files from disk directly and
   gets 50–60k chars per needle, recovering 4/8 content_full. The "accuracy
   gap" between helix_only and pure_rag is partially an assembly-ceiling
   artifact, not a ranking failure. (Agent 4, bench JSON evidence)

The latency regression (1.8s vs 151ms) is SEMA Mode B doing a 17K×20 numpy
cosine scan on every request, plus HTTP + 9-tier accumulation — at 8K genes
this was half the cost and half the test-file dilution.

---

## Determinism audit summary (Agent 1)

`query_genes()` output is deterministic given fixed `(query, genome_snapshot)`
**only if** 8 conditions all hold:

1. `_intent_cache` pre-populated (first query vs. warmed cache differs)
2. `_sema_cache` fully built with no pending upserts
3. `_corpus_size` TTL has not expired mid-run (otherwise IDF denominator drifts)
4. `_tcm_session.context_vector` is zero (no prior-turn drift)
5. `_pending` buffer is empty
6. SPLADE disabled OR torch in `use_deterministic_algorithms` mode
7. No concurrent `/context` writes racing against `last_query_scores`
8. Walking tie-break off, OR all scores strictly unique

**`clean=True` "bench mode" does NOT isolate** against these leaks:

- `_pending` buffer never cleared → background replications from prior rows
  leak into next row's express path
- `session_delivery_log` persists → same-session queries elide previously-
  delivered genes unless `ignore_delivered=True` passed explicitly
- `_sema_cache` not invalidated → genes added via `_pending` commits are
  invisible to SEMA retrieval until next rebuild
- `_corpus_size` memoization not reset → IDF frozen from row 1's corpus
- SPLADE model process-global state never reset

**For clean determinism under bench:** add `reset_session_state` extensions
to clear `_pending`, invalidate `_sema_cache`, and trigger `_corpus_size`
refresh. ~30 LOC, pure additive.

---

## Training state (Agent 2)

| Component | Type | Last trained | Functional? |
|---|---|---|---|
| SEMA codec | MiniLM + 20-anchor fixed projection | **Untrained by design** | Yes but sparse (cosines 0.10-0.20 on paraphrases) |
| Cymatics (W1) | Closed-form math | Parameter-free | Yes |
| **DeBERTa rerank** | DeBERTa-v3-small + regression | 2026-04-08, **3.5K-gene era, 1.6K pairs** | **OFF** (`rerank_enabled=false`) |
| **DeBERTa splice** | DeBERTa-v3-small + BCE | 2026-04-08, **179 labels** | Functional but critically under-trained |
| **DeBERTa NLI** | DeBERTa-v3-small, 7-class | **Never trained** — dir doesn't exist | Not trained |
| SPLADE | third-party pretrained | N/A | Yes, never fine-tuned |
| PLR head | GBC over 11+ features | 2026-04-22, AUC 0.631 | Query-quality head only (Option A) |

**Accuracy-lift ranking if trained on current in-domain data:**

1. **SEMA projection** — 20-dim bottleneck with cosines in 0.10-0.20 range
   cannot discriminate semantic paraphrases. Largest single win available.
   Blocker: needs per-(q,g) labels (Option B CWoLa refactor) for supervised
   fine-tune, or a contrastive weak-signal approach using current bucket-A/B
   labels (marginal — 161 A / 2048 B imbalance + single-party data).
2. **DeBERTa rerank** — re-export teacher labels on 17K-gene genome and
   flip `rerank_enabled=true`. Cheap (~30 min per 500 queries).
3. **DeBERTa splice** — 179 labels → several thousand. Directly shapes
   what survives into compressed context.

**cwola_log data is aggregate-per-query, not per-(q,g)** — unusable for
learning-to-rank without Option B refactor (~200 LOC + 3 weeks
re-accumulation). This is the same train/spec mismatch flagged in
[STATISTICAL_FUSION.md:263](../FUTURE/STATISTICAL_FUSION.md#L263).

---

## Matrix architecture (Agent 3)

**Natural tier grouping is 4+2+3, not 6+6+3:**

- **Lexical (4):** `fts5, lex_anchor, tag_exact, tag_prefix` — share BM25-like
  surface statistics; load together on PC0 of the pooled correlation
  ([LOCKSTEP_MATRIX_TEST_v2.md:42](../collab/comms/LOCKSTEP_MATRIX_TEST_v2.md#L42)).
- **Semantic (2):** `splade, sema_boost` — both cosine-derived.
- **Graph/trajectory (3):** `pki, harmonic, sr` — placed at non-gene levels
  in [HIERARCHICAL_MATH.md:66](../FUTURE/HIERARCHICAL_MATH.md#L66); PC2
  loadings confirm co-firing.

**Parameter accounting:**

- **Today (shipped):** 9 raws + 36 pairwise window correlations + 1 cos(q,c)
  = **46 features**. GBC cost ~1.4K split params + O(window·9²) correlation
  update per query.
- **6×6×3:** 108 features (54 if symmetric). +33% memory, +50% compute, no
  data-backed 6-axis split.
- **3×3×3:** 27 features. Cheaper but discards within-family pairs that
  Sprint 3's top-10 importances depend on (`fts5__tag_exact`,
  `tag_exact__harmonic`, etc.).

**Verdict: keep 46.** The signal lives in cross-family pairs that block
projections would pool away.

**Variable matrix view (third tensor dim keyed on query-type):** PHASE0_V2's
permutation T3 killed it — template obs=0.050 < null p95=0.073, mixed
obs=0.359 < 0.413, natural obs=0.165 < 0.213. Segmentation dilutes N without
unmasking signal. Precondition to revisit: Option B per-(q,g) labels + 3
weeks re-accumulation.

---

## Pareto proposals — ranked by expected lift (Agent 4)

All five are additive; they compose.

| # | Proposal | Acc Δ | Latency Δ | LOC | Risk |
|---|---|---|---|---|---|
| 1 | **BM25 as Tier 0 shortlist, Helix tiers rerank top-50** | +3-4/8 | **-1600 ms** (8-10× speedup) | ~80 | low |
| 2 | **Flip `filename_anchor_enabled=true`** (already shipped, off) | +2-3/8 | +5 ms | 0 | near-zero |
| 3 | **Fix `helix_only` 4555-char content ceiling** | +3/8 on helix_only | -200-400 ms | ~40 | low |
| 4 | **Fix `source_index unavailable` path resolution** | +2/8 | ~0 | ~20 | low |
| 5 | **Run `scripts/backfill_parent_genes.py` + `HELIX_LAYERED_FINGERPRINTS=1`** | +1-2/8 | +10 ms | 0 | low |

**Dark-shipped features worth flipping on immediately:**

- `filename_anchor_enabled` — Dewey spike showed +12pp at axis 2
- `HELIX_LAYERED_FINGERPRINTS=1` (after parent backfill)
- `sr_enabled` — Sprint 3 t07 showed `sema_boost__sr` at rank 7 importance
- `plr.enabled=true` — head works, calibration pass needed (prob_B skews
  ~0.9 due to 93% training-B base rate; tune `high_risk_threshold` or
  rebalance training sample)

**Single highest-leverage move: Proposal 1 + Proposal 2 together.** BM25
shortlist restructures the pipeline so Helix's precision tiers operate on a
50-candidate set where the right files are already present, not a 17K-gene
ocean where they're diluted. Filename anchor is free and addresses the exact
failure mode we observed (`helix.toml` losing to `test_stress.py` despite
being named `helix`).

---

## Determinism improvements (cross-cutting)

If "deterministic over speed" is the priority:

1. **Close the bench-mode leaks** (~30 LOC): clear `_pending`, invalidate
   `_sema_cache`, reset `_corpus_size` TTL when `clean=True` fires. Safe,
   additive, should have been there from day one.
2. **Make SPLADE gateable to CPU / deterministic** — torch
   `use_deterministic_algorithms(True)` + CPU fallback when `HELIX_DETERMINISTIC=1`.
3. **Warm `_intent_cache` before benchmark rows** — either pre-populate with
   all bench queries OR disable query expansion (`query_expansion_enabled=false`)
   during bench runs. Config flag already exists.
4. **Pin Ollama model at a specific tag** — the auto-detect at
   [ribosome.py:172](../../helix_context/ribosome.py#L172) silently swaps
   models between runs.

---

## Consolidated action list (prioritized)

**Free wins (zero LOC, flip-a-flag):**
1. `filename_anchor_enabled = true` in helix.toml
2. Run `scripts/backfill_parent_genes.py`, then `HELIX_LAYERED_FINGERPRINTS=1`
3. `sr_enabled = true` in helix.toml
4. Re-export rerank teacher labels on 17K genome; flip `rerank_enabled=true`

**Small-LOC structural wins:**
5. **Proposal 1:** BM25 shortlist + Helix re-rank on top-50 (~80 LOC, single
   highest-leverage move)
6. **Proposal 3:** Fix `helix_only` content ceiling (~40 LOC; exposes actual
   retrieval quality)
7. **Proposal 4:** Fix `source_index` path resolution (~20 LOC)
8. Bench-mode determinism closures (~30 LOC)
9. PLR calibration pass (rebalance training or tune `high_risk_threshold`)

**Medium-term training investments:**
10. Option B CWoLa refactor (per-(q,g) labels) — unblocks SEMA fine-tune and
    the real per-gene PLR ranker the §C3 spec originally described
11. Splice-label re-accumulation (179 → 2K+)
12. SEMA contrastive fine-tune once per-(q,g) labels land

**Not recommended:**
- 6×6×3 or 3×3×3 tensor restructure (data doesn't support; worse or
  equivalent parameters; no recoverable per-segment signal)
- Variable matrix view (killed by permutation tests; revisit only post
  Option B)

---

## What we didn't investigate

- Full bench mode isolation under concurrent `/context` writes (the race
  between `last_query_scores` consumers and producers is documented but not
  exercised in the current bench suite)
- Scale behavior beyond 17K genes — at 100K+ population dilution will
  compound and BM25's IDF advantage grows
- Whether filename_anchor introduces false positives on well-named test
  files (`test_helix.py` would boost for "helix" queries the same way
  `helix.toml` does)

---

*Research agents: 4 parallel × ~500-700 words each. Synthesis time: ~5 min.
Total agent runtime: ~460 s. Findings converged on "architectural restructure
over training/tensor expansion" independently.*

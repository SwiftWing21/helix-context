# Celestia × Helix — Joint Experiment Spec

> **Status:** proposal / pre-implementation
> **Date:** 2026-04-13 (revised 2026-04-13 pm after Fauxtrot cross-review)
> **Parties:** Fauxtrot (Celestia — perception-to-brain-aligned-salience) ×
> helix-context (cortical retrieval over genome-compressed knowledge)
>
> **Primary thesis:** helix is missing a **self-awareness signal** —
> the budget tiers (`tight/focused/broad`) don't know when their
> ranking is wrong, so they confidently serve tight-mode results on
> query patterns they've never seen. Celestia's **K** (running
> prediction-fidelity score) is the missing control loop: low K means
> "my scoring is broken, widen the net," high K means "trust the
> ranking." Adding K to helix is the bigger bet than replacing weights.
>
> **Secondary thesis:** while we're at it, Celestia's Mamba classifier
> — sitting **on top of** helix's existing SQL-backed retrieval and
> consuming `tier_features` — can learn per-dimension scaling from
> `cwola_log` feedback, replacing the hand-tuned dimension weights.
>
> **What this does NOT propose:** ripping out SQL for Mamba. FTS5,
> `promoter_index`, and chromatin gating are doing structured recall
> correctly. SIKE 10/10 is evidence they work. Mamba is a projector,
> not a `WHERE` clause — the two are complementary, not competing.

---

## 0. What each side actually has (ground truth)

### Celestia

- **Perception encoder** at 4 Hz, 11,522d per tick (visual/audio/semantic
  raw belts + learned features).
- **ProgressiveManifold v7** — 4.2M-param Mamba SSM mapping raw belts →
  25 brain-aligned ROI channels, trained against TRIBE v2 fMRI targets.
  Correlation with TRIBE ground truth: r=0.779 overall.
- **SalienceReactor v7** — 607K-param predictor of next salience state;
  prediction error = surprise signal. r=0.932 at +1 tick.
- **Coincidence detector** on raw 11,522d (no training, 22 channel
  groups, 33 ticks/sec CPU).
- **Emotional particle cloud** + **Hopfield consolidator** (three modes:
  reactive / reflective / generative).
- **8×25 transfer matrix** bridging ROI channels to 8 emotional axes.

### Helix-context

- **Nine retrieval dimensions** (D1–D9; see
  [`docs/DIMENSIONS.md`](../DIMENSIONS.md)): six active
  (semantic/promoter/provenance/working-set/chromatin/cymatics), three
  in-progress (attribution/co-activation/TCM).
- **SR (Successor Representation)** shipped dark in `c9367f8` behind
  `retrieval.sr_enabled`, γ=0.85 — the γ-discounted multi-hop
  generalisation of the harmonic boost (see
  [`docs/future/SUCCESSOR_REPRESENTATION.md`](../future/SUCCESSOR_REPRESENTATION.md)).
- **TCM** (Howard & Kahana 2002) assigned but not yet wired — see
  [`docs/future/TCM_VELOCITY.md`](../future/TCM_VELOCITY.md).
- **CWoLa log** — every retrieval is captured in `cwola_log` with
  `(query, tier_features, top_gene_id, bucket=A/B, requery_delta_s)`
  where A = accepted (no re-query within 60s), B = re-queried. Sprint 3
  training surface exists in [`helix_context/cwola.py`](../../helix_context/cwola.py).
- **Hand-tuned weights** across every dimension. The A/B test
  ([`docs/future/AB_TEST_PLAN.md`](../future/AB_TEST_PLAN.md))
  established that KV-harvest retrieval collapsed to 8% on a 17K-gene
  genome, far below the 40% predicted, because the hand-tuned weights
  over-fit the curated-SIKE query distribution.

---

## 1. The theses being tested

### Primary thesis — K as control loop

> Helix's budget tiers are a confident but blind controller. They
> decide `tight/focused/broad` based on `top_score` and `ratio`
> thresholds, which tell you how *separated* the top candidate is from
> the mean, not how *right* the ranking is. On query patterns the
> scoring was never tuned for, the tiers confidently pick tight mode
> and return three wrong genes.
>
> K — Celestia's running prediction-fidelity score, computed from
> reactor prediction error over a sliding window — is the missing
> self-awareness signal. High K = the scoring has been reliable for
> this pattern; go tight. Low K = predictions aren't matching; go
> broad. It replaces hand-tuned thresholds with a learned confidence
> that adapts to the actual data.

### Secondary thesis — learned per-dimension weights

> Helix's dimension weights (D1–D9 fusion constants) are hand-tuned.
> A small Mamba classifier trained on `cwola_log` can learn per-query
> per-dimension scaling: "for queries that look like this, trust the
> FTS5 signal more than cymatics" vs "for queries that look like this,
> cymatics carries the signal." The Mamba sits **on top of** the
> existing SQL retrieval, consuming `tier_features` at the fusion
> point. It does not replace any of the D1–D9 mechanisms.

### What this would falsify

Primary (K):
- **Falsified** if K-gated budget tiers don't lift helix/cosmic
  retrieval off 0% on KV-harvest. That would mean the problem isn't
  confidence miscalibration, it's that the dimensions themselves
  don't carry the signal for those query types.
- **Supported** if the learned K correlates with ground-truth
  retrieval success (B-bucket ≈ K collapse) — i.e., K goes low on
  exactly the queries that end up being re-queried.
- **Strongly supported** if flipping `tight/focused/broad` from
  threshold-based to K-based recovers ≥10pp on helix/cosmic retrieval
  with no SIKE regression.

Secondary (learned weights):
- **Falsified** if the learned manifold *loses* to hand-tuned across
  the board — dimensions don't carry the signal, or the training
  surface is too confounded (UI bias, session effects).
- **Drawn** if the manifold matches hand-tuned within ±2pp. Still a
  win operationally because it adapts as the genome grows; hand-tuning
  requires retuning.
- **Supported** if the manifold beats hand-tuned by ≥5pp on
  KV-harvest, and the gap grows as genome size doubles.

The K thesis can validate independently of the weights thesis and
vice versa. Both can hold. Neither depends on the other.

---

## 2. Architecture

### The layering — SQL underneath, Mamba on top

```
┌─ helix retrieval pipeline (UNCHANGED) ─────────────────┐
│                                                         │
│  query → Step 1 candidate recall                        │
│           ├─ Tier 1: genes_fts (FTS5 SQL)              │  D1
│           ├─ Tier 2: promoter_index + synonyms (SQL)   │  D2
│           ├─ Tier 3: SPLADE sparse expansion           │  D1
│           └─ Tier 3.5: ΣĒMA cosine + cold fallthrough  │  D1 + D5
│                                                         │
│         → Step 2 score fusion                           │
│           ├─ Provenance / authority boost               │  D3
│           ├─ Density gate                               │  D3 + D4
│           └─ Chromatin filter                           │  D5
│                                                         │
│           ┌──── tier_features[9d] ────┐                 │
│           │  per-dimension raw scores │                 │
│           └───────────┬────────────────┘                │
│                       ↓                                  │
└───────────────────────┼──────────────────────────────────┘
                        ↓
         ┌──── NEW: Mamba classifier head ────┐
         │                                     │
         │  Input  : tier_features[9d]         │
         │           + query_embed[20d]         │
         │           + candidate_embed[20d]     │
         │           + log(requery_delta_s)     │  ← Δ-gating
         │  Model  : single-stream Mamba SSM    │
         │           (Δ-gated by time gap)      │
         │  Output : scaling[9d] — per-dim      │
         │           relevance weights          │
         │           + K[1] — prediction        │
         │           confidence over history    │
         │                                     │
         └──────────┬────────────────────────────┘
                    ↓
┌─ helix retrieval pipeline (CONTINUES) ─────────────────┐
│                                                         │
│   weighted_scores = tier_features * scaling             │
│                                                         │
│   budget_tier     = K_gated_threshold(K)                │
│                      HIGH K → tight                      │
│                      MED K  → focused                    │
│                      LOW K  → broad                      │
│                                                         │
│         → Step 3 rerank                                 │
│           ├─ Cymatics resonance                         │  D6
│           ├─ Tier 5 harmonic boost                      │  D8
│           └─ SR multi-hop (flag-gated)                  │  D8
│                                                         │
│         → Step 4 gene expression → response             │
└─────────────────────────────────────────────────────────┘
```

### What we preserve vs. what we add

- **Preserved exactly:** FTS5, `promoter_index`, SPLADE, ΣĒMA cosine,
  provenance, density gate, chromatin tier, cymatics, harmonic boost,
  SR. All the D1–D9 *mechanisms* continue to do their job. SIKE 10/10
  is evidence they work — don't break what works.
- **Added:** (a) a small Mamba classifier that consumes
  `tier_features` and outputs per-dim scaling + K, (b) a K-gated
  budget tier decision replacing the hand-tuned `top_score`/`ratio`
  thresholds.
- **Removed:** nothing. The hand-tuned weights stay in `helix.toml`
  as fallback (`fallback_to_handtuned = true` when the learned head
  is unavailable or outputs low-confidence).

### Why this is a thin layer, not a substrate swap

> *"Mamba is a projector — it takes states over time and compresses
> them into data of another shape. It does terrible at replacing what
> a WHERE clause can do."* — Fauxtrot, 2026-04-13

SQL does structured recall (find rows matching this predicate).
Mamba does learned projection (turn a sequence of states into a
smaller representation). They're complementary tools for different
jobs. Helix's retrieval problem isn't that FTS5 can't find the right
rows — it's that the scoring stage doesn't weigh the rows correctly
for this query, and the budget controller doesn't know when to
abandon confidence. Both failure modes live after the recall step, in
fusion and thresholding. That's where Mamba fits.

### Parameter count

Target: ≤2M params. Much smaller than Celestia's 4.2M
ProgressiveManifold because the input is ≤50d (not 960d raw
perception). Trains on a 3060-class GPU in minutes. Inference
budget: ≤1ms per candidate on CPU, effectively free on GPU.

---

## 3. Training loop

### Data: `cwola_log` is already the signal

The CWoLa table (STATISTICAL_FUSION §C2, Metodiev/Nachman/Thaler 2017)
captures:

```sql
CREATE TABLE cwola_log (
    retrieval_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 REAL    NOT NULL,
    session_id         TEXT,
    party_id           TEXT,
    query              TEXT,
    tier_features      TEXT,     -- JSON: {tier_name: score}
    top_gene_id        TEXT,
    bucket             TEXT,     -- 'A' (accepted) | 'B' (re-queried) | NULL
    bucket_assigned_at REAL,
    requery_delta_s    REAL      -- seconds to next same-session query
);
```

- **Bucket A** (no re-query within 60s) = implicit accept signal.
- **Bucket B** (re-query within 60s) = implicit reject signal.
- **`tier_features`** — per-retrieval JSON of D1–D9 scores. This is
  the input-side signal.
- **`ts`, `session_id`, `party_id`** — crucial for Δ-gating below.

The CWoLa framing is crucial: we're training on **unlabeled mixtures**
(A and B are noisy, not ground truth), not per-document relevance
labels. This is cheaper than getting human labels and more robust
than explicit feedback.

### Tick representation — retrieval-event ticks, Δ-gated by time gap

Each `cwola_log` row is one tick. The Mamba SSM's Δ parameter
(`softplus(dt_raw)` in the MambaBlock — see
[`train_manifold_v7.py:42`](https://github.com/fauxtrot/celestia/train_manifold_v7.py)
for the reference implementation) is *literally designed* for
variable-rate input. Feed `log(requery_delta_s + 1)` as an explicit
input feature and the SSM learns to gate memory by the time gap
between retrievals:

- **Short gap** (seconds) → same thought, carry state forward
- **Medium gap** (minutes) → same session, partial state reset
- **Long gap** (hours/days) → context shift, flush memory

The fast/medium/slow/vslow timescales then become **accumulator
personas over the same timestamped stream**, not separate pipelines:

| Persona | Δ-weighting | What it learns |
|---|---|---|
| Fast | small Δ | "what's this query about?" |
| Medium | medium Δ | "what's this session about?" |
| Slow | large Δ | "what does this user care about?" |
| Very slow | largest Δ | "what does this codebase need?" |

No reshaping of `cwola_log` required. The `ts` column already gives
us the signal. `session_id` and `party_id` provide optional reset
boundaries.

### Input vector per tick

```
input[t] = concat(
    tier_features[9d],         # D1–D9 raw scores from SQL retrieval
    query_embed[20d],          # ΣĒMA of query text
    top_candidate_embed[20d],  # ΣĒMA of the winning gene
    log1p(requery_delta_s),    # 1d — time gap signal for Δ-gating
    party_id_onehot[P]         # optional — for per-party adaptation
)
```

Dim ≤ 50 + P. The SQL retrieval produces all of this already.

### Output per tick

```
output[t] = {
    'scaling':    [9d],   # per-D1..D9 relevance weights (sigmoid or softmax)
    'K':          [1d],   # prediction confidence, [0, 1]
    'K_internal': [1d],   # internal-only K (for reflection trigger)
}
```

### Loss

Multi-task CWoLa:

1. **Primary — bucket prediction.** Binary cross-entropy on (A, B)
   label given the full input. The scaling head's gradient comes from
   backprop through the score-fusion equivalent
   (`sum(tier_features * scaling)`).
2. **Secondary — K calibration.** MSE against empirical accuracy in
   a rolling window: `K_target = rolling_mean_of_A_bucket_rate`. So K
   learns to predict "what fraction of retrievals like this end up as
   A-bucket."
3. **Tertiary — reflection trigger.** Sparse label on retrievals
   that went through the `SR multi-hop` or `CC rerank` fallback and
   succeeded where the fast path would have failed. K_internal should
   go low on these *before* the fallback fired.

Baseline sanity check: confirm CWoLa classifier converges on
`tier_features` alone before adding query/candidate embeddings. If the
hand-tuned weights are already somewhat separating A and B, the
learned scaling should recover and improve on them.

### Honest caveat on the training signal

User behaviour is confounded by UI position, query formulation, prior
familiarity, and session state in ways fMRI-on-content isn't. The
`cwola_log` signal is *better than hand-tuning* but it's not a pure
retrieval-quality proxy. The manifold will inherit
UI/position/session biases as learned features unless we explicitly
strip them at training time.

**Mitigations:**
- Use `requery_delta_s` for **confidence-weighted labels**: strong A
  (no re-query for >300s) > weak A (no re-query within 60s but session
  ended) > weak B (re-queried in 30–60s) > strong B (re-queried in
  <10s). First training pass on strong labels only.
- **Stratify by query class.** Template-generated vs natural-language
  vs curated-SIKE queries have different base rates; train with
  per-class normalization to avoid the manifold just learning "SIKE
  queries succeed more."
- **Hold out a calibration set** (e.g., one party_id never seen
  during training) for K calibration check. If K generalises
  cross-party, it's tracking ranking quality. If it doesn't, it's
  overfit to session dynamics.

---

## 4. Success criteria (pre-registered)

These thresholds are locked before training so we can honestly check
whether the experiment beat the baseline or we backfit a story.

### Primary — K as control loop

| Metric | Hand-tuned baseline | K-gated win threshold | K-gated draw threshold |
|---|---|---|---|
| helix/cosmic KV-harvest retrieval | 0% / 0% | ≥10% / ≥10% | 0% / 0% |
| SIKE N=10 retrieval | 10/10 | ≥10/10 (tied) | 9/10 |
| K vs B-bucket correlation | n/a | Pearson r ≥ 0.4 | r ≥ 0.2 |
| Budget tier agreement with hand-tuned on high-K queries | n/a | ≥80% | ≥60% |
| Reflection-triggered fallback recall (SR/CC invoked when K_internal low) | n/a | precision ≥0.5, recall ≥0.3 | precision ≥0.3 |

**The headline metric for the primary thesis is helix/cosmic KV-harvest
retrieval.** Hand-tuning currently fails 0-for-13. If K-gated budget
tiers cannot lift those categories off the floor, the confidence-calibration
story is falsified and the problem lives in the dimensions themselves.

### Secondary — learned per-dimension weights

| Metric | Hand-tuned baseline (A post-fix) | Manifold win threshold | Manifold draw threshold |
|---|---|---|---|
| SIKE N=10 retrieval | 10/10 | ≥10/10 (tied) | 9/10 |
| SIKE N=10 answer (qwen3:8b) | 7/10 | ≥8/10 | ≥6/10 |
| KV-harvest N=50 retrieval | 12% | ≥17% (+5pp) | ≥10% (-2pp) |
| KV-harvest N=50 answer | 10% | ≥14% (+4pp) | ≥8% (-2pp) |
| Cross-genome generalization (train on 17K, test on regrown 30K) | n/a | ≤5pp degradation | ≤10pp degradation |

### Latency budget (both theses)

| Stage | Baseline | With Mamba head | Budget |
|---|---|---|---|
| Per-retrieval inference | ~0ms | ≤1ms on CPU, ≤0.1ms on GPU | ≤5ms added p95 |
| Full retrieval p95 | current | current + ≤5ms | hard cap: +20ms |

If latency blows past the budget, fall back to precomputed scaling
cells per `(query_class, party_id)` — degenerate but recovers speed.

---

## 5. What each side contributes

### Fauxtrot (Celestia side)

1. **Mamba classifier architecture** — adapted from the v7 manifold,
   single-stream (not three-pathway, confirmed post-cross-review),
   Δ-gated by `log(requery_delta_s)`. Input ≤50d, output 9d scaling +
   K. ≤400 LOC.
2. **K accumulator** — port of
   [`k_accumulator.py`](https://github.com/fauxtrot/celestia/k_accumulator.py):
   rolling K computation + per-group K (we pick the retrieval-analog
   of visual/auditory/internal) + window detection for sensory
   collapse and reflection trigger. ≤300 LOC.
3. **Training loop** — multi-task CWoLa classifier with K calibration
   and (optional) reflection-trigger head. Standard AdamW,
   confidence-weighted labels. ≤250 LOC.
4. **Sanity protocol** — converge on tier_features alone first,
   validate K correlates with B-bucket on held-out party_id.

### Helix side

1. **Data export pipeline** — `cwola_log` dump endpoint
   (`/admin/cwola-export`) with party-stratified splits and
   requery_delta_s confidence weights.
2. **Bucket backfill** — one-shot script to assign A/B buckets to
   existing `cwola_log` rows where `bucket IS NULL` and the next
   same-session query is known.
3. **Integration point** — a `RetrievalSalienceAdapter` class in
   `helix_context/` consuming Mamba output at Step 2 fusion in
   `context_manager._express()`. Gated by
   `retrieval.learned_salience_enabled` flag, default off.
4. **K-gated budget tier** — replace the threshold-based
   `tight/focused/broad` decision with K-based gating. Gated by
   `retrieval.k_gated_budget_enabled` flag, independent of the
   learned-weights flag so the two theses can be A/B'd separately.
5. **Benchmark extensions** — `benchmarks/bench_needle.py` and
   `bench_needle_1000.py` parameterized on the two flags; new test
   that measures K vs B-bucket correlation directly.
6. **Telemetry panel** — a `/admin/k-windows` view showing live K,
   per-group K, and detected sensory/reflection windows. Useful for
   both debugging and as UI for future Agentome wrapper.

### Open questions for both sides

1. **Sample size for CWoLa training.** How many retrievals before
   A/B distribution is stable? Celestia has empirical data from their
   training runs; helix has the accumulated `cwola_log` size. Need
   to check actual volume before scheduling Phase 1.
2. **Cold start.** First-day behaviour of a learned head is worse
   than hand-tuned. Ship with warm-start (distilled from hand-tuned
   weights over observed `cwola_log`) or accept cold-start penalty
   during shadow mode?
3. **Federation interaction.** Different parties have different
   retrieval patterns. One manifold with party_id as input vs
   per-party heads vs hierarchical (global + party residual)?
   Celestia has no analogue yet — they'd face this question only
   after adopting helix's federation layer.
4. **K reflection trigger semantics on the retrieval side.** What's
   the "evidence accumulated but not processed" equivalent? Candidate:
   the operator asking multiple related queries in quick succession
   without an SR/CC rerank ever firing, then K_internal drops — that's
   the signal to retroactively invoke deep retrieval on the
   accumulated query cluster.
5. **What does K_internal even mean on the retrieval side?** For
   Celestia it's DMN/valence/semantic_integration — cognitive
   post-processing channels. For helix the analog is probably D7
   (attribution), D8 (co-activation/SR), D9 (TCM) — the
   session-and-identity-scale lanes that need time to light up. Worth
   validating.

---

## 6. What Celestia gets back

The exchange is now genuinely symmetric. Each side has a thing the
other side actually needs:

### Immediate — load-bearing for Celestia

1. **CWoLa framework for viewer-behavior salience training**
   *(Fauxtrot's ask, 2026-04-13)*. Celestia's current training signal
   is TRIBE v2 fMRI — a 13GB overnight job per hour of footage, with
   brittle generalization to novel modalities. Our A/B bucket
   framework (`STATISTICAL_FUSION.md`, `helix_context/cwola.py`) can
   port wholesale to Celestia:
   - **A-bucket equivalent:** viewer kept watching / didn't skip / no
     re-seek within N seconds
   - **B-bucket equivalent:** viewer skipped / scrubbed / closed / opened
     a different content source
   - Same CWoLa classifier on the mixtures trains salience without
     ever needing fMRI data as ground truth.

   This is structurally important: it removes Celestia's dependency
   on the TRIBE encoder overnight loop and opens training to any
   dataset where viewer engagement is logged.

2. **Federation / attribution.** Celestia has no story for "whose
   experience is this" when multiple operators share an agent.
   Helix's 4-layer identity model (org / device / user / agent,
   shipped in `8990fb7`) plugs in directly.

3. **Cold-tier compression.** At 11,522d × 4Hz × indefinite runtime,
   Celestia's experience storage will balloon. Helix's pending zstd +
   int8 embedding migration is the same problem, same solution.

### Follow-on — conditional on primary phases validating

4. **Working 9-dim retrieval surface.** Celestia's "active ToM-driven
   retrieval" section is marked designed-not-built. If Celestia adopts
   helix's retrieval layer, the Hopfield wells become retrievable via
   the same 9 lanes + learned salience + K-gated confidence.

5. **Reflection trigger as a reciprocal control loop.** Celestia's
   K-reflection window machinery (sensory K stable + internal K
   starving → invoke LFM to process evidence) is exactly what helix
   needs for deciding when to invoke expensive fallbacks (SR multi-hop,
   CC rerank, full cold-tier scan). If we wire K-reflection on our
   side, we can hand Celestia back the same plumbing with a retrieval
   analogue (evidence = accumulated `cwola_log` rows, processing =
   batched deep retrieval pass).

### The frame

What started as "helix gets learned weights, Celestia gets validation"
is now closer to a genuine exchange:

- **Helix needs:** self-awareness signal, learned weights, reflection
  trigger → three things Celestia ships.
- **Celestia needs:** non-fMRI training signal, federation,
  compression, production retrieval surface → four things helix ships.

Both sides are pipelines trying to extract context and intent from
their respective substrates. The pairing works because the cognitive
architecture is analogous (perception → salience → memory) while the
substrates are complementary (biological signals vs symbolic retrieval).

---

## 7. Risks and honest limits

1. **The dimensions might be wrong, not the weights (or K).** If D1–D9
   don't carry the signal that KV-harvest needs, no amount of learned
   weighting or K-gating helps. The category-level 0% on
   helix/cosmic is suggestive of this failure mode. Test: include a
   control where the Mamba head can create new composite dimensions
   as learned linear combinations of tier_features; if those carry
   the lift, the problem was dimension design, not weighting.
2. **CWoLa requires bucket separation.** If A and B distributions are
   too similar (users re-query not because the first result was bad
   but because the task genuinely needed multiple lookups), the
   classifier has no signal. Test: train on an artificially separated
   subset first (A = dwell > 10s, B = closed tab in < 2s) and confirm
   non-trivial classification accuracy before scaling.
3. **K might collapse on novel queries but not on wrong rankings.**
   K is trained to predict B-bucket. If B-bucket correlates more with
   "query was unusual" than "ranking was bad," K-gated budget tiers
   widen the net on *anything novel* rather than *anything wrong*.
   Mitigation: calibration check in A3 — hold out a party_id and
   verify K correlates with B on queries the model *did* see patterns
   for.
4. **Generalisation off the training distribution.** A head trained
   on current retrieval patterns may degrade when helix adds new
   content domains. TRIBE has the same problem on novel modalities
   (Fauxtrot flagged it). Retraining cadence: monthly or on
   retrieval-quality regression.
5. **Compute cost at training time.** CWoLa training on ≤50d input
   should be minutes, not overnight. Risk: accumulated `cwola_log`
   is too sparse to train on for weeks. Check sample-size in A0
   before scheduling.
6. **Mamba Δ handling of extreme gaps.** `log(requery_delta_s)`
   saturates poorly at very long gaps (weeks+). Test gap buckets and
   confirm the SSM isn't just ignoring anything past a session. If
   it is, a hard session-reset signal (via `session_id` change) is
   more honest than trusting Δ to do all the work.
7. **We might be optimizing for the wrong signal entirely.** If
   helix's real bottleneck is extraction (LLM can't find the answer
   in the expressed context even when the right genes are retrieved
   — which the A/B test's "retrieval 8% but answer 8%" numbers are
   consistent with), then improving retrieval quality caps out fast.
   Parallel work on extraction/compression is independent and may be
   more load-bearing.

---

## 8. Proposed sequencing

The two theses can advance on partly-independent tracks. K-gated
budget tiers (primary) can be validated with a simpler model than the
full learned-scaling (secondary), so we sequence K first.

### Track A — K as control loop (primary thesis)

| Phase | What | Owner | Gate |
|---|---|---|---|
| A0 | Confirm `cwola_log` has ≥1k (A, B) pairs with mixed party_id | helix | data volume check |
| A1 | Bucket backfill script + export endpoint | helix | export runs cleanly |
| A2 | Port `k_accumulator.py` to helix with retrieval-analog channel grouping | Fauxtrot+helix | K computes per-retrieval |
| A3 | Offline bench — compute K for historical retrievals, check correlation with B-bucket label | joint | Pearson r ≥ 0.2 on held-out party |
| A4 | K-gated budget tier implementation behind flag | helix | passes test suite; SIKE unchanged |
| A5 | Shadow mode — log K + would-have-been tier alongside current tier | helix | ≥1 week production logs |
| A6 | A/B on live retrieval (K-gated vs threshold-based per session) | helix | §4 primary thresholds met |
| A7 | Default K-gated if A6 passes; retrospective writeup | joint | — |

A3 is the **primary go/no-go.** If K doesn't correlate with B-bucket,
the confidence-calibration story is falsified and Track A stops.

### Track B — learned per-dimension weights (secondary thesis)

Can start in parallel with Track A from A1 onwards. Depends on K
working if we want them to co-exist (K as gate to how aggressive the
learned weights get).

| Phase | What | Owner | Gate |
|---|---|---|---|
| B0 | — (shares A0/A1 with Track A) | — | — |
| B1 | Mamba classifier architecture port + Δ-gated training loop | Fauxtrot | converges on tier_features alone |
| B2 | Offline bench — swap hand-tuned for manifold on SIKE + KV-harvest fixed-genome snapshot | joint | §4 secondary thresholds met |
| B3 | `RetrievalSalienceAdapter` integration behind flag (separate from A4) | helix | passes test suite |
| B4 | Shadow mode — log manifold scores alongside hand-tuned | helix | ≥1 week production logs |
| B5 | A/B on live retrieval | helix | §4 secondary thresholds met |
| B6 | Default on if B5 passes | joint | — |

B2 is the **secondary go/no-go.** Independent of A — if weights don't
beat hand-tuned but K works, K ships and weights don't.

### Track C — reciprocal (Celestia gets CWoLa)

Runs in parallel, independent of A and B landing.

| Phase | What | Owner | Gate |
|---|---|---|---|
| C1 | Port CWoLa classifier framework from `helix_context/cwola.py` to Celestia's training pipeline | Fauxtrot | compiles + trains on viewer logs |
| C2 | A/B CWoLa-trained salience head vs TRIBE-trained on held-out content | Fauxtrot | matches or beats TRIBE |
| C3 | If C2 passes, replace/supplement TRIBE training loop | Fauxtrot | — |

---

## 9. Companion docs

- [`docs/DIMENSIONS.md`](../DIMENSIONS.md) — the 9 lanes in detail
- [`docs/future/AB_TEST_PLAN.md`](../future/AB_TEST_PLAN.md) — where
  we measured that hand-tuning misses on KV-harvest
- [`docs/future/STATISTICAL_FUSION.md`](../future/STATISTICAL_FUSION.md)
  — the CWoLa framework
- [`docs/future/SUCCESSOR_REPRESENTATION.md`](../future/SUCCESSOR_REPRESENTATION.md)
  — parallel "one clean high-ROI addition" lineage
- [`docs/collab/HELIX_CODEBASE_INTRO.md`](HELIX_CODEBASE_INTRO.md) —
  companion intro for Fauxtrot
- Fauxtrot's `CELESTIA_SALIENCE_BRIEF.md` (external)

---

## 10. Principles carried over

From `AB_TEST_PLAN.md`: **predictions before results, honest
retrospective.** The §4 thresholds are locked. When each phase gate
closes, fill in actuals next to predictions and assess calibration
regardless of direction.

From `MISSION.md`: **digital representation of biology's encoding.**
This experiment is the first test of whether a biology-trained
architecture generalises to a non-biology signal. If K validates, the
claim that helix is "cortical retrieval" earns a load-bearing piece
of evidence. If K doesn't validate but learned weights do, we learn
that cortical-style *self-awareness* was metaphor but cortical-style
*scoring* wasn't. If neither validates, we learn that the cortical
framing was aesthetic and act accordingly.

From Fauxtrot's cross-review (2026-04-13): **don't break what works.**
SQL tiers are doing structured recall and doing it well — SIKE 10/10
is evidence. The Mamba head is additive; the learned K-gate replaces
a hand-tuned threshold, not a working mechanism. Every change in this
spec should pass the question: "is this adding capacity the system
lacked, or replacing capacity that already works?"

---

## Revision history

- **2026-04-13 (initial):** single thesis (learned weights), three-pathway
  Mamba architecture, single-phase sequencing.
- **2026-04-13 (cross-review revision):** split into primary (K) and
  secondary (weights) theses per Fauxtrot's read; corrected
  architecture to show Mamba on top of SQL rather than replacing it;
  added Δ-gated single-stream Mamba with `log(requery_delta_s)`;
  added reciprocal Track C (CWoLa → Celestia); added reflection trigger
  as structural element; expanded honest limits.

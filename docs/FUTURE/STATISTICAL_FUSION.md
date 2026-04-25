> **Status (2026-04-13):** §C1 (W1 cymatics) and §C2 logger half shipped in `f4dcdcc` behind `cymatics.distance_metric`. §C2 trainer + §C3 PLR fusion remain blocked on label accumulation (~3 weeks from 2026-04-13).
>
> **Update (2026-04-21):** §C2 trainer shipped (`scripts/pwpc/sprint3.py`, GradientBoostingClassifier, best CV AUC = 0.631 on the t07-tightened label set — clears the 0.55 gate). §C3 **shipped as a query-quality head, not a per-(q,g) ranker** — see the addendum at the bottom of this document for the scope trade-off. Write-path in `helix_context/cwola.py::sweep_buckets` now applies the cosine intent-delta filter described in §C2.

# Statistical Fusion — From Hand-Tuned Caps to Calibrated Likelihood Ratios

> *"Each tier outputs a different statistic. We add them as if they
>  shared units. They don't."*
> — Researcher 1, 2026-04-13

A design note for replacing helix's hand-tuned additive tier fusion
with a **stacked, calibrated, label-free** statistical scorer drawn
from particle-physics multivariate analysis. Three pieces in
dependency order: Wasserstein cymatics distance (zero prereqs),
CWoLa-style implicit relevance labels (logger ships now, model
fires after 3 weeks of organic logging), and Profile Likelihood
Ratio fusion (consumes the labels CWoLa produces).

This addresses the empirical failure surfaced by
`bench_skill_activation.py`: lex_anchor uncapped reached +291 while
PKI capped at +12, both contributing to the same fused score as if
they shared units. They don't. The fix is structural, not a tuning
patch.

Status: **design note, with logger ready to ship Sprint 1.**
Date: 2026-04-13

---

## The problem with the current fuser

For each candidate gene `g` retrieved against query `q`:

```
score(g | q) = Σ_i w_i · t_i(g, q)
```

Eleven tiers, each producing different statistics in different units:
- PKI compound: capped at +12 (IDF-weighted compound match)
- tag_exact: ×3.0 per match (count-based)
- tag_prefix: ×1.5 per match (count-based)
- FTS5 content: capped at +6 (BM25-derived)
- SPLADE sparse: capped at +3.5 (cosine-derived)
- SEMA boost: bounded but uncapped (cosine ∈ [-1,1] × 2.0 × scale)
- SEMA cold: bounded but uncapped
- lex_anchor: **uncapped IDF accumulator** (observed +291 in skill bench)
- harmonic boost: capped at +3.0
- party_attr: +0.5 (indicator)
- access_rate: capped at +0.25

Adding these as `w_i · t_i` is dimensionally meaningless. The
empirical failure is not a tuning problem ("just lower lex_anchor's
weight") — it's categorical. Fix the categories, the tuning
disappears.

## Three changes, in the order they should ship

### Change 1 — Wasserstein-1 for cymatics (Sprint 1, ~25 LOC)

`cymatics.flux_score(a, b, w) = Σ_i a_i·b_i·w_i` is a weighted dot
product over 256 hashed bins. Cosine treats bin 5 and bin 7 as
exactly as different as bin 5 and bin 250 — there's no notion of
bin distance.

For two Gaussians peaked at hashed bin positions `μ_a, μ_b`:
- Cosine: `cos(a, b) ≈ exp(−(μ_a − μ_b)² / 4σ²)` — saturates at
  ~0 for distant peaks; cannot rank "far" vs "very far"
- Wasserstein-1 (closed form on 1-D ordered support):
  `W1(p, q) = Σ_i |F_p(i) − F_q(i)|` where F is the empirical
  CDF — for two equal-width Gaussians, `W1 ≈ |μ_a − μ_b|`. Linear
  in bin distance.

W1 wins in three cases:
1. Near-miss disambiguation (peaks 3 bins apart vs 150 bins apart)
2. Multi-peak spectra (multiple anchor terms)
3. Hash collisions near boundaries (circular W1 handles wraparound)

**Cost:** 25 LOC, parameter-free, drop-in. Forces spectrum
normalization (may surface latent bugs). Computational cost is the
same as cosine (one O(n) pass).

```python
def w1_circular(a, b):
    pa = a / (a.sum() + 1e-12)
    pb = b / (b.sum() + 1e-12)
    cdf_diff = np.cumsum(pa - pb)
    # Werman 1986: subtract median for circular W1
    cdf_diff -= np.median(cdf_diff)
    return np.abs(cdf_diff).sum()
```

**Validation:** A/B against cosine on `bench_dimensional_lock.py`
variant 2 queries (where partial-overlap is most common). Promote if
NDCG@10 lift > 1pp.

**Reference:** Singh et al. (2020), "Context Mover's Distance,"
arXiv:1808.09663.

### Change 2 — CWoLa logger (Sprint 1, ~80 LOC of the 180 total)

The blocker for any supervised tier weighting is labels. Helix has no
ground-truth relevance annotations. **CWoLa** (Metodiev/Nachman/Thaler
2017, arXiv:1708.02949) proves that the optimal classifier trained on
two unlabeled mixtures `M_1 = f_1·S + (1-f_1)·B` and `M_2 = f_2·S +
(1-f_2)·B` with `f_1 ≠ f_2` recovers the **optimal supervised
classifier**. We don't need labels — we need two buckets with
different signal fractions.

Proposed buckets (from session log):
- **Bucket A (signal-enriched):** retrieval where user accepted —
  no re-query within 60s, no thumbs-down, no edit of produced answer
  within 5 min
- **Bucket B (signal-depleted):** retrieval where user re-queried
  within 60s on a textually related topic (cosine sim > 0.4 on query
  embeddings)

These are not pure — Bucket A contains lucky-but-wrong retrievals,
Bucket B contains good-then-curious. CWoLa's whole point is that this
contamination is OK as long as `f_A ≠ f_B`.

**Convergence (Metodiev §3 / Fig. 3):**
`N_required ~ 1 / (f_A − f_B)²` per bucket to reach within ε of
optimal AUC. With realistic estimates `f_A ≈ 0.7, f_B ≈ 0.3`, gap²
= 0.16, so N ≈ 1.5K-5K per bucket.

**Failure mode:** if `f_A ≈ f_B` (acceptance uncorrelated with
quality — e.g., users always accept because they're tired), the
classifier learns noise. Gate by AUC > 0.55 on a held-out
same-distribution split. If degenerate, abort and fall back to
explicit HITL.

**Sprint 1 ships only the LOGGER** (~80 LOC) — captures
`(retrieval_id, tier_features, ts, session_id)` plus a
post-retrieval check at +60s for re-query event. The trainer
(~100 LOC more) ships in Sprint 3 once buckets accumulate.

**Cost:** Logger 80 LOC. Trainer 100 LOC. Calibration data: 3 weeks
organic at ~200 retrievals/day, or half-day manual sprint of ~500
triples if volume is too low.

### Change 3 — Profile Likelihood Ratio fusion (Sprint 3, ~170 LOC)

Once CWoLa labels exist, replace the additive fuser with PLR.
Cowan/Cranmer/Gross/Vitells (2011), arXiv:1007.1727, define the
profile likelihood ratio for "this gene is relevant" vs "not
relevant":

```
λ(g | q) = L(relevant=1, θ̂̂(1)) / L(relevant=0, θ̂̂(0))
```

For tier independence (a strong assumption — see Risk), this
factorizes:

```
log λ(g | q) = Σ_i log [ p(t_i | relevant) / p(t_i | irrelevant) ]
```

By Cranmer/Pavez/Louppe (2015), arXiv:1506.02169, if we have a
calibrated classifier `s_i(t_i) = P(relevant | t_i)`, the per-tier
likelihood ratio is `r_i = s_i / (1 − s_i)`, so:

```
fused(g | q) = Σ_i log r_i(t_i) = Σ_i logit(s_i(t_i))
```

This is **scale-free by construction**: each tier contributes
log-odds, units cancel, the lex_anchor +291 problem is impossible.

**Critical refinement (R1 finding):** per-tier independence is
empirically false. PKI ↔ tag_exact and FTS5 ↔ SPLADE are correlated.
The factorized log-LR overcounts. **Use the stacked variant** — a
single calibrated GradientBoostingClassifier over ALL 11 raw tier
outputs simultaneously, reading out one `s(g | q)`. This costs ~50
more LOC than per-tier independent calibrators, but removes the
independence assumption.

```python
# fleet/helix/fusion_plr.py
from sklearn.ensemble import GradientBoostingClassifier

class StackedPLRFuser:
    def __init__(self):
        self.clf = None  # trained from CWoLa labels

    def fit(self, tier_features, cwola_labels):
        self.clf = GradientBoostingClassifier(max_depth=3, n_estimators=200)
        self.clf.fit(tier_features, cwola_labels)

    def score(self, tier_outputs):
        # tier_outputs is the 11-vector of raw tier scores for one (q, g) pair
        s = self.clf.predict_proba(tier_outputs.reshape(1, -1))[0, 1]
        # Clip to keep logit finite
        s = max(min(s, 1 - 1e-3), 1e-3)
        return math.log(s / (1 - s))
```

**Per-octave calibration:** Either fit per-party calibrators (cheap,
~50 LOC) or include `party_id` as a stacked feature. R1 recommends
the latter — fewer models, automatic handling of new parties.

## Why this order

1. **W1 first** — 25 LOC, parameter-free, ship behind feature flag,
   measurable tomorrow on `bench_dimensional_lock.py`. Either it
   improves NDCG or it doesn't; either way we learn something cheap.
   Forces spectrum normalization which may surface latent bugs.

2. **CWoLa logger second** — enabling infrastructure for change 3.
   Even if PLR is delayed indefinitely, the logged accept/re-query
   buckets are independently useful for offline analysis,
   retrospective tier weight tuning, and HITL prioritization. Ship
   the logger immediately regardless of when the trainer fires.
   Three-week organic-logging clock starts the moment it lands.

3. **PLR third** — highest expected impact but blocked on label
   accumulation. The stacked variant structurally fixes the
   `lex_anchor +291` failure mode.

## What this does NOT solve

- Routing failures (CpuTagger producing empty domains+entities for
  natural-sentence queries → query_genes raises PromoterMismatch →
  pending buffer fallback bypasses the 12 retrieval signals
  entirely). PLR fixes the *fusion* of tier outputs but doesn't help
  if the tiers never fire.
- Tier completeness — if SR (see SUCCESSOR_REPRESENTATION.md) ships
  as a new Tier 5.5, it becomes a new feature in the stacked PLR.
  No re-architecture needed; just retrain.

## Risk profile

| Change | Risk | Mitigation |
|---|---|---|
| W1 cymatics | Scale mismatch with additive fuser pre-PLR | Feature flag, A/B against cosine |
| CWoLa logger | Buckets degenerate (f_A ≈ f_B) | AUC > 0.55 gate before PLR uses labels |
| PLR stacked | Distribution shift across parties / time | Per-party calibrators or party_id stacked |
| Combined | Migration friction during transition | Ship behind feature flag; cut over per-party once stable |

## Validation

Per change:
- W1: `bench_dimensional_lock.py` A/B at variant 2-3
- CWoLa logger: row count + bucket-fraction-gap measurement at 1-week
  intervals
- PLR: full bench suite with PLR enabled vs disabled, on `genome-bench-N50.db`

## References

- Cowan, G., Cranmer, K., Gross, E., & Vitells, O. (2011).
  Asymptotic formulae for likelihood-based tests of new physics.
  *Eur. Phys. J. C* 71:1554. arXiv:1007.1727
- Cranmer, K., Pavez, J., & Louppe, G. (2015). Approximating
  Likelihood Ratios with Calibrated Classifiers. arXiv:1506.02169
- Hoecker, A. et al. (2007). TMVA — Toolkit for Multivariate Data
  Analysis. arXiv:physics/0703039
- Metodiev, E. M., Nachman, B., & Thaler, J. (2017). Classification
  Without Labels. *JHEP* 10:174. arXiv:1708.02949
- Singh, S. et al. (2020). Context Mover's Distance. arXiv:1808.09663
- Werman, M., Peleg, S., & Rosenfeld, A. (1986). A distance metric
  for multidimensional histograms. *CGIP* 32(3), 328-336.

## Addendum 2026-04-21 — §C3 ships as a query-quality head, not a per-(q,g) fuser

When §C3 was drafted the intent was a per-(query, gene) stacked fuser that
replaces the additive gene ranker. During implementation the Sprint 1 CWoLa
logger (`server.py::log_query` call site ~line 900) landed storing
`tier_features` as the **sum of tier contributions across all genes in the
retrieval** — one row per query, not one row per (q, g) pair. All labels
accumulated between 2026-04-13 and 2026-04-15 (2209 rows) reflect that shape.

The Sprint 3 trainer (`scripts/pwpc/sprint3.py`) therefore learns a
**query-quality classifier** — predicted P(user re-queries within 60s | the
aggregate tier pattern that this retrieval produced). Every candidate gene in
a given retrieval shares the same feature vector, so the classifier cannot
order them; its output is a single per-query confidence score.

### What shipped (Option A)

- **`helix_context/fusion_plr.py::StackedPLRFuser.query_confidence()`** returns
  `{prob_B, logit, score_A}` for one query. `logit` is the scale-free
  log-odds statistic from §C3's original derivation; `score_A = 1 - prob_B`
  is the packet-friendly confidence.
- **`/context/packet`** attaches a `plr_confidence` block when
  `[plr] enabled = true` in `helix.toml`. Gene ranking is untouched — the
  additive fuser still ranks candidates.
- **Model artifact** at `training/models/stacked_plr.joblib` (sha256 sidecar
  written by the trainer; load refuses on mismatch).
- **Best CV AUC: 0.631** on the t07 label set (B ∩ cos(q_t, q_{t+1}) > 0.7);
  all four label sets (loose / t04 / t05 / t07) cleared the 0.55 gate.
  Details in `docs/collab/comms/SPRINT3_TRAINER_2026-04-21.md`.
- **Write-path fix:** `sweep_buckets()` now applies the §C2 cos-filter at
  bucket-assignment time (default 0.4, configurable). Previously it skipped
  the filter entirely — see the note in `project_cwola_bucket_accumulation`.

### What Option A does NOT solve

- The "lex_anchor uncapped reaches +291" failure from the original motivation
  is a gene-scoring problem. A query-quality head doesn't touch it. Separate
  fixes are available (rank-based per-tier normalisation, or a capped-sum
  prior over lex_anchor) that don't need CWoLa labels at all.
- Per-gene ranking quality is unchanged by this work.

### Option B — considered, not rejected

If the per-gene ranker remains the goal (e.g., additive-fusion pathology
doesn't respond to rank-normalisation, or downstream benchmarks need a
calibrated per-candidate score), Option B is the refactor path:

1. Change `cwola.log_query()` to emit one row per top-K candidate, each with
   the per-gene tier-score vector at retrieval time (rather than the
   aggregate).
2. Reset the ~2.2K-row label corpus; start accumulating under the new schema.
3. Retrain with `sprint3.py --per-gene` (would need a new flag + feature
   builder).
4. At that point the ranker becomes `score_gene = logit(s(tier_g | q))` and
   the additive fuser retires.

Cost estimate: schema + logger changes ~200 LOC; trainer feature-builder
refactor ~100 LOC; ≥3 weeks of re-accumulated labels at current volume.
Not cheap, but tractable if needed. Raise it when there's a downstream
benchmark signal that Option A's query-level head is insufficient.

## Companion docs

- [`SUCCESSOR_REPRESENTATION.md`](SUCCESSOR_REPRESENTATION.md) —
  trajectory layer; once SR ships, its output becomes another feature
  for the stacked PLR
- [`TCM_VELOCITY.md`](TCM_VELOCITY.md) — fixes TCM's divergence from
  Howard 2005; PLR doesn't affect TCM's update rule
- [`../BENCHMARK_RATIONALE.md`](../BENCHMARK_RATIONALE.md) — explains
  why the existing bench suite under-measures fusion quality

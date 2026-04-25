# Phase 0 v2 follow-ups — consolidated report

**Source:** `cwola_export_20260415_windowed.json` (N = 2209)
**Generated:** by `scripts/pwpc/v2_followups.py`

Executes the open "next moves" from PHASE0_V2_DRILLDOWN.md and LOCKSTEP_MATRIX_TEST_v2.md in one pass. Five tasks:

- T1 Template-shape density in top-sema_boost outliers
- T2 Query-length bias on the cos(q,c) A-B delta
- T3 Segmented 9x9 matrix test (template / mixed / natural)
- T4 10x10 matrix with cos(q,c) as 10th dim
- T5 Non-linear (RBF kernel-PCA) projection test

## T1 — template shape in sema_boost outliers

Heuristic classifier (regex): **template** = slot-fill shapes ("what is the value of X", "what is the Y configured in X"); **mixed** = generic question opener; **natural** = everything else.

### Top-10 labels

| bucket | template | mixed | natural |
|---|---|---|---|
| A | 9 | 0 | 1 |
| B | 10 | 0 | 0 |

### Top-50 labels

| bucket | template | mixed | natural |
|---|---|---|---|
| A | 42 | 1 | 7 |
| B | 48 | 2 | 0 |

### Population density (all rows, not just top-sema_boost)

| bucket | n | template | mixed | natural | template% |
|---|---|---|---|---|---|
| A | 161 | 135 | 6 | 20 | 83.9% |
| B | 2048 | 1758 | 81 | 209 | 85.8% |

### Top-10 B-bucket sema_boost queries (labeled)

| rid | score | label | cos(q,c) | query |
|---|---|---|---|---|
| 1564 | 2034.1 | template | 0.707 | What is the min tasks value in Education/fleet/outcome_tracker? |
| 1563 | 1962.5 | template | 0.785 | What is the min tasks configured in Education fleet? |
| 1412 | 1666.5 | template | 0.806 | What is the categories value in Education/fleet/digest_2026-03-26? |
| 1116 | 1604.0 | template | 0.855 | What is the index value in Education/fleet/api_lua_inventory? |
| 968 | 1585.6 | template | 0.722 | What is the days value in Education/fleet/metering_blueprint? |
| 1452 | 1580.4 | template | 0.680 | What is the 27t23 value in Education/fleet/source_meta? |
| 1611 | 1544.4 | template | 0.876 | What is the payments configured in Education fleet? |
| 1612 | 1535.1 | template | 0.711 | What is the payments value in Education/fleet/payments? |
| 967 | 1475.3 | template | 0.843 | What is the days configured in Education fleet? |
| 1584 | 1466.0 | template | 0.633 | What is the peers by url value in Education/fleet/discovery? |

### Top-10 A-bucket sema_boost queries (labeled)

| rid | score | label | cos(q,c) | query |
|---|---|---|---|---|
| 1958 | 1206.6 | template | 0.333 | What is the port in the cosmic source? |
| 2063 | 1104.1 | template | 0.333 | What is the port in the cosmic source? |
| 2180 | 587.1 | template | 0.417 | What is the value of exception? |
| 1752 | 509.8 | template | 0.680 | What is the value of report data? |
| 1178 | 481.5 | template | 0.464 | What is the value of all ranks in two-brain-audit? |
| 2000 | 398.3 | template | 0.368 | What is the value of state? |
| 2209 | 397.9 | natural | 0.778 | collab status PWPC Celestia next steps phase 0 phase 1 |
| 1377 | 321.5 | template | 0.648 | What is the value of gene b? |
| 1078 | 321.2 | template | 0.463 | What is the value of p3z in steamapps? |
| 960 | 316.2 | template | 0.555 | What is the classname value in CosmicTasha/web/page? |

**Read:** Template fraction in top-k vs population tells you whether sema_boost is preferentially latching onto slot-fill queries (the antiresonance signature). If top-50 template% >> population template%, the hypothesis lives.

## T2 — query-length bias on cos(q,c) A-B delta

### Query length by bucket

| bucket | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| A | 161 | 45.1 | 47 | 27 | 64 |
| B | 2048 | 43.1 | 44.0 | 26 | 62 |

### cos(q,c) within length bins

| bin (chars) | n_A | n_B | mean cos A | mean cos B | delta |
|---|---|---|---|---|---|
| [0, 30) | 35 | 456 | 0.483 | 0.548 | -0.0647 |
| [30, 45) | 40 | 626 | 0.598 | 0.603 | -0.0046 |
| [45, 60) | 58 | 694 | 0.634 | 0.679 | -0.0445 |
| [60, 80) | 26 | 258 | 0.583 | 0.662 | -0.0785 |
| [80, ∞) | 2 | 14 | 0.655 | 0.672 | -0.0175 |

**Read:** If the A-B delta flips sign or collapses inside each length bin, the raw -0.047 delta was Simpson-style length confounding. If the delta persists within bins, it's a real semantic-agreement signal independent of query length.

## T3 — segmented 9x9 tier matrix test

Re-runs LOCKSTEP_MATRIX_TEST_v2 within each query-shape segment. Reports max |r| (gate threshold = 0.20) and Frobenius ||C_A - C_B|| per segment.

| segment | n | n_A | n_B | max\|r\| | null p95 | perm p | top pair | frob ||ΔC|| |
|---|---|---|---|---|---|---|---|---|
| template | 1893 | 135 | 1758 | 0.0496 | 0.073 | 0.418 | fts5 × harmonic (r=-0.050) | 0.6288 |
| mixed | 87 | 6 | 81 | 0.3595 | 0.413 | 0.164 | splade × tag_prefix (r=-0.359) | 3.3288 |
| natural | 229 | 20 | 209 | 0.1649 | 0.213 | 0.232 | splade × tag_prefix (r=-0.165) | 1.4289 |

**Read:** Observed max|r| is pulled from 45 pairwise tests per segment. The permutation baseline shuffles bucket labels within the segment and recomputes max|r|; the `null p95` column is what 95% of random splits produce. A segment only shows real signal if `observed > null p95` and `perm p < 0.05`. Point estimates without this control are misleading for small n_A.

## T4 — 10×10 matrix with cos(q,c) as 10th dim

n=2209 (n_A=161, n_B=2048).

**cos(q,c) alone vs bucket:** r = +0.0508 (p = 0.0169)

### Top 10 pairs by |r|

| tier_i | tier_j | r | p |
|---|---|---|---|
| sema_boost | lex_anchor | -0.0420 | 0.0484 |
| fts5 | harmonic | -0.0344 | 0.106 |
| sema_boost | pki | -0.0333 | 0.118 |
| splade | harmonic | -0.0332 | 0.118 |
| lex_anchor | harmonic | -0.0315 | 0.139 |
| tag_exact | harmonic | -0.0315 | 0.139 |
| tag_prefix | harmonic | -0.0311 | 0.144 |
| sema_boost | tag_exact | -0.0309 | 0.146 |
| harmonic | harmonic | -0.0304 | 0.153 |
| fts5 | cos_qc | +0.0282 | 0.186 |

### Pairs involving cos(q,c)

| tier_i | tier_j | r | p |
|---|---|---|---|
| fts5 | cos_qc | +0.0282 | 0.186 |
| sr | cos_qc | -0.0278 | 0.191 |
| cos_qc | cos_qc | -0.0254 | 0.232 |
| sema_boost | cos_qc | +0.0211 | 0.322 |
| splade | cos_qc | +0.0194 | 0.363 |
| pki | cos_qc | -0.0174 | 0.414 |

**Gate (|r| >= 0.2):** FAIL

## T5 — non-linear (RBF kernel-PCA) projection

| kernel | top comp r | |r| | all comps |
|---|---|---|---|
| gamma=0.1 | -0.0443 | 0.0443 | k1: -0.044, k2: -0.036, k0: -0.011 |
| gamma=0.5 | -0.0213 | 0.0213 | k1: -0.021, k0: +0.020, k2: -0.017 |
| gamma=1.0 | +0.0215 | 0.0215 | k0: +0.022, k2: -0.013, k1: -0.008 |

**Read:** RBF kernel-PCA asks whether a non-linear combination of the 9 tier z-scores separates A from B. Still gated at |r| >= 0.2.

## Consolidated verdict

- **T1 — mild template enrichment in B-outliers:** top-50 B template = 96% vs population B = 86% (+10 pp). Top-50 A = 84% vs population A = 84% (+0 pp). sema_boost outliers in the B-bucket ARE preferentially slot-fill shapes, but the lift is small and consistent with the antiresonance direction, not a clean confirmation.
- **T2 — cos(q,c) delta is not length-confounded:** 5/5 length bins show A<B (same direction as the pooled -0.047). Simpson's-paradox control cleared — the semantic-agreement gap is a real population-level effect, but weak.
- **T3 — segmentation does not unlock lockstep:** ALL segments fall below their own permutation p95 — the per-segment max|r| is noise, not a washed-out signal. Per-segment detail: template: obs=0.050 p95=0.073 p=0.418 [noise]; mixed: obs=0.359 p95=0.413 p=0.164 [noise]; natural: obs=0.165 p95=0.213 p=0.232 [noise]
- **T4 — cos(q,c) as 10th dim doesn't help:** 10×10 gate FAIL; cos(q,c) alone r = +0.0508 (p = 0.0169). Weakly significant population-level effect consistent with T2, but far below the 0.20 gate for routing/head-training.
- **T5 — non-linearity isn't the missing ingredient:** best RBF kernel-PCA |r| = 0.0443 across γ∈{0.1, 0.5, 1.0} (FAIL). No hidden non-linear bucket axis in the 9-tier score manifold.

### What this means for Sprint 3

- The **9-tier score matrix alone does not carry bucket signal** at the |r| ≥ 0.2 gate, in any framing tried here (scalar, 9×9 pairwise, 10×10 + cos(q,c), segmented by query shape, or RBF-kernelized).
- Two real-but-weak effects survive statistical control: (1) B-bucket top-sema_boost outliers are **+10pp more template-shaped** than population baseline; (2) B-bucket cos(q,c) is **~0.05 higher than A** and persists across all length bins. These are directionally consistent with the antiresonance hypothesis but effect sizes are small.
- **Implication:** the agreement head can't be trained on these features alone. Next-move candidates: (a) add out-of-score-matrix features (query length, token overlap, path anchor quality); (b) switch from |r| gating to a classifier AUC-style evaluation on the current signal and accept a weaker ceiling; (c) accept that the re-query bucket label is too noisy a target and explore alternatives (explicit thumbs-up/down, downstream answer-quality signals).

— Generated by `scripts/pwpc/v2_followups.py`

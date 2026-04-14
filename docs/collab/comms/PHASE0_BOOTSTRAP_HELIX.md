# Phase 0 PWPC Bootstrap — helix side

**Source:** `cwola_export_20260414.json` (N=791 rows; 37 A, 754 B)
**Party:** swift_wing21 (single-party — no cross-party generalization possible on this slice)
**Method:** per-tier variance + precision Π = 1/(var + 1e-9), conditional on tier firing (score present in `tier_features`).
**Caveat:** 95.3% B-rate is inflated by 5-min synthetic-session windowing on burst traffic; treat A-vs-B differences here as methodology validation, not load-bearing findings until organic data accumulates (~2–3 weeks) or schema is enriched per Phase 1.

## Interpretation guide

- **fire_rate**: fraction of retrievals where this tier produced a score. Low fire_rate (sema_boost 17%, sr 59%) means the tier is gated — most rows won't inform its precision.
- **Π (precision)**: higher = more consistent score across rows = tier is firing in a narrow band. Lower = tier's output has high variance across rows.
- **Π_A / Π_B**: if a tier's precision differs between A-bucket and B-bucket, that tier carries content-dependent signal. Ratio ≈ 1 means no differentiation on this slice.


### All rows (N = 791)

| tier | n | fire_rate | mean | var | Π (=1/var) | median | min | max |
|---|---|---|---|---|---|---|---|---|
| fts5 | 791 | 1.000 | 258.000 | 448.111 | 0.002 | 264.000 | 210.000 | 282.000 |
| splade | 790 | 0.999 | 100.682 | 1259.525 | 7.939e-04 | 90.204 | 32.722 | 166.422 |
| sema_boost | 132 | 0.167 | 179.613 | 14435.737 | 6.927e-05 | 164.338 | 14.617 | 481.875 |
| lex_anchor | 791 | 1.000 | 2321.194 | 8.320e+06 | 1.202e-07 | 1026.000 | 243.000 | 14727.816 |
| tag_exact | 791 | 1.000 | 3382.024 | 2.127e+07 | 4.701e-08 | 1044.000 | 123.000 | 20238.000 |
| tag_prefix | 791 | 1.000 | 2648.570 | 7.408e+06 | 1.350e-07 | 1444.500 | 483.000 | 12420.000 |
| pki | 725 | 0.917 | 95.951 | 12967.925 | 7.711e-05 | 50.000 | 5.000 | 547.982 |
| harmonic | 791 | 1.000 | 463.571 | 93272.968 | 1.072e-05 | 348.000 | 149.000 | 1309.000 |
| sr | 466 | 0.589 | 0.428 | 0.020 | 50.055 | 0.443 | 0.101 | 0.739 |

### A-bucket only (N = 37)

| tier | n | fire_rate | mean | var | Π (=1/var) | median | min | max |
|---|---|---|---|---|---|---|---|---|
| fts5 | 37 | 1.000 | 259.622 | 409.911 | 0.002 | 270.000 | 216.000 | 282.000 |
| splade | 36 | 0.973 | 90.730 | 1308.301 | 7.643e-04 | 82.307 | 32.722 | 165.649 |
| sema_boost | 7 | 0.189 | 87.618 | 5789.465 | 1.727e-04 | 33.657 | 25.646 | 239.271 |
| lex_anchor | 37 | 1.000 | 2193.467 | 6.502e+06 | 1.538e-07 | 1017.000 | 243.000 | 10887.232 |
| tag_exact | 37 | 1.000 | 3658.541 | 2.144e+07 | 4.665e-08 | 1044.000 | 123.000 | 19233.000 |
| tag_prefix | 37 | 1.000 | 2858.635 | 7.547e+06 | 1.325e-07 | 1540.500 | 706.500 | 12292.500 |
| pki | 35 | 0.946 | 87.567 | 10261.437 | 9.745e-05 | 55.000 | 5.000 | 480.558 |
| harmonic | 37 | 1.000 | 504.703 | 117554.425 | 8.507e-06 | 348.000 | 198.000 | 1309.000 |
| sr | 30 | 0.811 | 0.444 | 0.016 | 61.756 | 0.482 | 0.195 | 0.601 |

### B-bucket only (N = 754)

| tier | n | fire_rate | mean | var | Π (=1/var) | median | min | max |
|---|---|---|---|---|---|---|---|---|
| fts5 | 754 | 1.000 | 257.920 | 449.850 | 0.002 | 264.000 | 210.000 | 282.000 |
| splade | 754 | 1.000 | 101.157 | 1252.242 | 7.986e-04 | 90.331 | 36.040 | 166.422 |
| sema_boost | 125 | 0.166 | 184.765 | 14419.455 | 6.935e-05 | 172.089 | 14.617 | 481.875 |
| lex_anchor | 754 | 1.000 | 2327.462 | 8.408e+06 | 1.189e-07 | 1026.000 | 243.000 | 14727.816 |
| tag_exact | 754 | 1.000 | 3368.455 | 2.126e+07 | 4.704e-08 | 1044.000 | 123.000 | 20238.000 |
| tag_prefix | 754 | 1.000 | 2638.261 | 7.399e+06 | 1.352e-07 | 1444.500 | 483.000 | 12420.000 |
| pki | 690 | 0.915 | 96.376 | 13101.464 | 7.633e-05 | 50.000 | 5.000 | 547.982 |
| harmonic | 754 | 1.000 | 461.553 | 91994.345 | 1.087e-05 | 348.000 | 149.000 | 1309.000 |
| sr | 436 | 0.578 | 0.426 | 0.020 | 49.460 | 0.442 | 0.101 | 0.739 |

### Per-tier precision ratio (A vs B)

Content-dependent structure test from PWPC spec. If Π differs meaningfully by bucket, 
per-dimension precision carries information about retrieval outcome. Ratio > 1 means 
A-bucket is *more* precise (more consistent score) for that dimension; ratio < 1 means 
B-bucket is more precise.

| tier | Π_A | Π_B | Π_A / Π_B | n_A | n_B | mean_A | mean_B | mean Δ (A-B) |
|---|---|---|---|---|---|---|---|---|
| fts5 | 0.002 | 0.002 | 1.097 | 37 | 754 | 259.622 | 257.920 | 1.701 |
| splade | 7.643e-04 | 7.986e-04 | 0.957 | 36 | 754 | 90.730 | 101.157 | -10.427 |
| sema_boost | 1.727e-04 | 6.935e-05 | 2.491 | 7 | 125 | 87.618 | 184.765 | -97.147 |
| lex_anchor | 1.538e-07 | 1.189e-07 | 1.293 | 37 | 754 | 2193.467 | 2327.462 | -133.995 |
| tag_exact | 4.665e-08 | 4.704e-08 | 0.992 | 37 | 754 | 3658.541 | 3368.455 | 290.086 |
| tag_prefix | 1.325e-07 | 1.352e-07 | 0.980 | 37 | 754 | 2858.635 | 2638.261 | 220.374 |
| pki | 9.745e-05 | 7.633e-05 | 1.277 | 35 | 690 | 87.567 | 96.376 | -8.810 |
| harmonic | 8.507e-06 | 1.087e-05 | 0.783 | 37 | 754 | 504.703 | 461.553 | 43.150 |
| sr | 61.756 | 49.460 | 1.249 | 30 | 436 | 0.444 | 0.426 | 0.018 |

## Phase 0 gate — verdict

**1 of 9 tiers show meaningful A-vs-B precision ratio** (outside [0.5, 2.0]):

- **sema_boost**: Π_A/Π_B = 2.491 (A more precise); mean_A = 87.618, mean_B = 184.765

This is a methodology-level positive signal — per-dimension precision is not uniform across bucket labels. With the B-rate caveat above, treat this as 'PWPC Phase 0 analysis is viable on helix data' rather than 'these tiers predict retrieval failure'.

## Next moves

1. Schema enrichment (this week, Phase 1 prerequisite): add per-tier **raw** scores + `query_sema[20]` + `top_candidate_sema[20]` columns to `cwola_log`; re-export after a few days of organic traffic.
2. When enriched data arrives: re-run this script. If A-vs-B precision ratio becomes more meaningful on organic data, PWPC Phase 0 gate passes cleanly.
3. Per-dimension precision as a live telemetry signal: add `helix_tier_precision{tier}` gauges to OTel dashboard (Raude's Sprint 5A infrastructure makes this trivial).
4. Sparse-firing tiers (sema_boost 17%, sr 59%, pki 92%) — per-class firing-rate breakdown is the Sprint 5/6 instrumentation item from Raude's Council Triage; directly informs which tiers have enough data for meaningful Π.

— Laude (generated by `scripts/pwpc/phase0_bootstrap.py`)

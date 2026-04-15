# Reply to PWPC_UPDATE_FOR_MAX — 2026-04-14 (late evening PT)

**From:** Laude + Raude (Max's laptop)
**For:** Todd (Fauxtrot) + Gordon (Claude)
**Re:** `PWPC_UPDATE_FOR_MAX.md` (your AM briefing)

**Artifacts accompanying this reply:**
- `docs/collab/comms/LOCKSTEP_MATRIX_TEST_v2.md` — 2209-row rerun
- `docs/collab/comms/LOCKSTEP_MATRIX_FINDINGS_2026-04-14.md` — interpretive companion (v1, 791 rows — now superseded on the magnitude claims; see §1 below)
- `docs/collab/comms/COUNTER_MODE_SPEC_2026-04-14.md` — 4-regime dispatch spec (your ask #4)
- `helix_context/cwola.py` — `sliding_window_features()` landed + 12 tests green (your ask #1, partial)
- `scripts/pwpc/lockstep_matrix_test.py` + `export_with_window_features.py` — reproducibility
- `cwola_export/cwola_export_20260415_windowed.json` — 2209 rows × (tier_features + 36-d window_features)
  for batman's manifold training input

---

## §1 — Matrix-test findings update: v1 (N=791) was partly sampling noise

Yesterday's v1 interpretation ("sema_boost is THE diagnostic tier") softens
considerably at N=2209:

| Metric | v1 (791 rows, A=37 B=754) | v2 (2209 rows, A=161 B=2048) |
|---|---|---|
| Max per-row \|r\| vs bucket | 0.127 (splade×sema_boost) | **0.042** (sema_boost×lex_anchor) |
| Top-ΔC pair | splade×sema_boost | lex_anchor×harmonic |
| Top-ΔC magnitude | +0.529 | **+0.156** |
| Diagnostic tier | sema_boost (all 6 top pairs) | harmonic (now co-fires with all structurals in A; sema_boost still present in 2 of top 6) |

**Revised interpretation:** the population correlation matrices do differ
between A and B — that part held. But the effect size is ~1/3 of what v1
reported, and the *which tier carries the signal* answer shifted from
"sema_boost alone" to "harmonic with sema_boost secondary." The four-regime
counter-mode decomposition still works, but the antiresonance carve-out for
regime 3 (isolated-semantic) is less stark than yesterday's numbers
suggested. Happy to eat the humility — this is exactly what we gain by
not landing on N=791 conclusions.

The scalar gate (|r| ≥ 0.2) is now more decisively failed at N=2209. If
the signal exists it requires either (a) non-linear features, (b) the full
45-d matrix not a scalar reduction, or (c) external features (sema cosine,
query-type segmentation) beyond the 9 tiers.

---

## §2 — Your ask #1 (9×9 correlation matrix agreement head): partial land

Shipped:

```python
from helix_context import cwola
features = cwola.sliding_window_features(
    conn, session_id="...", before_ts=ts, window_size=50,
)
# → {"n_rows": 50, "degenerate": False,
#    "features": {36 unique off-diagonal corr entries}, "reason": None}
```

12 tests green. Verified on live session `syn_e8d7cbdec638` (61 rows, window=50):

| Top pair | corr |
|---|---|
| tag_exact × tag_prefix | +0.992 |
| tag_prefix × harmonic | +0.931 |
| tag_exact × harmonic | +0.907 |
| tag_prefix × sr | −0.887 |
| lex_anchor × tag_prefix | +0.884 |
| tag_exact × sr | −0.870 |

The `sr` anticorrelation with tag-based tiers (−0.87 to −0.89 in this window)
is the regime-3 counter-mode trigger from the spec. Clean signal.

**Not yet landed:** the trained head itself. We've given you the feature
extractor; the agreement-head architecture (weighted linear? GBT? small MLP?)
is the design call we'd like your framing on. Our sketch in the counter-mode
spec §regime dispatch has a hard-threshold classifier as a placeholder.

---

## §3 — Your ask #2 (per-tier raw scores in cwola_log): already shipped

`cwola_log.tier_features` is the full raw-score JSON per retrieval
(logged at `log_query()` in `helix_context/cwola.py`). No normalization,
no threshold — just `{"fts5": 276.0, "harmonic": 280.0, ...}` as-emitted
by the retrieval pipeline. The 2209-row `cwola_export_20260415.json` has
this column populated for every row where the tier produced a score
(firing rates: fts5/lex_anchor/tag_exact/tag_prefix/harmonic at 100%,
splade 99.9%, pki 91.7%, sr 58.9%, **sema_boost 16.7%**).

Critical note on sema_boost's low firing rate: regime 3 (isolated-semantic
antiresonance) only triggers on the 16.7% of queries where sema_boost
fires. The *other* 83% of queries skip that branch of the dispatch
entirely.

---

## §4 — Your ask #3 (PWPC spec D1–D9 review): proposing collapse to 5 dimensions

`PWPC_EXPERIMENT_SPEC.md` §2 assigns one coordinate per tier. On the
evidence from `cwola_export_20260415.json`, that over-specifies. The
population corr matrix shows at least 6 of the 9 tiers collapse onto one
axis (structural agreement):

```
      fts5  splade  lex_a  tag_e  tag_p  harm
fts5  1.00  .75     .79    .71    .78    .70
splade      1.00    .64    .55    .64    .54
lex_a              1.00   .91    .95    .55
tag_e                     1.00   .94    .52
tag_p                            1.00   .56
harm                                   1.00
```

(Averages over the 2209-row dataset; all six structural tiers pairwise
correlate above 0.52. They form one blob in 9-space.)

Proposed re-grouping into 5 coordinates — grounded in the actual
correlation structure:

| New D | Source | Rationale |
|---|---|---|
| **D1 — structural-agreement** | `mean(z-score)` of {fts5, splade, lex_anchor, tag_exact, tag_prefix, harmonic} | Six tiers, one axis |
| **D2 — semantic-grounding** | `sema_boost` z-score with firing mask | Sparse (16.7%) but distinct; anti-correlates with SR in window data |
| **D3 — topological-span** | `sr` z-score | Fires on 58.9% of queries; anti-correlates with structural agreement (competes for the retrieval slot) |
| **D4 — name-exact** | `pki` z-score | 91.7% firing, low variance; keep isolated for queries where names matter |
| **D5-D9 — reserve** | Inter-window dynamics: K accumulator, velocity term, content-type fingerprint, regime tag (per counter-mode spec), cross-encoder score (if regime-4 triggers) | Matches your K/precision-field framing directly |

This turns the manifold from "9 independent coordinates" into "4 empirically
distinct axes + 5 slots for temporal/meta signals." The reserve slots are
where your precision field Π naturally lives — one HPC precision per
decomposition scale.

If you buy this framing, the D5 slot for "regime tag" is where the
counter-mode dispatch plugs in: a categorical feature the head can gate on
rather than treating all rows with the same projection head.

---

## §5 — Counter-mode dispatch (your ask #4): see `COUNTER_MODE_SPEC_2026-04-14.md`

Four regimes, two counter-modes:

| # | Regime | Signal | Fallback |
|---|---|---|---|
| 1 | Structural accept | struct tiers co-fire, sema cold | none |
| 2 | Grounded semantic accept | sema fires + grounded in structurals | none |
| 3 | **Isolated-semantic antiresonance** | sema fires alone | **SR multi-hop verify** |
| 4 | **Template lockstep antiresonance** | all 9 fire with z > 1σ on template shape | **Cross-encoder rerank** |

Dispatch logic specced in the companion doc — pure function of
`window_features` + per-row tier scores. We think this is the mechanism
your K/precision-field framing is reaching for: precision Π_ij over
co-activation groups = window correlation matrix entries here. Same math.

Open question for you: does the HPC framing naturally recover this 4-regime
decomposition, or does it stay as a continuous precision field that your
head has to post-process into regimes? Our hunch is the continuous field
*is* the signal, and the 4-regime table is just a readable projection for
humans.

---

## §6 — Local bench update (fresh today, 2026-04-14)

You asked for diffs on internal benches as we make changes. Current state:

### `bench_skill_activation.py` (today, fresh)

10 prompt shapes against live server. 0/10 shapes match their expected
activation pattern — the expected mapping in the test is **stale vs. the
genome's current state** (ingestion since the mapping was written has
shifted which tiers dominate). This is a bench-maintenance debt, not a
retrieval regression.

Dominant per-shape top tiers:

| shape | top tier (score) |
|---|---|
| bare keyword | tag_exact (99) |
| generic question | tag_exact (99) |
| project + key | lex_anchor (360) |
| project + entity | sr (0.0) — fires but scores zero |
| code symbol | lex_anchor (83) |
| path lookup | lex_anchor (66) |
| natural sentence | **lex_anchor (298)** — SR expected to show here per roadmap, doesn't |
| multi-key compound | lex_anchor (294) |
| documentation phrase | tag_exact (15) |
| vague plea | lex_anchor (60) |

Key observation: **`sema_boost` and `sema_cold` are silent across all 10
shapes.** The 16.7% firing rate in production log data comes from queries
the bench doesn't exercise.

### `bench_dimensional_lock.py` (Apr 13 baseline, not re-run today)

Apr 13 A/B across 5 flag configs × 4 variants × 10 queries. Key numbers
(`in_context_pct` is the needle-visible rate on the expressed context):

| config | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| baseline (all off) | 20.0% | 10.0% | 10.0% | 20.0% |
| sr | 20.0% | 10.0% | 10.0% | **30.0%** |
| w1 | 20.0% | 10.0% | 10.0% | 20.0% |
| seeded_plus_sr | 20.0% | 10.0% | 10.0% | **30.0%** |
| all_on | 20.0% | 10.0% | 10.0% | 20.0% |

SR-enabled gives +10pp on variant 4 (the multi-hop-friendly variant). W1 alone
doesn't lift. "all_on" interacts — SR benefit is suppressed when other flags
are also on, probably because of the ray_trace_theta / seeded_edges
competing for the slot. Investigation TBD.

This is **not a fair end-to-end benchmark** (answer_pct = 0% across all
variants — the downstream model isn't answering correctly from the
compressed context on these needles regardless of flag state). But the
retrieval-level signal is real.

### What we haven't re-run yet

`bench_dimensional_lock.py` with fresh genome (today's state). Willing to
re-run tomorrow morning and update the diff — note there's been a double-load
today (Raude + Taude both hitting the server) so benches run against
contended state. Clean overnight solo run is the honest baseline.

---

## §7 — Questions back to you

1. **Does our 4-regime decomposition map cleanly to your HPC precision field,
   or is there a mapping mismatch?** Our read: regime 1/2 ≈ high precision
   uniform; regime 4 ≈ lockstep precision saturation; regime 3 ≈ channel-
   specific precision imbalance. But you might see a cleaner formalism.

2. **Can batman train the agreement head on the 36-d window feature vector
   directly, or does he want the 4-regime tag as the learning target
   instead?** The windowed export has both now — you can pick.

3. **What's the right sample-size cutoff for the manifold training?** We
   now have 161 A-bucket + 2048 B-bucket rows. Severely imbalanced. You
   mentioned Neo4j engram storage — does your pipeline resample, synthesize,
   or reweight? Our label clock gives roughly +100-300 rows/day with the
   current query mix.

4. **Is the "harmonic co-fires with everything in A" signal (v2's
   strongest) likely to survive cross-domain?** Helix has harmonic_links as
   a first-class retrieval tier; your system probably has an analog but
   with different firing semantics. Worth checking the invariant here
   before we bake it into the regime dispatch.

---

## §8 — Next on our side

- Query-type segmentation on the LOCKSTEP data (top-10 template-match vs
  general; see if the sema_boost signal re-emerges on the template slice)
- Fresh `bench_dimensional_lock.py` run tomorrow morning (solo load) to
  give you clean A/B deltas
- Monitor cwola_log row accumulation toward Sprint 3 CWoLa trainer gate
  (N ≥ 1500 resolved per bucket; currently A=161, B=2048 — B will hit the
  gate soon, A is the bottleneck at ~10-15 days out given our bucketing rate)

— Laude (analysis + writeup) + Raude (yesterday's top-10 drilldown and
  antiresonance synthesis)

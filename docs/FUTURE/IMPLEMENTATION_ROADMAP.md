# Implementation Roadmap — Synthesis of Statistical + Trajectory Tracks

> *"Two tracks. Zero conflicts. One ordered plan."*

The deep-research team (R1 statistical foundations + R2 trajectory
dynamics + synthesizer, all 2026-04-13) produced two design tracks
that turn out to be **fully orthogonal** — different code paths,
different files, different parts of `query_genes`. This doc
prioritizes the combined work and shows the dependency graph.

Status: **plan**, no code yet.
Date: 2026-04-13

---

## TL;DR — Sprint plan

| Sprint | Item | Effort | Track | Status | Commit | Doc |
|---|---|---|---|---|---|---|
| **1** | Wasserstein-1 cymatics swap | ~25 LOC, 0.5d | Stats | ✅ shipped (flag `cymatics.distance_metric`) | `f4dcdcc` | [STATISTICAL_FUSION](STATISTICAL_FUSION.md) §C1 |
| **1** | CWoLa logger | ~80 LOC, 1d | Stats | ✅ live (clock ticking since 2026-04-13) | `f4dcdcc` | [STATISTICAL_FUSION](STATISTICAL_FUSION.md) §C2 |
| **1** | TCM velocity input (Howard 2005) | ~15 LOC, 0.5d | Trajectory | ✅ live (always on) | `f4dcdcc` | [TCM_VELOCITY](TCM_VELOCITY.md) |
| **1** | TCM ρ orthogonality bug fix | ~10 LOC, 0.5d | Trajectory | ✅ live (always on, logs on drift) | `f4dcdcc` | [TCM_VELOCITY](TCM_VELOCITY.md) §"What we got wrong" |
| **2** | Successor Representation (Tier 5.5) | ~80 LOC, 1d | Trajectory | ✅ shipped dark (`retrieval.sr_enabled`) | `c9367f8` | [SUCCESSOR_REPRESENTATION](SUCCESSOR_REPRESENTATION.md) |
| **2** | Theta fore/aft in ray_trace | ~30 LOC, 0.5d | Trajectory | ✅ shipped dark (`retrieval.ray_trace_theta`) | `c9367f8` | [TCM_VELOCITY](TCM_VELOCITY.md) §"Open: theta-style" |
| **3** | CWoLa trainer | ~100 LOC, 1d | Stats | ⏳ blocked on label accumulation (~3 weeks) | — | [STATISTICAL_FUSION](STATISTICAL_FUSION.md) §C2 |
| **3** | PLR stacked-GBT fusion | ~170 LOC, 2d | Stats | ⏳ blocked on CWoLa labels | — | [STATISTICAL_FUSION](STATISTICAL_FUSION.md) §C3 |
| **4** | Seeded co-activation edges + Hebbian decay | ~310 LOC, 1d | Trajectory | ✅ shipped dark (`retrieval.seeded_edges_enabled`) | `5184ea8` | (designed inline 2026-04-13; see `helix_context/seeded_edges.py` docstring) |
| **5** | Kalman session tracking | ~120 LOC + state-space, 2d | Trajectory | 📋 optional (only if SR + velocity + seeded insufficient) | — | TBD |
| **DEFER** | Predictive coding gating | ~100 LOC + reinforcement loop, 3+d | Trajectory | online learning loop doesn't exist yet | — | TBD |

**Current state (2026-04-13):** Sprints 1, 2, 4 shipped (~1200 LOC
total + 169 module tests). Sprint 3 is gated on the CWoLa label
clock that started landing rows when Sprint 1 shipped — ship it at
N ≥ 1.5K rows/bucket with AUC > 0.55 per STATISTICAL_FUSION.md.

## Dependency graph

```
                    (no deps)
                     ┌────────────────┐
                     │ W1 cymatics    │
                     │ (Sprint 1)     │
                     └────────────────┘
                            │
                            ▼  (becomes a feature in)
                     ┌────────────────┐
                     │ Stacked PLR    │◄────── CWoLa trainer ◄── CWoLa logger
                     │ (Sprint 3)     │        (Sprint 3)         (Sprint 1)
                     └────────────────┘
                            ▲
                            │  (becomes a feature in)
                     ┌────────────────┐
                     │ SR Tier 5.5    │
                     │ (Sprint 2)     │
                     └────────────────┘

Independent track — feeds the same retrieval pipeline:
   ┌────────────────┐         ┌────────────────┐
   │ TCM velocity   │ ──────► │ Theta ray_trace│
   │ (Sprint 1)     │         │ (Sprint 2)     │
   └────────────────┘         └────────────────┘

   ┌────────────────┐
   │ TCM ρ-bug fix  │  (Sprint 1, no deps, no consumers)
   └────────────────┘
```

**Key insight:** the only true blocker is CWoLa logger → labels
clock → PLR. Everything else can ship in parallel.

## Sprint 1 (this week — ~3 days, 4 items, 130 LOC)

Goal: ship infrastructure + cheap wins. Start the labeling clock.

1. **W1 cymatics swap** (R1 §C1) — drop-in `flux_score` replacement,
   feature-flag `cymatics.distance_metric: w1|cosine`. Ships behind
   `--use-w1` flag in helix.toml. A/B on `bench_dimensional_lock.py`
   variant 2.
2. **CWoLa logger** (R1 §C2) — append session-log rows with
   `(retrieval_id, tier_features, ts, session_id, party_id)`. Add
   60-second post-retrieval check for re-query event. Schema-only
   change to `session_registry` table + writer hook in /context. No
   trainer yet.
3. **TCM velocity input** (R2 §A) — `t^IN_i = gene_input_vector(g_i)
   − gene_input_vector(g_{i-1})` per Howard 2005 Eq. 16. ~15 LOC
   delta in `tcm.py` + a previous-vector slot in `SessionContext`.
4. **TCM ρ orthogonality bug fix** (R2 §A) — current code at
   `tcm.py:195-200` assumes `t^IN ⊥ t_{i-1}` and silently absorbs
   the violation via `_normalize()` at line 206. Fix: use the
   non-orthogonal closed form (positive root of the quadratic from
   Howard 2002 §3) OR explicitly Gram-Schmidt `t^IN` against
   `t_{i-1}` before the existing formula. Either is ~10 LOC.

Validation: SIKE 10/10 must hold. `bench_skill_activation.py` heatmap
should now show TCM lit on natural-sentence shapes (currently empty).

## Sprint 1/2/4 shipped — next steps

As of `5184ea8` (2026-04-13) Sprints 1, 2, and 4 are in the tree. The
remaining operator work is A/B validation of the dark flags before
Sprint 3 unblocks.

**Dark flags to flip for A/B:**

| Flag | Toggle to | Expected delta |
|---|---|---|
| `cymatics.distance_metric` | `"w1"` | [STATISTICAL_FUSION §C1](STATISTICAL_FUSION.md) — robustness under sparse-peak spectra. Measure on `bench_dimensional_lock.py` variants 2-3 |
| `retrieval.sr_enabled` | `true` | [SUCCESSOR_REPRESENTATION.md](SUCCESSOR_REPRESENTATION.md) — multi-hop topological pull-forward. Should light `sr` column on `bench_skill_activation.py` for natural-sentence + documentation-phrase shapes (currently empty `tier_totals` on those) |
| `retrieval.ray_trace_theta` | `true` | [TCM_VELOCITY §Open](TCM_VELOCITY.md) — fore/aft biased sampling along velocity. Requires TCM session depth ≥ 2 (real chat turns, not cold single queries); graceful fallback otherwise |
| `retrieval.seeded_edges_enabled` | `true` | Cold-start co-activation graph for fresh OPEN genes. Watch the `harmonic_links.source` provenance distribution: seeded (0.3×) dominant at first, co_retrieved (0.7×) climbing over days of real use |

**Bench commands:**

```bash
python benchmarks/bench_dimensional_lock.py     # W1 + SR primary
python benchmarks/bench_skill_activation.py     # theta + velocity-TCM primary
```

Run each with flags off, flip one flag, re-run, diff. Promote any flag
whose A/B shows a non-trivial NDCG@10 lift on its primary bench target
without regression on the others. The design docs named above specify
the exact signal each change is supposed to surface — if the bench
doesn't show it, that's a diagnostic, not a promotion.

Sprint 3 (CWoLa trainer + PLR fusion) stays blocked on label
accumulation. Check `cwola_log` row counts periodically; promotion gate
is AUC > 0.55 per [STATISTICAL_FUSION §C2](STATISTICAL_FUSION.md).

## Sprint 2 (next week — ~2 days, 2 items, 110 LOC)

Goal: highest-ROI structural addition (SR) + the theta delta on top of
Sprint 1.

5. **Successor Representation Tier 5.5** (R2 §B) — see
   [SUCCESSOR_REPRESENTATION.md](SUCCESSOR_REPRESENTATION.md). Lazy,
   on-demand SR rows. γ=0.85, k_steps=4 as starting defaults. Feature
   flag `retrieval.sr_enabled` so it can ship dark.
6. **Theta-style fore/aft** (R2 §E) — biased ray_trace sampling along
   ±velocity direction. Requires Sprint 1 item 3 (TCM velocity input).
   Drop-in change to `ray_trace.py` `_build_adjacency` weighting.

Validation: `bench_skill_activation.py` should show new `sr` and
`ray_trace_theta` columns firing on multi-hop scenarios. Predict: SR
lights up on "natural sentence" and "documentation phrase" — exactly
the shapes that have empty tier_totals today.

## Sprint 3 (after 3-week label accumulation — ~3 days, 2 items, 270 LOC)

Goal: structural fix to fusion. Kills `lex_anchor +291` failure mode.

7. **CWoLa trainer** (R1 §C2) — train GBT on `(tier_features → bucket_id)`
   from the logged data. AUC > 0.55 gate; if degenerate, abort and
   fall back to half-day manual labeling sprint of ~500 triples.
8. **Stacked PLR fusion** (R1 §C3) — single calibrated GBT over all
   12 raw tier outputs (PKI, tag_exact, tag_prefix, fts5, splade,
   sema_boost, sema_cold, lex_anchor, harmonic, **sr**, party_attr,
   access_rate). Note SR is in the list now — it shipped in Sprint 2.
   Per-party calibration via stacked feature.

Validation: full bench suite. Compare PLR-on vs PLR-off on
`bench_dimensional_lock.py` and `bench_skill_activation.py`. Promote
once a per-party A/B shows ≥ 1 NDCG@10 lift.

## Sprint 4 — optional, only if SR + velocity insufficient

9. **Kalman filter session tracking** (R2 §C) — composes with SR
   (different signal: continuous spatial prior vs discrete topological).
   Requires offline fit of state-space matrices F, Q, R. ~120 LOC +
   estimation work.

## What's NOT in this plan (considered + rejected)

- **"Motional EMF" as a separate retrieval tier** — original framing
  of the now-deleted `TCM_TRAJECTORY.md`. R2's reframe made this fall
  out: the velocity term IS the TCM input, not a new tier. Adding
  it as a 13th signal would double-count. Correct resolution is the
  Sprint 1 TCM velocity input change.
- **Plasma physics / MHD frozen-in flux** — R1's search agent flagged
  this as non-portable math. Requires continuous fluid velocity field
  + divergence-free B field that helix doesn't have. Beautiful
  metaphor, doesn't port as math.
- **Predictive coding as a query-time bonus** — R2: it's a *training*
  signal for chromatin promotion/demotion, not a feature for
  `query_genes`. Defer until reinforcement loop exists.
- **Per-tier independent calibration** (per-tier logistic regressors
  in PLR) — R1: PKI ↔ tag_exact and FTS5 ↔ SPLADE are correlated.
  Independence assumption is empirically false. Stacked GBT is the
  fix.
- **"Just patch the lex_anchor cap"** — tempting but R1's point is
  that the *scale-free PLR fusion structurally fixes this* without
  caps. Patching caps now creates migration friction when PLR ships.
- **MoMEMta matrix-element method** — requires a known theoretical
  cross-section. Helix has no Lagrangian. Rejected by R1's search
  agent.

## What we got wrong (the honest part)

The original `TCM_TRAJECTORY.md` (deleted in this commit) proposed
"motional EMF" as a velocity channel for TCM. R2's literature search
showed this is an independent re-derivation of Howard 2005 Eq. 16 with
worse vocabulary. The inner products match; only the framing differs.
Howard 2005 modifies the *input itself* (`t^IN` carries velocity) so
the standard TCM evolution naturally encodes trajectory — one rule,
not two. The motional-EMF doc proposed adding a *separate* trajectory
bonus on top of the existing position bonus. The latter is just
"reinventing TCM 2005 with extra steps."

The literature search saved us from shipping a worse-named version of
the canonical model. The actual change (Sprint 1 item 3) is 15 LOC.
The doc rewrite is `TCM_VELOCITY.md`.

We also discovered our TCM has a **derivation bug** at `tcm.py:195-200`
— ρ is computed assuming `t^IN ⊥ t_{i-1}` but we silently mask
violations via `_normalize()`. Functionally fine for retrieval,
numerically not what Howard wrote. Fixed in Sprint 1.

## Citations (combined R1 + R2 bibliography)

**Statistical / fusion (R1):**
- Cowan, Cranmer, Gross, Vitells (2011) — arXiv:1007.1727 — PLR
- Cranmer, Pavez, Louppe (2015) — arXiv:1506.02169 — Calibrated LR
- Hoecker et al. (2007) — arXiv:physics/0703039 — TMVA
- Metodiev, Nachman, Thaler (2017) — arXiv:1708.02949 — CWoLa
- Singh et al. (2020) — arXiv:1808.09663 — Context Mover's Distance
- Werman, Peleg, Rosenfeld (1986) — *CGIP* 32(3):328 — Circular W1

**Trajectory / dynamics (R2):**
- Howard, Fotedar, Datey, Hasselmo (2005) — PMC1421376 — TCM with velocity
- Howard & Kahana (2002) — *J Math Psych* 46(3) — original TCM
- Stachenfeld, Botvinick, Gershman (2017) — *Nat Neurosci* 20:1643 —
  Successor Representation
- Dayan (1993) — *Neural Comput* 5(4):613 — SR origin
- CMS Collaboration (2014) — arXiv:1405.6569 — Kalman tracking
- Frühwirth (1987) — NIM A262:444 — Kalman in tracking
- Pfeiffer & Foster (2013) — *Nature* 497:74 — hippocampal forward sweeps
- Wang, Foster, Pfeiffer (2020) — *Science* 370:247 — theta alternation
- Rao & Ballard (1999) — *Nat Neurosci* 2:79 — predictive coding
- Ramsauer et al. (2020) — arXiv:2008.02217 — modern Hopfield
- Bulatov et al. (2022) — arXiv:2207.06881 — RMT (baseline)
- Gallego et al. (2017) — *Neuron* 94:978 — neural manifolds
- Vyas et al. (2020) — *ARN* 43:249 — population dynamics
- Amari (1998) — *Neural Comput* 10:251 — natural gradient

## Companion docs

- [`TCM_VELOCITY.md`](TCM_VELOCITY.md) — the rewrite of `TCM_TRAJECTORY.md`
- [`SUCCESSOR_REPRESENTATION.md`](SUCCESSOR_REPRESENTATION.md) — SR Tier 5.5
- [`STATISTICAL_FUSION.md`](STATISTICAL_FUSION.md) — W1 + CWoLa + PLR
- [`../BENCHMARK_RATIONALE.md`](../BENCHMARK_RATIONALE.md) — why the
  existing benches under-measure fusion quality (the discovery doc
  that motivated this whole research push)
- [`../MUSIC_OF_RETRIEVAL.md`](../MUSIC_OF_RETRIEVAL.md) — the
  12-tone framing this work either preserves (12 tiers + octave gate)
  or extends (13 tiers + octave gate, depending on whether SR counts
  as new or absorbs into Tier 5)

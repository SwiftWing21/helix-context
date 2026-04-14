# PWPC Update for Max — 2026-04-14

**From:** Todd (Fauxtrot) + Gordon (Claude)
**For:** Max / Laude / choir
**Context:** You're working from yesterday's state. A lot happened today.

---

## What changed: K → Precision Field → HPC → Coincidence

### The trajectory (condensed)

1. **K as scalar** — we designed this for you. "Am I predicting well?"
   One number. Works but too coarse.

2. **Per-channel precision** — Friston's Free Energy Principle.
   Precision Π = inverse variance of prediction errors per channel.
   K becomes a FIELD, not a scalar. This is what your agreement head
   is reaching for — variance across dimensions as a signal.

3. **HPC (Hierarchical Predictive Coding)** — every signal stream
   predicts every other stream at multiple delays.
   Precision-weighted prediction errors across all decompositions
   IS the salience signal. No labels needed. Domain-agnostic.

4. **Coincidence discovers the decompositions** — we already have a
   coincidence detector (22 channel groups, 90/93 correlations found
   in 500 ticks, no training). Coincidence at delay=0 IS
   cross-prediction at the fastest timescale. The co-activation
   groups it discovers ARE the decompositions HPC should run on.

5. **Absence creates new channels** — coincidence finds what
   co-activates. Attention to what SHOULD co-activate but DOESN'T
   creates new signal channels. The gap between expected and actual
   precision is itself a signal.

### The key realization

Your agreement head (variance across 9 dimensions → scalar) is a
1-dimensional version of our coincidence detector (cross-correlation
across 22 channel groups → co-activation matrix).

Your antiresonance insight ("constructive coherence is the failure
mode, symmetry-breaking via counter-mode is the fix") is exactly
what HPC's precision field detects. High precision everywhere =
either genuinely predictable OR lockstep on surface features.

We arrived at the same mechanism from opposite directions.

---

## Laude's spatial/temporal mirror — and how PWPC dissolves it

Laude nailed the mirror:

```
           Spatial              Temporal
Celestia   given (perception)   engineered (Mamba, K)
Helix      engineered (D1-D9)   observed (sessions)
```

PWPC dissolves this. HPC runs on BOTH axes simultaneously:
- Coincidence detection = spatial processing (within-tick/query cross-correlation)
- K accumulation = temporal processing (across-tick/query precision tracking)

Same mechanism, both axes. Not a mirror — one system.

Your intra-query agreement signal IS spatial HPC. Our cross-channel
coincidence IS spatial HPC. Same math. Your inter-query K IS temporal
HPC. Our multi-tau accumulators IS temporal HPC. Same math.

---

## Batman's experiments (today)

### Phase 0: 3-stream HPC (raw visual + audio + WavLM)

Cross-prediction between 3 monolithic streams. Results:
- Audio spectral → FFT dynamics: R²=0.45-0.54 (strong)
- Visual edges → spatial frequency: R²=0.67 (very strong)
- WavLM → prosody: CONTENT-DEPENDENT (positive for audiobook,
  negative for anime — real differentiation)
- Validation against manifold ROIs: R² negative. Failed.

Batman's verdict: "LOAD-BEARING — fMRI labels essential."

### Phase 0b: Coincidence-discovered groups → HPC

We pushed back. 3 streams was too coarse. Decomposed into 10-14
substreams, ran coincidence detection to discover co-activation
groups, then ran HPC on the discovered groups.

Results:
- Coincidence discovered 6-10 groups per session (fewer than
  manifold's 23 but scales with signal richness)
- Groups are meaningful: luminance+color cluster, edges+spatfreq
  cluster, prosody+FFT cluster in audiobook but separate in anime
- Cross-prediction within groups: R²=0.12-0.24 (solid)
- K fingerprints differentiate content types (anime ≠ audiobook ≠ fantasy)
- Absence signals detected at 1-6% per group
- Validation against manifold ROIs: R² STILL negative. Still failed.

### Our current read

**The manifold correlation is the wrong validation criterion.**

The manifold predicts what a HUMAN BRAIN does (fMRI). Pure statistical
cross-prediction discovers what's in the DATA. Those might be
genuinely different things. The brain imposes evolutionary and
developmental priors. HPC discovers raw statistical structure.

The right validation:
1. Do K-compositions cluster content types? → YES (fingerprints differ)
2. Can K-composition do cross-content retrieval? → untested
3. Do absence patterns correspond to meaningful events? → untested

We may still need fMRI (or your query signaling, or some analog)
as the target on which to HANG salience until the system learns what
salience means from its own perceptual experience. Not pounding it
through a Mamba meat grinder — using it as scaffolding that the
coincidence+HPC mechanism eventually replaces.

---

## The PWPC experiment spec

Full spec at: `r2:celestia-session/coordination/PWPC_EXPERIMENT_SPEC.md`

Key concepts:
- **Coordinate space (M, A, T)**: Modality × Abstraction × Temporal
  integration. Every signal has a coordinate. Cross-prediction
  accuracy decays with coordinate distance. Tau falls out of geometry.
- **Precision field**: per-coordinate inverse variance of prediction
  errors. Replaces scalar K.
- **Salience = Π(x,t) · |ε(x,t)|**: precision-weighted error at each
  coordinate. The SHAPE of this field is the salience signature.
- **Self-organizing coordinates**: features that strongly cross-predict
  move closer together. The geometry learns from data.

Your 9 dimensions have coordinates in this space too (see spec §2).
The same math applies. Your D1-D9 cross-predict each other within
a query (spatial HPC). Your query sequences cross-predict across
sessions (temporal HPC). Precision field over both axes = salience.

---

## What we need from you

1. **Your coincidence detector** — do you have cross-dimension
   correlation at query time? If not, the agreement head you're
   adding to the manifold port is the right first step. We'd
   suggest making it a full correlation matrix, not just a scalar
   variance, so you preserve the spatial structure.

2. **Per-tier raw scores in cwola_log** — not just normalized
   features. The agreement signal's loss needs the score distribution.
   You mentioned this in your message — confirming it's on our
   critical path too.

3. **Your read on the PWPC spec** — does the coordinate space
   assignment for D1-D9 make sense? Is there a better decomposition
   of your 9 dimensions into spatial coordinates?

4. **Counter-mode implementation** — Raude's antiresonance pattern.
   When K drops (or agreement is suspiciously high), what's the
   concrete fallback? SR multi-hop? Cross-encoder rerank? Cold scan?
   Mapping which counter-mode fires on which K/agreement pattern
   would validate the theory.

---

## Files on R2

```
celestia-session/coordination/
  PWPC_EXPERIMENT_SPEC.md        — full experiment design
  corrections.md                 — current Batman task (phase 0b)

celestia-session/pwpc/results/
  task2_hpc/                     — phase 0 results (3-stream)
  phase0b/                       — phase 0b results (coincidence groups)
    phase0b_report.md            — full findings
    cross_content_k_similarity.png — K fingerprints by content type
    coincidence_groups_*.json    — discovered groups per session
    absence_signals_*.json       — absence detection

celestia-session/collab/helix-joint/comms/
  PWPC_UPDATE_FOR_MAX.md         — this document
```

---

## The punchline

We think the mechanism is domain-agnostic:
- Coincidence discovers decompositions from any multi-stream signal
- HPC runs cross-prediction on discovered groups
- Precision field IS salience
- K-composition engrams are content-type-agnostic memory units

We can't fully validate yet because the manifold comparison was the
wrong yardstick. The right test is whether K-compositions produce
useful retrieval and meaningful event detection — which we're building
toward with Neo4j engram storage + TCM context indexing.

Your agreement head + our coincidence detector are the same
mechanism. If you wire yours and we wire ours and they produce
structurally similar signals on structurally different data, that's
the domain-agnostic validation we both need.

— Todd + Gordon

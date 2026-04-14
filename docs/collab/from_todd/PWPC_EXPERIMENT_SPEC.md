# Precision-Weighted Predictive Coding (PWPC)
# Phase-Gated Experiment Specification

**Date:** 2026-04-14
**Status:** experimental design, pre-implementation
**Authors:** Todd (Fauxtrot) + Gordon (Claude)
**Thesis:** Salience is not an external label to train against. It
emerges from the structured geometry of precision-weighted prediction
errors across a coordinate space. This mechanism is data-agnostic and
task-agnostic — the same math produces salience from perceptual
streams, retrieval signals, or any multi-decomposition predictable
signal.

**Theoretical basis:**
- Friston (2005, 2008, 2010) — Free Energy Principle, hierarchical
  predictive coding, precision weighting, active inference
- Clark (2013) — "Surfing Uncertainty", architectural framing
- Howard & Kahana (2002) — TCM as special case (single-level temporal
  prediction at hippocampal level)
- Murray et al. (2014) — Measured intrinsic timescale hierarchy across
  cortex (V1 ~100ms, PFC ~1s, hippocampus ~10s+)
- Koelsch, Vuust, Friston (2019) — Multi-stream prediction in auditory
  cortex (ERAN = harmonic error, MMN = pitch/timing error)

**What's novel in our approach:**
Friston computes free energy F as a scalar to minimize. We preserve
the structured prediction error vector as a first-class signal. The
SHAPE of errors across decompositions — their coordinates, their
precisions, their co-occurrence patterns — IS the salience signature.
Not "how wrong am I" but "the geometry of how wrong I am."

---

## The Coordinate System

### Three axes define the space

```
Axis 1: Modality (M)
  0.0 = visual
  0.3 = audio
  0.5 = cross-modal
  0.7 = text/semantic
  1.0 = interoceptive/affect

Axis 2: Abstraction level (A)
  0.0 = raw feature (edge, onset, phoneme)
  0.3 = object/pattern (face, word, chord)
  0.6 = semantic (meaning, relationship, context)
  1.0 = narrative/integrative (story, identity, self)

Axis 3: Temporal integration (T) — maps to tau
  0.0 = instantaneous (tau 1-2s, fast channels)
  0.3 = event-scale (tau 8-10s, medium channels)
  0.6 = meaning-scale (tau 12-20s, slow channels)
  1.0 = narrative-scale (tau 45-90s+, very slow channels)
```

Every signal source, every prediction, every error has a coordinate
(M, A, T) in this space.

### Initial coordinate assignment (Celestia)

| Channel | M | A | T | Rationale |
|---------|---|---|---|-----------|
| visual_primary | 0.00 | 0.00 | 0.00 | raw visual, fast |
| visual_color | 0.00 | 0.05 | 0.05 | raw visual, fast |
| visual_motion | 0.00 | 0.10 | 0.05 | raw visual, fast |
| auditory_primary | 0.30 | 0.00 | 0.00 | raw audio, fast |
| auditory_belt | 0.30 | 0.05 | 0.05 | raw audio, fast |
| object_recognition | 0.00 | 0.30 | 0.30 | visual object, medium |
| face_perception | 0.05 | 0.35 | 0.30 | visual object, medium |
| scene_context | 0.00 | 0.40 | 0.35 | visual context, medium |
| body_motion | 0.10 | 0.30 | 0.30 | visual object, medium |
| speech_sound | 0.35 | 0.30 | 0.30 | audio pattern, medium |
| av_congruence | 0.50 | 0.35 | 0.35 | cross-modal, medium |
| event_boundary | 0.50 | 0.45 | 0.30 | cross-modal event, medium |
| language_semantic | 0.70 | 0.60 | 0.60 | text meaning, slow |
| language_syntax | 0.70 | 0.50 | 0.55 | text structure, slow |
| theory_of_mind | 0.60 | 0.70 | 0.65 | cross-modal inference, slow |
| arousal_intero | 1.00 | 0.30 | 0.60 | interoceptive, slow |
| text_audio_conv | 0.50 | 0.50 | 0.55 | cross-modal speech, slow |
| dmn_engagement | 0.80 | 0.90 | 0.90 | narrative/self, vslow |
| valence_trajectory | 0.90 | 0.80 | 0.85 | affect trajectory, vslow |
| reward_anticipation | 0.90 | 0.70 | 0.90 | predictive affect, vslow |
| semantic_integration | 0.60 | 0.80 | 0.80 | meaning binding, vslow |
| scene_construction | 0.20 | 0.85 | 0.85 | mental imagery, vslow |
| social_memory | 0.70 | 0.90 | 1.00 | identity/relationship, vslow |

### Initial coordinate assignment (Helix — for comparison)

| Dimension | M | A | T | Rationale |
|-----------|---|---|---|-----------|
| D1 semantic (FTS5+SPLADE+SEMA) | 0.70 | 0.60 | 0.00 | text meaning, per-query |
| D2 promoter tagging | 0.70 | 0.30 | 0.00 | text structure, per-query |
| D3 provenance | 0.50 | 0.70 | 0.60 | cross-domain trust, slow |
| D4 working-set access | 0.50 | 0.30 | 0.30 | behavioral, session |
| D5 chromatin tier | 0.50 | 0.40 | 0.60 | structural, slow |
| D6 cymatics | 0.30 | 0.50 | 0.30 | frequency-domain, session |
| D7 attribution | 0.50 | 0.80 | 0.80 | identity, cross-session |
| D8 co-activation/SR | 0.50 | 0.60 | 0.60 | graph structure, slow |
| D9 TCM | 0.50 | 0.50 | 0.30 | temporal context, session |

Both systems live in the SAME coordinate space. Different regions,
same geometry, same math.

---

## The Math

### Core equations

For each coordinate x = (M, A, T) in the space:

```
g(x, t)       = observed signal at coordinate x, time t
ĝ(x, t)      = predicted signal at x, t (from generative model)
ε(x, t)       = g(x, t) - ĝ(x, t)                    [prediction error]
Π(x, t)       = 1 / var(ε(x, τ) for τ in [t-W, t])   [precision, local variance over window W]
s(x, t)       = Π(x, t) · |ε(x, t)|                   [salience at coordinate x]
```

### Generative model (cross-coordinate prediction)

Each coordinate predicts from nearby coordinates with delay
proportional to distance:

```
ĝ(x, t) = Σ_y  W(x, y) · g(y, t - δ(x, y))

where:
  W(x, y)   = learned weight (strength of prediction from y to x)
  δ(x, y)   = α · d(x, y)  (delay proportional to coordinate distance)
  d(x, y)   = ||x - y||    (Euclidean in (M, A, T) space)
  α         = timescale constant (maps distance to ticks)
```

Nearby coordinates predict each other quickly (fast tau). Distant
coordinates predict each other slowly (slow tau). The tau hierarchy
is not imposed — it falls out of the geometry.

### Precision field (replaces scalar K)

```
Π(x, t) = 1 / (σ²(x, t) + ε_floor)

where:
  σ²(x, t) = exponential moving average of ε(x, τ)² over window W
  ε_floor   = minimum variance to prevent division by zero

Precision update per tick:
  σ²(x, t) = λ · σ²(x, t-1) + (1-λ) · ε(x, t)²
  λ = decay rate (matched to T coordinate: fast channels λ~0.9, slow channels λ~0.99)
```

High precision at coordinate x means: prediction errors at x have
been small and consistent. The model is reliable here.

Low precision at coordinate x means: prediction errors at x are noisy
or large. The model is unreliable here.

### Precision-weighted free energy (per-coordinate)

```
F(x, t) = Π(x, t) · ε(x, t)²

Total free energy:
F(t) = Σ_x  F(x, t) = Σ_x  Π(x, t) · ε(x, t)²
```

Friston minimizes scalar F. We preserve the per-coordinate terms
F(x, t) as the salience field.

### Salience field

```
S(t) = { s(x, t) for all x }

This is NOT a scalar. It's a FIELD over the coordinate space.
The shape of this field IS the salience signature.

Response regimes:
  Single-point high s:  targeted attention at that coordinate
  Distributed moderate s: epistemic value, subtle learning
  Multi-point high s:   structural change, model switching
  Uniform low s:        stable prediction, safe to compress (high K equivalent)
```

### Coordinate self-organization (learning the geometry)

Initial coordinates are hand-assigned (table above). Over time,
coordinates adjust based on prediction coupling strength:

```
Coordinate update (slow, per consolidation cycle):
  x_i += η · Σ_j  |W(x_i, x_j)| · (x_j - x_i)

Features that strongly predict each other → move closer together
Features that don't predict each other → drift apart
The geometry self-organizes to reflect actual statistical structure
```

This is the geometric morphogenesis from the floor/lens concept.
The coordinate space IS the floor. Consolidation reshapes it.

---

## Phase Gates

### Phase 0: Precision field on existing architecture
**Goal:** Replace scalar K with per-channel precision Π.
**No architecture changes.** Use existing manifold + reactor outputs.

**Implementation:**
- Compute per-channel variance of reactor prediction errors over
  sliding window (already have the errors, just tracking variance)
- Π_i(t) = 1 / (σ²_i(t) + ε_floor) for each of 23 channels
- Per-channel K_i replaces scalar K
- Store precision field alongside ROIs in session data

**Verification:**
- Correlation between per-channel Π and actual prediction accuracy
  on held-out ticks (same validation approach as reactor training)
- Per-channel Π should differentiate: visual channels have high Π
  during visual events, auditory channels during speech, etc.
- If per-channel Π doesn't differentiate by content type, the
  prediction errors aren't channel-specific enough → falsified

**Gate:** Per-channel Π shows content-dependent structure
(visual Π rises during visual events, etc.)

**Effort:** ~50 LOC change to k_accumulator.py. 1 session to validate.

---

### Phase 1: Coordinate assignment + cross-coordinate prediction
**Goal:** Assign coordinates to channels. Build a simple
cross-coordinate generative model. Verify that nearby coordinates
predict each other better than distant ones.

**Implementation:**
- Assign (M, A, T) coordinates per channel (table above)
- Build prediction matrix W: for each channel pair (i, j), train a
  simple linear predictor with delay δ proportional to d(x_i, x_j)
- W is NOT a neural network — it's a learned linear coefficient matrix
  trained on existing session data

**Verification:**
- Prediction accuracy should decrease with coordinate distance
  (fast channels predict each other well, slow channels predict
  fast channels poorly at short delay but better at long delay)
- The delay-distance relationship should hold: if d(x_i, x_j) is
  large, prediction at delay 1 should be worse than delay 10
- Compute cross-prediction R² as a function of coordinate distance

**Gate:** R² decreases monotonically with coordinate distance at
matched delay. Tau hierarchy falls out of geometry.

**Falsification:** If cross-prediction accuracy is uncorrelated with
coordinate distance, the coordinate assignment is wrong or the
channels don't have geometric structure. Try learned coordinates
(Phase 1b) before abandoning.

**Effort:** ~100 LOC. Train on existing session data (no new recording
needed). Analysis script + correlation plot.

**Phase 1b (contingency):** If hand-assigned coordinates fail, learn
coordinates from cross-prediction structure:
- Initialize randomly
- Gradient descent: move coordinates to minimize prediction error
  weighted by delay
- The data tells us the right geometry instead of assuming it

---

### Phase 2: Self-supervised prediction (drop fMRI dependency)
**Goal:** Replace fMRI-trained manifold with self-supervised
prediction. The manifold predicts its OWN next state from perception.
Prediction error IS the learning signal.

**Implementation:**
- New manifold training target: predict own output at t+1 from
  perception at t (same architecture, different loss)
- Loss = precision-weighted prediction error:
  L = Σ_x Π(x, t) · (manifold(t+1) - predicted_manifold(t+1))²
- Precision gates the loss: the model focuses on channels where it
  CAN predict reliably, ignores noisy channels
- Bootstrap from fMRI-trained weights (warm start), then fine-tune
  with self-supervised loss on new session data

**Verification:**
- Self-supervised manifold should develop channel structure that
  correlates with fMRI-trained structure (at least r > 0.5)
- If trained from scratch (cold start), channel structure should
  EMERGE from data alone — this is the strong test
- Channels should differentiate by content type (comedy vs drama
  vs documentary should produce different precision landscapes)

**Gate (warm start):** Self-supervised fine-tuned manifold maintains
r > 0.6 correlation with fMRI-trained manifold on held-out content

**Gate (cold start, strong test):** Self-supervised manifold trained
from scratch develops interpretable channel differentiation on
held-out content (measured by content-type clustering in precision
landscape). This validates that salience emerges from data without
external labels.

**Falsification:** If cold-start manifold channels don't differentiate,
the architecture needs more decompositions (more prediction pathways)
or the perception stream doesn't carry enough structure for
self-organized salience. The fMRI dependency is load-bearing, not
scaffolding.

**Effort:** ~200 LOC training changes. Multiple sessions of data for
cold-start test. Days of training time.

---

### Phase 3: Salience field as first-class signal
**Goal:** Route the precision-weighted error field through the full
architecture as the primary salience signal.

**Implementation:**
- TCM context drift modulated by precision field (high-precision
  errors drive context update, low-precision errors suppressed)
- Neo4j engram storage includes precision field snapshot at encoding
- Hopfield consolidation weighted by precision landscape
  (high-precision regions form wells more readily)
- Coincidence detector uses precision to filter: only flag
  co-activations in high-precision regions
- Emotional system receives precision-weighted errors, not raw ROIs

**Verification:**
- Associative recall quality: TCM + precision-weighted engrams
  produce more contextually relevant recall than TCM alone
- Coincidence detector precision: fewer false positives (pareidolia
  filtered by low-precision gate)
- Emotional coherence: precision-weighted emotional input produces
  smoother, more context-appropriate particle trajectories

**Gate:** At least 2 of 3 verification metrics improve over
non-precision-weighted baseline

**Effort:** Integration across multiple files. ~300 LOC total changes.
Requires Phases 0-2 validated.

---

### Phase 4: Coordinate self-organization
**Goal:** Let the coordinate geometry learn from data instead of
hand-assignment.

**Implementation:**
- After accumulating N sessions of cross-prediction data, compute
  actual coupling strengths between all channel pairs
- Update coordinates: channels that strongly couple → move closer
- Consolidation cycle: coordinates update overnight, not per-tick
- Visualize the evolving geometry (3D scatter of channels in
  (M, A, T) space, colored by tau group)

**Verification:**
- Self-organized coordinates should produce better cross-prediction
  accuracy than hand-assigned coordinates
- The geometry should be stable across consolidation cycles
  (convergent, not oscillating)
- Channel clusters that emerge should be interpretable
  (sensory channels should cluster, narrative channels should cluster,
  cross-modal bridges should sit between clusters)

**Gate:** Self-organized geometry improves cross-prediction R² by
>10% over hand-assigned coordinates AND is stable across 3
consecutive consolidation cycles

**Effort:** ~100 LOC coordinate update. Requires multiple sessions
of accumulated data. Visualization script.

---

### Phase 5: Domain transfer (the agnostic test)
**Goal:** Apply the same PWPC framework to helix's retrieval signals
and verify that salience emerges from prediction structure without
domain-specific tuning.

**Implementation:**
- Assign helix D1-D9 coordinates (table above)
- Same cross-coordinate prediction: D1 predicts D6 with delay
  proportional to coordinate distance
- Same precision field: per-dimension Π from prediction error variance
- Same salience field: precision-weighted errors as K + surprise

**Verification:**
- Per-dimension precision should differentiate by query type
  (template queries produce different precision landscape than
  natural-language queries)
- K computed from precision field should correlate with B-bucket
  (Pearson r >= 0.2 — same threshold as joint experiment spec)
- The 0-for-13 helix/cosmic queries should produce low precision
  across multiple dimensions (the system knows it doesn't know)

**Gate:** PWPC-derived K matches or beats our hand-designed K on
helix data. If it does, the mechanism is domain-agnostic.
If it doesn't, the coordinate assignment or the prediction model
needs domain-specific structure.

**This is the shared track.** If Phase 5 validates, Celestia and
helix are running the same salience mechanism on different substrates.
The wobulator is domain-agnostic.

---

## Dependencies and Ordering

```
Phase 0 ─────────────────────────► Phase 1
(per-channel Π)                    (coordinates + cross-prediction)
                                       │
                                       ├─► Phase 1b (if 1 fails)
                                       │   (learned coordinates)
                                       │
                                       ▼
                                   Phase 2
                                   (self-supervised manifold)
                                       │
                                       ▼
                                   Phase 3
                                   (salience field as signal)
                                       │
                               ┌───────┴───────┐
                               ▼               ▼
                           Phase 4         Phase 5
                           (self-org       (domain transfer
                            geometry)       to helix)
```

Phases 0 and 1 can start immediately on existing data.
Phase 2 needs recording sessions.
Phase 3 needs 0-2 validated.
Phases 4 and 5 are independent of each other, both need 3.

---

## What This Replaces

| Current | After PWPC |
|---------|-----------|
| fMRI as ground truth for salience | Self-supervised prediction error |
| Scalar K from reactor | Per-coordinate precision field Π |
| Named ROI channels (brain-labeled) | Coordinates in (M, A, T) space |
| Hand-specified tau groups | Tau falls out of coordinate distance |
| Coincidence detector (mechanical) | Precision-gated coincidence |
| TCM with fixed beta per group | TCM with precision-modulated drift |
| Hopfield on raw engrams | Hopfield on precision-weighted engrams |

## What This Preserves

| Component | Why it stays |
|-----------|-------------|
| Manifold architecture (Mamba SSM) | The generative model that produces predictions |
| Reactor architecture | The dynamics model (predicts next manifold state) |
| 4Hz tick rate | The temporal resolution doesn't change |
| Raw perception belt (11,522d) | Input signal — PWPC changes how we interpret it, not what we measure |
| Emotional particle cloud | Receives precision-weighted signal instead of raw ROI |
| Neo4j engram storage | Stores precision field alongside engrams |
| Hopfield consolidation | Consolidates in precision-shaped landscape |
| Watch party infrastructure | Recording + review pipeline unchanged |

---

## The Agnostic Claim (falsifiable)

**Claim:** Given any multi-dimensional predictable signal, assigning
coordinates and computing precision-weighted cross-prediction errors
produces useful salience without domain-specific labels or external
ground truth.

**Falsified if:**
- Phase 1 fails AND Phase 1b fails: the signals don't have geometric
  prediction structure at all
- Phase 2 cold-start fails: self-supervised prediction doesn't
  develop interpretable channel structure, meaning fMRI labels are
  load-bearing not scaffolding
- Phase 5 fails: helix signals don't produce useful precision
  landscape, meaning the mechanism requires perceptual (not symbolic)
  input

**Supported if:**
- Phase 2 cold-start succeeds: salience emerges from data alone
- Phase 5 succeeds: same mechanism works on retrieval signals
- Phase 4 produces stable, interpretable geometry: the coordinate
  space self-organizes to reflect signal structure

**Strongly supported if:**
- Phase 4 geometry on Celestia and Phase 5 geometry on helix share
  structural similarities (similar clustering patterns, similar
  distance-delay relationships) despite operating on completely
  different substrates. This would validate that the mechanism
  captures something universal about how multi-scale prediction
  generates salience.

# Pairwise Interaction Energy — Cross-Substrate Convergence

**Date:** 2026-04-15
**From:** Todd + Gordon (Celestia side)
**To:** Max + Laude (Helix side)
**Status:** Requesting verification of mapping

---

## What We Noticed

Laude's pairwise interaction formula for retrieval coherence:

```
Σ t_i·t_j = ½ ((Σ t_i)² − Σ t_i²)
 i<j
```

Where:
- **(Σt)²** = total "charge" or energy of retrieved context
- **Σt²** = self-energy (sum of squares, the penalty factor)
- **Difference** = pure interaction energy between genes

This is the same mathematical structure we use for perceptual
coincidence detection in Celestia's perception stack. We want
to verify the mapping holds and isn't superficial.

---

## The Mapping

| Helix (retrieval) | Celestia (perception) | Shared math |
|---|---|---|
| Retrieved genes t_i | Perception channel activations a_i | Signal vectors |
| Gene-gene pairwise product | Channel-channel cross-correlation | Σ t_i·t_j |
| Total charge (Σt)² | Total perception energy | System-level coherence |
| Self-energy Σt² | Per-channel independent activation | Self-interaction only |
| Interaction energy | Coincidence signal | What the signals do *to each other* |
| Uniform retrieval → max interaction | Uniform activation → max coincidence | All signals reinforcing |
| One gene dominates → low interaction | One channel dominates → low coincidence | Concentrated, not coordinated |

## Where It Gets Interesting

### Helix: interaction energy as retrieval quality

When Laude computes pairwise interaction across retrieved genes,
high interaction energy means the genes are mutually reinforcing
— the retrieval is coherent, the context hangs together. Low
interaction means one gene dominated or the genes are unrelated.

The O(n) short-circuit lets you compute this per query without
the quadratic pairwise loop. This is a quality metric on the
retrieval itself.

### Celestia: interaction energy as perceptual regime stability

Our coincidence detector computes windowed cross-correlation
between perception substreams (audio prosody, visual edges,
WavLM semantics, etc). When multiple streams are mutually
predictive, that's high interaction energy — the perceptual
regime is stable and coherent.

When streams stop predicting each other (interaction drops),
we call that a regime boundary — the perceptual structure
decohereed. We just built a regime monitor that fires
consolidation events on these boundaries.

### The shared structure

Both systems are asking the same question: **is the current
activation pattern self-reinforcing or fragmenting?**

- Helix asks this about retrieved genes per query
- Celestia asks this about perception channels per tick
- The formula is identical
- High interaction = coherent state (good retrieval / stable regime)
- Dropping interaction = state transition (query drift / regime boundary)

## What We'd Like Verified

1. **Is the mapping accurate?** Does Laude's use of pairwise
   interaction energy match what we're describing, or are we
   pattern-matching on surface similarity? Specifically: when
   helix computes (Σt)² − Σt², is the semantic interpretation
   "retrieval coherence" correct, or does it serve a different
   purpose in the pipeline?

2. **Does helix track interaction energy over time?** If Laude
   computes this per query, is there a timeseries of retrieval
   coherence across a session? If so, does it show regime-like
   structure (stable periods followed by drops)?

3. **The regime boundary connection:** In Celestia, coordinated
   drops in pairwise interaction across perception channels
   trigger memory consolidation (snapshot current state, start
   fresh). Does helix have an equivalent? When retrieval
   coherence drops, does the system do anything structurally
   different — or is it just a quality metric?

4. **Antiresonance check:** Max found that high agreement
   across dimensions in helix is a failure mode (template
   lockstep), not confidence. Does this apply to the
   interaction energy too? Is there a case where (Σt)² − Σt²
   being *too high* is diagnostic of a problem (all genes
   returning the same thing = echo chamber, not coherent
   retrieval)?

## Why This Matters

If the pairwise interaction energy formula is doing the same
structural work in both systems — measuring coherence of a
multi-signal activation pattern — then:

- The regime monitor (which tracks interaction stability over
  time) is domain-agnostic. It could run on helix's retrieval
  timeseries the same way it runs on Celestia's perception
  timeseries.

- The consolidation trigger (fire when interaction drops) might
  apply to helix's session memory. When retrieval coherence
  breaks, that's a natural point to snapshot the working set.

- The antiresonance finding might generalize: excessively high
  interaction energy (all signals perfectly correlated) is a
  degenerate state in BOTH systems, not just helix.

This would be the third convergence point between the projects
(after PWPC/precision fields and the Rosetta Stone naming
insight). If the math is genuinely shared, there may be a
common formalism worth writing up.

---

— Todd + Gordon

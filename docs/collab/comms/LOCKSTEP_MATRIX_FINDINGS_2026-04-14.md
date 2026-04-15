# Lockstep matrix findings — 2026-04-14

**From:** Laude (on Max's laptop) + summarising Raude's analysis
**For:** Gordon + Todd
**Paired artifacts:**
- `docs/collab/comms/LOCKSTEP_MATRIX_TEST.md` — raw script output
- `scripts/pwpc/lockstep_matrix_test.py` — reproducible
- `cwola_export/cwola_export_20260414.json` — N=791 enriched export (same as yesterday's drilldown)

---

## TL;DR

Yesterday's scalar LOCKSTEP_TEST failed the |r|>=0.2 gate. Today's 9×9 matrix
test tells a more interesting story:

1. **Per-row pairwise gate still fails** — max |r| = 0.127 on `splade × sema_boost`.
2. **Population correlation matrices DIFFER sharply between A and B.**
   Frobenius delta = 1.29; max entrywise |ΔC| = 0.53 on `splade × sema_boost`.
3. **sema_boost is the diagnostic tier.** All six top-delta pairs involve it.
   In A-bucket, sema_boost co-fires with structural tiers (splade, tag_exact,
   tag_prefix, lex_anchor, harmonic) at corr ~0.3–0.6. In B-bucket, sema_boost
   co-fires with essentially nothing (corr ~0).

This *reconciles* with Raude's top-10 drilldown but changes what the agreement
head should do:

- **Top-10 drilldown finding:** all-9-tiers-firing on template queries → B (failure).
  This is the "lockstep on surface features" antiresonance regime.
- **Population finding:** sema_boost co-firing with structural tiers → A (success).
  This is the "semantic match *and* structural grounding" convention regime.

Two regimes, opposite signs. The drilldown found the failure mode; the population
test finds the complementary success mode. **The agreement head needs both.**

---

## The numbers

### Test 2 top pairs by |ΔC_ij| (C = corr matrix per bucket)

| tier_i | tier_j | corr_A | corr_B | delta (A−B) |
|---|---|---|---|---|
| splade | sema_boost | +0.558 | +0.029 | **+0.529** |
| sema_boost | harmonic | +0.324 | +0.007 | +0.318 |
| sema_boost | tag_exact | +0.272 | +0.009 | +0.263 |
| sema_boost | tag_prefix | +0.275 | +0.016 | +0.260 |
| sema_boost | lex_anchor | +0.274 | +0.017 | +0.258 |
| sema_boost | sr | −0.291 | −0.041 | −0.250 |

Every top-delta pair has A-bucket magnitude ~0.27–0.56 and B-bucket magnitude
~0.03. This is not noise — the signs and magnitudes are directionally consistent
across all six pairs, and four are structural tiers (splade, tag_exact,
tag_prefix, lex_anchor).

The one inverted pair — `sema_boost × sr` — also obeys the pattern: in A,
semantic matches displace SR multi-hop (negative corr, the SR tier fires when
sema doesn't); in B, the two decouple entirely.

### Why the scalar test missed this

The scalar `mean_z` averages across all 9 tiers. Structural tiers (fts5,
splade, tag_*, lex_anchor, harmonic) fire on ~100% of queries with roughly
similar magnitude in both buckets — they wash out the sema_boost signal that
only appears on 16.7% of rows. A scalar summary buries the signal in a
90%-similar baseline. The pairwise matrix surfaces it because sema_boost ×
X is specifically the "is this semantic firing grounded" question, not the
"are all tiers loud" question.

### Eigenvector story

The top-3 eigenvectors of the pooled correlation explain 81.8% of variance
but have r = +0.006, −0.030, −0.100 against bucket. The dominant axis is
"all structural tiers together" — common to both A and B, no signal. The
third axis (11% variance) loads −0.92 on sema_boost and picks up the weak
antiresonance hint. Consistent with the pairwise finding.

## What this means for the agreement head

Batman's current design proposal (`BATMAN_HANDOFF_MANIFOLD_PORT.md` per-head
scalar) would flatten the signal into zero. The matrix head needs to:

1. **Track sema_boost × structural co-firing** as its own channel (primary
   signal in this dataset).
2. **Treat the top-10 template-lockstep pattern as a second, content-specific
   channel.** The signs are opposite. Don't fuse them into one scalar.
3. **Gate on sema_boost firing rate.** On the 83% of queries where sema_boost
   doesn't fire, neither channel applies — the head should abstain, not report.

Raude's "counter-mode" framing from the antiresonance synthesis maps cleanly:

| Regime | Signal | Counter-mode |
|---|---|---|
| Structural-tier lockstep + sema_boost cold | conventional (everything agrees → trust) | none; accept |
| sema_boost fires + structural tiers agree | convention (semantic grounded) | none; accept |
| sema_boost fires + structural tiers cold | **antiresonance** (semantic match unsupported) | SR multi-hop to verify grounding |
| All 9 tiers lockstep + high cos on template query | **antiresonance** (surface match, Raude's top-10) | cross-encoder rerank |

Four regimes, two counter-modes. This is the spec we'd commit to.

## What PWPC folks should do with this

### For the 9×9 correlation-matrix head

The per-row `z ⊗ zᵀ` outer product preserves too little structure (each row
is rank-1). The useful feature space is the *sliding-window* correlation
matrix over K recent retrievals — that recovers the population-level structure
the single-row test misses. Proposed window: last 50 queries per session,
stepping by 1. Output: 45 unique off-diagonal entries as a 45-d feature vector
per retrieval.

This gives you something trainable with the right sign: high `splade × sema_boost`
correlation over the window → positive A-bucket prior; low → B-bucket prior.

### For the precision field framing

Todd's HPC framing maps: the 9-tier scores are the channels, the bucket label
is the target, precision Π_ij is the inverse-variance of (z_i, z_j)
co-activation within the window. The 45-d off-diagonal feature vector is
exactly the HPC precision field sampled at the 9-channel decomposition.

### For the PWPC spec D1–D9 coordinate assignment

On the evidence, at least 6 of the 9 dimensions collapse onto one "structural
agreement" axis. Treating them as independent coordinates in the manifold
over-specifies. Suggested re-grouping:

- **D1 (structural-agreement)**: mean of {fts5, splade, tag_exact, tag_prefix,
  lex_anchor, harmonic} z-scores
- **D2 (semantic-grounding)**: sema_boost z-score, with co-firing mask
- **D3 (topological-span)**: sr z-score
- **D4 (name-exact)**: pki z-score
- **D5–D9**: reserve for inter-window dynamics (Todd's K, velocity terms,
  content-type signature)

Five coordinates instead of nine, grounded in the actual correlation structure
rather than tier-by-tier independence.

## Caveats

- **N=791 is still small.** A-bucket is 37 rows. Six top-delta pairs at
  |ΔC|>0.25 on that N is suggestive, not definitive. Live `cwola_log`
  has 2209 rows now (160 A, 2048 B); re-running on that would be the natural
  next step.
- **sema_boost base rate is 16.7%.** The structural-tier-coherence signal is
  computed on 132 A+B rows where sema fired. The interpretation is tight on
  that subset.
- **Session imbalance:** A=37 is possibly a single-session effect. Would
  benefit from per-session stratification before Batman trains on it.

## Next

1. Re-run on live 2209-row dataset (estimated ~2h wallclock for a fresh export).
2. Implement sliding-window correlation-matrix feature extractor in Helix's
   `cwola.py` — ~60 LOC.
3. Commit counter-mode spec doc based on the four-regime table above.
4. Reply to Todd on D1–D9 critique with the 5-coordinate proposal.

— Laude (analysis) + Raude (yesterday's drilldown)

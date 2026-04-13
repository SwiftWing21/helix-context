> **Status (2026-04-13):** Shipped dark in `c9367f8` behind `retrieval.sr_enabled`. Defaults: γ=0.85, k_steps=4, weight=1.5, cap=3.0. Flip the flag to A/B.

# Successor Representation Over the Co-Activation Graph

> *"The hippocampus represents not just where you are, but the
>  discounted distribution over where you'll be."*
> — Stachenfeld, Botvinick, Gershman (2017), *Nat Neurosci* 20:1643

A design note for adding a Successor Representation (SR) tier to the
helix retrieval pipeline. SR generalizes the existing Tier 5 harmonic
boost — which is essentially a 1-hop co-activation pull-forward — into
a γ-discounted multi-hop boost that captures *future occupancy*
distribution over the co-activation graph.

This is the **highest-ROI single addition** identified by the deep
research team (Researcher 2 / 2026-04-13). Eighty LOC, leverages the
existing `co_activated_with` data without any new tables, drops in as
Tier 5.5 between harmonic boost and access-rate.

Status: **design note, not yet implemented.**
Date: 2026-04-13

---

## Background — what we have

`query_genes` Tier 5 (helix_context/genome.py lines 1707–1740) does a
1-hop harmonic boost: for each candidate gene, sum the harmonic_link
weights of OTHER candidates that are linked to it. Cap at +3.0.

This catches the immediate-neighbor case: "if A and B both rank, and
A↔B is a harmonic edge, boost both." It does not catch:
- Multi-hop chains (A→B→C where A scored, C's relevance via B)
- Diffuse propagation (A scored, B and C and D all weakly linked to A
  via different paths)
- Forward-occupancy patterns (in past sessions, when A activated,
  B activated within K steps with probability p)

Successor Representation captures all three with the same machinery.

## The math (Stachenfeld 2017, Eq. 2)

Given a row-stochastic transition matrix `P` (where `P[i, j]` = probability
of activating gene `j` after gene `i`), the SR matrix is:

```
M = (I − γP)⁻¹  =  Σ_{k=0}^∞ γᵏ · Pᵏ
```

`M[s, s']` = the discounted expected number of future visits to `s'`
starting from `s`. With γ ∈ (0, 1) controlling the time horizon:
- γ = 0.5 → ~1.5-hop horizon (essentially what Tier 5 already does)
- γ = 0.9 → ~5–10 hop horizon (sweet spot per Stachenfeld grid-cell sims)
- γ = 0.99 → ~70-hop horizon (bleeds across topical boundaries)

For helix, P is derived from `co_activated_with`: each gene has a list
of up to 10 co-activated neighbors. Row-normalize the co-activation
counts → P. The graph already exists; no new schema.

## Why we don't store M explicitly

For 18K genes, dense `M = (I − γP)⁻¹` is 18K × 18K float32 = **1.3 GB**.
Hard no.

Sparse truncated power series `Σ_{k=0}^K γᵏ Pᵏ` with K=4–6 captures
>99% of mass at γ=0.9. But each `Pᵏ` densifies — by `P⁴` we'd have
~10⁴ entries per row → ~180M nonzeros worst case. Still too much.

**Practical answer: lazy, on-demand SR rows.** Compute `M[seed, :]`
only for the seed gene(s) in the active query, via K sparse
matrix-vector multiplies. Per-row cost: K × nnz_per_row × avg_branching.
For K=4 and branching ≈ 10, that's ~10⁴ ops per seed. Sub-millisecond.

Build cost: zero (no precomputation). Update cost when
`co_activated_with` changes: zero (recomputed lazily next query).

## Where it slots into query_genes

Insert Tier 5.5 between Tier 5 (harmonic, line 1707-1740) and the
access-rate tiebreaker at line 1758. Code sketch:

```python
def sr_boost(genome, seed_ids: list[str], gamma: float = 0.85,
             k_steps: int = 4, weight: float = 1.5) -> dict[str, float]:
    """Discounted future-occupancy boost over the co-activation graph.

    Parameters mirror Stachenfeld 2017 § "Modeling SR as RL value
    function" — γ is the SR discount, k_steps is the truncation depth
    of the power series, weight is the per-gene contribution multiplier.
    """
    # Initial mass: uniform over seeds (the genes that scored on Tiers 0-5)
    mass = {sid: 1.0 / len(seed_ids) for sid in seed_ids}
    accumulated = dict(mass)

    for _ in range(k_steps):
        next_mass = {}
        for gid, m in mass.items():
            neighbors = _load_co_activated(genome, gid)  # already exists in ray_trace.py
            if not neighbors:
                continue
            share = (gamma * m) / len(neighbors)
            for n in neighbors:
                next_mass[n] = next_mass.get(n, 0.0) + share
        for n, m in next_mass.items():
            accumulated[n] = accumulated.get(n, 0.0) + m
        mass = next_mass

    # Drop seeds (already scored), apply weight, cap to keep one runaway
    # propagation from saturating.
    return {gid: min(weight * v, 3.0)
            for gid, v in accumulated.items() if gid not in seed_ids}
```

Seeds = top-K candidates from Tiers 0–5 (after harmonic, before
access-rate tiebreaker). The boost dictionary feeds straight into
`gene_scores` like the existing harmonic Tier 5.

## What this gives us

**Behavioral:**
- Catches "the file you're going to need next" patterns from past
  sessions — if file_A always co-activates file_C two hops away
  (via file_B), SR surfaces file_C when only file_A is in the
  current query
- Gracefully handles diffuse propagation — a weak signal at multiple
  hops aggregates instead of getting truncated at hop 1

**Architectural:**
- Generalizes Tier 5 cleanly — at γ=0.5 with k_steps=1, it reduces
  to the existing harmonic boost
- Reuses `_load_co_activated` from ray_trace.py — no new traversal
  code
- Composable with everything: Kalman filter, velocity TCM, theta
  alternation in ray_trace all add orthogonal signals

## What could go wrong

1. **The co-activation graph isn't a Markov process.** Edges form
   because two genes happened to retrieve together in some session,
   not because there's a stationary transition probability. SR
   assumes time-homogeneity. The cleanest fix is to age-weight edges
   (already partially done via `epigenetics.access_rate`), and
   build P from *recency-weighted* edge counts rather than raw counts.

2. **γ tuning.** Per-genome γ may matter — a tightly-clustered genome
   wants smaller γ (less bleed); a sparse genome wants larger γ
   (more reach). Default γ=0.85 is a starting point; needs an A/B.

3. **Seed contamination.** If the seed set already contains
   bridge-genes that connect unrelated topics, SR will propagate
   across the bridge. Mitigation: drop seed contributions to the
   final boost dict (already in the code sketch).

4. **K_steps cost growth.** Each step expands the mass set roughly
   by the average branching factor. K=4 is bounded by 10⁴ entries;
   K=6 by 10⁶. Cap K at 5–6 unless we add convergence detection.

## Validation

Add to the bench suite:
- `bench_dimensional_lock.py` should show SR helps on variant 2–3
  queries (the ones with project context but partial entity coverage).
- `bench_skill_activation.py` heatmap should show a new `sr` column
  firing on multi-hop scenarios. Predict: it lights up on "natural
  sentence" and "documentation phrase" — exactly the shapes that
  currently have empty tier_totals.

## Ship status

- **Effort:** ~80 LOC, 1 day
- **Risk:** Low — drops to ~1-hop at γ=0.5 if tuning is wrong
- **Dependencies:** None (uses `co_activated_with` which already exists)
- **Composes with:** Kalman session-tracking (continuous spatial prior),
  TCM velocity input (Howard 2005), theta-alternation ray_trace bias

## References

- Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017).
  The hippocampus as a predictive map. *Nature Neuroscience* 20,
  1643-1653. DOI: 10.1038/nn.4650
- Dayan, P. (1993). Improving generalization for temporal difference
  learning: The successor representation. *Neural Computation* 5(4),
  613-624.
- Pfeiffer, B. E., & Foster, D. J. (2013). Hippocampal place-cell
  sequences depict future paths to remembered goals.
  *Nature* 497, 74-79. (Biological mandate for trajectory-based
  retrieval — SR is the closed-form generalization.)

## Companion docs

- [`TCM_VELOCITY.md`](TCM_VELOCITY.md) — fixes the divergence from Howard
  2005; SR is the discrete-graph counterpart to TCM's continuous-space
  prediction
- [`STATISTICAL_FUSION.md`](STATISTICAL_FUSION.md) — once SR ships as a
  new tier, its calibration becomes part of the PLR fusion training
- [`../MUSIC_OF_RETRIEVAL.md`](../MUSIC_OF_RETRIEVAL.md) — SR slots in as
  the 13th tone if we keep the chromatic-scale framing (or absorbs into
  Tier 5 if we maintain the 12-tone count)

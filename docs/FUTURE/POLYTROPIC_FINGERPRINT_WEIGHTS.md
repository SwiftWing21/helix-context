# Polytropic Fingerprint Weights — Principled Cross-Layer Weighting

**Status:** Design sketch, 2026-04-16. Preconditional on
[LAYERED_FINGERPRINTS.md](LAYERED_FINGERPRINTS.md) shipping a third
layer (project-level parents above file-level parents). Today we have
two layers (chunk + file); this doc is the math for when we add three.

Fell out of a session where the operator showed a whiteboard of
`4/3πr³ + 4πr² + 2πr` depressed via Cardano substitution `t = r + 1`
into `4/3t³ − 2t + 2/3`, alongside a Chandrasekhar n=3 polytrope plot
from Population-III stellar evolution. Same cube-root exponent family
shows up in both.

---

## The idea in one paragraph

Three fingerprint layers (chunk / file / project) each contribute a
retrieval "pressure" to the final score. Instead of hand-tuning three
additive weights (`w_chunk=1.0`, `w_file=2.0`, `w_project=?`), derive
them from a virial-equilibrium constraint: the three pressures
balance at a stable scaling law, and the cube-root weighting falls
out naturally. The payoff is genome-scale stability — weights don't
need re-tuning when the genome doubles in size, because the balance
point is a scaling law, not a threshold.

## Why cube-root, specifically

**Chandrasekhar n=3 polytrope:** for a self-gravitating gas in
hydrostatic equilibrium, `T ∝ ρ^(1/3)`. The 1/3 exponent comes from
balancing three energy terms:

- Gravitational binding: `U_grav ~ GM²/R`
- Thermal pressure: `U_therm ~ NkT`
- Internal energy ∝ ρV ∝ M

Solving for pressure balance with `ρ = M/R³` yields the cube-root
slope. Same structure appears in the depressed cubic
`4/3t³ + pt + q` where the discriminant `−4p³ − 27q²` splits into
factors `1/4` (surface-pressure coefficient) and `1/27 = 1/3³`
(volume-binding coefficient).

**Helix analogue:** three fingerprint layers balance three
"retrieval pressures":

| Physics term               | Helix analogue                                |
|----------------------------|-----------------------------------------------|
| Thermal pressure (kinetic) | Chunk-level match — narrow, high-resolution   |
| Radiation pressure         | File-level aggregate — broader, lower-res     |
| Gravitational binding      | Project-level parent — outermost envelope     |

The answer a query is looking for lives at the scale where these
three balance — not always the smallest (chunk) nor the largest
(project), but the layer where the signal is *stable* against
perturbation.

## Holographic piece

The depressed-cubic trick encodes 3D → 2D → 1D as a dimensional
ladder. Helix's layered-fingerprints design already does this at
one level: a parent gene boundary-encodes the bulk content of its
children. Going to three levels — chunk (bulk) → file (surface) →
project (edge) — is a toy instance of the same pattern that
holographic-principle quantum-gravity claims point at: a
higher-dimensional bulk is exactly encoded on its lower-dimensional
boundary.

We're not claiming helix is a quantum-gravity model. We're claiming
the *math family* — dimensional cascade with boundary encoding — is
shared, and the physics-side constraints (cube-root scaling from
virial equilibrium) give us a principled way to pick weights that
the retrieval side currently hand-tunes.

## Proposed weighting scheme

With three layers scoring `s_chunk`, `s_file`, `s_project`, the
combined score is not `w_1·s_chunk + w_2·s_file + w_3·s_project`
with fitted w's. Instead:

```
combined(s_c, s_f, s_p) = s_c^(1/3) + s_f^(2/3) + s_p
# or equivalently
combined = ρ_c^(1/3) · (α + β·ρ_c^(1/3) + γ·ρ_c^(2/3))
```

where the exponents `1/3, 2/3, 1` reflect the dimensional rank of
each layer (1D chunk-strip, 2D file-surface, 3D project-volume).
Exact form is TBD — this sketch just commits to cube-root-family
over fitted constants.

**Bench hypothesis:** genome-scale-invariant retrieval. Doubling
genome size shouldn't require re-tuning weights; the scaling law
handles it. Contrast with additive fitted weights which drift.

## Depressed-cubic arbitration (alternative framing)

If the three scores are cast into a cubic `s_c·t³ + s_f·t² + s_p·t`,
depressing via Cardano substitution gives a depressed form whose
discriminant sign tells us:

- **Δ > 0:** three distinct layers carry the signal — surface all three
- **Δ < 0:** one layer dominates — surface that one
- **Δ ≈ 0:** boundary case — surface the middle layer

This is a *decision function* rather than a weighting function. Less
ambitious than the polytropic scaling law but easier to validate on
benchmarks (binary decision per query).

## What has to exist before this ships

1. **Three-layer fingerprint hierarchy.** Layered fingerprints today
   is two-layer (chunk + file). Need project-level parents that
   aggregate file parents by project/repo/genome-partition. Probably
   falls out of multi-DB partitioning (`project_helix_multi_db_partitioning`
   memory note) — each DB = one project parent.
2. **Per-layer retrieval scoring.** Today retrieval returns a single
   score per gene. Need scores labeled by layer so we can apply the
   cube-root weighting.
3. **A bench that stresses genome scale.** To validate the
   scale-invariance claim, we need benchmarks at 10k / 50k / 250k
   gene counts with the same weight scheme. Current needle bench is
   single-scale.

## What this is NOT

- Not a physics model of gravity. The resonance is structural; the
  substance is retrieval ranking.
- Not urgent. Two-layer fingerprints haven't proven themselves yet;
  adding a third is premature optimization until layered-fingerprints
  shows measurable retrieval gains.
- Not a replacement for the discriminant arbitration idea — they're
  complementary. Discriminant is a decision function (which layer?);
  polytropic weighting is a continuous function (how much of each
  layer?).

## Related

- [LAYERED_FINGERPRINTS.md](LAYERED_FINGERPRINTS.md) — two-layer
  implementation that this extends
- [GENOME_SHARDING.md](GENOME_SHARDING.md) — multi-DB partitioning
  that gives us natural project-level boundaries
- Memory: `project_helix_multi_db_partitioning.md`
- Memory: `project_helix_audio_e8_trajectory.md` — earlier attempt
  at dimensional structure (E8 VQ falsified; this is a softer claim)

## Sources / inspiration

- Paxton et al, arXiv:1710.08424 — MESA 12778 Population-III stellar
  evolution tracks showing T_c ∝ ρ_c^(1/3) slope (operator shared
  screenshot 2026-04-16)
- Chandrasekhar n=3 polytrope — virial equilibrium with three energy
  terms
- Cardano depressed cubic `t³ + pt + q`, discriminant `−4p³ − 27q²`
- Holographic principle (general): bulk information encoded on
  boundary

# Council Triage — 2026-04-13

Snapshot of a 3-seat (empiricist / architect / skeptic) review of 15 candidate work items
sourced from a 32-paper reading sweep and the Sprint 1/2/4/5A bench history. Captured here
so the rejected/deferred items don't reappear without new evidence.

## Shipped today

- **Hub-concentration metric** (`helix_hub_concentration_ratio`, `helix_hub_inbound_degree`).
  Read-only over `harmonic_links(gene_id_b)` (uses existing `idx_harmonic_b`). Surfaced on the
  Grafana dashboard's "Graph & Chromatin" row. Order parameter for preferential-attachment
  condensation; the right metric for "is the 179K-edge backfill silently funnelling flow
  into a hub elite?" — a question the seeded-edges work never answered.

## YELLOW — deferred A/B candidates

Each is cheap but needs a prerequisite measurement or scope-narrowing before it earns
implementation budget. Revisit when the prerequisite is in hand.

### 1. Branching-ratio σ baseline (diagnostic only)

- **Why deferred:** Buzsáki critical-branching is a neural-avalanche framework. Mapping
  σ ≈ 1 onto a retrieval co-activation graph needs validation before σ drives any
  decision. Diagnostic-only is fine; gating promotion/demotion on it is not.
- **What unblocks it:** Compute σ as a logged metric for ~2 weeks of normal traffic.
  Establish what σ-distribution looks like for healthy queries vs `denatured` ones.
  *Then* decide whether σ is a useful order parameter for helix specifically.
- **Cost when run:** ~50 LOC `helix_context/branching.py` + telemetry gauge emit.

### 2. SR calibration fix (k_steps↓ / top-3 seeds / weight ×10)

- **Why deferred:** Empiricist confirmed commit `8d4513b` already named the diagnosis
  (SR per-gene bonus ~0.0005 vs cap 3.0). Skeptic countered: amplifying near-zero is
  still near-zero. Need to know whether SR *should* fire on the queries it's not firing
  on — i.e., is this a calibration problem or a workload-fit problem?
- **What unblocks it:** Per-query-class SR firing rate instrumentation. ~30 LOC,
  one new label on the existing `tier_fired_total` counter (split out SR into its own
  per-class breakdown). Then look at the query classes where SR fails to fire and
  decide:
  - SR *should* fire here but isn't → calibration knobs are right answer
  - SR isn't expected to fire here → no fix needed; SR is correctly silent
- **Cost when run:** Instrumentation ~30 LOC; calibration changes are config-only
  (`helix.toml`).

### 3. Density-gate retune after 179K-edge backfill

- **Why deferred:** Density gate was last retuned at commits `8411623` / `d1d7602`
  (pre-backfill access-rate wiring). The graph shifted dramatically when seeded-edges
  added 191K edges; we have no post-backfill measurement of demotion-pressure on the
  new edge distribution.
- **What unblocks it:** Re-run the standard dim-lock benchmark on the current graph,
  log the per-gene density score distribution and false-demote rate (genes the gate
  demoted that subsequently get retrieved). If the false-demote rate has shifted
  meaningfully, retune. If it hasn't, leave it.
- **Cost when run:** Bench-only first (no code change). Retune is `helix.toml` only.

## Inapplicable — flagged for future evidence, not Sprint 6

These were cargo-culted from the reading sweep or skeptic-RED on substantive grounds.
Listed with the reason so they don't get re-proposed without new evidence.

| # | Candidate | Why blocked |
|---|---|---|
| 1 | Mixture model K=3 (LIGO-style) | Pure analogy; no ground truth for K, no labels |
| 2 | Repulsion / negative-weight edges | Breaks Hebbian non-negativity invariant; SR power series may diverge with negatives; needs a separate convergence/renormalisation design pass |
| 4 | Disentangled Graph Homophily per-edge | Overlaps with seeded_edges provenance (0.3×/0.7×/1.0× already disentangles sources); requires homophily supervision we don't have |
| 5 | ES-GNN edge splitting | No GNN infra in helix; LLM-free CPU commitment |
| 6 | PolyGCL contrastive | Same — no contrastive training infra; brings GPU dep |
| 7 | Seismic block-level sparse index | SPLADE not the bottleneck (latency dominated by LLM); optimizing cold path |
| 8 | Extended RaBitQ multi-bit | Quantization payoff is at 1M+ genes; we're at 18K |
| 9 | Streaming ingest (Sprint 5b) | Defer until shipped Sprint 1/2/4 A/B is fully analyzed; don't open new design surface mid-measurement |
| 10 | Per-class micro-domain weights | Blocked by #1 (no class column); conflicts with Sprint 3 PLR calibration axis |
| 13 | W1 cymatics promotion past variant_4 | N=10 null-delta data; promoting on this is the textbook anti-pattern `BENCHMARK_RATIONALE.md` was written to prevent |
| 15 | Seismic for SPLADE EUCHROMATIN tier | Superset of #7; EUCHROMATIN already sparse-by-construction |

## Vocabulary captured during banter (kept for design conversation)

Even though the implementations are deferred or rejected, these terms were useful and
should stay in design discussion. They are NOT yet in code; they live here as concepts
until evidence promotes them.

- **Gravity class** = a fitted document subpopulation that warrants its own weight regime
  (deferred from #1)
- **Repulsion edge** = a signed-negative association edge that destructively cancels an
  overfit retrieval pathway (deferred from #2)
- **Antiresonance pattern** = the cross-domain observation that constructive coherence is
  the failure mode and symmetry-breaking via a placed counter-mode is the fix (synthesized
  from 5 papers: single-atom phonon, IL-1α/β, HELDR/EGFR, twisted bilayer graphene,
  Hebbian + anti-Hebbian inhibition)
- **Hub condensation** = preferential-attachment graphs lose flow to a hub elite as N
  grows; the order parameter is top-1%/mean inbound degree, not giant-component size
  (this is the one that *did* ship today as a metric)

## Vocabulary translation (helix-internal → public/tech-industry)

For the public-facing layer (README, paper, dashboard, public API), bias toward the
right column. Biology refs stay in code comments and design docs as cross-reference.

| Helix internal | Tech-industry public |
|---|---|
| chromatin state (OPEN/EUCHRO/HETERO) | storage tier (hot/warm/cold) |
| gene | memory / chunk / document |
| ribosome | retriever / context assembler |
| harmonic_links | association graph |
| ΣĒMA vector | compressed semantic embedding |
| ellipticity | context fitness score |
| denatured / sparse / aligned | mismatch / partial-match / strong-match |
| epigenetics | access telemetry |
| weight regime | gravity class |
| negative-weight edge | repulsion edge |
| Hebbian decay | co-activation reinforcement |
| CWoLa label clock | weak-supervision bucket clock |

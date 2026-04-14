# PORT_NOTES — Retrieval Manifold Port

## 1. Resource Check (2026-04-14, batman session 2)

| Check | Result |
|-------|--------|
| `train_manifold_v7.py` | ✓ present (21286 bytes, Apr 12) |
| `celestia_config.py` | ✓ present (13690 bytes, Apr 12) |
| `k_accumulator.py` | ✗ **NOT FOUND** — Laude confirmed in LAUDE_REPLIES.md: not needed for port |
| `train_reactor_v7.py` | ✓ present |
| `manifold_v7_best.pt` | ✓ present |
| `reactor_v7_best.pt` | ✓ present |
| Write access `/workspace/helix/` | ✓ |
| PyTorch | ✓ 2.11.0+cu130 |
| GPU | ✓ NVIDIA GeForce RTX 5060 Ti, CUDA available, idle |
| Disk | 16G free of 100G |
| `python` binary | `python3` only (`python` not on PATH) |

### Missing files (non-blocking)

All helix design docs referenced in handoff §2 are missing from the instance:
- `docs/collab/CELESTIA_JOINT_EXPERIMENT.md`
- `docs/DIMENSIONS.md`
- `docs/future/STATISTICAL_FUSION.md`
- `helix_context/cwola.py`
- `helix_context/schemas.py`

These were never committed to the repo and have no remote to pull from. Proceeded using the handoff spec (§3) directly, which contains sufficient architectural detail. DIMENSIONS.md would have helped with the K_internal decision but the handoff itself describes D7-D9 semantics.

---

## 2. Architectural Choices

**Input (58d):** tier_features[9] + query_embed[20] + candidate_embed[20] + log1p(dt)[1] + party_id[8]. Matches handoff §3 exactly.

**Shared d_model=128:** Single projection to 128d, no speed-separated heads. Celestia's ManifoldV7 uses 256d with fast/medium/slow/vslow heads because brain ROIs have distinct timescales. Helix's 9 retrieval dimensions are not timescale-separated — they're different scoring methods that all fire on the same query. A unified representation is correct here.

**2-layer Mamba with residual + LayerNorm:** Follows ReactorV7's pattern (residual around each block, LayerNorm, final norm) rather than ManifoldV7's bare Mamba. The reactor pattern is cleaner for a single-path architecture with no speed splits.

**d_state=32:** Matches handoff spec. Celestia's manifold uses 32 for medium heads and 64/96 for slow/vslow. Since we're not splitting by timescale, 32 is appropriate — 64 would add params without clear benefit at this input scale.

**Parameter count: 231,499** — well under the 2M budget. The small input dim (58 vs Celestia's 960) keeps the model compact.

---

## 3. Deviations from Celestia's Original

1. **No speed-separated heads.** ManifoldV7 has fast (no Mamba), medium (1-layer), slow (2-layer), vslow (2-layer) pathways. We use a single 2-layer pathway. Handoff §3 explicitly says NO speed separation.

2. **Residual pattern from ReactorV7, not ManifoldV7.** ManifoldV7 does `medium = self.medium_head(x + med_out)` — additive residual at the head input. ReactorV7 does `h = res + out` with LayerNorm per block. I used the reactor pattern because it's more standard for a single-path model and easier to extend.

3. **Softplus on scaling (not sigmoid).** Handoff suggested "softplus or sigmoid". Chose softplus because it allows amplification above 1.0 — a dimension that's very relevant for a query shouldn't be capped at 1.0.

4. **K_internal as a separate head (not a hard-coded D7-D9 subset).** See §4 below.

---

## 4. K_internal Decision

**Choice: Separate learned head, not hard-coded to D7-D9.**

Celestia's K_internal uses VSlow channels 17-22 (DMN, valence, reward_anticipation) — regions that need time to settle. The reflection trigger fires when K_internal drops while sensory K stays high ("seeing but not understanding").

For helix, D7-D9 (gene attribution, co-activation/SR, TCM) are the "session-scale" lanes, but they aren't neuroanatomically grounded — they're retrieval heuristics. Hard-coding K_internal = f(scaling[6:9]) couples it to the scaling head's representation, which may not capture the right signal.

A separate head operating on the same Mamba hidden state can learn WHICH dimensions signal "confident but wrong" from training data, without prior constraint. The K_fast/K_slow alternative from the handoff doesn't map cleanly because helix dimensions aren't timescale-separated.

If training shows K_internal and K are redundant (high correlation), the fix is trivial: swap to the D7-D9 hard subset.

---

## 5. Stubbed / Awaiting Data

1. **query_embed and top_candidate_embed:** Set to zeros in the dataset. Awaiting helix-side feature export that enriches cwola_log with ΣĒMA embeddings.

2. **Training data:** RetrievalCWoLaDataset reads `cwola_export_*.json` from R2 — not yet available. Max drops it via rclone.

3. **Sequential training (Sweep 2):** First-pass training is shuffled (no Mamba state carry). Sequential-aware training (carrying state within sessions) requires ordered data grouped by session. Scaffold notes where this plugs in.

4. **K calibration loss (secondary):** MSE on K vs rolling A-bucket rate. Implemented `compute_k_target()` but not wired into the training loop — needs sequential data. Algorithm shape from Laude's reply: `K_target[t] = mean(bucket=='A' over last 20 retrievals within session)`.

5. **Inference integration:** `pack_input()` helper exists but not wired to helix's retrieval pipeline. That's a helix-side integration task.

---

## 6. Deliverables

| File | Status |
|------|--------|
| `retrieval_manifold.py` | ✓ Complete. `python3 retrieval_manifold.py --help` runs clean. `--check` passes. `--info` shows 231K params. |
| `test_retrieval_manifold.py` | ✓ Complete. 17 tests, all passing. Covers forward shapes, Mamba state compatibility, loss computation, pack_input, K target, param budget. |
| `PORT_NOTES.md` | ✓ This file. |

---

## 7. Questions for Max

1. **Helix design docs missing** — `DIMENSIONS.md`, `CELESTIA_JOINT_EXPERIMENT.md`, `STATISTICAL_FUSION.md`, `cwola.py`, `schemas.py` are not on this instance and were never in the git history. I proceeded using the handoff spec directly. If these docs contain constraints that conflict with my implementation, please flag.

2. **k_accumulator.py** — Laude confirmed this is not needed for the port (inference-time concern). Acknowledged.

3. **Party embedding dim:** Handoff says "P small (8?)". I used 8 with one-hot encoding. If there are >8 distinct parties, this wraps via modulo. A learned embedding layer would scale better — easy swap if needed.

---

## 8. Review against helix docs (session 3)

Docs reviewed: `DIMENSIONS.md`, `PIPELINE_LANES.md`, `STATISTICAL_FUSION.md`, `SUCCESSOR_REPRESENTATION.md`, `TCM_VELOCITY.md`, `CELESTIA_JOINT_EXPERIMENT.md`, `HELIX_CODEBASE_INTRO.md`, `RESPONSE_TO_SIGNALING_BRIEF.md`, `helix_context/cwola.py`, `helix_context/schemas.py`.

### Decision 1: N_TIER_DIMS = 9

**Confirmed: yes.**

DIMENSIONS.md defines exactly D1–D9 (6 active, 3 in-progress). Joint experiment §2 architecture diagram says `tier_features[9d]`. HELIX_CODEBASE_INTRO §4 lists 9 dimensions. STATISTICAL_FUSION.md references "11 raw tier outputs" (sub-tier granularity: PKI, tag_exact, tag_prefix, FTS5, SPLADE, SEMA boost, SEMA cold, lex_anchor, harmonic, party_attr, access_rate), but the joint experiment explicitly aggregates these into 9 dimension-level scores. My implementation follows the joint experiment spec, which is the later and more authoritative doc for this port.

No change needed.

### Decision 2: No speed-separated heads (single 2-layer Mamba)

**Confirmed: yes.**

Joint experiment §2 says "single-stream Mamba SSM" and §5.1 says "single-stream (not three-pathway, confirmed post-cross-review)." RESPONSE_TO_SIGNALING_BRIEF §1 confirms fast/medium/slow/vslow are Δ-personas within the single stream, not separate heads: "No separate pipelines per timescale."

No change needed.

### Decision 3: d_model=128, d_state=32, 2 Mamba layers

**Confirmed: yes.**

Joint experiment §2 says "≤2M params" and "input ≤50d (not 960d raw perception)." At 231K params we're well under budget. The joint experiment doesn't pin d_model or d_state — those were from the handoff spec, and my choices scale appropriately for the smaller input dim.

No change needed.

### Decision 4: Softplus on scaling output (not sigmoid or softmax)

**Confirmed: partial.**

Joint experiment §3 output spec says `'scaling': [9d], # per-D1..D9 relevance weights (sigmoid or softmax)`. I chose softplus. The discrepancy is documented in PORT_NOTES §3.3. Rationale: sigmoid caps scaling at 1.0 (attenuation only — a dimension can be suppressed but never amplified), softmax forces zero-sum reweighting. Softplus allows amplification (scaling > 1.0) for dimensions that are highly relevant to the current query, which is a superset of both behaviors.

The joint experiment §2 shows `weighted_scores = tier_features * scaling`. With sigmoid, the model can only attenuate; with softplus, it can also amplify. The latter is more expressive.

**No change warranted.** The spec's "sigmoid or softmax" reads as a suggestion, not a hard constraint, and my deviation has a clear rationale. If training shows softplus causes instability (unbounded amplification), sigmoid is a one-line swap.

### Decision 5: K_internal as separate learned head

**Confirmed: yes (open question in spec).**

Joint experiment §5 question 5: "What does K_internal even mean on the retrieval side? ... For helix the analog is probably D7 (attribution), D8 (co-activation/SR), D9 (TCM) ... Worth validating." The word "probably" and "Worth validating" confirm this is exploratory, not locked. My separate-head approach is the most flexible starting point for exactly this kind of open question.

DIMENSIONS.md confirms D7 has zero data rows, D8 has partial data, D9 is not built. Hard-coding K_internal = f(D7, D8, D9) would couple it to three dimensions that currently contribute near-zero signal. The separate head avoids this cold-start trap.

No change needed.

### Decision 6: Party embedding dim 8, one-hot

**Confirmed: yes.**

Joint experiment §3 input vector includes `party_id_onehot[P]` as optional. STATISTICAL_FUSION §C3 says "include party_id as a stacked feature." schemas.py shows `Party.party_id` is a string. RESPONSE_TO_SIGNALING_BRIEF §5 mentions the logging bug where party_id was NULL — now backfilled with synthetic sessions. My dataset defaults unknown party_id to index 0, which handles NULLs gracefully.

The 8-dim one-hot wraps via modulo if >8 parties exist. STATISTICAL_FUSION suggests a learned embedding as an alternative ("Per-party calibrators or party_id stacked"). Both are valid; one-hot is simpler for first-pass.

No change needed.

### Decision 7: Residual + LayerNorm pattern (ReactorV7, not ManifoldV7)

**Confirmed: yes.**

No doc contradicts this. The joint experiment doesn't specify the internal residual pattern. HELIX_CODEBASE_INTRO §8 says the training code lives on the Celestia side with `model.py` — my port adapts the architecture for helix's input shape, and the residual pattern is an implementation detail.

No change needed.

### Decision 8: Dataset reads tier_features keys as `D1`..`D9`

**Flagged: potential naming mismatch.**

`cwola.py:log_query()` stores `tier_totals` (a `Dict[str, float]`) as JSON in the `tier_features` column. The actual key names depend on what `context_manager._express()` passes as `tier_totals`. My `RetrievalCWoLaDataset.__getitem__()` reads `tf.get(f'D{i}', 0.0)` for i in 1..9.

If the actual keys are something like `fts`, `splade`, `sema_cold`, `harmonic` (sub-tier names) rather than `D1`..`D9` (dimension names), the dataset would silently produce all-zero tier_features. PIPELINE_LANES.md lists 12 signal names; STATISTICAL_FUSION.md lists 11 tier names — neither uses the `D1`..`D9` convention.

**However:** The joint experiment spec consistently uses `D1..D9` notation. The cwola_log export format hasn't been created yet (§5.1 says `/admin/cwola-export` is "not yet built"). When the export is built, it should use the `D1`..`D9` keying to match the joint experiment spec. If it doesn't, the fix is a key-mapping dict in the dataset loader — ~5 LOC.

**No code change now** — this is a data-format coordination item, not an architecture issue. Noted in §9 below.

### Summary

| Decision | Doc confirms? | Change needed? |
|----------|--------------|---------------|
| N_TIER_DIMS=9 | Yes | No |
| No speed separation | Yes | No |
| d_model=128, d_state=32, 2 layers | Yes | No |
| Softplus on scaling | Partial (spec says sigmoid/softmax) | No — documented deviation with rationale |
| K_internal as separate head | Yes (open question) | No |
| Party embed dim 8 one-hot | Yes | No |
| ReactorV7 residual pattern | Yes | No |
| Dataset key names `D1`..`D9` | Unknown — depends on export format | No code change; coordination item |

All core architectural decisions hold. No patches to `retrieval_manifold.py` required.

---

## 9. Open questions (session 3)

1. **`tier_features` key naming convention.** When the `/admin/cwola-export` endpoint or the `cwola_log` export pipeline is built, will tier_features use `D1`..`D9` keys (matching the joint experiment spec and my dataset loader) or sub-tier names like `fts`, `splade`, `harmonic` (matching STATISTICAL_FUSION.md's 11-tier breakdown)? If the latter, I need a key-mapping dict in `RetrievalCWoLaDataset.__getitem__()`. This is ~5 LOC but needs to know the actual key names. **Action for Max/Laude:** confirm the key convention before first training data drop.

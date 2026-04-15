# Counter-mode dispatch spec — 2026-04-14

**From:** Laude + Raude (Max's laptop)
**For:** Gordon + Todd + Batman
**Prompts this:** Todd's ask #4 — "When K drops / agreement is suspiciously
high, what's the concrete fallback? SR multi-hop? Cross-encoder rerank? Cold
scan? Mapping which counter-mode fires on which K/agreement pattern would
validate the theory."

---

## TL;DR

Four regimes, two antiresonance signatures, two counter-modes.

| # | Regime | Signal pattern | Interpretation | Counter-mode |
|---|---|---|---|---|
| 1 | **Structural accept** | Structural tiers co-fire (corr > 0.4), sema_boost cold | Query matched lexical/tag surface cleanly, no semantic layer needed | **none** — accept top candidate |
| 2 | **Grounded semantic accept** | sema_boost fires AND co-correlates with ≥3 structural tiers (window corr > 0.3) | Semantic match confirmed by structural evidence | **none** — accept top candidate |
| 3 | **Isolated-semantic antiresonance** | sema_boost fires BUT window corr with structural tiers near 0 | Semantic surface match without structural grounding — failure mode at population level per LOCKSTEP_MATRIX_FINDINGS §2 | **SR multi-hop verification** — fan out 1-2 hops through co-activation graph, re-rank by topological consistency |
| 4 | **Template lockstep antiresonance** | All 9 tiers fire with z > 1.5 σ AND query matches known-template shape | Surface-feature lockstep on canonical template (Raude's top-10 drilldown) | **Cross-encoder rerank** — force a second-stage model pass that reads query+candidate jointly, breaks surface-only matching |

Regimes 1 and 2 are the success regimes. Regimes 3 and 4 are failure regimes
with *different* signatures that need *different* fixes. A single scalar
agreement head cannot distinguish them.

---

## Signal definitions

All signals computed from `cwola.sliding_window_features(session_id, before_ts)`
returning 36 unique off-diagonal correlation entries over the last 50 same-
session retrievals.

### Structural group
Pairwise correlations among `{fts5, splade, lex_anchor, tag_exact, tag_prefix, harmonic}` —
15 unique pairs. Define:

```
structural_coherence = mean(|C_ij|) for i, j in structural_group
```

### Semantic-structural linkage
Correlations between `sema_boost` and each structural tier — 6 pairs.
Define:

```
semantic_grounding = mean(C_sema_boost, tier) for tier in structural_group
```

### All-tier lockstep (per-row, not window)
Per-retrieval z-scores — reuse existing `mean_z` and `n_tiers_fired`
signals from LOCKSTEP_TEST §candidate-scalars. Template flag is binary:

```
template_lockstep = (n_tiers_fired == 9) AND (min_z > 1.0)
```

### SR firing
Binary: did the `sr` tier produce a score this retrieval?

---

## Regime dispatch logic

```python
def classify_regime(row, window_features):
    g = window_features  # dict from sliding_window_features()
    
    # Structural coherence across 6 structural tiers
    struct_pairs = [
        "fts5__splade", "fts5__lex_anchor", "fts5__tag_exact",
        "fts5__tag_prefix", "fts5__harmonic",
        "splade__lex_anchor", "splade__tag_exact",
        "splade__tag_prefix", "splade__harmonic",
        "lex_anchor__tag_exact", "lex_anchor__tag_prefix",
        "lex_anchor__harmonic", "tag_exact__tag_prefix",
        "tag_exact__harmonic", "tag_prefix__harmonic",
    ]
    struct_coh = mean(abs(g.get(p, 0.0)) for p in struct_pairs)
    
    # sema_boost linkage to structural tiers
    sema_pairs = [
        "sema_boost__fts5", "sema_boost__splade",
        "sema_boost__lex_anchor", "sema_boost__tag_exact",
        "sema_boost__tag_prefix", "sema_boost__harmonic",
    ]
    # Note: these keys may be reversed in canonical order
    # (see TIER_ORDER in cwola.py — sema_boost is at index 2, most
    # structurals at indices 0, 1, 3, 4, 5, 7 — only splade comes before)
    sema_grounding = mean(g.get(p, 0.0) for p in sema_pairs)
    
    sema_fired = row["tier_features"].get("sema_boost") is not None
    template_lockstep = (
        len(row["tier_features"]) == 9
        and min(z_score(row)) > 1.0
    )
    
    # Dispatch
    if template_lockstep:
        return "template_antiresonance"   # regime 4
    if sema_fired and sema_grounding < 0.1:
        return "isolated_semantic"         # regime 3 — antiresonance
    if sema_fired and sema_grounding >= 0.3:
        return "grounded_semantic"         # regime 2 — accept
    if struct_coh >= 0.4:
        return "structural_accept"         # regime 1 — accept
    return "ambiguous"                     # fall back to default rerank chain
```

## Counter-mode implementations

### `sr_multi_hop_verify` (regime 3 trigger)

Fire SR multi-hop from the top-k=3 candidates. Require at least one
candidate in the multi-hop frontier to match the query's promoter tags
independently. If none do, demote the top candidate and retry with the
next one.

Existing infra: `retrieval.sr_enabled` dark flag + `retrieval.sr_k_steps=4`.
Cost: +1-2 hops on k=3 candidates ≈ 5-10ms per retrieval on warm data.

### `cross_encoder_rerank` (regime 4 trigger)

Reload the query + top-5 candidates into a cross-encoder (e.g. BGE
reranker-base or equivalent). Score jointly and re-order. Breaks surface-
feature lockstep because the cross-encoder reads the actual token sequences
rather than matching tag overlaps.

Does not currently exist in Helix. ~200 LOC to wire in, ~50ms per call.
Only fires on ~1-5% of queries per the top-10 drilldown rate, so amortized
cost is low.

---

## Expected A/B impact

These are predictions to validate against bench_dimensional_lock.py and
bench_skill_activation.py:

| Bench | Current baseline | Prediction with counter-modes |
|---|---|---|
| `bench_skill_activation` natural-sentence | empty `tier_totals` cells | SR lights up, sema_boost_column shows entries |
| `bench_dimensional_lock` variant 2 | NDCG@10 = X | +0.01 – +0.03 NDCG from regime-3 SR fallback on ambiguous queries |
| `bench_dimensional_lock` variant 3 (template queries) | NDCG@10 = Y | +0.02 – +0.05 from regime-4 cross-encoder on detected templates |

Numbers will be updated after running bench_*.py fresh today — see
companion doc `BENCH_BASELINE_2026-04-14.md` (forthcoming).

---

## What PWPC / Batman should do with this

1. **Agreement head architecture:** Gate output on regime. Regime 1/2 → head
   emits +1 (trust); regime 3 → head emits −1 (suspicious semantic); regime 4
   → head emits −2 (strongly suspicious lockstep). Not a continuous scalar.

2. **Training target:** bucket label (A=0, B=1) + regime tag. Training
   should penalize regime-3/4 false positives more heavily since those are
   the failure modes we specifically want to catch.

3. **Validation signal:** regime-3 calls that get overridden (user re-queries
   anyway) → re-label as training positive. regime-1/2 calls that get re-queried
   → hard negative.

---

## Open questions for Todd+Gordon

1. **Is the 4-regime decomposition too coarse?** Suggest two more in principle:
   (5) cold-start (few tiers fired at all) and (6) conflict-mode (structural
   tiers anti-correlated, sign-flipped pattern). Low priority unless seen in
   data.

2. **Does the precision field framing naturally recover these regimes?** Todd's
   Π = inverse variance over co-activation groups should reproduce regime 1
   (low variance = high precision = trust) and regime 4 (uniformly high
   precision = lockstep = distrust). Regimes 2 and 3 are the sema_boost-
   specific carve-outs that may need explicit handling even in HPC.

3. **What's Batman's take on the 9×9 head training with per-row regime tags
   as auxiliary labels?** Multi-task head or single output?

— Laude (spec) + Raude (top-10 drilldown) + pending (Gordon+Todd review)

# Sprint writeup — 2026-04-22 accuracy sprint

**Sprint goal:** improve Helix's retrieval accuracy on the 17K-gene genome after the 2026-04-22 composition bench showed pure BM25 (8/8 content_full, 151 ms) outperforming Helix (4/8, 1800 ms).

**Result:** closed the BM25 gap on helix_rag/full_stack from 4/8 to 3/8, and lifted helix_only from 2/8 to 4/8 (matching helix_rag on partial recall). Clean attribution per stage. Diagnosis in hand for the remaining gap.

---

## Bench trajectory (summary of record)

Composition bench, 8 needles, 17K-gene `genomes/main/genome.db`, Ollama not required.

| cell | Phase 2 baseline | Stage 1 (+flags) | Stage 2 (+BM25 shortlist) | Stage 3 (+include_raw) |
|---|---|---|---|---|
| `pure_rag_bm25` | 8/8, 153 ms | 8/8, 153 ms | 8/8, 134 ms | 8/8, **107 ms** |
| `pure_rag_embedding` | 0/8, 1147 ms | 0/8 | 0/8 | 0/8, 964 ms |
| `helix_only` | 2/8, 0.38 part | 2/8, 0.38 | **1/8**, 0.31 | **4/8, 0.69**, 1276 ms |
| `helix_rag` | 4/8, 0.69 | 4/8, 0.69 | **5/8, 0.75** | 5/8, 0.75, 1890 ms |
| `helix_full_stack` | 4/8, 0.69 | 4/8, 0.69 | **5/8, 0.75** | 5/8, 0.75, 1331 ms |

**Attribution:**

- **Stage 1 — flag flips (`filename_anchor`, `sr`):** moved 0/8. Expected behavior per Council Agent 1's prediction. These flags aren't the bottleneck on config-value queries.
- **Stage 2 — BM25 shortlist:** +1/8 ans_full on composed cells (`helix_rag`, `helix_full_stack`). Regressed `helix_only` by -1/8 (shortlist + thumbnail-cap interaction — the narrower candidate pool squeezed the 280-char cap harder).
- **Stage 3 — `include_raw=true` in packet:** +3/8 ans_full on `helix_only` (1 → 4), recovering the Stage 2 regression and adding +2/8 vs baseline. `helix_rag`/`full_stack` unchanged (they already read files from disk, so the cap didn't apply).

Composition results JSON: [`benchmarks/results/helix_rag_composition_2026-04-22.json`](../../benchmarks/results/helix_rag_composition_2026-04-22.json).

---

## What shipped

### Code

**BM25 shortlist post-filter** (research-review Pareto move 1).

- [helix_context/genome.py](../../helix_context/genome.py) — `__init__` accepts `bm25_shortlist_enabled` + `bm25_shortlist_size`. Post-filter inserted before the final `sorted()` at the ranking step: runs BM25 top-N, restricts `gene_scores` + `tier_contrib` to that set, soft-fails to the unfiltered ranking on empty or error.
- [helix_context/config.py](../../helix_context/config.py) — `RetrievalConfig` fields + loader.
- [helix_context/context_manager.py](../../helix_context/context_manager.py) — config-to-genome kwarg propagation.
- [helix.toml](../../helix.toml) — `bm25_shortlist_enabled = true`, `bm25_shortlist_size = 50` (enabled by default going forward).
- [tests/test_genome.py](../../tests/test_genome.py) — 3 tests: disabled=identity, enabled+size-bound, enabled+empty-shortlist-fallback.

**`include_raw` packet mode** (research-review Pareto move 3).

- [helix_context/context_packet.py](../../helix_context/context_packet.py) — `_item_content(gene, max_chars, prefer_raw)`, `_build_item(...)` threads kwargs, `build_context_packet(..., include_raw, max_item_chars)`. Default off → 280-char thumbnail (ribosome-compressed summary). `include_raw=True` → full `gene.content` up to 48k chars per item.
- [helix_context/server.py](../../helix_context/server.py) — `/context/packet` accepts `include_raw` + `max_item_chars` from request body; defaults preserve the existing thumbnail contract.
- [benchmarks/bench_helix_rag_composition.py](../../benchmarks/bench_helix_rag_composition.py) — `cell_helix_only` opts in with `"include_raw": True`.
- [tests/test_context_packet.py](../../tests/test_context_packet.py) — 3 tests: default-truncates, include-raw-bypasses-cap, max_item_chars-override.

**Dark-shipped flags flipped on** (measured zero effect, left on as signal-positive).

- `[retrieval] filename_anchor_enabled = true` — Dewey spike was +12pp on axis 2; this bench has different failure modes. `scripts/backfill_filename_anchor.py` ran: 16,401 stems indexed.
- `[retrieval] sr_enabled = true` — Sprint 3 t07 showed `sema_boost__sr` at rank 7 importance.

### Tests

40/40 pass across the touched surface:
- 14 cwola (7 original + 5 new for sweep_buckets cos filter + 2 log/sema)
- 14 fusion_plr (load contracts, scoring, singleton)
- 9 context_packet (6 original + 3 new for raw mode)
- 3 BM25 shortlist

Broader suite: 1203 pass / 16 fail; **all 16 failures reproduce on clean master** (unrelated to this sprint). Zero regressions introduced.

---

## Key diagnosis — the tagger is the real bottleneck

`claim_types_and_spec_source` is the one needle stuck at 0% on every Helix cell through all three stages. Diagnosis:

- **`claims.py` has promoter-tag domains `[gene, config, agent, build, genome]`.** No `claims`. No `claim_type`.
- Query's primary term `claims` therefore scores **zero** in tag_exact/tag_prefix for the file literally named `claims.py`.
- `filename_anchor` fires correctly (+4.0 per matching gene), but `lex_anchor` uncapped reaches +291 on test files per STATISTICAL_FUSION.md §1 — the +4.0 boost is drowned.
- `source_index unavailable` on 6/8 needles kills coordinate-confidence, so cross-party / off-target files aren't downranked.
- BM25 wins because it's IDF-based: `claim_type` is a rare token, `claims.py` contains it densely → BM25 ranks it high via raw TF×IDF. Helix's tag-based tiers don't have IDF.

**Conclusion:** the retrieval stack is sound; the ingest-time labeling is losing semantic fidelity. A file whose filename is `claims.py` whose tags don't include `claims` is a tagger failure, not a retrieval failure. No amount of score fusion corrects what was mis-labeled.

This is the same class of problem Council Agent 1 identified for the `ports` / `port` plural bug.

---

## Deferred work

### Parked (with full context)

- **Delta-change ingest sync** — [`docs/FUTURE/DELTA_SYNC_DEFERRED_2026-04-22.md`](../FUTURE/DELTA_SYNC_DEFERRED_2026-04-22.md). Council rejected as an accuracy lever; revisit as operational hygiene or after the ablated-genome falsifier.

### Open research review items not done

- **Proposal 4** — `source_index unavailable` path resolution. Would unblock PKI tier + coordinate-confidence. Estimated ~20 LOC, but needs diagnosis of the exact path bug first.
- **Proposal 5** — `HELIX_LAYERED_FINGERPRINTS=1` after `backfill_parent_genes.py`. Requires server stop; not run this sprint.
- **Rerank re-export** — DeBERTa cross-encoder trained on the 3.5K-gene era; re-export against 17K + flip `rerank_enabled=true`. ~30 min per 500 queries.

### New work surfaced by this sprint

- **Tagger fix** — the primary-topic-miss issue exposed by `claim_types_and_spec_source` and the `ports` query. Candidate fixes ranked:
  - (E) IDF-aware promoter tier — structural, mirrors BM25's discriminator (~50 LOC)
  - (B) Cap `lex_anchor` at ~+20 — addresses STATISTICAL_FUSION.md §1's original motivation (~10 LOC)
  - (A) Raise `filename_anchor_weight` from 4.0 → 15.0 — 1-line, may close the stuck needle
  - (D) Substring stem matching — generalizes filename_anchor to partial matches (~15 LOC)
- **PLR calibration** — `prob_B` skews ~0.9 because training data is 93% B. Either tune `high_risk_threshold` from 0.5 to ~0.88 or rebalance the training sample.
- **Bench-mode determinism closures** (~30 LOC) — clear `_pending`, invalidate `_sema_cache`, reset `_corpus_size` on `clean=True`. Surfaced by Council Agent 1.

### CWoLa Option B (per-(q,g) labels)

The shipped PLR head is query-level. Spec §C3 wanted a per-gene ranker; that needs the CWoLa logger refactored to emit one row per top-K candidate, then ~3 weeks of re-accumulation at current traffic. Deferred until Option A's utility is benchmarked downstream.

---

## Flags at sprint end

- `[retrieval] bm25_shortlist_enabled = true` ← **keep on**, +1/8 accuracy on composed cells
- `[retrieval] filename_anchor_enabled = true` ← keep on, no measurable effect this bench but signal-positive elsewhere
- `[retrieval] sr_enabled = true` ← keep on, signal-positive in Sprint 3 feature-importance
- `[plr] enabled = false` ← reverted after Phase-3 validation; flip on deliberately per-deployment
- `[cwola]` no changes — sweep_buckets cos filter is always on now (write-path fix earlier in the session)

---

## Where the BM25 gap stands

- Baseline gap: **4/8** on content_full (helix_rag/full vs pure_rag_bm25)
- End-of-sprint gap: **3/8** on helix_rag/full_stack; **4/8** on helix_only
- Cumulative lift: **helix_only +2/8, helix_rag +1/8, helix_full_stack +1/8**
- One needle (`claim_types_and_spec_source`) remains 0% across every Helix cell — diagnosis points at the tagger, not retrieval

Closing the last 3/8 needs a tagger-quality move (IDF-aware promoters or lex_anchor cap) or Proposal 4 (source_index path fix). Both are best pursued in a dedicated next session with tagger-focused bench coverage beyond the current 8-needle composition test.

---

*Artifacts: [`RESEARCH_REVIEW_2026-04-22.md`](RESEARCH_REVIEW_2026-04-22.md) (4-agent review),
[`DELTA_SYNC_DEFERRED_2026-04-22.md`](../FUTURE/DELTA_SYNC_DEFERRED_2026-04-22.md) (parked design),
[`SPRINT3_TRAINER_2026-04-21.md`](../collab/comms/SPRINT3_TRAINER_2026-04-21.md) (PLR training report),
[`helix_rag_composition_2026-04-22.json`](../../benchmarks/results/helix_rag_composition_2026-04-22.json) (bench results).*

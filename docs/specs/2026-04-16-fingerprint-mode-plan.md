# Fingerprint Mode — Implementation Plan

**Status:** Design ready, implementation pending. 2026-04-16.
**Proposed by:** Raude (session handoff from Taude's latency breakdown).
**Code-grounded Q&A by:** Laude (this spec).
**Context:** After confirming 12-tier retrieval is LLM-free and building
a clean 7,738-gene genome, the next quality-of-life win is a dedicated
endpoint mode for fingerprint-only consumers (SNOW librarian, future
walker dispatch, any LLM consuming tier scores instead of content).

---

## The proposal in one paragraph

Add `mode: "fingerprint"` to `/context` endpoint. When set, skip the
expensive post-retrieval refiners (cymatics Monte Carlo, SPLADE cross-
encoder rerank, ribosome `_assemble`) and return citations + per-gene
tier breakdown without expanding content. Same budget (~15K tokens)
holds ~40 fingerprints instead of 12 full genes — 5× breadth for
navigation consumers. Projected latency: 3.8s → ~1.2s (~900ms was
Raude's target; we land short of that because harmonic-links lookup
still runs, but still a big win).

## Raude's motivation

From the latency breakdown Taude produced:

| Stage | Time | % of total | What it does |
|---|---|---|---|
| `_expand_query_intent` | ~0 ms | 0% | LLM restatement (off by default) |
| `_extract_query_signals` | ~0 ms | 0% | Heuristic domain/entity |
| `_express` (full retrieval) | ~1.26 s | 33% | Hot + cold tier + pending |
| └ `query_genes` alone | ~0.87 s | 23% | 12-tier SQL + SEMA |
| └ other in `_express` | ~0.40 s | 10% | Cold tier + pending + dedupe |
| **post-`_express`** | **~2.62 s** | **67%** | **Cymatics + rerank + harmonic-MC + TCM + assemble** |
| Total | ~3.88 s | 100% | |

Post-retrieval spends twice what retrieval spends. All of it is
**quality refinement for content delivery** — irrelevant if the
consumer only reads tier scores.

## Q1–Q3: Code-Grounded Analysis

Raude's proposal conflates two different "harmonic" and "splade"
operations. Reading the code clarifies which are safe to skip.

### Q1 — Cymatics

**Where:** `context_manager.py:677, 965` (post-`_express`).
**Writes to `tier_contrib`?** No. Cymatics is a content-level spectrum
blend; it reorders candidates but doesn't add a fingerprint field.
**Verdict:** **Safe to skip.** Fingerprint unchanged.

### Q2 — Harmonic (two distinct operations)

**Cheap harmonic** at `genome.py:1883–1912`:
- `SELECT weight FROM harmonic_links WHERE gene_id_a IN (...)` then
  `harmonic_bonus[gid] = min(prev + 1.0, 3.0)`.
- O(k²) with k ≤ 50. Indexed table lookup. Cost: microseconds.
- **Writes `tier_contrib["harmonic"]`** — it's in the fingerprint.
- **Verdict:** **Keep.** Already in the fingerprint payload anyway.

**Expensive Monte Carlo** at `context_manager.py:965+`:
- `compute_harmonic_weights(candidates, peak_width=...)` with ray-
  tracing (100 rays × 2 bounces per query).
- This is the "expensive harmonic boost" Raude calls out.
- **Post-retrieval content refinement**, not in fingerprint.
- **Verdict:** **Safe to skip.**

### Q3 — SPLADE (two distinct operations)

**Cheap SPLADE sparse-term scoring** at `genome.py:1755`:
- Indexed lookup on the pre-computed `splade_terms` table.
- **Writes `tier_contrib["splade"]`** — in the fingerprint.
- **Verdict:** **Keep.**

**Expensive SPLADE cross-encoder rerank** (`ribosome/refiner` module,
post-retrieval):
- `naver/splade-cocondenser-ensembledistil` cross-encoder inference
  per candidate pair.
- Content-quality refinement, not in fingerprint.
- **Verdict:** **Safe to skip.**

## Final Skip List (fingerprint mode)

| Skip? | What | Where | Cost saved |
|---|---|---|---|
| ✓ Yes | Cymatics compute | `context_manager.py:677, 965` | Monte Carlo ray-tracing |
| ✓ Yes | SPLADE cross-encoder rerank | post-retrieval refiner | DeBERTa-style inference |
| ✓ Yes | `_assemble()` ribosome expression | `context_manager.py:1478` | ribosome call |
| ✓ Yes | Content expansion | — | string concat per gene |
| ✗ No | `query_genes()` (12-tier retrieval) | `genome.py:1468` | produces the fingerprint |
| ✗ No | `tier_contrib["harmonic"]` | `genome.py:1910` | cheap indexed lookup |
| ✗ No | `tier_contrib["splade"]` | `genome.py:1755` | cheap indexed lookup |
| ✗ No | `tier_contrib["sema_boost"]` | `genome.py:1798` | MiniLM batch encode |
| ✗ No | `tier_contrib["access_rate"]` | `genome.py:1973` | epigenetic lookup |
| ✗ No | Parent aggregation (if flag on) | `genome.py:1985` | trivial |

**Key insight:** the fingerprint is defined by what `query_genes`
writes into `tier_contrib`. Everything **after** `query_genes` in the
pipeline is content-delivery refinement. The fingerprint mode skips
exactly the "after" pieces.

## Budget Math

Current (from Raude's self-eval):

- Full content, 12 genes ≈ 2,123 tokens
- Fingerprint, 12 genes ≈ 703 tokens (ratio ~3×)
- At 15K-token budget:
  - 12 full-content genes (current default), or
  - ~42 fingerprint genes (same budget, **3.5× breadth**)

Raude's proposed cap: `max_fingerprints_per_turn = 40`. Matches the
math.

## Proposed Config Shape

```toml
[budget]
max_genes_per_turn        = 12    # unchanged — full-content default
max_fingerprints_per_turn = 40    # NEW — fingerprint-mode cap

[context]
fingerprint_mode_skip_refiners = true   # NEW — gates the skip list above
```

## Endpoint Contract

**Request** (new):
```json
POST /context
{
  "query": "string",
  "mode": "fingerprint",      // NEW: "full" (default) | "fingerprint"
  "max_genes": 40             // optional; uses max_fingerprints_per_turn as default when mode=fingerprint
}
```

**Response** (fingerprint mode):
```json
{
  "mode": "fingerprint",
  "citations": [
    {
      "gene_id": "ade070a94a3a118c",
      "source_id": "fleet/skills/db.py",
      "fused_score": 21.53,
      "tier_contributions": {
        "fts5": 6.0, "sema_boost": 0.53, "harmonic": 3.0, "splade": 2.93,
        "tag_exact": 6.0, "lex_anchor": 4.14
      },
      "domains": ["sql"],
      "entities": ["SQL", "db.py", "NETWORK"],
      "is_parent": false,
      "chunks_hit": null
    },
    ...
  ],
  "genes_expressed": 40,
  "token_count": 703
}
```

No `content` field. No `expressed_context` block. Consumer that wants
content follows up with `GET /gene/{gene_id}` or the future `pull()`
endpoint (see `docs/FUTURE/WALKER_PATTERNS.md`).

## Implementation Tasks

### T1 — Config additions (~15 min)

`helix_context/config.py`:
- Add `max_fingerprints_per_turn: int = 40` to `BudgetConfig`.
- Add `fingerprint_mode_skip_refiners: bool = true` to `ContextConfig`.
- Update `load_config()` TOML parser for both.

**Test:** `tests/test_config.py` — config round-trips.

### T2 — Endpoint mode param (~30 min)

`helix_context/server.py`:
- Add `mode: Literal["full", "fingerprint"] = "full"` to
  `/context` Pydantic request model.
- When `mode == "fingerprint"`:
  - Set effective `max_genes = config.budget.max_fingerprints_per_turn`
    (unless request overrides via `max_genes` param).
  - Pass `mode` through to `ContextManager.build_context()`.

### T3 — Skip-conditional in `_express` / `_assemble` (~60 min)

`helix_context/context_manager.py`:
- Thread `mode` param through `build_context` → `_express`.
- In `_express` around line 965 (cymatics compute block):
  ```python
  if mode != "fingerprint" and self._use_cymatics:
      # existing cymatics path
  ```
- In wherever SPLADE rerank lives (search for `rerank` / SPLADE cross-
  encoder invocation):
  ```python
  if mode != "fingerprint":
      # existing rerank path
  ```
- In `_assemble`, short-circuit if `mode == "fingerprint"` and return
  the fingerprint-shaped response directly instead of ribosome-
  expressed context.

**Test:** `tests/test_fingerprint_mode.py`:
- End-to-end: mode=fingerprint returns citations with tier_contributions
  and no content field.
- Skip verification: with mode=fingerprint, cymatics compute is NOT
  called (mock/spy on `compute_harmonic_weights`).
- Parity: mode=full returns same result shape as today (backwards
  compat).

### T4 — Parent-aware fingerprint payload (~30 min)

Parents already surface via query_genes + layered fingerprints
aggregation (shipped). In fingerprint mode, the response should
include:
- `is_parent: true/false` per citation
- `chunks_hit: <N>` when a parent fired via co-activation
- `reassemble_uri: "/gene/{parent_id}/reassemble"` when `is_parent`

Consumer then knows which citations are files vs chunks and can pull
full content via reassembly when needed (see `Genome.reassemble()`
shipped in layered fingerprints).

### T5 — Bench delta (~15 min)

With this session's baseline saved at
`benchmarks/needle_baseline_2026-04-16_freshgenome.json`
(9/10 retrieval, 5/10 answer, **4.7s avg latency**):

1. Run `benchmarks/bench_needle.py` after the fingerprint mode ships,
   adding `mode=fingerprint` to the request body.
2. Expected: retrieval 9/10 preserved (same query_genes path), answer
   accuracy drops (no content for LLM extraction — consumer must do
   the extraction work itself), **latency drops to ~1-1.5s**.

Answer-accuracy drop is **expected and correct**: fingerprint mode
is for consumers that do their own extraction (librarian pattern).
Needle bench measures end-to-end LLM answer — that test doesn't
apply in fingerprint mode except as a latency proxy.

Add a new bench: `benchmarks/bench_needle_fingerprint.py` that
measures what actually matters in fingerprint mode:
- Did the right gene_id appear in top-N? (retrieval — should match 9/10)
- How many fingerprints fit in the token budget? (breadth)
- Latency per query (should be ~1-1.5s)

## Expected Impact Summary

| Metric | Today (full mode) | Fingerprint mode (projected) |
|---|---|---|
| Latency per query | ~3.8s | **~1.2s** |
| Genes surfaced in top-k | 12 | **40** |
| Tokens returned | ~2,123 | **~703** (or more genes at same budget) |
| Content for LLM extraction | ✓ included | ✗ consumer fetches via pull |
| Best for | Full LLM answer via expressed_context | Librarian pattern, SNOW navigation, Celestia manifold |

## Risk & Rollback

**Risk 1:** Backwards compat for existing consumers.
Mitigation: `mode` defaults to `"full"`. All existing clients get
identical behavior.

**Risk 2:** The skip list misses a refiner that writes to
`tier_contrib`. If so, fingerprint mode would return a "thinner"
fingerprint than full mode.
Mitigation: the implementation test (T3) spy-checks that the skipped
functions don't write to `tier_contrib` in either mode.

**Risk 3:** Downstream consumers of `_assemble` expect a specific
response shape; fingerprint-mode's response differs.
Mitigation: separate response model. Pydantic discriminated union on
`mode` field.

## Related

- `docs/FUTURE/PUSH_PULL_CONTEXT.md` — this is the push side of that
  contract. Fingerprint mode **is** the push channel optimized for
  its actual job.
- `docs/FUTURE/WALKER_PATTERNS.md` — librarian dispatch depends on
  fingerprint mode being fast + broad. This unblocks that pattern.
- `docs/FUTURE/LAYERED_FINGERPRINTS.md` — parent fingerprints already
  surface via query_genes; fingerprint mode simply returns them
  without content expansion.
- `docs/specs/2026-04-16-snow-benchmark-design.md` — SNOW's librarian
  variant B directly benefits from fingerprint mode.

## Open Decisions (for implementing session)

1. **Mode-specific config** — `fingerprint_mode_skip_refiners` is a
   single switch. Worth splitting into per-refiner flags
   (`skip_cymatics`, `skip_splade_rerank`, `skip_assemble`) for finer
   A/B? My vote: keep single switch for V1, split if tuning needs it.

2. **Response shape versioning** — add `"schema_version": "fingerprint-v1"`
   so future fingerprint-mode changes can be negotiated?

3. **Default-on or flag?** Should mode=fingerprint be gated behind a
   `HELIX_FINGERPRINT_MODE_ENABLED=1` flag like layered fingerprints,
   or live as a stable endpoint behavior from day one?
   My vote: stable from day one. It's additive (new mode, no behavior
   change for existing consumers), so no A/B gating needed.

# Cymatix-Context Retrieval Dimensions

> **Historical inventory review:** 2026-04-17 (working tree)
> **Historical knowledge store snapshot:** 19,738 documents (13,710 OPEN / 1,949 EUCHRO / 4,079 HETERO)
>
> **Defaults reviewed:** 2026-09-08. The lane graph describes implemented
> capabilities and historical measurements, including optional paths.
> The generated [configuration reference](../config-reference.md) is the
> authoritative defaults reference; the shipped settings are in
> [`cymatix.toml`](../../cymatix.toml). Benchmark results and store counts
> below retain their historical scope.
>
> **LLM-free pipeline milestones:** 2026-04-09 (CPU pipeline commit) +
> 2026-04-13 (Sprints 1-4). The "LLM re_rank" mentioned in §D6 is the
> legacy fallback path. Reranking is off by default
> (`[retrieval] rerank_enabled = false`). When enabled, the optional reranker
> is a pretrained MiniLM cross-encoder (classifier, not an LLM call).
> See [`PIPELINE_LANES.md`](PIPELINE_LANES.md) §"LLM boundary" for the
> full statement. Step 0 query-intent expansion is optional and disabled
> by default (`[ribosome] query_expansion_enabled = false`), keeping the
> default `/context` path LLM-free.

## Lane Graph

```
 PIPELINE STAGE ──►  Schema    Data       Wired to      Bench
                     exists    flowing    retrieval    measured
                    ────────  ────────   ──────────   ────────

 ACTIVE (7)
 ───────────────────────────────────────────────────────────────

 D1  Semantic        [████]    [████]     [████]       [████]
     FTS5+SPLADE     genes_fts  splade_    Tier 1-3.5   N=50 v2
     +ΣĒMA           +embed     terms      + cold ΣĒMA  runs #11-12

 D2  Promoter        [████]    [████]     [████]       [████]
     tagging         promoter_  auto-      Tier 1/2     Implicit in
                     index      populated  keyword+ent  all NIAH runs

 D3  Source          [████]    [████]     [████]       [████]
     provenance      source_id  per-gene   deny-list +  B→C recovery
                     on genes   at ingest  authority    measured

 D4  Working-set     [████]    [████]     [████]       [░░░░]
     access rate     recent_    ring buf   density      Phase 1 gate
                     accesses   c504265    gate 8411623 needs N=50

 D5  Chromatin       [████]    [████]     [████]       [████]
     tier            chromatin  3-tier     hot filter + Runs #11-12
                     column     populated  cold C.2     hot vs h+c

 D6  Cymatics        [████]    [████]     [████]       [░░░░]
     frequency       cymatics   harmonic_  Step 3       post-cymatics
     resonance       .py        links      resonance    N=50 pending

 D9  Temporal        [████]    [████]     [▓▓▓░]       [░░░░]
     context (TCM)   tcm.py     per-session Step 3.25   forward-recall
                     +session   drift vec   tiebreaker  bench pending


 PARTIAL / MIXED (2)
 ───────────────────────────────────────────────────────────────

 D7  Gene            [████]    [▓▓▓░]     [▓▓▓░]       [░░░░]
     attribution     gene_attr  live rows  party gate   scoring bonus
                     +registry  + /ingest   + citations still pending

 D8  Co-activation   [████]    [████]     [████]       [░░░░]
     graph           entity_    entity_gr  harmonic +   SR/entity
                     graph +    + 227k     ray-trace;   graph bench
                     harmonic   links      entity/SR   pending
                                           opt-in


 LEGEND
 ───────────────────────────────────────────────────────────────
  [████]  Done / active        [▓▓▓░]  Partial data
  [░░░░]  Not started
```

Auxiliary tiers not counted as separate D-lanes:

- `path_key_index` Tier 0 compound retrieval runs before D1/D2 fusion when enabled; `[retrieval] pki_enabled = false` by default.
- `filename_anchor` is a lexical boost enabled by default (`[retrieval] filename_anchor_enabled = true`).
- SR is an optional graph-expansion path under D8, disabled by default (`[retrieval] sr_enabled = false`).

---

## Dimension Reference

### D1 — Semantic Content (FTS5 + SPLADE + ΣĒMA)

Supported sub-tiers in a fusion pipeline:

| Sub-tier | Mechanism | Table/field |
|---|---|---|
| Tier 1 | FTS5 full-text with synonym expansion | `genes_fts` |
| Tier 2 | SPLADE sparse term expansion | `splade_terms` |
| Tier 3/3.5 | ΣĒMA 20-dim cosine similarity | `genes.embedding` |
| Cold fallthrough | ΣĒMA cosine on heterochromatin | `genes WHERE chromatin=2` |

SPLADE is opt-in (`[ingestion] splade_enabled = false`; both automatic
size toggles default to `0`). ΣĒMA embedding generation and its warm
retrieval boost are off by default (`[ingestion] sema_embed_on_ingest = false`).
The separate BGE-M3 dense path is also opt-in
(`[retrieval] dense_embedding_enabled = false`). The table lists supported
paths; their activation follows the configuration reference linked above.

Related adjuncts, not counted as separate dimensions:

- optional `path_key_index` Tier 0 compound retrieval (`path_token` + `kv_key`; disabled by default)
- `filename_anchor` lexical boost (enabled by default)

### D2 — Tagging (tag index; legacy: promoter index)

Keyword + entity tag matching at retrieval time. Documents tagged at ingest via `promoter_index`.
Query terms expanded through `cymatix.toml [synonyms]`.

### D3 — Source Provenance

Density gate at storage boundary uses `source_id` for deny-list filtering.
Source authority bonus applies per-source scoring weight during retrieval fusion.

### D4 — Working-Set Access Rate

Windowed access-rate ring buffer replaces monotonic `access_count`.

- `DocumentSignals.recent_accesses` (legacy: `EpigeneticMarkers`) — last 100 access timestamps
- `access_rate(gene, window_seconds)` — sliding-window rate computation
- Wired into density gate access-override path

### D5 — Lifecycle Tier (legacy: Chromatin)

Three-tier accessibility model with cold-tier reactivation:

| Tier | Chromatin | Default retrieval | Cold-tier fallthrough |
|---|---|---|---|
| OPEN | 0 | Always queried | — |
| EUCHROMATIN | 1 | Included with hot | — |
| HETEROCHROMATIN | 2 | Excluded by default | Explicit `include_cold: true`, or enabled automatic fallthrough (see below) |

Automatic fallthrough requires `[context] cold_tier_enabled = true` and
hot results ≤ `cold_tier_min_hot_genes` (default `0`). The master flag
defaults to `false`; explicit `include_cold: true` overrides both gates.

Content is preserved across all tiers (C.1 non-destructive compression).

### D6 — Cymatics (Frequency-Domain Resonance)

Maps retrieval onto wave physics. CPU-based (~5 ms) replacement for LLM re_rank (~2 s).

| Concept | Biology | Cymatics |
|---|---|---|
| Document | Resonant mode | Excited by query "frequencies" |
| Fragment weight | Spectral amplitude | Peak height in 256-bin spectrum |
| Co-activation | Harmonic coupling | Weighted spectral edges (`harmonic_links`) |
| Splice | Bandwidth filtering | Q-factor from `splice_aggressiveness` |

Integrated at Step 3 of `context_manager._express()` as a blended score bonus.
Current live path uses `query_spectrum()` + `flux_score_dispatch()` to add a
small bonus and re-sort candidates. The old "LLM fallback" language is legacy.

### D7 — Document Attribution (partial, live data path)

Schema and data flow are now live: `/ingest` can resolve or accept explicit
`org_id`, `party_id`, `participant_handle`, and `agent_handle`, and writes
`gene_attribution` rows when identity is known.

Current consumers:

1. **Per-party scoping** — `query_genes(..., party_id=...)` excludes documents attributed to other parties while still allowing unattributed legacy documents through.
2. **Citation enrichment** — `/context` citations can emit `authored_by_party` and `authored_by_handle`.

Still pending:

1. **Authorship-class scoring** — same-party or same-agent relevance bonus is not live yet.

### D8 — Co-Activation Graph (partial data)

Three data sources exist, and part of the graph stack is now read at query time:

| Source | Location | Rows |
|---|---|---|
| Legacy co-activation | `epigenetics.co_activated_with` | Per-document JSON |
| Entity graph | `entity_graph` table | Varies |
| Cymatics harmonics | `harmonic_links` table | 227k+ in the historical knowledge store snapshot |

Current live wiring:

- Tier 5 harmonic boost in `query_genes()`
- Step 3.20 harmonic-bin boost via `ray_trace.harmonic_bin_boost()`

Still partial:

- Entity memberships and ingest-time co-activation links are enabled (`[ingestion] entity_graph = true`). The Tier 5b query-time entity-graph boost (Step 3C, 2026-05-08) is a separate opt-in (`[retrieval] entity_graph_retrieval_enabled = false`).
- SR (`sr_boost`) is implemented but disabled by default (`[retrieval] sr_enabled = false`). The 2026-04-22 default-on experiment was reverted in the 2026-06-12 default-honesty pass; a fresh isolation receipt showing a delivered-recall gain is required before enabling it by default.
- seeded edges exist but are dark-shipped by default

**SR Gate Bench Result (2026-05-08, N=50, SEED=42, gemma4:e4b, same knowledge-store A/B):**

| Axis | Axes | SR-OFF retrieval | SR-ON retrieval | Delta |
|---|---|---|---|---|
| 1 | key | 10.0% | 10.0% | +0.0pp |
| 2 | key+project | 10.0% | 8.0% | -2.0pp |
| 3 | key+project+module | 14.0% | 14.0% | +0.0pp |
| 4 | key+project+module+filename | 32.0% | 32.0% | +0.0pp |

Gate criterion: ≥2pp gain on any axis without regression on others.
Result: **GATE NOT MET.** No retrieval_pct gain on any axis. Axis-2 shows -2pp regression
(within N=50 noise floor of ±1 needle). SR's in_context_pct shows marginal +2pp on axes 2-3
but this does not clear the gate on the primary recall@1 metric.

This historical gate did not justify default-on SR. SR is disabled in the
shipped configuration; any future default change needs a fresh
delivered-recall isolation receipt on a current bed.

### D9 — Temporal Context Model (built, lightly wired)

Howard & Kahana 2002 temporal context evolution equation now exists as a
per-session drift vector. It is not a primary recall tier inside
`query_genes()`; instead, it acts as a trajectory layer that lightly
reorders already-retrieved candidates.

Current live wiring:

- per-session TCM state is initialized in `CymatixContextManager`
- Step 3.25 applies `tcm_bonus(...)` as a tiebreaker over current candidates
- optional theta-biased ray tracing can use TCM velocity, but that flag is off by default

Reference: Howard, M. W., & Kahana, M. J. (2002). *A distributed representation of temporal context.* J. Math. Psych. 46(3), 269-299.

---

## Decision Gates

| Dimension | Test | Ship if | Drop if |
|---|---|---|---|
| D4 | N=50 access_rate ON vs OFF | ≥1pp retrieval | Worse than monotonic |
| D6 | N=50 post-cymatics | ≥0pp (non-regression) + latency win | Retrieval degrades vs LLM re_rank |
| D7 | Party scoping + attribution correctness | No cross-party leakage; citations correctly attributed | Wrongly-scoped or misattributed data |
| D8 | Harmonic/ray-trace/SR A/B | ≥2pp retrieval or clear robustness gain | <1pp and no failure-mode reduction |
| D9 | TCM forward-recall asymmetry | Asymmetry visible in benchmark | No asymmetry — wrong or N/A |

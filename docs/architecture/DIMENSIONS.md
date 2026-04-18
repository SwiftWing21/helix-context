# Helix-Context Retrieval Dimensions

> **Last reviewed:** 2026-04-17 (working tree)
> **Genome snapshot:** 19,738 genes (13,710 OPEN / 1,949 EUCHRO / 4,079 HETERO)
>
> **Status: LLM-free pipeline as of 2026-04-09 (CPU pipeline commit) +
> 2026-04-13 (Sprints 1-4).** Every dimension below — D1 through D9,
> including D6 cymatics rerank — runs on deterministic CPU math. The
> "LLM re_rank" mentioned in §D6 is the *legacy* fallback path; in the
> live config `rerank_enabled = false` and the active rerank, when on,
> is a pretrained cross-encoder MiniLM (classifier, not an LLM call).
> See [`PIPELINE_LANES.md`](PIPELINE_LANES.md) §"LLM boundary" for the
> full statement. Step 0 query-intent expansion is flag-gated
> (`[ribosome] query_expansion_enabled`); flip to `false` for a
> strictly LLM-free `/context`.

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

 D8  Co-activation   [████]    [████]     [▓▓▓░]       [░░░░]
     graph           entity_    entity_gr  harmonic +   SR/entity
                     graph +    + 227k     ray-trace    graph bench
                     harmonic   links      live; SR off pending


 LEGEND
 ───────────────────────────────────────────────────────────────
  [████]  Done / active        [▓▓▓░]  Partial data
  [░░░░]  Not started
```

Auxiliary live tiers not counted as separate D-lanes:

- `path_key_index` Tier 0 compound retrieval is live and fires before D1/D2 fusion.
- `filename_anchor` exists as a dark-shipped lexical boost, but remains off in the default `helix.toml`.
- `SR` exists as a graph-expansion path under D8, but remains dark-shipped in the default `helix.toml`.

---

## Dimension Reference

### D1 — Semantic Content (FTS5 + SPLADE + ΣĒMA)

Three sub-tiers in a fusion pipeline:

| Sub-tier | Mechanism | Table/field |
|---|---|---|
| Tier 1 | FTS5 full-text with synonym expansion | `genes_fts` |
| Tier 2 | SPLADE sparse term expansion | `splade_terms` |
| Tier 3/3.5 | ΣĒMA 20-dim cosine similarity | `genes.embedding` |
| Cold fallthrough | ΣĒMA cosine on heterochromatin | `genes WHERE chromatin=2` |

Related live adjuncts, not counted as separate dimensions:

- `path_key_index` Tier 0 compound retrieval (`path_token` + `kv_key`)
- optional `filename_anchor` lexical boost (dark-shipped by default)

### D2 — Promoter Tagging

Keyword + entity tag matching at retrieval time. Genes tagged at ingest via `promoter_index`.
Query terms expanded through `helix.toml [synonyms]`.

### D3 — Source Provenance

Density gate at storage boundary uses `source_id` for deny-list filtering.
Source authority bonus applies per-source scoring weight during retrieval fusion.

### D4 — Working-Set Access Rate

Windowed access-rate ring buffer replaces monotonic `access_count`.

- `EpigeneticMarkers.recent_accesses` — last 100 access timestamps
- `access_rate(gene, window_seconds)` — sliding-window rate computation
- Wired into density gate access-override path

### D5 — Chromatin Tier

Three-tier accessibility model with cold-tier reactivation:

| Tier | Chromatin | Default retrieval | Cold-tier fallthrough |
|---|---|---|---|
| OPEN | 0 | Always queried | — |
| EUCHROMATIN | 1 | Included with hot | — |
| HETEROCHROMATIN | 2 | Excluded by default | Opt-in via `include_cold` or automatic when hot returns ≤ `cold_tier_min_hot_genes` |

Content is preserved across all tiers (C.1 non-destructive compression).

### D6 — Cymatics (Frequency-Domain Resonance)

Maps retrieval onto wave physics. CPU-based (~5 ms) replacement for LLM re_rank (~2 s).

| Concept | Biology | Cymatics |
|---|---|---|
| Gene | Resonant mode | Excited by query "frequencies" |
| Codon weight | Spectral amplitude | Peak height in 256-bin spectrum |
| Co-activation | Harmonic coupling | Weighted spectral edges (`harmonic_links`) |
| Splice | Bandwidth filtering | Q-factor from `splice_aggressiveness` |

Integrated at Step 3 of `context_manager._express()` as a blended score bonus.
Current live path uses `query_spectrum()` + `flux_score_dispatch()` to add a
small bonus and re-sort candidates. The old "LLM fallback" language is legacy.

### D7 — Gene Attribution (partial, live data path)

Schema and data flow are now live: `/ingest` can resolve or accept explicit
`org_id`, `party_id`, `participant_handle`, and `agent_handle`, and writes
`gene_attribution` rows when identity is known.

Current consumers:

1. **Per-party scoping** — `query_genes(..., party_id=...)` excludes genes attributed to other parties while still allowing unattributed legacy genes through.
2. **Citation enrichment** — `/context` citations can emit `authored_by_party` and `authored_by_handle`.

Still pending:

1. **Authorship-class scoring** — same-party or same-agent relevance bonus is not live yet.

### D8 — Co-Activation Graph (partial data)

Three data sources exist, and part of the graph stack is now read at query time:

| Source | Location | Rows |
|---|---|---|
| Legacy co-activation | `epigenetics.co_activated_with` | Per-gene JSON |
| Entity graph | `entity_graph` table | Varies |
| Cymatics harmonics | `harmonic_links` table | 227k+ in current genome |

Current live wiring:

- Tier 5 harmonic boost in `query_genes()`
- Step 3.20 harmonic-bin boost via `ray_trace.harmonic_bin_boost()`

Still partial:

- `entity_graph` is populated but not a first-class read path in retrieval fusion
- SR (`sr_boost`) exists under D8 but is dark-shipped by default
- seeded edges exist but are dark-shipped by default

### D9 — Temporal Context Model (built, lightly wired)

Howard & Kahana 2002 temporal context evolution equation now exists as a
per-session drift vector. It is not a primary recall tier inside
`query_genes()`; instead, it acts as a trajectory layer that lightly
reorders already-retrieved candidates.

Current live wiring:

- per-session TCM state is initialized in `HelixContextManager`
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

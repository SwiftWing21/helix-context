# Helix-Context Retrieval Dimensions

> **Last updated:** 2026-04-11 (commit `059d902`)
> **Genome:** 17,623 genes (12,401 OPEN / 1,895 EUCHRO / 3,327 HETERO)

## Lane Graph

```
 PIPELINE STAGE ──►  Schema    Data       Wired to      Bench
                     exists    flowing    retrieval    measured
                    ────────  ────────   ──────────   ────────

 ACTIVE (6)
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


 IN PROGRESS (3)
 ───────────────────────────────────────────────────────────────

 D7  Gene            [████]    [░░░░]     [░░░░]       [░░░░]
     attribution     gene_attr  0 rows     not a        blocked on
                     +parties   schema     scoring      registry
                     +particip  only       signal yet   data flow

 D8  Co-activation   [████]    [▓▓▓░]     [░░░░]       [░░░░]
     graph           entity_    entity_gr  not read     blocked on
                     graph +    harmonic_  at query     ray-trace
                     harmonic   links(94)  time         wiring

 D9  Temporal        [░░░░]    [░░░░]     [░░░░]       [░░░░]
     context (TCM)   not built  —          —            —
                     assigned                           Howard &
                                                        Kahana 2002


 LEGEND
 ───────────────────────────────────────────────────────────────
  [████]  Done / active        [▓▓▓░]  Partial data
  [░░░░]  Not started
```

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

Integrated at Step 3 of `context_manager._express()`: `cymatics.resonance_rank()` preferred, LLM fallback.

### D7 — Gene Attribution (in progress)

Schema deployed (`gene_attribution`, `participants`, `parties`). Zero data rows — registry ingestion path not wired. Two planned consumers:

1. **Per-party scoping** — security-critical cross-tenant isolation
2. **Authorship-class scoring** — genes from user's own party get relevance bonus

### D8 — Co-Activation Graph (partial data)

Three data sources, none currently read at query time:

| Source | Location | Rows |
|---|---|---|
| Legacy co-activation | `epigenetics.co_activated_with` | Per-gene JSON |
| Entity graph | `entity_graph` table | Varies |
| Cymatics harmonics | `harmonic_links` table | 94+ |

Candidate wiring approaches: ScoreRift ray-trace port, or simpler harmonic-link boost.

### D9 — Temporal Context Model (not built)

Howard & Kahana 2002 temporal context evolution equation as a per-session drift vector.
Reframed as a **trajectory layer** operating across all retrieval dimensions, not a 9th
retrieval dimension competing with D1–D8 for ranking weight.

Reference: Howard, M. W., & Kahana, M. J. (2002). *A distributed representation of temporal context.* J. Math. Psych. 46(3), 269-299.

---

## Decision Gates

| Dimension | Test | Ship if | Drop if |
|---|---|---|---|
| D4 | N=50 access_rate ON vs OFF | ≥1pp retrieval | Worse than monotonic |
| D6 | N=50 post-cymatics | ≥0pp (non-regression) + latency win | Retrieval degrades vs LLM re_rank |
| D7 | Wire data flow, test party isolation | No cross-party leakage | (Must ship — security requirement) |
| D8 | Ray-trace or harmonic boost, N=50 | ≥2pp retrieval | <1pp — graph not load-bearing |
| D9 | TCM forward-recall asymmetry | Asymmetry visible in benchmark | No asymmetry — wrong or N/A |

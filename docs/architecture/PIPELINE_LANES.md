# Pipeline Lanes — Data In / Data Out (v2)

> *"Where do tags get added? What tools fire when? Who writes to what?"*

A swim-lane reference for the helix-context pipeline. Four flows
(ingest, context, packet, fingerprint), each broken down by which
component does what, what tools fire, and what gets written / read.

**v2, updated 2026-04-18**, after the SIKE pathway reframe. The
additions since v1:

- `/context/packet` — the **agent-safe** retrieval surface. Returns
  pointers + `verified / stale_risk / needs_refresh` labels +
  `refresh_targets`, not assembled content. See
  [`docs/specs/2026-04-17-agent-context-index-build-spec.md`](../specs/2026-04-17-agent-context-index-build-spec.md).
- `/context/refresh-plan` — thin convenience over packet; returns only
  the refresh plan.
- `/fingerprint` — navigation-first retrieval (scores + metadata, no
  content) with `score_floor` + honest accounting
  (`evaluated_total / above_floor_total / filtered_by_floor /
  truncated_by_cap`).
- **Weighing layer** — freshness × authority × specificity ×
  coord_confidence composing into a status label. First-class
  concept, not a post-hoc gate.
- **Provenance at ingest** — `source_kind`, `volatility_class`,
  `observed_at`, `last_verified_at` auto-populated from file
  extension (+ backfill script for existing rows).

v1 core (ingest + `/context` query) is unchanged and preserved below.

---

## LLM boundary (still the load-bearing frame)

```
═════════════════ LLM-FREE ZONE ═════════════════ │ ═══ LLM ═══
                                                  │
ingest → tag → encode → store → query → retrieve  │  /v1/chat/
weigh (freshness + coord) → label → emit          │  completions
(all CPU, deterministic, no model calls)          │  (Claude Haiku
                                                  │   via ribosome)
                                                  │
                  pipeline crosses the boundary ──┘
                  exactly once, at answer generation
```

**Every stage of the data pipeline — `/ingest`, the 12-tone signal
stack, cymatics flux/W1, TCM, SR, theta ray-trace, Hebbian
seeded-edges, splice, the cross-encoder rerank when enabled, the
packet builder's freshness + coord weighing — runs on pure CPU math
with zero LLM calls.** The only LLM call in the system is at
`/v1/chat/completions`, which consumes the already-built context and
emits the reply.

> **Step 0 query-intent expansion — optional, flag-gated.**
> `/context` can invoke `_expand_query_intent()` once per *novel* query
> string (LRU-cached, ~100 tokens out, falls back to the raw query on
> any failure). Controlled by `[ribosome] query_expansion_enabled` in
> `helix.toml`. Set to `false` for a strictly LLM-free `/context`
> pipeline. `/context/packet` does **not** invoke this step —
> the agent-safe surface is LLM-free end-to-end.

The math citations behind the LLM-free pipeline:

- Werman, Peleg, Rosenfeld (1986) — circular W1 histogram distance
- Howard & Kahana (2002); Howard, Fotedar, Datey, Hasselmo (2005) — TCM
- Dayan (1993); Stachenfeld, Botvinick, Gershman (2017) — SR
- Wang, Foster, Pfeiffer (2020) — theta forward/backward alternation
- Singh et al. (2020) — Context Mover's Distance (CMD)
- Metodiev, Nachman, Thaler (2017) — CWoLa (Sprint 3, deferred)

---

## Component map (the lanes)

```
CLIENT  ────►  SERVER  ────►  TAGGER  ────►  ENCODERS  ────►  GENOME (DB)
                  │              │              │                │
                  ├─► PACKET     │              │                │
                  │   BUILDER ◄──┴──────────────┴────────────────┤
                  │   (weighing layer: freshness + coord)        │
                  │                                              │
                  └─► REGISTRY ◄──────────────────────────────────┘
                       (orgs / parties / participants / agents /
                        gene_attribution + tz)
```

Lanes:

- **CLIENT** — IDE plugin / proxy / curl / MCP host / your code
- **SERVER** — `helix_context/server.py` FastAPI endpoints
- **TAGGER** — `helix_context/tagger.py` CpuTagger (no LLM, default)
- **RIBOSOME** — `helix_context/ribosome.py` answer-generation only
  (Claude Haiku via `[ribosome] backend = "claude"`); **not on the
  ingest path** since 2026-04-09 CPU pipeline commit
- **ENCODERS** — SPLADE / SEMA / cymatics (numerical, deterministic)
- **PACKET BUILDER** — `helix_context/context_packet.py` — weighing
  layer that labels retrieval results by freshness + coord confidence
- **GENOME** — SQLite tables: `genes`, `promoter_index`, `genes_fts`,
  `entity_graph`, `path_key_index`
- **REGISTRY** — federation tables: `orgs`, `parties`, `participants`,
  `agents`, `gene_attribution`

---

## INGEST flow

Ingest is unchanged from v1 except for the **9 provenance fields**
that now get auto-populated. These flow into both the `genes` table
(direct columns) and — when a shard is registered — `main.db`
`source_index` (when Phase 1 full ships; currently gene-local is the
single source of truth).

### Provenance fields (auto-populated since 3c6ead6)

Computed at ingest via `helix_context/provenance.apply_provenance()`
from the source path + ingestion timestamp:

| Field | Source | Default |
|---|---|---|
| `source_kind` | file extension → `code / config / doc / log / db` | `doc` (unknown ext) |
| `volatility_class` | derived from source_kind | code/doc=`stable` (7d), config/log=`hot` (15min), db=`medium` (12h) |
| `observed_at` | `time.time()` at ingest | fresh |
| `last_verified_at` | same as observed_at at ingest | fresh |
| `source_id` | caller-supplied `metadata["path"]` or `source_path` | `None` for non-path content |
| `authority_class` | not auto-set (defaults `primary` when consumed) | `None` |
| `repo_root` | not auto-set yet | `None` |
| `mtime` | not auto-set yet | `None` |
| `content_hash` | not auto-set yet | `None` |

Non-path source_ids (e.g. `"__session__"`, `"agent:laude"`) are
deliberately left NULL — the packet builder falls through to "freshness
unknown" for free-form identifiers, which is the honest answer.

### ASCII fallback (ingest)

```
client POST /ingest
  │
  ├─► Server: _local_attribution_defaults()         env vars + os
  │   └─► Registry.local_org/participant/agent      (registry tables)
  │
  ├─► CpuTagger ─► entities       (spaCy NER + EntityRuler)
  │             ─► domains        (regex)
  │             ─► key_values     ("key=value" list)
  │
  ├─► Encoders  ─► SPLADE sparse    (ModernBERT, deterministic)
  │             ─► SEMA 20D
  │             ─► cymatics 256-bin spectrum
  │
  ├─► Provenance ─► source_kind        (extension → kind)
  │   (new)      ─► volatility_class   (kind → half-life class)
  │              ─► observed_at
  │              ─► last_verified_at
  │
  ├─► Density gate ─► chromatin tier  (open/euchro/heterochro)
  │
  └─► WRITES:
        genes (+ 9 provenance columns), promoter_index, genes_fts,
        entity_graph, path_key_index,
        gene_attribution (org/dev/user/agent/tz/at)
```

---

## `/context` flow (decoder path — unchanged from v1)

The decoder surface: Helix owns the whole pipeline, returns assembled
context ready for a downstream LLM. Full retrieval + compression.

```
client POST /context (query, session_context)
  │
  ├─► Step 0: _expand_query_intent()        (LLM, cached, flag-gated)
  ├─► Step 1: CpuTagger.extract              → domains+entities
  ├─► Step 1b: session_context path_tokens   → injected into entities
  │
  ├─► Genome.query_genes (12 signals + 1 octave gate):
  │     Tier 0  path_key_index            (PKI compound, IDF-weighted)
  │     Tier 1  exact promoter tag        (3.0)
  │     Tier 2  prefix promoter tag       (1.5)
  │     Tier 3  FTS5 content              (≤6.0 cap)
  │     Tier 3.5 SPLADE sparse            (≤3.5)
  │     Tier 4  SEMA cold-tier            (cosine fallback)
  │     Tier 5  harmonic boost            (≤3.0)
  │     +     cymatics resonance          (Gaussian overlap)
  │     +     cymatics flux integral      (∫ B⃗·dA⃗)
  │     +     TCM session drift           (Howard&Kahana)
  │     +     ray-trace evidence          (Monte Carlo)
  │     +     access-rate tiebreaker      (≤0.25)
  │     gate: party_id filter             (octave — same shape, new identity)
  │
  ├─► Score-floor budget tier:
  │     top_score ≥ 5.0 + ratio ≥ 3.0  → tight  (3 genes, 6k tokens)
  │     top_score ≥ 2.5 + ratio ≥ 1.8  → focused (6 genes, 9k tokens)
  │     else                            → broad  (12 genes, 15k tokens)
  │
  ├─► Step 3:    cymatics blend bonus
  ├─► Step 3.20: harmonic bin boost (overtone series read)
  ├─► Step 3.25: TCM session re-sort
  │
  ├─► Step 4: Ribosome compress (Kompress/Headroom)
  │
  ├─► Weighing surface (Step 1b-iter2, 2026-04-18):
  │     coordinate_crispness, neighborhood_density
  │     top_score_raw, top_dominance
  │     path_token_coverage            ← the discriminator (Δ+0.48)
  │     resolution_confidence = pathcov × √max(coverage, 0.05)
  │
  └─► RETURN: expressed_context + citations + 4-axis attribution
             + ContextHealth (ellipticity + coord-resolution fields)
```

The weighing fields are also emitted on `/context` responses so
traditional decoder-path callers can act on confidence. Packet mode
(next section) makes them first-class.

---

## `/context/packet` flow (index path — NEW in v2)

The agent-safe surface per the 2026-04-17 build spec. Returns
pointers + status labels + refresh plan, not assembled content.
Caller fetches from `source_path` if they need the bytes.

```
client POST /context/packet (query, task_type, max_genes)
  │
  ├─► (reuses /context Steps 1 + 1b for signal extraction)
  ├─► (reuses Genome.query_genes for the 12 tiers)
  │
  ├─► Skip /context's Step 4 compression — no content bundle needed
  │
  ├─► Packet builder: context_packet.py
  │     │
  │     ├─► _coordinate_confidence(query, genes)
  │     │    └─► Path-token overlap: delivered source_paths ∩ query signals
  │     │        Hit mean 1.00 / miss mean 0.52 on 10-needle bench
  │     │
  │     ├─► For each gene, _effective_meta() + _build_item():
  │     │    ├─► freshness_score = exp(-age / half_life[volatility])
  │     │    ├─► authority_score (primary=1.0, derived=0.75, inferred=0.45)
  │     │    ├─► specificity_score (literal=1.0, support_span=0.9, doc=0.75)
  │     │    ├─► live_truth_score = freshness × authority × specificity
  │     │    └─► status = _status_for(task_type, freshness, authority)
  │     │              then _apply_coordinate_confidence(coord < 0.30 → downgrade)
  │     │
  │     └─► RefreshTarget for each non-verified item
  │
  └─► RETURN ContextPacket:
        {
          task_type, query,
          verified:          [ContextItem]     status="verified"
          stale_risk:        [ContextItem]     status="stale_risk" or "needs_refresh"
          contradictions:    [ContextItem]     (empty until Phase 2 claims land)
          refresh_targets:   [RefreshTarget]   prioritized reread list
          working_set_id,
          notes:             ["coord_conf=X..."]  threshold warnings
        }
```

### Task profiles (how weighing composes)

Status thresholds are task-sensitive:

| task_type | freshness ≥ verified | coord < 0.30 effect | notes |
|---|---|---|---|
| `plan` | 0.35 | stale_risk | low-risk, tolerant |
| `explain` | 0.35 | stale_risk | low-risk, tolerant |
| `review` | 0.55 | stale_risk | moderate |
| `edit` | 0.70 | **needs_refresh** | high-risk |
| `debug` | 0.70 | **needs_refresh** | high-risk |
| `ops` | 0.70 | **needs_refresh** | literal-answer, no tolerance |
| `quote` | 0.70 | **needs_refresh** | literal-answer, no tolerance |

### Validation — Phase 5 bench

10/10 scenarios pass across 5 families (see
[`benchmarks/bench_packet.py`](../../benchmarks/bench_packet.py) and
[`benchmarks/results/packet_bench_2026-04-18.json`](../../benchmarks/results/packet_bench_2026-04-18.json)):

- **stale_by_age** (3) — stable@30d, hot@1h, medium@2d → all flag
- **coordinate_mismatch** (2) — edit vs explain on off-target retrieval
- **task_sensitivity** (2) — same gene, different task → different verdict
- **authority_downgrade** (1) — inferred authority on ops → refresh
- **clean_verified** (2) — fresh + aligned + primary → stays verified

---

## `/context/refresh-plan` flow (NEW in v2)

Thin convenience over `/context/packet`. Returns only the
`refresh_targets` list, skipping the evidence buckets. Useful when the
caller already has content cached and just needs to decide which
sources to reread before a high-risk action.

```
client POST /context/refresh-plan (query, task_type=edit)
  │
  └─► get_refresh_targets(query, task_type, genome)
      └─► (reuses build_context_packet internals)
          └─► Filter to items where status != "verified"
              Emit RefreshTarget { target_kind, source_id, reason, priority }

RETURN: {
  query, task_type,
  refresh_targets: [{target_kind, source_id, reason, priority}, ...],
  response_mode: "refresh_plan"
}
```

---

## `/fingerprint` flow (NEW in v2)

Navigation-first retrieval per GT's fingerprint-mode plan. Returns
scored gene pointers + tier contribution metadata — no content, no
packet labels. Fast path for callers that want to inspect the
retrieval without paying for compression or weighing.

```
client POST /fingerprint (query, profile, max_results, score_floor)
  │
  ├─► Profile selection: fast / balanced / quality
  │     Affects eval-budget (50 / 100 / 200)
  │
  ├─► (reuses /context Steps 1 + 1b)
  ├─► (reuses Genome.query_genes 12-signal stack)
  ├─► Refiner pass: cymatics + harmonic + TCM bonuses applied
  │
  ├─► score_floor filter (optional, per bc47fb8):
  │     if final_score < score_floor: drop
  │     Eval budget expands to min(max(max_results*3, 50), 200) when set
  │
  ├─► Cap: top max_results by post-refiner score
  │
  └─► RETURN:
        score_floor, evaluated_total, above_floor_total,
        returned, filtered_by_floor, truncated_by_cap,
        response_hint,
        fingerprints: [{gene_id, source_id, score, tier_contributions, ...}]
```

**Accounting contract:** counts are defined over the *evaluated*
candidate set only. Helix cannot honestly claim how many whole-corpus
items fell below the floor without evaluating the whole corpus, so the
API does not pretend otherwise.

---

## Weighing layer (the conceptual center of gravity for v2)

The packet builder is not a post-retrieval filter. It is a **layer**
that takes retrieval output and emits a composed confidence:

```
coord_conf × (freshness × authority × specificity) = is-it-safe-to-act

  └─ location ──┘    └────── content trust ──────┘
     (pathcov)       (freshness/authority/specificity)

     "did we resolve     "is what we resolved to
      to the right       still trustworthy?"
      place?"
```

Both halves are needed:

- **Coord confidence alone** doesn't catch stale files in the right
  folder. A verified config from 3 days ago has `coord_conf = 1.0`
  but is still stale for an ops task.
- **Freshness alone** doesn't catch wrong-folder retrievals. A fresh
  gene with `last_verified_at = now` from the wrong repo is still the
  wrong answer.

The status label encodes the composition:

- `verified` — both halves pass the task's threshold
- `stale_risk` — freshness or coord is marginal for the task
- `needs_refresh` — freshness or coord is bad enough that action
  should pause for a reread

This is the SIKE pathway-layer identity ("Helix weighs, doesn't
retrieve") implemented. See
[`docs/specs/2026-04-17-agent-context-index-build-spec.md`](../specs/2026-04-17-agent-context-index-build-spec.md)
for the authoritative design.

---

## Where each kind of tag happens

| Tag type | Source | Created at | Used at |
|---|---|---|---|
| `domains` | regex + heuristics in CpuTagger | ingest | Tiers 1, 2, 3 |
| `entities` | spaCy NER + EntityRuler | ingest | Tiers 1, 2, 3, entity_graph |
| `key_values` | regex `key=value` extractor | ingest | path_key_index, ellipticity health |
| `complement` | Ribosome LLM (legacy) | ingest | retrieval display only (not score) |
| `codons` | Ribosome LLM (legacy) | ingest | expressed_context formatting |
| `path_token` | `path_tokens(source_id)` | ingest | path_key_index Tier 0 + packet coord_confidence |
| `cymatics spectrum` | term-hashed Gaussian | ingest | resonance + flux + harmonic bins |
| `embedding (SEMA)` | sentence-transformer | ingest | Tier 4 cold-tier |
| `SPLADE terms` | ModernBERT sparse | ingest | Tier 3.5 |
| `chromatin tier` | density_gate at ingest | ingest | hot/warm/cold partitioning |
| `source_kind` (new) | `provenance.infer_source_kind` | ingest | packet specificity + volatility |
| `volatility_class` (new) | `provenance.infer_volatility` | ingest | packet freshness half-life |
| `observed_at / last_verified_at` (new) | `time.time()` at ingest | ingest | packet freshness_score decay |
| `attribution row` | `Registry.attribute_gene` | ingest | filter scoping + audit |

---

## Read paths (where each table is touched at query time)

| Table | Read by | Purpose |
|---|---|---|
| `path_key_index` | `/context` Tier 0 | compound (path, key) lookup |
| `promoter_index` | `/context` Tier 1, 2 | tag exact / prefix match |
| `genes_fts` | `/context` Tier 3 | FTS5 full-text |
| `genes.embedding` | `/context` Tier 4 | SEMA cold-tier cosine scan |
| `harmonic_links` | `/context` Tier 5 | mutual reinforcement |
| `entity_graph` | `/context` post-rank | co-activation pull-forward |
| `gene_attribution` | `/context` filter | party_id scoping (octave gate) |
| `genes.epigenetics` | `/context` tiebreaker | access-rate / recent_accesses ring |
| `genes.source_id` (new) | `/context/packet` | path_tokens → coord_confidence |
| `genes.source_kind` (new) | `/context/packet` | specificity weighting |
| `genes.volatility_class` (new) | `/context/packet` | freshness half-life selection |
| `genes.last_verified_at` (new) | `/context/packet` | freshness_score decay input |
| `genes.authority_class` (new) | `/context/packet` | authority score lookup |
| `source_index` (future Phase 1 full) | `/context/packet` | overrides gene-local provenance |
| `agents` | all paths | citation enrichment |
| `parties` | all paths | citation enrichment + tz |
| `orgs` | analytics | cross-tenant aggregation |

---

## Companion docs

- [`../specs/2026-04-17-agent-context-index-build-spec.md`](../specs/2026-04-17-agent-context-index-build-spec.md) — authoritative packet-mode spec (657 lines)
- [`FEDERATION_LOCAL.md`](FEDERATION_LOCAL.md) — 4-layer + tz attribution model that every ingest writes through
- [`../research/MUSIC_OF_RETRIEVAL.md`](../research/MUSIC_OF_RETRIEVAL.md) — why the 12 signals + 1 octave gate is the chromatic structure
- [`DIMENSIONS.md`](DIMENSIONS.md) — formal retrieval dimension inventory
- [`../../benchmarks/bench_packet.py`](../../benchmarks/bench_packet.py) — Phase 5 bench source
- [`../../benchmarks/results/packet_bench_2026-04-18.json`](../../benchmarks/results/packet_bench_2026-04-18.json) — Phase 5 artifact

# Session Handoff — 2026-04-19

> **Previous handoff:** 2026-04-14 (late evening PT). See git history for
> commits through `daa85e6`. This handoff supersedes it.

---

## What landed today (8 commits on master, all pushed)

```
6f04e1c  feat(claims): edge detection + cache & external-retriever benches
8308956  feat(adapters): cache + retriever adapter + full-stack bench cell
e94d00e  feat(adapters): DAG walker + DAL reference adapter + router framing
532568b  bench+docs: embedding cell + Helix×RAG composition integration guide
157762b  bench: Helix + RAG composition NIAH (3-cell, dual-scored)
daa85e6  bench: multi-needle NIAH + headroom E2E latency
d05d62a  feat(launcher): make [headroom] autostart=true by default
9390403  feat(launcher): Tier 2 Headroom integration — tray menu + adoption
```

Plus a sibling session's `6db30e9 Hide paused ribosome from launcher tools`
pushed earlier in the day.

## Load-bearing reframe this session

**Helix is the router ABOVE the stack, not half of it.**

Prior framing ("Helix emits half of a RAG+DAG+DAL stack") was wrong.
The packet fields (task_type, coord_confidence, verdict, volatility,
contradictions, supersedes, refresh_targets) are routing signals; the
stack (RAG/DAG/DAL) is the execution layer below.

Example of the choice math Helix already does:
- `verified` + `coord_conf > 0.5` → RAG only
- `stale_risk` + `hot` volatility → DAL refetch, then RAG
- `contradictions` non-empty → DAG walk first, then DAL on winner
- `task_type=edit` + `needs_refresh` → all three in order

Documented as the central pattern in `docs/INTEGRATING_WITH_EXISTING_RAG.md`.

## New surface area

### Phase 2 claims layer — now fully operational

- **Extraction**: `helix_context/claims.py` (code/config/doc/benchmark
  extractors + key_values fallback). Shipped commit `bc5cc9f`.
- **Edges**: `helix_context/claims_analyze.py` (contradicts /
  duplicates / supersedes via Jaccard over entity_key groups).
  Shipped commit `6f04e1c`.
- **Walker**: `helix_context/claims_graph.py` (supersedes chain,
  contradiction clusters, topo sort, resolve + resolve_from_packet).
  Shipped commit `e94d00e`.
- **Backfill script**: `scripts/backfill_claims.py` now runs both
  extraction AND edge detection passes.

**Live state (genomes/main.db):** 78,472 claims + 95,382 edges
(50,362 contradicts + 45,020 duplicates + 0 supersedes) across 20,978
entity_key groups.

### Post-Helix composition adapters (reference)

All in `helix_context/adapters/`:
- **`dal.py`** — scheme-dispatch fetcher (`file://` + `http(s)://`
  default; `fetch_s3` opt-in). Soft-fail FetchResult.
- **`cache.py`** — TTL-bounded LRU wrapping a DAL. TTLs from Helix's
  `volatility_class` (stable=7d, medium=12h, hot=15min).
- **`retriever.py`** — duck-typed `Retriever` protocol + LlamaIndex
  and LangChain wrappers + `HelixNarrowedRetriever` for the
  shortlist-narrowing pattern.

### Launcher — Headroom integration (Tier 2)

- New `[headroom]` config section in `helix.toml`
- `HeadroomSupervisor` with orphan adoption (never spawns duplicates)
- Tray menu: Open Headroom Dashboard + Start/Restart/Stop Headroom
- Default `autostart=true` when `enabled=true`
- `start-helix-tray.bat` documented with HELIX_HEADROOM_* env opts

## Benchmark table (2026-04-19 snapshot)

### Multi-needle NIAH (8 needles, 7846-gene genome)

| Cell | ptr_partial | ans_full | ans_partial | latency |
|---|---|---|---|---|
| pure_rag_bm25 | 0.19 | 4/8 | 0.62 | 30 ms |
| pure_rag_embedding | 0.00 | 1/8 | 0.44 | 1083 ms |
| helix_only | 0.19 | 0/8 | 0.19 | 849 ms |
| helix_rag | 0.19 | 5/8 | **0.81** | 849 ms |
| helix_full_stack | 0.19 | 5/8 | **0.81** | 873 ms |

Full-stack matches `helix_rag` — DAG walks but content-presence
doesn't change. The right measurement for DAG value is
decision-quality (stale-claim avoidance), not content recall.

### External retriever — pattern 2 validation

| Metric | Raw SEMA | Helix-Narrowed |
|---|---|---|
| content_recall | 0.44 | **0.56** (+27%) |
| search space | 6,682 | ~13 (**516× smaller**) |
| latency | 903 ms | 1098 ms |

### Cache hit-rate (3 agents × 6 queries, 70/30 overlap)

41.67% hit rate, 4.5% wall savings (modest — local files are <1ms).
HTTP/S3 backends would show 10× or more.

### Headroom E2E

| Content | Headroom on | Fallback |
|---|---|---|
| code | 300ms | <1ms |
| doc | 460ms | <1ms |
| config | 275ms | <1ms |

Compression benefit flips by budget: at 200 chars, pure overhead;
at 1000, saves 9-17k chars/call for code+config.

## Open docs gap follow-on

[Issue #8](https://github.com/SwiftWing21/helix-context/issues/8) —
SETUP.md with 14-extra decision matrix, implicit-req callouts,
TROUBLESHOOTING.md, Phase 2 claims layer mention in README,
Linux/macOS launcher parity. Not blocking; filed for next owner.

## Stretch-move bench addendum (2026-04-19 evening)

All four queued stretch moves shipped. New files in this batch:

- `benchmarks/bench_stale_claim_avoidance.py` + results
- `benchmarks/bench_dal_http_s3.py` + results
- `benchmarks/bench_multi_needle_50.py` + results
- `benchmarks/bench_chroma_integration.py` + results

### 1. Stale-claim avoidance — DAG lift is now measurable

20-entity synthetic corpus (4 versioned-monotonic, 4 versioned-nonmono,
5 contradicted, 7 clean). Three retrieval modes:

| Mode | mono correct | nonmono correct | contradiction flag | p50 |
|---|---|---|---|---|
| raw_newest | 1.00 | **0.00** (100% stale leak) | 0.00 | 29 μs |
| raw_all | 1.00 | **0.00** (100% stale leak) | 0.00 | 30 μs |
| helix_dag | 1.00 | **1.00** | **1.00** | 136 μs |

The non-monotonic case (a stale claim ingested LATER than the current
one) is the realistic failure mode. Raw retrieval leaks stale 100% of
the time; DAG resolves it 100% of the time. Contradiction flagging
moves from 0 → 100%. Cost: ~100 μs per query.

### 2. DAL HTTP/S3 wall-savings curve

Sweep per-fetch latency from 1→200 ms on the same 3-agent × 24-fetch
workload (70% overlap):

| latency | cold wall | warm wall | saved | speedup |
|---|---|---|---|---|
| 1 ms   | 108 ms   | 62 ms    | 43.2% | 1.76× |
| 20 ms  | 1.47 s   | 841 ms   | 42.9% | 1.75× |
| 100 ms | 7.23 s   | 4.11 s   | 43.1% | 1.76× |
| 200 ms | 14.4 s   | 8.22 s   | 43.0% | 1.76× |

Hit rate = 0.431 is latency-invariant. Wall savings = hit_rate. At
200 ms/fetch (representative of cold-cache S3), the cache saves
6.2 seconds on a 14.4 s workload.

### 3. N=50 multi-needle — honest sampling finds real retrieval gaps

Went from N=8 (handpicked, 0.81 partial recall) to N=50 across 8
topic clusters. Raw result: 10/50 full, 18/50 any_hit, 0.28 avg
partial recall. Per-cluster:

| cluster | n | full | any | avg_partial |
|---|---|---|---|---|
| A helix core | 10 | 1 | 2 | 0.15 |
| B launcher | 7 | 3 | 3 | 0.43 |
| C adapters | 6 | 0 | 0 | **0.00** |
| D claims | 6 | 0 | 0 | **0.00** |
| E fleet config | 7 | 1 | 4 | 0.36 |
| F biged ops | 7 | 4 | 7 | 0.79 |
| G benches | 4 | 1 | 1 | 0.25 |
| H cross | 3 | 0 | 1 | 0.17 |

Clusters C+D sit at 0.00 because the helix-context files shipped 2026-04-19
(adapters/, claims_graph.py, claims_analyze.py) were re-ingested via
`/ingest` but the `metadata.source_id` did not flow through to
`gene.path`/`gene.source`, so `/context` can't rank them. This is a
real ingest bug, not a retrieval regression. Excluding those two
clusters (unfair sample): adjusted avg_partial = 0.37 over 38 needles.

The honest headline: on a diverse query set, N=8's 0.81 does NOT
generalize. Retrieval has room to grow; "publishable" numbers need
the ingest-metadata bug fixed first.

### 4. Third-party retriever (Chroma) integration

Wrapped chromadb 1.5.8 behind the `Retriever` protocol via a
~30-LOC `ChromaRetriever` adapter. Harvested 162 gene contents from
the running genome, indexed into Chroma with MiniLM embeddings, ran
15 benchmark queries through two cells:

| cell | recall@10 | p50 | mean candidate space |
|---|---|---|---|
| raw_chroma | 0.43 | 141 ms | 162 (full) |
| helix_narrowed_chroma | 0.36 | 786 ms | 20 (8× narrower) |

Narrowing worked (162 → 20 candidates, 8× search-space reduction),
but recall dropped 7 pp and latency rose 5× from the packet-fetch
tax. Interpretation: on a small, focused index (162 docs), narrowing
is counterproductive because the corpus was already Helix-curated.
The SEMA bench showed +27% recall on a 6,682-doc index where
narrowing genuinely reduces noise. Rule of thumb: narrowing wins
when the underlying index is noisy; unscoped wins when it isn't.

**Protocol validation is the real win here** — production Chroma
slotted behind `Retriever` with no changes to helix-context. The
LlamaIndex / pgvector / Weaviate adapter pattern is the same.

## Open follow-ups

- **Ingest-metadata bug** — `/ingest` payload's `metadata.source_id` is
  stored but not exposed on `gene.path`/`gene.source`; anything ingested
  through the HTTP endpoint is invisible to path-token scoring. Fix
  before re-running N=50 for publishable numbers.
- **[Issue #8](https://github.com/SwiftWing21/helix-context/issues/8)**
  docs gap still open (SETUP.md extras matrix, TROUBLESHOOTING.md,
  Phase 2 README mention). Not blocking.

## Live state at session close

- Server up at :11437 (pushed + restarted multiple times across session)
- Grafana panels populating if OTel collector is running
- main.db holds 78,472 claims + 95,382 edges — DO NOT drop these,
  they represent 3+ hours of compute and enable the DAG walker
- Test totals: ~180 tests, all green (37 claims_graph + 35 dal/retriever
  + 15 claims_analyze + 19 headroom_supervisor + 77 existing)
- Working tree has 10 unstaged files NOT from my session (ribosome /
  launcher UI work by sibling agents) — leave untouched

## For future sessions

- **Read `docs/INTEGRATING_WITH_EXISTING_RAG.md` first** if you're
  touching retrieval/adapter code. It's the authoritative
  composition guide now.
- **Don't treat Helix as half a stack** — it's the router above.
  The packet fields dispatch to RAG/DAG/DAL layers; Helix doesn't
  execute, it routes.
- **Don't re-measure DAG on content recall** — that bench is
  concluded (0.81 vs 0.81). Measure DAG on stale-claim avoidance or
  decision-quality metrics.
- **Adapters live in `helix_context/adapters/` as opt-in references.**
  They're meant to be copied / subclassed / swapped, not treated as
  core Helix dependencies.

— Laude, 2026-04-19

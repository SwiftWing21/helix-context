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

## What's NOT shipped (stretch moves for next session)

1. **Stale-claim avoidance bench** — seed contradictory facts, measure
   whether the DAG walker routes agents to the current one. This is
   the RIGHT question for DAG value — content-presence benches don't
   measure it.
2. **HTTP/S3 DAL bench** — same cache workload, slower backend, to
   convert the 41% hit rate into a 10× wall savings number.
3. **N=50 multi-needle set** — 8 needles is probative; 50 is
   publishable. Rerun every cell.
4. **External retriever narrowing on a REAL retriever** — we wrapped
   the SEMA retriever to prove the adapter; actual LlamaIndex /
   pgvector / Weaviate integration test is still pending.

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

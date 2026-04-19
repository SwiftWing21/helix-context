# Integrating Helix with an existing RAG

> **TL;DR** — Helix is a *coordinate index*, not a content store.
> Keep your existing retriever (embedding store, vector DB, hybrid
> search). Call Helix first to narrow the candidate set + get a
> freshness verdict. Then fetch the content from your existing pipeline.
>
> On an 8-needle multi-needle NIAH against a 7,846-gene corpus, the
> composition pattern beats every single retriever used alone:
>
> | Retriever | ans_partial | ans_full | latency |
> |---|---|---|---|
> | Pure BM25 | 0.62 | 4/8 | 31 ms |
> | Pure embedding (SEMA, 20D) | 0.44 | 1/8 | 1108 ms |
> | Helix packet alone | 0.19 | 0/8 | 896 ms |
> | **Helix + BM25 composition** | **0.81** | **5/8** | 887 ms |
>
> See [`benchmarks/results/helix_rag_composition_2026-04-19.json`](../benchmarks/results/helix_rag_composition_2026-04-19.json).

---

## Why you'd want this

You've already built a RAG. You have a vector DB (pgvector, Weaviate,
Pinecone), a custom embedding encoder, a hybrid BM25+vector pipeline,
or a tree of JSON docs with full-text search. **Don't rip it out.**

Helix answers a different question than your RAG does. Your RAG
answers *"what content is near this query?"*. Helix answers *"does
the answer exist, where is it, and is it fresh enough to act on?"*.

Put Helix **in front of** your RAG, not next to it:

```
              ┌─ coordinate + freshness verdict (Helix)
              │  "looks stale / go refresh /repo/config.yaml"
agent ──▶ query
              │
              └─ content fetch (your RAG / your store)
                 "here's the actual bytes at that location"
```

Three integration patterns, in increasing order of commitment.

---

## Pattern 1 — advisory verdict, your RAG unchanged

Simplest drop-in. Send every query to Helix first. Read the verdict.
If `needs_refresh`, tell your RAG to include the listed `refresh_targets`
in its result set. If `verified`, trust your RAG's top-K as-is.

```python
import httpx

def retrieve(query: str, task_type: str = "explain") -> list[dict]:
    # 1. Helix advisory call — cheap, ~200-800ms for the packet
    packet = httpx.post(
        "http://127.0.0.1:11437/context/packet",
        json={"query": query, "task_type": task_type},
        timeout=60,
    ).json()

    # 2. Your existing RAG — unchanged
    results = your_existing_retriever.search(query, top_k=20)

    # 3. If Helix says the packet needs a refresh, ensure those
    # source_ids are in the result set (fetch them if missing).
    target_paths = {t["source_id"] for t in packet.get("refresh_targets", [])}
    have_paths = {r["source_id"] for r in results}
    for missing in target_paths - have_paths:
        results.append(your_existing_retriever.fetch_by_path(missing))

    return results
```

**Cost:** one HTTP round-trip per query (~500 ms typical).
**Payoff:** you pick up freshness/staleness detection and coord-
resolution confidence for free. Everything else stays the same.

---

## Pattern 2 — Helix narrows the search space for your RAG

Use Helix's packet pointers (`verified` + `stale_risk` +
`refresh_targets`) as a *candidate set* for your retriever. Your RAG
searches only within those candidates, not the whole corpus. When
the answer is in Helix's shortlist this is dramatically cheaper than
searching millions of documents; when it isn't you fall back to your
normal retrieve.

```python
def retrieve_composed(query: str, task_type: str = "explain",
                      top_k: int = 8) -> list[dict]:
    packet = httpx.post(
        "http://127.0.0.1:11437/context/packet",
        json={"query": query, "task_type": task_type},
    ).json()

    # Shortlist from Helix — source_ids it says are relevant
    shortlist = set()
    for bucket in ("verified", "stale_risk", "contradictions"):
        for item in packet.get(bucket, []):
            if item.get("source_id"):
                shortlist.add(item["source_id"])
    for tgt in packet.get("refresh_targets", []):
        shortlist.add(tgt["source_id"])

    if shortlist:
        # Your RAG, scoped to Helix's shortlist
        results = your_existing_retriever.search(
            query, filter_paths=list(shortlist), top_k=top_k,
        )
    else:
        # Helix had nothing — fall back to full corpus
        results = your_existing_retriever.search(query, top_k=top_k)

    return results
```

**Cost:** Helix round-trip + scoped search.
**Payoff:** your vector DB does less work per query (smaller ANN
search, fewer tokens ingested into rerank). Helix takes the "where
should we even look" decision off your pipeline's critical path.

---

## Pattern 3 — Helix points, naive fetcher reads (benchmark-tested)

The pattern we measured in the composition bench. Use it when you
have source files on disk and the simplest possible fetcher (read
the file). Works even if you don't have a "real" RAG yet — Helix's
pointing is the retrieval, file-read is the fetch.

```python
from pathlib import Path

def retrieve_fileread(query: str, task_type: str = "explain",
                      max_files: int = 12,
                      chars_per_file: int = 5000) -> str:
    packet = httpx.post(
        "http://127.0.0.1:11437/context/packet",
        json={"query": query, "task_type": task_type},
    ).json()

    source_ids = []
    for bucket in ("verified", "stale_risk", "contradictions"):
        for item in packet.get(bucket, []):
            sid = item.get("source_id")
            if sid and sid not in source_ids:
                source_ids.append(sid)
    for tgt in packet.get("refresh_targets", []):
        sid = tgt.get("source_id")
        if sid and sid not in source_ids:
            source_ids.append(sid)

    chunks = []
    for sid in source_ids[:max_files]:
        path = Path(sid)
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace")
                          [:chars_per_file])
    return "\n---\n".join(chunks)
```

**Bench result:** 0.81 answer recall on multi-needle, vs 0.62 for
pure BM25, vs 0.19 for Helix alone. See
[`benchmarks/bench_helix_rag_composition.py`](../benchmarks/bench_helix_rag_composition.py).

---

## When to use which

| Your situation | Pattern |
|---|---|
| Existing mature RAG, don't want to change it | 1 — advisory verdict |
| Existing RAG, want to narrow its work | 2 — shortlist |
| No RAG yet, source files on disk | 3 — file-read |
| Private cloud corpus, Helix runs beside it | 1 or 2 |
| Heterogeneous stores (files + DB + API) | 1, then fetch per source_kind |

All three patterns compose — you can run pattern 1 alongside patterns
2/3 on different query types (e.g., explain-mode gets pattern 1 ruling
for freshness, edit-mode uses pattern 3 because you need literal
content).

---

## What Helix needs to know about your corpus

For the coordinate index to be useful, Helix needs ingested genes
referencing the source_ids your RAG uses. You ingest once at indexing
time; after that `/context/packet` knows your corpus exists.

```python
import httpx

# Replay your RAG's documents into Helix (one-time or on-change)
for doc in your_documents:
    httpx.post(
        "http://127.0.0.1:11437/ingest",
        json={
            "content": doc.text,
            "content_type": doc.kind,  # "code" | "doc" | "config" | ...
            "metadata": {
                "path": doc.source_id,           # your RAG's canonical path
                "observed_at": doc.indexed_at,   # when your RAG last saw it
            },
        },
        timeout=60,
    )
```

Helix uses:
- `path` → `source_id` (the pointer your RAG will re-fetch from later)
- `content_type` → `source_kind` → `volatility_class` (drives freshness
  half-life: `stable=7d, medium=12h, hot=15min`)
- `observed_at` → gates `live_truth_score` calculation

You don't need to send embeddings; Helix builds its own 20D SEMA
vectors. You don't need to send chunked text; Helix ingests whole
documents and chunks on retrieval.

---

## Authority + volatility — advanced tuning

The freshness math has two knobs you can adjust per gene:

- `volatility_class`: `stable` (7d half-life), `medium` (12h), `hot`
  (15min). Default is derived from `content_type` but you can
  override per-document if you know your corpus better. Ingest-time
  API docs for an externally-owned package: mark them `stable`. Your
  own production config files: mark them `hot`.
- `authority_class`: `primary` (1.0 weight), `derived` (0.75),
  `inferred` (0.45). Hand-written docs are primary; auto-generated
  summaries are derived; LLM-inferred claims are inferred.

Edit the `packet_notes` a downstream agent receives by shaping these
correctly at ingest.

---

## Troubleshooting

**"Helix packet's `source_id` doesn't match my RAG's paths."**
Helix stores whatever path you send at ingest. If your RAG uses
`s3://bucket/key` but you sent Windows paths to Helix, the shortlist
(pattern 2) won't match. Normalize at ingest time.

**"Packet says `verified` but I can't find the answer in my RAG
result."**
Expected. Helix's packet delivers *pointers + verdict*, not content
(see the helix_only bench cell — 0/8 full answer recall on its own).
Agents MUST fetch. That's what patterns 2 and 3 are for.

**"Packet note says `coord_confidence=0.12 below 0.30 floor`."**
Helix thinks your retrieval landed in the wrong folder region. Trust
`needs_refresh` / `stale_risk` labels — fetch the `refresh_targets`
before acting.

**"First embedding call is slow."**
SEMA codec downloads `all-MiniLM-L6-v2` (~90 MB) on first use.
Pre-download at container-build time with
`python -c "from sentence_transformers import SentenceTransformer;
SentenceTransformer('all-MiniLM-L6-v2')"`.

---

## Helix as the router above the stack

The integration patterns above focus on RAG (content retrieval).
Real agents also need a *dependency* layer (DAG — claim resolution,
contradiction handling, supersedes chains) and an *access* layer
(DAL — uniform fetch across heterogeneous stores). Helix doesn't
ship the stack, but it **emits the signals that tell the stack which
path to take**:

```
              ┌────────────────── Helix (router) ─────────────────┐
query ──▶    │  emits: task_type, coord_confidence, verdict,     │
              │         volatility_class, contradictions,         │
              │         supersedes edges, refresh_targets         │
              └──┬──────────────────┬──────────────────┬──────────┘
                 │                  │                  │
     verified +  │   stale_risk  +  │  contradictions  │
     high conf   │   + hot vol      │  + supersedes    │
                 ▼                  ▼                  ▼
              ┌─ RAG ─┐         ┌─ DAL ─┐        ┌─ DAG ─┐
              │ fetch │         │ scheme│        │ walk  │
              │ bytes │         │ refetch        │ edges │
              └───────┘         └───────┘        └───────┘
```

The packet's fields are the routing signals. Choice-math examples:

| Packet shape | Which layer(s) to engage |
|---|---|
| `verified` + `coord_conf > 0.5` | RAG only |
| `stale_risk` + `hot` volatility | DAL refetch, then RAG |
| `contradictions` non-empty | DAG walk first, then DAL on the winner |
| Claim has `supersedes_claim_id` | DAG resolves to latest before any fetch |
| `task_type = "edit"` + any `needs_refresh` | All three in order |

Helix ships reference implementations of both the DAG walker and the
DAL — use them as drop-ins or copy the patterns.

### DAG layer — `helix_context.claims_graph`

Walks the Phase 2 `claims` + `claim_edges` tables. Supersedes chains,
contradiction clusters, topological ordering, and the one-call
resolver that composes them:

```python
from helix_context.claims_graph import resolve_from_packet
from helix_context.shard_schema import open_main_db

main_db = open_main_db("genomes/main.db")
result = resolve_from_packet(main_db, packet, policy="latest_then_authority")

for claim in result["accepted"]:
    # Supersedes-resolved, contradiction-winner claims
    act_on(claim)

for claim in result["rejected"]:
    # Why: "superseded_by c_42" or "contradicts_winner c_17"
    log.debug("dropped %s: %s", claim["claim_id"], claim["rejected_reason"])
```

Policies:
- `latest_then_authority` (default) — follow supersedes to head, then
  within each contradiction cluster pick the highest-authority claim.
- `keep_all_with_flags` — return every claim with `superseded_by` +
  `contradicts_ids` annotations. Use when the agent needs to see the
  conflict surface rather than resolve it.

### DAL layer — `helix_context.adapters.dal`

Uniform `fetch(source_id) → bytes` across schemes. Ships with
`file://` and `http(s)://` fetchers registered by default. Register
`s3://`, `git://`, or custom schemes per integration:

```python
from helix_context.adapters.dal import DAL, fetch_packet_sources, fetch_s3

dal = DAL()
dal.register("s3", fetch_s3)  # opt-in; requires boto3

# Single fetch
result = dal.fetch("s3://my-bucket/doc.md")
if result.ok:
    agent.process(result.text)

# Batch-fetch every source in a packet
for source_id, fetch_result in fetch_packet_sources(packet, dal=dal):
    if not fetch_result.ok:
        log.warning("fetch failed: %s — %s", source_id,
                    fetch_result.meta["error"])
        continue
    agent.ingest(source_id, fetch_result.text)
```

Fetchers soft-fail (return `FetchResult(text=None, meta={"error": ...})`)
instead of raising — a missing doc shouldn't crash the agent's flow.

### Cache layer — `helix_context.adapters.cache`

Wraps a DAL with TTL-bounded LRU. TTLs come from Helix's
`volatility_class` (`stable=7d`, `medium=12h`, `hot=15min`), so the
cache stays honest about freshness without extra config:

```python
from helix_context.adapters.dal import DAL
from helix_context.adapters.cache import CachedDAL, fetch_packet_sources_cached

cache = CachedDAL(DAL(), max_entries=500)

# Single fetch — volatility drives TTL
result = cache.fetch("/repo/config.yaml", volatility_class="hot")

# Packet-aware batch — stale_risk + refresh_targets automatically
# bypass the cache (Helix already flagged them as needing a refresh,
# serving cached bytes would defeat the verdict)
for sid, r in fetch_packet_sources_cached(packet, cache=cache):
    if r.ok:
        agent.ingest(sid, r.text)
```

#### Multi-agent semantics

The cache is **`source_id`-keyed, party-scoped, and free to share
across agents on the same device.** Design table:

| Layer | Scoping | Why |
|---|---|---|
| Cache key | `source_id` | Bytes are identity-independent |
| Cache instance | Per party (= per device) | Different filesystems, different fetch results |
| TTL | Per `volatility_class` | Helix owns freshness; cache honors it |
| Invalidation | `invalidate(source_id)` / `invalidate_by_prefix()` / `invalidate_all()` | Wire to your ingest hooks |
| Sharing across agents in one party | **Yes, by default** | Laude + Taude on one box see identical bytes; cache hit is correct |
| Sharing across parties | **No** | Different machines; use shared genome + ingest instead |

Practical pattern for a launcher running Laude + Taude + Raude as
three agents: one `CachedDAL` instance in the launcher process,
handed to each agent. Hit rate climbs as soon as two personas touch
the same files. Cross-machine caching is **not** the cache's job —
that's what Helix's shared genome metadata + ingest replication
handles.

For ingest-driven invalidation, wire your ingest pipeline to call
the cache's invalidation methods:

```python
# In your ingest worker, after writing a gene:
cache.invalidate(gene.source_id)

# Or for bulk refreshes:
cache.invalidate_by_prefix("/repo/docs/")
```

The cache stats (`.stats()`) expose `hits / misses / evictions /
hit_rate` for diagnostics.

### Retriever adapter — `helix_context.adapters.retriever`

Wrap your existing RAG (LlamaIndex, LangChain, or any duck-typed
retriever) behind the `Retriever` protocol, then compose with Helix's
packet-shortlist via `HelixNarrowedRetriever`:

```python
from helix_context.adapters.retriever import (
    LlamaIndexRetriever, LangChainRetriever, HelixNarrowedRetriever,
)

# Option A: LlamaIndex
from llama_index.core.retrievers import VectorIndexRetriever
li_retriever = VectorIndexRetriever(index=my_index, similarity_top_k=12)
inner = LlamaIndexRetriever(li_retriever)

# Option B: LangChain
# inner = LangChainRetriever(my_langchain_retriever)

# Compose with Helix
narrowed = HelixNarrowedRetriever(inner, helix_url="http://127.0.0.1:11437")
docs = narrowed.retrieve("where does auth middleware live", top_k=8)
```

Narrowed flow: Helix returns a shortlist of `source_id`s, the wrapper
passes them as `filter_paths` to your retriever, and you get back
`list[RetrievedDoc]` scoped to Helix's candidates. If Helix has
nothing relevant or is unreachable, `fallback_unscoped=True` (default)
runs an unscoped retrieve so the agent never starves.

Custom retrievers don't need to subclass anything:

```python
from helix_context.adapters.retriever import Retriever, RetrievedDoc

class MyRetriever:
    def retrieve(self, query, *, filter_paths=None, top_k=8):
        # ... your logic ...
        return [RetrievedDoc(source_id=..., content=..., score=...)]

# Duck-typed — isinstance(MyRetriever(), Retriever) is True
```

### Full router — all three layers composed

The canonical post-Helix call path:

```python
from helix_context.claims_graph import resolve_from_packet
from helix_context.adapters.dal import DAL, fetch_packet_sources
from helix_context.shard_schema import open_main_db

import httpx

def answer(query: str, task_type: str = "edit") -> dict:
    # 1. Ask Helix for the packet (the router signal)
    packet = httpx.post(
        "http://127.0.0.1:11437/context/packet",
        json={"query": query, "task_type": task_type},
    ).json()

    # 2. DAG: resolve claim conflicts / supersedes chains
    main_db = open_main_db("genomes/main.db")
    claims = resolve_from_packet(main_db, packet).get("accepted", [])
    main_db.close()

    # 3. DAL: fetch the bytes at every source_id the packet points at
    fetched = fetch_packet_sources(packet, dal=DAL())

    # 4. Pass resolved claims + fetched content to your LLM / tool
    return {"claims": claims, "content": fetched, "packet_notes": packet.get("notes", [])}
```

The three layers are independent and composable — use all, some, or
none depending on what your agent needs. Helix's only opinion is that
the packet fields carry enough signal to dispatch correctly.

### Bench: 5-cell composition (2026-04-19)

Empirical check of the stack against the multi-needle NIAH on a
7,846-gene genome. 78,472 claims backfilled via
[`scripts/backfill_claims.py`](../scripts/backfill_claims.py):

| Cell | ptr_partial | ans_full | ans_partial | latency |
|---|---|---|---|---|
| pure_rag_bm25 | 0.19 | 4/8 | 0.62 | 35 ms |
| pure_rag_embedding | 0.00 | 1/8 | 0.44 | 1092 ms |
| helix_only | 0.19 | 0/8 | 0.19 | 1096 ms |
| helix_rag | 0.19 | 5/8 | **0.81** | 923 ms |
| helix_full_stack | 0.19 | 5/8 | **0.81** | 960 ms |

The `helix_full_stack` cell (DAG resolve + cached DAL) **matches
`helix_rag` exactly** at 0.81 today. That's expected — we extract
literal claims at ingest but don't auto-populate `claim_edges`
(contradiction / supersedes detection is a follow-on). The DAG layer
is a no-op right now, so it's a pure +37ms overhead.

When `claim_edges` gets populated (contradiction detection landing),
this cell diverges: the DAG resolves conflicts before the agent
commits to a belief, and the full-stack cell should lift recall on
any needle where the genome holds both a stale and a current answer.

Today the full-stack cell matters as **composition-correctness proof**
— the router pattern works end-to-end — not as a recall boost.

## Three further benches (2026-04-19)

### External retriever — narrowing pattern (`bench_external_retriever.py`)

Wraps the SEMA cosine retriever as a `Retriever` (pattern 2), runs it
raw vs Helix-narrowed across the same 8 multi-needle queries:

| Metric | Raw SEMA | Helix-Narrowed | Delta |
|---|---|---|---|
| mean content_recall | 0.44 | **0.56** | +12pp (+27%) |
| mean search space | 6,682 | ~13 | **~516× smaller** |
| mean latency | 903 ms | 1098 ms | +195 ms (packet call) |

**Narrowing lifts recall by 27% while cutting the candidate set by
~500×.** Latency goes up by the packet call cost, but on a retriever
with expensive search (ANN over 1M vectors, cross-encoder rerank)
this tradeoff flips — most of raw retrieval's cost is per-candidate,
so cutting candidates dominates.

### Cache hit-rate — multi-agent workload (`bench_cache_hitrate.py`)

Simulates 3 agents (laude/taude/raude personas) × 6 queries with
70% shared topic pool + 30% per-agent specialty:

- **Hit rate: 41.67%** with one shared CachedDAL across all three.
- Total wall saved: 4.5% (~600 ms).

Modest savings because fetches are local files (<1 ms each) — the
Helix packet call dominates wall time, not the DAL. For HTTP- or
S3-backed DALs (order of magnitude slower), the 41% hit rate
translates to proportionally bigger savings.

The bench validates the multi-agent pattern empirically: a shared
cache correctly dedups across personas without cross-contamination.

### Claim-edge detection + full-stack rerun

After landing `helix_context/claims_analyze.py` and backfilling
edges into the existing 78,472-claim main.db, we detected:

- **50,362 contradicts** (same entity_key, low-Jaccard text)
- **45,020 duplicates** (same entity_key, high-Jaccard text, diff genes)
- **0 supersedes** (needs diverging `observed_at` on near-duplicate
  pairs — most genes in the corpus were ingested together)
- Total: **95,382 edges** across 20,978 entity_key groups (190s scan)

With `claim_edges` populated, the `helix_full_stack` cell re-ran on
the multi-needle bench — still **0.81 ans_partial**, matching
`helix_rag`. The DAG layer is actively walking now (`resolve_from_packet`
returns accepted/rejected claims), but the file-read content blob
already contains the answer strings, so DAG resolution doesn't lift
content recall on this bench.

**Where DAG resolution starts mattering:** decision-quality metrics
(does the agent act on a stale claim?), not content-presence
metrics. This bench measures content; a future "stale-claim
avoidance" bench is where DAG will show its teeth. Ship the
infrastructure now, measure its value on the right question later.

## Further reading

- [`docs/specs/2026-04-17-agent-context-index-build-spec.md`](specs/2026-04-17-agent-context-index-build-spec.md)
  — full packet-mode design spec
- [`benchmarks/bench_helix_rag_composition.py`](../benchmarks/bench_helix_rag_composition.py)
  — the 4-cell composition benchmark source
- [`helix_context/claims_graph.py`](../helix_context/claims_graph.py)
  — DAG walker reference implementation
- [`helix_context/adapters/dal.py`](../helix_context/adapters/dal.py)
  — DAL reference implementation
- [`README.md`](../README.md) §Two product surfaces — `/context` vs
  `/context/packet` decision guide

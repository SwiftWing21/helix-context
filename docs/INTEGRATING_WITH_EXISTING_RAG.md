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

## Further reading

- [`docs/specs/2026-04-17-agent-context-index-build-spec.md`](specs/2026-04-17-agent-context-index-build-spec.md)
  — full packet-mode design spec
- [`benchmarks/bench_helix_rag_composition.py`](../benchmarks/bench_helix_rag_composition.py)
  — the 4-cell composition benchmark source
- [`README.md`](../README.md) §Two product surfaces — `/context` vs
  `/context/packet` decision guide

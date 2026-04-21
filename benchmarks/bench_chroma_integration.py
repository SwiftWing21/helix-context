"""Helix + Chroma — third-party retriever integration bench.

Validates the `Retriever` adapter against a *real* production RAG store
(Chroma 1.x with MiniLM embeddings), not just our internal SEMA wrapper.
This closes the gap left by `bench_external_retriever.py`, which proved
the adapter protocol works but only with a helix-internal retriever.

Setup:
    1. Index ~200 real gene contents fetched from the running genome
       (`/context/packet` with a broad query) into an EphemeralClient Chroma
       collection under the default all-MiniLM-L6-v2 embedding.
    2. Wrap the collection in a `ChromaRetriever` that conforms to the
       helix-context `Retriever` protocol (retrieve(query, filter_paths,
       top_k) -> list[RetrievedDoc]).
    3. For a set of benchmark queries, compare:
         - Raw Chroma (no helix) — baseline
         - HelixNarrowedRetriever(raw=Chroma) — Helix packet narrows the
           candidate space via filter_paths before Chroma scores.

Metrics:
    - recall@K — gold path appears in top-K
    - candidate_space — docs Chroma actually scored (shows narrowing ratio)
    - latency_ms — wall per query

Usage:
    python benchmarks/bench_chroma_integration.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

# Silence Chroma's ONNX download progress bars on stderr for clean output
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("CHROMA_TELEMETRY", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chromadb  # noqa: E402
import httpx  # noqa: E402

from helix_context.adapters.retriever import (  # noqa: E402
    HelixNarrowedRetriever,
    RetrievedDoc,
    Retriever,
)


HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
INDEX_SIZE_TARGET = 200


# ── Chroma retriever conforming to Helix Retriever protocol ─────────


class ChromaRetriever:
    """Adapter: wrap a chromadb Collection behind the Retriever protocol.

    Supports the optional ``filter_paths`` kwarg so Helix's narrowing
    pattern can pre-filter the candidate space by source_id.
    """

    def __init__(self, collection):
        self.collection = collection
        self.last_candidate_space = None

    def retrieve(
        self,
        query: str,
        *,
        filter_paths=None,
        top_k: int = 10,
    ) -> list[RetrievedDoc]:
        where = None
        if filter_paths:
            # Chroma's metadata filter: match any source_id in the list
            filter_list = list(filter_paths)
            self.last_candidate_space = len(filter_list)
            where = {"source_id": {"$in": filter_list}}
        else:
            self.last_candidate_space = self.collection.count()

        r = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )
        docs = []
        for i, doc_id in enumerate(r["ids"][0]):
            meta = r["metadatas"][0][i] if r.get("metadatas") else {}
            text = r["documents"][0][i] if r.get("documents") else ""
            dist = r["distances"][0][i] if r.get("distances") else None
            score = (1.0 / (1.0 + dist)) if dist is not None else 0.0
            docs.append(
                RetrievedDoc(
                    source_id=meta.get("source_id") or doc_id,
                    content=text,
                    score=score,
                    metadata=meta,
                )
            )
        return docs


def _protocol_check():
    coll = chromadb.EphemeralClient().get_or_create_collection("check")
    cr = ChromaRetriever(coll)
    assert isinstance(cr, Retriever), "ChromaRetriever violates Retriever protocol"


# ── Corpus construction from live Helix ─────────────────────────────


def harvest_corpus(client: httpx.Client, target_size: int) -> list[dict]:
    """Pull gene contents from running Helix to seed Chroma.

    We query a broad set of topics, collect unique gene contents + source_ids,
    stop when we have ``target_size`` docs.
    """
    seed_queries = [
        "helix packet verdict task_type",
        "claim extraction supersedes",
        "cache TTL volatility",
        "retriever protocol",
        "fleet dashboard port",
        "session handoff",
        "tray headroom",
        "dal scheme dispatch",
        "BM25 embedding retrieval",
        "benchmark multi needle",
        "ingest pipeline",
        "shard schema register",
        "confidence floor",
        "agent registry identity",
        "audit scorer",
        "ribosome paused",
        "dashboard endpoints",
        "smoke test fleet",
        "worker scaling ram tier",
        "db retry write",
    ]
    seen_sources: set[str] = set()
    docs: list[dict] = []

    for q in seed_queries:
        if len(docs) >= target_size:
            break
        try:
            r = client.post(
                f"{HELIX_URL}/context/packet",
                json={
                    "query": q,
                    "task_type": "explain",
                    "top_k": 30,
                    "read_only": True,
                },
                timeout=60,
            ).json()
        except Exception as exc:
            print(f"  harvest {q!r}: {exc}")
            continue
        for bucket in ("verified", "stale_risk", "contradictions"):
            for item in r.get(bucket) or []:
                sid = item.get("source_id")
                content = item.get("content") or item.get("preview") or ""
                if not sid or sid in seen_sources:
                    continue
                if not content or len(content) < 40:
                    continue
                seen_sources.add(sid)
                docs.append({
                    "source_id": sid,
                    "content": content[:2000],
                    "seed_query": q,
                })
                if len(docs) >= target_size:
                    break
    return docs


def build_chroma_collection(docs: list[dict]):
    client = chromadb.EphemeralClient()
    # Unique collection name — EphemeralClient is per-process but be safe
    coll = client.get_or_create_collection(f"helix_bench_{int(time.time())}")
    # Chroma requires string-safe IDs; hash source_id
    import hashlib
    ids = [hashlib.md5(d["source_id"].encode()).hexdigest() for d in docs]
    coll.add(
        documents=[d["content"] for d in docs],
        metadatas=[{"source_id": d["source_id"]} for d in docs],
        ids=ids,
    )
    return coll


# ── Benchmark queries ────────────────────────────────────────────────


BENCH_QUERIES = [
    {"query": "what port does helix listen on", "gold_substring": "helix.toml"},
    {"query": "how does claim extraction work", "gold_substring": "claims.py"},
    {"query": "what is the supersedes chain walker", "gold_substring": "claims_graph"},
    {"query": "cache TTL volatility classes", "gold_substring": "cache.py"},
    {"query": "DAL scheme dispatch file http s3", "gold_substring": "dal.py"},
    {"query": "headroom supervisor orphan adoption", "gold_substring": "headroom"},
    {"query": "fleet worker ram tier scaling", "gold_substring": "CLAUDE.md"},
    {"query": "SKILL_NAME contract", "gold_substring": "skill"},
    {"query": "db retry write WAL", "gold_substring": "db"},
    {"query": "dashboard endpoints blueprint", "gold_substring": "dashboard"},
    {"query": "packet verified stale_risk contradictions", "gold_substring": "context"},
    {"query": "retriever protocol LlamaIndex", "gold_substring": "retriever"},
    {"query": "ingest pipeline steps", "gold_substring": ""},
    {"query": "bench multi needle NIAH", "gold_substring": "bench"},
    {"query": "ribosome paused by default", "gold_substring": "ribosome"},
]


# ── Helix shortlist from packet ──────────────────────────────────────


def helix_shortlist(client: httpx.Client, query: str) -> list[str]:
    try:
        r = client.post(
            f"{HELIX_URL}/context/packet",
            json={
                "query": query,
                "task_type": "explain",
                "top_k": 30,
                "read_only": True,
            },
            timeout=60,
        ).json()
    except Exception:
        return []
    sources = []
    for bucket in ("verified", "stale_risk", "contradictions"):
        for item in r.get(bucket) or []:
            sid = item.get("source_id")
            if sid and sid not in sources:
                sources.append(sid)
    return sources


def score(docs: list[RetrievedDoc], gold_substring: str) -> int:
    if not gold_substring:
        return 0
    needle = gold_substring.lower()
    for i, d in enumerate(docs):
        if needle in (d.source_id or "").lower():
            return 1
    return 0


# ── Runner ───────────────────────────────────────────────────────────


def run_cell(label: str, retriever: Retriever, queries: list[dict],
             helix_client: httpx.Client = None) -> dict:
    per_q = []
    hits = 0
    latencies = []
    candidate_spaces = []
    for q in queries:
        gold = q["gold_substring"]
        query_text = q["query"]
        t0 = time.perf_counter()
        if label.startswith("helix_narrowed"):
            docs = retriever.retrieve(query_text, top_k=10)
        else:
            docs = retriever.retrieve(query_text, top_k=10)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        if isinstance(retriever, HelixNarrowedRetriever):
            cs = getattr(retriever._inner, "last_candidate_space", None)
        else:
            cs = getattr(retriever, "last_candidate_space", None)
        if cs is not None:
            candidate_spaces.append(cs)
        hit = score(docs, gold) if gold else None
        if hit:
            hits += 1
        per_q.append({
            "query": query_text,
            "gold": gold,
            "hit": hit,
            "latency_ms": round(dt, 2),
            "top_sources": [d.source_id for d in docs[:3]],
            "candidate_space": cs,
        })
    n_scorable = sum(1 for q in per_q if q["gold"])
    return {
        "label": label,
        "per_query": per_q,
        "hits": hits,
        "n_scorable": n_scorable,
        "recall_at_10": hits / n_scorable if n_scorable else None,
        "p50_latency_ms": round(statistics.median(latencies), 2),
        "mean_latency_ms": round(statistics.fmean(latencies), 2),
        "mean_candidate_space": (
            round(statistics.fmean(candidate_spaces), 1) if candidate_spaces else None
        ),
    }


def main():
    _protocol_check()

    client = httpx.Client(timeout=120)
    try:
        stats = client.get(f"{HELIX_URL}/stats").json()
    except Exception as exc:
        print(f"Cannot reach Helix at {HELIX_URL}: {exc}")
        return 1
    print(f"Genome: {stats.get('total_genes')} genes")

    print(f"Harvesting corpus (~{INDEX_SIZE_TARGET} docs from Helix)...")
    docs = harvest_corpus(client, INDEX_SIZE_TARGET)
    print(f"  collected {len(docs)} unique gene contents")

    print("Indexing into Chroma (MiniLM embeddings)...")
    t_idx = time.perf_counter()
    coll = build_chroma_collection(docs)
    idx_ms = (time.perf_counter() - t_idx) * 1000.0
    print(f"  indexed in {idx_ms:.0f} ms ({coll.count()} docs)")

    raw = ChromaRetriever(coll)
    narrowed = HelixNarrowedRetriever(
        inner=raw,
        helix_url=HELIX_URL,
        read_only=True,
    )

    print("\nRunning benchmark cells...")
    raw_stats = run_cell("raw_chroma", raw, BENCH_QUERIES)
    narrowed_stats = run_cell("helix_narrowed_chroma", narrowed, BENCH_QUERIES, client)

    print("\n-- Results ---------------------------------------------------")
    for s in (raw_stats, narrowed_stats):
        rec = s["recall_at_10"]
        print(f"[{s['label']}]  recall@10={rec:.2f}  "
              f"p50={s['p50_latency_ms']}ms  mean={s['mean_latency_ms']}ms  "
              f"candidate_space~{s['mean_candidate_space']}")

    out = {
        "genome_total_genes": stats.get("total_genes"),
        "corpus_size": len(docs),
        "chroma_index_ms": round(idx_ms, 1),
        "n_queries": len(BENCH_QUERIES),
        "cells": {"raw_chroma": raw_stats, "helix_narrowed_chroma": narrowed_stats},
    }
    out_path = REPO_ROOT / "benchmarks" / "results" / "chroma_integration_2026-04-19.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

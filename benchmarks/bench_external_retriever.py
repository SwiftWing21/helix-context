"""External retriever bench — measures what HelixNarrowedRetriever
adds when composed with a real external retriever.

We wrap the existing SEMA-embedding retriever as a ``Retriever`` and
run it both raw (unscoped over the full 7846-gene genome) and
narrowed (scoped to Helix's packet shortlist). The bench measures:

    - retrieval_candidate_count: how many docs the retriever
      searched over (raw = all genes; narrowed = ~12)
    - answer_recall_at_k: did the expected answer appear in the top-K
    - latency

This is pattern 2 in the integration doc. The value proposition is
NOT that narrowing lifts recall — BM25 already shows that a good
retriever hits respectable recall across the full corpus. The value
is that narrowing cuts the SEARCH SPACE by ~650x (7846 / ~12), which
matters for expensive retrievers (ANN with high `ef_search`,
cross-encoder rerank, etc.).

Usage:
    python benchmarks/bench_external_retriever.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from helix_context.adapters.retriever import (  # noqa: E402
    HelixNarrowedRetriever, RetrievedDoc, Retriever,
)

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
GENOME_PATH = os.environ.get(
    "HELIX_GENOME_PATH",
    str(Path(__file__).resolve().parents[1] / "genomes" / "main" / "genome.db"),
)


# Reuse the multi-needle needles (with expected answers) from the
# composition bench. Inline so this bench is self-contained.
NEEDLES = [
    {"name": "helix_and_headroom_ports",
     "query": "what ports do helix and headroom listen on",
     "expected": ["11437", "8787"]},
    {"name": "python_version_and_codec_extra",
     "query": "python version helix requires and extra that enables headroom",
     "expected": ["3.11", "codec"]},
    {"name": "pipeline_steps_and_compression_target",
     "query": "steps in helix pipeline and target compression ratio",
     "expected": ["6", "5x"]},
    {"name": "claim_types_and_spec_source",
     "query": "claim_type allowed values helix claims layer specification",
     "expected": ["path_value", "agent-context-index"]},
    {"name": "headroom_port_and_mode_default",
     "query": "headroom dashboard port default compression mode",
     "expected": ["8787", "token"]},
    {"name": "freshness_half_lives_stable_and_hot",
     "query": "freshness half-life stable hot volatility",
     "expected": ["7d", "15min"]},
    {"name": "coord_floor_and_file_grain_floor",
     "query": "coordinate confidence floor file-grain coverage floor",
     "expected": ["0.30", "0.15"]},
    {"name": "helix_port_and_fleet_port",
     "query": "helix listen port bigEd fleet dashboard port",
     "expected": ["11437", "5555"]},
]


# ── External retriever: SEMA cosine over genome.genes.embedding ─────


_SEMA_CODEC = None


def _get_codec():
    global _SEMA_CODEC
    if _SEMA_CODEC is None:
        from helix_context.sema import SemaCodec
        _SEMA_CODEC = SemaCodec()
    return _SEMA_CODEC


def _cosine(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class SemaEmbeddingRetriever:
    """A real external retriever that conforms to the Retriever protocol.

    This is what an integrator would wrap around their LlamaIndex /
    LangChain / custom retriever — just meets the duck-typed protocol.
    Uses the 20D SEMA embeddings already stored in the genome as its
    vector space, so we don't need to rebuild the index.
    """

    def __init__(self, genome_path: str) -> None:
        self.genome_path = genome_path
        self._vectors: list[tuple[str, str, str, list[float]]] = []
        self._load()

    def _load(self) -> None:
        conn = sqlite3.connect(self.genome_path)
        try:
            rows = conn.execute(
                """SELECT gene_id, source_id, content, embedding
                   FROM genes WHERE embedding IS NOT NULL"""
            ).fetchall()
        finally:
            conn.close()
        for gene_id, src, content, emb_blob in rows:
            try:
                emb = json.loads(emb_blob)
                self._vectors.append((gene_id, src, content or "", emb))
            except Exception:
                continue

    def retrieve(
        self,
        query: str,
        *,
        filter_paths=None,
        top_k: int = 8,
    ) -> list[RetrievedDoc]:
        codec = _get_codec()
        q_vec = codec.encode(query)
        allow = set(filter_paths) if filter_paths is not None else None

        scored = []
        for gene_id, src, content, emb in self._vectors:
            if allow is not None and src not in allow:
                continue
            scored.append((_cosine(q_vec, emb), gene_id, src, content))
        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        for sc, gid, src, content in scored[:top_k]:
            out.append(RetrievedDoc(
                source_id=src, content=content[:4000],
                score=float(sc), metadata={"gene_id": gid},
            ))
        return out

    @property
    def corpus_size(self) -> int:
        return len(self._vectors)


# ── Bench runner ────────────────────────────────────────────────────


def _content_recall(expected: list[str], content: str) -> float:
    if not expected:
        return 0.0
    low = content.lower()
    hits = sum(1 for a in expected if a.lower() in low)
    return hits / len(expected)


def run(raw: SemaEmbeddingRetriever,
        narrowed: HelixNarrowedRetriever,
        needles: list[dict],
        top_k: int = 8) -> dict:
    results = []
    for needle in needles:
        # A: raw — searches full corpus
        t0 = time.time()
        raw_docs = raw.retrieve(needle["query"], top_k=top_k)
        raw_latency = time.time() - t0
        raw_content = "\n".join(d.content for d in raw_docs)
        raw_recall = _content_recall(needle["expected"], raw_content)
        raw_candidates = raw.corpus_size  # Searches every vector

        # B: narrowed — Helix scopes, then retriever searches within
        t0 = time.time()
        nar_docs = narrowed.retrieve(needle["query"], top_k=top_k)
        nar_latency = time.time() - t0
        nar_content = "\n".join(d.content for d in nar_docs)
        nar_recall = _content_recall(needle["expected"], nar_content)
        # The actual candidate count for the narrowed retrieve is
        # Helix's shortlist size (we measure via the packet call).
        try:
            pkt = httpx.post(
                f"{HELIX_URL}/context/packet",
                json={
                    "query": needle["query"],
                    "task_type": "explain",
                    "read_only": True,
                },
                timeout=30,
            ).json()
            shortlist = set()
            for bucket in ("verified", "stale_risk", "contradictions"):
                for item in pkt.get(bucket, []) or []:
                    if item.get("source_id"):
                        shortlist.add(item["source_id"])
            for t in pkt.get("refresh_targets", []) or []:
                if t.get("source_id"):
                    shortlist.add(t["source_id"])
            nar_candidates = len(shortlist)
        except Exception:
            nar_candidates = 0

        results.append({
            "name": needle["name"],
            "raw_candidates": raw_candidates,
            "raw_latency_s": round(raw_latency, 3),
            "raw_content_recall": round(raw_recall, 3),
            "narrowed_candidates": nar_candidates,
            "narrowed_latency_s": round(nar_latency, 3),
            "narrowed_content_recall": round(nar_recall, 3),
            "narrowing_ratio": (raw_candidates / nar_candidates
                                if nar_candidates else float("inf")),
        })
    return {"needles": results}


def main() -> int:
    if not Path(GENOME_PATH).exists():
        print(f"ERROR: genome not found at {GENOME_PATH}")
        return 1
    try:
        stats = httpx.get(f"{HELIX_URL}/stats", timeout=5).json()
        print(f"Genome: {stats['total_genes']} genes\n")
    except Exception as exc:
        print(f"Cannot reach Helix at {HELIX_URL}: {exc}")
        return 1

    print("Loading SemaEmbeddingRetriever (this is the 'external' retriever)…",
          flush=True)
    raw = SemaEmbeddingRetriever(GENOME_PATH)
    print(f"  {raw.corpus_size} vectors loaded\n")

    narrowed = HelixNarrowedRetriever(
        raw,
        helix_url=HELIX_URL,
        fallback_unscoped=True,
        read_only=True,
    )

    print(f"=== External-retriever composition ({len(NEEDLES)} needles) ===\n")
    data = run(raw, narrowed, NEEDLES)

    # Report
    print(f"{'name':<45} {'raw_N':>6} {'nar_N':>6} "
          f"{'ratio':>8} {'raw_rec':>8} {'nar_rec':>8} "
          f"{'raw_ms':>7} {'nar_ms':>7}")
    print("-" * 100)
    for r in data["needles"]:
        ratio = (f"{r['narrowing_ratio']:>6.0f}x"
                 if r["narrowing_ratio"] != float("inf") else "—")
        print(f"{r['name']:<45} "
              f"{r['raw_candidates']:>6} {r['narrowed_candidates']:>6} "
              f"{ratio:>8} "
              f"{r['raw_content_recall']:>8.2f} {r['narrowed_content_recall']:>8.2f} "
              f"{r['raw_latency_s']*1000:>7.0f} {r['narrowed_latency_s']*1000:>7.0f}")

    # Aggregate
    n = len(data["needles"])
    mean_raw = sum(r["raw_content_recall"] for r in data["needles"]) / n
    mean_nar = sum(r["narrowed_content_recall"] for r in data["needles"]) / n
    mean_raw_lat = sum(r["raw_latency_s"] for r in data["needles"]) / n
    mean_nar_lat = sum(r["narrowed_latency_s"] for r in data["needles"]) / n
    mean_ratio = sum(r["narrowing_ratio"] for r in data["needles"]
                     if r["narrowing_ratio"] != float("inf")) / n
    print()
    print(f"mean raw_recall={mean_raw:.2f}  mean nar_recall={mean_nar:.2f}")
    print(f"mean raw_lat={mean_raw_lat*1000:.0f}ms  "
          f"mean nar_lat={mean_nar_lat*1000:.0f}ms")
    print(f"mean narrowing ratio: {mean_ratio:.0f}x")

    out = Path("benchmarks/results") / \
        f"external_retriever_{time.strftime('%Y-%m-%d')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "genome_size": raw.corpus_size,
        "needles": data["needles"],
        "summary": {
            "mean_raw_recall": mean_raw,
            "mean_narrowed_recall": mean_nar,
            "mean_raw_latency_s": mean_raw_lat,
            "mean_narrowed_latency_s": mean_nar_lat,
            "mean_narrowing_ratio": mean_ratio,
        },
    }, indent=2))
    print(f"\nsaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

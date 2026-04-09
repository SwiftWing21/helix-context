r"""
Benchmark: Compression Quality & Latency

Measures:
  1. Compression ratio (raw chars / compressed chars)
  2. Information retention (oracle grading via local model)
  3. Per-step latency (extract, express, re-rank, splice, assemble)
  4. Token savings (with vs without Helix)

Usage:
    # Requires Helix server running
    python benchmarks/bench_compression.py
    python benchmarks/bench_compression.py --queries 20 --output results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")

# Benchmark queries — diverse topics that should hit different genome regions
BENCHMARK_QUERIES = [
    # Architecture questions
    "How does the BigEd fleet supervisor manage worker processes?",
    "What is the ScoreRift divergence detection threshold?",
    "How does the Helix expression pipeline work?",
    "What database does BookKeeper use for tenant isolation?",
    # Implementation questions
    "How does the ribosome handle splice failures?",
    "What happens when a gene's decay score drops below 0.3?",
    "How does the co-activation pull-forward work in genome queries?",
    "What is the cold-start bootstrap threshold?",
    # Cross-domain questions
    "How does caching invalidation relate to context compression?",
    "What security patterns are shared across BigEd and BookKeeper?",
    # Specific detail questions (needle-in-haystack)
    "What is the exact name of the class that handles SQLite gene storage?",
    "What port does the Helix proxy server run on?",
    "How many audit dimensions does the Python preset in ScoreRift have?",
    "What is the default splice aggressiveness value?",
    # Impossible questions (should return 'no context')
    "What is the capital of France?",
    "How do I cook pasta?",
]


def bench_context_query(client, query, decoder_mode="none"):
    """Benchmark a single context query."""
    t0 = time.time()
    resp = client.post(f"{HELIX_URL}/context", json={
        "query": query,
        "decoder_mode": decoder_mode,
    })
    latency = time.time() - t0

    if resp.status_code != 200:
        return {
            "query": query,
            "error": f"HTTP {resp.status_code}",
            "latency_s": latency,
        }

    data = resp.json()
    entry = data[0] if data else {}
    content = entry.get("content", "")
    health = entry.get("context_health", {})

    return {
        "query": query,
        "latency_s": round(latency, 3),
        "genes_expressed": health.get("genes_expressed", 0),
        "genes_available": health.get("genes_available", 0),
        "ellipticity": health.get("ellipticity", 0),
        "coverage": health.get("coverage", 0),
        "density": health.get("density", 0),
        "freshness": health.get("freshness", 0),
        "status": health.get("status", "unknown"),
        "content_chars": len(content),
        "decoder_mode": decoder_mode,
    }


def bench_proxy_query(client, query, model="gemma4:e2b"):
    """Benchmark a full proxy pass (context + generation)."""
    t0 = time.time()
    resp = client.post(f"{HELIX_URL}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": False,
    })
    latency = time.time() - t0

    if resp.status_code != 200:
        return {
            "query": query,
            "error": f"HTTP {resp.status_code}",
            "latency_s": latency,
        }

    data = resp.json()
    choices = data.get("choices", [])
    response_text = choices[0].get("message", {}).get("content", "") if choices else ""

    return {
        "query": query,
        "latency_s": round(latency, 3),
        "response_chars": len(response_text),
        "model": model,
    }


def bench_token_savings(client, queries):
    """Compare token usage: raw file reads vs Helix context."""
    results = []

    for query in queries:
        # With Helix (decoder_mode=none for max savings)
        helix = bench_context_query(client, query, decoder_mode="none")

        # With Helix (decoder_mode=full for proxy comparison)
        helix_full = bench_context_query(client, query, decoder_mode="full")

        # Estimate raw file tokens (assume 8 genes * avg gene raw size)
        genes_available = helix.get("genes_available", 0)
        if genes_available > 0:
            # Get genome stats to calculate avg raw per gene
            stats = client.get(f"{HELIX_URL}/stats").json()
            avg_raw_per_gene = stats["total_chars_raw"] / max(stats["total_genes"], 1)
            genes_expressed = helix.get("genes_expressed", 0)
            estimated_raw_tokens = int(genes_expressed * avg_raw_per_gene / 4)
        else:
            estimated_raw_tokens = 0

        helix_tokens = helix["content_chars"] // 4
        helix_full_tokens = helix_full["content_chars"] // 4 + 750  # +decoder prompt

        results.append({
            "query": query,
            "raw_tokens_est": estimated_raw_tokens,
            "helix_none_tokens": helix_tokens,
            "helix_full_tokens": helix_full_tokens,
            "savings_none_pct": round((1 - helix_tokens / max(estimated_raw_tokens, 1)) * 100, 1),
            "savings_full_pct": round((1 - helix_full_tokens / max(estimated_raw_tokens, 1)) * 100, 1),
            "status": helix["status"],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Helix Context Benchmarks")
    parser.add_argument("--queries", type=int, default=len(BENCHMARK_QUERIES),
                        help="Number of queries to benchmark")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--proxy", action="store_true", help="Also benchmark full proxy pass")
    args = parser.parse_args()

    client = httpx.Client(timeout=300)

    # Check server
    try:
        stats = client.get(f"{HELIX_URL}/stats").json()
        print(f"Genome: {stats['total_genes']} genes, {stats['compression_ratio']:.1f}x")
        print(f"Raw: {stats['total_chars_raw']:,} chars, Compressed: {stats['total_chars_compressed']:,} chars")
    except Exception:
        print(f"Cannot reach Helix at {HELIX_URL}")
        sys.exit(1)

    queries = BENCHMARK_QUERIES[:args.queries]
    print(f"\nBenchmarking {len(queries)} queries...\n")

    # 1. Context query benchmarks
    print("=== Context Query Latency ===")
    context_results = []
    for q in queries:
        r = bench_context_query(client, q)
        context_results.append(r)
        status_icon = {"aligned": "+", "sparse": "~", "denatured": "!", "unknown": "?"}.get(r["status"], "?")
        print(f"  [{status_icon}] {r['latency_s']:>6.1f}s  e={r['ellipticity']:.2f}  "
              f"g={r['genes_expressed']}/{r['genes_available']}  {q[:50]}")

    # Summary stats
    latencies = [r["latency_s"] for r in context_results if "error" not in r]
    if latencies:
        latencies.sort()
        print(f"\n  p50={latencies[len(latencies)//2]:.1f}s  "
              f"p95={latencies[int(len(latencies)*0.95)]:.1f}s  "
              f"p99={latencies[-1]:.1f}s  "
              f"mean={sum(latencies)/len(latencies):.1f}s")

    status_counts = {}
    for r in context_results:
        s = r.get("status", "error")
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"  Status: {status_counts}")

    # 2. Token savings
    print("\n=== Token Savings ===")
    savings = bench_token_savings(client, queries)
    for s in savings:
        print(f"  {s['savings_none_pct']:>5.1f}%  (none)  {s['savings_full_pct']:>5.1f}%  (full)  "
              f"{s['status']:>10s}  {s['query'][:45]}")

    avg_none = sum(s["savings_none_pct"] for s in savings) / max(len(savings), 1)
    avg_full = sum(s["savings_full_pct"] for s in savings) / max(len(savings), 1)
    print(f"\n  Average savings: {avg_none:.1f}% (decoder=none), {avg_full:.1f}% (decoder=full)")

    # 3. Full proxy pass (optional)
    proxy_results = []
    if args.proxy:
        print("\n=== Full Proxy Pass ===")
        for q in queries[:5]:  # Only 5 for proxy (slow)
            r = bench_proxy_query(client, q)
            proxy_results.append(r)
            print(f"  {r['latency_s']:>6.1f}s  {r.get('response_chars', 0):>5d} chars  {q[:50]}")

    # Output
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "genome": stats,
        "context_queries": context_results,
        "token_savings": savings,
        "proxy_queries": proxy_results,
        "summary": {
            "total_queries": len(queries),
            "latency_p50": latencies[len(latencies)//2] if latencies else 0,
            "latency_p95": latencies[int(len(latencies)*0.95)] if latencies else 0,
            "latency_mean": sum(latencies)/len(latencies) if latencies else 0,
            "avg_savings_none_pct": avg_none,
            "avg_savings_full_pct": avg_full,
            "status_counts": status_counts,
        },
    }

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()

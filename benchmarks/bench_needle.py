r"""
Benchmark: Needle in a Haystack

The standard test for context systems: can you find a specific fact
buried in a large knowledge base?

Plants 10 "needles" (specific facts) across ingested content,
then queries for each one. Measures:
  - Retrieval rate (did the genome express the right gene?)
  - Answer accuracy (did the model answer correctly?)
  - Latency per retrieval

This is the benchmark KV cache papers and TurboQuant use to show
information retention at various compression ratios.

Usage:
    python benchmarks/bench_needle.py
"""

import json
import os
import sys
import time

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")

# Needles: specific facts that exist in the ingested content
# Each has a query and the expected answer substring
NEEDLES = [
    {
        "name": "helix_port",
        "query": "What port does the Helix proxy server listen on?",
        "expected": "11437",
        "accept": ["11437"],
        "source": "helix-context/helix.toml",
    },
    {
        "name": "scorerift_threshold",
        "query": "What is the divergence threshold that triggers alerts in ScoreRift?",
        "expected": "0.15",
        "accept": ["0.15", ".15"],
        "source": "two-brain-audit/README.md",
    },
    {
        "name": "biged_skills_count",
        "query": "How many skills does the BigEd fleet have?",
        "expected": "125",
        "accept": ["125", "129"],  # count changes between versions
        "source": "Education/CLAUDE.md",
    },
    {
        "name": "bookkeeper_monetary",
        "query": "What type should be used for monetary values in BookKeeper instead of float?",
        "expected": "Decimal",
        "accept": ["decimal", "Decimal"],
        "source": "BookKeeper/CLAUDE.md",
    },
    {
        "name": "helix_pipeline_steps",
        "query": "How many steps are in the Helix expression pipeline?",
        "expected": "6",
        "accept": ["6", "six"],
        "source": "helix-context/README.md",
    },
    {
        "name": "biged_rust_binary_size",
        "query": "What is the binary size of the Rust BigEd build in MB?",
        "expected": "11",
        "accept": ["11", "11mb", "11 mb"],
        "source": "Education/biged-rs/README.md",
    },
    {
        "name": "genome_compression_target",
        "query": "What is the target compression ratio for Helix Context?",
        "expected": "5x",
        "accept": ["5x", "5:1", "5 to 1"],
        "source": "helix-context design spec",
    },
    {
        "name": "scorerift_preset_dimensions",
        "query": "How many dimensions does the Python preset in ScoreRift check?",
        "expected": "8",
        "accept": ["8", "eight"],
        "source": "two-brain-audit/README.md",
    },
    {
        "name": "helix_ribosome_budget",
        "query": "How many tokens are allocated for the ribosome decoder prompt?",
        "expected": "3000",
        "accept": ["3000", "3k", "3,000"],
        "source": "helix-context design spec",
    },
    {
        "name": "biged_default_model",
        "query": "What is the default local model used by BigEd for conductor tasks?",
        "expected": "qwen3",
        "accept": ["qwen3", "qwen3:4b", "qwen"],
        "source": "Education/CLAUDE.md",
    },
]


def find_needle(client, needle):
    """Try to find a specific needle in the genome."""
    t0 = time.time()

    # Step 1: Context query
    try:
        resp = client.post(f"{HELIX_URL}/context", json={
            "query": needle["query"],
            "decoder_mode": "none",
        })
    except Exception:
        return {
            "name": needle["name"], "query": needle["query"],
            "expected": needle["expected"],
            "found_in_context": False, "answer_correct": False,
            "context_latency_s": time.time() - t0,
            "ellipticity": 0, "status": "error", "genes_expressed": 0,
            "answer_preview": "server unreachable",
        }
    context_latency = time.time() - t0

    if resp.status_code != 200:
        return {
            "name": needle["name"], "query": needle["query"],
            "expected": needle["expected"],
            "found_in_context": False, "answer_correct": False,
            "context_latency_s": context_latency,
            "ellipticity": 0, "status": "error", "genes_expressed": 0,
            "answer_preview": f"HTTP {resp.status_code}",
        }

    data = resp.json()
    entry = data[0] if data else {}
    content = entry.get("content", "")
    health = entry.get("context_health", {})

    # Check if any accepted answer appears in the expressed context
    accept = needle.get("accept", [needle["expected"]])
    found_in_context = any(a.lower() in content.lower() for a in accept)

    # Step 2: Full proxy query for answer accuracy
    t1 = time.time()
    proxy_resp = client.post(f"{HELIX_URL}/v1/chat/completions", json={
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": needle["query"]}],
        "stream": False,
    })
    proxy_latency = time.time() - t1

    answer_correct = False
    answer_text = ""
    if proxy_resp.status_code == 200:
        choices = proxy_resp.json().get("choices", [])
        if choices:
            answer_text = choices[0].get("message", {}).get("content", "")
            answer_correct = any(a.lower() in answer_text.lower() for a in accept)

    return {
        "name": needle["name"],
        "query": needle["query"],
        "expected": needle["expected"],
        "found_in_context": found_in_context,
        "answer_correct": answer_correct,
        "context_latency_s": round(context_latency, 3),
        "proxy_latency_s": round(proxy_latency, 3),
        "ellipticity": health.get("ellipticity", 0),
        "status": health.get("status", "unknown"),
        "genes_expressed": health.get("genes_expressed", 0),
        "answer_preview": answer_text[:200] if answer_text else "",
    }


def main():
    client = httpx.Client(timeout=300)

    # Check server
    try:
        stats = client.get(f"{HELIX_URL}/stats").json()
        print(f"Genome: {stats['total_genes']} genes, {stats['compression_ratio']:.1f}x")
    except Exception:
        print(f"Cannot reach Helix at {HELIX_URL}")
        sys.exit(1)

    print(f"\n=== Needle in a Haystack ({len(NEEDLES)} needles) ===\n")

    results = []
    found_context = 0
    found_answer = 0

    for needle in NEEDLES:
        r = find_needle(client, needle)
        results.append(r)

        icon_ctx = "+" if r["found_in_context"] else "-"
        icon_ans = "+" if r["answer_correct"] else "-"
        print(f"  ctx[{icon_ctx}] ans[{icon_ans}]  "
              f"{r['context_latency_s']:>5.1f}s  "
              f"e={r.get('ellipticity', 0):.2f}  "
              f"{r['name']}: \"{r['expected']}\"")

        if r["found_in_context"]:
            found_context += 1
        if r["answer_correct"]:
            found_answer += 1

    print(f"\n=== Results ===")
    print(f"Context retrieval:  {found_context}/{len(NEEDLES)} ({found_context/len(NEEDLES)*100:.0f}%)")
    print(f"Answer accuracy:    {found_answer}/{len(NEEDLES)} ({found_answer/len(NEEDLES)*100:.0f}%)")

    avg_latency = sum(r["context_latency_s"] for r in results) / len(results)
    print(f"Avg context latency: {avg_latency:.1f}s")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "genome_genes": stats["total_genes"],
        "compression_ratio": stats["compression_ratio"],
        "needles": results,
        "summary": {
            "context_retrieval_rate": found_context / len(NEEDLES),
            "answer_accuracy_rate": found_answer / len(NEEDLES),
            "avg_context_latency_s": round(avg_latency, 3),
        },
    }

    out_path = "benchmarks/needle_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

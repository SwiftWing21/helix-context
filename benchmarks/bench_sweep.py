r"""
Benchmark Sweep: Model Size vs Accuracy

Tests every available model against the needle-in-a-haystack benchmark.
Skips XL models (26B+) that overflow 12GB VRAM without extra config.

All models run with:
  - OLLAMA_KV_CACHE_TYPE=q4_0 (INT4 quantized KV cache)
  - Helix context injection (15K tokens expressed)
  - Same 10 needles

Usage:
    python benchmarks/bench_sweep.py
"""

import json
import os
import sys
import time

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Models to sweep — ordered smallest to largest
# Trimmed: dropped 0.6b (1/10 accuracy), e2b (7.2GB too heavy for q8_0 headroom)
# Skip 26B+ (overflow 12GB VRAM without offload config)
MODELS = [
    {"name": "qwen3:1.7b",  "params": "1.7B", "size_gb": 1.4},
    {"name": "qwen3:4b",    "params": "4B",   "size_gb": 2.5},
    {"name": "qwen3:8b",    "params": "8B",   "size_gb": 5.2},
    {"name": "gemma4:e4b",  "params": "4B",   "size_gb": 9.6},
]

NEEDLES = [
    {
        "name": "helix_port",
        "query": "What port does the Helix proxy server listen on?",
        "expected": "11437",
        "accept": ["11437"],
    },
    {
        "name": "scorerift_threshold",
        "query": "What is the divergence threshold that triggers alerts in ScoreRift?",
        "expected": "0.15",
        "accept": ["0.15", ".15"],
    },
    {
        "name": "biged_skills_count",
        "query": "How many skills does the BigEd fleet have?",
        "expected": "125",
        "accept": ["125", "129"],
    },
    {
        "name": "bookkeeper_monetary",
        "query": "What type should be used for monetary values in BookKeeper instead of float?",
        "expected": "Decimal",
        "accept": ["decimal", "Decimal"],
    },
    {
        "name": "helix_pipeline_steps",
        "query": "How many steps are in the Helix expression pipeline?",
        "expected": "6",
        "accept": ["6", "six"],
    },
    {
        "name": "biged_rust_binary_size",
        "query": "What is the binary size of the Rust BigEd build in MB?",
        "expected": "11",
        "accept": ["11", "11mb", "11 mb"],
    },
    {
        "name": "genome_compression_target",
        "query": "What is the target compression ratio for Helix Context?",
        "expected": "5x",
        "accept": ["5x", "5:1", "5 to 1"],
    },
    {
        "name": "scorerift_preset_dimensions",
        "query": "How many dimensions does the Python preset in ScoreRift check?",
        "expected": "8",
        "accept": ["8", "eight"],
    },
    {
        "name": "helix_ribosome_budget",
        "query": "How many tokens are allocated for the ribosome decoder prompt?",
        "expected": "3000",
        "accept": ["3000", "3k", "3,000"],
    },
    {
        "name": "biged_default_model",
        "query": "What is the default local model used by BigEd for conductor tasks?",
        "expected": "qwen3",
        "accept": ["qwen3", "qwen3:4b", "qwen"],
    },
]


def preload_model(client, model_name):
    """Ask Ollama to load the model into VRAM before benchmarking."""
    print(f"  Loading {model_name}...", end=" ", flush=True)
    t0 = time.time()
    try:
        resp = client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model_name,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1},
        }, timeout=120)
        if resp.status_code == 200:
            print(f"ready ({time.time()-t0:.1f}s)")
            return True
        else:
            print(f"FAILED (HTTP {resp.status_code})")
            return False
    except Exception as e:
        print(f"FAILED ({e})")
        return False


def unload_model(client, model_name):
    """Unload model from VRAM to make room for the next one."""
    try:
        client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model_name,
            "keep_alive": 0,
        }, timeout=10)
    except Exception:
        pass


def run_needle(client, model_name, needle):
    """Run a single needle test with a specific model."""
    # Step 1: Context retrieval (model-independent — same genome)
    t0 = time.time()
    try:
        resp = client.post(f"{HELIX_URL}/context", json={
            "query": needle["query"],
            "decoder_mode": "none",
        }, timeout=30)
    except Exception:
        return {"found_in_context": False, "answer_correct": False,
                "context_latency_s": time.time() - t0, "proxy_latency_s": 0,
                "answer_preview": "server unreachable"}
    context_latency = time.time() - t0

    if resp.status_code != 200:
        return {"found_in_context": False, "answer_correct": False,
                "context_latency_s": context_latency, "proxy_latency_s": 0,
                "answer_preview": f"HTTP {resp.status_code}"}

    data = resp.json()
    entry = data[0] if data else {}
    content = entry.get("content", "")

    accept = needle.get("accept", [needle["expected"]])
    found_in_context = any(a.lower() in content.lower() for a in accept)

    # Step 2: Proxy query (model-specific)
    t1 = time.time()
    try:
        proxy_resp = client.post(f"{HELIX_URL}/v1/chat/completions", json={
            "model": model_name,
            "messages": [{"role": "user", "content": needle["query"]}],
            "stream": False,
        }, timeout=120)
    except Exception:
        return {"found_in_context": found_in_context, "answer_correct": False,
                "context_latency_s": context_latency, "proxy_latency_s": time.time() - t1,
                "answer_preview": "proxy timeout"}
    proxy_latency = time.time() - t1

    answer_correct = False
    answer_text = ""
    if proxy_resp.status_code == 200:
        choices = proxy_resp.json().get("choices", [])
        if choices:
            answer_text = choices[0].get("message", {}).get("content", "")
            answer_correct = any(a.lower() in answer_text.lower() for a in accept)

    return {
        "found_in_context": found_in_context,
        "answer_correct": answer_correct,
        "context_latency_s": round(context_latency, 3),
        "proxy_latency_s": round(proxy_latency, 3),
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

    print(f"KV cache: q4_0 (INT4 quantized)")
    print(f"Models: {len(MODELS)} ({', '.join(m['name'] for m in MODELS)})")
    print(f"Needles: {len(NEEDLES)}")
    print()

    all_results = {}

    for model in MODELS:
        model_name = model["name"]
        print(f"=== {model_name} ({model['params']}, {model['size_gb']}GB) ===")

        # Preload model
        if not preload_model(client, model_name):
            print(f"  SKIPPING (failed to load)")
            all_results[model_name] = {"error": "load failed"}
            continue

        ctx_hits = 0
        ans_hits = 0
        total_proxy_latency = 0
        needle_results = []

        for needle in NEEDLES:
            r = run_needle(client, model_name, needle)
            needle_results.append({**needle, **r})

            icon_ctx = "+" if r["found_in_context"] else "-"
            icon_ans = "+" if r["answer_correct"] else "-"
            print(f"  ctx[{icon_ctx}] ans[{icon_ans}]  "
                  f"{r['proxy_latency_s']:>5.1f}s  {needle['name']}")

            if r["found_in_context"]:
                ctx_hits += 1
            if r["answer_correct"]:
                ans_hits += 1
            total_proxy_latency += r["proxy_latency_s"]

        avg_latency = total_proxy_latency / len(NEEDLES)
        all_results[model_name] = {
            "params": model["params"],
            "size_gb": model["size_gb"],
            "context_retrieval": f"{ctx_hits}/{len(NEEDLES)}",
            "answer_accuracy": f"{ans_hits}/{len(NEEDLES)}",
            "ctx_rate": ctx_hits / len(NEEDLES),
            "ans_rate": ans_hits / len(NEEDLES),
            "avg_proxy_latency_s": round(avg_latency, 1),
            "needles": needle_results,
        }

        print(f"  --> ctx={ctx_hits}/10  ans={ans_hits}/10  avg={avg_latency:.1f}s")
        print()

        # Unload to free VRAM for next model
        unload_model(client, model_name)
        time.sleep(2)

    # Summary table
    print()
    print("=" * 78)
    print(f"{'Model':<18} {'Params':>6} {'VRAM':>6} {'Retrieval':>10} {'Accuracy':>10} {'Latency':>8}")
    print("-" * 78)
    for model in MODELS:
        r = all_results.get(model["name"], {})
        if "error" in r:
            print(f"{model['name']:<18} {model['params']:>6} {model['size_gb']:>5.1f}G {'SKIP':>10} {'SKIP':>10} {'N/A':>8}")
        else:
            print(f"{model['name']:<18} {r['params']:>6} {r['size_gb']:>5.1f}G "
                  f"{r['context_retrieval']:>10} {r['answer_accuracy']:>10} "
                  f"{r['avg_proxy_latency_s']:>6.1f}s")
    print("=" * 78)
    print(f"KV cache: q4_0 | Genome: {stats['total_genes']} genes | Budget: 15K tokens/turn")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "genome_genes": stats["total_genes"],
        "kv_cache": "q4_0",
        "expression_budget": 15095,
        "models": all_results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

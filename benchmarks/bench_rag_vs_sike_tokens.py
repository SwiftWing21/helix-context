"""
RAG vs SIKE token-budget estimator.

For each query, capture what helix actually delivers (compressed
expressed_context + decoder + scaffolding) and compare to what a
naive RAG pipeline would deliver (raw chunks @ standard chunk size,
top-k from the same source files).

This is a token-COST comparison, not a quality comparison. SIKE
quality is measured by SIKE/dim-lock; this measures what each costs
to deliver to the downstream LLM.

Standard RAG assumptions (industry-typical):
  - chunk size: 1500 tokens (Pinecone/LangChain default)
  - top-k:      5 chunks
  - overhead:   ~500 tokens (system + question)
  - total:      ~8000 tokens per query

SIKE measurement (from /context response):
  - genes:                  N (typically 3-12)
  - expressed_context:      compressed via Kompress (~3-7x)
  - decoder prompt:         3000 tokens (broad) or ~750 (condensed)
  - scaffolding:            ~200 tokens
  - total:                  reported as agent.total_tokens_est

Output:
  - Per-query: (sike_tokens, rag_tokens, savings_x)
  - Aggregate: mean / median / p95 ratio
  - ASCII bar comparison

Usage:
  python benchmarks/bench_rag_vs_sike_tokens.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
OUTPUT_PATH = os.environ.get(
    "OUTPUT",
    str(Path(__file__).resolve().parent / "rag_vs_sike_tokens.json"),
)

# Industry-typical RAG defaults. Configurable so users can model their
# own RAG setup against SIKE.
RAG_CHUNK_TOKENS = int(os.environ.get("RAG_CHUNK_TOKENS", "1500"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
RAG_OVERHEAD = int(os.environ.get("RAG_OVERHEAD", "500"))


# ── Realistic test queries (mix of natural & technical) ───────────────
TEST_QUERIES = [
    "How does helix handle WAL checkpoints?",
    "What port does the helix proxy server listen on?",
    "How is the ribosome budget calculated?",
    "What does the path_key_index store?",
    "How does the density gate work?",
    "What is the 4-layer federation model?",
    "How does cymatics resonance contribute to retrieval?",
    "What does the chromatin tier control?",
    "How does TCM session drift work?",
    "What is the difference between SIKE and dimensional-lock benchmarks?",
    "How are gene attributions written?",
    "What does the access-rate tiebreaker do?",
    "How does the score-floor budget tier decide tight vs broad?",
    "What is the role of the harmonic_links table?",
    "How is the ΣĒMA cold-tier triggered?",
]


def estimate_rag_tokens(num_chunks: int = RAG_TOP_K) -> int:
    """RAG sends top-K raw chunks @ chunk size + overhead."""
    return num_chunks * RAG_CHUNK_TOKENS + RAG_OVERHEAD


def probe_helix(client: httpx.Client, query: str) -> dict:
    """Run query through /context, capture token usage."""
    out = {"query": query, "error": None}
    t0 = time.time()
    try:
        resp = client.post(
            f"{HELIX_URL}/context",
            json={
                "query": query,
                "decoder_mode": "condensed",  # realistic prod default
                "verbose": True,
                "clean": True,
            },
            timeout=30.0,
        )
    except Exception as e:
        out["error"] = f"context: {e}"
        return out

    out["latency_s"] = round(time.time() - t0, 2)
    if resp.status_code != 200:
        out["error"] = f"HTTP {resp.status_code}"
        return out

    body = resp.json()
    if isinstance(body, list):
        body = body[0] if body else {}

    agent = body.get("agent", {}) or {}
    health = body.get("context_health", {}) or {}
    content = body.get("content", "") or ""

    out["sike_tokens_est"] = agent.get("total_tokens_est", 0)
    out["compression_ratio"] = agent.get("compression_ratio", 0.0)
    out["genes_expressed"] = health.get("genes_expressed", 0)
    out["budget_tier"] = agent.get("budget_tier", "broad")
    # Cross-check via a rough char→token estimate (4 chars ~ 1 token)
    # Note: agent.total_tokens_est already includes decoder + scaffold;
    # this is just a sanity cross-check on the content portion.
    out["content_tokens_est"] = len(content) // 4
    return out


def render_bar(value: int, max_value: int, width: int = 30) -> str:
    if max_value <= 0:
        return ""
    filled = int(width * value / max_value)
    return "█" * filled + "░" * (width - filled)


def main() -> int:
    print(f"=== RAG vs SIKE token-budget estimator ===")
    print(f"Server:        {HELIX_URL}")
    print(f"RAG model:     top-{RAG_TOP_K} chunks @ {RAG_CHUNK_TOKENS} tokens "
          f"+ {RAG_OVERHEAD} overhead = {estimate_rag_tokens()} tokens/query")
    print(f"SIKE model:    measured from /context (decoder=condensed, broad budget)")
    print(f"Test queries:  {len(TEST_QUERIES)}")
    print()

    try:
        h = httpx.get(f"{HELIX_URL}/health", timeout=5).json()
        print(f"Server: ok, ribosome={h.get('ribosome')}, genes={h.get('genes')}\n")
    except Exception as e:
        print(f"Server unreachable: {e}", file=sys.stderr)
        return 2

    rag_tokens = estimate_rag_tokens()
    results = []

    print(f"{'#':>3} {'query':40s}  {'SIKE':>8s}  {'RAG':>8s}  {'savings':>9s}  {'tier':>10s}  {'genes':>5s}  {'cmpr':>6s}")
    print("-" * 110)

    with httpx.Client() as client:
        for i, q in enumerate(TEST_QUERIES, 1):
            r = probe_helix(client, q)
            results.append(r)
            if r.get("error"):
                print(f"{i:>3} {q[:40]:40s}  ERROR: {r['error']}")
                continue
            sike = r["sike_tokens_est"]
            ratio = rag_tokens / sike if sike > 0 else 0
            print(f"{i:>3} {q[:40]:40s}  {sike:>8d}  {rag_tokens:>8d}  "
                  f"{ratio:>7.1f}x  {r['budget_tier']:>10s}  {r['genes_expressed']:>5d}  "
                  f"{r['compression_ratio']:>5.1f}x")

    valid = [r for r in results if not r.get("error") and r.get("sike_tokens_est")]
    if not valid:
        print("\nNo valid measurements", file=sys.stderr)
        return 1

    sike_vals = sorted(r["sike_tokens_est"] for r in valid)
    n = len(sike_vals)
    sike_mean = sum(sike_vals) / n
    sike_median = sike_vals[n // 2]
    sike_p95 = sike_vals[int(n * 0.95)] if n > 1 else sike_vals[0]
    sike_min = sike_vals[0]
    sike_max = sike_vals[-1]

    savings_vals = [rag_tokens / r["sike_tokens_est"] for r in valid]
    savings_vals.sort()
    savings_mean = sum(savings_vals) / n
    savings_median = savings_vals[n // 2]

    print()
    print("=" * 70)
    print("== Summary ==")
    print()
    print(f"  RAG cost (fixed):      {rag_tokens:,} tokens/query")
    print()
    print(f"  SIKE cost (measured):")
    print(f"    min:                 {sike_min:,} tokens")
    print(f"    median:              {sike_median:,} tokens")
    print(f"    mean:                {sike_mean:,.0f} tokens")
    print(f"    p95:                 {sike_p95:,} tokens")
    print(f"    max:                 {sike_max:,} tokens")
    print()
    print(f"  Savings (RAG/SIKE):")
    print(f"    median:              {savings_median:.1f}x")
    print(f"    mean:                {savings_mean:.1f}x")
    print()
    print("== Visual comparison (median query) ==")
    print()
    bar_max = max(rag_tokens, sike_median)
    print(f"  RAG  ({rag_tokens:>5,} tok): |{render_bar(rag_tokens, bar_max)}|")
    print(f"  SIKE ({sike_median:>5,} tok): |{render_bar(sike_median, bar_max)}|")
    print()
    print(f"  Per-query token savings: {rag_tokens - sike_median:,} tokens "
          f"({100 * (rag_tokens - sike_median) / rag_tokens:.0f}% reduction)")

    # Annual cost projection (assume 1000 queries/day for an active dev)
    queries_per_day = 1000
    days = 365
    yearly_savings = (rag_tokens - sike_median) * queries_per_day * days
    print()
    print(f"== Cost projection (1,000 queries/day, 365 days) ==")
    print(f"  RAG annual:           {rag_tokens * queries_per_day * days:,} tokens")
    print(f"  SIKE annual:          {sike_median * queries_per_day * days:,} tokens")
    print(f"  Annual savings:       {yearly_savings:,} tokens")
    # At Anthropic Claude API pricing (rough — Sonnet input ~$3/M tokens)
    cost_per_m = 3.0  # Sonnet input
    annual_dollar_savings = yearly_savings * cost_per_m / 1_000_000
    print(f"  At ${cost_per_m}/M tokens (Sonnet input): ${annual_dollar_savings:,.0f}/year saved")

    out = {
        "timestamp": time.time(),
        "rag_assumptions": {
            "chunk_tokens": RAG_CHUNK_TOKENS,
            "top_k": RAG_TOP_K,
            "overhead": RAG_OVERHEAD,
            "total_per_query": rag_tokens,
        },
        "sike_stats": {
            "min": sike_min,
            "median": sike_median,
            "mean": round(sike_mean, 1),
            "p95": sike_p95,
            "max": sike_max,
        },
        "savings": {
            "median_x": round(savings_median, 2),
            "mean_x": round(savings_mean, 2),
            "tokens_saved_per_query_median": rag_tokens - sike_median,
        },
        "queries": results,
    }
    Path(OUTPUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

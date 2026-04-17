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
import re
import sys
import time

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")

# Needles: specific facts that exist in the ingested content
# Each has a query and the expected answer substring
# Each needle has `gold_source`: a list of case-insensitive path
# substrings identifying the source file(s) that contain the answer.
# `found_in_context` checks that one of these sources is in the
# delivered gene set AND its block body contains an accept substring.
# Payload-wide substring matches are retained as `false_positive_substring`
# diagnostics. Waude diagnostic 2026-04-17 showed the old pure-substring
# check inflated delivery by counting URL port numbers and compression
# metadata as hits.
NEEDLES = [
    {
        "name": "helix_port",
        "query": "What port does the Helix proxy server listen on?",
        "expected": "11437",
        "accept": ["11437"],
        "source": "helix-context/helix.toml",
        "gold_source": ["helix-context/helix.toml"],
    },
    {
        "name": "scorerift_threshold",
        "query": "What is the divergence threshold that triggers alerts in ScoreRift?",
        "expected": "0.15",
        "accept": ["0.15", ".15"],
        "source": "two-brain-audit/README.md",
        "gold_source": ["two-brain-audit/README.md"],
    },
    {
        "name": "biged_skills_count",
        "query": "How many skills does the BigEd fleet have?",
        "expected": "125",
        "accept": ["125", "129"],  # count changes between versions
        "source": "Education/CLAUDE.md",
        "gold_source": ["Education/CLAUDE.md"],
    },
    {
        "name": "bookkeeper_monetary",
        "query": "What type should be used for monetary values in BookKeeper instead of float?",
        "expected": "Decimal",
        "accept": ["decimal", "Decimal"],
        "source": "BookKeeper/CLAUDE.md",
        "gold_source": ["BookKeeper/CLAUDE.md"],
    },
    {
        "name": "helix_pipeline_steps",
        "query": "How many steps are in the Helix expression pipeline?",
        "expected": "6",
        "accept": ["6", "six"],
        "source": "helix-context/README.md",
        "gold_source": ["helix-context/CLAUDE.md", "helix-context/README.md"],
    },
    {
        "name": "biged_rust_binary_size",
        "query": "What is the binary size of the Rust BigEd build in MB?",
        "expected": "11",
        "accept": ["11", "11mb", "11 mb"],
        "source": "Education/biged-rs/README.md",
        "gold_source": ["Education/biged-rs/README.md"],
    },
    {
        "name": "genome_compression_target",
        "query": "What is the target compression ratio for Helix Context?",
        "expected": "5x",
        "accept": ["5x", "5:1", "5 to 1"],
        "source": "helix-context design spec",
        "gold_source": ["helix-context/README.md", "helix-context/docs"],
    },
    {
        "name": "scorerift_preset_dimensions",
        "query": "How many dimensions does the Python preset in ScoreRift check?",
        "expected": "8",
        "accept": ["8", "eight"],
        "source": "two-brain-audit/README.md",
        "gold_source": ["two-brain-audit/README.md"],
    },
    {
        "name": "helix_ribosome_budget",
        "query": "How many tokens are allocated for the ribosome decoder prompt?",
        "expected": "3000",
        "accept": ["3000", "3k", "3,000"],
        "source": "helix-context design spec",
        "gold_source": ["helix-context/helix.toml", "helix-context/README.md"],
    },
    {
        "name": "biged_default_model",
        "query": "What is the default local model used by BigEd for conductor tasks?",
        "expected": "qwen3",
        "accept": ["qwen3", "qwen3:4b", "qwen"],
        "source": "Education/CLAUDE.md",
        "gold_source": ["Education/CLAUDE.md"],
    },
]


# ── Gold-gene delivery check (Waude diagnostic 2026-04-17) ────────────

GENE_BLOCK_RE = re.compile(
    r'<GENE src="([^"]+)"[^>]*>(.*?)</GENE>', re.DOTALL,
)


def parse_delivered_genes(content: str):
    """Extract (src, body) tuples from the delivered /context payload.

    Helix embeds each expressed gene as ``<GENE src="path/to/file"
    facts="...">BODY</GENE>``. Parsing these gives an honest list of
    what was actually delivered, rather than relying on substring
    match against the whole payload (which picks up URL ports and
    compression metadata — see waude_diagnostic_2026-04-17.md).
    """
    return GENE_BLOCK_RE.findall(content or "")


def _body_contains_accept(body: str, accept) -> bool:
    """Word-boundary match to exclude substring-in-URL false positives.

    ``\\b11\\b`` matches "11 MB" but NOT "localhost:11434". Plain
    substring match was the 2026-04-16 bench bug — it counted
    ``11434`` as an "11" hit.
    """
    for a in accept:
        pattern = rf"\b{re.escape(a)}\b"
        if re.search(pattern, body, re.IGNORECASE):
            return True
    return False


def check_gold_delivery(content: str, gold_sources, accept):
    """Honest delivery check for a needle.

    Returns a dict with two independent dimensions:
      - ``gold_delivered``: gold source file is in delivered top-K
        (the retrieval-rank metric — addresses D-category failures
        directly, per Waude diagnostic 2026-04-17)
      - ``gold_has_answer``: gold-source block body contains accept
      - ``body_has_answer``: ANY delivered block body contains accept
        with word boundaries (what the consumer actually sees; more
        honest than raw substring match because it excludes URL
        ports and metadata headers)
      - ``false_positive_substring``: raw payload substring match
        fires BUT no gene body has a word-boundary match (i.e.,
        the match is pure metadata/header/URL noise)
    """
    blocks = parse_delivered_genes(content)

    def src_matches(src: str) -> bool:
        src_norm = src.replace("\\", "/").lower()
        return any(
            g.replace("\\", "/").lower() in src_norm for g in gold_sources
        )

    gold_blocks = [(src, body) for src, body in blocks if src_matches(src)]
    gold_delivered = len(gold_blocks) > 0

    gold_has_answer = any(
        _body_contains_accept(body, accept) for _, body in gold_blocks
    )
    body_has_answer = any(
        _body_contains_accept(body, accept) for _, body in blocks
    )

    old_substring_hit = any(a.lower() in (content or "").lower() for a in accept)
    false_positive = old_substring_hit and not body_has_answer

    return {
        "gold_delivered": gold_delivered,
        "gold_has_answer": gold_has_answer,
        "body_has_answer": body_has_answer,
        "old_substring_hit": old_substring_hit,
        "false_positive_substring": false_positive,
        "n_gold_blocks": len(gold_blocks),
        "n_delivered_blocks": len(blocks),
    }


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

    # Gold-gene delivery check (Waude diagnostic 2026-04-17): require
    # the answer's source file to be in the delivered gene set AND
    # that block's body to contain an accept substring. Fall back to
    # payload substring if a needle has no gold_source defined.
    accept = needle.get("accept", [needle["expected"]])
    gold_sources = needle.get("gold_source", [])
    if gold_sources:
        gold = check_gold_delivery(content, gold_sources, accept)
        # Primary metric: does any delivered gene BODY contain the
        # answer with word-boundary match. This is what the consumer
        # actually sees, minus metadata/URL false positives.
        found_in_context = gold["body_has_answer"]
        gold_delivered = gold["gold_delivered"]
        gold_has_answer = gold["gold_has_answer"]
        false_positive = gold["false_positive_substring"]
        n_gold_blocks = gold["n_gold_blocks"]
        n_delivered_blocks = gold["n_delivered_blocks"]
    else:
        found_in_context = any(a.lower() in content.lower() for a in accept)
        gold_delivered = found_in_context
        gold_has_answer = found_in_context
        false_positive = False
        n_gold_blocks = 0
        n_delivered_blocks = len(parse_delivered_genes(content))

    # Step 2: Full proxy query for answer accuracy
    t1 = time.time()
    model = os.environ.get("HELIX_MODEL", "qwen3:8b")
    proxy_resp = client.post(f"{HELIX_URL}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": needle["query"]}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 256},
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
        "gold_delivered": gold_delivered,
        "gold_has_answer": gold_has_answer,
        "false_positive_substring": false_positive,
        "n_gold_blocks": n_gold_blocks,
        "n_delivered_blocks": n_delivered_blocks,
        "context_latency_s": round(context_latency, 3),
        "proxy_latency_s": round(proxy_latency, 3),
        "ellipticity": health.get("ellipticity", 0),
        "status": health.get("status", "unknown"),
        "genes_expressed": health.get("genes_expressed", 0),
        # Step 1b weighing surface (2026-04-17): coordinate-resolution
        # confidence that tells the consumer whether to act on the pointer
        # or go fetch. Separate from ellipticity (retrospective).
        "coordinate_crispness": health.get("coordinate_crispness", 0),
        "neighborhood_density": health.get("neighborhood_density", 0),
        "resolution_confidence": health.get("resolution_confidence", 0),
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
        icon_gold = "+" if r.get("gold_delivered") else "-"
        conf = r.get("resolution_confidence", 0)
        print(f"  ctx[{icon_ctx}] ans[{icon_ans}] gold[{icon_gold}]  "
              f"{r['context_latency_s']:>5.1f}s  "
              f"e={r.get('ellipticity', 0):.2f}  "
              f"conf={conf:.2f}  "
              f"{r['name']}: \"{r['expected']}\"")

        if r["found_in_context"]:
            found_context += 1
        if r["answer_correct"]:
            found_answer += 1

    print(f"\n=== Results ===")
    print(f"Context retrieval (honest):  {found_context}/{len(NEEDLES)} ({found_context/len(NEEDLES)*100:.0f}%)")
    print(f"Answer accuracy:             {found_answer}/{len(NEEDLES)} ({found_answer/len(NEEDLES)*100:.0f}%)")

    gold_delivered = sum(1 for r in results if r.get("gold_delivered"))
    false_positives = sum(1 for r in results if r.get("false_positive_substring"))
    print(f"Gold source in top-K:        {gold_delivered}/{len(NEEDLES)} ({gold_delivered/len(NEEDLES)*100:.0f}%)")
    print(f"False-positive substring:    {false_positives}/{len(NEEDLES)} "
          f"(old-scoring would count these as hits)")

    # Weighing surface (Step 1b, 2026-04-17): know-vs-go quality
    # correctly_known_miss = gold missing AND helix's resolution_confidence
    # is below threshold. High rate means helix knows when it doesn't know.
    # silent_miss = gold missing AND confidence above threshold (dangerous).
    # overconfident_false_positive = substring false-positive AND confidence high.
    confidence_threshold = 0.30  # empirical; tune against distribution below
    misses = [r for r in results if not r.get("gold_delivered")]
    known_miss = sum(
        1 for r in misses
        if r.get("resolution_confidence", 0) < confidence_threshold
    )
    silent_miss = len(misses) - known_miss
    avg_conf_hit = (
        sum(r.get("resolution_confidence", 0) for r in results if r.get("gold_delivered"))
        / max(gold_delivered, 1)
    )
    avg_conf_miss = (
        sum(r.get("resolution_confidence", 0) for r in misses) / max(len(misses), 1)
    )
    print(
        f"Correctly-known miss:        {known_miss}/{len(misses)} "
        f"(confidence < {confidence_threshold} when gold absent)"
    )
    print(
        f"Silent miss (danger):        {silent_miss}/{len(misses)} "
        f"(confident but wrong)"
    )
    print(
        f"Avg resolution_confidence:   hit={avg_conf_hit:.3f}  miss={avg_conf_miss:.3f}  "
        f"(want hit >> miss)"
    )

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
            "gold_delivered_rate": gold_delivered / len(NEEDLES),
            "false_positive_substring_count": false_positives,
            "correctly_known_miss_count": known_miss,
            "silent_miss_count": silent_miss,
            "avg_resolution_confidence_hit": round(avg_conf_hit, 4),
            "avg_resolution_confidence_miss": round(avg_conf_miss, 4),
            "confidence_threshold": confidence_threshold,
            "avg_context_latency_s": round(avg_latency, 3),
            "scoring": "gold_source_in_top_K_and_body_substring",
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "needle_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

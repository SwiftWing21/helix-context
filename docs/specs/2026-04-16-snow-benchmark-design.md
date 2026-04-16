# SNOW Benchmark — Design Spec

**Scale-invariant Navigation on Organic Webs**

**Status:** Design approved 2026-04-16. Not yet implemented.
**Authors:** Max + Laude
**Depends on:** helix-context (in-process), Ollama (for local model ladder),
Claude API (for Haiku/Sonnet/Opus runs via sub-agents)

## Purpose

SNOW measures how efficiently an LLM consumer navigates helix's emergent
knowledge topology to find an answer. Unlike RAG benchmarks (one-hop
retrieval quality) or multi-hop QA benchmarks (reasoning across documents),
SNOW measures **navigation strategy** — hops through a data cascade, tokens
consumed, and latency per step.

The benchmark answers: given a mathematical fingerprint of the retrieval
landscape (tier scores, source files, domain tags, named entities), how
many escalation steps does it take to reach the answer, and at what cost?

## Origin

Discovered 2026-04-15 during a precision-probe investigation that started
with "does helix need Decimal arithmetic?" and ended with the finding that
LLM consumers can triage retrieval from tier-score fingerprints alone
(validated across model sizes from gemma4:e2b to Claude Opus). The
fingerprint acts as a magnifying glass — the consumer reads the mathematical
landscape, decides which genes to open, and makes targeted reads instead of
consuming 15K tokens of expressed content.

No existing benchmark measures this navigation pattern. SNOW is the first.

## Core concept — the cascade

Each query passes through a 5-tier data cascade. The consumer starts at
T0 (cheapest) and escalates until it finds the answer or exhausts all tiers.

```
T0: FINGERPRINT (~150 tok/gene)
    harness calls _extract_query_signals(query) → domains, entities
    then query_genes(domains, entities, max_genes) → List[Gene]
    tier data read from side-effects: genome.last_query_scores (gene_id→float)
    and genome.last_tier_contributions (gene_id→{tier→float})
    consumer receives: {gene_id, score, tiers{}, source_id, domains[], entities[]}
    consumer decides: answer / pick gene to escalate / mark unanswerable

T1: KEY_VALUES (~63 tok/gene median)
    consumer reads genes.key_values for selected gene_id(s)
    receives: pre-extracted key=value pairs (CPU regex at ingest)
    consumer decides: answer / escalate

T2: COMPLEMENT (~381 tok/gene median)
    consumer reads genes.complement for selected gene_id(s)
    receives: compressed semantic summary (2.7x smaller than raw)
    consumer decides: answer / escalate

T3: CONTENT (~864 tok/gene median)
    consumer reads genes.content for selected gene_id(s)
    receives: verbatim source text
    consumer decides: answer / escalate

T4: WALK (variable)
    consumer reads harmonic_links neighbors for selected gene_id
    receives: neighbor gene_ids + edge weights
    consumer reads neighbor content (always T3) for up to 3 neighbors
    sorted by edge weight descending
    consumer decides: answer / MISS
```

**Implementation note:** T0 data is NOT returned by `query_genes()` directly.
The harness must: (1) call `_extract_query_signals(query)` to get domains +
entities, (2) call `genome.query_genes(domains, entities, max_genes)` which
returns `List[Gene]`, (3) read `genome.last_query_scores` and
`genome.last_tier_contributions` which are populated as side-effects. See
`precision_probe.py` for this pattern.

At each hop, the LLM consumer receives a fresh prompt with ONLY the
current tier's data + the original query. Context does not accumulate
across tiers. This is an explicit v1 trade-off: it measures raw
single-tier signal quality rather than sequential navigation strategy.
Real consumers would accumulate context; a future v2 "stateful cascade"
extension would test that. The stateless design keeps token measurement
clean, removes a confound (LLM memory quality), and is simpler to
implement.

## Two consumers

### Oracle (theoretical floor)

A script with perfect knowledge of the ground-truth answer. For each
query, it string-matches the expected answer against each tier in order:

```python
def oracle_hop(query, gene_ids, expected_answer, conn):
    # T0: check fingerprint entities
    for gid in gene_ids:
        if answer_in_entities(gid, expected_answer):
            return 0

    # T1-T3: check stored fields
    for tier, field in [(1, 'key_values'), (2, 'complement'), (3, 'content')]:
        for gid in gene_ids:
            if answer_in_field(gid, field, expected_answer):
                return tier

    # T4: check 1-hop neighbor content
    for gid in gene_ids:
        for neighbor in get_harmonic_neighbors(gid):
            if answer_in_field(neighbor, 'content', expected_answer):
                return 4

    return -1  # MISS
```

No LLM, no tokens, instant. Gives the minimum possible hops for each
query **within the retrieved top-k set** — NOT the entire genome. If the
answer-bearing gene doesn't score highly enough to appear in top-k, the
oracle returns MISS. This is intentional: the oracle measures "how deep
IS the answer given helix's retrieval ranked it," not "does the answer
exist anywhere in the genome." Retrieval quality is measured by SIKE,
not SNOW.

The oracle's cascade profile shows where answers actually LIVE in the
data hierarchy of the retrieved candidates.

### Real LLM consumer

An actual model reads the fingerprint, makes triage decisions, and
escalates through tiers. Two prompt templates:

- **TRIAGE_PROMPT** (T0): "Here are fingerprints for top-k genes.
  Answer the question if you can, or say which gene_id to read and why."
- **EXTRACT_PROMPT** (T1-T3): "Here is [field] from gene [id].
  Can you answer the question now? If yes, answer. If no, say why."

For Ollama models: direct API call with `/no_think` suppression for
small models. For Claude models: sub-agent with appropriate model
parameter. Prompt templates are model-agnostic; delivery mechanism
differs.

## Query set (N=65)

### Existing needles (N=50)

From `benchmarks/needles_50_for_claude.json`. Each has `key`, `value`,
`query` fields. Already validated against the benchmark genome. These
land wherever they land in the cascade — no guarantee of tier coverage.

### Hand-crafted tier-stress queries (N=15)

Deliberately authored to plant answers at specific tiers:

| Target tier | Count | Example shape |
|---|---|---|
| T0 (fingerprint) | 3 | Answer visible in entity list |
| T1 (key_values) | 3 | Answer is a named scalar (port, threshold) |
| T2 (complement) | 3 | Answer requires semantic understanding |
| T3 (content) | 3 | Answer is exact code/quote/literal |
| T4 (walk) | 3 | Answer in a neighbor gene, not direct match |

These are authored against the benchmark genome by inspecting actual
gene data at each tier. Each has `expected_answer`, `oracle_tier`,
and `gene_id` fields.

## Model ladder

| Model | Type | Purpose |
|---|---|---|
| gemma4:e2b | Local MoE, ~2B active | Fingerprint floor (smallest that can triage) |
| qwen3:4b | Local dense | Mid-range local |
| qwen3:8b | Local dense | High-end local |
| Claude Haiku | API | Cheapest API tier |
| Claude Sonnet | API | Mid API tier |
| Claude Opus | API | Highest capability |

Each model gets its own SNOW scorecard. The comparison table is the
headline output.

## Reported metrics

### Per-model scorecard

```
SNOW Scorecard — qwen3:4b on helix-18K genome (N=65)
─────────────────────────────────────────────────
  Hops (avg):        2.3    oracle floor: 1.4    waste: 0.9
  Tokens (avg):      834    oracle floor: 276     overhead: 3.0x
  Latency (avg):     1.2s   oracle floor: 0.4s    overhead: 3.0x
  Cascade profile:   T0: 8%  T1: 22%  T2: 45%  T3: 20%  T4: 5%
  Answered@T0:       8%
  Triage accuracy:   74%
  Miss rate:         5%

  Per-step latency (avg):
    T0 fingerprint:  0.31s
    T1 key_values:   0.04s
    T2 complement:   0.18s
    T3 content:      0.42s
    T4 walk:         0.67s
─────────────────────────────────────────────────
```

**Metric definitions:**

- **Hops (avg):** Mean number of tier escalations to reach the answer.
  Excludes MISSes. Lower is better.
- **Tokens (avg):** Mean cumulative tokens across all hops (fingerprint +
  reads + LLM inference). Excludes MISSes. Lower is better.
- **Latency (avg):** Mean wall-clock time from query start to answer.
  Excludes MISSes. Lower is better (but trades off with model capability).
- **Oracle floor:** The oracle consumer's average for the same metric.
  The theoretical minimum. Waste/overhead is LLM minus oracle.
- **Cascade profile:** Distribution of which tier produced the answer.
  T0=fingerprint, T1=kv, T2=complement, T3=content, T4=walk.
- **Answered@T0:** Percentage of queries answered from fingerprint alone.
  The "magnifying glass without microscope" rate.
- **Triage accuracy:** Percentage of queries where the LLM chose ANY gene
  that the oracle also found the answer in. When the answer appears in
  multiple genes, the LLM gets credit for picking any of them. Measures
  fingerprint readability as a navigation signal.
- **Miss rate:** Percentage of queries where the consumer exhausted all
  tiers without finding the answer. Reported separately, not included
  in hop/token/latency averages. Cascade profiles also EXCLUDE misses
  and always sum to (N - miss_count) for both oracle and LLM consumers,
  keeping the distributions comparable.
- **Per-step latency:** Average time spent at each tier, across queries
  that reached that tier. Shows where the bottleneck is.

### Comparison table

All models side by side, oracle floor at top. Generated by a separate
`snow_compare.py` that reads per-model result JSONs.

## Token counting methodology

Tokens are counted differently depending on context:

- **Helix-side data (fingerprint, kv, complement, content):** character
  count / 4.0, rounded up. This is a standard approximation that holds
  within ~10% for English text across tokenizers. Used for oracle token
  counts AND for the data portion of LLM hop tokens.
- **Ollama LLM calls:** `prompt_eval_count` + `eval_count` from the Ollama
  API response. This is the actual token count the model processed/generated.
- **Claude LLM calls (sub-agents):** `total_tokens` from the agent usage
  metadata.
- **Oracle tokens:** sum of chars/4 for the data the oracle inspected at
  each tier to find the answer. The oracle makes no LLM calls, so there
  are no generation tokens — only data-read tokens.
- **Per-hop LLM tokens:** data tokens (chars/4 for the tier content shown
  to the model) + LLM tokens (prompt_eval + eval from the API response).
  Both components are reported in the hop_detail.

Cross-model comparisons use the data-token component (chars/4) for
apples-to-apples, since LLM prompt/generation tokens vary by tokenizer.
The total_tokens field includes both components for realistic cost modeling.

## Output format

One JSON file per model per run:
`benchmarks/snow/results/snow_<model>_<date>.json`

```json
{
  "benchmark": "SNOW",
  "version": "1.0",
  "genome_db": "genome-bench-2026-04-14.db",
  "genome_genes": 18254,
  "model": "qwen3:4b",
  "timestamp": "2026-04-16T...",
  "n_queries": 65,
  "oracle_summary": {
    "avg_hops": 1.4, "avg_tokens": 276, "avg_latency_s": 0.4,
    "cascade_profile": {"T0": 8, "T1": 18, "T2": 21, "T3": 14, "T4": 4},
    "miss_count": 0
  },
  "llm_summary": {
    "avg_hops": 2.3, "avg_tokens": 834, "avg_latency_s": 1.2,
    "cascade_profile": {"T0": 5, "T1": 14, "T2": 29, "T3": 13, "T4": 3},
    "miss_count": 3, "miss_rate": 0.046,
    "triage_accuracy": 0.74, "answered_at_t0": 0.08
  },
  "per_step_latency": {
    "T0": 0.28, "T1": 0.04, "T2": 0.22, "T3": 0.51, "T4": 0.73
  },
  "queries": [
    {
      "idx": 0,
      "query": "What port does helix listen on?",
      "expected_answer": "11437",
      "oracle_tier": 1,
      "oracle_gene_id": "abc123",
      "llm_tier": 1,
      "llm_gene_id": "abc123",
      "llm_answer": "11437",
      "correct": true,
      "hops": 1,
      "tokens": 213,
      "latency_s": 0.35,
      "hop_detail": [
        {"tier": "T0", "action": "READ abc123", "tokens": 150, "latency_s": 0.28},
        {"tier": "T1", "action": "ANSWER 11437", "tokens": 63, "latency_s": 0.07}
      ]
    }
  ]
}
```

## File structure

```
benchmarks/
  snow/
    bench_snow.py          # Main harness — runs oracle + LLM cascade
    snow_compare.py        # Reads result JSONs, prints comparison table
    snow_queries.json      # N=65 query set (50 existing + 15 tier-stress)
    prompts.py             # Triage + extraction prompt templates
    oracle.py              # Oracle consumer (string matching per tier)
    cascade.py             # LLM consumer (Ollama/Claude + tier escalation)
    results/               # Output JSONs
```

## Implementation approach

In-process (Approach B): import HelixContextManager directly, call
query_genes() for T0, read gene fields via SQLite for T1-T3, query
harmonic_links for T4. LLM calls go to Ollama API directly for local
models. Claude models run via sub-agents (deferred to Approach C for
production-realistic latency).

## Dependencies

- helix_context (existing)
- httpx (existing, for Ollama API)
- No new packages

## Future extensions (not in v1)

- **Approach C** — server-based LLM consumer for production-realistic
  latency (HTTP overhead, connection pooling, etc.)
- **Consumer memory** — LLM accumulates context across hops (changes
  the escalation strategy; currently each hop is independent)
- **Walk depth > 1** — T4 currently checks 1-hop neighbors only;
  deeper walks (2-hop, 3-hop) test graph navigation quality
- **Session-aware consumer** — consumer's recent retrieval history
  influences triage (the "walk from where you're standing" extension)
- **SNOW on different genome sizes** — 1K, 10K, 100K, 1M genes to
  test whether navigation efficiency scales with genome size

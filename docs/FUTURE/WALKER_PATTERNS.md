# Walker Patterns — Self-Walk vs Librarian Dispatch

**Status:** Design direction, 2026-04-16. No code yet. Fell out of a
session where we confirmed that consuming LLMs can triage from
tier-score fingerprints alone (see
`benchmarks/fingerprint_*_test.py`), and then asked: "if the cloud
model only needs the fingerprint, who actually does the walking when
content is needed?" This doc captures the execution-layer decision.
See [GENOME_SHARDING.md](GENOME_SHARDING.md) for the storage-layer
companion and [PUSH_PULL_CONTEXT.md](PUSH_PULL_CONTEXT.md) for the
API contract that makes the walker pluggable.

---

## The problem in one sentence

When a consumer LLM needs content deeper than the fingerprint (T1
`key_values`, T2 `complement`, T3 `content`, T4 neighbor walk), every
byte it fetches flows through its context window — and sequential
fetches in a context-pressured model are the slow, expensive,
miss-rate-inducing path helix was built to avoid.

## The pattern

Make the walker pluggable. Two deployments of the same helix API:

```
┌─ A: Self-walk ─────────────────────────────────────┐
│                                                    │
│  Consumer LLM ──── reads fingerprint (push)        │
│        │                                           │
│        ├──── decides "need T2 of gene X"           │
│        ├──── pulls T2 directly, into own context   │
│        └──── reasons over it                       │
│                                                    │
│  One model. Sequential fetches. Context grows.     │
└────────────────────────────────────────────────────┘

┌─ B: Librarian dispatch ────────────────────────────┐
│                                                    │
│  Consumer LLM ──── reads fingerprint (push)        │
│        │                                           │
│        ├──── emits directive:                      │
│        │      "gene X, tier T2, extract 'port'"    │
│        │                                           │
│        ▼                                           │
│  Local librarian LLM ──── opens T2                 │
│        │                  matches 'port'           │
│        │                  returns scalar packet    │
│        ▼                                           │
│  Consumer LLM ──── receives {answer, source_id}    │
│                                                    │
│  Two models. One packet hop. Context stays lean.   │
└────────────────────────────────────────────────────┘
```

The consumer's API is identical across both. Only the walker swaps.

## Why the directive form matters

A blind librarian (the one the SNOW benchmark currently tests) must
*decide* which tier to open and what to extract from the fingerprint
alone. That's a reasoning task, and it bottoms out around 2B active
params (our measured floor: `gemma4:e2b` passes, `qwen3:1.7b` fails).

A **directed** librarian gets told. "Gene ade070, tier T2, extract the
value of `queries_path`." That's not reasoning, it's execute-and-
return. Our prior that `qwen3:0.6b` can handle that is untested but
plausible — the hard decision already happened in the cloud model's
head before the directive was emitted.

This is the same shape as CPU speculative execution or database query
planning: the planner is expensive and runs once; the executor is
cheap and runs many times in parallel.

## Parallel dispatch is the killer use case

Single-file self-walk is already fine. The pattern earns its keep on
multi-file pulls. Example: a query that needs T2 of five different
genes to answer.

- **Self-walk**: five sequential T2 reads in consumer context, each
  ~381 tokens, total ~1905 tokens of paragraphs to reason over.
  Latency: 5 × fetch-and-read-time.
- **Librarian dispatch**: five parallel directives to five parallel
  librarian workers, each returns a ~30-token scalar packet. Consumer
  context sees ~150 tokens of scalars. Latency: one fetch-and-read-time
  (parallel).

This is map-reduce applied to retrieval. The consumer's "reduce" step
is cheap because it's working with scalars, not paragraphs.

## Decision rule

When the consuming LLM decides whether to self-walk or dispatch:

| Signal | Self-walk | Librarian dispatch |
|---|---|---|
| Gene count | 1 | ≥ 2–3 |
| Cloud context pressure | Low | Any |
| Latency budget | Tight (dispatch overhead matters) | Loose (parallel wins) |
| Answer type | Needs paragraph reasoning | Scalar extraction |
| Local LLM available | N/A | Yes |
| Cost per query | Higher context cost acceptable | Prefer dollar savings |

Rule of thumb: **dispatch whenever the walk is mechanical**. If the
consumer needs to *reason about* the content (e.g., "does this design
doc contradict this other design doc"), self-walk. If the consumer
needs to *extract from* the content ("what port does this config
use"), dispatch.

## Librarian model sizing

Our empirical data (from fingerprint-only triage testing) gives us
a known floor for **blind** triage:

| Model | Blind triage | Directed extraction (untested) |
|---|---|---|
| `qwen3:0.6b` | No — garbled output | Plausibly yes |
| `qwen3:1.7b` | Barely — tries but garbled | Almost certainly yes |
| `gemma4:e2b` | Yes — floor for blind | Yes — overkill |
| `qwen3:4b` | Yes | Yes |
| `qwen3:8b` | Yes | Yes |

The directed floor is probably **below** the blind floor. That's
worth validating in a follow-up benchmark (SNOW variant B) because
if the directed floor is 0.6b, the throughput-per-dollar math
changes significantly — you can run 10+ librarians on a single
consumer-grade GPU instead of 2–3.

## Directive schema (proposed)

The consumer → librarian contract:

```json
{
  "gene_id": "ade070a94a3a118c",
  "tier": "T2",
  "task": "extract",
  "target": "value of 'queries_path'",
  "fallback_tier": "T3",
  "timeout_ms": 2000
}
```

And the librarian → consumer return:

```json
{
  "gene_id": "ade070a94a3a118c",
  "tier_opened": "T2",
  "answer": "/opt/helix/queries",
  "confidence": "high",
  "tokens_used": 412,
  "elapsed_ms": 180
}
```

`confidence: low` escalates — the consumer may re-dispatch with
`tier: "T3"` or do the read itself. Escalation loop terminates at T3
because T3 is verbatim source.

## Sub-agent dispatch as a side benefit

Once the librarian pattern exists, the same machinery is reusable
for sub-agent workloads more generally. "Summarise these five files"
becomes five librarian calls with `task: "summarise"` instead of
`task: "extract"`. The genome becomes a unified index that any small
LLM can act upon on behalf of any large LLM.

That's the future direction, not the v1. V1 is just extract-from-
known-gene.

## Open questions

### Who runs the librarian?

Three deployment shapes:

1. **Embedded** — librarian is a local process the consumer launches
   (e.g., Ollama on the same machine). Lowest latency, highest
   throughput, requires local compute.
2. **Service** — librarian is an HTTP endpoint on the helix FastAPI
   proxy (already on :11437). Works across devices. Requires the
   proxy to have a backend model configured.
3. **Co-process** — launcher spawns librarian as a sidecar to the
   helix supervisor, shared across consumer sessions. Best for
   multi-session homes where several agents are active.

Likely answer: **embedded by default, service as fallback for thin
clients (mobile, browser)**.

### Dispatch vs agent framework

This overlaps with the Claude Code subagent idea. Should helix ship
its own dispatcher, or publish a protocol that lets
`Agent(subagent_type="helix-librarian")` work out of the box?

The protocol answer is cleaner but requires the consumer environment
to implement it. The dispatcher answer ships faster. Unclear which
wins; depends on where the first real consumer shows up.

### Retry / budget / failure

A librarian can time out, return low confidence, or hit a shard that's
cold-tier and slow. The consumer needs a retry budget and a fallback
path (escalate tier, or self-walk as final resort). Not hard, but
needs spelling out before first ship.

## Related

- [GENOME_SHARDING.md](GENOME_SHARDING.md) — sharding gives librarians
  natural parallelism boundaries (one librarian per shard scanning
  in parallel). Enables walker-per-shard dispatch patterns that are
  impossible on monolithic genome.db.
- [PUSH_PULL_CONTEXT.md](PUSH_PULL_CONTEXT.md) — the push/pull API is
  what makes the walker pluggable in the first place. Self-walk and
  librarian dispatch are two implementations of the "pull" side of
  the same contract.
- [../../benchmarks/snow/](../../benchmarks/snow) — the SNOW benchmark
  measures walker performance end-to-end. A variant B (directed
  cascade) is the logical follow-up once v1 lands.
- `FINGERPRINT_CONVERGENCE` (if it gets written up) — the empirical
  basis for trusting that small models can triage from fingerprints.

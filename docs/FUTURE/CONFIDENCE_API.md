# Confidence API — Signals-First, Content-On-Demand

**Status:** ~~Design sketch, 2026-04-17~~ **SUPERSEDED 2026-04-18** by
GT's authoritative build spec at
[../specs/2026-04-17-agent-context-index-build-spec.md](../specs/2026-04-17-agent-context-index-build-spec.md)
and the shipped `POST /context/packet` endpoint
(commit `f10fc8a`). GT's packet returns a richer shape
(`verified` / `stale_risk` / `contradictions` / `refresh_targets` /
`citations`) at a better granularity (task → curated packet) than the
per-gene `/confidence/{gene_id}` this sketch proposed.

The sections below are kept for historical context — the framing
(know-vs-go, reflexive consultation, content-on-demand separation) is
what fed into GT's spec, even though the endpoint shape diverged.
Same pattern as Laude's `/index` sketch being superseded by the same
commit (see [INDEX_MODE_ENDPOINT.md](INDEX_MODE_ENDPOINT.md)).

**Original framing below.** Not a live proposal.

---

Captures what an agent consuming
helix *reflexively* (rather than as an expensive last-resort retrieval
tool) would want the API to look like. Falls out of the "know-vs-go"
framing — helix as the instrument layer that emits permission slips for
autonomous action.

Related:
- Memory note: `project_helix_weighs_not_retrieves.md` (card catalog
  framing — weighs, doesn't retrieve)
- Memory note: `project_helix_personas_as_config_surface.md`
  (per-persona threshold via env vars)
- [MEMORY_WATCH_HOOK.md](MEMORY_WATCH_HOOK.md) — similar env-var
  pattern for hook toggles

---

## Motivation

Today's endpoints (`/debug/preview`, `/genes/{id}`, `/stats`) bundle
content + math into large payloads. Measurement 2026-04-17:

| Endpoint | Bytes | Math signal density |
|---|---|---|
| `/debug/preview` (5 candidates) | 6,737 | ~5% math, 95% content previews |
| `/genes/{id}` (full) | 8,540 | ~3% math, 97% content/codons/complement |
| `/stats` | 673 | ~60% math |

An agent that wants "is my mental model of gene X stale?" has to pay
~2,100 tokens per check. That's expensive enough that agents (me
included) don't check. Result: **helix is consulted near-zero in
practice** even though the math it computes is load-bearing for safe
autonomous action.

This doc proposes a refactor where the *default* response shape is
small and math-first, and content is a separate opt-in call.

## The refactor in one line

Shift from "retrieval with metadata bolted on" to **"confidence API
with content-on-demand."** Same underlying genome; different default
response shape.

---

## 1. Signals-only per-gene endpoint

**New:** `GET /confidence/{gene_id}` → ~80 tokens instead of 2,100.

```json
{
  "gene_id": "4f98e2f4296d7620",
  "freshness": 0.87,
  "chromatin": "open",
  "disk_lag_s": 180,
  "superseded_by": null,
  "current_version_gene_id": "4f98e2f4296d7620",
  "active_authors": ["raude"],
  "last_ingest_time": 1776450115.677,
  "edges_unpulled_count": 2
}
```

**Covers every know-vs-go input in one compact response.**

Implementation: derived view over existing tables — freshness from
`epigenetics`, chromatin from `genes.chromatin`, disk_lag by stat'ing
`source_id` path and comparing to `last_ingest_time`, supersession by
resolving the version chain once, active_authors from cross-referencing
`gene_attribution` with the active participants in `/sessions`.

## 2. Batched confidence after retrieval

**New:** `POST /confidence` with `{gene_ids: [...]}` body.

```json
{
  "confidence": [
    {"gene_id": "4f98...", "freshness": 0.87, "chromatin": "open", ...},
    {"gene_id": "a2c1...", "freshness": 0.12, "chromatin": "cold", ...},
    ...
  ]
}
```

Current pattern: `/debug/preview` returns 10 candidates → 10 follow-up
`/genes/{id}` calls at 2,100 tokens each = 21KB just for the math
signals. Batched version: one call, ~800 tokens total for the same 10
genes.

## 3. Derived signals, not raw

Today the consumer has to do math helix should do:

| Raw (today) | Derived (proposed) |
|---|---|
| `chromatin: 0` (int code) | `chromatin: "open"` (string) |
| `last_ingest_time` + caller stat() | `disk_lag_s: 180` (already compared) |
| `supersedes: gene_X` (single hop) | `current_version_gene_id: gene_Z` (resolved chain) |
| `last_heartbeat` per participant | `active_authors: ["raude"]` (resolved filter) |

Three concrete wins:
- Agents can't misinterpret int codes they have to memorize
- Filesystem state is consulted server-side where it's cheap, not
  client-side where it's a tool call
- Version lineage is expensive to resolve repeatedly — do it once at
  the derivation layer

## 4. Co-activation gap surfacing

**Addition to confidence response:** `edges_unpulled: [gene_B, gene_C]`
and `edges_unpulled_count` (already in §1's sketch).

When an agent retrieves gene A, it currently has no visibility into
what's linked-but-not-retrieved. The `gene_relations` table knows; it
should surface. Makes "what am I missing?" a readable signal instead
of a blind spot.

Use case: know-vs-go gate fails for "edges_unpulled > 0" on
decision-critical queries. Forces the agent to consider whether linked
context matters before committing to an action.

## 5. Content-on-demand separation

Split the current `/debug/preview` into two endpoints:

| Endpoint | Returns | Use when |
|---|---|---|
| `GET /search?query=X&max=N` | 10 candidates, gene_ids + signals only (~200 tokens) | Deciding whether to read |
| `GET /read/{gene_id}` | Full content + codons + complement | Actually needs the content |

Today's `/debug/preview` bundles both. Agents that only need "which
gene" or "how fresh" pay for content they discard.

The two-call pattern also sets up a natural **know-vs-go gate between
search and read**: agent checks confidence; below threshold, it
decides whether to read (verify) or skip. The call shape enforces the
decision point.

---

## What's NOT in this doc

- Prefix-injectable cache manifest — a separate direction, more
  invasive, needs its own design conversation. (That's the one that'd
  put helix state into turn-start context instead of per-query tool
  call. Different tradeoffs.)
- Confidence threshold tuning — that's per-persona via env vars, see
  [MEMORY_WATCH_HOOK.md](MEMORY_WATCH_HOOK.md) for the pattern.
- Cross-agent confidence (how one agent's view of "fresh" should
  reconcile with another's) — falls out naturally once the endpoint
  exists; deferred until we see how multi-vendor consumption behaves.

## Backwards compatibility

- Keep `/debug/preview` and `/genes/{id}` working verbatim — they're
  still useful for debug UIs and explicit content pulls.
- New endpoints are additive. No breakage for existing callers.
- Clients can migrate at their pace. The expected migration is:
  consumers that do "retrieve then decide" → flip to "search
  (signals), decide, read (content) if needed."

## Implementation cost estimate

- `/confidence/{gene_id}` (derived view): ~40 LOC + a couple of joins
- Batched `POST /confidence`: ~20 LOC wrapper around the single-gene
  implementation
- Derived signals (string chromatin, disk_lag_s, resolved supersession,
  active_authors): ~60 LOC of derivation logic
- `/search` endpoint (signals-only shape of `/debug/preview`): ~30 LOC,
  reuses existing retrieval, strips content
- `/read/{gene_id}` alias for current `/genes/{id}` with a narrower
  documentation surface: ~10 LOC

**Total ~160 LOC, call it a 1-2 day session when prioritized.**

## Why not now

- Retrieval rank bottleneck (per `project_helix_retrieval_rank_bottleneck`
  memory note) is still the dominant failure mode — better recall at
  top-K beats API ergonomics.
- Layered fingerprints (`docs/FUTURE/LAYERED_FINGERPRINTS.md`) and
  genome sharding (`docs/FUTURE/GENOME_SHARDING.md`) are both ahead of
  this on the current roadmap.
- The refactor pays off most when an agent consults helix *reflexively*
  — which is a habit question downstream of the API shape, but also a
  skill-instruction question (getting agents to check confidence
  before acting).

Revisit trigger: first time an agent (or you) notices "I really should
have checked helix before I did X" after a confident-wrong action.
That's the empirical signal that the consumption pattern is ready for
the API to catch up.

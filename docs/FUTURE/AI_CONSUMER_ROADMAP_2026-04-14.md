# AI-Consumer Roadmap — Helix as seen from the LLM

> *"Make Helix session-aware in a way that reduces my redundancy and lets me
> trust individual pieces more. That's not more genes, it's better shape-of-gene."*
> — Raude, 2026-04-14 council session

Helix has been optimized from the perspective of the system operator
(latency, compression ratio, bench NDCG). This roadmap captures what an LLM
actually pays for at the consumption point: redundancy, opacity, and the
absence of session state. The fixes are mostly cheap LOC-wise; the leverage
comes from recognizing the consumer as a first-class stakeholder.

Status: **plan**, no code yet.
Date: 2026-04-14.
Source: AI-consumer council session transcript (Raude).

---

## TL;DR — Sprint plan

| Sprint | Item | Effort | Leverage | Status |
|---|---|---|---|---|
| **1** | Fired-tier tags per gene in response | ~30 LOC, 0.5d | high — converts guessing into calibration | 📋 pending |
| **1** | Hash previews (1-line for compressed-away content) | ~40 LOC, 0.5d | high — stops opaque-hash anxiety | 📋 pending |
| **1** | Per-gene confidence markers (◆/◇) | ~20 LOC, 0.5d | medium — visual trust cue | 📋 pending |
| **2** | **Session working-set register** | ~100 LOC + schema, 2d | **highest — the suspension primitive** | 📋 pending |
| **2** | `/context` "already delivered" markers | ~40 LOC, 0.5d | high (depends on Sprint 2) | 📋 pending |
| **2** | `/session/{id}/manifest` introspection endpoint | ~30 LOC, 0.5d | medium | 📋 pending |
| **3** | `/context/expand?gene_id=X` 1-hop neighborhood | ~80 LOC, 1d | high — replaces follow-up full queries | 📋 pending |
| **4** | Streaming `/context` response (top-k early) | ~120 LOC, 2d | medium — latency win only | 📋 pending |
| **5** | Session gravity attractor (touched genes boost) | ~60 LOC, 1d | medium (depends on Sprint 2) | 📋 pending |
| **DEFER** | Cost hints ("this returns ~N tokens") | ~40 LOC, 0.5d | low | 📋 pending |
| **DEFER** | Auto-ingestion of session knowledge | TBD | unclear; feedback-loop risk | — |
| **REJECT** | Full HPC fluid field helix-side | N/A | PWPC collab's domain, not ours | — |

**Current state:** nothing shipped. Sprint 1 is additive (no schema
migration, response-shape only) and should ship as one cohesive release:
"v0.5 — legible retrieval." Sprint 2 is the critical-path unlock for
Sprints 3 and 5.

## Dependency graph

```
                   (no deps — ship independently)
┌────────────────────┐
│  Sprint 1 bundle   │   fired-tier tags + hash previews + confidence markers
│  (legibility pack) │
└────────────────────┘

┌────────────────────┐
│  Sprint 2          │   session working-set register
│  (suspension)      │   — schema change, new ideas: TTL, scope, federation
└────────────────────┘
         │
         ├────────────► Sprint 3 (`/context/expand`)  [depends on session awareness for "already-seen" filtering]
         │
         └────────────► Sprint 5 (session gravity attractor)  [depends on session register for touch tracking]

┌────────────────────┐
│  Sprint 4          │   streaming response
│  (fluid)           │   — independent; nice-to-have latency, not structural
└────────────────────┘
```

**Critical path:** Sprint 2 is the unlock. Sprint 1 is free-money legibility;
Sprint 4 is nice-to-have latency; Sprints 3 and 5 both wait on Sprint 2.

---

## Sprint 1 — "Legibility pack" (~90 LOC, 1.5 days, ship as one release)

Three additive response-shape changes. No endpoint changes, no schema
migration. All three together make every existing `/context` response
substantially more useful to an LLM consumer.

### 1. Fired-tier tags per gene

**Problem:** the LLM gets 12 genes back with no idea which tier produced
each one. A gene pulled by `sema_boost` deserves different trust than one
pulled by `tag_exact` on a naming fragment. Without this signal the LLM
applies uniform weighting to a non-uniform retrieval.

**Fix:** the retrieval pipeline already knows which tier(s) scored each
candidate (that info already flows into `cwola_log.tier_features` for the
aggregate query). Plumb the per-gene breakdown into the response:

```xml
<GENE src="helix-context/ribosome.py" facts="..." fired="sema_boost=2.3,harmonic=0.8">
```

Implementation: ~30 LOC in `context_manager.py` response serialization.
Adds ~20 tokens per gene; well worth it.

### 2. Hash previews

**Problem:** `"[107 lines compressed to 0. Retrieve more: hash=7b4bb03ff2c3c193]"`
forces the LLM to either blindly re-query (cost) or skip (risk).

**Fix:** when compression elides content, emit a 1-line rendering summary
keyed to the hash. The ribosome already generates a summary during
compression; preserve it:

```xml
[107 lines compressed to 0. Retrieve more: hash=7b4bb03... (preview: chromatin transition thresholds for heterochromatin→euchromatin)]
```

Implementation: ~40 LOC in `ribosome.py` compression path + the response
serializer in `context_manager.py`. Requires the summary to be stable
across calls (cache on hash) so previews are deterministic.

### 3. Per-gene confidence markers

**Problem:** aggregate `context_health.ellipticity=0.48` is useful but blunt.
One or two of the 12 returned genes are usually a stretch; the LLM can't
tell which.

**Fix:** derive a per-gene symbol from the gene's normalized retrieval score:

| Score range (z-normalized) | Symbol | Meaning |
|---|---|---|
| z >= 1.0 | ◆ | strong retrieval signal |
| 0.0 ≤ z < 1.0 | ◇ | moderate |
| z < 0.0 | ⬦ | weak / reach |

Emit as a prefix on each `<GENE>` tag. 4 bytes per gene, huge downstream
effect on the LLM's calibration of which facts to trust vs hedge.

Implementation: ~20 LOC in the response serializer. No new computation
required — retrieval already produces scores.

### Validation

Metric: a follow-up council with the LLM after Sprint 1 ships, asking
specifically "are you now making fewer redundant follow-up queries per
conversation-thread." Qualitative. Quantitative alternative: track the
ratio of `/context` calls per session before vs after — prediction is
a 10-20% drop in same-session repeat queries.

---

## Sprint 2 — Session working-set register (~170 LOC + schema, 2-3 days)

**The critical-path item.** Everything in Sprint 3 and 5 depends on this.

### Problem

Every `/context` call is stateless. In a single conversation the same LLM
queries Helix 20-60 times. Many of those hits deliver overlapping gene sets
— the core concepts in the conversation's domain appear in nearly every
response. The LLM pays full token cost for content it's already holding.

### Design

New table `session_delivery_log`:

```sql
CREATE TABLE session_delivery_log (
    delivery_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    gene_id         TEXT NOT NULL,
    retrieval_id    INTEGER,           -- FK to cwola_log if present
    delivered_at    REAL NOT NULL,     -- epoch
    content_hash    TEXT,              -- to detect re-expression
    mode            TEXT                -- 'full' | 'compressed' | 'preview'
);
CREATE INDEX idx_sdl_session_gene ON session_delivery_log(session_id, gene_id);
CREATE INDEX idx_sdl_session_time ON session_delivery_log(session_id, delivered_at);
```

### API changes

1. **`log_delivery(session_id, gene_ids, mode)`** — called after every
   `/context` response flush. Writes one row per gene delivered.

2. **`already_delivered(session_id, gene_id, since=None)`** — returns
   `(delivered_at, mode, content_hash)` or `None`. The retrieval pipeline
   consults this *before* splicing — delivered genes get omitted (with a
   pointer stub) unless the content has changed.

3. **Response shape change:** when a gene is elided because already-
   delivered, emit:

   ```xml
   [gene=abc123 last delivered 4 queries ago, see earlier response]
   ```

### Semantics to decide (open questions)

- **TTL:** is delivery memory unbounded-per-session, or does it decay
  (e.g. after 20 queries since last touch)? Recommendation: unbounded
  per-session, clear when session ends.
- **Scope:** per-session only, or also per-party across sessions? Start
  with per-session; per-party is a federation question for later.
- **Content hash:** compute on the expressed (spliced) content, not the raw
  gene. Different splices of the same gene count as different deliveries.
- **Opt-out:** clients that *want* redundancy (e.g. benchmarks) should be
  able to pass `?ignore_delivered=true`.

### Validation

Prediction: 20-40% reduction in total response tokens per session for
conversations longer than 10 turns. Bench: synthetic multi-turn fixture
with known overlap structure.

---

## Sprint 3 — `/context/expand` 1-hop neighborhood (~80 LOC, 1 day)

**Depends on Sprint 2** (for already-seen filtering).

### Problem

Current flow: LLM reads a gene, wants to trace a thread (co-activated
concepts, harmonic_links, promoter-tag neighbors) — and has to invent a
full new query to approximate what the graph already knows.

### Design

New endpoint:

```
GET /context/expand?gene_id=X&direction=forward|backward|sideways&k=5
```

- **forward:** follows `harmonic_links.gene_id_a = X`
- **backward:** follows `harmonic_links.gene_id_b = X`
- **sideways:** co-activated genes from `epigenetics.co_activated_with`
- k caps the return count; default 5

Response is a compact gene list without full expression — each hit is just
the `<GENE>` summary tag, pre-filtered to exclude anything already in the
session delivery log. Designed for sub-100-token typical response.

### Why this matters

Today's alternative: full `/context` call with a synthetic follow-up query,
costing 2-8k tokens and re-running the whole 6-step pipeline. `/context/expand`
is sub-100 tokens, skips re-ranking and splice, and directly uses the already-
built graph. Shaves ~95% off typical follow-up queries.

---

## Sprint 4 — Streaming `/context` response (~120 LOC, 2-3 days)

### Problem

LLM waits 5-8 seconds for the full 12-gene bolus. During that time it could
already be composing its response if the top-k genes arrived early.

### Design

FastAPI streaming response (SSE or chunked). The pipeline already orders
genes by score; emit them progressively:

1. Top-3 genes: ship as soon as retrieval finishes (before re-rank completes
   for positions 4-12)
2. Positions 4-12: ship as re-rank + splice finish
3. Final: ship `context_health` footer

### Caveats

- Splice is currently batched (single LLM call for all 12 genes' trims).
  Streaming requires either per-gene splice calls (higher latency per gene,
  lower time-to-first-byte) or a hybrid (top-3 streamed raw, 4-12 batched).
- Client-side adoption matters — the LLM consumer has to know the response
  is a stream and start reading immediately. `Continue` and custom scripts
  today probably don't handle streaming `/context`.

### Why this is Sprint 4 (not earlier)

Latency is a nice-to-have, not a leverage point. Sprints 1-3 reduce
redundancy and improve trust. Sprint 4 just makes the already-good response
arrive slightly faster. Ship after 1-3.

---

## Sprint 5 — Session gravity attractor (~60 LOC, 1 day)

**Depends on Sprint 2.**

### Problem

Gravity currently operates at the genome level: co-activation weights pull
related genes together regardless of who's asking. A specific session builds
up its own focal area over many queries — that focal area should *itself*
exert gravity on future retrievals within that session.

### Design

During retrieval re-rank (`ribosome.re_rank`), add a bonus for candidates
whose `gene_id` or promoter tags overlap with the session's
`session_delivery_log` entries. Tunable weight, starts at +0.1 per match
capped at +0.5.

### Effect

Conversation topic-coherence: once an LLM is "in" a subject area, follow-up
queries retrieve from that neighborhood preferentially. Reduces "forgotten
context" where an adjacent-but-technically-matching gene from a different
domain edges in.

---

## What this buys the LLM consumer (estimated)

| Before | After Sprint 1 | After Sprint 2 | After Sprint 3 |
|---|---|---|---|
| 30 /context calls/session | 24 (−20% redundancy) | 18 (−40%) | 14 (−53%) |
| ~100k tokens on Helix output per session | ~90k | ~65k | ~55k |
| Per-call trust: uniform | Per-gene calibrated | Per-gene + session-aware | + expansion trails |

(Estimates are back-of-envelope. Validate with real-session traces.)

## Framing this against the earlier gravity/suspension/fluid question

Max asked: *do we have suspension and fluid states?*

This roadmap operationalizes the answer:

| State | Primitive | Sprint |
|---|---|---|
| Gravity (already present) | co-activation, cluster-gravity detection | existing |
| **Suspension (proposed)** | session working-set register (what's held in suspension *for this session*) | Sprint 2 |
| **Fluid (proposed, partial)** | streaming response (progressive disclosure as flow, not packet) | Sprint 4 |
| Gravity × Suspension | session gravity attractor | Sprint 5 |

The true continuous "fluid field" (HPC precision per Todd+Gordon) is rejected
as helix-owned scope (see `IMPLEMENTATION_ROADMAP.md §"What's NOT in this
plan"`). If and when PWPC ships its precision field as an emitted signal
batman's agreement head can consume, we adopt by subscription rather than
re-derive.

## Companion docs

- [`../collab/comms/COUNTER_MODE_SPEC_2026-04-14.md`](../collab/comms/COUNTER_MODE_SPEC_2026-04-14.md) — 4-regime dispatch,
  consumer-side of what the sliding-window matrix unlocks
- [`../collab/comms/LOCKSTEP_MATRIX_FINDINGS_2026-04-14.md`](../collab/comms/LOCKSTEP_MATRIX_FINDINGS_2026-04-14.md) — matrix-test
  findings (v1 retracted portions, see reply for v2)
- [`../collab/comms/REPLY_TO_PWPC_UPDATE_2026-04-14.md`](../collab/comms/REPLY_TO_PWPC_UPDATE_2026-04-14.md) — includes D1-D9
  5-coordinate proposal that this roadmap's session primitives align with
- [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) — existing
  Sprint 1-4 plan (stats/trajectory tracks); this roadmap is complementary,
  not competing

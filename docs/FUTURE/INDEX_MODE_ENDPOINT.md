# Index-Mode Endpoint — Helix as Pathway, Not Store

**Status:** Superseded 2026-04-17 by GT's build spec at
`docs/specs/2026-04-17-agent-context-index-build-spec.md` (commit
`f10fc8a`), which landed context-packet API + schemas + implementation
+ tests in one go. This sketch is kept for the framing sections (what
stays in Helix vs caller, confidence-propagation from Step 1b) that
the authoritative spec doesn't duplicate. Read the build spec first;
come back here if you want the reframe context.

**How the two relate:**
- GT's `/context/packet` ≈ this doc's `/index` + freshness labeling.
  GT chose endpoint naming, byte-range shape (gene-grained via
  `support_span`), and the `verified/stale_risk/needs_refresh` verdict
  set. All my open questions are answered there.
- This doc's Step 1b confidence fields (`coordinate_crispness`,
  `neighborhood_density`, `resolution_confidence`) stay on
  `ContextHealth` and are orthogonal to the packet's `live_truth_score`
  — one measures "how confident was the coordinate resolution,"
  the other measures "how fresh/authoritative is what we resolved to."
  Both are load-bearing; they compose.
- The packet builder doesn't currently read Step 1b confidence; that's
  a future integration point (weighing ⊗ freshness = true know-vs-go).

**Original sketch below preserved as historical framing:**

---

**Status:** Design sketch, 2026-04-17. Not a commitment. Drafted by
Laude after the SIKE pathway-layer reframe (see
`~/.claude/.../project_helix_weighs_not_retrieves.md`) and Step 1b
weighing-surface commit (`6d16b7a`). Written to give GT/Raude a
concrete shape to react to as the "additive index tool" direction
crystallizes.

---

## Why this exists

Helix's current `/context` endpoint is a **store-identity** surface:
it assembles gene bodies, applies a decoder prompt, counts expression
tokens, returns a compression ratio. All of that positions Helix as
the thing that *holds* the content.

The reframe (operator + external model, 2026-04-17) is that Helix is
the **pathway layer** — the geometric coordinate index that resolves
queries to locations and emits confidence in the resolution. The
content itself can live anywhere: today's `genome.db`, tomorrow's
Postgres, a mempalace instance, an S3 bucket, a git repo.

That makes `/context` the wrong front door for agents who want to
consume Helix as a *card catalog over their existing store*. They
don't need the book; they need the shelf number and a confidence
that the shelf has what they want.

This doc sketches `/index` (or `/locate` — bikeshed below) as the
pathway-layer endpoint that makes Helix composable instead of
competitive.

---

## Shape

### Request

```json
POST /index
{
  "query": "What port does Helix listen on?",
  "session_context": { ... },         // optional, same as /context
  "k": 12,                             // top-K coordinates to return
  "confidence_floor": 0.3              // optional; return < k results if
                                       //   no coord meets this floor
}
```

### Response

```json
{
  "query": "What port does Helix listen on?",
  "coordinates": [
    {
      "gene_id": "a1b2c3d4e5f6a1b2",
      "source_path": "helix-context/helix.toml",
      "byte_range": [1420, 1587],    // optional — if known
      "chromatin": "open",
      "score_raw": 2.34,              // absolute score for this coord
      "score_rank": 1,
      "confidence_local": 0.87,       // per-coordinate crispness
      "last_accessed": 1712345678.1
    },
    ...
  ],
  "resolution": {
    "confidence": 0.75,                 // aggregate confidence for this query
    "coordinate_crispness": 0.82,
    "neighborhood_density": 0.67,
    "verdict": "high" | "medium" | "low" | "empty",
    "known_empty": false                // true if query resolves to sparse region
  },
  "genome_metadata": {
    "total_genes": 7807,
    "index_version": "2026-04-17",
    "shard_set": ["main"]              // when sharding lands
  }
}
```

### Key differences from `/context`

| Property | `/context` | `/index` |
|---|---|---|
| Returns content body? | Yes (gene.content, spliced, decoded) | **No** — returns pointers only |
| Computes compression ratio? | Yes | No (meaningless — no content returned) |
| Runs decoder prompt? | Yes | No |
| Reports confidence? | Retrospective via ellipticity | **Pre-delivery via resolution.confidence** |
| Emits "known empty"? | No — always returns something | **Yes** — first-class negative answer |
| Cache model | Response-is-the-content | Response-is-the-pointer (cache by coord) |
| Caller fetches content? | No, Helix already did | **Yes, from source_path + byte_range** |

`/context` is the **decoder path** — Helix owns the whole pipeline.
`/index` is the **indexer path** — Helix owns only the coordinate
resolution; content fetch is the caller's responsibility.

---

## Confidence propagation

Step 1b shipped `coordinate_crispness`, `neighborhood_density`,
`resolution_confidence` on `/context` via `ContextHealth`. Null result
on the first signal pass (see
`benchmarks/results/needle_step1b_conf_null_2026-04-17.json`) —
crispness × coverage does not correlate with ground truth on the
10-needle bench.

For `/index` to be useful, confidence must actually discriminate. The
bench rig is in place; the remaining work is signal iteration
(raw top-score magnitude, top-50 dispersion, path-token coverage —
see handoff part-2 for the three candidates).

**Contract:** `/index` returns `resolution.confidence` as a 0-1
probability that the top-K contains the right answer. Calibration is
a product obligation — if we can't compute a meaningful number, we
return `verdict: "unknown"` and omit the scalar rather than shipping
garbage.

---

## Fetch contract

**Caller's responsibility** once they have a coordinate:

1. Read `source_path` — the file on disk (or an external store if we
   later plumb that through).
2. Optionally seek to `byte_range` for the exact slice.
3. Apply their own transformation (if any) — summary, full content,
   context window around the match, etc.

**Helix does not promise** that `source_path` is globally resolvable
for every caller. It's resolvable *relative to the workspace Helix
ingested from*. Adapter patterns (e.g. "prepend this base-path",
"route to S3 bucket X") are a future concern and not part of the MVP
contract.

**Byte ranges** are optional — some genes (parent aggregates, HGT
merges, external-source pseudo-genes) don't have them. When absent,
the caller falls back to `source_path` + their own chunking.

---

## Known-empty handling

This is the quietly most important property.

When `resolution.confidence < confidence_floor`, the response shape is
unchanged but `verdict: "empty"` and `coordinates: []`. The caller
uses this as the "I should go fetch fresh context" signal — not an
error condition, a first-class answer.

This is what distinguishes SIKE from a probabilistic retrieval layer.
A vector DB returns K nearest neighbors regardless of whether they're
meaningfully close; Helix's job is to distinguish "populated
coordinate" from "empty space in the index."

---

## Backwards-compat / coexistence

`/context` stays. It's the right surface for callers who want Helix
to own the whole pipeline (the Claude CLI context hook, the launcher
panels, the existing MCP integrations). No deprecation.

`/index` is additive. It's for callers who already have content
infrastructure (vector store, their own file indexer, mempalace, a
database) and want to *add* Helix's coordinate resolution on top.

Internally both endpoints share the same `_prepare_query_signals`
and `_express` path. The divergence is at the Step 5 "assemble"
boundary:
- `/context` runs `_assemble` → `ContextWindow` → decoder
- `/index` serializes the post-`_express` candidate list + scores
  directly, skipping assembly

This is cheap to build — the expensive parts (retrieval) are shared;
the cheap parts (serialization) are what diverges.

---

## What stays in Helix vs goes to caller

**Helix's job:**
- Resolve query → coordinates (the math doesn't change)
- Compute confidence (the signal iteration needs to produce something
  meaningful first)
- Surface `known_empty` when the query's coordinate region is sparse
- Expose gene_id, source_path, byte_range as stable pointers

**Caller's job:**
- Fetch content from the pointer
- Decide what to do with low-confidence responses (fetch-fresh,
  degrade gracefully, prompt-for-clarification)
- Maintain any content cache (Helix doesn't cache content under this
  contract; it caches coordinates)

---

## Open questions — for GT/Raude to answer

1. **Naming:** `/index` vs `/locate` vs `/pathway` vs `/resolve`.
   Pathway framing says `/locate`; index-tool framing says `/index`.
   Strong preference either way?

2. **Byte ranges:** do we commit to providing them for all genes?
   Current ingest stores source_path but not offset. Layered
   fingerprints' parent genes are aggregates — no meaningful offset.
   Option: return `{source_path, anchor_type}` where anchor_type is
   `{file, chunk, parent, hgt}` and let the caller decide how to
   fetch.

3. **Mode flag vs separate endpoint:** alternative design is
   `/context?mode=index` (flag toggle) — same endpoint, different
   output shape. Pro: one code path. Con: response schema becomes
   conditional, OpenAPI gets messy.

4. **Sharding interplay:** when phase-2 sharding lands (paused
   pending Step 1 right now), `/index` naturally fits the
   fan-out/merge pattern — each shard returns coordinates, router
   merges by score. `/context` has to assemble content from every
   shard, which is a more complex merge. `/index` might actually
   unblock sharding by giving us a simpler merge path.

5. **Confidence calibration as a product obligation:** agree that
   `resolution.confidence` should return `null`/`"unknown"` when we
   can't compute something meaningful, rather than shipping a
   garbage scalar? Null result on current signals (Step 1b) makes
   this decision load-bearing.

6. **Push protocol:** once `/index` exists, pushing "hey, I updated
   the file at this path" from caller → Helix becomes cheap. Does
   that deprecate the ingest pipeline or live alongside?

---

## Where this fits in the sequence

1. **Step 1b.5+ — confidence signal iteration** (must happen first)
   — without a discriminating signal, `/index` is useless.

2. **Adapter surface skeleton** — define `/index` route, reuse
   `_express` output, skip `_assemble`. Should be ~1 session.

3. **Push-style ingest contract** — consumer-driven. "I just updated
   X, re-resolve any queries that pointed near its coordinate." A
   lot of this falls out of `/index` existing.

4. **External-store adapter protocol** — `source_path` resolution
   via pluggable resolver (filesystem, S3, git, external DB).

5. **Phase-2 sharding** — currently paused. `/index` may change what
   sharding's ideal merge looks like; re-evaluate after (1) lands.

---

## Related

- `project_helix_weighs_not_retrieves.md` (memory) — identity reframe
- `~/.helix/shared/handoffs/2026-04-17_session_close_laude_part2.md` — Step 1b arc
- `docs/FUTURE/GENOME_SHARDING.md` — how routing interacts with this
- `benchmarks/results/needle_step1b_conf_null_2026-04-17.json` —
  why confidence iteration is the gating step
- `helix_context/schemas.py::ContextHealth` — where new fields live
- `helix_context/context_manager.py::_compute_health` — where
  confidence currently computes (for `/context`; `/index` would reuse)

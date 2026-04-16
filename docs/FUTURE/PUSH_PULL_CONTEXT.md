# Push-Pull Context — The API Contract Between Consumer and Genome

**Status:** Design direction, 2026-04-16. No code yet. Captures the
integration-layer decision that falls out of two parallel design
moves: sharding the genome
([GENOME_SHARDING.md](GENOME_SHARDING.md)) and making the walker
pluggable ([WALKER_PATTERNS.md](WALKER_PATTERNS.md)). This doc says
*how* the consumer and the genome talk — the contract that makes
both decisions cohere.

---

## The contract in one paragraph

**Push the fingerprint eagerly, pull the content lazily.** Every
consumer turn gets tier-score fingerprints injected into its context
without being asked — helix has *presence*. Content (T1 key_values,
T2 complement, T3 source, T4 neighbor walk) is never pushed; it's
only fetched when the consumer explicitly asks for it. The fetch can
be executed by the consumer itself or delegated to a librarian —
that's a walker choice, not an API choice.

## Push: what helix puts in front of the consumer

On every query, helix emits a fingerprint payload *without the
consumer having to ask*. The payload is already the `/context`
endpoint's return today; the shift is semantic: treat it as a **push
channel**, not a pull response.

```
Fingerprint payload (push, ~150 tok/gene × top_k):
  ┌────────────────────────────────────────────┐
  │ gene_id: ade070a94a3a118c                  │
  │ source: fleet/skills/db.py                 │
  │ fused_score: 21.53                         │
  │ tier_contributions:                        │
  │   fts5: 6.0, sema_boost: 0.53,             │
  │   harmonic: 3.0                            │
  │ domains: [sql, python]                     │
  │ entities: [SQL, db.py, NETWORK]            │
  └────────────────────────────────────────────┘
```

The consumer now **knows helix is there** — it sees the top-k genes
ranked for the current turn, the structural signals, the entities.
It can reason about coverage ("do I have the right files?") and gaps
("this query is about X but no X-domain genes surfaced") without
spending a single content token.

Presence at near-zero cost is the unique property. RAG systems don't
have it: they push chunks (content) and make you pay for that content
whether or not you actually needed it.

## Pull: what the consumer fetches on demand

Four tiers of increasing cost, pulled only when the fingerprint
surfaces a gene worth investigating:

```
T1 key_values      ~63 tok median   structured k=v pairs
T2 complement     ~381 tok median   compressed semantic summary
T3 content        ~864 tok median   verbatim source
T4 neighbor walk  variable          follow harmonic_links / entity edges
```

Pull is addressable: `{gene_id, tier}` in, content out. No scanning,
no ranking, no side effects. The tier chosen is determined by what
the consumer needs — scalar extraction goes to T1, semantic reasoning
to T2, exact wording to T3, cross-gene navigation to T4.

A typical session uses push-heavy, pull-sparse: 10 pushes for every
1 pull is the target shape. RAG's ratio is roughly the inverse.

## Walker choice: orthogonal to the contract

The push-pull contract doesn't specify *who* executes the pull. Two
implementations of the same API:

- **Self-walk** — consumer issues `get_gene(gene_id, tier=T2)`,
  content lands in consumer context, consumer reasons.
- **Librarian dispatch** — consumer issues a directive
  `{gene_id, tier, task: extract, target: ...}`, a local LLM pulls
  content on its behalf, returns a scalar packet, consumer reasons
  over the packet.

Both satisfy the contract. The choice is a deployment knob, not an
API change. See [WALKER_PATTERNS.md](WALKER_PATTERNS.md) for the
decision rule and the directive schema.

The contract's job is to make sure the walker swap is invisible to
the consumer's code path above the "pull" call. Consumer-side code
looks the same either way:

```python
# Consumer code (unchanged across walker choice):
result = helix.pull(gene_id, tier="T2", task="extract", target="port")
# result: {"answer": "11437", "source": gene_id, "tokens": 412}

# Walker choice lives in helix.pull() dispatch:
#   - if context_pressure > threshold: dispatch to librarian
#   - elif multi_file and parallel_workers_available: dispatch
#   - else: self-walk (inline fetch)
```

## Why this is different from RAG

RAG push/pull is inverted from helix's. Traditional RAG:

```
Consumer: "I have a question: X"
  RAG:    "Here are 3 chunks of content, good luck."
  Consumer: [reasons over content, might also need more chunks]
  Consumer: "I also need chunks about Y"
  RAG:    "Here are 3 more chunks."
  ...
```

RAG pushes content (paragraph dumps). The consumer has no way to see
the retrieval landscape — is there a better chunk one hop away? What
domains did the retriever think this query touched? It's opaque.

Helix push/pull:

```
Consumer: "I have a question: X"
  Helix:  [pushes fingerprints — 10 genes, scores, entities, domains]
  Consumer: [reads landscape, decides: gene 3 has the entity I need]
  Consumer: "pull gene 3, T2, extract 'port'"
  Helix:  "11437"
  Consumer: [answers with confidence, cites gene_id]
```

Helix pushes *awareness* and pulls *answers*. The consumer's context
never bloats with paragraphs unless a paragraph is the answer.

## Packet discipline

The return from a pull is a **packet**, not a paragraph. Shape:

```json
{
  "gene_id": "ade070a94a3a118c",
  "tier": "T2",
  "answer": "11437",
  "confidence": "high",
  "source_span": "lines 24-26",
  "tokens_used": 412
}
```

Contrast with RAG's return, which is raw content. Packet discipline
is what keeps consumer context from bloating. A walker (self or
librarian) that returns a paragraph when a scalar was requested is
violating the contract.

For cases where the consumer *does* need a paragraph (summarisation,
rewriting, contradiction detection), `task: "passage"` in the pull
request returns the tier content verbatim. This is the escape hatch,
and using it is a deliberate context-spend.

## Implementation sketch

Today, `/context` is already the push channel — it emits fingerprints
on every turn. What's missing is:

1. **A pull endpoint with task semantics** — not just `get_gene(id)`
   but `pull(gene_id, tier, task, target)`. Task can be `extract`
   (scalar), `passage` (paragraph), `walk` (neighbor expansion),
   `summarise` (librarian-computed abstract).

2. **Walker dispatch** — `pull()` decides self-walk vs librarian based
   on context pressure, gene count, task type (see
   `WALKER_PATTERNS.md`).

3. **Packet formatter** — librarian returns are already packet-shaped
   (see directive schema in `WALKER_PATTERNS.md`). Self-walk needs a
   formatter that extracts the same shape from raw tier content.

4. **Sharded fetch** — `pull()` routes through
   `GENOME_SHARDING.md`'s shard router to open only the .db file the
   gene lives in. Noise isolation at the pull layer.

None of this breaks existing consumers. `/context` keeps working,
`get_gene()` keeps working. `pull()` is additive.

## What changes for the consumer

For a naive consumer (today's Laude/Taude sessions): nothing. The
fingerprints keep appearing in context, `get_gene()` keeps returning
raw content. The contract isn't visible until the consumer opts in.

For a **protocol-aware** consumer (future helix-library clients, SNOW
benchmark, Mamba head on Celestia's side): the contract is explicit.
The consumer decides when to pull, at what tier, with what task, and
whether to dispatch or self-walk.

That opt-in gradient is the right shipping path. Don't force the
contract on anyone; let consumers that want its benefits adopt it.

## Open questions

### How does the consumer learn the contract?

Probably a `/schema` endpoint that returns the push shape, the pull
shape, the task vocabulary, the walker options, and the escalation
rules. Helix documents itself to its consumers.

### Versioning

When the tier vocabulary changes (e.g., we add T5 graph-walks), the
contract needs to version cleanly. `/schema` includes a protocol
version; consumers negotiate.

### Streaming vs request/response

Today everything is request/response. For walker dispatch with
parallel librarians, streaming partial packets as they complete is
faster than waiting for the slowest. Worth spelling out before
implementation.

### Authentication / scope

Once shards have identity boundaries (per
`GENOME_SHARDING.md`), the push channel needs to respect scope.
Consumer in a participant session shouldn't see org-shard
fingerprints. This is entanglement with the identity registry, not
the contract itself, but the contract surface (`/context`,
`/pull`) is where the scope check sits.

## Related

- [GENOME_SHARDING.md](GENOME_SHARDING.md) — storage layer. The push
  channel reads fingerprints from `main.db`; the pull channel fetches
  content from the category shards. Sharding is how the contract is
  physically realised on disk.
- [WALKER_PATTERNS.md](WALKER_PATTERNS.md) — execution layer. The
  walker is the entity that actually services a pull. Self-walk and
  librarian dispatch are two implementations of the pull side of the
  same contract.
- [../../benchmarks/snow/](../../benchmarks/snow) — the SNOW
  benchmark exercises the pull side end-to-end. The cascade it
  measures (T0 → T1 → T2 → T3 → T4) *is* a trace of the pull
  protocol in action.
- [../DESIGN_TARGET.md](../DESIGN_TARGET.md) — the existing API
  surface (`/context`, `/query`, `/walk`) that this contract sits on
  top of.

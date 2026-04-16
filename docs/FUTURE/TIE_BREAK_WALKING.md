# Walking Tie-Break — Associative-Graph-Informed Ordering for Tied Top-k

**Status:** Prototype landed 2026-04-15, opt-in via `HELIX_WALKING_TIEBREAK=1`.
Not enabled by default. This document captures the investigation that
produced it, the empirical evidence, the ladder design, and the open
questions that remain before it becomes default behaviour.

**Origin:** a session that started out asking whether helix needed
`Decimal` arithmetic (like `BookKeeper`) for scoring precision, and
ended with the realisation that the *real* ordering question helix
faces isn't precision — it's what to do when 12-tier fusion produces
bitwise-identical scores for two or more genes. The investigation
walked a chain:

```
"Is helix a calculator?"
  → determinism probe (Pass 1):          CALCULATOR_GRADE
  → precision sensitivity (Pass 2):      FLAG_ONLY (not a problem)
  → tie inspection (Pass 3a):            96% of queries have ties
  → graph coverage of ties (Pass 3b):    94.6% of ties have signal
  → A/B of walking tie-break (Pass 3c):  48% of queries reorder
```

Each step was cheap (5–45 minutes). Each step either answered the
question or moved it. What follows is the summary — the raw probe
output is preserved under `benchmarks/precision_*_2026-04-15.json`.

## The question, honest version

**When 12-tier fusion gives gene A and gene B the same final score,
which one should helix surface first?**

Today the answer is "whichever Python's `dict` iterated first" —
effectively a roll of the dice determined by insertion order at the
accumulation step. For **96% of queries** (24 of 25 on the benchmark
genome) this coin flip happens at least once somewhere in top-k. For
**32% of queries** (8 of 25) it happens in the *head* of top-k
(ranks 0–3) — the slots the consumer LLM actually reads first.

The head ties matter most, because LLM consumers anchor on the first
result they see. A tie at rank 0 means "the most important answer is
arbitrary." That's the opposite of a calculator.

## The journey — why we didn't just build `Decimal` fusion

A reasonable first instinct, given the question: **swap the float
accumulator for `Decimal` and let bookkeeper-grade arithmetic decide.**

We investigated that path first, and three things ruled it out:

1. **Determinism is already clean.** Pass 1 showed 25/25 queries
   produce bitwise-identical top-k gene IDs and scores on back-to-back
   runs. `max_abs_delta = 0.00e+00` on every query. No hidden
   nondeterminism from numpy, dict iteration, or concurrent futures.
   So the baseline is already "calculator-grade" in the coarse sense
   — the fusion math is stable.

2. **Precision isn't where the problem is.** Pass 2's sensitivity
   analysis found **zero** adjacent score pairs in the float32-risky
   zone (relative gap between 1e-6 and 1e-15). The gap distribution
   is bimodal — pairs are either **exact ties** (13% of pairs) or
   **comfortably separated** (gaps ≥ 1e-4, 87% of pairs). There is
   no "middle ground where Decimal vs float could flip the ordering."
   The bookkeeper intuition was correct to check; it didn't fire.

3. **Decimal wouldn't change the tied pairs.** The tied pairs in
   Pass 3a have bitwise-identical per-tier contributions —
   `harmonic=3.0 + lex_anchor=6.0 + tag_exact=3.0 + tag_prefix=3.0`
   on both genes. Identical inputs produce identical outputs in
   any arithmetic mode. Decimal is the wrong tool.

The right framing: **ties are structural, not numerical.** The fusion
math isn't getting them wrong — it's correctly reporting that the
direct-evidence tiers give these genes equal weight. To order them,
we need *off-path* signal: information the direct-match tiers don't
carry. The associative graph is that information.

## The human-memory analogy that shaped the design

> "Human minds often recall the wrong thing first to 'walk' to the
> relevant item." — Max, 2026-04-15

This observation reframed tie-break from edge-case-handling into
*the mechanism of associative recall*. Humans don't pick between tied
candidates by coin flip. The "wrong-but-related" item is a stepping
stone — retrieval activates a neighborhood, and the walk starts from
whichever item is closest to the current context, then uses the
associative graph to move toward the target.

Helix has the substrate for the same behaviour:

- `harmonic_links` — Hebbian co-activation edges (192,612 rows on the
  bench genome)
- `entity_graph` — entity-to-gene index (128,346 rows)
- `gene_relations` — NLI-typed logical relations (82,245 rows)
- `path_key_index`, `promoter_index` — structural indices

These tables are already populated. They participate in retrieval via
specific tiers (SR multi-hop, harmonic, tag_exact). They do **not**
participate in tie-break. That's the gap the walking tie-break
closes — the graph signal becomes a tie-break substrate on top of
being a retrieval substrate.

## Empirical basis — graph coverage on real ties (Pass 3b)

We extracted the 74 tied adjacent pairs from Pass 1's top-k data and
measured how often helix's associative tables have enough signal to
distinguish them.

```
aggregate (N=74 tied pairs):
  direct harmonic edge present:    34   (45.9%)
  neighborhood size asymmetric:    70   (94.6%)   ← universal signal
  NLI relation between:            24   (32.4%)
  ANY signal present:              70   (94.6%)
  NO signal (graph-invisible):      4   (5.4%)    ← all tail ties in one query

  head ties (rank 0-3):            25
  head ties with ANY signal:       25   (100%)    ← every consequential tie
                                                    has graph signal
```

**Key finding: neighborhood-size asymmetry is the universal signal.**
When two genes are tied on direct evidence, one of them is nearly
always more graph-central than the other — often by 3-10×
(e.g. 288 neighbors vs 27). The graph is telling us which gene is
the "better-connected" answer even when the direct-match tiers can't.

## The ladder

The walking tie-break applies a ladder of pairwise comparators to
each tied score group. Each rule either decides (returns a preference)
or abstains (falls to the next rule). Single-gene groups pass through
unchanged. Score ordering is preserved — only within-tie ordering
changes.

```
Rule 1 — strong_edge_freshness
  If harmonic_link(a, b) exists with weight ≥ 0.7:
    These are graph-validated near-twins. Co-activation has
    verified they travel together. Pick the fresher of the two
    (larger rowid — more recent ingest).

Rule 2 — neighborhood_size
  Count distinct harmonic neighbors for each gene. Pick the one
  with MORE neighbors. Central genes beat peripheral ones —
  they're more likely to participate in the consumer's walk.

Rule 3 — nli_entailment
  If gene_relations has (a entails b), prefer a (strictly more
  informative). Contradictions abstain — should be surfaced
  explicitly, not resolved silently.

Rule 4 — freshness_fallback
  Prefer the higher rowid (more recent ingest). Recency is a
  real signal; never arbitrary.

Rule 5 — lexical_gene_id
  Compare gene_ids lexically. Deterministic total order.
  Only reached when every other rule abstained.
```

The order matters. `strong_edge_freshness` comes first because a
weight-0.95 edge is the graph declaring "these are functionally
equivalent from the retrieval side; you should pick on some other
axis." Neighborhood size is a weaker signal for graph-validated
twins (both may be central) but is decisive for the general case.
NLI sits below both because logical entailment is per-pair and
narrow; the earlier rules catch the broader cases.

## Prototype A/B results (Pass 3c)

Same 25 queries, same genome, two runs: baseline (insertion-order)
vs walking (`HELIX_WALKING_TIEBREAK=1`).

```
queries with ANY reorder:    12/25  (48%)
queries with HEAD reorder:    8/25  (32%)
total rank positions changed: 66

decisive rule per swap:
  neighborhood_size       54   (81.8%)    ← the workhorse
  strong_edge_freshness   11   (16.7%)    ← graph-twin cases
  lexical_gene_id          1   (1.5%)     ← one true fallthrough
  nli_entailment           0   (0.0%)     ← earlier rules always decided first
  freshness_fallback       0   (0.0%)     ← never reached
```

The prototype isn't a no-op — it reorders half the queries on this
benchmark. Most decisions come from the single cheapest signal
(neighbor count). The ladder holds together: no rule leaked wrong
decisions to the fallback, the lexical floor fired exactly once.

The **head reorders** (rank 0-3) are the consequential ones, because
that's what consumer LLMs see first. 32% of queries get a different
gene at the top when walking is enabled. Whether that's semantically
*better* requires downstream evaluation (see Open Questions).

## What this is NOT

- **Not a scoring fix.** The 12-tier fusion math is unchanged.
  Walking tie-break runs *after* `sorted(gene_scores, ...)`. Scores
  are preserved; only within-tie order changes.

- **Not a `Decimal` path.** Decimal was investigated and shelved
  (see "The journey" above). The ties are structural. Decimal of
  identical inputs = identical outputs.

- **Not dedup.** The near-twin cases (two Steam grid.json files, two
  drafts of the same file) are *legitimate distinct genes* — they
  have different `source_id`s and different content lengths. Deduping
  would lose real distinctions. Walking tie-break orders them; it
  doesn't collapse them.

- **Not a consumer-specific signal yet.** The ladder uses only
  intrinsic-to-helix signals (graph topology, ingest order, typed
  relations). A consumer LLM's *current context* — which genes it
  just retrieved, which entities are in its working memory — is a
  stronger signal, but one the consumer has to send in the request.
  The "session working-set" and `session_id` hooks exist; wiring
  them into tie-break is a future upgrade.

## Open questions

1. **Does walking tie-break produce semantically *better* top-k?**
   The A/B shows *different* top-k. To judge better/worse we need
   either:
   - Human-labeled relevance judgments on the 8 head-reorder queries,
     comparing baseline vs walking top-1 per query.
   - LLM-as-judge: same query, both top-k lists shown to a strong
     model, asked which ordering better supports the question.
   - End-to-end: measure answer accuracy on a fixed benchmark with
     flag on vs off. Most rigorous; most expensive.

2. **Should tie-break be consumer-aware?** Rule 1 could weight the
   freshness fallback by "did the consumer just retrieve a neighbor
   of one of these genes" (via session working-set). That's the
   "walk from where the consumer is standing" version of the
   human-memory analogy. Requires wiring the session_id through
   `query_genes` into `tie_break`.

3. **Is rank-0 stability a feature or bug?** Today, with insertion-
   order tie-break, rank-0 on a tied query is unstable — change
   the ingest order and the "best" answer shifts. Walking tie-break
   makes rank-0 deterministic from genome state. But when the genome
   changes (new ingests → new harmonic_links), rank-0 can shift
   even for the same query. This is probably *correct* (new evidence
   should matter), but worth documenting in API contracts.

4. **Should the tie itself be surfaced?** An alternative to "pick
   one" is "surface both and annotate the tie" — let the consumer
   LLM decide. This is the introspection-as-feature path from
   DESIGN_TARGET.md §3. Could be complementary: walking picks the
   *primary* ordering, but response includes `tied_with: [gene_id,
   gene_id]` for consumers who want the full picture.

5. **Does the tie frequency decrease as the genome matures?** On
   this 18K-gene benchmark, 96% of queries have ties. As ingest
   grows and per-gene scores spread across more contributors, ties
   may thin out. Worth re-measuring on a larger / more-mature genome
   before building infrastructure that assumes tie-break is always
   consequential.

6. **Is the 4 "graph-invisible" pairs (5.4%) a real gap?** All four
   are tail ties in a single query with genes that have zero
   harmonic_links neighbors — likely recent ingests before co-
   retrieval history built up. The freshness_fallback rule would
   handle them, but never fired in the A/B because earlier rules
   consumed the cases. Over time, these will resolve naturally as
   co-activation accumulates. For now: graceful fallback via
   lexical gene_id is fine.

## Revisit triggers

- After a larger-genome benchmark (100K+ genes) — does the tie
  frequency hold, drop, or rise?
- After the session working-set is wired through — consumer-aware
  walking is the natural next move.
- After we have relevance judgments — either human or LLM-as-judge
  — on the head-reorder queries.
- If a consumer reports rank-0 instability across genome-updates
  as a usability issue — then open question (3) needs a formal
  answer in the API contract.

## Implementation location

- **Core logic:** [helix_context/tie_break.py](../../helix_context/tie_break.py)
- **Hook site:** [helix_context/genome.py](../../helix_context/genome.py),
  immediately after `ranked_ids = sorted(...)`
- **Opt-in flag:** `HELIX_WALKING_TIEBREAK=1` environment variable
  (soft-fails to baseline on any exception; no permanent state)
- **Evidence files:** `benchmarks/precision_probe_2026-04-15.json`,
  `benchmarks/precision_sensitivity_2026-04-15.json`,
  `benchmarks/precision_tie_graph_coverage_2026-04-15.json`,
  `benchmarks/precision_tie_break_ab_2026-04-15.json`
- **Benchmark scripts:** `benchmarks/precision_probe.py`,
  `benchmarks/precision_sensitivity.py`,
  `benchmarks/precision_tie_inspection.py`,
  `benchmarks/precision_tie_graph_coverage.py`,
  `benchmarks/precision_tie_break_ab.py`

## Connection to `docs/DESIGN_TARGET.md`

This investigation exercises three of the design-target principles
explicitly:

- **§1 (token cost dominates)** — wrong tie-break resolution puts
  lower-quality genes in the first tokens the consumer reads.
  Token cost ≠ relevance cost; they compound.
- **§3 (introspection is a feature)** — open question (4) proposes
  surfacing the tie itself to the consumer as a first-class
  `tied_with` field. Pure application of the principle.
- **§5 (LLM-to-LLM coordination)** — the consumer-aware walking
  (open question 2) is a natural extension: "which gene is more
  associatively close to what this agent was *just* doing."

Walking tie-break is one of the first features designed explicitly
after the LLM-as-primary-consumer reframe. It's small — a few
hundred lines of code and five probe scripts — but it's the pattern
we expect to repeat: use signals helix *already computes* to close
gaps in the consumer-facing surface, without adding new ML, new
tables, or new complexity.

---

*Established 2026-04-15. Update with results from the open questions
as they're answered.*

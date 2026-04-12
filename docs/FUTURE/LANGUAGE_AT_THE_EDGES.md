# Language at the Edges, Math in the Middle

> *"The lightning at ingest IS our neuron firing. Then at reactivation
>  it's the same shape."*
> — Max, 2026-04-12

A design document for math-only ingest + lightning-strike co-activation
+ lazy LLM annotation. Status: **future direction, not yet implemented.**
The current pipeline uses LLMs at both ingest and expression. This
document argues the ingest-time LLM is redundant and removing it would
make the architecture both more efficient and more biologically accurate.

---

## The core observation

Biology's information substrate has no language layer. DNA base pairs
bind by chemical affinity. Proteins fold by geometry and charge.
Ribosomes don't "read" mRNA — they bind by steric fit. The entire
biological encoding-and-expression system runs on physics and geometry,
zero linguistics.

**Our current pipeline has LLMs doing two different jobs:**

1. **Ingest-time**: "Read this file and tell me what it's about" → produces
   `complement`, `intent`, `summary`, `codons`, `key_values`.
2. **Expression-time**: Downstream LLM reads `expressed_context` and
   reasons about the user's query.

Job #2 is load-bearing. The user asked a question in language, so some
language model must read the result and reason about it.

Job #1 is **anthropomorphic contamination** — we're projecting language
onto a substrate that doesn't need it. The downstream LLM at expression
time already performs interpretation. Pre-interpreting at ingest is
redundant computation.

## The neuron-firing insight

The lightning-strike ingest technique (see `ray_trace.py` origin story)
records the fractal pattern of which existing genes "light up" when a
new gene enters the genome. Rays cast from the new gene's embedding
position traverse the BVH of existing genes; the hit pattern is the
gene's innate functional neighborhood.

**This is neuron firing.**

When a biological neuron first fires in response to a novel stimulus,
it simultaneously activates a specific set of downstream neurons based
on synaptic weights and proximity. The pattern is not random — it
reflects the structural organization of the cortex at that moment.
This is the substrate Donald Hebb described in 1949:

> *"Neurons that fire together wire together."*

The lightning strike at ingest is exactly this:
- New gene enters the genome (new stimulus arrives)
- Rays cast through the BVH (action potential propagates)
- Hit pattern recorded (synaptic connections formed)
- Pattern becomes the gene's innate signature (engram)

**At reactivation, the same pattern fires again.** This is how
biological memory works — not as stored content, but as reproducible
firing patterns. Retrieval is pattern completion: you present a partial
pattern, the network fills in the rest by firing the same neurons that
fired during ingest.

Hopfield networks formalized this in 1982. Modern neuroscience has
identified engrams as specific sets of neurons whose coordinated firing
IS a memory. The memory isn't stored somewhere — the memory IS the
reproducible firing pattern.

## Mapping to helix-context

```
Biology                       helix-context
─────────────────────────────────────────────────────────────────
Novel stimulus arrives    →   New gene ingested
Action potential fires    →   Rays cast from new gene's position
Synaptic pattern forms    →   Innate co-activation neighbors recorded
Engram consolidated       →   Fractal signature stored with gene
─────────────────────────────────────────────────────────────────
Partial cue presented     →   Query enters pipeline
Pattern completion fires  →   Retrieval activates same fractal pattern
Same engram activates     →   Same neighbors light up
Memory is "recalled"      →   Gene expressed in context window
```

The shape IS the same. At ingest: the new gene's position in embedding
space fires a specific pattern into the genome's existing structure. At
retrieval: the query's position fires a pattern that overlaps with the
same structure. When the patterns overlap, the gene activates — not
because we "found" it via search, but because **the same neurons fired
that fired during ingest**.

This is substrate-level retrieval. No language required — just
reproducible firing patterns.

## What LLM ingest currently adds vs what math provides

| Dimension | Current LLM source | Math-only source |
|---|---|---|
| **FTS5 content** | raw text | ✅ pure tokenization |
| **SPLADE sparse** | ModernBERT encoder | ✅ neural encoder (deterministic, not generative) |
| **Embedding (20D SEMA)** | sentence-transformer | ✅ already math-only |
| **Promoter domains** | LLM extraction | ⚠️ regex + NER = ~60-70% quality |
| **Promoter entities** | LLM extraction | ⚠️ spaCy NER covers most cases |
| **Promoter intent** | LLM generation | ❌ no math analogue (requires interpretation) |
| **Promoter summary** | LLM generation | ❌ abstractive summarization needs an LLM |
| **Complement** | LLM summarization | ⚠️ extractive summary via spaCy sentence density |
| **Codons (semantic labels)** | LLM extraction | ⚠️ noun chunks + function names = partial |
| **Key-values** | LLM extraction | ⚠️ regex patterns (CpuTagger already has these) |
| **Innate co-activation** | not currently built | ✅ **lightning strike at ingest** (new!) |
| **Fractal signature** | not currently built | ✅ box-counting dim of strike pattern (new!) |
| **Source metadata** | filesystem | ✅ pure metadata |
| **Chromatin tier** | density gate | ✅ pure math |
| **Epigenetics** | access counters | ✅ pure math |

Of 15 signals:
- **10 are math-native** (no LLM needed)
- **4 degrade gracefully** with math-only fallbacks (60-80% quality)
- **2 genuinely require language** (intent, abstractive summary) — but
  neither is read during retrieval scoring; they're decoration for
  humans inspecting the DB.

## The architectural split

### Current (LLM at both ends)

```
┌─────────────────────────────────────────────────────────┐
│  INGEST                                                  │
│  ─────                                                   │
│  content → LLM.pack() → complement, codons, promoter    │
│          → LLM.extract_kv() → key_values                │
│          → SPLADE encode → sparse terms                 │
│          → Embed → 20D SEMA                             │
│          → upsert_gene()                                │
└─────────────────────────────────────────────────────────┘
                          ↓
                    [genome.db]
                          ↓
┌─────────────────────────────────────────────────────────┐
│  QUERY                                                   │
│  ─────                                                   │
│  query → 13-dim retrieval → top-k genes                 │
│        → Kompress compress → expressed_context          │
│        → Downstream LLM reads → answer                  │
└─────────────────────────────────────────────────────────┘

LLM cost: 2 API calls per gene at ingest + 1 API call per query.
```

### Proposed (language at the edges only)

```
┌─────────────────────────────────────────────────────────┐
│  INGEST (math only, no LLM)                             │
│  ─────                                                   │
│  content → CpuTagger (regex + spaCy) → promoter         │
│          → SPLADE encode → sparse terms                 │
│          → Embed → 20D SEMA                             │
│          → Ray-cast via OptiX → innate_coactivation     │
│                              → fractal_signature        │
│          → upsert_gene()                                │
└─────────────────────────────────────────────────────────┘
                          ↓
                    [genome.db]
                          ↓
┌─────────────────────────────────────────────────────────┐
│  QUERY (pattern completion)                              │
│  ──────                                                   │
│  query → Embed query → position in manifold             │
│        → Ray-cast from query position → matching pattern│
│        → 13-dim retrieval (innate graph available)      │
│        → Kompress compress → expressed_context          │
│        → Downstream LLM reads → answer                  │
└─────────────────────────────────────────────────────────┘
                          ↓
            [Lazy LLM annotation for hot genes]
                          ↓
  if gene.access_count > N and gene.complement is None:
      gene.complement = llm.summarize(gene.content)
      gene.promoter.intent = llm.extract_intent(gene.content)

LLM cost: 0 at ingest. 1 at query. Annotation only for genes that
earn it through usage (biological pattern: interpretation follows
activation, not precedes it).
```

## Benefits

| Metric | LLM ingest (current) | Math-only ingest |
|---|---|---|
| Ingest throughput | ~30 genes/sec | 200+ genes/sec |
| Ingest cost per 10K genes | $30-500 | $0 |
| Determinism | temperature-dependent | fully reproducible |
| Scalability | API rate limits | GPU memory bound |
| Failure mode | codon table mismatch when pipeline changes | stable |
| Hot genes get interpretation | always (wasteful) | only when needed (lazy) |
| Cold genes get interpretation | always (wasteful) | never (correct) |
| Biological accuracy | poor (anthropomorphic) | high (substrate-level) |

## The lazy annotation principle

Biology's actual pattern: **genes aren't pre-annotated by scientists.
They get annotated when someone studies them.** Most of GenBank is
math-derived sequencing data. Interpretation is retroactive and sparse.

For helix:

```python
def on_gene_access(gene_id: str):
    gene = genome.fetch(gene_id)
    # Hot gene threshold — interpretation cost is justified
    if (gene.access_count >= ANNOTATION_THRESHOLD
            and gene.complement is None):
        gene.complement = llm.summarize(gene.content)
        gene.promoter.intent = llm.extract_intent(gene.content)
        gene.promoter.summary = llm.one_line_gist(gene.content)
        genome.update(gene)
```

Effect: the 10% of genes that get accessed frequently get
human-readable metadata. The 90% that sit unused keep their math-only
signatures. Total LLM cost scales with ACTUAL information demand, not
with genome size.

This is exactly how biology organizes interpretation. You don't study
every gene — you study the ones that matter. Helix would be the same.

## Pattern completion — the retrieval insight

If ingest records the firing pattern (lightning strike), retrieval can
use **pattern completion** rather than search:

```python
def query_via_pattern_completion(query_text):
    # 1. Embed query to position in manifold
    q_pos = embed(query_text)

    # 2. Cast rays from query position (same technique as ingest)
    q_pattern = ray_cast(q_pos, genome_bvh)

    # 3. Find genes whose ingest pattern overlaps with query pattern
    #    (the genes that were "firing together" when those neurons fired)
    for gene_id, ingest_pattern in innate_patterns.items():
        overlap = pattern_intersection(q_pattern, ingest_pattern)
        if overlap > threshold:
            yield gene_id, overlap
```

This is Hopfield-network retrieval. No similarity scoring, no ranked
lists — just "the query fires a pattern; return the genes whose ingest
patterns overlap with it."

The shape at ingest IS the shape at retrieval. Same neurons firing,
same gene activates.

## What we'd lose

1. **Human interpretability at the DB level.** `SELECT complement FROM
   genes WHERE gene_id = 'abc'` returns `NULL` for cold genes. You'd
   need to either re-derive the summary on demand or accept
   math-signature-only metadata.

2. **Auditability.** "Why did retrieval return this gene?" is harder to
   explain when the answer is "it has fractal dim 1.3 and its engram
   overlaps 64% with the query's firing pattern." Mitigation: tooling
   to READ fractal signatures (visualization, cluster membership,
   co-activation graphs).

3. **Some tagging quality on specialist domains.** An LLM understands
   that "SPLADE" refers to a sparse retrieval model. Regex + spaCy will
   tag it as `UNKNOWN_ENTITY` or miss it entirely. Domain vocabularies
   (project-specific EntityRuler patterns) can close most of this gap.

## What we'd keep

1. **All 13 retrieval dimensions** (with potentially degraded tag quality
   on #3, #4 — recoverable via better CpuTagger).
2. **Kompress compression** (it's a neural encoder, not generative —
   still deterministic math).
3. **The downstream LLM at expression time** (language where it belongs:
   at the human boundary).
4. **Full auditability via math signatures** (fractal dim, innate
   coactivation graph, embeddings, SPLADE terms — all inspectable).

## Implementation path

**Not tonight.** Current priorities:
1. Gemini re-ingest completes (the codon-table rewrite)
2. Bench the current 13-dim engine against fresh needles
3. If current architecture delivers reliably (SIKE 10/10), this doc
   stays in FUTURE as a design direction
4. If retrieval quality has ceilings that LLM ingest can't break
   through, this becomes the next milestone

**When it's time:**

1. Build `helix_context/lightning.py` — ingest-time ray-casting with
   OptiX fallback to CPU Monte Carlo
2. Extend `schemas.py` Gene with `innate_coactivation`,
   `fractal_signature` fields
3. Improve CpuTagger with project-specific EntityRuler patterns
4. Add lazy annotation hook to `on_gene_access`
5. Build A/B bench: LLM-ingest genome vs math-ingest genome, same
   fresh needles, measure SIKE and KV-harvest scores
6. If math-only within 5-10% of LLM quality, adopt as default
7. If gap is bigger, keep LLM as opt-in per-project via config

## The philosophical punchline

Your mission statement says helix-context is "an attempt to digitally
represent nature's way of encoding and reading data." Nature's way
doesn't use language at the substrate level. Language is an emergent
property of specific information processing systems (brains, LLMs) that
operate ON the substrate.

When you put LLMs in the ingest path, you're saying "let's use a brain
to pre-process every cell before it enters the organism." That's not
how organisms work. Cells ingest nutrients via chemical gradients,
proteins, membrane channels — all physics. Brains get involved only
when the organism needs to REASON about something, usually at the
motor/perceptual boundary.

**Math in the middle, language at the edges** is the biologically
honest architecture. The middle is the substrate (math doing what
physics does). The edges are the interfaces (language doing what
consciousness does).

The lightning strike at ingest is the moment a new memory is laid down.
The same shape firing at retrieval is the moment the memory is recalled.
Between those two events, the memory lives as pure pattern — no words,
no summaries, no interpretation. Just the shape.

## References

- Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.
- Hopfield, J. J. (1982). Neural networks and physical systems with
  emergent collective computational abilities. *PNAS* 79(8).
- Ramirez, S., Tonegawa, S. et al. (2013). Creating a false memory in
  the hippocampus. *Science* 341(6144).
- Howard, M. W., & Kahana, M. J. (2002). A distributed representation
  of temporal context. *Journal of Mathematical Psychology* 46(3).
- Internal: [`docs/MISSION.md`](../MISSION.md)
- Internal: [`helix_context/ray_trace.py`](../../helix_context/ray_trace.py)
  (current Monte Carlo implementation)
- Internal: [`docs/DIMENSIONS.md`](../DIMENSIONS.md) (retrieval dimension
  inventory)

## Related but separate future docs

- `docs/FUTURE/LIGHTNING_STRIKE.md` — ingest-time ray-casting implementation
  (the mechanism)
- `docs/FUTURE/LAZY_ANNOTATION.md` — on-demand LLM interpretation trigger
  (the policy)

This doc (`LANGUAGE_AT_THE_EDGES.md`) is the philosophy that unifies
them.

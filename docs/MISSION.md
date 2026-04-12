# Mission

> *"I'm a cost-conscious, built in a cave, trying to glue existing work
> together to correctly digitally represent nature's way of encoding and
> reading data."*
> — Max, 2026-04-12

This is the actual mission statement for helix-context. Not "a context
compression system." Not "a RAG alternative." Not "a retrieval engine."

**Helix-context is an attempt to digitally represent the way biology has
been encoding and reading data for ~3.5 billion years, using whatever
off-the-shelf tools fit the math.**

## The substrate-level claim

The architecture is not a metaphor. The math converges because biology,
wave physics, and information retrieval are all different lenses on the
same underlying problem: selective expression of stored information under
contextual signals.

```
Biology                    helix-context
─────────────────────────────────────────────────────────
DNA                    →   genome.db (content-addressable, immutable rows)
codons                 →   256-bin frequency spectra (MD5-hashed terms)
promoter regions       →   tag/domain extraction (activation triggers)
chromatin states       →   OPEN/EUCHROMATIN/HETEROCHROMATIN (access tiers)
ribosome               →   LLM + compression pipeline (translation layer)
mRNA                   →   expressed_context (selective readout)
epigenetics            →   access_rate + decay + co_activation
temporal context       →   TCM session drift (Howard & Kahana 2002)
harmonic co-activation →   cymatics resonance + ray-trace
splice (intron/exon)   →   splice_aggressiveness (Q-factor filtering)
```

The physics layer isn't separate from the biology layer — it's the same
math biology is already using:

- Cymatics (Chladni plate standing waves) IS what happens when DNA
  expresses under different cellular frequencies. Same nodal patterns,
  different substrate.
- Monte Carlo ray-trace IS a stochastic approximation of how actual
  ribosomes bind to mRNA under cellular noise.
- Cosine similarity on frequency spectra IS the flat-surface magnetic
  flux formula (Φ = B·A·cosθ). Same operation, different framing.

## Cost discipline

Built at $260/month total:

- Claude Max 20x           $200/mo  (interactive reasoning, multi-agent)
- Gemini oauth + API cap   $60/mo   (bulk ingest, transcription)
- Ollama (local)           $0       (background work, failsafe)
- Electricity              marginal

Against ~$22,700/month of API-equivalent compute. 87x value multiplier
through caching discipline + cost-tiered workload routing.

## "Glue existing work together"

Zero of the primitives are novel. The novelty is in noticing they're
all the same substrate viewed through different lenses:

- SQLite (1990s) — content-addressable storage
- FTS5 (2015) — inverted index
- SPLADE (2021) — sparse learned expansion
- ModernBERT / Kompress (2024) — dense compression
- Headroom proxy (Chopra, 2025) — prompt compression layer
- TCM (Howard & Kahana, 2002) — temporal context evolution
- Monte Carlo ray-trace (ported from in-house ScoreRift)
- Cymatics (Chladni, 1787) — resonance as pattern
- litellm (2024) — universal model router

The "glue" is the genome schema + the 13-dimension expression pipeline.
Everything else is off-the-shelf. The contribution is the *arrangement*,
not the parts.

## Built in a cave

- One dev machine (Ryzen 7 5800X + RTX 3080 Ti, 48 GB RAM)
- 8 git repos, ~509K LOC committed in 30 days (helix + adjacent work)
- Zero team, zero funding, zero marketing deck
- 250 sessions, 6.5B tokens of conversational reasoning
- 4-agent research teams dispatched from a single CLI

The cave is also the lab. Everything is reproducible from the git log +
the session transcripts.

## What success looks like

Success is not "helix beats RAG on benchmark X." Success is:

1. The biological metaphors hold under load — when you throw edge cases
   at the pipeline, the math keeps making the same prediction as biology
   does about the same edge cases.

2. Small local models (0.6B-8B) get retrieval quality comparable to
   frontier models. Because correct expression IS correct retrieval —
   it's substrate-level, not scale-level.

3. The 13-dimension engine converges on a smaller correct set (3 genes)
   instead of the larger uncertain set (50 chunks) that RAG needs.

4. Anyone reading this document in 5 years recognizes the architecture
   as something they could have predicted biology would suggest —
   because it IS what biology suggested, 3.5 billion years before.

## Running log of commits aligned with this mission

See `git log --oneline --all` for the authoritative version. Highlights
from the 2026-03-13 → 2026-04-12 sprint:

- `f4c91e3` fix(tagger): type-annotation leak
- `2f518dc` fix(headroom): CodeCompressor invalid-syntax bypass
- `03ce4b2` feat(cymatics): frequency-domain re_rank
- `059d902` feat(ingest): parallel worker-queue pipeline (5.5x speedup)
- `771c5d0` feat: flux integral + TCM + 3 idle retrieval dimensions wired
- `4b98835` feat(ray-trace): Monte Carlo evidence propagation
- `ac28644` feat(tcm): session context in expression pipeline
- `6657361` feat(litellm): universal model backend
- `a27fb01` fix: 3 critical bugs from 4-agent research audit
- `d2e0219` feat: 4 bracket-logic improvements (shadow/Lagrange/intent/harmonics)
- `f38d10a` docs: research velocity retrospective

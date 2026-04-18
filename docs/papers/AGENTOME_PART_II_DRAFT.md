# Agentome, Part II: Thirteen Lanes and a Live Experiment

*A field report from running the Ribosome Hypothesis in production for a week.*

---

In [the original Agentome piece](https://mbachaud.substack.com/p/agentome), I proposed that an LLM's context window behaves like a biological cell — most of the genome is cold-stored, and a ribosome expresses only the proteins relevant to the current task. It was a hypothesis. A metaphor with some code behind it.

This is the follow-up. The hypothesis is now running on my machine, ingesting every repo I own, serving compressed context to local and frontier models, and producing numbers I can point at. This piece is about what happened when the metaphor met the measurement.

**The short version:** the biological architecture self-organized into thirteen retrieval lanes. All of them are wired. An A/B experiment is running right now that may cut the count back to eleven. Cymatics works. E8 does not. And the bottleneck is no longer where I thought it was.

---

## The genome is real now

When I wrote the original piece, the genome was a proof-of-concept database with a few hundred genes. Today the genome on my development machine is **17,623 genes, 670 MB**, distributed across three chromatin tiers:

| Tier | Count | Role |
|---|---|---|
| OPEN (hot) | 12,401 | Default retrieval target |
| EUCHROMATIN (warm) | 1,895 | Included with hot queries |
| HETEROCHROMATIN (cold) | 3,327 | Opt-in, queryable via ΣĒMA cosine fallthrough |

That growth happened in hours, not weeks. A parallel ingest pipeline landed this weekend — worker queues, deduplication, density-gated admission — and took throughput from ~1,000 genes per hour to ~5,500. Combined with a tagger fix that eliminated type-annotation leaks from 3,755 existing genes, the corpus went from "working demo" to "actually comprehensive" almost overnight.

I have a local LLM ingestion running right now on top of this. A Claude-backed ribosome that re-packs genes with high-quality complement summaries, codon weights, and key-value extraction. The CPU path is fast enough for routine ingest; the LLM path is for quality-critical content where you want the best possible expression.

This matters because **the biology metaphor was supposed to be about cold storage and selective expression.** Until you have enough genes to actually demote some to cold storage and pull them back on demand, you don't know if the metaphor is load-bearing or decorative.

It's load-bearing. I'll show you the numbers.

---

## The thirteen lanes that emerged

I didn't design this. It happened. Every time I tried to add a feature to improve retrieval, it ended up being its own scoring dimension with its own rules. By the time I stopped to count, there were thirteen:

1. **Tier 1** — exact promoter tag match
2. **Tier 2** — prefix promoter tag match
3. **Tier 3** — FTS5 full-text content search
4. **Tier 3.5** — SPLADE sparse term retrieval
5. **Tier 4** — ΣĒMA 20-dimensional semantic cosine + re-ranking
6. **Lexical anchoring** — IDF-weighted rare-term boost
7. **Authority boosts** — "about X" vs "mentions X" scoring
8. **Tier 5** — harmonic co-activation boost via cymatics-derived links
9. **Party attribution bonus** — small boost for genes authored by the querying party
10. **Access-rate tiebreaker** — small boost scaled by recent access rate
11. **Cymatics flux resonance** — frequency-domain scoring at expression time
12. **Cold-tier ΣĒMA fallthrough** — opt-in query over heterochromatin
13. **TCM session context** — temporal drift bonus from Howard & Kahana 2002

Each lane answers a different question about a candidate gene, and the scoring pipeline fuses them. Every one of the thirteen was added in response to a concrete problem — a query that should have hit but didn't, a result that was ranked too low, a gene that was demoted when it shouldn't have been. None of them were drawn on a whiteboard first.

The ray-trace Monte Carlo evidence-propagation mechanism shipped this weekend as well, providing a deeper co-activation scoring path layered on top of the harmonic-link boost. Whether it earns its own lane number or just augments #8 is a question I haven't settled yet.

The fact that this self-organized into thirteen independent scoring channels is, to me, the most interesting outcome. Nobody drew this architecture on a whiteboard. It emerged from "I need this to work better" decisions, one commit at a time, and it looks nothing like a conventional retrieval system.

---

## Cymatics: the biology metaphor holding up under load

The paper argued that biological compression mechanisms should inform engineering ones. The specific claim I was nervous about was that the metaphor would stay useful *after* the system was actually working. Metaphors are cheap when they're decorative. The real test is whether they produce engineering decisions you wouldn't have made otherwise.

Cymatics is that test.

Here's the setup. The retrieval pipeline's Step 3 is re-ranking — given 50 candidate genes from Step 2, which 12 should we actually express? The naive approach is to ask the LLM: "rank these by relevance to the query." This works. It takes about two seconds per query.

The biological framing asks a different question. A gene in biology isn't "ranked" by an executive — it's *excited* by the chemical environment around it. Transcription factors bind to promoter regions; the gene responds or doesn't based on resonance with the signal. That's not a ranking operation, it's a physics operation.

So we implemented it as physics. Each gene gets a 256-bin frequency spectrum derived from its content terms. Each query gets a spectrum too. Relevance is the cosine similarity between the two spectra — an interference pattern. Synonym expansion becomes harmonic overtones. The splice aggressiveness knob becomes a Q-factor controlling peak width.

The LLM re-rank was two seconds. Cymatics resonance ranking is **five milliseconds.** Four hundred times faster. And the retrieval quality is at least as good — we haven't observed a regression yet.

Would I have built this without the biology metaphor? Probably not. Without the framing, "replace the re-rank LLM call with frequency-domain cosine on synthesized spectra" sounds absurd. With the framing, it's the obvious thing. The metaphor earned its keep.

---

## E8: the honest failure, tested three times

This is the part I promised myself I'd include, because promoting only the wins is how you become a charlatan.

A few days ago, during a conversation about scaling the retrieval space, the question came up: could helix use the E8 lattice as its vector quantization codebook? E8 is the densest known sphere packing in eight dimensions — 240 roots, each touching 56 nearest neighbors at exactly 60 degrees, an object of genuine mathematical beauty. If retrieval lives in a spherical geometry, E8 should be the optimal discretization.

I wanted this to work. The symmetry was elegant. It would have been a beautiful result.

### Test 1 (April 11): E8 as primary codebook

Projected 780 cold-tier embeddings from 20-D to 8-D via PCA. Generated the 240 E8 roots. Quantized each gene to its nearest root. Compared against a learned K-means codebook with the same 240 centers.

| Metric | E8 | K-means(240) |
|---|---|---|
| Mean cos(gene, nearest codeword) | 0.856 | **0.962** |
| Codebook utilization | 154 of 240 (64%) | ~240 of 240 |
| Most-used codeword captures | 17.7% of genes | ~1% each |
| True top-10 neighbors within 60° cone | 36.9% | **92.3%** |

E8 loses decisively. The learned K-means codebook beats the mathematically optimal E8 lattice because real content is not uniformly distributed on the sphere — it's heavily clustered. E8's optimal-packing proof holds for uniform distributions. My genome is not a uniform distribution.

I wrote that result up and thought it was the end of the story.

### Test 2 (April 12): E8 on the larger, cleaner genome

Then the genome grew from 8K to 17.6K genes with a fixed tagger and a cleaner ingest pipeline. A natural question: did the distribution change enough to make E8 competitive? Re-ran the experiment with a 1,500-gene stratified sample across all chromatin tiers.

| Metric | Apr 11 | Apr 12 |
|---|---|---|
| E8 mean quant cosine | 0.856 | 0.855 |
| K-means mean quant cosine | 0.962 | 0.962 |
| E8 codebook utilization | 154/240 (64%) | **185/240 (77%)** |
| E8 most-used share (pathology) | 17.7% | **6.6%** |
| E8 neighbors within 60° cone | 36.9% | 42.2% |
| K-means neighbors within 60° cone | 92.3% | 94.4% |

**The pathology resolved.** The one-dominant-cluster problem went from 17.7% to 6.6% as the corpus grew and spread more evenly across the E8 lattice. Utilization climbed from 64% to 77%. E8 itself got a little better.

**But the gap didn't close.** 52 percentage points separated E8 from K-means on the retrieval-quality metric at April 11; 52 percentage points separated them at April 12. The learned codebook maintained its advantage as the corpus matured.

### Test 3 (April 12, same day): E8 as a safety floor

I still thought there might be a niche where E8 earns its keep. A reasonable-sounding hypothesis: even if K-means wins at steady state, surely at cold-start (N=50, N=100 genes) where K-means can barely train, E8's deterministic codebook provides a quality floor? The framing was elegant — E8 as the uninformed prior, K-means as the data-informed posterior.

I ran a crossover experiment: E8 vs K-means at N ∈ {50, 100, 200, 500, 1000, 1500, 5000}, with K-means forced to generalize (fixed K=50) rather than memorize.

| N | E8 neighbor cone | K-means cone | Gap |
|---|---|---|---|
| 50 | 8.6% | N/A (K-means can't run) | — |
| 100 | 20.1% | 73.1% | +53.0pp K-means |
| 200 | 21.7% | 72.8% | +51.1pp K-means |
| 500 | 27.5% | 81.1% | +53.6pp K-means |
| 1000 | 34.1% | 83.0% | +48.9pp K-means |
| 1500 | 31.2% | 80.9% | +49.7pp K-means |
| 5000 | 38.0% | 83.9% | +45.9pp K-means |

At N=50, where only E8 can run, E8's 8.6% recovery is actually **worse than random chance** (~23% would be expected from E8's neighbor-cone density). The PCA basis is too noisy at that sample size to align with the rigid E8 geometry. Above N=100, K-means crushes E8 by 46–54 percentage points uniformly.

**No crossover exists.** The "E8 as quality floor" hypothesis is as dead as the "E8 as primary codebook" hypothesis. There is no operational N where E8 provides a retrieval-quality win over K-means. The elegant quantum-state-before-observation framing is refuted by the measurement.

### What survives of the case for E8

Three non-quality arguments hold up:

- **Determinism** — E8's codebook is the same every time, with no seed dependence. K-means varies by initialization.
- **Zero-cost construction** — E8 requires no training. Useful for instant startup.
- **Auditability** — "Why did gene X land in codeword 47?" has a provable mathematical answer for E8, only an empirical one for K-means.

These are reliability and reproducibility arguments. None of them translate to retrieval quality. A system that needs an audit trail might still use E8 at its fringes; a system that retrieves context should use a learned codebook.

### The broader lesson

I ran this experiment three times across two days because I wanted the result to be different each time. The first test said E8 fails. The second test — with more data, a cleaner corpus, and a natural case for "maybe E8 works at scale now" — said E8 still fails. The third test — with a specifically-constructed "E8 as safety floor" hypothesis designed to find a niche where it could win — said E8 still fails.

Three tests, three falsifications. That's not a design flaw I'm working around; that's the measurement telling me the framing was wrong.

If you see someone claim that E8 or the Leech lattice or any structural codebook will revolutionize LLM retrieval, ask for the neighbor-recovery numbers on real clustered data. The mathematics are beautiful. The learned codebook still wins.

---

## The stack: helix + Headroom, zero LLM calls

Here's the architecture that actually ships:

```
skills.md file
      |
  INGEST (CPU)
      |
  Chunk + Tag + Embed + Gate --> genome.db
      |
      |  (later, on query)
      v
  /context POST (CPU)
      |
  Step 1  Signal extraction      (~20 us)
  Step 2  FTS5 + SPLADE + SEMA   (~14 ms)
  Step 3  Cymatics resonance     (~5 ms)
  Step 4  Headroom compress      (~265 ms * N genes)
  Step 5  Assembly               (<1 ms)
      |
      v
  Downstream model
  (local qwen3:4b or frontier API)
```

Everything from ingest to expression runs on CPU. No LLM calls in the retrieval path. The first LLM in the chain is the downstream model that consumes the compressed context.

Headroom is Tejas Chopra's compression library — `KompressCompressor` (ModernBERT extractive), `LogCompressor`, `DiffCompressor`. We stack it on top of helix as the expression-time compression layer. When helix returns 12 candidate genes, Headroom compresses each to a target of ~1,000 characters before they're assembled into the final context window.

This week I disabled Headroom's `CodeAwareCompressor` path after the 0.5.23 changelog documented a ~40% invalid syntax rate on real code files. All content now routes to `KompressCompressor` (ModernBERT). The measured impact on our N=20 benchmark:

- Answer accuracy: 16% → 25% (**+9 percentage points**)
- Retrieval rate: 20% → 25% (+5pp)
- Extraction misses per benchmark: 3-4 → 1

Nine points of accuracy from a one-line routing change. That's what I mean by "the bottleneck is not where I thought it was."

---

## The bottleneck that remains

Compression is slow. Not because it's LLM-based — it isn't — but because ModernBERT inference runs sequentially, one gene at a time, at about 265 ms per gene on CPU.

For a query returning 12 genes, that's **3.5 seconds.** Retrieval and scoring together take 19 ms. Compression takes 183 times longer than everything else combined.

I filed [an upstream issue](https://github.com/chopratejas/headroom/issues/151) requesting a batched compression API. ModernBERT natively supports batched inference — padding, attention masks, one forward pass for N sequences — and the estimated speedup is roughly 8× for a 3-gene batch and 17× for a 12-gene batch. That takes compression from "noticeable" to "imperceptible" and makes the whole pipeline interactive.

Until that lands, the user-visible latency story for helix + Headroom is dominated by Step 4. Retrieval is free; compression is expensive. This is almost exactly backwards from where most conventional RAG systems live, where retrieval is the slow part.

---

## The live experiment: math-only ingest

This is where the follow-up becomes a real field report instead of a retrospective — because as I'm writing, an A/B test is running on my machine that will determine whether two of those thirteen lanes survive in their current form.

The setup. The current ingest pipeline uses an LLM (Gemini Flash in the active run) to read every incoming gene and extract rich metadata: a dense complement summary, weighted codon labels, domain and entity tags in the promoter, a set of key-value facts. This is expensive — about 30 genes per second of throughput, roughly $0.30 per ten thousand genes, and the resulting tags carry a lot of the signal that downstream retrieval depends on.

The alternative (call it "math-only ingest") uses zero LLM calls at the ingest path. A deterministic CpuTagger based on spaCy and regex produces the promoter tags. SPLADE encodes sparse terms. ΣĒMA produces the 20-D semantic embedding. Monte Carlo ray-casting over co-activation structure records innate coupling between genes without asking any model what those genes are "about." Throughput jumps from ~30 genes per second to ~200+, and the cost per ten thousand genes drops to zero.

The hypothesis, locked in before results are in: math-only ingest scores within 10% of LLM ingest on retrieval quality. If it does, the LLM calls at ingest time weren't load-bearing — they were overhead. Kept around because they felt like they were doing useful work.

Here are the pre-results predictions, recorded in the project repository before the Gemini Flash run completes:

| Metric | LLM ingest (A) | Math-only (B) | Gap |
|---|---|---|---|
| SIKE N=10 retrieval | 10/10 | 9/10 | -1 |
| SIKE N=10 answer | 9/10 | 7/10 | -2 |
| KV-harvest N=50 retrieval | 40% | 35% | -5pp |
| KV-harvest N=50 answer | 35% | 30% | -5pp |
| Ingest throughput | ~30 genes/sec | ~200+ genes/sec | 6-12× |
| Ingest cost per 10K genes | ~$0.30 | $0 | -100% |

**If the measured B/A ratio lands above 0.95, math-only wins outright.** The LLM ingest path becomes optional, used only for "hot" genes that have earned interpretation through repeated access — a lazy annotation principle that mirrors how biological organisms interpret chemical signals only after the signal has already been activated by circumstance.

**If the ratio lands between 0.85 and 0.95, math-only is viable but not perfect.** It becomes the default for cost-sensitive workloads; LLM ingest stays available as an opt-in flag for projects where interpretability matters more than speed.

**If the ratio lands below 0.70, LLM ingest is load-bearing.** The current architecture stays, and the math-only work becomes additive rather than substitutive.

Two of the thirteen retrieval dimensions are specifically at risk in the math-only scenario — the ones most dependent on rich LLM-generated tag quality. Not deleted, but degraded, running on the CpuTagger's rougher output until the tagger catches up through better entity rules and project-specific patterns. Whether those two get dropped entirely or kept in degraded form is a decision I'll make *after* the numbers are in, not before. Thirteen peaked. Eleven or thirteen, depending on the experiment.

This is the part of the project I'm most uncertain about and most excited about. The biology framing has been productive for architecture (cold storage, selective expression, chromatin tiers, cymatics resonance). It also makes a specific claim about where intelligence belongs in the stack: at the edges, not in the middle. Language is how brains interface with the substrate; the substrate itself runs on physics. If math-only ingest holds up, the biology metaphor stops being a design intuition and starts being a falsifiable engineering claim about how context should be built.

We'll see what Gemini Flash finishes with in the next few hours.

---

## What running the system revealed that the paper didn't predict

Three things happened in the last week that I didn't anticipate when writing the original piece.

**First: the dimensional lanes self-organized.** I did not sit down and design "there will be six retrieval dimensions." I added features to solve concrete problems — cold-tier fallthrough for demoted content, access-rate tracking for hot-path preference, cymatics scoring for latency — and after the fact they turned out to be six clean scoring dimensions plus three in-progress ones. The biology metaphor may have been doing more work than I realized, quietly suggesting the right abstraction boundaries.

**Second: the biology metaphor kept generating engineering decisions.** Cymatics is the clearest example, but it's not the only one. Chromatin tier as a non-destructive accessibility control (not a deletion mechanism). Splice as a perceptual-coding operation (not a rank-and-truncate). Promoter tags as binding sites (not just keywords). Every time I asked "how does biology solve this?" the answer pointed at a cleaner architecture than the default software-engineering solution.

**Third: measurement showed us things we didn't feel.** The tagger fix that produced +9pp accuracy was not something I noticed in the benchmark numbers first — I noticed it in the corpus. Looking at `key_values` extractions, half of them were type annotations (`model=str`, `path=Optional`) instead of actual values. The benchmark had been under-reporting answer accuracy by ~15% because of phantom KV matches against docstrings and function calls. Once we fixed the extractor, re-ingested the corpus through the fixed path, and re-ran the benchmark, the numbers caught up to what the system was actually doing. There's a lesson here about trusting instrumented measurements over subjective performance impressions — when the instrument is wrong, your intuition can be more accurate than the numbers, and vice versa.

---

## Where it goes from here

Three near-term tracks:

1. **The A/B test resolves.** Math-only ingest versus LLM ingest, both benched on the same fresh needles, against a locked-in prediction table. The result determines whether we're running thirteen dimensions in full quality, thirteen with two degraded, or a different architecture entirely.
2. **Batched compression** lands when Headroom ships `compress_batch()`. That removes the 3.5-second Step 4 ceiling and makes the full pipeline sub-second.
3. **TCM forward-recall asymmetry.** The temporal context model is wired, but the signature property — that queries about earlier genes preferentially surface later genes from the same session — still needs to be verified on the benchmark. If the asymmetry shows up, the trajectory layer is real. If it doesn't, the implementation is wrong or the mechanism doesn't transfer from cognitive psychology.

Longer-term, the open question is whether this architecture holds at 100k genes or 1M genes. The compression pipeline scales linearly; the retrieval pipeline uses SQLite indexes and should scale sub-linearly; the chromatin tier management is an open empirical question. I've measured everything I can measure up to 17k genes. Beyond that, we'll find out by running it.

---

## Closing thought

The original Agentome piece was a hypothesis dressed in biological language. I wasn't sure, when I published it, whether the metaphor would generate engineering or just rhetoric. A week of running the system has mostly answered that question. Cymatics is real engineering. Chromatin tiers are real engineering. The thirteen-lane retrieval architecture that self-organized out of feature work is real architecture, not decorative framing.

E8 was the test case going the other direction — a mathematical structure I wanted to find useful, that the measurement rejected. Keeping that result in the public record is how the project stays honest.

The math-only ingest A/B is the next honest test. The biology framing makes a specific claim — language belongs at the edges where brains and LLMs live, not in the middle where the substrate lives. If the A/B results say LLM ingest was load-bearing all along, the framing takes a hit and we adjust. If they say math-only ingest is within 10% of LLM ingest, the framing earns more evidence and we adopt it as default.

Either way, the number we're looking at isn't 13. It's the B/A ratio. And we'll have it in a few hours.

---

*helix-context is at [github.com/SwiftWing21/helix-context](https://github.com/SwiftWing21/helix-context). The lane graph in this piece lives at [docs/DIMENSIONS.md](https://github.com/SwiftWing21/helix-context/blob/master/docs/DIMENSIONS.md). The skills-bundle architecture at [docs/SKILLS_BUNDLE.md](https://github.com/SwiftWing21/helix-context/blob/master/docs/SKILLS_BUNDLE.md). The knowledge graph at [docs/KNOWLEDGE_GRAPH.md](https://github.com/SwiftWing21/helix-context/blob/master/docs/KNOWLEDGE_GRAPH.md).*

*The original Agentome paper is [here](https://mbachaud.substack.com/p/agentome).*

*Headroom is by [Tejas Chopra](https://github.com/chopratejas/headroom). This write-up would not have a compression pipeline without it.*

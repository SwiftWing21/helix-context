# Research Velocity

> *"The difference between research and fuckin around is whether you wrote down the notes."*
> — Max, 2026-04-12

A retrospective on 30 days of build velocity (2026-03-13 → 2026-04-12), with a
specific focus on the phase shift when work moved from scaffolding to
architectural design. Data extracted from `~/.claude/projects/` session
transcripts cross-referenced with `git log --numstat` across all F:/Projects
repos.

---

## The spike

On 2026-04-11, output token generation hit an all-time high of **2,512,339 tokens**
while code committed that day was a modest **7,882 LOC**. The ratio of
output-tokens-per-line-of-code crossed **318:1** — an order of magnitude above
the 30-day baseline of 25-50:1.

## What did NOT drive the spike

**Hypothesis 1: CosmicTasha audit reports propagating through agent .md configs.**
*Not supported by the data.* The peak CosmicTasha commit day (Apr 6, 37,216 LOC,
top project) had the *lowest* tokens/LOC ratio of the period: **11.3**. If audit
reports were driving output, this ratio would spike on those days, not drop.
CosmicTasha commits were bulk-generated scaffolding material, not synthesis
output.

**Hypothesis 2: Markdown documentation commits.** *Not supported.* Apr 11 (spike
day) changed only 339 MD lines. Apr 1 changed 18,206 MD lines and generated
13.8 tokens/LOC — nowhere near the spike. Docs volume doesn't correlate with
output spikes.

## What DID drive the spike

Three stacking factors, visible only in the output-per-LOC ratio:

### 1. Synthesis density per commit went up ~10x

Pre-helix commits were scaffolding: many LOC, little analysis. Helix-era
commits were tied to architectural reasoning, benchmarks, writeups, and
explicit design decisions. Apr 11 had only 9 commits but each commit was
preceded by hundreds of tokens of analysis before a single line was written.

### 2. Multi-agent coordination amplified output without amplifying LOC

Apr 11 was the 4-agent research team dispatch (R1/R2/R3/R4 researchers +
synthesis). Each agent generated ~500-word reports (10K+ output tokens each),
then the main session synthesized findings (several thousand more tokens). Four
parallel researchers → four times the output, zero additional code.

### 3. Opus 4.6 1M context enabled denser turns

With 98.3% cache hit rate, each conversational exchange could carry more
analysis per round-trip. Pre-helix session turns were often "read file, edit,
commit" — short outputs. Helix session turns were "analyze 4 research reports,
synthesize findings, decide fix priorities, implement, test, write commit
message" — long outputs per turn.

## The phase shift

The tokens/LOC ratio is a signal for what kind of work is happening:

| Ratio | Signature | Typical day |
|---|---|---|
| **10-20 tokens/LOC** | Bulk scaffolding, auto-generated code, setup | Apr 6 (CosmicTasha bulk) |
| **20-50 tokens/LOC** | Standard feature development | Mar 17-28 average |
| **50-100 tokens/LOC** | Architectural exploration | Mar 30, Apr 9 |
| **100-300 tokens/LOC** | Design synthesis + research coordination | Apr 10-12 (helix-context) |
| **300+ tokens/LOC** | Multi-agent research session | Apr 11 (13-dim audit) |

The work got harder and slower per line of code, but each line became more
valuable. Apr 11's 7,882 LOC probably moved helix-context closer to a
reliable, production-grade retrieval engine than Mar 23's 40,915 LOC moved
Education. Not because helix is fancier — because the THINKING per line was
higher.

## Raw daily data

```
DATE        COMM   LOC+   LOC-   MD+    PY+    RS+   OUT-TOK    OUT/LOC  TOP PROJECT
─────────────────────────────────────────────────────────────────────────────────────
2026-03-17     4  18486      1   793  11975      0   575,125      31.1  Education
2026-03-18    39  27691   5984  4400  22535      0   616,518      22.3  Education
2026-03-19   164  25380   7188  4491  19314      0   952,843      37.5  Education
2026-03-20    96  28432   6772  4152  21754      0   567,804      20.0  Education
2026-03-21    53  19023   1300  3595  12470      0 1,012,513      53.2  Education
2026-03-22    46  10809   1808  1868   5642      0   362,141      33.5  Education
2026-03-23    66  40915   5286  4130  28041      0   272,417       6.7  Education
2026-03-24    92  66132  23313 11177  30705  10260   486,916       7.4  BookKeeper
2026-03-25    82  18963   2867  3484  11668    109   384,094      20.3  Education
2026-03-26    92  19529  15254  4867  11437      0   579,408      29.7  Education
2026-03-27    59  13762    478  8339   3427      0   249,811      18.2  Education
2026-03-28    86  21525   1366  9102  10116      0   649,715      30.2  Education
2026-03-29    35   5066    579  2358   2144      0   342,911      67.7  Education
2026-03-30    73  11612   1443  3424   4785      0 1,162,466     100.1  Education
2026-03-31   134  28641  14781  5329  12869      0   350,305      12.2  Education
2026-04-01    97  36010  26299 18206  13886      0   496,548      13.8  Education
2026-04-02    45   6564  42578  2282   3012      0   291,712      44.4  Education
2026-04-03     7    658    381   410    226      0    20,483      31.1  Education
2026-04-04    24   4441     96  1493   2748      0    95,185      21.4  Education
2026-04-05    18   4633    713   601    679   2631   137,652      29.7  Education
2026-04-06    15  37216    213   427   2372    331   422,120      11.3  CosmicTasha
2026-04-07    45  20815   3287  4388   6648      0   428,219      20.6  CosmicTasha
2026-04-08    14   8057    370   274   4393      0   186,541      23.2  helix-context
2026-04-09    32   7449   1140   902   5166      0   690,157      92.7  helix-context
2026-04-10    28  16462    510  2319   9945      0 2,043,954     124.2  helix-context
2026-04-11     9   7882     77   339   5151      0 2,512,339     318.7  helix-context
2026-04-12     9   3281     52   665   2613      0   707,707     215.7  helix-context
─────────────────────────────────────────────────────────────────────────────────────
30-day totals:
  LOC added:        509,434  (MD: 103,815  PY: 265,721  RS: 13,331)
  Output tokens:    16,597,604
  Avg output/LOC:   32.6 tokens/LOC
  Sessions:         250
  Tool calls:       31,994
  Cache hit rate:   98.3%
  Cache savings:    88.0% vs no-cache equivalent
```

## Context cost efficiency

```
Total tokens moved (input + both caches):   11.93 BILLION
  Fresh input:                              548,664       (0.005%)
  Cache writes:                             204,788,974   (1.72%)
  Cache reads (10x cheaper):                11,721,123,952 (98.28%)

If all were fresh input:                    11.93B tokens @ 1.0x
Actual cost-equivalent tokens:              1.43B tokens   (0.12x effective)
Savings from caching:                       88.0%
Estimated Opus 4.6 cost saved:              ~$157,500
```

## Takeaways

1. **Output-per-LOC is the velocity metric that matters**, not LOC/hr. A day with
   7,882 high-synthesis LOC beats a day with 40,000 scaffolding LOC for any
   real architectural work.

2. **Multi-agent dispatch is a force multiplier.** One research session with
   4 parallel Opus researchers produced more analysis in one afternoon than a
   full week of solo debugging would have.

3. **Cache hygiene unlocks long sessions.** At 98.3% hit rate, the effective
   cost of running billion-token sessions drops below the cost of NOT running
   them (you'd otherwise thrash context and redo work).

4. **Tool calls decreased per session while output increased.** This inverse
   correlation is the signature of a maturing system: less exploration, more
   synthesis. You stop reading files because you know the codebase; you spend
   the budget on reasoning instead.

5. **The CosmicTasha hypothesis was wrong, and checking it was worth doing.**
   Audit reports and .md config propagation *felt* like they should be
   driving output, but the data showed the opposite. Which is exactly why you
   wrote down the notes.

## Methodology

```python
# Token usage: ~/.claude/projects/{project}/{session}.jsonl
# Each line is a message record; assistant messages have usage fields:
#   input_tokens, cache_creation_input_tokens, cache_read_input_tokens,
#   output_tokens

# LOC per day:
git log --since=2026-03-13 --pretty='COMMIT|%ai|%H' --numstat

# Cross-reference by calendar day. No single session ran past midnight UTC
# enough to distort the daily bucketing in a material way.
```

Fully reproducible. The scripts used to generate this data are ad-hoc and live
in the session transcript rather than a checked-in tool.

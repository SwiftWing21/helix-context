# Waude's diagnostic — 2026-04-17

**Run at commit `9e6b718` (post baseline-lock, pre Task 3 extraction).**

## Setup

Waude proposed: for each `delivery=hit AND correct=miss` query in the
2026-04-16 needle baseline, check whether the expected answer string
is actually in the delivered /context output and in the gene source.
Three outcomes were enumerated: A (compression ate the answer),
B (consumer failed despite presence), C (overcounted delivery).

Source data: `benchmarks/needle_baseline_2026-04-16_freshgenome.json`
(9/10 delivery, 5/10 correct, genome=`genomes/main/genome.db`, 7,789 genes).

Bug-zone (delivery=hit, correct=miss):
- `biged_skills_count`
- `helix_pipeline_steps`
- `biged_rust_binary_size`
- `helix_ribosome_budget`

## Method

1. Re-ran `/context` against live helix (same genome) for each query.
2. Grepped the delivered content for `accept` substrings.
3. Grepped the genome directly for the literal answer-bearing phrase.
4. Cross-checked whether the answer-bearing gene was in the /context
   output.

## Results

| needle | accept in /context? | context of match | true-answer gene in genome? | in delivered set? |
|---|---|---|---|---|
| biged_skills_count | `129` yes (1x) | `"1296c"` — compression-chars metadata | gene `b8f20ab9` (Education/CLAUDE.md "Skills: 129") | **no** |
| helix_pipeline_steps | `6` yes (48x) | all in `566c`, `26.7`, `220c`, etc. — no "6-step" sentence | genes `003ebb96`, `60d6f020` ("6-step pipeline per turn") | **no** |
| biged_rust_binary_size | `11` yes (17x) | all in `localhost:11434` URLs | gene `4a6cbd63` (biged-rs/README.md "11 MB") | **no** |
| helix_ribosome_budget | `3k` yes (2x) | `"build 3k ribosome prompt"` — **real hit** | gene `b30f20a0` (helix.toml "ribosome_tokens = 3000") | **yes** (via 3k) |

## Outcome distribution

- **A** (compression ate answer): 0/4
- **B** (answer delivered, consumer failed): 1/4 — `helix_ribosome_budget`
- **C** (answer not in source at all): 0/4
- **D** (wrong gene retrieved; answer-bearing gene exists in genome but
  not in top-K delivered set): 3/4 — `biged_skills_count`,
  `helix_pipeline_steps`, `biged_rust_binary_size`

Waude did not enumerate D. It dominates the failure set.

## Implications

1. **Compression is not the primary bug.** Zero cases of the answer
   being in the source gene but eaten by compression. Step 2
   (query-aware compression) would not lift these numbers much.

2. **The 9/10 delivery rate is inflated.** Substring-match accept
   criteria on short numerics (`6`, `11`, `129`) produce false positives
   from metadata (compression counts) and unrelated content (URL port
   numbers). True delivery is closer to **2/4 bug-zone + 5 originally-correct
   = 7/10** at best — and for 3 of the bug-zone cases the "hit" was
   spurious.

3. **Batch N and uncompressed-top-K both miss the target.** Both
   assume the right gene is in the top-K. In 3/4 bug-zone cases it
   isn't. More genes at same rank won't help if rank itself is wrong.

4. **The fix is retrieval rank, not compression or context budget.**
   Ranking needs to surface gene `b8f20ab9` for "biged skills count",
   `003ebb96` for "6-step pipeline", `4a6cbd63` for "11 MB binary".
   They exist and have natural lexical overlap with the queries —
   something is ranking them below 12 unrelated chunks.

## Cleaner next experiment

Rather than batch-N or uncompressed-top-K, the experiment that actually
probes D is:

> For each bug-zone query, force-inject the manually-identified
> answer-bearing gene into position 1 of the /context output (via a
> new `force_gene_ids` parameter or via a separate script that
> prepends the gene). Re-run the model. If correctness jumps from 0/3
> to 3/3 on the D-category queries, the fix is a ranker improvement.

That's a 30-minute experiment with a clean outcome. Much cleaner
than scaling either N or compression budget.

## Also worth noting

The bench's `found_in_context` substring-match is a busted metric
for short-numeric accepts. Recommend tightening to require the
accept string to appear in a delivered-content window of ~60 chars
that also contains a query term, not globally. That alone would
re-rate the 2026-04-16 baseline from 9/10 → ~6-7/10 delivery.

## Follow-up from Waude council (appended 2026-04-17)

Waude reviewed the finding and flagged that my "cleaner experiment"
take above is partially wrong. Revised read:

1. **Batch-N is not a misallocation.** For D-category failures, raising
   top-K *would* pull the gold gene into the delivered set (currently
   invisible to the consumer). Batch-N trades precision for recall and
   directly addresses D.

2. **Consumer correctness is actually high.** Given real delivery
   ≈ 6/10 (not 9/10), the model got 5/6 ≈ 83% of delivered queries right.
   Opus is fine. The "consumer is the new bottleneck" framing from
   earlier sessions was wrong on this bench.

3. **Query-aware compression (step 2) doesn't touch D at all.**
   Compression only runs on retrieved genes. If the gold gene isn't in
   top-K, compression can't help. Re-sequence: query extraction
   hardening (step 1) is the single intervention that addresses
   D-category.

4. **The bench itself has a bug.** `found_in_context = substring in
   payload` counts metadata, URL ports, and hash prefixes as hits. Every
   delivery number from this bench is suspect until the check becomes
   "gold gene (by source path or gene_id) in top-K."

### Revised sequencing (from Waude)

- **Step 0**: Fix needle bench scoring — gold-gene-in-top-K, not
  payload substring. Re-run baseline. Expect the real number to be
  ~60-70%, not 90%.
- **Step 0.5**: Re-baseline SNOW on current 7.8k-gene genome as locked
  reference (SNOW already uses gene-ID scoring, so it's honest —
  27.7% oracle miss from today is the real number).
- **Step 1**: Query extraction hardening. Only now is the instrument
  honest enough to measure the lift.
- **Step 2**: Decide based on data. Compression work only lands if
  delivery still has gap after step 1.

### Pattern — query-side signal is too weak to discriminate

Same failure mode appearing across multiple observations:
- Needle-bench substring artifact (this diagnostic)
- SPARSE context_health warnings (Codex reports)
- 18k-vs-7.8k-gene scoping gap
- ROADMAP.md outranking helix_status.py on status queries

In all of these, coincidental token overlap beats semantic relevance
because the query-side signal is too weak to discriminate. Query
extraction hardening addresses the root cause across all of them —
rare property in a fix.

## Artifacts

- `/tmp/waude_diagnostic.json` — raw /context outputs per bug-zone query
  (local only, gitignored by default)
- this file

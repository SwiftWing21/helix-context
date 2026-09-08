# Retrieval stage provenance

Use the ERB ablation ladder's existing `--per-query` flag to attach
`stage_provenance` to each success or failure record. The default aggregate
receipt shape is unchanged. The pool-depth probe has a separate opt-in flag:

```powershell
python benchmarks/dogfood/erb/probe_pool_depth.py --stage-provenance --cache NEW_CACHE_DIRECTORY
```

These flags measure each timed `build_context` request. Warmup is outside the
capture. Each lexical retrieval call has its own entry in `retrievals`; do not
combine counts across calls as though they were one deduplicated pool. Capture
keeps candidate counts and watched gold IDs, not full candidate lists or text.

Capture runs inside the measured request and scans candidates to count them.
Instrumented wall times include that overhead, which can vary with pool size.
Use matched uninstrumented runs for performance comparisons.

Run requests serially per manager and store. Request-local capture does not
repair the existing shared `last_query_scores` publication used by the pipeline;
concurrent calls on the same manager cannot establish reliable post-blend
attribution or parallel latency evidence.

| Boundary | Meaning |
|---|---|
| `fts_raw` | Bounded SQL rows after an optional tier-0 BM25 prefilter, before lifecycle and party filters; see `fetch_depth` and `prefilter_applied` |
| `fts_eligible` | FTS membership after those filters |
| `pre_shortlist` | Candidate membership immediately before the shortlist |
| `post_shortlist` | Membership after the shortlist decision; inspect `filter_status` |
| `final_scoring` | IDs in the eligible scoring map before cross-encoder reranking and return expansion |
| `retrieval_returned` | IDs actually returned by that retrieval call |
| `post_blend_scores` | IDs in the manager's local score map after the configured blend branch, before budget, splice, and delivery; this need not equal the shared published score map |
| `post_blend_candidates` | Candidate IDs after the configured blend branch, before budget, splice, and delivery |

The version 1 report carries an overall `status`, `error`, per-call retrieval
statuses, request-level `stages`, and `unsupported` path reasons. A captured
stage has an integer `count` and a `gold_ids` list. A captured empty stage has
`count: 0` and `gold_ids: []`. Failed or unexecuted stages use `null` for both
fields. An overall `complete` report does not mean every optional stage ran;
check each stage's status. ANN union/gating, shard fanout/merge, and parallel
subqueries are unsupported and remain explicitly unmeasured. Any lexical
sub-call evidence they retain describes only that call, not the wider path.

Use membership at the boundary that answers the question. A gold document can
reach `post_shortlist` and disappear before `post_blend_scores`; absence from
the final score map does not establish admission failure. A shortlist that
was disabled or returned an empty fallback must be interpreted using its
`filter_status`. A caught shortlist failure invalidates an admission verdict
even if retrieval continues.

The converse also matters: gold may appear in `fts_raw` and `pre_shortlist`,
disappear at `post_shortlist`, then return through expansion in
`retrieval_returned` and rank highly in `post_blend_scores`. Downstream
reintroduction does not prove that gold survived the shortlist.

The pool-depth probe applies its registered threshold only when all 109
`pool_absent` cohort queries have usable `post_shortlist` evidence. It requires
exactly one completed lexical retrieval with that captured boundary per query.
Multiple independent retrieval calls remain visible in the report, but cannot
be collapsed into one admission pool for the verdict. Query
failures, missing stages, unsupported paths, and partial `--limit` runs produce
`INCONCLUSIVE`. `KILL` rejects the tested depth intervention; it does not prove
that gold is absent from the corpus or that other interventions cannot help.

The probe retains legacy final-map rank fields for inspection. Its histogram
field `absent_at_1000` means absence from a successfully measured final map in
the `deep1000` arm; it is not a raw FTS or shortlist absence measurement. Failed
and unmeasured queries have separate bins. BM25 proxy ranks describe a separate
raw FTS query and cannot identify a drop in the real pipeline. Proxy failures
and missing query terms are excluded from proxy absence counts.

When stage capture is requested, old arm or needle caches lacking structured
version 1 provenance are rejected with their path. Use a fresh cache directory
to capture those boundaries. Explicit failed or `not_captured` reports from
the new harness can be resumed; their status remains failed or unmeasured.
Existing receipts and real benchmark stores are not rewritten by this change.

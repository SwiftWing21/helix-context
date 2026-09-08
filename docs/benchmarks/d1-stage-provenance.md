# D1 measurement instrument on v0.9.0

This experiment branch starts at v0.9.0, commit
`55162c6b4a84070a29b9c02237714b07c1193b38`. It ports only the stage collector,
observation hooks, unsupported-path guards, and ERB ladder receipt fields from
PR #446. It does not import beta retrieval/configuration changes or Max's
bed-specific pool-depth probe. The executed revision must be recorded as
**v0.9.0 plus measurement instrumentation**, with its full commit SHA.

Run the parameterized ladder on Joe's existing 850,379-document blob and its
469-query needle/gold inputs. Do not substitute Max's 947k bed, 470-query bank,
or the separate 109-query pool-depth cohort. Resolve paths on the benchmark
host, verify every input, and freeze their hashes before starting an arm.

## Invocation

Each arm uses a fresh config derived from the v0.9.0 shipped `cymatix.toml`.
Set the registered depth, shortlist and fusion knobs explicitly and archive the
config with its SHA-256. Run one baseline arm per process with serial queries:

```text
python benchmarks/dogfood/erb/ablation_ladder.py
  --genome <verified-850k-blob.db>
  --resolved <verified-469-needles.json>
  --gold <verified-gold-by-needle.json>
  --config <frozen-arm.toml>
  --arms baseline --limit 0 --k 12 --per-query
  --out <new-run-directory/unique-arm-receipt.json>
```

The wrapped lines above are an argument template, not a ready-to-paste shell
command. Use the benchmark host's shell and bind actual paths in its manifest.
Set `CYMATIX_DISABLE_LEARN=1` and `PYTHONHASHSEED=0` in the process environment.
Record any other effective CYMATIX overrides; prevent inherited config, encoder,
or model flags from silently changing an arm. D1 is lexical: dense, SPLADE and
reranking stay off at the registered defaults. The ladder uses `read_only=True`
and `ignore_delivered=True`; ensure auxiliary state/telemetry writes use isolated
scratch paths and preserve the original benchmark bed and other workloads.

Before the full arm, run a small disposable smoke receipt with the same CLI and
config path. Verify configured depth reaches `fts_raw.fetch_depth`, observed
shortlist membership and status are present, and each successful lexical query
has exactly one retrieval observation. A smoke is not the full-cohort verdict.

## What the evidence means

`fts_raw` is the bounded SQL result after any optional prefilter and before
lifecycle/party eligibility. `fts_eligible` records the eligible IDs.
`pre_shortlist` and `post_shortlist` observe the actual shortlist boundary;
inspect `filter_status`, including `empty_fallback` and `failed`.
`final_scoring` precedes reranking and return expansion. `retrieval_returned`
observes returned candidates. Separate `post_blend_scores` and
`post_blend_candidates` observations precede budgets and delivery.

Counts and watched gold IDs are observations, not interchangeable rank maps.
Report raw entry, shortlist survival, final/head rank and delivered gold as
separate outcomes. Downstream reintroduction cannot establish shortlist survival.
`budget_tier` comes from the actual returned window metadata, alongside existing
`delivered_count`; absent metadata and failed queries use null.

Failed, unexecuted and unsupported observations cannot be treated as absence.
An overall complete report does not imply every optional stage ran. Keep
independent calls separate; ANN, sharding and parallel subqueries remain
unmeasured. Run c=1 on one manager/store per process. The new collector does not
repair shared score-map publication under concurrent requests.

Use D1's declared cohort and paired contrast, not the 109-query threshold from
Max's separate experiment. Require complete usable observations for the chosen
stage before interpreting its cohort result. A flat curve over depths 48 to
384 supports only a bounded finding about this intervention; it does not prove
corpus absence or establish entry-term coverage as the cause.

Capture runs inside timed requests. Instrumented wall time includes overhead
that varies with pool size. Throughput/latency claims need separately matched
uninstrumented runs and an uncontended measurement window. If the box is
contended, preserve the inspection and label comparative latency not measured.

## Verification scope

The backport is tested through the real ladder CLI, real TOML loading, a real
SQLite store and manager, with no retrieval/manager mocks in the child process.
Core tests cover raw/eligible filtering, shortlist loss and downstream return,
empty and failed stages, nested capture isolation and unchanged capture-off
results. The beta audit's optional-feature skips do not certify this branch.
Neither a tiny-store test nor source inspection certifies Joe's remote bed or
the start of a long run; the kickoff receipt must provide that evidence.

Local verification on 2026-09-08: **255 passed, no skips** across the stage,
real CLI, ablation helpers/metrics, BM25 prefilter, blend, determinism, ANN
threshold, additive-weight and shard-router test files. The later beta-only
bench-rank-hash test and Max-specific pool-depth probe test were not ported.
An AST comparison after removing only observer nodes reproduced the v0.9.0
store, manager and router implementations. Four boundary-label mutations and
a disabled-capture mutation were detected by the added tests.

Ruff's error/undefined-name checks pass on the other affected files. The old
store retains two F821 annotations (`BGEM3Codec` and `np`) that reproduce on the
unmodified v0.9.0 blob; no new lint diagnostic was introduced. Whitespace checks
and Python compilation passed. This is targeted verification, not a full old-
version suite pass or a hardware/corpus validation.

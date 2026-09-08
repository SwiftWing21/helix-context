# Measurement test applicability audit

Date: 2026-09-08. Scope: PR #446's measurement hooks and ERB harness, the
39 skips in its original full offline run, and the v0.9.0 instrument needed
for Joe's D1 experiment. Three independent audit agents traced the code,
exercised real paths, and checked deliberate faults; the coordinator reviewed
the changes and reran the final applicable selections.

## Findings and repairs

The original core tests exercised reachable code. Their real SQLite queries
reproduced different upstream histories with the same final score map, actual
shortlist loss, and subsequent co-activation/blend reintroduction. The blend
test replaces spectral inputs, not retrieval, expansion or the observer hooks.
ANN and shard tests assert unsupported-path guards; they do not certify ANN
union or parallel fanout. Parallel subqueries remain unmeasured.

The original harness tests mocked the manager, and the pool-depth CLI test
also replaced `run_arm`. They tested wrapper and receipt logic but did not
prove end-to-end wiring. The new real subprocess test uses parsed TOML, a
disposable SQLite bed, the actual manager/store, and JSON output. It checks
changed depth/shortlist settings, capture off, warmup isolation, a no-match
early return and an actual invalid-query exception. The child starts outside
the checkout with inherited configuration/import overrides cleared, so pytest
fixtures cannot substitute a fake manager.

Five additional core cases now assert real lifecycle/party filtering, an
eligibility SQL failure after a successful raw fetch, a genuinely empty
shortlist fallback, and a caught shortlist SQL failure that still returns
documents. Failed and unexecuted evidence must remain distinct from absence.

The audit also found that D1's required `budget_tier` was not exported. The
ladder now copies it from actual window metadata beside `delivered_count`;
failed requests and early returns without metadata emit null.

Two unrelated tests in the skipped set never reached their central assertions:

- The budget fixture used fewer tokens than its budget, and the shipped
  delivered-document floor would select truncation rather than eviction after
  merely reducing the budget. The test now explicitly disables that floor for
  this case, forces overflow, and requires a real eviction of the lowest score.
- The short citation fixture was lifecycle-filtered, and the query used a
  different party from the attributed ingest. Explicit fixture tags and the
  owning party now make the real document eligible; the test requires that
  exact ingested ID and its attribution in the response.

These changes repair test setup and assertions; they do not change production
budget or attribution behavior. The historical skip count remains 39.

## Evidence

| Check | Observed result |
|---|---|
| Final beta measurement and surrounding selection | 307 passed, no skips |
| Repaired budget/server test files | 89 passed, no skips |
| v0.9.0 D1 applicable selection | 255 passed, no skips |
| Deliberately disable capture in an isolated CLI copy | Test fails at complete vs not_captured |
| Label raw IDs as eligible | Both lifecycle and party assertions fail |
| Mislabel eligible-query failure as raw failure | Raw-evidence assertion fails |
| Mislabel empty fallback or caught shortlist failure | Status assertions fail |
| Require budget tier before adding its export | Real CLI test fails with missing field |
| Replace the two self-skip exits with strict assertions before fixing fixtures | Both tests fail, neither skips |

The coordinator reran the 307 beta tests, 89 budget/server tests and 255 D1
tests. Fresh import checks resolved each package to its intended worktree.
Ruff error/undefined-name and whitespace checks pass on the beta changes.
The D1 old store retains two pre-existing F821 type-annotation diagnostics,
reproduced directly from the unmodified v0.9.0 blob; its other affected files
pass lint, and compilation/whitespace checks pass.

The separate audit's AST comparison removed only the observer imports,
decorator, guards, assignments and exception-name binding. The remaining
store/manager/router AST matched v0.9.0 exactly. The D1 branch contains no
beta default, ranking, budget or admission changes.

## Skips and limits

[Every skipped test/module and its exact reason](2026-09-08-measurement-skipped-tests.md)
is listed separately: 35 testcase skips plus four collection skips. The 25
live deselections are a different count. No stage-provenance test was skipped.

Enabling the hash-seed golden exposed a pre-existing mismatch: query 0 expects
two documents but returns `score_below_floor` abstention. The same expected and
actual payloads reproduce on clean pre-PR beta `0891f5c`. This remains unresolved
and is not claimed as passing coverage. The installed console-command test
passes with its venv directory on PATH. Optional dependencies and unavailable
hardware coverage were not added by this audit. No new full-suite pass is claimed.

The original full run was 4,607 passed, 39 skipped and one Windows Bash-path
failure, with 25 live tests deselected. That Bash failure reproduces on clean
beta; all 21 installer tests pass with Git Bash selected. The new focused runs
above are separate evidence and do not retroactively rewrite those totals.

## D1 applicability

Joe's registered experiment uses his 850,379-document bed, 469-query bank and
v0.9.0 code. Max's pool-depth probe hardcodes a different bed and historical
109-query cohort; its threshold is not applicable to Joe's D1 experiment.

The separately verified D1 instrument is commit
`d0beec1ca75b775662da3200d712a7764a5fef3a`, based on v0.9.0
`55162c6b4a84070a29b9c02237714b07c1193b38`, with
[run instructions](https://github.com/mbachaud/Cymatix-Context/blob/d0beec1ca75b775662da3200d712a7764a5fef3a/docs/benchmarks/d1-stage-provenance.md).
It supports the registered serial lexical path and actual per-query admission
evidence. Its tests do not validate Joe's remote input identity, host state or
latency, and do not prove a remote process has started. Those require the
on-host preflight and kickoff receipt. Instrumented timings include capture
overhead; comparative cost claims require matched uninstrumented runs.

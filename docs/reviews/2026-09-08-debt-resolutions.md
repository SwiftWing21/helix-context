# Test, dependency and issue resolutions — 2026-09-08

Scope: beta `0891f5c`, the measurement repair in PR #446, and follow-up
maintenance. Three subagents investigated independent test/dependency areas;
the parent verified their changes and a second agent reviewed each fix.
The frozen D1 source and benchmark inputs were not modified.

## Resolutions

| Finding | Resolution | Evidence / remaining gate |
| --- | --- | --- |
| Historical Stage 5 test silently skipped unless hash seed was zero, then failed on clean beta | Preserve the original archive; actively check its 100-query generic rendering contract and current omitted/explicit-generic parity | All five classifier classes must deliver; abstention must also occur. Wrong-branch mutations fail. Archive bytes unchanged; see `tests/golden/README.md`. |
| Windows installer check selected WSL `bash.EXE` and passed it a Windows path | Resolve native Bash, reject known WSL shims, syntax-check script bytes through stdin | Actual installer parses; deliberately invalid shell syntax fails; Windows discovery regressions included in CI. |
| Installed CLI smoke skipped when its Scripts directory was absent from PATH | Search the current interpreter's scripts directory first; invoke the real launcher against this checkout | Actual launcher runs, with child-only PYTHONPATH; no persistent PATH changes. |
| Local MCP1 SDK, missing Markdown and BM25 hid optional coverage | Refresh only `mcp>=2,<3`, `markdown>=3.4`, `rank-bm25>=0.2.2`, `httpx2>=2.0.0` and required transitives | Main project environment now has MCP2.2.0, Markdown3.10.3, rank-bm25 0.2.2, httpx2 2.12.0; pip check passes. Existing FastAPI/Starlette/Pydantic versions retained. |
| Contributor setup omitted dev dependencies; BM25 was undeclared | Correct install instructions; add rank-bm25 and httpx2 to dev | Core and feature extras remain unchanged. CI checks all ten explicitly promised optional-module imports before tests. |
| PR446 citation test failed the intentional no-spaCy CI leg | Declare the ingestion model requirement and require real seeded citations in two model-independent cases | PR446 commit `f975039`; 101 affected tests with model, 68 server tests plus 15 declared model skips without spaCy. |
| [#411](https://github.com/mbachaud/Cymatix-Context/issues/411) remained open after implementation | Closed as completed | PR425 merge `29a1198` is an ancestor of beta. Configured cutoff 200, bare-store compatibility default 0. 72 existing checks passed; original paired receipts verified on disk. |
| [#418](https://github.com/mbachaud/Cymatix-Context/issues/418) malformed known config sections crash | [PR447](https://github.com/mbachaud/Cymatix-Context/pull/447) warns and discards malformed top-level section shapes before alias normalization | 120 new cases; 189 affected tests passed. Open pending review/merge. Nested-field validation remains separate. |
| [#421](https://github.com/mbachaud/Cymatix-Context/issues/421) dimensions document misstates defaults | PR447 corrects activation claims and points to generated config reference | 16 settings checked against both default dataclasses and shipped TOML; historical benchmark table unchanged. Open pending review/merge. |

MCP2 is a real supported dependency, not a package-range mistake. The project
already migrated to `mcp.server.mcpserver`; SDK2.0 was released July 28, 2026.
See the [official migration guide](https://py.sdk.modelcontextprotocol.io/migration/)
and [MCP2.0 release](https://pypi.org/project/mcp/2.0.0/). Starlette documents
[httpx2 as its TestClient client](https://starlette.dev/testclient/); the core
runtime's separate httpx dependency remains appropriate.

## Validation and skip accounting

The exact historical 39 records remain documented in
[PR446's inventory](https://github.com/mbachaud/Cymatix-Context/blob/101dadaf7ffc8841e341145ee0af9f6c84c56144/docs/reviews/2026-09-08-measurement-skipped-tests.md):
35 testcase skips and four whole-module skips. Eight MCP skip records masked
more than eight test functions, so subtracting skip records is not a valid
coverage count. Hardware/platform/live omissions remain legitimate boundaries.

- Parent affected selection in the repaired project environment: **196 passed,
  no skips** (golden/caller, CLI, installer, wiki, BM25, and four MCP files).
- Independent clean environment with latest compatible core dependencies:
  **163 passed, no skips**. Separate web selection: 82 passed, 16 explained
  skips (15 model dependencies and the old attribution skip repaired in PR446).
- Actual CI preflight extracted from YAML: all ten imports passed. Ruff and
  whitespace checks passed. Independent final diff review found no defects.
- Full offline suite in the repaired main environment: **4,672 passed,
  9 skipped, 25 live deselected, 80 warnings**, 380.65 seconds. Command:
  `python -m pytest tests/ -m "not live" -q -rs -p no:cacheprovider` with a
  disposable basetemp and JUnit receipt. No failures. This branch starts at
  beta and does not include PR446's separate test repairs or PR447.

The nine remaining skipped cases are five explicit hardware opt-ins (one MPS,
two CUDA, two ROCm), one graph-summary performance opt-in, one POSIX-only
supervisor case, and two weak-fixture skips already repaired in PR446:
`test_lowest_score_dropped_first_under_overflow` in score-aware budget trimming
and `test_citation_includes_attribution_when_present` in server citations.
Both must be evaluated with PR446's deterministic replacements, not accepted as covered
by this branch's full-suite pass. No whole-module dependency skips remain.

Local checks use Windows/Python3.14. The isolated stack additionally validated
FastAPI0.141.1/Starlette1.6.0; the shared environment remains on
FastAPI0.139.2/Starlette1.3.1. Starlette1.6's deprecated AnyIO BlockingPortal
alias is an upstream warning, separate from the resolved httpx fallback warning.
Linux/Python3.12 and macOS behavior require the corresponding CI jobs.

## Remaining work and dependencies

| Priority / issue | Concrete outstanding work | Dependency and completion evidence |
| --- | --- | --- |
| Hosted console prerequisite: [#434](https://github.com/mbachaud/Cymatix-Context/issues/434) | Bind observability compose ports to loopback; replace default Grafana credentials and make anonymous mode opt-in | Resolve before exposing the console/observability stack beyond the host. Test effective compose bindings, credential defaults and access behavior. |
| Concurrent correctness: [#439](https://github.com/mbachaud/Cymatix-Context/issues/439) | Serialize first SPLADE load, bound/clear freshness cache, copy score publications under lock | Port the reviewed changes onto current beta and exercise concurrent calls. Needed before claims about concurrent measurement; registered D1 is serial. |
| Silent measurement debt: [#431](https://github.com/mbachaud/Cymatix-Context/issues/431) | Bound harmonic-link SQL against SQLite variable limits and surface skipped/error states | Warning/bounds repair can be isolated. Restoring previously dropped tier contribution needs paired measurements on a bed with populated harmonic links; current empty-link beds cannot validate it. |
| Retrieval change: [#430](https://github.com/mbachaud/Cymatix-Context/issues/430) | Decide whether tight/focused tier caps should honor the delivered-document floor | Requires paired retrieval receipts with identical source/corpus/config/ingestion. Do not silently change the D1 baseline. Stage provenance now exports actual budget tier. |
| API compatibility: [#417](https://github.com/mbachaud/Cymatix-Context/issues/417) | Finish remaining wire vocabulary migration | Inventory clients and aliases first, then add contract/compatibility evidence. Not a prerequisite to the current serial D1 run. |
| CI dependency reproducibility | No tracked platform constraints/lock file | Preserve library version ranges; a constraints proposal needs a refresh cadence and a latest-compatible lane. Do not freeze an unrelated workstation environment as the project lock. |
| Extras documentation | SETUP says `all` includes launcher-native, but `all` lacks pywebview and launcher py-cpuinfo | Decide and align the advertised extras matrix; no large optional installation was needed for this audit. |
| Nested config validation | Non-table `vault.traces` is outside #418's top-level shape fix | Add a focused nested-schema contract before broadening loader recovery. |
| Upcoming dependency removals | Class-scoped pytest fixtures use instance methods; the semantic encoder uses SentenceTransformer's deprecated dimension method; Joblib emits NumPy shape warnings | Migrate affected fixtures before pytest10, use the supported encoder method with compatibility where needed, and track the upstream Joblib/NumPy warning. These warnings did not fail this run; they are not silently suppressed. |

No owner is assigned on the remaining GitHub issues as inspected today. The
table gives the implementation and evidence needed for a concrete next task;
these items are not claimed resolved by the test-maintenance PR.

## Joe / D1 handoff

The authenticated-console and status-panel specification was sent in exchange
turn0027. D1 start authorization and the verified observer-only v0.9.0 instrument
were sent in turn0029. A current-status follow-up was published as
[turn0030](https://github.com/addiplus/cc-exchange/blob/d463fff/threads/spark-erb-receipts/0030-max-debt-resolutions-and-kickoff-status.md),
exchange commit `d463fff`. It requests the console implementation/PR state and
the D1 host, PID/job, start time, source/config/input hashes, smoke result and
first full-arm progress.

No Joe kickoff receipt was present at the latest check. The run is dispatched,
but remote execution is **unconfirmed**. D1 remains pinned to
`d0beec1ca75b775662da3200d712a7764a5fef3a` on v0.9.0; none of these beta fixes
changes that experiment or establishes its results.

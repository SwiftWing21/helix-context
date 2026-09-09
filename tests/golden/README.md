# Historical Stage 5 responses

`pre_stage5_responses.jsonl` is the unchanged 100-query response archive added
by `d288f9bfe7761d8500e67eab9cb81378400f4304` (PR #47). That commit records an
initial capture at `66104e0`, followed by a refresh against the merged Stages
2/3/4/6/7 before Stage 5 landed. Its first parent is
`5fce44102ed873075ef6b6d29c3d418d609dc6f1`. The exact intermediate refresh
revision is not recorded separately in the squash commit. The historical
capture used `PYTHONHASHSEED=0`, an in-memory store, a mock compressor, and the
synthetic corpus in `_capture_pre_stage5_responses.py`.

The archive remains evidence of that response surface. Do not regenerate it
from current code to resolve a failure. Reproducing the original capture needs
the script, source, and dependencies from the historical checkout. The current
helper refuses standalone capture to prevent accidental replacement.

## Active invariants

`tests/test_caller_model_class.py` checks two separate contracts on every normal
pytest run, without a hash-seed or missing-fixture skip:

- Historical generic rendering: all 100 decoder prompts match the archive
  byte for byte for both omitted and explicit `generic` callers. Each row must
  preserve whether delivery occurred (92 deliveries and eight misses). The 92
  deliveries exercise all five classifier classes, preserve the recorded
  decoder selection and assembly cap, and retain generic context headers.
  This fixture explicitly selects historical additive retrieval, the full
  fallback decoder, a 6000-token expression budget, no session elision or
  foveation, no delivered-document floor, and the legacy splice target.
- Current default compatibility: two independent managers use current defaults
  for the same 100 queries. Omitted and explicit `generic` calls must produce
  identical content, document IDs and order, token counts, health, metadata,
  and the request-scoped retrieval evidence used for downstream citation scores.
  Only the independently generated `pipeline_request_id` is excluded by the
  serialization helper. Both delivery in all five classes and abstention must
  occur. These managers share a process and therefore its hash seed.

The historical test does **not** claim full pipeline byte identity across
later retrieval releases. RRF became the default in `e4889f6` (PR #247), default
decoder settings changed, the shared test compressor changed in `4653e7e`
(PR #238), and the delivered-document floor became 12 in `5c9f574` (PR #409).
Scores, selected document sets, score-bearing headers, health aggregates, and
new metadata can therefore differ legitimately from the archive. Retrieval
quality and default changes belong to their dedicated tests and measured
baselines in `docs/benchmarks/BASELINES.md`.

Other golden files in this directory retain their own capture scripts and
consuming tests; this archival policy concerns only the pre-Stage-5 responses.

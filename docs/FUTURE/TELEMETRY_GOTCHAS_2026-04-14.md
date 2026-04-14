# Telemetry Gotchas — 2026-04-14

End-to-end smoke test of the Sprint 5A OTel pipeline surfaced four real bugs that
were all hiding behind silent `log.debug` or default Python logging behavior.
This is the receipt so the same trap doesn't catch the next operator.

## 1. Stale replicas silently shadow master schema

**Symptom:** Every harmonic-related Grafana panel showed "No data" even after a
verified backfill committed 192K edges to the master genome.db. Direct `sqlite3`
query against the master file showed all 192,602 rows present and the schema
correct.

**Root cause:** `genome.read_conn` is a `@property` (genome.py:1125-1140) that
returns connections in priority order: **replication replica > local reader >
write connection**. The replicas at `C:/helix-cache/genome.db` and
`E:/helix-cache/genome.db` were **stale by 1-3 days**, missing both:
- The Sprint 4 schema migration (`source`, `co_count`, `miss_count`, `created_at`
  columns on `harmonic_links`)
- Today's 179K-edge backfill (only ~10K rows on each replica vs 192K on master)

`emit_gauges_snapshot` queried `SELECT source, COUNT(*) FROM harmonic_links` against
the replica → `sqlite3.OperationalError: no such column: source` → entire try
block bailed → all gauges after chromatin silently emitted nothing. (See gotcha
#2 for why this was silent.)

**Why replicas were stale:** `ReplicationManager.notify_write` syncs every
`replica_sync_interval` writes (default 100). Today's `backfill_seeded_edges.py`
ran in a **separate process** that bypasses the helix-side `ReplicationManager`
entirely — it writes directly to the master DB file. Replicas had no way to know.

**Immediate fix:** Stop helix → delete replica files → restart. The
`ReplicationManager.__init__` checks for missing replicas and triggers an
"Initial replica sync" on startup (replication.py:78-83), full-copying master.
Took ~3 seconds per replica via `sqlite3.Connection.backup()` (delta-page copy).

**Real follow-up fixes** (none shipped yet — pick one before next external
write to the master):

- **A.** Have `backfill_seeded_edges.py` (and any other external writer)
  call `mgr.sync_now()` after committing, OR write through helix's
  `WriteQueue` so `notify_write` fires per row.
- **B.** Add a startup integrity check: `PRAGMA schema_version` on each replica
  vs master, force re-sync if they diverge. ~10 LOC in `ReplicationManager`.
- **C.** Make `emit_gauges_snapshot` (and any other instrumentation read) use
  `genome.conn` (master) instead of `genome.read_conn` (replica). Trade-off:
  no concurrency benefit for telemetry reads, but telemetry is small and rare.
  ~5 LOC in `telemetry.py`.

Recommendation: **A + B**. A is correct-by-construction; B is the
defense-in-depth that catches anything A misses (e.g., manual sqlite3 sessions).

## 2. `log.debug` swallowed three different production bugs

**Symptom:** Three separate hidden failures during today's smoke test, each
wrapped in `try/except: log.debug(..., exc_info=True)`:

1. `emit_gauges_snapshot` failed with `no such column: source` → invisible
2. `OTel /context latency emit` skipped → invisible
3. `tier telemetry emit` skipped → invisible

All three exceptions logged at DEBUG level. Python's root logger defaults to
WARNING. Uvicorn's `--log-level info` only affects uvicorn's own loggers, not
`helix.*` loggers. So none of these failures appeared in any normal log output.

**Root cause pattern:** Defensive `try/except` that's correctly silent in
*expected-no-op* cases (OTel disabled, telemetry packages missing) but also
silent in *actual-failure* cases. Same swallow path for both.

**Fix:** Promote the `log.debug` to `log.warning` for these three sites.
Warnings appear at default logging levels; if the failure is benign (OTel off),
it just won't fire. Commits `f372866` and `512e281`.

**Pattern to apply elsewhere:** Anywhere we have `try/except → log.debug`
inside a function whose payload is observable side-effects (metric emit,
external write, cache update), promote the failure log to WARNING. The DEBUG
discipline is right when the function returns a value and the caller can
detect the no-op via the value; it's wrong when the function is fire-and-forget.

## 3. OTel "telemetry ON" confirmation was also at INFO level

**Symptom:** Operators set `HELIX_OTEL_ENABLED=1`, restart, look in the log,
see no confirmation. They can't tell whether OTel turned on or not, since the
absence of a log line is consistent with both "OTel disabled" (also INFO,
also invisible) and "no Python logging configured at all."

**Fix:** Promoted the `OTel telemetry ON, endpoint=...` log line in
`setup_telemetry()` from INFO to WARNING (commit `32f578e`). It's not
informational in the sense that matters — operators *need* this line to confirm
the desired state. Default INFO is hidden. WARNING is shown.

## 4. OTel histogram unit name is appended to the metric name

**Symptom:** Grafana panel "Per-tier contribution histogram" showed "No data"
despite the histogram firing on every /context call. Direct Prometheus query
for `helix_tier_contribution_bucket` returned empty.

**Root cause:** `tier_contribution_histogram()` in `telemetry.py` is created
with `meter.create_histogram("helix_tier_contribution", unit="score", ...)`.
The OTLP-to-Prometheus exporter appends the unit to histogram metric names by
convention, producing `helix_tier_contribution_score_bucket`,
`..._score_count`, `..._score_sum`. The dashboard JSON queried the
unit-less name. Series existed in Prometheus all along, just not under the
name the dashboard expected.

**Fix:** Updated dashboard JSON to query `helix_tier_contribution_score_bucket`
(commit `512e281`).

**Pattern to apply elsewhere:** Whenever creating a histogram with a `unit`
argument, the Prometheus name will be `<name>_<unit>_<bucket|count|sum>`. For
counters and gauges with units (e.g., `helix_genome_size_bytes` already has
`bytes` baked into the name; `unit="By"` is set but doesn't double-append),
verify in Prometheus before wiring the dashboard query.

## Operational checklist for the next telemetry sprint

When changing or adding a metric, before declaring it shipped:

1. **Run helix with `python -c "import logging; logging.basicConfig(level=logging.INFO); ..."`** wrapper at least once and grep startup output for `OTel telemetry ON` — confirm the right env vars made it to Python.
2. **`Test-Path` (or equivalent) all replica paths from `helix.toml`** — if any are stale (modified before the last schema migration), purge and let helix re-sync.
3. **Hit /stats once, then read the helix log for any `WARNING helix.telemetry: ... failed` lines** — if any appear, the gauge is silently broken.
4. **Direct Prometheus query against the bare metric name and the `_unit_bucket/count/sum` variants** — confirm the actual exported name before wiring a dashboard panel.
5. **Open the dashboard in "Last 1 hour"** — narrower time windows will show "No data" for low-rate metrics even when they're flowing correctly.

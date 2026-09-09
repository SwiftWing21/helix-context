# Observability — OTel → Grafana stack

Cymatix emits OpenTelemetry traces + metrics when telemetry is enabled — via `CYMATIX_OTEL_ENABLED=1`, `[telemetry] enabled = true` in cymatix.toml, or automatically under the tray launcher once the local stack is up. Everything upstream of the LLM answer boundary becomes visible on one dashboard.

## Quick start

**One-time setup** — bring up the observability stack via the native
sidecar (recommended; no Docker required):

```powershell
# Windows
scripts\setup-grafana-telem.ps1
```

```bash
# Linux / macOS
scripts/setup-grafana-telem.sh
```

That wrapper downloads the five pinned binaries into `tools/native-otel/`,
renders runtime configs, and wires dashboard provisioning. It is
idempotent — re-runs skip already-installed binaries and only refresh
configs. Then start the supervisor:

```bash
cymatix-launcher --tray              # daily-driver flow (tray icon + cymatix)
cymatix-launcher --no-autostart      # observability only, no cymatix backend
```

That launches:

| Service | Port | Purpose |
|---|---|---|
| OTel Collector | 4317 (gRPC), 4318 (HTTP) | receives OTLP from cymatix |
| Prometheus | 9090 | metrics storage |
| Tempo | 3200 | trace storage |
| Loki | 3100 | log storage |
| Grafana | 3000 | dashboards (`admin` / `admin`) |

**Install OTel client packages:**

```bash
pip install "cymatix-context[otel]"
```

**Enable on the cymatix server** — either via env vars:

```bash
export CYMATIX_OTEL_ENABLED=1
export CYMATIX_OTEL_ENDPOINT=localhost:4317   # default
python -m uvicorn cymatix_context._asgi:app --port 11437
```

or via the `[telemetry]` section in `cymatix.toml`:

```toml
[telemetry]
enabled = true
endpoint = "localhost:4317"
```

Precedence per knob: `CYMATIX_OTEL_*` env var > `[telemetry]` toml >
code default. Env wins in both directions — an explicit
`CYMATIX_OTEL_ENABLED=0` silences a toml `enabled = true`.

**Tray launcher:** no configuration needed. When the launcher starts
(or adopts) the native observability stack — and the collector's OTLP
port `:4317` is actually accepting connections — it exports
`CYMATIX_OTEL_ENABLED=1` into the cymatix child's environment, so the
default tray boot ships data to Grafana out of the box. Set
`CYMATIX_OTEL_ENABLED=0` yourself to keep the stack up with a silent
backend. If the stack fails to start, or a service spawns but never
becomes ready (red status), the collector-port probe fails and the
export is skipped — a backend dialing a dead collector would wedge its
gRPC channel. The endpoint itself is not exported; it resolves via the
normal env > toml > default chain, so an explicit `[telemetry]
endpoint` in cymatix.toml is respected.

Open <http://localhost:3000/d/cymatix-overview>. Retrieval latency, tier contributions, CWoLa f_gap, lifecycle-tier distribution (legacy: chromatin), co-activation edges by provenance (legacy: harmonic-edges-by-source) — all live.

**Docker-compose alternative.** If you prefer the containerized stack,
see [`deploy/otel/README.md`](../../deploy/otel/README.md). Both
runtimes are bit-for-bit compatible — same dashboard JSON, same
datasource UIDs, only the receiver runtime differs.

The Docker stack is a **local-only template**. Published ports bind to
`127.0.0.1`; Grafana anonymous Viewer access defaults off. The initial
Grafana password can be set with `CYMATIX_GRAFANA_ADMIN_PASSWORD` (local
fallback: `admin`); existing volumes retain their password. Loki and the
other telemetry APIs remain unauthenticated inside the Compose network.
Before exposing any service to other hosts, configure authenticated TLS
ingress and network isolation, rotate credentials, and keep raw service
ports private. See the
[hosted-session review](../reviews/2026-09-01-hosted-session-security-review.md).
Logs and query telemetry can include sensitive text even with query
redaction enabled.

## What's instrumented

### Traces (`/context` span tree)

Auto-instrumentation via `opentelemetry-instrumentation-fastapi` wraps every route in a span. Attributes are the request path + status code. Sampling is controlled by `CYMATIX_OTEL_SAMPLER_RATIO` (default `1.0`; drop to `0.1` at high QPS).

### Metrics

Two surfaces:

1. **Cymatix-domain** — `cymatix_*` metrics that capture the engine's
   internal mechanics (pipeline stages, retrieval tiers, knowledge-store
   health, A/B cluster convergence, co-activation graph). Vocabulary
   has bio-domain origins (chromatin, harmonic_links, CWoLa) and the
   metric names are stable contracts; dashboard panels translate to
   engineering names with inline references — see `docs/ROSETTA.md` for
   the full bidirectional table.
2. **OTel `gen_ai.*` standard** — `cymatix_context/telemetry/genai_telemetry.py`
   (#209). `cymatix_genai_*` token usage / TTFT / finish reasons /
   per-call cost following the upstream GenAI semantic conventions,
   plus `cymatix_context_cache_outcome_total`. Metric names carry the
   `cymatix_` namespace prefix; the spec name (`gen_ai.client.token.usage`,
   …) lives in each instrument's description. Emitting call sites today:
   the three `/v1/chat/completions` proxy forward paths (streaming,
   non-streaming, raw passthrough) and `CachedDAL.fetch` for the cache
   counter. Compressor/embedding backends are not yet instrumented —
   `llm_span` / `record_response` are the ready-made helpers when they
   are.

   | Metric | Type | Labels | Source |
   |---|---|---|---|
   | `cymatix_genai_client_token_usage` | histogram | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.operation.name`, `gen_ai.token.type` ∈ {input, output, cached, reasoning} | proxy forward paths |
   | `cymatix_genai_time_to_first_chunk_seconds` | histogram | same minus token.type | streaming proxy path |
   | `cymatix_genai_cost_usd` | histogram | same minus token.type | `estimate_cost_usd` over the module's `PRICE_TABLE`; 0.0 for local/unpriced models |
   | `cymatix_genai_finish_reasons_total` | counter | `finish_reason` | proxy forward paths |
   | `cymatix_context_cache_outcome_total` | counter | `outcome` ∈ {hit, miss, partial} | `CachedDAL.fetch` (partial = stale-then-refetched; not on the retrieval hot path yet) |

| Metric | Type | Labels | Source |
|---|---|---|---|
| `cymatix_context_latency_seconds` | histogram | `health`, `budget_tier`, `cold_tier_used` | `/context` endpoint |
| `cymatix_pipeline_stage_seconds` | histogram | `stage` | `_stage_timer` in `context_manager.py` records the histogram; `pipeline_stage_span` emits spans only — both cover all 7 stages |
| `cymatix_context_health_status_total` | counter | `status` ∈ {aligned, sparse, stale, denatured} | `/context` health classifier |
| `cymatix_context_ellipticity` | histogram | `party` | per-query coverage × density × freshness |
| `cymatix_tier_contribution` | histogram | `tier` | `query_genes` accumulation |
| `cymatix_tier_fired_total` | counter | `tier` | `query_genes` accumulation |
| `cymatix_rrf_fused_score` | histogram | — | RRF fused-score distribution, recorded per query per fused document (`query_genes`, default `fusion_mode = "rrf"` path); attribute-less by design — a per-document label would explode series cardinality |
| `cymatix_cwola_bucket_total` | counter | `bucket` ∈ {A, B, pending} | `cwola.log_query` + `sweep_buckets` |
| `cymatix_cwola_f_gap_sq` | gauge | — | `cwola.sweep_buckets` |
| `cymatix_harmonic_edges_total` | gauge | `source` ∈ {seeded, co_retrieved, cwola_validated} | `/stats` snapshot |
| `cymatix_chromatin_state_total` | gauge | `state` ∈ {open, euchromatin, heterochromatin} | `/stats` snapshot |
| `cymatix_genome_size_bytes` | gauge | `kind` ∈ {raw, compressed} | `/stats` snapshot |
| `cymatix_genome_wal_size_bytes` | gauge | — | `/stats` snapshot |
| `cymatix_genome_signal_seconds` | histogram | `signal` | per-signal SQLite query timing |
| `cymatix_genome_checkpoint_blocked_total` | counter | — | WAL checkpoint contention |
| `cymatix_hub_concentration_ratio` | gauge | — | `/stats` snapshot |
| `cymatix_hub_inbound_degree` | gauge | `stat` ∈ {max, p99, p95, p50, mean} | `/stats` snapshot |
| `cymatix_ribosome_call_seconds` | histogram | `backend`, `model`, `call_kind` | every compressor call |
| `cymatix_ribosome_info` | gauge | `backend`, `model`, `cost_class` | active compressor backend |
| `cymatix_dense_cosine` | histogram | `arm` ∈ {hot, cold} | dense-recall merge + cold-tier scan (#209) |
| `cymatix_shard_fanout` | histogram | — | shards consulted per `ShardRouter.query_genes` (#209) |
| `cymatix_shard_discrimination` | histogram | — | fraction of healthy shards hit per routed query (#209) |
| `cymatix_know_decision_total` | counter | `outcome` ∈ {know, miss, abstain}, `reason` | `decide_know_or_miss` (#209) |
| `cymatix_session_tokens_saved_total` | counter | — | session working-set elision savings (#209) |
| `cymatix_splice_ratio` | histogram | `caller_model_class` | assembled-window compression ratio (#209) |
| `cymatix_know_confidence` | histogram | — | KnowBlock confidence, know outcomes only (#209 ph2) |
| `cymatix_abstain_total` | counter | `gate` ∈ {floor_and_ratio, ratio_only}, `fusion_mode` | ABSTAIN gate in `pipeline/tier_logic.py` (#209 ph2) |
| `cymatix_freshness_demotion_total` | counter | `status` ∈ {stale, missing, unknown, superseded} | freshness revalidation + supersession (#209 ph2) |
| `cymatix_session_elided_total` | counter | — | elision-stub event count (#209 ph2) |
| `cymatix_pki_candidates` | histogram | — | docs hit by ≥1 path_key_index pair per query (#209 ph2) |
| `cymatix_pki_pairs_skipped_total` | counter | — | PKI pairs over the noise cutoff (#209 ph2) |
| `cymatix_fingerprint_filtered_total` | counter | `cause` ∈ {floor, cap} | /fingerprint floor/cap outcomes (#209 ph2) |
| `cymatix_ingest_vram_bytes` | gauge | — | CUDA memory per dense ingest batch (#209 ph2) |

The #209 ph2 rows are the hallucination-visibility completion set — gates
and alert queries in `docs/specs/2026-07-01-goal-gates-hallucination-visibility.md`.

`/stats`-sourced gauges are refreshed each time `/stats` is hit.
Prometheus scrapes every 15s — if nothing polls `/stats`, the gauges go
stale. `benchmark_monitor.py` or a cron scraping `/stats` keeps them
fresh.

### Spans

Two span families sit on top of FastAPI auto-instrumentation:

1. **Pipeline stage spans** — `cymatix.pipeline.<stage>` for each of
   the 7 pipeline stages (classify / extract / express / rerank /
   splice / assemble / persist), plus `cymatix.pipeline.build_context`
   as the request-level root wrapping `build_context`. Implemented via
   `cymatix_context.telemetry.pipeline_stage_span()`, which emits the
   span only; the matching `cymatix_pipeline_stage_seconds` histogram
   point comes from the `_stage_timer` context manager in
   `context_manager.py`. Both mechanisms cover all seven stages. The
   persist span is emitted from the background `learn()` task after
   the response ships, so it is **not** a child of the
   `cymatix.pipeline.build_context` root (which closed with the
   request) — on the server path it attaches to the enclosing HTTP
   request span, outside the build_context waterfall. Lets Tempo
   show the per-request waterfall instead of just the request
   boundary.
2. **GenAI client spans** — OTel GenAI semantic-convention spans
   (`<operation> <model>`, e.g. `chat qwen3:8b`) via
   `genai_telemetry.llm_span()` (#209). Attributes:
   `gen_ai.provider.name`, `gen_ai.operation.name`,
   `gen_ai.request.*`, and `gen_ai.response.*` /
   `gen_ai.usage.*` populated by `record_response()`. Emitted today
   on the `/v1/chat/completions` proxy forward paths; note the
   streaming path opens its span *after* the stream completes (the
   response was already forwarded chunk-by-chunk), so that span's
   duration is not the upstream latency — use the `total_ms` field
   on the `cymatix.proxy` log line or the TTFT histogram for timing.
   Compressor/embedding call sites are not yet wrapped.

### Logs

Cymatix's `log.warning` / `log.debug` calls propagate to stdout; when
running under the OTel SDK with a log handler configured, they flow to
Loki tagged with trace context so you can pivot from a slow span to
its logs.

Every `/v1/chat/completions` request additionally emits one
structured-JSON `cymatix.proxy` log line (`genai_telemetry.
emit_proxy_log_line()`, #209): request id, trace id, model + provider,
token counts split four ways (in/out/cached/reasoning), TTFT and total
latency, finish reason, cost estimate, and the prompt's SHA256-prefix
hash (never the prompt text). Filter in Loki with
`{logger="cymatix.proxy"} |= "proxy.call"`.

## Privacy

Query text is hashed by default — spans carry `query=<first-50-chars>[hash:<12-hex>]`. Set `CYMATIX_OTEL_REDACT_QUERY=0` to store raw query strings (dev only; do not enable in shared deployments).

## Configuration

Every knob is settable two ways — an env var or its `[telemetry]` key
in `cymatix.toml`. Resolution per knob: **env var > toml > default**
(`cymatix_context.telemetry.otel.resolve_telemetry_settings`). An env var
set to the empty string counts as unset.

| Env var | `[telemetry]` key | Default | Purpose |
|---|---|---|---|
| `CYMATIX_OTEL_ENABLED` | `enabled` | off | master switch (env: `1` = on) |
| `CYMATIX_OTEL_ENDPOINT` | `endpoint` | `localhost:4317` | OTLP gRPC endpoint |
| `CYMATIX_OTEL_INSECURE` | `insecure` | on | plain gRPC (local dev) |
| `CYMATIX_OTEL_SAMPLER_RATIO` | `sampler_ratio` | `1.0` | trace sampler 0.0–1.0 |
| `CYMATIX_OTEL_REDACT_QUERY` | `redact_query` | on | hash query strings (env: `0` = raw) |
| `CYMATIX_OTEL_LOGS_ENABLED` | `logs_enabled` | on | ship Python logs → collector → Loki |
| `CYMATIX_OTEL_LOGS_LEVEL` | `logs_level` | `INFO` | min log level forwarded |

The tray launcher exports `CYMATIX_OTEL_ENABLED=1` into the cymatix child's
environment after the observability stack is up and the collector's
OTLP port answers — that export is an env-layer value, so it beats the
shipped `[telemetry] enabled = false` default but never overrides an
explicit user `CYMATIX_OTEL_ENABLED`, and it does not touch
`CYMATIX_OTEL_ENDPOINT`.

## Dashboards

Seven dashboards under `deploy/otel/grafana/dashboards/`. All use
engineering vocabulary in panel titles; bio-domain legacy terms are
referenced inline in panel descriptions. See `docs/ROSETTA.md` for the
full bidirectional vocabulary table.

- **Cymatix — Operations Overview** (`cymatix-overview.json`) — default
  landing dashboard. Top-line operational KPIs: `/context` request
  rate, latency p50/p95/p99 by health, cache hit / miss / partial
  outcome, per-stage pipeline latency, compressor backend cost class
  + active model, knowledge-store size, WAL health, structured proxy
  log stream. Cross-links to the other dashboards.
- **Cymatix — Agent Usage** (`cymatix-agent-usage.json`) — `/context` call
  mix and latency bucketed by `caller_model_class`
  (`cymatix_context_calls_by_class_total`).
- **Cymatix — Pipeline Observatory** (`cymatix-pipeline-observatory.json`)
  — research panels reconciled to real instrument names in #209
  phase 1: tier activation share, per-signal retrieval latency, CWoLa
  bucket accumulation, p99 `/context` latency, co-activation edges by
  provenance, lifecycle tier distribution, hub concentration +
  inbound-degree stats, knowledge-store size.
- **Cymatix — Internals & Research** (`cymatix-internals.json`) — the #209
  phase-1 tuning signals: dense-cosine calibration distribution,
  shard-router fan-out + discrimination, know / miss / abstain
  decision mix, session-elision token savings, splice compression
  ratio by caller class. For deep-design work — not day-to-day
  operations.
- **Cymatix — GenAI** (`cymatix-genai.json`) — OTel `gen_ai.*` standard
  surface (#209): token throughput by type, TTFT quantiles per model,
  finish-reason mix, cost/hour + top spend by model, cache-outcome
  pie, and the `cymatix.proxy` structured log stream. Populated by
  `cymatix_context/telemetry/genai_telemetry.py`; a contract test
  (`tests/test_genai_telemetry.py::test_genai_dashboard_queries_are_covered`)
  keeps every panel query backed by an emitted metric.
- **Cymatix — Retrieval Quality + HITL** (`cymatix-retrieval-hitl.json`) —
  per-query ellipticity distribution, health-status pie, denatured-rate
  alert stat, budget-tier mix, ellipticity percentiles, HITL pause-event
  signals (alignment, override, escalation) with party-id breakdown.
  Uses technical-term vocabulary (ellipticity, denatured) that is on
  ROSETTA's "STAYS" list.
- **Cymatix — Know/Miss (Hallucination Visibility)** (`cymatix-know-miss.json`,
  #209 phase 2) — the G1 visibility-gate view: non-know share vs the 10%
  budget, miss rate by reason, know-confidence percentiles vs emit_floor,
  abstain fires by gate, freshness demotions, splice-ratio/abstain
  balancing pair, dense-cosine by arm, shard fan-out + discrimination,
  session-elision savings, latency row with load-annotation discipline.
  Overlaps cymatix-internals on the phase-1 series by design: internals is
  the tuning view, this is the goal-gates view. Companion spec:
  `docs/specs/2026-07-01-goal-gates-hallucination-visibility.md`.

The native install (`tools/native-otel/`) auto-syncs dashboards from
`deploy/otel/grafana/dashboards/` via
`cymatix_context.launcher.observability_render._wire_grafana_provisioning`.
After editing a dashboard JSON, re-run the launcher render step to
propagate, or restart the launcher.

## Verifying the stack

```bash
# Bring up the stack
cd deploy/otel && docker compose up -d

# In another shell, start cymatix with OTel on
export CYMATIX_OTEL_ENABLED=1
python -m uvicorn cymatix_context._asgi:app --port 11437

# Hit /context to emit spans + tier metrics
curl -s -X POST http://localhost:11437/context \
  -H "Content-Type: application/json" \
  -d '{"query":"what port does cymatix use","verbose":true}'

# Hit /stats to refresh chromatin + edge gauges
curl -s http://localhost:11437/stats > /dev/null

# Confirm Prometheus received metrics
curl -s 'http://localhost:9090/api/v1/query?query=cymatix_context_latency_seconds_count'
```

If the final `curl` returns `data.result[0].value`, metrics are flowing end-to-end.

## Troubleshooting

- **"OTel disabled" in logs** — set `CYMATIX_OTEL_ENABLED=1` in the cymatix server's env or `[telemetry] enabled = true` in cymatix.toml. If the server runs under the tray launcher this is exported automatically once the stack starts — check the launcher log for "Observability stack up" and make sure no stray `CYMATIX_OTEL_ENABLED=0` is set in the launcher's shell.
- **"OTel packages not installed"** — `pip install "cymatix-context[otel]"`.
- **Grafana shows no data** — check `http://localhost:9090/targets`; the `otel-collector` target should be `UP`. If it isn't, `docker compose logs otel-collector`.
- **Trace spans appear but metrics don't** — Prometheus remote-write endpoint needs `--web.enable-remote-write-receiver` (included in the provided `docker-compose.yml`). If running Prometheus outside the compose stack, add the flag.

## Shutting down

```bash
cd deploy/otel
docker compose down        # stop containers
docker compose down -v     # + discard volumes
```

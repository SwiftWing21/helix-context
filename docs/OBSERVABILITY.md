# Observability — OTel → Grafana stack

Helix emits OpenTelemetry traces + metrics when `HELIX_OTEL_ENABLED=1`. Everything upstream of the LLM answer boundary becomes visible on one dashboard.

## Quick start

**One-time setup** — bring up the observability stack:

```bash
cd deploy/otel
docker compose up -d
```

That starts:

| Service | Port | Purpose |
|---|---|---|
| OTel Collector | 4317 (gRPC), 4318 (HTTP) | receives OTLP from helix |
| Prometheus | 9090 | metrics storage |
| Tempo | 3200 | trace storage |
| Loki | 3100 | log storage |
| Grafana | 3000 | dashboards (`admin` / `admin`) |

**Install OTel client packages:**

```bash
pip install "helix-context[otel]"
```

**Enable on the helix server:**

```bash
export HELIX_OTEL_ENABLED=1
export HELIX_OTEL_ENDPOINT=localhost:4317   # default
python -m uvicorn helix_context.server:app --port 11437
```

Open <http://localhost:3000/d/helix-overview>. Retrieval latency, tier contributions, CWoLa f_gap, chromatin distribution, harmonic-edges-by-source — all live.

## What's instrumented

### Traces (`/context` span tree)

Auto-instrumentation via `opentelemetry-instrumentation-fastapi` wraps every route in a span. Attributes are the request path + status code. Sampling is controlled by `HELIX_OTEL_SAMPLER_RATIO` (default `1.0`; drop to `0.1` at high QPS).

### Metrics

| Metric | Type | Labels | Source |
|---|---|---|---|
| `helix_context_latency_seconds` | histogram | `health`, `budget_tier`, `cold_tier_used` | `/context` endpoint |
| `helix_tier_contribution` | histogram | `tier` | `query_genes` accumulation |
| `helix_tier_fired_total` | counter | `tier` | `query_genes` accumulation |
| `helix_cwola_bucket_total` | counter | `bucket` ∈ {A, B, pending} | `cwola.log_query` + `sweep_buckets` |
| `helix_cwola_f_gap_sq` | gauge | — | `cwola.sweep_buckets` |
| `helix_harmonic_edges_total` | gauge | `source` ∈ {seeded, co_retrieved, cwola_validated} | `/stats` snapshot |
| `helix_chromatin_state_total` | gauge | `state` ∈ {open, euchromatin, heterochromatin} | `/stats` snapshot |
| `helix_genome_size_bytes` | gauge | `kind` ∈ {raw, compressed} | `/stats` snapshot |

The gauges are refreshed each time `/stats` is hit. Prometheus scrapes every 15s — if nothing polls `/stats`, gauges stale. The `benchmark_monitor.py` or a cron scraping `/stats` keeps them fresh.

### Logs

Helix's `log.warning` / `log.debug` calls propagate to stdout; when running under the OTel SDK with a log handler configured, they flow to Loki tagged with trace context so you can pivot from a slow span to its logs.

## Privacy

Query text is hashed by default — spans carry `query=<first-50-chars>[hash:<12-hex>]`. Set `HELIX_OTEL_REDACT_QUERY=0` to store raw query strings (dev only; do not enable in shared deployments).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `HELIX_OTEL_ENABLED` | `0` | master switch |
| `HELIX_OTEL_ENDPOINT` | `localhost:4317` | OTLP gRPC endpoint |
| `HELIX_OTEL_INSECURE` | `1` | plain gRPC (local dev) |
| `HELIX_OTEL_SAMPLER_RATIO` | `1.0` | trace sampler 0.0–1.0 |
| `HELIX_OTEL_REDACT_QUERY` | `1` | hash query strings |

## What the dashboard shows

- **Retrieval row** — p50/p95/p99 `/context` latency split by health (`aligned` / `sparse` / `stale` / `denatured`), tier activation rate (which tiers are firing), per-tier contribution heatmap (score magnitude × tier). This is the live replacement for `bench_skill_activation.py`.
- **CWoLa Label Clock row** — bucket accumulation (A / B / pending) with a prominent f_gap_sq gauge. Red below 0.05, yellow 0.05–0.16, green ≥ 0.16. When it goes green, Sprint 3 PLR training is unblocked per `STATISTICAL_FUSION.md` §C2.
- **Graph & Chromatin row** — `harmonic_links` edge count by provenance (seeded vs co_retrieved vs cwola_validated) tracking the Sprint 4 Hebbian promotion cycle, and chromatin state pie (OPEN / EUCHROMATIN / HETEROCHROMATIN) tracking density-gate pressure over time.

## Verifying the stack

```bash
# Bring up the stack
cd deploy/otel && docker compose up -d

# In another shell, start helix with OTel on
export HELIX_OTEL_ENABLED=1
python -m uvicorn helix_context.server:app --port 11437

# Hit /context to emit spans + tier metrics
curl -s -X POST http://localhost:11437/context \
  -H "Content-Type: application/json" \
  -d '{"query":"what port does helix use","verbose":true}'

# Hit /stats to refresh chromatin + edge gauges
curl -s http://localhost:11437/stats > /dev/null

# Confirm Prometheus received metrics
curl -s 'http://localhost:9090/api/v1/query?query=helix_context_latency_seconds_count'
```

If the final `curl` returns `data.result[0].value`, metrics are flowing end-to-end.

## Troubleshooting

- **"OTel disabled" in logs** — set `HELIX_OTEL_ENABLED=1` in the helix server's env.
- **"OTel packages not installed"** — `pip install "helix-context[otel]"`.
- **Grafana shows no data** — check `http://localhost:9090/targets`; the `otel-collector` target should be `UP`. If it isn't, `docker compose logs otel-collector`.
- **Trace spans appear but metrics don't** — Prometheus remote-write endpoint needs `--web.enable-remote-write-receiver` (included in the provided `docker-compose.yml`). If running Prometheus outside the compose stack, add the flag.

## Shutting down

```bash
cd deploy/otel
docker compose down        # stop containers
docker compose down -v     # + discard volumes
```

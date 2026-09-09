# deploy/otel/ — Docker observability stack (advanced)

The default cymatix install ships native observability binaries managed
by the tray launcher. To set those up without bringing up the tray, run
[`scripts/setup-grafana-telem.ps1`](../../scripts/setup-grafana-telem.ps1)
(Windows) or [`scripts/setup-grafana-telem.sh`](../../scripts/setup-grafana-telem.sh)
(Linux / macOS).

This Docker Compose stack is the alternate path — useful for:

- Local containerized, declarative deployment
- Environments where native binaries don't fit (locked-down user dirs)
- Fallback testing against a known-good runtime

Both runtimes are first-class. Choose by deployment shape, not status.

## Components

| Service       | Image                                          | Port |
| ------------- | ---------------------------------------------- | ---- |
| OTel Collector| otel/opentelemetry-collector-contrib:0.105.0   | 4317, 4318, 8889 |
| Prometheus    | prom/prometheus:v2.54.1                        | 9090 |
| Tempo         | grafana/tempo:2.6.0                            | 3200 |
| Loki          | grafana/loki:3.2.0                             | 3100 |
| Grafana       | grafana/grafana:11.3.0                         | 3000 |

Wire format, ports, and dashboard provisioning are bit-for-bit
identical to the native sidecar — only the receiver runtime differs.

## Run

```bash
cd deploy/otel
docker-compose up -d
```

This is a **local-only template**: all published ports bind to `127.0.0.1`.
Grafana requires login; anonymous Viewer access is opt-in with
`CYMATIX_GRAFANA_ANON=true`. Set `CYMATIX_GRAFANA_ADMIN_PASSWORD` in your
environment before the first boot to override the local `admin` password.
The `admin` / `admin` fallback is only suitable for this loopback setup on a
trusted host. An existing Grafana data volume retains its password; rotate
it through Grafana instead of expecting the environment variable to reset it.

Loki and the other telemetry APIs have no authentication. Container peers
can reach them on the Compose network. Before allowing access from other
hosts, add authenticated TLS ingress, rotate credentials, keep the raw
service ports private, and restrict the network to trusted clients. See the
[hosted-session security review](../../docs/reviews/2026-09-01-hosted-session-security-review.md)
for the Proxmox and Ceph deployment controls. Query telemetry and logs can
contain sensitive text even when query redaction is enabled.

## Configs

- `otel-collector-config.yaml` — collector pipelines (used verbatim by Docker; templated for native).
- `prometheus.yml` — scrape config.
- `tempo.yaml` — Tempo storage + metrics-generator config.
- `loki-config.yaml` — explicit Loki config (mounted into the loki service).
- `grafana/provisioning/` — datasources + dashboard provisioning.
- `grafana/dashboards/` — committed dashboard JSON (runtime-agnostic).

The native sidecar (`tools/native-otel/`) reads these same files but
substitutes Docker-DNS hostnames → `localhost` and Linux container paths
→ per-user state dirs at install time. See
`docs/specs/2026-05-04-native-observability-sidecar-design.md` §6.3.

## Stop

```bash
docker-compose down
```

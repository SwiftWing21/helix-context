#!/usr/bin/env bash
# One-shot setup for the cymatix-context Grafana telemetry stack (native sidecar).
#
# Convenience wrapper around scripts/install-native-observability.sh that:
#   1. Verifies the [otel] + [launcher] extras are importable.
#   2. Downloads the pinned OTel collector + Prometheus + Tempo + Loki +
#      Grafana binaries into tools/native-otel/ (idempotent).
#   3. Renders runtime configs from deploy/otel/ for the native runtime.
#   4. Wires dashboards + provisioned datasources into Grafana's
#      conf/provisioning tree.
#   5. Smoke-tests Grafana (:3000) + Prometheus (:9090) if the supervisor
#      is already running.
#
# This script does NOT start the supervisor — it only prepares state.
# Run `cymatix-launcher --tray` (or start-cymatix-tray.bat on Windows) to
# spawn the five binaries; configs are reused across cymatix sessions.
#
# Spec: docs/specs/2026-05-04-native-observability-sidecar-design.md
#
# Flags:
#   --skip-download   skip binary download (binaries assumed present)
#   --verify-only     don't download or render; just smoke-test
#   --server-only     render configs only (no download, no smoke test)
#
# Exit codes:
#   0  success
#   1  setup error (missing extras, parse failure)
#   2  binary install failed
#   3  config render failed
#   4  smoke test failed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
fi

SKIP_DOWNLOAD=0
VERIFY_ONLY=0
SERVER_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --verify-only)   VERIFY_ONLY=1 ;;
        --server-only)   SERVER_ONLY=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) echo "[grafana-telem] Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

echo "[grafana-telem] Repo root: $REPO_ROOT"

test_extras() {
    echo "[grafana-telem] Verifying [otel] + [launcher] extras are importable..."
    if ! $PYTHON -c "import opentelemetry.sdk; import opentelemetry.exporter.otlp.proto.grpc; import jinja2; import psutil; import platformdirs" 2>/dev/null; then
        echo "[grafana-telem] Missing extras. Run:" >&2
        echo "    pip install -e \".[otel,launcher]\"" >&2
        return 1
    fi
    echo "[grafana-telem] Extras OK."
}

smoke_test() {
    echo "[grafana-telem] Smoke-testing endpoints..."
    local ok=0
    if curl -fsS --max-time 3 http://localhost:3000/api/health >/dev/null; then
        echo "  OK   Grafana      http://localhost:3000/api/health"
    else
        echo "  DOWN Grafana      http://localhost:3000/api/health"
        ok=1
    fi
    if curl -fsS --max-time 3 http://localhost:9090/-/healthy >/dev/null; then
        echo "  OK   Prometheus   http://localhost:9090/-/healthy"
    else
        echo "  DOWN Prometheus   http://localhost:9090/-/healthy"
        ok=1
    fi
    # Collector self-scrape is only present once the supervisor has run.
    # Absence on a fresh setup is expected, not an error.
    if curl -fsS --max-time 2 http://localhost:8889/metrics >/dev/null 2>&1; then
        echo "  OK   Collector    http://localhost:8889/metrics"
    else
        echo "  INFO Collector    not running (will be spawned by tray)"
    fi
    return $ok
}

if [ "$VERIFY_ONLY" -eq 1 ]; then
    test_extras
    smoke_test
    exit $?
fi

test_extras

# Step 1: binaries.
if [ "$SKIP_DOWNLOAD" -eq 0 ] && [ "$SERVER_ONLY" -eq 0 ]; then
    echo
    echo "[grafana-telem] Step 1/3: downloading native observability binaries..."
    install_script="$REPO_ROOT/scripts/install-native-observability.sh"
    if [ ! -x "$install_script" ]; then
        chmod +x "$install_script" 2>/dev/null || true
    fi
    if ! "$install_script"; then
        echo "[grafana-telem] Binary install failed" >&2
        exit 2
    fi
else
    echo
    echo "[grafana-telem] Step 1/3: skipped (binaries assumed present)"
    echo "[grafana-telem]   rendering configs explicitly..."
    if ! $PYTHON -m cymatix_context.launcher.observability_render render-all; then
        echo "[grafana-telem] Config render failed" >&2
        exit 3
    fi
fi

if [ "$SERVER_ONLY" -eq 1 ]; then
    echo
    echo "[grafana-telem] Step 2/3: rendering configs..."
    if ! $PYTHON -m cymatix_context.launcher.observability_render render-all; then
        echo "[grafana-telem] Config render failed" >&2
        exit 3
    fi
fi

echo
echo "[grafana-telem] Step 3/3: verifying stack reachability..."
smoke_test || true

cat <<'EOF'

== Grafana telemetry setup complete =====================
Dashboards:
  Overview     http://localhost:3000/d/cymatix-overview
  GenAI        http://localhost:3000/d/cymatix-genai
  Internals    http://localhost:3000/d/cymatix-internals
  Retrieval    http://localhost:3000/d/cymatix-retrieval-hitl

Defaults: admin / admin (set at first Grafana boot; rotate via UI).
Local trusted-host setup only. Before remote exposure, see docs/architecture/OBSERVABILITY.md.

To start the full stack (collector + Prom + Tempo + Loki + Grafana):
  cymatix-launcher --tray              # cross-platform
  start-cymatix-tray.bat               # Windows daily-driver wrapper

To enable telemetry on a headless backend:
  export CYMATIX_OTEL_ENABLED=1
  export CYMATIX_OTEL_ENDPOINT=localhost:4317
  python -m uvicorn cymatix_context._asgi:app --port 11437

Verify metrics are flowing (after first /context call):
  curl 'http://localhost:9090/api/v1/query?query=cymatix_context_latency_seconds_count'
EOF

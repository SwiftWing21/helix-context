<#
.SYNOPSIS
  One-shot setup for the cymatix-context Grafana telemetry stack (native sidecar).

.DESCRIPTION
  Convenience wrapper around scripts/install-native-observability.ps1 that:
    1. Verifies the [otel] + [launcher] extras are importable.
    2. Downloads the pinned OTel collector + Prometheus + Tempo + Loki +
       Grafana binaries into tools/native-otel/ (idempotent — re-runs
       skip components whose binary hash already matches).
    3. Renders runtime configs from deploy/otel/ into the rendered configs
       dir, substituting Docker DNS hostnames for localhost paths.
    4. Wires dashboard JSON + provisioned datasources into Grafana's
       conf/provisioning tree.
    5. Smoke-tests Grafana (:3000) + Prometheus (:9090) reachability if
       the supervisor is already running, otherwise prints next steps.

  This script does NOT start the supervisor — it only prepares the
  on-disk state so that `start-cymatix-tray.bat` (Windows) or the equivalent
  `cymatix-launcher --tray` invocation can spawn the five binaries.

  If you only want metrics and not the tray, call this script and then
  start cymatix with CYMATIX_OTEL_ENABLED=1 — the supervisor child binaries
  are reused across cymatix sessions and survive a backend restart.

.PARAMETER SkipDownload
  Skip the binary download step (use when binaries are already on disk).

.PARAMETER VerifyOnly
  Don't download or render — just smoke-test the running stack.

.PARAMETER ServerOnly
  Render configs only; skip binary download (useful for CI / build-time
  config validation).

.NOTES
  Spec: docs/specs/2026-05-04-native-observability-sidecar-design.md

  Exit codes:
    0  success (configs rendered, binaries present, smoke test passed
       if applicable)
    1  setup error (missing extras, .versions parse failure)
    2  binary install failed (propagated from install-native-observability)
    3  config render failed
    4  smoke test failed (Grafana or Prometheus unreachable)
#>

[CmdletBinding()]
param(
    [switch] $SkipDownload,
    [switch] $VerifyOnly,
    [switch] $ServerOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path -Parent $scriptPath
$RepoRoot = Split-Path -Parent $scriptDir
Write-Host "[grafana-telem] Repo root: $RepoRoot" -ForegroundColor Cyan

# Prefer venv python if present.
$python = "python"
if (Test-Path "$RepoRoot\.venv\Scripts\python.exe") {
    $python = "$RepoRoot\.venv\Scripts\python.exe"
}

function Test-Extras {
    Write-Host "[grafana-telem] Verifying [otel] + [launcher] extras are importable..." -ForegroundColor Cyan
    # PowerShell 5.1 with ErrorActionPreference=Stop turns ANY native-command
    # stderr into a script-terminating error. Python's import system writes
    # tracebacks to stderr on ImportError — so wrap the probe in a try/except
    # that silences stderr entirely and uses exit code to signal status.
    $probe = @'
import sys
try:
    import opentelemetry.sdk
    import opentelemetry.exporter.otlp.proto.grpc
    import jinja2
    import psutil
    import platformdirs
except Exception:
    sys.exit(1)
sys.exit(0)
'@
    & $python -c $probe
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        Write-Host "[grafana-telem] Missing extras. Run:" -ForegroundColor Red
        Write-Host "    pip install -e `".[otel,launcher]`"" -ForegroundColor Yellow
        return $false
    }
    Write-Host "[grafana-telem] Extras OK." -ForegroundColor Green
    return $true
}

function Invoke-SmokeTest {
    Write-Host "[grafana-telem] Smoke-testing endpoints..." -ForegroundColor Cyan
    $endpoints = @(
        @{ Name = "Grafana";    Url = "http://localhost:3000/api/health" },
        @{ Name = "Prometheus"; Url = "http://localhost:9090/-/healthy" }
    )
    $allOk = $true
    foreach ($e in $endpoints) {
        try {
            $r = Invoke-WebRequest -Uri $e.Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) {
                Write-Host ("  OK   {0,-12} {1}" -f $e.Name, $e.Url) -ForegroundColor Green
            } else {
                Write-Host ("  WARN {0,-12} {1} (HTTP {2})" -f $e.Name, $e.Url, $r.StatusCode) -ForegroundColor Yellow
                $allOk = $false
            }
        } catch {
            Write-Host ("  DOWN {0,-12} {1}" -f $e.Name, $e.Url) -ForegroundColor DarkYellow
            $allOk = $false
        }
    }
    # Optional: check the OTel collector self-scrape (only meaningful if
    # supervisor is already running; absent on a fresh setup).
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8889/metrics" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            Write-Host "  OK   Collector    http://localhost:8889/metrics" -ForegroundColor Green
        }
    } catch {
        Write-Host "  INFO Collector    not running (will be spawned by tray)" -ForegroundColor DarkGray
    }
    return $allOk
}

if ($VerifyOnly) {
    if (-not (Test-Extras)) { exit 1 }
    if (Invoke-SmokeTest) { exit 0 } else { exit 4 }
}

if (-not (Test-Extras)) { exit 1 }

# Step 1: download + extract binaries (idempotent).
if (-not $SkipDownload -and -not $ServerOnly) {
    Write-Host "`n[grafana-telem] Step 1/3: downloading native observability binaries..." -ForegroundColor Cyan
    $installScript = Join-Path $RepoRoot "scripts\install-native-observability.ps1"
    if (-not (Test-Path $installScript)) {
        Write-Host "[grafana-telem] Missing $installScript" -ForegroundColor Red
        exit 1
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $installScript -RepoRoot $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[grafana-telem] Binary install failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 2
    }
} else {
    Write-Host "`n[grafana-telem] Step 1/3: skipped (binaries assumed present)" -ForegroundColor DarkGray
    # install-native-observability already rendered configs as part of
    # its run; when we skip it we must still render explicitly so the
    # supervisor has fresh configs/datasources.yml to read.
    Write-Host "[grafana-telem]   rendering configs explicitly..." -ForegroundColor Cyan
    & $python -m cymatix_context.launcher.observability_render render-all
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[grafana-telem] Config render failed" -ForegroundColor Red
        exit 3
    }
}

# Step 2: re-render configs to pick up any edits to deploy/otel/ since
# binaries were installed. observability_render is idempotent + cheap
# (pure string substitution, six files). Skipped when install-native
# already ran above, since that script renders as its final step.
if ($ServerOnly) {
    Write-Host "`n[grafana-telem] Step 2/3: rendering configs..." -ForegroundColor Cyan
    & $python -m cymatix_context.launcher.observability_render render-all
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[grafana-telem] Config render failed" -ForegroundColor Red
        exit 3
    }
}

# Step 3: report next steps + smoke-test.
Write-Host "`n[grafana-telem] Step 3/3: verifying stack reachability..." -ForegroundColor Cyan
[void](Invoke-SmokeTest)

Write-Host "`n== Grafana telemetry setup complete =====================" -ForegroundColor Cyan
Write-Host "Dashboards:"
Write-Host "  Overview     http://localhost:3000/d/cymatix-overview"
Write-Host "  GenAI        http://localhost:3000/d/cymatix-genai"
Write-Host "  Internals    http://localhost:3000/d/cymatix-internals"
Write-Host "  Retrieval    http://localhost:3000/d/cymatix-retrieval-hitl"
Write-Host ""
Write-Host "Defaults: admin / admin (set at first Grafana boot; rotate via UI)." -ForegroundColor Gray
Write-Host "Local trusted-host setup only. Before remote exposure, see docs/architecture/OBSERVABILITY.md." -ForegroundColor Gray
Write-Host ""
Write-Host "To start the full stack (collector + Prom + Tempo + Loki + Grafana):"
Write-Host "  start-cymatix-tray.bat               # daily driver (Windows)"
Write-Host "  cymatix-launcher --tray              # cross-platform"
Write-Host ""
Write-Host "To enable telemetry on a headless backend:"
Write-Host "  `$env:CYMATIX_OTEL_ENABLED='1'; `$env:CYMATIX_OTEL_ENDPOINT='localhost:4317'"
Write-Host "  python -m uvicorn cymatix_context._asgi:app --port 11437"
Write-Host ""
Write-Host "Verify metrics are flowing (after first /context call):"
Write-Host "  curl http://localhost:9090/api/v1/query?query=cymatix_context_latency_seconds_count"

exit 0

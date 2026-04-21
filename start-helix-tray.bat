@echo off
REM ─────────────────────────────────────────────────────────────────
REM Helix tray launcher — double-click or pin to taskbar for 1-click start.
REM
REM This batch file sets the OTel + federation env vars and starts the
REM launcher in --tray mode, with Grafana + Prometheus links in the
REM right-click menu. Close the tray icon (Quit) to stop both the
REM launcher and helix.
REM
REM To customize: edit this file or create start-helix-tray.local.bat
REM (gitignored) alongside it and invoke via cmd /k.
REM ─────────────────────────────────────────────────────────────────

cd /d "%~dp0"

REM ── OpenTelemetry (optional — remove if you don't want metrics) ──
set "HELIX_OTEL_ENABLED=1"
set "HELIX_OTEL_ENDPOINT=localhost:4317"
set "HELIX_OTEL_INSECURE=1"
set "HELIX_OTEL_SAMPLER_RATIO=1.0"

REM ── Budget-zone gene-cap spike (2026-04-14) ─────────────────────
REM When set, the expression pipeline clamps max_genes based on the
REM caller's incoming prompt token count so big prompts don't get the
REM full BROAD tier when the caller is already near their window.
REM Zones (at 128k window): <25% none, 25-40% cap 12, 40-60% cap 6,
REM 60-80% cap 3, 80%+ cap 1. Harmless when unset. See budget_zone.py.
set "HELIX_BUDGET_ZONE=1"

REM ── 4-layer federation attribution (edit to your handle) ────────
REM HELIX_AGENT is the persona writing genes. If unset, ingests tag
REM as "manual / no AI persona involved." Set per shell/shortcut for
REM per-persona tagging (Laude/Taude/Raude each pin their own .bat).
if "%HELIX_USER%"=="" set "HELIX_USER=max"
REM set "HELIX_AGENT=raude"   REM uncomment + edit if you want persona tagging

REM ── Headroom proxy (OPTIONAL — requires helix-context[codec]) ───
REM When enabled, the launcher adopts or spawns a headroom proxy and
REM adds "Open Headroom Dashboard" + Start/Restart/Stop Headroom to
REM the tray menu. The launcher gracefully adopts an already-running
REM headroom proxy on the configured port (8787 by default) rather
REM than spawning a duplicate — adopted processes survive Quit.
REM
REM Master switch (overrides [headroom] enabled in helix.toml):
REM   set HELIX_HEADROOM_ENABLED=1
REM
REM Autostart (spawn if nothing is already running):
REM   set HELIX_HEADROOM_AUTOSTART=1
REM
REM See helix.toml [headroom] for host/port/mode configuration.
set "HELIX_HEADROOM_ENABLED=1"
set "HELIX_HEADROOM_AUTOSTART=1"

REM Auto-route Helix's OpenAI-compatible chat upstream through Headroom
REM only when the configured upstream is non-local. Local Ollama stays
REM direct; remote providers can benefit from Headroom's proxy layer.
set "HELIX_HEADROOM_ROUTE_UPSTREAM_AUTO=1"

REM ── Launch the tray ─────────────────────────────────────────────
start "helix-launcher" /B python -m helix_context.launcher.app ^
  --tray ^
  --grafana-url "http://localhost:3000/d/helix-overview/helix-overview" ^
  --prometheus-url "http://localhost:9090/graph"

REM /B = no new window. The launcher takes over stdout+stderr; the
REM tray icon is the persistent surface. Closing this cmd window does
REM NOT stop helix — only Quit from the tray menu does.

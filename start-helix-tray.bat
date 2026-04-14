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
set HELIX_OTEL_ENABLED=1
set HELIX_OTEL_ENDPOINT=localhost:4317
set HELIX_OTEL_INSECURE=1
set HELIX_OTEL_SAMPLER_RATIO=1.0

REM ── 4-layer federation attribution (edit to your handle) ────────
REM HELIX_AGENT is the persona writing genes. If unset, ingests tag
REM as "manual / no AI persona involved." Set per shell/shortcut for
REM per-persona tagging (Laude/Taude/Raude each pin their own .bat).
if "%HELIX_USER%"=="" set HELIX_USER=max
REM set HELIX_AGENT=raude   REM uncomment + edit if you want persona tagging

REM ── Launch the tray ─────────────────────────────────────────────
start "helix-launcher" /B python -m helix_context.launcher.app ^
  --tray ^
  --grafana-url "http://localhost:3000/d/helix-overview/helix-overview" ^
  --prometheus-url "http://localhost:9090/graph"

REM /B = no new window. The launcher takes over stdout+stderr; the
REM tray icon is the persistent surface. Closing this cmd window does
REM NOT stop helix — only Quit from the tray menu does.

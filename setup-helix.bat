@echo off
REM ─────────────────────────────────────────────────────────────────
REM Helix 1-click setup (Windows).
REM
REM Double-click this file from a fresh clone to install deps + create
REM desktop / start menu shortcuts + (optionally) bring up Grafana.
REM
REM Passes all args through to setup-helix.ps1. Useful flags:
REM   -WithObservability   also docker-compose up Grafana+Prometheus+OTel
REM   -NoShortcuts         headless install (no .lnk creation)
REM   -SkipPipInstall      just refresh shortcuts without reinstalling
REM
REM Example:
REM   setup-helix.bat -WithObservability
REM ─────────────────────────────────────────────────────────────────

cd /d "%~dp0"

where /q powershell
if errorlevel 1 (
    echo [setup-helix] PowerShell not found on PATH — required for shortcut creation.
    echo Install PowerShell 5+ or run deploy\windows\setup-helix.ps1 manually.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "deploy\windows\setup-helix.ps1" %*

echo.
pause

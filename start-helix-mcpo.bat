@echo off
REM ─────────────────────────────────────────────────────────────────
REM mcpo launcher — exposes helix MCP (stdio) as an OpenAPI server
REM so Open WebUI (and any other OpenAPI-consuming frontend) can call
REM helix tools from an Ollama chat session.
REM
REM Flow:
REM   Open WebUI  ──OpenAPI──▶  mcpo :8788  ──stdio MCP──▶  python -m helix_context.mcp_server
REM                                                          └──HTTP──▶  helix FastAPI :11437
REM
REM Prereqs:
REM   1. Helix FastAPI must be running on :11437 (start-helix-tray.bat
REM      or backend-with-otel.bat). This script waits for it.
REM   2. `pip install mcpo` in the same Python env that runs helix.
REM
REM To customize: copy to start-helix-mcpo.local.bat (gitignored) and
REM edit there — port, agent identity, log verbosity.
REM ─────────────────────────────────────────────────────────────────

cd /d "%~dp0"

REM ── mcpo port (Open WebUI registers this as an OpenAPI server) ──
if "%HELIX_MCPO_PORT%"=="" set HELIX_MCPO_PORT=8788

REM ── Helix upstream the MCP shim talks to ────────────────────────
if "%HELIX_MCP_URL%"=="" set HELIX_MCP_URL=http://127.0.0.1:11437

REM ── 4-layer federation identity — distinct from Claude Code's ──
REM Collision guard: if HELIX_AGENT is empty OR equals "laude", force
REM "openwebui" so this MCPO session doesn't merge with Claude Code's.
if "%HELIX_ORG%"==""        set HELIX_ORG=swiftwing
REM HELIX_PARTY_ID / HELIX_DEVICE identify this machine in CWoLa + session registry.
REM Change to your own party id (operator's preferred stable identifier).
if not defined HELIX_PARTY_ID set "HELIX_PARTY_ID=%COMPUTERNAME%"
if not defined HELIX_DEVICE set "HELIX_DEVICE=%COMPUTERNAME%"
if "%HELIX_USER%"==""       set HELIX_USER=max
if "%HELIX_AGENT%"=="laude" set HELIX_AGENT=openwebui
if "%HELIX_AGENT%"==""      set HELIX_AGENT=openwebui
if "%HELIX_AGENT_KIND%"=="" set HELIX_AGENT_KIND=ollama-chat
if "%HELIX_MCP_HANDLE%"=="" set HELIX_MCP_HANDLE=%HELIX_AGENT%
if "%HELIX_MCP_HOST%"==""   set HELIX_MCP_HOST=ollama-chat

REM ── Wait for helix :11437 to answer /health (up to ~60s) ────────
echo [mcpo] waiting for helix at %HELIX_MCP_URL% ...
set /a _tries=0
:wait_helix
curl.exe -s -f -o NUL --max-time 2 "%HELIX_MCP_URL%/health" && goto helix_ready
set /a _tries+=1
if %_tries% GEQ 30 (
  echo [mcpo] helix did not answer after 60s — start it with start-helix-tray.bat first.
  exit /b 1
)
timeout /t 2 /nobreak >NUL
goto wait_helix
:helix_ready
echo [mcpo] helix is up. Launching mcpo on :%HELIX_MCPO_PORT% as agent=%HELIX_AGENT%

REM ── Launch mcpo wrapping the stdio helix MCP ────────────────────
REM mcpo re-execs the inner command on every restart; env vars above
REM propagate to the child python process.
mcpo --port %HELIX_MCPO_PORT% -- python -m helix_context.mcp_server

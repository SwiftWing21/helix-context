@echo off
REM Wrapper for launch.json — sets OTel env vars, then runs the backend directly.
cd /d "%~dp0"
set "HELIX_OTEL_ENABLED=1"
set "HELIX_OTEL_ENDPOINT=localhost:4317"
set "HELIX_OTEL_INSECURE=1"
set "HELIX_OTEL_SAMPLER_RATIO=1.0"
set "HELIX_USER=max"
set "HELIX_AGENT=raude"
python -m uvicorn helix_context.server:app --host 127.0.0.1 --port 11437

#Requires -Version 5.0
<#
    setup-helix.ps1 -- one-shot setup for helix-context on Windows.

    What this does:
      1. Verifies Python 3.11+ is on PATH
      2. Installs helix-context in editable mode with the right optional
         extras for a full desktop deployment (tray, OTel, MCP)
      3. Creates Desktop + Start Menu shortcuts pointing at
         start-helix-tray.bat so the user gets a real 1-click launcher
      4. Optionally stands up the Grafana + Prometheus + OTel collector
         stack via docker compose (-WithObservability)

    Usage:
      # From the repo root:
      powershell -ExecutionPolicy Bypass -File deploy\windows\setup-helix.ps1

      # With observability stack:
      powershell -ExecutionPolicy Bypass -File deploy\windows\setup-helix.ps1 -WithObservability

      # Skip shortcut creation (e.g. headless install):
      powershell -ExecutionPolicy Bypass -File deploy\windows\setup-helix.ps1 -NoShortcuts

    This script is idempotent -- re-running it refreshes deps and
    overwrites shortcuts without harm. It does NOT start helix; it
    only prepares the machine so a click on the new shortcut can.
#>

[CmdletBinding()]
param(
    [switch] $WithObservability,
    [switch] $NoShortcuts,
    [switch] $SkipPipInstall
)

$ErrorActionPreference = "Stop"

# == Resolve repo root (two dirs up from this script) ================
$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
Write-Host "helix repo: $RepoRoot" -ForegroundColor Cyan
Write-Host "flags:      WithObservability=$WithObservability  NoShortcuts=$NoShortcuts  SkipPipInstall=$SkipPipInstall" -ForegroundColor DarkGray

# == Python check ====================================================
function Test-Python {
    try {
        $v = & python --version 2>&1
        if ($v -match "Python\s+(\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
                throw "Python 3.11+ required, found $v"
            }
            Write-Host "Python OK: $v" -ForegroundColor Green
            return $true
        }
        throw "Could not parse Python version: $v"
    } catch {
        Write-Host "Python check failed: $_" -ForegroundColor Red
        Write-Host "Install Python 3.11+ from https://www.python.org/downloads/ and rerun."
        return $false
    }
}

if (-not (Test-Python)) { exit 1 }

# == pip install with extras ========================================─
if (-not $SkipPipInstall) {
    Write-Host "`nInstalling helix-context + extras (this can take a few minutes)..." -ForegroundColor Cyan
    Push-Location $RepoRoot
    try {
        # Core package + the extras needed for tray, OTel, MCP, SemaCodec.
        # Quoted because .[a,b] syntax gets mangled by PowerShell arg parsing.
        & python -m pip install --upgrade pip
        & python -m pip install -e ".[otel,launcher-tray,accel,embeddings,cpu]"
        # MCP SDK is a separate top-level package, not an extra.
        & python -m pip install "mcp>=1.0"
        if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
        Write-Host "Dependencies installed." -ForegroundColor Green
    } finally {
        Pop-Location
    }
}

# == Shortcut creation ==============================================─
function New-HelixShortcut {
    param(
        [string] $LnkPath,
        [string] $TargetBat,
        [string] $WorkingDir,
        [string] $IconPath,
        [string] $Description
    )
    # Pointedly-verbose error handling: the silent-failure variant of
    # this function was the reason shortcuts didn't appear on the first
    # Windows deploy. Every failure path now prints exactly what broke.
    Write-Host "  target:    $LnkPath" -ForegroundColor DarkGray

    # Ensure the parent dir exists (GetFolderPath returns the path even
    # if the folder was customized away; unusual but survivable).
    $parent = Split-Path -Parent $LnkPath
    if (-not (Test-Path $parent)) {
        Write-Host "  FAIL:      parent dir missing: $parent" -ForegroundColor Red
        return $false
    }

    try {
        $shell = New-Object -ComObject WScript.Shell -ErrorAction Stop
    } catch {
        Write-Host "  FAIL:      could not create WScript.Shell COM object: $_" -ForegroundColor Red
        return $false
    }

    try {
        $lnk = $shell.CreateShortcut($LnkPath)
        $lnk.TargetPath = $TargetBat
        $lnk.WorkingDirectory = $WorkingDir
        $lnk.WindowStyle = 7    # 7 = minimized (the tray icon is the UI, not the cmd window)
        $lnk.Description = $Description
        if ($IconPath -and (Test-Path $IconPath)) {
            $lnk.IconLocation = $IconPath
        }
        $lnk.Save()
    } catch {
        Write-Host "  FAIL:      shortcut save threw: $_" -ForegroundColor Red
        return $false
    }

    if (Test-Path $LnkPath) {
        Write-Host "  OK:        $LnkPath" -ForegroundColor Green
        return $true
    }
    Write-Host "  FAIL:      .lnk did not appear on disk after Save()" -ForegroundColor Red
    return $false
}

if (-not $NoShortcuts) {
    Write-Host "`nCreating shortcuts..." -ForegroundColor Cyan
    $targetBat = Join-Path $RepoRoot "start-helix-tray.bat"
    if (-not (Test-Path $targetBat)) {
        Write-Host "  WARN: $targetBat not found -- skipping shortcuts" -ForegroundColor Yellow
    } else {
        # Icon: fall back to python.exe if no helix .ico is bundled yet.
        $iconCandidates = @(
            (Join-Path $RepoRoot "helix_context\launcher\static\helix.ico"),
            (Join-Path $RepoRoot "deploy\windows\helix.ico")
        )
        $icon = $iconCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

        $desktop = [Environment]::GetFolderPath("Desktop")
        $startMenuDir = [Environment]::GetFolderPath("Programs")
        Write-Host "  desktop:   $desktop" -ForegroundColor DarkGray
        Write-Host "  startmenu: $startMenuDir" -ForegroundColor DarkGray

        $desktopLnk = Join-Path $desktop "Helix.lnk"
        $startMenuLnk = Join-Path $startMenuDir "Helix.lnk"

        [void](New-HelixShortcut -LnkPath $desktopLnk -TargetBat $targetBat `
            -WorkingDir $RepoRoot -IconPath $icon `
            -Description "Start helix-context FastAPI server with system tray")
        [void](New-HelixShortcut -LnkPath $startMenuLnk -TargetBat $targetBat `
            -WorkingDir $RepoRoot -IconPath $icon `
            -Description "Start helix-context FastAPI server with system tray")
    }
}

# == Observability stack (optional) ==================================
if ($WithObservability) {
    Write-Host "`nStarting observability stack via docker compose..." -ForegroundColor Cyan
    $composeFile = Join-Path $RepoRoot "deploy\otel\docker-compose.yml"
    if (-not (Test-Path $composeFile)) {
        Write-Host "  $composeFile not found -- skipping" -ForegroundColor Yellow
    } else {
        try {
            & docker compose -f $composeFile up -d
            if ($LASTEXITCODE -ne 0) { throw "docker compose exited $LASTEXITCODE" }
            Write-Host "Observability stack up:" -ForegroundColor Green
            Write-Host "  Grafana     http://localhost:3000  (admin / admin)"
            Write-Host "  Prometheus  http://localhost:9090"
            Write-Host "  OTel coll.  localhost:4317 (gRPC)"
        } catch {
            Write-Host "  docker compose failed: $_" -ForegroundColor Yellow
            Write-Host "  Install Docker Desktop (https://www.docker.com/products/docker-desktop/) and rerun with -WithObservability"
        }
    }
}

# == Next-step summary ==============================================─
Write-Host "`n== Setup complete ==========================================" -ForegroundColor Cyan
Write-Host "1-click launcher:"
Write-Host "  Desktop    -> double-click 'Helix'"
Write-Host "  Start Menu -> search 'Helix'"
Write-Host "  Taskbar    -> right-click the Helix shortcut -> Pin to taskbar"
Write-Host ""
Write-Host "Customize env vars / persona in: $RepoRoot\start-helix-tray.bat"
Write-Host "  (or copy it to start-helix-tray-<persona>.bat for per-persona tagging)"
Write-Host ""
if (-not $WithObservability) {
    Write-Host "Optional observability stack (Grafana + Prometheus + OTel):"
    Write-Host "  rerun with -WithObservability"
    Write-Host ""
}
Write-Host "To remove: delete the .lnk files above + 'pip uninstall helix-context'." -ForegroundColor Gray

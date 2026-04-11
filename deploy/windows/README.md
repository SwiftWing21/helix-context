# Helix Launcher as a Windows Service

Windows doesn't ship with a clean Python service wrapper, so the
recommended path is **NSSM** (Non-Sucking Service Manager) — a tiny
~1 MB executable that adapts any console program into a proper Windows
service with restart-on-failure semantics.

This directory intentionally does **not** ship a service installer.
Instead, here's the recipe a user can follow once. A future
`helix-launcher install-service` CLI command may automate this.

## One-time setup

### 1. Install Python + helix-context with the launcher extra

```cmd
pip install helix-context[launcher]
where helix-launcher
```

Note the absolute path that `where` prints — you'll need it below.
Typical paths:

- `C:\Users\<you>\AppData\Local\Programs\Python\Python311\Scripts\helix-launcher.exe`
- `C:\Users\<you>\.venvs\helix\Scripts\helix-launcher.exe`

### 2. Install NSSM

Download from [https://nssm.cc](https://nssm.cc) or install via
chocolatey / scoop:

```cmd
choco install nssm
:: or
scoop install nssm
```

Verify:

```cmd
nssm --version
```

### 3. Register the service

Run an **elevated** command prompt (Run as administrator):

```cmd
nssm install HelixLauncher
```

NSSM will open a small GUI. Fill in:

| Field | Value |
|---|---|
| **Path** | The full path to `helix-launcher.exe` from step 1 |
| **Startup directory** | Leave blank or set to your venv `Scripts` folder |
| **Arguments** | `--no-browser` |

On the **Details** tab:

- **Display name:** `Helix Launcher`
- **Description:** `Helix Context supervisor + dashboard`
- **Startup type:** `Automatic` (or `Automatic (Delayed Start)` if you
  want it to wait for slower boot dependencies)

On the **I/O** tab (optional but recommended):

- **Output (stdout):** `C:\ProgramData\HelixLauncher\helix-launcher.log`
- **Error (stderr):** `C:\ProgramData\HelixLauncher\helix-launcher.err.log`

Click **Install service**.

### 4. Start it

```cmd
nssm start HelixLauncher
```

Or via the standard Services app (`services.msc`).

### 5. Verify

Open `http://127.0.0.1:11438/` in your browser. You should see the
launcher dashboard.

## Common operations

```cmd
:: Stop
nssm stop HelixLauncher

:: Restart
nssm restart HelixLauncher

:: Check status
nssm status HelixLauncher
sc query HelixLauncher

:: Uninstall
nssm stop HelixLauncher
nssm remove HelixLauncher confirm
```

## Why not `pywin32` or `pyinstaller`?

- **`pywin32`** can register a Python class as a Win32 service, but it
  pulls a heavy dependency into the helix-context project for a feature
  that only matters at install time. NSSM keeps the service shim
  external and OS-native.
- **`pyinstaller`** would let us ship a single `.exe`, but adds a build
  step and bloats releases by ~30 MB. Not worth it for a launcher
  binary.

If a future `helix-launcher install-service` CLI is added, it will
likely shell out to NSSM (downloading it on demand if missing) rather
than reimplement the service shim from scratch.

## Troubleshooting

**Service starts then immediately stops:**
Check `helix-launcher.err.log`. The most common cause is the venv path
being wrong, or `pip install helix-context[launcher]` not being run in
the same Python environment NSSM is invoking.

**Port 11438 in use:**
Pass `--port <other>` in NSSM's Arguments field. Reach the dashboard at
`http://127.0.0.1:<other>/` afterwards.

**Helix child process refuses to start:**
This usually means port 11437 is already in use by another helix
instance. Stop the other one, or pass `--helix-port <other>` in NSSM's
Arguments.

**Service is registered but `helix-launcher` isn't found:**
Re-check the absolute path on the **Application** tab. NSSM doesn't
inherit `PATH` from your interactive shell — you need the full path
to the `.exe`.

# Helix Launcher — Supervisor Process + Control UI

A standalone supervisor + dashboard that solves the "who starts helix"
problem and gives a live view of what's actually running.

**Status:** Design spec — not yet implemented.
**Target version:** `helix-context v0.4.0b2` (TBD — follows session registry).
**Depends on:** session registry (for party/participant counts on the dashboard).
**Related:** [`SESSION_REGISTRY.md`](SESSION_REGISTRY.md), [`RESTART_PROTOCOL.md`](RESTART_PROTOCOL.md).

---

## Motivation

Today, running a helix-context server means typing `python -m uvicorn
helix_context.server:app --host 127.0.0.1 --port 11437` into a terminal
and hoping nothing kills the shell. The gaps:

1. **No supervisor.** If the helix process dies — OOM, crash, deliberate
   kill — nothing brings it back. Empirically confirmed 2026-04-10:
   the live server at `:11437` was a backgrounded bash subprocess
   started by one Claude Code session, and would have died with that
   session's shell.
2. **No lifecycle controls.** Restarting helix (for code changes, model
   swaps, config reloads) requires manual process management.
3. **No live status surface.** `GET /stats` and `GET /health` exist, but
   you have to curl them. There is no at-a-glance view of what's loaded,
   who's connected, and what tools are active.
4. **Non-Claude users are left without a story.** Anyone using
   helix-context without a Claude Code agent to orchestrate the process
   has no straightforward "launch the thing" experience.

The launcher solves all four with one small supervisor process.

## Non-goals (to keep scope tight)

- Not a service/daemon installer. The launcher is a foreground app you
  run manually. Installing as a Windows service / systemd unit / macOS
  LaunchAgent is a follow-up.
- Not an observability dashboard. No charts, no time-series graphs, no
  historical drill-down. Only current state.
- Not an admin console. Admin actions like vacuum, consolidate, and
  ribosome pause stay on the existing `/admin/*` endpoints and are not
  exposed through the launcher UI.
- Not a config editor. `helix.toml` stays file-edited; launcher only
  reads it.
- Not a multi-helix manager. The launcher manages exactly one helix
  child process. Federation / multi-instance management is out of scope.

## Architecture

Two processes, two ports:

```
┌─────────────────────────────────────┐
│  helix-launcher      :11438         │  ← supervisor + UI (long-lived)
│  - FastAPI + Jinja templates        │
│  - HTMX polling                     │
│  - subprocess.Popen for helix       │
│  - psutil for liveness / cleanup    │
└────────────┬────────────────────────┘
             │
             │ spawns / monitors
             ▼
┌─────────────────────────────────────┐
│  helix-context      :11437          │  ← the actual server (restartable)
│  - unchanged                         │
└─────────────────────────────────────┘
```

The launcher is a separate process with its own lifecycle. When you
close the launcher, helix stops cleanly. When you click **Restart**, the
launcher announces the restart via `POST /admin/announce_restart`, waits
750ms, terminates the child, and spawns a fresh one.

The launcher NEVER imports `helix_context.server` directly. It talks to
helix exclusively over HTTP at `http://127.0.0.1:11437`. This keeps the
launcher's own process memory light (no genome load, no model) and means
the launcher keeps working when helix is stopped.

## Tech stack

No new heavy dependencies. Everything is a small addition under a new
optional extras group.

| Layer | Choice | New dep? | Rationale |
|---|---|---|---|
| Backend | FastAPI | already a core dep | Match the existing project stack |
| Templates | Jinja2 | **new** (~350KB) | Declarative HTML, no layout math in Python |
| Reactivity | HTMX (served locally) | **new** (~14KB single JS file, vendored) | Polls endpoints, no build step |
| Styling | CSS custom properties | none | All visual tokens in `:root {}` |
| Process control | `subprocess` + `psutil` | `psutil` **new** (~1MB) | Cross-platform liveness, kill trees |
| Window (optional) | `pywebview` | **new, optional** | Native window wrapper — opt-in only |

New optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
launcher = ["jinja2>=3.1", "psutil>=5.9"]
launcher-native = ["jinja2>=3.1", "psutil>=5.9", "pywebview>=5.0"]
```

Users run `pip install helix-context[launcher]` for browser-based UI, or
`pip install helix-context[launcher-native]` for a native window.

## No hardcoded UI — what this means in practice

The user's explicit requirement: no Tkinter, no hardcoded UIs. Translated
into concrete rules for this codebase:

1. **No layout math in Python.** Widths, heights, margins, padding,
   grid tracks — all live in CSS, never in Python.
2. **No colors in Python.** Every color is a CSS custom property
   defined in one `:root {}` block. Python never touches pixel values.
3. **No imperative draw calls.** No `canvas.draw_rect`, no
   `immediate_mode.button`. The UI is declarative HTML rendered from
   Jinja templates. Python produces data; templates produce markup.
4. **Components are named and reusable.** Each panel is its own
   template file (`templates/components/*.html`), included from the
   main dashboard. Changing one panel's layout is one file edit.
5. **Theming happens in one place.** Switching to a dark theme means
   flipping CSS variables in one block, not hunting through Python code.

If a future contributor is tempted to write `x=5, y=10, width=200` in
Python, they are doing it wrong.

## Project layout

```
helix_context/
  launcher/
    __init__.py
    app.py              # FastAPI app factory + CLI entry point (main())
    supervisor.py       # helix subprocess lifecycle (Start/Restart/Stop)
    state.py            # ~/.helix/launcher/state.json read/write + adoption
    models.py           # Pydantic models for launcher state + API responses
    templates/
      layout.html       # base template (head, htmx, css link, body shell)
      dashboard.html    # extends layout.html — full dashboard
      components/
        controls.html   # start/restart/stop buttons
        status_banner.html
        parties_panel.html
        participants_panel.html
        models_panel.html
        tools_panel.html
        genes_panel.html
        tokens_panel.html
    static/
      launcher.css      # one :root{} block + component classes
      htmx.min.js       # vendored, so no CDN fetch at runtime
```

## Entry point

New console script in `pyproject.toml`:

```toml
[project.scripts]
helix = "helix_context.server:main"            # unchanged
helix-launcher = "helix_context.launcher.app:main"   # NEW
```

Usage:

```bash
# Browser mode (default)
helix-launcher

# System tray icon — persistent, "close to tray" experience
# (requires [launcher-tray] extra)
helix-launcher --tray

# Close-to-tray with native window (Windows only — see platform notes)
helix-launcher --tray --native

# Native window (requires [launcher-native] extra)
helix-launcher --native

# Install as a system service (systemd / launchd / NSSM recipe)
helix-launcher install-service
helix-launcher uninstall-service
helix-launcher install-service --dry-run

# Don't auto-start helix; just show the UI with a Start button
helix-launcher --no-autostart

# Use a non-default helix port
helix-launcher --helix-port 11439

# Use a non-default launcher port
helix-launcher --port 11438
```

## Launcher REST API (on :11438)

All endpoints are localhost-only. No authentication (the launcher binds
to 127.0.0.1 exclusively).

### `GET /`

Renders the dashboard HTML. This is the only HTML endpoint — all other
data flows as JSON through the endpoints below, pulled by HTMX.

### `GET /api/state`

The single source of truth for the HTMX poll. Returns the full
launcher-view-of-the-world as one JSON document. HTMX polls this every
2 seconds and replaces dashboard panels with server-rendered partials.

Response:

```json
{
  "helix": {
    "running": true,
    "pid": 56792,
    "port": 11437,
    "uptime_s": 412.5,
    "version": "0.4.0b2",
    "last_restart_reason": "session registry DAL fix",
    "last_restart_at": 1775881217.9
  },
  "parties": {
    "count": 1,
    "party_ids": ["max@local"]
  },
  "participants": {
    "count": 3,
    "handles": ["taude", "laude", "raude"]
  },
  "models": {
    "loaded": [
      {"name": "gemma4:e4b", "size_mb": 4200, "source": "ollama"}
    ]
  },
  "tools": [
    {"name": "ribosome", "kind": "decoder", "status": "running"},
    {"name": "cpu_tagger", "kind": "encoder", "status": "idle"},
    {"name": "splade", "kind": "encoder", "status": "running"},
    {"name": "sema", "kind": "encoder", "status": "idle"}
  ],
  "genes": {
    "raw_chars": 47086009,
    "compressed_chars": 17516283,
    "compression_ratio": 2.69,
    "total": 8107
  },
  "tokens": {
    "session": 184293,
    "lifetime": 1_734_551,
    "tracked": true
  }
}
```

Panels whose underlying data is empty (e.g., `tools` empty, `models`
empty, `parties.count == 0`) are simply absent from the response. The
dashboard then conditionally renders them.

### `POST /api/control/start`

Start the helix child if not running. Returns 200 with new state on
success, 409 if already running.

### `POST /api/control/stop`

Announce via helix `/admin/announce_restart`, wait 750ms, terminate the
process tree, wait for port 11437 to be free. Returns 200 when the
process is fully down, 408 if port never releases within 10s.

### `POST /api/control/restart`

Combined stop + start in sequence. Announces with `reason="manual
restart from launcher"`. Returns 200 when new helix answers `GET /stats`.

## Data sources (how the launcher populates `/api/state`)

Every field comes from an existing endpoint (or a trivial extension of
one). No new endpoints on the helix side are strictly required for the
first launcher slice, with two exceptions flagged below.

| State field | Source |
|---|---|
| `helix.running` | `psutil` check on stored PID + `GET /stats` reachability |
| `helix.pid` | launcher state file |
| `helix.port` | launcher config |
| `helix.uptime_s` | stored `start_time` subtracted from `now()` |
| `helix.version` | read from `helix_context.__version__` in child's env (or GET /stats if a version field is added — see note below) |
| `parties.count` | derived — count unique `party_id` values in `GET /sessions?status=all` |
| `parties.party_ids` | same query, projected to unique set |
| `participants.count` | `GET /sessions?status=active`, count |
| `participants.handles` | same, projected to handle list |
| `models.loaded` | Ollama `GET /api/ps` (from helix.toml `[ribosome] base_url`) + helix `/health` for non-Ollama backends |
| `tools` | **NEW endpoint on helix side** — `GET /admin/components` returns the running subsystem list. See below. |
| `genes.raw_chars` | `GET /stats` → `total_chars_raw` |
| `genes.compressed_chars` | `GET /stats` → `total_chars_compressed` |
| `genes.compression_ratio` | `GET /stats` → `compression_ratio` |
| `genes.total` | `GET /stats` → `total_genes` |
| `tokens.session` | **NEW endpoint on helix side** — session counter (see below) |
| `tokens.lifetime` | **NEW endpoint on helix side** — persisted counter (see below) |

### Two small helix-side additions

**`GET /admin/components`** — new endpoint. Returns the list of running
subsystems with their current status:

```json
{
  "components": [
    {"name": "ribosome", "kind": "decoder", "status": "running"},
    {"name": "cpu_tagger", "kind": "encoder", "status": "idle"},
    {"name": "splade", "kind": "encoder", "status": "running"},
    {"name": "sema", "kind": "encoder", "status": "idle"}
  ]
}
```

Status derivation: a subsystem is `running` if it has processed a call
in the last 60 seconds, `idle` otherwise. A subsystem that is disabled
in config is simply absent from the list (matches the "only
active/online" policy).

**Token tracking** — flagged in the original ask as "questionable but
maybe do it anyway." Treating it as optional Phase 2:

- `helix_context.metrics` module accumulates `tokens_in` and
  `tokens_out` counters on every `/v1/chat/completions` call.
- Session counter: in-memory, reset on helix process restart.
- Lifetime counter: persisted to a tiny `metrics.json` next to
  `genome.db`, updated every N seconds.
- New endpoint: `GET /metrics/tokens` returns both counters.
- Launcher's `/api/state` includes `tokens: null` until this lands,
  and the tokens panel simply doesn't render.

Recommendation: ship the launcher without token tracking first, add
metrics in a follow-up. The launcher does not block on this.

## Dashboard layout

A single page, rendered by `dashboard.html`. Layout in CSS grid, with
one header row + one content area that stacks the panels. All measurements
in CSS variables; no pixel values in Python.

```
┌─────────────────────────────────────────────────────────────┐
│  Helix Launcher       [Status: running, pid 56792, 6m42s]  │
│  [▶ Start]  [↻ Restart]  [■ Stop]                          │
├─────────────────────────────────────────────────────────────┤
│  Parties connected: 1                                       │
│    └─ max@local                                             │
│                                                             │
│  Participants: 3                                            │
│    ├─ taude   (active, 4s)                                  │
│    ├─ laude   (active, 12s)                                 │
│    └─ raude   (idle, 45s)                                   │
│                                                             │
│  Models loaded:                                             │
│    • gemma4:e4b   4.2 GB   (ollama)                         │
│                                                             │
│  Tools:                                                     │
│    • ribosome    decoder  running                           │
│    • splade      encoder  running                           │
│    • cpu_tagger  encoder  idle                              │
│    • sema        encoder  idle                              │
│                                                             │
│  Genes:  raw 44.9 MB  →  compressed 16.7 MB  (2.69×)        │
│          total 8,107 genes                                  │
│                                                             │
│  Tokens:  session 184.2K  |  lifetime 1.73M                 │
│          (only if tokens panel is enabled)                  │
└─────────────────────────────────────────────────────────────┘
```

Panels hide themselves conditionally:

- No participants registered → participants panel omitted
- No models reported by helix → models panel omitted
- `tools` list empty → tools panel omitted
- Token tracking not yet shipped → tokens panel omitted
- helix not running → everything below the controls is replaced with a
  single "helix is stopped" banner

This matches the user's explicit rule: "any data not active/online
doesn't need to be displayed."

## HTMX polling model

The dashboard uses HTMX's `hx-get` with a 2-second trigger to re-fetch
the JSON state and replace the panels. A simplified example:

```html
<div id="dashboard-panels"
     hx-get="/api/state/panels"
     hx-trigger="load, every 2s"
     hx-swap="innerHTML">
  <!-- Jinja-rendered panel HTML goes here -->
</div>
```

`/api/state/panels` is a server-rendered HTML partial (not the JSON
endpoint) that runs the same state gathering logic and produces the
`{% if %}`-gated panels as a single HTML blob. The JSON endpoint at
`/api/state` remains for programmatic consumers and debugging.

Poll interval is configurable via a CSS-free `--launcher-poll-interval`
(default `2s`). Users can set `helix-launcher --poll-interval 5` to
reduce chatter.

## Orphan adoption

The launcher adopts already-running helix processes in two situations:

**1. On launcher startup** — `supervisor.adopt()` runs a two-stage check:

   1. **State file**: if `~/.helix/launcher/state.json` has a `helix_pid`,
      verify the process is alive and the command line still matches
      `helix_context.server:app`. If yes → adopted, no further scan.
   2. **Orphan scan**: use `psutil.net_connections()` to find the PID
      listening on `helix_host:helix_port`. Verify its command line
      matches helix uvicorn. Walk up to the uvicorn parent process (the
      one `subprocess.Popen` would hand us). Write its PID + command
      line to the state file.

   This means: **the launcher adopts any helix running outside of it,
   as long as the port matches**. No coordination required between
   externally-started helix processes and the launcher.

**2. On Start button click** — `supervisor.start()` checks port
availability. If the port is busy:

   - If the occupying process IS a helix uvicorn → adopt it (same
     behavior as the startup orphan scan), return its PID, no spawn
   - If the occupying process is something else (another dev server,
     unrelated Python script) → raise `SupervisorError` and record it
     as a `last_error` for the diagnostics panel

This fixes the common UX trap where the launcher and a developer's
`python -m uvicorn helix_context.server:app` race on port 11437 and
the Start button returns a 500. Instead the launcher quietly adopts
the external helix and the dashboard shows it as running.

### Why we adopt without confirming

A user asking "Start this" when a helix is already running on the
target port almost always means "make the launcher aware of the
existing helix," not "start a second helix" (which would fail anyway
since both can't bind the same port). Silent adoption matches the
user's intent; a confirmation prompt would be friction for the
common case.

If you want to explicitly refuse adoption and fail instead, kill
the orphan first, then click Start.

## Diagnostics panel

A footer panel on the dashboard that always renders, regardless of
helix state. Surfaces three kinds of information:

**Last error:** if any `start`/`stop`/`restart` operation failed since
the launcher started, its error message is displayed in a red-bordered
strip. Cleared when the next operation succeeds. Fed by
`supervisor.get_last_error()`.

**Orphan warning:** when `supervisor.is_running()` is False but an
unmanaged helix is detected on the port (e.g. started from a terminal
outside the launcher), a yellow-bordered strip says:

> **Orphan helix detected** — PID X is listening on port 11437 but
> is not managed by this launcher. Click *Start* to adopt it.

**Paths:** the state file location and the helix log location are
always displayed so you can tail the log or inspect the state file
without hunting for them.

## State file

`~/.helix/launcher/state.json` (atomic write + rename):

```json
{
  "helix_pid": 56792,
  "helix_port": 11437,
  "helix_start_time": 1775881217.9,
  "helix_command": ["python", "-m", "uvicorn", "helix_context.server:app", "--host", "127.0.0.1", "--port", "11437"],
  "launcher_pid": 48213,
  "launcher_start_time": 1775881200.0,
  "last_restart_reason": "session registry DAL fix",
  "last_restart_at": 1775881217.9
}
```

On launcher startup:

1. Read state file.
2. If `helix_pid` is set, check `psutil.pid_exists(pid)` + verify the
   process command line matches expected uvicorn invocation.
3. If alive and matching, **adopt** the process (no spawn, just track).
4. If alive but mismatch (PID reused for something else), clear and
   spawn fresh.
5. If not alive, clear and spawn fresh (unless `--no-autostart`).

This means you can restart the launcher without killing helix. The
launcher reattaches to the already-running helix on next start. Nice
UX property.

## Supervisor lifecycle

Pseudocode for `supervisor.py`:

```python
class HelixSupervisor:
    def __init__(self, state: LauncherState, config: LauncherConfig):
        self.state = state
        self.config = config
        self._helix_pid: int | None = None

    def start(self) -> None:
        if self.is_running():
            raise AlreadyRunning()
        proc = subprocess.Popen(
            self._command(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdout=open(self.config.log_path, "ab"),
            stderr=subprocess.STDOUT,
        )
        self._helix_pid = proc.pid
        self.state.write(helix_pid=proc.pid, helix_start_time=time.time())
        self._wait_for_ready(timeout=30)

    def stop(self, reason: str = "manual stop from launcher") -> None:
        if not self.is_running():
            return
        self._announce_restart(reason, expected_downtime_s=10)
        time.sleep(0.75)
        self._kill_tree(self._helix_pid)
        self._wait_for_port_free(port=self.config.helix_port, timeout=10)
        self._helix_pid = None
        self.state.clear_helix()

    def restart(self, reason: str) -> None:
        self.stop(reason=reason)
        self.start()

    def is_running(self) -> bool:
        if self._helix_pid is None:
            return False
        if not psutil.pid_exists(self._helix_pid):
            self._helix_pid = None
            self.state.clear_helix()
            return False
        return True
```

### Graceful kill on Windows vs POSIX

- **Windows:** `subprocess.Popen.terminate()` sends `SIGTERM` equivalent
  but uvicorn's Windows handler is unreliable. Use `taskkill /F /T /PID
  {pid}` via subprocess for reliable tree kill.
- **POSIX:** `os.killpg(os.getpgid(pid), signal.SIGTERM)` kills the
  whole process group cleanly.

Both paths are encapsulated in `_kill_tree(pid)`.

### Window console flash

All `subprocess.Popen` calls MUST include `creationflags=getattr(
subprocess, "CREATE_NO_WINDOW", 0)` to prevent the console window from
flashing when spawning helix on Windows (matches the user's global
CLAUDE.md rule and the existing project pattern).

## Failure modes

| Failure | Behavior |
|---|---|
| Launcher starts, helix PID in state file is dead | Clear state, spawn fresh helix (or stay idle if `--no-autostart`) |
| Launcher starts, helix PID in state file is alive but wrong command | Clear state, log warning, spawn fresh helix on different port if old one holds :11437 |
| helix refuses to start (port in use, import error, bad config) | Surface error in dashboard status banner; keep Start button enabled for retry |
| helix crashes while launcher is running | `is_running()` returns False on next poll → UI shows "stopped" → user clicks Start |
| Launcher itself crashes | helix keeps running (orphaned). Next launcher start adopts it from state file. |
| helix `/admin/announce_restart` fails during Stop | Log warning, proceed with kill anyway (announce is best-effort) |
| psutil not installed | Launcher refuses to start with a clear error pointing at `pip install helix-context[launcher]` |
| Port 11438 already in use | Launcher exits with error, suggests `--port` flag |
| Poll interval to `/api/state` exceeds helix response time | Dashboard shows stale data with a faded style; new poll supersedes |
| User closes browser tab but launcher keeps running | helix keeps running; reopen browser to resume. Launcher does not self-terminate on tab close. |
| User Ctrl+C's the launcher | Launcher stops helix cleanly via the same announce + kill path, then exits. |

## Native window mode

Opt-in via `--native`. Requires the `launcher-native` extras. Uses
`pywebview` to wrap a WebView (WebView2 on Windows, WebKit on macOS,
GTK WebKit on Linux) pointed at `http://127.0.0.1:11438/`. The window
is a thin shim — all rendering still happens in the browser engine,
all data flows through the same FastAPI endpoints.

```python
if args.native:
    import webview
    thread = start_fastapi_in_thread()
    webview.create_window(
        "Helix Launcher",
        f"http://127.0.0.1:{args.port}",
        width=1000, height=720,
        resizable=True,
    )
    webview.start()
```

If `pywebview` is not installed, `--native` prints a helpful error and
points at the install extras. Default behavior (no `--native`) opens
the user's default browser at the launcher URL.

## System tray mode

Opt-in via `--tray`. Requires the `launcher-tray` extras (pystray +
Pillow). Puts a persistent icon in the system notification area with
a menu for controlling helix. Uvicorn runs in a daemon thread; pystray
owns the main thread for its message pump.

```bash
pip install helix-context[launcher-tray]
helix-launcher --tray
```

Tray menu:

```
┌──────────────────┐
│ Open Dashboard   │  ← default action (left-click on icon)
│──────────────────│
│ Start helix      │  (disabled when running)
│ Restart helix    │  (disabled when stopped)
│ Stop helix       │  (disabled when stopped)
│──────────────────│
│ Quit             │  (stops helix + exits launcher)
└──────────────────┘
```

In tray mode the **tray icon is the persistent surface**. You can
open and close the browser tab freely — the launcher keeps running
because the icon is alive. Only clicking **Quit** from the tray menu
actually stops the launcher (and helix via the normal announce-then-
kill path).

**License note:** pystray is LGPL-3. It is installed as an optional
runtime dep by the user — the helix-context wheel itself does not
bundle pystray, so the core package stays Apache-2.0-clean. See
`pyproject.toml` for the `launcher-tray` extras definition.

### `--tray --native` combined (Windows only)

When both flags are set on Windows, the launcher runs in **close-to-tray**
mode: a native pywebview window AND a persistent tray icon, with the
window's close button intercepted to hide the window instead of
exiting. The tray menu gains `Show Window` / `Hide to Tray` items
and `Quit` is the only way to fully stop.

Threading model:

    - Main thread      pywebview (WebView2 message pump)
    - Background       uvicorn (daemon)
    - Background       pystray (daemon, tray icon message pump)

Close-to-tray flow:

    User clicks X          →  window.events.closing returns False
                              → window.hide()
    Tray "Show Window"     →  window.show()
    Tray "Hide to Tray"    →  window.hide()
    Tray "Quit"            →  set quitting flag
                              → window.destroy()
                              → closing handler returns True
                              → webview.start() returns
                              → main() exits
                              → daemon threads (uvicorn + pystray) die with process

**Not supported on macOS or Linux** in this release:

- macOS: pystray's Cocoa backend requires main-thread + NSApplication
  event loop, which directly conflicts with pywebview's WebKit main
  loop. The two libraries cannot share the main thread.
- Linux: depends on the pystray backend (AppIndicator can run from a
  thread, Xlib cannot). Opt-in once the Linux backend story is tested.

On these platforms, passing `--tray --native` returns exit code 2
with a clear error message. Pick one or the other.

## Service install — `helix-launcher install-service`

Once the deploy templates are validated for your platform, the
launcher can install them for you in one command:

```bash
helix-launcher install-service
```

What it does (by platform):

| Platform | Action |
|---|---|
| Linux | Writes `~/.config/systemd/user/helix-launcher.service` with `ExecStart` substituted to the actual `helix-launcher` binary path. Prints the `systemctl --user daemon-reload && systemctl --user enable --now` next steps. |
| macOS | Writes `~/Library/LaunchAgents/com.swiftwing21.helix-launcher.plist` with `ProgramArguments` and `StandardOutPath` substituted. Prints the `launchctl load` next step. |
| Windows | Prints the NSSM recipe. Does NOT install NSSM (licensing + download), does NOT register the service. User follows the printed steps. |

The installer deliberately **never runs** `systemctl enable`,
`launchctl load`, or `nssm install` — those are side effects that
deserve explicit user consent, and it makes the installer reversible.

Dry-run mode shows what would happen without writing anything:

```bash
helix-launcher install-service --dry-run
```

Uninstall removes the file and prints the disable command (again,
doesn't run it):

```bash
helix-launcher uninstall-service
```

Windows users: see [`deploy/windows/README.md`](../deploy/windows/README.md)
for the NSSM walkthrough.

## Helix-side endpoint: `POST /admin/shutdown`

Complements `POST /admin/announce_restart`. Where announce_restart
signals an *intentional restart*, shutdown signals a clean
*stop-and-stay-down*:

```bash
curl -X POST http://127.0.0.1:11437/admin/shutdown \
  -H "Content-Type: application/json" \
  -d '{"actor": "launcher", "reason": "user quit from tray menu"}'
```

Behavior:

1. Stamps `server_state.json` with `state=stopped`
2. Logs the shutdown reason
3. Fires `SIGINT` on the helix process
4. Uvicorn catches SIGINT and runs its graceful-shutdown path,
   invoking the lifespan cleanup (WAL checkpoint, token metrics flush,
   background tasks cancelled)
5. Returns `200` immediately — the actual shutdown happens
   asynchronously as uvicorn processes the signal
6. Callers poll `GET /stats` until connection refused to confirm
   the shutdown completed

This is the endpoint whose 404 prompted "someone was trying
/admin/shutdown" during an earlier test session — now it exists.

## Implementation checklist

Rough ordering for the first PR. Each is independently testable.

1. **Package scaffold.** `helix_context/launcher/` directory, empty
   modules, add `[launcher]` + `[launcher-native]` extras to
   `pyproject.toml`, add `helix-launcher` script entry.
2. **State module.** `state.py` with atomic read/write/clear of
   `~/.helix/launcher/state.json`.
3. **Supervisor module.** `supervisor.py` with Start / Stop / Restart
   / is_running / adopt. Unit tests with a dummy `sleep 60` child
   process.
4. **FastAPI app + CLI.** `app.py` with `main()`, argparse, browser
   launch, `/api/state` JSON endpoint, `/api/control/*` endpoints.
5. **Templates.** `layout.html` + `dashboard.html` + all panel
   components. One CSS file. HTMX vendored to `static/htmx.min.js`.
6. **Dashboard HTMX wiring.** `/api/state/panels` server-rendered
   partial endpoint, 2s polling, conditional panel rendering.
7. **Helix data integration.** Launcher's state collector hits
   `GET /stats`, `GET /sessions`, `GET /health`, and Ollama
   `GET /api/ps`. All with timeouts.
8. **New helix endpoint: `GET /admin/components`.** Component
   introspection — small addition to `server.py`.
9. **Tests.** Unit tests for supervisor + state + adoption. Integration
   test with `TestClient` + a mocked subprocess.
10. **README update.** Quick Start gains a "Run it with the launcher"
    section pointing at this doc.

**Phase 2 (follow-up, not first slice):**

- Token metrics (`helix_context/metrics.py` + `GET /metrics/tokens`)
- Native window via `pywebview`
- Service/daemon install (systemd unit, Windows service, launchd plist)

## Related

- [`SESSION_REGISTRY.md`](SESSION_REGISTRY.md) — parties + participants
  feed the launcher's dashboard.
- [`RESTART_PROTOCOL.md`](RESTART_PROTOCOL.md) — the launcher uses
  `POST /admin/announce_restart` before every Stop/Restart so observer
  sessions (Claude panels, external agents) don't misread the outage.
- `helix_context/server.py::main` — existing thin entry point; stays
  unchanged. The launcher is additive.

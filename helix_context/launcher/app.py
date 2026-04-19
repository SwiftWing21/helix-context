"""
Launcher FastAPI app + CLI entry point.

Run via the ``helix-launcher`` console script. See ``docs/LAUNCHER.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .collector import StateCollector
from .state import StateStore
from .supervisor import (
    AlreadyRunning,
    HelixSupervisor,
    NotRunning,
    ShutdownTimeout,
    StartupTimeout,
    SupervisorError,
)
from .headroom_supervisor import (
    HeadroomSupervisor,
    HeadroomSupervisorError,
    HeadroomNotInstalled,
    is_headroom_installed,
)

log = logging.getLogger("helix.launcher.app")

LAUNCHER_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = LAUNCHER_DIR / "templates"
STATIC_DIR = LAUNCHER_DIR / "static"


def _get_templates():
    """Lazy-import Jinja2 so the module loads without the [launcher] extra."""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as e:
        raise SupervisorError(
            "jinja2 is required. Install with: pip install helix-context[launcher]"
        ) from e
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def create_app(
    store: StateStore,
    supervisor: HelixSupervisor,
    collector: StateCollector,
) -> FastAPI:
    """Build the launcher FastAPI app."""
    templates = _get_templates()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # On startup, try to adopt an already-running helix.
        try:
            supervisor.adopt()
        except Exception:
            log.warning("Adoption check failed", exc_info=True)
        yield
        # On shutdown, stop only processes this launcher spawned itself.
        # Adopted Helix instances should keep running when the launcher exits.
        if supervisor.is_running() and supervisor.owns_process():
            try:
                log.info("Launcher shutting down — stopping helix")
                supervisor.stop(reason="launcher shutdown")
            except Exception:
                log.warning("Graceful helix stop failed during launcher shutdown", exc_info=True)
        elif supervisor.is_running():
            log.info("Launcher shutting down — leaving adopted helix running")

    app = FastAPI(title="Helix Launcher", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.supervisor = supervisor
    app.state.collector = collector

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── dashboard HTML ─────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_root(request: Request) -> HTMLResponse:
        state = collector.collect()
        template = templates.get_template("dashboard.html")
        html = template.render(state=state, launcher_port=_launcher_port(request))
        return HTMLResponse(html)

    @app.get("/api/state/panels", response_class=HTMLResponse)
    async def panels_partial() -> HTMLResponse:
        """Server-rendered HTML partial — just the panels, for polling."""
        state = collector.collect()
        template = templates.get_template("components/panels.html")
        html = template.render(state=state)
        return HTMLResponse(html)

    # ── JSON state API ─────────────────────────────────────────────

    @app.get("/api/state")
    async def api_state():
        return collector.collect()

    # ── control endpoints ──────────────────────────────────────────

    @app.post("/api/control/start")
    async def api_control_start():
        try:
            pid = supervisor.start()
            return {"ok": True, "pid": pid}
        except AlreadyRunning as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        except (StartupTimeout, SupervisorError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/control/stop")
    async def api_control_stop():
        try:
            supervisor.stop(reason="manual stop from launcher UI")
            return {"ok": True}
        except NotRunning as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        except ShutdownTimeout as exc:
            return JSONResponse({"error": str(exc)}, status_code=408)
        except SupervisorError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/control/restart")
    async def api_control_restart():
        try:
            pid = supervisor.restart(reason="manual restart from launcher UI")
            return {"ok": True, "pid": pid}
        except (StartupTimeout, ShutdownTimeout, SupervisorError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


def _launcher_port(request: Request) -> int:
    """Extract the port the launcher is running on from the request."""
    try:
        return request.url.port or 11438
    except Exception:
        return 11438


# ── CLI entry ──────────────────────────────────────────────────────


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="helix-launcher",
        description="Supervisor + dashboard for a helix-context server.",
    )
    p.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "install-service", "uninstall-service"],
        help="Subcommand. 'run' (default) starts the launcher. "
             "'install-service' writes a systemd/launchd service file for "
             "the current platform. 'uninstall-service' removes it.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Launcher UI bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=11438, help="Launcher UI port (default: 11438)")
    p.add_argument("--helix-host", default="127.0.0.1", help="Host for supervised helix (default: 127.0.0.1)")
    p.add_argument("--helix-port", type=int, default=11437, help="Port for supervised helix (default: 11437)")
    p.add_argument("--no-autostart", action="store_true", help="Don't spawn helix on launcher start")
    p.add_argument("--no-browser", action="store_true", help="Don't open the dashboard in a browser")
    p.add_argument("--native", action="store_true", help="Use pywebview native window instead of browser")
    p.add_argument(
        "--tray", action="store_true",
        help="Run with a system tray icon (pystray required). "
             "Click the tray icon to open the dashboard, click Quit from "
             "its menu to stop the launcher. Mutually exclusive with --native.",
    )
    p.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL for model discovery (default: http://127.0.0.1:11434)",
    )
    p.add_argument(
        "--grafana-url",
        default=None,
        help="If set, tray menu gains 'Open Grafana' item pointing here "
             "(e.g. http://localhost:3000/d/helix-overview/helix-overview)",
    )
    p.add_argument(
        "--prometheus-url",
        default=None,
        help="If set, tray menu gains 'Open Prometheus' item pointing here "
             "(e.g. http://localhost:9090/graph)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="For install-service / uninstall-service: show what would happen without making changes",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)


def _check_native_available() -> bool:
    """Return True if pywebview can be imported. Used to fail-fast in main()."""
    try:
        import webview  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


def _check_tray_available() -> bool:
    """Return True if pystray + PIL can be imported. Used for --tray fail-fast."""
    from .tray import is_tray_available
    return is_tray_available()


def _open_ui(url: str, native: bool, window_title: str = "Helix Launcher") -> None:
    """Open the dashboard — browser tab (default) or native webview window.

    This is a blocking call when native=True (webview owns the main thread
    until the user closes the window). Browser mode is non-blocking — it
    just opens a tab and returns.

    Caller is responsible for verifying pywebview availability via
    _check_native_available() BEFORE starting any background work, so
    that --native with no pywebview installed fails fast.
    """
    if native:
        import webview  # noqa: F401  # type: ignore  -- caller already verified
        webview.create_window(
            window_title, url, width=1000, height=720, resizable=True,
        )
        webview.start()
    else:
        try:
            webbrowser.open(url)
        except Exception:
            log.warning("Failed to open browser — navigate manually to %s", url)


def _env_truthy(name: str) -> Optional[bool]:
    """Parse an env var as tristate: True/False/None (unset)."""
    v = os.environ.get(name)
    if v is None:
        return None
    return v.strip().lower() in ("1", "true", "yes", "on")


def _maybe_build_headroom(
    store,
    autostart_override: Optional[bool] = None,
    enabled_override: Optional[bool] = None,
) -> tuple[Optional["HeadroomSupervisor"], Optional[str]]:
    """Build a HeadroomSupervisor + dashboard URL, or (None, None) if
    the feature isn't enabled for this environment.

    Resolution order:
        1. `helix-context[codec]` must be installed (headroom importable)
        2. `[headroom] enabled = true` in helix.toml (or HELIX_HEADROOM_ENABLED=1)
        3. Try to adopt an existing headroom proxy on the configured port.
        4. If no orphan found AND (autostart=true OR HELIX_HEADROOM_AUTOSTART=1),
           spawn a new headroom child.

    Never raises — returns (None, None) on any failure. Headroom is an
    optional enhancement; a broken install should never block launcher start.
    """
    if not is_headroom_installed():
        log.debug("Headroom not installed — skipping headroom supervisor")
        return None, None

    try:
        from helix_context.config import load_config
        cfg = load_config()
        hcfg = cfg.headroom
    except Exception as exc:
        log.warning("Headroom: failed to load config, skipping (%s)", exc)
        return None, None

    enabled = hcfg.enabled if enabled_override is None else enabled_override
    if not enabled:
        log.debug("Headroom: [headroom] enabled=false — skipping")
        return None, None

    autostart = hcfg.autostart if autostart_override is None else autostart_override

    headroom = HeadroomSupervisor(
        store=store,
        host=hcfg.host,
        port=hcfg.port,
        mode=hcfg.mode,
    )

    # Stage 1: adopt if a headroom is already running.
    if headroom.adopt():
        log.info(
            "Headroom: adopted existing process on %s:%d — will NOT stop on Quit",
            hcfg.host, hcfg.port,
        )
    elif autostart:
        # Stage 2: spawn a new one.
        try:
            log.info("Headroom: starting on %s:%d (mode=%s)",
                     hcfg.host, hcfg.port, hcfg.mode)
            headroom.start()
        except HeadroomNotInstalled:
            log.warning("Headroom: package not installed; disabling")
            return None, None
        except Exception as exc:
            log.warning("Headroom: autostart failed (%s); continuing without", exc)
            # Keep the supervisor so the tray still shows Start Headroom.

    dashboard_url = f"http://{hcfg.host}:{hcfg.port}{hcfg.dashboard_path}"
    return headroom, dashboard_url


def _handle_service_command(command: str, dry_run: bool) -> int:
    """Handle install-service / uninstall-service subcommands.

    Returns exit code 0 on success, non-zero on failure. Prints the
    result (which includes next-step instructions) to stdout.
    """
    from .installer import install_service, uninstall_service
    if command == "install-service":
        ok, msg = install_service(dry_run=dry_run)
    else:  # uninstall-service
        ok, msg = uninstall_service(dry_run=dry_run)
    print(msg)
    return 0 if ok else 1


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    # Service install/uninstall subcommands — do not start any server.
    if args.command in ("install-service", "uninstall-service"):
        return _handle_service_command(args.command, dry_run=args.dry_run)

    # --tray and --native combined is only supported on Windows today.
    #
    # Approach: pywebview on main thread, pystray on a detached background
    # thread. Windows allows pystray's Win32 message pump to run outside
    # the main thread when spawned via threading.Thread + icon.run(). On
    # macOS this is fundamentally impossible because pystray's Cocoa
    # backend requires main-thread + NSApplication event loop, which
    # conflicts with pywebview's WebKit main loop. On Linux it depends
    # on the pystray backend (AppIndicator works from a thread, Xlib
    # does not), so we reject it too — opt into it explicitly later
    # once the Linux-backend story is tested.
    if args.tray and args.native:
        if sys.platform != "win32":
            log.error(
                "--tray --native is only supported on Windows in this "
                "release. On macOS and Linux, pystray's backend needs "
                "the main thread and cannot be combined with pywebview. "
                "Pick --tray or --native, not both."
            )
            return 2

    # Fail fast if --native is requested but pywebview isn't installed.
    if args.native and not _check_native_available():
        log.error(
            "--native requires pywebview. Install with: "
            "pip install helix-context[launcher-native]"
        )
        return 1

    # Fail fast if --tray is requested but pystray/PIL aren't installed.
    if args.tray and not _check_tray_available():
        log.error(
            "--tray requires pystray + Pillow. Install with: "
            "pip install helix-context[launcher-tray]"
        )
        return 1

    store = StateStore()
    store.set_launcher(pid=_current_pid())

    supervisor = HelixSupervisor(
        store=store,
        helix_host=args.helix_host,
        helix_port=args.helix_port,
    )

    collector = StateCollector(
        supervisor=supervisor,
        ollama_base_url=args.ollama_base_url,
    )

    # Adopt or start helix before the UI comes up.
    if not supervisor.adopt() and not args.no_autostart:
        try:
            log.info("Starting helix on %s:%d", args.helix_host, args.helix_port)
            supervisor.start()
        except AlreadyRunning:
            pass
        except Exception as exc:
            log.error("Failed to start helix: %s", exc)
            log.info("Launcher will continue; use the Start button once the issue is fixed")

    # Optional Headroom proxy — only provisioned when the package is
    # importable AND [headroom] enabled=true in helix.toml (or
    # HELIX_HEADROOM_ENABLED=1). Adoption is always attempted first so
    # we never spawn a duplicate on top of an existing user-run proxy.
    headroom_supervisor, headroom_dashboard_url = _maybe_build_headroom(
        store=store,
        autostart_override=_env_truthy("HELIX_HEADROOM_AUTOSTART"),
        enabled_override=_env_truthy("HELIX_HEADROOM_ENABLED"),
    )

    app = create_app(store=store, supervisor=supervisor, collector=collector)

    url = f"http://{args.host}:{args.port}/"

    if args.tray and args.native:
        # Windows-only combined mode — see platform guard above.
        return _run_tray_native_combined(
            app, args.host, args.port, url, supervisor,
            grafana_url=args.grafana_url,
            prometheus_url=args.prometheus_url,
        )

    if args.tray:
        # Uvicorn in daemon thread, pystray on main.
        # pystray owns the process lifecycle — Quit from the tray menu
        # calls icon.stop() which unblocks this main thread; main() returns
        # and the daemon thread dies with the process.
        from .tray import HelixTrayIcon

        server_thread = threading.Thread(
            target=_run_uvicorn,
            args=(app, args.host, args.port),
            daemon=True,
            name="launcher-uvicorn",
        )
        server_thread.start()
        time.sleep(0.4)  # let uvicorn bind

        tray_icon = HelixTrayIcon(
            supervisor=supervisor,
            dashboard_url=url,
            grafana_url=args.grafana_url,
            prometheus_url=args.prometheus_url,
            headroom_supervisor=headroom_supervisor,
            headroom_dashboard_url=headroom_dashboard_url,
        )
        log.info("Tray mode active — dashboard at %s", url)
        log.info("Click the tray icon to open the dashboard; Quit from its menu to exit.")
        tray_icon.run()  # blocks until Quit
        return 0

    if args.native:
        # Start uvicorn in a background thread so pywebview can own the main thread.
        server_thread = threading.Thread(
            target=_run_uvicorn,
            args=(app, args.host, args.port),
            daemon=True,
            name="launcher-uvicorn",
        )
        server_thread.start()
        # Tiny delay to let uvicorn bind.
        time.sleep(0.4)
        _open_ui(url, native=True)
    else:
        if not args.no_browser:
            # Browser tab opened just before uvicorn blocks the main thread.
            _schedule_open(url)
        _run_uvicorn(app, args.host, args.port)

    return 0


def _current_pid() -> int:
    import os
    return os.getpid()


def _run_tray_native_combined(
    app: FastAPI,
    host: str,
    port: int,
    url: str,
    supervisor: HelixSupervisor,
    grafana_url: Optional[str] = None,
    prometheus_url: Optional[str] = None,
) -> int:
    """Close-to-tray mode: pywebview window + persistent tray icon.

    Windows-only (see _run_args guard). Threading model:

      - Main thread:        pywebview (WebView2 message pump)
      - Background thread:  uvicorn (daemon)
      - Background thread:  pystray (daemon, owns tray icon message pump)

    Window close behavior:
      - User clicks X → closing event returns False, window.hide()
      - Tray "Show Window" → window.show()
      - Tray "Hide to Tray" → window.hide()
      - Tray "Quit" → set flag, window.destroy() → closing returns True
                     → webview.start() returns → main() exits

    On quit, the daemon threads (uvicorn + pystray) are cleaned up
    automatically when the process exits. The tray icon.stop() is also
    called explicitly for good measure.
    """
    import webview  # type: ignore
    import pystray  # type: ignore
    from .tray import _build_icon_image

    log.info("Tray + native mode — pywebview on main, pystray detached")

    # ── background uvicorn ────────────────────────────────────────
    server_thread = threading.Thread(
        target=_run_uvicorn,
        args=(app, host, port),
        daemon=True,
        name="launcher-uvicorn",
    )
    server_thread.start()
    time.sleep(0.4)  # let uvicorn bind

    # ── shared state for the close-to-tray dance ──────────────────
    quitting = threading.Event()
    window_holder: list = [None]  # mutable holder shared between threads
    tray_holder: list = [None]

    def on_window_closing():
        """Hook for window.events.closing — intercept close, hide instead."""
        if quitting.is_set():
            return True  # allow close
        w = window_holder[0]
        if w is not None:
            try:
                w.hide()
            except Exception:
                log.warning("Hide-on-close failed", exc_info=True)
        return False  # cancel the close

    # ── tray menu action handlers ─────────────────────────────────
    def _show_window(icon, item):  # noqa: ARG001
        w = window_holder[0]
        if w is not None:
            try:
                w.show()
            except Exception:
                log.warning("Show window failed", exc_info=True)

    def _hide_window(icon, item):  # noqa: ARG001
        w = window_holder[0]
        if w is not None:
            try:
                w.hide()
            except Exception:
                log.warning("Hide window failed", exc_info=True)

    def _open_browser(icon, item):  # noqa: ARG001
        try:
            webbrowser.open(url)
        except Exception:
            log.warning("Open browser failed", exc_info=True)

    def _start_helix(icon, item):  # noqa: ARG001
        try:
            supervisor.start()
        except Exception:
            log.warning("Tray start failed", exc_info=True)

    def _restart_helix(icon, item):  # noqa: ARG001
        try:
            supervisor.restart(reason="manual restart from tray menu")
        except Exception:
            log.warning("Tray restart failed", exc_info=True)

    def _stop_helix(icon, item):  # noqa: ARG001
        try:
            supervisor.stop(reason="manual stop from tray menu")
        except Exception:
            log.warning("Tray stop failed", exc_info=True)

    def _quit_all(icon, item):  # noqa: ARG001
        log.info("Tray Quit — destroying window")
        quitting.set()

        try:
            if supervisor.is_running():
                supervisor.stop(reason="launcher quit from tray menu (native)")
        except Exception:
            log.warning("Helix stop during quit failed", exc_info=True)

        w = window_holder[0]
        if w is not None:
            try:
                w.destroy()
            except Exception:
                log.warning("Window destroy failed", exc_info=True)

        tray = tray_holder[0]
        if tray is not None:
            try:
                tray.stop()
            except Exception:
                log.warning("Tray stop failed", exc_info=True)

    def _open_grafana(icon, item):  # noqa: ARG001
        if grafana_url:
            try:
                webbrowser.open(grafana_url)
            except Exception:
                log.warning("Open Grafana failed", exc_info=True)

    def _open_prometheus(icon, item):  # noqa: ARG001
        if prometheus_url:
            try:
                webbrowser.open(prometheus_url)
            except Exception:
                log.warning("Open Prometheus failed", exc_info=True)

    # ── build the tray menu ───────────────────────────────────────
    menu_items = [
        pystray.MenuItem("Show Window", _show_window, default=True),
        pystray.MenuItem("Hide to Tray", _hide_window),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open in Browser", _open_browser),
    ]
    if grafana_url:
        menu_items.append(pystray.MenuItem("Open Grafana", _open_grafana))
    if prometheus_url:
        menu_items.append(pystray.MenuItem("Open Prometheus", _open_prometheus))
    menu_items.extend([
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            "Start helix", _start_helix,
            enabled=lambda item: not supervisor.is_running(),  # noqa: ARG005
        ),
        pystray.MenuItem(
            "Restart helix", _restart_helix,
            enabled=lambda item: supervisor.is_running(),  # noqa: ARG005
        ),
        pystray.MenuItem(
            "Stop helix", _stop_helix,
            enabled=lambda item: supervisor.is_running(),  # noqa: ARG005
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _quit_all),
    ])
    tray_menu = pystray.Menu(*menu_items)

    tray_icon = pystray.Icon(
        name="helix-launcher",
        icon=_build_icon_image(),
        title="Helix Launcher",
        menu=tray_menu,
    )
    tray_holder[0] = tray_icon

    # ── start pystray on a background thread ──────────────────────
    tray_thread = threading.Thread(
        target=tray_icon.run,
        daemon=True,
        name="launcher-tray",
    )
    tray_thread.start()

    # ── create the pywebview window on main thread ────────────────
    window = webview.create_window(
        "Helix Launcher",
        url,
        width=1000,
        height=720,
        resizable=True,
    )
    window_holder[0] = window

    # Hook closing BEFORE start so the first close attempt is intercepted.
    try:
        window.events.closing += on_window_closing
    except Exception:
        log.warning("Could not hook window.events.closing — close-to-tray disabled", exc_info=True)

    log.info("Combined mode active — dashboard at %s", url)
    log.info("Close the window to hide to tray; Quit from tray menu to exit.")

    webview.start()  # blocks until window.destroy() is called by _quit_all

    # Cleanup
    try:
        tray_icon.stop()
    except Exception:
        pass
    log.info("Combined mode: exited cleanly")
    return 0


def _schedule_open(url: str) -> None:
    """Fire a browser open after a short delay so uvicorn has time to bind."""
    def _worker() -> None:
        time.sleep(0.6)
        _open_ui(url, native=False)
    t = threading.Thread(target=_worker, daemon=True, name="launcher-browser-open")
    t.start()


def _run_uvicorn(app: FastAPI, host: str, port: int) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    sys.exit(main())

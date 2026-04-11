"""
Launcher FastAPI app + CLI entry point.

Run via the ``helix-launcher`` console script. See ``docs/LAUNCHER.md``.
"""

from __future__ import annotations

import argparse
import logging
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
        # On shutdown, stop helix cleanly (best-effort).
        if supervisor.is_running():
            try:
                log.info("Launcher shutting down — stopping helix")
                supervisor.stop(reason="launcher shutdown")
            except Exception:
                log.warning("Graceful helix stop failed during launcher shutdown", exc_info=True)

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
    p.add_argument("--host", default="127.0.0.1", help="Launcher UI bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=11438, help="Launcher UI port (default: 11438)")
    p.add_argument("--helix-host", default="127.0.0.1", help="Host for supervised helix (default: 127.0.0.1)")
    p.add_argument("--helix-port", type=int, default=11437, help="Port for supervised helix (default: 11437)")
    p.add_argument("--no-autostart", action="store_true", help="Don't spawn helix on launcher start")
    p.add_argument("--no-browser", action="store_true", help="Don't open the dashboard in a browser")
    p.add_argument("--native", action="store_true", help="Use pywebview native window instead of browser")
    p.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL for model discovery (default: http://127.0.0.1:11434)",
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


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    # Fail fast if --native is requested but pywebview isn't installed.
    if args.native and not _check_native_available():
        log.error(
            "--native requires pywebview. Install with: "
            "pip install helix-context[launcher-native]"
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

    app = create_app(store=store, supervisor=supervisor, collector=collector)

    url = f"http://{args.host}:{args.port}/"

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

"""A minimal HTTP router with middleware support."""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class Request:
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def json(self) -> Any:
        if self.body is None:
            return None
        return json.loads(self.body)


@dataclass
class Response:
    status: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""

    @classmethod
    def json(cls, data: Any, status: int = 200) -> Response:
        return cls(
            status=status,
            headers={"Content-Type": "application/json"},
            body=json.dumps(data),
        )

    @classmethod
    def text(cls, text: str, status: int = 200) -> Response:
        return cls(
            status=status,
            headers={"Content-Type": "text/plain"},
            body=text,
        )

    @classmethod
    def not_found(cls) -> Response:
        return cls.json({"error": "not found"}, status=404)


Handler = Callable[[Request], Response]
Middleware = Callable[[Request, Handler], Response]


class Router:
    """A simple path-based HTTP router with middleware chain."""

    def __init__(self):
        self._routes: List[Tuple[str, str, Handler]] = []
        self._middleware: List[Middleware] = []

    def add_route(self, method: str, path: str, handler: Handler) -> None:
        self._routes.append((method.upper(), path, handler))

    def get(self, path: str) -> Callable:
        def decorator(fn: Handler) -> Handler:
            self.add_route("GET", path, fn)
            return fn
        return decorator

    def post(self, path: str) -> Callable:
        def decorator(fn: Handler) -> Handler:
            self.add_route("POST", path, fn)
            return fn
        return decorator

    def use(self, middleware: Middleware) -> None:
        self._middleware.append(middleware)

    def handle(self, request: Request) -> Response:
        handler = self._match(request.method, request.path)
        if handler is None:
            return Response.not_found()

        chain = handler
        for mw in reversed(self._middleware):
            prev = chain
            chain = lambda req, _mw=mw, _prev=prev: _mw(req, _prev)

        return chain(request)

    def _match(self, method: str, path: str) -> Optional[Handler]:
        for route_method, route_path, handler in self._routes:
            if route_method == method and self._path_matches(route_path, path):
                return handler
        return None

    @staticmethod
    def _path_matches(pattern: str, path: str) -> bool:
        pattern_parts = pattern.strip("/").split("/")
        path_parts = path.strip("/").split("/")

        if len(pattern_parts) != len(path_parts):
            return False

        for pp, rp in zip(pattern_parts, path_parts):
            if pp.startswith(":"):
                continue
            if pp != rp:
                return False
        return True


def logging_middleware(request: Request, next_handler: Handler) -> Response:
    response = next_handler(request)
    elapsed = (time.time() - request.start_time) * 1000
    log.info("%s %s -> %d (%.1fms)", request.method, request.path, response.status, elapsed)
    return response


def cors_middleware(request: Request, next_handler: Handler) -> Response:
    response = next_handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


def rate_limit_middleware(max_requests: int = 100, window_seconds: float = 60.0) -> Middleware:
    buckets: Dict[str, List[float]] = {}

    def middleware(request: Request, next_handler: Handler) -> Response:
        client = request.headers.get("X-Real-IP", "unknown")
        now = time.time()
        cutoff = now - window_seconds

        if client not in buckets:
            buckets[client] = []

        buckets[client] = [t for t in buckets[client] if t > cutoff]

        if len(buckets[client]) >= max_requests:
            return Response.json(
                {"error": "rate limit exceeded", "retry_after": window_seconds},
                status=429,
            )

        buckets[client].append(now)
        return next_handler(request)

    return middleware

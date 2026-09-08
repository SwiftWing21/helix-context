"""Opt-in, request-local benchmark evidence; never part of a response schema.

Only counts and watched gold IDs are retained, not whole candidate pools.
Stages describe observed membership, not interchangeable rankings. Contexts
are deliberately not propagated into executor threads: unsupported paths
remain unmeasured instead of being mistaken for empty retrievals.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from typing import Iterable


_RETRIEVAL_STAGES = (
    "fts_raw", "fts_eligible", "pre_shortlist", "post_shortlist",
    "final_scoring", "retrieval_returned",
)
_REQUEST_STAGES = ("post_blend_scores", "post_blend_candidates")
_active: ContextVar["StageCapture | None"] = ContextVar("bench_stages", default=None)
_retrieval: ContextVar["_Stages | None"] = ContextVar("bench_retrieval", default=None)


class _Stages:
    def __init__(self, gold_ids: frozenset[str], names: tuple[str, ...]):
        self.gold_ids = gold_ids
        self.stages = {
            name: {"status": "not_executed", "count": None, "gold_ids": None}
            for name in names
        }

    def record(self, name: str, ids: Iterable[str], **details) -> None:
        count = 0
        found = set()
        for gid in ids:
            count += 1
            if gid in self.gold_ids:
                found.add(gid)
        self.stages[name] = {
            "status": "captured", "count": count, "gold_ids": sorted(found),
            **details,
        }

    def unavailable(self, name: str, *, status: str, reason: str) -> None:
        self.stages[name] = {
            "status": status, "count": None, "gold_ids": None, "reason": reason,
        }


class StageCapture(_Stages):
    def __init__(self, gold_ids: Iterable[str], enabled: bool):
        super().__init__(frozenset(gold_ids), _REQUEST_STAGES)
        self.enabled = enabled
        self.retrievals: list[dict] = []
        self.error: str | None = None
        self.unsupported: list[str] = []

    def report(self) -> dict:
        failed = self.error is not None or any(
            call["status"] == "failed"
            or any(stage["status"] == "failed"
                   or stage.get("filter_status") == "failed"
                   for stage in call["stages"].values())
            for call in self.retrievals
        )
        status = "failed" if failed else (
            "complete" if self.enabled and self.retrievals and not self.unsupported
            else "not_captured"
        )
        return deepcopy({
            "version": 1, "status": status, "error": self.error,
            "retrievals": self.retrievals, "stages": self.stages,
            "unsupported": self.unsupported,
        })


@contextmanager
def capture_stages(gold_ids: Iterable[str], *, enabled: bool = True):
    """Capture one benchmark query, including partial evidence on failure."""
    capture = StageCapture(gold_ids, enabled)
    token = _active.set(capture if enabled else None)
    retrieval_token = _retrieval.set(None)
    try:
        yield capture
    except BaseException as exc:
        capture.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _retrieval.reset(retrieval_token)
        _active.reset(token)


def current_capture() -> StageCapture | None:
    return _active.get()


def current_retrieval() -> _Stages | None:
    return _retrieval.get()


def trace_retrieval(function):
    """Keep independent lexical calls separate and mark propagated errors."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        capture = _active.get()
        if capture is None:
            return function(*args, **kwargs)
        observation = _Stages(capture.gold_ids, _RETRIEVAL_STAGES)
        call = {"path": "query_docs", "status": "complete", "error": None,
                "stages": observation.stages}
        capture.retrievals.append(call)
        token = _retrieval.set(observation)
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            call["status"] = "failed"
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _retrieval.reset(token)
    return wrapped

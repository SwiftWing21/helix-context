"""
Token metrics — session + lifetime token counters with disk persistence.

Counts prompt + completion tokens flowing through `/v1/chat/completions`.
The session counter resets on each helix process restart; the lifetime
counter is persisted to a small JSON file next to ``genome.db`` and
survives restarts.

Wired into the proxy from server.py — see ``_forward_and_replicate``,
``_forward_raw``, and ``_stream_and_tee``. Exposed via the
``GET /metrics/tokens`` endpoint and consumed by the launcher's tokens
panel.

Persistence model:
    - Mutations bump in-memory counters atomically (under a Lock).
    - Lifetime totals are flushed to disk opportunistically: every
      ``persist_interval_s`` seconds OR on shutdown via ``flush()``.
    - Atomic write via tempfile + os.replace, so readers (e.g. an
      external dashboard) never see partial files.

Token-source priority:
    1. Upstream `usage.prompt_tokens` / `usage.completion_tokens` if
       provided (OpenAI / Ollama compatible). This is the authoritative
       count.
    2. Heuristic estimation from character counts when usage is absent
       (~4 chars/token for English). Tagged as ``estimated`` in the
       counter so callers can filter if they want hard accuracy.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("helix.metrics")


# Rough chars-per-token estimate for English text. Used only as a fallback
# when upstream doesn't report a usage object.
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(text: str) -> int:
    """Crude character-count token estimate. Used as a fallback only."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


class TokenCounter:
    """Thread-safe session + lifetime token counter with disk persistence.

    Designed for the helix proxy's hot path. ``add()`` is O(1) under a
    short critical section; the actual disk write only happens every
    ``persist_interval_s`` seconds.
    """

    def __init__(
        self,
        persist_path: Path,
        persist_interval_s: float = 30.0,
    ) -> None:
        self.persist_path = Path(persist_path)
        self.persist_interval_s = persist_interval_s
        self._lock = threading.Lock()

        # Session counters reset on every process start.
        self._session_in: int = 0
        self._session_out: int = 0
        self._session_estimated_in: int = 0
        self._session_estimated_out: int = 0
        self._session_started_at: float = time.time()

        # Lifetime counters loaded from disk (or zero if no file yet).
        loaded = self._load()
        self._lifetime_in: int = loaded.get("prompt_tokens", 0)
        self._lifetime_out: int = loaded.get("completion_tokens", 0)
        self._lifetime_estimated_in: int = loaded.get("estimated_prompt_tokens", 0)
        self._lifetime_estimated_out: int = loaded.get("estimated_completion_tokens", 0)
        self._lifetime_started_at: float = loaded.get("first_seen_at", time.time())
        self._last_persist_at: float = time.time()

    # ── public surface ─────────────────────────────────────────────

    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        estimated: bool = False,
    ) -> None:
        """Add a single proxy call's worth of tokens.

        Pass ``estimated=True`` when the count came from the char-count
        fallback rather than an authoritative ``usage`` field. Estimated
        and exact counts are tracked in separate buckets so the panel
        can show both.
        """
        if prompt_tokens < 0 or completion_tokens < 0:
            log.warning("TokenCounter.add called with negative values; ignoring")
            return
        with self._lock:
            if estimated:
                self._session_estimated_in += prompt_tokens
                self._session_estimated_out += completion_tokens
                self._lifetime_estimated_in += prompt_tokens
                self._lifetime_estimated_out += completion_tokens
            else:
                self._session_in += prompt_tokens
                self._session_out += completion_tokens
                self._lifetime_in += prompt_tokens
                self._lifetime_out += completion_tokens

            if time.time() - self._last_persist_at >= self.persist_interval_s:
                try:
                    self._persist_locked()
                except Exception:
                    log.warning("TokenCounter persist failed", exc_info=True)
                self._last_persist_at = time.time()

    def add_from_usage(self, usage: Optional[dict]) -> bool:
        """Convenience: add from an OpenAI-compatible usage dict.

        Returns True on success (usage was a dict with token fields),
        False if the dict was missing or malformed (caller should fall
        back to estimation).
        """
        if not isinstance(usage, dict):
            return False
        try:
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
        except (TypeError, ValueError):
            return False
        if prompt == 0 and completion == 0:
            return False
        self.add(prompt, completion, estimated=False)
        return True

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of the current counters."""
        with self._lock:
            session_total_exact = self._session_in + self._session_out
            session_total_est = self._session_estimated_in + self._session_estimated_out
            lifetime_total_exact = self._lifetime_in + self._lifetime_out
            lifetime_total_est = self._lifetime_estimated_in + self._lifetime_estimated_out
            return {
                "session": {
                    "prompt_tokens": self._session_in,
                    "completion_tokens": self._session_out,
                    "total": session_total_exact,
                    "estimated_prompt_tokens": self._session_estimated_in,
                    "estimated_completion_tokens": self._session_estimated_out,
                    "estimated_total": session_total_est,
                    "started_at": self._session_started_at,
                },
                "lifetime": {
                    "prompt_tokens": self._lifetime_in,
                    "completion_tokens": self._lifetime_out,
                    "total": lifetime_total_exact,
                    "estimated_prompt_tokens": self._lifetime_estimated_in,
                    "estimated_completion_tokens": self._lifetime_estimated_out,
                    "estimated_total": lifetime_total_est,
                    "first_seen_at": self._lifetime_started_at,
                },
                "persist_path": str(self.persist_path),
                "persist_interval_s": self.persist_interval_s,
            }

    def flush(self) -> None:
        """Force a disk write — call on graceful shutdown."""
        with self._lock:
            try:
                self._persist_locked()
                self._last_persist_at = time.time()
            except Exception:
                log.warning("TokenCounter flush failed", exc_info=True)

    def reset_lifetime(self) -> None:
        """Wipe the lifetime counter to zero (and write it). Use with care."""
        with self._lock:
            self._lifetime_in = 0
            self._lifetime_out = 0
            self._lifetime_estimated_in = 0
            self._lifetime_estimated_out = 0
            self._lifetime_started_at = time.time()
            try:
                self._persist_locked()
            except Exception:
                log.warning("TokenCounter persist failed during reset", exc_info=True)

    # ── persistence ────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self.persist_path.exists():
            return {}
        try:
            return json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning(
                "TokenCounter could not read %s — starting from zero",
                self.persist_path, exc_info=True,
            )
            return {}

    def _persist_locked(self) -> None:
        """Caller MUST hold ``self._lock`` before calling this."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt_tokens": self._lifetime_in,
            "completion_tokens": self._lifetime_out,
            "estimated_prompt_tokens": self._lifetime_estimated_in,
            "estimated_completion_tokens": self._lifetime_estimated_out,
            "first_seen_at": self._lifetime_started_at,
            "updated_at": time.time(),
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="metrics_", suffix=".tmp", dir=str(self.persist_path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp_path, self.persist_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

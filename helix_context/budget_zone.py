"""
Budget-zone gene expression cap — spike.

Complements the existing confidence-tier logic in ``context_manager.py``
(TIGHT=3 / FOCUSED=6 / BROAD=12) by adding a second axis driven by the
caller's remaining context budget, signalled via incoming prompt size.

Thresholds mirror the CLAUDE.md context-hygiene zones so the two systems
speak the same language:

    clean     (<25%)     no cap        (use full max_genes)
    soft      (25-40%)   cap = 12      (= broad max, no-op by default)
    pressure  (40-60%)   cap = 6       (= focused max)
    cap       (60-80%)   cap = 3       (= tight max)
    emergency (80%+)     cap = 1       (single strongest gene)

The returned value is a CEILING, not a set-point. The confidence tier
continues to pick the actual number within that ceiling, so behavior
only narrows under pressure — never widens past what the current
max_genes_per_turn would allow.

Enabled via env var ``HELIX_BUDGET_ZONE=1``. When unset or false, the
helper is a no-op. This keeps the spike safe to merge behind a flag.
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_WINDOW_TOKENS = 128_000

_TRUTHY = {"1", "true", "yes", "on"}

# Zone boundaries as fractions of window_tokens. Keep in sync with
# CLAUDE.md context-hygiene: Clean / Soft / Pressure / Cap / Emergency.
_ZONE_BOUNDARIES = (
    (0.25, None, "clean"),     # <25% — no cap
    (0.40, 12,   "soft"),      # 25-40% — cap at broad max
    (0.60, 6,    "pressure"),  # 40-60% — clamp to focused
    (0.80, 3,    "cap"),       # 60-80% — clamp to tight
    (float("inf"), 1, "emergency"),  # 80%+ — single gene
)


def is_enabled() -> bool:
    """Env-flag check, re-read every call so tests can toggle in-process."""
    return os.environ.get("HELIX_BUDGET_ZONE", "").lower() in _TRUTHY


def zone_for(prompt_tokens: int, window_tokens: int = DEFAULT_WINDOW_TOKENS) -> str:
    """Return the zone name for a given prompt/window pair."""
    if window_tokens <= 0 or prompt_tokens < 0:
        return "clean"
    ratio = prompt_tokens / window_tokens
    for boundary, _cap, name in _ZONE_BOUNDARIES:
        if ratio < boundary:
            return name
    return "emergency"  # pragma: no cover (inf boundary catches all)


def zone_cap(
    prompt_tokens: Optional[int],
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
) -> Optional[int]:
    """Return the gene-count ceiling for a given prompt size.

    Returns None when no cap should apply (clean zone, missing signal,
    or the feature flag is off). Callers should do
    ``max_genes = min(max_genes, zone_cap(...) or max_genes)``.
    """
    if not is_enabled():
        return None
    if prompt_tokens is None or window_tokens <= 0:
        return None
    ratio = prompt_tokens / window_tokens
    for boundary, cap, _name in _ZONE_BOUNDARIES:
        if ratio < boundary:
            return cap
    return 1  # pragma: no cover


def zone_metadata(
    prompt_tokens: Optional[int],
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
) -> dict:
    """Structured telemetry payload — emitted alongside each expression."""
    if prompt_tokens is None:
        return {"enabled": is_enabled(), "zone": None, "cap": None, "ratio": None}
    ratio = prompt_tokens / max(window_tokens, 1)
    return {
        "enabled": is_enabled(),
        "zone": zone_for(prompt_tokens, window_tokens),
        "cap": zone_cap(prompt_tokens, window_tokens),
        "ratio": round(ratio, 3),
        "prompt_tokens": prompt_tokens,
        "window_tokens": window_tokens,
    }

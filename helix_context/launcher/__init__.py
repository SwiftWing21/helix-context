"""
Helix Launcher — supervisor process + control UI.

See ``docs/LAUNCHER.md`` for the full design.

Entry point: ``helix-launcher`` (console script) → ``launcher.app:main``.

Dependencies (optional extras):
    pip install helix-context[launcher]         # browser mode
    pip install helix-context[launcher-native]  # + pywebview native window
"""

from __future__ import annotations

__all__ = ["app", "state", "supervisor"]

"""
Tests for helix_context.headroom_bridge.

Covers the pure-Python dispatcher logic, the fallback path when headroom is
unavailable, and a live round-trip per specialist when it IS available.

The live specialist tests run unconditionally when headroom-ai is installed.
They exercise real model loading and ONNX inference — expect 1-3 seconds of
runtime on first invocation (cached on subsequent calls).
"""

from __future__ import annotations

import pytest

from helix_context import headroom_bridge
from helix_context.headroom_bridge import (
    _detect_language,
    _pick_specialist,
    compress_text,
    is_headroom_available,
)


# ── Pure dispatch logic (no dependency on headroom being installed) ────

class TestPickSpecialist:
    def test_empty_domains_defaults_to_kompress(self):
        assert _pick_specialist([]) == "kompress"

    def test_unknown_domain_defaults_to_kompress(self):
        assert _pick_specialist(["markdown", "docs"]) == "kompress"

    def test_python_routes_to_kompress(self):
        # CodeCompressor disabled (40% invalid syntax — see 2f518dc).
        assert _pick_specialist(["python"]) == "kompress"

    def test_rust_routes_to_kompress(self):
        assert _pick_specialist(["rust", "cargo"]) == "kompress"

    def test_typescript_routes_to_kompress(self):
        assert _pick_specialist(["typescript"]) == "kompress"

    def test_log_routes_to_log(self):
        assert _pick_specialist(["log"]) == "log"

    def test_pytest_routes_to_log(self):
        assert _pick_specialist(["pytest"]) == "log"

    def test_diff_routes_to_diff(self):
        assert _pick_specialist(["diff"]) == "diff"

    def test_patch_routes_to_diff(self):
        assert _pick_specialist(["patch"]) == "diff"

    def test_diff_wins_over_code_when_both_present(self):
        # diff is checked first in _pick_specialist
        assert _pick_specialist(["diff", "python"]) == "diff"

    def test_log_wins_over_code_when_both_present(self):
        # log is checked before code
        assert _pick_specialist(["pytest", "python"]) == "log"

    def test_case_insensitive(self):
        assert _pick_specialist(["PYTHON"]) == "kompress"
        assert _pick_specialist(["Log"]) == "log"


class TestDetectLanguage:
    def test_python_detected(self):
        assert _detect_language(["python", "code"]) == "python"

    def test_rust_detected(self):
        assert _detect_language(["rust"]) == "rust"

    def test_py_shortform(self):
        assert _detect_language(["py"]) == "python"

    def test_no_language_returns_none(self):
        assert _detect_language(["markdown", "docs"]) is None

    def test_empty_returns_none(self):
        assert _detect_language([]) is None


# ── Shortcut behaviors (work regardless of headroom availability) ──────

class TestCompressTextShortcuts:
    def test_empty_content_returns_empty(self):
        assert compress_text("", target_chars=1000) == ""

    def test_content_under_budget_returns_unchanged(self):
        short = "def foo(): return 42"
        assert compress_text(short, target_chars=1000) == short

    def test_content_exactly_at_budget_returns_unchanged(self):
        content = "x" * 500
        assert compress_text(content, target_chars=500) == content

    def test_none_content_type_falls_through_to_kompress(self):
        """content_type=None should not crash the dispatcher."""
        # Only runs if headroom available, otherwise hits truncation fallback
        out = compress_text("hello world " * 200, target_chars=100, content_type=None)
        assert isinstance(out, str)
        assert len(out) > 0


# ── Fallback path (simulate headroom unavailable) ───────────────────────

class TestFallbackPath:
    def test_fallback_when_headroom_unavailable(self, monkeypatch):
        """When is_headroom_available returns False, compress_text truncates."""
        monkeypatch.setattr(headroom_bridge, "_HEADROOM_AVAILABLE", False)
        content = "a" * 3000
        out = compress_text(content, target_chars=1000, content_type=["python"])
        assert len(out) == 1000
        assert out == content[:1000]

    def test_fallback_strips_whitespace(self, monkeypatch):
        monkeypatch.setattr(headroom_bridge, "_HEADROOM_AVAILABLE", False)
        content = "   " + "word " * 500
        out = compress_text(content, target_chars=1000)
        # Should not start with whitespace after .strip()
        assert not out.startswith(" ")

    def test_env_toggle_disables_headroom(self, monkeypatch):
        """HELIX_DISABLE_HEADROOM=1 should bypass Headroom even if installed."""
        monkeypatch.setenv("HELIX_DISABLE_HEADROOM", "1")
        assert is_headroom_available() is False
        # Compressing should fall back to truncation
        content = "a" * 3000
        out = compress_text(content, target_chars=500)
        assert out == content[:500]

    def test_env_toggle_accepts_truthy_variants(self, monkeypatch):
        """Env toggle should accept 1, true, yes, on (case insensitive)."""
        for val in ("1", "true", "True", "TRUE", "yes", "on"):
            monkeypatch.setenv("HELIX_DISABLE_HEADROOM", val)
            assert is_headroom_available() is False, f"Expected False for {val!r}"

    def test_env_toggle_off_leaves_headroom_active(self, monkeypatch):
        """Empty or falsy env var should NOT disable Headroom (when installed)."""
        monkeypatch.delenv("HELIX_DISABLE_HEADROOM", raising=False)
        # Only meaningful if headroom is actually installed; otherwise the
        # probe will naturally return False.
        # This test asserts the env override doesn't interfere with normal
        # probe behavior — it's a no-op when unset.
        result = is_headroom_available()
        # In our test env headroom IS installed, so expect True
        if _headroom_installed:
            assert result is True


# ── Live specialist round-trips (require headroom-ai installed) ────────

_headroom_installed = is_headroom_available()
requires_headroom = pytest.mark.skipif(
    not _headroom_installed,
    reason="headroom-ai not installed; install with pip install helix-context[codec]",
)


@requires_headroom
class TestLiveSpecialists:
    def test_kompress_generic_text(self):
        content = (
            "The helix-context project implements a genome-based approach "
            "to LLM context compression. It uses a SQLite database to store "
            "compressed chunks called genes, which are retrieved and expressed "
            "per query to fit within a small context window. "
        ) * 20
        out = compress_text(content, target_chars=300, content_type=["text"])
        assert isinstance(out, str)
        assert len(out) > 0
        # Compressed output should be shorter than original
        assert len(out) <= len(content)

    def test_code_specialist_python(self):
        code = """
def calculate_ellipticity(coverage, density, freshness, logical_coherence):
    import math
    factors = [coverage, density, freshness, logical_coherence]
    if any(f <= 0 for f in factors):
        return 0.0
    product = 1.0
    for f in factors:
        product *= f
    return math.pow(product, 1.0 / len(factors))

class ContextHealth:
    def __init__(self, status, ellipticity):
        self.status = status
        self.ellipticity = ellipticity
    def is_aligned(self):
        return self.ellipticity >= 0.7
""" * 3
        out = compress_text(code, target_chars=500, content_type=["python", "code"])
        assert isinstance(out, str)
        assert len(out) > 0

    def test_log_specialist(self):
        log_content = """
2026-04-10 10:15:26 [INFO] Starting helix-context server on port 11437
2026-04-10 10:15:27 [INFO] Loaded 7990 genes from genome.db
2026-04-10 10:15:27 [WARNING] Ribosome warmup disabled per config
2026-04-10 10:15:28 [ERROR] Failed to connect to Ollama at localhost:11434
Traceback (most recent call last):
  File "helix_context/ribosome.py", line 318, in pack
    result = self.backend.complete(prompt)
ConnectionRefusedError: [Errno 61] Connection refused
""" * 10
        out = compress_text(log_content, target_chars=400, content_type=["log"])
        assert isinstance(out, str)
        assert len(out) > 0

    def test_diff_specialist(self):
        diff_content = """
diff --git a/helix_context/context_manager.py b/helix_context/context_manager.py
index abc123..def456 100644
--- a/helix_context/context_manager.py
+++ b/helix_context/context_manager.py
@@ -492,7 +492,11 @@ class HelixContextManager:
             src_attr = f' src="{short}"' if short else ""
-            content = g.content[:1000].strip()
+            content = compress_text(
+                g.content,
+                target_chars=1000,
+                content_type=g.promoter.domains,
+            )
             spliced_map[g.gene_id] = f"<GENE{src_attr}{kv_attrs}>..."
""" * 2
        out = compress_text(diff_content, target_chars=400, content_type=["diff"])
        assert isinstance(out, str)
        assert len(out) > 0


# ── Error-path robustness ──────────────────────────────────────────────

@requires_headroom
class TestRobustness:
    def test_malformed_diff_does_not_crash(self):
        """Feeding non-diff content to diff specialist should not raise."""
        # Router won't pick diff without the hint — force it by passing ['diff']
        bad_diff = "this is not a diff at all " * 100
        out = compress_text(bad_diff, target_chars=200, content_type=["diff"])
        assert isinstance(out, str)
        assert len(out) > 0

    def test_binary_like_content_does_not_crash(self):
        """Compressing what looks like binary bytes should not raise."""
        # Only printable characters — compress_text takes str, not bytes
        weird = "\x01\x02\x03 lots of weird characters " * 50
        out = compress_text(weird, target_chars=200)
        assert isinstance(out, str)

    def test_very_long_content_does_not_crash(self):
        """Kompress chunks at ~350 words; verify multi-chunk input works."""
        long_content = "The quick brown fox jumps over the lazy dog. " * 500
        out = compress_text(long_content, target_chars=300, content_type=["text"])
        assert isinstance(out, str)
        assert len(out) > 0

"""scripts/ingest_all.py no longer ships a machine-specific default source
list (six hardcoded personal drive paths). Without ``--sources`` (and
without ``--agent-source`` in sharded mode) the script now exits with a
usage message instead of silently completing a 0-file, 0-gene run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable (same convention as tests/test_dense_pool_floor.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import ingest_all


def test_no_sources_exits_with_the_usage_message(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ingest_all.py"])
    with pytest.raises(SystemExit) as exc_info:
        ingest_all.main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--sources" in captured.err


def test_agent_source_alone_exits_in_monolithic_mode(monkeypatch, capsys):
    """``--agent-source`` is ignored outside sharded mode, so on its own it
    would otherwise reach a 0-file run rather than the usage message."""
    monkeypatch.setattr(sys, "argv", ["ingest_all.py", "--agent-source", "max=/srv/max"])
    with pytest.raises(SystemExit) as exc_info:
        ingest_all.main()
    assert exc_info.value.code == 2
    assert "--sources" in capsys.readouterr().err


def test_agent_source_alone_is_accepted_in_sharded_mode(monkeypatch):
    """The sharded path is the one place ``--agent-source`` stands alone."""
    monkeypatch.setattr(sys, "argv",
                        ["ingest_all.py", "--sharded", "--agent-source", "max=/srv/max"])
    monkeypatch.setattr(ingest_all, "CpuTagger", lambda: object())
    monkeypatch.setattr(ingest_all, "CodonChunker", lambda: object())
    calls = []
    monkeypatch.setattr(ingest_all, "_run_sharded",
                        lambda args, tagger, chunker: calls.append(args))
    ingest_all.main()
    assert len(calls) == 1
    assert calls[0].agent_source == ["max=/srv/max"]


def test_agent_only_does_not_probe_default_corpus_paths(monkeypatch, tmp_path):
    """An agent-only sharded run walks only the agent path; the removed
    default corpus roots are never probed."""
    from types import SimpleNamespace

    seen = []
    conn = SimpleNamespace(
        execute=lambda *a: SimpleNamespace(fetchall=lambda: []),
        close=lambda: None,
    )
    monkeypatch.setattr(ingest_all, "open_main_db", lambda *a: conn)
    monkeypatch.setattr(ingest_all, "init_main_db", lambda *a: None)
    real_isdir = ingest_all.os.path.isdir

    def source_isdir(root):
        # Keep pathlib's own directory checks real.
        if isinstance(root, Path):
            return real_isdir(root)
        seen.append(root)
        return False

    monkeypatch.setattr(ingest_all.os.path, "isdir", source_isdir)
    args = SimpleNamespace(genomes_root=str(tmp_path), sources=None,
                           agent_source=["agent=memory"], skip_models=False)
    ingest_all._run_sharded(args, object(), object())
    assert seen == ["memory"]


def test_module_defines_no_default_source_list():
    assert not hasattr(ingest_all, "_DEFAULT_SOURCES")


def test_explicit_sources_parse_unchanged():
    assert ingest_all._parse_source_arg("D:/Projects=projects:participant") == (
        "D:/Projects", "projects", "participant",
    )
    assert ingest_all._parse_source_arg("/srv/corpus") == (
        "/srv/corpus", "corpus", "reference",
    )

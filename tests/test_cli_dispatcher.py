"""Tests for the top-level `cymatix` CLI dispatcher (no subcommand work yet)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from cymatix_context.cli import dispatcher
from cymatix_context.cli import main
from tests.conftest import run_cli as _run


def test_no_args_prints_help_and_returns_nonzero():
    rc, out, err = _run([])
    assert rc != 0, "running `cymatix` with no subcommand should be an error"
    combined = (out + err).lower()
    assert "usage:" in combined
    assert "cymatix" in combined


def test_help_flag_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    # argparse exits 0 on --help
    assert exc.value.code == 0


def test_unknown_subcommand_returns_two():
    with pytest.raises(SystemExit) as exc:
        main(["does-not-exist"])
    # argparse exits 2 on unknown choice
    assert exc.value.code == 2


def test_main_consults_sys_argv_when_argv_is_none(monkeypatch):
    """Regression for commit 7647b72: main() with no argument must read sys.argv[1:].

    Previously ``main()`` (no argument) bypassed the `if argv is None: argv = sys.argv[1:]`
    branch in some entry-point setups, so the installed `cymatix` console-script
    crashed when invoked without explicit args. We exercise the path by setting
    sys.argv to a benign ``["cymatix", "--help"]`` value and calling ``main()``
    with no positional argument — argparse should still see ``--help`` and
    SystemExit(0). If main() failed to consult sys.argv, it would instead see
    an empty argv and return EXIT_ERROR via the no-args branch.
    """
    import sys
    monkeypatch.setattr(sys, "argv", ["cymatix", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    # argparse --help → exit 0; proves sys.argv was consulted (the no-args
    # branch would have returned EXIT_ERROR=1 without raising SystemExit).
    assert exc.value.code == 0


# ── prog derives from the invoked script name (rename cosmetic fix) ────


@pytest.mark.parametrize("argv0,expected_prog", [
    ("/usr/local/bin/cymatix", "cymatix"),
    (r"C:\env\Scripts\cymatix.exe", "cymatix"),
])
def test_parser_prog_derives_from_argv0(monkeypatch, argv0, expected_prog):
    """The console script should show its own invoked name in usage/help
    output (P3 finding)."""
    monkeypatch.setattr(sys, "argv", [argv0, "--help"])
    parser = dispatcher._build_parser()
    assert parser.prog == expected_prog


def test_parser_prog_falls_back_when_argv_empty(monkeypatch):
    monkeypatch.setattr(sys, "argv", [])
    parser = dispatcher._build_parser()
    assert parser.prog == "cymatix"


def _find_console_script(script_name: str) -> str | None:
    # Running an environment's Python by absolute path does not activate it.
    return shutil.which(script_name, path=sysconfig.get_path("scripts")) or shutil.which(script_name)


def test_installed_console_script_found_without_scripts_on_path(monkeypatch, tmp_path):
    """Invoking an environment's Python directly must still test its launcher."""
    scripts = Path(sysconfig.get_path("scripts"))
    installed = shutil.which("cymatix", path=str(scripts))
    if installed is None:
        pytest.skip(f"cymatix console script not installed in {scripts}")
    monkeypatch.setenv("PATH", str(tmp_path))

    assert _find_console_script("cymatix") == installed


@pytest.mark.parametrize("script_name,expected_prog", [
    ("cymatix", "cymatix"),
])
def test_installed_console_script_prog_matches_invoked_name(script_name, expected_prog):
    """Smoke-test the real installed console script: `cymatix --help` must
    say `usage: cymatix`."""
    exe = _find_console_script(script_name)
    if exe is None:
        pytest.skip(f"{script_name} console script not installed in this Python environment or on PATH")
    # The environment can be shared with another checkout. Exercise this
    # checkout's code through the real installed launcher in that case too.
    env = os.environ.copy()
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (repo, env.get("PYTHONPATH"))))
    proc = subprocess.run(
        [exe, "--help"], capture_output=True, text=True, timeout=60,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.returncode == 0, proc.stderr
    first_line = proc.stdout.splitlines()[0]
    assert first_line.startswith(f"usage: {expected_prog}"), first_line


# ── subcommand parsers follow the invoked alias too (0.8.0 rename) ─────


_SUBCOMMAND_MODULES = [
    ("query", "cmd_query"),
    ("packet", "cmd_packet"),
    ("refresh-targets", "cmd_refresh_targets"),
    ("gene", "cmd_gene"),
    ("neighbors", "cmd_neighbors"),
    ("ingest", "cmd_ingest"),
    ("config", "cmd_config"),
    ("diag", "cmd_diag"),
    ("status", "cmd_status"),
]


@pytest.mark.parametrize("alias", ["cymatix"])
@pytest.mark.parametrize("sub,module_name", _SUBCOMMAND_MODULES)
def test_subcommand_parser_prog_derives_from_argv0(monkeypatch, alias, sub, module_name):
    """`cymatix query --help` must say `usage: cymatix query`, deriving the
    prog from the invoked argv0."""
    import importlib
    monkeypatch.setattr(sys, "argv", [rf"C:\env\Scripts\{alias}.exe", sub, "--help"])
    mod = importlib.import_module(f"cymatix_context.cli.{module_name}")
    parser = mod._build_parser()
    assert parser.prog == f"{alias} {sub}"


@pytest.mark.parametrize("sub,module_name", _SUBCOMMAND_MODULES)
def test_subcommand_descriptions_say_cymatix_not_helix(monkeypatch, sub, module_name):
    """Post-rename, no subcommand --help description should brand itself helix."""
    import importlib
    monkeypatch.setattr(sys, "argv", ["cymatix", sub])
    mod = importlib.import_module(f"cymatix_context.cli.{module_name}")
    parser = mod._build_parser()
    assert "helix" not in (parser.description or "").lower(), parser.description

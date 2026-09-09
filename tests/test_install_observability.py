"""Tests for the bootstrap script's Python helper entry-points.

Both install-native-observability.ps1 and .sh delegate hash verification
and archive extraction to a Python helper module so the platform-specific
shell wrapper stays small and the testable surface is one place.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def _real_sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def test_verify_hash_accepts_match(tmp_path):
    from cymatix_context.launcher._install_helpers import verify_hash
    f = tmp_path / "binary.bin"
    f.write_bytes(b"hello world")
    expected = _real_sha256(f)
    # Returns None on success (no raise).
    verify_hash(f, expected)


def test_verify_hash_raises_on_mismatch(tmp_path):
    from cymatix_context.launcher._install_helpers import (
        HashMismatch,
        verify_hash,
    )
    f = tmp_path / "binary.bin"
    f.write_bytes(b"corrupt")
    with pytest.raises(HashMismatch):
        verify_hash(f, "0" * 64)


def test_verify_hash_rejects_placeholder(tmp_path):
    """A `TODO_<platform>` placeholder must NOT be accepted as a valid hash.

    Bootstrap path: refuses to install on platforms where the row is still
    placeholder, so a Linux user running this on day 1 (before §11.6 work
    lands) gets a clean error instead of an unverified binary.
    """
    from cymatix_context.launcher._install_helpers import (
        HashPlaceholder,
        verify_hash,
    )
    f = tmp_path / "binary.bin"
    f.write_bytes(b"hello")
    with pytest.raises(HashPlaceholder):
        verify_hash(f, "TODO_linux_amd64")


def test_should_skip_when_existing_binary_matches(tmp_path):
    """Idempotency: present-and-correct binary returns True from
    should_skip; download is not re-run."""
    from cymatix_context.launcher._install_helpers import should_skip
    f = tmp_path / "binary.bin"
    f.write_bytes(b"present and correct")
    expected = _real_sha256(f)
    assert should_skip(f, expected) is True


def test_should_skip_false_when_binary_absent(tmp_path):
    from cymatix_context.launcher._install_helpers import should_skip
    assert should_skip(tmp_path / "nope.bin", "0" * 64) is False


def test_should_skip_false_when_hash_drifts(tmp_path):
    """Version bump → hash drift → re-download triggers."""
    from cymatix_context.launcher._install_helpers import should_skip
    f = tmp_path / "binary.bin"
    f.write_bytes(b"old version")
    assert should_skip(f, "0" * 64) is False


# ── Script-presence / parseability checks ───────────────────────────
# These guarantee the platform-specific wrappers exist alongside the
# Python helpers tested above. They don't try to invoke the install
# end-to-end (network + extraction), but they DO verify the shells parse
# without syntax errors so a typo doesn't slip through.

PS_SCRIPT = REPO / "scripts" / "install-native-observability.ps1"
SH_SCRIPT = REPO / "scripts" / "install-native-observability.sh"


def _find_bash() -> str | None:
    """Find a host shell, without launching Windows' WSL forwarding shims."""
    if sys.platform != "win32":
        return shutil.which("bash")

    shim_dirs = set()
    for variable in ("SystemRoot", "WINDIR"):
        if root := os.environ.get(variable):
            shim_dirs.update((Path(root) / name).resolve() for name in ("System32", "Sysnative"))
    if local := os.environ.get("LOCALAPPDATA"):
        shim_dirs.add((Path(local) / "Microsoft" / "WindowsApps").resolve())

    search_dirs = [Path(entry) for entry in os.get_exec_path() if entry]
    # Git for Windows commonly puts only Git/cmd on PATH, not Git/bin.
    if git := shutil.which("git"):
        git_root = Path(git).resolve().parent.parent
        search_dirs.extend((git_root / "bin", git_root / "usr" / "bin"))
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        if root := os.environ.get(variable):
            search_dirs.append(Path(root) / "Git" / "bin")
    if local:
        search_dirs.append(Path(local) / "Programs" / "Git" / "bin")

    for directory in search_dirs:
        bash = shutil.which("bash", path=str(directory))
        if bash and Path(bash).resolve().parent not in shim_dirs:
            return bash
    return None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows executable discovery")
@pytest.mark.parametrize("native_source", ["path", "git", "absent"])
def test_bash_discovery_excludes_wsl_shims(monkeypatch, tmp_path, native_source):
    """An earlier WSL shim must not hide an available native Bash."""
    windows = tmp_path / "Windows"
    shim_dir = windows / "System32"
    git_cmd = tmp_path / "Git" / "cmd"
    native_dir = tmp_path / "Git" / "bin"
    for directory in (shim_dir, git_cmd, native_dir):
        directory.mkdir(parents=True)
    (shim_dir / "bash.EXE").touch()
    (git_cmd / "git.EXE").touch()
    if native_source != "absent":
        (native_dir / "bash.EXE").touch()
    monkeypatch.setenv("WINDIR", str(windows))
    monkeypatch.setenv("SystemRoot", str(windows))
    monkeypatch.setenv("PATHEXT", ".EXE")
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "unused"))
    search_dirs = [shim_dir, native_dir if native_source == "path" else git_cmd]
    monkeypatch.setenv("PATH", os.pathsep.join(map(str, search_dirs)))

    bash = _find_bash()

    if native_source == "absent":
        assert bash is None
    else:
        assert bash is not None
        assert Path(bash) == native_dir / "bash.EXE"


def test_powershell_install_script_exists():
    assert PS_SCRIPT.exists(), f"missing: {PS_SCRIPT}"


def test_bash_install_script_exists():
    assert SH_SCRIPT.exists(), f"missing: {SH_SCRIPT}"


def test_bash_install_script_parses():
    """`bash -n` checks syntax without executing. Skipped if host Bash missing."""
    bash = _find_bash()
    if bash is None:
        pytest.skip("native Bash not available (Windows WSL shims are not host shells)")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # Stdin avoids passing a Windows path to a POSIX shell. Read in text
    # mode to normalize checkout line endings, then send bytes to retain LF.
    proc = subprocess.run(
        [bash, "-n"],
        input=SH_SCRIPT.read_text(encoding="utf-8").encode("utf-8"),
        capture_output=True,
        timeout=10,
        creationflags=creationflags,
    )
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr.decode(errors='replace')}"


def test_powershell_install_script_handles_targz():
    """Plan-fix carry-forward: Tempo ships .tar.gz on Windows, NOT .zip.

    The script must dispatch on archive extension rather than assuming
    every Windows artifact is a zip.
    """
    text = PS_SCRIPT.read_text(encoding="utf-8")
    assert ".zip" in text, "PowerShell script missing .zip branch"
    assert "tar" in text.lower(), "PowerShell script missing tar branch (Tempo Windows is .tar.gz)"


def test_powershell_install_script_handles_loki_archive_name():
    """Loki's Windows archive contains the binary as loki-windows-amd64.exe,
    not loki.exe. Naive Get-ChildItem -Filter loki.exe misses it and the
    install fails with 'loki.exe not found inside ... cymatix-native-otel-loki.zip'.

    Caught a real regression. Pin: the script must special-case Loki's
    in-archive filename. Copy-Item renames it to the canonical on-disk
    name so binary_path() in the supervisor still resolves loki.exe.
    """
    text = PS_SCRIPT.read_text(encoding="utf-8")
    assert "loki-windows-amd64.exe" in text, (
        "PowerShell install script must handle Loki's in-archive filename "
        "(loki-windows-amd64.exe) as a special case. The archive does NOT "
        "contain loki.exe — Get-ChildItem -Filter loki.exe will miss it."
    )


def test_powershell_install_script_names_temp_archive_with_real_extension():
    """Expand-Archive (the PS1 unzipper) requires the input file to literally
    end in .zip; a generic .tmp suffix fails with
    NotSupportedArchiveFileExtension. tar.exe sniffs content and tolerates
    any extension, so a generic .tmp suffix masked this for tar.gz only.

    Caught a real regression where prometheus (a .zip download) failed
    with: 'Expand-Archive : .tmp is not a supported archive file format'.

    Pin: the script MUST derive the temp archive's extension from the
    URL, not hardcode .tmp.
    """
    text = PS_SCRIPT.read_text(encoding="utf-8")
    # The simple-minded form `cymatix-native-otel-$svc.tmp` was the bug.
    # Allow the substring inside a fallback branch but require explicit
    # `.zip` and `.tar.gz` cases too.
    assert '$url.EndsWith(".zip")' in text, (
        "PowerShell install script must derive the archive extension "
        "from the URL (e.g., $url.EndsWith(\".zip\") branch). "
        "Hardcoding .tmp breaks Expand-Archive for .zip downloads."
    )
    assert '$url.EndsWith(".tar.gz")' in text, (
        "PowerShell install script must handle the .tar.gz extension "
        "explicitly so the named temp file has the right suffix."
    )


def test_powershell_install_script_resolves_repo_root_in_body_not_param():
    r"""Param default `(Resolve-Path "$PSScriptRoot\..").Path` is unreliable
    under `powershell.exe -File <script>` because $PSScriptRoot can be
    empty during param-block evaluation. When empty, "$PSScriptRoot\.."
    collapses to "\.." which Resolve-Path interprets as the drive root,
    making $RepoRoot equal to "F:\" instead of the repo path.

    Caught a real regression where the tray-launched install hit
    `F:\tools\native-otel\.versions not found` because the resolution
    happened in the param default.

    Pin: the file MUST NOT contain the broken pattern in the param block
    AND MUST resolve via $MyInvocation.MyCommand.Path in the script body
    (which is reliably set after param parsing).
    """
    text = PS_SCRIPT.read_text(encoding="utf-8")
    assert 'Resolve-Path "$PSScriptRoot' not in text, (
        "PowerShell install script uses the unreliable "
        "(Resolve-Path \"$PSScriptRoot\\..\") pattern in the param block. "
        "Move repo-root resolution into the script body using "
        "$MyInvocation.MyCommand.Path to ensure it works under -File."
    )
    assert "$MyInvocation.MyCommand.Path" in text, (
        "PowerShell install script must resolve repo root via "
        "$MyInvocation.MyCommand.Path in the script body."
    )


def test_powershell_install_script_is_ascii_only():
    """PowerShell 5.1 (the default `powershell.exe` on Windows) reads .ps1
    files without a BOM as ANSI/CP1252. UTF-8 multi-byte chars (em-dashes,
    section signs, smart quotes) become mojibake that breaks the parser
    mid-string. Pin the script to ASCII so it parses on every Windows box.

    Caught a real regression where an em-dash in a Write-Error string
    broke the install action at runtime.
    """
    script = REPO / "scripts" / "install-native-observability.ps1"
    text = script.read_text(encoding="utf-8")
    non_ascii = [
        (i + 1, c)
        for i, line in enumerate(text.splitlines())
        for c in line
        if ord(c) > 127
    ]
    assert not non_ascii, (
        f"PowerShell install script contains non-ASCII chars that will "
        f"mojibake under powershell.exe (Windows PowerShell 5.1, no BOM): "
        f"{non_ascii[:5]}. Use ASCII alternatives (-- for em-dash, "
        f"'Section' for §)."
    )


@pytest.mark.skipif(sys.platform != "win32", reason="powershell.exe only on Windows")
def test_powershell_install_script_parses_cleanly():
    """Spawn powershell.exe in parse-only mode to verify the script is
    syntactically valid. Mirrors `bash -n` for the .sh sibling. Catches
    syntax errors AND encoding-induced parse failures (e.g., the em-dash
    bug that the ASCII-only test above pins, viewed from the parser side).
    """
    script = REPO / "scripts" / "install-native-observability.ps1"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # `[scriptblock]::Create` parses without executing — equivalent of
    # bash -n. Exit 0 on parse success, non-zero on parse error.
    proc = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"$null = [scriptblock]::Create((Get-Content -Raw -Path '{script}'))",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=creationflags,
    )
    assert proc.returncode == 0, (
        f"PowerShell parse failed:\nstderr={proc.stderr}\nstdout={proc.stdout}"
    )


def test_install_helpers_cli_verify_hash_returns_zero_on_match(tmp_path):
    """The CLI surface is what the shell scripts shell out to. Smoke-test
    it end-to-end so wiring drift between Python and shell is caught."""
    f = tmp_path / "binary.bin"
    f.write_bytes(b"cli round-trip")
    expected = _real_sha256(f)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        [sys.executable, "-m", "cymatix_context.launcher._install_helpers",
         "verify-hash", str(f), expected],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=creationflags,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"


def test_install_helpers_cli_verify_hash_returns_nonzero_on_mismatch(tmp_path):
    f = tmp_path / "binary.bin"
    f.write_bytes(b"cli mismatch")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        [sys.executable, "-m", "cymatix_context.launcher._install_helpers",
         "verify-hash", str(f), "0" * 64],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=creationflags,
    )
    assert proc.returncode != 0
    assert "MISMATCH" in proc.stderr or "ERROR" in proc.stderr


def test_install_helpers_cli_should_skip_returns_zero_on_match(tmp_path):
    """should-skip succeeds (exit 0) when binary exists and hash matches —
    install script uses this to skip the download."""
    f = tmp_path / "binary.bin"
    f.write_bytes(b"cli should-skip match")
    expected = _real_sha256(f)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        [sys.executable, "-m", "cymatix_context.launcher._install_helpers",
         "should-skip", str(f), expected],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=creationflags,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    # Critical: silent on success too — no informational chatter.
    assert proc.stderr == ""


def test_install_helpers_cli_should_skip_silent_on_missing_file(tmp_path):
    """should-skip MUST be silent (no stderr) when the file doesn't exist.
    PowerShell 5.1 with ErrorActionPreference=Stop turns native-command
    stderr into a script-terminating error, which would defeat the
    install script's `if ($LASTEXITCODE -eq 0)` skip-gate. Pin this:
    missing file → exit 1, stderr empty."""
    missing = tmp_path / "does-not-exist.bin"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        [sys.executable, "-m", "cymatix_context.launcher._install_helpers",
         "should-skip", str(missing), "0" * 64],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=creationflags,
    )
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}"
    assert proc.stderr == "", (
        f"should-skip must be silent on missing file (PS5.1 stderr-as-error "
        f"contract); got stderr={proc.stderr!r}"
    )


def test_install_helpers_cli_should_skip_silent_on_hash_drift(tmp_path):
    """Same silence contract for hash-drift case — also an expected
    "please download" outcome, not an error condition."""
    f = tmp_path / "binary.bin"
    f.write_bytes(b"cli drift")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        [sys.executable, "-m", "cymatix_context.launcher._install_helpers",
         "should-skip", str(f), "0" * 64],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=creationflags,
    )
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}"
    assert proc.stderr == "", (
        f"should-skip must be silent on hash drift; got stderr={proc.stderr!r}"
    )


def test_install_scripts_use_should_skip_not_verify_hash():
    """Install scripts MUST use should-skip for the existing-binary check
    so PowerShell 5.1's stderr-as-error semantics don't trip the install
    flow. verify-hash stays available for downloaded-archive verification
    where missing IS a real error."""
    ps_text = PS_SCRIPT.read_text(encoding="utf-8")
    sh_text = (REPO / "scripts" / "install-native-observability.sh").read_text(encoding="utf-8")
    # The existing-binary check should use should-skip.
    assert "should-skip $absPath $expected" in ps_text or 'should-skip "$abspath"' in ps_text or "should-skip $absPath" in ps_text, (
        "PowerShell install script must call `should-skip` for the existing-binary check."
    )
    assert 'should-skip "$abspath"' in sh_text, (
        "Bash install script must call `should-skip` for the existing-binary check."
    )

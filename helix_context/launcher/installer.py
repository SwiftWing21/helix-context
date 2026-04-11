"""
Service installer — platform-aware wrapper around the deploy/ templates.

Supports three install targets:

    Linux    → systemd user unit at ~/.config/systemd/user/helix-launcher.service
    macOS    → launchd plist at ~/Library/LaunchAgents/com.swiftwing21.helix-launcher.plist
    Windows  → prints NSSM recipe (cannot bundle NSSM due to licensing +
               download requirements; user runs the steps themselves)

The installer only **writes the file**. It does NOT run systemctl enable,
launchctl load, or nssm install — those are side effects that deserve
explicit user consent. The installer prints the exact next command so
the user can paste it into their own shell.

Invoked via the CLI:

    helix-launcher install-service
    helix-launcher uninstall-service

Each subcommand walks through:

    1. Detect platform
    2. Locate the template file in the installed package (or repo `deploy/`)
    3. Substitute placeholders (USER, ABSOLUTE_PATH_TO_HELIX_LAUNCHER, etc.)
    4. Write the target file with sane perms
    5. Print the next-step command
"""

from __future__ import annotations

import getpass
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger("helix.launcher.installer")


# ── platform detection ────────────────────────────────────────────

def current_platform() -> str:
    """Return 'linux', 'darwin', 'win32', or 'unsupported'."""
    p = sys.platform
    if p.startswith("linux"):
        return "linux"
    if p == "darwin":
        return "darwin"
    if p == "win32":
        return "win32"
    return "unsupported"


# ── template discovery ───────────────────────────────────────────

def _deploy_dir() -> Optional[Path]:
    """Locate the deploy/ directory relative to the installed package.

    The templates ship in the source tree at `<repo>/deploy/`. When
    installed via pip, the templates are not inside the wheel by
    default (deploy/ is not part of the package). Fall back to
    walking up from __file__ to find a sibling `deploy/`.
    """
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent.parent, here.parent.parent.parent.parent]:
        deploy = candidate / "deploy"
        if deploy.exists() and deploy.is_dir():
            return deploy
    return None


def _find_template(platform: str) -> Optional[Path]:
    """Return the template path for the given platform, or None if missing."""
    deploy = _deploy_dir()
    if deploy is None:
        return None
    if platform == "linux":
        return deploy / "systemd" / "helix-launcher.service"
    if platform == "darwin":
        return deploy / "launchd" / "com.swiftwing21.helix-launcher.plist"
    return None  # Windows has no installable template (see install_service below)


# ── target paths ─────────────────────────────────────────────────

def target_path(platform: str) -> Optional[Path]:
    """Return the path the service file should be written to."""
    if platform == "linux":
        return Path.home() / ".config" / "systemd" / "user" / "helix-launcher.service"
    if platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / "com.swiftwing21.helix-launcher.plist"
    return None


# ── helix-launcher binary discovery ──────────────────────────────

def find_launcher_binary() -> Optional[Path]:
    """Locate the helix-launcher console script on the current PATH."""
    path = shutil.which("helix-launcher")
    if path:
        return Path(path)
    # Fall back to the same Python that's running us + its Scripts/bin dir.
    py = Path(sys.executable)
    for candidate in (
        py.parent / "helix-launcher",
        py.parent / "helix-launcher.exe",
        py.parent / "Scripts" / "helix-launcher.exe",
    ):
        if candidate.exists():
            return candidate
    return None


# ── template substitution ────────────────────────────────────────

def _substitute(template_text: str, replacements: dict) -> str:
    out = template_text
    for key, value in replacements.items():
        out = out.replace(key, str(value))
    return out


def _linux_replacements(launcher_path: Path) -> dict:
    # The systemd template uses %h (systemd's home placeholder) so we
    # only need to swap the ExecStart path.
    return {
        "%h/.local/share/helix/.venv/bin/helix-launcher --no-browser":
            f"{launcher_path} --no-browser",
    }


def _darwin_replacements(launcher_path: Path) -> dict:
    user = getpass.getuser()
    return {
        "/usr/local/bin/helix-launcher": str(launcher_path),
        "/Users/USERNAME": str(Path.home()),
    }


# ── install ──────────────────────────────────────────────────────

def install_service(dry_run: bool = False) -> Tuple[bool, str]:
    """Install the service file for the current platform.

    Returns (success, message). On Windows, returns (False, instructions)
    because there is no NSSM auto-install path; the message contains the
    recipe.
    """
    platform = current_platform()
    if platform == "unsupported":
        return False, f"Unsupported platform: {sys.platform}"

    if platform == "win32":
        msg = (
            "Windows service installation is not automated — NSSM must be\n"
            "installed and the service registered manually. Recipe:\n\n"
            "    1. Install NSSM:\n"
            "         choco install nssm   (or scoop install nssm)\n\n"
            "    2. Find your helix-launcher.exe:\n"
            "         where helix-launcher\n\n"
            "    3. Register as a Windows service (run as admin):\n"
            "         nssm install HelixLauncher\n"
            "       ... then fill in the NSSM GUI with the launcher path + --no-browser\n"
            "       See deploy/windows/README.md for the complete walkthrough.\n\n"
            "    4. Start the service:\n"
            "         nssm start HelixLauncher"
        )
        return False, msg

    template_path = _find_template(platform)
    if template_path is None or not template_path.exists():
        return False, (
            f"Template not found for {platform}. "
            "Is this install missing the deploy/ directory? "
            "Templates ship in the source tree — try reinstalling from source."
        )

    launcher_path = find_launcher_binary()
    if launcher_path is None:
        return False, (
            "Could not locate the `helix-launcher` executable on PATH. "
            "Install the launcher extras first:\n\n"
            "    pip install helix-context[launcher]"
        )

    target = target_path(platform)
    assert target is not None  # platform check above ensures this

    # Substitute placeholders
    template_text = template_path.read_text(encoding="utf-8")
    if platform == "linux":
        content = _substitute(template_text, _linux_replacements(launcher_path))
    else:  # darwin
        content = _substitute(template_text, _darwin_replacements(launcher_path))

    if dry_run:
        return True, (
            f"[dry run] Would write {target}\n"
            f"[dry run] Template: {template_path}\n"
            f"[dry run] Launcher: {launcher_path}\n"
            f"\n--- content preview ---\n{content[:800]}"
            f"{'...' if len(content) > 800 else ''}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    # Print next-step command based on platform
    if platform == "linux":
        next_steps = (
            f"Wrote {target}\n\n"
            "Next steps:\n\n"
            "    systemctl --user daemon-reload\n"
            "    systemctl --user enable --now helix-launcher.service\n\n"
            "Verify:\n\n"
            "    systemctl --user status helix-launcher.service\n"
            "    journalctl --user -u helix-launcher.service -f\n\n"
            "Open http://127.0.0.1:11438/ in your browser."
        )
    else:  # darwin
        next_steps = (
            f"Wrote {target}\n\n"
            "Next steps:\n\n"
            f"    launchctl load {target}\n\n"
            "Verify:\n\n"
            "    launchctl list | grep helix-launcher\n"
            "    tail -f ~/Library/Logs/helix-launcher.log\n\n"
            "Open http://127.0.0.1:11438/ in your browser."
        )

    return True, next_steps


# ── uninstall ────────────────────────────────────────────────────

def uninstall_service(dry_run: bool = False) -> Tuple[bool, str]:
    """Remove the service file for the current platform.

    Does NOT run the platform's disable command — the user is
    responsible for stopping the running service first. We just
    unlink the file.
    """
    platform = current_platform()
    if platform == "unsupported":
        return False, f"Unsupported platform: {sys.platform}"

    if platform == "win32":
        msg = (
            "Windows uninstall is not automated. Run:\n\n"
            "    nssm stop HelixLauncher\n"
            "    nssm remove HelixLauncher confirm"
        )
        return False, msg

    target = target_path(platform)
    assert target is not None

    if not target.exists():
        return False, f"Not installed — {target} does not exist"

    if dry_run:
        return True, f"[dry run] Would remove {target}"

    # Warn the user to stop the service first if applicable
    if platform == "linux":
        post = (
            f"Removed {target}\n\n"
            "If the service was running, stop and disable it:\n\n"
            "    systemctl --user disable --now helix-launcher.service\n"
            "    systemctl --user daemon-reload"
        )
    else:  # darwin
        post = (
            f"Removed {target}\n\n"
            "If the service was loaded, unload it:\n\n"
            f"    launchctl unload {target}\n"
            "    (it's already removed, so the unload may say 'No such process')"
        )

    target.unlink()
    return True, post

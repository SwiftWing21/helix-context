"""
Tests for helix_context.launcher.installer — platform detection,
template substitution, install + uninstall paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from helix_context.launcher import installer


class TestCurrentPlatform:
    def test_returns_one_of_known_values(self):
        assert installer.current_platform() in ("linux", "darwin", "win32", "unsupported")


class TestTargetPath:
    def test_linux_target(self):
        assert installer.target_path("linux") == Path.home() / ".config" / "systemd" / "user" / "helix-launcher.service"

    def test_darwin_target(self):
        assert installer.target_path("darwin") == Path.home() / "Library" / "LaunchAgents" / "com.swiftwing21.helix-launcher.plist"

    def test_win32_target_is_none(self):
        assert installer.target_path("win32") is None


class TestFindTemplate:
    def test_linux_template_exists(self):
        p = installer._find_template("linux")
        assert p is not None
        assert p.exists()
        assert p.name == "helix-launcher.service"

    def test_darwin_template_exists(self):
        p = installer._find_template("darwin")
        assert p is not None
        assert p.exists()
        assert p.name == "com.swiftwing21.helix-launcher.plist"

    def test_win32_template_is_none(self):
        assert installer._find_template("win32") is None


class TestFindLauncherBinary:
    def test_returns_none_or_path(self):
        result = installer.find_launcher_binary()
        assert result is None or isinstance(result, Path)


class TestInstallWin32:
    def test_windows_install_returns_instructions(self):
        with patch("helix_context.launcher.installer.current_platform", return_value="win32"):
            ok, msg = installer.install_service()
        assert ok is False
        assert "NSSM" in msg
        assert "choco install nssm" in msg or "scoop install nssm" in msg

    def test_windows_uninstall_returns_instructions(self):
        with patch("helix_context.launcher.installer.current_platform", return_value="win32"):
            ok, msg = installer.uninstall_service()
        assert ok is False
        assert "nssm remove" in msg


class TestInstallUnsupported:
    def test_unknown_platform_errors(self):
        with patch("helix_context.launcher.installer.current_platform", return_value="unsupported"):
            ok, msg = installer.install_service()
        assert ok is False
        assert "Unsupported platform" in msg


class TestInstallMissingBinary:
    def test_linux_install_without_binary_errors(self):
        with patch("helix_context.launcher.installer.current_platform", return_value="linux"):
            with patch("helix_context.launcher.installer.find_launcher_binary", return_value=None):
                ok, msg = installer.install_service()
        assert ok is False
        assert "helix-launcher" in msg.lower()


class TestLinuxDryRun:
    def test_linux_dry_run_does_not_write_file(self, tmp_path):
        fake_target = tmp_path / "helix-launcher.service"
        fake_launcher = tmp_path / "helix-launcher"
        fake_launcher.write_text("#!/bin/bash\necho hi")
        fake_launcher.chmod(0o755)

        with patch("helix_context.launcher.installer.current_platform", return_value="linux"):
            with patch("helix_context.launcher.installer.target_path", return_value=fake_target):
                with patch("helix_context.launcher.installer.find_launcher_binary", return_value=fake_launcher):
                    ok, msg = installer.install_service(dry_run=True)

        assert ok is True
        assert "dry run" in msg.lower()
        assert not fake_target.exists()


class TestLinuxInstallAndUninstall:
    def test_linux_install_writes_file_and_substitutes_path(self, tmp_path):
        fake_target = tmp_path / "systemd" / "user" / "helix-launcher.service"
        fake_launcher = tmp_path / "helix-launcher"
        fake_launcher.write_text("#!/bin/bash\necho hi")

        with patch("helix_context.launcher.installer.current_platform", return_value="linux"):
            with patch("helix_context.launcher.installer.target_path", return_value=fake_target):
                with patch("helix_context.launcher.installer.find_launcher_binary", return_value=fake_launcher):
                    ok, msg = installer.install_service()

        assert ok is True
        assert fake_target.exists()
        content = fake_target.read_text(encoding="utf-8")
        # The launcher path should be substituted in
        assert str(fake_launcher) in content
        assert "systemctl --user" in msg

    def test_linux_uninstall_removes_file(self, tmp_path):
        fake_target = tmp_path / "systemd" / "user" / "helix-launcher.service"
        fake_target.parent.mkdir(parents=True, exist_ok=True)
        fake_target.write_text("stub content")

        with patch("helix_context.launcher.installer.current_platform", return_value="linux"):
            with patch("helix_context.launcher.installer.target_path", return_value=fake_target):
                ok, msg = installer.uninstall_service()

        assert ok is True
        assert not fake_target.exists()
        assert "systemctl --user disable" in msg

    def test_linux_uninstall_errors_when_not_installed(self, tmp_path):
        fake_target = tmp_path / "not-there.service"
        with patch("helix_context.launcher.installer.current_platform", return_value="linux"):
            with patch("helix_context.launcher.installer.target_path", return_value=fake_target):
                ok, msg = installer.uninstall_service()
        assert ok is False
        assert "does not exist" in msg


class TestDarwinInstall:
    def test_darwin_install_substitutes_username_and_launcher_path(self, tmp_path):
        fake_target = tmp_path / "LaunchAgents" / "com.swiftwing21.helix-launcher.plist"
        fake_launcher = tmp_path / "bin" / "helix-launcher"
        fake_launcher.parent.mkdir(parents=True)
        fake_launcher.write_text("#!/bin/bash\necho hi")

        with patch("helix_context.launcher.installer.current_platform", return_value="darwin"):
            with patch("helix_context.launcher.installer.target_path", return_value=fake_target):
                with patch("helix_context.launcher.installer.find_launcher_binary", return_value=fake_launcher):
                    ok, msg = installer.install_service()

        assert ok is True
        assert fake_target.exists()
        content = fake_target.read_text(encoding="utf-8")
        assert str(fake_launcher) in content
        # USERNAME placeholder should be replaced (with $HOME, not the literal string)
        assert "/Users/USERNAME" not in content
        assert "launchctl load" in msg


class TestCLIIntegration:
    def test_install_service_command_dispatches(self, capsys):
        from helix_context.launcher import app as app_mod
        with patch("helix_context.launcher.installer.install_service",
                   return_value=(True, "ok message")):
            rc = app_mod.main(["install-service"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "ok message" in captured.out

    def test_uninstall_service_command_dispatches(self, capsys):
        from helix_context.launcher import app as app_mod
        with patch("helix_context.launcher.installer.uninstall_service",
                   return_value=(True, "removed")):
            rc = app_mod.main(["uninstall-service"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "removed" in captured.out

    def test_install_service_failure_returns_nonzero(self, capsys):
        from helix_context.launcher import app as app_mod
        with patch("helix_context.launcher.installer.install_service",
                   return_value=(False, "error")):
            rc = app_mod.main(["install-service"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "error" in captured.out

    def test_dry_run_flag_passed_through(self):
        from helix_context.launcher import app as app_mod
        with patch("helix_context.launcher.installer.install_service",
                   return_value=(True, "dry")) as mock_install:
            app_mod.main(["install-service", "--dry-run"])
        mock_install.assert_called_once_with(dry_run=True)

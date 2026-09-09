"""Doc tests for the native-observability-sidecar README scope.

Asserts that the docs say what the spec promises:

1. Root README:
   - has a "Native observability (default)" section under Quick Start.
   - mentions the bootstrap script run, tray-managed lifecycle, balloon
     notification, and the CYMATIX_OBSERVABILITY=0 opt-out.
   - demotes the docker-compose path to an "Advanced" footnote pointing at
     deploy/otel/README.md.
2. deploy/otel/README.md:
   - exists; framed as the alternate Docker path (not deprecated).
   - lists all five services with the same ports as the native sidecar.
   - links back to the native sidecar / spec for the shared-config story.
3. tools/native-otel/README.md:
   - still accurate post Tasks 3-10: per-user state dir, install script,
     idempotent re-run, .versions pinning.

The tests are textual — they do not import any module — so they run on
any host without the launcher extras installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Root README
# ---------------------------------------------------------------------------


# NOTE (post README-v3, commit 97e1ed6): the proof-first restructure moved
# the deep observability / tray / Docker content out of the root README and
# into docs/SETUP.md + docs/architecture/OBSERVABILITY.md. These four tests
# now pin the v3 contract: README keeps a compact "## Observability" section
# that links out, and the moved content must actually exist at the link
# targets so the chain README -> SETUP/OBSERVABILITY -> deploy/otel stays
# unbroken.


def test_root_readme_has_observability_section():
    body = _read("README.md")
    assert "## Observability" in body, (
        "Root README is missing the '## Observability' section"
    )
    assert "setup-grafana-telem" in body, (
        "Root README's Observability section must show the sidecar setup script"
    )
    assert "docs/architecture/OBSERVABILITY.md" in body, (
        "Root README must link to the full observability doc"
    )


def test_setup_doc_carries_tray_lifecycle_and_opt_out():
    # v3 moved tray lifecycle + opt-out documentation to docs/SETUP.md;
    # the README must link there and the content must exist.
    readme = _read("README.md")
    assert "docs/SETUP.md" in readme, "Root README must link to docs/SETUP.md"
    setup = _read("docs/SETUP.md")
    assert "tray" in setup.lower()
    assert "CYMATIX_OBSERVABILITY" in setup, (
        "docs/SETUP.md must document the CYMATIX_OBSERVABILITY=0 opt-out"
    )
    assert "tools/native-otel" in setup


def test_setup_doc_keeps_docker_as_alternate_path():
    # The docker-compose stack stays documented as the alternate (not
    # primary) path, now from docs/SETUP.md rather than the root README.
    setup = _read("docs/SETUP.md")
    assert "deploy/otel" in setup, (
        "docs/SETUP.md must point at the deploy/otel Docker stack"
    )
    assert "docker" in setup.lower()


def test_root_readme_preserves_canonical_launch_section():
    """Regression: don't rip out the launch/get-started content.

    README v3 renamed 'Quick Start' to 'Get started'; the tray entry point
    (start-cymatix-tray.bat) is documented in docs/SETUP.md.
    """
    body = _read("README.md")
    assert "## Get started" in body or "## Quick Start" in body, (
        "Root README must keep a Get started / Quick Start section"
    )
    setup = _read("docs/SETUP.md")
    assert "start-cymatix-tray.bat" in setup.lower()


# ---------------------------------------------------------------------------
# deploy/otel/README.md
# ---------------------------------------------------------------------------


def test_deploy_otel_readme_exists():
    assert (REPO / "deploy" / "otel" / "README.md").is_file(), (
        "deploy/otel/README.md must exist as the Docker-stack reference"
    )


def test_deploy_otel_readme_is_alternate_not_deprecated():
    body = _read("deploy/otel/README.md")
    # Frames itself as alternate / advanced, not deprecated.
    assert "alternate" in body.lower()
    assert "deprecated" not in body.lower(), (
        "deploy/otel/README.md must NOT call the Docker path deprecated"
    )


def test_deploy_otel_readme_lists_all_five_services():
    body = _read("deploy/otel/README.md").lower()
    for svc in ("otel", "prometheus", "tempo", "loki", "grafana"):
        assert svc in body, f"deploy/otel/README.md is missing service: {svc}"


def test_deploy_otel_readme_documents_shared_ports():
    body = _read("deploy/otel/README.md")
    # Same ports as native sidecar.
    for port in ("4317", "4318", "8889", "9090", "3200", "3100", "3000"):
        assert port in body, (
            f"deploy/otel/README.md is missing port {port}"
        )


def test_deploy_otel_readme_links_to_native_sidecar_or_spec():
    body = _read("deploy/otel/README.md")
    assert "tools/native-otel" in body or "native sidecar" in body.lower()
    # Spec back-reference.
    assert "2026-05-04-native-observability-sidecar-design" in body


# ---------------------------------------------------------------------------
# tools/native-otel/README.md
# ---------------------------------------------------------------------------


def test_native_otel_readme_still_accurate():
    body = _read("tools/native-otel/README.md")
    # Install script reference.
    assert "install-native-observability" in body
    # Per-user state dir story (platformdirs).
    assert "platformdirs" in body or "user_data_dir" in body
    # Pinned versions file.
    assert ".versions" in body
    # Idempotent re-run is called out.
    assert "idempotent" in body.lower() or "skip" in body.lower()


def test_docker_observability_ports_are_loopback_only():
    """Unauthenticated telemetry services must stay on the local host."""
    services = yaml.safe_load(_read("deploy/otel/docker-compose.yml"))["services"]
    ports = [(name, port) for name, service in services.items()
             for port in service.get("ports", [])]
    assert ports
    for name, port in ports:
        assert isinstance(port, str) and port.startswith("127.0.0.1:"), (name, port)


def test_docker_grafana_anonymous_access_is_opt_in():
    services = yaml.safe_load(_read("deploy/otel/docker-compose.yml"))["services"]
    environment = dict(item.split("=", 1) for item in services["grafana"]["environment"])
    assert environment["GF_AUTH_ANONYMOUS_ENABLED"] == "${CYMATIX_GRAFANA_ANON:-false}"
    assert environment["GF_SECURITY_ADMIN_PASSWORD"] == "${CYMATIX_GRAFANA_ADMIN_PASSWORD:-admin}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

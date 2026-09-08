"""Known TOML sections must be tables (issue #418)."""

import logging

import pytest

from cymatix_context.config import load_config
from cymatix_context.exceptions import ConfigError


@pytest.mark.parametrize("section", [
    "ribosome", "budget", "genome", "server", "encoder_daemon", "telemetry",
    "ingestion", "context", "cymatics", "retrieval", "abstain", "session",
    "plr", "headroom", "classifier", "know", "hardware", "vault", "synonyms",
    "mem_sync", "compressor", "knowledge_store",
])
@pytest.mark.parametrize("value", [
    '"not a table"', "7", "false", "[]", "[{ enabled = true }]",
], ids=["string", "integer", "boolean", "array", "table-array"])
def test_non_table_known_section_warns_and_preserves_other_settings(
    tmp_path, caplog, section, value,
):
    """Bad section shapes must not crash or discard unrelated valid settings."""
    path = tmp_path / "cymatix.toml"
    valid_section = (
        "[budget]\nexpression_tokens = 4321\n" if section == "server"
        else "[server]\nport = 1234\n"
    )
    path.write_text(valid_section, encoding="utf-8")
    expected = load_config(str(path))
    caplog.clear()

    path.write_text(f"{section} = {value}\n{valid_section}", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="cymatix_context.config"):
        actual = load_config(str(path))

    assert actual == expected
    if section == "server":
        assert actual.budget.expression_tokens == 4321
    else:
        assert actual.server.port == 1234
    warnings = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1, warnings
    assert f"[{section}] is not a table; ignoring" in warnings[0]


@pytest.mark.parametrize(("legacy", "alias", "field"), [
    ("ribosome", "compressor", "model"),
    ("genome", "knowledge_store", "path"),
])
@pytest.mark.parametrize(("legacy_value", "alias_value", "expected"), [
    ('"malformed"', '{ FIELD = "alias-value" }', "alias-value"),
    ('{ FIELD = "legacy-value" }', '"malformed"', "legacy-value"),
    ('{ FIELD = "legacy-value" }', '{ FIELD = "alias-value" }', "legacy-value"),
])
def test_section_alias_precedence_ignores_only_malformed_tables(
    tmp_path, caplog, monkeypatch, legacy, alias, field, legacy_value, alias_value, expected,
):
    """A malformed legacy value must not hide a valid alias; valid legacy wins."""
    # conftest sets the legacy store-path environment override to :memory:;
    # isolate TOML precedence from that independent, higher-priority override.
    monkeypatch.delenv("CYMATIX_GENOME_PATH", raising=False)
    monkeypatch.delenv("CYMATIX_STORE_PATH", raising=False)
    path = tmp_path / "cymatix.toml"
    path.write_text(
        f"{legacy} = {legacy_value.replace('FIELD', field)}\n"
        f"{alias} = {alias_value.replace('FIELD', field)}\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="cymatix_context.config"):
        cfg = load_config(str(path))

    assert getattr(getattr(cfg, legacy), field) == expected
    warnings = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1, warnings
    assert not any("Unknown" in warning for warning in warnings)


def test_non_table_section_preserves_environment_override(tmp_path, monkeypatch):
    path = tmp_path / "cymatix.toml"
    path.write_text('genome = "malformed"\n', encoding="utf-8")
    monkeypatch.delenv("CYMATIX_GENOME_PATH", raising=False)
    monkeypatch.setenv("CYMATIX_STORE_PATH", "env/selected.db")

    assert load_config(str(path)).genome.path == "env/selected.db"


@pytest.mark.parametrize(("text", "error", "message"), [
    ('ribosome = "malformed"\n[budget]\nmin_delivered_docs = "many"\n',
     ValueError, "invalid literal"),
    ('hardware = false\n[abstain]\nmode = "per_classifier"\n',
     ConfigError, "requires an"),
    ('[ribosome]\ntimeout = "slow"\n[compressor]\ntimeout = 12\n',
     ValueError, "could not convert"),
])
def test_table_field_validation_errors_still_propagate(tmp_path, text, error, message):
    """Shape recovery must not hide invalid fields or override valid precedence."""
    path = tmp_path / "cymatix.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(error, match=message):
        load_config(str(path))

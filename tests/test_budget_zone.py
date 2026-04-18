"""Unit tests for budget_zone — spike."""

from __future__ import annotations

import os

import pytest

from helix_context.budget_zone import (
    DEFAULT_WINDOW_TOKENS,
    is_enabled,
    zone_cap,
    zone_for,
    zone_metadata,
)


@pytest.fixture
def zone_on(monkeypatch):
    monkeypatch.setenv("HELIX_BUDGET_ZONE", "1")
    assert is_enabled()


@pytest.fixture
def zone_off(monkeypatch):
    monkeypatch.delenv("HELIX_BUDGET_ZONE", raising=False)
    assert not is_enabled()


class TestZoneFor:
    @pytest.mark.parametrize("tokens, expected", [
        (0,       "clean"),
        (10_000,  "clean"),
        (31_999,  "clean"),
        (32_000,  "soft"),
        (50_000,  "soft"),
        (51_200,  "pressure"),
        (70_000,  "pressure"),
        (76_800,  "cap"),
        (100_000, "cap"),
        (102_400, "emergency"),
        (200_000, "emergency"),
    ])
    def test_boundaries_at_128k(self, tokens, expected):
        assert zone_for(tokens, DEFAULT_WINDOW_TOKENS) == expected

    def test_negative_prompt_defaults_clean(self):
        assert zone_for(-100) == "clean"

    def test_zero_window_defaults_clean(self):
        assert zone_for(50_000, 0) == "clean"


class TestZoneCap:
    def test_disabled_returns_none(self, zone_off):
        assert zone_cap(100_000) is None

    def test_clean_returns_none(self, zone_on):
        assert zone_cap(10_000) is None

    def test_soft_caps_at_12(self, zone_on):
        assert zone_cap(40_000) == 12

    def test_pressure_caps_at_6(self, zone_on):
        assert zone_cap(60_000) == 6

    def test_cap_caps_at_3(self, zone_on):
        assert zone_cap(90_000) == 3

    def test_emergency_caps_at_1(self, zone_on):
        assert zone_cap(120_000) == 1

    def test_none_prompt_returns_none(self, zone_on):
        assert zone_cap(None) is None

    def test_cap_is_monotone_nonincreasing(self, zone_on):
        prior = 10**9
        for tokens in range(0, 130_000, 5_000):
            cap = zone_cap(tokens)
            effective = cap if cap is not None else 10**9
            assert effective <= prior, (tokens, effective, prior)
            prior = effective


class TestZoneMetadata:
    def test_shape_when_disabled(self, zone_off):
        meta = zone_metadata(50_000)
        assert meta["enabled"] is False
        assert meta["zone"] == "soft"
        assert meta["cap"] is None  # cap is None because feature off
        assert meta["ratio"] == pytest.approx(0.391, abs=1e-3)

    def test_shape_when_enabled(self, zone_on):
        meta = zone_metadata(90_000)
        assert meta["enabled"] is True
        assert meta["zone"] == "cap"
        assert meta["cap"] == 3
        assert meta["prompt_tokens"] == 90_000
        assert meta["window_tokens"] == DEFAULT_WINDOW_TOKENS

    def test_none_prompt_returns_empty_shape(self, zone_on):
        meta = zone_metadata(None)
        assert meta["zone"] is None
        assert meta["cap"] is None
        assert meta["ratio"] is None

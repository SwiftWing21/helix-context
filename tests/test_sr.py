"""Tests for Successor Representation Tier 5.5 (Stachenfeld 2017)."""

from __future__ import annotations

import pytest

from helix_context.sr import sr_boost, DEFAULT_GAMMA, DEFAULT_K_STEPS


class _FakeGenome:
    """Minimal genome stub for SR unit tests — sr_boost only needs
    access to a co-activation neighbour function."""

    def __init__(self, graph):
        self._graph = graph


@pytest.fixture(autouse=True)
def _patch_loader(monkeypatch):
    """Redirect _load_co_activated to the fake graph so we don't need a DB."""
    from helix_context import ray_trace

    def fake_load(genome, gene_id):
        return genome._graph.get(gene_id, [])

    monkeypatch.setattr(ray_trace, "_load_co_activated", fake_load)


def test_empty_seeds_returns_empty():
    g = _FakeGenome({})
    assert sr_boost(g, []) == {}


def test_seeds_are_excluded_from_result():
    # Seed points to itself; boost must not self-reinforce the seed.
    graph = {"a": ["b"], "b": []}
    g = _FakeGenome(graph)
    result = sr_boost(g, ["a"], gamma=0.9, k_steps=3, weight=1.0)
    assert "a" not in result
    assert "b" in result


def test_one_hop_matches_harmonic_style_pull():
    """At k_steps=1 with gamma=1.0, SR on a single chain reduces to a
    1-hop pull-forward (the regime Tier 5 handles today)."""
    graph = {"seed": ["n1", "n2"], "n1": [], "n2": []}
    g = _FakeGenome(graph)
    result = sr_boost(g, ["seed"], gamma=1.0, k_steps=1, weight=1.0, cap=100.0)
    assert result["n1"] == pytest.approx(0.5, abs=1e-9)
    assert result["n2"] == pytest.approx(0.5, abs=1e-9)


def test_multi_hop_captures_indirect_neighbours():
    """SR's whole point: a gene 2 hops from a seed via an intermediate
    should get some boost. Tier 5 misses this."""
    graph = {"seed": ["bridge"], "bridge": ["far"], "far": []}
    g = _FakeGenome(graph)
    result = sr_boost(g, ["seed"], gamma=0.9, k_steps=3, weight=1.0, cap=100.0)
    assert "bridge" in result
    assert "far" in result
    # Closer neighbour should accumulate more mass than the far one
    assert result["bridge"] > result["far"]


def test_gamma_decay_controls_reach():
    """Low gamma should attenuate multi-hop contributions more than
    high gamma does."""
    graph = {"seed": ["a"], "a": ["b"], "b": ["c"], "c": []}
    g = _FakeGenome(graph)
    low = sr_boost(g, ["seed"], gamma=0.3, k_steps=4, weight=1.0, cap=100.0)
    high = sr_boost(g, ["seed"], gamma=0.9, k_steps=4, weight=1.0, cap=100.0)
    assert high["c"] > low["c"]  # farther gene benefits more from larger gamma


def test_k_steps_bounds_traversal_depth():
    """k_steps=1 should NOT reach genes 2 hops away."""
    graph = {"seed": ["a"], "a": ["b"], "b": []}
    g = _FakeGenome(graph)
    result = sr_boost(g, ["seed"], gamma=0.9, k_steps=1, weight=1.0, cap=100.0)
    assert "a" in result
    assert "b" not in result


def test_cap_bounds_runaway_propagation():
    """Weight * accumulated mass is capped; high weight must not
    blow up any single gene's boost."""
    graph = {"seed": ["a"]}
    g = _FakeGenome(graph)
    result = sr_boost(g, ["seed"], gamma=0.9, k_steps=4, weight=1000.0, cap=3.0)
    assert all(v <= 3.0 + 1e-9 for v in result.values())


def test_defaults_match_spec():
    assert DEFAULT_GAMMA == 0.85
    assert DEFAULT_K_STEPS == 4

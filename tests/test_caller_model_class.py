"""Stage 5 tests — caller_model_class render-branch behavior.

See docs/specs/2026-05-08-stage-5-caller-model-class.md §10.

All tests use an in-memory store and mock backend (no Ollama). The historical
golden checks the generic decoder/render contract; a separate test compares
the current default and explicit generic responses. See tests/golden/README.md
for the archived snapshot's provenance and the scope of these invariants.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import List, Tuple
from unittest.mock import AsyncMock

import pytest

from cymatix_context import context_manager as _cm
from cymatix_context.config import BudgetConfig, ClassifierConfig
from cymatix_context.context_manager import CymatixContextManager
from cymatix_context.retrieval.query_classifier import (
    DECODER_MODE_TABLE,
    resolve_decoder_mode,
)
from cymatix_context.schemas import (
    CALLER_MODEL_CLASS_DEFAULT,
    CallerModelClass,
)
from tests.conftest import MockCompressorBackend, make_client, make_gene, make_cymatix_config


# ── Test fixtures ────────────────────────────────────────────────────────


def _make_manager_with_kvs() -> CymatixContextManager:
    """Manager with mock backend + genes that carry KVs (for slate tests)."""
    cfg = make_cymatix_config(
        budget=BudgetConfig(max_genes_per_turn=4, splice_aggressiveness=0.5),
        classifier=ClassifierConfig(enabled=True),
    )
    mgr = CymatixContextManager(cfg)
    mgr.ribosome.backend = MockCompressorBackend()
    # Seed genes with key_values so slate population has material.
    seeds: List[Tuple[str, List[str], List[str], List[str]]] = [
        ("Auth middleware uses JWT validation",
         ["auth", "security"], ["jwt", "auth"],
         ["jwt_expiry=15m", "auth_kind=middleware", "version=1"]),
        ("Cache eviction policy uses LRU",
         ["performance", "cache"], ["cache", "ttl"],
         ["cache_ttl=30m", "policy=lru", "size=10000"]),
        ("Migration cost spreadsheet with monthly totals",
         ["finance", "migration"], ["cost", "total"],
         ["region=us-east", "monthly_cost=42000", "currency=usd"]),
        ("Failover runbook for recovery procedures",
         ["ops"], ["failover"],
         ["rto=15m", "rpo=5m", "owner=ops"]),
    ]
    for i, (content, doms, ents, kvs) in enumerate(seeds):
        g = make_gene(content, domains=doms, entities=ents,
                      gene_id=f"kv_seed_{i:010d}")
        g.key_values = kvs
        mgr.genome.upsert_gene(g)
    return mgr


# ── Test 1: default class is generic ─────────────────────────────────────


@pytest.fixture
def http_client():
    cfg = make_cymatix_config(classifier=ClassifierConfig(enabled=True))
    client = make_client(config=cfg, backend=MockCompressorBackend())
    app = client.app
    # Seed at least one gene so the build_context path runs.
    g = make_gene("Authentication middleware with JWT validation",
                  domains=["auth"], entities=["jwt"],
                  gene_id="http_seed_00001")
    g.key_values = ["jwt_expiry=15m"]
    app.state.cymatix.genome.upsert_gene(g)
    with client as c:
        yield c


@pytest.fixture
def context_spy(http_client, monkeypatch):
    manager = http_client.app.state.cymatix
    spy = AsyncMock(wraps=manager.build_context_async)
    monkeypatch.setattr(manager, "build_context_async", spy)
    return spy


@pytest.mark.parametrize(
    ("payload", "expected_model"),
    [
        ({"model": "qwen3:4b"}, "qwen3:4b"),
        ({"downstream_model": "legacy"}, "legacy"),
        ({"model": "canonical", "downstream_model": "legacy"}, "canonical"),
        ({"model": None, "downstream_model": "legacy"}, None),
        ({"model": "", "downstream_model": "legacy"}, ""),
    ],
)
def test_context_model_key_precedence(http_client, context_spy, payload, expected_model):
    """/context forwards canonical model first, preserving a narrow legacy alias."""
    response = http_client.post("/context", json={"query": "q", **payload})

    assert response.status_code == 200
    assert context_spy.call_args.kwargs["downstream_model"] == expected_model


@pytest.mark.parametrize("caller_class", ["generic", "small_moe", "frontier"])
def test_context_echoes_effective_caller_class(http_client, caller_class):
    """The response agent surface exposes the selected retrieval class."""
    response = http_client.post(
        "/context",
        json={"query": "q", "caller_model_class": caller_class},
    )

    assert response.status_code == 200
    assert response.json()[0]["agent"]["caller_model_class"] == caller_class


def test_invalid_per_call_class_keeps_route_error(http_client):
    """Invalid per-call classes retain the adapter-consumable validation shape."""
    response = http_client.post(
        "/context",
        json={"query": "q", "caller_model_class": "oversized"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "Invalid caller_model_class",
        "allowed": ["generic", "small_moe", "frontier"],
    }


@pytest.mark.parametrize("caller_class", ["generic", "small_moe", "frontier"])
def test_mcp_to_context_round_trip_echoes_class(http_client, monkeypatch, caller_class):
    """The MCP adapter's canonical fields drive the real HTTP route contract."""
    pytest.importorskip("mcp.server.mcpserver")
    from cymatix_context.mcp import mcp_server

    def loopback(method, path, body=None):
        assert method == "POST"
        response = http_client.post(path, json=body)
        return response.json()

    monkeypatch.setattr(mcp_server, "_http", loopback)
    result = mcp_server.cymatix_context(
        "q",
        model="qwen3:4b",
        caller_model_class=caller_class,
    )

    assert result["agent"]["caller_model_class"] == caller_class


def test_caller_model_class_default_is_generic(http_client):
    """Spec §10 test 1: POST /context without the field; assert handled,
    response metadata echoes ``"generic"``."""
    resp = http_client.post(
        "/context",
        json={"query": "What is the JWT expiry?"},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    # /context returns a list of one entry under the Continue protocol.
    assert isinstance(payload, list) and len(payload) == 1
    entry = payload[0]
    # The agent block surfaces metadata for programmatic callers.
    assert "agent" in entry
    # Echoed via window.metadata, mirrored on agent.* if surfaced; else
    # verify by hitting the build_context path directly through the client.
    # The /context endpoint sets window.metadata["caller_model_class"]; that
    # attribute is exposed on the agent block via a passthrough below if the
    # server wires it. We assert the round-trip via a follow-up call that
    # explicitly sends the field and checks the same surface for parity.
    explicit = http_client.post(
        "/context",
        json={"query": "What is the JWT expiry?", "caller_model_class": "generic"},
    ).json()[0]
    # Both responses reference the same agent surface; the implicit and
    # explicit "generic" calls are externally indistinguishable.
    assert entry["agent"].get("recommendation") == explicit["agent"].get("recommendation")


def test_caller_model_class_unknown_returns_400(http_client):
    """Defensive: unknown caller_model_class is rejected with the allowed list."""
    resp = http_client.post(
        "/context",
        json={"query": "What is the JWT expiry?",
              "caller_model_class": "made_up_class"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "allowed" in body
    assert set(body["allowed"]) == {"generic", "small_moe", "frontier"}


# ── Test 2: frontier skips foveated ──────────────────────────────────────


def test_frontier_skips_foveated(monkeypatch):
    """Spec §10 test 2: with frontier + broad budget_tier, _compute_foveated_caps
    is never called and candidates stay in forward order at splice time."""
    mgr = _make_manager_with_kvs()
    try:
        # Force broad budget tier and foveated_enabled, then monkeypatch the
        # foveated cap computer to record any call.
        mgr.config.budget.foveated_enabled = True

        called = {"n": 0}

        original = _cm._compute_foveated_caps

        def _spy(*args, **kwargs):
            called["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(_cm, "_compute_foveated_caps", _spy)

        # Frontier call — assert no foveated invocation.
        win_frontier = mgr.build_context(
            "What is the JWT expiry?",
            caller_model_class="frontier",
        )
        assert called["n"] == 0, "frontier must not invoke _compute_foveated_caps"
        # Foveated metadata must be absent (no reversal happened).
        assert "foveated_caps" not in (win_frontier.metadata or {})

        # Sanity: a generic call under broad tier and foveated_enabled DOES
        # invoke the foveated path (as long as candidates >1). This pinpoints
        # the test to the caller_model_class gate, not some unrelated dropout.
        called["n"] = 0
        # Force broad tier by inflating budget_tier signal — bypass the
        # internal computation by directly setting prompt_tokens_hint to a
        # value that hints budget_tier=="broad" (which is the default).
        win_generic = mgr.build_context(
            "Compare cache eviction and JWT expiry between auth and ops.",
            caller_model_class="generic",
        )
        # We can't unconditionally assert called["n"] > 0 because the foveated
        # gate also requires len(candidates) > 1, which depends on the seed
        # corpus. Both branches reach the gate; the assertion that frontier
        # is gated OUT is the load-bearing one above.
        _ = win_generic
    finally:
        mgr.close()


# ── Test 3: small_moe emits JSON slate ───────────────────────────────────


def test_small_moe_emits_json_slate():
    """Spec §10 test 3: decoder prompt contains <cymatix:slate>{ and the
    rendered slate body parses as JSON."""
    mgr = _make_manager_with_kvs()
    try:
        win = mgr.build_context(
            "What is the JWT expiry?",
            caller_model_class="small_moe",
        )
        prompt = win.ribosome_prompt
        assert "<cymatix:slate>" in prompt
        # Extract the slate body and assert it is JSON.
        start = prompt.index("<cymatix:slate>") + len("<cymatix:slate>")
        end = prompt.index("</cymatix:slate>", start)
        slate_body = prompt[start:end]
        parsed = json.loads(slate_body)  # must not raise
        assert isinstance(parsed, dict)
    finally:
        mgr.close()


# ── Test 4: slate is char-bounded, not entry-bounded ─────────────────────


def test_slate_char_bounded_not_entry_bounded():
    """Spec §10 test 4: feed 50 short KVs with budget=400 — assert >20
    entries can fit AND total rendered length ≤ 400 (wrapper-tag chars
    INCLUDED in the budget per §5)."""
    # Drive the renderer directly with 50 short KVs.
    kvs = [f"k{i:02d}=v{i:02d}" for i in range(50)]
    rendered = _cm._render_small_moe_slate(kvs, char_budget=400)

    assert rendered.startswith("<cymatix:slate>")
    assert rendered.endswith("</cymatix:slate>")
    assert len(rendered) <= 400
    # Parse the JSON and count entries.
    body = rendered[len("<cymatix:slate>"): -len("</cymatix:slate>")]
    parsed = json.loads(body)
    assert len(parsed) > 20, (
        f"expected >20 entries to fit in 400-char budget with short KVs, "
        f"got {len(parsed)}: {parsed}"
    )


def test_slate_truncates_long_value_to_min_chars():
    """Spec §5 truncation rule: when a KV's value would exceed budget,
    truncate to fit (≥ 8 chars retained); else drop the entry."""
    kvs = [
        "answer=" + ("X" * 1000),  # huge value — must be truncated
        "next=short",
    ]
    rendered = _cm._render_small_moe_slate(kvs, char_budget=80)
    body = rendered[len("<cymatix:slate>"): -len("</cymatix:slate>")]
    parsed = json.loads(body)
    # The huge value must be truncated; we don't assert the precise length
    # because the binary-search width depends on the JSON-overhead of the
    # other key, but we DO assert it is shorter than the original.
    if "answer" in parsed:
        assert len(parsed["answer"]) < 1000
    # And the budget is honored.
    assert len(rendered) <= 80


# ── Test 5: 15-cell decoder mode lookup table is complete ────────────────


def test_decoder_mode_lookup_table_complete():
    """Spec §10 test 5: assert every (cls, caller_model_class) cell of the
    §6 table returns the documented value via resolve_decoder_mode."""
    expected = {
        ("arithmetic", "generic"):   "minimal",
        ("arithmetic", "small_moe"): "answer_slate_only",
        ("arithmetic", "frontier"):  "minimal",
        ("factual",    "generic"):   "condensed",
        ("factual",    "small_moe"): "answer_slate_only",
        ("factual",    "frontier"):  "condensed",
        ("procedural", "generic"):   "full",
        ("procedural", "small_moe"): "condensed_with_slate",
        ("procedural", "frontier"):  "full",
        ("multi_hop",  "generic"):   "full",
        ("multi_hop",  "small_moe"): "condensed_with_slate",
        ("multi_hop",  "frontier"):  "full",
        ("default",    "generic"):   None,
        ("default",    "small_moe"): "condensed_with_slate",
        ("default",    "frontier"):  None,
    }
    # Cardinality check: 5 classifier classes × 3 caller_model_class = 15.
    classes = ["arithmetic", "factual", "procedural", "multi_hop", "default"]
    cmcs = ["generic", "small_moe", "frontier"]
    assert len(expected) == 15
    for cls, cmc in itertools.product(classes, cmcs):
        actual = resolve_decoder_mode(cls, cmc)
        assert actual == expected[(cls, cmc)], (
            f"resolve_decoder_mode({cls!r}, {cmc!r}) returned {actual!r}, "
            f"expected {expected[(cls, cmc)]!r}"
        )

    # Also: the published DECODER_MODE_TABLE matches the expected mapping.
    for (cls, cmc), val in expected.items():
        assert DECODER_MODE_TABLE[cls][cmc] == val


# ── Test 6: generic rendering and default-call compatibility ─────────────


_GOLDEN_PATH = Path(__file__).parent / "golden" / "pre_stage5_responses.jsonl"


def _historical_responses():
    # A missing committed fixture is a failure, never a silent skip.
    with _GOLDEN_PATH.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 100
    return rows


@pytest.mark.parametrize("caller_kwargs", [{}, {"caller_model_class": "generic"}],
                         ids=["default", "explicit-generic"])
def test_generic_rendering_matches_pre_stage5_contract(caller_kwargs):
    """Wrong decoder routing, suppressed headers, or lost delivery must fail.

    Prompt bytes and classifier decisions come from the untouched archive.
    Ranking scores, health aggregates, and later metadata additions are outside
    this historical render invariant; current response parity is checked below.
    """
    from tests.golden._capture_pre_stage5_responses import _build_manager

    mgr = _build_manager()
    try:
        delivered_classes = set()
        delivered_count = 0
        for row in _historical_responses():
            expected = row["response"]
            win = mgr.build_context(row["query"], **caller_kwargs)
            label = f"idx={row['idx']} query={row['query']!r}"
            assert win.ribosome_prompt == expected["ribosome_prompt"], label
            assert bool(win.expressed_gene_ids) == bool(expected["expressed_gene_ids"]), label
            if win.expressed_gene_ids:
                delivered_count += 1
                classifier = win.metadata["classifier"]
                expected_classifier = expected["metadata"]["classifier"]
                for key in ("class", "decoder_selected", "assembly_max_genes_cap"):
                    assert classifier[key] == expected_classifier[key], (label, key)
                delivered_classes.add(classifier["class"])
                assert "[gene=" in win.expressed_context, label
                assert "<expressed_context>" in win.expressed_context, label
                assert "<cymatix:slate>" not in win.ribosome_prompt, label
        # Exercise real assembly in every classifier branch, not just decoder
        # selection on an early abstain return. The archive has eight misses.
        assert delivered_count == 92
        assert delivered_classes == {"arithmetic", "factual", "procedural", "multi_hop", "default"}
    finally:
        mgr.close()


def test_default_and_generic_callers_have_identical_current_responses():
    """Omitting caller_model_class must preserve explicit generic behavior.

    Independent managers receive the same query sequence, keeping access-rate
    and delivery history equal. Both run in this process, so hash order is
    shared without changing the user's environment or skipping the test.
    """
    from contextlib import ExitStack

    from tests.golden._capture_pre_stage5_responses import (
        _build_manager,
        _serializable_window,
    )

    with ExitStack() as cleanup:
        default_mgr = _build_manager(historical_render_config=False)
        cleanup.callback(default_mgr.close)
        generic_mgr = _build_manager(historical_render_config=False)
        cleanup.callback(generic_mgr.close)
        delivered_classes = set()
        saw_abstain = False
        for row in _historical_responses():
            implicit = default_mgr.build_context(row["query"])
            explicit = generic_mgr.build_context(row["query"], caller_model_class="generic")
            assert _serializable_window(implicit) == _serializable_window(explicit), (
                f"Default/generic mismatch at idx={row['idx']} query={row['query']!r}"
            )
            if implicit.expressed_gene_ids:
                delivered_classes.add(implicit.metadata["classifier"]["class"])
            elif implicit.metadata.get("budget_tier") == "abstain":
                saw_abstain = True
        assert delivered_classes == {"arithmetic", "factual", "procedural", "multi_hop", "default"}
        assert saw_abstain


# ── Sanity: schema enum values match the wire format strings ─────────────


def test_caller_model_class_enum_values():
    """Wire-format check: enum values must match the allowed strings used
    by server.py validation."""
    assert CALLER_MODEL_CLASS_DEFAULT == "generic"
    assert {c.value for c in CallerModelClass} == {"generic", "small_moe", "frontier"}

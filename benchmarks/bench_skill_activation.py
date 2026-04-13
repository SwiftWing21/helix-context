"""
Skill / tool activation profiler.

For a curated set of distinct prompt SHAPES, probe the /context endpoint
and capture WHICH retrieval signals fire (and how strongly). Output is
a 2D activation matrix: prompt-shape × retrieval-tier → contribution.

This makes the lane graph from docs/PIPELINE_LANES.md MEASURABLE rather
than asserted. We can verify that the engine actually picks the tools
we think it picks for each query type — and detect drift if a refactor
silently changes the routing.

Requires server-side per-tier breakdown surfaced via /context's
agent.tier_totals (verbose=true). See helix_context/genome.py
last_tier_contributions and helix_context/server.py /context endpoint
for the wiring.

Usage:
    python benchmarks/bench_skill_activation.py
    HELIX_MODEL=qwen3:8b python benchmarks/bench_skill_activation.py

Output:
    benchmarks/skill_activation_results.json — per-shape activation
        matrix with raw contributions + heatmap-friendly aggregates
    Console — ASCII heatmap (10 shapes × N tiers)

Companion: benchmarks/bench_dimensional_lock.py (precision curve);
this bench is the ingredient profile.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
OUTPUT_PATH = os.environ.get(
    "OUTPUT",
    str(Path(__file__).resolve().parent / "skill_activation_results.json"),
)
REQUEST_TIMEOUT_S = 30.0


# ── Curated prompt shapes ────────────────────────────────────────────
# Each shape is designed to engage a specific subset of the 12 retrieval
# signals. The "expected_strong" field is the prediction; the bench
# measures whether that prediction holds. Mismatches between expected
# and observed are diagnostic — they reveal that the routing isn't
# what we think it is.
PROMPT_SHAPES = [
    {
        "id": "bare_keyword",
        "label": "bare keyword",
        "query": "port",
        "expected_strong": ["fts5", "splade"],
        "rationale": "no context, just a single token — content match dominates",
    },
    {
        "id": "generic_question",
        "label": "generic question",
        "query": "what is the value of port?",
        "expected_strong": ["fts5"],
        "rationale": "natural-language scaffolding, no project narrowing",
    },
    {
        "id": "project_plus_key",
        "label": "project + key",
        "query": "helix port",
        "expected_strong": ["pki", "tag_exact", "lex_anchor"],
        "rationale": "compound (helix, port) hits PKI; both terms are tag-like",
    },
    {
        "id": "project_plus_entity",
        "label": "project + entity",
        "query": "helix ribosome",
        "expected_strong": ["tag_exact", "pki", "lex_anchor"],
        "rationale": "two named entities, narrows hard",
    },
    {
        "id": "code_symbol",
        "label": "code symbol",
        "query": "def upsert_gene",
        "expected_strong": ["fts5", "splade"],
        "rationale": "code-shaped token, lives in raw content not tags",
    },
    {
        "id": "path_lookup",
        "label": "path lookup",
        "query": "helix_context genome.py",
        "expected_strong": ["pki", "tag_prefix", "fts5"],
        "rationale": "file path tokens — PKI loves these",
    },
    {
        "id": "natural_sentence",
        "label": "natural sentence",
        "query": "How does helix handle WAL checkpoints?",
        "expected_strong": ["fts5", "splade", "lex_anchor"],
        "rationale": "long natural language; multiple weak signals stack",
    },
    {
        "id": "multi_key_compound",
        "label": "multi-key compound",
        "query": "helix port and ribosome model",
        "expected_strong": ["pki", "tag_exact", "lex_anchor"],
        "rationale": "two PKI compound hits, two tag matches",
    },
    {
        "id": "documentation_phrase",
        "label": "documentation phrase",
        "query": "the four-layer attribution model",
        "expected_strong": ["fts5", "splade"],
        "rationale": "doc-shaped, no project anchor",
    },
    {
        "id": "vague_plea",
        "label": "vague plea",
        "query": "show me everything about caching",
        "expected_strong": ["splade", "sema_cold", "fts5"],
        "rationale": "broad semantic — semantic tiers should carry it",
    },
]


# ── Probe one shape ──────────────────────────────────────────────────

def probe_shape(client: httpx.Client, shape: dict) -> dict:
    """Send the shape's query to /context with verbose=true and capture
    the per-tier activation totals."""
    out = dict(shape)  # shallow copy
    t0 = time.time()
    try:
        resp = client.post(
            f"{HELIX_URL}/context",
            json={
                "query": shape["query"],
                "decoder_mode": "none",
                "verbose": True,
                "clean": True,  # fresh state per probe
            },
            timeout=REQUEST_TIMEOUT_S,
        )
    except Exception as e:
        out["error"] = f"context: {e}"
        out["latency_s"] = time.time() - t0
        return out

    out["latency_s"] = round(time.time() - t0, 2)
    if resp.status_code != 200:
        out["error"] = f"HTTP {resp.status_code}"
        return out

    body = resp.json()
    if isinstance(body, list):
        body = body[0] if body else {}

    agent = body.get("agent", {}) or {}
    health = body.get("context_health", {}) or {}

    out["tier_totals"] = agent.get("tier_totals", {}) or {}
    out["genes_expressed"] = health.get("genes_expressed", 0)
    out["ellipticity"] = health.get("ellipticity", 0.0)
    out["budget_tier"] = agent.get("budget_tier", "")
    out["moe_mode"] = agent.get("moe_mode", False)
    out["compression_ratio"] = agent.get("compression_ratio", 0.0)

    # Top citation source (helps interpretation)
    cits = agent.get("citations", []) or []
    if cits:
        out["top_source"] = (cits[0].get("source") or "").rsplit("/", 1)[-1]
        out["top_score"] = round(cits[0].get("score", 0.0), 2)
    else:
        out["top_source"] = ""
        out["top_score"] = 0.0

    return out


# ── Heatmap rendering ────────────────────────────────────────────────

# Order tiers by their conceptual position in the pipeline (matches
# PIPELINE_LANES.md). Tiers that don't fire at all stay in the column
# order so absence is visually obvious.
ALL_TIERS = [
    "pki",
    "tag_exact",
    "tag_prefix",
    "fts5",
    "splade",
    "sema_boost",
    "sema_cold",
    "lex_anchor",
    "harmonic",
    "party_attr",
    "access_rate",
]


def _glyph(value: float, max_value: float) -> str:
    """Map a contribution to one of █▓░. relative to the row max."""
    if max_value <= 0:
        return " . "
    ratio = value / max_value
    if value < 0.01:
        return " . "
    if ratio >= 0.66:
        return " █ "
    if ratio >= 0.33:
        return " ▓ "
    return " ░ "


def render_heatmap(results: list[dict]) -> str:
    """ASCII heatmap: prompt-shape × tier."""
    lines = []
    # Header
    head = f"  {'shape':24s}  " + "".join(f"{t[:6]:>6s} " for t in ALL_TIERS)
    lines.append(head)
    lines.append("  " + "─" * (24 + 2 + 7 * len(ALL_TIERS)))
    for r in results:
        if "error" in r and r.get("error"):
            row = f"  {r['label'][:24]:24s}  ERROR: {r['error']}"
            lines.append(row)
            continue
        totals = r.get("tier_totals", {}) or {}
        row_max = max(totals.values()) if totals else 0
        row = f"  {r['label'][:24]:24s}  " + "".join(
            f"{_glyph(totals.get(t, 0), row_max):>6s} " for t in ALL_TIERS
        )
        lines.append(row)
    lines.append("")
    lines.append("  legend: █ strong  ▓ moderate  ░ weak  . silent  (row-relative)")
    return "\n".join(lines)


def render_predictions(results: list[dict]) -> str:
    """Per-shape comparison: expected_strong vs observed-strong tiers."""
    lines = ["", "== Prediction-vs-observation ==", ""]
    for r in results:
        if r.get("error"):
            lines.append(f"  {r['label']}: ERROR")
            continue
        expected = set(r.get("expected_strong", []))
        totals = r.get("tier_totals", {}) or {}
        if not totals:
            lines.append(f"  {r['label']}: no tiers fired")
            continue
        # "Strong" = >= 33% of row max
        row_max = max(totals.values())
        observed = {t for t, v in totals.items() if row_max > 0 and v / row_max >= 0.33}
        match = expected & observed
        only_expected = expected - observed
        only_observed = observed - expected
        verdict = "✓" if match == expected and not only_expected else "≠"
        lines.append(
            f"  {verdict}  {r['label']:24s}  "
            f"expected={sorted(expected)}  observed={sorted(observed)}"
        )
        if only_expected:
            lines.append(f"     missed: {sorted(only_expected)}")
        if only_observed:
            lines.append(f"     bonus:  {sorted(only_observed)}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    print(f"=== Skill / tool activation profiler ===")
    print(f"Server: {HELIX_URL}")
    print(f"Shapes: {len(PROMPT_SHAPES)}")
    print()

    # Sanity ping
    try:
        h = httpx.get(f"{HELIX_URL}/health", timeout=5).json()
        print(f"Server: ok, ribosome={h.get('ribosome')}, genes={h.get('genes')}\n")
    except Exception as e:
        print(f"Server unreachable: {e}", file=sys.stderr)
        return 2

    results = []
    with httpx.Client() as client:
        for i, shape in enumerate(PROMPT_SHAPES, 1):
            print(f"  [{i}/{len(PROMPT_SHAPES)}] {shape['label']:24s} → ", end="", flush=True)
            r = probe_shape(client, shape)
            results.append(r)
            if r.get("error"):
                print(f"ERROR ({r['error']})")
            else:
                tt = r.get("tier_totals") or {}
                top = max(tt.items(), key=lambda kv: kv[1]) if tt else ("none", 0)
                print(f"top_tier={top[0]} ({top[1]:.1f})  genes={r.get('genes_expressed', 0)}  "
                      f"top_src={r.get('top_source', '')[:30]}")

    print()
    print("== Activation heatmap ==")
    print()
    print(render_heatmap(results))
    print(render_predictions(results))

    out = {
        "timestamp": time.time(),
        "n_shapes": len(PROMPT_SHAPES),
        "results": results,
    }
    Path(OUTPUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

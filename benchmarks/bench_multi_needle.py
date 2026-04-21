"""Multi-needle NIAH — retrieval bench for queries that require 2-3 distinct
genes to answer correctly, against a genome containing semantically-similar
distractors.

Extends `bench_needle.py`: each needle carries a list of *groups* instead
of a single gold_source. A query is fully answered only if at least one
gene from **each** group is delivered. Partial delivery is tracked separately.

The distractor signal is implicit — the existing genome already holds
multiple `port` mentions (11437 for helix, 8787 for headroom, 11434 for
ollama), multiple `model =` configs, multiple `CLAIM_TYPES`-like
constants. That's the adversarial surface: folder/file-grain routing has
to disambiguate between *semantically identical* phrases in different
contexts.

Usage:
    python benchmarks/bench_multi_needle.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
GENE_SRC_RE = re.compile(r'<GENE src="([^"]+)"')


# Each needle requires ALL groups to be delivered for full recall.
# Within a group, any path substring match counts. Groups are the
# *distinct facts* the query needs — same-group paths are equivalents.
# Adversarial distractors are implicit in the genome (other files that
# mention 'port', 'model', 'version' in different contexts).
NEEDLES_MULTI = [
    {
        "name": "helix_and_headroom_ports",
        "query": "what ports do helix and headroom listen on",
        "gold_source_groups": [
            ["helix-context/helix.toml"],                  # 11437
            ["helix-context/start-helix-tray.bat", "helix-context/helix.toml"],  # 8787 (headroom section)
        ],
    },
    {
        "name": "python_version_and_codec_extra",
        "query": "what python version does helix require and which extra enables headroom integration",
        "gold_source_groups": [
            ["helix-context/pyproject.toml"],              # requires-python
            ["helix-context/pyproject.toml", "helix-context/README.md"],  # [codec]
        ],
    },
    {
        "name": "pipeline_steps_and_compression_target",
        "query": "how many steps are in the helix pipeline and what is the target compression ratio",
        "gold_source_groups": [
            ["helix-context/docs/architecture/PIPELINE_LANES.md", "helix-context/README.md"],
            ["helix-context/docs/DESIGN_TARGET.md", "helix-context/README.md"],
        ],
    },
    {
        "name": "claim_types_and_spec_source",
        "query": "what are the allowed claim_type values and where is the claims layer specified",
        "gold_source_groups": [
            ["helix-context/helix_context/schemas.py", "helix-context/helix_context/claims.py"],
            ["helix-context/docs/specs/2026-04-17-agent-context-index-build-spec.md"],
        ],
    },
    {
        "name": "helix_port_and_fleet_port",
        "query": "what port does helix listen on and what port does the bigEd fleet dashboard use",
        "gold_source_groups": [
            ["helix-context/helix.toml"],                  # 11437
            ["Education/fleet/fleet.toml", "Education/fleet/CLAUDE.md",
             "Education/CLAUDE.md"],                        # bigEd dashboard port
        ],
    },
    {
        "name": "headroom_dashboard_port_and_mode_default",
        "query": "what port does the headroom dashboard serve on and what is the default compression mode",
        "gold_source_groups": [
            ["helix-context/helix.toml", "helix-context/README.md"],  # 8787
            ["helix-context/helix.toml", "helix-context/helix_context/launcher/headroom_supervisor.py"],  # token mode
        ],
    },
    {
        "name": "freshness_half_lives_stable_and_hot",
        "query": "what are the half-lives for stable and hot volatility classes",
        "gold_source_groups": [
            ["helix-context/README.md",
             "helix-context/docs/specs/2026-04-17-agent-context-index-build-spec.md",
             "helix-context/helix_context/context_packet.py"],                   # stable=7d
            ["helix-context/README.md",
             "helix-context/docs/specs/2026-04-17-agent-context-index-build-spec.md",
             "helix-context/helix_context/context_packet.py"],                   # hot=15min
        ],
    },
    {
        "name": "coord_confidence_floor_and_file_grain_floor",
        "query": "what is the coordinate confidence floor and the file-grain coverage floor",
        "gold_source_groups": [
            ["helix-context/helix_context/context_packet.py"],   # 0.30 floor
            ["helix-context/helix_context/context_packet.py"],   # 0.15 floor (file-grain)
        ],
    },
]


def _norm(s: str) -> str:
    return s.replace("\\", "/").lower()


def fetch_delivered_srcs(client: httpx.Client, query: str,
                         task_type: str = "explain") -> tuple[list[str], dict]:
    """Issue /context, return (delivered source_ids, context_health)."""
    r = client.post(
        f"{HELIX_URL}/context",
        json={"query": query, "task_type": task_type, "read_only": True},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, list):
        content = "\n".join(
            b.get("content", "") for b in payload if isinstance(b, dict)
        )
        health = {}
    elif isinstance(payload, dict):
        content = payload.get("expressed_context") or payload.get("content") or ""
        health = payload.get("context_health") or {}
    else:
        content = ""
        health = {}
    srcs = GENE_SRC_RE.findall(content)
    return srcs, health


def score_needle(delivered: list[str], groups: list[list[str]]) -> dict:
    """Return per-group + overall metrics for a multi-needle query."""
    delivered_norm = [_norm(s) for s in delivered]
    group_hits: list[bool] = []
    group_hit_counts: list[int] = []
    for group in groups:
        gold_norm = [_norm(g) for g in group]
        n_match = sum(
            1 for s in delivered_norm
            if any(g in s for g in gold_norm)
        )
        group_hit_counts.append(n_match)
        group_hits.append(n_match > 0)
    n_groups = len(groups)
    n_hit_groups = sum(group_hits)
    return {
        "n_groups": n_groups,
        "n_hit_groups": n_hit_groups,
        "all_delivered": n_hit_groups == n_groups,
        "any_delivered": n_hit_groups > 0,
        "partial_recall": n_hit_groups / n_groups if n_groups else 0.0,
        "group_hit_counts": group_hit_counts,
    }


def run_needle(client: httpx.Client, needle: dict) -> dict:
    t0 = time.time()
    try:
        srcs, health = fetch_delivered_srcs(client, needle["query"])
    except Exception as exc:
        return {
            "name": needle["name"],
            "error": str(exc),
            "all_delivered": False,
            "partial_recall": 0.0,
        }
    latency = time.time() - t0
    score = score_needle(srcs, needle["gold_source_groups"])
    return {
        "name": needle["name"],
        "query": needle["query"],
        "n_delivered": len(srcs),
        "context_latency_s": round(latency, 3),
        "resolution_confidence": health.get("resolution_confidence", 0),
        "path_token_coverage": health.get("path_token_coverage", 0),
        "file_token_coverage": health.get("file_token_coverage", 0),
        **score,
    }


def main(out_path: Optional[Path] = None) -> int:
    client = httpx.Client(timeout=120)
    try:
        stats = client.get(f"{HELIX_URL}/stats").json()
        print(
            f"Genome: {stats['total_genes']} genes, "
            f"{stats['compression_ratio']:.2f}x compression"
        )
    except Exception as exc:
        print(f"Cannot reach helix at {HELIX_URL}: {exc}")
        return 1

    print(f"\n=== Multi-needle NIAH ({len(NEEDLES_MULTI)} needles) ===\n")
    print(
        f"{'all':>3} {'partial':>7}  {'conf':>5} {'name':<45} {'query':<60}"
    )
    print("-" * 125)

    results: List[dict] = []
    for needle in NEEDLES_MULTI:
        r = run_needle(client, needle)
        results.append(r)
        if "error" in r:
            print(f"  ERR  {r['name']}: {r['error']}")
            continue
        mark = "YES" if r["all_delivered"] else (
            "PARTIAL" if r["any_delivered"] else "NO"
        )
        print(
            f" {mark:>3} "
            f"{r['partial_recall']:>7.2f}  "
            f"{r['resolution_confidence']:>5.2f} "
            f"{r['name']:<45} "
            f"{r['query'][:60]}"
        )

    # Aggregates
    n = len(results)
    valid = [r for r in results if "error" not in r]
    full = sum(1 for r in valid if r["all_delivered"])
    any_ = sum(1 for r in valid if r["any_delivered"])
    none_ = sum(1 for r in valid if not r["any_delivered"])
    avg_partial = (
        sum(r["partial_recall"] for r in valid) / len(valid) if valid else 0.0
    )
    avg_conf = (
        sum(r["resolution_confidence"] for r in valid) / len(valid)
        if valid else 0.0
    )
    avg_latency = (
        sum(r["context_latency_s"] for r in valid) / len(valid)
        if valid else 0.0
    )

    print()
    print(f"n={n}  full_recall={full}/{n}  any_hit={any_}/{n}  "
          f"zero_hit={none_}/{n}")
    print(f"avg_partial_recall={avg_partial:.2f}  "
          f"avg_resolution_conf={avg_conf:.2f}  "
          f"avg_context_latency={avg_latency*1000:.0f}ms")

    out = out_path or Path("benchmarks/results") / f"multi_needle_{time.strftime('%Y-%m-%d')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "genome": {
            "total_genes": stats.get("total_genes"),
            "compression_ratio": stats.get("compression_ratio"),
        },
        "needles": results,
        "summary": {
            "n": n,
            "full_recall": full,
            "any_hit": any_,
            "zero_hit": none_,
            "avg_partial_recall": avg_partial,
            "avg_resolution_confidence": avg_conf,
            "avg_context_latency_s": avg_latency,
        },
    }, indent=2))
    print(f"\nsaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

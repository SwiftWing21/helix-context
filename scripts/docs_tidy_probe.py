"""Docs-tidy helix path-sensitivity probe.

Hits /debug/preview for a fixed query set, resolves source_id per returned
gene_id, and writes a JSON report. Run before moves, then after, then diff.

Usage:
    python scripts/docs_tidy_probe.py --out benchmarks/docs_tidy_pre.json
    python scripts/docs_tidy_probe.py --out benchmarks/docs_tidy_post.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HELIX = "http://127.0.0.1:11437"

PROBES: list[tuple[str, str]] = [
    ("RESTART_PROTOCOL", "restart protocol"),
    ("MUSIC_OF_RETRIEVAL", "music of retrieval"),
    ("DIMENSIONS", "9 dimensions helix"),
    ("FEDERATION_LOCAL", "4-layer federation identity"),
    ("KNOWLEDGE_GRAPH", "knowledge graph entity links"),
    ("PIPELINE_LANES", "pipeline lanes"),
    ("SESSION_REGISTRY", "session registry org party participant"),
    ("LAUNCHER", "launcher tray setup"),
    ("OBSERVABILITY", "observability prometheus otel"),
    ("RESEARCH_VELOCITY", "research velocity"),
    ("AGENTOME_PART_II", "agentome part two"),
    ("PAPER_FIGURE_SPECS", "paper figure specs"),
    ("PAPER_THREE_CONSTRAINTS", "three constraints paper"),
    ("SIKE_POST_DRAFT", "sike post draft"),
    ("ECONOMICS", "economics helix"),
    ("ENTERPRISE", "enterprise compliance"),
    ("SKILLS_BUNDLE", "skills bundle"),
    ("BENCHMARKS", "benchmarks"),
    ("BENCHMARK_RATIONALE", "benchmark rationale"),
]

MAX_GENES = 10


def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def preview(query: str) -> dict:
    qs = urllib.parse.urlencode({"query": query, "max_genes": MAX_GENES})
    return _get_json(f"{HELIX}/debug/preview?{qs}") or {}


def gene_meta(gene_id: str) -> dict:
    d = _get_json(f"{HELIX}/genes/{gene_id}") or {}
    return {
        "source_id": d.get("source_id"),
        "chromatin": d.get("chromatin"),
        "is_fragment": d.get("is_fragment"),
        "sequence_index": (d.get("promoter") or {}).get("sequence_index"),
    }


def run() -> dict:
    results = []
    for label, query in PROBES:
        prev = preview(query)
        candidates = prev.get("candidates", []) or []
        enriched = []
        for c in candidates:
            gid = c.get("gene_id")
            meta = gene_meta(gid) if gid else {}
            enriched.append(
                {
                    "rank": c.get("rank"),
                    "gene_id": gid,
                    "score": c.get("score"),
                    "source_id": meta.get("source_id"),
                    "chromatin": meta.get("chromatin"),
                    "is_fragment": meta.get("is_fragment"),
                    "sequence_index": meta.get("sequence_index"),
                }
            )
        results.append(
            {
                "label": label,
                "query": query,
                "extracted": prev.get("extracted"),
                "candidates": enriched,
                "error": prev.get("error"),
            }
        )
    stats = _get_json(f"{HELIX}/stats") or {}
    return {
        "timestamp": time.time(),
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "genome": {
            "total_genes": stats.get("total_genes"),
            "open": stats.get("open"),
            "euchromatin": stats.get("euchromatin"),
            "heterochromatin": stats.get("heterochromatin"),
        },
        "probes": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="JSON output path")
    args = p.parse_args()

    report = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    hits = sum(
        1
        for probe in report["probes"]
        for c in probe["candidates"]
        if c.get("source_id") and probe["label"].replace("_", "").lower()
        in (c["source_id"] or "").replace("_", "").lower()
    )
    print(f"[docs_tidy_probe] wrote {args.out}")
    print(f"[docs_tidy_probe] probes: {len(report['probes'])}")
    print(f"[docs_tidy_probe] target-file hits (loose stem match): {hits}")


if __name__ == "__main__":
    sys.exit(main() or 0)

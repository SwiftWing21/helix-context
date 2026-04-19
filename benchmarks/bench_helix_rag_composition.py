"""Helix + RAG composition benchmark — 3-cell NIAH.

Tests the `project_helix_weighs_not_retrieves.md` thesis:
Helix narrows the search space (card catalog), classical RAG fetches
the bytes (library). Together they should out-recall either alone.

## Cells

1. **pure_rag** — direct FTS5/BM25 query against genes_fts, no Helix
   pipeline. Baseline "what does raw RAG get?"
2. **helix_only** — /context/packet (Helix's weighing layer). The
   agent-safe index surface as it stands today.
3. **helix_rag** — /context/packet for pointers, then read the source
   files from disk. The composition: Helix points, naive fetcher reads.

## Dual scoring per needle

- **pointer_precision** — did the gold source_ids appear in the cell's
  delivered set? (card catalog test)
- **content_recall** — did the expected answer string appear in the
  fetched/delivered content? (library test — what an agent would
  actually see)

Both signals are tracked so we can see where each cell fails:
high pointer / low content = pointed right but didn't fetch enough.
low pointer / high content = dumb luck (content overlap without
coordinate resolution).

Requires helix-context server running at 127.0.0.1:11437 AND raw
access to the genome.db file for the pure-RAG cell.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

HELIX_URL = "http://127.0.0.1:11437"
GENOME_PATH = os.environ.get(
    "HELIX_GENOME_PATH",
    str(Path(__file__).resolve().parents[1] / "genomes" / "main" / "genome.db"),
)
GENE_SRC_RE = re.compile(r'<GENE src="([^"]+)"')

# FTS5 has a tokenizer but natural-language queries aren't valid MATCH
# syntax (operators like "what", "does" are fine as literal tokens but
# punctuation and too many stopwords hurt ranking). Strip stopwords +
# OR-join significant tokens.
_STOPWORDS = {
    "what", "when", "where", "who", "how", "why", "which", "do", "does",
    "is", "are", "the", "a", "an", "and", "or", "of", "in", "on", "to",
    "for", "with", "at", "by", "from", "this", "that", "be", "use",
}


def _fts_query(natural: str) -> str:
    """Turn a natural-language query into an FTS5 MATCH expression."""
    toks = re.findall(r"[A-Za-z0-9_]+", natural.lower())
    keep = [t for t in toks if t not in _STOPWORDS and len(t) > 1]
    # OR-join so we get any-match ranking via BM25, not strict AND
    return " OR ".join(keep) if keep else natural


NEEDLES = [
    {
        "name": "helix_and_headroom_ports",
        "query": "what ports do helix and headroom listen on",
        "expected": ["11437", "8787"],
        "gold_source_groups": [
            ["helix-context/helix.toml"],
            ["helix-context/start-helix-tray.bat", "helix-context/helix.toml"],
        ],
    },
    {
        "name": "python_version_and_codec_extra",
        "query": "python version helix requires and extra that enables headroom",
        "expected": ["3.11", "codec"],
        "gold_source_groups": [
            ["helix-context/pyproject.toml"],
            ["helix-context/pyproject.toml", "helix-context/README.md"],
        ],
    },
    {
        "name": "pipeline_steps_and_compression_target",
        "query": "steps in helix pipeline and target compression ratio",
        "expected": ["6", "5x"],
        "gold_source_groups": [
            ["helix-context/docs/architecture/PIPELINE_LANES.md",
             "helix-context/README.md"],
            ["helix-context/docs/DESIGN_TARGET.md",
             "helix-context/README.md"],
        ],
    },
    {
        "name": "claim_types_and_spec_source",
        "query": "claim_type allowed values helix claims layer specification",
        "expected": ["path_value", "agent-context-index"],
        "gold_source_groups": [
            ["helix-context/helix_context/schemas.py",
             "helix-context/helix_context/claims.py"],
            ["helix-context/docs/specs/2026-04-17-agent-context-index-build-spec.md"],
        ],
    },
    {
        "name": "headroom_port_and_mode_default",
        "query": "headroom dashboard port default compression mode",
        "expected": ["8787", "token"],
        "gold_source_groups": [
            ["helix-context/helix.toml", "helix-context/README.md"],
            ["helix-context/helix.toml",
             "helix-context/helix_context/launcher/headroom_supervisor.py"],
        ],
    },
    {
        "name": "freshness_half_lives_stable_and_hot",
        "query": "freshness half-life stable hot volatility",
        "expected": ["7d", "15min"],
        "gold_source_groups": [
            ["helix-context/README.md",
             "helix-context/docs/specs/2026-04-17-agent-context-index-build-spec.md",
             "helix-context/helix_context/context_packet.py"],
            ["helix-context/README.md",
             "helix-context/docs/specs/2026-04-17-agent-context-index-build-spec.md",
             "helix-context/helix_context/context_packet.py"],
        ],
    },
    {
        "name": "coord_floor_and_file_grain_floor",
        "query": "coordinate confidence floor file-grain coverage floor",
        "expected": ["0.30", "0.15"],
        "gold_source_groups": [
            ["helix-context/helix_context/context_packet.py"],
            ["helix-context/helix_context/context_packet.py"],
        ],
    },
    {
        "name": "helix_port_and_fleet_port",
        "query": "helix listen port bigEd fleet dashboard port",
        "expected": ["11437", "5555"],
        "gold_source_groups": [
            ["helix-context/helix.toml"],
            ["Education/fleet/fleet.toml",
             "Education/CLAUDE.md",
             "Education/fleet/CLAUDE.md"],
        ],
    },
]


def _norm(s: str) -> str:
    return (s or "").replace("\\", "/").lower()


# ── Cell A: pure RAG via raw FTS5 ────────────────────────────────────


def cell_pure_rag(needle: dict, top_k: int = 12) -> dict:
    t0 = time.time()
    match_expr = _fts_query(needle["query"])
    conn = sqlite3.connect(GENOME_PATH)
    try:
        rows = conn.execute(
            """SELECT g.gene_id, g.source_id, g.content
               FROM genes_fts f JOIN genes g ON g.gene_id = f.gene_id
               WHERE f.genes_fts MATCH ?
               ORDER BY bm25(genes_fts) LIMIT ?""",
            (match_expr, top_k),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        return {"cell": "pure_rag", "error": f"FTS query failed: {exc}"}
    finally:
        conn.close()

    delivered_srcs = [r[1] for r in rows if r[1]]
    content = "\n---\n".join((r[2] or "")[:4000] for r in rows)
    return {
        "cell": "pure_rag",
        "latency_s": round(time.time() - t0, 3),
        "delivered_srcs": delivered_srcs,
        "n_delivered": len(delivered_srcs),
        "content": content,
        "content_chars": len(content),
    }


# ── Cell B: Helix only (packet mode) ────────────────────────────────


def cell_helix_only(client: httpx.Client, needle: dict) -> dict:
    t0 = time.time()
    try:
        resp = client.post(
            f"{HELIX_URL}/context/packet",
            json={"query": needle["query"], "task_type": "explain"},
            timeout=60,
        )
        resp.raise_for_status()
        packet = resp.json()
    except Exception as exc:
        return {"cell": "helix_only", "error": str(exc)}

    items = []
    for bucket in ("verified", "stale_risk", "contradictions"):
        items.extend(packet.get(bucket, []) or [])
    delivered_srcs = [i.get("source_id") for i in items if i.get("source_id")]
    content = "\n---\n".join(
        (i.get("content") or i.get("title") or "") for i in items
    )
    return {
        "cell": "helix_only",
        "latency_s": round(time.time() - t0, 3),
        "delivered_srcs": delivered_srcs,
        "n_delivered": len(delivered_srcs),
        "content": content,
        "content_chars": len(content),
        "packet_notes": packet.get("notes", []),
        "n_refresh_targets": len(packet.get("refresh_targets", [])),
    }


# ── Cell C: Helix + naive RAG (file-read) ───────────────────────────


def _resolve_path(source_id: str) -> Optional[Path]:
    """Map a source_id to a readable file path."""
    if not source_id:
        return None
    p = Path(source_id)
    if p.exists() and p.is_file():
        return p
    # Try workspace roots (common on Windows dev box)
    for root in (Path("F:/Projects"), Path.home() / "Projects"):
        cand = root / source_id
        if cand.exists() and cand.is_file():
            return cand
    return None


def cell_helix_rag(client: httpx.Client, needle: dict, max_files: int = 12,
                   chars_per_file: int = 5000) -> dict:
    t0 = time.time()
    try:
        resp = client.post(
            f"{HELIX_URL}/context/packet",
            json={"query": needle["query"], "task_type": "explain"},
            timeout=60,
        )
        resp.raise_for_status()
        packet = resp.json()
    except Exception as exc:
        return {"cell": "helix_rag", "error": str(exc)}

    source_ids: list[str] = []
    for bucket in ("verified", "stale_risk", "contradictions"):
        for item in packet.get(bucket, []) or []:
            sid = item.get("source_id")
            if sid and sid not in source_ids:
                source_ids.append(sid)
    for tgt in packet.get("refresh_targets", []) or []:
        sid = tgt.get("source_id")
        if sid and sid not in source_ids:
            source_ids.append(sid)

    fetched = {}
    n_read = 0
    n_missing = 0
    for sid in source_ids[:max_files]:
        path = _resolve_path(sid)
        if path is None:
            n_missing += 1
            continue
        try:
            fetched[sid] = path.read_text(encoding="utf-8", errors="replace")[:chars_per_file]
            n_read += 1
        except Exception:
            n_missing += 1

    content = "\n---\n".join(fetched.values())
    return {
        "cell": "helix_rag",
        "latency_s": round(time.time() - t0, 3),
        "delivered_srcs": source_ids[:max_files],
        "n_delivered": len(source_ids[:max_files]),
        "n_read": n_read,
        "n_missing": n_missing,
        "content": content,
        "content_chars": len(content),
    }


# ── Scoring — dual signal ───────────────────────────────────────────


def score_cell(result: dict, needle: dict) -> dict:
    if "error" in result:
        return {
            "pointer_full": False,
            "pointer_partial": 0.0,
            "content_full": False,
            "content_partial": 0.0,
            "error": result["error"],
        }
    # Pointer precision
    delivered_norm = [_norm(s) for s in result.get("delivered_srcs", [])]
    groups = needle["gold_source_groups"]
    group_hits = []
    for group in groups:
        gold_norm = [_norm(g) for g in group]
        group_hits.append(any(
            any(g in s for g in gold_norm) for s in delivered_norm
        ))
    n_groups = len(groups)
    pointer_partial = sum(group_hits) / n_groups if n_groups else 0.0
    pointer_full = all(group_hits) if group_hits else False

    # Content recall
    content_lower = (result.get("content") or "").lower()
    expected = needle.get("expected") or []
    if isinstance(expected, str):
        expected = [expected]
    answer_found = [a.lower() in content_lower for a in expected]
    content_partial = sum(answer_found) / len(expected) if expected else 0.0
    content_full = all(answer_found) if expected else False

    return {
        "pointer_full": pointer_full,
        "pointer_partial": pointer_partial,
        "content_full": content_full,
        "content_partial": content_partial,
        "group_hits": group_hits,
        "answer_found": answer_found,
    }


# ── Runner + reporting ──────────────────────────────────────────────


def run_needle(client: httpx.Client, needle: dict) -> dict:
    cells = {
        "pure_rag": cell_pure_rag(needle),
        "helix_only": cell_helix_only(client, needle),
        "helix_rag": cell_helix_rag(client, needle),
    }
    scores = {name: score_cell(r, needle) for name, r in cells.items()}
    return {
        "name": needle["name"],
        "query": needle["query"],
        "expected": needle["expected"],
        "cells": {
            name: {**r, "score": scores[name]}
            for name, r in cells.items()
        },
    }


def _fmt_pct(x: float) -> str:
    return f"{x*100:>4.0f}%"


def print_per_needle(results: list[dict]) -> None:
    print(f"{'needle':<45} "
          f"{'pure_rag':<17} {'helix_only':<17} {'helix_rag':<17}")
    print(f"{'':<45} "
          f"{'ptr':<7} {'ans':<8} "
          f"{'ptr':<7} {'ans':<8} "
          f"{'ptr':<7} {'ans':<8}")
    print("-" * 96)
    for r in results:
        line = f"{r['name']:<45} "
        for cell_name in ("pure_rag", "helix_only", "helix_rag"):
            s = r["cells"][cell_name]["score"]
            if "error" in s:
                line += f"{'ERR':<7} {'ERR':<8} "
            else:
                line += (f"{_fmt_pct(s['pointer_partial']):<7} "
                         f"{_fmt_pct(s['content_partial']):<8} ")
        print(line)


def print_aggregate(results: list[dict]) -> None:
    print("\n=== Aggregate (across {} needles) ===".format(len(results)))
    print(f"{'cell':<12} {'ptr_full':<10} {'ptr_partial':<12} "
          f"{'ans_full':<10} {'ans_partial':<12} {'mean_latency_ms':<16}")
    print("-" * 75)
    for cell_name in ("pure_rag", "helix_only", "helix_rag"):
        ptr_full = 0
        ans_full = 0
        ptr_partial_sum = 0.0
        ans_partial_sum = 0.0
        lat_sum = 0.0
        n = 0
        for r in results:
            s = r["cells"][cell_name]["score"]
            if "error" in s:
                continue
            n += 1
            ptr_full += int(s["pointer_full"])
            ans_full += int(s["content_full"])
            ptr_partial_sum += s["pointer_partial"]
            ans_partial_sum += s["content_partial"]
            lat_sum += r["cells"][cell_name].get("latency_s", 0) or 0
        if not n:
            continue
        print(f"{cell_name:<12} "
              f"{f'{ptr_full}/{n}':<10} "
              f"{ptr_partial_sum/n:>5.2f}        "
              f"{f'{ans_full}/{n}':<10} "
              f"{ans_partial_sum/n:>5.2f}        "
              f"{(lat_sum/n)*1000:>6.0f} ms")


def main() -> int:
    if not Path(GENOME_PATH).exists():
        print(f"ERROR: genome not found at {GENOME_PATH}")
        print("Set HELIX_GENOME_PATH or run from helix-context dir.")
        return 1

    client = httpx.Client(timeout=120)
    try:
        stats = client.get(f"{HELIX_URL}/stats").json()
        print(f"Genome: {stats['total_genes']} genes, "
              f"{stats['compression_ratio']:.2f}x")
    except Exception as exc:
        print(f"Cannot reach helix at {HELIX_URL}: {exc}")
        return 1

    print(f"\n=== Helix + RAG composition NIAH "
          f"({len(NEEDLES)} needles, 3 cells) ===\n")

    results = []
    for needle in NEEDLES:
        print(f"  running: {needle['name']:<45} ", end="", flush=True)
        r = run_needle(client, needle)
        results.append(r)
        # Tiny inline status so we see progress
        marks = []
        for cell_name in ("pure_rag", "helix_only", "helix_rag"):
            s = r["cells"][cell_name]["score"]
            if "error" in s:
                marks.append("E")
            else:
                marks.append(
                    "F" if s["content_full"] else
                    ("P" if s["content_partial"] > 0 else "-")
                )
        print("  ans: " + " ".join(marks))

    print()
    print_per_needle(results)
    print_aggregate(results)

    out = Path("benchmarks/results") / f"helix_rag_composition_{time.strftime('%Y-%m-%d')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Strip content blobs from saved JSON to keep size reasonable
    trimmed = []
    for r in results:
        r_copy = {"name": r["name"], "query": r["query"],
                  "expected": r["expected"], "cells": {}}
        for name, cell in r["cells"].items():
            cell_copy = {k: v for k, v in cell.items() if k != "content"}
            r_copy["cells"][name] = cell_copy
        trimmed.append(r_copy)
    out.write_text(json.dumps({
        "genome": {
            "total_genes": stats.get("total_genes"),
            "compression_ratio": stats.get("compression_ratio"),
        },
        "n_needles": len(NEEDLES),
        "results": trimmed,
    }, indent=2))
    print(f"\nsaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

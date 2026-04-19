"""Offline file-grain diagnostic — runs 10-needle queries against the
live helix server, parses delivered gene source paths, and computes
file_token_coverage in-python to validate the new signal WITHOUT
requiring a server restart.

Compares folder-grain (path_tokens) vs file-grain (file_tokens) on
the hit-vs-miss split. If file-grain is the right signal, we expect
miss mean < hit mean with a wider gap than the current folder-grain.

Usage:
    python scripts/diagnose_file_grain.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from benchmarks.bench_needle import NEEDLES  # noqa: E402
from helix_context.accel import extract_query_signals  # noqa: E402
from helix_context.genome import file_tokens, path_tokens  # noqa: E402

HELIX_URL = "http://localhost:11437"
GENE_SRC_RE = re.compile(r'<GENE src="([^"]+)"')


def fetch_delivered_srcs(client: httpx.Client, query: str) -> list[str]:
    r = client.post(
        f"{HELIX_URL}/context",
        json={"query": query, "task_type": "explain"},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    # /context returns list-of-blocks; each has .content with embedded <GENE> tags
    if isinstance(payload, list):
        content = "\n".join(b.get("content", "") for b in payload if isinstance(b, dict))
    elif isinstance(payload, dict):
        content = payload.get("expressed_context") or payload.get("content") or ""
    else:
        content = ""
    return GENE_SRC_RE.findall(content)


def coverage(q_set: set, srcs: list[str], token_fn) -> float:
    if not srcs or not q_set:
        return 0.0
    hits = sum(1 for s in srcs if token_fn(s) & q_set)
    return hits / len(srcs)


def main() -> None:
    client = httpx.Client(timeout=60)
    rows = []
    for needle in NEEDLES:
        query = needle["query"]
        name = needle["name"]
        expected = needle["expected"]
        gold_sources = [g.lower() for g in needle.get("gold_source", [])]
        try:
            srcs = fetch_delivered_srcs(client, query)
        except Exception as e:
            print(f"  {name}: request failed — {e}")
            continue
        domains, entities = extract_query_signals(query)
        q_set = {t.lower() for t in (domains + entities) if t}
        folder_cov = coverage(q_set, srcs, path_tokens)
        file_cov = coverage(q_set, srcs, file_tokens)
        gold_delivered = any(
            any(g in s.replace("\\", "/").lower() for g in gold_sources)
            for s in srcs
        )
        rows.append({
            "name": name,
            "query": query,
            "expected": expected,
            "n_delivered": len(srcs),
            "folder_cov": round(folder_cov, 3),
            "file_cov": round(file_cov, 3),
            "gold_delivered": gold_delivered,
        })
        print(
            f"  folder={folder_cov:.2f}  file={file_cov:.2f}  "
            f"gold={'Y' if gold_delivered else 'N'}  "
            f"{name} — {query[:60]}"
        )

    hits = [r for r in rows if r["gold_delivered"]]
    misses = [r for r in rows if not r["gold_delivered"]]

    def mean(xs, k):
        return sum(x[k] for x in xs) / len(xs) if xs else 0.0

    print()
    print(f"n={len(rows)}  hits={len(hits)}  misses={len(misses)}")
    print(f"folder_cov  hit_mean={mean(hits,'folder_cov'):.3f}  "
          f"miss_mean={mean(misses,'folder_cov'):.3f}  "
          f"delta={mean(hits,'folder_cov')-mean(misses,'folder_cov'):+.3f}")
    print(f"file_cov    hit_mean={mean(hits,'file_cov'):.3f}  "
          f"miss_mean={mean(misses,'file_cov'):.3f}  "
          f"delta={mean(hits,'file_cov')-mean(misses,'file_cov'):+.3f}")

    out_path = Path("benchmarks/results/file_grain_diagnostic_2026-04-18.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "needles": rows,
        "summary": {
            "n": len(rows),
            "hits": len(hits),
            "misses": len(misses),
            "folder_hit_mean": mean(hits, "folder_cov"),
            "folder_miss_mean": mean(misses, "folder_cov"),
            "folder_delta": mean(hits, "folder_cov") - mean(misses, "folder_cov"),
            "file_hit_mean": mean(hits, "file_cov"),
            "file_miss_mean": mean(misses, "file_cov"),
            "file_delta": mean(hits, "file_cov") - mean(misses, "file_cov"),
        },
    }, indent=2))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()

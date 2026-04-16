"""
Pass 3a — inspect the exact-tie pairs surfaced by the precision probe.

Question: when two genes in top-k have bitwise-identical scores, are they
    (a) semantically similar — both valid answers, order doesn't matter much
    (b) semantically distinct — different answers, arbitrary pick loses info

(b) is the interesting case: it's empirical motivation for smart tie-break
via associative-graph distance rather than dict insertion order.

Also reports:
    - WHY the pair is tied (same tiers, same scores? different tiers summing
      to same total?) — tells us whether the ties are structural accidents
      of the accumulator math or genuine equivalences at the evidence level.
    - Content previews of each tied gene — the human-readable "are these
      actually equivalent or not" check.

Input:  benchmarks/precision_probe_2026-04-15.json
        genome-bench-2026-04-14.db
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INPUT_JSON = REPO / "benchmarks" / "precision_probe_2026-04-15.json"
GENOME_DB = REPO / "genome-bench-2026-04-14.db"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def tier_breakdown_diff(contribs_a: dict, contribs_b: dict) -> str:
    """Pretty-print why two tied genes scored identically.

    Shows per-tier contributions side by side. Same tiers + same values
    = structural tie (identical evidence). Different tiers summing to
    the same total = coincidental tie (different evidence, same math).
    """
    all_tiers = sorted(set(contribs_a.keys()) | set(contribs_b.keys()))
    lines = [f"    {'tier':25s} {'gene_a':>10s} {'gene_b':>10s}"]
    for t in all_tiers:
        a = contribs_a.get(t, 0.0)
        b = contribs_b.get(t, 0.0)
        marker = "  " if a == b else " *"
        lines.append(f"    {t:25s} {a:10.4f} {b:10.4f}{marker}")
    sum_a = sum(contribs_a.values())
    sum_b = sum(contribs_b.values())
    lines.append(f"    {'TOTAL':25s} {sum_a:10.4f} {sum_b:10.4f}")
    return "\n".join(lines)


def main() -> int:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    conn = sqlite3.connect(str(GENOME_DB))
    conn.row_factory = sqlite3.Row

    def gene_summary(gene_id: str) -> dict:
        row = conn.execute(
            "SELECT gene_id, source_id, content, complement FROM genes WHERE gene_id = ?",
            (gene_id,),
        ).fetchone()
        if not row:
            return {"gene_id": gene_id, "missing": True}
        content = row["content"] or ""
        complement = row["complement"] or ""
        return {
            "gene_id": row["gene_id"],
            "source_id": row["source_id"],
            "content_len": len(content),
            "content_preview": content[:200].replace("\n", " | "),
            "complement_preview": complement[:200].replace("\n", " | "),
        }

    # Walk all queries and find those with any tied adjacent pair (gap==0,
    # scores both nonzero — not padding).
    tie_hits = []
    for q in data["results"]:
        scores = q["run_a"]["scores"]
        contribs = q["run_a"]["tier_contributions"]
        for i in range(len(scores) - 1):
            a = scores[i]
            b = scores[i + 1]
            if a["score"] == b["score"] and a["score"] != 0.0:
                tie_hits.append({
                    "idx": q["idx"],
                    "query": q["query"],
                    "rank_a": i,
                    "rank_b": i + 1,
                    "score": a["score"],
                    "gene_a": a["gene_id"],
                    "gene_b": b["gene_id"],
                    "contribs_a": contribs.get(a["gene_id"], {}),
                    "contribs_b": contribs.get(b["gene_id"], {}),
                })

    print(f"[tie-inspect] found {len(tie_hits)} tied adjacent pairs "
          f"across {len(set(t['idx'] for t in tie_hits))} queries\n")

    # For brevity, inspect at most the first tied pair per query — enough
    # to classify each query as "ties look similar" or "ties look distinct".
    seen_queries = set()
    for hit in tie_hits:
        if hit["idx"] in seen_queries:
            continue
        seen_queries.add(hit["idx"])

        print(f"=== query {hit['idx']}: {hit['query']!r}")
        print(f"    tied at ranks {hit['rank_a']}-{hit['rank_b']}  score={hit['score']:.4f}")

        sum_a = gene_summary(hit["gene_a"])
        sum_b = gene_summary(hit["gene_b"])

        print(f"\n  gene_a = {hit['gene_a']}")
        print(f"    source: {sum_a.get('source_id')}")
        print(f"    content ({sum_a.get('content_len', 0)} chars):")
        print(f"      {sum_a.get('content_preview', '')}")

        print(f"\n  gene_b = {hit['gene_b']}")
        print(f"    source: {sum_b.get('source_id')}")
        print(f"    content ({sum_b.get('content_len', 0)} chars):")
        print(f"      {sum_b.get('content_preview', '')}")

        print(f"\n  tier breakdown (* = differs between a and b):")
        print(tier_breakdown_diff(hit["contribs_a"], hit["contribs_b"]))
        print()

    # Aggregate: how many queries have tied pairs, how deep do the ties go,
    # are they always at the tail (rank 11+) or also in the head (rank 0-3)?
    by_rank = {}
    for hit in tie_hits:
        by_rank.setdefault(hit["rank_a"], 0)
        by_rank[hit["rank_a"]] += 1

    print(f"[tie-inspect] tied pairs by rank_a (where in top-k ties occur):")
    for r in sorted(by_rank):
        bar = "#" * by_rank[r]
        print(f"  rank {r:2d}: {by_rank[r]:3d}  {bar}")

    head_ties = sum(v for r, v in by_rank.items() if r < 4)
    tail_ties = sum(v for r, v in by_rank.items() if r >= 4)
    print(f"\n  head ties (rank 0-3): {head_ties}")
    print(f"  tail ties (rank 4+):  {tail_ties}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

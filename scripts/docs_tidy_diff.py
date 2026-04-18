"""Compare pre- and post-move probe outputs and write a human-readable diff.

Usage:
    python scripts/docs_tidy_diff.py \
        --pre benchmarks/docs_tidy_pre.json \
        --post benchmarks/docs_tidy_post.json \
        --out benchmarks/docs_tidy_diff.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def target_match(label: str, source_id: str | None) -> bool:
    if not source_id:
        return False
    stem = label.replace("_", "").lower()
    sid = source_id.replace("_", "").replace("/", "").replace("\\", "").lower()
    return stem in sid


def summarize(probe: dict) -> dict:
    label = probe["label"]
    cands = probe.get("candidates", []) or []
    target_hits = [c for c in cands if target_match(label, c.get("source_id"))]
    top = cands[0] if cands else None
    return {
        "label": label,
        "query": probe["query"],
        "top_rank0": (top or {}).get("source_id"),
        "top_score": (top or {}).get("score"),
        "target_in_top10": bool(target_hits),
        "target_best_rank": target_hits[0]["rank"] if target_hits else None,
        "target_best_score": target_hits[0]["score"] if target_hits else None,
        "target_source_id": target_hits[0]["source_id"] if target_hits else None,
        "target_count_in_top10": len(target_hits),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pre", required=True)
    p.add_argument("--post", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pre = json.loads(Path(args.pre).read_text())
    post = json.loads(Path(args.post).read_text())

    lines = []
    lines.append("# Docs-Tidy Helix Probe — Pre/Post Diff")
    lines.append("")
    lines.append(f"**Pre:**  {pre['timestamp_human']}  ({pre['genome']['total_genes']} genes)")
    lines.append(f"**Post:** {post['timestamp_human']}  ({post['genome']['total_genes']} genes)")
    lines.append("")
    delta_genes = (post["genome"]["total_genes"] or 0) - (pre["genome"]["total_genes"] or 0)
    lines.append(f"**Δ genes:** {delta_genes:+d} (new chunks for re-ingested files)")
    lines.append("")

    lines.append("## Per-query results")
    lines.append("")
    lines.append("| Query | Pre target rank | Post target rank | Pre top score | Post top score | Score Δ | Notes |")
    lines.append("|---|---|---|---|---|---|---|")

    improved = regressed = unchanged = 0
    new_found = new_lost = 0
    for pre_probe, post_probe in zip(pre["probes"], post["probes"]):
        assert pre_probe["label"] == post_probe["label"]
        pre_s = summarize(pre_probe)
        post_s = summarize(post_probe)
        label = pre_s["label"]

        pre_rank = pre_s["target_best_rank"]
        post_rank = post_s["target_best_rank"]
        pre_score = pre_s["target_best_score"]
        post_score = post_s["target_best_score"]

        if pre_rank is None and post_rank is None:
            notes = "miss both"
        elif pre_rank is None:
            new_found += 1
            notes = "FOUND post-move"
        elif post_rank is None:
            new_lost += 1
            notes = "LOST post-move"
        elif pre_rank == post_rank:
            unchanged += 1
            notes = "stable rank"
        elif pre_rank > post_rank:
            improved += 1
            notes = f"rank improved {pre_rank}→{post_rank}"
        else:
            regressed += 1
            notes = f"rank regressed {pre_rank}→{post_rank}"

        score_delta = ""
        if pre_score is not None and post_score is not None:
            score_delta = f"{(post_score - pre_score):+.2f}"
        pre_score_str = f"{pre_score:.2f}" if pre_score is not None else ""
        post_score_str = f"{post_score:.2f}" if post_score is not None else ""

        lines.append(
            f"| {label} | {pre_rank} | {post_rank} | "
            f"{pre_score_str} | {post_score_str} | "
            f"{score_delta} | {notes} |"
        )

    lines.append("")
    lines.append("## Summary counts")
    lines.append("")
    lines.append(f"- **Improved rank:** {improved}")
    lines.append(f"- **Regressed rank:** {regressed}")
    lines.append(f"- **Unchanged rank:** {unchanged}")
    lines.append(f"- **Found only post:** {new_found}")
    lines.append(f"- **Lost only post:** {new_lost}")
    lines.append("")

    # Orphan analysis: old source_ids that still appear post-move
    lines.append("## Old-path orphans still retrievable post-move")
    lines.append("")
    orphan_rows = []
    for post_probe in post["probes"]:
        label = post_probe["label"]
        for c in post_probe.get("candidates", []) or []:
            sid = c.get("source_id") or ""
            # Old path = docs/FOO.md (flat). New path = docs/subdir/FOO.md.
            # Orphan = retrieval returns a source_id where the file no longer exists at that path.
            if target_match(label, sid) and sid:
                path = Path(sid)
                # Check if file still exists at that exact path
                import os
                exists = os.path.exists(sid) if ":" in sid else False
                if not exists:
                    orphan_rows.append((label, c.get("rank"), c.get("score"), sid))
    if orphan_rows:
        lines.append("| Query | Rank | Score | Orphan source_id |")
        lines.append("|---|---|---|---|")
        for row in orphan_rows[:30]:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]:.2f} | `{row[3]}` |")
    else:
        lines.append("_No old-path orphans detected in top-10 results._")
    lines.append("")

    # Top-1 stability: did the rank-0 gene change?
    lines.append("## Rank-0 stability")
    lines.append("")
    lines.append("| Query | Pre rank-0 source_id | Post rank-0 source_id | Same? |")
    lines.append("|---|---|---|---|")
    for pre_probe, post_probe in zip(pre["probes"], post["probes"]):
        pre_top = (pre_probe["candidates"] or [{}])[0].get("source_id", "—")
        post_top = (post_probe["candidates"] or [{}])[0].get("source_id", "—")
        same = "yes" if pre_top == post_top else "no"
        lines.append(f"| {pre_probe['label']} | `{pre_top}` | `{post_top}` | {same} |")
    lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[diff] wrote {out}")
    print(f"[diff] improved={improved} regressed={regressed} unchanged={unchanged} "
          f"found_post={new_found} lost_post={new_lost}")


if __name__ == "__main__":
    main()

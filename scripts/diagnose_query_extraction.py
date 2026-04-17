"""Per-needle diagnostic: what does extract_query_signals pull out of each
bench query vs what tokens the gold source path carries.

Answers Waude's Step 1 on-ramp question:
    "Which tokens extract from each query vs which source tokens the gold
    gene has?"

No server needed. Reads NEEDLES from benchmarks/bench_needle.py and runs
the extraction in-process.

Usage::

    python scripts/diagnose_query_extraction.py
    python scripts/diagnose_query_extraction.py --json > ext.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from helix_context.accel import extract_query_signals, STOP_WORDS
from helix_context.genome import path_tokens
from bench_needle import NEEDLES


PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*\b")
QUOTED_RE = re.compile(r'["\']([^"\']+)["\']')
FILENAME_RE = re.compile(r"\b[\w-]+\.[a-zA-Z0-9]{1,6}\b")
CAMEL_COMPOUND_RE = re.compile(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z])")


def case_sensitive_tokens(query: str) -> dict:
    """What structural signals does the raw (pre-lowercase) query carry?"""
    return {
        "proper_nouns": PROPER_NOUN_RE.findall(query),
        "quoted": QUOTED_RE.findall(query),
        "filenames": FILENAME_RE.findall(query),
    }


def diagnose_needle(needle: dict) -> dict:
    query = needle["query"]
    gold_sources = needle.get("gold_source") or []

    domains, entities = extract_query_signals(query)
    extracted = set(domains) | set(entities)

    gold_tok_union: set[str] = set()
    per_source = {}
    for src in gold_sources:
        toks = path_tokens(src)
        per_source[src] = sorted(toks)
        gold_tok_union |= toks

    overlap = extracted & gold_tok_union
    missed_in_query = gold_tok_union - extracted
    extra_in_query = extracted - gold_tok_union

    case = case_sensitive_tokens(query)

    return {
        "name": needle["name"],
        "query": query,
        "gold_source": gold_sources,
        "extract": {
            "domains": domains,
            "entities": entities,
            "all": sorted(extracted),
        },
        "gold_path_tokens": sorted(gold_tok_union),
        "gold_per_source": per_source,
        "overlap": sorted(overlap),
        "missed_in_extract": sorted(missed_in_query),
        "noise_in_extract": sorted(extra_in_query),
        "case_signals": case,
        "overlap_count": len(overlap),
        "gold_token_count": len(gold_tok_union),
    }


def print_row(r: dict) -> None:
    name = r["name"]
    q = r["query"]
    ov = r["overlap_count"]
    gt = r["gold_token_count"]
    print(f"\n=== {name}  ({ov}/{gt} path tokens hit) ===")
    print(f"  query:     {q!r}")
    print(f"  domains:   {r['extract']['domains']}")
    print(f"  entities:  {r['extract']['entities']}")
    print(f"  gold toks: {r['gold_path_tokens']}")
    print(f"  overlap:   {r['overlap']}")
    print(f"  MISSED:    {r['missed_in_extract']}")
    print(f"  case sigs: "
          f"proper={r['case_signals']['proper_nouns']} "
          f"quoted={r['case_signals']['quoted']} "
          f"filenames={r['case_signals']['filenames']}")


def summarize(results: list[dict]) -> dict:
    n = len(results)
    any_hit = sum(1 for r in results if r["overlap_count"] > 0)
    zero_hit = n - any_hit
    total_tokens_missed = sum(len(r["missed_in_extract"]) for r in results)
    total_tokens_gold = sum(r["gold_token_count"] for r in results)

    # proper-noun-detectable queries (capitalized non-stop words present)
    with_proper = sum(
        1 for r in results
        if any(
            p.lower() not in STOP_WORDS and p.lower() not in {"how", "what", "which"}
            for p in r["case_signals"]["proper_nouns"]
        )
    )
    with_filename = sum(
        1 for r in results if r["case_signals"]["filenames"]
    )

    return {
        "n_needles": n,
        "needles_any_path_token_hit": any_hit,
        "needles_zero_path_token_hit": zero_hit,
        "path_token_recall": (
            (total_tokens_gold - total_tokens_missed) / max(total_tokens_gold, 1)
        ),
        "needles_with_proper_noun_signal": with_proper,
        "needles_with_filename_in_query": with_filename,
        "total_gold_path_tokens": total_tokens_gold,
        "total_missed_tokens": total_tokens_missed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of table.")
    args = ap.parse_args()

    results = [diagnose_needle(n) for n in NEEDLES]
    summary = summarize(results)

    if args.json:
        print(json.dumps({"needles": results, "summary": summary}, indent=2))
        return 0

    print(f"[diag] extraction vs gold-path tokens  ({summary['n_needles']} needles)")
    for r in results:
        print_row(r)

    print("\n=== summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>40s} : {v:.3f}")
        else:
            print(f"  {k:>40s} : {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

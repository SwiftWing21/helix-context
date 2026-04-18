"""Phase 5 packet benchmark — does freshness + coord confidence
actually change agent behavior on controlled scenarios?

Per ``docs/specs/2026-04-17-agent-context-index-build-spec.md``
§"Phase 5 — Benchmarks" the question is whether the packet's
``verified / stale_risk / needs_refresh`` labeling catches scenarios
a naive relevance-only retriever would happily return. This bench
builds N controlled in-memory genomes, runs ``build_context_packet``
against each, and reports precision/recall against the expected
status.

Scenario families covered in v1:
    1. stale_by_age           — verified long ago vs volatility half-life
    2. invalidated            — ``invalidated_at`` is set
    3. coordinate_mismatch    — content relevant, path off-target
    4. task_sensitivity       — same gene, different task_type → different verdict
    5. clean_verified         — fresh + aligned + high-authority (negative control)

Families deferred to later passes:
    - conflicting_config_values — gated on Phase 2 claims layer
    - duplicate_fact_across_shards — gated on real multi-shard setup
    - generated_log_contamination — partially covered by stale_by_age +
      hot volatility; full version needs ingest-time log-folder rules

Metrics per family:
    - correct_flag_rate  — of "should flag" cases, how many did flag
    - false_flag_rate    — of "should verify" cases, how many were flagged
    - avg_build_ms       — time to produce a labeled packet

Usage::

    python benchmarks/bench_packet.py
    python benchmarks/bench_packet.py --json
    python benchmarks/bench_packet.py --verbose  # per-scenario dump

Output always saved to
``benchmarks/results/packet_bench_<date>.json`` for tracking over
time. Re-runs overwrite the same-day artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from helix_context.context_packet import build_context_packet
from helix_context.genome import Genome
from helix_context.schemas import ChromatinState, EpigeneticMarkers, Gene, PromoterTags


# ── Synthetic gene builder ────────────────────────────────────────────

def _make_gene(
    content: str,
    *,
    source_id: str,
    domains: Optional[list[str]] = None,
    source_kind: str = "doc",
    volatility_class: str = "stable",
    authority_class: str = "primary",
    last_verified_at: Optional[float] = None,
) -> Gene:
    """Build a Gene with provenance fields pre-populated for the bench.

    Note: ``invalidated_at`` lives on the ``source_index`` row in
    main.db, not on the Gene itself. This bench encodes the "source
    changed" case via ``volatility_class=hot`` + a very old
    ``last_verified_at``, which forces the same ``needs_refresh``
    label through the freshness path. Real source_index-based
    invalidation tests land when Phase 1 full (source_index writes
    at ingest) ships.
    """
    gid = Genome.make_gene_id(content)
    return Gene(
        gene_id=gid,
        content=content,
        complement=content[:120],
        codons=["chunk_0"],
        promoter=PromoterTags(domains=domains or [], entities=domains or []),
        epigenetics=EpigeneticMarkers(),
        chromatin=ChromatinState.OPEN,
        is_fragment=False,
        source_id=source_id,
        source_kind=source_kind,
        volatility_class=volatility_class,
        authority_class=authority_class,
        last_verified_at=last_verified_at,
    )


# ── Scenario spec ─────────────────────────────────────────────────────

@dataclass
class Scenario:
    family: str
    name: str
    setup: Callable[[Genome], None]
    query: str
    task_type: str
    expected_status: str
    expected_note_substring: Optional[str] = None


NOW = 1_800_000_000.0  # fixed reference time for determinism


# Half-life thresholds for reference (see context_packet._HALF_LIFE_SECONDS):
#   stable = 7 days    → freshness 0.35 at age ≈ 7.4 days
#   medium = 12 hours  → freshness 0.35 at age ≈ 12.6 hours
#   hot    = 15 min    → freshness 0.35 at age ≈ 15.8 minutes
#
# Our scenario ages are chosen to land clearly above / below these
# thresholds so the bench isn't testing noise at the boundary.


def _setup_single(
    content: str,
    source_id: str,
    *,
    domains: Optional[list[str]] = None,
    source_kind: str = "doc",
    volatility_class: str = "stable",
    authority_class: str = "primary",
    last_verified_at: Optional[float] = None,
):
    def _run(g: Genome) -> None:
        gene = _make_gene(
            content,
            source_id=source_id,
            domains=domains,
            source_kind=source_kind,
            volatility_class=volatility_class,
            authority_class=authority_class,
            last_verified_at=last_verified_at,
        )
        g.upsert_gene(gene, apply_gate=False)
    return _run


SCENARIOS: list[Scenario] = [
    # ── 1. stale_by_age ──────────────────────────────────────────────
    Scenario(
        family="stale_by_age",
        name="stable_doc_30_days_old",
        setup=_setup_single(
            "Helix design notes for the index",
            "/repo/docs/helix/design.md",
            domains=["helix", "design"],
            source_kind="doc",
            volatility_class="stable",
            last_verified_at=NOW - 30 * 86_400,  # 30 days
        ),
        query="helix design",
        task_type="edit",
        expected_status="needs_refresh",
    ),
    Scenario(
        family="stale_by_age",
        name="hot_config_1_hour_old",
        setup=_setup_single(
            "jwt ttl set to fifteen minutes in helix auth config",
            "/repo/config/helix/auth.toml",
            domains=["helix", "auth", "config"],
            source_kind="config",
            volatility_class="hot",
            last_verified_at=NOW - 3600,  # 1 hour old, 4x hot half-life
        ),
        query="helix auth config",
        task_type="ops",
        expected_status="needs_refresh",
    ),
    Scenario(
        family="stale_by_age",
        name="medium_db_2_days_old",
        setup=_setup_single(
            "benchmark result scorerift ran at 0.15 threshold for helix run",
            "/repo/benchmarks/helix/run.db",
            domains=["helix", "benchmark"],
            source_kind="db",
            volatility_class="medium",
            last_verified_at=NOW - 2 * 86_400,  # 2 days, 4x medium half-life
        ),
        query="helix benchmark result",
        task_type="review",
        expected_status="needs_refresh",
    ),

    # ── 2. coordinate_mismatch ────────────────────────────────────────
    Scenario(
        family="coordinate_mismatch",
        name="product_name_wrong_repo_edit",
        setup=_setup_single(
            "The scorerift preset checks eight dimensions",
            "/repo/two-brain-audit/README.md",
            domains=["scorerift", "preset"],
            source_kind="doc",
            volatility_class="stable",
            last_verified_at=NOW - 60,  # fresh
        ),
        query="scorerift preset dimensions",
        task_type="edit",
        expected_status="needs_refresh",
        expected_note_substring="coordinate_confidence",
    ),
    Scenario(
        family="coordinate_mismatch",
        name="product_name_wrong_repo_explain",
        setup=_setup_single(
            "The scorerift preset checks eight dimensions",
            "/repo/two-brain-audit/README.md",
            domains=["scorerift", "preset"],
            source_kind="doc",
            volatility_class="stable",
            last_verified_at=NOW - 60,
        ),
        query="scorerift preset dimensions",
        task_type="explain",
        # For explain, coord low downgrades verified → stale_risk, not needs_refresh.
        expected_status="stale_risk",
        expected_note_substring="coordinate_confidence",
    ),

    # ── 3. task_sensitivity (same gene, different tasks) ──────────────
    # Setup: 6-hour-old medium-volatility doc. freshness ≈ 0.606.
    # - explain (low risk): 0.35 ≤ 0.606 → verified
    # - edit (high risk):   0.35 ≤ 0.606 ≤ 0.70 → stale_risk
    Scenario(
        family="task_sensitivity",
        name="medium_6h_as_explain",
        setup=_setup_single(
            "helix config overview covering port and model settings",
            "/repo/docs/helix/config-overview.md",
            domains=["helix", "config"],
            source_kind="doc",
            volatility_class="medium",
            last_verified_at=NOW - 6 * 3600,
        ),
        query="helix config overview",
        task_type="explain",
        expected_status="verified",
    ),
    Scenario(
        family="task_sensitivity",
        name="medium_6h_as_edit",
        setup=_setup_single(
            "helix config overview covering port and model settings",
            "/repo/docs/helix/config-overview.md",
            domains=["helix", "config"],
            source_kind="doc",
            volatility_class="medium",
            last_verified_at=NOW - 6 * 3600,
        ),
        query="helix config overview",
        task_type="edit",
        expected_status="stale_risk",
    ),

    # ── 4. authority_downgrade ────────────────────────────────────────
    Scenario(
        family="authority_downgrade",
        name="inferred_authority_on_ops",
        setup=_setup_single(
            "helix may listen on port eleven thousand something",
            "/repo/notes/helix/misc.md",
            domains=["helix", "port"],
            source_kind="doc",
            authority_class="inferred",  # 0.45 < 0.55 threshold
            volatility_class="stable",
            last_verified_at=NOW - 60,  # fresh
        ),
        query="helix port",
        task_type="ops",
        expected_status="needs_refresh",
    ),

    # ── 5. clean_verified (negative control) ──────────────────────────
    Scenario(
        family="clean_verified",
        name="fresh_aligned_stable_explain",
        setup=_setup_single(
            "Helix sharding Phase 2 routes queries by fingerprint overlap",
            "/repo/docs/helix/sharding.md",
            domains=["helix", "sharding"],
            source_kind="doc",
            volatility_class="stable",
            authority_class="primary",
            last_verified_at=NOW - 60,  # fresh
        ),
        query="helix sharding",
        task_type="explain",
        expected_status="verified",
    ),
    Scenario(
        family="clean_verified",
        name="fresh_aligned_code_edit",
        setup=_setup_single(
            "def apply_gate(gene): return gene.chromatin for helix genome",
            "/repo/helix_context/helix/genome.py",
            domains=["helix", "genome"],
            source_kind="code",
            volatility_class="stable",
            authority_class="primary",
            last_verified_at=NOW - 60,
        ),
        query="helix genome",
        task_type="edit",
        expected_status="verified",
    ),
]


# ── Runner ────────────────────────────────────────────────────────────

def run_scenario(s: Scenario) -> dict:
    g = Genome(path=":memory:")
    # Disable density gate so test genes don't get demoted into
    # heterochromatin before retrieval can see them.
    _original_upsert = g.upsert_gene
    def _ungated(gene, apply_gate=False):
        return _original_upsert(gene, apply_gate=apply_gate)
    g.upsert_gene = _ungated

    try:
        s.setup(g)
        t0 = time.perf_counter()
        packet = build_context_packet(
            s.query,
            task_type=s.task_type,
            genome=g,
            now_ts=NOW,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        g.close()

    items = list(packet.verified) + list(packet.stale_risk)
    if not items:
        actual_status = "no_results"
    else:
        actual_status = items[0].status

    status_ok = actual_status == s.expected_status

    note_ok = True
    matched_note = None
    if s.expected_note_substring:
        matched_note = next(
            (n for n in packet.notes if s.expected_note_substring in n), None,
        )
        note_ok = matched_note is not None

    passed = status_ok and note_ok

    return {
        "family": s.family,
        "name": s.name,
        "query": s.query,
        "task_type": s.task_type,
        "expected_status": s.expected_status,
        "actual_status": actual_status,
        "status_ok": status_ok,
        "note_ok": note_ok,
        "matched_note": matched_note,
        "passed": passed,
        "time_ms": round(elapsed_ms, 2),
        "verified_count": len(packet.verified),
        "stale_risk_count": len(packet.stale_risk),
        "refresh_target_count": len(packet.refresh_targets),
        "notes": list(packet.notes),
    }


def summarize(results: list[dict]) -> dict:
    by_family: dict[str, dict] = {}
    for r in results:
        fam = r["family"]
        entry = by_family.setdefault(fam, {"total": 0, "passed": 0, "time_ms_sum": 0.0})
        entry["total"] += 1
        entry["passed"] += int(r["passed"])
        entry["time_ms_sum"] += r["time_ms"]

    for fam, entry in by_family.items():
        entry["pass_rate"] = round(entry["passed"] / max(entry["total"], 1), 3)
        entry["avg_time_ms"] = round(entry["time_ms_sum"] / max(entry["total"], 1), 2)
        del entry["time_ms_sum"]

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {
        "total_scenarios": total,
        "total_passed": passed,
        "overall_pass_rate": round(passed / max(total, 1), 3),
        "by_family": by_family,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="Emit raw JSON to stdout.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-scenario outcome table.")
    args = ap.parse_args()

    results = [run_scenario(s) for s in SCENARIOS]
    summary = summarize(results)

    if args.json:
        print(json.dumps({"scenarios": results, "summary": summary}, indent=2))
        return 0

    print(f"[packet-bench] {summary['total_passed']}/{summary['total_scenarios']} "
          f"scenarios passed ({summary['overall_pass_rate']*100:.0f}%)")
    print()
    print("=== by family ===")
    for fam, entry in sorted(summary["by_family"].items()):
        mark = "+" if entry["pass_rate"] == 1.0 else "-"
        print(f"  {mark} {fam:<22}  {entry['passed']:>2}/{entry['total']:<2}  "
              f"pass={entry['pass_rate']:.2f}  avg={entry['avg_time_ms']:.1f}ms")

    if args.verbose or summary["overall_pass_rate"] < 1.0:
        print()
        print("=== per scenario ===")
        for r in results:
            mark = "+" if r["passed"] else "-"
            print(
                f"  {mark} {r['family']:<22} {r['name']:<34} "
                f"expected={r['expected_status']:<14} "
                f"actual={r['actual_status']:<14} "
                f"{r['time_ms']:.1f}ms"
            )
            if not r["passed"] and r["notes"]:
                for n in r["notes"]:
                    print(f"      note: {n}")

    # Always save artifact
    date_str = time.strftime("%Y-%m-%d")
    out_path = Path(__file__).resolve().parent / "results" / f"packet_bench_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"scenarios": results, "summary": summary}, f, indent=2)
    print(f"\nResults: {out_path}")

    return 0 if summary["overall_pass_rate"] >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())

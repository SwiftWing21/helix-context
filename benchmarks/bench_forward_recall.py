"""bench_forward_recall.py -- Phase 4 TCM forward-recall asymmetry probe.

Empirically verifies the Howard & Kahana (2002) prediction that a session's
drifting context vector preferentially surfaces later-encoded items over
earlier-encoded ones when queried. This is the decision gate for Phase 4
of the 2026-04-11 8D dimensional roadmap.

The shipped mechanism (helix_context/tcm.py):
    - SessionContext.update() integrates each accessed gene's input vector
      via Howard 2005 Eq. 16 velocity input + Gram-Schmidt orthogonalization.
    - tcm_bonus() returns a per-gene cosine-similarity bonus (weight * cos
      between current context_vector and gene_input_vector), clamped >= 0.
    - context_manager.py:648-659 re-sorts retrieval candidates using that
      bonus when a SessionContext is attached.

Procedure:
    1. Fresh :memory: Genome, N synthetic genes sharing one domain tag
       (so tier-1 retrieval scores are comparable) and distinct per-gene
       entities (so each gene has a distinct gene_input_vector under the
       hash-based fallback path).
    2. Walk a SessionContext forward, updating with genes[0..N-1] in order.
    3. Call genome.query_genes(domain), then apply tcm_bonus and re-sort,
       replicating context_manager's Step 3.25 logic.
    4. Compute the forward-asymmetry index (fai): for every pair (i, j)
       with i < j (earlier, later), fraction where gene[j] ranks higher
       (lower rank-number) than gene[i] under TCM ordering. fai = 1.0 is
       perfect forward asymmetry; 0.5 is random; 0.0 is perfect backward.
    5. Report tier-1 baseline fai (without TCM bonus) alongside to rule
       out insertion-order artifacts.

Aggregate across multiple seeds for stability.

Decision gate (per roadmap §Phase 4):

    The relevant question is whether TCM ADDS forward asymmetry over
    the tier-1 baseline, not whether absolute fai > 0.5 -- tier-1 on
    a synthetic genome with tied scores has its own structural order
    (SQLite rowid, etc) that gives a non-random baseline. Phase 4's
    claim is that TCM produces asymmetry via velocity drift.

    delta_fai = mean_tcm_fai - mean_tier1_fai

    delta_fai > +0.05  -> PASS; TCM adds forward asymmetry as claimed
    delta_fai < -0.05  -> REGRESSION; TCM reverses the ordering
    |delta_fai| <= 0.05 -> NEUTRAL; TCM bonus has no measurable effect
                            on rank order. Options: raise bonus weight,
                            revisit the velocity projection, or revisit
                            the docstring claim about forward asymmetry.

Usage:
    python benchmarks/bench_forward_recall.py
    N=20 SEEDS=10 python benchmarks/bench_forward_recall.py
    TCM_WEIGHT=0.5 python benchmarks/bench_forward_recall.py  # weight sweep
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure the repo is importable when invoked from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from helix_context.genome import Genome
from helix_context.schemas import (
    ChromatinState,
    EpigeneticMarkers,
    Gene,
    PromoterTags,
)
from helix_context.tcm import SessionContext, tcm_bonus


SHARED_DOMAIN = "ephemeron"


def _make_gene(i: int, rng: random.Random) -> Gene:
    """Synthetic gene at sequence position i.

    All genes share one domain (so they're all tier-1 candidates for the
    same query), but carry position-specific entities so the hash-based
    gene_input_vector produces distinct vectors for each gene.
    """
    entities = [
        f"item_{i:02d}",
        f"position_{i:02d}",
        f"token_{rng.randint(0, 9_999)}",
    ]
    codons = [f"seq_{i:02d}_a", f"seq_{i:02d}_b", f"seq_{i:02d}_c"]
    content = f"Synthetic sequence item {i} for forward-recall bench"
    gid = Genome.make_gene_id(f"forward_recall_gene_{i}")
    return Gene(
        gene_id=gid,
        content=content,
        complement=f"Gene {i} summary",
        codons=codons,
        promoter=PromoterTags(
            domains=[SHARED_DOMAIN],
            entities=entities,
            intent="test",
            summary=content[:80],
        ),
        epigenetics=EpigeneticMarkers(),
        chromatin=ChromatinState.OPEN,
        is_fragment=False,
    )


def _fai(rank_map: Dict[str, int], genes: List[Gene]) -> float:
    """Forward-asymmetry index: frac of pairs (i,j), i<j, where j ranks higher.

    Genes missing from ``rank_map`` (e.g., dropped by a retrieval cap)
    are treated as rank = +inf so they always lose pairwise comparisons
    against retrieved genes. Pairs of two missing genes are skipped.
    """
    n = len(genes)
    if n < 2:
        return 0.0
    inf = float("inf")
    count = 0
    total = 0
    for i in range(n):
        ri = rank_map.get(genes[i].gene_id, inf)
        for j in range(i + 1, n):
            rj = rank_map.get(genes[j].gene_id, inf)
            if ri == inf and rj == inf:
                continue  # both missing, uninformative pair
            total += 1
            if rj < ri:
                count += 1
    return count / total if total else 0.0


def run_probe(n: int, weight: float, seed: int) -> Dict:
    rng = random.Random(seed)

    genome = Genome(path=":memory:")
    try:
        genes = [_make_gene(i, rng) for i in range(n)]
        for g in genes:
            genome.upsert_gene(g, apply_gate=False)

        # Walk session through all N items in encoding order.
        session = SessionContext()
        for g in genes:
            session.update_from_gene(g)

        # Retrieve candidates for the shared domain. Pass max_genes=n so
        # every synthetic gene is considered (default cap is 8 and the
        # bench probes need every item in the ranking).
        candidates = genome.query_genes([SHARED_DOMAIN], [], max_genes=n)
        scores = dict(genome.last_query_scores or {})
        bonuses = tcm_bonus(session, candidates, weight=weight)

        # TCM-active ranking: tier-1 score + TCM bonus (replicates
        # context_manager.py:654-657).
        tcm_sorted = sorted(
            candidates,
            key=lambda g: scores.get(g.gene_id, 0) + bonuses.get(g.gene_id, 0),
            reverse=True,
        )
        tcm_rank = {g.gene_id: r for r, g in enumerate(tcm_sorted)}

        # Tier-1 baseline ranking (no TCM bonus applied).
        tier1_sorted = sorted(
            candidates,
            key=lambda g: scores.get(g.gene_id, 0),
            reverse=True,
        )
        tier1_rank = {g.gene_id: r for r, g in enumerate(tier1_sorted)}

        tcm_fai = _fai(tcm_rank, genes)
        tier1_fai = _fai(tier1_rank, genes)

        last_gene_tcm_rank = tcm_rank[genes[-1].gene_id]
        first_gene_tcm_rank = tcm_rank[genes[0].gene_id]

        bonus_values = list(bonuses.values())
        max_bonus = max(bonus_values) if bonus_values else 0.0
        mean_bonus = (
            sum(bonus_values) / len(bonus_values) if bonus_values else 0.0
        )

        return {
            "seed": seed,
            "n": n,
            "weight": weight,
            "tcm_fai": tcm_fai,
            "tier1_fai": tier1_fai,
            "last_gene_tcm_rank": last_gene_tcm_rank,
            "first_gene_tcm_rank": first_gene_tcm_rank,
            "mean_bonus": mean_bonus,
            "max_bonus": max_bonus,
            "n_candidates": len(candidates),
        }
    finally:
        genome.close()


def main() -> int:
    n = int(os.environ.get("N", "10"))
    base_seed = int(os.environ.get("SEED", "42"))
    n_seeds = int(os.environ.get("SEEDS", "5"))
    weight = float(os.environ.get("TCM_WEIGHT", "0.3"))
    ts = time.strftime("%Y-%m-%d_%H%M")
    out_path = os.environ.get(
        "OUTPUT",
        str(_REPO_ROOT / "benchmarks" / f"forward_recall_{ts}.json"),
    )

    runs = [run_probe(n=n, weight=weight, seed=base_seed + s) for s in range(n_seeds)]

    tcm_fais = [r["tcm_fai"] for r in runs]
    tier1_fais = [r["tier1_fai"] for r in runs]
    last_ranks = [r["last_gene_tcm_rank"] for r in runs]
    first_ranks = [r["first_gene_tcm_rank"] for r in runs]
    max_bonuses = [r["max_bonus"] for r in runs]

    mean_tcm = statistics.mean(tcm_fais)
    mean_tier1 = statistics.mean(tier1_fais)
    delta_fai = mean_tcm - mean_tier1

    aggregate = {
        "mean_tcm_fai": mean_tcm,
        "stdev_tcm_fai": statistics.stdev(tcm_fais) if len(tcm_fais) > 1 else 0.0,
        "mean_tier1_fai": mean_tier1,
        "delta_fai": delta_fai,
        "mean_last_gene_tcm_rank": statistics.mean(last_ranks),
        "mean_first_gene_tcm_rank": statistics.mean(first_ranks),
        "mean_max_bonus": statistics.mean(max_bonuses),
    }

    if delta_fai > 0.05:
        decision = "PASS"
        rationale = (
            f"TCM adds forward asymmetry over tier-1 baseline "
            f"(delta_fai = {delta_fai:+.3f} > +0.05). Phase 4 "
            f"delivers its intended effect."
        )
    elif delta_fai < -0.05:
        decision = "REGRESSION"
        rationale = (
            f"TCM reverses the tier-1 ordering "
            f"(delta_fai = {delta_fai:+.3f} < -0.05). Phase 4 is "
            f"degrading retrieval; investigate."
        )
    else:
        decision = "NEUTRAL"
        rationale = (
            f"TCM bonus has no measurable effect on rank order "
            f"(delta_fai = {delta_fai:+.3f}, |delta| <= 0.05). "
            f"Either the bonus weight is too low relative to tier-1 "
            f"score spread, or the velocity-integrated context vector "
            f"is not systematically closer to later-encoded genes' "
            f"input vectors than to earlier ones. The docstring claim "
            f"of forward-recall asymmetry is not substantiated by this "
            f"setup."
        )

    summary = {
        "bench": "forward_recall_asymmetry",
        "timestamp": ts,
        "n_items": n,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "tcm_bonus_weight": weight,
        "decision": decision,
        "rationale": rationale,
        "aggregate": aggregate,
        "runs": runs,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))

    print()
    print(
        f"Phase 4 TCM forward-recall asymmetry probe "
        f"-- N={n}, seeds={n_seeds}, weight={weight}"
    )
    print("-" * 78)
    print(
        f"{'seed':>6}  {'tcm_fai':>8}  {'tier1_fai':>10}  "
        f"{'last_rank':>10}  {'first_rank':>10}  {'max_bonus':>10}"
    )
    for r in runs:
        print(
            f"{r['seed']:>6}  "
            f"{r['tcm_fai']:>8.3f}  "
            f"{r['tier1_fai']:>10.3f}  "
            f"{r['last_gene_tcm_rank']:>10d}  "
            f"{r['first_gene_tcm_rank']:>10d}  "
            f"{r['max_bonus']:>10.4f}"
        )
    print("-" * 78)
    print(
        f"{'mean':>6}  "
        f"{aggregate['mean_tcm_fai']:>8.3f}  "
        f"{aggregate['mean_tier1_fai']:>10.3f}  "
        f"{aggregate['mean_last_gene_tcm_rank']:>10.2f}  "
        f"{aggregate['mean_first_gene_tcm_rank']:>10.2f}  "
        f"{aggregate['mean_max_bonus']:>10.4f}"
    )
    print()
    print(f"  delta_fai = tcm_fai - tier1_fai = {delta_fai:+.3f}")
    print(f"  Decision: {decision}")
    print(f"    {rationale}")
    print()
    print(f"Full result: {out_path}")

    # Non-zero exit on regression so CI/watchers can alarm. NEUTRAL
    # and PASS both exit 0.
    return 1 if decision == "REGRESSION" else 0


if __name__ == "__main__":
    raise SystemExit(main())

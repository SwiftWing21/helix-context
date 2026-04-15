"""Lockstep 9x9 matrix test — follow-up to scripts/pwpc/lockstep_test.py.

The scalar test (mean_z, min_z, all9_fired, n_tiers_fired) failed the |r| >= 0.2
gate on the 791-row enriched export. LOCKSTEP_TEST.md §3 noted: "A scalar
summary of the 9-tier pattern loses the structure; the 9x9 matrix may still
carry the signal but not any single projection of it."

This script tests the full matrix. Hypothesis: pairwise products of per-tier
z-scores (the 81 entries of the per-row outer-product z ⊗ zᵀ, or the 45
unique ones by symmetry) carry bucket signal that single scalar reductions
lose.

Three complementary tests:

  1. **Per-row pairwise features (81 or 45)**
     For each row, build z ∈ R^9 (z_i = 0 when tier didn't fire).
     Compute the 9x9 rank-1 outer product M = z zᵀ. Flatten to 45 unique
     entries. Test each against bucket label with point-biserial r.
     Gate: any entry |r| >= 0.2?

  2. **Population correlation matrix delta (A vs B)**
     Stack all A-bucket z-vectors into matrix Z_A ∈ R^(n_A, 9).
     Same for Z_B. Compute C_A = corr(Z_A), C_B = corr(Z_B).
     Compare: Frobenius distance, max entrywise delta, eigenvalue spectra.
     This is a global "do the tiers co-fire differently in failure vs success"
     test.

  3. **Top-k eigenvalue features**
     Per-row, compute the 9-vector z, then project onto top-3 eigenvectors
     of the *pooled* correlation matrix. Those 3 projections become 3
     features per row. Test each against bucket.

Usage:
    python scripts/pwpc/lockstep_matrix_test.py <export.json> [--out <md>]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pointbiserialr

TIER_KEYS = [
    "fts5", "splade", "sema_boost", "lex_anchor",
    "tag_exact", "tag_prefix", "pki", "harmonic", "sr",
]
N_TIERS = len(TIER_KEYS)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tier_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, fired_mask, bucket) arrays.

    X shape (N, 9): raw tier scores, 0 where tier didn't fire.
    fired_mask (N, 9): 1 where tier fired, 0 otherwise.
    bucket (N,): 0 for A, 1 for B, -1 for unbuckted (filtered out).
    """
    X = np.zeros((len(rows), N_TIERS), dtype=float)
    fired = np.zeros((len(rows), N_TIERS), dtype=float)
    bucket = np.full(len(rows), -1, dtype=int)
    for i, r in enumerate(rows):
        b = r.get("bucket")
        if b == "A":
            bucket[i] = 0
        elif b == "B":
            bucket[i] = 1
        feats = r.get("tier_features") or {}
        for j, t in enumerate(TIER_KEYS):
            v = feats.get(t)
            if v is None:
                continue
            try:
                X[i, j] = float(v)
                fired[i, j] = 1.0
            except (TypeError, ValueError):
                continue
    keep = bucket != -1
    return X[keep], fired[keep], bucket[keep]


def z_score_rows(X: np.ndarray, fired: np.ndarray) -> np.ndarray:
    """Per-tier z-score, conditional on the tier firing.

    Non-firing tiers stay at 0 (their population mean once z-scored conditional
    on firing would be NaN; 0 is the neutral "no contribution" value we want).
    """
    Z = np.zeros_like(X)
    for j in range(N_TIERS):
        mask = fired[:, j] > 0
        if mask.sum() < 2:
            continue
        vals = X[mask, j]
        mu = vals.mean()
        sigma = vals.std(ddof=0) or 1.0
        Z[mask, j] = (X[mask, j] - mu) / sigma
    return Z


def test1_per_row_pairwise(Z: np.ndarray, bucket: np.ndarray) -> list[dict]:
    """Test 1: each unique (i, j) pairwise product as a feature."""
    results = []
    for i in range(N_TIERS):
        for j in range(i, N_TIERS):  # unique pairs including diagonal
            feat = Z[:, i] * Z[:, j]
            if np.ptp(feat) < 1e-12:
                r, p = 0.0, 1.0
            else:
                r, p = pointbiserialr(bucket, feat)
                if math.isnan(r):
                    r, p = 0.0, 1.0
            results.append({
                "tier_i": TIER_KEYS[i],
                "tier_j": TIER_KEYS[j],
                "r": r,
                "abs_r": abs(r),
                "p": p,
                "mean_A": float(feat[bucket == 0].mean()) if (bucket == 0).any() else 0.0,
                "mean_B": float(feat[bucket == 1].mean()) if (bucket == 1).any() else 0.0,
            })
    results.sort(key=lambda d: d["abs_r"], reverse=True)
    return results


def test2_population_corr(Z: np.ndarray, bucket: np.ndarray) -> dict:
    """Test 2: do tiers co-fire differently in A vs B?"""
    ZA = Z[bucket == 0]
    ZB = Z[bucket == 1]
    if len(ZA) < 2 or len(ZB) < 2:
        return {"error": f"insufficient bucket sizes A={len(ZA)} B={len(ZB)}"}
    CA = np.corrcoef(ZA, rowvar=False)
    CB = np.corrcoef(ZB, rowvar=False)
    # Handle NaN (zero-variance column)
    CA = np.nan_to_num(CA)
    CB = np.nan_to_num(CB)
    delta = CA - CB
    frob = float(np.linalg.norm(delta, ord="fro"))
    max_abs = float(np.max(np.abs(delta)))
    ai, aj = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    # Top-k biggest entrywise deltas
    flat = delta.copy()
    flat[np.tril_indices(N_TIERS)] = 0  # upper triangle only (symmetric)
    top = []
    flat_abs = np.abs(flat)
    for _ in range(6):
        if flat_abs.max() < 1e-12:
            break
        i, j = np.unravel_index(np.argmax(flat_abs), flat_abs.shape)
        top.append({
            "tier_i": TIER_KEYS[i],
            "tier_j": TIER_KEYS[j],
            "corr_A": float(CA[i, j]),
            "corr_B": float(CB[i, j]),
            "delta": float(delta[i, j]),
        })
        flat_abs[i, j] = 0
    # Eigenvalue spectra
    eigs_A = np.sort(np.linalg.eigvalsh(CA))[::-1]
    eigs_B = np.sort(np.linalg.eigvalsh(CB))[::-1]
    return {
        "n_A": int(len(ZA)),
        "n_B": int(len(ZB)),
        "frob_delta": frob,
        "max_abs_delta": max_abs,
        "max_delta_pair": f"{TIER_KEYS[ai]}<->{TIER_KEYS[aj]}",
        "top_delta_pairs": top,
        "eigs_A_top3": [float(x) for x in eigs_A[:3]],
        "eigs_B_top3": [float(x) for x in eigs_B[:3]],
        "cond_A": float(eigs_A[0] / max(eigs_A[-1], 1e-12)),
        "cond_B": float(eigs_B[0] / max(eigs_B[-1], 1e-12)),
    }


def test3_eigen_projection(Z: np.ndarray, bucket: np.ndarray, k: int = 3) -> list[dict]:
    """Test 3: project onto top-k eigenvectors of the pooled correlation matrix."""
    C_pool = np.corrcoef(Z, rowvar=False)
    C_pool = np.nan_to_num(C_pool)
    eigvals, eigvecs = np.linalg.eigh(C_pool)
    # Descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    results = []
    for m in range(min(k, N_TIERS)):
        proj = Z @ eigvecs[:, m]
        r, p = pointbiserialr(bucket, proj)
        if math.isnan(r):
            r, p = 0.0, 1.0
        results.append({
            "eigen_index": m,
            "eigval": float(eigvals[m]),
            "pct_var": float(eigvals[m] / eigvals.sum()),
            "top_loadings": {
                TIER_KEYS[idx]: float(eigvecs[idx, m])
                for idx in np.argsort(np.abs(eigvecs[:, m]))[::-1][:3]
            },
            "r": r,
            "abs_r": abs(r),
            "p": p,
            "mean_A": float(proj[bucket == 0].mean()) if (bucket == 0).any() else 0.0,
            "mean_B": float(proj[bucket == 1].mean()) if (bucket == 1).any() else 0.0,
        })
    return results


def format_report(path: Path, test1: list[dict], test2: dict, test3: list[dict], n_total: int) -> str:
    lines: list[str] = []
    lines.append("# Lockstep 9x9 matrix test — results")
    lines.append("")
    lines.append(f"**Source:** `{path.name}` (N = {n_total}; A = {test2.get('n_A', '?')}, B = {test2.get('n_B', '?')})")
    lines.append("")
    lines.append("Follow-up to LOCKSTEP_TEST.md: the scalar gate failed at |r|>=0.2. This "
                 "test checks whether the full 9x9 structure carries signal.")
    lines.append("")
    lines.append("## Test 1 — per-row pairwise products of z-scores (45 unique features)")
    lines.append("")
    lines.append("For each row, z in R^9 (tier didn't fire -> 0). Feature_ij = z_i * z_j. "
                 "Point-biserial r vs bucket label (A=0, B=1).")
    lines.append("")
    lines.append("### Top 10 features by |r|")
    lines.append("")
    lines.append("| tier_i | tier_j | r | mean_A | mean_B | p |")
    lines.append("|---|---|---|---|---|---|")
    for d in test1[:10]:
        lines.append(f"| {d['tier_i']} | {d['tier_j']} | {d['r']:+.4f} | "
                     f"{d['mean_A']:+.3f} | {d['mean_B']:+.3f} | {d['p']:.3g} |")
    passed = [d for d in test1 if d["abs_r"] >= 0.2]
    lines.append("")
    if passed:
        lines.append(f"**Gate PASSES — {len(passed)} features at |r| >= 0.2:**")
        lines.append("")
        for d in passed:
            sign = "ANTIRES" if d["r"] > 0 else "CONVENT"
            lines.append(f"- {d['tier_i']} x {d['tier_j']}: r={d['r']:+.4f} [{sign}]")
    else:
        top = test1[0]
        lines.append(f"**Gate FAILS** — max |r| = {top['abs_r']:.4f} on "
                     f"{top['tier_i']} x {top['tier_j']}. Signal is directional but sub-threshold.")
    lines.append("")
    lines.append("## Test 2 — A vs B correlation-matrix delta")
    lines.append("")
    if "error" in test2:
        lines.append(f"{test2['error']}")
    else:
        lines.append(f"- **Frobenius ||C_A - C_B||:** {test2['frob_delta']:.4f} "
                     f"(baseline: 0 if identical, ~1-2 typical for small samples)")
        lines.append(f"- **Max entrywise |ΔC|:** {test2['max_abs_delta']:.4f} "
                     f"on {test2['max_delta_pair']}")
        lines.append(f"- **Top eigenvalues A:** {[f'{x:.3f}' for x in test2['eigs_A_top3']]}")
        lines.append(f"- **Top eigenvalues B:** {[f'{x:.3f}' for x in test2['eigs_B_top3']]}")
        lines.append(f"- **Condition number A:** {test2['cond_A']:.1f}, B: {test2['cond_B']:.1f}")
        lines.append("")
        lines.append("### Top pairs by |ΔC|")
        lines.append("")
        lines.append("| tier_i | tier_j | corr_A | corr_B | delta |")
        lines.append("|---|---|---|---|---|")
        for d in test2.get("top_delta_pairs", []):
            lines.append(f"| {d['tier_i']} | {d['tier_j']} | {d['corr_A']:+.3f} | "
                         f"{d['corr_B']:+.3f} | {d['delta']:+.3f} |")
    lines.append("")
    lines.append("## Test 3 — projection onto top-k eigenvectors of pooled correlation")
    lines.append("")
    lines.append("| k | eigval | %var | r vs bucket | top loadings |")
    lines.append("|---|---|---|---|---|")
    for d in test3:
        loads = ", ".join(f"{k}={v:+.2f}" for k, v in d["top_loadings"].items())
        lines.append(f"| {d['eigen_index']} | {d['eigval']:.3f} | {d['pct_var']*100:.1f}% | "
                     f"{d['r']:+.4f} | {loads} |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    any_pass = any(d["abs_r"] >= 0.2 for d in test1) or any(d["abs_r"] >= 0.2 for d in test3)
    if any_pass:
        lines.append("**At least one matrix-derived feature clears the |r|>=0.2 gate.** The "
                     "scalar LOCKSTEP_TEST was losing structure; the pairwise / eigen "
                     "projection preserves it. This unblocks the agreement head design — we "
                     "have a feature to train against.")
    else:
        lines.append("**Gate still fails on matrix features.** Next moves: (1) query-type "
                     "segmentation, (2) include semantic cos(query, candidate) as a 10th "
                     "dimension, (3) try non-linear projections (e.g. kernelised). Even in "
                     "this case, test 2 tells Gordon+Todd whether the A and B correlation "
                     "structures are meaningfully different at the population level.")
    lines.append("")
    lines.append("— Generated by `scripts/pwpc/lockstep_matrix_test.py`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path, help="cwola export JSON")
    ap.add_argument("--out", type=Path, default=None, help="Markdown output path")
    args = ap.parse_args()

    rows = load_rows(args.export)
    X, fired, bucket = tier_matrix(rows)
    Z = z_score_rows(X, fired)

    test1 = test1_per_row_pairwise(Z, bucket)
    test2 = test2_population_corr(Z, bucket)
    test3 = test3_eigen_projection(Z, bucket)

    report = format_report(args.export, test1, test2, test3, len(bucket))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()

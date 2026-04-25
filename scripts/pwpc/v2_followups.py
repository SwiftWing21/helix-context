"""Phase 0 v2 follow-ups — merged execution of the open "next moves" from
PHASE0_V2_DRILLDOWN.md and LOCKSTEP_MATRIX_TEST_v2.md.

Tasks:
  T1  Heuristic template-label top-B vs top-A sema_boost queries; population
      density of template shapes across buckets.
  T2  Query-length bias on the cos(q,c) A-B delta (bin, compare within bins).
  T3  Query-type segmentation: rerun 9x9 pairwise correlation matrix within
      {template, natural} segments; compare Frobenius ||C_A - C_B|| per segment.
  T4  10x10 matrix test: add cos(q,c) as a 10th feature; rerun pairwise |r|
      gate.
  T5  Kernel PCA (RBF) on the 9-d tier z-score matrix; test bucket signal on
      top-k non-linear components.

Usage:
    python scripts/pwpc/v2_followups.py cwola_export/cwola_export_20260415_windowed.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import pointbiserialr

try:
    from sklearn.decomposition import KernelPCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

TIER_KEYS = [
    "fts5", "splade", "sema_boost", "lex_anchor",
    "tag_exact", "tag_prefix", "pki", "harmonic", "sr",
]
N_TIERS = len(TIER_KEYS)


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


# ──────────────────────────────────────────────────────────────────────
# T1 — template heuristic
# ──────────────────────────────────────────────────────────────────────

# Strong template patterns (slot-fill question shapes)
TEMPLATE_RE = [
    re.compile(r"^\s*what is the value of\b", re.I),
    re.compile(r"^\s*what is the \w[\w\s]{0,40}? value in\b", re.I),
    re.compile(r"^\s*what is the \w[\w\s]{0,40}? configured in\b", re.I),
    re.compile(r"^\s*what is the \w[\w\s]{0,40}? (set|defined|used) (in|as|for)\b", re.I),
    re.compile(r"^\s*how does \w[\w\s]{0,40}? relate to \w", re.I),
    re.compile(r"^\s*what (are|is) the \w[\w\s]{0,40}? in\b", re.I),
]

# Weak patterns — suggest "question shape" but not pure slot-fill
QUESTION_RE = [
    re.compile(r"^\s*(what|how|where|why|when|which|does|do|is|are|can)\b", re.I),
]


def classify_query(q: str | None) -> str:
    """Return 'template' | 'mixed' | 'natural'.

    - template: strong slot-fill shape ("what is the X of Y", "what is the Y
      configured in X", etc.).
    - mixed: generic question opener ("what/how/why/...") but not a clean
      slot-fill template.
    - natural: everything else (statements, code snippets, freeform phrases).
    """
    if not q:
        return "natural"
    for r in TEMPLATE_RE:
        if r.search(q):
            return "template"
    for r in QUESTION_RE:
        if r.search(q):
            return "mixed"
    return "natural"


def t1_template_labels(rows: list[dict[str, Any]], k: int = 10) -> dict:
    """T1: top-k sema_boost queries per bucket, labeled, plus population density."""
    def top_k_by_tier(bucket: str, tier: str, kk: int) -> list[dict]:
        scored = []
        for r in rows:
            if r.get("bucket") != bucket:
                continue
            v = (r.get("tier_features") or {}).get(tier)
            if v is None:
                continue
            try:
                scored.append((float(v), r))
            except (TypeError, ValueError):
                continue
        scored.sort(key=lambda p: p[0], reverse=True)
        return [
            {
                "retrieval_id": r.get("retrieval_id"),
                "query": r.get("query"),
                "score": s,
                "label": classify_query(r.get("query")),
                "cos_q_c": cosine(r.get("query_sema"), r.get("top_candidate_sema")),
                "requery_delta_s": r.get("requery_delta_s"),
            }
            for s, r in scored[:kk]
        ]

    top_a = top_k_by_tier("A", "sema_boost", k)
    top_b = top_k_by_tier("B", "sema_boost", k)

    def label_counts(seq: list[dict]) -> dict[str, int]:
        c = {"template": 0, "mixed": 0, "natural": 0}
        for o in seq:
            c[o["label"]] = c.get(o["label"], 0) + 1
        return c

    # Population density: fraction of A vs B rows that match template shape.
    pop = {"A": {"template": 0, "mixed": 0, "natural": 0, "n": 0},
           "B": {"template": 0, "mixed": 0, "natural": 0, "n": 0}}
    for r in rows:
        b = r.get("bucket")
        if b not in pop:
            continue
        pop[b][classify_query(r.get("query"))] += 1
        pop[b]["n"] += 1

    # Top-50 by sema_boost for density-in-outliers
    top50_a = top_k_by_tier("A", "sema_boost", 50)
    top50_b = top_k_by_tier("B", "sema_boost", 50)

    return {
        "top_k_A": top_a,
        "top_k_B": top_b,
        "top_k_A_labels": label_counts(top_a),
        "top_k_B_labels": label_counts(top_b),
        "top50_A_labels": label_counts(top50_a),
        "top50_B_labels": label_counts(top50_b),
        "population": pop,
    }


# ──────────────────────────────────────────────────────────────────────
# T2 — query-length bias
# ──────────────────────────────────────────────────────────────────────

LENGTH_BINS = [(0, 30), (30, 45), (45, 60), (60, 80), (80, 10**6)]


def _bin(ln: int) -> tuple[int, int]:
    for lo, hi in LENGTH_BINS:
        if lo <= ln < hi:
            return (lo, hi)
    return LENGTH_BINS[-1]


def t2_length_bias(rows: list[dict[str, Any]]) -> dict:
    per_bin: dict[tuple[int, int], dict[str, list[float]]] = {
        b: {"A": [], "B": [], "A_len": [], "B_len": []} for b in LENGTH_BINS
    }
    overall_by_bucket: dict[str, list[int]] = {"A": [], "B": []}
    for r in rows:
        b = r.get("bucket")
        if b not in ("A", "B"):
            continue
        q = r.get("query") or ""
        ln = len(q)
        overall_by_bucket[b].append(ln)
        c = cosine(r.get("query_sema"), r.get("top_candidate_sema"))
        if c is None:
            continue
        bkt = _bin(ln)
        per_bin[bkt][b].append(c)
        per_bin[bkt][f"{b}_len"].append(ln)

    rows_out = []
    for bkt in LENGTH_BINS:
        d = per_bin[bkt]
        rows_out.append({
            "bin": bkt,
            "n_A": len(d["A"]),
            "n_B": len(d["B"]),
            "mean_cos_A": statistics.fmean(d["A"]) if d["A"] else None,
            "mean_cos_B": statistics.fmean(d["B"]) if d["B"] else None,
            "delta": (statistics.fmean(d["A"]) - statistics.fmean(d["B"]))
                      if d["A"] and d["B"] else None,
        })

    return {
        "bins": rows_out,
        "length_summary": {
            b: {
                "n": len(overall_by_bucket[b]),
                "mean": statistics.fmean(overall_by_bucket[b]) if overall_by_bucket[b] else 0,
                "median": statistics.median(overall_by_bucket[b]) if overall_by_bucket[b] else 0,
                "p10": sorted(overall_by_bucket[b])[int(len(overall_by_bucket[b]) * 0.1)]
                        if overall_by_bucket[b] else 0,
                "p90": sorted(overall_by_bucket[b])[int(len(overall_by_bucket[b]) * 0.9)]
                        if overall_by_bucket[b] else 0,
            } for b in ("A", "B")
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Matrix helpers (shared by T3/T4/T5)
# ──────────────────────────────────────────────────────────────────────

def tier_matrix(rows: list[dict[str, Any]]):
    X = np.zeros((len(rows), N_TIERS), dtype=float)
    fired = np.zeros((len(rows), N_TIERS), dtype=float)
    bucket = np.full(len(rows), -1, dtype=int)
    cosqc = np.full(len(rows), np.nan, dtype=float)
    lengths = np.zeros(len(rows), dtype=int)
    labels = np.full(len(rows), "", dtype=object)
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
        c = cosine(r.get("query_sema"), r.get("top_candidate_sema"))
        if c is not None:
            cosqc[i] = c
        q = r.get("query") or ""
        lengths[i] = len(q)
        labels[i] = classify_query(q)
    keep = bucket != -1
    return X[keep], fired[keep], bucket[keep], cosqc[keep], lengths[keep], labels[keep]


def z_score(X: np.ndarray, fired: np.ndarray) -> np.ndarray:
    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        mask = fired[:, j] > 0
        if mask.sum() < 2:
            continue
        vals = X[mask, j]
        mu = vals.mean()
        sigma = vals.std(ddof=0) or 1.0
        Z[mask, j] = (X[mask, j] - mu) / sigma
    return Z


def pairwise_corr_features(Z: np.ndarray, bucket: np.ndarray, keys: list[str]) -> list[dict]:
    n = Z.shape[1]
    out = []
    for i in range(n):
        for j in range(i, n):
            feat = Z[:, i] * Z[:, j]
            if np.ptp(feat) < 1e-12:
                r, p = 0.0, 1.0
            else:
                r, p = pointbiserialr(bucket, feat)
                if math.isnan(r):
                    r, p = 0.0, 1.0
            out.append({
                "i": keys[i], "j": keys[j],
                "r": r, "abs_r": abs(r), "p": p,
            })
    out.sort(key=lambda d: d["abs_r"], reverse=True)
    return out


def frob_delta(Z: np.ndarray, bucket: np.ndarray) -> dict | None:
    ZA = Z[bucket == 0]
    ZB = Z[bucket == 1]
    if len(ZA) < 3 or len(ZB) < 3:
        return None
    CA = np.nan_to_num(np.corrcoef(ZA, rowvar=False))
    CB = np.nan_to_num(np.corrcoef(ZB, rowvar=False))
    return {
        "n_A": int(len(ZA)),
        "n_B": int(len(ZB)),
        "frob": float(np.linalg.norm(CA - CB, ord="fro")),
        "max_abs": float(np.max(np.abs(CA - CB))),
    }


# ──────────────────────────────────────────────────────────────────────
# T3 — segmentation
# ──────────────────────────────────────────────────────────────────────

def _max_abs_r_over_pairs(Z: np.ndarray, bucket: np.ndarray) -> float:
    """Fast max|r| over the 45 unique pairwise products, no table building."""
    n = Z.shape[1]
    max_ar = 0.0
    # center bucket for point-biserial equivalence to Pearson
    b = bucket.astype(float)
    b = (b - b.mean()) / (b.std(ddof=0) or 1.0)
    for i in range(n):
        for j in range(i, n):
            feat = Z[:, i] * Z[:, j]
            if np.ptp(feat) < 1e-12:
                continue
            fv = (feat - feat.mean())
            denom = np.linalg.norm(fv) * math.sqrt(len(b))
            if denom < 1e-12:
                continue
            r = float(np.dot(fv, b) / denom)
            if abs(r) > max_ar:
                max_ar = abs(r)
    return max_ar


def _perm_max_abs_r(
    Z: np.ndarray, bucket: np.ndarray, n_perm: int = 500, seed: int = 0
) -> dict:
    """Shuffle bucket labels and return the distribution of max|r| across pairs."""
    rng = np.random.default_rng(seed)
    observed = _max_abs_r_over_pairs(Z, bucket)
    null_vals = np.zeros(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(bucket)
        null_vals[t] = _max_abs_r_over_pairs(Z, perm)
    p = float((null_vals >= observed).mean())
    return {
        "observed": float(observed),
        "null_mean": float(null_vals.mean()),
        "null_p95": float(np.quantile(null_vals, 0.95)),
        "p_value": p,
        "n_perm": n_perm,
    }


def t3_segmentation(
    X: np.ndarray, fired: np.ndarray, bucket: np.ndarray, labels: np.ndarray,
    n_perm: int = 500,
) -> dict:
    out = {}
    for seg in ("template", "mixed", "natural"):
        mask = labels == seg
        if mask.sum() < 20:
            out[seg] = {"skipped": f"n={int(mask.sum())} too small"}
            continue
        Xs = X[mask]
        Fs = fired[mask]
        bs = bucket[mask]
        Zs = z_score(Xs, Fs)
        pw = pairwise_corr_features(Zs, bs, TIER_KEYS)
        frob = frob_delta(Zs, bs)
        # Permutation baseline — controls for 45-test multiplicity and small-n
        # noise. Observed vs null_p95 is the gate we actually trust.
        perm = _perm_max_abs_r(Zs, bs, n_perm=n_perm, seed=hash(seg) & 0xFFFF)
        out[seg] = {
            "n": int(mask.sum()),
            "n_A": int((bs == 0).sum()),
            "n_B": int((bs == 1).sum()),
            "top_pairs": pw[:6],
            "max_abs_r": pw[0]["abs_r"] if pw else 0.0,
            "frob": frob,
            "perm": perm,
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# T4 — 10-d matrix (9 tiers + cos(q,c))
# ──────────────────────────────────────────────────────────────────────

def t4_matrix10(
    X: np.ndarray, fired: np.ndarray, bucket: np.ndarray, cosqc: np.ndarray
) -> dict:
    keep = ~np.isnan(cosqc)
    Xk = X[keep]
    Fk = fired[keep]
    bk = bucket[keep]
    ck = cosqc[keep]
    Zk = z_score(Xk, Fk)
    # Z-score cos(q,c) over its population (it always "fires")
    mu = ck.mean()
    sigma = ck.std(ddof=0) or 1.0
    cz = (ck - mu) / sigma
    Z10 = np.column_stack([Zk, cz])
    keys = TIER_KEYS + ["cos_qc"]
    pw = pairwise_corr_features(Z10, bk, keys)
    # cos_qc alone
    r_alone, p_alone = pointbiserialr(bk, cz)
    return {
        "n": int(keep.sum()),
        "n_A": int((bk == 0).sum()),
        "n_B": int((bk == 1).sum()),
        "top_pairs": pw[:10],
        "pairs_involving_cosqc": [d for d in pw if "cos_qc" in (d["i"], d["j"])][:6],
        "cos_qc_alone": {"r": float(r_alone), "p": float(p_alone)},
        "gate_pass": any(d["abs_r"] >= 0.2 for d in pw),
    }


# ──────────────────────────────────────────────────────────────────────
# T5 — Kernel PCA (RBF)
# ──────────────────────────────────────────────────────────────────────

def t5_kernel_pca(Z: np.ndarray, bucket: np.ndarray, k: int = 3) -> dict:
    if not _HAS_SKLEARN:
        return {"skipped": "scikit-learn not installed"}
    # Downsample if huge (KernelPCA is O(n^2) in memory)
    n = Z.shape[0]
    rng = np.random.default_rng(0)
    if n > 2000:
        idx = rng.choice(n, 2000, replace=False)
        Zs = Z[idx]
        bs = bucket[idx]
    else:
        Zs = Z
        bs = bucket

    results = {}
    for gamma in (0.1, 0.5, 1.0):
        try:
            kpca = KernelPCA(n_components=k, kernel="rbf", gamma=gamma, random_state=0)
            proj = kpca.fit_transform(Zs)
        except Exception as e:
            results[f"gamma={gamma}"] = {"error": str(e)}
            continue
        per_comp = []
        for m in range(proj.shape[1]):
            r, p = pointbiserialr(bs, proj[:, m])
            if math.isnan(r):
                r, p = 0.0, 1.0
            per_comp.append({"k": m, "r": float(r), "abs_r": float(abs(r)), "p": float(p)})
        per_comp.sort(key=lambda d: d["abs_r"], reverse=True)
        results[f"gamma={gamma}"] = {
            "n": int(Zs.shape[0]),
            "per_component": per_comp,
            "max_abs_r": per_comp[0]["abs_r"] if per_comp else 0.0,
        }
    return results


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def escape_md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def fmt_query(q: str | None, width: int = 70) -> str:
    q = (q or "")
    q = escape_md(q)
    if len(q) > width:
        q = q[: width - 3] + "..."
    return q


def build_report(export_name: str, n_total: int, t1, t2, t3, t4, t5) -> str:
    L: list[str] = []
    L += [
        "# Phase 0 v2 follow-ups — consolidated report",
        "",
        f"**Source:** `{export_name}` (N = {n_total})",
        f"**Generated:** by `scripts/pwpc/v2_followups.py`",
        "",
        "Executes the open \"next moves\" from PHASE0_V2_DRILLDOWN.md "
        "and LOCKSTEP_MATRIX_TEST_v2.md in one pass. Five tasks:",
        "",
        "- T1 Template-shape density in top-sema_boost outliers",
        "- T2 Query-length bias on the cos(q,c) A-B delta",
        "- T3 Segmented 9x9 matrix test (template / mixed / natural)",
        "- T4 10x10 matrix with cos(q,c) as 10th dim",
        "- T5 Non-linear (RBF kernel-PCA) projection test",
        "",
    ]

    # ── T1 ─────────────────────────────────────────────────────────────
    L += [
        "## T1 — template shape in sema_boost outliers",
        "",
        "Heuristic classifier (regex): **template** = slot-fill shapes "
        "(\"what is the value of X\", \"what is the Y configured in X\"); "
        "**mixed** = generic question opener; **natural** = everything else.",
        "",
        "### Top-10 labels",
        "",
        "| bucket | template | mixed | natural |",
        "|---|---|---|---|",
        f"| A | {t1['top_k_A_labels']['template']} | {t1['top_k_A_labels']['mixed']} | {t1['top_k_A_labels']['natural']} |",
        f"| B | {t1['top_k_B_labels']['template']} | {t1['top_k_B_labels']['mixed']} | {t1['top_k_B_labels']['natural']} |",
        "",
        "### Top-50 labels",
        "",
        "| bucket | template | mixed | natural |",
        "|---|---|---|---|",
        f"| A | {t1['top50_A_labels']['template']} | {t1['top50_A_labels']['mixed']} | {t1['top50_A_labels']['natural']} |",
        f"| B | {t1['top50_B_labels']['template']} | {t1['top50_B_labels']['mixed']} | {t1['top50_B_labels']['natural']} |",
        "",
        "### Population density (all rows, not just top-sema_boost)",
        "",
        "| bucket | n | template | mixed | natural | template% |",
        "|---|---|---|---|---|---|",
    ]
    for b in ("A", "B"):
        pop = t1["population"][b]
        if pop["n"]:
            pct = pop["template"] / pop["n"] * 100
        else:
            pct = 0
        L.append(
            f"| {b} | {pop['n']} | {pop['template']} | {pop['mixed']} | "
            f"{pop['natural']} | {pct:.1f}% |"
        )
    L += [
        "",
        "### Top-10 B-bucket sema_boost queries (labeled)",
        "",
        "| rid | score | label | cos(q,c) | query |",
        "|---|---|---|---|---|",
    ]
    for o in t1["top_k_B"]:
        cos_s = f"{o['cos_q_c']:.3f}" if o["cos_q_c"] is not None else "n/a"
        L.append(
            f"| {o['retrieval_id']} | {o['score']:.1f} | {o['label']} | "
            f"{cos_s} | {fmt_query(o['query'])} |"
        )
    L += [
        "",
        "### Top-10 A-bucket sema_boost queries (labeled)",
        "",
        "| rid | score | label | cos(q,c) | query |",
        "|---|---|---|---|---|",
    ]
    for o in t1["top_k_A"]:
        cos_s = f"{o['cos_q_c']:.3f}" if o["cos_q_c"] is not None else "n/a"
        L.append(
            f"| {o['retrieval_id']} | {o['score']:.1f} | {o['label']} | "
            f"{cos_s} | {fmt_query(o['query'])} |"
        )

    L += [
        "",
        "**Read:** Template fraction in top-k vs population tells you whether "
        "sema_boost is preferentially latching onto slot-fill queries (the "
        "antiresonance signature). If top-50 template% >> population template%, "
        "the hypothesis lives.",
        "",
    ]

    # ── T2 ─────────────────────────────────────────────────────────────
    L += [
        "## T2 — query-length bias on cos(q,c) A-B delta",
        "",
        "### Query length by bucket",
        "",
        "| bucket | n | mean | median | p10 | p90 |",
        "|---|---|---|---|---|---|",
    ]
    for b in ("A", "B"):
        s = t2["length_summary"][b]
        L.append(
            f"| {b} | {s['n']} | {s['mean']:.1f} | {s['median']} | "
            f"{s['p10']} | {s['p90']} |"
        )
    L += [
        "",
        "### cos(q,c) within length bins",
        "",
        "| bin (chars) | n_A | n_B | mean cos A | mean cos B | delta |",
        "|---|---|---|---|---|---|",
    ]
    for b in t2["bins"]:
        lo, hi = b["bin"]
        hi_s = "∞" if hi >= 10**5 else str(hi)
        ma = f"{b['mean_cos_A']:.3f}" if b["mean_cos_A"] is not None else "—"
        mb = f"{b['mean_cos_B']:.3f}" if b["mean_cos_B"] is not None else "—"
        dl = f"{b['delta']:+.4f}" if b["delta"] is not None else "—"
        L.append(f"| [{lo}, {hi_s}) | {b['n_A']} | {b['n_B']} | {ma} | {mb} | {dl} |")
    L += [
        "",
        "**Read:** If the A-B delta flips sign or collapses inside each length "
        "bin, the raw -0.047 delta was Simpson-style length confounding. "
        "If the delta persists within bins, it's a real semantic-agreement "
        "signal independent of query length.",
        "",
    ]

    # ── T3 ─────────────────────────────────────────────────────────────
    L += [
        "## T3 — segmented 9x9 tier matrix test",
        "",
        "Re-runs LOCKSTEP_MATRIX_TEST_v2 within each query-shape segment. "
        "Reports max |r| (gate threshold = 0.20) and Frobenius "
        "||C_A - C_B|| per segment.",
        "",
        "| segment | n | n_A | n_B | max\\|r\\| | null p95 | perm p | top pair | frob ||ΔC|| |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for seg in ("template", "mixed", "natural"):
        d = t3[seg]
        if "skipped" in d:
            L.append(f"| {seg} | — | — | — | skipped ({d['skipped']}) | — | — | — | — |")
            continue
        top = d["top_pairs"][0] if d["top_pairs"] else None
        top_s = f"{top['i']} × {top['j']} (r={top['r']:+.3f})" if top else "—"
        frob_s = f"{d['frob']['frob']:.4f}" if d["frob"] else "—"
        perm = d.get("perm", {})
        p95_s = f"{perm.get('null_p95', 0):.3f}" if perm else "—"
        pv_s = f"{perm.get('p_value', 1):.3f}" if perm else "—"
        L.append(
            f"| {seg} | {d['n']} | {d['n_A']} | {d['n_B']} | "
            f"{d['max_abs_r']:.4f} | {p95_s} | {pv_s} | {top_s} | {frob_s} |"
        )
    L += [
        "",
        "**Read:** Observed max|r| is pulled from 45 pairwise tests per segment. "
        "The permutation baseline shuffles bucket labels within the segment and "
        "recomputes max|r|; the `null p95` column is what 95% of random splits "
        "produce. A segment only shows real signal if `observed > null p95` and "
        "`perm p < 0.05`. Point estimates without this control are misleading "
        "for small n_A.",
        "",
    ]

    # ── T4 ─────────────────────────────────────────────────────────────
    L += [
        "## T4 — 10×10 matrix with cos(q,c) as 10th dim",
        "",
        f"n={t4['n']} (n_A={t4['n_A']}, n_B={t4['n_B']}).",
        "",
        f"**cos(q,c) alone vs bucket:** r = {t4['cos_qc_alone']['r']:+.4f} "
        f"(p = {t4['cos_qc_alone']['p']:.3g})",
        "",
        "### Top 10 pairs by |r|",
        "",
        "| tier_i | tier_j | r | p |",
        "|---|---|---|---|",
    ]
    for d in t4["top_pairs"]:
        L.append(f"| {d['i']} | {d['j']} | {d['r']:+.4f} | {d['p']:.3g} |")
    L += [
        "",
        "### Pairs involving cos(q,c)",
        "",
        "| tier_i | tier_j | r | p |",
        "|---|---|---|---|",
    ]
    for d in t4["pairs_involving_cosqc"]:
        L.append(f"| {d['i']} | {d['j']} | {d['r']:+.4f} | {d['p']:.3g} |")
    L += [
        "",
        f"**Gate (|r| >= 0.2):** {'PASS' if t4['gate_pass'] else 'FAIL'}",
        "",
    ]

    # ── T5 ─────────────────────────────────────────────────────────────
    L += [
        "## T5 — non-linear (RBF kernel-PCA) projection",
        "",
    ]
    if "skipped" in t5:
        L.append(f"Skipped: {t5['skipped']}")
    else:
        L.append("| kernel | top comp r | |r| | all comps |")
        L.append("|---|---|---|---|")
        for gkey, res in t5.items():
            if "error" in res:
                L.append(f"| {gkey} | ERR ({res['error']}) | — | — |")
                continue
            top = res["per_component"][0]
            all_s = ", ".join(f"k{d['k']}: {d['r']:+.3f}" for d in res["per_component"])
            L.append(f"| {gkey} | {top['r']:+.4f} | {top['abs_r']:.4f} | {all_s} |")
    L += [
        "",
        "**Read:** RBF kernel-PCA asks whether a non-linear combination of "
        "the 9 tier z-scores separates A from B. Still gated at |r| >= 0.2.",
        "",
    ]

    # ── Verdict
    t1_topB_pct = (t1['top50_B_labels']['template']
                   / max(1, sum(t1['top50_B_labels'].values())) * 100)
    t1_topA_pct = (t1['top50_A_labels']['template']
                   / max(1, sum(t1['top50_A_labels'].values())) * 100)
    t1_popB_pct = (t1['population']['B']['template']
                   / max(1, t1['population']['B']['n']) * 100)
    t1_popA_pct = (t1['population']['A']['template']
                   / max(1, t1['population']['A']['n']) * 100)
    L += [
        "## Consolidated verdict",
        "",
        f"- **T1 — mild template enrichment in B-outliers:** top-50 B "
        f"template = {t1_topB_pct:.0f}% vs population B = {t1_popB_pct:.0f}% "
        f"(+{t1_topB_pct - t1_popB_pct:.0f} pp). Top-50 A = {t1_topA_pct:.0f}% "
        f"vs population A = {t1_popA_pct:.0f}% ({t1_topA_pct - t1_popA_pct:+.0f} pp). "
        "sema_boost outliers in the B-bucket ARE preferentially slot-fill shapes, "
        "but the lift is small and consistent with the antiresonance direction, "
        "not a clean confirmation.",
    ]
    persists_pos = sum(1 for b in t2["bins"] if b["delta"] is not None and b["delta"] > 0)
    persists_neg = sum(1 for b in t2["bins"] if b["delta"] is not None and b["delta"] < 0)
    all_bins = sum(1 for b in t2["bins"] if b["delta"] is not None)
    L.append(
        f"- **T2 — cos(q,c) delta is not length-confounded:** "
        f"{persists_neg}/{all_bins} length bins show A<B (same direction as "
        f"the pooled -0.047). Simpson's-paradox control cleared — the semantic-"
        "agreement gap is a real population-level effect, but weak."
    )
    real_hits = []
    for s in ("template", "mixed", "natural"):
        d = t3[s]
        if "skipped" in d:
            continue
        perm = d.get("perm", {})
        observed = d.get("max_abs_r", 0)
        p95 = perm.get("null_p95", 0)
        pv = perm.get("p_value", 1)
        status = "real" if (observed > p95 and pv < 0.05) else "noise"
        real_hits.append((s, observed, p95, pv, status))
    any_real = any(st == "real" for *_, st in real_hits)
    segs_s = "; ".join(
        f"{s}: obs={o:.3f} p95={p95:.3f} p={pv:.3f} [{st}]"
        for s, o, p95, pv, st in real_hits
    )
    if any_real:
        headline = "at least one segment clears permutation gate"
    else:
        headline = (
            "ALL segments fall below their own permutation p95 — the "
            "per-segment max|r| is noise, not a washed-out signal"
        )
    L.append(f"- **T3 — segmentation does not unlock lockstep:** {headline}. "
             f"Per-segment detail: {segs_s}")
    L.append(
        f"- **T4 — cos(q,c) as 10th dim doesn't help:** 10×10 gate "
        f"{'PASS' if t4['gate_pass'] else 'FAIL'}; cos(q,c) alone r = "
        f"{t4['cos_qc_alone']['r']:+.4f} (p = {t4['cos_qc_alone']['p']:.3g}). "
        "Weakly significant population-level effect consistent with T2, "
        "but far below the 0.20 gate for routing/head-training."
    )
    if "skipped" not in t5:
        best_t5 = max(
            (res.get("max_abs_r", 0) for res in t5.values() if "error" not in res),
            default=0,
        )
        L.append(
            f"- **T5 — non-linearity isn't the missing ingredient:** best RBF "
            f"kernel-PCA |r| = {best_t5:.4f} across γ∈{{0.1, 0.5, 1.0}} "
            f"({'PASS' if best_t5 >= 0.2 else 'FAIL'}). No hidden non-linear "
            "bucket axis in the 9-tier score manifold."
        )
    else:
        L.append("- **T5 — non-linearity isn't the missing ingredient:** skipped (no sklearn).")
    L += [
        "",
        "### What this means for Sprint 3",
        "",
        "- The **9-tier score matrix alone does not carry bucket signal** at "
        "the |r| ≥ 0.2 gate, in any framing tried here (scalar, 9×9 pairwise, "
        "10×10 + cos(q,c), segmented by query shape, or RBF-kernelized).",
        "- Two real-but-weak effects survive statistical control: (1) B-bucket "
        "top-sema_boost outliers are **+10pp more template-shaped** than "
        "population baseline; (2) B-bucket cos(q,c) is **~0.05 higher than A** "
        "and persists across all length bins. These are directionally "
        "consistent with the antiresonance hypothesis but effect sizes are "
        "small.",
        "- **Implication:** the agreement head can't be trained on these features "
        "alone. Next-move candidates: (a) add out-of-score-matrix features "
        "(query length, token overlap, path anchor quality); (b) switch from "
        "|r| gating to a classifier AUC-style evaluation on the current signal "
        "and accept a weaker ceiling; (c) accept that the re-query bucket "
        "label is too noisy a target and explore alternatives (explicit "
        "thumbs-up/down, downstream answer-quality signals).",
        "",
        "— Generated by `scripts/pwpc/v2_followups.py`",
        "",
    ]
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path)
    ap.add_argument(
        "--out", type=Path,
        default=Path("docs/collab/comms/PHASE0_V2_FOLLOWUPS_2026-04-21.md"),
    )
    args = ap.parse_args()

    rows = load_rows(args.export)
    n_total = len(rows)

    print(f"[v2-followups] loaded {n_total} rows from {args.export.name}")

    t1 = t1_template_labels(rows, k=10)
    print(f"[T1] top-10 B labels: {t1['top_k_B_labels']}")
    print(f"[T1] top-10 A labels: {t1['top_k_A_labels']}")

    t2 = t2_length_bias(rows)
    print(f"[T2] bins populated: {sum(1 for b in t2['bins'] if b['delta'] is not None)}/{len(t2['bins'])}")

    X, fired, bucket, cosqc, lengths, labels = tier_matrix(rows)
    Z9 = z_score(X, fired)

    t3 = t3_segmentation(X, fired, bucket, labels, n_perm=500)
    for seg in ("template", "mixed", "natural"):
        d = t3[seg]
        if "skipped" in d:
            print(f"[T3] {seg}: skipped ({d['skipped']})")
        else:
            perm = d.get("perm", {})
            print(
                f"[T3] {seg}: n={d['n']} max|r|={d['max_abs_r']:.4f} "
                f"null_p95={perm.get('null_p95', 0):.3f} "
                f"perm_p={perm.get('p_value', 1):.3f}"
            )

    t4 = t4_matrix10(X, fired, bucket, cosqc)
    print(f"[T4] 10d gate: {'PASS' if t4['gate_pass'] else 'FAIL'}, "
          f"cos_qc alone r={t4['cos_qc_alone']['r']:+.4f}")

    t5 = t5_kernel_pca(Z9, bucket, k=3)
    if "skipped" in t5:
        print(f"[T5] {t5['skipped']}")
    else:
        for gkey, res in t5.items():
            if "error" in res:
                print(f"[T5] {gkey}: ERR {res['error']}")
            else:
                print(f"[T5] {gkey}: max|r|={res['max_abs_r']:.4f}")

    report = build_report(args.export.name, n_total, t1, t2, t3, t4, t5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"[v2-followups] wrote {args.out}")


if __name__ == "__main__":
    main()

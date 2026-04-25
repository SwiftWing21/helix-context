"""Sprint 3 — tightened-label CWoLa training, Stacked PLR per STATISTICAL_FUSION.md §C3.

Pipeline:

  1. Load windowed export (rows with tier_features + window_features + sema).
  2. Re-derive the next-query cosine intent-delta (cos(q_t, q_{t+1}) within
     the same session, delta ≤ 60s).
  3. Produce four label sets:
        loose  — original bucket from helix_context.cwola.sweep_buckets
        t04    — loose B restricted to cos(q_t, q_{t+1}) > 0.4  (spec default)
        t05    — same with threshold 0.5
        t07    — same with threshold 0.7 (unambiguous reformulation)
  4. For each label set, train GradientBoostingClassifier on the tier features
     + window features under stratified 5-fold CV. Report AUC, feature
     importance, f_gap² (bucket contamination gap, Metodiev §3).
  5. Gate: AUC > 0.55 on held-out (STATISTICAL_FUSION.md §C2 failure-mode gate).

Usage:
    python scripts/pwpc/sprint3.py cwola_export/cwola_export_20260415_windowed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

# Bump this when the feature layout changes so older artifacts are rejected
# at load time instead of silently scoring against the wrong vector shape.
MODEL_SCHEMA_VERSION = 1

TIER_KEYS = [
    "fts5", "splade", "sema_boost", "lex_anchor",
    "tag_exact", "tag_prefix", "pki", "harmonic", "sr",
]


def cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def load_and_derive(path: Path) -> list[dict[str, Any]]:
    """Load export; annotate each B row with cos(q_t, q_{t+1})."""
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    by_session = defaultdict(list)
    for r in rows:
        sid = r.get("session_id")
        if sid:
            by_session[sid].append(r)
    for sid, rs in by_session.items():
        rs.sort(key=lambda x: x.get("ts") or 0)

    n_b_matched = 0
    for r in rows:
        r["next_query_cos"] = None
        if r.get("bucket") != "B":
            continue
        sid = r.get("session_id")
        ts = r.get("ts")
        if not sid or ts is None:
            continue
        rs = by_session.get(sid, [])
        nxt = [x for x in rs if (x.get("ts") or 0) > ts
               and (x.get("ts") or 0) - ts <= 60.0]
        if not nxt:
            continue
        nxt_row = min(nxt, key=lambda x: x.get("ts"))
        c = cosine(r.get("query_sema"), nxt_row.get("query_sema"))
        r["next_query_cos"] = c
        if c is not None:
            n_b_matched += 1
    print(f"[derive] B rows matched to next query cosine: {n_b_matched}")
    return rows


# ──────────────────────────────────────────────────────────────────────
# Feature construction
# ──────────────────────────────────────────────────────────────────────

def build_features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    """Feature vector = 9 tier scores (0 if not fired) + 36 window features +
    cos(q, top_candidate).
    Total: 9 + 36 + 1 = 46 features.
    """
    # Discover window-feature keys from any row that has them
    wf_keys = None
    for r in rows:
        wf = r.get("window_features") or {}
        if wf:
            wf_keys = sorted(wf.keys())
            break
    if wf_keys is None:
        wf_keys = []

    feat_names = list(TIER_KEYS) + wf_keys + ["cos_q_c"]
    X = np.zeros((len(rows), len(feat_names)), dtype=float)

    for i, r in enumerate(rows):
        tf = r.get("tier_features") or {}
        for j, k in enumerate(TIER_KEYS):
            v = tf.get(k)
            if v is not None:
                try:
                    X[i, j] = float(v)
                except (TypeError, ValueError):
                    pass
        wf = r.get("window_features") or {}
        for k, name in enumerate(wf_keys):
            v = wf.get(name)
            if v is not None:
                try:
                    X[i, len(TIER_KEYS) + k] = float(v)
                except (TypeError, ValueError):
                    pass
        c = cosine(r.get("query_sema"), r.get("top_candidate_sema"))
        if c is not None:
            X[i, -1] = c
    return X, feat_names


# ──────────────────────────────────────────────────────────────────────
# Label sets
# ──────────────────────────────────────────────────────────────────────

def make_label_sets(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Produce multiple (y, keep_mask) pairs.

    Encoding: A=0, B=1; keep_mask drops rows excluded from the current label set.
    Returned dict: label_set_name -> {"y": labels, "keep": mask, "desc": str}.
    """
    base_bucket = np.array([r.get("bucket") for r in rows], dtype=object)
    next_cos = np.array([
        r.get("next_query_cos") if r.get("next_query_cos") is not None else np.nan
        for r in rows
    ], dtype=float)

    def assemble(b_mask: np.ndarray, desc: str) -> dict:
        y = np.zeros(len(rows), dtype=int)
        keep = np.zeros(len(rows), dtype=bool)
        # A = kept, label 0
        a_mask = base_bucket == "A"
        keep |= a_mask
        # B = kept only if b_mask, label 1
        full_b_mask = (base_bucket == "B") & b_mask
        keep |= full_b_mask
        y[full_b_mask] = 1
        n_a = int(a_mask.sum())
        n_b = int(full_b_mask.sum())
        return {"y": y, "keep": keep, "n_A": n_a, "n_B": n_b, "desc": desc}

    B_all = base_bucket == "B"
    label_sets: dict[str, dict] = {
        "loose": assemble(B_all, "all B rows (no cosine filter)"),
        "t04":   assemble(B_all & (next_cos > 0.4), "B AND cos(q_t, q_{t+1}) > 0.4 (spec default)"),
        "t05":   assemble(B_all & (next_cos > 0.5), "B AND cos(q_t, q_{t+1}) > 0.5"),
        "t07":   assemble(B_all & (next_cos > 0.7), "B AND cos(q_t, q_{t+1}) > 0.7 (unambiguous reformulation)"),
    }
    return label_sets


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_one(
    X: np.ndarray, y: np.ndarray, keep: np.ndarray, feat_names: list[str],
    n_splits: int = 5, seed: int = 0,
) -> dict:
    Xk = X[keep]
    yk = y[keep]
    n_A = int((yk == 0).sum())
    n_B = int((yk == 1).sum())
    if n_A < 5 or n_B < 5:
        return {"error": f"insufficient samples A={n_A} B={n_B}"}

    # f_gap² diagnostic from Metodiev §3
    frac_A = n_A / (n_A + n_B)
    f_gap_sq = (frac_A - (1 - frac_A)) ** 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, briers = [], []
    fold_importances = []
    for tr, te in skf.split(Xk, yk):
        clf = GradientBoostingClassifier(
            max_depth=3, n_estimators=200, learning_rate=0.05, random_state=seed,
        )
        clf.fit(Xk[tr], yk[tr])
        p = clf.predict_proba(Xk[te])[:, 1]
        aucs.append(roc_auc_score(yk[te], p))
        briers.append(brier_score_loss(yk[te], p))
        fold_importances.append(clf.feature_importances_)

    # Train on full keep for feature ranking
    full_clf = GradientBoostingClassifier(
        max_depth=3, n_estimators=200, learning_rate=0.05, random_state=seed,
    ).fit(Xk, yk)
    importances = full_clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]

    return {
        "n_A": n_A,
        "n_B": n_B,
        "f_gap_sq": f_gap_sq,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_folds": [float(a) for a in aucs],
        "brier_mean": float(np.mean(briers)),
        "top_features": [(feat_names[i], float(importances[i])) for i in top_idx],
    }


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def build_report(
    export_name: str, n_total: int, cos_stats: dict, label_results: dict,
    feat_names: list[str],
) -> str:
    L: list[str] = []
    L += [
        "# Sprint 3 — CWoLa trainer with tightened labels",
        "",
        f"**Source:** `{export_name}` (N = {n_total})",
        f"**Generated:** by `scripts/pwpc/sprint3.py`",
        "",
        "Trains the Stacked PLR classifier specified in STATISTICAL_FUSION.md "
        "§C3 (GradientBoostingClassifier over raw tier features + window "
        "correlations + cos(q, top_candidate)). Compares four label sets to "
        "isolate whether the re-query bucket label is noisy and whether "
        "tightening to same-intent reformulations raises the AUC gate.",
        "",
        "### Gate (STATISTICAL_FUSION.md §C2 degenerate-bucket gate)",
        "",
        "**AUC > 0.55** on held-out split — below this, `f_A ≈ f_B`, the bucket "
        "labels are mostly noise, and the trainer learns the background.",
        "",
        "## Next-query cosine distribution on B-bucket",
        "",
        f"- Matched B rows: {cos_stats['n_matched']} / {cos_stats['n_b']}",
        f"- mean: {cos_stats['mean']:.3f}   median: {cos_stats['median']:.3f}",
        f"- p10: {cos_stats['p10']:.3f}   p90: {cos_stats['p90']:.3f}",
        "",
        "| threshold | retained B | % of loose B |",
        "|---|---|---|",
    ]
    for t, n in cos_stats["thresholds"]:
        pct = n / max(1, cos_stats["n_b"]) * 100
        L.append(f"| > {t:.1f} | {n} | {pct:.1f}% |")
    L += [
        "",
        "## Training results per label set",
        "",
        "| label set | n_A | n_B | f_gap² | AUC mean | AUC std | Brier | gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in label_results.items():
        if "error" in r:
            L.append(f"| {name} | — | — | — | ERR ({r['error']}) | — | — | — |")
            continue
        gate = "PASS" if r["auc_mean"] > 0.55 else "FAIL"
        L.append(
            f"| {name} | {r['n_A']} | {r['n_B']} | {r['f_gap_sq']:.4f} | "
            f"{r['auc_mean']:.3f} | {r['auc_std']:.3f} | {r['brier_mean']:.3f} | "
            f"{gate} |"
        )
    L += [
        "",
        "### Label-set descriptions",
        "",
    ]
    for name, r in label_results.items():
        desc = r.get("desc", "")
        L.append(f"- **{name}** — {desc}")

    L += [
        "",
        "## Top features by importance (full-fit, per label set)",
        "",
    ]
    for name, r in label_results.items():
        if "error" in r:
            continue
        L.append(f"### {name}")
        L.append("")
        L.append("| feature | importance |")
        L.append("|---|---|")
        for fname, imp in r["top_features"]:
            L.append(f"| {fname} | {imp:.4f} |")
        L.append("")

    # Pick best label set
    best_name, best_auc = None, 0.0
    for name, r in label_results.items():
        if "error" in r:
            continue
        if r["auc_mean"] > best_auc:
            best_auc = r["auc_mean"]
            best_name = name

    L += [
        "## Verdict",
        "",
    ]
    if best_name is None:
        L.append("All label sets errored — see table above.")
    else:
        best = label_results[best_name]
        gate = "PASS" if best_auc > 0.55 else "FAIL"
        L.append(f"- **Best label set:** `{best_name}` "
                 f"(AUC mean = {best_auc:.3f} ± {best['auc_std']:.3f}). Gate: **{gate}**.")
        if gate == "PASS":
            L.append(
                "- The Stacked PLR classifier extracts above-chance signal on "
                "this label set. Ship the trained model as `fleet/helix/"
                "fusion_plr.py::StackedPLRFuser` per STATISTICAL_FUSION.md §C3; "
                "wire its `score()` behind a `plr.enabled` feature flag."
            )
        else:
            L.append(
                "- No label set clears the AUC > 0.55 gate. Options: (a) "
                "wait for more A-bucket volume (current A is heavily "
                "outnumbered — see class imbalance); (b) relabel on "
                "explicit user signals (thumbs, edits) rather than re-query "
                "timing; (c) treat this as a known null and focus Sprint 3 "
                "on non-classifier tier-weight tuning (e.g., rank-based "
                "normalisation per tier)."
            )
    L += [
        "",
        "- **Class imbalance note:** n_A = 161 across all label sets (A is "
        "unaffected by the B-tightening filter). All tightening reduces n_B, "
        "which *improves* the balance but reduces absolute sample size. "
        "Stratified CV handles the imbalance; AUC is the right metric to "
        "report here (accuracy is misleading).",
        "",
        "- **Convergence budget (Metodiev §3):** `N ~ 1/(f_A - f_B)²` per "
        "bucket for ε-close-to-optimal AUC. Current f_gap² values are in "
        "the table; the closer to 1, the less data you need. The tight "
        "label sets reduce contamination (raising `f_A - f_B`) but also "
        "reduce n — it's a real tradeoff.",
        "",
        "— Generated by `scripts/pwpc/sprint3.py`",
        "",
    ]
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _fit_full(X: np.ndarray, y: np.ndarray, keep: np.ndarray, seed: int = 0):
    """Fit a single classifier on the full kept set — for artifact export."""
    clf = GradientBoostingClassifier(
        max_depth=3, n_estimators=200, learning_rate=0.05, random_state=seed,
    )
    clf.fit(X[keep], y[keep])
    return clf


def save_model_artifact(
    path: Path, clf: Any, feat_names: list[str], label_set: str,
    auc_mean: float, auc_std: float, n_A: int, n_B: int,
    source_export: str, cos_threshold_used: float | None,
) -> dict:
    """Persist the trained classifier + metadata for fusion_plr.py to load.

    Artifact format (joblib dict; first-party trust assumption — only load
    from operator-controlled paths gated by the [plr] config section):

        {
          "schema_version": int,
          "feat_names": list[str],      # names in fit order (for load-time check)
          "classifier": sklearn estimator,
          "label_set": str,             # 'loose' | 't04' | 't05' | 't07'
          "cos_threshold": float | None,
          "auc_mean": float, "auc_std": float,
          "n_A": int, "n_B": int,
          "source_export": str,         # basename of training corpus
          "trained_at": ISO8601,
        }
    """
    from datetime import datetime, timezone
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feat_names": list(feat_names),
        "classifier": clf,
        "label_set": label_set,
        "cos_threshold": cos_threshold_used,
        "auc_mean": float(auc_mean),
        "auc_std": float(auc_std),
        "n_A": int(n_A),
        "n_B": int(n_B),
        "source_export": source_export,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    # Emit a SHA256 sidecar so operators can verify the artifact on load.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_path = path.with_suffix(path.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return {"path": str(path), "sha256": digest, "sha_path": str(sha_path)}


LABEL_SET_COS_THRESHOLDS = {
    "loose": None, "t04": 0.4, "t05": 0.5, "t07": 0.7,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path)
    ap.add_argument(
        "--out", type=Path,
        default=Path("docs/collab/comms/SPRINT3_TRAINER_2026-04-21.md"),
    )
    ap.add_argument(
        "--save-model", type=Path, default=None,
        help="Write best-label-set classifier to this path as joblib artifact. "
             "A companion .sha256 sidecar is emitted for integrity verification.",
    )
    ap.add_argument(
        "--save-label-set", type=str, default=None,
        choices=["loose", "t04", "t05", "t07", "best"],
        help="Which label set to export (default: 'best' — highest CV AUC).",
    )
    args = ap.parse_args()

    rows = load_and_derive(args.export)
    n_total = len(rows)

    # next-query cosine stats on B
    next_cos = [r.get("next_query_cos") for r in rows
                if r.get("bucket") == "B" and r.get("next_query_cos") is not None]
    next_cos_sorted = sorted(next_cos)
    n_b = sum(1 for r in rows if r.get("bucket") == "B")
    if next_cos_sorted:
        n = len(next_cos_sorted)
        cos_stats = {
            "n_b": n_b,
            "n_matched": n,
            "mean": float(np.mean(next_cos_sorted)),
            "median": float(np.median(next_cos_sorted)),
            "p10": next_cos_sorted[n // 10],
            "p90": next_cos_sorted[(n * 9) // 10],
            "thresholds": [
                (t, sum(1 for c in next_cos_sorted if c > t))
                for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
            ],
        }
    else:
        cos_stats = {
            "n_b": n_b, "n_matched": 0, "mean": 0, "median": 0,
            "p10": 0, "p90": 0, "thresholds": [],
        }
    print(f"[cos] n_matched={cos_stats['n_matched']} mean={cos_stats['mean']:.3f}")

    X, feat_names = build_features(rows)
    print(f"[features] shape={X.shape}  names[:5]={feat_names[:5]}... +{len(feat_names)-5} more")

    label_sets = make_label_sets(rows)
    results: dict[str, dict] = {}
    for name, spec in label_sets.items():
        print(f"[train] {name}: n_A={spec['n_A']} n_B={spec['n_B']} — {spec['desc']}")
        r = train_one(X, spec["y"], spec["keep"], feat_names, n_splits=5, seed=0)
        r["desc"] = spec["desc"]
        results[name] = r
        if "error" in r:
            print(f"  ERR: {r['error']}")
        else:
            print(f"  AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
                  f"Brier={r['brier_mean']:.3f}  f_gap²={r['f_gap_sq']:.4f}")
            top3 = r["top_features"][:3]
            print(f"  top3: " + ", ".join(f"{n}={i:.3f}" for n, i in top3))

    report = build_report(args.export.name, n_total, cos_stats, results, feat_names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"[sprint3] wrote {args.out}")

    if args.save_model:
        target = args.save_label_set or "best"
        if target == "best":
            target = max(
                (n for n, r in results.items() if "error" not in r),
                key=lambda n: results[n]["auc_mean"],
                default=None,
            )
        if target is None or "error" in results.get(target, {"error": "missing"}):
            print(f"[sprint3] cannot save model — label set '{target}' unusable")
            return
        print(f"[sprint3] saving '{target}' classifier (AUC {results[target]['auc_mean']:.3f}) "
              f"to {args.save_model}")
        spec = label_sets[target]
        full_clf = _fit_full(X, spec["y"], spec["keep"], seed=0)
        meta = save_model_artifact(
            args.save_model,
            clf=full_clf,
            feat_names=feat_names,
            label_set=target,
            auc_mean=results[target]["auc_mean"],
            auc_std=results[target]["auc_std"],
            n_A=spec["n_A"],
            n_B=spec["n_B"],
            source_export=args.export.name,
            cos_threshold_used=LABEL_SET_COS_THRESHOLDS.get(target),
        )
        print(f"[sprint3] artifact sha256: {meta['sha256']}")
        print(f"[sprint3] sidecar:         {meta['sha_path']}")


if __name__ == "__main__":
    main()

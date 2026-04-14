"""
E8 vs K-means crossover curve.

For each sample size N in {50, 100, 200, 500, 1000, 1500, 5000}:
  - Random-sample N genes from the genome
  - PCA 20-D → 8-D using that sample's basis
  - Quantize via E8 (240 roots, always available)
  - Quantize via K-means(K) where K scales with N
  - Measure neighbor recovery within 60° cone

Answers: at what N does K-means start beating E8 decisively?
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from itertools import combinations, product

import numpy as np
from sklearn.cluster import KMeans

DB_PATH = "F:/Projects/helix-context/genome.db"
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 1500, 5000]
N_TRIALS = 3           # Average over N random samples per size (smooths noise)
N_PROBE = 50           # Neighbor-recovery probes per trial
K_CODEBOOK = 240       # E8 codebook size; K-means uses min(K, N-1)
RNG_SEED = 42


# ─────────────────────────────────────────────────────────
# Build E8 root system
# ─────────────────────────────────────────────────────────
def build_e8_roots() -> np.ndarray:
    type1 = []
    for i, j in combinations(range(8), 2):
        for si, sj in product([1, -1], repeat=2):
            v = np.zeros(8)
            v[i], v[j] = si, sj
            type1.append(v)
    type2 = [
        np.array(signs)
        for signs in product([0.5, -0.5], repeat=8)
        if sum(1 for s in signs if s < 0) % 2 == 0
    ]
    roots = np.array(type1 + type2, dtype=np.float32)
    return roots / np.sqrt(2)  # unit-normalized


E8_UNIT = build_e8_roots()
assert E8_UNIT.shape == (240, 8)


# ─────────────────────────────────────────────────────────
# Load all embeddings once (reused across trials)
# ─────────────────────────────────────────────────────────
def load_all_embeddings() -> np.ndarray:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA query_only = TRUE")
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM genes WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    parsed = []
    for (emb,) in rows:
        try:
            vec = json.loads(emb)
            if len(vec) == 20:
                parsed.append(vec)
        except Exception:
            pass
    return np.asarray(parsed, dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Neighbor recovery measurement
# ─────────────────────────────────────────────────────────
def neighbor_recovery(
    orig_unit: np.ndarray,       # (N, 20) L2-normalized
    assignments: np.ndarray,      # (N,) codeword index per gene
    centers_unit: np.ndarray,     # (K, 8) L2-normalized codebook
    probe_idx: np.ndarray,
) -> tuple[float, float]:
    """Return (same_codeword_rate, within_60deg_cone_rate)."""
    same = 0
    cone = 0
    for pi in probe_idx:
        sims = orig_unit @ orig_unit[pi]
        sims[pi] = -2
        gt_top10 = np.argsort(-sims)[:10]
        pr_code = assignments[pi]
        for nbr in gt_top10:
            nbr_code = assignments[nbr]
            if nbr_code == pr_code:
                same += 1
                cone += 1
            elif centers_unit[pr_code] @ centers_unit[nbr_code] >= 0.5:
                cone += 1
    total = len(probe_idx) * 10
    return same / total, cone / total


# ─────────────────────────────────────────────────────────
# Run one trial at a given N
# ─────────────────────────────────────────────────────────
def run_trial(all_embs: np.ndarray, N: int, rng: np.random.Generator) -> dict:
    if N > all_embs.shape[0]:
        return {"N": N, "skipped": True, "reason": f"genome has {all_embs.shape[0]} < {N}"}

    # Sample
    idx = rng.choice(all_embs.shape[0], size=N, replace=False)
    mat = all_embs[idx]

    # Ground truth in 20-D
    orig_unit = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    # PCA 20 -> 8 (sample-specific basis)
    mean = mat.mean(axis=0)
    centered = mat - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    top8 = Vt[:8]
    proj = centered @ top8.T
    proj_n = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-9)

    # Probe indices
    n_probe = min(N_PROBE, N)
    probe_idx = rng.choice(N, size=n_probe, replace=False)

    # E8 quantize
    sims_e8 = proj_n @ E8_UNIT.T
    nearest_e8 = np.argmax(sims_e8, axis=1)
    e8_same, e8_cone = neighbor_recovery(orig_unit, nearest_e8, E8_UNIT, probe_idx)
    e8_util = len(set(nearest_e8.tolist()))
    top_e8 = Counter(nearest_e8.tolist()).most_common(1)[0][1]
    e8_top_pct = top_e8 / N

    # K-means
    K = min(K_CODEBOOK, max(2, N - 1))
    km_trainable = N >= K + 1
    if km_trainable:
        try:
            km = KMeans(n_clusters=K, random_state=int(rng.integers(0, 1_000_000)),
                       n_init=3, max_iter=100)
            km.fit(proj_n)
            km_centers = km.cluster_centers_
            km_centers /= (np.linalg.norm(km_centers, axis=1, keepdims=True) + 1e-9)
            nearest_km = km.labels_
            km_same, km_cone = neighbor_recovery(orig_unit, nearest_km, km_centers, probe_idx)
            km_util = len(set(nearest_km.tolist()))
            top_km = Counter(nearest_km.tolist()).most_common(1)[0][1]
            km_top_pct = top_km / N
        except Exception as exc:
            return {"N": N, "skipped": True, "reason": f"kmeans failed: {exc}"}
    else:
        km_same = km_cone = km_util = km_top_pct = None

    return {
        "N": N,
        "K": K,
        "km_trainable": km_trainable,
        "e8_cone": e8_cone,
        "e8_same": e8_same,
        "e8_util": e8_util,
        "e8_top_pct": e8_top_pct,
        "km_cone": km_cone,
        "km_same": km_same,
        "km_util": km_util,
        "km_top_pct": km_top_pct,
    }


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("Loading genome embeddings...")
    all_embs = load_all_embeddings()
    print(f"Loaded {all_embs.shape[0]} embeddings ({all_embs.shape[1]}-dim)")
    print()

    rng = np.random.default_rng(RNG_SEED)

    # Aggregate results
    print(f"{'N':>6}  {'K':>4}  {'trials':>6}  "
          f"{'E8 cone':>8}  {'KM cone':>8}  {'gap':>7}  "
          f"{'E8 util':>10}  {'KM util':>10}  {'E8 top%':>8}  {'KM top%':>8}  {'verdict':>15}")
    print("-" * 130)

    for N in SAMPLE_SIZES:
        if N > all_embs.shape[0]:
            print(f"{N:>6}  (genome has only {all_embs.shape[0]} embeddings — skipping)")
            continue

        trial_results = []
        for trial in range(N_TRIALS):
            r = run_trial(all_embs, N, rng)
            if not r.get("skipped"):
                trial_results.append(r)

        if not trial_results:
            print(f"{N:>6}  (all trials skipped)")
            continue

        # Average
        e8_cone_avg = np.mean([r["e8_cone"] for r in trial_results])
        e8_util_avg = np.mean([r["e8_util"] for r in trial_results])
        e8_top_avg = np.mean([r["e8_top_pct"] for r in trial_results])

        km_trainable = trial_results[0]["km_trainable"]
        if km_trainable:
            km_cone_avg = np.mean([r["km_cone"] for r in trial_results])
            km_util_avg = np.mean([r["km_util"] for r in trial_results])
            km_top_avg = np.mean([r["km_top_pct"] for r in trial_results])
            gap = km_cone_avg - e8_cone_avg

            if gap < 0.05:
                verdict = "E8 competitive"
            elif gap < 0.20:
                verdict = "K-means wins"
            else:
                verdict = "K-means CRUSHES"

            K = trial_results[0]["K"]
            print(
                f"{N:>6}  {K:>4}  {len(trial_results):>6}  "
                f"{100*e8_cone_avg:>7.1f}%  {100*km_cone_avg:>7.1f}%  "
                f"{100*gap:>+6.1f}pp  "
                f"{e8_util_avg:>8.0f}/240  {km_util_avg:>6.0f}/{min(K_CODEBOOK, N-1)}  "
                f"{100*e8_top_avg:>7.1f}%  {100*km_top_avg:>7.1f}%  {verdict:>15}"
            )
        else:
            print(
                f"{N:>6}  {'—':>4}  {len(trial_results):>6}  "
                f"{100*e8_cone_avg:>7.1f}%  {'N/A':>8}  "
                f"{'N/A':>7}  "
                f"{e8_util_avg:>8.0f}/240  {'N/A':>10}  "
                f"{100*e8_top_avg:>7.1f}%  {'N/A':>8}  {'E8 ONLY':>15}"
            )

    print()
    print("Notes:")
    print(f"  Each row is averaged over {N_TRIALS} random trials.")
    print(f"  'cone' = fraction of true top-10 neighbors that land within 60° cone of the probe.")
    print(f"  'util' = distinct codewords used out of total codebook size.")
    print(f"  K-means requires N > K, so K adapts as min(240, N-1).")


if __name__ == "__main__":
    main()

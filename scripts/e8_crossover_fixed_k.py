"""
E8 vs K-means — FIXED K=20 across all N.

The adaptive-K test degenerated to memorization at small N. This fixed-K
variant forces K-means to actually cluster rather than memorize. Fair
apples-to-apples against E8's fixed 240-root codebook (E8 naturally
sub-utilizes at low N, which is the honest comparison).

Alternative: fix K=50 as a middle ground — enough capacity for real
clustering, not enough to memorize.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from itertools import combinations, product

import numpy as np
from sklearn.cluster import KMeans

DB_PATH = "F:/Projects/helix-context/genome.db"
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 1500, 5000]
N_TRIALS = 3
N_PROBE = 50
FIXED_K = 50   # Force K-means to generalize, not memorize
RNG_SEED = 42


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
    return roots / np.sqrt(2)


E8_UNIT = build_e8_roots()


def load_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA query_only = TRUE")
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM genes WHERE embedding IS NOT NULL")
    parsed = []
    for (emb,) in cur.fetchall():
        try:
            vec = json.loads(emb)
            if len(vec) == 20:
                parsed.append(vec)
        except Exception:
            pass
    return np.asarray(parsed, dtype=np.float32)


def neighbor_recovery(orig_unit, assignments, centers_unit, probe_idx):
    same = 0
    cone = 0
    for pi in probe_idx:
        sims = orig_unit @ orig_unit[pi]
        sims[pi] = -2
        gt_top10 = np.argsort(-sims)[:10]
        pr = assignments[pi]
        for nbr in gt_top10:
            nb = assignments[nbr]
            if nb == pr:
                same += 1
                cone += 1
            elif centers_unit[pr] @ centers_unit[nb] >= 0.5:
                cone += 1
    total = len(probe_idx) * 10
    return same / total, cone / total


def run_trial(all_embs, N, K, rng):
    if N > all_embs.shape[0]:
        return None

    idx = rng.choice(all_embs.shape[0], size=N, replace=False)
    mat = all_embs[idx]
    orig_unit = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    mean = mat.mean(axis=0)
    centered = mat - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    top8 = Vt[:8]
    proj = centered @ top8.T
    proj_n = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-9)

    n_probe = min(N_PROBE, N)
    probe_idx = rng.choice(N, size=n_probe, replace=False)

    # E8 (240 roots)
    nearest_e8 = np.argmax(proj_n @ E8_UNIT.T, axis=1)
    e8_same, e8_cone = neighbor_recovery(orig_unit, nearest_e8, E8_UNIT, probe_idx)
    e8_util = len(set(nearest_e8.tolist()))

    # K-means with fixed K
    if N < K + 1:
        km_same = km_cone = km_util = None
    else:
        km = KMeans(n_clusters=K, random_state=int(rng.integers(0, 1_000_000)),
                    n_init=3, max_iter=100)
        km.fit(proj_n)
        km_centers = km.cluster_centers_
        km_centers /= (np.linalg.norm(km_centers, axis=1, keepdims=True) + 1e-9)
        km_same, km_cone = neighbor_recovery(orig_unit, km.labels_, km_centers, probe_idx)
        km_util = len(set(km.labels_.tolist()))

    return {
        "e8_cone": e8_cone, "e8_same": e8_same, "e8_util": e8_util,
        "km_cone": km_cone, "km_same": km_same, "km_util": km_util,
    }


def main():
    print("Loading embeddings...")
    all_embs = load_all_embeddings()
    print(f"Loaded {all_embs.shape[0]} embeddings")
    print()
    print(f"Fixed K = {FIXED_K} for K-means (vs 240 for E8) — forcing generalization, not memorization")
    print()

    rng = np.random.default_rng(RNG_SEED)

    print(f"{'N':>6}  {'E8 cone':>8}  {'KM cone':>8}  {'gap':>8}  "
          f"{'E8 util':>10}  {'KM util':>10}  {'winner':>15}")
    print("-" * 95)

    for N in SAMPLE_SIZES:
        trial_results = []
        for _ in range(N_TRIALS):
            r = run_trial(all_embs, N, FIXED_K, rng)
            if r is not None:
                trial_results.append(r)

        if not trial_results:
            print(f"{N:>6}  (insufficient data)")
            continue

        e8_cone = np.mean([r["e8_cone"] for r in trial_results])
        e8_util = np.mean([r["e8_util"] for r in trial_results])

        km_available = trial_results[0]["km_cone"] is not None
        if km_available:
            km_cone = np.mean([r["km_cone"] for r in trial_results])
            km_util = np.mean([r["km_util"] for r in trial_results])
            gap = km_cone - e8_cone
            if abs(gap) < 0.03:
                winner = "~tie"
            elif gap > 0:
                winner = "K-means"
            else:
                winner = "E8"
            print(
                f"{N:>6}  {100*e8_cone:>7.1f}%  {100*km_cone:>7.1f}%  "
                f"{100*gap:>+7.1f}pp  "
                f"{e8_util:>8.0f}/240  {km_util:>6.0f}/{FIXED_K}  {winner:>15}"
            )
        else:
            print(
                f"{N:>6}  {100*e8_cone:>7.1f}%  {'N/A':>8}  {'N/A':>8}  "
                f"{e8_util:>8.0f}/240  {'N/A':>10}  {'E8 ONLY':>15}"
            )


if __name__ == "__main__":
    main()

"""
E8 VQ re-test — April 12 snapshot.

Compares E8 lattice VQ vs learned K-means(240) codebook on the current
live genome. Stratified sample (500 OPEN, 500 EUCHRO, 500 HETERO) with
per-tier breakdown and April 11 baseline comparison.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from itertools import combinations, product

import numpy as np

DB_PATH = "F:/Projects/helix-context/genome.db"

# ============================================================
# Snapshot
# ============================================================
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA query_only = TRUE")
cur = conn.cursor()

t0 = time.perf_counter()
samples: dict[str, list] = {}
for tier_id, tier_name in [(0, "OPEN"), (1, "EUCHRO"), (2, "HETERO")]:
    cur.execute(
        """SELECT gene_id, embedding FROM genes
           WHERE chromatin = ? AND embedding IS NOT NULL
           ORDER BY gene_id LIMIT 500""",
        (tier_id,),
    )
    samples[tier_name] = cur.fetchall()
    print(f"  {tier_name}: {len(samples[tier_name])} genes snapshotted")

load_sec = time.perf_counter() - t0
print(f"Snapshot load time: {load_sec:.2f}s")
print()

# Parse
all_embs = []
tier_labels = []
for tier_name in ["OPEN", "EUCHRO", "HETERO"]:
    for gid, emb in samples[tier_name]:
        try:
            vec = json.loads(emb)
            if len(vec) == 20:
                all_embs.append(vec)
                tier_labels.append(tier_name)
        except Exception:
            pass

mat = np.asarray(all_embs, dtype=np.float32)
tier_labels = np.array(tier_labels)
print(f"Parsed {mat.shape[0]} embeddings ({mat.shape[1]}-dim)")
for t in ["OPEN", "EUCHRO", "HETERO"]:
    print(f"  {t}: {(tier_labels == t).sum()}")
print()

# ============================================================
# PCA 20 -> 8
# ============================================================
mean = mat.mean(axis=0)
centered = mat - mean
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
explained = (S ** 2 / (S ** 2).sum())
print(f"PCA variance explained (top-8): {explained[:8].sum():.3f}")
print(f"Singular values (top 10): {S[:10].round(2)}")
print()

top8 = Vt[:8]
proj_8d = centered @ top8.T
proj_n = proj_8d / (np.linalg.norm(proj_8d, axis=1, keepdims=True) + 1e-9)

# ============================================================
# E8 roots
# ============================================================
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
e8_roots = np.array(type1 + type2, dtype=np.float32)
e8_unit = e8_roots / np.sqrt(2)
assert e8_unit.shape == (240, 8)

# ============================================================
# Quantize
# ============================================================
sims_e8 = proj_n @ e8_unit.T
nearest_e8 = np.argmax(sims_e8, axis=1)
max_e8 = sims_e8[np.arange(len(proj_n)), nearest_e8]

from sklearn.cluster import KMeans  # noqa: E402
km = KMeans(n_clusters=240, random_state=42, n_init=5, max_iter=100)
km.fit(proj_n)
km_centers = km.cluster_centers_ / (
    np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-9
)
sims_km = proj_n @ km_centers.T
nearest_km = np.argmax(sims_km, axis=1)
max_km = sims_km[np.arange(len(proj_n)), nearest_km]

# ============================================================
# Overall
# ============================================================
usage_e8 = Counter(nearest_e8.tolist())
usage_km = Counter(nearest_km.tolist())
top_e8 = usage_e8.most_common(1)[0][1]
top_km = usage_km.most_common(1)[0][1]

print("=" * 65)
print(f"OVERALL (all tiers, N={len(proj_n)})")
print("=" * 65)
print(f"                           E8       K-means   gap")
print(f"Mean cos(gene, nearest):   {max_e8.mean():.3f}    {max_km.mean():.3f}     {max_km.mean()-max_e8.mean():+.3f}")
print(f"Median:                    {np.median(max_e8):.3f}    {np.median(max_km):.3f}     {np.median(max_km)-np.median(max_e8):+.3f}")
print(f"P10:                       {np.quantile(max_e8, 0.1):.3f}    {np.quantile(max_km, 0.1):.3f}     {np.quantile(max_km, 0.1)-np.quantile(max_e8, 0.1):+.3f}")
print(f"P90:                       {np.quantile(max_e8, 0.9):.3f}    {np.quantile(max_km, 0.9):.3f}     {np.quantile(max_km, 0.9)-np.quantile(max_e8, 0.9):+.3f}")
print(f"Codebook utilization:      {len(usage_e8)}/240  {len(usage_km)}/240")
print(f"Most-used captures:        {100*top_e8/len(proj_n):.1f}%    {100*top_km/len(proj_n):.1f}%")
print()

# ============================================================
# Neighbor recovery
# ============================================================
rng = np.random.default_rng(1)
n_probe = min(150, len(proj_n))
probe_idx = rng.choice(len(proj_n), size=n_probe, replace=False)

orig_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

same_e8 = cone_e8 = same_km = cone_km = 0
for pi in probe_idx:
    sims_gt = orig_norm @ orig_norm[pi]
    sims_gt[pi] = -2
    gt_top10 = np.argsort(-sims_gt)[:10]

    pr_e8 = nearest_e8[pi]
    pr_km = nearest_km[pi]

    for nbr in gt_top10:
        ne8 = nearest_e8[nbr]
        if ne8 == pr_e8:
            same_e8 += 1
            cone_e8 += 1
        elif e8_unit[pr_e8] @ e8_unit[ne8] >= 0.5:
            cone_e8 += 1

        nkm = nearest_km[nbr]
        if nkm == pr_km:
            same_km += 1
            cone_km += 1
        elif km_centers[pr_km] @ km_centers[nkm] >= 0.5:
            cone_km += 1

total = n_probe * 10
print(f"Neighbor recovery (N={n_probe} probes x 10 neighbors = {total}):")
print(f"Same codeword:             {100*same_e8/total:.1f}%    {100*same_km/total:.1f}%     {100*(same_km-same_e8)/total:+.1f}pp")
print(f"Within 60-deg cone:        {100*cone_e8/total:.1f}%    {100*cone_km/total:.1f}%     {100*(cone_km-cone_e8)/total:+.1f}pp")
print()

# ============================================================
# Per-tier
# ============================================================
print("=" * 65)
print("PER-TIER BREAKDOWN")
print("=" * 65)
print(f"{'tier':<8} {'n':>5}  {'E8 mean':>8}  {'KM mean':>8}  {'gap':>6}  {'E8 util':>9}  {'KM util':>9}  {'E8 top%':>8}")
for tier_name in ["OPEN", "EUCHRO", "HETERO"]:
    mask = tier_labels == tier_name
    if mask.sum() == 0:
        continue
    tier_e8_cos = max_e8[mask]
    tier_km_cos = max_km[mask]
    tier_e8_use = len(set(nearest_e8[mask].tolist()))
    tier_km_use = len(set(nearest_km[mask].tolist()))
    tier_e8_top = Counter(nearest_e8[mask].tolist()).most_common(1)[0][1]
    tier_pct = 100 * tier_e8_top / mask.sum()
    print(f"{tier_name:<8} {mask.sum():>5}  {tier_e8_cos.mean():>8.3f}  {tier_km_cos.mean():>8.3f}  {tier_km_cos.mean()-tier_e8_cos.mean():>+6.3f}  {tier_e8_use:>5}/240  {tier_km_use:>5}/240  {tier_pct:>7.1f}%")
print()

# ============================================================
# April 11 comparison
# ============================================================
print("=" * 65)
print("APRIL 11 vs APRIL 12")
print("=" * 65)
print(f"                              Apr 11      Apr 12    delta")
print(f"Sample size                   780         {len(proj_n)}       +{len(proj_n)-780}")
print(f"E8 mean quant cosine          0.856       {max_e8.mean():.3f}     {max_e8.mean() - 0.856:+.3f}")
print(f"K-means mean quant cosine     0.962       {max_km.mean():.3f}     {max_km.mean() - 0.962:+.3f}")
print(f"E8 codebook utilization       154/240     {len(usage_e8)}/240   {len(usage_e8) - 154:+d}")
print(f"E8 most-used share            17.7%       {100*top_e8/len(proj_n):.1f}%")
print(f"E8 neighbors same codeword    36.9%       {100*same_e8/total:.1f}%")
print(f"E8 neighbors within 60-deg    36.9%       {100*cone_e8/total:.1f}%")
print(f"KM neighbors within 60-deg    92.3%       {100*cone_km/total:.1f}%")

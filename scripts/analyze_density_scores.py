"""
Ad-hoc diagnostic: compute the current compute_density_score on random samples
from each source category, see whether it actually separates noise from signal.

Read-only against genome.db. Safe to run while a benchmark is in progress.
"""
import sqlite3
import json
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "F:/Projects/helix-context/genome.db"

# Mirror of Genome.compute_density_score from helix_context/genome.py:1486
def compute_density(content_len, n_domains, n_entities, n_kv, access_count, has_complement_50plus):
    tag_count = n_domains + n_entities
    tag_density = tag_count / max(content_len / 1000.0, 0.001)
    kv_density = n_kv / max(content_len / 1000.0, 0.001)
    return (
        tag_density * 0.4
        + kv_density * 0.3
        + min(access_count / 10.0, 1.0) * 0.2
        + (1.0 if has_complement_50plus else 0.0) * 0.1
    )

STEAM_MARKERS = ("steamlibrary", "steamapps", "beamng.drive", "hades/", "hades\\", "dyson sphere")
HELIX_MARKERS = ("helix-context", "helix_context")
COSMIC_MARKERS = ("cosmictasha", "novabridge")
TALLY_MARKERS = ("bookkeeper",)
SCORERIFT_MARKERS = ("two-brain-audit", "scorerift")
EDUCATION_MARKERS = ("biged", "education", "fleet/", "autoresearch")


def bucket(src):
    s = (src or "").lower()
    if any(m in s for m in STEAM_MARKERS) and "education" not in s and "fleet" not in s:
        return "steam_noise"
    if any(m in s for m in HELIX_MARKERS):
        return "helix"
    if any(m in s for m in COSMIC_MARKERS):
        return "cosmic"
    if any(m in s for m in TALLY_MARKERS):
        return "tally"
    if any(m in s for m in SCORERIFT_MARKERS):
        return "scorerift"
    if any(m in s for m in EDUCATION_MARKERS):
        return "education"
    return "other"


def main():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row

    samples = c.execute(
        """
        SELECT gene_id, source_id, content, key_values, promoter,
               epigenetics, complement, chromatin, compression_tier
        FROM genes
        WHERE source_id IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 500
        """
    ).fetchall()

    by_bucket = defaultdict(list)

    for r in samples:
        try:
            promoter = json.loads(r["promoter"]) if r["promoter"] else {}
            n_domains = len(promoter.get("domains", []))
            n_entities = len(promoter.get("entities", []))
            epi = json.loads(r["epigenetics"]) if r["epigenetics"] else {}
            access = epi.get("access_count", 0)
            kvs = json.loads(r["key_values"]) if r["key_values"] else []
            n_kv = len(kvs)
            content_len = len(r["content"] or "")
            has_comp = bool(r["complement"] and len(r["complement"]) > 50)
            score = compute_density(content_len, n_domains, n_entities, n_kv, access, has_comp)
            by_bucket[bucket(r["source_id"])].append({
                "score": score,
                "content_len": content_len,
                "n_tags": n_domains + n_entities,
                "n_kv": n_kv,
                "chromatin": r["chromatin"],
                "tier": r["compression_tier"],
                "source": r["source_id"] or "",
            })
        except Exception:
            pass

    print(f"sample size: {sum(len(v) for v in by_bucket.values())}")
    print()
    print(f"{'bucket':<15} {'n':>4} {'med_score':>10} {'p10':>7} {'p90':>7} {'med_clen':>9} {'med_tags':>9} {'med_kvs':>8}")
    print("-" * 80)

    def med(L): return sorted(L)[len(L) // 2] if L else 0
    def pct(L, p): return sorted(L)[int(len(L) * p)] if L else 0

    for name in sorted(by_bucket):
        rows = by_bucket[name]
        if len(rows) < 3:
            continue
        scores = [r["score"] for r in rows]
        clens = [r["content_len"] for r in rows]
        tags = [r["n_tags"] for r in rows]
        kvs = [r["n_kv"] for r in rows]
        print(
            f"{name:<15} {len(rows):>4} {med(scores):>10.2f} {pct(scores,0.1):>7.2f} {pct(scores,0.9):>7.2f} "
            f"{med(clens):>9} {med(tags):>9} {med(kvs):>8}"
        )

    print()
    print("=== Lowest-scoring genes (should be the noise pile) ===")
    all_rows = [(r, b) for b, rs in by_bucket.items() for r in rs]
    all_rows.sort(key=lambda x: x[0]["score"])
    for r, b in all_rows[:12]:
        src = (r["source"] or "")[-60:]
        print(f"  score={r['score']:>5.2f}  {b:<12} clen={r['content_len']:>6}  tags={r['n_tags']:>3}  kvs={r['n_kv']:>3}  {src}")

    print()
    print("=== Highest-scoring genes (should be the signal pile) ===")
    all_rows.sort(key=lambda x: -x[0]["score"])
    for r, b in all_rows[:8]:
        src = (r["source"] or "")[-60:]
        print(f"  score={r['score']:>5.2f}  {b:<12} clen={r['content_len']:>6}  tags={r['n_tags']:>3}  kvs={r['n_kv']:>3}  {src}")

    # Key question: what percentage of noise vs signal would be caught at threshold 0.25?
    print()
    print("=== Threshold sweep: what fraction of each bucket falls below threshold? ===")
    print(f"{'bucket':<15} {'t=0.15':>8} {'t=0.25':>8} {'t=0.50':>8} {'t=1.00':>8} {'t=2.00':>8}")
    for name in sorted(by_bucket):
        rows = by_bucket[name]
        if len(rows) < 3:
            continue
        scores = [r["score"] for r in rows]
        parts = []
        for t in [0.15, 0.25, 0.50, 1.00, 2.00]:
            pct_below = sum(1 for s in scores if s < t) / len(scores) * 100
            parts.append(f"{pct_below:>6.0f}%")
        print(f"{name:<15} {' '.join(parts)}")

    c.close()


if __name__ == "__main__":
    main()

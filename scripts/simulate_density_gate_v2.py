"""
Dry-run simulation: apply a proposed new density gate to the existing genome
and report what would happen. Read-only against genome.db.

Proposes three changes to the current gate:
  1. Recalibrated thresholds (< 0.5 heterochromatin, < 1.0 euchromatin)
  2. Source-path deny list (immediate heterochromatin)
  3. Gate at the storage boundary (upsert_gene) instead of orchestration layer

This script does NOT modify the database. It only counts what WOULD be demoted.
"""

import sqlite3
import json
import sys
import re
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "F:/Projects/helix-context/genome.db"

# ─────────────────────────────────────────────────────────────────────────
# Proposed Fix 2: Source-path deny list (fast-reject to HETEROCHROMATIN)
# ─────────────────────────────────────────────────────────────────────────
# Paths that are structurally noise regardless of content. These should
# never participate in hot retrieval.
DENY_PATTERNS = [
    # Steam + game content
    r"[\\/]SteamLibrary[\\/]",
    r"[\\/]steamapps[\\/]common[\\/]",
    r"[\\/]BeamNG\.drive[\\/]",
    r"[\\/]Hades[\\/]Content[\\/]Subtitles[\\/]",
    r"[\\/]Factorio[\\/]data[\\/]base[\\/]",
    r"[\\/]Dyson Sphere",
    # Build artifacts
    r"[\\/]\.next[\\/]",
    r"[\\/]node_modules[\\/]",
    r"[\\/]__pycache__[\\/]",
    r"[\\/]dist[\\/](?!helix)",   # dist/ but not e.g. distance/, or dist/ in helix itself
    r"[\\/]build[\\/](?!helix)",
    r"[\\/]target[\\/]debug[\\/]",
    r"[\\/]target[\\/]release[\\/]",
    # Non-English localization only — English locale often has signal value
    r"[\\/]locale[\\/](?!en)",
    # NOTE: .csv is NOT in the deny list. Business CSVs (customer data,
    # financial records, invoice exports) are legitimate ingest targets.
    # Game-localization CSVs are already caught by the Hades/Factorio path
    # patterns above. Low-density generic CSVs will be caught by the score
    # gate, which is the correct layer for content-based (not path-based)
    # filtering.
    # Lockfiles / manifests / minified
    r"[\\/]package-lock\.json$",
    r"[\\/]yarn\.lock$",
    r"[\\/]Cargo\.lock$",
    r"[\\/]uv\.lock$",
    r"\.min\.(js|css)$",
    r"app-paths-manifest\.json$",
    r"app-build-manifest\.json$",
    # Serialized/binary-ish data
    r"\.(bin|pack|idx|lock|log)$",
]

DENY_RE = re.compile("|".join(DENY_PATTERNS), re.IGNORECASE)


def deny_source(source_id):
    if not source_id:
        return False
    return bool(DENY_RE.search(source_id))


# ─────────────────────────────────────────────────────────────────────────
# Existing compute_density_score (mirror of genome.py:1486)
# ─────────────────────────────────────────────────────────────────────────
def compute_density(content_len, n_domains, n_entities, n_kv, access_count, has_complement_50plus):
    tag_count = n_domains + n_entities
    # Content-length floor to prevent tiny-content explosion
    effective_len = max(content_len, 100)
    tag_density = tag_count / (effective_len / 1000.0)
    kv_density = n_kv / (effective_len / 1000.0)
    return (
        tag_density * 0.4
        + kv_density * 0.3
        + min(access_count / 10.0, 1.0) * 0.2
        + (1.0 if has_complement_50plus else 0.0) * 0.1
    )


# ─────────────────────────────────────────────────────────────────────────
# Proposed Fix 1: Recalibrated thresholds
# ─────────────────────────────────────────────────────────────────────────
OLD_THRESH_HETERO = 0.15
OLD_THRESH_EUCHRO = 0.25
NEW_THRESH_HETERO = 0.50
NEW_THRESH_EUCHRO = 1.00


def main():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row

    total = c.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
    print(f"total genes: {total}")
    print()

    # Stream all genes (not just a sample) — the result of gating is per-gene
    rows = c.execute(
        """
        SELECT gene_id, source_id, content, key_values, promoter,
               epigenetics, complement, chromatin, compression_tier
        FROM genes
        """
    ).fetchall()

    # Stats buckets
    outcomes = defaultdict(int)
    by_reason = defaultdict(int)
    by_source_bucket = defaultdict(lambda: defaultdict(int))

    def source_bucket(src):
        s = (src or "").lower()
        if any(m in s for m in ("steamlibrary", "steamapps", "beamng.drive")): return "steam"
        if "\\.next\\" in s or "/.next/" in s or "node_modules" in s: return "build_artifacts"
        if "helix-context" in s or "helix_context" in s: return "helix"
        if "cosmictasha" in s: return "cosmic"
        if "bookkeeper" in s: return "bookkeeper"
        if "education" in s or "biged" in s or "fleet/" in s: return "education"
        if "two-brain-audit" in s or "scorerift" in s: return "scorerift"
        if "autoresearch" in s: return "autoresearch"
        if not src: return "no_source"
        return "other"

    for r in rows:
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

            bucket = source_bucket(r["source_id"])

            # Fix 2: path deny-list check first
            if deny_source(r["source_id"]):
                outcomes["heterochromatin"] += 1
                by_reason["deny_list"] += 1
                by_source_bucket[bucket]["demoted"] += 1
                continue

            # Fix 1: recalibrated thresholds
            score = compute_density(content_len, n_domains, n_entities, n_kv, access, has_comp)

            if access >= 5:
                # Never demote genes with meaningful access history
                outcomes["kept_open"] += 1
                by_reason["high_access"] += 1
                by_source_bucket[bucket]["open"] += 1
            elif score < NEW_THRESH_HETERO:
                outcomes["heterochromatin"] += 1
                by_reason["low_score_hetero"] += 1
                by_source_bucket[bucket]["demoted"] += 1
            elif score < NEW_THRESH_EUCHRO:
                outcomes["euchromatin"] += 1
                by_reason["low_score_euchro"] += 1
                by_source_bucket[bucket]["demoted"] += 1
            else:
                outcomes["kept_open"] += 1
                by_reason["high_score"] += 1
                by_source_bucket[bucket]["open"] += 1
        except Exception:
            outcomes["parse_error"] += 1

    c.close()

    print("=== Proposed gate outcomes ===")
    for k, v in outcomes.items():
        pct = v / total * 100
        print(f"  {k:<18} {v:>6}  ({pct:>5.1f}%)")
    print()
    print("=== Reasons ===")
    for k, v in by_reason.items():
        print(f"  {k:<22} {v:>6}")
    print()
    print("=== By source bucket ===")
    print(f"  {'bucket':<18} {'total':>6} {'open':>6} {'demoted':>8} {'demote_pct':>10}")
    for b in sorted(by_source_bucket):
        cnts = by_source_bucket[b]
        t = cnts["open"] + cnts["demoted"]
        pct = cnts["demoted"] / t * 100 if t else 0
        print(f"  {b:<18} {t:>6} {cnts['open']:>6} {cnts['demoted']:>8} {pct:>9.1f}%")

    print()
    print("=== Retention rate of signal categories ===")
    signal_buckets = ["helix", "cosmic", "bookkeeper", "education", "scorerift", "autoresearch"]
    for b in signal_buckets:
        cnts = by_source_bucket.get(b, {})
        t = cnts.get("open", 0) + cnts.get("demoted", 0)
        if t:
            retained = cnts.get("open", 0) / t * 100
            print(f"  {b:<18} {cnts.get('open',0):>6} / {t:>6}  = {retained:>5.1f}% retained")

    print()
    print("=== Impact summary ===")
    total_demoted = outcomes["heterochromatin"] + outcomes["euchromatin"]
    print(f"  Before: 8063 genes in retrieval pool (9 euchromatin, 0 heterochromatin)")
    print(f"  After:  {outcomes['kept_open']} genes in retrieval pool ({total_demoted} demoted)")
    print(f"  Reduction: {total_demoted / total * 100:.1f}% of genes removed from hot retrieval")
    print(f"  Deny-list caught: {by_reason['deny_list']} genes (structural noise)")
    print(f"  Score-based caught: {by_reason['low_score_hetero'] + by_reason['low_score_euchro']} genes (dilute content)")


if __name__ == "__main__":
    main()

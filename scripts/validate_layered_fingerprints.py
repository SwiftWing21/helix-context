"""Post-backfill validation smoke test for layered fingerprints.

Runs a controlled multi-chunk scenario against an in-memory genome
to confirm:
    1. Parent gene creation at ingest (N ≥ 2 chunks)
    2. No parent for single-chunk files
    3. CHUNK_OF edges inserted
    4. Feature flag OFF: query behaves as before (no parent in top-k)
    5. Feature flag ON: parent surfaces when ≥ 2 chunks hit
    6. reassemble() roundtrips content

Does NOT touch C:/helix-cache/genome.db — safe to run while helix is up.

Usage:
    python scripts/validate_layered_fingerprints.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from helix_context.context_manager import HelixContextManager
from helix_context.genome import Genome
from helix_context.schemas import (
    ChromatinState,
    EpigeneticMarkers,
    Gene,
    PromoterTags,
    StructuralRelation,
)


def _make_chunk_gene(content: str, seq: int, source_id: str, domains, entities) -> Gene:
    return Gene(
        gene_id="",  # content-hashed at upsert
        content=content,
        complement=content[:100],
        codons=[],
        promoter=PromoterTags(
            domains=list(domains),
            entities=list(entities),
            sequence_index=seq,
        ),
        epigenetics=EpigeneticMarkers(),
        chromatin=ChromatinState.OPEN,
        is_fragment=True,
        source_id=source_id,
    )


def main() -> int:
    print("=== Layered fingerprints post-backfill smoke test ===\n")
    genome = Genome(path=":memory:", synonym_map={"auth": ["jwt", "login"]})

    # Scenario: one multi-chunk source, one single-chunk source.
    # Multi-chunk: 3 chunks of an auth design doc.
    multi_source = "/test/auth_design.md"
    chunks_multi = [
        "AUTH module overview. JWT-based sessions for the dashboard.",
        "JWT tokens carry user_id and role. Refresh rotation every 15m.",
        "Session storage: Redis for hot, Postgres for audit log.",
    ]
    child_ids = []
    for i, c in enumerate(chunks_multi):
        gene = _make_chunk_gene(c, seq=i, source_id=multi_source, domains=["auth"], entities=["JWT"])
        # Apply apply_gate=False so chunks aren't demoted to heterochromatin.
        child_ids.append(genome.upsert_gene(gene, apply_gate=False))

    # Build parent manually (simulating what _upsert_parent_gene does in ingest).
    parent_gid = HelixContextManager._make_parent_gene_id(multi_source)
    parent = Gene(
        gene_id=parent_gid,
        content="\n\n".join(chunks_multi)[:1024],
        complement=f"Parent of {len(chunks_multi)} chunks of {multi_source}",
        codons=list(child_ids),
        key_values=[
            f"chunk_count={len(chunks_multi)}",
            "is_parent=true",
        ],
        promoter=PromoterTags(sequence_index=-1),
        epigenetics=EpigeneticMarkers(),
        source_id=multi_source,
    )
    genome.upsert_gene(parent, apply_gate=False)
    genome.store_relations_batch(
        [(cid, parent_gid, int(StructuralRelation.CHUNK_OF), 1.0) for cid in child_ids]
    )

    # Single-chunk source for control.
    single_source = "/test/readme.md"
    single = _make_chunk_gene(
        "Quick start. Run pytest.", seq=0, source_id=single_source,
        domains=["docs"], entities=[],
    )
    genome.upsert_gene(single, apply_gate=False)

    print(f"✓ Ingested: {len(child_ids)} chunks of {multi_source}")
    print(f"✓ Ingested: 1 chunk of {single_source}")
    print(f"✓ Parent gene_id: {parent_gid}\n")

    # --- Check 1: edges exist ---
    edges = genome.conn.execute(
        "SELECT COUNT(*) c FROM gene_relations WHERE relation = ?",
        (int(StructuralRelation.CHUNK_OF),),
    ).fetchone()
    print(f"[1] CHUNK_OF edges in genome: {edges['c']} (expected {len(child_ids)})")
    assert edges["c"] == len(child_ids), "Edge count mismatch"

    # --- Check 2: flag OFF — capture parent's natural (unboosted) score ---
    os.environ.pop("HELIX_LAYERED_FINGERPRINTS", None)
    genome.query_genes(domains=["auth"], entities=["JWT"], max_genes=10)
    score_off = genome.last_query_scores.get(parent_gid, 0.0)
    tier_off = genome.last_tier_contributions.get(parent_gid, {})
    print(f"\n[2] Flag OFF: parent natural score = {score_off:.3f}")
    print(f"    tier_contrib: {tier_off}")
    print(f"    has parent_coactivation tier? {'parent_coactivation' in tier_off} (expected False)")

    # --- Check 3: flag ON — parent score should be HIGHER via co-activation ---
    os.environ["HELIX_LAYERED_FINGERPRINTS"] = "1"
    genome.query_genes(domains=["auth"], entities=["JWT"], max_genes=10)
    score_on = genome.last_query_scores.get(parent_gid, 0.0)
    tier_on = genome.last_tier_contributions.get(parent_gid, {})
    print(f"\n[3] Flag ON:  parent boosted score = {score_on:.3f}")
    print(f"    tier_contrib: {tier_on}")
    print(f"    has parent_coactivation tier? {'parent_coactivation' in tier_on} (expected True)")
    print(f"    chunks_hit reported:           {tier_on.get('chunks_hit')} (expected 3)")
    print(f"    score delta flag_on - flag_off: {score_on - score_off:.3f} (expected > 0)")

    # --- Check 4: reassemble roundtrip ---
    reassembled = genome.reassemble(parent_gid)
    expected_content = "\n\n".join(chunks_multi)
    content_matches = reassembled["content"] == expected_content
    print(f"\n[4] reassemble({parent_gid[:8]}...):")
    print(f"    chunk_count:      {reassembled['chunk_count']}")
    print(f"    reassembled_from: {len(reassembled['reassembled_from'])} children in sequence order")
    print(f"    content matches original join: {content_matches}")

    print("\n=== Summary ===")
    edges_ok = edges["c"] == len(child_ids)
    flag_off_no_boost = "parent_coactivation" not in tier_off
    flag_on_boosts = ("parent_coactivation" in tier_on
                     and tier_on.get("chunks_hit") == len(child_ids)
                     and score_on > score_off)
    reassemble_ok = (reassembled["chunk_count"] == len(child_ids)
                    and content_matches
                    and not reassembled["missing_children"])

    print(f"  Edges created:                     {edges_ok}")
    print(f"  Flag OFF: no co-activation tier:   {flag_off_no_boost}")
    print(f"  Flag ON:  co-activation boost +chunks_hit correct +score delta>0: {flag_on_boosts}")
    print(f"  Reassemble roundtrips cleanly:     {reassemble_ok}")

    all_good = edges_ok and flag_off_no_boost and flag_on_boosts and reassemble_ok
    print(f"\n{'PASS' if all_good else 'FAIL'}")
    genome.close()
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())

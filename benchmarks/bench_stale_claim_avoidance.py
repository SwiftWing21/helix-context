"""Stale-claim avoidance bench — measures DAG walker value on decision quality.

Content-presence benches (multi-needle, composition) showed 0.81 vs 0.81 for
helix_rag vs helix_full_stack because DAG traversal does not change what's
*retrievable*. The question DAG actually answers is: "of the candidates
retrieval surfaced, which one is current?"

This bench seeds a synthetic corpus with three failure modes that pure
retrieval cannot distinguish:

    1. Versioned facts — claim A about entity X exists, then claim A' at
       later observed_at supersedes it. Pure retrieval returns BOTH; only
       DAG can drop A for being superseded.
    2. Unresolved contradictions — two claims assert incompatible values
       for entity X with no supersedes edge. DAG must SURFACE the conflict
       to the agent (via clusters), not silently pick one.
    3. Clean controls — single claim per entity, no conflicts. All modes
       should agree.

Three retrieval modes are compared:

    - ``raw_newest``  — query_claims() newest-first, top-1 wins
    - ``raw_all``     — query_claims(), no resolution; agent "acts on" first
    - ``helix_dag``   — resolve(policy='latest_then_authority')

Metrics:
    - correct_current_rate     — top accepted matches the intended current truth
    - stale_leak_rate          — at least one superseded claim is exposed unflagged
    - contradiction_flag_rate  — unresolved contradictions are surfaced as clusters
    - latency_ms_per_query     — wall time per entity_key lookup
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix_context.claims_graph import (  # noqa: E402
    contradiction_clusters,
    resolve,
)
from helix_context.shard_schema import (  # noqa: E402
    init_main_db,
    open_main_db,
    query_claims,
    register_shard,
    upsert_claim,
    upsert_claim_edge,
)


# ── Seeded corpus ─────────────────────────────────────────────────────


def _seed(db, shard, claim_id, entity_key, text, observed_at,
          supersedes=None, claim_type="path_value"):
    upsert_claim(
        db,
        claim_id=claim_id,
        gene_id=f"g_{claim_id}",
        shard_name=shard,
        claim_type=claim_type,
        claim_text=text,
        entity_key=entity_key,
        observed_at=observed_at,
        supersedes_claim_id=supersedes,
    )


def seed_corpus(db_path: Path) -> dict:
    """Build the synthetic test fixture. Returns the ground-truth map."""
    db = open_main_db(db_path)
    init_main_db(db)
    register_shard(db, "test", "reference", str(db_path.parent / "test.db"))

    truth: dict[str, dict] = {}

    # ── 1a. Versioned, time-monotonic (baseline case) ─────────────────
    # Supersedes chain is also in ascending observed_at — raw time-order
    # retrieval gets this right "by accident," so it's the easy case.
    versioned_mono = [
        ("config.max_workers", ["4", "8", "14"]),
        ("config.model_name", ["qwen3:4b", "qwen3:8b"]),
        ("config.timeout_s", ["10", "20", "30"]),
        ("config.port", ["5000", "5500", "5555"]),
    ]
    for entity, versions in versioned_mono:
        prev_id = None
        for i, val in enumerate(versions):
            cid = f"v_{entity.replace('.', '_')}_{i}"
            _seed(
                db, "test", cid, entity,
                text=f"{entity} = {val}",
                observed_at=100.0 + i * 10,
                supersedes=prev_id,
            )
            prev_id = cid
        truth[entity] = {
            "kind": "versioned",
            "subkind": "monotonic",
            "current_claim": f"v_{entity.replace('.', '_')}_{len(versions) - 1}",
            "current_value": versions[-1],
            "stale_claims": [
                f"v_{entity.replace('.', '_')}_{i}"
                for i in range(len(versions) - 1)
            ],
        }

    # ── 1b. Versioned, NON-monotonic observed_at (the hard case) ──────
    # Supersedes chain inverts observed_at — e.g. a stale fact was
    # ingested recently (observed_at=500) while the current fact was
    # ingested long ago (observed_at=100). This models real ingest
    # reality: backfill, out-of-order scrapes, time-drifted upstreams.
    # Raw newest-first retrieval picks the stale claim; DAG picks the
    # head of the supersedes chain regardless of timestamp.
    versioned_nonmono = [
        # (entity, [(value, observed_at, is_head)])
        ("api.endpoint", [
            ("/v2/tasks", 100.0, True),   # current truth (head, OLD timestamp)
            ("/v1/tasks", 500.0, False),  # stale but INGESTED LATER
        ]),
        ("schema.version", [
            ("4", 120.0, True),
            ("3", 300.0, False),
            ("2", 450.0, False),
            ("1", 600.0, False),
        ]),
        ("retry.backoff", [
            ("exponential", 50.0, True),
            ("linear", 400.0, False),
        ]),
        ("cache.ttl", [
            ("900s", 80.0, True),
            ("300s", 250.0, False),
            ("60s", 550.0, False),
        ]),
    ]
    for entity, versions in versioned_nonmono:
        # Seed in supersedes-chain order (head last), regardless of observed_at.
        # supersedes chain walks: oldest_stale -> ... -> head (current).
        ordered = list(reversed(versions))  # start with oldest stale
        prev_id = None
        head_cid = None
        stale_cids = []
        for i, (val, ts, is_head) in enumerate(ordered):
            cid = f"vn_{entity.replace('.', '_')}_{i}"
            _seed(
                db, "test", cid, entity,
                text=f"{entity} = {val}",
                observed_at=ts,
                supersedes=prev_id,
            )
            prev_id = cid
            if is_head:
                head_cid = cid
            else:
                stale_cids.append(cid)
        truth[entity] = {
            "kind": "versioned",
            "subkind": "nonmonotonic",
            "current_claim": head_cid,
            "current_value": next(v for v, _, h in versions if h),
            "stale_claims": stale_cids,
        }

    # ── 2. Unresolved contradictions ──────────────────────────────────
    # Two claims assert incompatible values; NO supersedes edge.
    # A well-behaved DAG surfaces both as a contradiction cluster.
    contradicted = [
        ("feature.flag_color", ["red", "blue"]),
        ("deploy.env_name", ["staging", "preview"]),
        ("auth.provider", ["okta", "auth0"]),
        ("db.flavor", ["postgres", "mysql"]),
        ("ui.theme", ["dark", "light"]),
    ]
    for entity, values in contradicted:
        cids = []
        for i, val in enumerate(values):
            cid = f"c_{entity.replace('.', '_')}_{i}"
            _seed(
                db, "test", cid, entity,
                text=f"{entity} = {val}",
                observed_at=200.0 + i,
            )
            cids.append(cid)
        # wire contradicts edges (undirected, write once)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                upsert_claim_edge(db, cids[i], cids[j], "contradicts")
        truth[entity] = {
            "kind": "contradicted",
            "claim_ids": cids,
            "expected_cluster_size": len(cids),
        }

    # ── 3. Clean single-claim controls ────────────────────────────────
    clean = [
        ("log.level", "INFO"),
        ("telemetry.enabled", "true"),
        ("region.primary", "us-east-1"),
        ("encoding.default", "utf-8"),
        ("build.tool", "cargo"),
        ("license.spdx", "MIT"),
        ("runtime.py_version", "3.11"),
    ]
    for entity, val in clean:
        cid = f"k_{entity.replace('.', '_')}"
        _seed(db, "test", cid, entity, text=f"{entity} = {val}", observed_at=300.0)
        truth[entity] = {"kind": "clean", "current_claim": cid, "current_value": val}

    db.commit()
    return {"db_path": str(db_path), "truth": truth, "conn": db}


# ── Retrieval modes ───────────────────────────────────────────────────


def mode_raw_newest(db, entity_key):
    """Top-1 by observed_at desc — what a naive RAG does."""
    t0 = time.perf_counter()
    rows = query_claims(db, entity_key=entity_key, limit=1)
    dt = (time.perf_counter() - t0) * 1000.0
    return {"top": rows[0] if rows else None, "all": rows, "clusters": [], "dt_ms": dt}


def mode_raw_all(db, entity_key):
    """Return everything newest-first; agent picks the first."""
    t0 = time.perf_counter()
    rows = query_claims(db, entity_key=entity_key, limit=50)
    dt = (time.perf_counter() - t0) * 1000.0
    return {"top": rows[0] if rows else None, "all": rows, "clusters": [], "dt_ms": dt}


def mode_helix_dag(db, entity_key):
    """DAG resolve — supersedes chains walked, contradictions clustered."""
    t0 = time.perf_counter()
    rows = query_claims(db, entity_key=entity_key, limit=50)
    claim_ids = [r["claim_id"] for r in rows]
    result = resolve(db, claim_ids, policy="latest_then_authority")
    clusters = contradiction_clusters(db, claim_ids) if len(claim_ids) > 1 else []
    dt = (time.perf_counter() - t0) * 1000.0
    # "top" for answer-correctness metric = first accepted (head of chain)
    accepted = result["accepted"]
    return {
        "top": accepted[0] if accepted else None,
        "accepted": accepted,
        "rejected": result["rejected"],
        "clusters": clusters,
        "dt_ms": dt,
    }


MODES = {
    "raw_newest": mode_raw_newest,
    "raw_all": mode_raw_all,
    "helix_dag": mode_helix_dag,
}


# ── Scoring ───────────────────────────────────────────────────────────


def score_versioned(result, truth_entry):
    """correct=top is current; stale_leak=any stale claim surfaced as top."""
    top = result.get("top")
    current_id = truth_entry["current_claim"]
    stale_ids = set(truth_entry["stale_claims"])
    correct = bool(top and top.get("claim_id") == current_id)
    # stale leak = the top we'd "act on" is a stale version
    stale_leak = bool(top and top.get("claim_id") in stale_ids)
    return {"correct": correct, "stale_leak": stale_leak, "contradiction_flagged": None}


def score_contradicted(result, truth_entry):
    """contradiction_flagged=cluster with >=2 of the contradicted claim_ids is present."""
    expected_set = set(truth_entry["claim_ids"])
    clusters = result.get("clusters", []) or []
    flagged = any(
        len([c for c in cluster if c in expected_set]) >= 2
        for cluster in clusters
    )
    # There IS no unique "correct" answer for an unresolved contradiction;
    # we only care that the conflict is visible.
    return {"correct": None, "stale_leak": False, "contradiction_flagged": flagged}


def score_clean(result, truth_entry):
    top = result.get("top")
    correct = bool(top and top.get("claim_id") == truth_entry["current_claim"])
    return {"correct": correct, "stale_leak": False, "contradiction_flagged": None}


SCORERS = {
    "versioned": score_versioned,
    "contradicted": score_contradicted,
    "clean": score_clean,
}


# ── Runner ────────────────────────────────────────────────────────────


def run_bench(db, truth):
    summary: dict[str, dict] = {}
    for mode_name, fn in MODES.items():
        buckets: dict[str, list[dict]] = {
            "versioned_mono": [], "versioned_nonmono": [],
            "contradicted": [], "clean": [],
        }
        latencies: list[float] = []
        for entity_key, entry in truth.items():
            result = fn(db, entity_key)
            latencies.append(result["dt_ms"])
            score = SCORERS[entry["kind"]](result, entry)
            if entry["kind"] == "versioned":
                bucket = "versioned_mono" if entry.get("subkind") == "monotonic" else "versioned_nonmono"
            else:
                bucket = entry["kind"]
            buckets[bucket].append({"entity": entity_key, **score})

        def _frac(items, field):
            vals = [it[field] for it in items if it[field] is not None]
            return (sum(1 for v in vals if v) / len(vals)) if vals else None

        summary[mode_name] = {
            "latency_ms_p50": round(statistics.median(latencies), 3) if latencies else None,
            "latency_ms_mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "versioned_mono": {
                "n": len(buckets["versioned_mono"]),
                "correct_current_rate": _frac(buckets["versioned_mono"], "correct"),
                "stale_leak_rate": _frac(buckets["versioned_mono"], "stale_leak"),
            },
            "versioned_nonmono": {
                "n": len(buckets["versioned_nonmono"]),
                "correct_current_rate": _frac(buckets["versioned_nonmono"], "correct"),
                "stale_leak_rate": _frac(buckets["versioned_nonmono"], "stale_leak"),
            },
            "contradicted": {
                "n": len(buckets["contradicted"]),
                "contradiction_flag_rate": _frac(buckets["contradicted"], "contradiction_flagged"),
            },
            "clean": {
                "n": len(buckets["clean"]),
                "correct_current_rate": _frac(buckets["clean"], "correct"),
            },
        }
    return summary


def main():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="stale_bench_"))
    db_path = tmp / "main.db"
    fixture = seed_corpus(db_path)
    db = fixture["conn"]
    truth = fixture["truth"]

    n_mono = sum(1 for v in truth.values() if v["kind"] == "versioned" and v.get("subkind") == "monotonic")
    n_nonmono = sum(1 for v in truth.values() if v["kind"] == "versioned" and v.get("subkind") == "nonmonotonic")
    n_contradicted = sum(1 for v in truth.values() if v["kind"] == "contradicted")
    n_clean = sum(1 for v in truth.values() if v["kind"] == "clean")

    print(f"Seeded {len(truth)} entity_keys: "
          f"{n_mono} versioned-mono, {n_nonmono} versioned-nonmono, "
          f"{n_contradicted} contradicted, {n_clean} clean")

    summary = run_bench(db, truth)

    print("\n-- Stale-claim avoidance results -------------------------")
    for mode, stats in summary.items():
        print(f"\n[{mode}]  p50={stats['latency_ms_p50']}ms  mean={stats['latency_ms_mean']}ms")
        vm = stats["versioned_mono"]
        print(f"  versioned_mono (n={vm['n']}): correct={vm['correct_current_rate']}  "
              f"stale_leak={vm['stale_leak_rate']}")
        vn = stats["versioned_nonmono"]
        print(f"  versioned_nonmono (n={vn['n']}): correct={vn['correct_current_rate']}  "
              f"stale_leak={vn['stale_leak_rate']}")
        c = stats["contradicted"]
        print(f"  contradicted (n={c['n']}): contradiction_flag_rate="
              f"{c['contradiction_flag_rate']}")
        k = stats["clean"]
        print(f"  clean (n={k['n']}): correct={k['correct_current_rate']}")

    out = {
        "fixture": {
            "n_entities": len(truth),
            "n_versioned_mono": n_mono,
            "n_versioned_nonmono": n_nonmono,
            "n_contradicted": n_contradicted,
            "n_clean": n_clean,
        },
        "modes": summary,
    }
    out_path = REPO_ROOT / "benchmarks" / "results" / "stale_claim_avoidance_2026-04-19.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")

    db.close()


if __name__ == "__main__":
    main()

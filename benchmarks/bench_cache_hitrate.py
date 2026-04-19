"""Cache hit-rate benchmark — simulates a multi-agent workload where
overlapping queries hit overlapping source_ids.

Design:
    - N "agents" (distinct session_ids) run a shared pool of queries.
    - Queries are drawn with replacement so overlap is explicit.
    - Each query → Helix packet → DAL fetch for every source_id.
    - Two runs: cold (fresh cache per call) and warm (shared cache).
    - Reports: per-agent + aggregate hit-rate + latency savings.

Answers the question "does the cache actually help multi-agent flows"
empirically for a realistic query pattern.

Usage:
    python benchmarks/bench_cache_hitrate.py
    python benchmarks/bench_cache_hitrate.py --n-agents 3 --queries-per-agent 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from helix_context.adapters.cache import CachedDAL  # noqa: E402
from helix_context.adapters.dal import DAL  # noqa: E402

HELIX_URL = "http://127.0.0.1:11437"

# Queries overlap — shared interest pool across agents. Some unique
# per agent to simulate working sets that diverge.
SHARED_QUERIES = [
    "where does auth config live",
    "what port does helix listen on",
    "how does the packet freshness model work",
    "claim extraction pipeline",
    "headroom dashboard configuration",
    "cache invalidation semantics",
    "file grain coordinate signal",
    "ingest provenance fields",
]

# Per-agent specialties — each agent has a handful of unique queries
# so the cache isn't 100% overlap (unrealistic).
AGENT_SPECIALTIES = {
    "laude": ["vscode persona layout", "session handoff protocol"],
    "taude": ["tcm velocity drift", "cymatic resonance detection"],
    "raude": ["phased array hardware", "rust biged gui state"],
    "gemini": ["file_token coverage bench", "antigravity ide integration"],
}


def _get_packet(client, query: str) -> dict:
    try:
        r = client.post(
            f"{HELIX_URL}/context/packet",
            json={"query": query, "task_type": "explain"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _fetch_all_sources(packet: dict, dal) -> tuple[int, int, float]:
    """Returns (n_ok, n_err, total_latency_s) for one packet's sources."""
    source_ids = []
    for bucket in ("verified", "stale_risk", "contradictions"):
        for item in packet.get(bucket, []) or []:
            sid = item.get("source_id")
            if sid and sid not in source_ids:
                source_ids.append(sid)
    for tgt in packet.get("refresh_targets", []) or []:
        sid = tgt.get("source_id")
        if sid and sid not in source_ids:
            source_ids.append(sid)

    t0 = time.time()
    n_ok = 0
    n_err = 0
    for sid in source_ids[:12]:
        r = dal.fetch(sid)
        if r.ok:
            n_ok += 1
        else:
            n_err += 1
    return n_ok, n_err, time.time() - t0


def run_workload(
    n_agents: int,
    queries_per_agent: int,
    seed: int = 42,
    use_shared_cache: bool = True,
) -> dict:
    """Simulate agents issuing queries. Returns stats dict.

    ``use_shared_cache=True`` uses one CachedDAL across all agents
    (the recommended multi-agent pattern). False creates a fresh DAL
    per call (no cache at all — baseline).
    """
    random.seed(seed)
    client = httpx.Client(timeout=120)

    shared_cache = CachedDAL(DAL(), max_entries=500) if use_shared_cache else None
    agent_names = list(AGENT_SPECIALTIES.keys())[:n_agents]
    if len(agent_names) < n_agents:
        agent_names += [f"agent_{i}" for i in range(len(agent_names), n_agents)]

    per_agent_stats = {}
    total_t0 = time.time()

    for agent in agent_names:
        # Each agent's query pool: 70% shared + 30% specialty
        specialty = AGENT_SPECIALTIES.get(agent, [f"{agent} specific topic"])
        pool = SHARED_QUERIES + specialty * 3  # boost specialty weight

        t0 = time.time()
        n_fetched = 0
        n_errs = 0
        for _ in range(queries_per_agent):
            q = random.choice(pool)
            packet = _get_packet(client, q)
            if "error" in packet:
                n_errs += 1
                continue
            if use_shared_cache:
                ok, err, _ = _fetch_all_sources(packet, shared_cache)
            else:
                # No cache: fresh DAL each call
                ok, err, _ = _fetch_all_sources(packet, DAL())
            n_fetched += ok
            n_errs += err
        per_agent_stats[agent] = {
            "wall_time_s": round(time.time() - t0, 2),
            "fetches_ok": n_fetched,
            "fetches_err": n_errs,
        }

    total_wall = time.time() - total_t0

    result = {
        "config": {
            "n_agents": n_agents,
            "queries_per_agent": queries_per_agent,
            "use_shared_cache": use_shared_cache,
            "seed": seed,
        },
        "per_agent": per_agent_stats,
        "total_wall_s": round(total_wall, 2),
    }
    if shared_cache is not None:
        result["cache_stats"] = shared_cache.stats()
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--queries-per-agent", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        default=f"benchmarks/results/cache_hitrate_"
                f"{time.strftime('%Y-%m-%d')}.json",
    )
    args = p.parse_args()

    # Probe Helix
    try:
        stats = httpx.get(f"{HELIX_URL}/stats", timeout=5).json()
        print(f"Genome: {stats['total_genes']} genes")
    except Exception as exc:
        print(f"Cannot reach Helix at {HELIX_URL}: {exc}")
        return 1

    print(f"\n=== Cache hit-rate bench "
          f"(n_agents={args.n_agents}, queries/agent={args.queries_per_agent}) ===\n")

    print("Run A — no cache (baseline)")
    no_cache = run_workload(
        args.n_agents, args.queries_per_agent,
        seed=args.seed, use_shared_cache=False,
    )
    print(f"  total wall: {no_cache['total_wall_s']:.2f}s")

    print("\nRun B — shared cache (recommended multi-agent pattern)")
    with_cache = run_workload(
        args.n_agents, args.queries_per_agent,
        seed=args.seed, use_shared_cache=True,
    )
    cstats = with_cache["cache_stats"]
    print(f"  total wall: {with_cache['total_wall_s']:.2f}s")
    print(f"  cache entries={cstats['entries']}  "
          f"hits={cstats['hits']}  misses={cstats['misses']}  "
          f"hit_rate={cstats['hit_rate']:.2%}")

    # Per-agent compare
    print("\nPer-agent wall time (cache on vs off):")
    print(f"  {'agent':<10} {'no_cache_s':<12} {'cached_s':<10} {'speedup':<8}")
    print("  " + "-" * 44)
    for agent in with_cache["per_agent"]:
        a = no_cache["per_agent"][agent]["wall_time_s"]
        b = with_cache["per_agent"][agent]["wall_time_s"]
        speedup = f"{a/b:.2f}x" if b > 0 else "inf"
        print(f"  {agent:<10} {a:<12.2f} {b:<10.2f} {speedup:<8}")

    delta = no_cache["total_wall_s"] - with_cache["total_wall_s"]
    print(f"\nTotal saved: {delta:.2f}s "
          f"({delta / no_cache['total_wall_s'] * 100:.1f}% of baseline)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "no_cache": no_cache,
        "with_cache": with_cache,
        "delta_s": round(delta, 2),
    }, indent=2))
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

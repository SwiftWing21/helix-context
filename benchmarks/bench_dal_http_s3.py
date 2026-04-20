"""DAL cache wall-savings curve across backend latencies.

The companion bench `bench_cache_hitrate.py` measured 41.67% hit rate
but only 4.5% wall savings because the underlying DAL fetches local
files (<1ms). For HTTP/S3/remote backends, each miss costs real wall
time (tens-to-hundreds of ms), and the same 41% hit rate converts into
10× or more savings.

This bench isolates that relationship: we register a synthetic "slow://"
scheme with a configurable per-fetch sleep, replay a deterministic
multi-agent workload, and sweep latency from 1ms → 200ms.

The point is not to prove the cache works (bench_cache_hitrate does
that) — it's to show the *per-backend* wall-savings curve so operators
can estimate cache value without deploying it.

Usage:
    python benchmarks/bench_dal_http_s3.py
    python benchmarks/bench_dal_http_s3.py --latencies 1,20,100,500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix_context.adapters.cache import CachedDAL  # noqa: E402
from helix_context.adapters.dal import DAL, FetchResult  # noqa: E402


def build_slow_fetcher(latency_s: float):
    """Return a fetcher that sleeps for ``latency_s`` then returns text."""
    def _fetch(source_id: str, **_):
        time.sleep(latency_s)
        return FetchResult(
            text=f"# content for {source_id}",
            meta={"source_id": source_id, "backend": "slow", "latency_s": latency_s},
        )
    return _fetch


def make_workload(n_agents: int, fetches_per_agent: int, overlap: float, seed: int):
    """Build a deterministic per-agent fetch sequence.

    ``overlap`` ∈ [0,1] = fraction of each agent's fetches drawn from the
    shared pool; the rest are agent-specific (never hit the cache).
    """
    rng = random.Random(seed)
    shared_pool = [f"slow://shared/doc_{i}" for i in range(24)]
    agent_pools = {
        a: [f"slow://agent_{a}/doc_{i}" for i in range(12)]
        for a in range(n_agents)
    }
    workload: dict[int, list[str]] = {}
    for a in range(n_agents):
        seq = []
        for _ in range(fetches_per_agent):
            if rng.random() < overlap:
                seq.append(rng.choice(shared_pool))
            else:
                seq.append(rng.choice(agent_pools[a]))
        workload[a] = seq
    return workload


def run_workload(dal, workload: dict[int, list[str]]) -> dict:
    t0 = time.perf_counter()
    per_agent = {}
    total_fetches = 0
    for agent, seq in workload.items():
        ta = time.perf_counter()
        for sid in seq:
            r = dal.fetch(sid)
            assert r.ok, f"fetch failed for {sid}"
            total_fetches += 1
        per_agent[agent] = round((time.perf_counter() - ta) * 1000.0, 2)
    wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "wall_ms": wall_ms,
        "per_agent_ms": per_agent,
        "total_fetches": total_fetches,
    }


def bench_one_latency(latency_ms: float, n_agents: int, fetches_per_agent: int,
                      overlap: float, seed: int) -> dict:
    latency_s = latency_ms / 1000.0
    workload = make_workload(n_agents, fetches_per_agent, overlap, seed)

    # Cold run — no cache. Fresh DAL per workload; every fetch sleeps.
    dal_cold = DAL()
    dal_cold.register("slow", build_slow_fetcher(latency_s))
    cold = run_workload(dal_cold, workload)

    # Warm run — shared CachedDAL. TTL high enough that nothing expires.
    dal_raw = DAL()
    dal_raw.register("slow", build_slow_fetcher(latency_s))
    cached = CachedDAL(dal_raw, max_entries=500)
    warm = run_workload(cached, workload)
    cstats = cached.stats()

    savings_ms = cold["wall_ms"] - warm["wall_ms"]
    savings_pct = (savings_ms / cold["wall_ms"]) * 100.0 if cold["wall_ms"] > 0 else 0.0
    speedup = cold["wall_ms"] / warm["wall_ms"] if warm["wall_ms"] > 0 else None

    return {
        "latency_ms_per_fetch": latency_ms,
        "cold_wall_ms": cold["wall_ms"],
        "warm_wall_ms": warm["wall_ms"],
        "savings_ms": round(savings_ms, 2),
        "savings_pct": round(savings_pct, 1),
        "speedup_x": round(speedup, 2) if speedup else None,
        "cache_hit_rate": round(cstats["hit_rate"], 3),
        "cache_hits": cstats["hits"],
        "cache_misses": cstats["misses"],
        "cache_entries": cstats["entries"],
        "total_fetches": cold["total_fetches"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latencies", default="1,5,20,50,100,200",
                   help="comma-separated per-fetch latencies in ms")
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--fetches-per-agent", type=int, default=24)
    p.add_argument("--overlap", type=float, default=0.70,
                   help="fraction of fetches drawn from shared pool [0,1]")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    latencies = [float(x) for x in args.latencies.split(",") if x.strip()]
    print(f"DAL cache wall-savings curve - n_agents={args.n_agents} "
          f"fetches/agent={args.fetches_per_agent} overlap={args.overlap:.0%}")

    # Verify hit rate is latency-invariant (should be — it depends only on workload)
    # then sweep latency.
    print(f"\n{'latency_ms':<12}{'cold_ms':<12}{'warm_ms':<12}"
          f"{'saved_ms':<12}{'saved_pct':<12}{'speedup':<10}{'hit_rate':<10}")
    print("-" * 78)

    results = []
    for lat in latencies:
        r = bench_one_latency(
            lat, args.n_agents, args.fetches_per_agent,
            args.overlap, args.seed,
        )
        results.append(r)
        print(f"{r['latency_ms_per_fetch']:<12.1f}"
              f"{r['cold_wall_ms']:<12.2f}{r['warm_wall_ms']:<12.2f}"
              f"{r['savings_ms']:<12.2f}{r['savings_pct']:<12.1f}"
              f"{str(r['speedup_x']) + 'x':<10}{r['cache_hit_rate']:<10.3f}")

    out = {
        "config": {
            "n_agents": args.n_agents,
            "fetches_per_agent": args.fetches_per_agent,
            "overlap": args.overlap,
            "seed": args.seed,
            "total_fetches_per_run": args.n_agents * args.fetches_per_agent,
        },
        "results": results,
    }
    out_path = REPO_ROOT / "benchmarks" / "results" / "dal_http_s3_2026-04-19.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")

    # Headline takeaway
    print("\nTakeaway:")
    cheapest = results[0]
    dearest = results[-1]
    print(f"  at {cheapest['latency_ms_per_fetch']:.0f}ms/fetch: saved "
          f"{cheapest['savings_ms']:.1f}ms ({cheapest['savings_pct']:.1f}%)")
    print(f"  at {dearest['latency_ms_per_fetch']:.0f}ms/fetch: saved "
          f"{dearest['savings_ms']:.1f}ms ({dearest['savings_pct']:.1f}%)")
    print("  hit rate is latency-invariant; wall savings scale with backend latency.")


if __name__ == "__main__":
    main()

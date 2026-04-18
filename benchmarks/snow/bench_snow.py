"""
SNOW benchmark harness — oracle + LLM cascade runner.

Runs T0 retrieval through the live Helix `/fingerprint` endpoint for each
query, fetches deeper gene fields from SQLite, runs oracle, optionally runs
LLM cascade, aggregates into a SNOW scorecard, writes JSON to results/.

Usage:
    python benchmarks/snow/bench_snow.py --model oracle-only
    python benchmarks/snow/bench_snow.py --model qwen3:4b
    python benchmarks/snow/bench_snow.py --model all
    python benchmarks/snow/bench_snow.py --model qwen3:4b --limit 5
    python benchmarks/snow/bench_snow.py --model qwen3:4b --profile all
    python benchmarks/snow/bench_snow.py --genome path/to/genome.db
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Windows cp1252 can't encode special chars; force UTF-8 for console output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ["HELIX_DISABLE_HEADROOM"] = "1"

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.snow.oracle import oracle_cascade  # noqa: E402
from benchmarks.snow.cascade import llm_cascade  # noqa: E402
from benchmarks.snow.prompts import clean_response  # noqa: E402

GENOME_DB_DEFAULT = "genome-bench-2026-04-14.db"
QUERIES_JSON = REPO / "benchmarks" / "snow" / "snow_queries.json"
RESULTS_DIR = REPO / "benchmarks" / "snow" / "results"
TOP_K = 12
DEFAULT_HELIX_TIMEOUT_S = 120.0

ALL_MODELS = ["gemma4:e2b", "qwen3:4b", "qwen3:8b"]
ALL_PROFILES = ["fast", "balanced", "quality"]


# ---------------------------------------------------------------------------
# Ollama model wrapper
# ---------------------------------------------------------------------------

class OllamaModel:
    """Thin wrapper around Ollama /api/chat with timeout."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        import httpx

        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=timeout)

    def chat(self, messages: List[Dict]) -> Dict:
        resp = self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 500},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        data["message"]["content"] = clean_response(data["message"]["content"])
        return data


# ---------------------------------------------------------------------------
# Helix fingerprint client
# ---------------------------------------------------------------------------

def default_helix_url() -> str:
    """Resolve the default Helix base URL from local config."""
    from helix_context import load_config  # noqa: WPS433

    cfg = load_config()
    host = cfg.server.host or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{cfg.server.port}"


class HelixFingerprintClient:
    """Thin HTTP client for the live `/fingerprint` benchmark surface."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_HELIX_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def fingerprint(
        self,
        query: str,
        max_results: int = TOP_K,
        profile: str = "balanced",
    ) -> Dict[str, Any]:
        resp = self.client.post(
            f"{self.base_url}/fingerprint",
            json={
                "query": query,
                "max_results": max_results,
                "profile": profile,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self.client.close()


def run_t0(client: HelixFingerprintClient, query: str, profile: str, max_results: int):
    """Run T0 retrieval via /fingerprint and normalize the response."""
    response = client.fingerprint(query=query, max_results=max_results, profile=profile)
    fingerprints = response.get("fingerprints", [])
    gene_ids = [fp.get("gene_id") for fp in fingerprints if fp.get("gene_id")]
    scores = {
        gid: float(fp.get("score", 0.0))
        for gid, fp in ((fp.get("gene_id"), fp) for fp in fingerprints)
        if gid
    }
    tier_contribs = {
        gid: dict(fp.get("tier_contributions", {}))
        for gid, fp in ((fp.get("gene_id"), fp) for fp in fingerprints)
        if gid
    }
    return response, gene_ids, scores, tier_contribs


def fetch_gene_fields(conn: sqlite3.Connection, gene_ids: List[str]) -> Dict:
    """Fetch key_values, complement, content, promoter for each gene."""
    if not gene_ids:
        return {}
    placeholders = ",".join("?" * len(gene_ids))
    rows = conn.execute(
        f"SELECT gene_id, key_values, complement, content, promoter "
        f"FROM genes WHERE gene_id IN ({placeholders})",
        tuple(gene_ids),
    ).fetchall()
    result = {}
    for r in rows:
        gid = r[0]
        try:
            promo = json.loads(r[4]) if r[4] else {}
        except Exception:
            promo = {}
        result[gid] = {
            "key_values": r[1] or "",
            "complement": r[2] or "",
            "content": r[3] or "",
            "entities": promo.get("entities", []),
            "domains": promo.get("domains", []),
        }
    return result


def fetch_neighbors(conn: sqlite3.Connection, gene_ids: List[str]) -> Dict:
    """Fetch harmonic_links neighbors for each gene."""
    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    for gid in gene_ids:
        rows = conn.execute(
            "SELECT gene_id_b, weight FROM harmonic_links WHERE gene_id_a = ? "
            "UNION SELECT gene_id_a, weight FROM harmonic_links WHERE gene_id_b = ?",
            (gid, gid),
        ).fetchall()
        if rows:
            neighbors[gid] = [(r[0], float(r[1])) for r in rows]
    return neighbors


def build_fingerprints(
    retrieval_fps: List[Dict[str, Any]],
    gene_ids: List[str],
    scores: Dict[str, float],
    tier_contribs: Dict[str, Dict],
    gene_fields: Dict[str, Dict],
) -> List[Dict]:
    """Build fingerprint dicts for the cascade."""
    fingerprint_map = {
        fp["gene_id"]: fp for fp in retrieval_fps if fp.get("gene_id")
    }
    fps = []
    for gid in gene_ids:
        f = gene_fields.get(gid, {})
        fp = fingerprint_map.get(gid, {})
        fps.append({
            "gene_id": gid,
            "source": fp.get("source") or fp.get("path"),
            "score": scores.get(gid, 0.0),
            "tiers": tier_contribs.get(gid, {}),
            "domains": fp.get("domains") or f.get("domains", []),
            "entities": fp.get("entities") or f.get("entities", []),
        })
    return fps


def build_oracle_fingerprints(
    gene_ids: List[str],
    gene_fields: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Build fingerprint dict keyed by gene_id for the oracle."""
    fps = {}
    for gid in gene_ids:
        f = gene_fields.get(gid, {})
        fps[gid] = {
            "entities": f.get("entities", []),
            "key_values": f.get("key_values", ""),
            "complement": f.get("complement", ""),
            "content": f.get("content", ""),
        }
    return fps


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def run_single_query(
    helix_client: HelixFingerprintClient,
    conn: sqlite3.Connection,
    q: Dict,
    profile: str,
    max_results: int,
    model: Optional[Any] = None,
) -> Dict:
    """Run a single SNOW query: T0 retrieval + oracle + optional LLM cascade."""
    query_text = q["query"]
    expected = q["expected_answer"]
    accept = q["accept"]

    # T0 retrieval
    t0_start = time.perf_counter()
    t0_payload, gene_ids, scores, tier_contribs = run_t0(
        helix_client,
        query=query_text,
        profile=profile,
        max_results=max_results,
    )
    t0_elapsed = time.perf_counter() - t0_start
    retrieved_fingerprints = t0_payload.get("fingerprints", [])

    # Fetch gene fields + neighbors from SQLite
    gene_fields = fetch_gene_fields(conn, gene_ids)

    # Also fetch neighbor gene fields for T4 walk
    neighbors = fetch_neighbors(conn, gene_ids)
    nb_ids = set()
    for nbs in neighbors.values():
        for nb_id, _ in nbs:
            nb_ids.add(nb_id)
    nb_ids -= set(gene_ids)  # don't re-fetch already loaded
    if nb_ids:
        nb_fields = fetch_gene_fields(conn, list(nb_ids))
        gene_fields.update(nb_fields)

    # Build fingerprints
    oracle_fps = build_oracle_fingerprints(gene_ids, gene_fields)
    cascade_fps = build_fingerprints(
        retrieved_fingerprints,
        gene_ids,
        scores,
        tier_contribs,
        gene_fields,
    )

    # Oracle
    oracle_result = oracle_cascade(
        expected_answer=expected,
        accept=accept,
        gene_ids=gene_ids,
        fingerprints=oracle_fps,
        neighbors=neighbors,
    )

    # LLM cascade (optional)
    llm_result = None
    if model is not None:
        llm_result = llm_cascade(
            query=query_text,
            fingerprints=cascade_fps,
            model=model,
            gene_fields=gene_fields,
            neighbors=neighbors,
        )

    return {
        "idx": q["idx"],
        "query": query_text,
        "expected_answer": expected,
        "accept": accept,
        "source": q.get("source", ""),
        "target_tier": q.get("target_tier"),
        "retrieval": {
            "profile": profile,
            "extracted": t0_payload.get("extracted", {}),
            "gene_ids": gene_ids,
            "scores": {gid: scores.get(gid, 0.0) for gid in gene_ids},
            "tier_contribs": {gid: tier_contribs.get(gid, {}) for gid in gene_ids},
            "t0_latency_s": t0_elapsed,
            "t0_endpoint_latency_ms": t0_payload.get("agent", {}).get("latency_ms"),
            "num_genes": len(gene_ids),
        },
        "oracle_result": oracle_result,
        "llm_result": llm_result,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_scorecard(
    query_results: List[Dict],
    model_name: str,
    genome_path: str,
    fingerprint_profile: str,
    helix_url: str,
) -> Dict:
    """Aggregate individual query results into a SNOW scorecard."""
    n = len(query_results)
    if n == 0:
        return {"error": "no queries"}

    # Oracle aggregation
    oracle_hits = [q for q in query_results if q["oracle_result"]["tier"] >= 0]
    oracle_misses = n - len(oracle_hits)
    oracle_miss_rate = oracle_misses / n if n else 0.0

    oracle_tiers = [q["oracle_result"]["tier"] for q in oracle_hits]
    oracle_tokens = [q["oracle_result"]["tokens"] for q in oracle_hits]
    oracle_latencies = [q["oracle_result"]["latency_s"] for q in oracle_hits]

    avg_oracle_tier = mean(oracle_tiers) if oracle_tiers else 0.0
    avg_oracle_tokens = mean(oracle_tokens) if oracle_tokens else 0.0
    avg_oracle_latency = mean(oracle_latencies) if oracle_latencies else 0.0

    # Oracle cascade profile
    oracle_cascade_profile = {f"T{t}": 0 for t in range(5)}
    for t in oracle_tiers:
        key = f"T{t}"
        if key in oracle_cascade_profile:
            oracle_cascade_profile[key] += 1

    # LLM aggregation (if present)
    llm_stats = None
    has_llm = any(q["llm_result"] is not None for q in query_results)

    if has_llm:
        llm_queries = [q for q in query_results if q["llm_result"] is not None]
        successful = [q for q in llm_queries if not q["llm_result"]["miss"]]
        llm_misses = len(llm_queries) - len(successful)
        llm_miss_rate = llm_misses / len(llm_queries) if llm_queries else 0.0

        avg_hops = mean([q["llm_result"]["hops"] for q in successful]) if successful else 0.0
        avg_tokens = mean([q["llm_result"]["tokens"] for q in successful]) if successful else 0.0
        avg_latency = mean([q["llm_result"]["latency_s"] for q in successful]) if successful else 0.0

        # Cascade profile
        cascade_profile = {f"T{t}": 0 for t in range(5)}
        for q in successful:
            tier = q["llm_result"]["tier"]
            key = f"T{tier}"
            if key in cascade_profile:
                cascade_profile[key] += 1

        # Per-step latency: average across queries that reached each tier
        per_step_latency: Dict[str, Dict] = {}
        tier_labels = {0: "T0 fingerprint", 1: "T1 key_values", 2: "T2 complement",
                       3: "T3 content", 4: "T4 walk"}
        for tier_num, label in tier_labels.items():
            step_latencies = []
            for q in llm_queries:
                lr = q["llm_result"]
                if lr is None:
                    continue
                for hop in lr.get("hop_detail", []):
                    if hop["tier"] == tier_num:
                        step_latencies.append(hop["latency_s"])
            if step_latencies:
                per_step_latency[label] = {
                    "avg_s": mean(step_latencies),
                    "count": len(step_latencies),
                }

        # Triage accuracy: % of queries where LLM's gene_id matches any oracle gene
        triage_correct = 0
        triage_total = 0
        for q in llm_queries:
            lr = q["llm_result"]
            orc = q["oracle_result"]
            if lr is None or lr["miss"]:
                continue
            triage_total += 1
            # Oracle gene_id is the gene where the answer was found
            if orc["gene_id"] and lr.get("gene_id"):
                if lr["gene_id"] == orc["gene_id"]:
                    triage_correct += 1
            # Also count as correct if LLM answered correctly at T0
            # (no gene_id but answered from fingerprint)
            elif lr["tier"] == 0 and not lr["miss"]:
                triage_correct += 1

        triage_accuracy = triage_correct / triage_total if triage_total else 0.0

        # Waste/overhead vs oracle
        hop_waste = avg_hops - avg_oracle_tier if oracle_tiers else avg_hops
        token_overhead = avg_tokens / avg_oracle_tokens if avg_oracle_tokens > 0 else 0.0
        latency_overhead = avg_latency / avg_oracle_latency if avg_oracle_latency > 0 else 0.0

        answered_t0_pct = cascade_profile.get("T0", 0) / len(successful) * 100 if successful else 0.0

        llm_stats = {
            "avg_hops": round(avg_hops, 2),
            "avg_tokens": round(avg_tokens, 1),
            "avg_latency_s": round(avg_latency, 2),
            "miss_rate": round(llm_miss_rate, 4),
            "cascade_profile": cascade_profile,
            "cascade_profile_pct": {
                k: round(v / len(successful) * 100, 1) if successful else 0.0
                for k, v in cascade_profile.items()
            },
            "answered_t0_pct": round(answered_t0_pct, 1),
            "triage_accuracy": round(triage_accuracy, 4),
            "hop_waste": round(hop_waste, 1),
            "token_overhead_x": round(token_overhead, 1),
            "latency_overhead_x": round(latency_overhead, 1),
            "per_step_latency": per_step_latency,
        }

    # Genome info
    genome_name = Path(genome_path).stem
    gene_count = "unknown"
    try:
        tmp_conn = sqlite3.connect(genome_path, timeout=10)
        row = tmp_conn.execute("SELECT COUNT(*) FROM genes").fetchone()
        gene_count = row[0] if row else "unknown"
        tmp_conn.close()
    except Exception:
        pass

    return {
        "meta": {
            "model": model_name,
            "genome": genome_name,
            "gene_count": gene_count,
            "n_queries": n,
            "fingerprint_profile": fingerprint_profile,
            "helix_url": helix_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "oracle": {
            "avg_tier": round(avg_oracle_tier, 2),
            "avg_tokens": round(avg_oracle_tokens, 1),
            "avg_latency_s": round(avg_oracle_latency, 4),
            "miss_rate": round(oracle_miss_rate, 4),
            "cascade_profile": oracle_cascade_profile,
        },
        "llm": llm_stats,
        "queries": query_results,
    }


# ---------------------------------------------------------------------------
# Scorecard printer
# ---------------------------------------------------------------------------

def print_scorecard(scorecard: Dict) -> None:
    """Print the SNOW scorecard to stdout."""
    meta = scorecard["meta"]
    orc = scorecard["oracle"]
    llm = scorecard.get("llm")
    n = meta["n_queries"]

    print()
    profile = meta.get("fingerprint_profile", "balanced")
    header = (
        f"SNOW Scorecard -- {meta['model']} [{profile}] "
        f"on {meta['genome']} genome (N={n})"
    )
    print(header)
    sep = "-" * len(header)
    print(sep)

    # Oracle stats always shown
    print(f"  Oracle tier (avg):   {orc['avg_tier']:.1f}")
    print(f"  Oracle tokens (avg): {orc['avg_tokens']:.0f}")
    print(f"  Oracle latency:      {orc['avg_latency_s']:.4f}s")
    print(f"  Oracle miss rate:    {orc['miss_rate']:.1%}")
    ocp = orc["cascade_profile"]
    ocp_str = "  ".join(f"{k}: {v}" for k, v in ocp.items())
    print(f"  Oracle profile:      {ocp_str}")

    if llm:
        print()
        print(f"  Hops (avg):          {llm['avg_hops']:.1f}    "
              f"oracle floor: {orc['avg_tier']:.1f}    "
              f"waste: {llm['hop_waste']:.1f}")
        print(f"  Tokens (avg):        {llm['avg_tokens']:.0f}    "
              f"oracle floor: {orc['avg_tokens']:.0f}     "
              f"overhead: {llm['token_overhead_x']:.1f}x")
        print(f"  Latency (avg):       {llm['avg_latency_s']:.1f}s   "
              f"oracle floor: {orc['avg_latency_s']:.4f}s    "
              f"overhead: {llm['latency_overhead_x']:.1f}x")

        cp = llm["cascade_profile_pct"]
        cp_str = "  ".join(f"{k}: {v:.0f}%" for k, v in cp.items())
        print(f"  Cascade profile:     {cp_str}")
        print(f"  Answered@T0:         {llm['answered_t0_pct']:.0f}%")
        print(f"  Triage accuracy:     {llm['triage_accuracy']:.0%}")
        print(f"  Miss rate:           {llm['miss_rate']:.1%}")

        psl = llm.get("per_step_latency", {})
        if psl:
            print()
            print("  Per-step latency (avg):")
            for label, info in psl.items():
                print(f"    {label:20s} {info['avg_s']:.2f}s  (n={info['count']})")

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SNOW benchmark harness")
    parser.add_argument(
        "--model", default="oracle-only",
        help="Model name (e.g. qwen3:4b), 'oracle-only', or 'all'",
    )
    parser.add_argument(
        "--genome", default=GENOME_DB_DEFAULT,
        help="Path to genome DB file",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Run only first N queries (0 = all)",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--helix-url", default=None,
        help="Helix base URL serving POST /fingerprint (default: local helix.toml host/port)",
    )
    parser.add_argument(
        "--profile", default="balanced",
        choices=["fast", "balanced", "quality", "all"],
        help="Fingerprint retrieval profile to benchmark",
    )
    parser.add_argument(
        "--max-results", type=int, default=TOP_K,
        help="Fingerprint count requested from /fingerprint",
    )
    args = parser.parse_args()

    # Resolve genome path
    genome_path = args.genome
    if not Path(genome_path).is_absolute():
        genome_path = str(REPO / genome_path)
    if not Path(genome_path).exists():
        print(f"ERROR: genome not found: {genome_path}", file=sys.stderr)
        sys.exit(1)

    helix_url = args.helix_url or default_helix_url()

    # Load queries
    with open(QUERIES_JSON, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if args.limit > 0:
        queries = queries[: args.limit]

    print(f"Loaded {len(queries)} queries from {QUERIES_JSON.name}")
    print(f"Genome: {genome_path}")
    print(f"Helix:  {helix_url}")

    # Determine model(s)
    if args.model == "all":
        models_to_run = ALL_MODELS
    else:
        models_to_run = [args.model]

    if args.profile == "all":
        profiles_to_run = ALL_PROFILES
    else:
        profiles_to_run = [args.profile]

    # Open a raw SQLite connection for field fetching
    conn = sqlite3.connect(genome_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    row = conn.execute("SELECT COUNT(*) FROM genes").fetchone()
    gene_count = row[0] if row else "unknown"
    print(f"Genome loaded: {gene_count} genes")

    helix_client = HelixFingerprintClient(helix_url)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        try:
            warmup = helix_client.fingerprint(
                query="helix benchmark warmup",
                max_results=min(args.max_results, 3),
                profile=profiles_to_run[0],
            )
            print(
                "Helix ready:"
                f" {warmup.get('count', 0)} warmup fingerprints"
                f" via profile={profiles_to_run[0]}"
            )
        except Exception as e:
            print(f"ERROR: cannot reach Helix /fingerprint at {helix_url}: {e}", file=sys.stderr)
            sys.exit(1)

        for profile in profiles_to_run:
            for model_name in models_to_run:
                print(f"\n{'='*60}")
                print(f"Running: {model_name}  profile={profile}")
                print(f"{'='*60}")

                # Create model wrapper (unless oracle-only)
                model = None
                if model_name != "oracle-only":
                    try:
                        model = OllamaModel(
                            model=model_name,
                            base_url=args.base_url,
                        )
                        # Warm up: verify model is reachable
                        print(f"  Warming up {model_name}...")
                        model.chat([{"role": "user", "content": "hi"}])
                        print("  Model ready.")
                    except Exception as e:
                        print(f"  ERROR: cannot reach {model_name}: {e}", file=sys.stderr)
                        print("  Skipping this model.", file=sys.stderr)
                        continue

                # Run queries
                query_results = []
                for i, q in enumerate(queries):
                    t_q = time.perf_counter()
                    try:
                        result = run_single_query(
                            helix_client,
                            conn,
                            q,
                            profile=profile,
                            max_results=args.max_results,
                            model=model,
                        )
                    except Exception as e:
                        print(f"  [{i+1}/{len(queries)}] FAIL q{q['idx']}: {e}", file=sys.stderr)
                        result = {
                            "idx": q["idx"],
                            "query": q["query"],
                            "expected_answer": q["expected_answer"],
                            "accept": q["accept"],
                            "source": q.get("source", ""),
                            "target_tier": q.get("target_tier"),
                            "retrieval": {
                                "profile": profile,
                                "extracted": {},
                                "gene_ids": [],
                                "scores": {},
                                "tier_contribs": {},
                                "t0_latency_s": 0,
                                "t0_endpoint_latency_ms": None,
                                "num_genes": 0,
                            },
                            "oracle_result": {"tier": -1, "gene_id": None, "tokens": 0, "latency_s": 0},
                            "llm_result": {"tier": -1, "hops": 0, "answer": None, "miss": True,
                                           "tokens": 0, "latency_s": 0, "hop_detail": [],
                                           "gene_id": None} if model else None,
                            "error": str(e),
                        }
                    elapsed = time.perf_counter() - t_q
                    query_results.append(result)

                    # Progress
                    orc_tier = result["oracle_result"]["tier"]
                    llm_tier = result["llm_result"]["tier"] if result.get("llm_result") else "-"
                    status = f"oracle=T{orc_tier}"
                    if model:
                        status += f" llm=T{llm_tier}"
                    print(
                        f"  [{i+1}/{len(queries)}] q{q['idx']:02d} {elapsed:.1f}s"
                        f"  {status}  {q['query'][:50]}"
                    )

                # Aggregate + print scorecard
                scorecard = aggregate_scorecard(
                    query_results,
                    model_name,
                    genome_path,
                    fingerprint_profile=profile,
                    helix_url=helix_url,
                )
                print_scorecard(scorecard)

                # Write JSON
                date_str = datetime.now().strftime("%Y-%m-%d")
                safe_model = model_name.replace(":", "_").replace("/", "_")
                out_path = RESULTS_DIR / f"snow_{safe_model}_{profile}_{date_str}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(scorecard, f, indent=2, default=str)
                print(f"  Results written to {out_path}")
    finally:
        conn.close()
        helix_client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

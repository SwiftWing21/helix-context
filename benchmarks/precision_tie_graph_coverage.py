"""
Pass 3b — associative-graph coverage on the tied pairs.

Question: do helix's existing associative tables have enough signal to
distinguish the tied genes? Three checks per tied pair:

    1. Direct edge — is there a harmonic_links row between gene_a and
       gene_b? If yes, they're explicitly co-activated; we already know
       they travel together and can tie-break on edge weight or direction.

    2. Neighborhood asymmetry — count distinct harmonic_links neighbors
       for each gene. If gene_a has 50 neighbors and gene_b has 200,
       gene_b is more "central" in the graph and may be the better
       surfacing choice under a centrality-aware tie-break.

    3. Query-entity affinity — for each query, find the entities that
       appear both in the query text and in entity_graph. Count how
       many map to gene_a vs gene_b. The gene with more query-entity
       matches wins the tie-break.

    4. (Bonus) gene_relations — NLI-typed logical relations. If
       gene_a entails / contradicts gene_b, that's structural info
       the tie-break should probably respect.

If any of (1)-(4) produces a clear asymmetry on most tied pairs, the
"walking" tie-break is implementable with data already in the genome.
If they come back flat, the graph isn't dense enough yet and tie-break
needs different signal (freshness, provenance, party_id, etc.).

Input:  benchmarks/precision_probe_2026-04-15.json
        genome-bench-2026-04-14.db
Output: benchmarks/precision_tie_graph_coverage_2026-04-15.json
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
INPUT_JSON = REPO / "benchmarks" / "precision_probe_2026-04-15.json"
OUTPUT_JSON = REPO / "benchmarks" / "precision_tie_graph_coverage_2026-04-15.json"
GENOME_DB = REPO / "genome-bench-2026-04-14.db"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Crude query → entity-candidate split. The real helix extractor does
# more work (stopword removal, filepath tokens, etc.); we approximate
# with lowercase word tokens of length >= 3.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "the", "and", "what", "is", "of", "in", "on", "for", "to", "with",
    "mentioned", "code", "value", "that", "from", "this", "which", "are",
}


def query_terms(query: str) -> Set[str]:
    tokens = {m.group(0).lower() for m in _WORD_RE.finditer(query)}
    return {t for t in tokens if t not in _STOPWORDS}


def load_tied_pairs() -> List[Dict]:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for q in data["results"]:
        scores = q["run_a"]["scores"]
        for i in range(len(scores) - 1):
            a, b = scores[i], scores[i + 1]
            if a["score"] == b["score"] and a["score"] != 0.0:
                pairs.append({
                    "idx": q["idx"],
                    "query": q["query"],
                    "query_terms": sorted(query_terms(q["query"])),
                    "rank_a": i,
                    "rank_b": i + 1,
                    "score": a["score"],
                    "gene_a": a["gene_id"],
                    "gene_b": b["gene_id"],
                })
    return pairs


def inspect_pair(conn: sqlite3.Connection, pair: Dict) -> Dict:
    ga, gb = pair["gene_a"], pair["gene_b"]

    # Check 1: direct harmonic_links edge (either direction)
    row = conn.execute(
        "SELECT weight, co_count, miss_count, source FROM harmonic_links "
        "WHERE (gene_id_a = ? AND gene_id_b = ?) OR (gene_id_a = ? AND gene_id_b = ?) "
        "LIMIT 1",
        (ga, gb, gb, ga),
    ).fetchone()
    direct_edge = None
    if row:
        direct_edge = {
            "weight": row[0],
            "co_count": row[1],
            "miss_count": row[2],
            "source": row[3],
        }

    # Check 2: neighborhood size + overlap
    nb_a = {r[0] for r in conn.execute(
        "SELECT gene_id_b FROM harmonic_links WHERE gene_id_a = ? "
        "UNION SELECT gene_id_a FROM harmonic_links WHERE gene_id_b = ?",
        (ga, ga),
    )}
    nb_b = {r[0] for r in conn.execute(
        "SELECT gene_id_b FROM harmonic_links WHERE gene_id_a = ? "
        "UNION SELECT gene_id_a FROM harmonic_links WHERE gene_id_b = ?",
        (gb, gb),
    )}
    shared = nb_a & nb_b
    nb_a_only = len(nb_a - nb_b)
    nb_b_only = len(nb_b - nb_a)

    # Check 3: query-entity affinity
    terms = pair["query_terms"]
    entities_a: Set[str] = set()
    entities_b: Set[str] = set()
    if terms:
        placeholders = ",".join("?" * len(terms))
        entities_a = {
            r[0] for r in conn.execute(
                f"SELECT entity FROM entity_graph WHERE gene_id = ? AND entity IN ({placeholders})",
                (ga, *terms),
            )
        }
        entities_b = {
            r[0] for r in conn.execute(
                f"SELECT entity FROM entity_graph WHERE gene_id = ? AND entity IN ({placeholders})",
                (gb, *terms),
            )
        }
    query_match_a = len(entities_a)
    query_match_b = len(entities_b)

    # Check 4: gene_relations between them (NLI-typed logical relation)
    rel = conn.execute(
        "SELECT relation, confidence FROM gene_relations "
        "WHERE (gene_id_a = ? AND gene_id_b = ?) OR (gene_id_a = ? AND gene_id_b = ?) "
        "LIMIT 1",
        (ga, gb, gb, ga),
    ).fetchone()
    relation = {"relation": rel[0], "confidence": rel[1]} if rel else None

    # Tie-break verdict: can any signal distinguish these two?
    signals = []
    if direct_edge:
        signals.append("direct_edge")
    if len(nb_a) != len(nb_b):
        signals.append("neighborhood_size")
    if query_match_a != query_match_b:
        signals.append("query_entity_affinity")
    if relation:
        signals.append("nli_relation")

    # Who wins the tie-break under each signal?
    winner_by_signal: Dict[str, str] = {}
    if "neighborhood_size" in signals:
        winner_by_signal["neighborhood_size"] = "a" if len(nb_a) > len(nb_b) else "b"
    if "query_entity_affinity" in signals:
        winner_by_signal["query_entity_affinity"] = "a" if query_match_a > query_match_b else "b"

    return {
        "gene_a": ga,
        "gene_b": gb,
        "direct_edge": direct_edge,
        "neighbors_a": len(nb_a),
        "neighbors_b": len(nb_b),
        "shared_neighbors": len(shared),
        "nb_a_only": nb_a_only,
        "nb_b_only": nb_b_only,
        "query_match_a": query_match_a,
        "query_match_b": query_match_b,
        "entities_matched_by_a": sorted(entities_a),
        "entities_matched_by_b": sorted(entities_b),
        "nli_relation": relation,
        "tie_break_signals_present": signals,
        "winner_by_signal": winner_by_signal,
    }


def main() -> int:
    pairs = load_tied_pairs()
    print(f"[graph-coverage] loaded {len(pairs)} tied pairs\n")

    conn = sqlite3.connect(str(GENOME_DB))

    results = []
    for p in pairs:
        report = inspect_pair(conn, p)
        results.append({**p, **report})

    # Aggregate statistics
    n = len(results)
    n_direct_edge = sum(1 for r in results if r["direct_edge"])
    n_neighborhood_asym = sum(1 for r in results if r["neighbors_a"] != r["neighbors_b"])
    n_query_affinity_asym = sum(1 for r in results if r["query_match_a"] != r["query_match_b"])
    n_nli = sum(1 for r in results if r["nli_relation"])
    n_any_signal = sum(1 for r in results if r["tie_break_signals_present"])
    n_no_signal = n - n_any_signal

    # For the ties with at least one signal, which signal is most useful?
    signal_counts: Dict[str, int] = {}
    for r in results:
        for s in r["tie_break_signals_present"]:
            signal_counts[s] = signal_counts.get(s, 0) + 1

    # Head ties (rank 0-3) specifically
    head = [r for r in results if r["rank_a"] < 4]
    head_n = len(head)
    head_any_signal = sum(1 for r in head if r["tie_break_signals_present"])

    # Print per-pair detail (condensed)
    for r in results:
        sig = ",".join(r["tie_break_signals_present"]) or "NONE"
        edge = f"edge_w={r['direct_edge']['weight']:.2f}" if r["direct_edge"] else "no_edge"
        print(
            f"  q{r['idx']:2d} r{r['rank_a']:2d}-{r['rank_b']:<2d} "
            f"nb=({r['neighbors_a']:3d}|{r['neighbors_b']:3d},shared={r['shared_neighbors']:3d}) "
            f"qmatch=({r['query_match_a']}|{r['query_match_b']}) "
            f"{edge:<15s} signals={sig}"
        )

    print()
    print(f"[graph-coverage] aggregate (N={n} tied pairs):")
    print(f"  direct harmonic edge present:    {n_direct_edge:3d} ({100*n_direct_edge/n:5.1f}%)")
    print(f"  neighborhood size asymmetric:    {n_neighborhood_asym:3d} ({100*n_neighborhood_asym/n:5.1f}%)")
    print(f"  query-entity affinity asym:      {n_query_affinity_asym:3d} ({100*n_query_affinity_asym/n:5.1f}%)")
    print(f"  NLI relation between:            {n_nli:3d} ({100*n_nli/n:5.1f}%)")
    print(f"  ANY tie-break signal present:    {n_any_signal:3d} ({100*n_any_signal/n:5.1f}%)")
    print(f"  NO signal (graph-invisible tie): {n_no_signal:3d} ({100*n_no_signal/n:5.1f}%)")
    print()
    print(f"  head ties (rank 0-3): {head_n}")
    print(f"  head ties with any signal: {head_any_signal} ({100*head_any_signal/head_n if head_n else 0:.1f}%)")
    print()
    print(f"[graph-coverage] signal frequency:")
    for s, c in sorted(signal_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:25s} {c:3d} ({100*c/n:5.1f}%)")

    # Verdict
    if n_any_signal / n >= 0.8:
        verdict = "BUILD_ASSOC_TIE_BREAK - graph has signal on 80pct+ of ties; walking tie-break is implementable"
    elif n_any_signal / n >= 0.5:
        verdict = "PARTIAL_SIGNAL - build associative tie-break for the pairs with signal; fall back to alt-signal (freshness/provenance) for the rest"
    else:
        verdict = "GRAPH_SPARSE - co-activation graph doesn't distinguish most ties; tie-break needs different signal (freshness, party_id, recent history)"
    print()
    print(f"[graph-coverage] verdict: {verdict}")

    output = {
        "input": INPUT_JSON.name,
        "n_tied_pairs": n,
        "n_direct_edge": n_direct_edge,
        "n_neighborhood_asym": n_neighborhood_asym,
        "n_query_affinity_asym": n_query_affinity_asym,
        "n_nli_relation": n_nli,
        "n_any_signal": n_any_signal,
        "n_no_signal": n_no_signal,
        "signal_frequency": signal_counts,
        "head_ties": head_n,
        "head_ties_with_signal": head_any_signal,
        "verdict": verdict,
        "pairs": results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[graph-coverage] wrote {OUTPUT_JSON}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

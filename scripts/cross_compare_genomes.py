"""
Cross-compare multiple genome.db files side-by-side.

Takes N paths and emits a markdown report covering:
  * Bulk stats (gene count, harmonic_links density, avg chars/gene)
  * Top-10 outbound-hub gene IDs per genome (surfaces bench-fixture skew)
  * Gene-ID overlap across the set (what's shared vs what's unique)
  * Same 3 canonical queries against each genome via direct Python API
    (not HTTP — so the live server on :11437 stays undisturbed)

Usage:
  python scripts/cross_compare_genomes.py \\
      genome.db genome_bench_helix.db genome_bench_organic.db \\
      --out docs/FUTURE/genome_cross_compare_2026-04-15.md

The report is intentionally standalone — hand it to a human or to a
follow-up session and it should read cleanly without needing the DBs.

Never writes to any of the input genomes. Read-only access throughout.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench.compare")


# Canonical queries we run against each genome to see what it surfaces.
# Picked to probe different retrieval behaviors without being so specific
# that ONE genome has unfair advantage (e.g. "helix" is present in all
# three corpora through docs + code mentions, but expresses differently).
CANONICAL_QUERIES = [
    "session delivery working-set register",
    "harmonic links retrieval tier scoring",
    "cwola bucket accumulation",
    "context manager pipeline step assemble",
    "legibility per-gene header fired tiers",
]


def _open_readonly(path: str) -> sqlite3.Connection:
    """Open a sqlite DB in read-only URI mode so nothing mutates."""
    uri = f"file:{os.path.abspath(path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def stats_for(path: str) -> Dict[str, int]:
    """Basic counts + size distribution without touching the Genome class."""
    out: Dict[str, int] = {"path": path}
    try:
        out["size_bytes"] = os.path.getsize(path)
    except OSError:
        out["size_bytes"] = -1

    conn = _open_readonly(path)
    try:
        # Genes
        row = conn.execute("SELECT COUNT(*) AS n FROM genes").fetchone()
        out["genes"] = int(row["n"]) if row else 0

        # Chromatin split — column is an IntEnum (0=OPEN, 1=EUCHROMATIN, 2=HETEROCHROMATIN)
        row = conn.execute(
            "SELECT chromatin, COUNT(*) AS n FROM genes GROUP BY chromatin"
        ).fetchall()
        chromatin: Dict[int, int] = {int(r["chromatin"]): int(r["n"]) for r in row}
        out["open_genes"] = chromatin.get(0, 0)
        out["euchromatin"] = chromatin.get(1, 0)
        out["heterochromatin"] = chromatin.get(2, 0)

        # Harmonic links
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM harmonic_links"
            ).fetchone()
            out["harmonic_links"] = int(row["n"]) if row else 0
        except sqlite3.OperationalError:
            out["harmonic_links"] = 0

        # cwola_log rows (not content — just a pollution signal)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM cwola_log"
            ).fetchone()
            out["cwola_rows"] = int(row["n"]) if row else 0
        except sqlite3.OperationalError:
            out["cwola_rows"] = 0

        # session_delivery_log rows
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM session_delivery_log"
            ).fetchone()
            out["session_delivery_rows"] = int(row["n"]) if row else 0
        except sqlite3.OperationalError:
            out["session_delivery_rows"] = 0

        # Char totals (compression ratio proxy)
        row = conn.execute(
            "SELECT "
            "  SUM(length(content)) AS raw, "
            "  SUM(length(complement)) AS compressed "
            "FROM genes"
        ).fetchone()
        out["chars_raw"] = int(row["raw"] or 0) if row else 0
        out["chars_compressed"] = int(row["compressed"] or 0) if row else 0
    finally:
        conn.close()
    return out


def top_hubs(path: str, n: int = 10) -> List[Dict]:
    """Top-N outbound-hub genes: who has the most harmonic_links?"""
    conn = _open_readonly(path)
    try:
        rows = conn.execute(
            "SELECT gene_id_a AS gid, COUNT(*) AS out_deg, "
            "       AVG(weight) AS avg_weight "
            "FROM harmonic_links "
            "GROUP BY gene_id_a "
            "ORDER BY out_deg DESC LIMIT ?",
            (n,),
        ).fetchall()
        result = []
        for r in rows:
            # Lookup the gene's promoter summary for human-readable context
            g = conn.execute(
                "SELECT promoter FROM genes WHERE gene_id = ?",
                (r["gid"],),
            ).fetchone()
            summary = ""
            if g and g["promoter"]:
                import json
                try:
                    pro = json.loads(g["promoter"])
                    summary = (pro.get("summary") or "")[:80]
                except Exception:
                    summary = ""
            result.append({
                "gene_id": r["gid"],
                "out_degree": int(r["out_deg"]),
                "avg_weight": round(float(r["avg_weight"] or 0.0), 3),
                "summary": summary,
            })
        return result
    finally:
        conn.close()


def gene_id_set(path: str) -> set:
    """All gene_ids in the genome. Used to compute overlap."""
    conn = _open_readonly(path)
    try:
        rows = conn.execute("SELECT gene_id FROM genes").fetchall()
        return {r["gene_id"] for r in rows}
    finally:
        conn.close()


def run_query(genome_path: str, query: str, max_genes: int = 5) -> Dict:
    """Run a single /context-like query against a genome via Python API.

    Uses HelixContextManager directly so the live :11437 server stays
    undisturbed. Just the retrieval step — no splice, no ribosome calls.
    """
    from helix_context.config import (
        HelixConfig, BudgetConfig, GenomeConfig, RibosomeConfig,
    )
    from helix_context.context_manager import HelixContextManager

    cfg = HelixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=max_genes),
        genome=GenomeConfig(path=genome_path, cold_start_threshold=5),
        synonym_map={},
    )
    mgr = HelixContextManager(cfg)
    try:
        t0 = time.perf_counter()
        # Just call _express (Step 2) directly — returns ranked candidates
        # without incurring the ribosome splice cost.
        from helix_context.accel import extract_query_signals
        domains, entities = extract_query_signals(query)
        candidates = mgr._express(
            domains=domains,
            entities=entities,
            max_genes=max_genes,
            query_text=query,
            include_cold=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        scores = mgr.genome.last_query_scores or {}
        top = [
            {
                "gene_id": g.gene_id,
                "score": round(float(scores.get(g.gene_id, 0.0)), 3),
                "summary": (g.promoter.summary or "")[:70],
                "domains": list(g.promoter.domains or [])[:3],
            }
            for g in candidates[:max_genes]
        ]
        return {
            "query": query,
            "elapsed_ms": round(elapsed_ms, 1),
            "top": top,
        }
    except Exception as exc:
        log.warning("query %r failed on %s: %s", query, genome_path, exc)
        return {"query": query, "error": str(exc), "top": []}
    finally:
        mgr.close()


def render_markdown(
    report: Dict,
    *,
    queries: List[str],
    top_n: int = 10,
) -> str:
    lines: List[str] = []
    lines.append(f"# Genome cross-compare — {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("Generated by `scripts/cross_compare_genomes.py`. Compares")
    lines.append("`genome.db` (current working state) against reproducible")
    lines.append("bench-target genomes built earlier tonight.")
    lines.append("")

    # Stats table
    paths = list(report["stats"].keys())
    lines.append("## Bulk stats")
    lines.append("")
    lines.append("| metric | " + " | ".join(os.path.basename(p) for p in paths) + " |")
    lines.append("|---|" + "|".join(["---"] * len(paths)) + "|")

    def row(label: str, fn) -> str:
        cells = []
        for p in paths:
            v = fn(report["stats"][p])
            cells.append(str(v))
        return f"| {label} | " + " | ".join(cells) + " |"

    lines.append(row("size on disk (MB)",
                     lambda s: f"{s['size_bytes'] / 1_000_000:.1f}"))
    lines.append(row("genes", lambda s: f"{s['genes']:,}"))
    lines.append(row("open",
                     lambda s: f"{s.get('open_genes',0):,}"))
    lines.append(row("euchromatin",
                     lambda s: f"{s.get('euchromatin',0):,}"))
    lines.append(row("heterochromatin",
                     lambda s: f"{s.get('heterochromatin',0):,}"))
    lines.append(row("harmonic_links", lambda s: f"{s['harmonic_links']:,}"))
    lines.append(row("cwola_log rows",
                     lambda s: f"{s.get('cwola_rows',0):,}"))
    lines.append(row("session_delivery rows",
                     lambda s: f"{s.get('session_delivery_rows',0):,}"))
    lines.append(row("chars raw (M)",
                     lambda s: f"{s['chars_raw'] / 1_000_000:.1f}"))
    lines.append(row("chars compressed (M)",
                     lambda s: f"{s['chars_compressed'] / 1_000_000:.1f}"))
    lines.append(row("compression ratio",
                     lambda s: f"{s['chars_raw'] / max(s['chars_compressed'], 1):.1f}x"))
    lines.append(row("links / gene",
                     lambda s: f"{s['harmonic_links'] / max(s['genes'], 1):.2f}"))
    lines.append("")

    # Overlap matrix
    lines.append("## Gene-ID overlap")
    lines.append("")
    lines.append("Cells are |A ∩ B| / |A ∪ B| (Jaccard).")
    lines.append("")
    gid_sets = report["gid_sets"]
    names = [os.path.basename(p) for p in paths]
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|---|" + "|".join(["---"] * len(names)) + "|")
    for i, pi in enumerate(paths):
        cells = [names[i]]
        si = gid_sets[pi]
        for pj in paths:
            sj = gid_sets[pj]
            inter = len(si & sj)
            union = len(si | sj)
            cells.append(f"{inter / max(union, 1):.2%}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Top hubs per genome
    lines.append(f"## Top-{top_n} outbound hubs per genome")
    lines.append("")
    for p in paths:
        lines.append(f"### `{os.path.basename(p)}`")
        lines.append("")
        hubs = report["hubs"][p]
        if not hubs:
            lines.append("(no harmonic_links)")
            lines.append("")
            continue
        lines.append("| gene_id (12) | out-deg | avg w | summary |")
        lines.append("|---|---|---|---|")
        for h in hubs:
            short = h["gene_id"][:12]
            s = (h.get("summary") or "").replace("|", "\\|")[:60]
            lines.append(
                f"| `{short}` | {h['out_degree']} | {h['avg_weight']} | {s} |"
            )
        lines.append("")

    # Canonical queries (only when actually run)
    any_queries = any(report["queries"][p] for p in paths)
    if not any_queries:
        lines.append("## Canonical queries")
        lines.append("")
        lines.append("_Skipped (`--skip-queries`)._")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append("## Canonical queries — same prompt, different genomes")
    lines.append("")
    lines.append("Direct Python API calls (no HTTP), retrieval only — no")
    lines.append("ribosome splice. Shows what each genome *surfaces* for the")
    lines.append("same question.")
    lines.append("")

    for q in queries:
        lines.append(f"### `{q}`")
        lines.append("")
        lines.append("| genome | ms | top results (gene_id · score · summary) |")
        lines.append("|---|---|---|")
        for p in paths:
            result = report["queries"][p].get(q, {})
            if "error" in result:
                lines.append(
                    f"| {os.path.basename(p)} | — | ERROR: {result['error']} |"
                )
                continue
            ms = result.get("elapsed_ms", "?")
            top = result.get("top", [])
            if not top:
                cells = "(no hits)"
            else:
                parts = []
                for t in top[:3]:
                    short = t["gene_id"][:12]
                    sm = (t.get("summary") or "")[:40].replace("|", "\\|")
                    parts.append(
                        f"`{short}` · {t['score']} · {sm}"
                    )
                cells = "<br>".join(parts)
            lines.append(f"| {os.path.basename(p)} | {ms} | {cells} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+",
        help="Genome .db paths to compare (2 or more)",
    )
    parser.add_argument(
        "--out", default="docs/FUTURE/genome_cross_compare.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="How many hubs to list per genome (default 10)",
    )
    parser.add_argument(
        "--skip-queries", action="store_true",
        help="Skip the canonical-query sweep (saves a minute or two).",
    )
    args = parser.parse_args()

    # Validate paths
    missing = [p for p in args.paths if not os.path.exists(p)]
    if missing:
        log.error("Missing genome paths: %s", missing)
        return 1

    report: Dict = {"stats": {}, "hubs": {}, "gid_sets": {}, "queries": {}}

    # Stats + hubs per genome
    for p in args.paths:
        log.info("Stats for %s ...", p)
        report["stats"][p] = stats_for(p)
        log.info("Hubs for %s ...", p)
        report["hubs"][p] = top_hubs(p, n=args.top_n)
        log.info("Gene-ID set for %s ...", p)
        report["gid_sets"][p] = gene_id_set(p)

    # Canonical queries per genome
    if args.skip_queries:
        log.info("Skipping canonical queries per --skip-queries")
        for p in args.paths:
            report["queries"][p] = {}
    else:
        for p in args.paths:
            log.info("Running canonical queries against %s ...", p)
            report["queries"][p] = {}
            for q in CANONICAL_QUERIES:
                report["queries"][p][q] = run_query(p, q)

    # Render + write
    md = render_markdown(report, queries=CANONICAL_QUERIES, top_n=args.top_n)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    log.info("Report written: %s (%d bytes)", args.out, len(md.encode("utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

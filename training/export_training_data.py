"""
Export training data from the Helix genome for DeBERTa ribosome fine-tuning.

Produces two datasets:
  1. rerank_pairs.jsonl  — (query, gene_summary, relevance_score) for cross-encoder
  2. splice_labels.jsonl — (query, gene_id, codon_meanings, keep_indices) for token classification

Uses the existing Ollama ribosome as "teacher" to generate labels.
Run this once to produce the training set, then fine-tune DeBERTa offline.

Usage:
    python training/export_training_data.py --genome genome.db --out training/data/
    python training/export_training_data.py --genome genome.db --out training/data/ --queries training/queries.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add parent to path for helix_context imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_context.schemas import Gene, PromoterTags, EpigeneticMarkers
from helix_context.ribosome import Ribosome, OllamaBackend, _EXPRESS_SYSTEM, _splice_system, _parse_json

log = logging.getLogger("helix.training.export")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_genes(genome_path: str) -> List[Gene]:
    """Load all genes from a genome SQLite database."""
    conn = sqlite3.connect(genome_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT gene_id, content, complement, codons, promoter, epigenetics, "
        "chromatin, is_fragment FROM genes"
    ).fetchall()
    conn.close()

    genes = []
    for r in rows:
        try:
            genes.append(Gene(
                gene_id=r["gene_id"],
                content=r["content"],
                complement=r["complement"],
                codons=json.loads(r["codons"]),
                promoter=PromoterTags(**json.loads(r["promoter"])),
                epigenetics=EpigeneticMarkers(**json.loads(r["epigenetics"])),
                chromatin=r["chromatin"],
                is_fragment=bool(r["is_fragment"]),
            ))
        except Exception:
            log.warning("Skipping malformed gene %s", r["gene_id"], exc_info=True)
    return genes


def generate_synthetic_queries(genes: List[Gene], n: int = 500) -> List[str]:
    """Generate training queries from gene promoter metadata.

    Creates diverse queries by combining domains, entities, intents,
    and summaries from randomly sampled genes.
    """
    queries = []

    for g in random.sample(genes, min(n, len(genes))):
        p = g.promoter

        # Direct domain query
        if p.domains:
            queries.append(f"How does {random.choice(p.domains)} work?")

        # Entity-based query
        if p.entities:
            ent = random.choice(p.entities)
            queries.append(f"What is {ent} used for?")

        # Intent-derived query
        if p.intent:
            queries.append(p.intent)

        # Summary as natural question
        if p.summary:
            queries.append(f"Explain: {p.summary}")

    # Deduplicate, shuffle
    queries = list(set(queries))
    random.shuffle(queries)
    return queries[:n]


def teacher_rerank(
    ribosome: Ribosome,
    query: str,
    candidates: List[Gene],
) -> Dict[str, float]:
    """Use Ollama ribosome to score genes against a query (teacher labels)."""
    summaries = {
        g.gene_id: f"{g.promoter.summary} [{','.join(g.promoter.domains)}]"
        for g in candidates
    }

    prompt = (
        f"Query: {query}\n\n"
        f"Gene summaries:\n"
        + "\n".join(f"  {gid}: {s}" for gid, s in summaries.items())
    )

    try:
        raw = ribosome.backend.complete(prompt, system=_EXPRESS_SYSTEM)
        scores = _parse_json(raw)
        if isinstance(scores, dict):
            return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
    except Exception:
        log.warning("Teacher re-rank failed for query: %s", query[:80], exc_info=True)
    return {}


def teacher_splice(
    ribosome: Ribosome,
    query: str,
    genes: List[Gene],
    aggressiveness: float = 0.5,
) -> Dict[str, List[int]]:
    """Use Ollama ribosome to label codon keep/drop (teacher labels)."""
    gene_sections = []
    for g in genes:
        fragment_note = " [fragment]" if g.is_fragment else ""
        codon_list = "\n".join(f"    [{i}] {c}" for i, c in enumerate(g.codons))
        gene_sections.append(f"  Gene {g.gene_id}{fragment_note}:\n{codon_list}")

    prompt = (
        f"Query context: {query}\n\n"
        f"Genes and their codons:\n"
        + "\n\n".join(gene_sections)
        + "\n\nFor each gene, which codon indices should be KEPT?"
    )

    system = _splice_system(aggressiveness)

    try:
        raw = ribosome.backend.complete(prompt, system=system)
        parsed = _parse_json(raw)
        if isinstance(parsed, dict):
            return {
                k: [int(i) for i in v if isinstance(i, int)]
                for k, v in parsed.items()
                if isinstance(v, list)
            }
    except Exception:
        log.warning("Teacher splice failed for query: %s", query[:80], exc_info=True)
    return {}


def export(
    genome_path: str,
    out_dir: str,
    query_file: str | None = None,
    n_queries: int = 500,
    candidates_per_query: int = 16,
    batch_size: int = 8,
    timeout: float = 30.0,
) -> None:
    """Main export pipeline: genome → teacher-labeled training data."""
    os.makedirs(out_dir, exist_ok=True)

    log.info("Loading genes from %s", genome_path)
    genes = load_genes(genome_path)
    log.info("Loaded %d genes", len(genes))

    if not genes:
        log.error("No genes found — cannot generate training data")
        return

    # Load or generate queries
    if query_file and os.path.exists(query_file):
        with open(query_file) as f:
            queries = [line.strip() for line in f if line.strip()]
        log.info("Loaded %d queries from %s", len(queries), query_file)
    else:
        queries = generate_synthetic_queries(genes, n=n_queries)
        # Save generated queries for reproducibility
        queries_path = os.path.join(out_dir, "queries.txt")
        with open(queries_path, "w") as f:
            f.write("\n".join(queries))
        log.info("Generated %d synthetic queries → %s", len(queries), queries_path)

    # Initialize teacher ribosome
    backend = OllamaBackend(timeout=timeout, warmup=True)
    ribosome = Ribosome(backend=backend)
    log.info("Teacher ribosome ready (model: %s)", backend.model)

    # Build gene lookup
    gene_map = {g.gene_id: g for g in genes}

    rerank_path = os.path.join(out_dir, "rerank_pairs.jsonl")
    splice_path = os.path.join(out_dir, "splice_labels.jsonl")

    rerank_count = 0
    splice_count = 0

    with open(rerank_path, "w") as f_rerank, open(splice_path, "w") as f_splice:
        for qi, query in enumerate(queries):
            if qi % 10 == 0:
                log.info("Progress: %d/%d queries (rerank=%d, splice=%d)",
                         qi, len(queries), rerank_count, splice_count)

            # Sample random candidates (mix of relevant and irrelevant)
            candidates = random.sample(genes, min(candidates_per_query, len(genes)))

            # ── Teacher re-rank ──
            scores = teacher_rerank(ribosome, query, candidates)
            for gid, score in scores.items():
                if gid in gene_map:
                    g = gene_map[gid]
                    record = {
                        "query": query,
                        "gene_id": gid,
                        "summary": g.promoter.summary,
                        "domains": g.promoter.domains,
                        "entities": g.promoter.entities,
                        "score": round(score, 3),
                    }
                    f_rerank.write(json.dumps(record) + "\n")
                    rerank_count += 1

            # Also record negatives (genes not scored or scored 0)
            for g in candidates:
                if g.gene_id not in scores:
                    record = {
                        "query": query,
                        "gene_id": g.gene_id,
                        "summary": g.promoter.summary,
                        "domains": g.promoter.domains,
                        "entities": g.promoter.entities,
                        "score": 0.0,
                    }
                    f_rerank.write(json.dumps(record) + "\n")
                    rerank_count += 1

            # ── Teacher splice (batch genes that got scored) ──
            scored_genes = [gene_map[gid] for gid in scores if gid in gene_map]
            if scored_genes:
                # Process in batches to avoid huge prompts
                for i in range(0, len(scored_genes), batch_size):
                    batch = scored_genes[i : i + batch_size]
                    keep_map = teacher_splice(ribosome, query, batch)
                    for gid, indices in keep_map.items():
                        if gid in gene_map:
                            g = gene_map[gid]
                            record = {
                                "query": query,
                                "gene_id": gid,
                                "codons": g.codons,
                                "keep_indices": indices,
                                "total_codons": len(g.codons),
                            }
                            f_splice.write(json.dumps(record) + "\n")
                            splice_count += 1

            # Throttle to avoid overwhelming Ollama
            time.sleep(0.3)

    log.info("Export complete:")
    log.info("  Re-rank pairs: %d → %s", rerank_count, rerank_path)
    log.info("  Splice labels: %d → %s", splice_count, splice_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DeBERTa training data from Helix genome")
    parser.add_argument("--genome", default="genome.db", help="Path to genome.db")
    parser.add_argument("--out", default="training/data/", help="Output directory")
    parser.add_argument("--queries", default=None, help="Optional queries file (one per line)")
    parser.add_argument("--n-queries", type=int, default=500, help="Number of synthetic queries")
    parser.add_argument("--candidates", type=int, default=16, help="Candidates per query")
    parser.add_argument("--timeout", type=float, default=30.0, help="Ollama timeout (seconds)")
    args = parser.parse_args()

    export(
        genome_path=args.genome,
        out_dir=args.out,
        query_file=args.queries,
        n_queries=args.n_queries,
        candidates_per_query=args.candidates,
        timeout=args.timeout,
    )

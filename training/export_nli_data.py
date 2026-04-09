"""
Export NLI training data from the Helix genome.

Generates (gene_summary_a, gene_summary_b, relation) triples using three
complementary strategies:

  A. Heuristic labels from genome structure (domain/entity overlap)
  B. Teacher labels from Ollama (high quality, low volume)
  C. Codon-level labels from existing splice decisions

Output: training/data/nli_pairs.jsonl

The 7 MacCartney-Manning relations:
  0=entailment, 1=reverse_entailment, 2=equivalence,
  3=alternation, 4=negation, 5=cover, 6=independence

Usage:
    python training/export_nli_data.py --genome genome.db --out training/data/
    python training/export_nli_data.py --genome genome.db --out training/data/ --teacher --teacher-count 3000
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
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_context.schemas import Gene, PromoterTags, EpigeneticMarkers

log = logging.getLogger("helix.training.nli")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Relation labels
ENTAILMENT = 0
REVERSE_ENTAILMENT = 1
EQUIVALENCE = 2
ALTERNATION = 3
NEGATION = 4
COVER = 5
INDEPENDENCE = 6

RELATION_NAMES = [
    "entailment", "reverse_entailment", "equivalence",
    "alternation", "negation", "cover", "independence",
]


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


def _domain_overlap(a: Gene, b: Gene) -> float:
    """Jaccard similarity of domain sets."""
    da, db = set(a.promoter.domains), set(b.promoter.domains)
    if not da and not db:
        return 0.0
    union = da | db
    return len(da & db) / len(union) if union else 0.0


def _entity_overlap(a: Gene, b: Gene) -> float:
    """Jaccard similarity of entity sets."""
    ea, eb = set(a.promoter.entities), set(b.promoter.entities)
    if not ea and not eb:
        return 0.0
    union = ea | eb
    return len(ea & eb) / len(union) if union else 0.0


def _is_subset(small: set, big: set) -> bool:
    """True if small is a proper subset of big."""
    return len(small) > 0 and small < big


def _co_activated(a: Gene, b: Gene) -> bool:
    """Check if genes have mutual co-activation links."""
    return (
        b.gene_id in a.epigenetics.co_activated_with
        or a.gene_id in b.epigenetics.co_activated_with
    )


# ── Strategy A: Heuristic labels ────────────────────────────────────


def heuristic_nli_labels(genes: List[Gene], max_per_class: int = 3000) -> List[dict]:
    """Generate NLI labels from genome structure heuristics."""
    log.info("Generating heuristic NLI labels from %d genes...", len(genes))

    equivalence = []
    entailment = []
    reverse_entailment = []
    alternation = []
    independence = []

    # Build lookup for efficient sampling
    gene_map = {g.gene_id: g for g in genes}

    # Sample pairs (full C(n,2) is too large for 3500+ genes)
    all_pairs = list(combinations(genes, 2))
    random.shuffle(all_pairs)

    # Limit to avoid excessive processing
    sample_size = min(len(all_pairs), 50000)
    sampled = all_pairs[:sample_size]

    for a, b in sampled:
        d_overlap = _domain_overlap(a, b)
        e_overlap = _entity_overlap(a, b)
        co_act = _co_activated(a, b)
        da, db = set(a.promoter.domains), set(b.promoter.domains)
        ea, eb = set(a.promoter.entities), set(b.promoter.entities)

        # Equivalence: high domain + entity overlap + co-activation
        if d_overlap >= 0.8 and e_overlap >= 0.5 and co_act:
            if len(equivalence) < max_per_class:
                equivalence.append((a, b, EQUIVALENCE))
            continue

        # Entailment: A's domains ⊂ B's domains
        if _is_subset(da, db) and (not ea or ea <= eb):
            if len(entailment) < max_per_class:
                entailment.append((a, b, ENTAILMENT))
            continue

        # Reverse entailment: B's domains ⊂ A's domains
        if _is_subset(db, da) and (not eb or eb <= ea):
            if len(reverse_entailment) < max_per_class:
                reverse_entailment.append((a, b, REVERSE_ENTAILMENT))
            continue

        # Alternation: shared domain, zero entity overlap, no co-activation
        if d_overlap > 0.0 and e_overlap == 0.0 and not co_act:
            if len(alternation) < max_per_class:
                alternation.append((a, b, ALTERNATION))
            continue

        # Independence: no shared domains, entities, or co-activation
        if d_overlap == 0.0 and e_overlap == 0.0 and not co_act:
            if len(independence) < max_per_class:
                independence.append((a, b, INDEPENDENCE))

    all_labeled = equivalence + entailment + reverse_entailment + alternation + independence
    random.shuffle(all_labeled)

    log.info(
        "Heuristic labels: eq=%d ent=%d rev=%d alt=%d ind=%d total=%d",
        len(equivalence), len(entailment), len(reverse_entailment),
        len(alternation), len(independence), len(all_labeled),
    )

    records = []
    for a, b, label in all_labeled:
        records.append({
            "text_a": _gene_text(a),
            "text_b": _gene_text(b),
            "label": label,
            "level": "gene",
            "gene_id_a": a.gene_id,
            "gene_id_b": b.gene_id,
            "source": "heuristic",
        })
    return records


def _gene_text(g: Gene) -> str:
    """Build a representative text for a gene (summary + domains + entities)."""
    parts = []
    if g.promoter.summary:
        parts.append(g.promoter.summary)
    if g.promoter.domains:
        parts.append(f"[{', '.join(g.promoter.domains)}]")
    if g.promoter.entities:
        parts.append(f"({', '.join(g.promoter.entities[:5])})")
    return " ".join(parts) if parts else g.complement[:200]


# ── Strategy B: Teacher labels ──────────────────────────────────────

_NLI_SYSTEM = """You classify the logical relation between two gene summaries.

Relations (respond with the number):
  0 = entailment: A implies B (A is more specific than B)
  1 = reverse_entailment: B implies A (B is more specific than A)
  2 = equivalence: A and B say the same thing
  3 = alternation: A and B are mutually exclusive (both about same topic, different aspects)
  4 = negation: A and B are direct opposites
  5 = cover: A and B overlap and together cover the full topic
  6 = independence: no reliable relation

Respond ONLY with a JSON object: {"relation": <0-6>, "confidence": <0.0-1.0>}"""


def teacher_nli_labels(
    genes: List[Gene],
    backend,
    n_pairs: int = 3000,
    timeout: float = 30.0,
) -> List[dict]:
    """Use Ollama to generate high-quality NLI labels."""
    from helix_context.ribosome import _parse_json

    log.info("Generating teacher NLI labels for %d pairs...", n_pairs)

    # Sample diverse pairs (mix of related and unrelated)
    pairs = []
    gene_list = list(genes)

    for _ in range(n_pairs):
        a, b = random.sample(gene_list, 2)
        pairs.append((a, b))

    records = []
    for i, (a, b) in enumerate(pairs):
        if i % 100 == 0:
            log.info("Teacher NLI progress: %d/%d", i, len(pairs))

        prompt = (
            f"Gene A: {_gene_text(a)}\n"
            f"Gene B: {_gene_text(b)}\n\n"
            f"What is the logical relation between Gene A and Gene B?"
        )

        try:
            raw = backend.complete(prompt, system=_NLI_SYSTEM)
            parsed = _parse_json(raw)
            if isinstance(parsed, dict) and "relation" in parsed:
                rel = int(parsed["relation"])
                conf = float(parsed.get("confidence", 0.5))
                if 0 <= rel <= 6 and conf >= 0.5:
                    records.append({
                        "text_a": _gene_text(a),
                        "text_b": _gene_text(b),
                        "label": rel,
                        "level": "gene",
                        "gene_id_a": a.gene_id,
                        "gene_id_b": b.gene_id,
                        "source": "teacher",
                        "confidence": conf,
                    })
        except Exception:
            log.warning("Teacher NLI failed for pair %d", i, exc_info=True)

        time.sleep(0.3)

    log.info("Teacher labels: %d (from %d attempts)", len(records), len(pairs))
    return records


# ── Strategy C: Codon-level labels ──────────────────────────────────


def codon_nli_labels(
    splice_path: str,
    genes: List[Gene],
    max_pairs: int = 5000,
) -> List[dict]:
    """Generate codon-level NLI labels from existing splice decisions."""
    gene_map = {g.gene_id: g for g in genes}
    records = []

    if not os.path.exists(splice_path):
        log.info("No splice labels found at %s, skipping codon NLI", splice_path)
        return records

    with open(splice_path) as f:
        for line in f:
            if len(records) >= max_pairs:
                break

            rec = json.loads(line)
            gid = rec.get("gene_id")
            codons = rec.get("codons", [])
            keep_set = set(rec.get("keep_indices", []))

            if gid not in gene_map or len(codons) < 2:
                continue

            gene = gene_map[gid]
            gene_summary = gene.promoter.summary or ""

            # Pairs of codons within the same gene
            for i, ci in enumerate(codons):
                for j, cj in enumerate(codons):
                    if i >= j:
                        continue
                    if len(records) >= max_pairs:
                        break

                    i_kept = i in keep_set
                    j_kept = j in keep_set

                    # Both kept → likely entailment or equivalence
                    if i_kept and j_kept:
                        label = ENTAILMENT if abs(i - j) == 1 else EQUIVALENCE
                    # One kept, one dropped → alternation or independence
                    elif i_kept and not j_kept:
                        label = ALTERNATION
                    elif not i_kept and j_kept:
                        label = REVERSE_ENTAILMENT
                    else:
                        # Both dropped → independence
                        label = INDEPENDENCE

                    text_a = f"{ci} [{gene_summary}]" if gene_summary else ci
                    text_b = f"{cj} [{gene_summary}]" if gene_summary else cj

                    records.append({
                        "text_a": text_a,
                        "text_b": text_b,
                        "label": label,
                        "level": "codon",
                        "gene_id_a": gid,
                        "gene_id_b": gid,
                        "source": "codon_splice",
                    })

    random.shuffle(records)
    log.info("Codon-level NLI labels: %d", len(records))
    return records


# ── Main export ─────────────────────────────────────────────────────


def export(
    genome_path: str,
    out_dir: str,
    use_teacher: bool = False,
    teacher_count: int = 3000,
    splice_path: str = "",
    timeout: float = 30.0,
) -> None:
    """Main export pipeline."""
    os.makedirs(out_dir, exist_ok=True)

    log.info("Loading genes from %s", genome_path)
    genes = load_genes(genome_path)
    log.info("Loaded %d genes", len(genes))

    if not genes:
        log.error("No genes found")
        return

    all_records = []

    # Strategy A: Heuristic
    all_records.extend(heuristic_nli_labels(genes))

    # Strategy B: Teacher (optional, slow)
    if use_teacher:
        from helix_context.ribosome import OllamaBackend
        backend = OllamaBackend(timeout=timeout, warmup=True)
        all_records.extend(teacher_nli_labels(genes, backend, n_pairs=teacher_count))

    # Strategy C: Codon-level (if splice labels exist)
    if splice_path:
        all_records.extend(codon_nli_labels(splice_path, genes))

    # Write output
    out_path = os.path.join(out_dir, "nli_pairs.jsonl")
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Stats
    label_counts = {}
    for rec in all_records:
        name = RELATION_NAMES[rec["label"]]
        label_counts[name] = label_counts.get(name, 0) + 1

    log.info("Export complete: %d pairs → %s", len(all_records), out_path)
    for name, count in sorted(label_counts.items()):
        log.info("  %s: %d", name, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NLI training data from Helix genome")
    parser.add_argument("--genome", default="genome.db", help="Path to genome.db")
    parser.add_argument("--out", default="training/data/", help="Output directory")
    parser.add_argument("--teacher", action="store_true", help="Use Ollama teacher labels")
    parser.add_argument("--teacher-count", type=int, default=3000, help="Number of teacher pairs")
    parser.add_argument("--splice-labels", default="", help="Path to splice_labels.jsonl for codon NLI")
    parser.add_argument("--timeout", type=float, default=30.0, help="Ollama timeout")
    args = parser.parse_args()

    export(
        genome_path=args.genome,
        out_dir=args.out,
        use_teacher=args.teacher,
        teacher_count=args.teacher_count,
        splice_path=args.splice_labels,
        timeout=args.timeout,
    )

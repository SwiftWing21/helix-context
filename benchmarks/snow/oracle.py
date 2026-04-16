"""Oracle consumer — string matching per data tier."""
from __future__ import annotations
import time
from typing import Dict, List, Tuple


def _answer_in_text(text: str, accept: List[str]) -> bool:
    lower = text.lower()
    return any(a.lower() in lower for a in accept)


def _answer_in_entities(entities: List[str], accept: List[str]) -> bool:
    joined = ' '.join(entities).lower()
    return any(a.lower() in joined for a in accept)


def oracle_cascade(
    expected_answer: str,
    accept: List[str],
    gene_ids: List[str],
    fingerprints: Dict[str, Dict],
    neighbors: Dict[str, List[Tuple[str, float]]],
) -> Dict:
    """Walk the cascade with perfect knowledge. Return the first tier
    where the answer appears.

    Returns: {tier: int (-1=MISS, 0-4), gene_id: str|None,
              tokens: int, latency_s: float}
    """
    t0 = time.perf_counter()
    tokens = 0

    # T0: fingerprint entities
    for gid in gene_ids:
        fp = fingerprints.get(gid, {})
        entities = fp.get("entities", [])
        tokens += len(' '.join(entities)) // 4 + 1
        if _answer_in_entities(entities, accept):
            return {"tier": 0, "gene_id": gid, "tokens": tokens,
                    "latency_s": time.perf_counter() - t0}

    # T1: key_values
    for gid in gene_ids:
        kv = fingerprints.get(gid, {}).get("key_values", "{}")
        tokens += len(kv) // 4 + 1
        if _answer_in_text(kv, accept):
            return {"tier": 1, "gene_id": gid, "tokens": tokens,
                    "latency_s": time.perf_counter() - t0}

    # T2: complement
    for gid in gene_ids:
        comp = fingerprints.get(gid, {}).get("complement", "")
        tokens += len(comp) // 4 + 1
        if _answer_in_text(comp, accept):
            return {"tier": 2, "gene_id": gid, "tokens": tokens,
                    "latency_s": time.perf_counter() - t0}

    # T3: content
    for gid in gene_ids:
        content = fingerprints.get(gid, {}).get("content", "")
        tokens += len(content) // 4 + 1
        if _answer_in_text(content, accept):
            return {"tier": 3, "gene_id": gid, "tokens": tokens,
                    "latency_s": time.perf_counter() - t0}

    # T4: walk — check 1-hop neighbors (top 3 by weight), content only
    for gid in gene_ids:
        nbs = neighbors.get(gid, [])
        for nb_id, _weight in sorted(nbs, key=lambda x: -x[1])[:3]:
            nb_fp = fingerprints.get(nb_id, {})
            content = nb_fp.get("content", "")
            tokens += len(content) // 4 + 1
            if _answer_in_text(content, accept):
                return {"tier": 4, "gene_id": nb_id, "tokens": tokens,
                        "latency_s": time.perf_counter() - t0}

    return {"tier": -1, "gene_id": None, "tokens": tokens,
            "latency_s": time.perf_counter() - t0}

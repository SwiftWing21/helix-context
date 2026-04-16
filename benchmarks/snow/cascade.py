"""LLM cascade consumer — tier escalation with real model calls."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .prompts import (
    TRIAGE_SYSTEM,
    triage_prompt,
    EXTRACT_SYSTEM,
    extract_prompt,
    clean_response,
)

TIER_FIELDS = [
    (1, "key_values", "Key-Value"),
    (2, "complement", "Complement"),
    (3, "content", "Content"),
]


def _parse_response(text: str) -> tuple:
    """Parse cleaned model response into (action, value)."""
    text = text.strip()
    for prefix in ("ANSWER:", "READ:", "ESCALATE", "MISS"):
        if text.upper().startswith(prefix):
            value = text[len(prefix):].strip()
            action = prefix.rstrip(":")
            return action, value
    return "UNKNOWN", text


def llm_cascade(
    query: str,
    fingerprints: List[Dict],
    model: Any,
    gene_fields: Dict[str, Dict],
    neighbors: Optional[Dict] = None,
) -> Dict:
    """Run the LLM cascade: triage -> read tiers -> walk neighbors."""
    hops = 0
    tokens = 0
    hop_detail: List[Dict] = []
    t_start = time.perf_counter()

    # --- T0: triage from fingerprint ---
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": TRIAGE_SYSTEM},
        {"role": "user", "content": triage_prompt(query, fingerprints)},
    ]
    resp = model.chat(messages)
    raw_content = resp["message"]["content"]
    content = clean_response(raw_content)
    tok = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
    tokens += tok
    elapsed = time.perf_counter() - t0
    action, value = _parse_response(content)
    hop_detail.append({
        "tier": 0, "action": f"{action} {value}",
        "tokens": tok, "latency_s": elapsed,
    })

    if action == "ANSWER":
        return {
            "tier": 0, "hops": 0, "answer": value, "miss": False,
            "tokens": tokens, "latency_s": time.perf_counter() - t_start,
            "hop_detail": hop_detail, "gene_id": None,
        }

    if action == "MISS":
        return {
            "tier": -1, "hops": 0, "answer": None, "miss": True,
            "tokens": tokens, "latency_s": time.perf_counter() - t_start,
            "hop_detail": hop_detail, "gene_id": None,
        }

    # action == READ — fuzzy match gene_id (LLM may return truncated 12-char)
    gene_to_read = value.strip()
    matched = None
    for fp in fingerprints:
        fid = fp["gene_id"]
        if fid.startswith(gene_to_read) or gene_to_read.startswith(fid[:12]):
            matched = fid
            break
    if not matched and fingerprints:
        matched = fingerprints[0]["gene_id"]
    gene_id = matched

    # --- T1-T3: escalate through tiers ---
    fields = gene_fields.get(gene_id, {})
    for tier_num, field_name, tier_label in TIER_FIELDS:
        data = fields.get(field_name, "")
        if not data:
            continue
        hops += 1  # only count tiers we actually read

        t_hop = time.perf_counter()
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": extract_prompt(query, tier_label, data)},
        ]
        resp = model.chat(messages)
        raw_content = resp["message"]["content"]
        content = clean_response(raw_content)
        tok = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
        tokens += tok
        elapsed = time.perf_counter() - t_hop
        action, value = _parse_response(content)
        hop_detail.append({
            "tier": tier_num, "action": f"{action} {value}",
            "tokens": tok, "latency_s": elapsed,
        })

        if action == "ANSWER":
            return {
                "tier": tier_num, "hops": hops, "answer": value,
                "miss": False, "tokens": tokens, "gene_id": gene_id,
                "latency_s": time.perf_counter() - t_start,
                "hop_detail": hop_detail,
            }

    # --- T4: walk — read top-3 neighbor content ---
    if neighbors and gene_id and gene_id in neighbors:
        hops += 1
        nb_list = sorted(neighbors[gene_id], key=lambda x: -x[1])[:3]
        for nb_id, _w in nb_list:
            nb_content = gene_fields.get(nb_id, {}).get("content", "")
            if not nb_content:
                continue
            t_hop = time.perf_counter()
            messages = [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": extract_prompt(
                    query, "Neighbor content", nb_content,
                )},
            ]
            resp = model.chat(messages)
            raw_content = resp["message"]["content"]
            content = clean_response(raw_content)
            tok = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
            tokens += tok
            elapsed = time.perf_counter() - t_hop
            action, value = _parse_response(content)
            hop_detail.append({
                "tier": 4, "action": f"{action} {value}",
                "tokens": tok, "latency_s": elapsed,
            })
            if action == "ANSWER":
                return {
                    "tier": 4, "hops": hops, "answer": value,
                    "miss": False, "tokens": tokens, "gene_id": nb_id,
                    "latency_s": time.perf_counter() - t_start,
                    "hop_detail": hop_detail,
                }

    return {
        "tier": -1, "hops": hops, "answer": None, "miss": True,
        "tokens": tokens, "latency_s": time.perf_counter() - t_start,
        "hop_detail": hop_detail, "gene_id": gene_id,
    }

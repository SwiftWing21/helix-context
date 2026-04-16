"""Prompt templates for the SNOW LLM consumer."""
from __future__ import annotations
import re
from typing import Dict, List

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def clean_response(text: str) -> str:
    """Strip qwen3 think tags and normalize whitespace."""
    return _THINK_RE.sub("", text).strip()

TRIAGE_SYSTEM = (
    "You consume retrieval fingerprints from a knowledge store. "
    "For each query, you see gene metadata: tier scores, source file, "
    "domains, and entities. You do NOT see content.\n\n"
    "If you can answer from the fingerprint: respond with ANSWER: <value>\n"
    "If you need to read a gene: respond with READ: <gene_id>\n"
    "If no gene seems relevant: respond with MISS\n"
    "Be concise. One line only. /no_think"
)

def triage_prompt(query: str, fingerprints: List[Dict]) -> str:
    lines = [f"QUERY: {query}\n"]
    for fp in fingerprints:
        gid = fp.get("gene_id", "unknown")[:12]
        src = fp.get("source", "unknown")
        if src and len(src) > 45:
            src = "..." + src[-42:]
        score = fp.get("score", 0.0)
        tiers = fp.get("tiers", {})
        domains = fp.get("domains", [])
        entities = fp.get("entities", [])
        lines.append(
            f"Gene {gid}: src={src} s={score:.1f} "
            f"tiers={tiers} domains={domains} entities={entities}"
        )
    return "\n".join(lines)


EXTRACT_SYSTEM = (
    "You receive data from a knowledge store gene. "
    "Answer the question using ONLY this data.\n\n"
    "If you can answer: respond with ANSWER: <value>\n"
    "If this data doesn't contain the answer: respond with ESCALATE\n"
    "One line only. /no_think"
)

def extract_prompt(query: str, tier_name: str, data: str) -> str:
    return f"QUERY: {query}\n\n{tier_name} data:\n{data}"

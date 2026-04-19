"""Literal claim extraction over genes — Phase 2 of the agent-context-index.

Turns gene content into structured facts (claims) that the packet
builder can query without reopening bulk content. V1 prioritizes
literal claims: exact path/value/symbol/port extraction via regex.

Dispatch on ``Gene.source_kind`` with a per-kind extractor. If
``source_kind`` is unset, we still harvest ``Gene.key_values`` (already
pre-extracted ``k=v`` facts at ingest) as config-value claims — that
alone answers "what ports/paths/models does the genome know about?"
without any per-kind logic.

See ``docs/specs/2026-04-17-agent-context-index-build-spec.md`` §279
for the full claim-model design and scoring rationale.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional

from .schemas import Claim, Gene

# ── Entity-key extraction ────────────────────────────────────────────

_FILE_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:[A-Za-z]:[\\/])?"                 # optional drive letter (Windows)
    r"(?:[\w./\\-]+[\\/])+"                 # at least one dir separator
    r"[\w./\\-]+"                           # final path component
    r"\.[A-Za-z][A-Za-z0-9]{0,5}"           # extension (1-6 alnum chars)
    r")"
)
_PORT_RE = re.compile(r"\bport[\s:=]+(?P<port>\d{2,5})\b", re.IGNORECASE)
_URL_RE = re.compile(r"\bhttps?://(?P<url>[\w.\-/:]+)", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"\b(?P<sym>[A-Z][A-Z0-9_]{3,})\b")  # CONST-CASE
_KV_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z_][\w.-]*)\s*[:=]\s*(?P<val>.+?)\s*$"
)


def extract_entity_keys(claim_text: str) -> list[str]:
    """Heuristic entity anchors: file paths, symbols, config keys, URLs, ports.

    Returns a de-duped list in discovery order. Best-effort — don't
    trust for exact lookup, trust for coarse filter.
    """
    seen: list[str] = []

    def _add(s: Optional[str]) -> None:
        if s and s not in seen:
            seen.append(s)

    for m in _FILE_PATH_RE.finditer(claim_text):
        _add(m.group("path"))
    for m in _URL_RE.finditer(claim_text):
        _add(f"http://{m.group('url')}")
    for m in _PORT_RE.finditer(claim_text):
        _add(f"port:{m.group('port')}")
    for m in _SYMBOL_RE.finditer(claim_text):
        _add(m.group("sym"))
    kv_match = _KV_RE.match(claim_text)
    if kv_match:
        _add(kv_match.group("key"))
    return seen


# ── Claim ID ─────────────────────────────────────────────────────────


def claim_id_for(
    gene_id: str,
    claim_type: str,
    claim_text: str,
    entity_key: Optional[str] = None,
) -> str:
    """Deterministic 16-char hash so re-ingestion is idempotent."""
    h = hashlib.sha1()
    h.update(gene_id.encode("utf-8"))
    h.update(b"|")
    h.update(claim_type.encode("utf-8"))
    h.update(b"|")
    h.update((entity_key or "").encode("utf-8"))
    h.update(b"|")
    h.update(claim_text.encode("utf-8"))
    return h.hexdigest()[:16]


# ── Per-kind extractors ──────────────────────────────────────────────


_CODE_DEF_RE = re.compile(
    r"^\s*(?:def|class|async\s+def)\s+(?P<name>[A-Za-z_]\w*)\s*[\(:]",
    re.MULTILINE,
)
_CODE_CONST_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z0-9_]{2,})\s*=\s*(?P<val>[^\n]+)$",
    re.MULTILINE,
)
_MD_HEADER_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>[^\n]+)$", re.MULTILINE)


def _extract_code_claims(gene: Gene, shard_name: str) -> list[Claim]:
    out: list[Claim] = []
    for m in _CODE_DEF_RE.finditer(gene.content):
        name = m.group("name")
        text = f"{name} defined in {gene.source_id or gene.gene_id}"
        out.append(_mk_claim(
            gene, shard_name, "api_contract", text,
            entity_key=name, specificity=0.9, confidence=0.95,
        ))
    for m in _CODE_CONST_RE.finditer(gene.content):
        name, val = m.group("name"), m.group("val").strip()
        if len(val) > 200:
            continue
        text = f"{name} = {val}"
        out.append(_mk_claim(
            gene, shard_name, "config_value", text,
            entity_key=name, specificity=1.0, confidence=0.95,
        ))
    return out


def _extract_doc_claims(gene: Gene, shard_name: str) -> list[Claim]:
    out: list[Claim] = []
    # Markdown headers → version_marker / operational_state claims
    for m in _MD_HEADER_RE.finditer(gene.content):
        title = m.group("title").strip()
        # H1/H2 only — deeper headers are usually sub-sections of the same fact
        if len(m.group("hashes")) > 2:
            continue
        if len(title) > 200:
            continue
        out.append(_mk_claim(
            gene, shard_name, "operational_state", f"section: {title}",
            entity_key=title[:64], specificity=0.7, confidence=0.8,
        ))
    # Port mentions in prose → path_value claims
    for m in _PORT_RE.finditer(gene.content):
        port = m.group("port")
        text = f"port {port}"
        out.append(_mk_claim(
            gene, shard_name, "path_value", text,
            entity_key=f"port:{port}", specificity=1.0, confidence=0.9,
        ))
    return out


def _extract_benchmark_claims(gene: Gene, shard_name: str) -> list[Claim]:
    """Parse ``metric: value`` and ``metric = value`` pairs from gene content."""
    out: list[Claim] = []
    for line in gene.content.splitlines():
        m = _KV_RE.match(line)
        if not m:
            continue
        key, val = m.group("key"), m.group("val").strip()
        # Must contain at least one digit to be benchmark-shaped
        if not any(c.isdigit() for c in val):
            continue
        if len(val) > 120:
            continue
        text = f"{key} = {val}"
        out.append(_mk_claim(
            gene, shard_name, "benchmark_result", text,
            entity_key=key, specificity=1.0, confidence=0.9,
        ))
    return out


def _extract_config_claims(gene: Gene, shard_name: str) -> list[Claim]:
    """TOML/YAML/INI-style top-level key = value pairs.

    Deliberately conservative: skip indented lines (nested keys) and
    anything that looks like prose (no assignment operator).
    """
    out: list[Claim] = []
    for line in gene.content.splitlines():
        if line.startswith((" ", "\t", "#", ";")):
            continue
        m = _KV_RE.match(line)
        if not m:
            continue
        key, val = m.group("key"), m.group("val").strip().strip('"').strip("'")
        if not val or len(val) > 200:
            continue
        text = f"{key} = {val}"
        out.append(_mk_claim(
            gene, shard_name, "config_value", text,
            entity_key=key, specificity=1.0, confidence=0.95,
        ))
    return out


# ── key_values fallback (source_kind-agnostic) ───────────────────────


def _extract_key_value_claims(gene: Gene, shard_name: str) -> list[Claim]:
    """Turn ``Gene.key_values`` (pre-extracted at ingest) into config claims.

    ``key_values`` is already a list of ``"port=11437"``-style strings —
    the cheapest possible source of literal facts, available regardless
    of ``source_kind``.
    """
    out: list[Claim] = []
    for kv in gene.key_values or []:
        m = _KV_RE.match(kv)
        if not m:
            continue
        key, val = m.group("key"), m.group("val").strip()
        if not val:
            continue
        text = f"{key} = {val}"
        out.append(_mk_claim(
            gene, shard_name, "config_value", text,
            entity_key=key, specificity=1.0, confidence=0.95,
        ))
    return out


# ── Dispatch ─────────────────────────────────────────────────────────

_EXTRACTORS = {
    "code": _extract_code_claims,
    "config": _extract_config_claims,
    "doc": _extract_doc_claims,
    "benchmark": _extract_benchmark_claims,
}


def extract_literal_claims(
    gene: Gene,
    shard_name: str = "main",
    dedup: bool = True,
) -> list[Claim]:
    """Extract literal claims from a gene.

    Always harvests ``gene.key_values`` (pre-extracted facts at ingest).
    Then dispatches on ``gene.source_kind`` for kind-specific claims.
    Unknown or missing ``source_kind`` produces only key_values claims —
    still useful, just narrower.

    ``dedup=True`` removes claims with identical ``claim_id``, which can
    happen when the same ``key = value`` appears in both ``key_values``
    and the extracted content.
    """
    claims: list[Claim] = []
    claims.extend(_extract_key_value_claims(gene, shard_name))
    extractor = _EXTRACTORS.get(gene.source_kind or "")
    if extractor is not None:
        claims.extend(extractor(gene, shard_name))
    if dedup:
        seen: set[str] = set()
        unique: list[Claim] = []
        for c in claims:
            if c.claim_id in seen:
                continue
            seen.add(c.claim_id)
            unique.append(c)
        return unique
    return claims


# ── Helpers ──────────────────────────────────────────────────────────


def _mk_claim(
    gene: Gene,
    shard_name: str,
    claim_type: str,
    claim_text: str,
    entity_key: Optional[str] = None,
    specificity: float = 0.8,
    confidence: float = 0.9,
) -> Claim:
    # If no entity key was supplied, try to pull one out of the claim text.
    if entity_key is None:
        keys = extract_entity_keys(claim_text)
        entity_key = keys[0] if keys else None
    cid = claim_id_for(gene.gene_id, claim_type, claim_text, entity_key)
    return Claim(
        claim_id=cid,
        gene_id=gene.gene_id,
        shard_name=shard_name,
        claim_type=claim_type,
        entity_key=entity_key,
        claim_text=claim_text,
        extraction_kind="literal",
        specificity=specificity,
        confidence=confidence,
        observed_at=gene.observed_at,
    )


def persist_claims(
    conn,
    claims: Iterable[Claim],
) -> int:
    """Write claims to ``main.db`` via ``shard_schema.upsert_claim``.

    Returns the number of rows upserted. Takes a sqlite connection, not
    a Genome — claims live in main.db, not the per-shard content dbs.
    """
    from .shard_schema import upsert_claim
    n = 0
    for c in claims:
        upsert_claim(
            conn,
            claim_id=c.claim_id,
            gene_id=c.gene_id,
            shard_name=c.shard_name,
            claim_type=c.claim_type,
            claim_text=c.claim_text,
            entity_key=c.entity_key,
            extraction_kind=c.extraction_kind,
            specificity=c.specificity,
            confidence=c.confidence,
            observed_at=c.observed_at,
            supersedes_claim_id=c.supersedes_claim_id,
        )
        n += 1
    return n

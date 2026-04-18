"""Agent-safe context packet builder.

Additive surface for the Helix index work: it reuses existing promoter
retrieval, then labels results by freshness/authority so callers can
decide whether to trust, reread, or refresh before acting.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import PurePath
from typing import Optional

from .accel import extract_query_signals
from .genome import path_tokens
from .schemas import ContextItem, ContextPacket, Gene, RefreshTarget

_HALF_LIFE_SECONDS = {
    "stable": 7 * 24 * 60 * 60,
    "medium": 12 * 60 * 60,
    "hot": 15 * 60,
}

_AUTHORITY_WEIGHTS = {
    "primary": 1.0,
    "derived": 0.75,
    "inferred": 0.45,
}

_TASK_RISK = {
    "plan": 0.30,
    "explain": 0.45,
    "review": 0.60,
    "edit": 0.85,
    "debug": 0.90,
    "ops": 1.00,
    "quote": 0.95,
}

_HIGH_RISK_TASKS = {"edit", "debug", "ops", "quote"}
_LITERAL_SOURCE_KINDS = {"code", "config", "db", "benchmark", "tool_output"}


def _row_value(row: sqlite3.Row | None, key: str):
    if row is None:
        return None
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _lookup_source_row(
    main_conn: sqlite3.Connection | None,
    gene_id: str,
) -> sqlite3.Row | None:
    if main_conn is None:
        return None
    try:
        return main_conn.execute(
            "SELECT * FROM source_index WHERE gene_id = ?",
            (gene_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None


def _effective_meta(gene: Gene, source_row: sqlite3.Row | None) -> dict:
    observed_at = _row_value(source_row, "observed_at")
    if observed_at is None:
        observed_at = gene.observed_at
    if observed_at is None and gene.epigenetics is not None:
        observed_at = gene.epigenetics.created_at

    last_verified_at = _row_value(source_row, "last_verified_at")
    if last_verified_at is None:
        last_verified_at = gene.last_verified_at
    if last_verified_at is None:
        last_verified_at = observed_at

    return {
        "source_id": _row_value(source_row, "source_id") or gene.source_id,
        "repo_root": _row_value(source_row, "repo_root") or gene.repo_root,
        "source_kind": _row_value(source_row, "source_kind") or gene.source_kind,
        "observed_at": observed_at,
        "mtime": _row_value(source_row, "mtime") or gene.mtime,
        "content_hash": _row_value(source_row, "content_hash") or gene.content_hash,
        "volatility_class": (
            _row_value(source_row, "volatility_class")
            or gene.volatility_class
            or "medium"
        ),
        "authority_class": (
            _row_value(source_row, "authority_class")
            or gene.authority_class
            or "primary"
        ),
        "support_span": _row_value(source_row, "support_span") or gene.support_span,
        "last_verified_at": last_verified_at,
        "invalidated_at": _row_value(source_row, "invalidated_at"),
    }


def _freshness_score(last_verified_at: float | None, volatility_class: str, now_ts: float) -> float:
    if last_verified_at is None:
        return 0.0
    half_life = _HALF_LIFE_SECONDS.get(volatility_class or "medium", _HALF_LIFE_SECONDS["medium"])
    age_seconds = max(0.0, now_ts - float(last_verified_at))
    return math.exp(-age_seconds / max(half_life, 1.0))


def _authority_score(authority_class: str | None) -> float:
    return _AUTHORITY_WEIGHTS.get(authority_class or "primary", 0.75)


def _specificity_score(meta: dict) -> float:
    source_kind = meta.get("source_kind")
    source_id = meta.get("source_id")
    support_span = meta.get("support_span")

    if source_kind in _LITERAL_SOURCE_KINDS and source_id:
        return 1.0
    if support_span and source_id:
        return 0.9
    if source_kind in {"doc", "log", "session_note"} and source_id:
        return 0.75
    if source_kind == "user_assertion":
        return 0.45
    return 0.60


def _status_for(
    *,
    task_type: str,
    freshness_score: float,
    authority_score: float,
    invalidated_at: float | None,
    freshness_known: bool,
) -> str:
    if invalidated_at is not None:
        return "needs_refresh"

    if not freshness_known:
        return "needs_refresh" if task_type in _HIGH_RISK_TASKS else "stale_risk"

    if authority_score < 0.55:
        return "needs_refresh" if task_type in _HIGH_RISK_TASKS else "stale_risk"

    if task_type in _HIGH_RISK_TASKS:
        if freshness_score >= 0.70:
            return "verified"
        if freshness_score >= 0.35:
            return "stale_risk"
        return "needs_refresh"

    if task_type == "review":
        if freshness_score >= 0.55:
            return "verified"
        if freshness_score >= 0.20:
            return "stale_risk"
        return "needs_refresh"

    if freshness_score >= 0.35:
        return "verified"
    if freshness_score >= 0.12:
        return "stale_risk"
    return "needs_refresh"


def _action_risk_score(task_type: str, freshness_score: float, source_kind: str | None) -> float:
    base = _TASK_RISK.get(task_type, 0.50)
    exactness_penalty = 0.0
    if task_type in _HIGH_RISK_TASKS and source_kind not in _LITERAL_SOURCE_KINDS:
        exactness_penalty = 0.15
    return min(1.0, base * (1.0 - freshness_score) + exactness_penalty)


def _item_title(gene: Gene, meta: dict) -> str:
    source_id = meta.get("source_id")
    if source_id:
        name = PurePath(source_id).name
        if name:
            return name
        return source_id
    if gene.promoter.summary:
        return gene.promoter.summary[:80]
    return gene.gene_id


def _item_content(gene: Gene) -> str:
    text = (gene.complement or gene.content or "").strip()
    if len(text) <= 280:
        return text
    return text[:277] + "..."


def _item_citations(gene: Gene, meta: dict) -> list[str]:
    citations = [f"gene:{gene.gene_id}"]
    if meta.get("source_id"):
        citations.insert(0, str(meta["source_id"]))
    return citations


def _coordinate_confidence(query: str, genes: list[Gene]) -> float:
    """Path-token overlap between query and delivered gene source paths.

    Step 1b-iter2 signal (2026-04-18): measures whether retrieval landed
    in the coordinate region the query names, independent of content
    freshness. Folder-grain, not file-grain. Hit mean 1.00 vs miss mean
    0.52 on the 10-needle bench.
    """
    if not genes:
        return 0.0
    domains, entities = extract_query_signals(query)
    q_set = {t.lower() for t in (domains + entities) if t}
    if not q_set:
        return 0.0
    hits = 0
    for g in genes:
        sid = getattr(g, "source_id", None)
        if sid and (path_tokens(sid) & q_set):
            hits += 1
    return hits / len(genes)


_COORDINATE_CONFIDENCE_FLOOR = 0.30


def _apply_coordinate_confidence(
    status: str, task_type: str, coordinate_confidence: float,
) -> str:
    """Downgrade status when coordinate confidence is below the floor.

    Freshness answers "is what we resolved to trustworthy?" — but if the
    resolution itself landed in the wrong region, freshness is a category
    error. Low coordinate confidence forces a refresh cue regardless of
    how fresh the delivered content is.
    """
    if coordinate_confidence >= _COORDINATE_CONFIDENCE_FLOOR:
        return status
    if task_type in _HIGH_RISK_TASKS:
        return "needs_refresh"
    if status == "verified":
        return "stale_risk"
    return status


def _build_item(
    gene: Gene,
    *,
    relevance_score: float,
    meta: dict,
    task_type: str,
    now_ts: float,
    coordinate_confidence: float = 1.0,
) -> tuple[ContextItem, str]:
    freshness_known = meta.get("last_verified_at") is not None
    freshness_score = _freshness_score(
        meta.get("last_verified_at"),
        meta.get("volatility_class") or "medium",
        now_ts,
    )
    authority_score = _authority_score(meta.get("authority_class"))
    specificity_score = _specificity_score(meta)
    live_truth_score = freshness_score * authority_score * specificity_score

    if meta.get("invalidated_at") is not None:
        live_truth_score *= 0.25

    status = _status_for(
        task_type=task_type,
        freshness_score=freshness_score,
        authority_score=authority_score,
        invalidated_at=meta.get("invalidated_at"),
        freshness_known=freshness_known,
    )
    status = _apply_coordinate_confidence(status, task_type, coordinate_confidence)

    item = ContextItem(
        kind="gene",
        gene_id=gene.gene_id,
        title=_item_title(gene, meta),
        content=_item_content(gene),
        relevance_score=float(relevance_score),
        live_truth_score=float(live_truth_score),
        source_id=meta.get("source_id"),
        source_kind=meta.get("source_kind"),
        volatility_class=meta.get("volatility_class"),
        authority_class=meta.get("authority_class"),
        last_verified_at=meta.get("last_verified_at"),
        status=status,
        citations=_item_citations(gene, meta),
    )
    return item, status


def _refresh_target(item: ContextItem, task_type: str) -> RefreshTarget | None:
    if not item.source_id:
        return None
    if item.status == "verified":
        return None
    priority = _action_risk_score(task_type, item.live_truth_score, item.source_kind)
    reason = "stale_risk"
    if item.status == "needs_refresh":
        reason = "fresh verification required before action"
    elif item.status == "stale_risk":
        reason = "relevant evidence is aging or weakly grounded"
    return RefreshTarget(
        target_kind=item.source_kind or "source",
        source_id=item.source_id,
        reason=reason,
        priority=priority,
    )


def _query_genes(query: str, *, genome=None, router=None, max_genes: int = 8) -> tuple[list[Gene], dict]:
    domains, entities = extract_query_signals(query)
    if not domains and not entities and query.strip():
        fallback = query.strip().lower()
        if len(fallback) > 2:
            domains = [fallback]

    if router is not None:
        genes = router.query_genes(domains=domains, entities=entities, max_genes=max_genes)
        score_map = dict(getattr(router, "last_query_scores", {}))
        return genes, score_map

    if genome is not None:
        genes = genome.query_genes(domains=domains, entities=entities, max_genes=max_genes)
        score_map = dict(getattr(genome, "last_query_scores", {}))
        return genes, score_map

    raise ValueError("build_context_packet requires a genome or router")


def build_context_packet(
    query: str,
    *,
    task_type: str = "explain",
    genome=None,
    router=None,
    main_conn: sqlite3.Connection | None = None,
    max_genes: int = 8,
    now_ts: float | None = None,
) -> ContextPacket:
    """Return a freshness-labeled packet for the given query."""
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    effective_main_conn = main_conn
    if effective_main_conn is None and router is not None:
        effective_main_conn = getattr(router, "main_conn", None)

    now_ts = float(now_ts) if now_ts is not None else 0.0
    if now_ts <= 0.0:
        import time
        now_ts = time.time()

    genes, score_map = _query_genes(query, genome=genome, router=router, max_genes=max_genes)
    packet = ContextPacket(task_type=task_type, query=query)

    if effective_main_conn is None:
        packet.notes.append("source_index unavailable; using gene-local metadata only")

    # Step 1b-iter2: coordinate_confidence downgrades status when
    # retrieval lands outside the coordinate region the query names.
    coordinate_confidence = _coordinate_confidence(query, genes)
    if coordinate_confidence < _COORDINATE_CONFIDENCE_FLOOR:
        packet.notes.append(
            f"coordinate_confidence={coordinate_confidence:.2f} below "
            f"{_COORDINATE_CONFIDENCE_FLOOR:.2f} floor — retrieval may not "
            "have located the right coordinate region"
        )

    for gene in genes:
        source_row = _lookup_source_row(effective_main_conn, gene.gene_id)
        meta = _effective_meta(gene, source_row)
        item, status = _build_item(
            gene,
            relevance_score=score_map.get(gene.gene_id, 0.0),
            meta=meta,
            task_type=task_type,
            now_ts=now_ts,
            coordinate_confidence=coordinate_confidence,
        )
        if status == "verified":
            packet.verified.append(item)
        elif status == "stale_risk":
            packet.stale_risk.append(item)
        else:
            packet.stale_risk.append(item)

        target = _refresh_target(item, task_type)
        if target is not None:
            packet.refresh_targets.append(target)

    packet.verified.sort(
        key=lambda item: (item.live_truth_score, item.relevance_score),
        reverse=True,
    )
    packet.stale_risk.sort(
        key=lambda item: (item.status == "needs_refresh", item.live_truth_score, item.relevance_score),
        reverse=True,
    )
    packet.refresh_targets.sort(key=lambda target: target.priority, reverse=True)
    return packet


def get_refresh_targets(
    query: str,
    *,
    task_type: str = "edit",
    genome=None,
    router=None,
    main_conn: sqlite3.Connection | None = None,
    max_genes: int = 8,
    now_ts: float | None = None,
) -> list[RefreshTarget]:
    """Convenience helper for just the reread plan."""
    packet = build_context_packet(
        query,
        task_type=task_type,
        genome=genome,
        router=router,
        main_conn=main_conn,
        max_genes=max_genes,
        now_ts=now_ts,
    )
    return packet.refresh_targets

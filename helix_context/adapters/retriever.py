"""External retriever adapter — wrap your existing RAG (LlamaIndex,
LangChain, custom) behind a uniform ``Retriever`` protocol so Helix's
shortlist-narrowing pattern (pattern 2 in the integration doc) works
with any backend.

Design:

- ``Retriever`` is a duck-typed protocol (``retrieve(query, filter_paths,
  top_k) -> list[RetrievedDoc]``), not an abstract base. Anything that
  matches the signature is a retriever — you don't need to inherit.
- Two reference wrappers: ``LlamaIndexRetriever``, ``LangChainRetriever``.
  Lazy imports so neither framework is a hard dep.
- ``HelixNarrowedRetriever`` composes Helix's ``/context/packet`` with
  your retriever: packet returns source_ids, your retriever searches
  *only within those source_ids*, you get a scoped top-K back.

Typical integration::

    from helix_context.adapters.retriever import (
        LlamaIndexRetriever, HelixNarrowedRetriever,
    )

    my_retriever = LlamaIndexRetriever(llama_index_retriever)
    narrowed = HelixNarrowedRetriever(
        my_retriever,
        helix_url="http://127.0.0.1:11437",
    )
    docs = narrowed.retrieve("where does auth middleware live", top_k=8)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

log = logging.getLogger("helix.adapters.retriever")


@dataclass
class RetrievedDoc:
    """Normalized retriever output.

    Fields mirror what both LlamaIndex ``NodeWithScore`` and LangChain
    ``Document`` surface, so adapters can target one shape.
    """
    source_id: Optional[str] = None
    content: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Duck-typed protocol for external retrievers.

    Any object with this ``retrieve`` signature is a valid Retriever.
    Use ``isinstance(obj, Retriever)`` for opt-in validation.
    """

    def retrieve(
        self,
        query: str,
        *,
        filter_paths: Optional[Iterable[str]] = None,
        top_k: int = 8,
    ) -> list[RetrievedDoc]: ...


# ── Reference wrappers ──────────────────────────────────────────────


class LlamaIndexRetriever:
    """Wrap a ``llama_index.core.retrievers.BaseRetriever``.

    LlamaIndex returns ``NodeWithScore`` — we unpack ``node.text`` +
    ``node.metadata["file_path"]`` + the score field.

    If ``filter_paths`` is given, we filter the LlamaIndex results
    post-retrieval rather than relying on retriever-specific filters
    (every LlamaIndex retriever supports its own filter mechanism
    differently). Users with a tighter integration should pass
    ``filter_paths`` through their retriever's native API.
    """

    def __init__(self, base_retriever: Any,
                 source_id_key: str = "file_path") -> None:
        self._inner = base_retriever
        self._source_id_key = source_id_key

    def retrieve(
        self,
        query: str,
        *,
        filter_paths: Optional[Iterable[str]] = None,
        top_k: int = 8,
    ) -> list[RetrievedDoc]:
        try:
            nodes = self._inner.retrieve(query)
        except Exception as exc:
            log.warning("LlamaIndex retrieve failed: %s", exc)
            return []

        allow = set(filter_paths) if filter_paths else None
        out: list[RetrievedDoc] = []
        for n in nodes:
            node_obj = getattr(n, "node", n)
            meta = getattr(node_obj, "metadata", None) or {}
            sid = meta.get(self._source_id_key) or getattr(node_obj, "id_", None)
            if allow is not None and sid not in allow:
                continue
            out.append(RetrievedDoc(
                source_id=sid,
                content=getattr(node_obj, "text", "") or "",
                score=float(getattr(n, "score", 0.0) or 0.0),
                metadata=dict(meta),
            ))
            if len(out) >= top_k:
                break
        return out


class LangChainRetriever:
    """Wrap a ``langchain_core.retrievers.BaseRetriever``.

    LangChain retrievers return ``Document`` objects. We map:

    - ``doc.page_content`` → ``content``
    - ``doc.metadata.get(source_id_key)`` → ``source_id``
      (defaults to ``"source"`` which is LangChain's conventional key)
    """

    def __init__(self, base_retriever: Any,
                 source_id_key: str = "source") -> None:
        self._inner = base_retriever
        self._source_id_key = source_id_key

    def retrieve(
        self,
        query: str,
        *,
        filter_paths: Optional[Iterable[str]] = None,
        top_k: int = 8,
    ) -> list[RetrievedDoc]:
        try:
            # LangChain 0.2+: .invoke(query); older: .get_relevant_documents(query)
            if hasattr(self._inner, "invoke"):
                docs = self._inner.invoke(query)
            else:
                docs = self._inner.get_relevant_documents(query)
        except Exception as exc:
            log.warning("LangChain retrieve failed: %s", exc)
            return []

        allow = set(filter_paths) if filter_paths else None
        out: list[RetrievedDoc] = []
        for d in docs:
            meta = dict(getattr(d, "metadata", {}) or {})
            sid = meta.get(self._source_id_key)
            if allow is not None and sid not in allow:
                continue
            out.append(RetrievedDoc(
                source_id=sid,
                content=getattr(d, "page_content", "") or "",
                score=float(meta.get("score", 0.0) or 0.0),
                metadata=meta,
            ))
            if len(out) >= top_k:
                break
        return out


# ── Helix-narrowed composition ──────────────────────────────────────


class HelixNarrowedRetriever:
    """Composes Helix's packet-shortlist with an underlying Retriever.

    Flow:
        1. Query Helix's ``/context/packet`` → extract source_ids
           (verified + stale_risk + contradictions + refresh_targets).
        2. Pass the shortlist to the underlying retriever as
           ``filter_paths``.
        3. If the shortlist is empty OR the retriever returned nothing,
           fall back to an unscoped retrieve so we never starve the agent.

    The ``filter_paths`` parameter is still honored if the caller
    supplies their own filter — it intersects with the Helix shortlist.
    """

    def __init__(
        self,
        inner: Retriever,
        helix_url: str = "http://127.0.0.1:11437",
        *,
        task_type: str = "explain",
        fallback_unscoped: bool = True,
        read_only: bool = False,
    ) -> None:
        self._inner = inner
        self._helix_url = helix_url.rstrip("/")
        self._task_type = task_type
        self._fallback_unscoped = fallback_unscoped
        self._read_only = read_only

    def _get_packet(self, query: str) -> dict:
        try:
            import httpx
        except ImportError:
            log.warning("httpx not installed; bypassing Helix narrowing")
            return {}
        try:
            resp = httpx.post(
                f"{self._helix_url}/context/packet",
                json={
                    "query": query,
                    "task_type": self._task_type,
                    "read_only": self._read_only,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("Helix packet fetch failed: %s", exc)
            return {}

    def _shortlist(self, packet: dict) -> set[str]:
        sids: set[str] = set()
        for bucket in ("verified", "stale_risk", "contradictions"):
            for item in packet.get(bucket, []) or []:
                sid = item.get("source_id")
                if sid:
                    sids.add(sid)
        for tgt in packet.get("refresh_targets", []) or []:
            sid = tgt.get("source_id")
            if sid:
                sids.add(sid)
        return sids

    def retrieve(
        self,
        query: str,
        *,
        filter_paths: Optional[Iterable[str]] = None,
        top_k: int = 8,
    ) -> list[RetrievedDoc]:
        packet = self._get_packet(query)
        helix_set = self._shortlist(packet)
        caller_set = set(filter_paths) if filter_paths else None

        if helix_set and caller_set:
            effective = helix_set & caller_set
        else:
            effective = helix_set or caller_set

        if effective:
            docs = self._inner.retrieve(
                query, filter_paths=effective, top_k=top_k,
            )
        else:
            docs = []

        # Fallback: nothing back from scoped retrieve → run unscoped
        if not docs and self._fallback_unscoped:
            log.debug("Helix-scoped retrieve returned 0; falling back unscoped")
            docs = self._inner.retrieve(query, top_k=top_k)

        return docs

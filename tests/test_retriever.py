"""Tests for retriever adapters (helix_context.adapters.retriever)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from helix_context.adapters.retriever import (
    HelixNarrowedRetriever,
    LangChainRetriever,
    LlamaIndexRetriever,
    Retriever,
    RetrievedDoc,
)


# ── Protocol ────────────────────────────────────────────────────────


def test_retriever_protocol_duck_types():
    """Any object with the right signature satisfies Retriever."""
    class _DuckRetriever:
        def retrieve(self, query, *, filter_paths=None, top_k=8):
            return []

    obj = _DuckRetriever()
    assert isinstance(obj, Retriever)


def test_retriever_protocol_rejects_non_matching():
    class _NoRetrieve:
        pass

    assert not isinstance(_NoRetrieve(), Retriever)


# ── RetrievedDoc ─────────────────────────────────────────────────────


def test_retrieved_doc_defaults():
    d = RetrievedDoc()
    assert d.source_id is None
    assert d.content == ""
    assert d.score == 0.0
    assert d.metadata == {}


# ── LlamaIndexRetriever ─────────────────────────────────────────────


def _fake_llama_node(text: str, file_path: str, score: float = 1.0):
    """Shape-compatible stub for NodeWithScore."""
    inner = MagicMock()
    inner.text = text
    inner.metadata = {"file_path": file_path}
    inner.id_ = file_path
    outer = MagicMock()
    outer.node = inner
    outer.score = score
    return outer


def test_llamaindex_retriever_unpacks_nodes():
    fake_base = MagicMock()
    fake_base.retrieve.return_value = [
        _fake_llama_node("content A", "/a.py", 0.9),
        _fake_llama_node("content B", "/b.py", 0.7),
    ]
    wrapper = LlamaIndexRetriever(fake_base)
    docs = wrapper.retrieve("q")
    assert len(docs) == 2
    assert docs[0].source_id == "/a.py"
    assert docs[0].content == "content A"
    assert docs[0].score == 0.9


def test_llamaindex_retriever_filters_by_paths():
    fake_base = MagicMock()
    fake_base.retrieve.return_value = [
        _fake_llama_node("A", "/a.py"),
        _fake_llama_node("B", "/b.py"),
        _fake_llama_node("C", "/c.py"),
    ]
    wrapper = LlamaIndexRetriever(fake_base)
    docs = wrapper.retrieve("q", filter_paths=["/a.py", "/c.py"])
    assert {d.source_id for d in docs} == {"/a.py", "/c.py"}


def test_llamaindex_retriever_honors_top_k():
    fake_base = MagicMock()
    fake_base.retrieve.return_value = [
        _fake_llama_node(f"x{i}", f"/f{i}.py") for i in range(10)
    ]
    wrapper = LlamaIndexRetriever(fake_base)
    docs = wrapper.retrieve("q", top_k=3)
    assert len(docs) == 3


def test_llamaindex_retriever_softfails_on_exception():
    fake_base = MagicMock()
    fake_base.retrieve.side_effect = Exception("kaboom")
    wrapper = LlamaIndexRetriever(fake_base)
    assert wrapper.retrieve("q") == []


def test_llamaindex_retriever_custom_source_id_key():
    inner = MagicMock()
    inner.text = "body"
    inner.metadata = {"url": "https://x.com"}
    outer = MagicMock()
    outer.node = inner
    outer.score = 1.0
    fake_base = MagicMock()
    fake_base.retrieve.return_value = [outer]
    wrapper = LlamaIndexRetriever(fake_base, source_id_key="url")
    docs = wrapper.retrieve("q")
    assert docs[0].source_id == "https://x.com"


# ── LangChainRetriever ───────────────────────────────────────────────


def _fake_lc_doc(content: str, source: str, score: float = 0.0):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "score": score}
    return doc


def test_langchain_retriever_prefers_invoke():
    fake_base = MagicMock()
    fake_base.invoke.return_value = [
        _fake_lc_doc("A", "/a.py", 0.9),
    ]
    wrapper = LangChainRetriever(fake_base)
    docs = wrapper.retrieve("q")
    assert len(docs) == 1
    assert docs[0].source_id == "/a.py"
    assert docs[0].score == 0.9


def test_langchain_retriever_falls_back_to_get_relevant():
    """Older LangChain — no .invoke, has .get_relevant_documents"""
    fake_base = MagicMock(spec=["get_relevant_documents"])
    fake_base.get_relevant_documents.return_value = [
        _fake_lc_doc("A", "/a.py"),
    ]
    wrapper = LangChainRetriever(fake_base)
    docs = wrapper.retrieve("q")
    assert len(docs) == 1


def test_langchain_retriever_filters_by_paths():
    fake_base = MagicMock()
    fake_base.invoke.return_value = [
        _fake_lc_doc("A", "/a.py"),
        _fake_lc_doc("B", "/b.py"),
    ]
    wrapper = LangChainRetriever(fake_base)
    docs = wrapper.retrieve("q", filter_paths=["/b.py"])
    assert [d.source_id for d in docs] == ["/b.py"]


def test_langchain_retriever_softfails_on_exception():
    fake_base = MagicMock()
    fake_base.invoke.side_effect = Exception("net")
    wrapper = LangChainRetriever(fake_base)
    assert wrapper.retrieve("q") == []


# ── HelixNarrowedRetriever ──────────────────────────────────────────


def _mock_packet(source_ids):
    return {
        "verified": [{"source_id": s} for s in source_ids],
        "stale_risk": [],
        "contradictions": [],
        "refresh_targets": [],
    }


def test_narrowed_passes_helix_shortlist_to_inner():
    inner = MagicMock(spec=Retriever)
    inner.retrieve.return_value = [RetrievedDoc(source_id="/a.py")]
    narrowed = HelixNarrowedRetriever(inner)
    fake_resp = MagicMock()
    fake_resp.json.return_value = _mock_packet(["/a.py", "/b.py"])
    fake_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=fake_resp):
        docs = narrowed.retrieve("q")
    inner.retrieve.assert_called_once()
    call_kwargs = inner.retrieve.call_args.kwargs
    assert call_kwargs["filter_paths"] == {"/a.py", "/b.py"}
    assert docs[0].source_id == "/a.py"


def test_narrowed_intersects_caller_filter_with_helix_shortlist():
    inner = MagicMock(spec=Retriever)
    inner.retrieve.return_value = []
    narrowed = HelixNarrowedRetriever(inner, fallback_unscoped=False)
    fake_resp = MagicMock()
    fake_resp.json.return_value = _mock_packet(["/a.py", "/b.py", "/c.py"])
    fake_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=fake_resp):
        narrowed.retrieve("q", filter_paths=["/b.py", "/d.py"])
    call_kwargs = inner.retrieve.call_args.kwargs
    # Only /b.py is in both sets
    assert call_kwargs["filter_paths"] == {"/b.py"}


def test_narrowed_falls_back_unscoped_when_helix_empty():
    inner = MagicMock(spec=Retriever)
    inner.retrieve.return_value = [RetrievedDoc(source_id="/x.py")]
    narrowed = HelixNarrowedRetriever(inner, fallback_unscoped=True)
    fake_resp = MagicMock()
    fake_resp.json.return_value = _mock_packet([])
    fake_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=fake_resp):
        docs = narrowed.retrieve("q")
    # Inner called once, without filter_paths
    inner.retrieve.assert_called_once()
    assert "filter_paths" not in inner.retrieve.call_args.kwargs or \
        inner.retrieve.call_args.kwargs.get("filter_paths") in (None, set())
    assert docs[0].source_id == "/x.py"


def test_narrowed_survives_helix_unreachable():
    """Helix down → fall back unscoped to the inner retriever."""
    inner = MagicMock(spec=Retriever)
    inner.retrieve.return_value = [RetrievedDoc(source_id="/x.py")]
    narrowed = HelixNarrowedRetriever(inner, fallback_unscoped=True)
    with patch("httpx.post", side_effect=Exception("connection refused")):
        docs = narrowed.retrieve("q")
    assert docs[0].source_id == "/x.py"


def test_narrowed_no_fallback_returns_empty_when_helix_empty():
    inner = MagicMock(spec=Retriever)
    inner.retrieve.return_value = [RetrievedDoc(source_id="/x.py")]
    narrowed = HelixNarrowedRetriever(inner, fallback_unscoped=False)
    fake_resp = MagicMock()
    fake_resp.json.return_value = _mock_packet([])
    fake_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=fake_resp):
        docs = narrowed.retrieve("q")
    assert docs == []


def test_narrowed_forwards_read_only_flag_to_packet_request():
    inner = MagicMock(spec=Retriever)
    inner.retrieve.return_value = []
    narrowed = HelixNarrowedRetriever(inner, read_only=True, fallback_unscoped=False)
    fake_resp = MagicMock()
    fake_resp.json.return_value = _mock_packet([])
    fake_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=fake_resp) as post:
        narrowed.retrieve("q")
    assert post.call_args.kwargs["json"]["read_only"] is True

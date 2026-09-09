"""SPLADE precompute plumbing - issue #92, Phase 1.

Verifies that callers can pass a precomputed SPLADE sparse vector to
``sync_splade_index`` and ``upsert_doc`` instead of letting them call
``splade_backend.encode`` inline. Used by the parallel/shard-pool ingest
paths to batch SPLADE encoding outside the per-document upsert.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cymatix_context.backends import splade_backend
from cymatix_context.storage.indexes import sync_splade_index


@pytest.fixture
def splade_loader(monkeypatch):
    """Exercise the real lazy loader without importing or downloading ML weights."""
    from cymatix_context import hardware

    loader = SimpleNamespace(device=object(), tokenizer=object(), model=Mock())
    loader.make_device = Mock(return_value=loader.device)
    loader.load_tokenizer = Mock(return_value=loader.tokenizer)
    loader.load_model = Mock(return_value=loader.model)
    loader.model.to.return_value = loader.model
    loader.model.eval.return_value = loader.model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=loader.make_device))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=loader.load_tokenizer),
        AutoModelForMaskedLM=SimpleNamespace(from_pretrained=loader.load_model),
    ))
    monkeypatch.setattr(hardware, "resolve_layer_device", lambda layer: "cpu")
    for name in ("_model", "_tokenizer", "_device"):
        monkeypatch.setattr(splade_backend, name, None)
    return loader


@pytest.mark.concurrency
def test_ensure_loaded_concurrent_first_use_loads_once(monkeypatch, splade_loader):
    """A waiting first-use caller reuses the completed model and tokenizer."""
    first_loading = threading.Event()
    contender_entered = threading.Event()
    real_lock = getattr(splade_backend, "_load_lock", threading.Lock())

    class ObservedLock:
        def __enter__(self):
            if first_loading.is_set():
                contender_entered.set()
            return real_lock.__enter__()

        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    # Observe a contended acquisition without sleeps or scheduler assumptions.
    monkeypatch.setattr(splade_backend, "_load_lock", ObservedLock(), raising=False)

    def load_tokenizer(model_name):
        if first_loading.is_set():
            # The unlocked implementation enters the loader a second time.
            contender_entered.set()
        else:
            first_loading.set()
            assert contender_entered.wait(5), "second caller never entered"
        return splade_loader.tokenizer

    splade_loader.load_tokenizer.side_effect = load_tokenizer

    def load():
        splade_backend._ensure_loaded("test/splade")
        return splade_backend._model, splade_backend._tokenizer, splade_backend._device

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(load)
        assert first_loading.wait(5), "first caller never entered the loader"
        second = pool.submit(load)
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert results == [(splade_loader.model, splade_loader.tokenizer, splade_loader.device)] * 2
    splade_loader.load_tokenizer.assert_called_once_with("test/splade")
    splade_loader.load_model.assert_called_once_with("test/splade")
    splade_loader.model.to.assert_called_once_with(splade_loader.device)
    splade_loader.model.eval.assert_called_once_with()


def test_ensure_loaded_publishes_only_after_eval(splade_loader):
    """No cached initialization state is visible before eval succeeds."""
    def check_unpublished():
        assert splade_backend._model is None
        assert splade_backend._tokenizer is None
        assert splade_backend._device is None

    splade_loader.model.eval.side_effect = check_unpublished
    splade_backend._ensure_loaded("test/splade")

    assert splade_backend._model is splade_loader.model
    assert splade_backend._tokenizer is splade_loader.tokenizer
    assert splade_backend._device is splade_loader.device
    # A warm call does not touch the model or repeat evaluation.
    splade_backend._ensure_loaded("test/splade")
    splade_loader.model.eval.assert_called_once_with()


@pytest.mark.parametrize("stage", ["device", "tokenizer", "model", "to", "eval"])
def test_ensure_loaded_failure_leaves_cache_empty_and_can_retry(splade_loader, stage):
    operations = {
        "device": splade_loader.make_device,
        "tokenizer": splade_loader.load_tokenizer,
        "model": splade_loader.load_model,
        "to": splade_loader.model.to,
        "eval": splade_loader.model.eval,
    }
    operation = operations[stage]
    operation.side_effect = RuntimeError("initialization failed")

    with pytest.raises(RuntimeError, match="initialization failed"):
        splade_backend._ensure_loaded("test/splade")

    assert splade_backend._model is None
    assert splade_backend._tokenizer is None
    assert splade_backend._device is None

    operation.side_effect = None
    splade_backend._ensure_loaded("test/splade")
    assert splade_backend._model is splade_loader.model
    assert splade_backend._tokenizer is splade_loader.tokenizer
    assert splade_backend._device is splade_loader.device
    assert operation.call_count == 2


def _fresh_splade_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    splade_backend.create_splade_table(conn)
    return conn


def test_sync_splade_index_uses_provided_sparse():
    """When splade_sparse is provided, no inline encode happens."""
    conn = _fresh_splade_db()
    provided = {"alpha": 1.5, "beta": 0.75}

    sync_splade_index(
        conn.cursor(),
        gene_id="g1",
        content="this content should be ignored",
        splade_enabled=True,
        splade_sparse=provided,
    )
    conn.commit()

    rows = conn.execute(
        "SELECT term, weight FROM splade_terms WHERE gene_id = ? ORDER BY term",
        ("g1",),
    ).fetchall()
    assert rows == [("alpha", 1.5), ("beta", 0.75)]


def test_sync_splade_index_disabled_is_noop_even_with_sparse():
    conn = _fresh_splade_db()
    sync_splade_index(
        conn.cursor(),
        gene_id="g1",
        content="x",
        splade_enabled=False,
        splade_sparse={"alpha": 1.0},
    )
    conn.commit()
    rows = conn.execute("SELECT COUNT(*) FROM splade_terms").fetchone()
    assert rows[0] == 0


def test_sync_splade_index_empty_sparse_dict_clears_existing_rows():
    """Pre-existing rows for gene_id get DELETE'd even when sparse is empty."""
    conn = _fresh_splade_db()
    conn.execute(
        "INSERT INTO splade_terms (gene_id, term, weight) VALUES (?, ?, ?)",
        ("g1", "stale", 1.0),
    )
    conn.commit()

    sync_splade_index(
        conn.cursor(),
        gene_id="g1",
        content="x",
        splade_enabled=True,
        splade_sparse={},
    )
    conn.commit()

    rows = conn.execute(
        "SELECT COUNT(*) FROM splade_terms WHERE gene_id = ?", ("g1",)
    ).fetchone()
    assert rows[0] == 0


from cymatix_context.knowledge_store import KnowledgeStore
from cymatix_context.schemas import Gene


def _make_test_gene(content: str = "hello world parallel ingest") -> Gene:
    return Gene(
        gene_id=KnowledgeStore.make_gene_id(content),
        content=content,
        complement=f"Summary: {content[:40]}",
        codons=["chunk_0"],
        source_id="test://splade-precompute",
    )


def test_upsert_doc_forwards_splade_sparse(tmp_path):
    """Pre-computed SPLADE sparse dict ends up in the splade_terms table."""
    db = tmp_path / "g.db"
    ks = KnowledgeStore(path=str(db), synonym_map={}, splade_enabled=True)
    gene = _make_test_gene()

    provided = {"semantic": 2.5, "expansion": 1.1}
    gene_id = ks.upsert_doc(gene, apply_gate=False, splade_sparse=provided)

    rows = [
        (r["term"], r["weight"])
        for r in ks.conn.execute(
            "SELECT term, weight FROM splade_terms WHERE gene_id = ? ORDER BY term",
            (gene_id,),
        )
    ]
    ks.close()

    assert sorted(rows) == [("expansion", 1.1), ("semantic", 2.5)]


def test_upsert_doc_inline_encode_when_sparse_not_provided(tmp_path, monkeypatch):
    """No splade_sparse -> falls back to splade_backend.encode."""
    db = tmp_path / "g.db"
    ks = KnowledgeStore(path=str(db), synonym_map={}, splade_enabled=True)

    sentinel = {"sentinel": 9.99}
    calls: list[str] = []

    def fake_encode(text, top_k=128, **kw):
        calls.append(text)
        return sentinel

    monkeypatch.setattr(splade_backend, "encode", fake_encode)
    gene_id = ks.upsert_doc(_make_test_gene(), apply_gate=False)

    rows = [
        (r["term"], r["weight"])
        for r in ks.conn.execute(
            "SELECT term, weight FROM splade_terms WHERE gene_id = ?", (gene_id,)
        )
    ]
    ks.close()

    assert calls, "splade_backend.encode should have been called once"
    assert rows == [("sentinel", 9.99)]

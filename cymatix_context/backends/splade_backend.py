"""
SPLADE Backend — Learned sparse expansion for the knowledge store.

Bio analogue (legacy term: epigenetics):
    SPLADE is like an epigenetic mark that makes hidden genes visible.
    Where BM25 only sees exact words, SPLADE expands each chunk with
    semantically related terms at index time — making documents findable
    by meaning, not just surface text.

Implementation:
    Uses naver/splade-cocondenser-ensembledistil (or compatible) to
    produce sparse vocabulary-space weight vectors. These get stored
    in a splade_terms table (inverted index in SQLite) and queried
    via dot-product scoring at retrieval time.

Performance:
    ~5ms per chunk on CPU (single forward pass, no autoregressive generation).
    ~60s for 3,500 chunks (batch mode).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Dict, List, Optional, Tuple

from . import encoder_client

log = logging.getLogger("cymatix.splade")

# Lazy-loaded model + tokenizer
_model = None
_tokenizer = None
_device = None
_load_lock = threading.Lock()

_SPECIAL_TOKENS = frozenset({"[CLS]", "[SEP]", "[PAD]", "[UNK]"})

# Encoder-daemon transport (Fork 1 slice 1), as a single ``(url, client)``
# tuple so the rebind on a URL change is atomic. This is lock-free module
# state, and a benign race just rebuilds a cheap stdlib
# client. ``None`` until the first remote encode; stays ``None`` forever when
# no daemon URL is configured.
_remote_client = None


def _remote_sparse(
    texts: List[str], top_k: int, timeout_s: Optional[float] = None,
) -> Optional[List[Dict[str, float]]]:
    """Daemon-encoded sparse vectors, or ``None`` meaning "run the local body".

    Deliberately reached BEFORE the ``import torch`` in ``encode`` /
    ``encode_batch``: with a daemon serving this process, importing
    splade_backend — and encoding with it — must work on a host that has no
    transformers installed at all.

    Any failure (transport, malformed payload, wrong item count) records the
    failure on the shared circuit breaker and returns ``None``, so the caller
    re-encodes in process. Never skip: ``sync_splade_index`` soft-fails to
    ``log.debug``, so a "skip" would silently drop terms from the index.

    ``timeout_s`` (Finding 1): ``encode()`` passes None (flat client
    default); ``encode_batch()`` passes
    ``encoder_client.batch_timeout_s(len(texts))`` — a whole-batch POST under
    the flat 10s default times out client-side on large batches, which dumps
    every worker onto the local fallback and opens the shared circuit.
    """
    url = encoder_client.active_url()
    if not url:
        return None
    if not texts:
        return []
    if not encoder_client.should_attempt():
        return None

    global _remote_client
    cached = _remote_client
    if cached is None or cached[0] != url:
        client = encoder_client.EncoderClient(
            url, timeout_s=encoder_client.default_timeout_s(),
        )
        _remote_client = (url, client)
    else:
        client = cached[1]

    # Shared one-shot readiness gate: SPLADE usually encodes ahead of dense on
    # the ingest path, so skipping it here would let a cold daemon open the
    # process-wide circuit before dense ever probes.
    if not encoder_client.ready_gate(client):
        return None

    try:
        sparse = client.encode_splade(list(texts), top_k=top_k, timeout_s=timeout_s)
    except Exception as exc:  # EncoderClientError + any stray transport error
        encoder_client.record_failure()
        log.debug("remote SPLADE encode failed (%s); encoding in process", exc)
        return None
    if (
        not isinstance(sparse, list)
        or len(sparse) != len(texts)
        or not all(isinstance(d, dict) for d in sparse)
    ):
        encoder_client.record_failure()
        log.debug("remote SPLADE returned an unusable payload; encoding in process")
        return None
    encoder_client.record_success()
    # Weights and descending-weight dict order pass through untouched: the
    # daemon already applied round(w, 4) and the topk ordering, and JSON
    # round-trips both.
    return sparse


def _ensure_loaded(model_name: str = "naver/splade-cocondenser-ensembledistil"):
    """Load SPLADE model on first use. Cached for process lifetime."""
    global _model, _tokenizer, _device

    if _model is not None:
        return

    with _load_lock:
        if _model is not None:
            return

        import time as _time

        # Timer starts BEFORE the torch/transformers imports — on a cold
        # process those imports cost seconds and belong to the load bill.
        _load_t0 = _time.monotonic()

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        from cymatix_context.hardware import resolve_layer_device
        device = torch.device(resolve_layer_device("splade"))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()

        # Publish the ready marker last. Failed initialization leaves the
        # cache untouched, and readers can only observe a fully ready model.
        _device = device
        _tokenizer = tokenizer
        _model = model
        # Perf slice 1 (W0.2): record the lazy weight load so cold-start
        # receipts can subtract it from the express stage.
        try:
            from cymatix_context.telemetry import record_model_load
            record_model_load("splade", _time.monotonic() - _load_t0)
        except Exception:
            pass
        log.info("SPLADE model loaded: %s on %s", model_name, _device)


def encode(text: str, top_k: int = 128, model_name: str = "naver/splade-cocondenser-ensembledistil") -> Dict[str, float]:
    """
    Encode text into a sparse SPLADE vector.

    Returns {token: weight} dict with top_k non-zero entries.
    Each token is from the BERT vocabulary; weight indicates how
    strongly that term is associated with this content.
    """
    remote = _remote_sparse([text], top_k)
    if remote is not None:
        return remote[0]

    import torch

    _ensure_loaded(model_name)

    tokens = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False,
    ).to(_device)

    with torch.no_grad():
        output = _model(**tokens)

    # SPLADE activation: ReLU + log(1 + x) over MLM logits, max-pool over tokens
    logits = output.logits  # (1, seq_len, vocab_size)
    activated = torch.log1p(torch.relu(logits))  # sparse activation
    pooled = activated.max(dim=1).values.squeeze(0)  # (vocab_size,)

    # Extract top-k non-zero entries
    nonzero_count = int((pooled > 0).sum().item())
    if nonzero_count == 0:
        # torch.Tensor.topk(0) raises on some backends; also there's nothing
        # to return if the pooled activation is entirely zero.
        return {}
    top_values, top_indices = pooled.topk(min(top_k, nonzero_count))

    sparse: Dict[str, float] = {}
    for idx, val in zip(top_indices.cpu().tolist(), top_values.cpu().tolist()):
        if val > 0:
            token = _tokenizer.decode([idx]).strip()
            # Skip special tokens and single chars
            if len(token) > 1 and token not in _SPECIAL_TOKENS:
                sparse[token] = round(val, 4)

    return sparse


def encode_batch(
    texts: List[str],
    top_k: int = 128,
    batch_size: Optional[int] = None,
    model_name: str = "naver/splade-cocondenser-ensembledistil",
) -> List[Dict[str, float]]:
    """
    Batch-encode texts into SPLADE sparse vectors.
    More efficient than calling encode() in a loop due to batched forward passes.

    ``batch_size`` defaults to ``recommended_batch_size("splade")`` when unset,
    sized for the detected hardware (VRAM tier on CUDA/ROCm, system RAM on
    MPS/CPU). Pass an explicit value to override — advisory only when an
    encoder daemon serves the request, since the daemon owns micro-batching.
    """
    # Finding 1: scale the timeout with len(texts) for the whole-batch POST
    # — see _remote_sparse's docstring for the rationale.
    remote = _remote_sparse(texts, top_k, timeout_s=encoder_client.batch_timeout_s(len(texts)))
    if remote is not None:
        return remote

    import torch

    _ensure_loaded(model_name)

    if batch_size is None:
        from cymatix_context.hardware import recommended_batch_size
        batch_size = recommended_batch_size("splade")

    results: List[Dict[str, float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = _tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(_device)

        with torch.no_grad():
            output = _model(**tokens)

        logits = output.logits  # (batch, seq_len, vocab_size)
        activated = torch.log1p(torch.relu(logits))
        pooled = activated.max(dim=1).values  # (batch, vocab_size)

        for j in range(pooled.size(0)):
            vec = pooled[j]
            nonzero_count = int((vec > 0).sum().item())
            if nonzero_count == 0:
                results.append({})
                continue
            top_values, top_indices = vec.topk(min(top_k, nonzero_count))

            sparse: Dict[str, float] = {}
            for idx, val in zip(top_indices.cpu().tolist(), top_values.cpu().tolist()):
                if val > 0:
                    token = _tokenizer.decode([idx]).strip()
                    if len(token) > 1 and token not in _SPECIAL_TOKENS:
                        sparse[token] = round(val, 4)
            results.append(sparse)

    return results


# ── SQLite integration helpers ────────────────────────────────────

def create_splade_table(conn) -> None:
    """Create the splade_terms inverted index table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS splade_terms (
            gene_id  TEXT NOT NULL,
            term     TEXT NOT NULL,
            weight   REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_splade_term
        ON splade_terms (term, weight DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_splade_gene
        ON splade_terms (gene_id)
    """)
    # 2026-08-03 ERB step-test fix: idx_splade_term is NOT covering — every
    # posting hit paid a table b-tree hop for gene_id, which at 147M rows /
    # 829k genes measured 43s per query (60-177s in isolation). The covering
    # index serves the same term=? lookups index-only: 3.0-4.5s measured,
    # 111.7s -> 88.9s end-to-end median with zero query changes
    # (docs/benchmarks/2026-08-03-erb-step-curve.md).
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_splade_term_cover
        ON splade_terms (term, gene_id, weight)
    """)
    # Per-term document frequency, maintained incrementally by
    # upsert_splade_terms. Consumed by query_splade's DF cap. An empty
    # table means "no DF data" and the cap stays inactive (legacy
    # behavior) — old beds that predate the counters are unaffected until
    # backfilled: INSERT INTO splade_term_df
    #             SELECT term, COUNT(*) FROM splade_terms GROUP BY term.
    # Gene deletes outside upsert_splade_terms can drift the counters
    # slightly; the cap is a threshold heuristic, so drift is tolerable
    # and a re-backfill resyncs.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS splade_term_df (
            term TEXT PRIMARY KEY,
            df   INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()


def upsert_splade_terms(conn, gene_id: str, sparse: Dict[str, float]) -> None:
    """Store SPLADE sparse vector for a document (replaces existing entries)."""
    old_terms = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT term FROM splade_terms WHERE gene_id = ?",
            (gene_id,),
        )
    ]
    conn.execute("DELETE FROM splade_terms WHERE gene_id = ?", (gene_id,))
    if old_terms:
        conn.executemany(
            "UPDATE splade_term_df SET df = MAX(df - 1, 0) WHERE term = ?",
            [(t,) for t in old_terms],
        )
    if sparse:
        conn.executemany(
            "INSERT INTO splade_terms (gene_id, term, weight) VALUES (?, ?, ?)",
            [(gene_id, term, weight) for term, weight in sparse.items()],
        )
        conn.executemany(
            "INSERT INTO splade_term_df (term, df) VALUES (?, 1) "
            "ON CONFLICT(term) DO UPDATE SET df = df + 1",
            [(term,) for term in sparse],
        )
    conn.commit()


def query_splade(
    conn,
    query_sparse: Dict[str, float],
    limit: int = 50,
    min_score: float = 0.01,
    max_df_fraction: float = 0.0,
    total_docs: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Query the SPLADE inverted index with a sparse query vector.

    Returns [(gene_id, score)] ranked by dot-product similarity.

    ``max_df_fraction`` (2026-08-03 ERB fix, the #327 tag_df_cap pattern
    applied to SPLADE): drop query terms whose document frequency exceeds
    ``max_df_fraction * total_docs`` before the postings join. On the
    829k-gene ERB blob one real 58-term query vector touched 5.06M posting
    rows, dominated by terms present in >70% of documents ('phone' 1.03M,
    'number' 905k) that contribute score mass but no discrimination.
    0.0 (default) disables the cap. DF comes from ``splade_term_df``
    (maintained by ``upsert_splade_terms``); a missing/empty table means no
    DF data and the cap stays inactive. If EVERY query term is over the
    cap, the cap disables itself for that query rather than empty a tier.
    """
    if not query_sparse:
        return []

    if max_df_fraction > 0.0 and total_docs:
        try:
            cap = max_df_fraction * total_docs
            ph = ",".join("?" * len(query_sparse))
            dfs = dict(conn.execute(
                f"SELECT term, df FROM splade_term_df WHERE term IN ({ph})",
                list(query_sparse),
            ).fetchall())
            kept = {
                t: w for t, w in query_sparse.items() if dfs.get(t, 0) <= cap
            }
            if kept:
                query_sparse = kept
        except sqlite3.OperationalError:
            pass  # pre-counter bed (no splade_term_df) — cap inactive

    # Single SQL computing the query-weighted dot product per document:
    # join the inverted index against the query vector (VALUES CTE) so
    # LIMIT and min_score apply to the TRUE score. The previous two-pass
    # version pre-filtered candidates by unweighted SUM(weight) term mass,
    # which could discard a high-weighted-score document in favor of one
    # with more raw mass but near-zero query overlap.
    values_clause = ",".join("(?, ?)" for _ in query_sparse)
    params: List[object] = []
    for term, qw in query_sparse.items():
        params.extend((term, qw))

    rows = conn.execute(
        f"WITH q(term, qw) AS (VALUES {values_clause}) "
        f"SELECT st.gene_id, SUM(st.weight * q.qw) AS score "
        f"FROM splade_terms st JOIN q ON st.term = q.term "
        f"GROUP BY st.gene_id "
        f"HAVING score > ? "
        f"ORDER BY score DESC "
        f"LIMIT ?",
        params + [min_score, limit],
    ).fetchall()

    return [(gene_id, score) for gene_id, score in rows]

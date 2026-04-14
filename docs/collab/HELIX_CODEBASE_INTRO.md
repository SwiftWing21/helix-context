# Helix-Context — Codebase Intro for Celestia Collaboration

> **Audience:** Fauxtrot — builder of Celestia, senior systems/ML.
> **Purpose:** fast orientation so you can read the relevant 10% of the
> repo without trawling the other 90%.
> **Companion to:** [`CELESTIA_JOINT_EXPERIMENT.md`](CELESTIA_JOINT_EXPERIMENT.md)

---

## 1. What helix actually is, in 3 sentences

Helix-context is a local-first retrieval system that compresses
codebases and documents into "genes" (token-compressed chunks with
promoter tags, embeddings, and co-activation links), stored in SQLite
with WAL. A ribosome loop consumes a query, expresses the top-k genes
into a compressed context window, and ships it to a local or remote
LLM. Architecture is explicitly borrowed from molecular biology —
chromatin tiers for hot/cold residency, cymatics for frequency-domain
co-activation, TCM/SR for hippocampal-style temporal context.

It runs as a FastAPI server on `localhost:11437` with a
`ribosome` dev persona for CLI, and a launcher on `:11438`.

---

## 2. Where things live

```
helix-context/
├── helix_context/              ← main package (what you care about)
│   ├── server.py               ← FastAPI app, routes, lifecycle
│   ├── ribosome.py             ← the top-level retrieval loop
│   ├── context_manager.py      ← query → genes → expressed context
│   ├── genome.py               ← SQLite DAL, schema (2,940 lines, big)
│   ├── schemas.py              ← pydantic types for everything
│   │
│   ├── sema.py                 ← ΣĒMA 20d embeddings (query + gene)
│   ├── splade_backend.py       ← sparse term expansion
│   ├── deberta_backend.py      ← cross-encoder reranker (when on)
│   │
│   ├── cymatics.py             ← D6: frequency-domain rerank
│   ├── sr.py                   ← Successor Representation tier
│   ├── tcm.py                  ← Temporal Context Model (partial)
│   ├── cwola.py                ← Statistical-fusion A/B classifier
│   │
│   ├── tagger.py               ← CpuTagger — regex + spaCy ingest
│   ├── tree_chunker.py         ← AST-aware gene chunking
│   ├── codons.py               ← token packing into genes
│   ├── ray_trace.py            ← Monte Carlo co-activation bonus
│   ├── seeded_edges.py         ← co-activation graph builder
│   │
│   ├── headroom_bridge.py      ← Kompress compressor integration
│   ├── registry.py             ← project/party/participant registry
│   ├── config.py               ← helix.toml loader
│   ├── write_queue.py          ← async SQLite writes with batching
│   └── integrations/
│       └── scorerift.py        ← ray-trace integration
│
├── docs/
│   ├── DIMENSIONS.md           ← D1–D9 reference (read first)
│   ├── PIPELINE_LANES.md       ← retrieval flow diagram
│   ├── MISSION.md              ← the why
│   ├── future/                 ← unimplemented / in-progress
│   │   ├── AB_TEST_PLAN.md     ← the measurement that motivated this collab
│   │   ├── STATISTICAL_FUSION.md  ← CWoLa framework details
│   │   ├── SUCCESSOR_REPRESENTATION.md  ← SR design note
│   │   ├── TCM_VELOCITY.md     ← TCM wiring plan
│   │   └── LANGUAGE_AT_THE_EDGES.md  ← math-only ingest thesis
│   └── collab/
│       └── (this directory)
│
├── benchmarks/                 ← needle-in-a-haystack + KV-harvest
├── scripts/                    ← one-off tools
└── helix.toml                  ← runtime config (ports, flags, weights)
```

The three files to read first, in order:

1. [`docs/DIMENSIONS.md`](../DIMENSIONS.md) — what the 9 lanes are and
   what's implemented.
2. [`helix_context/context_manager.py`](../../helix_context/context_manager.py)
   — the retrieval pipeline, start at `_express()`. This is where all
   9 lanes get composed into a ranked top-k.
3. [`helix_context/genome.py`](../../helix_context/genome.py) lines
   340–900 — the SQLite schema. Every persistent piece of state is
   here.

---

## 3. The retrieval pipeline (what happens per query)

```
HTTP /context (query, options)
    ↓
server.py                         ← FastAPI route handler
    ↓
ribosome.py ::ribosome_loop       ← top-level orchestration
    ↓
context_manager._express(query)   ← the actual retrieval
    ↓
  Step 0  Query expansion (optional, flag-gated)
  Step 1  Candidate recall
          ├─ Tier 1:  genes_fts (FTS5)            ← D1
          ├─ Tier 2:  promoter_index + synonyms    ← D2
          ├─ Tier 3:  SPLADE sparse expansion      ← D1
          └─ Tier 3.5: ΣĒMA cosine + cold fallthrough ← D1 + D5
  Step 2  Filter + score fusion
          ├─ Provenance / authority boost          ← D3
          ├─ Density gate (source_id, access rate)  ← D3 + D4
          └─ Chromatin filter (hot vs hot+cold)     ← D5
  Step 3  Rerank
          ├─ Cymatics resonance                    ← D6
          ├─ Tier 5 harmonic boost (1-hop)          ← D8 partial
          └─ SR multi-hop boost (flag-gated)        ← future D8
  Step 4  Gene expression (token compression)
          └─ Headroom Kompress → expressed_context
    ↓
  return {context, genes, scores, tier_trace, retrieval_id}
    ↓
cwola_log insert (ts, query, tier_features, top_gene_id, bucket=NULL)
    ↓
respond to client

[async, later]
  when next query in same session arrives, the PREVIOUS retrieval's
  bucket gets assigned:
    bucket = 'B' if requery_delta_s < 60 else 'A'
```

The hand-tuned weights live in `helix.toml` under `[retrieval.weights]`
and in `context_manager.py` as literal constants. Both surfaces would
be replaced by the learned salience head.

---

## 4. The 9 dimensions, with file pointers

| Dim | Name | File | State |
|---|---|---|---|
| D1 | Semantic (FTS5 + SPLADE + ΣĒMA) | `context_manager.py`, `sema.py`, `splade_backend.py` | active |
| D2 | Promoter tagging | `tagger.py`, `genome.py` (`promoter_index`) | active |
| D3 | Source provenance | `genome.py` (`source_id`), `context_manager.py` | active |
| D4 | Working-set access rate | `genome.py` (`recent_accesses` ring) | active |
| D5 | Chromatin tier | `genome.py` (`chromatin` column, tier logic) | active |
| D6 | Cymatics resonance | `cymatics.py` | active |
| D7 | Gene attribution | `genome.py` (`gene_attribution`, `parties`, `participants`) | schema only, 0 rows |
| D8 | Co-activation graph + SR | `sr.py`, `ray_trace.py`, `seeded_edges.py` | SR shipped dark, harmonic boost active |
| D9 | TCM (temporal context) | `tcm.py` | partial, not wired |

For the joint experiment, the salience manifold consumes the `tier_features`
dict produced at Step 2 of `_express()` — that's the per-dimension
raw-score snapshot before fusion. Swapping the fusion weights is a
single-point integration, not a rewrite.

---

## 5. The feedback surface — where to get training data

### `cwola_log` (the main signal)

Schema at `helix_context/genome.py:805–818`. Every `/context` call
writes a row. Columns:

- `query` — the raw query text
- `tier_features` — JSON of per-dimension raw scores for that retrieval
- `top_gene_id` — the winning gene
- `bucket` — 'A' (accepted), 'B' (re-queried within 60s), NULL (pending)
- `requery_delta_s` — seconds to the next same-session query
- `session_id` — for grouping
- `party_id` — for federation-aware training

A clean export for Celestia training:

```sql
SELECT
    retrieval_id,
    query,
    tier_features,      -- parse as JSON
    top_gene_id,
    bucket,
    requery_delta_s,
    party_id
FROM cwola_log
WHERE bucket IN ('A', 'B')
  AND ts > strftime('%s', '2026-04-01')
ORDER BY ts DESC;
```

Exposed via a future `/admin/cwola-export` endpoint — not yet built;
for now use direct SQLite read against `genome.db`.

### `hitl_events` (secondary signal)

Schema at `helix_context/genome.py:762–784`. Captures operator pauses —
when the agent stopped to ask the human, what the task was, whether the
operator intervened. Useful as a "hard negative" signal (the agent got
stuck enough to pause → the retrieval probably wasn't strong enough).

### `epigenetics.recent_accesses`

Ring buffer of last 100 access timestamps per gene. Not a feedback
signal per se but a working-set residency indicator. Read for D4.

---

## 6. How to run it locally

### Prereqs
- Python 3.11+
- A local Ollama instance on `:11434` with at least one model pulled
  (`qwen3:8b` is the default bench model)
- ~2 GB for a representative genome

### Quick start

```bash
git clone <helix-context repo>
cd helix-context
pip install -e .

# Ingest a project
python -m helix_context.cli ingest /path/to/some/codebase

# Start the server
python -m helix_context.server
# → FastAPI on 127.0.0.1:11437

# Health check
curl http://127.0.0.1:11437/health

# Query
curl -X POST http://127.0.0.1:11437/context \
     -H "Content-Type: application/json" \
     -d '{"query": "how does retrieval fusion work?"}'
```

### Query the currently-running genome (17,959 genes as of this writing)

If the server on `:11437` is already running against `genome.db`, you
can query directly — no ingest needed. The existing genome includes
helix's own source, plus several other projects (education_public,
scorerift, Steam metadata, BeamNG, CosmicTasha, GGUF).

```bash
curl -s -X POST http://127.0.0.1:11437/context \
     -H "Content-Type: application/json" \
     -d '{"query": "where is the successor representation implemented?"}' \
     | jq '.genes[:3], .tier_trace'
```

The `tier_trace` field in the response shows per-dimension scores for
top candidates — this is what the manifold would consume.

### Run the benchmarks

```bash
# Curated SIKE (N=10, fast)
python benchmarks/bench_needle.py

# KV-harvest (N=50, ~12 min)
N=50 SEED=42 HELIX_MODEL=qwen3:8b python benchmarks/bench_needle_1000.py
```

Both benches log to `benchmarks/needle_*.json` and print a summary. The
KV-harvest bench is the one the joint experiment is graded against
(§4 of the design doc).

---

## 7. Configuration surface

`helix.toml` is the runtime config. Relevant sections for the
experiment:

```toml
[retrieval]
sr_enabled = true             # flip to A/B the SR tier
rerank_enabled = false        # cross-encoder rerank
query_expansion_enabled = true

[retrieval.weights]
# hand-tuned per-dimension scale factors — this is what we're replacing
promoter_exact = 3.0
promoter_entity = 2.5
fts_high = 6.0
sema_cosine = 2.0
harmonic_boost = 3.0
source_authority = 1.5
# ... etc

[retrieval.budget]
# budget tiers (tight/focused/broad) with absolute score floors
tight_top_score_min = 5.0
tight_ratio_min = 3.0
focused_top_score_min = 2.5
focused_ratio_min = 1.8

[ribosome]
# LLM usage — default path is LLM-free
query_expansion_enabled = true
```

For the joint experiment, add a new section:

```toml
[retrieval.learned_salience]
enabled = false               # default off, matches SR rollout pattern
model_path = "models/salience_v1.pt"
fallback_to_handtuned = true  # if inference fails, don't regress
```

---

## 8. Where the joint experiment's code would live

Minimum-surface-area integration:

```
helix_context/
├── retrieval_salience.py     ← NEW: RetrievalSalienceAdapter class
│                               wraps the Celestia manifold, consumes
│                               tier_features dict, returns per-dim
│                               scaling factors
└── context_manager.py        ← 1 patch in _express() Step 2 to call
                                the adapter when flag is on
```

Training code (Celestia side, not in the helix repo):

```
celestia/
└── retrieval_manifold/
    ├── train.py              ← CWoLa binary classifier, reads
    │                           exported cwola_log JSON
    ├── model.py              ← 3-pathway Mamba, 49d → 9d
    └── export.py             ← save torch model to helix/models/
```

Clean separation: helix doesn't import Celestia code, it loads a
serialised model. If the model changes substrate (PyTorch → ONNX →
whatever), helix's adapter handles the deserialization.

---

## 9. Known sharp edges

1. **`genome.py` is 2,940 lines.** It's the whole DAL + schema + some
   business logic. Split is pending. For now: schema is lines 340–900,
   retrieval is 1,200–1,900, write paths are 1,900–2,400. Ignore the
   rest on first read.
2. **Windows-native project.** Shell is bash-on-Windows. Subprocess
   calls use `CREATE_NO_WINDOW`. If you're on Linux/macOS most of it
   still works, but a handful of path manipulations are Windows-first.
3. **SQLite WAL mode, busy-retry with jittered backoff.** Don't
   `sqlite3.connect()` directly — go through `genome.py` helpers.
   Concurrent writes are fine; concurrent schema changes are not.
4. **Uncommitted WIP in `genome.py`.** As of this writing, another
   collaborator has staged changes to genome.py. Don't rebase on main
   until that lands or coordinate via `~/.helix/shared/handoffs/`.
5. **The 17K-gene genome is partly contaminated.** Steam manifests,
   BeamNG configs, and GGUF metadata were ingested during a broad
   sweep that turned out to hurt retrieval (AB_TEST_PLAN §"Struggle
   1"). A clean re-ingest is pending. The learned salience manifold
   will train on current `cwola_log` which reflects this — good and
   bad.

---

## 10. Who to ping for what

- **Retrieval logic, weights, dimensions, benchmarks** — helix-context
  primary maintainer (author of this doc).
- **`genome.py` / schema / raude's WIP** — coordinate via
  `~/.helix/shared/handoffs/` off-git. A Claude persona called Raude
  works in the genome.py layer; handoffs live there.
- **Phase 6 TCM / trajectory work** — assigned to another Claude
  persona (Taude). Temporal context is their lane.
- **Headroom / Kompress compression** — upstream is `chopratejas/headroom`;
  helix pins a version via `headroom_bridge.py`. PR #152 landed the
  `compress_batch` path recently.

---

## 11. Philosophy notes

A few things that will save you a conversation cycle:

- **"Local file" means off-git.** Coordination handoffs default to
  `~/.helix/shared/handoffs/` not `docs/`. Docs are public-surface.
- **No destructive ops without explicit consent.** Don't force-push,
  drop tables, rewrite migrations, or rm genome.db. The
  `cwola_log` data is accumulating experimental signal — treat it
  as write-once append-only.
- **Predictions before results.** The `AB_TEST_PLAN.md` pattern is
  load-bearing: lock predictions before running the experiment, then
  honestly check which parts of the mental model were calibrated.
  Apply the same pattern to the joint experiment (§4 of the design
  doc is pre-registered).
- **Honest limits in claim docs.** If a thing doesn't work, the doc
  says so. If a thing hasn't been measured, the doc says so. Don't
  backfit.

---

## 12. What to read next

If you want to engage with the joint experiment:

1. [`docs/collab/CELESTIA_JOINT_EXPERIMENT.md`](CELESTIA_JOINT_EXPERIMENT.md) — the design doc
2. [`docs/future/AB_TEST_PLAN.md`](../future/AB_TEST_PLAN.md) — the measurement that motivated it
3. [`docs/future/STATISTICAL_FUSION.md`](../future/STATISTICAL_FUSION.md) — CWoLa framework
4. `helix_context/context_manager.py::_express()` — where integration would land
5. `helix_context/cwola.py` — existing classifier training surface

If you want the broader picture:

6. [`docs/MISSION.md`](../MISSION.md)
7. [`docs/RESEARCH.md`](../RESEARCH.md)
8. [`docs/AGENTOME_PART_II_DRAFT.md`](../AGENTOME_PART_II_DRAFT.md) — the
   public-facing Substack draft (uncommitted), where the E8/13-dim
   experiments are written up

Welcome aboard.

# Cymatix Context

[![Website](https://img.shields.io/badge/website-cymatixcontext.com-ff9f43.svg)](https://cymatixcontext.com)
[![Discord](https://img.shields.io/badge/discord-join-5865F2.svg?logo=discord&logoColor=white)](https://discord.gg/pX7x7pA3Da)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/pypi/v/cymatix-context.svg)](https://pypi.org/project/cymatix-context/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 4000+](https://img.shields.io/badge/tests-4000%2B-brightgreen.svg)](tests/)
[![LLM-free pipeline](https://img.shields.io/badge/pipeline-LLM--free-brightgreen.svg)](docs/architecture/PIPELINE_LANES.md)

> Coordinate-index engine for LLM agents. Retrieves, weighs, and compresses
> your codebase into a context window — without a single LLM call on the
> retrieval path.

One SQLite knowledge store, a seven-stage pipeline, and an explicit `know` / `miss` contract on every response. *The engine's namesake cymatics stage — an MD5-binned 256-dimensional term spectrum — is a candidate-reordering signal that has **not yet been isolated** against hashed bag-of-words or random-bin controls; treat it as an experimental cheap feature, not a proven one.*

## Proof (30 seconds)

**Token economics** — compressor disabled (the default LLM-free config), N=15 query shapes, May 2026:

| Query shape | Tokens per turn | vs standard RAG |
|---|---|---|
| Best — focused query | 1,410 | **5.7×** fewer |
| Median | **2,757** | **2.9×** fewer |
| Worst — broad 12-document window | 3,755 | **2.1×** fewer |

That denominator is a *configurable modeled baseline*, not a measured competitor run: top-5 × 1,500 + 500 overhead = 8,000 tokens. Reproducer: `benchmarks/bench_rag_vs_sike_tokens.py`, against your own store. The multi-turn session-delivery figures (~40% savings, 37× on repeated retrievals) are **unverified design estimates** pending the `cymatix_session_tokens_saved_total` counter.

**Historical v0.9.1 defaults (released 2026-08-30)** — 829K-fragment [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) bed, n=470, delivered basis, on a fully algorithmic retrieval path (dense, SPLADE and PKI default-off since 2026-08-15 / -16 / -17, each flip receipt-gated). Two defaults graduated in v0.9.1, each measured as a paired row: the wave-1 ranking flip ([#407](https://github.com/mbachaud/Cymatix-Context/pull/407), `rrf_k` 60 → 20 plus all-classes `eps_band` combinators) moved gold-document delivery **0.555 → 0.630** and recall@12 0.651 → 0.681 with zero question-type regressions; the delivered-seat floor ([#409](https://github.com/mbachaud/Cymatix-Context/pull/409), `[budget] min_delivered_docs = 12`) then moved delivery **0.630 → 0.668** (+18/−0) with the ranking bases byte-identical, confirmed on two more corpora with zero paired losses. Release gate: `benchmarks/dogfood/receipts/sweep_v091_gate_2026-08-30.json` (ALL PASS). These are retrieval-layer measurements, *not* end-to-end grades — the ERB judge protocol has not been re-run on these defaults.

**Shipped defaults (v0.9.2, released 2026-09-08)** — one ingestion default changes: `[ingestion] entity_autolink_hub_cutoff = 200` excludes entities with more than 200 existing postings from auto-link probes during ingestion. Set it to `0` to restore legacy linking. The paired receipts and scope are documented in [Configuration](wiki/Configuration.md) and the [CHANGELOG](CHANGELOG.md).

**0.9.0 shipped defaults, for reference** — 829k bed, n=469: **56.5% gold-document delivery** (265/469), recall@12 **0.659** (`benchmarks/dogfood/erb/receipts/sema_readgate_829k_n469.json`). That is a different ledger row from the 0.9.1 lines above — a different bed build and ingest concurrency — so the two are not a before/after pair.

Methodology, the ERB correctness/delivery pair-quote rule, the sharded gap ([#275](https://github.com/mbachaud/Cymatix-Context/issues/275)), and the dense-off latency disclosure (×2.5–2.6 at 100k, shrinking at the 829k operating point; receipts in the CHANGELOG, [#374](https://github.com/mbachaud/Cymatix-Context/issues/374)) all live on [Benchmarks and Receipts](https://github.com/mbachaud/Cymatix-Context/wiki/Benchmarks-and-Receipts).

## Get started

Python 3.11+. The core install is dependency-light (FastAPI + SQLite, no torch); extras add what you turn on — `cpu` (spaCy ingest tagging), `embeddings` (opt-in dense recall), `mcp`, `ast`, `otel`, `launcher-tray`, `all`. Full extras matrix, GPU detection, and three worked workflows: [Getting Started](https://github.com/mbachaud/Cymatix-Context/wiki/Getting-Started) · [docs/SETUP.md](docs/SETUP.md).

```bash
pip install "cymatix-context[embeddings,cpu,mcp]"    # recommended working set
python -m spacy download en_core_web_sm              # ingest tagger model

cymatix ingest path/to/your/project/ --recursive     # 1. build the store
cymatix query "how does the splice step work?"       # 2. ask it — no server
cymatix packet "edit the splice step" --task-type edit --json   # 3. agent bundle
cymatix-server                                       # 4. proxy on 127.0.0.1:11437
```

## Pipeline

Seven stages per turn, all LLM-free except the optional splice. Stage by stage, and where the model boundary actually sits: [Pipeline](https://github.com/mbachaud/Cymatix-Context/wiki/Pipeline).

```
  query
    ▼
  0. Classify   rule-based: decoder mode + assembly cap
    ▼
  1. Extract    heuristic keyword + entity extraction
    ▼
  2. Retrieve   FTS5 BM25 + tags (+ opt-in BGE-M3 dense) + synonym expansion
    │           + co-activation + SR + cymatics 256-bin spectrum scoring,
    ▼           ranked via RRF (default) or additive fusion
  3. Re-rank    CPU classifier scores (optional)
    ▼
  4. Splice     Headroom Kompress (CPU) or LLM compressor (optional)
    ▼
  5. Assemble   token budget + legibility headers (fired tiers, confidence
  + Stage 7     ◆/◇/⬦, compression ratio) + freshness gate (stale/cold/
    ▼           superseded → miss) + session delivery (elide seen docs)
  6. Persist    query+response → knowledge store (background)
    ▼
  know { } or miss { }
```

## Surfaces

Three ways in, same retrieval primitives, same JSON shapes. Direct MCP needs neither the model proxy nor the tray — a healthy headless server is sufficient. Configuration lives in `cymatix.toml`; env vars use the `CYMATIX_*` prefix. Reference: [CLI](https://github.com/mbachaud/Cymatix-Context/wiki/CLI) · [HTTP API](https://github.com/mbachaud/Cymatix-Context/wiki/HTTP-API) · [MCP and IDE Integration](https://github.com/mbachaud/Cymatix-Context/wiki/MCP-and-IDE-Integration) · [Configuration](https://github.com/mbachaud/Cymatix-Context/wiki/Configuration).

| Surface | Best for | Example |
|---|---|---|
| **CLI** | Scripts, CI, cold-start agents | `cymatix document get abc123 --json` *(legacy: `cymatix gene get`)* |
| **MCP** | Claude Code, Codex, Gemini CLI, Antigravity | `python -m cymatix_context.mcp_server` · [client guides](docs/clients/cymatix-context.md) |
| **HTTP** | Continue IDE, `OPENAI_BASE_URL` redirect | `POST /context/packet` |

## The know/miss contract

- **`know { found, confidence }`** — the context is grounded; the agent may answer.
- **`miss { reason, escalate_to }`** — don't answer from the knowledge store; escalate, or refetch from `refresh_targets`. The freshness gate downgrades stale / cold / superseded results into a `miss`.
- **The shape is stable; the confidence is provisional.** The contract shape is load-bearing, but the `confidence` scalar is under active recalibration and is not yet a reliable trust signal on current internal beds ([#287](https://github.com/mbachaud/Cymatix-Context/issues/287), [#239](https://github.com/mbachaud/Cymatix-Context/issues/239)) — rely on `found` / `reason`. Full semantics: [Agent Contract](https://github.com/mbachaud/Cymatix-Context/wiki/Agent-Contract).

## Gotchas

- **Knowledge store path** is `genomes/main/genome.db`, not the project root. Delete it to start fresh; it auto-creates on first use.
- **The synonym map is critical.** "No relevant context" usually means the query keywords don't map to the tags assigned at ingest — add them under `[synonyms]`.
- **Session delivery** (`session_delivery_enabled = true`) elides already-delivered documents per session; the ~40% multi-turn saving is an unverified design estimate. Pass `ignore_delivered: true` in the `/context` body for benchmarks.
- **Sharded scale gap.** The sharded path trails the unsharded engine by ~31pp recall@10 / ~30pp MRR on the xl bed ([#275](https://github.com/mbachaud/Cymatix-Context/issues/275)) — prefer unsharded for accuracy-sensitive corpora.
- **The agent prompt fragment is load-bearing.** Without it, frontier models confabulate past a `miss`. Import `cymatix_context.agent_prompt.full_fragment()`.

## Observability

Optional Grafana/Tempo/Loki sidecar: `scripts\setup-grafana-telem.ps1` (Windows) or `scripts/setup-grafana-telem.sh` (Linux/macOS), dashboards at `localhost:3000`. Full surface: [docs/architecture/OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) · [wiki: Observability](https://github.com/mbachaud/Cymatix-Context/wiki/Observability).

## Documentation

The **[wiki](https://github.com/mbachaud/Cymatix-Context/wiki)** is the narrative documentation — 15 pages, [Troubleshooting](https://github.com/mbachaud/Cymatix-Context/wiki/Troubleshooting) included, also rendered at <https://cymatixcontext.com/wiki/>. Questions, and arguments about the receipts: [Discord](https://discord.gg/pX7x7pA3Da).

| Repo doc | What it is |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | The canonical install path, with every nuance |
| [docs/config-reference.md](docs/config-reference.md) | Every `cymatix.toml` key, default, and flip date |
| [docs/api/endpoints.md](docs/api/endpoints.md) | Full HTTP schema |
| [docs/clients/cli.md](docs/clients/cli.md) | Full CLI reference |
| [docs/benchmarks/BASELINES.md](docs/benchmarks/BASELINES.md) | The receipt ledger and its comparability rules |
| [wiki: Lexicon](https://github.com/mbachaud/Cymatix-Context/wiki/Lexicon) | Biology-to-software lexicon ([docs/ROSETTA.md](docs/ROSETTA.md) is now a stub that points there) |

Built on [spaCy](https://spacy.io/) NER, SQLite FTS5 BM25, [BGE-M3](https://huggingface.co/BAAI/bge-m3), [Kompress](https://huggingface.co/chopratejas/kompress-base), [Headroom](https://github.com/chopratejas/headroom), and the Howard 2005 TCM / Stachenfeld 2017 SR literature — full attributions in [NOTICE](NOTICE).

## How this was built

Cymatix Context is architected and QA-directed by Michael Bachaud. Implementation,
refactoring, draft documentation, and test generation are produced by AI coding
agents under spec- and benchmark-gated review. The human owns the product thesis,
architecture selection, acceptance criteria, experiment design, and falsification
authority; the models own the code production.

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for third-party attributions.

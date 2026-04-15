# Helix Context

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0b1-orange.svg)](https://pypi.org/project/helix-context/)
[![SIKE: 10/10](https://img.shields.io/badge/SIKE-10%2F10_retrieval-brightgreen.svg)](docs/RESEARCH.md#benchmark-results--scale-invariant-knowledge-engine-sike)
[![Compression: 769x](https://img.shields.io/badge/inference_compression-769x-brightgreen.svg)](docs/RESEARCH.md)
[![Paper: Agentome](https://img.shields.io/badge/paper-Agentome-purple.svg)](https://mbachaud.substack.com/p/agentome)

**A local-first knowledge store for LLM context compression.**
**Scale-Invariant Knowledge Engine (SIKE) — 10/10 retrieval from 0.6B to 26B parameters.**

> **The `/context` pipeline is LLM-free.** Ingest, tagging, candidate
> selection, fusion, chromatin gating, and context assembly are all pure
> CPU math — Howard 2005 TCM, Stachenfeld 2017 SR, Werman 1986 W1, Hebbian
> co-activation, spaCy NER. MiniLM (SEMA encoder, 384d → 20d) and DeBERTa
> (optional rerank/splice classifiers) are **not** LLMs — they're small
> transformer encoders, no generation. The only LLM call on the hot path
> is the final answer-generation step at `/v1/chat/completions`. Helix
> used to depend on an LLM to serve another LLM; now it just uses
> well-understood retrieval math.
>
> An optional "subconscious" layer — the ribosome — sits off the hot path
> for idle-time re-processing (tighter complements, cross-gene pattern
> noticing) and is how partners plug a hosted model (Anthropic, Google,
> etc.) into the compression seam without forking. Default: off. See
> [`[ribosome]`](#configuration) in the config.

> A persistent SQLite-backed **knowledge store** holds compressed documents.
> 7,200 documents (44 MB raw knowledge) compress to ~15K tokens of retrieved
> context per turn — a **769× inference compression ratio**. Retrieval is
> scale-invariant: the same store delivers 10/10 needle accuracy to qwen3:0.6b
> and Claude Opus alike. The retrieval engine does the work; the LLM just extracts.

> **📖 Terminology note** — Helix's original vocabulary borrowed from molecular
> biology (gene, genome, ribosome, chromatin, splice, promoter). The **canonical
> lexicon is now standard software terminology**; the biology terms remain valid
> aliases for older docs and commit history. See [docs/ROSETTA.md](docs/ROSETTA.md)
> for the full bidirectional mapping. Quick cheat-sheet:
>
> | Biology (legacy) | Software (canonical) |
> |---|---|
> | gene | document |
> | genome | knowledge store (kb) |
> | ribosome | compressor |
> | promoter tags | document tags |
> | chromatin tier / OPEN·EUCHROMATIN·HETEROCHROMATIN | lifecycle tier / OPEN·WARM·COLD |
> | expression | retrieval |
> | replication | persistence |
> | harmonic_links | co-activation edges |
> | horizontal gene transfer (HGT) | cross-store import |
>
> Both forms are used interchangeably throughout this README; older sections
> skew biology, newer sections skew software. `Document is Gene` is literally
> `True` at the Python class level ([helix_context/aliases.py](helix_context/aliases.py)).

<details>
<summary><b>📑 Table of Contents</b></summary>

- [Benchmark Highlights](#benchmark-highlights)
- [Quick Start](#quick-start)
- [What You'll See](#what-youll-see)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
  - [Context Health Monitor (Delta-Epsilon)](#context-health-monitor-delta-epsilon)
  - [Cross-Store Import (HGT)](#cross-store-import-hgt)
  - [Associative Memory](#associative-memory)
  - [Multi-Agent Identity (4-layer attribution)](#multi-agent-identity-4-layer-attribution)
  - [Task-Conditioned Retrieval (MoE + Small Models)](#task-conditioned-retrieval-moe--small-models)
  - [Synonym Expansion](#synonym-expansion)
- [HTTP Endpoints](#http-endpoints)
- [Continue IDE Integration](#continue-ide-integration)
- [Python API](#python-api)
- [ScoreRift Integration](#scorerift-integration)
- [Configuration](#configuration)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Architecture](#architecture)
- [Origin](#origin)
- [License](#license)

</details>

```
  Client (Continue, Cursor, any OpenAI client)
         |
         v
  +----------------------------+
  |  Helix Proxy (FastAPI)     |  Port 11437
  |  /v1/chat/completions      |  OpenAI-compatible
  |                            |
  |  1. Extract query          |
  |  2. Retrieval pipeline     |  <-- Knowledge store (SQLite)
  |     (seam compression)     |      optional: Headroom codec
  |  3. Inject context         |  <-- Compressor (CPU model)
  |  4. Forward to Ollama      |  --> localhost:11434
  |  5. Stream tee response    |
  |  6. Background persistence |
  +----------------------------+
```

Instead of stuffing your entire codebase into the prompt, Helix compresses it into a persistent SQLite knowledge store and retrieves only the relevant documents per turn. The model sees compressed context, not raw text. Conversations persist back into the store automatically, building institutional memory over time.

## Benchmark Highlights

> 🎯 **10/10 needle retrieval** from 0.6B to 26B parameters (43x range)
> 🚀 **769x inference compression** (11.6M-token genome → 15K expressed per turn)
> 💎 **Claude Haiku + Helix matches Opus** — all three API tiers hit 10/10 accuracy
> 🧠 **Local 4B model beats blind Opus 2.25x** on domain-specific extraction

### Test Corpus Composition

The benchmark genome is a real developer's working data, not a curated eval set.
**65.8% of the corpus is pure noise** — game data, subtitles, blueprints — and Helix
still hits 10/10 on project-specific needles hidden in the remaining 34%.

| Source Category | Genes | Tokens | % | Repo Visibility |
|---|---:|---:|---:|:---|
| 🎮 Steam / game data (Hades subtitles, BeamNG configs, Dyson Sphere blueprints, Factorio saves) | 2,905 | ~7.7M | **65.8%** | — |
| 🌐 [`SwiftWing21/BigEd`](https://github.com/SwiftWing21/BigEd) — BigEd fleet (Education dir) | 2,405 | ~1.8M | 15.4% | **public** (private worktree ahead by 2 commits) |
| 🔒 [`CosmicTasha/CosmicTasha`](https://github.com/CosmicTasha/CosmicTasha) | 944 | ~1.6M | 13.9% | private |
| 🔒 Project Tally (private financial ledger — repo URL withheld) | 242 | ~0.2M | 2.0% | private |
| 🌐 [`SwiftWing21/helix-context`](https://github.com/SwiftWing21/helix-context) — this repo | 161 | ~0.1M | 1.2% | public |
| 🌐 [`SwiftWing21/scorerift`](https://github.com/SwiftWing21/scorerift) — ScoreRift / two-brain-audit | 110 | ~0.1M | 0.7% | public |
| Unclassified / session memory | 497 | ~0.1M | 1.0% | — |
| **Total** | **7,264** | **~11.6M** | **100%** | |

**Source breakdown (software only, excluding game noise):**

- 🌐 **Public GitHub repos: ~2.0M tokens (50.0%)** — BigEd, helix-context, scorerift
- 🔒 **Private GitHub repos: ~1.8M tokens (45.6%)** — CosmicTasha, BookKeeper
- 🔄 **Unclassified / session memory: ~0.2M tokens (4.4%)**

**Signal-to-noise:** Only ~33% of the 11.6M-token corpus is relevant software knowledge.
The other ~66% is game data the Agentome had to learn to ignore via chromatin state
(`HETEROCHROMATIN` tier) and promoter-tag discrimination. The 10/10 retrieval holds
*despite* the noise — arguably *because* of it, since real-world retrieval systems have
to survive mixed-domain corpora.

> 💡 **How this table was measured:** Claude (co-authoring this repo) had workspace
> access to the user's local project directories during the benchmark session, including
> private repos that never leave the machine. The genome file itself is gitignored —
> only aggregate counts and the benchmark queries are public. This demonstrates a real
> use case for Helix: **your proprietary code participates in retrieval without being
> uploaded anywhere**. Even the Education directory is split — the bulk lives in the
> public [`BigEd`](https://github.com/SwiftWing21/BigEd) repo, with a private worktree
> ahead by 2 unreleased commits.

### Database Storage Breakdown (post-VACUUM)

The on-disk `genome.db` is **523 MB** for 7,264 genes (~46 MB of raw content).
Why the ~12x gap between raw content and DB file? Because the genome isn't just storage —
it's a **4-tier retrieval engine** (promoter tags → FTS5 → SPLADE → ΣĒMA semantic), and
each tier carries its own index.

| Component | Size | % of DB | Purpose |
|---|---:|---:|---|
| **FTS5 posting lists** (`genes_fts_data`) | 187.3 MB | 35.8% | Full-text inverted index for keyword retrieval |
| **Raw content** (`gene.content`) | 44.5 MB | 8.5% | Original source text, verbatim |
| **SPLADE sparse index** (`splade_terms`) | 35.7 MB | 6.8% | 1.73M term weights for lexical expansion |
| **Ribosome complements** (`gene.complement`) | 16.5 MB | 3.2% | Small-model compressed summaries (2.69x storage ratio) |
| **Gene relations** (NLI) | 6.6 MB | 1.3% | 108K typed logical relations between genes |
| **Entity graph** | 5.6 MB | 1.1% | 117K entity-to-gene edges for co-activation |
| **Promoter index** (retrieval tags) | 3.8 MB | 0.7% | 73,815 domain/entity tags across all genes |
| **Codons + metadata JSON** | 8.2 MB | 1.6% | Semantic tags, promoter JSON, epigenetics |
| **ΣĒMA embeddings** (20D vectors) | 0.34 MB | 0.1% | Semantic primes — 80 bytes per gene |
| **Key-value facts** (pre-extracted) | 1.4 MB | 0.3% | Pre-parsed `key=value` pairs for answer slate |
| **Accounted payload subtotal** | **310.0 MB** | **59.3%** | Actual data across all indexes |
| **SQLite B-tree + page overhead** | 212.7 MB | 40.7% | Index structure, not fragmentation |
| **Total file size** | **522.7 MB** | **100%** | |

> 💾 **VACUUM impact:** This table reflects post-`VACUUM` state. Before VACUUM, the
> database was **752 MB** — the extra 229 MB (30.4%) was free pages from thinning
> 11,529 genes down to 7,264 during tuning. SQLite holds deleted pages until a
> VACUUM reclaims them. The ~213 MB of "B-tree overhead" that remains is *structural*:
> page headers, cell pointers, interior nodes of the index B-trees. That's not
> reclaimable without changing the indexing strategy.

**Observations:**

- **FTS5 dominates storage** (35.8% of the file). The full-text index holds position
  data for every token across all 7K genes — it's what enables the sub-5ms content
  queries that make the ~1s total retrieval latency possible.
- **Raw content is only 8.5% of the file**. The rest is indexes. This is the expected
  tradeoff for a retrieval-optimized database vs a flat text archive.
- **Accounted payload is 310 MB (59.3%)**. The remaining 213 MB (40.7%) is legitimate
  B-tree structure overhead — page headers, cell pointers, and internal index nodes.
  SQLite can't compress this further without sacrificing query speed.
- **ΣĒMA embeddings are essentially free** — 20 floats per gene = 80 bytes. A 1M-gene
  genome would cost only 80 MB for the semantic tier.
- **Inference cost is unchanged by DB size**: the LLM only ever sees ~15K tokens
  per turn regardless of whether the genome is 50 MB or 50 GB.

**Compression summary:**

| Metric | Ratio | Meaning |
|---|---:|---|
| Storage (raw → complement) | **2.69x** | How much the ribosome compresses each gene's summary |
| Expression (full corpus → single turn) | **776x** | How much of the genome the LLM sees per query |
| DB file / raw content | 11.76x (post-VACUUM) | Index overhead for 4-tier retrieval |
| DB file / raw content | 16.90x (pre-VACUUM) | With fragmentation from thinning |
| vs 128K-stuffed context | 8.5x fewer tokens | Baseline "dump everything" approach |
| vs chunked RAG (25K tokens) | 1.7x fewer tokens | Standard vector-search RAG |

The headline number — **776x inference compression** — is what matters for cost and
latency. Everything else is a bookkeeping detail of how the Librarian files its books.

**Needle-in-a-haystack on this 7,264-gene genome (~46MB raw knowledge):**

| Model | Params | VRAM | Retrieval | Accuracy |
|-------|--------|------|-----------|----------|
| qwen3:0.6b | 0.6B | 0.5 GB | **10/10** | 2/10 |
| qwen3:1.7b | 1.7B | 1.4 GB | **10/10** | 3/10 |
| **qwen3:4b** | **4B** | **2.5 GB** | **10/10** | **9/10** |
| gemma4:e4b (MoE) | 8B / 4B active | 9.6 GB | **10/10** | **9/10** |
| qwen3:8b | 8B | 5.2 GB | **10/10** | **9/10** |
| gemma4:26b-a4b (MoE + DDR4 offload) | 26B / 4B active | 8 GB + 13 GB RAM | **10/10** | 6/10 |
| **Claude Haiku + Helix** | — | API | **10/10** | **10/10** |
| **Claude Sonnet + Helix** | — | API | **10/10** | **10/10** |
| **Claude Opus + Helix** | — | API | **10/10** | **10/10** |

**Without Helix, the same Claude models score 3-4/10** (hand-curated reference only).
The genome is a universal uplift: identical gains at every price tier and parameter count.
See [docs/RESEARCH.md](docs/RESEARCH.md#benchmark-results--scale-invariant-knowledge-engine-sike) for the full SIKE analysis.

## Quick Start

```bash
# Install from PyPI (beta)
pip install helix-context --pre

# Pull a small model for the ribosome (context codec)
ollama pull gemma4:e2b

# Start the proxy
helix
# or: python -m uvicorn helix_context.server:app --host 127.0.0.1 --port 11437

# Seed the genome with your own project files
python examples/seed_genome.py path/to/your/project/

# Check genome health
curl http://127.0.0.1:11437/stats
```

Point any OpenAI-compatible client at `http://127.0.0.1:11437/v1` and start chatting. Context compression happens transparently.

### Run it with the launcher (supervisor + dashboard)

Instead of babysitting a terminal, use the bundled launcher — a
separate supervisor process that spawns and monitors helix, with a
live dashboard for status and Start/Restart/Stop controls.

```bash
# Install with the launcher extras
pip install helix-context[launcher] --pre

# Run it — opens http://127.0.0.1:11438/ in your browser
helix-launcher

# Or as a system tray app (requires [launcher-tray] for pystray)
pip install helix-context[launcher-tray] --pre
helix-launcher --tray
```

The launcher:

- Spawns `helix` as a supervised child process on `:11437`
- Shows parties (devices) / participants (humans) / models / tools /
  genes / tokens panels in a live dashboard (polls every 2 seconds).
  Org and agent panels are on the roadmap — attribution captures all
  four layers already; the UI surfaces the two middle ones today.
- Wires Start / Restart / Stop buttons to the helix process via the
  restart-protocol-compliant announce + kill path
- **Adopts** an already-running helix via state file on startup —
  you can restart the launcher without killing helix
- Runs with a **system tray icon** in `--tray` mode: close browser
  tabs freely, the launcher keeps running in the tray until you
  click Quit

System-service templates for running the launcher unattended live in
[`deploy/`](deploy/) (systemd, launchd, and NSSM for Windows). See
[`docs/LAUNCHER.md`](docs/LAUNCHER.md) for the full architecture.

## What You'll See

After seeding the genome, `/stats` shows the state of your knowledge base:

```bash
$ curl -s http://127.0.0.1:11437/stats | jq
{
  "total_genes": 7264,
  "open": 7264,
  "compression_ratio": 2.69,
  "health": {
    "total_queries": 503,
    "avg_ellipticity": 0.62,
    "status_counts": {"aligned": 143, "sparse": 267, "denatured": 93}
  }
}
```

A `/context` query returns the expressed context window — exactly what gets injected
into the downstream LLM:

```bash
$ curl -s http://127.0.0.1:11437/context \
    -H "Content-Type: application/json" \
    -d '{"query":"What port does the Helix proxy listen on?"}' | jq '.[0]'
{
  "name": "Helix Genome Context",
  "description": "12 genes expressed, 3.1x compression, health=aligned (Δε=0.66)",
  "content": "<expressed_context>\n<GENE src=\"helix-context/README.md\" facts=\"port=11437\">\n# Helix Context\n...",
  "context_health": {
    "ellipticity": 0.66,
    "coverage": 0.85,
    "density": 0.42,
    "freshness": 1.0,
    "genes_expressed": 12,
    "status": "aligned"
  }
}
```

A chat request through the proxy gets the context injected automatically — your
client doesn't need to know Helix exists:

```bash
$ curl -s http://127.0.0.1:11437/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3:4b",
      "messages": [{"role":"user","content":"What port does the Helix proxy use?"}]
    }' | jq -r '.choices[0].message.content'

The Helix proxy server listens on **port 11437**, as specified in helix.toml
under [server]. This is configured in the repository at helix-context/README.md.
```

The model answered from the retrieved genes, not its training data — which doesn't
contain your project.

## How It Works

**`/context` pipeline per query (LLM-free today — zero `backend.complete()` calls on this path):**

| Step | What | LLM? |
|------|------|------|
| 1. Query parse | Heuristic keyword / entity extraction | No |
| 2. SEMA encode | MiniLM 384d → project to 20d (encoder, not generator) | No |
| 3. 12-tier scorers | FTS5, SPLADE, SEMA cosine, lex_anchor, tag_exact/prefix, pki, harmonic, sr (multi-hop), cymatics — all pure math | No |
| 4. Fusion | Weighted sum across tier scores | No |
| 5. Top-k + chromatin gate | Filter by retrieval score + tier residency | No |
| 6. Gene fetch | SQLite read | No |
| 7. Codon decompression | Deterministic expansion | No |
| 8. Context assembly | Sort + join spliced parts, wrap in tags, emit session stubs | No |
| 9. CWoLa log write | Telemetry insert for tier-contribution analysis | No |

> Two optional LLM hooks exist off the default path. `rerank_enabled`
> (off) routes candidate rerank through DeBERTa cross-encoder (still
> not a generator) or an LLM backend. `query_expansion_enabled` (off)
> fires one Step 0 ribosome call per query for intent expansion —
> worth ~2-3pp on ambiguous queries, not worth it on specific ones.

**Token budget:**
- 12k tokens: expressed context (dense XML gene format, 12 genes per turn)
- 11M+ tokens: genome cold storage (SQLite, ~46MB raw on a mature project)
- 3k tokens: optional decoder prompt (only emitted when the downstream
  model benefits from an explicit codon-reading preamble — API models
  get `decoder_mode = "none"` for free savings)

**Compression metrics:**
- Storage: 2.7x (raw content → stored complements)
- Expression: **769x** (full genome → what the LLM sees per turn)
- vs naive RAG at 25K tokens: 1.7x fewer tokens, 10/10 vs ~6/10 accuracy

## Key Features

### Context Health Monitor (Delta-Epsilon)

Every query computes a health signal measuring how well the genome served it:

```json
{
  "context_health": {
    "ellipticity": 0.82,
    "coverage": 0.75,
    "density": 0.68,
    "freshness": 1.0,
    "genes_expressed": 3,
    "genes_available": 42,
    "status": "aligned"
  }
}
```

| Status | Ellipticity | Meaning |
|--------|-------------|---------|
| `aligned` | >= 0.7 | Genome is well-grounded, model is informed |
| `sparse` | >= 0.3 | Gaps exist, model may guess on some topics |
| `stale` | any | Expressed genes are outdated (low freshness) |
| `denatured` | < 0.3 | Context is unreliable, high hallucination risk |

### Cross-Store Import (HGT)

Export a genome and import it into another Helix instance:

```bash
# Export
python examples/hgt_transfer.py export -d "Project knowledge snapshot"

# Preview what an import would change
python examples/hgt_transfer.py diff genome_export.helix

# Import into another instance
python examples/hgt_transfer.py import genome_export.helix
```

Three merge strategies: `skip_existing` (safe default), `overwrite`, `newest`.
Content-addressed gene IDs ensure deduplication across instances.

### Associative Memory

Genes that are frequently expressed together build co-activation links. When you query for topic A, the genome also pulls in topic B if they've been co-expressed before. This creates an organic associative memory that grows smarter over time.

### Multi-Agent Identity (4-layer attribution)

Every gene is attributed across a 4-layer identity chain at ingest time.
This makes multi-agent deployments first-class: one agent's context can
surface another agent's work, queries can scope by whichever layer is
load-bearing, and enterprise deployments can audit authorship without
retrofitting a permissions model.

| Layer | Meaning | Example |
|---|---|---|
| `org` | External account / oauth / email-level identity | `swift_wing21@github` |
| `party` | Device | `max-desktop`, `max-laptop` |
| `participant` | Human user on that device | `max`, `todd` |
| `agent` | Agent session / tool call / sub-agent | `laude-vscode-left`, `raude-mcp-pid42` |

**Resolution is trust-on-first-use.** Clients identify themselves via
env vars (`HELIX_ORG` / `HELIX_DEVICE` / `HELIX_USER` / `HELIX_AGENT`)
with OS-level fallbacks (`getpass`, hostname) — no auth layer required
for local/single-user deployments. The registry is additive: older
pre-4-layer clients still work and just inherit local-tier defaults.

**What the layering buys you:**

- **Scoped retrieval** — "what did Laude do on this device tonight"
  (agent + party), "what does Max's org know about X" (org), "what
  did the human type themselves vs what did an agent write"
  (participant vs agent).
- **Presence genes** — each participant gets a `presence:{participant_id}`
  gene that agents write to via heartbeat. One agent's query can retrieve
  another's current state by direct lookup, bypassing BM25 noise.
- **LLM-to-LLM coordination** — sibling sessions (Laude / Raude / Taude)
  see each other's recent work via `GET /sessions/{handle}/recent`,
  which is chronological and doesn't drown in the larger genome corpus.

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions/register` | POST | Register a participant under a party (trust-on-first-use) |
| `/sessions/{participant_id}/heartbeat` | POST | Keepalive + optional presence-gene body emit |
| `/sessions` | GET | List live participants (active / idle / stale / gone) |
| `/sessions/{handle}/recent` | GET | Chronological recent genes authored by `handle` (bypasses BM25) |

Full design spec (and historical context on the 2-layer → 4-layer
migration): [`docs/SESSION_REGISTRY.md`](docs/SESSION_REGISTRY.md).

### Task-Conditioned Retrieval (MoE + Small Models)

MoE models (Gemma 4) and sub-3.2B models can't reliably "look back" across a 15K context
window. Helix auto-detects these architectures and switches to a tissue-specific expression
mode inspired by how cell types selectively express genes from the same genome:

1. **Answer slate** — pre-extracted `key=value` facts front-loaded in the first ~200 tokens,
   inside every sliding-window attention layer (Gemma 4's 5:1 SWA ratio means 5 of 6 layers
   only see 1,024-token windows).
2. **Relevance-first gene ordering** — highest-scoring gene at position 0, not sorted by
   source sequence. Guarantees the best match lands inside every attention window.
3. **Think suppression** — `/no_think` injection + temp=0 for small models that otherwise
   waste their output budget on reasoning loops.

Measured impact on gemma4:e4b:

| Mode | Retrieval | Accuracy |
|------|-----------|----------|
| Standard expression | 10/10 | 5/10 |
| MoE tissue expression | 10/10 | **9/10** |

Dense models (qwen3 family) automatically use the standard expression path and are
unaffected. Detection is per-request based on the downstream model name, so the same
server can handle mixed clients.

### Synonym Expansion

Configure lightweight query expansion in `helix.toml`:

```toml
[synonyms]
cache = ["redis", "ttl", "invalidation", "cdn"]
auth = ["jwt", "login", "security", "token"]
```

When a user asks about "cache", the genome also searches for "redis", "ttl", etc.

## HTTP Endpoints

### Core endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy (primary integration) |
| `/ingest` | POST | Ingest content into genome: `{content, content_type, metadata?}` |
| `/context` | POST | Query genome for context: `{query}` (Continue format) |
| `/consolidate` | POST | Distill session buffer into knowledge genes |
| `/stats` | GET | Genome metrics, compression ratio, health |
| `/health` | GET | Server status, ribosome model, gene count |
| `/health/history` | GET | Recent query health signals (`?limit=N`) |

### Admin / maintenance endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/refresh` | POST | Reopen the genome connection to see external writes |
| `/admin/vacuum` | POST | Reclaim free SQLite pages after thinning (returns before/after size) |
| `/admin/kv-backfill` | POST | Run CPU regex KV extraction on genes missing `key_values` |
| `/replicas` | GET | List replica status (sync lag, paths) |
| `/replicas/sync` | POST | Force-sync all replicas from the master genome |
| `/bridge/status` | GET | Shared-memory bridge status (inbox, signals) |
| `/bridge/collect` | POST | Ingest pending files from the shared bridge inbox |
| `/bridge/signal` | POST | Write a named signal to the shared bridge |

### Four operations that sound similar — but do different things

These are the most confused operations in the admin surface. Know which one to reach for:

| Operation | What it does | When to use |
|-----------|--------------|-------------|
| **`checkpoint(mode)`** | Flush WAL log into the main DB file. No file size change. | During/after bulk ingest, to guarantee data is durable before a crash. Automatic every 50 inserts. |
| **`refresh()`** / `/admin/refresh` | Close and reopen the long-lived DB connection so it picks up writes made by external processes. | After running a thinning script, ingest worker, or any out-of-band write. Cheap, non-destructive. |
| **`compact()`** | Scan every gene's `source_id`, mtime-check the file, mark source-changed genes as `AGING`. **Does not delete or shrink anything.** | Periodic source-staleness detection (runs automatically every `compact_interval` seconds). |
| **`vacuum()`** / `/admin/vacuum` | Rewrite the SQLite file to reclaim free pages from previous deletions. **Shrinks the file.** | After large thinning operations. Blocking — run during maintenance windows only. Our 7.2K-gene genome reclaimed 229 MB (30%) on first VACUUM. |

**Rule of thumb:**
- If you care about **durability** → `checkpoint()`
- If you care about **visibility** (seeing external writes) → `refresh()`
- If you care about **staleness** (detecting changed sources) → `compact()`
- If you care about **disk space** → `vacuum()`

## Continue IDE Integration

Add to `~/.continue/config.yaml`:

```yaml
models:
  - name: Helix (Local)
    provider: openai
    model: gemma4:e4b
    apiBase: http://127.0.0.1:11437/v1
    apiKey: EMPTY
    roles: [chat]
    defaultCompletionOptions:
      contextLength: 128000
      maxTokens: 4096
```

Use **Chat** mode (not Agent mode). Set `contextLength` high so Continue sends the full message; Helix handles compression downstream.

## Python API

```python
from helix_context import HelixContextManager, load_config

config = load_config()
helix = HelixContextManager(config)

# Ingest content
helix.ingest("Your document text here", content_type="text")
helix.ingest(open("src/main.py").read(), content_type="code")

# Build context for a query
window = helix.build_context("How does auth work?")
print(window.expressed_context)
print(window.context_health.status)  # "aligned" / "sparse" / "denatured"

# Learn from an exchange
helix.learn("How does auth work?", "JWT middleware validates tokens...")

# Export genome
from helix_context.hgt import export_genome
export_genome(helix.genome, "project.helix", description="Auth system knowledge")
```

## ScoreRift Integration

Helix includes a bridge to [ScoreRift](https://github.com/SwiftWing21/scorerift) for divergence-based context health monitoring:

```python
from helix_context.integrations.scorerift import GenomeHealthProbe, cd_signal

# Probe genome health
probe = GenomeHealthProbe("http://127.0.0.1:11437")
report = probe.full_scan()

# Register as ScoreRift dimensions
from helix_context.integrations.scorerift import make_genome_dimensions
engine.register_many(make_genome_dimensions())

# Feed divergence resolutions back into the genome
from helix_context.integrations.scorerift import resolution_to_gene
resolution_to_gene("security", auto_score=0.85, manual_score=1.0,
                   resolution="False positives in auth module scanner rules")
```

## Configuration

All config in `helix.toml`:

```toml
# [ribosome] — OPTIONAL. The /context retrieval path is LLM-free by
# default. This section only kicks in if you explicitly enable one of
# the ribosome ops (query expansion, rerank, or ingest-time pack).
# Think "subconscious layer" — reflective re-processing during idle,
# not a hot-path dependency.
[ribosome]
backend = "ollama"                  # "ollama" (local) | "claude" | "deberta" | "litellm" — only consulted when a ribosome op explicitly runs
model = "gemma4:e2b"                # Ollama model for pack/replicate (light, ~2GB VRAM)
base_url = "http://localhost:11434"
timeout = 30
keep_alive = "30m"
warmup = false                      # pre-load on server start; false keeps /context zero-LLM out of the box
query_expansion_enabled = false     # Step 0 LLM query-intent expansion. Flip true for ~2-3pp on ambiguous queries at the cost of one ribosome call per request.

# Partner / vendor hook — route the ribosome through a hosted model
# (Anthropic, Google, etc.) for compression/extraction testing without
# forking. Uncomment and set your key (HELIX / ANTHROPIC_API_KEY).
# backend = "claude"
# claude_model = "claude-haiku-4-5-20251001"   # haiku = cost-effective bulk; swap to claude-sonnet-4-6 for higher resolution
# claude_base_url = ""                         # "" = direct Anthropic API; set to a proxy URL to route through a gateway

[budget]
ribosome_tokens = 3000
expression_tokens = 12000   # 15K total per turn (decoder + expression)
max_genes_per_turn = 12
splice_aggressiveness = 0.3
decoder_mode = "condensed"  # full | condensed | minimal | none

[genome]
path = "genome.db"
cold_start_threshold = 10
replicas = ["C:/helix-cache/genome.db", "E:/helix-cache/genome.db"]
replica_sync_interval = 100

[ingestion]
backend = "cpu"             # "cpu" (spaCy+regex, fast) | "ollama" (LLM, slow)
splade_enabled = true       # SPLADE sparse expansion at index time
entity_graph = true         # entity-based co-activation links

[server]
host = "127.0.0.1"
port = 11437
upstream = "http://localhost:11434"

[synonyms]
cache = ["redis", "ttl", "invalidation", "cdn"]
auth = ["jwt", "login", "security", "token"]
```

**Environment variables:**
- `OLLAMA_KV_CACHE_TYPE=q4_0` — INT4 KV cache quantization (recommended).
  q8_0 tested but produced WORSE accuracy (gave models more room to hallucinate in
  think mode). q4_0 is faster, more accurate, and uses less VRAM.
- `HELIX_CONFIG=/path/to/helix.toml` — override config file location

## Testing

```bash
# Mock tests only (no Ollama needed, ~8s)
pytest tests/ -m "not live"

# Live tests (requires Ollama)
pytest tests/ -m live -v -s

# Full suite
pytest tests/ -v
```

## Benchmarks

```bash
# Needle-in-a-haystack (single model)
HELIX_MODEL=qwen3:4b python benchmarks/bench_needle.py

# Full sweep across all local models
python benchmarks/bench_sweep.py
```

See [docs/RESEARCH.md](docs/RESEARCH.md#benchmark-results--scale-invariant-knowledge-engine-sike)
for full SIKE analysis and results across 7 local models + 3 Claude API tiers.

## Architecture

| Module | Role |
|--------|------|
| `schemas.py` | Gene, ContextWindow, ContextHealth, ChromatinState |
| `codons.py` | CodonChunker (text/code splitting) + CodonEncoder (serialization) |
| `genome.py` | SQLite genome with promoter-tag retrieval + co-activation |
| `ribosome.py` | Small-model codec: pack, re_rank, splice, replicate |
| `context_manager.py` | 6-step pipeline orchestrator + pending replication buffer |
| `server.py` | FastAPI proxy + standalone endpoints |
| `config.py` | TOML config loader with synonym map |
| `hgt.py` | Genome export/import (Horizontal Gene Transfer) |
| `integrations/scorerift.py` | CD spectroscope bridge to ScoreRift |

## Origin

Built as a standalone package extracted from [BigEd CC](https://github.com/SwiftWing21/Education). Implements the "Ribosome Hypothesis" for local LLM context management.

## Acknowledgments

Helix Context uses the following third-party libraries; we are grateful to
their authors and maintainers.

- **[Headroom](https://github.com/chopratejas/headroom)** by **Tejas Chopra** ([@chopratejas](https://github.com/chopratejas)) — CPU-resident semantic compression for gene content at the retrieval seam. Kompress (ModernBERT ONNX), LogCompressor, DiffCompressor, and CodeAwareCompressor replace the legacy character-level truncation in the expression pipeline. Optional dependency, installed via `pip install helix-context[codec]`. Apache-2.0. See [NOTICE](NOTICE) for full attribution.

## License

Apache 2.0


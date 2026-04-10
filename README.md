# 🧬 Helix Context

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0b1-orange.svg)](https://pypi.org/project/helix-context/)
[![SIKE: 10/10](https://img.shields.io/badge/SIKE-10%2F10_retrieval-brightgreen.svg)](docs/RESEARCH.md#benchmark-results--scale-invariant-knowledge-engine-sike)
[![Compression: 769x](https://img.shields.io/badge/inference_compression-769x-brightgreen.svg)](docs/RESEARCH.md)
[![Paper: Agentome](https://img.shields.io/badge/paper-Agentome-purple.svg)](https://mbachaud.substack.com/p/agentome)

**Genome-based context compression for local LLMs.**
**Scale-Invariant Knowledge Engine (SIKE) — 10/10 retrieval from 0.6B to 26B parameters.**

> Treats context like a genome instead of a flat text buffer. A 7,200-gene SQLite
> database (44MB raw knowledge) compresses to ~15K tokens of expressed context
> per turn — a **769x inference compression ratio**. Retrieval is perfectly
> scale-invariant: the same genome delivers 10/10 needle accuracy to qwen3:0.6b
> and Claude Opus alike. *The Librarian does the work; the Reader just extracts.*

> **📖 Quick glossary** — If the biological metaphor is new to you:
> **gene** = one knowledge chunk (content + metadata) · **genome** = the full SQLite store ·
> **ribosome** = small model that packs/ranks/splices context · **promoter** = retrieval tags ·
> **expression** = selecting + formatting genes for one query · **chromatin** = gene accessibility tier
> (open / euchromatin / heterochromatin) · **replication** = packing conversations back into the genome.

<details>
<summary><b>📑 Table of Contents</b></summary>

- [Benchmark Highlights](#benchmark-highlights)
- [Quick Start](#quick-start)
- [What You'll See](#what-youll-see)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
  - [Context Health Monitor (Delta-Epsilon)](#context-health-monitor-delta-epsilon)
  - [Horizontal Gene Transfer (HGT)](#horizontal-gene-transfer-hgt)
  - [Associative Memory](#associative-memory)
  - [Tissue-Specific Expression (MoE + Small Models)](#tissue-specific-expression-moe--small-models)
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
  +--------------------------+
  |  Helix Proxy (FastAPI)   |  Port 11437
  |  /v1/chat/completions    |  OpenAI-compatible
  |                          |
  |  1. Extract query        |
  |  2. Express pipeline     |  <-- Genome (SQLite)
  |  3. Inject context       |  <-- Ribosome (CPU model)
  |  4. Forward to Ollama    |  --> localhost:11434
  |  5. Stream tee response  |
  |  6. Background replicate |
  +--------------------------+
```

Instead of stuffing your entire codebase into the prompt, Helix compresses it into a persistent SQLite genome and expresses only the relevant genes per turn. The model sees compressed context, not raw text. Conversations replicate back into the genome automatically, building institutional memory over time.

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

### Database Storage Breakdown

The on-disk `genome.db` is **752 MB** for 7,264 genes (~46 MB of raw content).
Why the 16x gap between raw content and DB file? Because the genome isn't just storage —
it's a **4-tier retrieval engine** (promoter tags → FTS5 → SPLADE → ΣĒMA semantic), and
each tier carries its own index.

| Component | Size | Purpose |
|---|---:|---|
| **Raw content** (`gene.content`) | 44.5 MB | Original source text, verbatim |
| **Ribosome complements** (`gene.complement`) | 16.5 MB | Small-model compressed summaries (2.69x storage ratio) |
| **FTS5 posting lists** (`genes_fts_data`) | 187 MB | Full-text inverted index for keyword retrieval |
| **SPLADE sparse index** (`splade_terms`) | 36 MB | 1.73M term weights for lexical expansion |
| **Promoter index** (retrieval tags) | 3.8 MB | 73,815 domain/entity tags across all genes |
| **Entity graph** | 5.6 MB | 117K entity-to-gene edges for co-activation |
| **Gene relations** (NLI) | 6.6 MB | 108K typed logical relations between genes |
| **ΣĒMA embeddings** (20D vectors) | 0.34 MB | Semantic primes — 80 bytes per gene |
| **Key-value facts** (pre-extracted) | 1.4 MB | Pre-parsed `key=value` pairs for answer slate |
| **Codons + promoter JSON + epigenetics** | 8.2 MB | Gene metadata (tags, access counts, decay) |
| **SQLite B-tree overhead + free pages** | ~441 MB | Post-thinning fragmentation (7,075 of 11,529 genes deleted, space not reclaimed) |
| **Total file size** | **752 MB** | |

**Observations:**

- **FTS5 dominates storage** (25% of the file). The full-text index holds position
  data for every token across all 7K genes — it's what enables the sub-5ms content
  queries that make the 1s total retrieval latency possible.
- **Raw content is only 6% of the file**. The rest is indexes. This is the expected
  tradeoff for a retrieval-optimized database vs a flat text archive.
- **~440 MB is fragmentation**, not real data. The genome was thinned from 11,529
  to 7,264 genes during tuning, and SQLite holds those pages until a `VACUUM`. A
  compacted genome would land around **~310 MB for the same content**.
- **ΣĒMA embeddings are essentially free** — 20 floats per gene = 80 bytes. A 1M-gene
  genome would cost only 80MB for the semantic tier.
- **Inference cost is unchanged by DB size**: the LLM only ever sees ~15K tokens
  per turn regardless of whether the genome is 50MB or 50GB.

**Compression summary:**

| Metric | Ratio | Meaning |
|---|---:|---|
| Storage (raw → complement) | **2.69x** | How much the ribosome compresses each gene's summary |
| Expression (full corpus → single turn) | **776x** | How much of the genome the LLM sees per query |
| DB file / raw content | 16x (6x post-VACUUM) | Index overhead for 4-tier retrieval |
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

**6-step expression pipeline per turn:**

| Step | What | Cost | Blocking? |
|------|------|------|-----------|
| 1. Extract | Heuristic keyword extraction from query | 0 tokens | No |
| 2. Express | SQLite promoter lookup + synonym expansion + co-activation | 0 tokens | No |
| 3. Re-rank | Small CPU model scores candidates by relevance | ~300 tokens | Yes |
| 4. Splice | Small CPU model trims introns, keeps exons (batched) | ~600 tokens | Yes |
| 5. Assemble | Join spliced parts, enforce token budget, wrap in tags | 0 tokens | No |
| 6. Replicate | Pack query+response exchange back into genome | ~300 tokens | No (background) |

**Token budget:**
- 3k tokens: ribosome decoder prompt (fixed, tells the big model how to read codons)
- 12k tokens: expressed context (dense XML gene format, 12 genes per turn)
- 11M+ tokens: genome cold storage (SQLite, ~46MB raw on a mature project)

**Compression metrics:**
- Storage: 2.7x (raw content → ribosome complements)
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

### Horizontal Gene Transfer (HGT)

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

### Tissue-Specific Expression (MoE + Small Models)

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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy (primary integration) |
| `/ingest` | POST | Ingest content into genome: `{content, content_type, metadata?}` |
| `/context` | POST | Query genome for context: `{query}` (Continue format) |
| `/stats` | GET | Genome metrics, compression ratio, health |
| `/health` | GET | Server status, ribosome model, gene count |

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
[ribosome]
model = "gemma4:e4b"        # context codec for pack/re_rank/splice
backend = "ollama"          # or "deberta" for faster CPU-only ribosome
timeout = 30                # seconds before fallback
keep_alive = "30m"          # keep model loaded (eliminates swap latency)
warmup = true               # pre-load model on server start

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

## License

Apache 2.0

# Helix Context

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/badge/pypi-v0.4.0b1-orange.svg)](https://pypi.org/project/helix-context/)
[![Packet bench: 10/10](https://img.shields.io/badge/packet_bench-10%2F10_5_families-brightgreen.svg)](benchmarks/results/packet_bench_2026-04-18.json)
[![LLM-free pipeline](https://img.shields.io/badge/pipeline-LLM--free-brightgreen.svg)](docs/architecture/PIPELINE_LANES.md)
[![Paper: Agentome](https://img.shields.io/badge/paper-Agentome-purple.svg)](https://mbachaud.substack.com/p/agentome)

**A coordinate index layer for LLM context — Helix weighs, doesn't retrieve.**

The card catalog for your existing stores. Helix returns a
pointer, a confidence, and a *verdict* — not a bag of content.
Your agent asks "do I know this, or do I need to go look?" and Helix
answers deterministically: `verified / stale_risk / needs_refresh`.
Fetching, if any, is the caller's job.

Composes on top of the bundled SQLite genome today. Stacks on
Postgres / S3 / git repos / mempalace-style stores tomorrow. The
value isn't where the bytes live — it's that the *locating* is
crisp and the *confidence* is real.

> **The whole data pipeline is LLM-free** — and this is load-bearing.
> Ingest, tagging, scoring, fusion, chromatin gating, freshness
> weighing, and packet labeling are pure CPU math. spaCy NER, Howard
> 2005 TCM, Stachenfeld 2017 SR, Werman 1986 W1, Hebbian co-activation.
> MiniLM (SEMA, 384d → 20d) and DeBERTa (optional rerank) are small
> transformer encoders, not generators. The only LLM call in the whole
> system is the final answer at `/v1/chat/completions`. Pre-2026-04-09
> the pipeline had an LLM pack step adding ~30 s per ingest; now there
> is zero LLM cost on the retrieval or weighing hot paths. Agents
> consume `/context/packet` without LLM-in-the-loop latency.

<details>
<summary><b>📑 Table of Contents</b></summary>

- [Two Product Surfaces](#two-product-surfaces)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Weighing Layer](#weighing-layer)
- [Key Features](#key-features)
- [HTTP Endpoints](#http-endpoints)
- [MCP Tools](#mcp-tools)
- [Python API](#python-api)
- [Continue IDE Integration](#continue-ide-integration)
- [ScoreRift Integration](#scorerift-integration)
- [Configuration](#configuration)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Architecture](#architecture)
- [Acknowledgments](#acknowledgments)
- [License](#license)

</details>

## Two product surfaces

Helix exposes two retrieval surfaces. Pick by caller type.

```
                ┌─ /context/packet ──► agent-safe.  Pointers + verdict + refresh plan.
CLIENT ─► HELIX ┤                     Agent decides know-vs-go.
                └─ /context ────────► decoder.  Helix assembles + compresses the
                                      context window.  Downstream LLM consumes directly.
```

|  | `/context/packet` | `/context` |
|---|---|---|
| Returns | gene_id + source_id + **verdict** + refresh plan | assembled `expressed_context` (compressed) |
| Caller fetches content? | **yes** (from source_path) | no — Helix did it |
| Task-sensitive? | **yes** — plan / explain / review / edit / debug / ops / quote | no — one compression profile |
| Emits "known-empty"? | **yes** — first-class verdict | no — always returns something |
| Use for | MCP agents, tool use, programmatic decisions | chat clients, Continue-style prompts |

Both compose with the **weighing layer** (freshness + coordinate
confidence). Packet mode makes the verdict primary; `/context` surfaces
the same signals in `ContextHealth` so decoder callers can still
inspect.

See [`docs/architecture/PIPELINE_LANES.md`](docs/architecture/PIPELINE_LANES.md) for the full swim-lane
reference and
[`docs/specs/2026-04-17-agent-context-index-build-spec.md`](docs/specs/2026-04-17-agent-context-index-build-spec.md)
for the authoritative packet-mode spec.

## Quick Start

### Launch

Three launch modes, three different scopes:

| Launcher | OTel | Budget Zone | Tray | Supervisor | Use case |
|---|---|---|---|---|---|
| `start-helix-tray.bat` | ✅ | ✅ | ✅ | ✅ | **canonical daily driver** |
| `backend-with-otel.bat` | ✅ | ❌ | ❌ | ❌ | dev / debug — direct uvicorn |
| `python -m uvicorn helix_context.server:app --host 127.0.0.1 --port 11437` | ❌ | ❌ | ❌ | ❌ | minimal (no metrics, no zone cap) |

```bash
pip install helix-context[launcher] --pre

# Canonical path (tray + OTel + budget zone + supervisor)
start-helix-tray.bat       # Windows
./start-helix-tray.sh      # Linux/macOS (coming — use helix-launcher for now)

# Check status from another terminal
helix-status
```

Point any OpenAI-compatible client at `http://127.0.0.1:11437/v1` and
chat. Context compression happens transparently through `/context`.

### Seed the genome

```bash
# Seed from a real project
python examples/seed_genome.py path/to/your/project/

# Confirm it landed
curl -s http://127.0.0.1:11437/stats | jq '.total_genes, .compression_ratio'
```

Fresh ingests auto-populate provenance metadata (`source_kind`,
`volatility_class`, `observed_at`, `last_verified_at`) via file
extension inference. For existing genomes ingested before
`v0.4.0b1`, run the backfill once:

```bash
python scripts/backfill_gene_provenance.py --dry-run  # preview
python scripts/backfill_gene_provenance.py            # apply
```

### Agent / MCP setup

```json
{
  "mcpServers": {
    "helix-context": {
      "command": "python",
      "args": ["-m", "helix_context.mcp_server"],
      "env": {
        "HELIX_MCP_URL": "http://127.0.0.1:11437",
        "HELIX_AGENT": "your-agent-handle"
      }
    }
  }
}
```

The canonical MCP entrypoint is `helix_context.mcp_server`. It
exposes `helix_context_packet` and `helix_refresh_targets` (the
agent-safe tools) alongside `helix_context`, `helix_stats`,
`helix_ingest`, `helix_resonance`, and the session/HITL toolkit.

## How It Works

### The 3-layer view

```
┌──────────────────────────────────────────────────────────────────┐
│  Weighing layer (LLM-free)                                       │
│                                                                  │
│    coord_conf  ×  (freshness × authority × specificity)          │
│    └─ location ─┘   └─────── content trust ───────┘              │
│    "did we resolve   "is what we resolved to still trustworthy?" │
│     to the right                                                 │
│     place?"                                                      │
│                                                                  │
│    Output: verified / stale_risk / needs_refresh + refresh plan  │
└──────────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────────┐
│  Retrieval layer (LLM-free, 12 signals + 1 octave gate)          │
│                                                                  │
│    path_key_index • promoter tags • FTS5 • SPLADE • SEMA cold   │
│    harmonic_links • cymatics resonance+flux • TCM drift • ...    │
│    party_id octave gate (multi-tenant scoping)                   │
└──────────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────────┐
│  Storage layer (genome.db — replaceable)                         │
│                                                                  │
│    genes + provenance + encoders + attribution + relations       │
└──────────────────────────────────────────────────────────────────┘
```

Helix is the top two layers. The bottom layer is a genome.db today
and could be Postgres / S3 / an external vector store tomorrow — the
coordinate resolution math lives in Helix, the content can live
anywhere.

### Packet mode in 30 seconds

```bash
curl -s http://127.0.0.1:11437/context/packet \
  -H "Content-Type: application/json" \
  -d '{"query":"helix auth config port","task_type":"ops"}' | jq
```

```json
{
  "task_type": "ops",
  "query": "helix auth config port",
  "verified": [
    {
      "kind": "gene",
      "gene_id": "4f98e2f4296d7620",
      "title": "helix.toml",
      "source_id": "/repo/helix-context/helix.toml",
      "source_kind": "config",
      "volatility_class": "hot",
      "last_verified_at": 1776539103.1,
      "status": "verified",
      "relevance_score": 8.42,
      "live_truth_score": 0.96
    }
  ],
  "stale_risk": [],
  "refresh_targets": [],
  "notes": []
}
```

For an off-target query (retrieval in the wrong folder region), the
coordinate-confidence warning fires and items downgrade to
`needs_refresh`:

```json
{
  "notes": [
    "coordinate_confidence=0.12 below 0.30 floor — retrieval may not have located the right coordinate region"
  ],
  "refresh_targets": [
    {"target_kind": "doc", "source_id": "/repo/two-brain-audit/README.md",
     "reason": "fresh verification required before action", "priority": 0.85}
  ]
}
```

## Weighing Layer

The center of gravity for agent interactions. Two half-signals
compose into one verdict.

**Freshness × authority × specificity (content trust):**

| Signal | Source | Half-lives |
|---|---|---|
| `freshness_score` | `exp(-age / half_life[volatility])` | stable=7d, medium=12h, hot=15min |
| `authority_score` | `authority_class` lookup | primary=1.0, derived=0.75, inferred=0.45 |
| `specificity_score` | source_kind + support_span | literal=1.0, span=0.9, doc=0.75, assertion=0.45 |

**Coord confidence (location):**

`path_token_coverage(query, delivered_genes)` — fraction of
delivered top-K whose `source_path` tokens overlap the extracted query
signals. Validated on the 10-needle bench (hit mean 1.00 / miss mean
0.52, Δ +0.48). Below 0.30 the verdict downgrades regardless of
freshness.

**Task sensitivity:**

| task_type | freshness ≥ verified | coord < 0.30 effect | intent |
|---|---|---|---|
| `plan` / `explain` | 0.35 | stale_risk | low-risk, tolerant |
| `review` | 0.55 | stale_risk | moderate |
| `edit` / `debug` | 0.70 | **needs_refresh** | high-risk action |
| `ops` / `quote` | 0.70 | **needs_refresh** | literal-answer, no tolerance |

Full spec: [`docs/specs/2026-04-17-agent-context-index-build-spec.md`](docs/specs/2026-04-17-agent-context-index-build-spec.md).
Bench validation: [`benchmarks/bench_packet.py`](benchmarks/bench_packet.py) — 10/10 across 5 families.

## Key Features

### Provenance at Ingest

Every ingest auto-populates 4 provenance fields based on the source
path. No backfill needed after `v0.4.0b1`:

- `source_kind` — inferred from extension (40+ mappings → code/config/doc/log/db)
- `volatility_class` — derived from kind (matches packet half-lives)
- `observed_at` + `last_verified_at` — ingest timestamp

Non-path source_ids (like `"__session__"` or `"agent:laude"`) are
deliberately left NULL — the packet builder treats unknown
provenance as unknown, not as fresh. Centralized inference in
[`helix_context/provenance.py`](helix_context/provenance.py).

### 4-Layer Multi-Agent Attribution

Every gene is attributed across 4 identity layers at ingest:

| Layer | Meaning | Example |
|---|---|---|
| `org` | External account / oauth / email | `swiftwing21@github` |
| `party` | Device | `max-desktop` |
| `participant` | Human user on that device | `max` |
| `agent` | Agent session / tool call | `laude-vscode-left` |

**Trust-on-first-use.** Clients identify via env vars (`HELIX_ORG /
HELIX_DEVICE / HELIX_USER / HELIX_AGENT`) with OS-level fallbacks
(`getpass`, hostname) — no auth layer required for local
deployments. Opt out per request with `"local_federation": false`.

Spec: [`docs/architecture/SESSION_REGISTRY.md`](docs/architecture/SESSION_REGISTRY.md).

### Associative Memory

Genes that are frequently expressed together build co-activation
links (Hebbian `harmonic_links`, seeded-edge updates). Querying for
topic A pulls in topic B if they've been co-expressed before. Grows
smarter with use.

### Cross-Store Import (HGT)

Export a genome and import it into another Helix instance:

```bash
python examples/hgt_transfer.py export -d "Project knowledge snapshot"
python examples/hgt_transfer.py diff genome_export.helix       # dry-run
python examples/hgt_transfer.py import genome_export.helix
```

Three merge strategies: `skip_existing` (safe default),
`overwrite`, `newest`. Content-addressed gene IDs ensure dedup.

### Task-Conditioned Retrieval (MoE + Small Models)

Sub-3.2B models and MoE architectures (Gemma 4) can't reliably
"look back" across a 15k context window. Helix auto-detects these
architectures and switches to a tissue-specific expression mode:

1. **Answer slate** — pre-extracted `key=value` facts front-loaded
   in the first ~200 tokens, inside every sliding-window layer.
2. **Relevance-first gene ordering** — highest-scoring gene at
   position 0 (not sequence-sorted).
3. **Think suppression** — `/no_think` + temp=0 for models that
   waste their output budget on reasoning loops.

Auto-detected per-request from the downstream model name.

### Synonym Expansion

Configure lightweight query expansion in `helix.toml`:

```toml
[synonyms]
cache = ["redis", "ttl", "invalidation", "cdn"]
auth = ["jwt", "login", "security", "token"]
```

## HTTP Endpoints

### Retrieval

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/context` | POST | Decoder path — assembled compressed context for downstream LLM |
| `/context/packet` | POST | **Agent-safe index path** — pointers + verdict + refresh plan |
| `/context/refresh-plan` | POST | Just the reread plan (thin wrapper over `/context/packet`) |
| `/fingerprint` | POST | Navigation-first — scored gene pointers with `score_floor` + accounting |
| `/v1/chat/completions` | POST | OpenAI-compatible proxy (primary chat integration) |

### Ingest / Lifecycle

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | `{content, content_type, metadata?}` — provenance auto-populated |
| `/consolidate` | POST | Distill session buffer into knowledge genes |
| `/stats` | GET | Genome metrics, compression ratio, health |
| `/health` | GET | Server status, ribosome model, gene count |

### Admin / Maintenance

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/refresh` | POST | Reopen genome connection to see external writes |
| `/admin/vacuum` | POST | Reclaim free SQLite pages after thinning |
| `/admin/kv-backfill` | POST | Run CPU regex KV extraction on legacy genes |
| `/admin/announce_restart` | POST | Signal downstream observers before a planned kill |

### Four operations that sound similar

| Operation | What | When |
|---|---|---|
| `checkpoint()` | Flush WAL → main DB file | After bulk ingest, for durability |
| `refresh()` | Reopen DB connection | See writes from an external process |
| `compact()` | Mark source-changed genes `AGING` | Periodic staleness detection (auto) |
| `vacuum()` | Rewrite file, reclaim free pages | After large thinning — blocking |

## MCP Tools

The canonical MCP server (`python -m helix_context.mcp_server`) exposes:

**Retrieval:**
- `helix_context` — main retrieval (decoder path)
- `helix_context_packet` — agent-safe packet with freshness + coord verdict
- `helix_refresh_targets` — just the reread plan for edit/ops tasks
- `helix_stats`, `helix_ingest`, `helix_resonance`, `helix_consolidate`

**Session / attribution:**
- `helix_sessions_list`, `helix_session_recent`

**HITL events:**
- `helix_hitl_emit`, `helix_hitl_recent`

**Operational:**
- `helix_health`, `helix_metrics_tokens`, `helix_bridge_status`

**Introspection:**
- `helix_gene_get`, `helix_neighbors`, `helix_splice_preview`

Plus `document_*` aliases per [`docs/ROSETTA.md`](docs/ROSETTA.md)
for software-vocabulary consumers.

## Python API

```python
from helix_context import HelixContextManager, load_config
from helix_context.context_packet import build_context_packet, get_refresh_targets

config = load_config()
helix = HelixContextManager(config)

# Ingest — provenance auto-populated
helix.ingest(open("src/main.py").read(), content_type="code",
             metadata={"path": "/repo/src/main.py"})

# Decoder path — full assembled context
window = helix.build_context("How does auth work?")
print(window.expressed_context)
print(window.context_health.status)  # aligned / sparse / denatured
print(window.context_health.resolution_confidence)  # 0.0-1.0

# Index path — agent-safe packet
packet = build_context_packet("How does auth work?",
                              task_type="edit",
                              genome=helix.genome)
for item in packet.verified:
    print(f"{item.status}: {item.source_id} (truth={item.live_truth_score:.2f})")
for target in packet.refresh_targets:
    print(f"reread: {target.source_id} ({target.reason})")

# Learn from an exchange
helix.learn("How does auth work?", "JWT middleware validates tokens...")
```

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

Use **Chat** mode. Set `contextLength` high so Continue sends the
full message; Helix handles compression downstream.

## ScoreRift Integration

```python
from helix_context.integrations.scorerift import (
    GenomeHealthProbe, make_genome_dimensions, resolution_to_gene,
)

# Probe genome health
report = GenomeHealthProbe("http://127.0.0.1:11437").full_scan()

# Register as ScoreRift dimensions
engine.register_many(make_genome_dimensions())

# Feed divergence resolutions back into the genome
resolution_to_gene("security", auto_score=0.85, manual_score=1.0,
                   resolution="False positives in auth scanner rules")
```

## Configuration

All config in `helix.toml`. Defaults are LLM-free.

```toml
[budget]
ribosome_tokens = 3000
expression_tokens = 12000      # 15K total per turn (decoder + expression)
max_genes_per_turn = 12
splice_aggressiveness = 0.3
decoder_mode = "condensed"     # full | condensed | minimal | none

[genome]
path = "genomes/main/genome.db"
cold_start_threshold = 10
replicas = []

[ingestion]
backend = "cpu"                # cpu (spaCy+regex) | ollama (LLM, slow)
splade_enabled = true
entity_graph = true

[server]
host = "127.0.0.1"
port = 11437
upstream = "http://localhost:11434"

# [ribosome] — OPTIONAL. The retrieval + weighing paths are LLM-free.
# Only kicks in if you explicitly enable a ribosome op (query expansion,
# rerank, ingest-time pack). Think "subconscious layer."
[ribosome]
backend = "ollama"
model = "gemma4:e2b"
base_url = "http://localhost:11434"
warmup = false                 # keeps /context zero-LLM out of the box
query_expansion_enabled = false

[synonyms]
cache = ["redis", "ttl", "invalidation", "cdn"]
auth = ["jwt", "login", "security", "token"]
```

**Environment variables:**
- `HELIX_OTEL_ENABLED=1` — emit metrics to the collector at `HELIX_OTEL_ENDPOINT`
- `HELIX_BUDGET_ZONE=1` — adaptive gene-cap based on caller's prompt token count
- `HELIX_{ORG,DEVICE,USER,AGENT}` — 4-layer attribution defaults
- `HELIX_CONFIG=/path/to/helix.toml` — override config file location

`start-helix-tray.bat` sets the first two automatically.

## Testing

```bash
# Mock tests only (no Ollama needed, ~30s)
pytest tests/ -m "not live"

# Packet + pipeline + server (no live deps)
pytest tests/test_context_packet.py tests/test_pipeline.py tests/test_server.py

# Full suite including live Ollama (slow, ~15 min)
pytest tests/
```

## Benchmarks

```bash
# Phase 5 packet bench — freshness + coord labeling (10/10 across 5 families)
python benchmarks/bench_packet.py

# Needle-in-a-haystack with coord confidence
HELIX_MODEL=qwen3:4b python benchmarks/bench_needle.py

# Query extraction diagnostic (per-needle path-token overlap)
python scripts/diagnose_query_extraction.py
```

Artifacts land in `benchmarks/results/`. Same-day runs overwrite;
dated artifacts are regression baselines.

Historical scale-invariance (SIKE) analysis lives in
[`docs/research/RESEARCH.md`](docs/research/RESEARCH.md). SIKE
framed Helix's 2025-era uplift pattern when an LLM was on the
ingest path; the current LLM-free pipeline reframes the value
proposition around pathway resolution + confidence rather than raw
retrieval accuracy.

## Architecture

| Module | Role |
|---|---|
| [`helix_context/schemas.py`](helix_context/schemas.py) | `Gene`, `ContextWindow`, `ContextHealth`, `ContextItem`, `ContextPacket`, `RefreshTarget` |
| [`helix_context/genome.py`](helix_context/genome.py) | SQLite genome with promoter-tag retrieval + co-activation + provenance columns |
| [`helix_context/context_manager.py`](helix_context/context_manager.py) | Decoder-path pipeline + pending replication + coord-confidence fields |
| [`helix_context/context_packet.py`](helix_context/context_packet.py) | Weighing layer — freshness × authority × specificity + coord confidence |
| [`helix_context/provenance.py`](helix_context/provenance.py) | Extension → source_kind → volatility_class inference (shared with backfill) |
| [`helix_context/server.py`](helix_context/server.py) | FastAPI: `/context`, `/context/packet`, `/context/refresh-plan`, `/fingerprint`, `/ingest`, session registry |
| [`helix_context/mcp_server.py`](helix_context/mcp_server.py) | MCP tools (retrieval, packet, session, HITL, introspection) |
| [`helix_context/shard_schema.py`](helix_context/shard_schema.py) | Phase-2 sharding scaffolding (`main.db` + `source_index`) |
| [`helix_context/hgt.py`](helix_context/hgt.py) | Genome export / import |
| [`helix_context/integrations/scorerift.py`](helix_context/integrations/scorerift.py) | CD spectroscope bridge to ScoreRift |

Full pipeline walkthrough:
[`docs/architecture/PIPELINE_LANES.md`](docs/architecture/PIPELINE_LANES.md).

> **Biology → software vocabulary.** Helix originally used biology
> terms (gene, genome, ribosome, chromatin, splice, promoter). Those
> remain valid as legacy aliases; canonical names are the software
> forms (document, knowledge store, compressor, lifecycle tier,
> assemble, tags). `Document is Gene` is literally `True` at the
> Python class level. Full mapping:
> [`docs/ROSETTA.md`](docs/ROSETTA.md).

## Acknowledgments

Helix Context uses the following third-party libraries; we are
grateful to their authors and maintainers.

- **[Headroom](https://github.com/chopratejas/headroom)** by
  **Tejas Chopra**
  ([@chopratejas](https://github.com/chopratejas)) — CPU-resident
  semantic compression for gene content at the retrieval seam.
  Kompress (ModernBERT ONNX), LogCompressor, DiffCompressor, and
  CodeAwareCompressor replace the legacy character-level truncation
  in the expression pipeline. Optional dependency, installed via
  `pip install helix-context[codec]`. Apache-2.0. See
  [NOTICE](NOTICE) for full attribution.

## License

Apache 2.0

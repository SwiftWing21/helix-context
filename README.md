# Helix Context

**Genome-based context compression for local LLMs.**

Makes 9k tokens of context window feel like 600k by treating context like a genome instead of a flat text buffer.

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

## Quick Start

```bash
# Install
pip install -e .

# Pull a small model for the ribosome (context codec)
ollama pull gemma4:e2b

# Start the proxy
python -m uvicorn helix_context.server:app --host 127.0.0.1 --port 11437

# Seed the genome with your project files
python examples/seed_genome.py path/to/your/project/

# Check genome health
curl http://127.0.0.1:11437/stats
```

Point any OpenAI-compatible client at `http://127.0.0.1:11437/v1` and start chatting. Context compression happens transparently.

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
- 6k tokens: expressed context (compressed, spliced)
- 600k+: genome cold storage (SQLite, never fully loaded)

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

Helix includes a bridge to [ScoreRift](https://github.com/SwiftWing21/two-brain-audit) for divergence-based context health monitoring:

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
model = "auto"              # auto-detect from Ollama
timeout = 10                # seconds before fallback
keep_alive = "30m"          # keep model loaded (eliminates swap latency)

[budget]
ribosome_tokens = 3000
expression_tokens = 6000
max_genes_per_turn = 8
splice_aggressiveness = 0.5  # 0=keep all, 1=ruthless trim

[genome]
path = "genome.db"
cold_start_threshold = 10   # genes needed before history stripping

[server]
port = 11437
upstream = "http://localhost:11434"

[synonyms]
cache = ["redis", "ttl", "invalidation", "cdn"]
auth = ["jwt", "login", "security", "token"]
```

## Testing

```bash
# Mock tests only (no Ollama needed, ~8s)
pytest tests/ -m "not live"

# Live tests (requires Ollama)
pytest tests/ -m live -v -s

# Full suite
pytest tests/ -v
```

165 tests across 7 test files, 18 diverse fixtures (code, essays, poems, science).

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

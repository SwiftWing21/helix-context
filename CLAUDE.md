# Helix Context

Genome-based context compression for local LLMs. Makes 9k tokens of context window feel like 600k.

## Quick Start

```bash
# Install
pip install -e .

# Ensure Ollama is running with at least one small model
ollama pull gemma4:e2b

# Start the proxy server
python -m uvicorn helix_context.server:app --host 127.0.0.1 --port 11437

# Seed the genome (in another terminal)
python examples/seed_genome.py path/to/your/docs/

# Or use the Python API directly
python examples/quickstart.py
```

## How It Works

A transparent OpenAI-compatible proxy that intercepts LLM requests and injects compressed context from a persistent SQLite genome.

**6-step pipeline per turn:**
1. **Extract** — heuristic keyword extraction from query (no model call)
2. **Express** — SQLite promoter-tag lookup + synonym expansion + co-activation
3. **Re-rank** — small CPU model scores candidates by relevance
4. **Splice** — small CPU model trims introns, keeps exons (batched single call)
5. **Assemble** — join spliced parts, enforce token budget, wrap in tags
6. **Replicate** — pack query+response exchange back into genome (background)

## Structure

| Module | Role |
|--------|------|
| `schemas.py` | Gene, ContextWindow, PromoterTags, ChromatinState |
| `exceptions.py` | 5 error types, all with fallbacks |
| `config.py` | TOML loader, synonym map, cold-start threshold |
| `codons.py` | CodonChunker (RawStrand) + CodonEncoder (Codon) |
| `genome.py` | SQLite DDL, promoter index, synonym expansion, co-activation |
| `ribosome.py` | pack/re_rank/splice/replicate + timeout fallbacks |
| `context_manager.py` | 6-step pipeline orchestrator + pending replication buffer |
| `server.py` | FastAPI proxy + /ingest, /context, /stats, /health endpoints |
| `integrations/scorerift.py` | CD spectroscope bridge to ScoreRift audit system |

## Configuration

All config lives in `helix.toml`. Key sections:

- `[ribosome]` — model, base_url, timeout, keep_alive, warmup
- `[budget]` — ribosome_tokens (3k), expression_tokens (6k default in config.py; helix.toml ships 12k override), max_genes_per_turn, splice_aggressiveness
- `[genome]` — path, compact_interval, cold_start_threshold, replicas, replica_sync_interval
- `[server]` — host, port, upstream, upstream_timeout
- `[synonyms]` — lightweight query expansion (e.g., "cache" -> ["redis", "ttl", "invalidation"])

## HTTP Endpoints

```
POST /v1/chat/completions  — OpenAI-compatible proxy (primary integration)
POST /ingest               — { content, content_type, metadata? } → gene_ids
POST /context              — { query } → Continue HTTP context provider format
POST /context/packet       — agent-safe bundle: verified / stale_risk / refresh_targets
POST /context/refresh-plan — refresh_targets only (reread plan, no evidence items)
POST /fingerprint          — navigation-first payload with tier scores, not content
POST /consolidate          — rewrite stale gene bodies from their source fingerprints
POST /sessions/register    — register an agent participant (taude/laude/...) for attribution
POST /admin/refresh        — force a retrieval-layer refresh (admin only)
POST /admin/vacuum         — reclaim SQLite pages after compaction (admin only)
GET  /stats                — genome metrics + compression ratio
GET  /health               — ribosome model, gene count, upstream URL
```

## Testing

```bash
# All mock tests (no Ollama needed, ~6s)
python -m pytest tests/ -m "not live" -v

# Live tests (requires Ollama running)
python -m pytest tests/ -m live -v -s

# Full suite
python -m pytest tests/ -v
```

## Continue IDE Integration

Add to `~/.continue/config.yaml`:

```yaml
models:
  - name: Helix (Local)
    provider: openai
    model: gemma4:e4b          # or whatever model Ollama is running
    apiBase: http://127.0.0.1:11437/v1
    apiKey: EMPTY
    roles: [chat]
    defaultCompletionOptions:
      contextLength: 128000    # Helix handles compression downstream
      maxTokens: 4096
```

## Gotchas

- **Model swap latency:** The ribosome (small model) and the generation model share Ollama. Use `keep_alive = "30m"` in helix.toml to pin the ribosome in memory.
- **Synonym map is critical:** If queries return "no relevant context", check that your query keywords map to the promoter tags the ribosome assigned. Add synonyms in `[synonyms]` section of helix.toml.
- **Short content may fail ingestion:** The ribosome struggles with very short inputs (<200 chars). Pad with context or combine small files before ingesting.
- **genome.db persists:** Delete it to start fresh. It auto-creates on first use.
- **Continue Agent mode:** Use Chat mode, not Agent mode. The proxy doesn't handle tool routing.

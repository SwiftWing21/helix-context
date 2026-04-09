# Helix Context — Gemini Integration

You have access to the **Agentome** — a genome-based context compression system that stores project knowledge from across the entire F:\ drive.

## Querying the Genome

The Helix server runs at `http://127.0.0.1:11437`. You can query it:

```bash
# Get compressed context for a query
curl -s http://127.0.0.1:11437/context -X POST -H "Content-Type: application/json" \
  -d '{"query": "your question here", "decoder_mode": "none"}'

# Check genome stats
curl http://127.0.0.1:11437/stats

# Check health
curl http://127.0.0.1:11437/health

# Check replication status
curl http://127.0.0.1:11437/replicas
```

## Sharing Knowledge with Claude

Both you and Claude Code share context through the Agentome bridge:

- **Shared context:** Read `~/.helix/shared/SHARED_CONTEXT.md` for current genome state
- **Share a fact:** Write a `.md` file to `~/.helix/shared/inbox/` — it will be auto-ingested
  - Filename format: `gemini_<timestamp>.md`
  - Content: any knowledge worth preserving (facts, decisions, discoveries)
- **Signals:** Read/write JSON files in `~/.helix/shared/signals/` for coordination
  - `ingesting.json` — an assistant is currently ingesting data
  - `query.json` — a query was just made (includes the query text)

## What's in the Genome

The genome contains compressed knowledge from:
- F:\Projects\ (BigEd, BookKeeper, CosmicTasha, helix-context, etc.)
- F:\Factorio\ (game Lua scripts, config files)
- F:\SteamLibrary\ (game data)
- Session memories (distilled conversation facts)

## Key Architecture

- **Genes:** Content units with promoter tags (domains/entities), ΣĒMA vectors (20D semantic), and key-value facts
- **Retrieval:** 4-tier (exact tag → prefix tag → FTS5 content → ΣĒMA semantic)
- **Compression:** ~7x ratio, raw content preserved for factual extraction
- **CpuTagger:** spaCy + regex ingestion (no LLM needed for pack)
- **Replication:** Delta-sync to read-only clones on C:, D:, E: drives

## Don't

- Don't modify genome.db directly — use the /ingest endpoint or inbox
- Don't ingest API keys, secrets, or credentials
- Don't assume genome content is current — check source file mtimes

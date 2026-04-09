# Session Handoff — 2026-04-08

## What We Built Today (single session, ~20 hours)

### Helix Context — Complete System
- **From:** Design spec + reference files in Downloads
- **To:** Published PyPI package, 3,500+ gene genome, Claude Code skill, DeBERTa training pipeline

### Key Files
| Component | Path |
|-----------|------|
| Package | `helix_context/` (10 modules) |
| Tests | `tests/` (7 test files, 170+ tests) |
| Benchmarks | `benchmarks/bench_compression.py`, `benchmarks/bench_needle.py` |
| Training | `training/export_training_data.py`, `training/finetune_rerank.py`, `training/finetune_splice.py` |
| DeBERTa backend | `helix_context/deberta_backend.py` |
| MCP server | `helix_context/mcp/server.py` |
| Claude Code skill | `~/.claude/skills/helix-context/SKILL.md` |
| Ingest scripts | `scripts/deep_ingest.py`, `scripts/deep_ingest_parallel.py` |
| Agentome wrapper | `F:\Projects\agentome/` (separate repo) |

### Current State
- **Genome:** 3,516 genes, 7.1x compression, 11MB raw → 1.6MB compressed
- **Server:** Running on port 11437 with DeBERTa backend enabled
- **DeBERTa models:** Trained, saved in `training/models/rerank` and `training/models/splice`
- **Deep ingest:** ~1,400/1,648 files done, progress tracked in `scripts/.ingest_progress`
- **PyPI:** `helix-context` v0.1.0b1 published, `agentome` pending publisher registration

### Benchmark Results (Pre vs Post DeBERTa)
| Metric | Pre-DeBERTa (Ollama) | Post-DeBERTa |
|--------|---------------------|--------------|
| p50 latency | 23.3s | **0.6s** (39x faster) |
| Mean latency | 16.9s | **0.5s** (34x faster) |
| Token savings | 89% | **96%** |
| Needle retrieval | 1/10 (10%) | 1/10 (10%) |
| Health status | 5 sparse, 11 denatured | 16 denatured |

### What's Working
- 6-step expression pipeline (extract, express, re-rank, splice, assemble, replicate)
- DeBERTa hybrid backend (re_rank + splice at 0.6s, pack + replicate via Ollama)
- Change-based decay (source file mtime detection, no time-based decay)
- Adaptive decoder modes (full/condensed/minimal/none — 96% savings on none)
- Delta-epsilon health monitor with history logging
- HGT genome export/import
- ScoreRift CD spectroscope bridge
- Continue IDE integration (E4B model)
- Claude Code `/helix` skill
- Parallel deep ingest with skeleton extraction

### What Needs Work (Next Session Priorities)
1. **Retrieval quality** — DeBERTa re-ranker producing all `denatured` results. Needs:
   - Re-export training data with 500 queries (was 100)
   - More epochs (rerank loss still dropping at epoch 3)
   - Possibly larger model (deberta-v3-base instead of small)
2. **Needle-in-haystack at 10%** — promoter tag coverage issue. The genes exist but query keywords don't match tags. Fix path: better synonyms OR semantic search fallback
3. **Splice accuracy at 77.5%** — needs more training data (was only 179 → 804 after expansion)
4. **Deep ingest remaining** — ~248 files left, restart with `python scripts/deep_ingest_parallel.py --workers 2`
5. **Agentome PyPI** — register trusted publisher on pypi.org, then `gh release create v0.1.0b1`
6. **MCP server not loading in VS Code** — works as skill instead, MCP needs debugging

### Architecture Decisions Made
- **No time-based decay** — genes never expire from age. Only source file changes trigger staleness
- **Splice handles relevance** — intron/exon filtering at expression time, not storage time
- **DeBERTa for hot path** — re_rank + splice (0.6s). Ollama for generative tasks (pack, replicate)
- **Adaptive decoder** — skip the 750-token decoder prompt when Claude is the downstream model
- **Content-addressed gene IDs** — SHA256[:16], enables dedup across HGT transfers

### Config State (helix.toml)
- `backend = "deberta"` (hybrid mode enabled)
- `splice_aggressiveness = 0.3`
- `decoder_mode = "full"` (override to "none" via /context API for Claude)
- `compact_interval = 3600` (hourly source-change checks)
- `OLLAMA_NUM_PARALLEL=4` (system env var)

### Git State
- Repo: SwiftWing21/helix-context (public)
- Latest commit: DeBERTa training pipeline + server health fix
- Uncommitted: benchmark results, helix.toml backend=deberta, server.py health fix
- Branch: master

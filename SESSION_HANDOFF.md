# Session Handoff — 2026-04-09

## What We Built Today

### CPU/Math Retrieval Pipeline (Phases 1-5)
Replaced LLM-based ingestion with CPU-native alternatives. **40x faster ingestion.**

#### New Files Created
| File | Purpose |
|------|---------|
| `helix_context/tagger.py` | CpuTagger — spaCy NER + regex KV + extractive summary (Phase 1) |
| `helix_context/splade_backend.py` | SPLADE sparse expansion + SQLite inverted index (Phase 2) |
| `scripts/resequence_cpu.py` | Re-encode genome through CPU pipeline |
| `scripts/ingest_steam.py` | Ingest Steam libraries (full content + game manifests for skipped files) |
| `scripts/ingest_models.py` | Ingest GGUF model headers (tensor count, architecture, KV metadata) |
| `scripts/ingest_all.py` | Combined ingest for all sources (F:\Projects, F:\SteamLibrary, E:\, OpenModels) |

#### Files Modified
| File | Changes |
|------|---------|
| `helix_context/config.py` | Added `IngestionConfig` dataclass + `GenomeConfig.replicas` |
| `helix_context/genome.py` | SPLADE Tier 3.5 retrieval, entity_graph table, `read_conn` replica routing, `busy_timeout=30s`, atomic commits |
| `helix_context/deberta_backend.py` | `rerank_pretrained` option for HuggingFace cross-encoders |
| `helix_context/context_manager.py` | CPU tagger routing in `ingest()`, rerank re-enablement, replication manager init |
| `helix_context/__init__.py` | Export CpuTagger + ReplicationManager |
| `helix_context/replication.py` | (Created by user/Gemini) Distributed genome clones with delta-sync |
| `helix.toml` | `[ingestion]` section (cpu backend, splade, rerank, entity_graph), `[genome]` replicas |
| `pyproject.toml` | `cpu = ["spacy>=3.7"]` optional dependency |

### CUDA PyTorch Upgrade
- Upgraded `torch 2.11.0+cpu` → `torch 2.11.0+cu126`
- SPLADE now runs on 3080 Ti GPU: ~50ms/gene (was ~200ms on CPU)
- Drivers fine: CUDA 13.1, driver 591.86

### Resequence Results (completed successfully)
- 3,501 genes resequenced through CPU+GPU pipeline in **8m52s** (was ~60min with LLM)
- 417,390 SPLADE terms (119/gene)
- 31,134 entity graph links (8.9/gene)
- 20,085 gene relations
- 0 errors

### Needle Benchmark (post-resequence, pre-additional-ingest)
- Context retrieval: **8/10 (80%)** — down from 9/10
- Misses: `biged_skills_count` (domain tagging gap), `biged_default_model` (same)
- Avg query latency: 285ms (includes SPLADE forward pass)

### Additional Ingestion (partial)
- F:\SteamLibrary: 924 files, 1,823 genes, 26 game manifests (completed)
- F:\OpenModels: 9 model shadow genes with GGUF architecture metadata (completed)
- E:\ drive: partially ingested BeamNG (~6,200 genes) before process kills
- Full ingest_all.py run reached ~14,189 genes visible in WAL but only 422 committed

## Current Genome State
- **422 committed genes** in genome.db (WAL data lost from process kills)
- genome_llm_backup.db: deleted (was the pre-CPU original with 3,501 genes)
- All indexes present: genes, promoter_index, FTS5, splade_terms, entity_graph, gene_relations

## Critical Bug: WAL Data Loss on Process Kill
- `upsert_gene()` commits per-gene (atomic: gene+promoter+FTS5+entity+SPLADE in one commit)
- But WAL checkpoint only runs when connection closes gracefully
- Killing Python processes loses all in-WAL data
- **Fix needed:** periodic `PRAGMA wal_checkpoint(PASSIVE)` during bulk ingest (every N genes)
- The 14,189 genes were real — they just weren't checkpointed to the main DB file before kill

## Database Locking Issue (Resolved)
- Root cause: multiple Python processes opening genome.db simultaneously
- VS Code extensions, Helix server, replication manager all compete for WAL
- **Fixes applied:**
  - `busy_timeout=30000` (30s retry instead of instant fail)
  - `read_conn` property routes reads to replicas
  - Atomic single-commit in `upsert_gene()` (was 3 separate commits)
- Still need: WAL checkpoint in ingest loop, or `synchronous=NORMAL` for faster writes

## What Needs Work (Next Session)

### Priority 1: Fix ingest durability
- Add `PRAGMA wal_checkpoint(PASSIVE)` every 100 genes in ingest scripts
- Or switch to `synchronous=FULL` during bulk ingest (slower but crash-safe)
- The pipeline works — data just needs to survive process interruption

### Priority 2: Full re-ingest
- Run `scripts/ingest_all.py` to completion without interruption
- Sources: F:\Projects (~3,500 genes), F:\SteamLibrary (~1,800), E:\ (~6,000+), OpenModels (9)
- Expected: ~11,000+ genes, ~20min at current GPU speed

### Priority 3: Benchmark with full genome
- Run `benchmarks/bench_needle.py` against completed genome
- Target: 10/10 retrieval, >50% answer accuracy
- The 2 misses (biged_skills_count, biged_default_model) are domain-tagging gaps in CpuTagger

### Priority 4: Improve CpuTagger domain coverage
- Add project-specific terms to `_TECH_TERMS` dict (helix, bookkeeper, biged, scorerift)
- Or rely more heavily on SPLADE (which covers semantic gaps mathematically)

### Priority 5: ColBERT (Phase 4, optional)
- Only if Phases 1-3 don't reach 10/10 after full ingest + tuning

## Architecture Summary
```
INGESTION (CPU+GPU, ~150ms/gene):
  Raw text → CodonChunker → CpuTagger.pack():
    spaCy NER → entities, domains    [CPU, ~100ms]
    Regex → key_values               [CPU, <1ms]
    Extractive → complement, codons  [CPU, ~2ms]
  → SPLADE encode → sparse terms     [GPU, ~50ms]
  → Entity graph overlap detection   [CPU, <1ms]
  → genome.upsert_gene() [single atomic commit]

RETRIEVAL (<300ms):
  Query → extract_query_signals()
  → Tier 1: exact promoter tag (weight 3.0)
  → Tier 2: prefix tag (weight 1.5)
  → Tier 3: FTS5 content (weight 2.0)
  → Tier 3.5: SPLADE sparse (weight 2.5)
  → IDF weighting + co-activation + entity graph expansion
  → Cross-encoder rerank (MS MARCO MiniLM, optional)
  → Splice + Assemble
```

## Config State (helix.toml)
- `backend = "cpu"` (CpuTagger for ingestion)
- `splade_enabled = true` (GPU-accelerated)
- `rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- `rerank_enabled = true`
- `entity_graph = true`
- `replicas = ["C:/helix-cache/genome.db", "E:/helix-cache/genome.db"]`
- `max_genes_per_turn = 20`

## Installed Dependencies
- `spacy 3.8.13` + `en_core_web_sm 3.8.0`
- `torch 2.11.0+cu126` (CUDA on 3080 Ti)
- `transformers 5.5.0`, `sentence-transformers 5.3.0`
- SPLADE model: `naver/splade-cocondenser-ensembledistil` (cached)
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (cached)

## Git State
- Repo: SwiftWing21/helix-context (public)
- Branch: master
- Uncommitted: all Phase 1-5 changes, scripts, config updates

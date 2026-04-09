# Helix Context — Ingest Timeline & Benchmark Notes

## Deep Ingest Timeline (2026-04-08)

**Started:** ~12:45am April 8, 2026
**Target:** F:\Projects (entire workspace — 8 projects, ~1,955 source files)
**Hardware:** RTX 3080 Ti (12GB VRAM), 48GB RAM, 16-core CPU, Windows 11

### Phase 1: Single-threaded (deep_ingest.py)
- **Time:** 12:45am - ~8:00am (~7 hours)
- **Ribosome:** gemma4:e2b
- **Workers:** 1
- **Rate:** ~69 files/hour
- **Files completed:** ~568
- **Genes produced:** ~1,845
- **Issues:**
  - Process died twice (no parent shell keepalive)
  - Large files (34KB+ pdf_report.py, xlsx_report.py) caused timeouts
  - No resume support initially — added .ingest_progress file
  - Added chunking for files >15KB (split on function/class boundaries)

### Phase 2: Parallel attempt with 5 workers (deep_ingest_parallel.py)
- **Time:** ~2:00pm - ~2:30pm
- **Ribosome:** gemma4:e2b
- **Workers:** 5
- **OLLAMA_NUM_PARALLEL:** not set (default 1)
- **Rate:** ~44 files/hour (SLOWER — workers queued on single Ollama slot)
- **Issues:** Workers timing out at 300s, producing 0 genes
- **Action:** Killed, reduced to 3 workers

### Phase 3: Parallel with 3 workers
- **Time:** ~2:30pm - ~4:00pm
- **Ribosome:** gemma4:e2b
- **Workers:** 3
- **OLLAMA_NUM_PARALLEL:** not set
- **Rate:** ~264 files/hour (estimated from early throughput)
- **Issues:** Still some timeouts on large files

### Phase 4: Set OLLAMA_NUM_PARALLEL=4, 5 workers
- **Time:** ~4:00pm - ~8:00pm (~4 hours)
- **Ribosome:** gemma4:e2b
- **Workers:** 5
- **OLLAMA_NUM_PARALLEL:** 4
- **Files completed:** ~1,094 total
- **Genes:** ~2,114
- **Issues:**
  - 25 zombie Python processes accumulated (httpx connections stuck)
  - Several files producing 0 genes (ribosome overwhelmed by concurrent requests)
  - Large research docs (60-120KB) only reduced 5-6% by markdown skeleton extraction

### Phase 5: Skeleton extraction + hard-cut chunking, 5 workers
- **Time:** ~8:00pm - ~10:00pm
- **Workers:** 5
- **Added:** Skeleton extraction for files >50KB (keep signatures, imports, headings)
- **Added:** Hard-cut fallback (force slice at 15KB if chunks still too large)
- **Added:** Skip .next/ build artifacts
- **Issues:** Markdown skeleton not aggressive enough, still timing out
- **Action:** Killed, reduced to 2 workers

### Phase 6: Stable 2 workers (current)
- **Time:** ~10:00pm - ongoing
- **Ribosome:** gemma4:e2b
- **Workers:** 2
- **Rate:** ~70 files/hour (stable, no timeouts)
- **Files completed:** 1,407 / 1,648
- **Genes:** 2,853
- **Remaining:** ~241 files
- **Status:** Stable, processing fleet/knowledge/ directory

### Crash/Error Summary
| Event | Cause | Resolution |
|-------|-------|------------|
| Process death x2 | nohup + no parent shell | Added resume via .ingest_progress |
| 0-gene timeouts | Files >30KB overwhelming ribosome | Added chunking at 15KB boundaries |
| Zombie processes (25) | 5 workers + httpx stuck connections | Reduced to 2 workers |
| Markdown skeleton ineffective | Headings/lists/tables are most of content | Hard-cut fallback at 15KB |
| .next artifacts ingested | Build output in CosmicTasha | Added .next to SKIP_DIRS |

### Final Genome State (as of 2:51pm)
- **Genes:** 2,853
- **Compression ratio:** 7.17x
- **Raw content:** 8.8MB (9,269,721 chars)
- **Compressed:** 1.2MB (1,293,271 chars)
- **Files ingested:** 1,407 / 1,648

## Benchmark Plan

### 1. Compression Quality (vs KV Cache / TurboQuant baselines)
- **Metric:** Information retention at various compression ratios
- **Method:** Oracle grading (Opus 1M) — read raw file + compressed output, score preservation
- **Targets:** 5x, 7x, 10x compression ratios
- **Baselines:** Raw context (1x), LLMLingua (~2-3x), KV cache quantization

### 2. Retrieval Accuracy
- **Metric:** Precision@k, Recall@k for promoter-tag retrieval
- **Method:** Prepare 50 queries with known-relevant files, measure gene expression accuracy
- **Compare:** vs standard RAG (vector search), vs BM25, vs full-file read

### 3. Latency Profile
- **Metric:** Time per pipeline step (extract, express, re-rank, splice, assemble)
- **Method:** Instrument each step, run 100 queries, report p50/p95/p99
- **Variants:** e2b vs e4b ribosome, 1 vs 4 parallel Ollama slots

### 4. Token Savings (API cost reduction)
- **Metric:** Input tokens consumed with vs without Helix
- **Method:** Record actual token usage across 100 representative queries
- **Report:** % savings by decoder mode (full/condensed/minimal/none)

### 5. Genome Growth & Decay
- **Metric:** Gene count, chromatin state distribution, co-activation density over time
- **Method:** Log genome stats during extended conversation sessions
- **Report:** Growth curve, compaction rate, associative link formation

### 6. Standard HuggingFace Benchmarks (if applicable)
- LongBench (long-context comprehension)
- MTEB (retrieval accuracy — adapt for promoter matching)
- Needle-in-haystack (can the genome express the right gene for a specific fact?)

# DeBERTa Ribosome Training

Fine-tunes DeBERTa-v3-small as a small-model replacement for Ollama
in the re-rank, splice, and NLI stages of the expression pipeline.

## Why

Ollama ribosome runs an autoregressive LLM for what is fundamentally a
classification/extraction task. A fine-tuned encoder-only DeBERTa head
processes the entire sequence in parallel, replacing generative token
loops with a single forward pass.

**Expected speedup:** ~400x for batch re-rank operations (80s → 200ms).

## Heads

Three task-specific DeBERTa-v3-small heads, each ~44M params:

| Head | Task | Loss | Data |
|------|------|------|------|
| `rerank` | Cross-encoder `(query, gene_summary) → score` | MSE | `rerank_pairs.jsonl` |
| `splice` | Binary codon keep/drop classifier | BCE | `splice_labels.jsonl` |
| `nli` | 7-class natural logic relation classifier | CrossEntropy | `nli_pairs.jsonl` |

All three fit simultaneously on 12GB VRAM (~264MB FP16 combined)
alongside an Ollama gemma4:e2b (~2-4GB) for PACK operations.

## Workflow

### 1. Export teacher labels from the current genome

The existing Ollama ribosome acts as teacher. This step generates
training pairs by running the production pipeline over sampled queries
and recording its outputs as ground truth.

```bash
# Produces training/data/rerank_pairs.jsonl + splice_labels.jsonl
python training/export_training_data.py \
    --genome genome.db \
    --out training/data/ \
    --n-queries 500
```

Requirements:
- Ollama running at `localhost:11434`
- At least one model loaded (default: `gemma4:e2b`)
- Current genome at `genome.db` with ≥1000 genes

Time: ~30 min for 500 queries (mostly Ollama teacher inference).

### 2. Fine-tune each head

```bash
# Re-rank head (cross-encoder)
python training/finetune_rerank.py \
    --data training/data/rerank_pairs.jsonl \
    --epochs 5 \
    --lr 2e-5 \
    --output training/models/rerank/

# Splice head (binary classifier)
python training/finetune_splice.py \
    --data training/data/splice_labels.jsonl \
    --epochs 5 \
    --lr 2e-5 \
    --output training/models/splice/

# NLI head (optional, 7-class relation classifier)
python training/finetune_nli.py \
    --data training/data/nli_pairs.jsonl \
    --epochs 5 \
    --output training/models/nli/
```

Time: ~10-15 min per head on RTX 3080 Ti.

### 3. Wire into helix.toml

```toml
[ribosome]
backend = "deberta"
rerank_model = "training/models/rerank"
splice_model = "training/models/splice"
nli_model = "training/models/nli"  # optional
fallback = "ollama"  # keep Ollama for PACK operations
```

Restart the Helix server to pick up the new backend.

### 4. Verify with benchmarks

```bash
python benchmarks/bench_needle.py      # Should match or beat prior results
python benchmarks/bench_babilong.py    # Multi-hop, sensitive to re-rank quality
python benchmarks/bench_compression.py # Compression ratio shouldn't regress
```

## When to re-train

- **Genome grows significantly** (e.g., 2x gene count) — teacher labels
  from the old genome miss new concepts
- **Retrieval quality plateaus** in benchmarks — maybe the re-rank head
  is the bottleneck, not retrieval
- **New content domain added** (e.g., a whole new codebase ingested)
  — existing model didn't see those patterns during training

## Current training data snapshot

| File | Count | Last regenerated |
|------|-------|------------------|
| `data/queries.txt` | 99 | v0.1.0b1 era (~3,500 gene genome) |
| `data/rerank_pairs.jsonl` | 1,600 | v0.1.0b1 era |
| `data/splice_labels.jsonl` | 179 | v0.1.0b1 era |

**Staleness note:** the current genome has ~7,300 genes. The data was
generated when the genome was ~3,500 genes. A full re-export would
roughly double the training set and cover concepts added since (SIKE,
MoE decoder, cold-storage tiers, etc.).

## Troubleshooting

- **`CUDA out of memory`** — lower `--batch-size` (default 16), or fall
  back to CPU training with `--device cpu` (~10x slower but works)
- **Teacher returns empty labels** — Ollama model is too small or too
  busy; try `OLLAMA_NUM_PARALLEL=1` and a larger teacher model
- **Val loss diverges** — training data is noisy; check for duplicate
  queries or contradictory labels in the same query

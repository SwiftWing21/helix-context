# Helix + Headroom Skills Bundle

> How a `skills.md` file becomes queryable, compressed context
> through the helix-context + Headroom stack.

## The Simple Version

A skill is just a markdown file. It enters the genome through the normal ingest
pipeline, gets chunked into genes, tagged with promoters, embedded with ΣĒMA,
and becomes retrievable via SIKE scoring. Headroom compresses the output when
a downstream model requests context.

There is no special "skills runtime." The genome IS the skills database.

---

## Lane Graph — Skill Lifecycle

```
 skills.md file
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  INGEST                                                     │
 │                                                             │
 │  /ingest POST  ──or──  scripts/ingest_all.py                │
 │       │                       │                             │
 │       └───────┬───────────────┘                             │
 │               ▼                                             │
 │  ┌─────────────────┐                                        │
 │  │  Chunker         │  Split into strands (≤4000 chars)     │
 │  │  (CodonChunker)  │  Preserves section structure          │
 │  └────────┬────────┘                                        │
 │           ▼                                                 │
 │  ┌─────────────────┐                                        │
 │  │  Tagger          │  CpuTagger (spaCy + regex, fast)      │
 │  │                  │  Extracts: codons, promoter tags,     │
 │  │                  │  key_values, complement (summary)     │
 │  └────────┬────────┘                                        │
 │           ▼                                                 │
 │  ┌─────────────────┐                                        │
 │  │  ΣĒMA Encoder    │  20-dim semantic embedding            │
 │  │  (SemaCodec)     │  Enables cosine retrieval + cold-tier │
 │  └────────┬────────┘                                        │
 │           ▼                                                 │
 │  ┌─────────────────┐                                        │
 │  │  Density Gate    │  Admit / demote to heterochromatin    │
 │  │                  │  Based on source, score, access rate  │
 │  └────────┬────────┘                                        │
 │           ▼                                                 │
 │      genome.db                                              │
 │      (genes table)                                          │
 └─────────────────────────────────────────────────────────────┘

       ▼  (later, on query)

 ┌─────────────────────────────────────────────────────────────┐
 │  RETRIEVAL  (/context POST)                                 │
 │                                                             │
 │  Step 1 ─── Extract query signals (keywords, entities)      │
 │                                                             │
 │  Step 2 ─── Genome query                                    │
 │             Hot-tier: FTS5 + SPLADE + promoter match        │
 │             Cold-tier: ΣĒMA cosine fallthrough (opt-in)     │
 │                                                             │
 │  Step 3 ─── Score & rank                                    │
 │             Cymatics resonance_rank (CPU, ~5ms)             │
 │             Fallback: ribosome re_rank (LLM, ~2s)           │
 │                                                             │
 │  Step 4 ─── Compress (HEADROOM)                             │
 │             ┌──────────────────────────────────┐            │
 │             │  Headroom specialist dispatch:    │            │
 │             │  logs → LogCompressor             │            │
 │             │  diffs → DiffCompressor           │            │
 │             │  code → Kompress (ModernBERT)     │            │
 │             │  prose → Kompress                 │            │
 │             │  target: ~1000 chars per gene     │            │
 │             └──────────────────────────────────┘            │
 │                                                             │
 │  Step 5 ─── Assemble                                        │
 │             Token budget enforcement                        │
 │             MoE answer-slate with top-5 KVs per gene        │
 │             <expressed_context> wrapper                     │
 └─────────────────────────────────────────────────────────────┘

       ▼

 ┌─────────────────────────────────────────────────────────────┐
 │  DOWNSTREAM MODEL                                           │
 │                                                             │
 │  Receives compressed, scored, budget-fitted context         │
 │  from the skill's genes. Doesn't know or care that the     │
 │  source was a skills.md file.                               │
 └─────────────────────────────────────────────────────────────┘
```

---

## What Each Layer Does

### Helix (storage + retrieval)

| Component | Role | CPU/LLM |
|---|---|---|
| **Chunker** | Split skill content into gene-sized strands | CPU |
| **CpuTagger** | Extract codons, promoters, KVs, complement | CPU (spaCy) |
| **ΣĒMA** | 20-dim semantic embedding per gene | CPU |
| **Density gate** | Admit or demote genes based on quality/source | CPU |
| **FTS5 + SPLADE** | Full-text + sparse term retrieval | CPU |
| **Cymatics** | Frequency-domain relevance scoring | CPU |
| **Cold-tier** | ΣĒMA cosine fallthrough for demoted genes | CPU |
| **SIKE** | Scale-invariant scoring (model-agnostic) | CPU |

### Headroom (compression at expression time)

| Component | Role | CPU/LLM |
|---|---|---|
| **Kompress** | ModernBERT extractive compression | CPU |
| **LogCompressor** | Log-specific compression | CPU |
| **DiffCompressor** | Diff/patch compression | CPU |
| **~~CodeCompressor~~** | ~~Code-aware compression~~ (disabled: 40% syntax corruption) | — |

### The Bundle

helix + Headroom together = **fully CPU-based context pipeline.** No LLM calls
required at any step from ingest through expression. The downstream model is the
first and only LLM in the chain.

```
skills.md → [helix: chunk, tag, embed, store] → [helix: retrieve, score]
          → [headroom: compress] → downstream model
```

---

## How to Ingest a Skill

### Single file
```bash
curl -X POST http://localhost:11437/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "$(cat skills/code_review.md)", "content_type": "text"}'
```

### Bulk ingest (all .md files in a directory)
```bash
python scripts/ingest_all.py --roots skills/ --ext .md
```

### From another helix instance (HGT)
```python
from helix_context.hgt import export_genome, import_genome

# Export skills from source genome
export_genome("source_genome.db", "skills_export.helix")

# Import into target genome (skip duplicates)
import_genome("target_genome.db", "skills_export.helix", strategy="skip_existing")
```

---

## What a Skill Gene Looks Like in the Genome

After ingesting `skills/code_review.md`:

```
gene_id:     a7f3b2c1...
content:     "## Code Review Checklist\n\n1. Check for..."
complement:  "Code review skill covering security, style..."
codons:      ["code_review:1.0", "security:0.8", "style:0.5"]
promoter:    {"domains": ["code", "review"], "entities": ["OWASP"]}
key_values:  {"checklist_items": "12", "severity_levels": "3"}
embedding:   [0.082, 0.014, ...]  (20-dim ΣĒMA)
chromatin:   0  (OPEN — hot tier)
source_id:   "skills/code_review.md"
```

Retrieval query `"review this PR for security issues"` would:
1. Match promoter tags `code`, `review`, `security`
2. Score via cymatics resonance on frequency spectrum
3. Compress via Headroom Kompress to ~1000 chars
4. Deliver as `<GENE src="skills/code_review.md">...</GENE>`

---

## Future: Federation Layer (not built yet)

When federation ships, the bundle extends:

```
 External org
       │
  API key → party_id
       │
  Headroom SSS (trust scoring)
       │
  Federation router
       │
  genome.query_genes(... WHERE party_id = ?)
       │
  Same 5-step pipeline
       │
  HGT export for "copy this skill"
```

This is the D7 (gene attribution) dependency. Until `gene_attribution` has data
flowing, federation is schema-only. See [DIMENSIONS.md](DIMENSIONS.md) for the
lane graph.

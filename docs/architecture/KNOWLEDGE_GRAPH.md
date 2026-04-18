# Helix Knowledge Graph

> How genes, dimensions, and connections form a queryable knowledge graph
> in the helix-context + Headroom stack.

## Graph Structure

```
                        ┌─────────────┐
                        │   QUERY     │
                        │  "how does  │
                        │  auth work?"│
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
        ┌──────────┐   ┌──────────┐      ┌──────────┐
        │ PROMOTER │   │   FTS5   │      │   ΣĒMA   │
        │ D2 match │   │ D1 match │      │ D1 cos.  │
        └────┬─────┘   └────┬─────┘      └────┬─────┘
             │              │                  │
             └──────────────┼──────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Candidate Genes │
                   │  (from genome)   │
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼              ▼
        ┌──────────┐ ┌──────────┐  ┌──────────┐
        │ CYMATICS │ │ CHROMATIN│  │ WORKING  │
        │ D6 score │ │ D5 tier  │  │ SET D4   │
        └────┬─────┘ └────┬─────┘  └────┬─────┘
             │             │              │
             └─────────────┼──────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Ranked Genes    │
                  │  (top-k scored)  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  HEADROOM        │
                  │  compress → out  │
                  └─────────────────┘
```

---

## Node Types

### Gene (primary node)

The fundamental unit. Every piece of knowledge in helix is a gene.

```
┌─────────────────────────────────────────────┐
│  GENE: a7f3b2c1                             │
│                                             │
│  content:    "## Auth Middleware\n..."       │
│  complement: "JWT auth with session..."     │
│  source_id:  "fleet/skills/auth.py"         │
│  chromatin:  0 (OPEN)                       │
│                                             │
│  ┌─ Codons ─────────────────────────────┐   │
│  │  auth:1.0  jwt:0.8  session:0.5     │   │
│  │  security:0.8  middleware:0.6        │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌─ Promoter ───────────────────────────┐   │
│  │  domains: [auth, security, backend]  │   │
│  │  entities: [JWT, OAuth, OWASP]       │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌─ Key-Values ─────────────────────────┐   │
│  │  token_expiry: "30m"                 │   │
│  │  algorithm: "HS256"                  │   │
│  │  session_store: "redis"              │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌─ Embedding (ΣĒMA 20-dim) ───────────┐   │
│  │  [0.08, 0.01, 0.07, -0.05, ...]     │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌─ Epigenetics ────────────────────────┐   │
│  │  access_rate: 2.3/hour               │   │
│  │  decay_score: 0.95                   │   │
│  │  co_activated_with: [b3e1..., c4f2]  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Edge Types (connections between genes)

```
 Gene A ────── co_activated_with ──────► Gene B
               (epigenetics field)
               "retrieved together in same query"

 Gene A ────── harmonic_link ──────────► Gene B
               (harmonic_links table)
               "spectral similarity via cymatics"
               weight: cosine of frequency spectra

 Gene A ────── entity_link ────────────► Gene B
               (entity_graph table)
               "share a named entity"
               entity: "JWT", "OAuth", etc.

 Gene A ────── supersedes ─────────────► Gene B
               (genes.supersedes field)
               "newer version of same content"
               version: gene.version

 Gene A ────── relation ───────────────► Gene B
               (gene_relations table)
               "semantic relation"
               type: entails | contradicts | neutral
```

---

## The Six Active Dimensions as Graph Filters

Each dimension acts as a filter or scoring function over the graph:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  D1  Semantic    ─── FTS5 term match + SPLADE expansion     │
│                      + ΣĒMA cosine similarity               │
│                      "does this gene CONTAIN relevant terms?"│
│                                                             │
│  D2  Promoter    ─── domain + entity tag intersection       │
│                      + synonym expansion (helix.toml)       │
│                      "is this gene ABOUT the right topic?"  │
│                                                             │
│  D3  Source      ─── deny-list filter + authority bonus     │
│                      "is this gene FROM a trusted source?"  │
│                                                             │
│  D4  Working-set ─── access_rate(gene, window) tiebreaker   │
│                      "is this gene RECENTLY active?"        │
│                                                             │
│  D5  Chromatin   ─── tier filter (hot/warm/cold)            │
│                      + cold-tier ΣĒMA fallthrough           │
│                      "is this gene ACCESSIBLE right now?"   │
│                                                             │
│  D6  Cymatics    ─── frequency-domain resonance scoring     │
│                      "does this gene RESONATE with query?"  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Traversal Example

Query: `"how does the auth middleware handle expired tokens?"`

```
Step 1 — Signal extraction
    keywords: [auth, middleware, handle, expired, tokens]
    entities: [auth, middleware, tokens]

Step 2 — Graph traversal (candidate retrieval)
    D1 FTS5:    "auth" OR "middleware" OR "tokens"  → 47 genes
    D1 SPLADE:  expanded terms from splade_terms    → +12 genes
    D2 Promoter: domains ∩ {auth, security, backend} → 31 genes
    D1 ΣĒMA:    cosine(query_embed, gene_embed) > 0.3 → +8 genes
    Deduplicate → 52 unique candidate genes

Step 3 — Scoring (dimension fusion)
    D6 Cymatics resonance_rank:
        Gene "auth.py chunk 1"    → score 0.89  (auth + jwt peaks resonate)
        Gene "redis session mgr"  → score 0.72  (session + token peaks)
        Gene "OWASP top 10 ref"   → score 0.41  (security broad match)
        ...
    D5 Chromatin: filter out heterochromatin (unless include_cold)
    D4 Working-set: tiebreak by access_rate for equal scores
    D3 Source: authority bonus for fleet/skills/* sources
    Top-12 selected (budget.max_genes_per_turn)

Step 4 — Compression (Headroom)
    Gene "auth.py chunk 1" (3200 chars) → Kompress → 980 chars
    Gene "redis session mgr" (2100 chars) → Kompress → 720 chars
    ...

Step 5 — Assembly
    <expressed_context>
      <GENE src="fleet/skills/auth.py" score="0.89" facts="token_expiry=30m...">
        compressed auth middleware content...
      </GENE>
      <GENE src="fleet/session_manager.py" score="0.72" facts="store=redis...">
        compressed session content...
      </GENE>
      ...
    </expressed_context>
```

---

## Graph Storage Layout

```
genome.db (SQLite)
│
├── genes              ─── Primary node table (17,623 rows)
│   ├── gene_id            (TEXT PK, content-addressed hash)
│   ├── content            (TEXT, full original content)
│   ├── complement         (TEXT, dense summary)
│   ├── codons             (TEXT JSON, weighted term labels)
│   ├── promoter           (TEXT JSON, {domains, entities})
│   ├── epigenetics        (TEXT JSON, {access_rate, decay, co_activated_with})
│   ├── chromatin          (INT, 0=OPEN 1=EUCHRO 2=HETERO)
│   ├── embedding          (TEXT JSON, 20-dim ΣĒMA vector)
│   ├── key_values         (TEXT JSON, extracted KV pairs)
│   ├── source_id          (TEXT, file path or ingest source)
│   ├── version            (INT, monotonic per gene lineage)
│   └── supersedes         (TEXT, gene_id of prior version)
│
├── genes_fts          ─── FTS5 full-text index (D1)
├── splade_terms       ─── Sparse expansion terms (D1)
├── promoter_index     ─── Promoter tag index (D2)
│
├── harmonic_links     ─── Cymatics spectral edges (D6 → D8)
│   ├── gene_a, gene_b
│   └── weight             (cosine of frequency spectra)
│
├── entity_graph       ─── Named entity co-occurrence (D8)
│   ├── gene_a, gene_b
│   └── entity             (shared entity string)
│
├── gene_relations     ─── Semantic relations (D8)
│   ├── gene_a, gene_b
│   └── relation           (entails | contradicts | neutral)
│
├── gene_attribution   ─── Party/participant ownership (D7)
│   ├── gene_id
│   ├── party_id           (org/tenant)
│   └── participant_id     (agent/skill/human)
│
├── parties            ─── Tenant registry (D7, 0 rows)
├── participants       ─── Agent/skill registry (D7, 0 rows)
├── hitl_events        ─── HITL pause logger (0 rows)
└── health_log         ─── Genome health snapshots
```

---

## Helix + Headroom Bundle Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     HELIX + HEADROOM                        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    HELIX                               │  │
│  │                                                       │  │
│  │  Ingest ─── Chunk + Tag + Embed + Gate ──► genome.db  │  │
│  │                                                       │  │
│  │  Retrieve ── FTS5 + SPLADE + ΣĒMA + Promoter          │  │
│  │              + Cymatics resonance scoring              │  │
│  │              + Chromatin tier filtering                │  │
│  │              + Working-set access rate                 │  │
│  │              + Cold-tier ΣĒMA fallthrough              │  │
│  │                                                       │  │
│  │  ALL CPU. No LLM calls.                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  HEADROOM                              │  │
│  │                                                       │  │
│  │  Compress ── Kompress (ModernBERT) for prose + code   │  │
│  │              LogCompressor for build/test output       │  │
│  │              DiffCompressor for patches                │  │
│  │              Target: ~1000 chars per gene              │  │
│  │                                                       │  │
│  │  ALL CPU. No LLM calls.                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│               Compressed, scored context                    │
│               ready for ANY downstream model                │
│               (local 4B or frontier API)                    │
└─────────────────────────────────────────────────────────────┘
```

**Zero LLM calls from ingest to expression.** The first LLM in the chain is
the downstream model that receives the compressed context. This is the core
value proposition: helix + Headroom turn raw files into query-ready, compressed,
scored context using only CPU, making it model-agnostic and cost-free at
retrieval time.

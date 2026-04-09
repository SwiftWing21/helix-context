# Agentome Research Log

Theoretical foundations, referenced papers, and design decisions that shaped the
Agentome (formerly Helix Context) architecture. Every external idea is credited
to its source.

---

## Core Thesis: The Ribosome Hypothesis

> A local LLM's context window is a cell: most of the genome is cold-stored,
> and a ribosome expresses only the relevant proteins per turn.

**Origin:** Michael Bachaud, "Agentome: The Ribosome Hypothesis" (2026).
Published at [mbachaud.substack.com/p/agentome](https://mbachaud.substack.com/p/agentome).

The hypothesis maps molecular biology's central dogma onto LLM context management:

| Biology | Agentome | Role |
|---------|----------|------|
| DNA genome | SQLite gene store | Cold storage, never fully loaded |
| mRNA transcription | Promoter-tag retrieval | Find relevant genes |
| Ribosome translation | Small model codec | Compress / rank / splice |
| Protein | Expressed context window | What the big model actually sees |
| Epigenetics | Access counts, decay, co-activation | Usage-based gene evolution |
| Chromatin state | OPEN / EUCHROMATIN / HETEROCHROMATIN | Accessibility tiers |
| Introns / exons | Codon keep/drop via splice | Per-query relevance filtering |
| Horizontal gene transfer | `.helix` export/import | Genome sharing across instances |

---

## Referenced Papers & Sources

### 1. Goldman et al. — DNA Digital Data Storage (2013)

**Paper:** Goldman, N., Bertone, P., Chen, S., Dessimoz, C., LeProust, E.M.,
Sipos, B., & Birney, E. (2013). Towards practical, high-capacity, low-maintenance
information storage in synthesized DNA. *Nature*, 494(7435), 77-80.
doi:[10.1038/nature11875](https://doi.org/10.1038/nature11875)

**Key contributions used:**
- **Trit encoding without homopolymer runs**: Maps to our codon weight system
  (ternary: 0.1 filler / 0.5 useful / 1.0 critical)
- **Overlapping segment indexing**: Validates our `is_fragment` + `sequence_index`
  gene reassembly strategy
- **Error correction via redundancy**: Informs future HGT export resilience
  (Reed-Solomon-style redundancy for genome transfers)

**Where it applies:** The paper provides the physical-layer blueprint for what
Agentome does at the semantic layer. Both systems solve the same problem:
high-density information storage with error-tolerant retrieval.

---

### 2. Bowman, Potts & Manning — Recursive Neural Networks Can Learn Logical Semantics (2015)

**Paper:** Bowman, S.R., Potts, C., & Manning, C.D. (2015). Recursive Neural
Networks Can Learn Logical Semantics. *Proceedings of the 3rd Workshop on
Continuous Vector Space Models and their Compositionality*, Association for
Computational Linguistics.
[arxiv.org/abs/1406.1827](https://arxiv.org/abs/1406.1827)

**Key contributions used:**
- **MacCartney-Manning 7-class natural logic relations**: Entailment, reverse
  entailment, equivalence, alternation, negation, cover, independence. Adopted
  as `NLRelation` enum in `schemas.py` for typed inter-gene relationships.
- **TreeRNTN achieves 99.7% on quantifier reasoning**: Demonstrates that
  fixed-length vectors can encode set-theoretic logical operations. Validates
  using DeBERTa embeddings for relation classification.
- **Tensor interaction term (NTN) critical for set operations**: Informed our
  choice of cross-encoder architecture over bi-encoder for NLI classification.
- **Compositional generalization**: Models trained on short structures generalize
  to longer unseen ones. Supports our approach of training on gene summaries
  and applying to full content.

**Where it applies:**
- `helix_context/nli_backend.py` — 7-class NLI classifier
- `helix_context/schemas.py` — `NLRelation`, `TypedCoActivation`
- `helix_context/genome.py` — `gene_relations` table, typed co-activation
  expansion (entailment links always pull forward, alternation links skipped)
- `helix_context/context_manager.py` — Step 3.5 NLI classification,
  `logical_coherence` health metric
- `training/finetune_nli.py` — 7-class DeBERTa fine-tuning

**Design decision:** We started with 5 classes (dropping negation and cover)
because these are genuinely rare in codebase contexts. The schema defines all 7
for forward compatibility.

---

### 3. He et al. — DeBERTa: Decoding-enhanced BERT with Disentangled Attention (2021)

**Paper:** He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa:
Decoding-enhanced BERT with Disentangled Attention. *Proceedings of ICLR 2021*.
[arxiv.org/abs/2006.03654](https://arxiv.org/abs/2006.03654)

**Also:** He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa
using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.
[arxiv.org/abs/2111.09543](https://arxiv.org/abs/2111.09543)

**Key contributions used:**
- **Disentangled attention** (content + relative position): Superior to BERT/RoBERTa
  for understanding structural relationships between short text spans. Critical
  for codon-level splice decisions where position within a gene matters.
- **44M params (v3-small)**: Fits trivially alongside Ollama on 12GB VRAM.
  Three heads (rerank + splice + NLI) total ~264MB in FP16.
- **SentencePiece Unigram tokenizer**: We keep the native tokenizer rather than
  swapping to BPE — the pretrained embedding space is the whole point of
  fine-tuning.

**Where it applies:**
- `helix_context/deberta_backend.py` — `DeBERTaRibosome` class
- `training/finetune_rerank.py` — Cross-encoder for gene re-ranking
- `training/finetune_splice.py` — Binary classifier for codon keep/drop
- `training/finetune_nli.py` — 7-class NLI relation classifier

**Design decision:** DeBERTa-v3-small over alternatives:
- vs. BERT/RoBERTa: DeBERTa's disentangled attention handles our (query, codon)
  pairs better — position of a codon within a gene is load-bearing information.
- vs. DeBERTa-base/large: Diminishing returns for our short inputs (codons are
  3-15 words). Small fits 3x on 12GB VRAM.
- vs. Byte-level (ByT5): 5-6x sequence explosion for marginal accuracy gain on
  short text. Not worth the compute.
- vs. Training from scratch: We're fine-tuning, not pretraining. Swapping the
  tokenizer loses the embedding space.

---

### 4. MacCartney & Manning — An Extended Model of Natural Logic (2009)

**Paper:** MacCartney, B., & Manning, C.D. (2009). An Extended Model of Natural
Logic. *Proceedings of the Eighth International Conference on Computational
Semantics (IWCS-8)*.

**Key contributions used:**
- **7 mutually exclusive semantic relations** defined set-theoretically:
  entailment (x ⊂ y), reverse entailment (x ⊃ y), equivalence (x = y),
  alternation (x ∩ y = ∅, x ∪ y ≠ D), negation (x ∩ y = ∅, x ∪ y = D),
  cover (x ∩ y ≠ ∅, x ∪ y = D), independence (else).
- **Composition rules for projecting relations**: Table 2 in the Bowman paper
  (derived from this work) defines valid inferences from pairs of relations.
  Informs future transitive relation inference in the genome.

**Where it applies:** The theoretical foundation for all NLI work in Agentome.
Every `NLRelation` enum value maps directly to MacCartney's definitions.

---

### 5. Sennrich et al. — BPE / Subword Tokenization Research

**Papers referenced during tokenizer evaluation:**

- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of
  Rare Words with Subword Units. *ACL 2016*.
  [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) — Original BPE
  for NLP.
- Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation
  Models with Multiple Subword Candidates. *ACL 2018*.
  [arxiv.org/abs/1804.10959](https://arxiv.org/abs/1804.10959) — Unigram LM
  (SentencePiece), used by DeBERTa-v3.
- Xue, L., et al. (2022). ByT5: Towards a Token-Free Future with Pre-trained
  Byte-to-Byte Models. *TACL*.
  [arxiv.org/abs/2105.13626](https://arxiv.org/abs/2105.13626) — Byte-level
  alternative (evaluated, not adopted).

**Design decision:** Keep DeBERTa-v3's native SentencePiece Unigram tokenizer.
Swapping to BPE or byte-level loses pretrained embeddings, which is the whole
value of fine-tuning. Our inputs are short enough (< 512 tokens) that tokenizer
efficiency is irrelevant. If training from scratch in the future, BPE-dropout
(Provilkov et al., 2020) is worth A/B testing for paraphrase robustness.

---

## Architecture Decisions Log

### 2026-04-08: DeBERTa Ribosome (replace Ollama for re_rank + splice)

**Problem:** Ollama ribosome takes 3-8s per turn for re_rank + splice. This is
an autoregressive model doing a classification/extraction job.

**Decision:** Fine-tune DeBERTa-v3-small as three task-specific heads:
1. Cross-encoder for re-ranking (MSE regression)
2. Binary classifier for splice (BCE)
3. 7-class NLI classifier (CrossEntropy)

**Rationale:** Encoder models process the entire sequence simultaneously vs.
autoregressive token-by-token. Expected ~400x speedup (80s → 200ms for batch
operations).

**Training data source:** The existing Ollama ribosome serves as "teacher" —
3,507 genes in the genome provide labeled examples. Teacher-student distillation
without manual annotation.

**VRAM budget:** 3 x DeBERTa-v3-small (~264MB FP16) + Ollama gemma4:e2b
(~2-4GB) = ~4.3GB total. Well within 12GB RTX 3080 Ti.

### 2026-04-08: Natural Logic Relations (MacCartney-Manning NLI)

**Problem:** Co-activation links are untyped (pure co-occurrence). Splice treats
codons atomically. No logical coherence metric.

**Decision:** Add 7-class NLI classification at Step 3.5 in the expression
pipeline. Store typed relations in `gene_relations` table. Use relations to:
- Bias splice decisions (entailment → keep together, alternation → drop one)
- Filter co-activation expansion (skip alternation links)
- Compute logical coherence metric (4th factor in ellipticity)

**Training data strategy (3 sources, no manual labels):**
- Heuristic: domain/entity overlap patterns → ~15k pairs
- Teacher: Ollama classifies sampled pairs → ~3k high-quality pairs
- Codon-level: splice decision patterns → ~5k pairs

**Risk:** Heuristic labels are noisy. Mitigation: teacher labels weighted higher,
confidence threshold (>0.6) for storage, start with 5 classes.

---

## Biological Metaphor Reference

For contributors unfamiliar with the biology:

| Term | Biology | Agentome Meaning |
|------|---------|------------------|
| **Gene** | Unit of heredity encoding a protein | A chunk of content with metadata |
| **Codon** | 3-nucleotide sequence encoding an amino acid | Semantic label for a content group |
| **Exon** | Coding region (kept after splicing) | Load-bearing content |
| **Intron** | Non-coding region (removed) | Filler content, removed per-query |
| **Promoter** | DNA sequence controlling gene expression | Tags that enable retrieval (domains, entities) |
| **Ribosome** | Molecular machine translating mRNA to protein | Small model that compresses/ranks/splices |
| **Epigenetics** | Heritable changes without DNA alteration | Access counts, decay, co-activation links |
| **Chromatin** | DNA packaging affecting accessibility | Gene state: OPEN, EUCHROMATIN, HETEROCHROMATIN |
| **Complement** | Paired DNA strand | Dense summary fallback |
| **Splice** | Removing introns from pre-mRNA | Dropping irrelevant codons per query |
| **Replication** | DNA copying | Packing Q&A exchanges back into genome |
| **HGT** | Horizontal Gene Transfer between organisms | Genome export/import between instances |
| **Ellipticity** | Measure of orbital shape | Composite context health score |
| **Denatured** | Protein losing its structure | Context is unreliable, high hallucination risk |
| **Co-activation** | Genes expressed together | Associative memory links |
| **Delta-epsilon** | CD spectroscopy signal | Divergence between auto and manual scores |

---

## Future Research Directions

### Sequence-to-sequence PACK replacement
The DeBERTa ribosome accelerates re_rank and splice but PACK still requires
Ollama (generative). A seq2seq encoder (e.g., T5-small fine-tuned for structured
extraction) could replace this, eliminating the Ollama dependency for ingest.

### Transitive relation inference
MacCartney's composition rules (Table 2 in Bowman) define valid inferences from
pairs of relations. E.g., if gene A entails gene B, and gene B entails gene C,
then gene A entails gene C. Currently we only store direct pairwise relations.

### Codon dependency graphs
Codons within a gene are currently a flat list. Adding parent/child relationships
(which codons are prerequisites for understanding others) would make splice
decisions context-aware rather than atomic.

### Tree-sitter integration for code chunking
`codons.py` line 127 notes: "MVP heuristic — swap for tree-sitter later." AST-aware
chunking would preserve class/method nesting, improving code gene quality.

### DNA-scale cold storage
Goldman et al.'s trit encoding could theoretically be applied to genome exports
for extreme archival durability. This is speculative but architecturally aligned.

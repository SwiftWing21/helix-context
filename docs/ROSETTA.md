# Rosetta Stone — biology lexicon ↔ software lexicon

Helix's original vocabulary borrowed from molecular biology (gene, genome,
ribosome, chromatin, splice, codon, promoter, epigenetics, transcription,
expression, replication). The metaphor is evocative and shaped a lot of
the original architecture, but it imposes a real cognitive tax: every
reader — human or LLM — has to hold two mental models in parallel.

The **canonical lexicon is now standard software terminology.** This
document is the bidirectional mapping. Legacy biology terms remain valid
references (the Python aliases live in `helix_context/aliases.py`), so
older handoffs, papers, and commit messages stay readable without
modification. New code, new docs, and new tool surfaces should use the
software vocabulary.

This is a *living document* during the rename effort. As new biology
terms surface in the codebase, add them here.

---

## Mapping table

The table is bidirectional: scan the left column to translate legacy
references, scan the right column to find the canonical term for new
work.

| Biology term (legacy) | Software term (canonical) | Notes |
|---|---|---|
| `Gene` | `Document` | The atomic unit of stored knowledge. Pydantic class identity unchanged; `Document is Gene` is True. |
| `genome` / `Genome` | `KnowledgeStore` / `kb` | The persistent SQLite-backed store. |
| `gene_id` | `document_id` / `doc_id` | Hash-derived identifier on the atomic unit. |
| `genes` (collection / table) | `documents` | SQL table name stays `genes` for now (deferred — see "Schema rename" below). |
| `PromoterTags` | `DocumentTags` | The `(domains, entities, intent, summary)` tuple attached to each document. |
| `promoter` (field on Gene/Document) | `tags` | |
| `EpigeneticMarkers` | `DocumentSignals` | Access rate, co-activation list, decay score. The behavioural metadata for retrieval scoring. |
| `epigenetics` (field) | `signals` | |
| `ChromatinState` | `LifecycleTier` | The hot/warm/cold storage tier enum. |
| `chromatin` (field) | `tier` | |
| `OPEN` / `EUCHROMATIN` / `HETEROCHROMATIN` | `OPEN` / `WARM` / `COLD` | Tier values. The numeric IntEnum values stay the same. |
| `codon` / `Codon` | `Fragment` / `Chunk` | The within-document compressed unit. |
| `codons` (field) | `fragments` | |
| `Ribosome` | `Compressor` | The small-model pipeline that encodes raw text into compressed documents. |
| `ribosome.pack(...)` | `compressor.encode(...)` | Take raw text → emit a compressed document. |
| `ribosome.splice(...)` | `compressor.trim(...)` | Drop low-value fragments from a candidate set. (Currently unwired in the pipeline; see ribosome.py:32.) |
| `ribosome.replicate(...)` | `compressor.persist(...)` | Pack a query+response exchange back into the store. |
| `ribosome.re_rank(...)` | `compressor.rerank(...)` | Re-score retrieval candidates with a small cross-encoder. |
| `transcription` | `encoding` | The text → compressed-document conversion. |
| `express` / `_express` / `expression` | `retrieve` / `_retrieve` / `retrieval` | The candidate-selection step in the retrieval pipeline. |
| `expression_tokens` (budget config) | `retrieval_tokens` | The token budget for retrieved context. |
| `replicate` / `replication` | `persist` / `persistence` | Saving query exchanges back into the store as new documents. |
| `harmonic_links` | `coactivation_edges` | Graph edges connecting documents that have been retrieved together. |
| `harmonic_bin_boost` | `random_walk_boost` | The Monte Carlo neighbour-expansion tier. |
| `gene_attribution` | `document_attribution` | The party/participant authorship metadata on each document. |
| `GeneAttribution` | `DocumentAttribution` | |
| `HGT` (horizontal gene transfer) | `cross_store_import` | Importing documents from another helix instance. |

---

## Terms that STAY (not biology, not tax)

These are domain-specific technical terms with established meaning
outside biology. Renaming them would lose precision or trade a
small cognitive cost for a bigger one.

- **`SEMA`** — semantic embedding alignment. Helix-coined but not biological;
  stays.
- **`TCM`** — Temporal Context Model (Howard & Kahana, 2002). Established
  psych-literature acronym; stays.
- **`cymatics`** — vibrational pattern math; the `cymatics.py` module
  references real physics, not biology.
- **`SPLADE`** — Sparse Lexical AnD Expansion model. Established sparse-
  retrieval term from the IR community.
- **`CWoLa`** — Classification Without Labels. Established weakly-
  supervised-learning acronym.
- **`ScoreRift`** — proper noun for the audit subsystem.
- **`PWPC`** — proper noun for the joint experiment with Todd's Celestia.
- **`helix`** itself — established product name. Renaming the package
  would be too disruptive for the cognitive-tax payoff.

---

## What gets renamed when

The rename ships in waves so back-compat stays solid throughout.

| Phase | Scope | Status |
|---|---|---|
| **R1** | Rosetta Stone doc + Python alias module + new MCP tool aliases (additive only) | shipping now |
| **R2** | Docstring + comment sweep — Python files and `docs/*.md` prose use canonical terms | next |
| **R3** | Internal symbol rename — non-exported helpers (e.g. `gene_input_vector` → `document_vector`). Pydantic class FIELDS stay (renaming them breaks the SQL schema). | after R2 |
| **R4** | Soft-deprecate legacy MCP tool names with docstring nudge. No removal. | after R3 |

### What we are explicitly NOT doing

- **No SQL schema rename.** Tables (`genes`, `gene_attribution`,
  `harmonic_links`, etc.) and columns stay. Renaming would force a
  migration on every existing helix instance; the cognitive tax at
  the SQL layer is paid by ~rare readers.
- **No removal of legacy class or tool names.** Only additions and
  docstring nudges. A future major-version cleanup may remove the
  legacy surface; until then both names work and resolve to the same
  underlying objects.
- **No rename of dated handoffs, papers, commit messages, or git
  history.** These are immutable historical artifacts; this Rosetta
  Stone makes them readable without modification.
- **No rename of the `helix-context` package itself.**

---

## How to use this document

**Reading legacy code or docs:** scan the left column for the term you
hit, get the canonical term from the right.

**Writing new code or docs:** use the right column. If a term you need
isn't listed yet, add a row.

**Importing canonical names:**

```python
# After R1 ships:
from helix_context.aliases import Document, KnowledgeStore, Compressor
from helix_context.aliases import DocumentTags, DocumentSignals, LifecycleTier
from helix_context.aliases import DocumentAttribution

# These are pure aliases for the legacy names. Identity holds:
from helix_context.schemas import Gene, PromoterTags
assert Document is Gene
assert DocumentTags is PromoterTags
```

**Calling MCP tools:** both names work. Prefer the canonical for new
client code.

```
helix_document_get(doc_id)   # canonical
helix_gene_get(gene_id)       # legacy alias, same behavior
```

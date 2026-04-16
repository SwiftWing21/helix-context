# Layered Fingerprints — Implementation Plan

**Design doc:** [../FUTURE/LAYERED_FINGERPRINTS.md](../FUTURE/LAYERED_FINGERPRINTS.md)
**Author:** Laude, 2026-04-16
**Target:** file-level parent genes that aggregate chunk fingerprints
and enable one-call reassembly

---

## Scope (v1)

File-level parent genes only. Directory+/project+/codebase+ parents
are deferred to a follow-up.

A parent is created at ingest time when and only when a file chunks
into **N ≥ 2** strands. Single-chunk files get no parent.

## Acceptance criteria

1. A multi-chunk file ingested via `context_manager.ingest()` produces:
   - N chunk genes (unchanged from today)
   - 1 parent gene with deterministic `gene_id = sha256(source_id + "::parent")[:16]`
   - N `CHUNK_OF` edges in `gene_relations`
2. Re-ingesting the same file (same content) is idempotent:
   - Same N chunk genes via content hash (unchanged)
   - Parent gene UPSERTed with same `gene_id` (content may differ if file changed; codons list refreshed)
   - Edges UPSERTed
3. `query_genes()` surfaces parent fingerprints when N ≥ 2 chunks of
   the same parent hit top-k. Co-activation bonus is applied.
4. `reassemble(parent_gene_id)` returns the file content stitched
   from children in sequence order.
5. Existing tests pass. New tests cover the 4 acceptance criteria
   above.

## Non-goals (v1)

- Directory/project/codebase-level parents
- Lazy parent creation at query time
- Garbage collection of orphaned children on re-ingest
- Parent handling in walking tie-break (defer to follow-up — `CHUNK_OF`
  as tie-break signal can be added later without breaking v1)
- Parent pull via librarian dispatch (the reassembly endpoint works;
  integrating with WALKER_PATTERNS is a separate task)

---

## Tasks

### T1 — Schema: add CHUNK_OF relation code

**File:** `helix_context/genome.py`

Find the existing `relation` enum/constants for `gene_relations`
(harmonic, semantic, etc.). Add:

```python
RELATION_CHUNK_OF = <next_available_int>
```

No schema migration needed — `gene_relations.relation` is already
`INTEGER`.

**Test:** `tests/test_layered_fingerprints.py::test_chunk_of_constant_is_unique`
asserts the new constant doesn't collide with existing relation codes.

**Est:** 15 min.

---

### T2 — Parent gene creation hook in ingest

**File:** `helix_context/context_manager.py`

In `ingest()` (around line 385-440), after the `for i, strand in
enumerate(strands):` loop completes, if `len(strands) >= 2`:

1. Compute `parent_gene_id = sha256((source_path + "::parent").encode()).hexdigest()[:16]`
2. Build the parent `Gene`:
   - `content` = first 1024 chars of the original `content` arg (not
     the assembled chunk content — the original file content)
   - `complement` = aggregated summary (for v1: join the top 3 child
     complements with `\n---\n`; ribosome-level aggregation is a
     follow-up)
   - `codons` = JSON list of child `gene_id`s in the order they
     appeared in `gene_ids`
   - `source_id` = `source_path`
   - `is_fragment` = 0
   - `key_values` = `{"chunk_count": len(strands), "total_size_bytes": len(content), "is_parent": True}`
   - Promoter fields: copy from first child (domains + entities), set
     `sequence_index = -1`
   - Epigenetics: inherit from first child
   - Embedding: None for v1 (could aggregate later)
3. `parent_gid = self.genome.upsert_gene(parent, apply_gate=False)`
   — bypass density gate; parents are metadata, not content
4. Insert `CHUNK_OF` edges: for each child `gid` in `gene_ids`, insert
   `(gid, parent_gid, RELATION_CHUNK_OF, 1.0, now())` into `gene_relations`

**Error handling:** wrap the parent-creation block in a try/except; a
parent failure must not fail the ingest. Log a warning and continue.

**Test:**
- `tests/test_layered_fingerprints.py::test_multi_chunk_file_creates_parent`
- `test_single_chunk_file_creates_no_parent`
- `test_parent_gene_id_deterministic_across_reingest`
- `test_chunk_of_edges_inserted`

**Est:** 45 min.

---

### T3 — Parent-aware query aggregation

**File:** `helix_context/genome.py`

In `query_genes()`, after per-chunk scoring finishes (around line 1982
where `last_tier_contributions` is assigned) and before the final top-k
truncation:

1. Find all candidate chunks that have a `CHUNK_OF` edge to a parent
   (single query: `SELECT DISTINCT gene_id_b FROM gene_relations WHERE
   gene_id_a IN (<candidates>) AND relation = RELATION_CHUNK_OF`)
2. For each parent with ≥ 2 chunks hit:
   - Aggregate `tier_contributions`: per-tier sum of children's contributions
   - Aggregate score: `parent_score = sum(child_scores) * (1 + 0.1 * log(n_hits))`
     (co-activation bonus, tuned later)
   - Insert parent into candidate set with aggregated fields
3. Apply dedup rule: if parent rank > any of its children's ranks,
   drop the children below position K. If parent rank < any child's
   rank, keep the child (it's a "this specific part matters more" signal)
4. Re-sort the final top-k

**Feature flag:** `HELIX_LAYERED_FINGERPRINTS=1` for A/B testing.
Default off in v1 until we measure impact.

**Test:**
- `test_two_chunks_hit_surfaces_parent`
- `test_one_chunk_hit_does_not_surface_parent`
- `test_parent_fingerprint_has_aggregated_entities`
- `test_dedup_drops_redundant_children_below_parent`

**Est:** 2 hrs (this is the trickiest piece — aggregation math and
dedup rule interact)

---

### T4 — Parent reassembly endpoint

**File:** `helix_context/genome.py`

New method:

```python
def reassemble(self, parent_gene_id: str) -> dict:
    """Reassemble a parent gene's full content from its chunks.

    Returns:
        {"content": <full text>,
         "source_id": <path>,
         "chunk_count": <n>,
         "reassembled_from": [<child_gene_ids in sequence order>]}

    Raises ValueError if gene_id isn't a parent or has no children.
    """
```

Implementation:
1. Load parent by gene_id, verify `is_parent: True` in key_values
2. Parse `codons` to get ordered child gene_ids
3. Batched SELECT: `SELECT gene_id, content, promoter FROM genes WHERE gene_id IN (?, ?, ...)`
4. Sort children by `promoter.sequence_index`
5. Concatenate `content` fields with `\n\n` separator
6. Return the dict

**FastAPI endpoint** (`helix_context/server.py` or wherever the proxy
lives): add `GET /reassemble/{gene_id}` → calls `genome.reassemble()`.

**Test:**
- `test_reassemble_roundtrip_matches_original_content` — ingest a
  3-chunk file, reassemble the parent, assert reassembled content
  equals original (up to chunk-boundary whitespace)
- `test_reassemble_rejects_non_parent_gene`
- `test_reassemble_handles_missing_child` (one child was deleted) —
  should log warning and skip, not crash

**Est:** 45 min.

---

### T5 — Tests + harness run

**File:** `tests/test_layered_fingerprints.py` (new)

All tests from T1–T4 consolidated in one file. Also:
- `test_fingerprint_push_includes_parent_when_multi_chunk_hit` —
  integration test against `/context` endpoint
- `test_feature_flag_off_preserves_current_behaviour` — with
  `HELIX_LAYERED_FINGERPRINTS=0`, query results bitwise-identical to
  pre-patch

Run `pytest tests/test_layered_fingerprints.py -v` — all green.
Also run full `pytest` to confirm no regressions in existing 158 tests.

**Est:** 30 min to write missing tests + run.

---

### T6 — Spot-check on real genome

Run a probe query against `C:/helix-cache/genome.db` (with backup
safety net in `E:\Helix-backup`):

1. Backfill script: for existing multi-chunk files in genome, create
   retroactive parents. `scripts/backfill_parent_genes.py`
2. Enable feature flag: `HELIX_LAYERED_FINGERPRINTS=1`
3. Run a representative query that previously returned 3+ chunks of
   one file in top-10 — verify parent surfaces, children get deduped
4. Measure: compare top-10 with/without flag on N=5 queries. Capture
   before/after in a short report.

**Est:** 45 min including backfill.

---

## Implementation order

T1 → T2 → T4 → T3 → T5 → T6.

T3 (query-time aggregation) is the hardest piece but is also
independent of T2 (ingest). We can build T2 + T4 first (easy wins
that just create parents and enable reassembly), land them behind
the feature flag, then do T3 as a separate review.

This lets us ship parent creation + reassembly as a minimal V1
without perturbing query behaviour at all — the flag gates the
behaviour change.

## Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| T2 breaks ingest | High | try/except around parent creation; feature flag |
| T3 changes retrieval quality | High | feature flag off by default; A/B before default-on |
| Parent gene_id collision with existing gene_id | Low | `sha256(source_id + "::parent")[:16]` is distinct from `sha256(content)[:16]` space |
| Reassembly whitespace mismatch | Low | Known limitation; reassembled content ≠ original bitwise due to chunk-boundary normalisation. Document. |
| Orphaned children on re-ingest | Low | GC is future work; orphans take space but don't affect correctness |

## Estimated total

T1 (15m) + T2 (45m) + T3 (2h) + T4 (45m) + T5 (30m) + T6 (45m) = **~5 hours**

Likely 1-2 sessions depending on context.

## Review checklist (before commit)

- [ ] All tests green
- [ ] Feature flag defaults to off
- [ ] `CHUNK_OF` relation code documented in the constants block
- [ ] `reassemble()` method documented with example
- [ ] No regressions in existing 158+ tests
- [ ] Spot-check on real genome confirms parent surfaces for multi-chunk hits
- [ ] FUTURE doc references this plan under "Open questions" if anything changed during implementation

# Delta-Change Ingest Sync — DEFERRED

**Status:** Deferred (2026-04-22). Council gate did not approve as an accuracy lever. Revisit after Pareto moves land and we have bench evidence on whether drift actually moves retrieval.

**Owner decision:** stub this design, pursue the Pareto moves from [RESEARCH_REVIEW_2026-04-22.md](../research/RESEARCH_REVIEW_2026-04-22.md) first. If those close the BM25-vs-Helix gap, delta-sync's accuracy framing falls away and this reduces to an operational-hygiene project; if they don't, re-open with the falsifier experiment at the top.

---

## What was proposed

Replace ~10 manual backfill scripts with a unified delta-change sync service:

- Watches configured folders; SHA256-based change detection (generalization of `mem_sync`)
- **Fill-empty-only invariant** on derived-index fields (filename_index, parent edges, kv tags, etc.)
- **Tombstone-and-reinsert** on source content change (preserves audit trail via `supersedes`)
- Unified with `mem_sync` — one daemon, one state file, one config section

Design questions answered during brainstorming (2026-04-22):

- **Scope:** Option C — unified with mem_sync; both content-sync AND derived-index refresh.
- **Content-change policy:** Option C — tombstone (chromatin=2) + reinsert; preserves audit trail, respects the existing volatility signal.

Design questions NOT yet answered:

- Trigger model (daemon poll vs. filesystem watcher vs. manual "sync now" vs. startup scan)
- Watch-target configuration surface
- Server-integration model (in-process thread vs. sidecar daemon)

---

## Council verdict (2026-04-22)

4 parallel research agents independently interrogated the accuracy claim. Convergent findings:

| Agent | Verdict | Key critique |
|---|---|---|
| Accuracy | **REJECT** as accuracy lever | BM25 gap driven by population dilution / broken PKI / content-delivery ceiling — none of these are sync problems. Estimated lift from closing the sync story: ≤1/8 content_full. |
| Schema carve-out | **Scope needs sharpening** | Only 5/10 backfills fill-empty-qualified. |
| Measurability | **Partly measurable** | Without a 4-cell factorial (control / flags-only / delta-sync / delta-sync+BM25), any win rides the coattails of other moves. |
| Ops reality | **Conditional approve for hygiene** | Current mem_sync has latent bugs (network-drive dismount → en masse tombstone), no observability, no first-run throttling. |

Full council transcripts preserved in task output files (2026-04-22 conversation context).

---

## Carve-outs (from council agent 2)

**Migrate to delta-sync (fill-empty qualified):**

- `backfill_filename_anchor.py`
- `backfill_path_key_index.py`
- `backfill_sema_embeddings.py`
- `backfill_gene_provenance.py`
- `backfill_parent_genes.py`
- `backfill_claims.py`

**Stay as one-shots (schema migrations — rewrite non-NULL by design):**

- `backfill_kv_tagger_fix.py` — formula changed post-f4c91e3, overwrites `key_values`
- `backfill_cwola_sema.py` — new columns per ALTER, not source-driven
- `backfill_cwola_sessions.py` — synthesizes session_id; bug-fix rewrite

**Retire or gate as manual rebuild (tunable heuristic, not a backfill):**

- `backfill_seeded_edges.py` — rename `rebuild_seeded_edges`, keep out of sync critical path

---

## Hard blockers before implementation

Must land before this is safe to ship, even as hygiene-only:

1. **Schema:** add `provenance_source` enum to `genes` table (`sync | operator | ingest`) so fill-empty can distinguish "operator deliberately cleared" from "never populated." Without this, operator-tuned fields get silently re-stamped.

2. **Safety:** tombstone-rate guard on sync pass — if >N% of tracked files would tombstone in a single pass, **abort the pass and alert**. Protects against network-drive dismount / UNC drop / typo-deleted watch_dirs. This bug exists in current `mem_sync` at [mem_sync.py:257-262](../../helix_context/mem_sync.py#L257).

3. **Observability:** `/mem_sync/status` endpoint + Prometheus counters (`helix_mem_sync_pass_total`, `..._errors_total`, `..._tombstones_total`, `..._last_success_timestamp`). Without these the daemon is a silent footgun.

4. **First-run throttling:** Cold-state scan against 17K-gene genome costs 30-150 min of writer-lock churn. Need rate-limited or "seed" mode that skips unchanged-by-mtime.

---

## Tombstone-reinsert break surface

Content change → new `gene_id` (SHA256[:16] of content at [genome.py:1344](../../helix_context/genome.py#L1344)) breaks these FKs (no CASCADE in DDL):

- `claims.claim_id_for` — mixes `gene_id.encode()` into hash at [claims.py:80](../../helix_context/claims.py#L80); every claim orphans
- `gene_relations`, `harmonic_links`, `entity_graph`, `promoter_index`, `path_key_index`, `filename_index`, `session_delivery_log`, `cwola_log.top_gene_id`
- `claim_edges.supersedes_claim_id` at [shard_schema.py:196](../../helix_context/shard_schema.py#L196)
- `genes.supersedes` intended for this case but nothing writes it except legacy `version` path — sync must populate `supersedes = old_gene_id` at tombstone time

CWoLa trainer tolerates tombstoned `top_gene_id` as NULL but historical buckets shrink.

---

## Falsifier experiment (from council agent 3)

**Cheapest single test** to decide whether drift-and-stale actually moves retrieval before committing to the sync design:

1. Snapshot current genome: `genome-2026-04-22-17022`
2. Ablate: drop 50% of `filename_index` rows + 30% of `provenance` rows at random (seeded)
3. Re-run `bench_multi_needle_50` against the ablated snapshot vs. the clean one
4. If no degradation > 1σ: drift hypothesis falsified. Abandon the sync angle; delta-sync ships only if operational cleanliness justifies it on its own.
5. If degradation clears 2σ: drift-and-stale is real; reopen this spec.

Bench cost: ~2 hours single-seeded run. Cheap relative to the ~500 LOC sync service with its prereqs.

---

## When to revisit

- After Pareto moves land (BM25 shortlist, `filename_anchor` flip, `helix_only` content ceiling fix, `source_index` path fix) — if the BM25-vs-Helix gap closes, the accuracy framing is moot.
- If operational pain (manual backfill cadence, drift confusion) surfaces repeatedly in on-call.
- If the falsifier experiment shows ablated-index retrieval degrades meaningfully.

Otherwise leave parked. Better designs may emerge once the Pareto work exposes actual bottlenecks.

---

*Brainstorming output preserved 2026-04-22 per superpowers:brainstorming skill. Council transcripts in task-output files from that session.*

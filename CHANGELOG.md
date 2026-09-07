# Changelog

## Unreleased

- **fix(scripts): `scripts/ingest_all.py` takes its sources from the command
  line.** Remove the built-in list of six source roots. The stale
  `cymatix_context.provenance` import had also stopped the script before
  argument parsing; it is repaired here (the module moved into the identity
  package in an earlier restructure). Missing `--sources` now prints usage and
  exits 2 unless `--sharded --agent-source` is supplied. Monolithic agent-only
  runs now error; sharded agent-only runs no longer also consider the removed
  corpus roots. Explicit-source parsing and selection are unchanged once the
  module is reachable. Test: `tests/test_ingest_all_sources.py`.

- **feat(launcher): graph summary panel on the Knowledge store tab.** The
  dashboard now shows how much of each graph layer the active genome actually
  holds: documents and how many of them a COVER relation touches, distinct
  undirected COVER links beside the stored row count, `CHUNK_OF` structure
  links, harmonic co-activation links, and distinct entities beside their
  membership rows. Implements
  `docs/superpowers/specs/2026-07-13-launcher-graph-summary-design.md`; it is a
  ledger, not a renderer, and nothing is drawn. New
  `cymatix_context/launcher/graph_summary.py` reads the genome over a
  short-lived `mode=ro` connection with `PRAGMA query_only=ON`, does not modify
  the genome, and caches by resolved path for 30 seconds, so the 2-second poll
  costs a dict lookup. Concurrent requests share one refresh per path, with
  the 30-second TTL starting when the read finishes. The read-only URI comes
  from `Path.as_uri()`, the form
  `cymatix_context/cli/cmd_status.py` uses: it percent-encodes, so a genome
  path containing `#` cannot re-parse and drop `?mode=ro`. The
  distinct-pair count needs a temp B-tree over every COVER row (measured 1.7 s
  at 800,000 rows, ~10 s at 7.2M), so it is skipped above
  `COVER_LINKS_MAX_ROWS = 500_000` and the panel reports stored rows instead.
  An unreadable store renders the unavailable state with the reason in the log
  and never on the page. Panel omits itself when the active genome file is not
  on disk. No new dependency, endpoint, config key or client-side script.
  `tests/test_launcher_graph_summary.py`.

- **fix(launcher): the collector publishes the real active genome path.**
  `_database_panel` lowercased `active_genome_path()` before publishing it as
  `state["database"]["active_path"]`. On a case-sensitive filesystem the result
  names a file that does not exist, so anything opening that string fails; the
  graph summary hands it straight to sqlite, which is how it surfaced. The
  registry's own `is_active` still compares case-insensitively, so a Windows
  genome selected as `F:/Genomes/...` and discovered as `f:/genomes/...` is
  unaffected.

## 0.9.2 (2026-09-05)

**Ingest at scale and the #411 default flip (PRs #424, #425), a
default-inert W2.2 combinator with its kill receipt (#428), a never-swallow
fix on the harmonic tier (#432), and the 0.9.2 bench ledger (#426, #427,
#429). One default moves — `[ingestion] entity_autolink_hub_cutoff` — and it
supersedes the 0.9.1 entry below that shipped the same knob at `0`.**

- **docs: hosted-session security, RBAC, and encryption review** lands as
  `docs/reviews/2026-09-01-hosted-session-security-review.md` — read-only
  source review of the auth/identity/retrieval-scope/encryption surface for a
  hosted multi-VM deployment, with a P0/P1/P2 roadmap; pointers re-verified
  against `beta` (drift noted in-doc) and the top finding filed as #434
  (`deploy/otel` compose publishes all ports with Grafana `admin/admin` +
  anonymous Viewer). No code or default changes.
- **DEFAULT FLIP: `[ingestion] entity_autolink_hub_cutoff` `0` → `200`
  (#425, closes #411).** The 0.9.1 entry below ("default `0` = off = legacy,
  byte-identical; flip proposal tracked in #411") is superseded — a fresh
  install now drops entities with more than 200 `entity_graph` postings from
  the auto-link probe set. Both #411 flip conditions are receipted:
  (1) **retrieval null** —
  `benchmarks/dogfood/receipts/entity_hub_cutoff_retrieval_ab_enronqa_2026-08-30.json`,
  the `enronqa_bed_v2` / `v2c` cutoff twins (0 vs 200, gene-id-set digests
  identical `158fa8a3…` = content-equality proof, n=500): delivered / r@12 /
  fr@12 / median rank identical to full precision, **0/500 per-needle
  delivered flips in both the 0.9.0 and 0.9.1 configs**; (2) **post-tagger-v2
  re-measure** —
  `benchmarks/dogfood/receipts/entity_hub_cutoff_reprobe_85k_taggerv1_2026-08-30.json`
  and `…_reprobe_85k_taggerv2_2026-08-31.json`, paired 500-document read-only
  replays on content-identical 84,677-gene beds (v2 = tagger v1, v3 = tagger
  v2): tagger v2 removes 15.2% of distinct entities (239,302 → 202,844) but the
  natural hub population persists and grows slightly (`enron` 15,397 → 16,048
  postings — junk removal frees slots in the 15-entity cap); at cutoff 200,
  **0.22% of entities (441) still hold 30.1% of postings**, and the mean
  per-link call falls **4.15 → 0.42 ms** at 85k (~340× in the #412 padded-bed
  receipt). What changes on disk: which relation=5
  COVER edges form (v3 replay sample: 4,548 → 2,984 edges, 2,746 lost / 1,182
  gained) — query-inert at shipped defaults (W2.1 cover-walk kill receipt,
  #408), and the padded 597k bed built at cutoff 200 is the same bed the 0.9.1
  gate sweep certified. [BASELINES](docs/benchmarks/BASELINES.md) rows:
  `2026-08-30-entity-hub-cutoff-retrieval-ab` and
  `2026-08-31-entity-hub-cutoff-reprobe-85k`. Scope: `IngestionConfig`, shipped
  `cymatix.toml`, and `docs/config-reference.md` move to `200`; the bare
  `KnowledgeStore` kwarg
  default stays `0` (non-config constructors keep byte-identical legacy
  linking — the `rrf_k` flip pattern); the bench builder is unchanged and reads
  the cutoff only from `CYMATIX_BFM_HUB_CUTOFF`. **Opt back in to legacy
  linking: `entity_autolink_hub_cutoff = 0`.** Tests:
  `tests/test_entity_autolink_hub_cutoff.py` (default assertion flipped),
  `tests/test_config_default_honesty.py` ratchet.

- **perf(storage): `idx_filename_gene` index on `filename_index(gene_id)` +
  `KnowledgeStore(entity_autolink=False)` scheduling knob (#424) — the
  residual >150k-document ingest decay is closed.** The per-upsert
  `DELETE FROM filename_index WHERE gene_id = ?`
  (`storage/indexes.py::sync_filename_index`) was a full-table scan (`EXPLAIN`
  = `SCAN filename_index`; 571,999 rows on the finished bed), i.e. O(N) per
  insert: **42.7 ms warm scan at 572k rows vs 0.002 ms indexed (~20,000×)**.
  Receipt
  `benchmarks/dogfood/receipts/ingest_residual_decay_enronqa_2026-08-30.json`
  (the cutoff-200 full rebuild, 596,707 documents in 4h58m against a ~26h
  uncapped trajectory): the O(N²) signature is gone, worker tagging throughput
  is flat ~50 docs/s to 580k, and the writer's residual decay is linear in N
  (38.4 → 32.7 docs/s over 479k → 656k processed) — it reconciles exactly with
  the unindexed DELETE; FTS5 stays refuted (0.07% of writer wall). The index
  is content-neutral (no document, tag, or edge changes); **existing stores
  gain it on their next writable open** (the DDL pass calls
  `filename_anchor.ensure_schema`). With it, bulk builds become tagging-bound
  (~3.3 h projected for the 517k-file bed). The scheduling knob —
  `KnowledgeStore(entity_autolink=False)`, and `--no-entity-autolink` /
  `CYMATIX_BFM_ENTITY_AUTOLINK=0` on `scripts/build_fixture_matrix.py` — skips
  COVER-edge formation while still writing `entity_graph` rows (equivalence
  receipt: only default-inert relation=5 edges differ); `sync_entity_graph`
  threads it with the hub cutoff as **off > cutoff > legacy**. Default
  unchanged (auto-link on). Narrative:
  `docs/benchmarks/2026-08-30-enronqa-ingest-decay.md`. Tests:
  `tests/test_entity_autolink_schedule.py`.

- **perf(ingest): `bed_provenance` table + `cymatix diag bed`, the
  `bulk_load` SQLite memory profile, and the bench-builder commit batcher —
  port of the never-committed 2026-08-24..27 perf/ingest lane (#438;
  worktree `claude/v0-9-0-planning-dc7564`, base `5f5c677`).** Every
  writable open
  now creates the append-only `bed_provenance` table (`CREATE TABLE IF NOT
  EXISTS`; content-neutral — no document, tag, or edge changes) so a bed
  carries its own history: `scripts/stamp_bed_provenance.py` and
  `scripts/build_erb_blob_v09x.py` append build / ingest / stamp rows
  (cymatix version, git sha + dirty flag, `ingest_c`, the ingest-affecting
  config subset, gene/FTS counts, optional gene-id digest), and
  `cymatix diag bed [--db PATH] [--identity] [--json]` reads them back over a
  `mode=ro` connection with config drift between rows — the bed-identity /
  `ingest_c` house rule in `docs/benchmarks/BASELINES.md` now has code behind
  it. The snapshotted knob set includes `[ingestion]
  entity_autolink_hub_cutoff` (#411/#425, added to `beta` after this lane's
  base), since it decides which `gene_relations` relation=5 COVER edges a bed
  carries. `CYMATIX_MEM_PROFILE=bulk_load` (single-writer ingest only; NOT for the
  server) gives the writer a page cache up to 4 GiB: 64 MiB vs 4 GiB measured
  **6.47 vs 16.16 genes/s** on the 22.7 GiB ERB bed with identical write
  volume, i.e. the whole 2.5× is upsert-path lookups
  (`benchmarks/dogfood/receipts/ingest_commit_batch_ab.json`,
  `…_cache64.json`). This closes two latent breaks in beta's already-shipped
  `scripts/build_erb_blob_v09x.py`, which imports
  `cymatix_context.storage.provenance` unconditionally (ImportError after the
  multi-hour build) and defaults `--mem-profile bulk_load` (silently fell
  back to the 64 MiB `auto` cache via `_MEM_PROFILES.get(profile, auto)`).
  Bench builder `scripts/build_fixture_matrix.py` (env knobs, no
  `cymatix.toml` surface): `CYMATIX_BFM_COMMIT_BATCH` (default `0` = inert
  commit-per-gene; `K` = one durable commit per K genes with
  `CYMATIX_BFM_WAL_VALVE_MB` / `CYMATIX_BFM_RAM_FLOOR_MB` early-flush valves;
  1.234 → 0.929 write MB/gene at 5000, content-equivalent across arms),
  `CYMATIX_BFM_MAX_FILES` (deterministic prefix cap for scale curves),
  **blob-mode `--rebuild` now honoured under `--parallel`** — a killed
  parallel blob build resumes into the existing `.db` (`_filter_to_unseen`
  drops only files with durable `bfm_completed_files` markers)
  instead of losing every gene; before, blob mode always rebuilt and silently
  ignored the flag. Completion markers commit with the file's final successful
  gene, so both blob and sharded resume retry files interrupted mid-chunk-batch
  by a pause or crash. Legacy beds without markers replay through idempotent
  upserts once instead of assuming any `source_id` proves the file complete.
  The **sequential** blob path is unchanged: it walks roots
  through `ingest_tree` and never materialises a file list, so there is
  nothing for `_filter_to_unseen` to filter and `build_profile` keeps the
  historical delete-and-rebuild there (with a warning naming `--parallel`)
  rather than re-ingesting every root into a bed that already holds them.
  So `--mode blob` with no flags behaves exactly as it did before this
  entry. Parallel drains are
  **ordered** by default (`imap` rather than `imap_unordered`) so
  `gene_relations` matches the sequential path (28,767 vs 28,765 rows
  observed; `ingest_equivalence_erb.json`: content_equal=true, 4.85× at
  workers=6). Also ships `scripts/ingest_commit_batch_ab.py`,
  `scripts/sqlite_cache_ab.py`, and the `ingest_write_path_ab.json`
  (deferred-index arm: negative, 0.05×), `ingest_scale_100k_curve.csv` /
  `_250k_curve.csv` receipts. Shipped server and CLI defaults unchanged.
  Tests: `tests/test_bed_provenance.py` (17), `tests/test_mem_budget.py`
  (+2), `tests/test_build_fixture_matrix.py` (+25).

- **feat(retrieval): `eps_band_coverage` combinator — opt-in, default-inert
  (#428; W2.2-narrow, KILLED by its own receipt, kept as an arm).** Identical
  ε-band walk to `eps_band`; inside a band, distinct-query-term coverage breaks
  the tie first (then rerank, then fused, then id). Coverage is computed only
  when a query's effective combinator selects it (top-60 fused head, one
  SELECT, `coverage_tiebreak` signal), so the shipped map (all five classes →
  `eps_band`) pays nothing. `retrieval/rerank_combinators.py::VALID_COMBINATORS`
  is now the single source of truth for the combinator names (validated at
  config load and in `KnowledgeStore.__init__`, replacing a drifted literal
  tuple). The evidence for it —
  `benchmarks/dogfood/erb/receipts/semantic_above_gold_947k_2026-08-31.json`,
  gold out-covers the interloper majority on 16/21 rank-reachable ERB semantic
  misses — and the kill —
  `benchmarks/dogfood/erb/receipts/w22_coverage_kill_2026-08-31.json`, full
  470-needle arm on the 947k bed vs the floor-12 reference: **delivered 0.6681
  → 0.6681, +0/−0 paired in every class**, 0/21 audit targets converted, fr@12
  +1 needle, 67/470 final-rank moves (the mechanism ran as designed). Cause of
  death is band geometry, not signal validity: at `rrf_k = 20` the fused head
  decays ~36% relative by rank 13, so rank-13+ gold never shares an ε band with
  the top-12, and widening δ reintroduces the unbanded-additive failure wave-1
  measured (semantic r@12 0.32 → 0.16). Opt in per class with
  `rerank_combinator_by_class = { <class> = "eps_band_coverage" }`. Tests: 8
  new in the combinator suite; `tests/test_sharded_adapter_parity.py`
  whitelists `_coverage_for` (per-shard `query_docs` helper, same convention
  as `_rerank_effective` / `_score_rerank`).

- **fix(retrieval): the harmonic tier warns and skips over
  `SQLITE_LIMIT_VARIABLE_NUMBER` instead of silently failing (#432, closes
  #431 tier 1).** Tier 5 binds the whole pre-shortlist candidate pool twice
  (`gene_id_a IN (...) AND gene_id_b IN (...)`). On bulk beds that pool comes
  off the unbounded tag lanes — receipt
  `benchmarks/dogfood/erb/receipts/harmonic_off_dilution_2026-09-01.json`
  (`harmonic_tier_liveness`, PR #429): ERB 947k median 171,745 candidates, max
  550,202, i.e. ~343k bound parameters against `SQLITE_LIMIT_VARIABLE_NUMBER`
  = 32,766 — so SQLite raised `OperationalError: too many SQL variables` and
  the bare `except Exception:` swallowed it at DEBUG; the tier never fired on
  any large bed and nobody was told (a never-swallow-rule violation).
  `query_docs` now probes `connection.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER)`
  (new `_sqlite_variable_limit` helper, fallback 999 where `getlimit` is
  unavailable), skips the harmonic query with **one `log.warning` per store
  instance** when `2 × len(candidate_ids)` exceeds it, and the bare except now
  logs at WARNING with `exc_info`. **Ranking is byte-identical below the
  limit** — same SQL, same bonus arithmetic, same `fuser.add_tier` call, pinned
  by `tests/test_harmonic_tier_limit.py` (dict-equal `last_query_scores` /
  `last_tier_contributions` against a never-skip run). `harmonic_links` holds 0
  rows on every current bench bed, so shipped ranking is unaffected either way;
  the behavior-changing variant (bound or batch the list so Tier 5 can fire on
  large populated beds) stays open on #431.

- **fix(retrieval): the ΣĒMA vectorless auto-gate now arms by a one-shot
  probe on full-pool queries, not only on the undersized-pool cycle (port of
  campaign-branch `2259cbf`; default-inert).** The gate (`_sema_vectorless`)
  was armed exclusively from `_build_sema_cache()`, which runs only when the
  lexical tiers leave the candidate pool undersized (`len(gene_scores) <
  limit // 2`) — so on a vectorless bed whose queries always fill the pool
  (100k / 829k) it never armed and every query paid the Tier-4
  `codec.encode()` RTT for a tier that cannot produce a candidate.
  `KnowledgeStore._probe_sema_vectorless()` closes that with a single
  `SELECT 1 FROM genes WHERE embedding IS NOT NULL LIMIT 1`, memoized per
  store lifetime by `_sema_probe_done` and re-armed by
  `invalidate_sema_cache()`; a failing probe degrades to the pre-gate
  behaviour. Beds with vectors are byte-identical (the probe finds a row and
  arms nothing). **At shipped defaults this is unreachable**: `[ingestion]
  sema_embed_on_ingest = false` (#371) leaves `_sema_codec` `None`, so only
  opt-in ΣĒMA beds are affected. `ShardedGenomeAdapter` gets the matching
  no-op shim. Tests: `tests/test_sema_vectorless_gate.py` (4 new; 3 were
  red on beta before the port), plus two entity-graph delivered-set
  blast-radius regression tests in `tests/test_d8_entity_graph.py` (port of
  `cf101bc`; already green on beta, pure coverage).

- **docs: the 2026-08-09 A/B data campaign's analysis and plan are ported
  from `claude/ab-data-cymatics-pki-dff9c6`** —
  `docs/benchmarks/2026-08-09-candidate-cascade-map.md` (`daaa6af`, "PKI
  admits zero gold") and `docs/superpowers/plans/2026-08-09-ab-data-campaign.md`
  (`fe2ef73`/`b164be4`), each with a port-provenance block; their ~5.4 MB of
  wave0a/wave3b/wave3d ladder receipts are archived in the `cymatix-receipts`
  repository (`mirror-branches/claude__ab-data-cymatics-pki-dff9c6/`), not
  ported, and ledger Addendum 7's receipt pointer now cites that mirror.

- **bench (no default moves):**
  - **#427 — 947k `min_delivered_docs` width curve, 12 confirmed:** delivered
    **0.6298 → 0.6447 → 0.6681** at floor 0 / 6 / 12, delivered sets strictly
    nested (+7/−0, then +11/−0), map + final ranks byte-identical across all
    three arms; floor 6 forfeits 11 needles because the classifier's 8-cap
    class is never lifted (BASELINES row `2026-08-31-floor-width-curve-947k`,
    receipt `benchmarks/dogfood/erb/receipts/floor_width_curve_947k_2026-08-31.json`).
    The EnronQA paraphrase crater is closed as a lane — lexical anchor loss on
    25/31 orig-hit/rephrase-miss pairs, coverage discrimination killed for
    that corpus (`docs/benchmarks/2026-08-30-enronqa-paraphrase-crater.md`);
    the ERB semantic-cap decomposition (`semantic_vocab_gap_947k` /
    `semantic_above_gold_947k` receipts) is what revived W2.2 for #428.
  - **#426 — semantic-portfolio lane: LoCoMo(+Plus), MULocBench, and
    FinanceBench adapters + baselines** (emitter → profile → resolver → ladder,
    the EnronQA pattern; all beds tagger-v2). Baselines on shipped v0.9.1
    defaults: LoCoMo delivered 0.4348 / r@12 0.4857 (n=2,378; the cognitive
    split is a 3/401 blackout that the paper's own dense baselines fail
    identically), MULocBench delivered 0.4485 / r@12 0.4824 cold-gated (the 0.444 / 0.481 warm run is marked deprecated in `muloc_budget_ab_2026-08-31.json`; n=680; exact-commit
    subset n=70: 0.500 / 0.571), FinanceBench delivered 0.153 (n=150;
    metrics-generated 0/50 — a doc-routing blackout). **New-corpus rows, NOT
    comparable to any ERB or EnronQA row.** Receipts under
    `benchmarks/dogfood/{locomo,muloc,financebench}/receipts/`; campaign doc
    `docs/research/2026-08-31-semantic-portfolio-lane.md`; the
    `expression_tokens` 7000 → 14000 cell is NULL on both corpus families.
  - **#429 — ERB 947k Phase-1/2 admission campaign, verdict doc
    `docs/research/2026-09-01-erb-phase2-admission-verdict.md`:** the
    admission thesis LIVES with corrections — 66 of the 109 pool-absent misses
    have gold at candidate depth 1000 (pre-registered kill under 30), but depth
    never improves gold's fused rank (0 better / 38 same / 69 worse of 107) and
    a bare depth-1000 flip projects **0.634 [0.572, 0.661] vs 0.668 shipped**;
    no default moves. Receipts
    `benchmarks/dogfood/erb/receipts/pool_depth_forensics_2026-09-01.json`,
    `pool_depth_addendum_2026-09-01.json` (+ `.NOTE.md`), and the four
    `verify_a1_*_2026-09-01.json` lenses.

- **Known (0.9.2):** `[budget] min_delivered_docs` does not floor the
  TIGHT/FOCUSED budget-tier cuts (`pipeline/tier_logic.py` `candidates[:3]` /
  `candidates[:6]`, applied before the floored classifier cap): **152 of 470
  ERB 947k needles deliver 6 seats at floor 12**, including 119 of the 314
  hits, and the seat count flips 6 ↔ 12 on 33/216 needles when only admission
  knobs move — every delivery-based A/B on that bed is confounded until the
  floor reaches the tier cut. Receipt
  `benchmarks/dogfood/erb/receipts/verify_a1_completeness_2026-09-01.json`.
  The fix (extend the same knob to the tier cuts) changes shipped seat counts
  on ~32% of ERB needles, so it is receipt-gated and tracked in #430, not
  patched here.

**ROSETTA Tier 2 — software-lexicon aliases (spec:
`docs/superpowers/specs/2026-08-30-v091-readme-wiki-rosetta-design.md`).**
Additive-only canonical-name aliases across the config, CLI, HTTP, and MCP
surfaces — every legacy bio-named surface keeps working unchanged. On the
config/env surfaces, collision (both names present) resolves legacy-wins
with a warning naming both, so no default or shipped behavior changes; the
CLI is deliberately exempt from that rule (see the `feat(cli)` entry
below — explicit dual-flag CLI input gets argparse's native last-wins
instead). Tier 3 — the wire-surface rename (`<GENE>` assembly blocks,
decoder prompts, response field names, `/stats` keys) — is out of scope
here because it alters delivered bytes; deferred to its own byte-level A/B
gate and tracked as issue #417.

- **feat(config): `[compressor]`/`[knowledge_store]` section aliases,
  `[budget]` key aliases, `CYMATIX_STORE_PATH` env alias, unknown-section
  warning (Task A1).** `cymatix_context/config.py` gains
  `_SECTION_ALIASES` (`compressor` → `ribosome`, `knowledge_store` →
  `genome`) and `_KEY_ALIASES` (`[budget] retrieval_tokens` →
  `expression_tokens`, `max_docs_per_turn` → `max_genes_per_turn`);
  `CYMATIX_STORE_PATH` aliases legacy `CYMATIX_GENOME_PATH`. Any
  top-level `cymatix.toml` section this loader doesn't recognize now
  warns via `_warn_unknown_sections` instead of vanishing silently —
  closes the sharpest pre-existing gap (an unrecognized `[compressor]`
  used to produce no signal at all). Collision rule: legacy name wins, a
  warning names both, so no default or shipped behavior changes. Review
  round 1 fixed two follow-on bugs: a `[budget]` alias/legacy collision
  was leaking a spurious "Unknown keys in [budget]" warning alongside the
  intended collision warning (the alias key is now always popped once
  handled, not only on the non-collision path), and a non-table alias
  value (e.g. `compressor = "notadict"`) was crashing `load_config` with
  `AttributeError` inside `RibosomeConfig` construction because the
  section-alias loop lacked the `isinstance` guard the key-alias loop
  already had — it now warns and leaves the section alone. That guard
  only covers the alias names; the same crash reproduces today for
  *legacy* section values (`ribosome = "notadict"`) on unmodified master
  with no Tier-2 code involved — filed separately as issue #418. Tests:
  `tests/test_lexicon_aliases_config.py`.

- **fix(api): `StatsResult` tier counts read the keys `stats()` actually
  emits (Task A2).** `CymatixSession`'s `StatsResult` construction read
  `chromatin_open`/`chromatin_euchromatin`/`chromatin_heterochromatin`,
  but `knowledge_store.stats()` emits `open`/`euchromatin`/
  `heterochromatin` — every tier count in `cymatix diag corpus` printed 0
  regardless of corpus state. Now reads the new keys with the old ones as
  fallback. Tests: `tests/test_stats_result_tiers.py`.

- **feat(cli): `cymatix document get|preview` alias, `--max-docs` flag
  alias (Task A3).** `document` is a first-class top-level subcommand in
  `dispatcher.py`, resolved to the same `cmd_gene.run` function object as
  legacy `gene` (identity-asserted in tests, not just behaviorally
  equivalent). `--max-docs` shares the `max_genes` argparse dest on
  `packet` and `refresh-targets`. CLI flags are deliberately exempt from
  the config/env legacy-wins collision rule: passing both `--max-genes`
  and `--max-docs` in one invocation is explicit user input, so
  argparse's native last-flag-wins applies as-is — pinned by
  `test_flag_alias_collision_last_wins_is_intentional_cli_exemption` plus
  a comment at each shared-dest `add_argument()` call, so the exemption
  is deliberate rather than an oversight. Tests:
  `tests/test_cli_document_alias.py`.

- **feat(server): `GET /documents/{id}` alias of `/genes/{id}` (Task
  A4).** Registered via `add_api_route` against the same handler with a
  distinct operation id, so there's no OpenAPI collision; JSON responses
  are byte-identical to `/genes/{gene_id}` for both the 200 and 404
  cases. `/genes/{gene_id}` itself is untouched. Tests:
  `tests/test_http_documents_alias.py`.

- **feat(mcp): `cymatix_document_*` tools join the lean core set, new
  `cymatix_document_neighbors`, R4 deprecation nudges (#87, Task A5).**
  `_MCP_CORE_TOOLS` grows to 11 — the five `cymatix_document_*`
  tools (`get`, `query`, `preview`, `fingerprint`, and the new
  `neighbors`) now ship in the default MCP surface alongside the
  original agent-loop five and `cymatix_announce` (added to the core set
  by the 0.9.1 model-contract work); the legacy bio-named tools stay registered
  only under `CYMATIX_MCP_FULL=1`. `cymatix_document_neighbors` is a
  pass-through-equivalent alias of `cymatix_neighbors` (same `query`/`k`
  params, same `/debug/neighbors` call). `cymatix_document_preview`'s own
  parameter is renamed `max_genes` → `max_docs` (still forwarded as
  `max_genes` on the wire) — safe since it has no prior callers pinning
  the old name; `cymatix_splice_preview` keeps `max_genes` untouched. R4
  (#87) soft-deprecation nudges added to the first docstring line of
  `cymatix_gene_get`, `cymatix_splice_preview`, and `cymatix_neighbors`,
  each naming its `cymatix_document_*` replacement — no removals. Tests:
  `tests/test_mcp_document_aliases.py`;
  `tests/test_mcp_tool_names.py`/`tests/test_mcp_server.py` updated for
  the 11-tool core. `cymatix_document_query` tracks its canonical
  original's 0.9.1 signature (`model` / `caller_model_class`, replacing
  `downstream_model`) so the alias stays signature-identical to
  `cymatix_context`.

- **docs: `mcp-tools.md` rewrite (document_* primary), alias-aware
  config-reference headings (Task A6).** `docs/api/mcp-tools.md` replaces
  a stale 1.8KB stub with the full 11-tool core (canonical names
  primary), a legacy-name back-compat table noting the R4 nudge policy,
  the 15 full-surface-only tools grouped by category, env vars (flagging
  that `CYMATIX_MCP_COMPAT` is documented but dead code), and a note that
  Tier 3 wire fields stay legacy-named for now (issue #417).
  `scripts/gen_config_reference.py` learns a canonical-heading map for
  sections with a Tier 2 alias — importing
  `cymatix_context.config._SECTION_ALIASES` off the loaded module rather
  than duplicating it — emitting `[compressor] (legacy alias:
  [ribosome])` and `[knowledge_store] (legacy alias: [genome])` headings;
  `docs/config-reference.md` regenerated, plus hand-authored prose for
  the `[budget]` key aliases (not separate dataclass fields, so the
  generated table can't express them).

- **docs(wiki): `wiki/` becomes the source of truth for both the GitHub
  wiki and cymatixcontext.com/wiki.** 15 pages (`Home`, `Getting-Started`,
  `Architecture-Map`, `Pipeline`, `Retrieval-Dimensions`, `Configuration`,
  `HTTP-API`, `CLI`, `MCP-and-IDE-Integration`, `Agent-Contract`,
  `Observability`, `Benchmarks-and-Receipts`, `Roadmap-and-Releases`,
  `Troubleshooting`, `Lexicon`) plus `_Sidebar`/`_Footer` and 4 shared SVG
  diagrams (pipeline flow, surfaces, token economics, defaults
  switchboard), all in engineering vocabulary with legacy biology terms
  called out inline. Two consumers render from the same Markdown:
  `scripts/sync_github_wiki.py` mirrors it verbatim into the
  `*.wiki` GitHub wiki repo (`--dry-run` supported), and
  `scripts/build_wiki_site.py` renders it into chrome-wrapped static HTML
  for `cymatixcontext.com/wiki/` (tag-scoped href/img-src rewrites so
  wiki-relative links and the shared SVGs resolve under the site's own
  path prefix). 14 new tests in `tests/test_build_wiki_site.py` cover the
  site-render path, including the Home-link and asset-src rewrites.
  The site render also builds a navigation sidebar from `wiki/_Sidebar.md`
  (the same file GitHub uses): a sticky desktop rail with the current page
  highlighted, collapsing to a `<details>` "Pages" block on mobile — JS-free.
- **readme: README.md rewritten 503 → 124 lines as a minimal landing
  page.** Install/quickstart/pipeline-summary/links only; the
  configuration tables, endpoint catalogue, gotchas, and lexicon detail
  that used to live in the README now live on the wiki (`Configuration`,
  `HTTP-API`, `CLI`, `Lexicon`, ...) instead of being duplicated — one
  place to keep current instead of two drifting copies.
- **docs: ROSETTA Tier-1 sweep — software-term prose pass across current
  (non-dated) docs.** Seven commits swapping biology-metaphor prose
  (gene/genome/ribosome/chromatin/splice, ...) for the canonical
  software vocabulary in `CLAUDE.md`, `docs/SETUP.md`,
  `docs/TROUBLESHOOTING.md`, `docs/api/context-endpoint.md`,
  `docs/api/endpoints.md`, `docs/architecture/{DIMENSIONS,
  KNOWLEDGE_GRAPH,OBSERVABILITY,PIPELINE_LANES,SESSION_REGISTRY}.md`,
  `docs/benchmarks/BASELINES.md`, `docs/clients/cli.md`,
  `docs/config-reference.md`, `docs/operations/DENSE_VRAM.md`,
  `docs/operator-runbooks.md`, the launcher
  dashboard/database-panel templates, and `cymatix_context/mcp/server.py`'s
  tool-description strings. Dated docs (benchmarks, council verdicts,
  dated plans, `docs/archive/`) are deliberately left in their
  point-in-time vocabulary. Notable fixes carried in the same sweep
  (found while touching the surrounding prose, not a separate audit):
  - `CLAUDE.md`: `pki_enabled` default-flip date corrected
    2026-08-19 → 2026-08-17 (matches issue #370).
  - `docs/TROUBLESHOOTING.md`: issues URL corrected to
    `github.com/mbachaud/Cymatix-Context/issues` (was pointing at a
    stale fork).
  - `docs/api/endpoints.md`: `/context/packet` response shape corrected
    to the actual `ContextPacket` (`verified`/`stale_risk` buckets, not
    a per-item `verdict` field), `GET /fingerprint` corrected to
    `POST`, the non-existent `/replicate` and top-level `/compact`
    routes corrected to `/consolidate` and `/admin/compact`, `/stats`
    tier keys corrected from `chromatin_*`-prefixed to bare
    (`open`/`euchromatin`/`heterochromatin`), and the
    `ANTHROPIC_BASE_URL` example removed (cymatix registers no
    `/v1/messages` route — OpenAI-chat-completions-shaped only).
  - `docs/architecture/OBSERVABILITY.md`: dashboard count corrected
    6 → 7 (stale since a dashboard was added).
  - `docs/api/context-endpoint.md`: `know_decision.py` /
    `know_calibration.py` source links repointed to their current
    `cymatix_context/scoring/` location (post-restructure dead links).
  - `docs/benchmarks/BASELINES.md`: de-duplicated a repeated
    `sema-readgate-decider` row.
  - `docs/architecture/KNOWLEDGE_GRAPH.md`: ASCII diagrams translated
    to engineering vocabulary (padding/headings preserved).
  - `docs/operator-runbooks.md`: a chromatin SQL literal (an actual
    column value, not prose) restored after an earlier pass
    over-translated it.
- **docs: `docs/ROSETTA.md` retires to a stub.** The full
  biology-to-software lexicon table moves to the wiki's `Lexicon` page
  (mirrored to both `github.com/mbachaud/Cymatix-Context/wiki/Lexicon`
  and `cymatixcontext.com/wiki/lexicon/`); the stub links there, links
  the Agentome paper (<https://mbachaud.substack.com/p/agentome>) for
  the "why biology in the first place" backstory, and tracks the
  remaining un-renamed wire/SQL surface (`gene_id`, the `genes` table,
  `/stats` keys, the legacy `<GENE .../>` inline tag — intentionally
  not renamed to avoid an on-the-wire break) on issue #417. `CLAUDE.md`
  pointers updated to match.
- **docs:** the wiki pages treat PR #419's canonical spellings as primary;
  that branch is merged into this one, so the two land in either order.
- **process: beta-branch release flow.** `beta` is the standing integration
  branch and, since 2026-09-01, the repository's **default branch** — new pull
  requests and fresh clones land on it. `master` holds released code and tags
  only; releases are `release/vX.Y.Z` cuts from `beta` merged to `master` by
  PR, hotfixes are `hotfix/*` from `master` back-merged into `beta`. CI runs on
  pushes to `beta` and `release/**` and on PRs targeting `beta` (master
  unchanged); `cla.yml` was already unfiltered. A sixth CI job,
  `release-source`, fails a pull request to `master` whose head is not
  `release/**` or `hotfix/**`, and is scoped by a job-level `if` to PRs against
  `master` so `beta` pushes and `beta` PRs skip it. **Branch protection is
  applied** on both branches (classic, `enforce_admins` false, no force-push,
  no deletion): `beta` requires `contributor-signup` plus the five test jobs;
  `master` requires those six plus `release-source` and a pull request with 0
  approvals. Pre-releases (`vX.Y.ZbN` on `beta`, GitHub pre-release,
  `publish.yml` to PyPI, `pip install --pre cymatix-context`) are the new way
  to ship an integration snapshot. Runbook: `docs/RELEASING.md` (branch model,
  receipt gate, exact commands, and the applied protection JSON as the reapply
  record); helper: `scripts/release.py` (`check` / `pre --n N` /
  `final --version X.Y.Z` / `tag-message [--body-only]`, dry-run by default,
  never runs git itself; `tests/test_release_script.py`); PR template
  `.github/PULL_REQUEST_TEMPLATE.md`; `CONTRIBUTING.md` "Branches and
  releases".
- **ci: ruff error-gate + `[nli]` extra in the full-suite job.** New fast
  `lint` job runs `ruff check --select E9,F63,F7,F82 .` (syntax errors,
  invalid comparisons, logic errors, undefined names — deliberately not a
  style linter; the selection is pinned in `[tool.ruff.lint]` so local
  `ruff check .` and CI agree). `test-full-suite` now installs
  `.[dev,ast,mcp,nli]` so the fully-mocked transformers-gated tests (incl.
  the #341 rerank device-routing test) stop silently skipping — torch is
  already the CPU wheel, so no CUDA wheel is pulled in. The gate's only
  hits on `beta` were 4 annotation-only forward references
  (`knowledge_store.py` `BGEM3Codec`/`np`, `tagger.py` `IntentClass`,
  `tests/conftest.py` `TestClient`), fixed with `TYPE_CHECKING` imports —
  no runtime import changes. Ported from the 2026-08-08 audit branch
  (`claude/codebase-audit-weaknesses-96a7da`, c193c20 + the f2b8a06
  `knowledge_store.py` rider).
- **bench: `benchmarks/bench_cymatix_rag_composition.py` imports repaired.**
  The script still imported `cymatix_context.lexical_rescue` /
  `chunk_fetch` / `relevance_window`, which moved under `retrieval/` and
  `encoding/` in the #90 restructure — it has been import-broken since.
  Repointed at the current module locations. `bench_needle.py` and
  `bench_claude_matrix.py`: `helix-context/helix.toml` restored as the gold
  label for the 11 toml needles — the frozen bench beds predate the 0.8.5
  rename, so the codemodded `helix-context/cymatix.toml` label matched no
  bed (`docs/benchmarks/MULTI_VALID_GOLD.md`); `cymatix.toml` stays as an
  ANY-match sibling for post-rename beds. Ported from audit commit 5c12c16
  minus its `ablate_cymatics.py` "peak_width is inert" annotation (false
  since #357 wired the knob; beta already carries the #354/#357 label fix).
- **chore: dead code removed.** `scripts/codemod_cymatix_rename.py`
  (self-neutralized since the 0.8.5 clean break — `OLD_PKG == NEW_PKG`;
  working version recoverable at 2e3f90e), `scripts/_write_tcm.py` (4-line
  dead scaffold), and `filename_anchor.remove_gene` (zero callers).
  Document deletes already sweep `filename_index` via
  `KnowledgeStore.delete_gene`'s `optional_tables` loop; the new
  `tests/test_filename_index_delete.py` pins that so it cannot regress
  silently. Ported from audit commit 363f700. Not ported from that branch:
  **f2b8a06 (concurrency) — deferred, not superseded**: only its
  `knowledge_store.py` `TYPE_CHECKING` rider is here (see the ci entry
  above), and three fixes it carries are still absent on `beta` — the SPLADE
  `_ensure_loaded` first-load lock (`splade_backend.py:112-142`, N-thread
  cold start constructs the model N times), the bounded freshness mtime
  cache with its `/admin/refresh` clear (`freshness.py:133,145` grow for the
  process lifetime; `routes_admin.py:653-658` never clears the cache that
  `context_manager.py:1211` documents it as clearing), and `blend.py`'s
  copy-under-lock score-map publication (`blend.py:204-207,235,283` still
  alias the #350 request-local dict into the shared
  `genome.last_query_scores`, plus the unlocked cold-tier write at
  `knowledge_store.py:1571-1574`). They need a hand re-apply against #350
  rather than a cherry-pick — **tracked in #439**. 44d0327 (docs) is
  genuinely superseded — stale claims retired by #354/#388, #357, #396 and
  865d34d — as is the README "Security" section, where the wiki
  `Configuration` and `HTTP-API` pages already carry a fuller, current
  equivalent.

- **bench: beta witness sweep + four new code corpora (2026-09-04).**
  `benchmarks/dogfood/sweeps/run_sweep_beta.py` re-ran every frozen v0.9.1
  receipt on the beta head `21606a0`: bit-exact on LoCoMo, FinanceBench,
  EnronQA v2 and EnronQA padded (with the 947k check: five corpora, 3,998
  paired needles, zero field diffs). MULoc's residual movement is the
  `PYTHONHASHSEED` hazard in `filename_anchor.py:164` (`list({...})[:64]`):
  139/141 movers are >64-term queries and seed 1 vs 0 alone moves 165, so the
  runner now pins and records the seed. RepoBench-R python reproduces the
  June arm; Java is measured for the first time. Four new code beds land with
  builders (`scripts/build_{coderag,cosqa,swebench}_corpus.py`), a generic
  gold resolver (`scripts/resolve_bench_needles.py`), fixture profiles, a
  document-level BM25 foil (`benchmarks/dogfood/code/bm25_doc_foil.py`) and
  their own BASELINES rows: CodeRAG-Bench solutions + docs, CoIR CosQA,
  SWE-bench Verified. Two defects surfaced and are tracked separately, no
  default moves: queries with >500 Stage-1 terms raise through
  `build_context` (Tier-2 UNION ALL vs `SQLITE_LIMIT_COMPOUND_SELECT`), and
  the Tier-1/2 tag lanes cost 12–25 NDCG@10 points on small-file code corpora
  under RRF (diagnostic arms). Write-up
  `docs/benchmarks/2026-09-04-beta-witness-sweep.md`.

## 0.9.1 (2026-08-30)

The retrieval-quality release: wave-1 ranking graduation (rrf_k 60→20 +
all-five-classes eps_band) and the W2.4 delivered-seat floor
(min_delivered_docs 0→12) ship as defaults, each receipt-gated on paired
delivered-basis evidence across ERB and the new EnronQA second-corpus
bench lane (gate receipt:
benchmarks/dogfood/receipts/sweep_v091_gate_2026-08-30.json, ALL PASS).
Ingest at scale: entity auto-link hub cutoff (default-off knob, ~340×
per-link-call, retrieval-null A/B) and tagger v2 entity hygiene
(TAGGER_VERSION=2, bed-comparability versioned). Cross-host client
unification (#406). Release gate: PRs #406–#409, #412, #413, #422.

- **tagger v2 — entity hygiene (#410), bed-comparability break.** The CPU
  tagger no longer emits email/MIME plumbing as entities: entities containing
  newlines/tabs are rejected, and standard header field names
  (`content-type`, `mime-version`, …), RFC 822 `x-*` extension headers, and
  transport artifacts (`javamail`, `quoted-printable`, …) are denylisted.
  Root cause of the EnronQA `entity_graph` hubs that made per-insert entity
  auto-linking O(N²) at build time (receipt
  `benchmarks/dogfood/receipts/ingest_decay_enronqa_2026-08-30.json`).
  Because tags are part of the bed-content digest, this ships as
  `TAGGER_VERSION = 2` (`cymatix_context/tagger.py`), recorded per bed in the
  fixture-matrix manifest; beds built at different tagger versions are not
  cross-comparable (rule added to `docs/benchmarks/BASELINES.md`). Fresh
  before/after receipt:
  `benchmarks/dogfood/receipts/tagger_v2_entity_hygiene_2026-08-30.json`.


- **perf(storage): `[ingestion] entity_autolink_hub_cutoff` — posting-count
  hub cutoff for entity auto-linking (default `0` = off = legacy,
  byte-identical; flip proposal tracked in #411).** The 2026-08-30 enronqa_padded profiling receipt
  (`benchmarks/dogfood/receipts/ingest_decay_enronqa_2026-08-30.json`)
  measured `auto_link_by_entity` at 89.5% of writer wall time — the
  per-insert GROUP BY sweeps every posting row of every probe entity, and
  MIME-header hubs (`content-type` 171k+ rows) make that O(N²) per build
  (148.9 ms/insert vs 7.4 ms without the two hubs). With the cutoff > 0,
  entities whose `entity_graph` posting count exceeds it are dropped from
  the probe set before the GROUP BY (`PKI_NOISE_CUTOFF` precedent —
  cardinality strictly greater is skipped); the count probe is
  LIMIT-bounded, so per-insert cost is O(n_entities × cutoff). This
  changes which relation=5 COVER edges form (default-inert at query time —
  W2.1 cover-walk kill receipt), so the flip proposal ships behind the
  default-off knob with an A/B edge-delta + perf receipt
  (`benchmarks/dogfood/receipts/entity_hub_cutoff_ab_enronqa_2026-08-30.json`,
  `scripts/receipt_entity_hub_cutoff.py`).


- **`[budget] min_delivered_docs` GRADUATED 0 → 12 (`[w24-floor-flip]`).**
  The W2.4 delivered-seat floor is now a shipped default: the classifier's
  per-rule assembly cap is lifted to at least 12 seats and the Stage-5
  budget trimmer truncates the largest parts instead of evicting whole
  documents (token budget still wins last — eviction resumes below the
  floor on overflow). Receipt-gated on three corpora with zero paired
  delivered losses: ERB 829k .630→.668 (+18/−0,
  `benchmarks/dogfood/erb/receipts/ladder_v09x_w24_min_delivered_2026-08-28.json`),
  EnronQA v2 .820→.856 (+18/−0) and EnronQA padded 597k .776→.792
  (+8/−0) (BASELINES row `2026-08-30-v091-gate-sweep`); r@12/fr@12
  identical in every cell. `min_delivered_docs = 0` restores legacy;
  revert = `git revert` of the `[w24-floor-flip]` commit.
## 0.9.0 (2026-08-20)

The post-flip release: the shipped default retrieval path is now fully
algorithmic — `dense_embedding_enabled`, `splade_enabled`, `pki_enabled`,
`dense_embed_on_ingest`, and `sema_embed_on_ingest` all default `false`,
each flip receipt-gated (entries below). Release-gate tracker: #377.

**Release-notes disclosures:**

- **know surface:** KnowBlock never fires at shipped defaults (max confidence
  ≈0.28 vs emit_floor 0.45; the `lexical_dense_agree` calibration feature is
  structurally dead on the neural-free path) — fail-safe direction (0%
  false-KNOW); recalibration tracked as #287 in the 0.9.x backlog (receipts:
  `benchmarks/dogfood/erb/receipts/postflip_know_abstain_sanity_*.json`).
- **p50 latency:** the dense-off default carries the measured p50 regression
  disclosed under #374 (see the `dense_embedding_enabled` entry below —
  ×2.5–2.6 at 100k shrinking to ×1.09–1.38 at 829k); the mitigation (lex-branch
  candidate cap) has not landed and stays open on the retrieval-layer ledger.

**Known deferred (0.9.x):**

- Deferred work is sequenced in #377's 0.9.x deferral-ledger comment
  (2026-08-19) — #366, #349, #344, #340, #336, #287, #275, #260, #205, #373,
  #356, #355, plus the #351 knob decisions.
- entity_graph layer ships ON but has never been ablated on the delivered
  basis (ledger REMOVE-CANDIDATE); the arm is tracked in #377's 0.9.x backlog.

**Ops:**

- **ops(beds): 2026-08-20 bench-bed maintenance** — 36.6 GB reclaimed across
  the two 829k benchmark beds (`path_key_index` drop + FTS5 external-content
  migration + vacuum); receipt-comparability flag recorded on #370; new bed
  bytes and reproduction notes in `docs/benchmarks/BASELINES.md`
  (bed-state note, 2026-08-20).
- **security: `[budget] neutralize_control_tags` default flipped `false` →
  `true` (#351 decision 2).** Scope: assembly-time escaping of `<cymatix:`
  control tags in retrieved content — a document whose content contains the
  literal `<cymatix:` can no longer forge the genuine
  `<cymatix:no_match/>` / `<cymatix:slate>` control tags that downstream
  agents adopting `CYMATIX_NO_MATCH_FRAGMENT` trust (the escape rewrites it
  to `&lt;cymatix:`; the genuine no-match tag, emitted on the separate
  parts-empty branch, is never escaped). The escape runs strictly AFTER
  retrieval/fusion/tie-break/rank publication, so retrieval metrics are
  unchangeable by construction; the flip gate was the delivered basis and
  assembled bytes. Receipt
  (`benchmarks/dogfood/erb/receipts/neutralize_ab_100k.json`, 141-needle
  100k-carve replay, both arms explicit): r@12 0.6879 = 0.6879, fr@12
  0.6950 = 0.6950, delivered_gold_rate 0.6099 = 0.6099, per-needle
  ranks/delivered ids identical, **141/141 windows byte-identical**, and
  the 5 genuine abstain-tag windows untouched in both arms. The bed carries
  zero control-tag docs (receipt `bed_control_tag_scan`), so the positive
  case — the escape fires where a tag exists and only there — is carried by
  the synthetic assembly-function A/B unit tests in
  `tests/test_control_tag_neutralization.py`. Opt out with
  `neutralize_control_tags = false` (documents legitimately containing the
  literal `<cymatix:` string then ship verbatim — byte-identical to
  pre-knob behavior). Scope limit unchanged: closes tag forgery only;
  general indirect prompt injection via document text is inherent to
  context injection.
- **security: startup warning on non-loopback bind with empty
  `admin_token` (#351 decision 1).** `create_app` now logs a prominent
  `NETWORK-EXPOSED ADMIN SURFACE` warning when `[server] host` is not
  loopback (`127.0.0.1` / `localhost` / `::1`) while `[server] admin_token`
  is empty — that configuration leaves `/admin/*` (including
  `/admin/shutdown` and `/admin/swap-db`), `/ingest` and `/consolidate`
  reachable by anyone on the network with no authentication. Warning only:
  the bind is honored unchanged, loopback binds and token-set configs stay
  silent.

- **ingestion: `[ingestion] sema_embed_on_ingest` default flipped `true` →
  `false` (#371).** Ingest-time behavior change only — this completes #371's
  sema half (the dense half is the entry below). Consumer audit first
  (PR #399): at shipped defaults the ONLY live retrieval-path consumer of
  `gene.embedding` is the Tier-4 `sema_boost` re-rank — cymatics reads tags
  (`[cymatics] use_embeddings` off), cold tier is opt-in, TCM's tag-hash
  fallback is clean, and the freshness gate / packet know/PLR read no
  embeddings. Deciding cell (PR #400,
  `benchmarks/dogfood/erb/receipts/sema_readgate_829k_n469.json` — `no_sema`
  read-gate arm on the 829k bed, n=469, delivered basis, paired per-needle):
  a wash — delivered gold 264/469 → 265/469 (+1 needle), r@12/fr@12/median/
  mean rank byte-identical, 0 recall flips, 0 final-rank moves, abstain set
  byte-identical (16/469, all pre-existing). The dogfood-scale A/B (PR #399,
  n=18) had measured −2 delivered needles (one `[abstain]` floor coupling,
  one budget-boundary cut); the coupling does not reproduce at 829k. Riders:
  cold start −~9.7 s (the SEMA/MiniLM load was 97.7% of first-request
  latency, `cold_start_postflip.json`), ingest wall 835.9 s → 761.5 s on the
  1,181-file dogfood corpus, −0.48% bed bytes. Re-opt-in story: set
  `sema_embed_on_ingest = true` — new ingests embed; docs ingested while the
  knob was off have NULL `gene.embedding` until `scripts/backfill_sema.py`
  (or re-ingest/consolidate), and TCM's tag-hash fallback covers vectorless
  docs by design (#227). Explicit `sema_embed_on_ingest = true` configs keep
  their behavior. Follow-up (not a flip blocker): `[abstain]`
  floor-recalibration is tracked on #371's closing comment / the #377
  ledger — 13/16 delivered-adjacent floor abstains pre-exist at shipped
  defaults and are not moved by this knob.
- **ingestion: `[ingestion] dense_embed_on_ingest` default flipped `true` →
  `false` (#371).** Ingest-time behavior change only — retrieval is untouched:
  at shipped defaults (`[retrieval] dense_embedding_enabled = false` since
  2026-08-15) nothing reads `embedding_dense_v2`, and the sharded read path
  hard-wires dense off, so the inline BGE-M3 encode (~2 GB model load, one
  *unbatched* extra encode per multi-chunk file for the layered-fingerprint
  parent) was a dead write. PR #379's harness smoke
  (`benchmarks/dogfood/run_ingest_throughput.py`, 12 files, CPU encode):
  defaults 126.8 s wall / 4084 ms steady per-file / 3.47 GB peak RSS vs
  `no_dense_ingest` 20.9 s / 494 ms / 0.82 GB — a 6.1× wall speedup and
  −2.6 GB RSS from this knob alone. Re-opt-in story: set
  `dense_embed_on_ingest = true` to resume writing vectors at ingest; docs
  ingested while the knob was off have NULL vectors, so re-enabling dense
  RETRIEVAL on such a bed additionally requires `scripts/backfill_bgem3_v2.py`.
  Existing beds with backfilled vectors are unaffected, and explicit
  `dense_embed_on_ingest = true` configs keep their behavior. The sema half of
  #371 (`sema_embed_on_ingest`) was deliberately NOT flipped here — it
  needed its own receipt, which landed via PRs #399/#400 (see the entry
  above).
- **config: #219 slice 5 — dark-feature labels, dead-knob removal, and the
  `[plr] expected_sha256` wiring gap.** The default-honesty pass for knobs
  the loader accepted but the runtime ignored or half-honored:
  - **Removed five dead knobs** (parsed with ZERO runtime readers — verified
    repo-wide, the only Python references were `config.py`'s own parse
    lines): `[ingestion] colbert_enabled` (Phase-4 scaffold, never
    implemented), `[retrieval] seeded_edge_weight` (the seed-insertion
    weight is hardcoded `0.7 + 0.3*cos` in
    `scripts/backfill_seeded_edges.py`), and `[cymatics] n_bins` /
    `use_embeddings` / `splice_threshold_scale` (the spectrum resolution is
    the module constant `scoring/cymatics.N_BINS`; the splice threshold
    derives from `[budget] splice_aggressiveness`; no embedding path ever
    read `use_embeddings`). The loader warns-and-ignores unknown keys
    (`_warn_unknown`), so existing configs still carrying them get a startup
    WARNING, not a failure — remove the lines to silence it.
  - **Wired `[plr] expected_sha256`** (previously parsed but never passed):
    the server's `get_fuser` call now threads the configured pin through to
    `StackedPLRFuser.load`, so an operator-pinned artifact hash actually
    verifies and a mismatch refuses to load (PLR soft-fails to
    "unavailable", packets ship without `plr_confidence`). Byte-identical at
    the shipped default `""` — empty falls back to the sidecar-`.sha256`
    path exactly as before.
  - **Labeled the three dark-but-wired features EXPERIMENTAL** with evidence
    and graduation criteria instead of bare "Dark ship" comments:
    `[retrieval] sr_enabled` (measured zero retrieval effect),
    `ray_trace_theta` (requires a TCM velocity input the default pipeline
    does not populate; never measured), `seeded_edges_enabled` (starts
    evidence writes on every retrieval; never measured). All three stay
    default-off; each graduates only with a fresh isolation receipt.
  - `docs/config-reference.md` regenerated; its "Default cymatix.toml"
    appendix no longer shows `sr_enabled = true` (stale since the
    2026-06-12 honesty pass).

- **retrieval: `[retrieval] pki_enabled` default flipped `true` → `false`
  (#370).** The 829k exact-shipped-default confirm at full statistical power
  (n=469, delivered basis,
  `benchmarks/dogfood/erb/receipts/postflip_default_confirm_829k.json`)
  measured PKI on as **−1 delivered needle** vs `no_pki` (263/469 = 0.5608 vs
  264/469 = 0.5629; r@12 0.6546 vs 0.6588, fr@12 0.6588 vs 0.6631) —
  null-to-hair-negative. Honest caveat: the earlier n=30 cells were
  delivered-identical at 250k/500k/829k-probe but the 100k cell was **+1
  delivered needle for PKI** (the one positive cell, replicated in both run
  orders); the full-power n=469 cell is −1. Opt back in per store with
  `pki_enabled = true` — one config line. Note the flip does NOT reclaim the
  8.56 GB `path_key_index` on existing beds; the scorer-neutral compaction is
  #370's other checkbox and stays open. Receipts:
  `docs/benchmarks/2026-08-14-encoder-isolation-scale-curve.md`.
- **feat(mcp): migrate the MCP server to mcp 2.x and lift the #325 pin (#326).**
  `cymatix_context/mcp/mcp_server.py` now uses `mcp.server.mcpserver.MCPServer`
  (2.x home of the removed `mcp.server.fastmcp.FastMCP`); requirement is
  `mcp>=2,<3` in both the `mcp` and `all` extras. Tool surface is unchanged —
  lean 5-tool core, `CYMATIX_MCP_FULL=1` opt-in, schemas byte-identical vs 1.x.
  Spec: `docs/design/2026-08-16-mcp-2x-migration.md`.

- **ingestion: `[ingestion] splade_enabled` default flipped `true` → `false`.**
  The n=469 isolation receipts on the 829k blob measured query-side SPLADE
  over the full 147M-row expansion index as null-to-negative vs the lexical
  floor (pool −0.004, delivered −0.006, ~15% slower; both repeats identical),
  and the ingest-side bed A/B showed the expansion index contributes nothing
  passively — the all-off floor is byte-identical with and without it. With
  dense also flipped (below), **the default retrieval path is now fully
  neural-free**. Opt back in per store with `splade_enabled = true`; existing
  beds keep their expansion tables (unused until opted in). Receipts:
  `docs/benchmarks/2026-08-14-encoder-isolation-scale-curve.md`.
- **retrieval: `[retrieval] dense_embedding_enabled` default flipped `true` →
  `false`.** Four-scale isolation receipts (100k/250k/500k carves + the 829k
  blob at the full 469-needle set, every order-reversed pair replicated
  exactly) measured BGE-M3 dense recall *displacing gold documents from the
  delivered top-k* at every scale — final recall −0.20..−0.33 vs the lexical
  floor (−0.207 at n=469, ~97/469 needles) — while adding at most +0.02
  candidate-pool recall. **Disclosed cost (#374):** the same receipts measure
  a p50 latency regression from the flip — the dense-off floor is **×2.5–2.6
  slower than dense_on at 100k** (p50 10.4s/9.7s vs 3.9s/3.8s,
  `benchmarks/dogfood/erb/receipts/enc_iso_100k_run{1_fwd,2_rev}.json`),
  ×2.3–2.4 at 250k, ×1.27–1.37 at 500k, and ×1.09–1.10 at 829k on the n=30
  pairs (×1.15–1.38 on the full-power n=469
  `enc_iso_829k_dense_confirm_run{1,2}.json` pair). Mechanism (retrieval-layer
  ledger claim 1, `docs/benchmarks/2026-08-06-retrieval-layer-ledger.md`):
  dense was load-bearing as a *latency device* — its ANN gate capped the
  candidate list feeding splice, and dense-off degrades the median query to an
  uncapped lex pool. The ledger's precondition — "cap the non-ANN branch's
  candidate list before dense-skip is viable" — has **not** landed (the
  2026-08-12 refinement plan kept dense load-bearing until it did); the
  lex-branch cap stays open in the ledger's claim-1 line, alongside the
  related #336 splice char-target dynamics. Dense is now opt-in per store; beds with backfilled
  vectors are unaffected until you set the flag. Existing configs that
  explicitly set `dense_embedding_enabled = true` keep their behavior.
  Receipts + method: `docs/benchmarks/2026-08-14-encoder-isolation-scale-curve.md`.

- **perf(storage): #338 FTS5 external-content rebuild.** `genes_fts` no longer
  stores its own full copy of every document (the `genes_fts_content` shadow —
  8.67 GB / 18.5% of the 829K blob bed): new stores are created as
  external-content FTS5 backed by the `genes_fts_source` view, which reproduces
  the exact composite text the contentful table indexed, so tokenizer, BM25
  stats, and reader queries are unchanged (receipted byte-identical on the ERB
  10k bed: `benchmarks/dogfood/erb/receipts/fts5_external_content_migration_10k.json`).
  Legacy contentful stores keep working untouched;
  `scripts/migrate_fts5_external_content.py` converts them offline (idempotent,
  `--dry-run`, `--vacuum`). Write paths follow the external-content
  delete-with-prior-values discipline; `rebuild_fts` uses the FTS5 `'rebuild'`
  command on migrated stores. **Stub caveat** (cc-exchange 0017-joe §6): the
  contentful shadow accidentally retained pre-compression original text for
  stubbed genes; the migration refuses on stub-bearing stores unless
  `--allow-stub-loss` is passed — signal-neutrality is conditional on a
  stub-free store.

- **feat: query-time cross-encoder rerank, default-off (#341).** New
  `[retrieval] rerank_enabled` (default `false`), `rerank_depth` (`50`),
  `rerank_model` (`"cross-encoder/ms-marco-MiniLM-L-6-v2"`) and
  `rerank_enabled_by_class` (`{}`, the #255 per-class map). The cross-encoder
  runs **pre-cap** inside the knowledge store (`query_docs_ann` pre-gate;
  `query_docs` post-fusion on the dense-off profile) and is
  count-preserving — the store returns the same number of documents either
  way, only membership and order change. `[hardware] rerank_device` is now
  actually consumed (`resolve_layer_device("rerank")` at model load). The
  encoder daemon gained `POST /encode/rerank` with
  `EncoderClient.encode_rerank` and a `rerank_backend` remote branch at
  circuit/ready-gate/fallback parity with the SPLADE seam; `/ready` now
  waits on the rerank family too. Default-off is deliberate: it pays only on
  populations with near-cutoff rerank headroom, and costs ~270–560ms/query on
  a GPU daemon at 829k documents (~285–315ms at 100k). Receipts and the
  population caveat:
  `docs/benchmarks/2026-08-07-rerank-wiring-receipts.md`.

- **Breaking: removed `[ingestion] rerank_enabled`.** The knob was dead or
  wrong on every backend: under `deberta` the serving path looked for a
  `rerank` attribute while `DeBERTaRibosome` only defines `re_rank`, so the
  cross-encoder never ran; under `ollama`/`litellm` it misrouted to the LLM
  `Compressor.rerank`. The post-cap branch in `scoring/blend.py` is gone with
  it. **Configs that set it now get no rerank** unless they also set
  `[retrieval] rerank_enabled = true`. `[ingestion] rerank_model` is
  unchanged (still the legacy DeBERTa constructor input).

## 0.8.6 — 2026-08-05

GPU encoder daemon, executor sizing, ERB scale fixes. (This header was added
retroactively on 2026-08-14 — the tag and GitHub Release shipped 2026-08-05/06
without a changelog section.)

- **feat: `[encoder_daemon]` — shared GPU encoder daemon (fork1 slice 1).**
  New `cymatix_context.encoder_daemon` FastAPI process (default port 11439)
  with an `EncoderRegistry` + per-family micro-batcher; dense/SPLADE/SEMA
  seams route through it when `[encoder_daemon] url` (or
  `CYMATIX_ENCODER_URL`) is set, byte-identical when off. Client side ships
  `RemoteBGEM3Codec`/`RemoteSemaCodec` drop-ins with URL resolution, a circuit
  breaker, readiness gating, and text-count-scaled batch timeouts.
  `CYMATIX_ENCODER_BATCH_WINDOW_MS` default lowered 8 ms → 2 ms
  (receipt-driven).
- **perf(hardware): per-layer encoder device knobs + CUDA A/B** — BGE-M3
  22.7× batched on a 3080 Ti; deconfounded GPU-vs-CPU ladder receipts
  (−41% at dogfood scale, wash at ERB scale). GPU-daemon worker sweep found
  the x4 executor sweet spot.
- **feat(retrieval): #327 `authority_path_selectivity` knob** (default-off —
  it regresses alone; the seam landed, not the fix).
- **fix(server): #329 three-state upstream health** — an inactive ribosome no
  longer degrades `/health`.
- **bench(erb): per-layer ablation ladder + storage audit** — 7 arms survive
  gate verification (3 dropped ungateable, 3 identity); blob-scale storage
  audit receipts.

## 0.8.5 — 2026-07-25

Completes the helix → cymatix rename as a **clean break** (0.8.0 was the soft
rename; 0.8.5 removes all of its back-compat). **Breaking:** if you still use
the old names, migrate or pin `cymatix-context<0.8.5` (0.8.0 keeps the aliases).

- **rename: removed all helix back-compat.** Deleted the `helix_context` alias
  package (`import helix_context` now raises `ModuleNotFoundError`), the `helix*`
  console scripts, the `helix_*` MCP tool aliases (and the `CYMATIX_MCP_COMPAT`
  machinery), the `CYMATIX_*→HELIX_*` env mirror (internal reads are `CYMATIX_*`
  only), and the `helix.toml` config fallback (`cymatix.toml` / `$CYMATIX_CONFIG`
  only). The MCP surface is the 5 canonical `cymatix_*` core tools, no aliases.

- **rename: internal identifiers + observability.** `Helix*` classes → `Cymatix*`
  (`CymatixConfig`, `CymatixContextManager`, `CymatixError`, `CymatixSession`);
  loggers `helix.*` → `cymatix.*`; OTel metrics `helix_*` → `cymatix_*`, spans
  `helix.pipeline.*` → `cymatix.pipeline.*`, and the 7 Grafana dashboards
  (uids/titles/queries) renamed to match; agent protocol tags `<helix:…>` →
  `<cymatix:…>`; portable bundle format is now `.cymatix` /
  `cymatix_format_version` (import still accepts the legacy `helix_format_version`
  key).

- **compat: data stays readable.** Existing `genome.db` opens unchanged; new
  launcher/observability state lands under `~/.cymatix` / `cymatix-context` with
  a read-fallback to the old `~/.helix` / `helix-context` locations.

- **feat: `cymatix_context.__version__`.** Single-source version export (pinned
  to `pyproject.toml` by test); `/health` carries a `version` field.

- **ci: full offline suite job.** `pytest tests/ -m "not live"` (~3.4k tests)
  runs on every push/PR (the eager app lives only in `_asgi.py`).

- **docs: drop dangling `*_DAEMON_DESIGN.md` pointers.** References now point at
  `cymatix-server` for the long-lived HTTP surface.

## 0.8.0 — 2026-07-22

- **rename: helix-context → cymatix-context.** Canonical package is now
  `cymatix_context`; the old `helix_context` package is a live alias —
  every submodule import resolves to the identical `cymatix_context`
  module object (no copies, isinstance-safe) and emits a
  `DeprecationWarning`. CLI entry points are now `cymatix`/`cymatix-server`/
  `cymatix-launcher`/`cymatix-status`/`cymatix-vault`, with the old
  `helix*` names kept as console-script aliases. Env vars: `CYMATIX_*` is
  canonical and mirrored to `HELIX_*` unless an explicit `HELIX_*` value
  is already set (old deployments untouched). Config: `cymatix.toml` is
  canonical, `helix.toml` still loads as a fallback. MCP server identifies
  as `cymatix` (was `helix`) — a client-visible tool-namespace change.
  OTel `service.name` is now `cymatix-context`; metric names, logger
  names, and dashboard UIDs are deliberately unchanged. The wheel ships
  both packages. Knowledge-store format is unchanged — no re-ingest
  needed.

- **rename: Windows launcher surface follows the 0.8.0 cymatix rename.**
  `setup-helix.bat` → `setup-cymatix.bat`, `start-helix-tray.bat` →
  `start-cymatix-tray.bat`, `start-helix-mcpo.bat` →
  `start-cymatix-mcpo.bat`, `deploy/windows/setup-helix.ps1` →
  `setup-cymatix.ps1`; thin forwarders remain at the old .bat names for
  the deprecation window. The launchers now invoke
  `python -m cymatix_context.*` directly and set the canonical
  `CYMATIX_*` env vars (adopting any `HELIX_*` already present in the
  shell, so old-prefix deployments are untouched) — this makes the
  SETUP.md claims about which vars the tray bat sets actually true.
  Setup now creates/refreshes `Cymatix` shortcuts and retires stale
  repo-local `Helix` ones; the NSSM service recipe
  (`deploy/windows/README.md`) moves to `cymatix-launcher` /
  `CymatixLauncher`. Also adds the `*.local.bat` gitignore rule the
  launcher comments always promised.

- **rename: CLI `--help` surfaces follow the invoked alias.** Subcommand
  parsers hardcoded `prog="helix <sub>"`, so `cymatix query --help` printed
  `usage: helix query`. All nine subcommands, `cymatix-status`,
  `cymatix-vault`, and `cymatix-launcher` now derive prog from argv[0]
  (`cli.dispatcher.invoked_prog`) — the `cymatix` entry points brand
  themselves cymatix while the deprecated `helix*` aliases keep showing
  the name they were invoked as. Help descriptions drop the helix brand.

- **fix(sharding): mirror the WS2 symbol-graph + delete surface on
  `ShardedGenomeAdapter`.** `KnowledgeStore` grew `store_symbol_defs` /
  `resolve_symbol` / `_sweep_symbol_orphans` (WS2) and `delete_gene`
  without adapter counterparts, so any of those calls on a sharded store
  (`HELIX_USE_SHARDS=1`) raised `AttributeError` — caught by
  `test_adapter_covers_full_knowledgestore_surface`, which CI's
  file-scoped test selection never ran. `resolve_symbol` fans out across
  shards (soft-fail, deduped) like `term_doc_frequencies`; the three
  writes are V1 read-only-adapter no-ops, with `delete_gene` returning
  False so admin callers don't believe a hard-delete happened.

- **config: shipped `cymatix.toml` genome default back to
  `genomes/main/genome.db`.** The 2026-07-13 dogfooding commit pointed the
  shipped default at the dogfood genome (and broke
  `test_shipped_toml_matches_code_defaults`). Per-machine genome choice
  lives in the launcher's durable last-used selection
  (`~/.helix/launcher/selected_genome.json`, #286), which wins over the
  toml on every startup after the first — fresh installs get `main`,
  existing setups keep whatever they last selected.

- **fix(bench): `sweep_splade_scale_curve.py` on-arm never ran query-side
  SPLADE (#204).** The harness constructed `Genome(path=...,
  dense_embedding_enabled=False)` without threading `splade_enabled`, and
  the `KnowledgeStore` constructor default is `False` (config default is
  `True` — the #256-family layer-default disagreement biting a bench
  script). Tier 3.5 is gated on `self._splade_enabled`, so every
  "SPLADE-on vs off" delta this script ever reported — including the
  2026-07-11 overnight P9 smoke — was an A/A comparison, not a SPLADE
  ablation. The on-arm now constructs with `splade_enabled=True` and each
  arm's metrics embed a firing receipt (`splade_fire`: encode /
  query_splade call counts + hit totals) so artifacts self-certify that
  the tier engaged. Also adds `--query-shape raw|extracted` (`extracted` =
  stage-1 `extract_query_signals`, the serving shape — serving SPLADE
  encodes the extracted keyword bag, not the raw question; `raw` absolute
  levels are not serving-representative), plus
  `benchmarks/build_striptwins.py` (copy bed → `DROP TABLE splade_terms` →
  `VACUUM`, ~1 min/scale point) and the curated paraphrase gold-query
  fixture `benchmarks/_splade_curve_queries.json` (+ provenance sidecar).
  Curve results: `docs/research/2026-07-13-splade-scale-curve.md`.

- **fix(bench): seed sharded-fixture `harmonic_links` so cross-shard co-activation is reachable (#223).**
  `scripts/build_fixture_matrix.py --mode sharded` shipped every fixture
  with zero `harmonic_links` rows — `seed_edges()` existed but was
  test-only, leaving `ShardRouter._expand_cross_shard_coactivation` (#120)
  and the `coact_reserved_slots`/`coact_link_boost` knobs (#270)
  permanently unreachable in any sharded receipt. Now seeds intra-shard
  edges per shard (`seed_edges`) and cross-shard edges once across all
  shards (new `seed_cross_shard_edges`, bucketed by shared domain/entity
  token, bidirectional writes), both `ON CONFLICT DO NOTHING` and
  default-on (`HELIX_BFM_SEED_EDGES=0` to disable). `scripts/reseed_sharded_fixture.py`
  backfills an already-built fixture without a re-ingest. Re-run of the
  #223 A/B on a seeded medium fixture confirms `_apply_coact_reserve`
  now fires (previously unreachable); see issue #223 for the receipt.

- **fix(retrieval): harmonize `fusion_mode` layer defaults (#256).**
  `KnowledgeStore`/`Genome` direct construction now defaults
  `fusion_mode="rrf"`, matching `RetrievalConfig` (the #247 default flip).
  Config-built servers are unaffected (they always passed the config value
  explicitly); directly-constructed stores — including most test fixtures
  and micro-bench scripts — previously ran the legacy additive accumulator
  silently. Pass `fusion_mode="additive"` explicitly for legacy physics
  (scheduled for removal in v(N+2)). `test_layer_defaults_agree` now guards
  the two layers' equality permanently.

## 0.7.2b1 — 2026-07-06 (beta)

Efficiency + bench-validity wave. Beta cut for cross-host bench validation
(Max's box runs the trimmed local ladder; Joe's DGX Spark runs the heavy gemma4
rungs). Merge order: this bump lands **after** #241 (merged), #242, #243.

- **feat: fp32-BLOB SEMA embeddings + lean MCP surface (#241).** `genes.embedding`
  (20-d ΣĒMA) now packs as a little-endian fp32 BLOB in the same TEXT-affinity
  column — ~5x on that column, **no schema migration**, dual-decode keeps legacy
  JSON rows readable. MCP defaults to a lean 5-tool surface
  (`helix_context`/`_packet`/`_ingest`/`_health`/`_sessions_list`); set
  `HELIX_MCP_FULL=1` for the full 24 (~4–5K schema tokens/turn saved).
- **feat(server): `HELIX_DISABLE_LEARN` read-only serving (#243).** Skips Stage-6
  persist so answering a query never mutates the store — read-only / ephemeral /
  eval serving. Keystone fix for bench self-contamination (echo genes).
- **feat(bench): SIKE Run-1 validity + per-rung checkpoint/resume/pause (#243).**
  Decontaminated beds, probe A3 fixes, `SIKE_OLLAMA_MODELS` / `SIKE_SKIP_CLAUDE`
  ladder split, `sike_ctl.ps1` control, and a runner that checkpoints after every
  rung (a stop loses at most the in-flight rung).
- **fix(config): repair mojibake `helix.toml` (#242).** A UTF-8 BOM + double-encoded
  comments made the TOML parser fall back to code defaults; repaired to clean
  UTF-8 (config values unchanged).
- **docs:** efficiency/cost-reduction design memo (binary storage · algorithm-vs-model
  · MCP token cost), honesty baseline (encoder defaults are ON), root-tidy.

## 0.7.1 — 2026-06-09

- **fix(launcher): usable stdio under pythonw (#199).** The 0.7.0
  headless tray (`start-helix-tray.bat` via `pythonw`) failed to bind
  the dashboard port: detached pythonw has `sys.stdout`/`sys.stderr` =
  None and uvicorn's default loggers write to them, killing the
  launcher's server thread before bind (tray icon alive, dashboard
  dead, no traceback). `_ensure_streams()` now routes the missing
  streams into `--log-file` (or `os.devnull`) right after arg parsing.
  Caught during a live v0.7.0 install.

## 0.7.0 — 2026-06-09

Dashboard + UX release: the tray-hosted web dashboard graduates from a
read-only status page to the primary control surface, and the launcher
grows a dev/configuration dual-port mode. Driven by a live full-stack QA
session (clean-worktree boot, OTel routing verified end-to-end into
Prometheus / Tempo / Loki / Grafana).

- **feat(launcher): dev-mode dual main+bench ports (#197).**
  `[server] bench_enabled / bench_port (11439) / bench_genome_path` — the
  launcher supervises a second helix pinned to the bench genome via a
  per-instance env overlay, with its own state file, log, web controls
  (`/api/control/bench/start|stop`) and a dashboard card. Primary chat
  stays on the main genome/port; a subagent's bench-harness targets the
  bench port. Default OFF (`--bench` / `HELIX_BENCH_ENABLED=1`
  override) — final deployments get exactly one server. Verified live:
  ingest against :11439 lands only in the bench store.

- **feat(launcher): startup + first-boot UX (#197).** Alive-but-not-
  ready now renders a loading spinner ("encoders warming up") instead of
  "stopped"; when no genome exists at the active path the launcher skips
  autostart and the dashboard pops a select-or-create database dialog
  that dismisses itself once helix is up. `start-helix-tray.bat` goes
  headless via `pythonw` + new `--log-file` flag (python /B fallback).

- **feat(launcher): dashboard wiring sweep (#195).** Genome management
  from the web UI (`GET /api/genomes`, `POST /api/genome/select|create`,
  Select buttons + Create-and-switch form — heavy work on worker
  threads, 202 responses); Observability panel with per-service status
  dots + direct links to all six Grafana dashboards and Prometheus;
  `/api/state` carries the same snapshots for automation.

- **fix(store): fresh-checkout boot crash (#195).** `KnowledgeStore`
  mkdirs the genome's parent chain — a clean clone/worktree previously
  died with `sqlite3.OperationalError: unable to open database file`
  behind a silent 90s supervisor timeout.

- **feat(obs): grafana sidecar hardening (#195).** Pinned to 127.0.0.1
  with anonymous Admin for the local single-operator surface (GF_*
  overrides win) — telemetry links land on dashboards, not a login
  wall. Prometheus pinned to 127.0.0.1:9090; update/analytics
  phone-home disabled. The duplicate `helix-overview` dashboard uid is
  now a real **Helix — Overview** entry point.

## 0.6.5 — 2026-06-09

Eleven PRs landed same-day on top of 0.6.4 — the open-PR backlog merge
train (#182–#190), the full-suite QA de-flake that validated it, and the
two storage fixes that came out of the #165 fingerprint-index audit.

- **perf(storage): path_key_index Option-B compaction (#165, #193).** The
  fingerprint-routing index was 34.1% of the v2 Onyx corpus; probes
  showed the live Tier-0 lookup never uses `idx_pki_lookup` (covering
  PK scan), 38% of rows sit in pairs above `PKI_NOISE_CUTOFF` (hard-
  skipped by the scorer — zero score contribution), and the rowid
  table + 3-col-PK autoindex stored every row twice. New DBs now create
  `path_key_index` WITHOUT ROWID and skip the dead index; existing DBs
  convert via `storage.indexes.compact_path_key_index` /
  `POST /admin/compact-pki` (transactional, dry-run supported; follow
  with `/admin/vacuum`). ~21% corpus reduction on the audit fixture,
  score-invariant (40/40-query ablation). Tier-0 scoring constants
  (`PKI_BASE/FLOOR/NOISE_CUTOFF`) moved to `storage.indexes` as the
  canonical home so the scorer and compactor cannot drift.

- **fix(sharding): MAX_PATH overflow guard (#192).** The mirrored
  corpus-shard layout could exceed Windows' 260-char MAX_PATH when deep
  source roots mirror under deep `genomes_root`s. `corpus_shard_db` now
  caps at `HELIX_SHARD_PATH_MAX` (default 240) and falls back to a
  deterministic `_overflow/<label>-<sha1[:10]>.genome.db`; resume/salvage
  and routing unaffected.

- **fix(tests): full-suite de-flake (#191).** Two suite-hangers fixed:
  the sharded-parity test no longer loads SPLADE+BGE-M3 in parent + both
  spawn workers (three CUDA contexts = the #176 WDDM livelock on <=12 GB
  rigs) — lean-ingest env kill-switches `HELIX_BFM_SPLADE` /
  `HELIX_BFM_DENSE_BACKFILL` force the lean path; the metrics atomicity
  test no longer does 200K locked disk persists. Four
  `test_observability_docs` contracts re-pinned to the README-v3 layout;
  WSL-relay `bash.EXE` probed before use (skip when non-functional).

- **feat(ingest): size-aware SPLADE auto-toggle (#164, #189).**
  `splade_auto_enable_below_genes` / `splade_auto_disable_above_genes`
  knobs in `[ingestion]` (default 0 = off, byte-identical) +
  `benchmarks/sweep_splade_scale_curve.py` scaffold.

- **feat(bench): dense_additive_weight sweep harness (#138, #188).**
  `benchmarks/sweep_dense_additive_weight.py` across {0.0–6.0} with
  `gold_evicted_vs_baseline`; w=0.0 pinned as a true dense-off floor.
  Default stays 4.0 pending EnterpriseRAG-class data.

- **feat(bench): auto-subshard large source roots (#147, #186).**
  `_decompose_oversized_root` splits any single-root shard above
  ~5 GB / 100K files along top-level subdirs; silent-fail logging
  guards; `enterprise_rag_500k` profile.

- **fix(packet): preserve source-type prefix in `<GENE src=...>`
  (#146, #185).** Path shortener now anchors on the last `sources/`
  segment so `confluence/...`, `gmail/...` prefixes survive verbatim.

- **fix(bench): BenchServer import-source identity guard (#153, #184).**
  Spawn pins cwd + PYTHONPATH to the repo root, logs the resolved
  `helix_context` path at RUN START, and probes the fixture schema
  before swap — wrong-worktree mismatches fail in milliseconds, not as
  `retr=err` x 50.

- **feat(bench): file-level resume + SIGINT pause-then-resume
  (#150/#151, #183).** Partial shards resume at the file boundary
  (`_filter_to_unseen`); Ctrl+C finishes the in-flight batch, writes a
  `.paused-at-*` checkpoint, exits cleanly; `--rebuild` restores
  nuke-and-start-fresh.

- **feat(hardware): GB10 / Grace+Blackwell launch-blocking shim
  (#190).** Opt-in `HELIX_CUDA_LAUNCH_BLOCKING=1` forces synchronous
  CUDA launches before any torch import to dodge the sm_121
  async-dispatch livelock; byte-identical embeddings, default-off.
  Plus `docs/hardware/grace-blackwell.md`. (Contributed by @addiplus.)

- **docs(operations): dense ingest VRAM tuning matrix (#178, #182).**
  `docs/operations/DENSE_VRAM.md` — the <=12 GB / 16–24 GB / >=48 GB
  runbook with the confirmed failure modes and env-knob reference.

## 0.6.4 — 2026-06-09

Three landed PRs since v0.6.2. v0.6.3 remains the frozen Onyx
external-validation snapshot (tag-only, not on PyPI); 0.6.4 is the
public sibling that pulls forward the master-bound subset.

- **perf(dense): bound CUDA VRAM during batch ingest via periodic
  `empty_cache` (#177).** `BGEM3Codec.encode_batch` now releases
  torch's caching allocator every `HELIX_DENSE_VRAM_RELEASE_EVERY`
  batches (default 256, set `0` to disable). Holds dense ingest at
  ~6 GB plateau on a 12 GB 3080 Ti — previously climbed to 11.7 GB
  and spilled to shared-mem (the slow path that looked like a hang).
  Vectors are byte-identical (`empty_cache` only frees unused
  blocks). CUDA-only; CPU path untouched. Pairs well with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for
  fragmentation. Closes #176.

- **fix(config): wire `semantic_dense_additive_weight` +
  `semantic_broaden_routing` through retrieval (#180).** Two
  `RetrievalConfig` fields plus the consumer code that reads them
  — env-gated by `HELIX_SEMANTIC_ARM=1` AND `query_type ==
  "semantic"`, default-off so the stock path is byte-identical to
  v0.6.2. Lets downstreams running the v0.6.3 fixed-pipeline TOML
  compare arm-on vs arm-off on a master-derived build without the
  loader silently dropping their config keys. `query_type` is read
  from POST body on `/context` and `/fingerprint`. Backports JUST
  the semantic-arm hunks from the v0.6.3 chain — `_dense_w` swap in
  `knowledge_store`, LIKE-gate bypass in `shard_router.route()`,
  `query_type` thread through `build_context` /
  `build_context_async` / `_retrieve`. Explicitly NOT included:
  question-conditioned dense, fp16 matrix, RAM-cap PRAGMAs,
  IN-clause batching.

- **feat(launcher): Manage Database tray submenu + dashboard
  switchboard + pipeline viewer (#179).** Three connected launcher
  features:
  - System-tray "Manage Database ▸" submenu discovers genome `.db`
    files under `genomes/**` / `benchmarks/` / repo root, shows the
    active marker, per-genome sub-submenu with folder breakdown
    sampled from `source_id`, and a one-click "Use this database
    (restart helix)" action that sets `HELIX_GENOME_PATH` and
    restarts the supervised helix on a background thread so the
    pystray pump stays responsive. Win32 `MessageBoxW` confirmation
    on Windows (avoids the tkinter-on-pystray-thread deadlock);
    tkinter fallback elsewhere.
  - Dashboard **Switchboard** panel surfaces 11 operationally-
    interesting retrieval/budget/classifier/ribosome knobs from
    `load_config()` so an operator can read the live pipeline shape
    without `cat helix.toml`.
  - Dashboard **Pipeline viewer** (dev-mode toggle, default on)
    renders the last ~20 `build_context()` calls with per-stage
    timings. Backed by a `contextvars`-scoped per-request id and a
    bounded `_pipeline_events` deque in `context_manager`
    (`HELIX_PIPELINE_RING=0` disables) + new
    `GET /debug/pipeline/recent`.
  - New `GET /admin/genome` reports the `.db` the running helix
    actually opened; the dashboard cross-checks it against the
    on-disk registry so drift between "selected" and "running" is
    visible.

## 0.6.2 — 2026-05-30

Make the v0.6.1 SQLite memory posture **host-aware** instead of unconditionally
conservative. v0.6.1 hard-coded `mmap_size=0` + 2/4 MB page caches on every host
as a 100-shard fan-out commit guard — but the BGE-M3 model singleton was the
actual 120 GB → 7 GB fix, so that I/O posture only over-throttled RAM-rich hosts
(reading 46 GB of shards through a 4 MB cache with mmap off). This scales the
per-shard `mmap_size` / `cache_size` to the host.

- **perf(memory): RAM-aware SQLite budget, `HELIX_MEM_PROFILE` (default `auto`).**
  `hardware.sqlite_memory_budget(n_shards)` derives a per-connection
  `mmap_size` / `cache_size` from *available* RAM: `budget = (available − 25%
  reserve) / n_shards`, split into file-backed mmap (≤ 2 GiB/shard) + a bounded
  2–64 MB page cache. Because the budget is a fraction of free RAM, it can never
  claim more than exists, and it self-throttles when shard count is high or RAM
  is scarce (the 105-shard / 48 GB stress case falls back toward mmap-off). The
  plan is resolved once in `ShardRouter` from the registered shard count and
  threaded into each shard's `KnowledgeStore` + `main.db`; standalone stores
  resolve a single-DB budget.
  - `HELIX_MEM_PROFILE`: `auto` (default) · `aggressive` (15% reserve, 4 GiB cap)
    · `conservative` (**byte-identical to v0.6.1** — the escape hatch) · `<N>gb`
    (pin the total SQLite budget, host-independent — useful where psutil
    over-reports inside a constrained container).
  - Hard overrides: `HELIX_SQLITE_MMAP_SIZE` (bytes) and `HELIX_SQLITE_CACHE_SIZE`
    (raw pragma value) win over the profile.
  - PRD: `docs/prds/2026-05-30-dynamic-ram-scaling.md`. Unit-tested (budget
    contract + plan-application through Genome / ShardRouter / main.db); the
    end-to-end 100-shard perf delta is validated separately on the bench fixture.

## 0.6.1 — 2026-05-30

Performance release: concurrent shard fan-out + a daemon RAM collapse at
100-shard scale. Both land off the EnterpriseRAG-Bench v2 work (850K genes /
100 shards) and were verified end-to-end on that fixture (resident RAM
~120 GB → 7 GB; per-query 125s → 57.6s median at `HELIX_SHARD_WORKERS=8`;
0 daemon deaths; ranked output byte-identical to the serial path).

- **perf(memory): share ONE BGE-M3 model process-wide (PR #173).**
  `KnowledgeStore._get_dense_codec()` built its own ~2 GB `BGEM3Codec` per
  instance, so a query touching ~100 shards loaded up to ~100 copies of the
  model — the dominant driver of the daemon's 47 GB-on-disk → ~120 GB-resident
  ramp (legitimate heap is only ~6 GB). `get_shared_codec()` returns one
  instance per `(model_name, dim, device)`; inference is stateless so sharing
  across concurrent fan-out workers is safe (double-checked load lock). Default
  on; `HELIX_SHARE_DENSE_CODEC=0` reverts to per-instance for an A/B.

- **perf(retrieval): concurrent shard fan-out (PR #172).** The 100-shard
  fan-out in `ShardRouter.query_genes` ran as a serial loop. It now parallelizes
  the per-shard fetch (open + `query_docs` + IDF probe) via a
  `ThreadPoolExecutor` — the dense matmul releases the GIL through BLAS, so
  threads genuinely parallelize, with BLAS pinned to 1 thread (`threadpoolctl`)
  to avoid oversubscription. Accumulation/merge/sort stays sequential in
  original shard order, so ranked output is byte-identical to serial. Gated by
  `HELIX_SHARD_WORKERS` (default 1 = serial). Pair with the BGE-M3 singleton —
  without it the per-shard model duplication thrashes the pagefile and caps the
  speedup at ~1.5x.

- **perf(memory): optional fp16 dense matrix (PR #173).**
  `HELIX_DENSE_MATRIX_DTYPE=float16` halves the resident per-shard dense matrix
  (~3.3 GB → ~1.65 GB); numpy promotes to fp32 inside the matmul so cosine
  precision is unchanged. Default `float32` = byte-identical to 0.6.0.

- **perf(memory): bound SQLite memory + guard mmap (PR #173).** Explicit
  `cache_size` caps on both per-shard connections (−2 MB writer / −4 MB reader;
  previously the unbounded 2 MB/conn default × 200 connections) and explicit
  `PRAGMA mmap_size=0` on every connection (writer/reader/main.db) as a process-
  commit guard for concurrent shard opens under fan-out.

- **fix(retrieval): WAL checkpoint on shard close (PR #172).**
  `ShardRouter.close()` now calls `genome.close()` (which runs
  `checkpoint(TRUNCATE)`) instead of closing the connections directly, which
  skipped the checkpoint and left up to 64 MB of un-truncated WAL per shard.

## 0.6.0 — 2026-05-28

Substantial release covering corpus-scale retrieval, bench rebuilding,
storage audits, and a stack of stability + portability fixes. Headline
work: EnterpriseRAG-Bench (Layer 3) shipped with 100q variant-A
results (recall@10 = 28% on the 850K-gene v2 fixture); two scaling-wall
bugs (regex ReDoS at ingest, SQL-variable cap at retrieval) fixed and
cross-validated on x86 + ARM64 hardware; ~400 lines of new bench docs.

- **fix(tagger): eliminate catastrophic backtracking in `_KV_PAIR_PATTERN`
  (PR #155 / PR #162).** Pre-fix, `(\w+(?:_\w+)*)` had redundant
  nested-quantifier ambiguity that triggered catastrophic backtracking on
  underscore-heavy content. A single worker spinning on
  `tagger.py:439`'s `_KV_PAIR_PATTERN.finditer(content[:5000])` hung the
  EnterpriseRAG-Bench-Onyx-full corpus build for 60+ minutes on a single
  google-drive shared-drives file (underscore-rich JSON keys like
  `expected_doc_ids`, `data_source_id`). The fix `(\w+)` is functionally
  identical (same match set, since `\w` includes `_`) but has no nested
  quantifier. Verified on the 3 worst-offender files from the bench corpus:
  0.40-0.52 ms each, down from >60 min hung. 200-underscore stress test:
  0.02 ms. **Cross-validated** under two independent py-spy investigations
  on different hardware classes (x86 Ryzen + RTX 3080 Ti on 2026-05-19,
  ARM64 Grace + GB10 on 2026-05-27) — same line, same root cause, same fix.

- **perf(ddl): skip FTS5 orphan cleanup when delta is < 5% of gene count
  (PR #156 / PR #162).** The previous cleanup ran
  `DELETE FROM genes_fts WHERE gene_id NOT IN (SELECT gene_id FROM genes)`
  — an O(N·M) correlated subquery that hung the daemon's first-query
  response for hours on the 850K-gene / 105-shard EnterpriseRAG-Bench
  fixture. On a single 18K-gene shard with ~40 orphans (0.2% noise) the
  `NOT IN` scan pegged a core for 5-10 minutes against a cold OS-cache
  page set. Orphan FTS5 entries are harmless at query time (downstream
  `gene_id` joins return NULL and filter out before delivery), so
  skipping cleanup for trivial deltas costs nothing in retrieval quality
  and unblocks first-query latency entirely. For the rare significant-drift
  case (delta ≥ 5%), the rewritten query uses indexed `NOT EXISTS` instead
  of `NOT IN`, turning O(N²) into O(N log N). Daemon `/health` response
  on the 850K-gene fixture went from "hangs forever" to milliseconds.

- **feat(build): salvage already-complete shards on rebuild (PR #157 /
  PR #162).** Adds `_try_salvage_complete_shard()` which opens an existing
  shard `.db` read-only, verifies the `genes` table has 100% dense
  backfill coverage and no live WAL sidecar, and returns the same
  result-dict shape that `_build_one_shard` would normally produce —
  letting the parent's `_commit_shard_result` re-register the shard via
  `INSERT OR REPLACE`. Designed for the kill+restart cycle: if
  `build_fixture_matrix.py` is interrupted (Ctrl+C, OOM, planned restart),
  fully-complete shards on disk are re-registered in seconds instead of
  rebuilt from scratch. Verified at scale: 21 of 22 already-complete
  shards re-registered into a fresh `main.genome.db` in 2 min 19 sec
  (vs ~13 hours to rebuild from scratch) during a mid-build restart of
  the EnterpriseRAG-Bench-Onyx-full 850K-gene build.

- **fix(knowledge_store): batch IN-clause queries to stay under SQLite cap
  (PR #163).** SQLite caps `WHERE col IN (?, ?, ...)` placeholders at
  `SQLITE_LIMIT_VARIABLE_NUMBER` (999 legacy, 2000 on the Python 3.12 /
  SQLite 3.50 builds we ship to, 32766 on newer compile defaults). Four
  call sites on the `gene_scores` fan-out path in `query_docs`
  (`_apply_authority_boosts`, sema-boost embedding lookup, party-attribution
  lookup, access-rate epigenetics lookup) build the IN clause from a
  caller-determined candidate set that can exceed the cap in production.
  Observed in the 2026-05-28 v2-fixture 100q bench: 3 of 29 queries had a
  per-shard query raise `OperationalError: too many SQL variables`, which
  the daemon's per-shard try/except swallowed as "shard X query failed;
  skipping" — biasing recall@K by silently dropping shards. Variants where
  SPLADE or the prefilter narrows `gene_scores` don't hit this; only the
  no-filter SPLADE-off path produces sets large enough to blow up. Adds
  `_iter_in_batches(items, batch_size=500)` helper and refactors the four
  hot sites. Includes TDD'd regression test at
  `tests/test_knowledge_store_batched_in.py` that probes the runtime cap
  via `conn.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER)` and exercises at
  `cap`, `cap + 1`, and `4*cap + 7` boundaries.

- **fix(mcp): registry handshake is best-effort, don't kill subprocess on
  failure (PR #169).** On Windows, `claude -p` MCP attempts were failing
  with "Connection closed" after ~2 s even when helix was alive on
  `http://127.0.0.1:11437`. Root cause: `_register_with_registry()` was
  called synchronously before `mcp.run()` entered the stdio handshake; an
  exception from `register_participant()` (auto-heartbeat thread init,
  etc.) propagated out of `main()` and killed the MCP subprocess before
  the host could complete its handshake. The registry is not load-bearing
  for tool calls — tool calls proxy directly to the helix HTTP API.
  Registry is only used by `helix_announce` + dashboards. This patch
  wraps `_register_with_registry()` in a try/except inside `main()`:
  happy path unchanged, failure path logs the exception and continues
  to `mcp.run()` rather than exiting. Closes #167.

- **feat(bench): add `--isolated` flag to `bench_claude_matrix` for
  leak-free measurement (PR #170).** When set, the `claude -p` sub-agent
  is launched with `--tools ""` (all built-in tools disabled),
  `--strict-mcp-config`, and `--mcp-config '{"mcpServers":{}}'` (no MCP
  servers). Pair with a sterile `--cwd` (e.g. `F:/tmp/bench_sandbox`) to
  also block CLAUDE.md auto-discovery. Isolates retrieval-driven answer
  quality from filesystem-tool access. Records `isolated` + `claude_cwd`
  in the per-run JSON so post-hoc analysis can distinguish leak-free runs
  from contaminated runs. Brings shipped code into agreement with
  shipped docs (`docs/benchmarks/BENCHMARKS.md` §"Layer 3 —
  EnterpriseRAG-Bench" and `BENCHMARK_RATIONALE.md` addendum already
  described this isolation mode). Closes #168.

- **docs(benchmarks): add Layer 3 (EnterpriseRAG-Bench) + EnterpriseRAG
  fixtures (PR #166).** ~400 lines across four files. `BENCHMARKS.md`
  gets a new "Layer 3 — EnterpriseRAG-Bench" section covering the
  2026-05-20→21 bench investigation rebuild (`isolated=True` mode,
  +32.4 pp helix lift, 65% hallucination reduction), cross-corpus results
  (60% recall@10 @ 10K → 71% @ 50K → 28% @ 850K), the expression-budget
  clamp fix (4%→43% correctness), Wall-1 / Wall-2 scaling-wall framing,
  the v2 100q variant-A result table, and cross-host validation of the
  tagger fix. `GENOME_FIXTURE_MATRIX.md` gets a new EnterpriseRAG-Bench
  fixtures section (5-row fixture table, shared 9-source-root scope,
  excluded-from-ingest list, auto-subsharding behavior, path-portability
  gotcha, branch/PR routing). `BENCHMARK_RATIONALE.md` gets an addendum
  on how Layer 3 answered the rationale's NIAH-doesn't-fit problems.
  `MULTI_VALID_GOLD.md` gets an EnterpriseRAG-Bench gold-path matching
  section (schema diff, `_rel_after_sources` normalization, prefix-tolerant
  match fix).

### Prior work consolidated into this release

The following entries were already in `## Unreleased` at the start of
this release cycle and ship together as part of 0.6.0:

- **fix(launcher): `[headroom] route_upstream` is now a separate, default-off
  config knob; routing no longer happens implicitly from "upstream is remote".**
  Pre-fix, `_should_route_helix_upstream_via_headroom` returned True for any
  remote (non-loopback) upstream as long as `HELIX_HEADROOM_ROUTE_UPSTREAM_AUTO`
  wasn't explicitly set falsy. So an operator with `cfg.server.upstream =
  "https://api.openai.com/v1"` and `cfg.headroom.enabled = false` (defaults!)
  would have the launcher rewrite `HELIX_SERVER_UPSTREAM` to
  `http://127.0.0.1:8787` and start helix pointing at a Headroom proxy that
  was never started — every chat call then failed with ECONNREFUSED, with
  no clear diagnostic. `route_upstream` is now an explicit `[headroom]` bool
  (default `false`) gating the rewrite. `HELIX_HEADROOM_ROUTE_UPSTREAM_AUTO`
  remains as a per-launch override (truthy → on, falsy → off, unset →
  defer to config). The existing test
  `test_remote_upstream_routes_helix_via_headroom` (which pinned the buggy
  behavior) is replaced with four tests covering the new precedence rules.

- **fix(launcher): `POST /api/control/start` no longer reports success on a
  hung backend; returns `202 Accepted` with `started_pending=true`.** PR #68
  made `supervisor.start()` non-fatal on `/stats` timeout (proc left
  running so the tray's next poll picks it up). The REST handler still
  treated this as success and returned `{ok: true, pid}`, so external
  automation hitting `/api/control/start` directly couldn't distinguish
  ready from alive-but-not-ready. New `supervisor.last_start_pending`
  flag flips on the timeout path; REST surface returns 202 with a
  `started_pending: true` field and a hint to poll `/api/state` or
  `GET /stats`. Same treatment on `/api/control/restart`. Closes #72.
- **fix(hardware): summary `WARNING` line when explicit-device probe
  falls back to CPU.** The tray fires a balloon, but headless deployments
  (server, supervisor-managed, agents) miss that signal. `_detect()` now
  emits one `log.warning("Hardware fallback: requested=X active=cpu — ...")`
  alongside the per-candidate probe failures so operators tailing logs
  see the cause in line-of-sight. `auto`→cpu is unchanged (not noteworthy).
  Closes #65 SF2 — SF1, SF3, SF4 were already addressed on master
  (per-rewrite log.info, `cost_class` in `/health` + Prometheus info
  metric + startup WARN, and the WAL-bloat section in
  `docs/TROUBLESHOOTING.md` + `/admin/checkpoint` admin endpoint).

- **fix(api): `open_session()` now honors `HELIX_CONFIG` / `HELIX_GENOME_PATH`.**
  Pre-fix, every cold-start CLI subcommand (query, diag corpus, packet,
  gene, neighbors, refresh-targets) called `HelixConfig()` (defaults) and
  silently created/read `./genome.db` regardless of what the operator had
  configured — so `helix status` looked at the configured genome but
  `helix query` looked at an empty one. Now routes through `load_config()`
  the same way `helix status` does. Surfaced by AI-user testing on
  `93deaf2`.
- **fix(status): bump `/health` probe timeout default 1.5s → 10s, override
  via `HELIX_STATUS_TIMEOUT_S`.** Cold-start `/health` can take 5-10s
  under model warmup + manager init + WAL replay; the old 1.5s timeout
  silently reported a healthy-but-slow server as `unreachable` in
  `helix status --json`.
- **fix(mcp): unwrap the Continue list shape in `helix_context` /
  `helix_document_query` tools.** `POST /context` returns the
  Continue-IDE HTTP context-provider list (`[{name, description, content,
  ...}]`) so the FastAPI endpoint stays drop-in compatible with Continue.
  MCP hosts validate tool returns against the declared `Dict[str, Any]`
  schema and rejected the list. New `_unwrap_context_list` helper flattens
  the single-entry list, passes error envelopes through, and wraps
  unexpected shapes with a diagnostic note.
- **fix(config): auto-fallback `ingestion.backend` → `"cpu"` when
  `ribosome.enabled = false`.** The two settings contradict each other —
  ingest with the ribosome disabled raised
  `TranscriptionError: Pack failed: Ribosome is disabled` on the first
  chunk. `load_config()` now flips ingestion to the spaCy/heuristic
  CpuTagger path and logs a WARNING. Honors explicit `cpu` / `hybrid`
  settings without override.
- **feat(cli): `python -m helix_context.cli` works as a console-script
  fallback.** Adds `helix_context/cli/__main__.py` so an agent or
  operator with a broken pip-installed `helix.exe` (deleted editable
  source path, Scripts dir off PATH) always has a module-direct
  invocation. Documented in `docs/clients/cli.md`.

- **feat(cli): agent walk-aware surface — `packet` / `gene` / `neighbors` /
  `refresh-targets`.** Four new subcommands that complete the v1 CLI as a
  full agent surface — agents drive genome lookups via subprocess CLI
  calls instead of MCP-injected context. JSON shapes match the
  corresponding HTTP endpoints (`/context/packet`, `/genes/<id>`,
  `/debug/neighbors`, `/context/refresh-plan`) and MCP tools
  (`helix_context_packet`, `helix_gene_get`, `helix_neighbors`,
  `helix_refresh_targets`) so callers can swap surfaces without changing
  call logic. Read-only by default (no genome mutation from inspection).
- **feat(api): walk-aware methods on `HelixSession`.** `gene_get`,
  `packet`, `refresh_targets`, `neighbors` — previously deferred to v1.1
  per `helix_context/api.py:352-360`, now in v1 to back the CLI surface
  above. Pure in-process wrappers over `Genome.get_gene`,
  `build_context_packet`, and the existing SEMA codec; no HTTP server
  required.
- **docs: README — agent-CLI callout + benchmark sourcing.** New "Agent
  CLI surface (no server required)" section advertises the CLI as a
  first-class agent surface alongside MCP, with the full subcommand
  surface inline. The "28.7× headline / 5.4× median" claim now cites the
  reproducer at `benchmarks/bench_rag_vs_sike_tokens.py` and the
  methodology doc at `docs/benchmarks/BENCHMARKS.md`; the overnight
  result file referenced in the README-v2 spec is documented as internal.
- **docs: ROSETTA — response-types section, dead-entry fix, ChromatinState
  note.** Adds a "Response & routing types (STAYS — no biology twin)"
  section covering `ContextWindow`, `ContextPacket`, `KnowBlock`,
  `MissBlock`, `RefreshTarget`, `ContextHealth`, `ContextItem`,
  `QueryResult`, `IngestResult`, `StatsResult`. The `HGT →
  cross_store_import` row is annotated as a forward-pointer (no code under
  either name today). The `OPEN/EUCHROMATIN/HETEROCHROMATIN →
  OPEN/WARM/COLD` row notes the rename is deferred to R3 because
  `ChromatinState` in `schemas.py` still emits the bio names.

- **feat(cli): v1 cold-start CLI shipped.** `helix query`, `helix ingest`,
  `helix status`, `helix diag corpus`, `helix config show`. Mocked unit
  tests for every subcommand plus a live `@pytest.mark.live` integration
  test against an `:memory:` genome. The legacy FastAPI launcher is now
  reachable as `helix-server` (previous `helix` entry point). Daemon
  (`helix serve`) remains deferred per `docs/architecture/HELIX_DAEMON_DESIGN.md`
  — the subcommand prints a pointer and exits 4. See `docs/clients/cli.md`
  for the operator reference.

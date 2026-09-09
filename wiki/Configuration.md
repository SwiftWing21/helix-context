# Configuration

- Everything lives in one file: **`cymatix.toml`**, read once at process start
  by `cymatix_context.config.load_config()`. There is no second config source
  and no dotfile.
- Malformed TOML logs an error and uses defaults. Unknown keys and non-table
  known sections log warnings and are ignored. Invalid fields inside valid
  tables can still raise errors — for example, a nonnumeric timeout or
  `[abstain] mode = "per_classifier"` without an `[abstain.default]` block.
  Section-shape recovery in v0.9.2 preserves this field validation.
- **Defaults are the interesting part.** Almost every expensive feature in this
  project ships off, and every default-off flip is attached to a receipt. This
  page gives the tour and the flip dates; the exhaustive per-key reference is
  [`docs/config-reference.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/config-reference.md).
- **Canonical names are shown first.** `[compressor]` and `[knowledge_store]`
  are the software-lexicon spellings; `[ribosome]` and `[genome]` are the
  legacy spellings the file on disk still uses, and both resolve. See
  [Lexicon](Lexicon). The aliases come from the Tier-2 lexicon pass
  ([#419](https://github.com/mbachaud/Cymatix-Context/pull/419)), which follows the 0.9.1 tag — on a 0.9.1 wheel only the legacy
  spellings resolve.
- This page documents **v0.9.2**. The wave-1 ranking
  defaults ([#407](https://github.com/mbachaud/Cymatix-Context/pull/407)),
  the delivered-seat floor and its graduation to 12
  ([#409](https://github.com/mbachaud/Cymatix-Context/pull/409)), the entity
  auto-link hub cutoff knob
  ([#412](https://github.com/mbachaud/Cymatix-Context/pull/412)), and the
  tagger v2 behavior change
  ([#413](https://github.com/mbachaud/Cymatix-Context/pull/413)) all shipped
  in v0.9.1 (released 2026-08-30) and are cited where they appear.

## Where a value comes from

Highest priority wins:

1. **Per-call request fields** — `caller_model_class`, `include_cold`,
   `decoder_mode`, `ignore_delivered` and friends on the `/context` body.
   These never mutate the loaded config.
2. **Environment variables** — the `CYMATIX_*` set below.
3. **`cymatix.toml`** — the path in `CYMATIX_CONFIG`, else `cymatix.toml` in
   the current working directory.
4. **Dataclass defaults** in `cymatix_context/config.py`, returned as-is when
   no file is found.

Two typo guards run at load, both warning-only:

- An unrecognized **key** inside a known section logs
  `Unknown-key check` output naming the section and key.
- An unrecognized **top-level section** logs
  `Unknown top-level section [<name>] in cymatix.toml`. Added by the Tier-2
  lexicon pass ([#419](https://github.com/mbachaud/Cymatix-Context/pull/419), after the 0.9.1 tag) — it
  catches a mistyped `[retreival]` that would previously have been silently
  ignored while you wondered why the flip did nothing.

## Canonical and legacy names

The Tier-2 lexicon pass ([#419](https://github.com/mbachaud/Cymatix-Context/pull/419), merged after the 0.9.1 tag — not in the
0.9.1 wheel) adds canonical software-term aliases across the config surface.
They are **additive**: every legacy spelling keeps working, unchanged.

| Canonical | Legacy | Surface |
|---|---|---|
| `[compressor]` | `[ribosome]` | TOML section |
| `[knowledge_store]` | `[genome]` | TOML section |
| `[budget] retrieval_tokens` | `expression_tokens` | TOML key |
| `[budget] max_docs_per_turn` | `max_genes_per_turn` | TOML key |
| `CYMATIX_STORE_PATH` | `CYMATIX_GENOME_PATH` | environment variable |

**Collision rule: the legacy name wins, and a `WARNING` names both.** If a
config carries both `[compressor]` and `[ribosome]`, the `[ribosome]` block is
the one that binds and the alias is dropped with a log line. Same for the key
pairs and for the two env vars. The rule is deliberately conservative — an
operator who has both spellings in play gets the behavior their existing file
already had, not a silent change.

(The CLI is the one exception: `--max-docs` and `--max-genes` share an argparse
destination, so passing both is plain last-wins. See [CLI](CLI).)

## Every section

| Section | What it governs | Key settings (shipped defaults) |
|---|---|---|
| `[compressor]` *(legacy `[ribosome]`)* | The optional compression / enrichment layer. Off the retrieval hot path entirely. | `enabled = false`, `backend = "none"` (only `"litellm"` and `"deberta"` are honored when enabled; anything else, including `"claude"` and legacy `"ollama"`, dispatches the no-op DisabledBackend), `model = "gemma4:e2b"`, `timeout = 120`, `query_expansion_enabled = false` |
| `[hardware]` | Device picker and per-model batch-size policy. | `device = "auto"` (cuda → rocm → mps → cpu), `batch_sizes = "auto"`, `low_vram_threshold_gb = 4.0`, `lazy_encoders = true` |
| `[budget]` | Token caps, per-turn document caps, decoder mode, assembly-time safety. | `retrieval_tokens = 7000` *(legacy `expression_tokens`)*, `max_docs_per_turn = 12` *(legacy `max_genes_per_turn`)*, `decoder_mode = "condensed"`, `splice_aggressiveness = 0.3`, `legibility_enabled = true`, `session_delivery_enabled = true`, `abstain_enabled = true`, `neutralize_control_tags = true`, `min_delivered_docs = 12` *(0.9.1; graduated from 0)* |
| `[session]` | Session and party attribution fallbacks for the CWoLa label logger. | `default_party_id = "default"`, `synthetic_session_enabled = true`, `synthetic_session_window_s = 300` |
| `[knowledge_store]` *(legacy `[genome]`)* | Store path, compaction cadence, write durability. | `path = "genomes/main/genome.db"`, `compact_interval = 3600`, `cold_start_threshold = 10`, `replicas = []`, `synchronous = "NORMAL"` (WAL-safe; skips SQLite's per-commit fsync) |
| `[server]` | HTTP bind, chat upstream, and the two admin-surface hardening knobs. | `host = "127.0.0.1"`, `port = 11437`, `upstream = "http://localhost:11434"`, `upstream_timeout = 180`, `admin_token = ""`, `swap_db_roots = []`, `bench_enabled = false`, `bench_port = 11439` |
| `[telemetry]` | OpenTelemetry export defaults. | `enabled = false`, `endpoint = "localhost:4317"`, `insecure = true`, `sampler_ratio = 1.0`, `redact_query = true`, `logs_enabled = true`, `logs_level = "INFO"` |
| `[headroom]` | Optional Headroom compression-proxy lifecycle — launcher only. | `enabled = true`, `autostart = true`, `port = 8787`, `mode = "token"`, `route_upstream = false` |
| `[ingestion]` | What happens to content on the way in. | `backend = "cpu"` (spaCy + regex), `entity_graph = true`, `splade_enabled = false`, `sema_embed_on_ingest = false`, `dense_embed_on_ingest = false`, `symbol_graph = false`, `entity_autolink_hub_cutoff = 200` *(0.9.2; was 0)*, `locale_demotion_enabled = true` |
| `[context]` | Cold-tier fallthrough and the fingerprint profile. | `cold_tier_enabled = false`, `cold_tier_k = 3`, `cold_tier_min_cosine = 0.15`, `fingerprint_mode_profile = "balanced"` |
| `[cymatics]` | The 256-bin term-spectrum scorer. Blends as a bonus, never re-sorts. | `enabled = true`, `harmonic_links = true`, `distance_metric = "cosine"` (or `"w1"`), `peak_width` unset (derives from `[budget] splice_aggressiveness`) |
| `[classifier]` | The Stage-0 rule-based query classifier. | `enabled = true` — and that is the whole surface. Per-class caps and decoder hints are code constants pending [#205](https://github.com/mbachaud/Cymatix-Context/issues/205) |
| `[retrieval]` | The big one: every recall tier, the fuser, the rerank layer. | `fusion_mode = "rrf"`, `rrf_k = 20` *(0.9.1; was 60)*, `rerank_combinator_by_class` = all five classes → `eps_band` *(0.9.1; was `{multi_hop, default}`)*, `cover_walk_enabled = false` *(0.9.1, W2.1 — shipped inert, killed by its receipt, [#408](https://github.com/mbachaud/Cymatix-Context/pull/408))*, `dense_embedding_enabled = false`, `pki_enabled = false`, `rerank_enabled = false`, `filename_anchor_enabled = true`, `bm25_shortlist_enabled = true`, `sr_enabled = false`, `seeded_edges_enabled = false`, `blend_mode = "scale_relative"` |
| `[plr]` | The PLR query-confidence head that decorates `/context/packet`. | `enabled = true`, `model_path = "training/models/stacked_plr.joblib"`, `expected_sha256 = ""` (empty trusts the trainer's `.sha256` sidecar), `high_risk_threshold = 0.5` |
| `[know]` | The KnowBlock confidence logistic and its calibration provenance. | `emit_floor = 0.45`, `s_ref = 4.2503`, `g_ref = 0.4386`, `betas` (6 coefficients), `calibrated_at` / `calibrated_on_n` written by `scripts/calibrate_know_confidence.py`, `stale_after_days = 30` |
| `[mem_sync]` | Auto-sync of a watched notes directory into the store. Consumed by `scripts/run_mem_sync.py`, not by the server. | `enabled = false`, `watch_dirs = []`, `sync_interval_s = 60`, `agent_kind = "claude-code"` |
| `[synonyms]` | Query-keyword → tag expansion. Flat `key = [values]` map; read once at start. | Ships ~40 rows (`cache`, `auth`, `db`, `api`, …). **Add your own project vocabulary here** — see the note below |
| `[abstain]` | Confidence floors for the ABSTAIN tier. | `mode = "global"` (legacy hard-coded floors, byte-for-byte), `ratio_threshold = 1.8` (additive), `ratio_threshold_rrf_norm = 1.5` (RRF). `"per_classifier"` requires an `[abstain.default]` sub-table |
| `[vault]` | Obsidian-style read-only export plus diagnostic `/context` traces. Shipped commented out. | `enabled = false`, `path = "~/.cymatix/vault"`, `redact_body = false`, `[vault.traces] retention_hours = 48` |
| `[encoder_daemon]` | Route dense / SPLADE / SEMA encoding through one shared process instead of loading weights per seam. Absent from the shipped file. | `url = ""` (off — every seam encodes in-process, byte-identical). Set e.g. `"http://127.0.0.1:11440"`; `CYMATIX_ENCODER_URL` wins over the TOML value |

**The synonym map earns its own warning.** Retrieval matches query keywords
against the tags assigned at ingest time. When a query returns "no relevant
context", the most common cause by far is that your vocabulary and your
corpus's tags never meet. Add a row under `[synonyms]` and restart before
concluding that retrieval is broken.

## Default-off, and when it flipped

Every one of these was flipped by a measurement, not a preference. The default
retrieval path in 0.9.1 makes **zero model calls** as a result.

| Knob | Default | Since | Why |
|---|---|---|---|
| `[retrieval] dense_embedding_enabled` | `false` | 2026-08-15 | Four-scale isolation receipts (100k / 250k / 500k carves + the 829k blob, every pair replicated) measured dense **displacing** gold from the delivered top-k at every scale — final recall −0.20..−0.33 against the lexical floor — for at most +0.02 pool recall |
| `[ingestion] splade_enabled` | `false` | 2026-08-16 | At n=469 on the 829k blob, query-side SPLADE over the full 147M-row expansion index was null-to-negative against the lexical floor, and the index contributed nothing passively (floors byte-identical with and without it) |
| `[retrieval] pki_enabled` | `false` | **2026-08-17** ([#370](https://github.com/mbachaud/Cymatix-Context/issues/370)) | The exact-shipped-default confirm at full power (n=469, delivered basis) measured PKI on as **−1 delivered needle** — null-to-hair-negative. Flipping it back does not reclaim an existing `path_key_index` table |
| `[ingestion] sema_embed_on_ingest` | `false` | 2026-08-19 ([#371](https://github.com/mbachaud/Cymatix-Context/issues/371)) | Its only live retrieval consumer at shipped defaults is the Tier-4 `sema_boost` re-rank, and the deciding read-gate cell was a wash (+1 delivered needle, ranks byte-identical). Riders: ~9.7 s cold start removed, ingest wall 835.9 s → 761.5 s on the 1,181-file corpus |
| `[ingestion] dense_embed_on_ingest` | `false` | 2026-08-19 ([#371](https://github.com/mbachaud/Cymatix-Context/issues/371)) | A dead write at neural-free retrieval defaults — nothing read the vectors |
| `[retrieval] rerank_enabled` | `false` | always ([#341](https://github.com/mbachaud/Cymatix-Context/issues/341)) | The cross-encoder costs roughly 270–560 ms/query at 829k documents on a GPU encoder daemon; enable it only on populations with near-cutoff rerank headroom |
| `[compressor] enabled` | `false` | always | The design pillar. With it false the backend string is ignored entirely and `/context` never touches an LLM |

**Two consequences worth internalizing.** Turning a retrieval knob back on is
not enough on its own — dense recall additionally needs the vectors to exist,
so a store ingested with `dense_embed_on_ingest = false` returns nothing from
the dense tier until `scripts/backfill_bgem3_v2.py` has run (same story for
`scripts/backfill_sema.py`). And the dense-off default carries a disclosed p50
latency regression — dense's ANN gate was load-bearing as a candidate-list cap
— ×2.5–2.6 at 100k fragments, shrinking to ×1.09–1.38 at the 829k operating
point ([#374](https://github.com/mbachaud/Cymatix-Context/issues/374)); the
mitigation has not landed.

## New in 0.9.1

### `[budget] min_delivered_docs` — the delivered-seat floor

```toml
[budget]
min_delivered_docs = 12   # default since 2026-08-30 ([w24-floor-flip]); 0 = legacy eviction-only trim
```

- Guards **both** mechanisms that cost a document its seat. It lifts the
  classifier's per-rule assembly cap to at least N, and it makes the Stage-5
  budget trimmer shrink the largest parts (tail-truncated, header-preserving,
  marked `...[budget-trimmed]`) instead of evicting whole documents.
- The token budget still wins last: if every part is already at the truncation
  floor and the prompt still overflows, eviction resumes below the floor.
- Model-class caps (`small_moe = 4`) are a model-capability concern and stay
  untouched.
- **Measured at floor 12 on the 829k bed:** gold-document delivery
  **0.630 → 0.668**, **+18 / −0** needles — zero paired losses, and the map and
  final rank bases came back byte-identical per needle, so this is a pure
  delivery change ([#409](https://github.com/mbachaud/Cymatix-Context/pull/409)).
- **Graduated 0 → 12 in the same release** (the commit subject-tagged
  `[w24-floor-flip]`), gated on a cross-corpus confirm with zero paired
  delivered losses: EnronQA v2 **0.820 → 0.856** (+18/−0) and EnronQA padded
  597k **0.776 → 0.792** (+8/−0) alongside the 829k row; recall@12 / final
  recall@12 identical in every cell. `min_delivered_docs = 0` restores the
  legacy eviction-only trim byte-for-byte; the revert is one `git revert` of
  the flip commit.
- **What the receipts do not grade:** answer quality under a widened seat
  count. More documents in the window is not automatically a better answer,
  and twelve full-length documents per window costs real tokens — that lane is
  still owed.

### `[ingestion] entity_autolink_hub_cutoff` — bound the auto-link sweep

```toml
[ingestion]
entity_autolink_hub_cutoff = 200  # default since 0.9.2 (#425, closes #411); 0 = off = legacy linking
```

- Ingest-time entity auto-linking sweeps every `entity_graph` posting row of
  every probe entity before its per-insert `GROUP BY`. On hub-heavy corpora
  that is O(N²) per build: on a 289k-document EnronQA bed, `auto_link_by_entity`
  was **89.5% of writer wall time** — 148.9 ms/insert, versus 7.4 ms without
  the two MIME-header hubs.
- Above `0`, entities whose posting count exceeds the bound are dropped from
  the probe set before the `GROUP BY`. The count probe is `LIMIT`-bounded, so
  per-insert cost becomes O(n_entities × cutoff). The shipped `200` is the
  `PKI_NOISE_CUTOFF` precedent value. The knob shipped **off** in 0.9.1
  ([#412](https://github.com/mbachaud/Cymatix-Context/pull/412)); the flip to `200` lands in 0.9.2
  ([#425](https://github.com/mbachaud/Cymatix-Context/pull/425)) with both conditions on
  [#411](https://github.com/mbachaud/Cymatix-Context/issues/411) receipted — see the two bullets below.
- **Measured on a 500-document read-only replay at cutoff 200:** 0.53% of
  distinct entities held 56.9% of all postings; the per-link call went
  **210.5 ms → 0.62 ms (~340×)**
  ([#412](https://github.com/mbachaud/Cymatix-Context/pull/412)).
- **What the cutoff changes, and why it was gated:** which relation=5 COVER
  edges form — 4,842 → 2,867 on that bed. A store built with the cutoff on is
  not edge-identical to one built without it. Those edges are default-inert at
  query time (the W2.1 cover-walk arm was killed by its own receipt), but
  "inert" is not "absent" — which is why the flip waited for the retrieval-side
  measurement.
- **Flip condition 1 — retrieval null:** a paired EnronQA A/B (n=500; two beds
  differing only in the cutoff, gene-id sets identical) was null in both the
  0.9.0 and 0.9.1 configs — **0/500 per-needle delivered flips** (ledger row
  `2026-08-30-entity-hub-cutoff-retrieval-ab`, receipt
  `benchmarks/dogfood/receipts/entity_hub_cutoff_retrieval_ab_enronqa_2026-08-30.json`).
- **Flip condition 2 — the hubs survive tagger v2:** paired 500-document
  replays on content-identical 85k beds, one tagger-v1 and one tagger-v2
  (`benchmarks/dogfood/receipts/entity_hub_cutoff_reprobe_85k_taggerv1_2026-08-30.json`,
  `…_reprobe_85k_taggerv2_2026-08-31.json`, ledger row
  `2026-08-31-entity-hub-cutoff-reprobe-85k`). Tagger v2 removes 15.2% of
  distinct entities, but the natural hub population persists: at cutoff 200,
  **0.22% of entities still hold 30.1% of postings**, and the mean per-link
  call falls **4.15 → 0.42 ms**.
- **Opt out:** `entity_autolink_hub_cutoff = 0` restores byte-identical legacy
  linking. The bare `KnowledgeStore` constructor default stays `0`, and the
  bench builder reads the cutoff only from `CYMATIX_BFM_HUB_CUTOFF`.

### Wave-1 ranking defaults ([#407](https://github.com/mbachaud/Cymatix-Context/pull/407))

- `[retrieval] rrf_k` graduates **60 → 20**, tightening the rank-position
  discount so a high-ranked candidate from any single signal survives fusion.
  Set `60` to restore the Cormack 2009 default.
- `[retrieval] rerank_combinator_by_class` extends from `{multi_hop, default}`
  to **all five classifier classes**, all mapped to `eps_band`. Pass an
  explicit `{}` to restore global-combinator behavior.
- Together, on the 829k bed at n=470: delivered gold **0.555 → 0.630**,
  recall@12 **0.651 → 0.681**, paired **+18 / −4**, zero question-type
  regressions. The band also protects paraphrastic gold — the unbanded arm
  dropped semantic recall@12 0.32 → 0.16.
- Both halves ship as one commit, so the whole thing reverts with one
  `git revert`. Full context: [Pipeline](Pipeline).

### Tagger v2 ([#413](https://github.com/mbachaud/Cymatix-Context/pull/413)) — a behavior change, not a knob

- The CPU ingest tagger now rejects entities containing newlines or tabs, and
  no longer emits standard email/MIME header field names, `x-*` extension
  headers, or MIME transport artifacts as entities. It fixes the EnronQA
  entity-graph hub problem at the source.
- There is **no config knob**. The version is the module constant
  `TAGGER_VERSION = 2` in `cymatix_context/tagger.py`.
- **Comparability rule:** the tagger determines document boundaries and tags,
  so beds built under different tagger versions are **not cross-comparable**.
  Every bed built before 2026-08-30 is `tagger_version = 1`.
  `scripts/build_fixture_matrix.py` records the value in each bed's manifest
  and [`docs/benchmarks/BASELINES.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/benchmarks/BASELINES.md)
  requires matching versions for a cross-bed claim. See
  [Benchmarks and Receipts](Benchmarks-and-Receipts).

## Security knobs

A loopback bind is **not** authentication — any process on the same host can
reach the port. Three knobs harden the surface; two are opt-in, one is now on
by default.

```toml
[server]
admin_token = "your-secret"              # 401s /admin/*, /ingest, /consolidate without the bearer
swap_db_roots = ["F:/cymatix/genomes"]   # 403s /admin/swap-db targets outside these roots

[budget]
neutralize_control_tags = true           # default since 2026-08-19 (#351) — set false to opt out
```

- **`admin_token`** (default `""` = no auth). When non-empty, every `/admin/*`
  route plus `/ingest` and `/consolidate` demands
  `Authorization: Bearer <token>` and returns 401 otherwise. Read-only routes
  (`/stats`, `/health`, `/context`, …) stay open.
- **`swap_db_roots`** (default `[]` = unrestricted). When non-empty,
  `POST /admin/swap-db` only opens `.db` paths that resolve under one of the
  listed roots; anything else is rejected 403 before a store is opened. A root
  on a different drive than the target never matches.
- **`neutralize_control_tags`** (default `true` since 2026-08-19,
  [#351](https://github.com/mbachaud/Cymatix-Context/issues/351)). The assembly
  loop escapes a content-sourced `<cymatix:` to `&lt;cymatix:` so an ingested
  document cannot forge the genuine `<cymatix:no_match/>` / `<cymatix:slate>`
  control tags that downstream agents trust. The genuine no-match tag is
  emitted on a separate branch and is never escaped. The flip was
  receipt-gated: the escape runs strictly after retrieval, fusion, and rank
  publication, and the 141-needle 100k-carve A/B measured the delivered basis
  and **141/141 assembled windows byte-identical** across arms. Set `false` and
  documents legitimately containing the literal `<cymatix:` string ship
  verbatim.
  **Scope limit:** this closes control-tag forgery only. General indirect
  prompt injection through document text is inherent to context injection and
  is not addressed.
- **Startup warning.** When `[server] host` is non-loopback (not `127.0.0.1` /
  `localhost` / `::1`) while `admin_token` is empty, the app logs a prominent
  `NETWORK-EXPOSED ADMIN SURFACE` warning at startup. Warning only — the bind
  is honored unchanged.

## Environment variables

All process-level overrides use the `CYMATIX_` prefix and beat the TOML file.

| Variable | Overrides |
|---|---|
| `CYMATIX_CONFIG` | Path to the TOML file (default `./cymatix.toml`) |
| `CYMATIX_STORE_PATH` *(legacy `CYMATIX_GENOME_PATH`, which wins on conflict)* | `[knowledge_store] path`. `:memory:` for ephemeral runs |
| `CYMATIX_SERVER_UPSTREAM` / `CYMATIX_SERVER_UPSTREAM_TIMEOUT` | `[server] upstream` / `upstream_timeout` |
| `CYMATIX_DEVICE` | One-shot device pin, consumed by `hardware.init_from_config()` |
| `CYMATIX_PARTY_ID` | `[session] default_party_id` — the preferred way to set it |
| `CYMATIX_USE_SHARDS`, `CYMATIX_SHARD_WORKERS`, `CYMATIX_SHARD_FETCH_FACTOR`, `CYMATIX_SHARD_COACT_RESERVE` | Sharded read-path routing and fan-out. Inert in blob mode |
| `CYMATIX_FILENAME_ANCHOR_ENABLED` | `[retrieval] filename_anchor_enabled` |
| `CYMATIX_ABSTAIN_DISABLE` | Forces `[budget] abstain_enabled` off without a redeploy |
| `CYMATIX_ENCODER_URL` | `[encoder_daemon] url` |
| `CYMATIX_BENCH_ENABLED` | `[server] bench_enabled` |
| `CYMATIX_HEADROOM_ROUTE_UPSTREAM_AUTO` | `[headroom] route_upstream`, per launch |
| `CYMATIX_MCP_FULL` | Exposes the full MCP tool surface — see [MCP and IDE Integration](MCP-and-IDE-Integration) |
| `CYMATIX_OTEL_*` | `[telemetry]` — see the precedence note below |

**The OTel rule is env-wins in both directions.** `CYMATIX_OTEL_ENABLED=0`
silences a `[telemetry] enabled = true`, and the tray launcher's auto-export of
`CYMATIX_OTEL_ENABLED=1` beats a `[telemetry] enabled = false` once the
observability stack's collector port is up. Resolution happens in
`otel.resolve_telemetry_settings()`, not in the TOML loader. See
[Observability](Observability).

```bash
CYMATIX_STORE_PATH=genomes/dogfood/genome.db cymatix-server
CYMATIX_OTEL_ENABLED=1 CYMATIX_OTEL_ENDPOINT=localhost:4317 cymatix-server
```

## When a change takes effect

- **Restart required** for most of the surface: `[compressor]`, `[hardware]`,
  `[server]`, `[knowledge_store]`, `[ingestion]`, `[cymatics]`, `[classifier]`,
  `[plr]`, `[abstain]`, `[mem_sync]`, `[synonyms]`, `[vault]`, `[headroom]`,
  the `[budget]` token caps, and the `[retrieval]` weights and mode.
- **Read per request:** `[budget] foveated_alpha` / `foveated_c_min` /
  `foveated_base_chars`, `[retrieval] ann_threshold_mode`, and
  `[retrieval] fusion_mode`.
- **`POST /admin/refresh`** invalidates the retrieval-layer caches and the
  `[know]` calibration table. That route is one of the ones `admin_token`
  gates.
- When in doubt, restart. The hot-reload paths exist for the highest-churn
  knobs, not as a general contract.

*A note on names in code blocks: the shipped `cymatix.toml` on disk still uses
the legacy spellings (`[ribosome]`, `[genome]`, `genome.db`,
`max_genes_per_turn`), and this page reproduces them verbatim where it quotes
the file. Renaming the on-disk file and the SQL columns is a tracked Tier-3
change, deliberately not taken in 0.9.1 because it breaks every existing store
and consumer. See [Lexicon](Lexicon).*

## Go deeper

- [`docs/config-reference.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/config-reference.md) — the canonical per-key reference: type, default, and the rationale for every knob, plus loading order and hot-reload semantics
- [`cymatix.toml`](https://github.com/mbachaud/Cymatix-Context/blob/master/cymatix.toml) — the shipped file, with the flip receipts written inline as comments
- [`docs/benchmarks/2026-08-14-encoder-isolation-scale-curve.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/benchmarks/2026-08-14-encoder-isolation-scale-curve.md) — the four-scale receipts behind the dense and SPLADE flips
- [`CHANGELOG.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/CHANGELOG.md) — every default flip with its receipt path and disclosure
- [`docs/operator-runbooks.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/operator-runbooks.md) — backfill and calibration procedures with timings
- Next: [Retrieval Dimensions](Retrieval-Dimensions) · [HTTP API](HTTP-API) · [Benchmarks and Receipts](Benchmarks-and-Receipts) · [Observability](Observability)

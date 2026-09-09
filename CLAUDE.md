# Cymatix Context

Knowledge-store-based context compression for local LLMs. v0.9.2 (clean break in 0.8.5: renamed from helix-context in July 2026; the old helix CLI/env/import names, `helix.toml`, and `helix_*` MCP tools are removed — cymatix only).

## Quick Start

```bash
# Install
pip install cymatix-context

# Ingest content
cymatix ingest path/to/your/docs/ --recursive

# Query the store
cymatix query "what does the splice step do?"

# Inspect corpus state
cymatix diag corpus
cymatix status

# Start the FastAPI proxy (IDE integrations, MCP hosts)
cymatix-server
# or
python -m uvicorn cymatix_context._asgi:app --host 127.0.0.1 --port 11437
```

Full CLI reference: `docs/clients/cli.md`.

Cross-host MCP and Agent Skill guidance: [`docs/clients/cymatix-context.md`](docs/clients/cymatix-context.md),
with native guides for [Claude Code](docs/clients/claude-code.md),
[Codex](docs/clients/codex.md), [Gemini CLI](docs/clients/gemini-cli.md),
and [Antigravity](docs/clients/antigravity.md). Direct MCP needs neither the
model proxy nor the tray; a healthy headless server is sufficient.

## How It Works

A transparent OpenAI-compatible proxy that intercepts LLM requests and injects compressed context from a persistent SQLite knowledge store.

**7-stage pipeline per turn:**

0. **Classify** — rule-based query classifier picks decoder mode + assembly cap (no model call)
1. **Extract** — heuristic keyword extraction from query (no model call)
2. **Retrieve** — FTS5 lexical + tag lookup + synonym expansion + co-activation + cymatics 256-bin spectrum scoring (all algorithmic, no model inference); BGE-M3 dense recall (`[retrieval] dense_embedding_enabled`, **default OFF since 2026-08-15** — four-scale isolation receipts measured dense displacing gold from the delivered top-k at every scale, −0.20..−0.33 final recall; see `docs/benchmarks/2026-08-14-encoder-isolation-scale-curve.md`; set true to opt in) and SPLADE sparse expansion (`[ingestion] splade_enabled`, **default OFF since 2026-08-16** — n=469 receipts measured it null-to-negative over the full expansion index, and the index contributes nothing passively) are both **opt-in** — the default retrieval path is now fully algorithmic/neural-free (see `docs/design/2026-07-05-efficiency-cost-reduction.md` and `docs/benchmarks/2026-08-14-encoder-isolation-scale-curve.md`); ranks via Reciprocal Rank Fusion when `[retrieval] fusion_mode = "rrf"` (the default since 2026-07-06; `"additive"` is the legacy accumulator)
3. **Re-rank** — CPU model scores candidates by relevance (optional, off by default)
4. **Splice** — CPU model compresses each candidate, keeping high-value fragments (batched)
5. **Assemble** — join spliced parts, enforce token budget, attach per-document legibility headers (fired tiers, confidence marker, compression ratio), elide already-delivered documents via session working-set register
6. **Persist** — pack query+response exchange into knowledge store (background)

Additionally, a **freshness gate** (Stage 7) runs during assembly to demote stale, cold, or superseded documents.

The pipeline emits a **know/miss agent contract** on `/context/packet`: every response carries `know { found, confidence, gene_id_match }` or `miss { reason }` so downstream agents can calibrate trust without guessing.

## Package Structure (post-PR #90)

After the repo restructure, `cymatix_context/` is organized into 16 sub-packages plus a handful of top-level orchestration modules. (The old `helix_context` alias package was removed in 0.8.5.)

| Package | Purpose |
|---------|---------|
| `adapters/` | Cache, DAL, retriever abstractions |
| `backends/` | Compressor (formerly ribosome), DeBERTa, NLI, SEMA codec, SPLADE |
| `cli/` | `cymatix` CLI: query, packet, ingest, gene, neighbors, diag, config, status |
| `encoding/` | Chunking, fragment encoding, legibility headers, Headroom bridge |
| `identity/` | CWoLa logger, session delivery, registry, provenance, claims |
| `okf/` | Open Knowledge Format (OKF v0.1) bundle reader, canonical digest, ingest adapter (`cymatix ingest --okf`) |
| `pipeline/` | Tier logic, pipeline-stage helpers |
| `retrieval/` | Expand, freshness gate, RRF/additive fusion, PLR, intent router, SR, seeded edges, query classifier, tie-break |
| `scoring/` | Cymatics, know-calibration, know-decision, ray-trace, TCM, blending |
| `server/` | FastAPI app factory + route modules (context, ingest, registry, admin) |
| `storage/` | DDL, indexes, co-activation graph |
| `telemetry/` | OTel metrics, histogram instrumentation |
| `vault/` | Obsidian vault export (read-only diagnostic traces) |
| `launcher/` | System-tray supervisor (backend + telemetry stack) |
| `mcp/` | MCP tool surface for Claude Code, Codex, Gemini CLI, and Antigravity |
| `integrations/` | ScoreRift bridge |

Top-level modules: `context_manager.py` (pipeline orchestrator), `config.py` (TOML loader), `schemas.py` (Pydantic models), `knowledge_store.py` (SQLite DDL + retrieval), `codons.py` (chunker + encoder), `tagger.py` (CPU ingest tagger).

**Biology-lexicon shims:** `genome.py`, `ribosome.py`, `replication.py`, `hgt.py`, `server.py`, `mcp_server.py` re-export from their new locations (the domain metaphor, not the helix brand). See the wiki Lexicon page (docs/ROSETTA.md is a stub) for the full biology-to-software lexicon. The `helix_context` alias package was removed in 0.8.5 — import `cymatix_context`.

## Configuration

All config lives in `cymatix.toml` (the `helix.toml` fallback was removed in 0.8.5). Sections:

| Section | Key settings |
|---------|-------------|
| `[ribosome]` | model, backend (`"none"` default / `"litellm"` / `"deberta"` — only litellm and deberta are honored when enabled; any other value, including legacy `"ollama"` and `"claude"`, dispatches the no-op DisabledBackend), timeout, query_expansion_enabled |
| `[hardware]` | device auto-detection (CUDA, MPS, ROCm, CPU) |
| `[budget]` | expression_tokens (default 7000 — code and cymatix.toml unified in the 2026-06-12 default-honesty pass), max_genes_per_turn, splice_aggressiveness, decoder_mode, legibility_enabled, session_delivery_enabled, min_delivered_docs (**default 12 since 2026-08-30**, W2.4 seat-floor graduation, commit subject-tagged [w24-floor-flip] — lifts the classifier's per-rule assembly cap to at least N and switches the Stage-5 trimmer to largest-part truncation instead of whole-document eviction, the token budget still wins last; receipt-gated on three corpora with zero paired delivered losses: ERB 829k .630→.668 (+18/−0), EnronQA v2 .820→.856 (+18/−0), EnronQA padded 597k .776→.792 (+8/−0), ranking bases untouched in every cell; 0 = legacy eviction-only trim; revert = git revert of the flip commit) |
| `[session]` | synthetic_session_enabled, synthetic_session_window_s, default_party_id |
| `[genome]` | path (`genomes/main/genome.db`), compact_interval, cold_start_threshold, replicas |
| `[server]` | host, port, upstream |
| `[telemetry]` | OTel export defaults: enabled (default false), endpoint (`"localhost:4317"`), insecure, sampler_ratio, redact_query, logs_enabled, logs_level. Precedence: `CYMATIX_OTEL_*` env > toml > default (env wins both directions); the tray launcher auto-exports `CYMATIX_OTEL_ENABLED=1` once the observability stack's collector port is up |
| `[headroom]` | route_upstream toggle for Headroom proxy integration |
| `[encoder_daemon]` | url (default `""` = off — every seam encodes in-process, byte-identical to today; set to route dense/SPLADE/SEMA encoding through a shared `cymatix_context.encoder_daemon` process instead, e.g. `"http://127.0.0.1:11440"`; `CYMATIX_ENCODER_URL` env wins over the TOML value; start the daemon with `python -m cymatix_context.encoder_daemon`, default port 11440 — distinct from `[server] bench_port` 11439, see #376) |
| `[ingestion]` | backend (`"cpu"` / `"ollama"` / `"hybrid"`), splade_enabled, rerank_model, entity_graph, sema_embed_on_ingest (#227 knob, **default false since 2026-08-19**, #371 — Tier-4 sema_boost was its only live consumer at shipped defaults and the 829k read-gate cell was a wash (+1 delivered needle), PR #399/#400 receipts; set true to re-opt-in, then `scripts/backfill_sema.py` for docs ingested while off; TCM falls back to tag-hash/text by design — governs MiniLM only, not BGE-M3), dense_embed_on_ingest (**default false since 2026-08-19**, #371 — dead write at neural-free retrieval defaults, PR #379 receipt; set true to re-opt-in, then `scripts/backfill_bgem3_v2.py` for docs ingested while off), entity_autolink_hub_cutoff (**default 200 since 2026-08-31**, PR #425 closes #411 — posting-count hub cutoff for ingest-time entity auto-linking; entities with more than that many entity_graph postings are dropped from the probe set, so per-insert cost is O(n_entities × cutoff) instead of O(N²) on hub-heavy corpora; both flip conditions receipted: retrieval null 0/500 delivered flips on content-identical EnronQA twins (`benchmarks/dogfood/receipts/entity_hub_cutoff_retrieval_ab_enronqa_2026-08-30.json`) and hubs persist under tagger v2 — 0.22% of entities hold 30.1% of postings at 85k, per-link 4.15 → 0.42 ms (`entity_hub_cutoff_reprobe_85k_taggerv{1,2}` receipts); changes only which relation=5 COVER edges form, query-inert at defaults; `0` = legacy linking; the bare `KnowledgeStore` kwarg default stays 0 and the bench builder reads only `CYMATIX_BFM_HUB_CUTOFF`; sibling constructor-only scheduling knob — not a TOML key — `KnowledgeStore(entity_autolink=False)`, PR #424, skips COVER-edge formation while still writing entity_graph rows, precedence off > cutoff > legacy, bench builder `--no-entity-autolink` / `CYMATIX_BFM_ENTITY_AUTOLINK=0`) |
| `[context]` | cold_tier_enabled, cold_tier_k, cold_tier_min_cosine |
| `[cymatics]` | enabled, distance_metric (`"cosine"` / `"w1"`), harmonic_links |
| `[classifier]` | Rule-based query classifier: `enabled` toggle only; per-class caps/decoder hints are code constants pending #205 |
| `[retrieval]` | fusion_mode (`"rrf"` default / `"additive"` legacy), rrf_k (**default 20 since 2026-08-28**, wave-1 ranking-under-width graduation — delivered 0.555→0.630 at 829k with the all-classes eps_band map; Cormack default was 60), rerank_combinator_by_class (**all five classes → eps_band since 2026-08-28**, same receipts; was `{multi_hop, default}` since 2026-07-16), dense_embedding_enabled (**default false since 2026-08-15**), pki_enabled (**default false since 2026-08-17**, #370 — −1 delivered needle at n=469; the flip does not reclaim an existing path_key_index table), sr_enabled, sr_gamma, ray_trace_theta, seeded_edges_enabled, cover_walk_enabled (**default false** — W2.1 query-time walk over the ingest-built gene_relations COVER graph: eps-band-confined rerank class + append-below-head rescue lane; flip is receipt-gated on the w2_cover_walk arm), fts5_candidate_depth (#205: FTS content-tier fetch depth; 0 = auto = max_genes*4), blend_mode (`"scale_relative"` default / `"legacy"` DEPRECATED-FOR-REMOVAL, condition-gated — not calendar-based; target v(N+2) at earliest) |
| `[plr]` | Piecewise linear reranker: enabled, model_path |
| `[know]` | KnowBlock confidence logistic: emit_floor, betas, s_ref, g_ref, stale_after_days (+ calibrated_at / calibrated_on_n written by scripts/calibrate_know_confidence.py) |
| `[mem_sync]` | Auto-memory-to-cymatix sync: watch_dirs, sync_interval_s |
| `[synonyms]` | Lightweight query expansion (e.g., "cache" -> ["redis", "ttl", "invalidation"]) |
| `[abstain]` | Abstention thresholds for low-confidence responses |

## HTTP Endpoints

**Core retrieval:**
```
POST /v1/chat/completions  — OpenAI-compatible proxy (primary integration)
POST /context              — { query } → expressed context (Continue-compatible)
POST /context/packet       — agent-safe bundle: verified / stale_risk / refresh_targets
POST /context/refresh-plan — refresh_targets only (reread plan, no evidence items)
POST /fingerprint          — navigation-first payload with tier scores, not content
GET  /context/expand       — 1-hop neighborhood from a gene_id (forward/backward/sideways)
```

**Ingestion + maintenance:**
```
POST /ingest               — { content, content_type, metadata? } → gene_ids
POST /consolidate          — rewrite stale documents from their source fingerprints
POST /admin/refresh        — force retrieval-layer refresh
POST /admin/vacuum         — reclaim SQLite pages
POST /admin/compact        — run compaction pass
POST /admin/checkpoint     — WAL checkpoint
```

**Identity + sessions:**
```
POST /sessions/register    — register an agent participant for attribution
GET  /sessions             — list registered participants
GET  /sessions/{handle}/recent — recent documents by agent handle
GET  /session/{id}/manifest — session delivery log (what was shipped to this session)
POST /hitl/emit            — record a human-in-the-loop pause event
GET  /hitl/recent          — recent HITL events
```

**Diagnostics:**
```
GET  /stats                — corpus metrics + compression ratio
GET  /health               — model, document count, upstream URL
GET  /genes/{gene_id}      — single document detail
GET  /debug/resonance      — tier activation profile
GET  /debug/neighbors      — co-activation graph neighbors
GET  /metrics/tokens       — token usage counters
GET  /vault/status         — Obsidian vault export state
```

## Testing

```bash
# ~4,165 tests, no external services needed
python -m pytest tests/ -m "not live" -v

# Live tests (requires Ollama running)
python -m pytest tests/ -m live -v -s
```

## Observability (Grafana telemetry)

Cymatix ships a native OTel sidecar (collector + Prometheus + Tempo + Loki + Grafana):

```powershell
scripts\setup-grafana-telem.ps1     # Windows
scripts/setup-grafana-telem.sh      # Linux / macOS
```

Enable telemetry on the backend:

```bash
CYMATIX_OTEL_ENABLED=1 CYMATIX_OTEL_ENDPOINT=localhost:4317 \
  python -m uvicorn cymatix_context._asgi:app --port 11437
```

Dashboards: <http://localhost:3000/d/cymatix-overview>.
See [`docs/architecture/OBSERVABILITY.md`](docs/architecture/OBSERVABILITY.md) for the full instrumentation surface.

## Gotchas

- **Knowledge store path:** Default is `genomes/main/genome.db` (not project root). Delete to start fresh; auto-creates on first use.
- **Synonym map is critical:** If queries return "no relevant context", check that query keywords map to the tags assigned at ingest. Add synonyms in `[synonyms]`.
- **Stage 2 backfill:** only relevant if you opt in to dense (`[retrieval] dense_embedding_enabled = true`, default off since 2026-08-15). Ingest-time dense writes are also off by default (`[ingestion] dense_embed_on_ingest = false` since 2026-08-19, #371), so BGE-M3 vectors (`embedding_dense_v2`) are NULL until you run `scripts/backfill_bgem3_v2.py` (or re-ingest with the knob on) — the dense tier retrieves nothing without them.
- **Fusion mode:** Default is `"rrf"` (Reciprocal Rank Fusion) since 2026-07-06 — SIKE Run-2 measured +12pp gold_delivered over additive on xl. Set `fusion_mode = "additive"` in `[retrieval]` to restore the legacy pre-Stage-3 accumulator (scheduled for removal in v(N+2)). Under RRF the abstain gates run ratio-only (absolute floors were additive-calibrated).
- **Session delivery:** `session_delivery_enabled = true` tracks what documents each session has already received and elides repeats. Saves ~40% tokens on multi-turn conversations (unverified design estimate — pending the `cymatix_session_tokens_saved_total` counter). Flip to false or pass `ignore_delivered: true` in /context body for benchmarks.
- **Ingest concurrency changes the bed:** parallel LLM tagging (client pool > 1 and/or `OLLAMA_NUM_PARALLEL` > 1) produces different gene content and tags than sequential ingest of the same bytes, even at temp 0 (LLM-batching drift, byte-receipted 2026-08-08: content differed on 105/150 shared genes — cc-exchange EXP-A refutation). Cross-bed comparisons must pin ingest concurrency (`ingest_c` in `docs/benchmarks/BASELINES.md`); comparable rebuilds ingest sequentially unless the comparator was built parallel.
- **Continue IDE:** Use Chat mode, not Agent mode. The proxy doesn't handle tool routing.
- **Naming lexicon:** Biology terms (gene, genome, ribosome, chromatin, splice) have canonical software equivalents (document, knowledge store, compressor, etc.). See the wiki Lexicon page (docs/ROSETTA.md is a stub) for the full mapping. Both vocabularies work in code (back-compat shims); new code should use the software terms.

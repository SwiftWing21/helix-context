# Roadmap and Releases

- **Releases are receipt-gated, not calendar-gated.** There is no sprint
  cadence and no scheduled ship date. A default changes when a measurement
  says it should, and the measurement ships with it — including the
  measurements that cost something.
- **The canonical record is
  [`CHANGELOG.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/CHANGELOG.md).**
  Every default flip there carries its receipt path, its caveat, and its
  opt-back-in line. This page is the reading order, not a replacement.
- **The forward plan is
  [`docs/ROADMAP.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/ROADMAP.md)** —
  the sequencing layer that says which track a piece of work sits under.
- **Install from PyPI:** `pip install cymatix-context`. Released versions are
  git-tagged (`v0.9.2`, `v0.9.1`, `v0.9.0`, …) on the repository; pre-releases
  (`vX.Y.ZbN`, cut from `beta`) install with `pip install --pre cymatix-context`.
- This wiki documents **v0.9.2**. Each item below carries
  its PR or issue number. The first section highlights changes since the
  published v0.9.1 release (2026-08-30).

## 0.9.2 (2026-09-08) — release contents

The highlights below follow the 0.9.1 tag. One ingestion default changes —
`[ingestion] entity_autolink_hub_cutoff` 0 → 200 — alongside additive features,
fixes, and bench work. Every number here is copied from the receipt named in
the matching `CHANGELOG.md` 0.9.2 entry.

| Thread | What it is | Default impact |
|---|---|---|
| Tier-2 lexicon aliases ([#419](https://github.com/mbachaud/Cymatix-Context/pull/419)) | `[compressor]` / `[knowledge_store]` config sections, `retrieval_tokens` / `max_docs_per_turn` keys, `CYMATIX_STORE_PATH`, `cymatix document get`, `--max-docs`, `GET /documents/{id}`, canonical `cymatix_document_*` MCP tools | Additive only; legacy wins on collision |
| Wiki, README, Tier-1 prose sweep ([#420](https://github.com/mbachaud/Cymatix-Context/pull/420)) | This wiki as the docs source of truth (GitHub wiki + cymatixcontext.com/wiki), README cut to a landing page, `docs/ROSETTA.md` retired to a stub | Docs only |
| `filename_index` gene_id index + auto-link scheduling knob ([#424](https://github.com/mbachaud/Cymatix-Context/pull/424)) | The per-upsert `DELETE` was a full-table scan — **42.7 ms → 0.002 ms** at 572k rows; `KnowledgeStore(entity_autolink=False)` skips COVER-edge formation | Content-neutral; existing stores gain the index on their next writable open |
| Entity auto-link hub cutoff flip ([#425](https://github.com/mbachaud/Cymatix-Context/pull/425), closes [#411](https://github.com/mbachaud/Cymatix-Context/issues/411)) | `[ingestion] entity_autolink_hub_cutoff` **0 → 200**; both flip conditions receipted (retrieval null 0/500; hubs persist under tagger v2, per-link 4.15 → 0.42 ms) | **Changes defaults** — ingest-side only; `0` opts back out |
| Semantic-portfolio bench lane ([#426](https://github.com/mbachaud/Cymatix-Context/pull/426)) | LoCoMo(+Plus), MULocBench, and FinanceBench adapters + baselines (delivered 0.4348 / 0.4485 cold-gated / 0.153) | Bench only — new-corpus rows, not comparable to ERB or EnronQA rows |
| Paraphrase crater, floor width curve, semantic-cap decomposition ([#427](https://github.com/mbachaud/Cymatix-Context/pull/427)) | Floor 0 / 6 / 12 → delivered **0.6298 / 0.6447 / 0.6681** on the 947k bed, strictly nested — 12 confirmed | Bench only |
| `eps_band_coverage` combinator ([#428](https://github.com/mbachaud/Cymatix-Context/pull/428)) | W2.2-narrow distinct-term tie-break inside the ε band; killed by its own receipt (**+0 / −0** delivered on 947k) | Default-inert; opt in per class |
| ERB Phase-1/2 admission campaign ([#429](https://github.com/mbachaud/Cymatix-Context/pull/429)) | Pool-depth forensics: the admission thesis lives with corrections (66/109 pool-absent misses have gold at depth 1000), but a bare depth-1000 flip projects 0.634 vs 0.668 shipped | Bench only, no default moves |
| Harmonic-tier bind-limit warning ([#432](https://github.com/mbachaud/Cymatix-Context/pull/432), closes [#431](https://github.com/mbachaud/Cymatix-Context/issues/431) tier 1) | Tier 5 skips with one warning per store when the candidate list would exceed `SQLITE_LIMIT_VARIABLE_NUMBER`, instead of swallowing the error at DEBUG | Ranking byte-identical below the limit |
| Admission provenance and test audit ([#446](https://github.com/mbachaud/Cymatix-Context/pull/446)) | Opt-in per-stage candidate counts and watched-gold membership, explicit unmeasured states, actual budget tiers, and repaired filtering/eviction/citation assertions | Bench instrumentation; serial per manager/store; retrieval defaults unchanged |
| Configuration shape recovery and dimension defaults ([#447](https://github.com/mbachaud/Cymatix-Context/pull/447)) | Malformed known TOML section shapes warn and fall back; valid settings and aliases survive; dimension-default claims match the shipped configuration | Invalid-config recovery and docs; no default changes |
| Golden, Windows, and dependency coverage ([#448](https://github.com/mbachaud/Cymatix-Context/pull/448)) | Active archived-rendering and current-caller parity checks, native Bash/CLI discovery, declared contributor dependencies and CI import preflight | Tests and contributor setup; runtime defaults unchanged |
| Observability exposure and concurrency fixes ([#449](https://github.com/mbachaud/Cymatix-Context/pull/449)) | Docker ports bind to loopback, anonymous Grafana access is opt-in; SPLADE loads once, freshness caches are bounded and refreshed, and score publication uses locked copies | Docker access defaults tightened; retrieval arithmetic and defaults unchanged |
| Final merged-stack witness ([#450](https://github.com/mbachaud/Cymatix-Context/pull/450)) | September 8 at `8cab199`: all 470 queries complete with zero errors; all 11 compared fields reproduce the frozen reference; 314/470 delivered, r@12 0.6809, final r@12 0.6830 | Release evidence; ranks and delivery only, no latency claim |

- **Known, tracked as [#430](https://github.com/mbachaud/Cymatix-Context/issues/430):**
  `[budget] min_delivered_docs` does not floor the TIGHT/FOCUSED budget-tier
  cuts — **152 of 470 ERB 947k needles deliver 6 seats at floor 12**, including
  119 of the 314 hits, and the seat count flips 6 ↔ 12 on 33/216 needles when
  only admission knobs move. The fix changes shipped seat counts on ~32% of ERB
  needles, so it is receipt-gated (a paired ERB 947k ladder plus the v0.9.1
  gate beds) rather than patched in 0.9.2.

## 0.9.1 (2026-08-30) — the retrieval-quality release

Six threads shipped. Two move retrieval defaults, one changes what a fresh
ingest produces, and the rest are additive or ship off. The release gate was
PRs [#406](https://github.com/mbachaud/Cymatix-Context/pull/406)–[#409](https://github.com/mbachaud/Cymatix-Context/pull/409), [#412](https://github.com/mbachaud/Cymatix-Context/pull/412), [#413](https://github.com/mbachaud/Cymatix-Context/pull/413), [#422](https://github.com/mbachaud/Cymatix-Context/pull/422), certified by the three-arm
sweep in ledger row `2026-08-30-v091-gate-sweep` (ALL PASS) — see
[Benchmarks and Receipts](Benchmarks-and-Receipts).

| Thread | What it is | Default impact |
|---|---|---|
| Wave-1 ranking flip ([#407](https://github.com/mbachaud/Cymatix-Context/pull/407)) | `rrf_k` 60 → 20, `eps_band` combinator extended to all five classifier classes | **Changes defaults** |
| Delivered-seat floor ([#409](https://github.com/mbachaud/Cymatix-Context/pull/409)) | New `[budget] min_delivered_docs` knob, graduated **0 → 12** in the same release | **Changes defaults** |
| Entity auto-link hub cutoff ([#412](https://github.com/mbachaud/Cymatix-Context/pull/412)) | New `[ingestion] entity_autolink_hub_cutoff` knob | Default `0` = off in 0.9.1 (flipped to `200` in 0.9.2 by [#425](https://github.com/mbachaud/Cymatix-Context/pull/425)) |
| Tagger v2 ([#413](https://github.com/mbachaud/Cymatix-Context/pull/413)) | CPU ingest tagger rejects newline/tab entities and email/MIME header plumbing | **Changes new ingests** |
| Cross-host client unification ([#406](https://github.com/mbachaud/Cymatix-Context/pull/406)) | One MCP contract and native guides for Claude Code, Codex, Gemini CLI, and Antigravity under `docs/clients/`; `cymatix status` reports host readiness and live MCP state | Additive only |
| EnronQA second-corpus bench lane ([#422](https://github.com/mbachaud/Cymatix-Context/pull/422)) | 73,772-email corpus, 250 paired original / rephrased needles, a padded 597k bed, and the v0.9.1 gate receipts | Bench only |

### Wave-1 ranking flip (#407)

- The first of the release's two default changes. On the 829k bed at full needle power,
  gold-document delivery moves **0.555 → 0.630** (+43 / −8 paired), recall@12
  **0.651 → 0.681**, median gold rank **3 → 2**, with **zero question-type
  regressions**.
- It ships as a **single named commit** so `git revert` is a one-step
  rollback, and the post-flip repository `cymatix.toml` was verified
  field-identical to the measured arm config.
- The cross-shard merge constant deliberately stays at **60** — that surface
  was never measured, and is annotated as unmeasured in code rather than
  flipped on faith.
- The full receipt, the bed identity, and why 0.630 is *not* a successor to the
  shipped-defaults 0.565 row are on
  [Benchmarks and Receipts](Benchmarks-and-Receipts).

### Delivered-seat floor (#409) — knob and flip, both in the release

- **`[budget] min_delivered_docs`** ships at **12**, graduated from `0` on
  2026-08-30 by the commit subject-tagged `[w24-floor-flip]`; `0` restores the
  byte-identical eviction-only legacy trim. At floor 12 the receipts measure
  delivery **0.630 → 0.668** on the 829k bed with **zero losses**, and
  per-needle ranking bases byte-identical — a pure delivery change.
- The flip was gated on a cross-corpus confirm, not the 829k row alone:
  EnronQA v2 **0.820 → 0.856** (+18/−0) and EnronQA padded 597k
  **0.776 → 0.792** (+8/−0) — three corpora, zero paired delivered losses,
  recall@12 / final recall@12 identical in every cell.
- **What the receipts still do not grade:** answer quality under a wider seat
  count. Twelve full-length documents per window costs real tokens, and ERB
  does not grade that trade — the answer-quality lane is still owed.

### Entity auto-link hub cutoff (#412) — knob shipped off

- **`[ingestion] entity_autolink_hub_cutoff`** (default `0` = off, pinned
  byte-identical by test). On a hub-heavy 289k-document bed it takes the mean
  per-link call from 210.5 ms to 0.62 ms — roughly **340×**. **The 0.9.1
  default stays 0** because the edge delta is real: the graph that comes out is
  a different graph, so the knob ships opt-in with the receipt attached. Flip
  conditions live on
  [#411](https://github.com/mbachaud/Cymatix-Context/issues/411); the
  retrieval-side condition has since been measured null (a paired EnronQA A/B,
  0/500 delivered flips in both configs), the post-tagger-v2 re-measure shows
  the hubs persist, and the flip to `200` ships in 0.9.2 via
  [#425](https://github.com/mbachaud/Cymatix-Context/pull/425) — see the 0.9.2 section above.

### Tagger v2 (#413) — a behavior change with a version bump

- The CPU ingest tagger now rejects entities containing newlines or tabs, plus
  email/MIME header field names, `x-*` extension headers, and MIME transport
  artifacts. On a 2,000-file EnronQA sample that is **−16.7% entity-graph
  rows** and zero multi-line entities, with document ids unchanged.
- **This is a comparability break by design.** Tags are part of the bed-content
  digest, so `TAGGER_VERSION = 2` is declared in the tagger, recorded in every
  bed manifest, and added to the BASELINES rules. **Every bed built before
  2026-08-30 is `tagger_version = 1`** — internally valid, but not
  cross-comparable with a v2 bed.
- There is **no backfill script on purpose**: stripping entities in place would
  produce hybrid v1/v2 beds. v2 beds are fresh builds.
- Origin issue: [#410](https://github.com/mbachaud/Cymatix-Context/issues/410).

### After the 0.9.1 tag — Lexicon Tier 2 and the docs pass

Neither of these is in the 0.9.1 wheel: both follow the 0.9.1 tag on `master`
and ship in v0.9.2. This wiki presents their spellings as canonical; on a
0.9.1 install, use the legacy spellings.

- **Tier 2 aliases** ([#419](https://github.com/mbachaud/Cymatix-Context/pull/419))
  land on every operator-facing surface — `[compressor]` /
  `[knowledge_store]` config sections, `retrieval_tokens` /
  `max_docs_per_turn` keys, `CYMATIX_STORE_PATH`, `cymatix document get`,
  `--max-docs`, `GET /documents/{id}`, canonical `cymatix_document_*` MCP
  tools. Additive only; every legacy spelling keeps working, with legacy
  winning on collision. This closes the long-deferred R4 phase
  ([#87](https://github.com/mbachaud/Cymatix-Context/issues/87)).
- **Tier 3 — the wire surface — was deliberately not taken.** See
  [#417](https://github.com/mbachaud/Cymatix-Context/issues/417) and the
  [Lexicon](Lexicon).
- **Documentation** ([#420](https://github.com/mbachaud/Cymatix-Context/pull/420)): this wiki (also rendered at
  <https://cymatixcontext.com/wiki/>),
  [`docs/ROSETTA.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/ROSETTA.md)
  retired to a stub pointing at [Lexicon](Lexicon), a repo-wide software-term
  prose sweep, and a shorter README that hands depth to the wiki.

## 0.9.0 (2026-08-20) — the post-flip release

The release where the shipped default retrieval path became **fully
algorithmic**. Five encoder-related defaults flipped off, each behind its own
receipt.

| Default flipped off | Date | One-line reason |
|---|---|---|
| `[retrieval] dense_embedding_enabled` | 2026-08-15 | Four-scale isolation measured BGE-M3 dense **displacing gold** from the delivered top-k at every scale |
| `[ingestion] splade_enabled` | 2026-08-16 | Null-to-negative against the lexical floor at n=469, and the expansion index contributed nothing passively |
| `[retrieval] pki_enabled` | **2026-08-17** ([#370](https://github.com/mbachaud/Cymatix-Context/issues/370)) | −1 delivered needle at full power; the flip does not reclaim an existing `path_key_index` table |
| `[ingestion] sema_embed_on_ingest` | 2026-08-19 ([#371](https://github.com/mbachaud/Cymatix-Context/issues/371)) | The deciding read-gate cell was a wash; ~9.7 s of cold start removed as a rider |
| `[ingestion] dense_embed_on_ingest` | 2026-08-19 ([#371](https://github.com/mbachaud/Cymatix-Context/issues/371)) | A dead write at neural-free retrieval defaults |

Every one of these is **one config line to opt back in**. The receipts and
their caveats are on [Benchmarks and Receipts](Benchmarks-and-Receipts); the
per-knob view is on [Configuration](Configuration).

**The two disclosures shipped with the release notes, not after them:**

- **The know surface never fires at shipped defaults**
  ([#287](https://github.com/mbachaud/Cymatix-Context/issues/287)). Max
  confidence lands around 0.28 against an `emit_floor` of 0.45, and the
  `lexical_dense_agree` calibration feature is structurally dead on the
  neural-free path. The failure direction is fail-safe — 0% false-KNOW — but
  it means **the `confidence` scalar is not a usable trust signal today**;
  rely on `found` / `reason`. See [Agent Contract](Agent-Contract).
- **The dense-off default carries a measured p50 latency regression**
  ([#374](https://github.com/mbachaud/Cymatix-Context/issues/374)): **×2.5–2.6
  slower at 100k** fragments, shrinking to **×1.09–1.38 at the 829k** operating
  point. Dense was load-bearing as a *latency device* — its ANN gate capped the
  candidate list feeding splice. The named mitigation, a lex-branch candidate
  cap, **has not landed**.

Also in 0.9.0: `[budget] neutralize_control_tags` flipped **on** (assembly-time
escaping of forged `<cymatix:` control tags, gated on a 141/141
byte-identical-window receipt), a startup warning for a non-loopback bind with
an empty `admin_token`, the MCP 2.x migration, and the #219 slice-5 config
honesty pass that removed five knobs with zero runtime readers.

## Earlier releases

| Version | Date | Headline |
|---|---|---|
| **0.8.6** | 2026-08-05 | Shared GPU encoder daemon (`[encoder_daemon]`), executor sizing, ERB scale fixes |
| **0.8.5** | 2026-07-25 | **Breaking.** The helix → cymatix rename completed as a clean break — all 0.8.0 back-compat removed. Migrate, or pin `cymatix-context<0.8.5` |
| **0.8.0** | 2026-07-22 | The helix → cymatix soft rename: canonical package `cymatix_context`, old names kept as live aliases |
| **0.7.2b1** | 2026-07-06 (beta) | Efficiency and bench-validity wave: fp32-BLOB SEMA embeddings, the lean 5-tool MCP surface, read-only serving |

Full entries, including the pre-0.7 line, are in
[`CHANGELOG.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/CHANGELOG.md).
The vocabulary shift these releases are often confused with — biology terms to
software terms — is a separate story, told on [Lexicon](Lexicon).

## What is deferred, and where it is tracked

- **The 0.9.x deferral ledger** is the comment dated 2026-08-19 on the v0.9.0
  release-gate issue
  [#377](https://github.com/mbachaud/Cymatix-Context/issues/377). That comment
  is the sequencing record for the work 0.9.0 knowingly did not do — the
  know-gate recalibration (#287), the sharded structural gap (#275), the
  retrieval-profile layer (#205), and the rest. Read it before proposing that
  something be picked up; it usually says why it was not.
- **The `entity_graph` layer ships on and has never been ablated** on the
  delivered basis. It sits on the retrieval-layer ledger as a
  REMOVE-CANDIDATE, and the arm is still owed.
- **The sharded path trails the unsharded engine** by roughly 31pp recall@10
  and 30pp MRR on the xl bed
  ([#275](https://github.com/mbachaud/Cymatix-Context/issues/275)). Prefer
  unsharded for accuracy-sensitive corpora; the gap is disclosed, not fixed.
- **Issue ledger carried forward from v0.9.1:**

| Item | What it is |
|---|---|
| [#417](https://github.com/mbachaud/Cymatix-Context/issues/417) | Tier 3 lexicon — `<GENE>` blocks, decoder prompts, wire field names, `/stats` keys. Needs its own byte-level A/B gate; v1.0-scale work |
| [#418](https://github.com/mbachaud/Cymatix-Context/issues/418) | Resolved in v0.9.2 by [#447](https://github.com/mbachaud/Cymatix-Context/pull/447): non-table known config sections warn and fall back instead of crashing |
| [#411](https://github.com/mbachaud/Cymatix-Context/issues/411) | Flip proposal and conditions for `entity_autolink_hub_cutoff`; both conditions receipted, the flip itself (0 → 200) is [#425](https://github.com/mbachaud/Cymatix-Context/pull/425) in 0.9.2 |
| [#421](https://github.com/mbachaud/Cymatix-Context/issues/421) | Resolved in v0.9.2 by [#447](https://github.com/mbachaud/Cymatix-Context/pull/447): dimension-default claims agree with shipped settings and point to the generated reference |
| [#410](https://github.com/mbachaud/Cymatix-Context/issues/410) | The tagger-hygiene root cause that #413 fixes |

## How releases work here

- **Two branches carry the code.** `master` holds released code and release
  tags only, so `master` always equals the latest tag. `beta` is the
  integration branch and the default base for pull requests; everything
  headed for the next release lands there first. A release is a
  `release/vX.Y.Z` branch cut from `beta` and merged to `master` by pull
  request; a `hotfix/*` branch is cut from `master` and merged back into
  `beta` afterwards.
- **Pre-releases come from `beta`.** A `vX.Y.ZbN` tag on `beta` is published
  as a GitHub pre-release and lands on PyPI as a beta. Opt in with
  `pip install --pre cymatix-context`; a plain install keeps resolving to the
  last final release.
- **Releases are receipt-gated, not calendar-gated.** A shipped-default
  change needs its paired receipt and `BASELINES.md` row before it merges to
  `beta`, and a final release needs a merged-stack witness ladder on the 947k
  bed. The [September 8 final witness](https://github.com/mbachaud/Cymatix-Context/blob/master/benchmarks/dogfood/receipts/sweep_v092_witness_947k_2026-09-08.json)
  at `8cab199` is **EXACT_REPRODUCE** on all 11 compared fields. The
  September 4 witness remains historical evidence for its own pinned
  stack; v0.9.1's gate remains `sweep_v091_gate_2026-08-30.json` (ALL PASS).
  When the receipt says a change is null or negative, the change does not
  ship: the wave-2
  COVER-walk arm in the 0.9.x window was killed by its own receipt
  ([#408](https://github.com/mbachaud/Cymatix-Context/pull/408) shipped the
  infrastructure inert, `cover_walk_enabled = false`).
- **Defaults move separately from features.** A knob and its default flip
  are different pull requests: the knob ships default-inert with a test
  pinning byte-identical legacy behavior, and the flip is its own
  receipt-gated change. #412's knob shipped in 0.9.1 without its flip, which
  followed in 0.9.2 as #425.
- **Versioning is `MAJOR.MINOR.PATCH`, but the number alone will not tell
  you whether an upgrade is safe.** 0.8.5 was a deliberate breaking change
  shipped as a patch-level bump, called out as breaking in the changelog with
  a pin instruction. Read the changelog entry, not the digits.
- **Disclosures ship with the release, not after it.** If a flip costs
  latency, or a surface silently stops working at defaults, it goes in the
  release notes next to the win. Comparability rules (`tagger_version`,
  `ingest_c`) are part of the same contract; see
  [Benchmarks and Receipts](Benchmarks-and-Receipts).
- **The runbook is
  [`docs/RELEASING.md`](https://github.com/mbachaud/Cymatix-Context/blob/beta/docs/RELEASING.md)**:
  exact commands for pre-releases, the final cut, the post-release wiki
  sync, and the `scripts/release.py` helper that bumps the version and rolls
  the changelog. It links to `beta`, not `master`, because it describes the
  process rather than a shipped version.

## Go deeper

- [`CHANGELOG.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/CHANGELOG.md) — every release entry with its receipt paths, caveats, and opt-back-in lines
- [`docs/ROADMAP.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/ROADMAP.md) — the forward-planning doc: live tracks, issue triage, and the cross-box data pipeline
- [`docs/benchmarks/BASELINES.md`](https://github.com/mbachaud/Cymatix-Context/blob/master/docs/benchmarks/BASELINES.md) — the bed-identity and receipt ledger every number above is anchored to
- [#377](https://github.com/mbachaud/Cymatix-Context/issues/377) — the v0.9.0 release gate, whose 2026-08-19 comment is the 0.9.x deferral ledger
- <https://pypi.org/project/cymatix-context/> — released versions
- Next: [Benchmarks and Receipts](Benchmarks-and-Receipts) · [Configuration](Configuration) · [Lexicon](Lexicon) · [Home](Home)

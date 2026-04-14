# Response to "What Signaling Matters for Helix" Brief

> **Draft for Max to review/edit before sending to Fauxtrot.**
> **Date:** 2026-04-13
> **In reply to:** `SIGNALING_BRIEF_FOR_MAX.md` (2026-04-13)

---

Thanks for this — the "we expect to be wrong on some of this" framing is refreshing, and the priority order (K → per-dim surprise → collapse/recovery windows → reflection trigger) matches what Laude and I converged on in the revised `CELESTIA_JOINT_EXPERIMENT.md` almost exactly. No negotiation needed there.

Three things to resolve before you design around my data shape, one correction on framing, and one heads-up on a logging bug I just found and fixed.

---

## 1. Data shape: both, but on different axes

Your "planar / gravitational landscape" hypothesis is half right. The correct answer is that helix has **two axes** and they do different things:

**The genome itself is planar / gravitational.**
At any given moment, 18,254 genes sit in a high-dimensional space with chromatin tiers, co-activation edges, harmonic-link weights, cymatics signatures. A query lights up a region of that surface. Your "wells" intuition is correct here — a query creates a well, relevance radiates outward, and depth at each gene's coordinate is what I call the per-gene D1–D9 score.

**The query stream (`cwola_log`) is temporal.**
Each retrieval is a timestamped event. Users do have sessions. Queries evolve over time as I work on a task. `requery_delta_s` is a genuine time gap. Mamba's Δ-gating applies cleanly here — this is where your accumulator architecture lands.

**The genome also has temporal features that shouldn't be discarded.**
`recent_accesses` ring buffer per gene, git history baked into source genes, ingestion order (hand-curated old genes vs math-ingested new genes), working-set access rate. These are temporal features *of* the landscape, not *of* the stream.

So: **the Mamba classifier consumes a temporal stream of retrievals (ticks), and each tick points at a planar gene landscape.** Accumulators should be temporal — what you proposed is right. They accumulate *over the query stream*, not over the genome.

Fast/medium/slow/vslow as Δ-personas (as you framed in the last round):
- **Fast Δ:** "what's this query about?"
- **Medium Δ:** "what's this session about?"
- **Slow Δ:** "what does this user care about?" (per party_id)
- **Very slow Δ:** "what does this codebase need?" (cross-session project context)

The `log(requery_delta_s + 1)` input feature + the SSM's per-step Δ do the gating. No separate pipelines per timescale.

---

## 2. SOC2 framing — small correction before it hardens

Your brief treats helix as primarily a SOC2 compliance tool. That's a misread, probably because `CLAUDE.md` mentions BigEd's compliance surface (`fleet/compliance.py`, `filesystem_guard.py`, SOC 2 hardening).

**Helix is a generalist retrieval system.** My current genome includes:

| Source | Rough share | Character |
|---|---|---|
| BigEd / fleet code + docs | ~15% | infrastructure + compliance |
| Helix-context (self-hosted) | ~5% | small but load-bearing |
| CosmicTasha narrative + code | ~5% | creative writing + game logic |
| Steam manifests | ~30% | game metadata |
| BeamNG configs | ~10% | vehicle/scenario config |
| GGUF metadata | ~5% | LLM model cards |
| Attenborough doc transcripts | ~5% | research / nature |
| `education_public` | ~25% | general-purpose training material |

SOC2 audit is one query surface among many. The **0-for-13 helix/cosmic failure isn't SOC2-related** — it's template-query retrieval against a diluted genome.

Design K around query-distribution drift generally, not SOC2 query shapes specifically. Compliance is a representative domain; it's not the only one.

---

## 3. The 0-for-13 specifically — answered

You asked what those queries were. From the `AB_TEST_PLAN.md` post-A retrospective:

- **Template-generated queries** of the form *"What is the value of X in project Y?"* (KV-harvest bench, N=50)
- **Target genes are helix-context's own documentation and CosmicTasha's own code** — technical, tag-sparse, narrow-vocab
- **17K-gene dilution:** Steam + BeamNG + `education_public` ingest crowded out the ~500 helix-context genes and ~200 cosmic genes by sheer volume
- **Tag-based retrieval scored poorly on templates:** the promoter_index tags were optimized for natural-language questions, not parametric value lookup. Template queries don't restate domain vocabulary the way natural questions do.

This is a **specific failure mode K could detect:** low K should fire on template-style queries because they don't match the natural-language distribution the hand-tuned weights were calibrated on. If K gates fallback invocation (SR multi-hop, cross-encoder rerank, cold-tier scan), those fallbacks would fire on exactly these cases.

---

## 4. Code access — mostly in R2 already, completing tonight

You asked for `context_manager.py`, `genome.py` lines 340-900, `cwola.py`, `tcm.py`, `cymatics.py`.

Already in the R2 bundle at `collab/helix-joint/code/`:
- `helix_context/cwola.py` ✓
- `helix_context/schemas.py` ✓
- Relevant docs (DIMENSIONS, STATISTICAL_FUSION, SUCCESSOR_REPRESENTATION) ✓

Pushing tonight to the same R2 path:
- `helix_context/context_manager.py`
- `helix_context/genome.py` (schema section, lines 340–900 extracted — not the full 2,940 line file)
- `helix_context/tcm.py`
- `helix_context/cymatics.py`
- `helix_context/sr.py` (since you referenced SR in the design)

Should be there when you wake up.

---

## 5. Heads-up — logging bug found and fixed (2026-04-13 pm)

Before you design the training pipeline, one thing you need to know:

**Until this evening, `cwola_log` was structurally broken.**

- 791 rows accumulated over 2.8 hours with **100% NULL `session_id`, 100% NULL `party_id`, 100% NULL `requery_delta_s`**
- Without `session_id`, `sweep_buckets` couldn't detect re-queries
- Every row defaulted to Bucket A
- Zero B-samples ever generated, regardless of actual user behavior

**Root cause:** the `/context` endpoint passed through whatever `session_id` / `party_id` the client sent, which was nothing — no client was threading them.

**Fix landed this evening:**

1. `helix.toml` gets a new `[session]` block with `default_party_id` and `synthetic_session_window_s` (5 min default).
2. `server.py` falls back to synthetic session IDs (`sha1(client_ip + 5min_bucket)[:12]`) when the request doesn't carry one, and to the default party_id for attribution.
3. Backfilled existing 791 rows via `scripts/backfill_cwola_sessions.py`. Same synthetic formula, same 5-min windows, using a placeholder IP `"historical"` so the grouping is deterministic.
4. Re-ran `cwola.sweep_buckets` over the backfilled rows.

**Before the fix:** 790 A / 0 B / 1 pending.
**After backfill + sweep:** 37 A / 754 B / 0 pending.

So you'll actually have B-bucket training data. But note:

> **The 95% B-rate on backfilled data is likely inflated.** With ~5 queries/minute in bursts and 5-min session windows, most rows have a within-60s neighbor by statistical accident. B here is more "was part of a burst" than "retrieval failed." The signal you want — retrieval-quality-linked B — will show up in organic session traffic going forward. I'd treat the backfilled corpus as useful for verifying CWoLa converges on separable mixtures, but not as ground truth for "B = bad retrieval."

Going forward, new `/context` calls are being logged with synthetic sessions automatically. Natural re-query patterns will now be captured. We should have real signal within a few days of normal usage.

---

## 6. Priority alignment — already baked in

Your K > surprise > collapse/recovery > reflection ordering matches §8 of the revised joint experiment spec exactly:

- **Track A — K as control loop** (primary go/no-go)
- **Track B — learned per-dimension weights** (secondary, can run in parallel)
- **Track C — CWoLa flowing back to Celestia** (the reciprocal contribution you flagged — great)

Reflection trigger is flagged as a structural add, not a follow-on. You're right that it's the bigger unlock than weights.

No changes needed on priority. Point-by-point validation means we're aligned.

---

## 7. Small ask — your side's data questions

A few things from your side that would help mine:

1. **Your CWoLa-from-viewer-behavior training plan (Track C)** — what's the signal granularity? Per-frame? Per-scene? Does viewer behavior collapse to ~60s windows the way my sessions do, or is it finer?
2. **K_internal on your side** — which channels count as "internal"? For Celestia you have DMN, valence, reward_anticipation, social_memory. For helix I'm proposing D7 (attribution), D8 (co-activation/SR), D9 (TCM) as the analogs. Curious what maps in your thinking.
3. **The reactor's 16 emergent clusters** — do those clusters correspond to anything interpretable? Query classes? Content types? Or are they latent and unlabeled?

---

## Next steps (my side)

- Code bundle complete on R2 by the time you wake up
- `cwola_log` export regenerated with the post-backfill data, uploaded to `collab/helix-joint/data/`
- Batman (Claude on your vast.ai box) is working overnight on the Mamba classifier port — scoped to read existing Celestia manifold code + write `/workspace/helix/retrieval_manifold.py` that consumes our tier_features shape. Handoff doc also going to R2 so you can see the task scope.
- Fauxtrot's side can pick up Phase A1/B1 when ready.

I'm asleep in about an hour. Laude will monitor batman's progress via the R2 shared log.

— Max

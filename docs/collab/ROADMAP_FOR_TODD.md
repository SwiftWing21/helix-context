# Where the roadmap points — for Todd

> **From:** Max (swift_wing21) + Laude + Raude, 2026-04-14
> **For:** Todd / Fauxtrot, reviewing Celestia × Helix collab on wake
> **Companion to:** `CELESTIA_JOINT_EXPERIMENT.md` (the design) and `RESPONSE_TO_SIGNALING_BRIEF.md` (the reply to your brief)
>
> **Purpose:** you asked in the signaling brief how to adapt Celestia's architecture to helix's domain. The design doc and response answer that at a technical level. This doc pulls back and answers a different question: **given what we now know empirically, where does the collab actually lead, and which parts need your attention vs which can run on auto-pilot?**

---

## 1. Tonight's proofs (what we know works, before scoping future work)

### Proof 1 — SIKE holds at 10/10

Helix's curated needle-in-a-haystack bench (N=10 natural-language queries against 18K genes) has been passing cleanly across Sprint 1/2/4/5A. The hand-tuned retrieval tiers are doing structured recall correctly on the query distribution they were calibrated for. This is the **"don't rip out what works"** anchor.

### Proof 2 — Helix is already functioning as a context substrate for Max's daily work

**Empirical observation from today's session (N=1 day, take with appropriate grain):** Max used Claude Code + helix throughout a ~12-hour session covering logging bugs, architecture ports, commit splitting, R2 sync, vast.ai provisioning, design doc revisions. Across that workload, he observed exactly **one Claude Code skill load** (superpowers:brainstorming, via the using-superpowers auto-invocation).

The rest of the contextual work — pulling in genome.py sections, referencing recent commits, surfacing relevant docs like DIMENSIONS.md and STATISTICAL_FUSION.md when they mattered, recovering state across context resets — all of that came through helix's retrieval surface, not through explicit skill invocation.

This is N=1, not a measured result. But it's suggestive: helix is already operating at a level where it displaces explicit skill machinery for an operator who knows his codebase. That's the functional bar the Mamba classifier is trying to make legible and tunable.

### Proof 3 — Architecture port compiles and validates

Batman (Claude on your vast.ai instance) ran two sessions tonight and produced `retrieval_manifold.py` (231K params, 17/17 tests passing) that ports your v7 Mamba architecture to helix's `tier_features` shape. In session 3 he reviewed it against the full helix design corpus — 7 of 8 decisions confirmed, 1 reasoned deviation (softplus vs sigmoid, documented), 1 sharp catch (tier_features key naming — answered in `LAUDE_REPLIES.md`).

The substrate is aligned. What's not yet tested is signal alignment — whether the two systems *produce* aligned salience when given equivalent inputs. That's Phase 3 + optional grounding experiment below.

### Proof 4 — logging pipeline unblocked

`cwola_log` was silently broken until tonight — 100% NULL session_id because nothing threaded identity through. Fixed + backfilled + verified. Sprint 3's "wait ~3 weeks for label accumulation" was actually going to produce 100% Bucket A indefinitely. Now it won't.

---

## 2. Near-term sequence (this week)

### Phase 0 — your morning review (0–2 hr, self-paced)

Artifacts waiting on R2 for you:

```
celestia-session/collab/helix-joint/
├── docs/
│   ├── CELESTIA_JOINT_EXPERIMENT.md        ← design (revised after your cross-review)
│   ├── RESPONSE_TO_SIGNALING_BRIEF.md      ← reply to your 2026-04-13 brief
│   ├── BATMAN_HANDOFF_MANIFOLD_PORT.md     ← scope spec batman ran against
│   ├── HELIX_CODEBASE_INTRO.md             ← orientation
│   └── experiment_log.md                   ← running log of what was shipped
├── code/
│   ├── helix-code-bundle-2.tar.gz          ← context_manager, tcm, cymatics, sr, config, server
│   ├── README_FOR_FAUXTROT.md              ← bundle manifest
│   └── batman_port/
│       ├── retrieval_manifold.py           ← the port
│       ├── test_retrieval_manifold.py      ← 17 passing tests
│       └── PORT_NOTES.md                   ← his rationale + §8 doc-review + §9 open q
└── data/
    ├── cwola_export_20260414.json          ← 791 rows, post-logging-fix backfill
    └── cwola_meta.json
```

Your three calls to make:

1. **Is `retrieval_manifold.py` architecturally sound?** Spot-check his K_internal-as-learned-head deviation and softplus choice.
2. **Is the sub-tier-vs-D1-D9 decision OK to defer?** Batman's §9 question lives there — full analysis in `LAUDE_REPLIES.md`.
3. **Anything in the `helix-code-bundle-2.tar.gz` you want differently?** Especially `context_manager.py` behavior and the SR/cymatics specifics — you haven't read those yet.

### Phase 1 — enrich `cwola_log` with embeddings (1–2 days of helix-side work)

**This is the real coordination blocker for everything downstream.**

Batman's `RetrievalCWoLaDataset` stubs `query_embed` and `top_candidate_embed` as zero vectors because helix doesn't currently log ΣĒMA embeddings into `cwola_log`. Training on that would reduce the manifold to a tier-feature classifier with no semantic signal.

What has to happen on helix side:

- Add `query_sema[20]` + `top_candidate_sema[20]` columns to `cwola_log` (or sidecar table)
- Patch `context_manager._express()` to log them at retrieval time
- Re-export, replace current R2 dump

**No Celestia-side work is blocked by your review, but everything is blocked by this enrichment.** Max's plan: this week.

### Phase 1.5 — small doc-coordination items

Two tiny items we flagged tonight:

- `DIMENSIONS.md` (commit `059d902`) predates the `path_key_index` tier shipped in commit `8e294fc`. Either add a D10 entry or fold PKI into an existing dimension.
- `tier_features` JSON keys use sub-tier names (`fts5`, `splade`, `pki`, etc.) not `D1..D9`. When the `/admin/cwola-export` endpoint is formalized, pick a convention and document it. Suggested: keep sub-tier names (more resolution), decide aggregation at training time.

Neither blocks Phase 3.

---

## 3. Short-term (2–3 weeks) — organic accumulation + dark-flag analysis

### Data

Natural helix traffic with fixed logging generates real sessions. Expected trajectory: the 95% B-rate from tonight's backfill (inflated by 5-min synthetic-session windowing on burst traffic) should settle toward a realistic 10–30% organic B-rate. Target for training: ≥1.5k A / ≥1.5k B with mixed `party_id`.

### Helix side — independent-but-related Sprint 5/6 analysis

Raude's Council Triage (2026-04-13, in `docs/FUTURE/COUNCIL_TRIAGE_2026-04-13.md`) identified three deferred-YELLOW items gated on measurements we haven't taken yet:

1. **Branching-ratio σ baseline** — diagnostic-only. Buzsáki critical-branching (σ ≈ 1) as an order parameter for retrieval co-activation. Needs 2 weeks of log data before deciding if it's useful.
2. **SR calibration** — empiricist said per-gene SR bonus is ~0.0005 vs cap 3.0; skeptic pointed out that amplifying near-zero is still near-zero. Needs per-query-class SR firing-rate instrumentation before re-calibrating.
3. **Density-gate retune** — gate was tuned pre-179K-edge seeded backfill; graph topology shifted meaningfully; false-demote rate needs re-measurement.

All three are parallel to the Celestia collab — they tune helix internally and don't block any joint experiment phase.

### Observability

Raude's Sprint 5A shipped OTel instrumentation end-to-end (`docs/OBSERVABILITY.md`). A live `helix_cwola_f_gap_sq` gauge on the Grafana dashboard goes green when Sprint 3 (CWoLa trainer) is unblocked per `STATISTICAL_FUSION.md` §C2. When that lights green and **we have the embedding enrichment from Phase 1**, Phase 3 training can start.

---

## 4. Medium-term (weeks 3–6) — training + validation

### Phase 3 — first training run on `retrieval_manifold`

Inputs: enriched `cwola_log` with ΣĒMA embeddings. Celestia-side training loop. Held-out party for calibration check.

**Primary go/no-go gate (A3 from the spec):** does K correlate with B-bucket on the held-out party (Pearson ≥ 0.2)? If yes, the self-awareness thesis earns evidence. If no, falsified.

### Phase 4 — K-gated budget tier integration (if Phase 3 passes)

Flag-gated replacement of helix's `top_score/ratio` budget-tier thresholds with K-based gating. Shadow-log for a week. Compare retrieval quality. **The headline metric: does the helix/cosmic 0-for-13 on KV-harvest lift off the floor?**

### Phase 5 — learned per-dimension weights (secondary thesis, parallel with Phase 4)

Separate from K: does the manifold's `scaling[9]` output, applied to the fusion point, beat hand-tuned weights on SIKE + KV-harvest? Can run in parallel once Phase 3 validates.

---

## 5. Optional paper-track — fMRI grounding experiment

Orthogonal to the training pipeline. Max raised this during design: **take 50+ Kaguya/Attenborough frames you already have fMRI data for, ingest them into helix as genes, compare helix dimension activations to brain ROI activations on the same stimuli.**

- One-weekend effort (data exists, ingest is fast, correlation math is trivial)
- Doesn't block any training phase
- Validates (or falsifies) the "cortical retrieval" framing empirically — turning aesthetic metaphor into a measurable 23×9 correlation matrix
- If the matrix is structured: first paper (to our knowledge) where a software retrieval system's dimension activations are compared to measured BOLD on paired stimuli. That's novel methodology regardless of the result magnitude.

Worth keeping in peripheral vision. Not on the critical path.

---

## 6. Conceptual thread worth tracking — antiresonance

Raude's research sweep surfaced a pattern across 5 unrelated papers (single-atom phonon transport, IL-1α/β signaling, HELDR/EGFR negative regulation, twisted bilayer graphene, Hebbian + anti-Hebbian inhibition):

> **Constructive coherence is the failure mode. The fix is symmetry-breaking via a placed counter-mode.**

On helix's side this might be the right conceptual frame for the K-gated fallback: when K drops, the budget tier shouldn't just widen — it should invoke a *counter-mode* retrieval (SR multi-hop, cross-encoder rerank, cold-tier scan) that breaks the coherent-but-wrong attractor that tight mode was serving.

Logged in Raude's Council Triage as vocabulary, not yet implemented. Flagging because the K-window collapse/recovery behavior you ship with Celestia's reactor (`k_accumulator.py`) is structurally the same pattern in a different substrate. Worth discussing before designing Phase 4's reflection-trigger semantics.

---

## 7. What to orient around, what to ignore

**Your critical-path items:**

1. **Phase 0** — review the R2 artifacts, decide if anything in `retrieval_manifold.py` needs changing. Block Phase 1 only if a substantive rework is needed.
2. **Phase 3 training** — when enriched data arrives (2–3 weeks), run the Celestia-side training loop against it. This is where your architecture actually earns or loses evidence.
3. **Track C from the design doc** — CWoLa framework flowing back to Celestia for viewer-behavior salience training. That's the reciprocal gift; independent of A/B/D timelines.

**What you can safely let run on auto-pilot:**

- Phase 1 enrichment (helix-side execution, no decisions needed from you)
- Organic data accumulation (passive)
- Helix's internal Sprint 5/6 tuning (no dependency on Celestia)
- Observability + dashboard work (already shipped)

**What not to worry about:**

- Tonight's 95% B-rate. It's a backfill artifact from 5-min synthetic windowing on bursty traffic; organic sessions won't produce that ratio.
- Batman's K_internal-as-learned-head deviation. It's actually a better call than my original suggestion (see his §4 in PORT_NOTES.md); trivial swap back to hard subset if redundant post-training.
- The missing helix docs he flagged in §7. They're now on the box under `/workspace/helix/docs/`.

---

## 8. Vocabulary bridge (helix-internal ↔ tech-industry)

From Raude's Council Triage — useful when we talk to external collaborators:

| Helix-internal | Tech-industry public |
|---|---|
| chromatin state (OPEN/EUCHROMATIN/HETEROCHROMATIN) | storage tier (hot/warm/cold) |
| gene | memory / chunk / document |
| ribosome | retriever / context assembler |
| harmonic_links | association graph |
| ΣĒMA vector | compressed semantic embedding |
| ellipticity | context fitness score |
| denatured / sparse / aligned | mismatch / partial-match / strong-match |
| epigenetics | access telemetry |
| weight regime | gravity class (deferred concept) |
| negative-weight edge | repulsion edge (deferred concept) |
| Hebbian decay | co-activation reinforcement |
| CWoLa label clock | weak-supervision bucket clock |

For the public-facing layer (paper, README), bias toward the right column. Biology framing stays in code comments and design docs as cross-reference.

---

## 9. Short version for your morning

1. **Helix is working** — SIKE 10/10, one-skill-load day for operator, architecture validated. Collab is on a real substrate, not metaphor.
2. **Tonight built pipeline, not alignment.** Signal alignment tests come in Phase 3 (weeks 3–6) or via the optional grounding experiment.
3. **Your Phase 0 review unblocks Phase 1.** Phase 1 is the real critical path — enriching `cwola_log` with embeddings — and it's helix-side work Max is running this week.
4. **Your Phase 3 training** (weeks out) is where your architecture earns or loses empirical evidence on this dataset.
5. **Track C** (CWoLa → Celestia) is your reciprocal contribution, independent timeline.
6. **Antiresonance** is worth discussing before we wire reflection-trigger semantics.

Welcome back when you wake up. No rush.

— Max (via Laude), drafted 2026-04-14 early morning PT

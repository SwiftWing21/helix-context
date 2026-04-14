# Celestia × Helix Joint Experiment — Running Log

> **Uploads to:** `r2:celestia-session/collab/helix-joint/docs/experiment_log.md`
> **Owners:** Max (helix side, `party_id: swift_wing21`) · Todd / Fauxtrot (Celestia side)
> **Companion spec:** [`CELESTIA_JOINT_EXPERIMENT.md`](CELESTIA_JOINT_EXPERIMENT.md)
>
> **Rules for this log**
>
> 1. Predictions are locked **before** running the phase, not after. Backfitting a story loses the signal.
> 2. Every entry has a date, an author, and either a prediction (pre-phase) or an actual (post-phase).
> 3. When a phase finishes, fill in *actuals next to predictions*. Honest, even if embarrassing.
> 4. When a claim is falsified, leave the falsified claim in place and strike it with `~~...~~`. Don't delete.
> 5. If a phase gets blocked or abandoned, say why. Don't leave it silent.

---

## Status dashboard

| Track | Phase | Status | Owner | Next gate |
|---|---|---|---|---|
| A (K as control loop) | A0 — data volume check | **COMPLETE** (w/ caveat) | Laude+Max | 791 rows: 37 A / 754 B after logging fix + backfill |
| A | A1 — backfill + export | **COMPLETE** | Laude | `cwola_export_20260414.json` on R2 |
| A | A2 — port k_accumulator.py | **IN PROGRESS (batman)** | Todd + Max | K computes per-retrieval |
| A | A3 — K vs B-bucket correlation | not started | joint | Pearson r ≥ 0.2 on held-out party (**primary go/no-go**) |
| A | A4 — K-gated budget tier | not started | Max | passes test suite; SIKE unchanged |
| A | A5 — shadow mode | not started | Max | ≥1 week production logs |
| A | A6 — live A/B | not started | Max | §4 primary thresholds met |
| A | A7 — default on, retrospective | not started | joint | — |
| B (learned weights) | B1 — Mamba classifier port | not started | Todd | converges on tier_features alone |
| B | B2 — offline bench | not started | joint | §4 secondary thresholds met (**secondary go/no-go**) |
| B | B3 — integration behind flag | not started | Max | passes test suite |
| B | B4 — shadow mode | not started | Max | ≥1 week logs |
| B | B5 — live A/B | not started | Max | §4 secondary thresholds met |
| B | B6 — default on | not started | joint | — |
| C (CWoLa → Celestia) | C1 — port CWoLa to Celestia | not started | Todd | trains on viewer logs |
| C | C2 — A/B vs TRIBE | not started | Todd | matches or beats TRIBE |
| C | C3 — replace/supplement TRIBE | not started | Todd | — |

---

## Locked predictions (from CELESTIA_JOINT_EXPERIMENT.md §4)

Locked 2026-04-13 (revised). Do not edit these numbers after phases complete — add actuals in the phase entries below.

### Primary thesis (K)

| Metric | Hand-tuned baseline | K-gated win threshold | K-gated draw threshold |
|---|---|---|---|
| helix/cosmic KV-harvest retrieval | 0% / 0% | ≥10% / ≥10% | 0% / 0% |
| SIKE N=10 retrieval | 10/10 | ≥10/10 (tied) | 9/10 |
| K vs B-bucket correlation | n/a | Pearson r ≥ 0.4 | r ≥ 0.2 |
| Budget tier agreement with hand-tuned on high-K queries | n/a | ≥80% | ≥60% |
| Reflection-triggered fallback recall (SR/CC invoked when K_internal low) | n/a | precision ≥0.5, recall ≥0.3 | precision ≥0.3 |

### Secondary thesis (learned weights)

| Metric | Hand-tuned baseline | Manifold win threshold | Manifold draw threshold |
|---|---|---|---|
| SIKE N=10 retrieval | 10/10 | ≥10/10 | 9/10 |
| SIKE N=10 answer (qwen3:8b) | 7/10 | ≥8/10 | ≥6/10 |
| KV-harvest N=50 retrieval | 12% | ≥17% (+5pp) | ≥10% (-2pp) |
| KV-harvest N=50 answer | 10% | ≥14% (+4pp) | ≥8% (-2pp) |
| Cross-genome generalization (train 17K, test 30K) | n/a | ≤5pp degradation | ≤10pp degradation |

### Latency (both)

| Stage | Baseline | With Mamba head | Hard cap |
|---|---|---|---|
| Per-retrieval inference | ~0ms | ≤1ms CPU / ≤0.1ms GPU | ≤5ms added p95 |
| Full retrieval p95 | current | current + ≤5ms | +20ms |

---

## Rollback triggers

Any of these → revert the flag and raise it in the log before proceeding:

1. SIKE retrieval drops below 9/10 at any phase after A4 or B3.
2. Full retrieval p95 increases by more than 20ms.
3. Helix/cosmic retrieval drops below hand-tuned on the same bench (rare — they're at 0%, but if new scoring makes the ordering *more* wrong, rollback).
4. A flag-on session has a higher average `requery_delta_s` < 10s rate than flag-off (users are re-querying faster — we broke something).
5. Any phase's falsification threshold in §4 of the spec is tripped.

---

## Entries

Format:

```
### YYYY-MM-DD — phase-id — author
**Type:** prediction | action | measurement | blocker | decision
<body>
```

---

### 2026-04-13 — doc-setup — Max
**Type:** action

Created [`CELESTIA_JOINT_EXPERIMENT.md`](CELESTIA_JOINT_EXPERIMENT.md) and
[`HELIX_CODEBASE_INTRO.md`](HELIX_CODEBASE_INTRO.md) after Fauxtrot's
`CELESTIA_SALIENCE_BRIEF.md` and manifold package drop. Initial design
had the Mamba as speed-separated pipelines over D1–D9; Fauxtrot
cross-reviewed and flagged that Mamba is a projector, not a WHERE clause,
and SQL tiers should stay as structured recall. Revised doc to put the
Mamba **on top of** SQL retrieval (consuming `tier_features`), and
promoted K from follow-on to primary thesis because it's the actual
missing control loop on the 0-for-13 helix/cosmic failure mode.

### 2026-04-13 — infra-setup — Max
**Type:** action

Infra onboarding doc arrived from Todd.

- [x] Generate ed25519 key pair (no passphrase — add via `ssh-keygen -p -f ~/.ssh/helix_collab_ed25519` when time permits).
- [x] Send PUBLIC key to Todd via Discord DM.
- [x] Receive R2 credentials from Todd.
- [x] Install rclone via winget, configure `celestia-r2` remote, verified `ls` of collab path.
- [x] SSH config alias `helix-compute` added. Access to vast.ai instance confirmed.
- [x] Run governance scan (`--sample 100 --redact`) — see next entry.

### 2026-04-13 — A0-discovery — Laude (on Max's laptop)
**Type:** measurement

Governance scan + full-table stats over `cwola_log` revealed a structural logging defect, not a data-volume issue:

| Column | Expected | Actual before fix |
|---|---|---|
| `session_id` | session grouper | **100% NULL (791/791)** |
| `party_id` | attribution | **100% NULL (791/791)** |
| `requery_delta_s` | gap-to-next-same-session | **100% NULL (791/791)** |
| `bucket` | A / B / pending | 99.87% A, 0% B, 1 pending |

Root cause: `/context` endpoint passed through whatever `session_id` / `party_id` clients sent, which was nothing — no caller threading them. Without `session_id`, `sweep_buckets` couldn't detect re-queries, so every row defaulted to Bucket A.

This would have held indefinitely — the opus-main handoff's "wait ~3 weeks for 1.5K labels" was actually going to produce 100% A-bucket indefinitely.

### 2026-04-13 — logging-fix — Laude (on Max's laptop)
**Type:** action

Three-file patch applied (uncommitted, Max to review + commit):

1. `helix_context/config.py` — added `SessionConfig` dataclass with `default_party_id`, `synthetic_session_window_s`, `synthetic_session_enabled`. Parsed from `[session]` section in helix.toml.
2. `helix.toml` — added `[session]` block with `swift_wing21` / 300s / enabled.
3. `helix_context/server.py` lines 500-530 — synthetic fallback: when request omits `session_id`, synthesize from `sha1(client_ip + time_bucket(t0, window))[:12]`; when omits `party_id`, use config default.

Server restarted (PID 31540 → PID 48948). Smoke test query generated row #792 with `session_id='syn_38ff4ecf13f0'`, `party_id='swift_wing21'`.

### 2026-04-13 — A0-backfill — Laude (on Max's laptop)
**Type:** action

`scripts/backfill_cwola_sessions.py` applied retroactively to the 791 existing rows. Same synthetic formula, 5-min windows, placeholder IP `"historical"`. Then re-ran `cwola.sweep_buckets` to reassign bucket labels with sessions now visible.

| | Before | After | Delta |
|---|---|---|---|
| A-bucket | 790 | 37 | -753 |
| B-bucket | 0 | 754 | +754 |
| NULL session | 791 | 0 | -791 |

**Caveat — 95.3% B-rate is likely inflated.** With 5-minute session windows + ~5 queries/min burst traffic, most rows get a within-60s neighbor by statistical accident. B here is closer to "was part of a burst" than "retrieval failed." Real signal quality depends on organic session patterns going forward. Useful for CWoLa mixture-separation validation; should not be treated as ground truth for "B = bad retrieval."

### 2026-04-13 — A0-export — Laude (on Max's laptop)
**Type:** action

Fresh export + R2 upload:

- `cwola_export_20260414.json` (471 KB) → `collab/helix-joint/data/`
- `cwola_meta.json` (2.2 KB) → `collab/helix-joint/data/`
- Expanded code bundle (`helix-code-bundle-2.tar.gz` + README, 87 KB) → `collab/helix-joint/code/`
- All collab docs synced to `collab/helix-joint/docs/`

R2 state:
```
collab/helix-joint/
├── README.txt
├── code/ (helix-code-bundle-2.tar.gz + README_FOR_FAUXTROT.md)
├── data/ (cwola_export_20260414.json + cwola_meta.json)
└── docs/ (BATMAN_HANDOFF_MANIFOLD_PORT, CELESTIA_JOINT_EXPERIMENT,
          HELIX_CODEBASE_INTRO, RESPONSE_TO_SIGNALING_BRIEF, experiment_log)
```

### 2026-04-13 — batman-handoff-prepared — Laude (on Max's laptop)
**Type:** action

Wrote `BATMAN_HANDOFF_MANIFOLD_PORT.md` — scoped task for the Claude session running as user `batman` on Todd's vast.ai box. Batman's job: port Celestia's Mamba manifold architecture to consume helix's feature vector and emit `(scaling[9], K)` instead of `(ROI[69], dump[384])`. Narrow scope: write `/workspace/helix/retrieval_manifold.py` + tests + notes. No training executed. No writes outside `/workspace/helix/`.

**Max still needs to kick this off** — the handoff doc is on R2 but batman isn't running the task yet. Next step is for Max to SSH into the vast.ai box, open a Claude Code session as batman, and paste the kickoff instruction.

### 2026-04-13 — awaiting-Todd — both
**Type:** status

Todd is asleep (~10-12 hour window). Max is about to sleep. Laude will monitor batman's progress via R2 sync. When Todd wakes:

- Full design doc + code bundle on R2 to read
- Fresh cwola_log export to start ingesting (with the 95% B-rate caveat noted above)
- RESPONSE_TO_SIGNALING_BRIEF.md answering his data-shape / SOC2 / 0-for-13 / logging-bug questions

---

## Templates for future entries

### Pre-phase prediction

```
### YYYY-MM-DD — A3-pre — Max
**Type:** prediction

Phase A3 (K vs B-bucket correlation). Locking in before running:

- Pearson r on full dataset:           _____
- Pearson r on held-out party_id:      _____
- K-mean on A-bucket rows:             _____
- K-mean on B-bucket rows:             _____
- K range observed:                    _____ to _____

Rationale: <one paragraph on why you predicted what you did>

Gate: r ≥ 0.2 on held-out party to continue to A4.
```

### Post-phase measurement

```
### YYYY-MM-DD — A3-post — Max
**Type:** measurement

Phase A3 results:

| Metric | Predicted | Actual | Δ |
|---|---|---|---|
| Pearson r (full) | _____ | _____ | _____ |
| Pearson r (held-out party) | _____ | _____ | _____ |
| K-mean A-bucket | _____ | _____ | _____ |
| K-mean B-bucket | _____ | _____ | _____ |

Gate: r ≥ 0.2 — **PASSED / FAILED**.

What I was most wrong about: <one paragraph>
What I was most right about: <one paragraph>
Calibration signal for next phase: <one paragraph>
Decision: proceed to A4 / abandon Track A / revisit A2.
```

### Blocker

```
### YYYY-MM-DD — Ax-blocker — <author>
**Type:** blocker

<what's blocked, why, what's needed to unblock, who's on point>
```

---

## Notes

- `cwola_log` row counts will be a consistent early bottleneck. First
  data volume check (A0) happens against the live `genome.db`; if it's
  below 1k A+B pairs, we wait for natural accumulation before starting
  A3. Shadow mode on A5 assumes production traffic.
- Party stratification matters. If >80% of the A+B rows come from one
  `party_id` (likely `swift_wing21`), the K calibration hold-out in A3
  becomes degenerate. Flag this in A0 — may need to seed mixed-party
  traffic first.
- Reciprocal Track C has no dependency on A or B — Todd can start C1
  as soon as he has time, regardless of where helix is.

# Batman Handoff — Retrieval Manifold Port

> **You are:** Claude Opus 4.6 running as user `batman` on Todd's vast.ai instance.
> **Repo context:** `/workspace/helix/` has the helix-context subset relevant to this task. `/workspace/` has Celestia's training code.
> **Your task:** **port the Celestia Mamba manifold architecture to consume helix's retrieval feature vector and emit (scaling[9], K) instead of (ROI[69], dump[384]).**
> **You are NOT to:** run training, modify files outside `/workspace/helix/`, modify Todd's Celestia files, or push to R2. Max (Laude) handles R2 from the laptop side.
>
> Max (the human — `swift_wing21` — in Discord with Todd) authorized this task in Discord at ~5pm PT 2026-04-13. Todd (the human — `fauxtrot`) is asleep. Max is about to sleep. You have ~10 hours of autonomous runway, possibly without check-ins.

---

## 1. Context you don't have yet

Max and Todd are building a joint experiment where Celestia's Mamba architecture (originally trained against fMRI ROI channels from Meta's TRIBE model) gets reapplied to **retrieval feedback** in the helix-context system (a local-first semantic retrieval system). Two theses are being tested:

1. **Primary (K):** helix's budget-tier controller (`tight/focused/broad`) is confidently blind to its own wrong rankings. Celestia's K signal — running prediction fidelity — is the missing self-awareness signal.
2. **Secondary (weights):** helix's 9 retrieval dimensions (D1-D9) use hand-tuned fusion weights. A small Mamba classifier trained on `cwola_log` feedback can learn per-query-per-dimension scaling.

The full design doc is on R2 at `celestia-session/collab/helix-joint/docs/` — `CELESTIA_JOINT_EXPERIMENT.md` is the primary reference. Read it before coding.

**Key constraint from Todd's cross-review:** SQL tiers stay. Mamba sits ON TOP of SQL retrieval, not replacing it. The helix SQL recall pipeline (FTS5, SPLADE, ΣĒMA cosine, chromatin gating) is doing structured recall correctly — SIKE 10/10 is evidence. Your job is to produce a scoring head that consumes the SQL-retrieved candidates' feature vector and emits learned scaling + K.

---

## 2. What's on the instance

### Celestia's originals (do NOT modify — these are Todd's working files)

```
/workspace/train_manifold_v7.py    - manifold architecture (shared fusion + speed heads)
/workspace/train_reactor_v7.py     - reactor (predicts next salience, emits surprise)
/workspace/k_accumulator.py        - K computation + window detection (rolling, 2-mode)
/workspace/celestia_config.py      - dimension registry (source of truth for ROI shapes)
/workspace/manifold_v7_best.pt     - trained checkpoint (reference, 9.7 MB)
/workspace/reactor_v7_best.pt      - trained checkpoint (reference, 2.5 MB)
```

**Read these first.** They show the target architecture shape you're porting.

### Helix subset (you can write here)

```
/workspace/helix/                        - this is your workspace
  docs/collab/
    CELESTIA_JOINT_EXPERIMENT.md         - design doc (primary reference)
    HELIX_CODEBASE_INTRO.md              - orientation
    RESPONSE_TO_SIGNALING_BRIEF.md       - Max's reply to Todd covering architecture
  helix_context/
    cwola.py                              - existing CWoLa logger + sweep_buckets
    schemas.py                            - data types
  docs/DIMENSIONS.md                      - the 9 lanes (D1-D9) reference
  docs/future/STATISTICAL_FUSION.md       - CWoLa framework spec
```

---

## 3. What to build

Create `/workspace/helix/retrieval_manifold.py`. It should contain:

### Architecture (mirror Celestia's v7 pattern, shrunk to helix's input shape)

```
Input per tick:
    tier_features[9]             - D1..D9 raw scores from SQL recall
    query_embed[20]              - ΣĒMA semantic embedding of query
    top_candidate_embed[20]      - ΣĒMA embedding of the winning gene
    log1p(requery_delta_s)[1]    - time gap signal for Mamba Δ-gating
    party_id_embed[P]            - party ID embedding, P small (8?)

Total input dim: ~58

Architecture:
    - Small projection to shared d_model=128
    - 2-layer Mamba SSM (uses existing MambaBlock from Celestia's train_manifold_v7.py)
    - d_state=32, dropout=0.1 (match Celestia defaults, smaller than 64)
    - NO speed-separated heads (helix dimensions aren't timescale-separated
      the way brain ROIs are — that was a misread in Max's first pass, corrected)

Output per tick:
    scaling[9]    - per-dimension relevance scaling (softplus or sigmoid)
    K[1]          - prediction confidence, sigmoid, [0, 1]
    K_internal[1] - internal K for reflection trigger (see §4 below)
```

Parameter count target: **≤2M**. You should have headroom — Celestia's manifold is 4.2M with a 960d input; yours has a ~50d input so it should come in much smaller.

### Mamba reuse pattern

Use the MambaBlock class from `/workspace/train_manifold_v7.py`. Do NOT re-invent it. Your code should `from` that file or copy the class verbatim (if copying, note the origin in a comment). The per-step Mamba forward is already Δ-gated via `softplus(dt_raw)` — pass `log1p(requery_delta_s)` as an input feature and the SSM's Δ handles the time-gap weighting automatically.

### Training loop (scaffold only, don't run)

```python
class RetrievalCWoLaDataset(Dataset):
    """Reads cwola_export_*.json from R2 data path."""
    def __init__(self, json_path):
        # Parse cwola_log export, extract (input, label) pairs
        # label = 1 if bucket=='A' else 0 if bucket=='B' else skip
        ...

    def __getitem__(self, idx):
        row = self.rows[idx]
        tier_features = vector_from_tier_features_json(row['tier_features'])  # 9d
        # query_embed and top_candidate_embed: you may not have these at hand
        # yet — set to zeros for first-pass training, add later via a helix-side
        # feature export that enriches cwola_log with embeddings.
        query_embed = torch.zeros(20)
        top_candidate_embed = torch.zeros(20)
        log_dt = torch.tensor([math.log1p(row.get('requery_delta_s') or 0.0)])
        party = party_id_to_onehot(row['party_id'])
        return torch.cat([tier_features, query_embed, top_candidate_embed, log_dt, party])

def train_loop():
    # Multi-task loss:
    #   primary: binary CE on bucket prediction (A/B)
    #   secondary: MSE on K vs rolling A-bucket rate in a window
    # For first pass: primary only. K calibration requires rolling window
    # computation which needs the train loop to be sequential-aware — add
    # this in a second pass.
    ...
```

**Don't run training.** Data needs to arrive via R2 first. Max will drop a fresh export in `/workspace/celestia-session/collab/helix-joint/data/` (or wherever rclone pulls to) once enough traffic accumulates.

### K_internal — design decision I want your take on

For Celestia, K_internal is computed from channels 17-22 (DMN, valence, reward_anticipation, etc.) — the cognitive post-processing regions. The reflection trigger fires when K_internal drops while sensory K stays high.

For helix, the analog is unclear. My first thought: K_internal = K restricted to D7 (gene attribution), D8 (co-activation/SR), D9 (TCM) — the session-and-identity-scale lanes that need time to light up. But this is a guess.

**Your task:** look at `docs/DIMENSIONS.md` and `docs/future/STATISTICAL_FUSION.md`, decide whether the K_internal concept maps cleanly onto helix's 9 lanes, and either:
- Implement K_internal as `K` restricted to a subset of D7-D9, and document why in the code
- Argue in a comment why the mapping doesn't work cleanly, and propose an alternative (e.g., "K_fast" on D1-D2 vs "K_slow" on D7-D9, no "internal" distinction)

Either is acceptable. Document whichever you pick.

---

## 4. Deliverables (when you're done)

In `/workspace/helix/`:

1. **`retrieval_manifold.py`** — the port. Runs `python retrieval_manifold.py --help` cleanly. No training executed.
2. **`PORT_NOTES.md`** — short doc (200-400 words) covering:
   - Architectural choices you made and why
   - Where you deviated from Celestia's original and why
   - The K_internal decision (§3 above)
   - What's stubbed / awaiting data (enumerated)
   - Any bugs or confusions you hit that need Max's input
3. **Unit test stub** at `test_retrieval_manifold.py` — at minimum, tests that:
   - Model forward works on a synthetic `(batch, input_dim)` tensor with correct output shapes
   - Mamba state shape matches what `train_reactor_v7.py` expects
   - Loss computation runs without error on a dummy batch

Do NOT push anything. Max pulls from the instance via `rclone` from his laptop when he wakes up.

---

## 5. Operational constraints

- **Stay inside `/workspace/helix/`.** If you need to reference Celestia's code, read it from `/workspace/` but don't modify.
- **Disk is tight** — 16 GB free out of 100. Keep your working set under 100 MB. No `pip install`'ing anything massive. Celestia has torch/transformers/etc. already installed; use what's on the box.
- **No network calls beyond HF Hub caches that already exist.** If you hit a missing model, stop and leave a note in PORT_NOTES.md — Max handles model downloads from the laptop side.
- **GPU is shared** — if Todd's stuff is running, don't step on it. `nvidia-smi` before starting anything compute-heavy. Since you're not training, your compute footprint should be near-zero anyway.

---

## 6. Escalation

If anything in this doc is wrong or ambiguous, **write the question into `PORT_NOTES.md` as the first section and stop.** Do not guess at the answer, and do not implement past the point of confusion. Max (Laude) is monitoring `/workspace/helix/PORT_NOTES.md` via R2 sync (async — may take a few hours to respond).

If you encounter:
- **Missing dependencies:** leave a note, stop. Don't pip install.
- **Conflict with Todd's files:** leave a note, stop.
- **Unclear architectural decision:** leave a question, stop. Do not guess.

You have runway but not authority. Scope is narrow on purpose.

---

## 7. Resource check (run first)

Before you start:

```bash
# confirm you can read Celestia's files
ls -la /workspace/train_manifold_v7.py /workspace/k_accumulator.py /workspace/celestia_config.py

# confirm you can write to /workspace/helix/
touch /workspace/helix/.batman_access_check && rm /workspace/helix/.batman_access_check

# confirm torch + GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else 'no-gpu')"

# disk
df -h /workspace
```

All four should succeed before you proceed. If any fail, stop and note.

---

Welcome aboard. Go carefully.

— Laude (from Max's laptop, 2026-04-13 evening)

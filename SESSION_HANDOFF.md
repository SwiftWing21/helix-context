# Session Handoff — 2026-04-13

> **Architecture note.** With Sprints 1-4 in, the entire data pipeline
> (ingest → tag → encode → 12 tones + cymatics + TCM + SR + Hebbian →
> splice) is LLM-free CPU math. The only LLM call is at
> `/v1/chat/completions`. See [`docs/PIPELINE_LANES.md`](docs/PIPELINE_LANES.md)
> §"LLM boundary" for the full statement. Step 0 query-intent expansion
> is flag-gated via `[ribosome] query_expansion_enabled` (default `true`
> for backward compat; flip to `false` for strict LLM-free `/context`).

## What Landed Today

Three sprints from `docs/FUTURE/IMPLEMENTATION_ROADMAP.md` shipped in one
session (~1200 LOC, 169 module tests pass). Sprint 3 is still gated on
CWoLa label accumulation.

| Commit | Sprint | One-line summary |
|---|---|---|
| `f4dcdcc` | Sprint 1 | W1 cymatics + CWoLa logger + Howard 2005 TCM velocity + TCM ρ Gram-Schmidt fix |
| `c9367f8` | Sprint 2 | SR Tier 5.5 + theta fore/aft ray_trace bias (both dark) |
| `5184ea8` | Sprint 4 | Seeded co-activation edges with Laplace-smoothed Hebbian decay + dense-rank miss weighting (dark) |

See each commit message for the detailed change list. Design docs under
`docs/FUTURE/` describe the math; the commit messages describe what
actually shipped.

## Current Flag State (`helix.toml`)

| Flag | Default | Status | Notes |
|---|---|---|---|
| `cymatics.distance_metric` | `"cosine"` | live, opt-in | Flip to `"w1"` to A/B Werman 1986 W1 distance |
| CWoLa logger | always on | live | Writing to `cwola_log` table since commit |
| TCM velocity input | always on | live | Howard 2005 Eq. 16 |
| TCM ρ Gram-Schmidt | always on | live | Logs warning on drift instead of silent mask |
| `retrieval.sr_enabled` | `false` | dark | Tier 5.5 SR boost (γ=0.85, k=4, w=1.5, cap=3.0) |
| `retrieval.ray_trace_theta` | `false` | dark | Theta alternation — requires TCM session depth ≥ 2 |
| `retrieval.seeded_edges_enabled` | `false` | dark | Hebbian accrual on `harmonic_links` with provenance tagging |

Everything "dark" is safe to flip on any single session for A/B. The
feature paths fall back silently if prerequisite data isn't there (e.g.
empty ΣĒMA cache → theta falls back to uniform sampling).

## What's Pending

### Sprint 3 — CWoLa trainer + Stacked PLR fusion (blocked)
Gated on the CWoLa label clock that started ticking with `f4dcdcc`.
Baseline throughput expectation: ~3 weeks to N ≥ 1.5K rows/bucket.
Promotion gate is AUC > 0.55 per `docs/FUTURE/STATISTICAL_FUSION.md` §C2.
Check label accumulation with:

```sql
SELECT bucket_id, COUNT(*) FROM cwola_log GROUP BY bucket_id;
```

If rows aren't accumulating the way the design expected, that's the
first thing to investigate — the trainer can't ship without them.

### Sprint 5 — Kalman session tracking (optional, unscheduled)
Only pull this in if Sprint 2 + 4 benches show SR + velocity + seeded
edges are insufficient. No one is assigned.

## Where to Pick Up Next Session

The useful work right now is **empirical, not new code**:

1. **A/B the dark flags** against the existing benches:
   - `benchmarks/bench_dimensional_lock.py` — measures tier activation
     on synthetic dimensional probes. Primary target for SR + seeded
     edges; variant 2-3 for W1 cymatics.
   - `benchmarks/bench_skill_activation.py` — measures tier lighting
     on realistic query shapes. Primary target for theta ray_trace and
     TCM velocity; already surfaces the "TCM empty on natural sentence"
     pathology that velocity input is supposed to fix.
2. **Watch CWoLa labels accumulate** — cheap, no code. Just a SQL poll
   every few days. When N crosses 1.5K/bucket, Sprint 3 unblocks.
3. **Look at `harmonic_links` provenance distribution** once
   `seeded_edges_enabled` has been true for a week of real use. The
   design expects seeded (0.3×) to dominate initially, with
   co_retrieved (0.7×) climbing as real co-expression accrues.

## Files Touched Today (reference)

- `helix_context/cymatics.py`, `cwola.py`, `tcm.py`, `sr.py`,
  `ray_trace.py`, `seeded_edges.py`
- `helix_context/genome.py` (schema: `cwola_log`, `harmonic_links`
  provenance columns)
- `helix_context/context_manager.py`, `config.py`, `server.py`
- `helix.toml` — new `[retrieval]` flags
- `tests/test_cwola.py`, `test_cymatics_flux.py`, `test_tcm.py`,
  `test_sr.py`, `test_ray_trace_theta.py`, `test_seeded_edges.py`

## Git State

- Branch: `main`
- Commits: `f4dcdcc`, `c9367f8`, `5184ea8` — pushed
- No uncommitted changes in tracked code (docs on this branch only)
- Pre-existing failures unchanged: 1 registry citation test, 3 live-LLM
  stress flakes. Unrelated to this session's work.

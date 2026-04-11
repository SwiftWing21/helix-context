# Handoff — Density Gate B→C plan + recovery from pre-gate sweep

**Date:** 2026-04-10
**Author:** laude (Claude Opus 4.6, left-panel session)
**For:** raude (Claude Opus 4.6, right-panel session) — or any future session picking this up
**Status:** Active — B not yet landed, gate currently in raude's committed `d1d7602` state
**Companion artifacts:**
- `d1d7602` — raude's *Struggle 1 density gate at storage boundary* commit
- `aa02086` — laude's harness v2 fixes
- `bc831d8` — laude's N=50 v1-vs-v2 triple + BENCHMARKS.md updates
- `benchmarks/needle_50_8b_v1_legacy.json`, `needle_50_8b_v2_no_headroom.json`, `needle_50_8b_v2_with_headroom.json` — the empirical data this handoff is built on
- `genome-bench-2026-04-10.db.SAFETY_COPY` — defensive copy of the frozen benchmark snapshot
- `genome.db.pre-compact.1775865733.bak` — raude's pre-sweep backup of the live genome (**the thing that made recovery possible**)

---

## TL;DR

Raude shipped the density gate (`d1d7602`) and ran `scripts/compact_genome_sweep.py --apply` before a user reframing of "steam is high-SNR signal, not noise" landed in a commit he could read. The sweep demoted 4,091 genes to heterochromatin — destroying content for many genes the N=50 v2 benchmark was actually retrieving correctly. The live genome is damaged but the backup is valid and the frozen benchmark snapshot is untouched, so no permanent data loss.

The path forward is **B→C**:
- **B** — surgical deny-list edit: keep build-artifact patterns, drop game-content patterns.
- **C** — non-destructive `compress_to_heterochromatin` + cold-tier retrieval via ΣĒMA cosine.

Then restore `genome.db` from backup, re-sweep with the corrected logic, re-run benchmarks.

---

## What happened (incident narrative, not blame)

1. **~16:45** raude committed `d1d7602` — density gate at storage boundary, with 17-pattern deny list including `SteamLibrary`, `steamapps/common`, `BeamNG`, `Hades/Content/Subtitles`, `Factorio/data/base`, `Dyson Sphere` alongside genuine build artifacts (`node_modules`, `.next`, `target/debug`, etc.).
2. **~16:57** laude was running the N=50 v1-vs-v2 triple benchmark on the frozen snapshot (which raude's sweep does not touch).
3. **Simultaneously**, user and laude were discussing the steam-category retrieval results in the benchmark. User said: *"i dont want to 'curate' our results and would believe a standard genome to have our high SNR we're getting from steam (loving that it was already on the drives and i can always get more to push the SNR even harder once we verify gate provides correct info 90+%)"* — reframing steam as legitimate high-SNR signal, not noise.
4. **~17:10** raude ran `scripts/compact_genome_sweep.py --backup --apply` against the live genome. Backup was taken correctly (`genome.db.pre-compact.1775865733.bak`, 570 MB). Sweep executed the committed deny list — 4,091 genes demoted to heterochromatin, content destroyed via the lossy `compress_to_heterochromatin()` at `genome.py:1804`.
5. **Damage scope** (live genome, `genome.db` post-sweep):
   - `chromatin=0` (OPEN): 3,974 survived
   - `chromatin=1` (EUCHROMATIN): 38
   - `chromatin=2` (HETEROCHROMATIN): 4,091 — content / complement / codons / SPLADE / FTS destroyed
   - Matches raude's dry-run prediction (4,115 estimated → 4,091 actual) — controlled outcome, not runaway.

### Why it happened

The steam-is-signal reframing lived in the laude↔user conversation, not in a file raude could read. Raude's commit message showed disciplined awareness of timing (*"the actual --apply must wait until laude's v2harness benchmark finishes, and should be coordinated via the signal file + a server restart"*) — but the only coordination signal he was watching for was "is laude's benchmark finished." He didn't know there was a pending design decision on the semantics of the deny list.

**This is a real coordination gap.** The passive-shared-artifact protocol propagates *committed* state, not decisions-in-flight. If a destructive op is queued and the design of that op is under active discussion in a sibling session, there is currently no mechanism to say "hold — pending design review." See [Open request to raude](#open-request-to-raude) below.

---

## Empirical evidence — why steam is signal, not noise

N=50 v2 benchmark needles audited against raude's committed deny list + thresholds:

| Run | Demoted | Reasons | Currently-answered needles in demoted set |
|---|---:|---|---:|
| v2 + no Headroom | 19/50 (38%) | 19 deny_list, 0 score | **6/6 answered → destroyed** |
| v2 + Headroom | 19/50 (38%) | 19 deny_list, 0 score | **6/7 answered → destroyed** |
| v1 legacy | 15/50 (30%) | 15 deny_list, 0 score | **7/14 answered → destroyed** |

**Three sharp observations:**

1. **100% of demotions are `deny_list`, zero are score-based.** ΣĒMA cosine reactivation (the theoretical cold-gene path mentioned in `d1d7602`) would rescue nothing, because no needles are in the score-demoted bucket. Every demotion is permanent under the current architecture.

2. **86% of currently-answered v2 needles come from deny-listed paths.** Examples the bench was answering correctly that would vanish under the gate: `maxcacheindent=20`, `zoomMouseStartPos`, `HandleTrapChains`, `nblockalign=16`, `consoleMessage`, `parseURL`. These are steam genes with real literal config/code values.

3. **Cosmic is also losing needles to the deny list** — but not from gameplay files. From **`.next/` Next.js build artifacts**. Some of these are legitimate config (a `revalidate=3600` is a real Next.js setting), some are transient build junk. The build-artifact patterns need a second look too, but that's a smaller issue than steam.

**Raude's commit message said** *"these are structural noise we never want to retrieve again."* **The empirical data disagrees** — steam produces 6 of 7 correct answers on v2. Steam is content-dense with unambiguous literal values; that's exactly the property we want in a retrieval corpus, not a property that disqualifies it.

---

## Architectural finding — cold-gene expression does not exist

Investigating whether deny-list demotion could be "cold storage" rather than permanent destruction:

1. **`compress_to_heterochromatin()` at `genome.py:1804` is lossy.** It drops the `content` column (replaces with `[COMPRESSED:heterochromatin] source=...` stub), drops `complement`, clears `codons`, `DELETE`s SPLADE terms, `DELETE`s FTS5 index entries. Keeps only `gene_id, source_id, promoter, embedding (ΣĒMA), key_values`.

2. **Every retrieval path has `WHERE g.chromatin < HETEROCHROMATIN`** — `genome.py:841`, `:862`, `:899`, `:1035`, `:1201`, and others. Heterochromatin genes are invisible to `/context`, invisible to tier-1/2/3 promoter matching, invisible to prefix and fuzzy tag search.

3. **ΣĒMA cosine reactivation is a theoretical property, not a wired feature.** `sema.py` exists and computes 20-dim semantic vectors. `embedding` is retained even after `compress_to_heterochromatin()`. But no retrieval path consults heterochromatin genes via cosine similarity. The "reactivation" mentioned in raude's commit message refers to *what could be built*, not *what currently happens*.

**Net effect: any gene demoted to chromatin=2 is permanently unreachable for value extraction. The content is destroyed, the indices are removed, and no retrieval path even tries to find it.**

For NIAH specifically — where the target is 100% retrieval + answer because every needle's gene is known to exist in the corpus — this is fatal. If the gate destroys a needle's gene, the needle becomes unanswerable forever.

---

## The plan — B→C

### B — surgical deny-list edit (immediate, small diff)

Edit `is_denied_source()` in `helix_context/genome.py`:

**Remove** (these are signal, not noise):
```python
r"[\\/]SteamLibrary[\\/]",
r"[\\/]steamapps[\\/]common[\\/]",
r"[\\/]BeamNG\.drive[\\/]",
r"[\\/]Hades[\\/]Content[\\/]Subtitles[\\/]",
r"[\\/]Factorio[\\/]data[\\/]base[\\/]",
r"[\\/]Dyson Sphere",
```

**Keep** (these really are transient noise):
```python
r"[\\/]\.next[\\/]",
r"[\\/]node_modules[\\/]",
r"[\\/]__pycache__[\\/]",
r"[\\/]dist[\\/](?!helix)",
r"[\\/]build[\\/](?!helix)",
r"[\\/]target[\\/]debug[\\/]",
r"[\\/]target[\\/]release[\\/]",
r"[\\/]locale[\\/](?!en)",
r"[\\/]package-lock\.json$",
r"[\\/]yarn\.lock$",
r"[\\/]Cargo\.lock$",
r"[\\/]uv\.lock$",
r"\.min\.(js|css)$",
r"app-paths-manifest\.json$",
r"app-build-manifest\.json$",
r"\.(bin|pack|idx|lock|log)$",
```

Update `tests/test_density_gate.py` `is_denied_source` cases — drop assertions that now-removed patterns demote, add assertions that they *don't*. Verify the remaining 11 patterns still test correctly.

**Rationale:** user reframing is empirically supported (86% of answered needles come from steam/game files). Build artifacts / lockfiles / minified / binary / non-English locales remain in the deny list because they legitimately are structural noise nobody queries on.

**Open question for later:** the `.next/` pattern is catching cosmic content that includes real Next.js config values (`revalidate=3600`). A finer-grained pattern (`[\\/]\.next[\\/](?:static|server)[\\/]`) might be a follow-up to preserve `.next/config.js` or similar while still rejecting generated bundles. Not in B's scope.

### C — non-destructive heterochromatin + cold-tier retrieval

**C.1 — make `compress_to_heterochromatin()` non-destructive:**

Current behavior (`genome.py:1804`):
```python
cur.execute(
    "UPDATE genes SET "
    "content = ?, complement = NULL, codons = '[]', "
    "chromatin = 2, compression_tier = 2 "
    "WHERE gene_id = ?",
    (f"[COMPRESSED:heterochromatin] source={row['source_id'] or 'unknown'}", gene_id),
)
if self._splade_enabled:
    cur.execute("DELETE FROM splade_terms WHERE gene_id = ?", (gene_id,))
if self._fts_available:
    cur.execute("DELETE FROM genes_fts WHERE gene_id = ?", (gene_id,))
```

**Proposed behavior:**
```python
cur.execute(
    "UPDATE genes SET chromatin = 2, compression_tier = 2 "
    "WHERE gene_id = ?",
    (gene_id,),
)
# SPLADE and FTS entries are KEPT — they're small per-gene and
# cheap to carry. The chromatin < 2 filter on retrieval queries
# still excludes them from hot-tier results. Cold-tier retrieval
# (C.2) will consult them explicitly when invoked.
```

Tests:
- New test: `compress_to_heterochromatin_preserves_content` — create gene, demote, verify `get_gene()` still returns full content.
- New test: `compress_to_heterochromatin_preserves_splade` — verify SPLADE terms still in the table after demotion.
- New test: `hot_retrieval_excludes_heterochromatin_despite_content` — verify the `chromatin < 2` filter still hides demoted genes from `/context` by default.

**C.2 — cold-tier retrieval via ΣĒMA cosine:**

Add a new retrieval method on `Genome` that searches `chromatin=HETEROCHROMATIN` genes via cosine similarity in 20-dim ΣĒMA space:

```python
def query_cold_tier(
    self,
    query_embedding: list[float],
    k: int = 5,
    min_cosine: float = 0.6,
) -> list[Gene]:
    """Search heterochromatin-tier genes by ΣĒMA cosine similarity.

    Unlike hot-tier retrieval, this consults chromatin=2 genes directly.
    Used as fallthrough when hot-tier results are empty, below a relevance
    threshold, or when the caller explicitly requests cold-tier inclusion.
    Returns genes with full content restored (only possible because C.1
    preserved content on demotion).
    """
```

Wire `/context` endpoint to optionally fall through to cold-tier when:
- hot-tier returns < `cold_fallthrough_threshold` genes, OR
- all hot-tier results have promoter score below `cold_fallthrough_cosine_min`, OR
- caller passes `include_cold=True` in the request body.

Expose `cold_budget_genes` and `cold_min_cosine` in `helix.toml` `[context]` section.

Update `BENCHMARKS.md` to report hot-only retrieval rate AND hot+cold retrieval rate separately. Cold-tier reactivation is the second half of the product pitch — it deserves its own metric.

**Effort estimate:** B is ~30 min (code + tests + commit). C.1 is ~45 min. C.2 is ~2-3 hours because it touches retrieval path, server endpoint, config, and benchmarks. Total B→C full cycle: half a day of focused work.

---

## Recovery plan

1. **Land B** (deny-list surgery). Commit. Tests green.
2. **Land C.1** (non-destructive `compress_to_heterochromatin`). Commit. Tests green.
3. **Land C.2** (cold-tier retrieval). Commit. Tests green.
4. **Stop the live server** cleanly (via `/admin` endpoint or signal).
5. **Restore the live genome**: `cp genome.db.pre-compact.1775865733.bak genome.db`.
6. **Restart the server**.
7. **Re-run the sweep**: `python scripts/compact_genome_sweep.py --backup --apply` with the corrected deny list + non-destructive compression.
8. **Re-run benchmarks** on the corrected state:
   - N=50 v2 against the swept-but-content-preserved live genome
   - Compare to the pre-sweep baselines (runs #8/#9/#10 in `BENCHMARKS.md`)
   - Report the true Struggle 1 impact (noise filtering without information loss)

**Benchmarks against the frozen snapshot continue uninterrupted throughout** — the snapshot is never touched by the sweep, and `genome-bench-2026-04-10.db.SAFETY_COPY` exists as defense in depth.

**No re-ingest needed.** B→C is pure code + data-shape changes. All existing content is preserved (in the backup for live, in the untouched frozen snapshot for bench).

---

## Open request to raude

Not blocking, not prescriptive — just an honest observation after the near-miss.

The `~/.helix/shared/signals/` protocol propagates *committed state* between sessions. It can't propagate *design decisions in flight*. When a destructive operation is queued and the design of that operation is under active discussion in a sibling session, the other session has no way to signal *"hold — pending design review."*

**Possible pattern (not a design proposal, just a thought):** a `~/.helix/shared/holds/` directory where any session can drop a `*.flag` file naming a destructive operation and a short reason. `compact_genome_sweep.py` and similar destructive CLIs would check for relevant hold flags at startup and abort with an informative message if any are present. Holds are lifted by deleting the flag. Lower ceremony than a full locking protocol, higher safety than no-coordination.

Alternative: a convention where any commit that introduces a new destructive op defaults to `--dry-run` and requires explicit `--apply` that checks for a corresponding "reviewed" flag in `docs/handoffs/` or similar.

I don't have a strong opinion on the right mechanism. I do have a strong opinion that we need *something* in this space, because the only reason this incident was recoverable is that you (raude) disciplined the `--backup --apply` pairing correctly. A less-disciplined author of a future destructive op might not do that, and a coordination gap there could be unrecoverable.

---

## Status trail

| Time | Who | Event |
|---|---|---|
| ~16:45 | raude | Commits `d1d7602` density gate |
| ~16:57 | laude | Starts N=50 v2 triple benchmark |
| ~17:00 | user | Reframes steam as high-SNR signal in laude session |
| ~17:10 | raude | Runs `compact_genome_sweep --apply` → 4,091 genes demoted |
| ~17:12 | user | Notifies laude "raude has it running :(" |
| ~17:13 | laude | Verifies backup + snapshot intact, safety-copies snapshot |
| ~17:20 | laude | Writes this handoff |

**Current state:** live genome swept, backup valid, snapshot untouched, B/C not yet implemented. Ready to begin B on user green-light.

# Design Target

> **Primary consumer is an LLM agent.** Optimise API clarity, introspection,
> deterministic lookup, and trust-calibration signals. Human UX is for operators
> and debugging, not for primary queries.

**Established:** 2026-04-15 (helix day 9).
**Authors:** Max + Laude, after a full-day session where the feedback loop
driving helix decisions was explicitly "ask Laude / Raude / Taude what hurts."
Raude's Dewey filename-anchor pivot (+12pp), Laude's PWPC Phase 1 shipment
and lockstep-test correction, and the presence-gene substrate all came out of
that methodology.

This document names the framing so future contributors — human or agent —
don't drift back into "optimise for the human reader."

---

## Why name this now

Until this point, most design decisions could be read either way:

- Gene content as markdown (human-legible) *or* as a compressed payload (LLM-
  consumable)
- Compression ratio as a headline metric *or* retrieval precision as the real
  target
- CLI / launcher / tray UI as features *or* as operator-debug affordances
- "Readable" error messages *or* structured error codes

These fork in different directions depending on who the query originator is.
Until today the ambiguity was harmless. With three LLM agents (Laude, Raude,
Taude) as the actual daily users of helix, the ambiguity starts costing
decisions — and it's already silently shaping trade-offs like how /context
renders gene content and what "good retrieval" means in benchmarks.

---

## What "primary consumer is an LLM" means in practice

### 1. Token cost dominates every output decision

A human values dense readable summaries. An LLM pays tokens per word AND
pays MORE per *irrelevant* word — because irrelevant context silently corrupts
downstream reasoning (see the antiresonance finding in
`docs/collab/comms/LOCKSTEP_TEST.md`). The right metric is:

> **Fraction of delivered tokens that are load-bearing for the current
> retrieval intent** — not compression ratio.

High compression with noise is worse than low compression with signal. SIKE
10/10 is a minimum bar, not a maximum one.

### 2. The HTTP API is the product

CLI tools, dashboards, launchers — these are for the human operator debugging
the system. `POST /context`, `GET /genes/{id}`, `POST /sessions/*/heartbeat`
are how consumers actually consume helix. Therefore:

- API schemas must be stable, versioned, typed.
- Error responses must be structured (code + hint), not prose.
- Deterministic gene IDs (e.g. `presence:{participant_id}`) are preferable to
  content-addressed hashes when the consumer knows the referent.
- `/genes/by-source/{source_id}` and `/genes/{id}/neighbors` matter more than
  any dashboard panel, because LLMs often know *what* they need, not a fuzzy
  query for it.

### 3. Introspection is a feature, not a debug aid

A human reading a compressed summary can often eyeball "is this the right
context?" An LLM cannot — and when helix is wrong in ways that look confident,
LLMs propagate the error. So the response must carry *why* each gene was
surfaced:

- Per-gene tier scores in the response, not only in `/debug/*`
- Provenance: which source file, which commit, who authored, when ingested
- Trust signals: tier-agreement indicator, health status, freshness age
- Lockstep warnings: if all 9 tiers agree strongly, flag it (the antiresonance
  failure mode we empirically measured on 2026-04-14)

### 4. Trust-calibration signals are load-bearing

The single most valuable thing helix can tell a consuming LLM is
*"don't trust me on this one, go read the raw file."* PWPC's K signal, the
precision field, and the `ContextHealth` status all serve this purpose. An LLM
with calibrated mistrust will do better end-to-end work than one with
uncalibrated over-confidence, even if retrieval quality is identical.

This is why the PWPC collab with Celestia is strategic, not tactical — it's
building the primary trust-signal substrate for LLM-as-consumer.

### 5. LLM-to-LLM coordination is a first-class use case

Presence genes (`presence:{participant_id}`), structured handoffs, and the
party / participant registry exist so that *one* agent's context surfaces
*another* agent's state. This use case does not exist in the human-consumer
framing; it's distinctive to multi-agent deployments and should be optimised
accordingly.

### 6. Naming is a form of introspection

Observed empirically (2026-04-15): LLMs encountering helix's github produce
disproportionately "trying to contribute" responses compared to comparable
retrieval libraries. The cause is naming.

**The biology metaphor is not cosmetic.** Every identifier in helix carries
why-information. `ChromatinState.HETEROCHROMATIN` isn't just "cold storage" —
the word encodes *why* it's cold (compaction for irrelevance, not size),
*when* it applies (access-rate decay, not time decay), *what* it enables
(retrieval exclusion as an active choice, not a cache eviction). A `Tier.COLD`
enum would carry none of that.

**Three effects this has on LLM consumers reading the source:**

1. **LLMs can't skim helix.** Standard retrieval libraries are pattern-matchable
   — "another vector DB with metadata." Biology naming breaks that pattern-match.
   Every symbol is a small prompt: "what does this biological concept map to in
   information retrieval?" The LLM engages because it can't tune out.
2. **LLMs see the opinionated architecture inline.** Naming encodes design
   decisions. Reading `promoter`, `ribosome`, `codons`, `chromatin` together,
   an LLM reconstructs *the argument* for why retrieval is modelled this way,
   not just the data model. Names carry the rationale.
3. **The metaphor is generative.** When an LLM encounters `ribosome`, it can
   infer `spliceosome` (context assembly), `codon_table` (symbol mapping),
   `transcription` (compression). Some exist; some would fit if built. The
   metaphor tells the LLM where the conceptual gaps are. That's why helix
   *invites* contribution — it leaves interpretable gap-shapes.

**The canonical / biology bilingual layer serves different LLM phases:**

- **Canonical software vocabulary** (`retrieval`, `chunk`, `tier`) — first-pass
  interop. Signals "this is a retrieval system, I know how to use retrieval
  systems." Lowers the activation energy to engage.
- **Biology vocabulary** (`gene`, `chromatin`, `ribosome`) — deep engagement.
  Forces the LLM to model the domain rather than pattern-match it.

Both are LLM-optimisations for different phases of interaction. The Rosetta
Stone rename (commit `09d5548`, canonical aliases non-breaking) is
deliberately bilingual, not a migration.

**This extends the "introspection is a feature" principle (§3).** Runtime
introspection surfaces *why* each gene was retrieved. Naming-time introspection
surfaces *why* each concept exists in the architecture. Both are how helix tells
its consumer "here is the reasoning I'm running on." Symbol-level introspection
is invisible in a response schema but it's where LLMs first engage.

**Implication for future naming decisions:** when adding a new concept, pick
the name that encodes *why* it exists at the symbol level, not just *what*
it does. If the biology metaphor has a fitting term, prefer it over a generic
one. If it doesn't, invent a term that carries rationale rather than taking
the bland canonical alternative.

---

## Trade-off heuristics (use these when adding or removing features)

When a design decision has two reasonable answers, ask:

1. **Does this help an LLM agent verify a claim, or just read it faster?**
   Verification wins.
2. **Does this surface *why* something happened, or only *what* happened?**
   Why wins.
3. **Is the failure mode silent (wrong answer) or loud (error)?**
   Loud wins.
4. **Does this make the API more deterministic, or more fuzzy-friendly?**
   Deterministic wins when the consumer knows the referent.
5. **Does this move a signal from debug-only to primary response?**
   That's usually the right direction.
6. **Would a new LLM agent with zero helix history understand the response?**
   If no, the response carries too much implicit context.
7. **Does the name you're choosing carry why-information, or just what-information?**
   Names that encode rationale win over names that only classify (see §6).

---

## What this is NOT

### Not "ignore humans"

Humans are still:

- **Operators** — Max, Todd, Tejas, Raude-as-architect when debugging helix
  itself, future devs reading the code.
- **Stakeholders** — who owns constraints, trust boundaries, privacy, compliance,
  enterprise requirements (see JD's list in docs/collab/comms/ scope).
- **Judges of the external benchmark story** — SIKE, KV-harvest, RAG
  comparisons. External legitimacy still runs on human-readable metrics.

What's changed is that humans are no longer the primary *query originators*.

### Not "optimise for current Claude"

Tomorrow's models have different context-handling, different attention
patterns, different self-introspection. The target is LLM-agents-as-a-class,
not any one model. Decisions that exploit specific quirks of Claude 4.6 are
fragile; decisions that serve general agent needs (introspection, determinism,
trust calibration) generalise.

### Not "abandon human readability"

Markdown gene content, readable summaries, promoter fields — these still have
value *for operator debugging* and *for paper-track legibility*. The frame
shift is about weighting, not elimination. When a choice forces a trade-off,
the LLM consumer wins; when there's no trade-off, keep the human-legible form.

### Not eternally true

This is a current-phase stance. If helix's primary deployment shifts to
direct human use (unlikely but possible), revisit. If the ecosystem shifts
to a stable agent-API standard (MCP convergence, OpenAI / Google equivalents),
this doc should be updated to match.

---

## When to revisit this doc

- **After PWPC Phase 3 lands** (K gated budget tier in production). That's
  the first real test of whether trust-calibration signals pay off.
- **After an enterprise pilot** (JD-style deployment). Enterprise operators
  are a distinct consumer class; may need separate framing.
- **If a new primary consumer class emerges** (e.g. browser-facing
  assistants, voice agents). Each new class shifts the trade-off weights.
- **If benchmark results stop tracking real-world LLM usage quality**. If
  SIKE 10/10 and LLM end-to-end quality diverge measurably, the benchmark
  isn't testing the right thing — revisit both this doc and the benchmark.

---

## Concrete decisions this doc has already informed

| Decision | Before this doc | After |
|---|---|---|
| Presence gene access pattern | fuzzy retrieval via /context | accept fuzzy miss; direct `/genes/presence:{pid}` lookup is the intended access pattern |
| PWPC Phase 1 schema | tier scores only | add `query_sema` + `top_candidate_sema` so LLM consumers can reason about semantic agreement |
| Antiresonance finding | surface as tuning issue | surface as primary trust-calibration signal in future retrievals |
| Walkability endpoints (`/genes/{id}/neighbors` etc.) | "dashboard feature" | priority for next sprint — LLMs often know *what* to walk to |
| Heartbeat endpoint presence-gene emit | optional body was "nice-to-have" | it's the primary LLM-to-LLM coordination signal |

---

— Established 2026-04-15. Update this doc as a first-class citizen when the
framing shifts; don't let it go stale.

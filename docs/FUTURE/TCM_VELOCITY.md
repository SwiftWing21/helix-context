> **Status (2026-04-13):** Velocity input + ρ Gram-Schmidt fix shipped in `f4dcdcc` (always on). Theta fore/aft ray_trace addendum shipped in `c9367f8` behind `retrieval.ray_trace_theta` (dark).

# TCM Velocity — Velocity Input for the Temporal Context Model (Howard 2005)

> *"Position is what the session is thinking about. Velocity is where
>  it's going."*

A design note for promoting helix's TCM implementation from the **2002**
model (position only) to the **2005** model (velocity input). The
current code ships Howard & Kahana 2002 — the session vector drifts
toward recently accessed items. Howard 2005 generalizes this: the input
pattern `t^IN_i` becomes a *difference* of item representations, which
is mathematically a velocity term in representation space.

This doc supersedes the earlier `TCM_TRAJECTORY.md`. See §"What we got
wrong" at the bottom.

Status: **design note, not yet implemented.**
Date: 2026-04-13

---

## What we have today

`helix_context/tcm.py` implements Howard & Kahana 2002:

```
t_i = ρ_i · t_{i-1} + β · t^IN_i
```

where `t_i` is the 20-D session context vector, `ρ_i` is the drift
factor set by the normalization constraint `||t_i|| = 1`, and `t^IN_i`
is the input pattern derived from the current gene:

```python
# tcm.py, gene_input_vector()
t^IN_i = gene_input_vector(gene_i)   # static current-item embedding
```

Each candidate gene gets a bonus proportional to `context_similarity` —
cosine between the current `t_i` and the gene's own 20-D vector. This
is pure *position alignment*.

## What Howard 2005 changes

Howard (2005, Eq. 16) redefines the input pattern as a **velocity** in
item-representation space:

```
t^IN_i(s) = || p(s+1) − p(s) ||  ·  f(direction)
```

In words: the input pattern is not the current item's embedding, it is
the *difference* between the current item's embedding and the prior
item's embedding, optionally with a direction-tuned kernel `f`. The
rest of the TCM equation is unchanged — `t_i = ρ_i · t_{i-1} + β · t^IN_i`
still runs — but now the drift is **powered by inter-item motion**, not
by absolute position.

Behaviorally this means the session vector tracks *where the user is
heading* rather than *where the user just was*.

## Proposed implementation

Minimal change — roughly 15 LOC:

```python
# Replace in tcm.py update_from_gene():
def update_from_gene(self, gene: Gene) -> None:
    curr = gene_input_vector(gene)
    if self.item_history:
        prev_vec = self.item_history[-1][1]
        # Velocity in representation space (Howard 2005 Eq. 16)
        delta = [c - p for c, p in zip(curr, prev_vec)]
        self.update(gene.gene_id, delta)
    else:
        self.update(gene.gene_id, curr)
```

State change required: none. `item_history` already stores the prior
item's vector. No new tables, no new indexes, no new dependencies.

## What this gives us, diagnostically

The velocity vector becomes a **queryable session-state object**:

- `||v||` large across consecutive queries → topic pivot detected.
  Could trigger a stale-context warning, a chromatin demotion, or a
  context-vector flush.
- `||v||` small across consecutive queries → deep dive. Velocity term
  dominates, predictive surfacing kicks in.
- Velocity vector itself can be logged, visualized, and compared across
  sessions — gives us a per-session "direction of thought" signal we
  can mine offline.

This is what the original `TCM_TRAJECTORY.md` was reaching for; it just
picked the wrong physics analog to justify it.

## Bug found during this review — orthogonality violation

While writing this doc we noticed `tcm.py:195–200` derives `ρ` from:

```
ρ = sqrt(1 − β² · ||t^IN||²) / ||t_{i-1}||
```

This formula is the normalization constant *assuming* `t^IN ⊥ t_{i-1}`.
In practice `t^IN` and `t_{i-1}` are almost never orthogonal — they're
both drawn from the same 20-D semantic space. The code silently masks
the violation with a final `_normalize()` call at line 206 that
re-projects the result onto the unit sphere.

The output stays on the unit sphere, but the *direction* of the
resulting `t_i` is subtly biased away from what Howard & Kahana
intended. The canonical form is to compute `ρ` via the explicit
quadratic equation that does *not* assume orthogonality (solve
`||ρ·t_{i-1} + β·t^IN||² = 1` for `ρ`), then take the positive root.

Functionally this fix is likely to change answer-given-retrieval by a
tiny amount. It should be fixed as a hygiene item alongside the
velocity-input work. Documented here so we don't forget it.

## What this does NOT solve

- **Lex_anchor +291 dominance** (empirical finding on the additive
  fusion) — this is a calibration problem in the fusion layer, not a
  TCM problem. See `STATISTICAL_FUSION.md`.
- **Topological 2-hop relevance** — when the right answer is a gene
  that's 2 hops away in the co-activation graph but not similar in
  embedding space, velocity doesn't help. See `SUCCESSOR_REPRESENTATION.md`
  for the topological prior that composes with this one.
- **Cymatics partial-overlap regime** — different signal, different
  metric. See `WASSERSTEIN_CYMATICS.md`.

## Companion docs

- `SUCCESSOR_REPRESENTATION.md` — topological prior (discrete graph);
  **composes with velocity TCM, does not replace it.** KF on session
  vector is the continuous spatial analog.
- `STATISTICAL_FUSION.md` — how all 11 raw tier outputs (including the
  velocity-TCM bonus and SR bonus) ultimately get combined via a
  stacked GBT calibrated on CWoLa labels.
- `MUSIC_OF_RETRIEVAL.md` — TCM is the 10th of 12 retrieval signals.
- `helix_context/tcm.py` — current 2002 implementation.

## What we got wrong

The prior version of this doc (`TCM_TRAJECTORY.md`, same date) proposed
a "motional EMF" analog — Faraday's law, `EMF = ∮ (v × B) · dl`, Lenz's
stability bias. The math worked out to:

```
traj_align = velocity @ gene_position
```

i.e. a dot product between the session velocity vector and each
candidate gene's position vector.

**This is Howard 2005 Eq. 16 with worse vocabulary.** The "induced
bonus" is literally the TCM drift term with `t^IN` redefined as a
difference. A parallel literature scour surfaced the 2005 paper as
existing prior art; the EMF framing was an independent re-derivation
dressed in electromagnetic clothing. It's not wrong, exactly — the
inner products match — but it introduces `B`, `dl`, Lenz's law, and
flux integration as conceptual baggage that the canonical model
doesn't need.

The corrected framing ships the same code with ~5× cleaner citation,
matches the vocabulary used by the rest of the memory-modeling
literature, and makes the orthogonality bug above visible because we
re-read the 2005 derivation carefully.

Honest lesson: **search for the canonical model before inventing a
metaphor.** Faraday gave us a working analogy; Howard gave us the
actual equation.

## References

- **Howard, M. W. (2005). A distributed representation of temporal
  context for episodic memory.** *Psychological Review*. — Eq. 16
  introduces the velocity/difference input pattern.
- Howard, M. W., & Kahana, M. J. (2002). A distributed representation
  of temporal context. *Journal of Mathematical Psychology* 46(3). —
  The 2002 model currently implemented.
- Gershman, S. J., Moore, C. D., Todd, M. T., Norman, K. A., & Sederberg,
  P. B. (2012). The successor representation and temporal context.
  *Neural Computation*. — Connects TCM to SR; motivates the
  `SUCCESSOR_REPRESENTATION.md` companion doc.

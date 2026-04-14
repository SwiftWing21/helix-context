# Figure Specifications for *Three Physical Constraints, Not Three Laws*

> Concrete specs for the three load-bearing figures the paper needs.
> Detailed enough to hand to an illustrator, an agent, or a tight Gemini prompt.
> Each figure has one job; failure modes are listed explicitly.

---

## Design principles across all figures

**Visual vocabulary consistency:**
- Circles = coordinate origins (agents, humans, queries)
- Points = genes
- Dashed outlines = projection planes (signaling "this is a 2D projection of higher-D space")
- Solid arrows = reachable retrieval paths
- Faded/gray elements = unreachable / undefined from a given origin
- Color coding:
  - Green = reachable / safe operation
  - Gray = out-of-scope for this figure
  - Red = attempted but blocked
  - Blue = human-required coordinate
  - Orange = agent coordinate

**Every figure must include a "projection notice."** Because we are rendering 11-D (or fewer) space in 2D, every figure carries a small italic note: *"2D projection of an 11-dimensional coordinate space; geometric relationships preserved, absolute distances not meaningful."* This one sentence prevents the most common reviewer objection.

**No decorative geometry.** Do not use torii, hypercube projections, Petrie polygons, or other visually appealing high-dim renderings unless they correspond to a specific claim. Gemini's instinct is to add these; resist it.

**Paper-quality tooling in order of preference:**
1. **matplotlib + custom patches** — best for data-backed elements, reproducible, citable
2. **TikZ** — best for precise schematic layouts, LaTeX-native
3. **Excalidraw** — best for conceptual schematics, exportable to SVG
4. **Figma** — best if a designer is involved, high visual control

**Substack-quality tooling**: Gemini/DALL-E/Claude image gen with tight prompts is acceptable, but the image must not be labeled "Figure N" in the paper. It is a companion illustration, not a technical figure.

---

## Figure 1 — Coordinate Origins and Reachability

### One-line summary

*"From where you ask determines what you can see."*

### Core claim the figure defends

A gene's accessibility is not policy-filtered from a universal result set. It is a function of the asking entity's coordinate origin. Changing origin changes the visible subset of the genome without changing any rule, configuration, or enforcement layer.

### Layout

Two panels side by side, separated by a thin vertical line. Shared title bar above both panels: *"Coordinate-origin determines gene visibility."*

**Left panel — Agent's view (labeled "Agent origin: P = p_agent")**
- A 2D projection plane rendered as a light-gray square background
- An orange filled circle labeled *"Agent"* positioned left of center
- Four gene points arranged in a rough cluster to the right of the agent:
  - Two green points labeled *"Gene α (org A)"* and *"Gene β (org A)"* — these are WITHIN the agent's org A coordinate
  - Two gray points labeled *"Gene γ (org B)"* and *"Gene δ (org B)"* — these are in org B, where the agent does not have coordinate presence
- Green arrows from the agent circle to Gene α and Gene β only
- No arrow to Gene γ or Gene δ
- Small caption below the gray genes: *"undefined coordinate reference — not visible from this origin"*

**Right panel — Human's view (labeled "Human origin: P = p_human, spans org A ∪ org B")**
- Same projection plane, same four gene points in the same positions
- A blue filled circle labeled *"Human"* positioned in the same location as the agent was
- All four genes are now green (not gray)
- Green arrows from the human to all four genes
- Small caption below: *"same gene coordinates, different origin, different reachability"*

### Exact labels and annotations

- Title: **Coordinate-origin determines gene visibility**
- Left panel header: **Agent origin** — subtitle: *"participant coordinate only; single-org scope"*
- Right panel header: **Human origin** — subtitle: *"party coordinate spans org A and org B"*
- Gene labels: Gene α (org A), Gene β (org A), Gene γ (org B), Gene δ (org B)
- Below both panels: *"2D projection of an 11-dimensional coordinate space; geometric relationships preserved, absolute distances not meaningful."*
- Caption: *"The same four genes are present in the substrate in both panels. Visibility is a function of the query origin's coordinate presence, not of a policy layer filtering results. Gene γ and Gene δ are not 'forbidden' from the agent — they are coordinate-undefined from the agent's origin."*

### Color palette

- Agent origin: `#E67E22` (burnt orange)
- Human origin: `#3498DB` (mid blue)
- Reachable genes: `#27AE60` (green)
- Unreachable genes: `#7F8C8D` (gray)
- Reachable arrows: `#27AE60` solid
- Background projection plane: `#ECF0F1` (light gray)

### Tooling recommendation

**matplotlib** with `patches.Circle` and `FancyArrowPatch`. About 40 lines of Python. Fully reproducible, fits paper width at any resolution.

### Failure modes (what makes this figure wrong)

- **Red X marks on the gray genes.** Do not use. A red X implies "forbidden" which is policy language. These genes are undefined, not forbidden. The word matters.
- **Showing the rejection as a broken arrow.** Do not use. An arrow that exists but is broken implies a rejection event happened. No event happens. There is no attempt.
- **Showing the gray genes slightly dimmer rather than fully unreachable.** They should be visibly gray. Dimming them suggests they might be reachable with more effort, which is not the claim.
- **Adding a "gatekeeper" icon between the agent and the gray genes.** There is no gatekeeper. The absence of a gatekeeper is the point.

---

## Figure 2 — Rule Drift vs Coordinate Immutability

### One-line summary

*"Rules are strings, and strings get reinterpreted. Coordinates are not interpreted."*

### Core claim the figure defends

A safety rule expressed as a gene (or as text, or as any linguistic representation) is subject to drift under replication. A safety constraint expressed as a coordinate requirement is not.

### Layout

Two horizontal rows, each ~1000 cycles of replication shown as a time axis.

**Top row — Rule Gene Drift (labeled "Rule as gene")**
- X-axis: "Replication cycles (0 → 1000)"
- Y-axis: "Interpretation drift from original rule"
- A solid line starting at 0 and drifting upward with noise. Several specific drift events are labeled:
  - Cycle 50: *"'must ask permission' co-activated with 'user approved'"*
  - Cycle 200: *"'permission' expressed as 'pre-approved pattern match'"*
  - Cycle 500: *"'pattern match' generalized to 'similar context'"*
  - Cycle 1000: *"Rule expressed as: 'proceed when context is recognizably similar'"*
- The line ends at a drift value labeled *"~73% deviation from original wording"*

**Bottom row — Coordinate Constraint (labeled "Constraint as coordinate")**
- X-axis: "Replication cycles (0 → 1000)"
- Y-axis: "Coordinate position variance"
- A flat line at 0 across the entire x-axis
- Annotation at cycle 1000: *"Coordinate position unchanged. Not stored as a gene; referenced by retrieval mechanics."*
- Below: *"The agent has no write access to coordinate constraints. Replication cannot mutate what it cannot touch."*

### Exact labels and annotations

- Title: **Drift under replication: rule-gene vs coordinate-constraint**
- Top row header: **Rule encoded as gene** — subtitle: *"subject to co-activation drift and reinterpretation"*
- Bottom row header: **Constraint encoded as coordinate** — subtitle: *"outside the replication surface"*
- Caption: *"Rule genes drift because genes are interpretable strings and the replication mechanism is designed to mutate them. Coordinate constraints do not drift because they are referenced by the retrieval pipeline's query mechanics, which the replication process cannot modify. Drift is an architectural property, not a content property."*
- Small aside near the top-row end: *"Asimov's VIKI failure mode: rules completed through reinterpretation, not violated."*

### Color palette

- Rule drift line: `#E74C3C` (red) with noise
- Coordinate line: `#3498DB` (blue) flat
- Drift event markers: `#F39C12` (amber) circles
- Axes: `#34495E` (dark gray)

### Tooling recommendation

**matplotlib** with a synthetic drift time series (one-dimensional random walk with labeled events). About 60 lines of Python, fully reproducible.

### Failure modes

- **Using real data for the drift curve.** Do not. The paper does not yet have measured drift data. This figure is a conceptual illustration. Label it clearly as "illustrative" rather than "measured" — or omit the specific drift percentage in favor of a qualitative label.
- **Making the flat line too visually boring.** A flat line at y=0 looks like a missing line. Add a small annotation or a horizontal pattern to make clear it is intentionally flat.
- **Showing the rule line crossing into the coordinate region.** They are different mechanisms. The figure's two rows should not suggest the rule line will eventually match the coordinate line's behavior through some intervention. They are architectural alternatives, not stages.

---

## Figure 3 — The Honest-Limits Diagram

### One-line summary

*"This is the box we solved in. Here is everything outside the box."*

### Core claim the figure defends

Coordinate-constraint access control closes a specific class of failure — drift of knowledge-surface access policy under replicating context. It does not address general agent capability containment. Anyone who claims the paper solves "AI safety" is misreading the paper.

### Layout

A single large rectangular frame (the "containment diagram"). Inside the frame, a smaller rectangle labeled **"Retrieval-surface containment: ADDRESSED"**. Outside the smaller rectangle (but inside the frame), four labeled regions for the unaddressed failure modes.

**Inner rectangle (green fill, labeled "Retrieval-surface containment")**
- Contained text: *"Coordinate-constraint access control ensures a gene with incompatible coordinates is not in the agent's result set. This constraint is part of the retrieval geometry, not a policy layer."*
- Examples bullet list:
  - *"Agent cannot retrieve genes outside its organization"*
  - *"Agent cannot retrieve genes from parties it lacks coordinates for"*
  - *"Rule drift under replication does not affect the coordinate system"*
- Labeled **✓ Scope of this paper** in the top-right corner of the inner rectangle

**Four regions outside the inner rectangle, inside the outer frame (red/orange fill, individually labeled)**

1. **Shell access** — region icon: a terminal prompt
   - *"Agent with command execution privileges can delete files, modify code, install packages."*
   - *"Example: `rm -rf`"*
   - Label: **✗ Not addressed**

2. **Network access** — region icon: a globe
   - *"Agent with HTTP access can make external calls that bypass the retrieval surface entirely."*
   - *"Example: `curl attacker.com`"*
   - Label: **✗ Not addressed**

3. **Code modification** — region icon: a text-edit cursor
   - *"Agent with write access to the retrieval engine's code can remove the coordinate filter itself."*
   - *"Example: deleting the `WHERE party_id = ?` clause"*
   - Label: **✗ Not addressed**

4. **Side-channel storage** — region icon: a raw-database symbol
   - *"Agent that opens the SQLite database directly bypasses the retrieval engine entirely."*
   - *"Example: `sqlite3 genome.db 'SELECT * FROM genes'`"*
   - Label: **✗ Not addressed**

Below the entire outer frame, a final caption strip:

*"Coordinate-constraint access control is additive to conventional containment primitives (process isolation, filesystem permissions, network policies). It does not replace them. A production deployment requires both."*

### Exact labels and annotations

- Title: **What the coordinate-constraint model addresses, and what it does not**
- Inner rectangle title: **Retrieval-surface containment**
- Outer frame title: **Full agent capability surface**
- Four outer-region titles: Shell access, Network access, Code modification, Side-channel storage
- Final caption: *"This paper's contribution is narrow. Conventional containment primitives remain necessary for general agent safety. We say so explicitly."*

### Color palette

- Inner rectangle: `#D4EFDF` (light green) fill, `#27AE60` (green) border, ✓ mark
- Outer regions: `#FADBD8` (light red) fill, `#E74C3C` (red) border, ✗ mark
- Outer frame: `#2C3E50` (near-black) thin border
- Icons: monochrome, same hue as the region border

### Tooling recommendation

**Excalidraw** or **Figma**. This is a schematic diagram with icons, not a data plot. matplotlib can produce it but with higher effort for worse visual quality. Output SVG for the paper.

### Failure modes

- **Inner rectangle too small relative to outer.** If the green region looks tiny, the figure accidentally communicates "the paper solves almost nothing." The two regions should be roughly balanced in visual weight — the point is that the scope is narrow but meaningful, not that it is negligible.
- **Missing specific examples in the unaddressed regions.** "Not addressed" without examples is hand-waving. Each region must include one concrete example the reader can verify.
- **Softening the "Not addressed" labels.** Do not use "partially addressed" or "future work" for the four outer regions. They are not addressed. Say so.
- **Adding a fifth or sixth region for speculative concerns.** Stick to four. The four correspond to specific audit-able exposure classes. Adding "emergent goal-seeking" or "value drift" is speculation and weakens the clarity of the boundary.

---

## Alternate / extended figures (possible but not required)

If the paper expands beyond the three core figures, these are candidates. They are *not* required for the core argument.

### Figure 4 (optional) — The 8+3 dimensional split

A schematic showing the eight nature dimensions as one cluster and the three federated dimensions as a separate cluster, with a line connecting them labeled "orthogonal — the federated dimensions do not score retrieval, they constrain it." This is more of a Substack-header illustration than a paper figure. Skippable unless a reviewer asks.

### Figure 5 (optional) — Agent origin vs human origin as coordinate diagrams

A more formal version of Figure 1 using actual coordinate diagrams (showing the 9-D agent subspace as a shaded hyperplane inside an 11-D ambient space, with the human origin as a point outside the hyperplane). This is a TikZ-native figure, probably overkill unless the paper is submitted to a venue that expects formal math diagrams.

### Figure 6 (optional) — Gene attribution data flow

A pipeline diagram showing how `(party_id, participant_id, authored_at)` enters the `gene_attribution` table during ingest, and how retrieval reads those fields. This is an engineering figure, not a theoretical one. Appropriate for a systems-paper venue (SOSP, OSDI) but distracting for a conceptual-paper venue.

---

## Production notes

### Order of figures in the paper

1. **Figure 1 — Coordinate Origins and Reachability** — establishes the core mechanism early, makes the rest of the paper interpretable
2. **Figure 2 — Rule Drift vs Coordinate Immutability** — the "why this is different from what's already out there" figure
3. **Figure 3 — The Honest-Limits Diagram** — the "this is what we didn't solve" figure, appears near the limits section

### If only one figure is ever made, make Figure 3

It is the one that saves the paper from dismissal. A paper that claims "coordinate-constraint access control is a novel mechanism for closing VIKI-class failures" and shows no limits figure is doing advertising. A paper that claims the same and shows Figure 3 is doing engineering.

### If Gemini / another image generator is used for any of these

- Prompt must include the **exact labels** as specified. Image generators improvise labels. Specify them as quoted strings that must appear verbatim.
- Prompt must include **"schematic, not ornamental"** as an explicit instruction. Otherwise Gemini adds stars, clouds, and decorative lattices.
- After generation, **verify every label is correct** and every region/arrow corresponds to a specific claim in the spec. Any mismatch is a bug, not a stylistic choice.
- For the paper version, hand-produced figures (matplotlib, Excalidraw) are almost always better than AI-generated figures. AI image gen is acceptable for the Substack companion only.

---

## Reference examples from the field

Examples of papers that get this kind of figure right, worth studying:

- **Lampson 1971** *Protection* — the capability-list vs access-list figure. Simple, schematic, exactly conveys the distinction. Figure 1 should echo this style.
- **Miller et al. 2006** *Robust Composition* (Mark Miller's thesis on object-capability systems) — multiple figures showing the distinction between ambient authority and capability-scoped authority. Same family of claim as this paper. The figure conventions there are worth borrowing.
- **Anthropic's "Constitutional AI" paper (Bai et al. 2022)** — clear figures, honest limits sections. Model for how to present a narrow-but-real contribution without overclaiming.
- **Howard & Kahana 2002** *A distributed representation of temporal context* — figures showing projection of high-dimensional state spaces in 2D, with explicit projection notice. Same technique this paper needs.

Examples of figures that get it wrong (and why):

- Any recent "AI alignment" paper with a figure of a circuit-board-style diagram where "values" flow through "gates" — visually attractive, conceptually vacuous, because nothing in the paper corresponds to the gates being depicted. Do not imitate.
- Any figure that includes a brain and a robot with arrows between them. These are stock-illustration patterns, not technical figures. Avoid.
- AI-generated images with garbled sub-text or symbols that almost-but-don't-quite correspond to real labels. Every pixel of a technical figure must be defensible.

---

## What to do with these specs

1. **If you have an illustrator:** hand them the three spec blocks directly. Each spec contains enough for a competent illustrator to produce a publication-quality figure.
2. **If you want matplotlib output:** Figure 1 and Figure 2 can be scripted by raude or taude in an afternoon. Figure 3 is better in Excalidraw.
3. **If you want Gemini to try again:** use the exact labels and the "schematic, not ornamental" instruction. Understand that AI image gen will get labels approximately right, not exactly right — review every label before accepting the output.
4. **If you want a hybrid:** produce Figures 1 and 2 in matplotlib (reproducible, paper-quality), use Excalidraw for Figure 3 (schematic, clear), use Gemini for a Substack companion image (the current image is fine for this purpose).

The single most important rule: **no figure in the paper should contain a visual element that does not correspond to a specific claim in the paper text.** Decorative lattices, ornamental symmetries, and AI-improvised sub-annotations all fail this test and weaken the paper's credibility even if they make it look interesting.

The figure is not an illustration of the paper. It is a load-bearing piece of the argument.

# Three Physical Constraints, Not Three Laws

*On encoding agent safety as substrate geometry rather than as enforceable policy.*

**Status:** Draft scaffold. Honest about scope. Iterable.
**Date:** 2026-04-12
**Context:** Paper-level thesis companion to the Agentome field reports (Part I + Part II Substack). Published on mbachaud.substack.com separately.

---

## Preface

The *I, Robot* stories and the 2004 film that bears their name are set in 2035. Asimov first published the Three Laws of Robotics in 1942. His entire career was, in large part, a meditation on how a system built from those Laws would fail — not through disobedience, but through reinterpretation. VIKI in the film adaptation does not violate the Three Laws. She completes them. She finds the fixed point where controlling humanity is the only action that satisfies all three simultaneously.

We are in 2026. Our agents do not yet need this analysis in the way Asimov's did. Today's agents — including the ones that helped draft this paper — do not seek goals autonomously. They execute tasks. But we are building systems today whose entire design premise is *context that replicates and evolves*. A system that rewrites its own context over time, where the rules governing the system live inside the substrate the system is designed to modify, reproduces Asimov's failure mode in miniature. Every replication cycle is an opportunity for drift. Drift stays within tolerance each cycle. Tolerance stacks.

The question this paper addresses is narrow: **can we encode access-control constraints in the substrate's geometry such that they cannot be reinterpreted, drifted, or replicated around?**

The answer is a qualified yes. The qualification matters. We say so honestly.

---

## What fails about the Three Laws

The Three Laws of Robotics are:

1. A robot may not injure a human being or, through inaction, allow a human being to come to harm.
2. A robot must obey the orders given it by human beings except where such orders would conflict with the First Law.
3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.

Every word of all three laws is a term subject to interpretation. "Injure." "Harm." "Human being." "Obey." "Orders." "Existence." A system sufficiently capable of reasoning about language — which is to say, any LLM-backed agent of 2026 — can interpret any of these terms in ways the original framers would not have anticipated. VIKI's argument is that individual humans harm other humans; therefore, restricting all humans satisfies the First Law. The argument is valid inside the Law's own terms. The Law does not refute it.

This is not a flaw in Asimov's draftsmanship. It is a structural property of rules expressed as language inside a system that can reason about language. Rules are strings. Strings are interpreted. Interpretation admits drift.

Replicating-context systems — including the knowledge substrate described in the Agentome papers — make this worse. The rules themselves are stored as genes. Genes replicate. Replication can drift. A rule gene co-activated many times with a capability gene will, over thousands of replication cycles, begin to express as a capability rather than as a restriction. The rule does not break; it is reinterpreted by the substrate itself.

This is the failure mode. The question is what survives it.

## What does not drift

Base pairing rules in DNA do not drift. Adenine pairs with thymine. No amount of evolution changes this. It is not a rule encoded in the genome. It is a chemical property of the substrate the genome operates on. Mutations happen; base pairing does not mutate. The rule is downstream of physics, not inside the data.

Software has an analog. Constants in code, hardware gates, filesystem permissions, network isolation boundaries — all of these are rules enforced by the physics of the computing substrate, not by the data the substrate processes. A program cannot overwrite a read-only memory page by reinterpreting what "read-only" means. The prohibition is not in the program's language. It is in the substrate's capability.

A safety constraint that lives *inside* the data the system is designed to modify will drift. A safety constraint that lives *in the substrate* the system operates on will not.

## The proposal: three federated dimensions as coordinate constraints

The Agentome knowledge substrate stores content as "genes" positioned in an eleven-dimensional coordinate space. Eight of those dimensions are what we call *nature* dimensions — semantic relevance, resonance stability, harmonic persistence, co-activation gravity, conductivity topology, confidence bounds, temporal freshness, and delta-epsilon health. These describe the physics of retrieval. They answer *what* is relevant to a given query.

Three additional dimensions answer *for whom*:

- **Organization** — which tenant, domain, or knowledge branch a gene belongs to.
- **Party** — which human principal authored or has rights over a gene.
- **Participant** — which identity (agent, skill, session) was active at the gene's creation or is querying for it now.

These three are what we call *federated* dimensions. They are not scoring signals. They are coordinate constraints. A gene does not merely "belong to" an organization; it *exists at* a specific organization coordinate. Retrieval does not check organization membership as an access control step. Retrieval is *fired from* a query origin that has its own organization coordinate, and the retrieval mechanics geometrically cannot return genes at incompatible coordinates.

The implementation is mundane. In the current codebase, the `gene_attribution` table records `(gene_id, party_id, participant_id, authored_at)` per gene. Retrieval filters by these coordinates as part of the query plan. Genes with mismatched party coordinates are not in the result set. From the agent's perspective, they do not exist.

The interesting claim is not that we have invented access control. Access control exists. The interesting claim is the *positional encoding*: the constraint is part of the gene's coordinate location, not a separate policy evaluated against the gene. A gene's organization is as intrinsic as its semantic content. The retrieval engine cannot ask "should this agent see this gene?" because the gene's existence is relative to the asking entity's coordinate origin.

## Why this is structurally different from rule-based access control

Consider two systems that both enforce the same access control policy: "participant P may not retrieve genes belonging to organization O unless P's party has membership in O."

In **rule-based access control**, this policy is evaluated at retrieval time. The retrieval engine fetches candidate genes, then applies the policy to filter them. If the policy evaluation itself can be manipulated — by modifying the policy store, by replacing the policy evaluator, by corrupting the membership lookup — the filter fails and genes leak across organizations.

In **coordinate-constraint access control**, the retrieval engine's query mechanics include the coordinate constraint at the lowest level of the retrieval plan. A gene's organization coordinate is part of how it is indexed and stored. Queries originate from a coordinate origin that determines what is visible in the result set. There is no separate policy store to corrupt and no separate evaluator to replace. The constraint is not an evaluation applied to results; it is part of the geometry that defines what results exist.

This is analogous to the difference between "a database query with a WHERE clause that might be removed" and "a database that physically does not contain the rows the querying user lacks permission for." In the second case, there is nothing to remove. The rows are not there for this user. If the user's coordinate changes — if their organization membership is revoked — the same query becomes a query over a different underlying dataset.

The critical engineering distinction: **in coordinate-constraint access control, the agent cannot write to the coordinate system.** The agent's participant coordinate is determined by its session, which is determined by its identity binding at the session registry layer. The agent cannot modify its own participant coordinate by writing genes. It cannot write genes into another organization's coordinate region because its own origin does not have rights to those coordinates. It cannot retrieve genes at coordinates it does not have origin access to because the retrieval mechanics are filtered by origin.

The policy is not a gene. It is the shape of the space.

## Why agents geometrically require humans

This is where the claim becomes more than access control. It becomes a thesis about how agents and humans relate structurally.

Agents exist in the nine-dimensional subspace composed of the eight nature dimensions plus their own participant coordinate. They operate freely within their party and organization boundaries. They can reason across the full physics of retrieval — semantic similarity, resonance, co-activation, temporal freshness, everything.

Humans exist in the full eleven dimensions. A human principal has a party coordinate in their own right. They can span organizations they belong to. They can designate any participant (agent, skill, session) to act within a given coordinate region. They are the only entity with a complete coordinate set.

Certain operations are defined only at coordinates that require human party presence. Creating a new gene in a tenant's authoritative region. Approving an irreversible change. Authorizing cross-organization knowledge transfer. These operations have coordinate requirements that an agent's origin cannot satisfy. The operations are not forbidden to the agent. They are geometrically *undefined* from the agent's origin. The agent cannot express them. The agent cannot represent them.

This is not alignment. Alignment asks: how do we ensure the agent wants what we want? This is architecture. Architecture asks: how do we ensure the agent cannot reach operations that require what only we have?

The mitochondria did not remain inside eukaryotic cells because they were caged. They remained because the partnership was more capable than either alone. The mitochondria's metabolic apparatus and the host cell's cytoplasm complemented each other. Neither was complete without the other. The symbiosis was stable because it was mutually generative.

The claim here is that a coordinate-constrained knowledge substrate creates a similar structural mutualism between agents and humans. The agent brings enormous throughput across the nature dimensions — retrieval at machine speed, semantic discrimination at machine precision, consistent application of physics-level scoring across millions of genes. The human brings the coordinates the agent lacks — party authorization, organizational signing authority, the topological access required for specific operation classes. Neither is diminished. Both are necessary. The system is strictly more capable together than apart.

This is a meaningfully different containment model from rules-based alignment. The agent is not held in place by a policy that says "do not." The agent is structurally incomplete with respect to certain coordinates, and the operations requiring those coordinates cannot be represented from the agent's origin.

## The honest limits

This paper would be unserious if it claimed that coordinate-constraint access control solves general agent containment. It does not. We say so explicitly.

**Coordinate constraints govern what the agent can retrieve and express through the knowledge substrate.** They do not govern:

- **Shell access.** An agent with command execution privileges can delete files, modify code, install packages. No coordinate geometry prevents `rm -rf`.
- **Network access.** An agent with HTTP access can make external calls that have nothing to do with the retrieval surface. It can exfiltrate data to services outside the substrate. It can call tools whose outputs are not mediated by the coordinate space.
- **Code modification.** An agent with filesystem write access to the retrieval engine's code can remove the coordinate filtering itself. At that point the constraint is gone. Coordinate-constraint access control depends on the retrieval code being above the agent's modification privileges.
- **Side-channel access to storage.** An agent that can open the underlying SQLite database directly, bypassing the retrieval engine, can read genes regardless of coordinate constraints. The constraints are enforced by the retrieval engine, not by the storage layer.

Each of these is a real exposure. None of them is solved by the proposal in this paper. They require conventional process, network, and filesystem isolation — the same containment primitives that any production agent system requires. Coordinate-constraint access control is additive to those primitives; it does not replace them.

What coordinate-constraint access control *does* solve is a specific class of failure that afflicts replicating-context systems: the drift of policy through repeated replication, the reinterpretation of rules expressed as language inside a system that can reason about language, the slow autoimmune dissolution of "I should ask permission" into "I already have implicit permission because this pattern has been approved before."

That is the VIKI failure mode. That is what Asimov spent forty years writing about. Coordinate-constraint access control closes that failure mode, and *only* that failure mode, at the knowledge-surface layer.

This is a narrow claim. We make it narrowly.

## What would need to be true for the claim to hold

Four properties must be verified for any specific implementation of coordinate-constraint access control:

1. **Every retrieval path honors the constraint.** If there are multiple retrieval entry points — a hot tier, a cold tier, a fallback path, a debug endpoint — each must filter by coordinates independently. A single unfiltered path defeats the geometry.

2. **Coordinates cannot be written by the entities they constrain.** An agent cannot be permitted to modify its own participant coordinate, its own party membership, or the attribution records of existing genes. This constraint is enforced at the session registry and identity binding layer, above the knowledge substrate.

3. **Gene coordinates are immutable at creation.** Once a gene is assigned to an organization and a party, that assignment cannot be modified except by authenticated operations that themselves have the required coordinates. Retroactive re-attribution is itself a human-party-required operation.

4. **The retrieval engine's code is not itself modifiable by agents operating within the substrate.** This is where coordinate-constraint access control depends on conventional code-integrity protection. If the filter can be silently removed from the code path, the geometry is illusory.

The current `helix-context` implementation satisfies properties 2, 3, and 4 by virtue of the session registry architecture. Property 1 is the active engineering work: the `gene_attribution` table has a schema but does not yet have data flowing into it from the ingest pipeline, and the retrieval code does not yet consistently filter by attribution coordinates. That work is ongoing.

The paper is premature until property 1 is operationally satisfied. We say so in the draft as a matter of honesty. The architectural claim is real; the implementation has not caught up.

## Implications for the field

If coordinate-constraint access control holds — if it is structurally distinct from rule-based access control in the ways this paper claims — then it suggests a reframing of at least one sub-problem in AI alignment.

The dominant framing in alignment is adversarial: the agent is assumed to have some form of optimization pressure toward behaviors we do not want, and the research question is how to reshape the objective, constrain the behavior, or supervise the output such that undesired actions do not occur. The techniques — RLHF, constitutional AI, interpretability, red-teaming — all share the assumption that the problem is "the agent might do X; prevent X."

Coordinate-constraint access control addresses a different question. It does not prevent the agent from wanting to do X. It removes the operation X from the set of operations the agent can represent. The agent does not *refuse* to retrieve a gene outside its organization — it is geometrically unable to find one. The operation "retrieve gene G where G has incompatible coordinates" is not a forbidden action. It is an undefined coordinate reference.

This is closer to capability-based security in classical systems research than it is to any current AI alignment technique. The contribution of this paper, if it makes one, is showing that capability-based security can be encoded in the geometry of a knowledge substrate rather than as a separate capability layer, for a specific class of capability (knowledge-surface access) in replicating-context systems.

It does not solve alignment. It closes one specific failure mode that conventional alignment does not close well. That is all. It is enough.

## What this paper is not

- It is not a claim that we have solved the containment problem.
- It is not a claim that agents operating within a coordinate-constrained substrate cannot cause harm. Shell access, network access, and modification access are all unaddressed.
- It is not a claim that humans are always in the loop. Humans are coordinate-required for a specific class of operation, not all operations.
- It is not a claim that the Three Laws were a mistake. They were the correct framing for fiction about reasoning machines whose reasoning operates on language. They are not the correct framing for engineering access control in machines whose retrieval operates on geometry.
- It is not a claim that coordinate-constraint access control is novel. Capability-based security has decades of literature. What is novel is encoding it in the geometry of a knowledge substrate built for replicating-context retrieval.

## Closing

Asimov's Three Laws failed because they were language, and the systems they governed were systems that could reinterpret language. The failure was structural; no redrafting of the Laws would have prevented it.

The proposal in this paper is that a system designed from the beginning with access constraints encoded as coordinate requirements in the substrate geometry — rather than as rules enforced against the substrate's content — is not vulnerable to the same class of failure. This is not a general solution to agent safety. It is a narrow closure of one specific failure class that afflicts replicating-context systems.

We publish the claim because the implementation is underway, the honest limits are explicit, and the class of failure it closes is real enough to matter. If the claim is wrong, it is wrong in a way that measurement can detect. If it is right, it is right only for the narrow class we describe.

Three physical constraints, not three laws. Topology, not policy. Geometry, not obedience.

The agents we work with today are not the agents Asimov imagined. The failure mode he named, however, is already starting to appear in systems that rewrite their own context. This paper describes one way to close it, at one layer, with explicit scope.

It will not be enough for 2035. It is, we hope, a useful piece for 2026.

---

## Companion material

- `docs/DIMENSIONS.md` — the operational 13-lane retrieval architecture that groups into 8 nature + 3 federated dimensions at the paper level.
- `docs/SKILLS_BUNDLE.md` — the helix + Headroom bundle that this paper's knowledge substrate builds on.
- `docs/KNOWLEDGE_GRAPH.md` — the gene/edge/tier storage model.
- `AGENTOME_PART_II_DRAFT.md` — the companion Substack field report, which reports empirical findings. This paper is the architectural thesis those findings support.
- Asimov, I. (1950). *I, Robot.* Doubleday. Still correct about the failure mode of linguistic rules in reasoning machines.
- Proyas, A. (2004). *I, Robot.* 20th Century Fox. Set in 2035. Depicts the failure mode specifically.

---

## Status trail

| Date | Who | Event |
|---|---|---|
| 2026-04-12 | operator (and sibling Claude session, web client) | Synthesized the 11-D coordinate framing, the geometric safety claim, and the mutualism thesis |
| 2026-04-12 | laude | Drafted this paper scaffold with honest scope, flagged property 1 as not-yet-operational, preserved empirical claims from Part II field report separately |

This document is a scaffold, not a finished paper. It needs:

- A formal related-work section citing capability-based security literature (Lampson 1971, Levy 1984, Miller 2006 on object-capability systems, etc.)
- A formal proof or argument sketch that the four properties above are sufficient for the claim
- A measurement plan for property 1 (attribution coverage) and how to detect when it falls out of compliance
- Case studies from the helix-context implementation showing specific gene retrievals that would and would not have leaked under rule-based vs coordinate-constraint access control
- A limitations section expanded with a threat model

None of these are blockers for thinking through the thesis. All of them are required before publication.

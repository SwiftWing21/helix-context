# From RAG to SIKE (Got You) — draft hook

Status: C-option (300-word probe), paused 2026-04-14. B (full draft) deferred.

## Open questions before committing
1. Voice — does "Sike. Got me." land?
2. Self-criticism level — "should have written *before* I published Agentome, not after" — too harsh publicly?
3. Ending — keep "the next post will" (commits sequel) or soften?
4. Date on Agentome — [N days ago] needs real value.

## Draft

### From RAG to SIKE (Got You)

*Second post. First since Agentome [N days ago].*

The `helix-context` repo is seven days old. I shipped Sprints 1, 2, and 4 yesterday. Somewhere in the middle of that sprint I started calling the retrieval layer a "Scale-Invariant Knowledge Engine." SIKE. I liked how it sounded. I also liked that saying it out loud brought back the 90s playground cadence — *sike, got you* — because I was pretty sure I'd landed on something nobody else was doing.

Then I read the KVzip paper out of Seoul National this week, and my brain did the thing brains do when you've been moving fast and haven't looked up. *Wait — is nobody pursuing this because it's that hard?* Fifteen minutes of honest literature-checking later: no. The actual answer is that everyone is pursuing pieces of it, under different names, and I had built a narrative wrapper around a crowded field without admitting that to myself.

Sike. Got me.

This is the post I should have written *before* I published Agentome, not after. Two parts:

1. **What SIKE-the-framing actually overlaps with**, honestly. SPLADE, LLMLingua, RAPTOR, GraphRAG, MemGPT, Letta, Mem0, Headroom — I'll walk each one and show where it intersects Helix. Some of them do what I'm doing, better. Some of them don't do what I thought they did. It matters which is which.

2. **What's actually narrow, defensible, and mine.** Not "the bridge to embodied AGI." Not "synthetic nervous system." Something much smaller and much more likely to survive a reviewer: persistent sparse state across process restarts, cross-agent shared substrate, and biologically-motivated promotion/demotion rules that aren't just vibes.

The embarrassing part is that Agentome could have included this list. The useful part is that the next post will.

## Companion artifacts to build for B
- Overlap matrix table: {SPLADE, LLMLingua, RAPTOR, GraphRAG, MemGPT, Letta, Mem0, Headroom} × {sparse, hierarchical, persistent, cross-agent, biology-motivated}
- The "narrow defensible" claim with a pointer to the specific Helix code (chromatin promotion, cold-tier, shared genome across Laude/Taude/Raude)
- Reference links: KVzip (Seoul National, techxplore 2025-11), KVPress (NVIDIA)

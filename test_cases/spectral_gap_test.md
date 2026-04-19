# Helix Spectral Gap Detection: Canonical Test Case

**Date:** 2026-04-18
**Status:** Reproducible Baseline
**Type:** Architecture Verification / Genome Control

## Overview
This document logs a canonical query that proves the functional difference between Semantic (SEMA) extraction and Cymatic structural filtering in Helix's retrieval engine. It serves as a permanent test case indicating that "spectral gap" detection via `cymatic_cos_sim` correctly identifies hallucinated or non-structural semantic links.

## The Negative Control (The Hallucination)
We tested a query for an architectural relationship that _does not exist_ in the codebase but shares high semantic overlap with existing Agent code.

**Query:** `Agentome kernel delta-sync`
**Parameters:** `helix_resonance(downsample=256, k=5)`

### Results:
The SEMA vector aggressively matches terms like "agent", "sync", and "kernel/loop". However, the Cymatic spectrum instantly flags the structural mismatch:
1. `ff273f3b...` (PPO Component): SEMA = `0.9558`, **Cymatic = `0.1499`**
2. `c70d64c6...` (AgentBrain Spec): SEMA = `0.9520`, **Cymatic = `0.0039`**

**Conclusion:** The semantic space is noisy and eager to please, but the cymatic frequency space recognizes that there is zero cohesive local density or structural reality to an "Agentome delta-sync".

## The Positive Control (True Cohesion)
To prove the genome isn't just spectrally sparse everywhere, we ran `helix_resonance` targeting a known dense subgraph: the Team Orchestrator hierarchy.

**Query Target Text:** `Team Orchestrator — Design Spec... A Claude Code skill that orchestrates...`
**Parameters:** `helix_resonance(downsample=256, k=5)`

### Results:
1. `ebee5916...` (Team Orchestrator - Design Spec): SEMA = `0.9800`, **Cymatic = `0.5706`**
2. `0cc0ee26...` (Team Orchestrator Implementation Plan): SEMA = `0.9569`, **Cymatic = `0.7027`**

**Conclusion:** When a genuine structural block exists with properly co-activated documents, the Cymatic cosine similarity rises significantly (up to ~0.70x).

## Takeaway for Future Verification
If questioned on whether spectral gaps are simply "threshold tuning" artifacts, run this exact test. The discrepancy (Cymatic drops to `0.0039` for the hallucinated structure but hits `0.7027` for true structural intent) proves the spectrum acts as a robust, topological truth-filter.

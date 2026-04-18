# Helix Code Bundle for Celestia Collab — 2026-04-13

This is the second code drop, expanding on `helix-collab-bundle.tar.gz`.
Covers what you asked for in `SIGNALING_BRIEF_FOR_MAX.md`.

## Contents

helix_context/
  context_manager.py         - retrieval pipeline, start at _express()
  cwola.py                    - CWoLa classifier surface (same as bundle 1)
  cymatics.py                 - D6 frequency-domain scoring
  tcm.py                      - D9 temporal context (partial, gate-ready)
  sr.py                       - Successor Representation (Tier 5.5, flag-gated)
  schemas.py                  - data types (same as bundle 1)
  genome_schema_section_lines_340_900.py  - just the SQL schema portion; full genome.py is ~2940 lines
  config.py                   - config loader, NOTE [session] block is new
  server.py                   - FastAPI endpoints, NOTE cwola_session_id block is newly patched

docs/
  DIMENSIONS.md              - the 9 lanes (D1-D9) reference
  PIPELINE_LANES.md          - retrieval flow diagram
  future/
    STATISTICAL_FUSION.md    - CWoLa framework spec (what cwola.py implements)
    SUCCESSOR_REPRESENTATION.md - SR design note (what sr.py implements)
    TCM_VELOCITY.md          - TCM wiring plan (what tcm.py partially implements)

## Things to note

1. The [session] block in helix.toml + the synthetic session fallback
   in server.py:500-530 is a NEW FIX as of 2026-04-13 pm. Prior to this,
   every cwola_log row had NULL session_id so sweep_buckets treated
   them all as Bucket A. The fix is documented in RESPONSE_TO_SIGNALING_BRIEF.md.

2. TCM is partial. tcm.py has the Howard & Kahana 2002 drift math but
   isn't wired into Step 3 of _express() yet. That's D9 in DIMENSIONS.md.

3. SR (sr.py) shipped DARK behind retrieval.sr_enabled flag. gamma=0.85,
   k_steps=4, weight=1.5, cap=3.0. Flip the flag for A/B.

4. The full genome.py is 2940 lines — schema is 340-900, retrieval 1200-1900,
   write paths 1900-2400. I only bundled the schema section. If you want
   the retrieval or write-path sections, ping me and I'll push those too.

Questions: see docs/collab/RESPONSE_TO_SIGNALING_BRIEF.md in the main R2
path for my reply to your brief.

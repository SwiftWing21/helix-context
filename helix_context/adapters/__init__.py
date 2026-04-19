"""Helix adapters — reference composition-layer modules.

This subpackage is deliberately thin and opt-in: Helix's core stays a
coordinate index, and these adapters are example consumers of the
packet surface (see ``docs/INTEGRATING_WITH_EXISTING_RAG.md``).

    - ``helix_context.adapters.dal`` — Data Access Layer with file / HTTP
      reference fetchers (S3 via optional dep). Turns opaque source_ids
      into bytes regardless of transport.

These modules are meant to be copied, subclassed, or replaced. They
exist so integrators have a starting point, not a required dependency.
"""

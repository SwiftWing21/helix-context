"""Canonical software-vocabulary aliases for the legacy biology lexicon.

Helix's original vocabulary borrowed from molecular biology (gene, genome,
ribosome, chromatin, etc). This module exposes the same underlying
classes and modules under standard software names so new code can
sidestep the cognitive tax of holding two mental models in parallel.

All aliases preserve identity — ``Document is Gene`` is True; importing
``DocumentTags`` and ``PromoterTags`` from the two surfaces returns the
same class object. There is no subclassing, no wrapping, and no runtime
cost to using the canonical names.

The legacy names remain valid for back-compat: existing imports, docs,
papers, handoffs, and git messages stay readable without modification.
The full bidirectional mapping lives in ``docs/ROSETTA.md``.

Usage:

    from helix_context.aliases import (
        Document, KnowledgeStore, Compressor,
        DocumentTags, DocumentSignals, LifecycleTier,
        DocumentAttribution,
    )

Lexicon: see ``docs/ROSETTA.md`` for the legacy biology terms and their
canonical software equivalents used elsewhere in the codebase.
"""

from __future__ import annotations

# ── Schemas (pydantic classes — identity-preserving aliases) ─────────────
from helix_context.schemas import (
    ChromatinState as LifecycleTier,
    EpigeneticMarkers as DocumentSignals,
    Gene as Document,
    GeneAttribution as DocumentAttribution,
    PromoterTags as DocumentTags,
)

# ── Core modules (module-level class aliases) ────────────────────────────
from helix_context.genome import Genome as KnowledgeStore
from helix_context.ribosome import Ribosome as Compressor


__all__ = [
    "Compressor",        # was Ribosome
    "Document",          # was Gene
    "DocumentAttribution",  # was GeneAttribution
    "DocumentSignals",   # was EpigeneticMarkers
    "DocumentTags",      # was PromoterTags
    "KnowledgeStore",    # was Genome
    "LifecycleTier",     # was ChromatinState
]


# Per-alias provenance comments, useful for code-search tools that surface
# rename history. Read once at import; not used at runtime.
_RENAME_LOG = {
    "Document": "renamed from Gene (helix_context.schemas)",
    "DocumentAttribution": "renamed from GeneAttribution (helix_context.schemas)",
    "DocumentSignals": "renamed from EpigeneticMarkers (helix_context.schemas)",
    "DocumentTags": "renamed from PromoterTags (helix_context.schemas)",
    "KnowledgeStore": "renamed from Genome (helix_context.genome)",
    "LifecycleTier": "renamed from ChromatinState (helix_context.schemas)",
    "Compressor": "renamed from Ribosome (helix_context.ribosome)",
}

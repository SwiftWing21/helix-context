"""
Cymatics — Frequency-domain context compression.

Maps the biological metaphors of helix-context onto wave physics:
    Gene          → Resonant mode (excited by query "frequencies")
    Codon weight  → Spectral amplitude
    Co-activation → Harmonic coupling
    Chromatin     → Mode damping level
    Splice        → Bandwidth filtering (Q-factor)
    Decay score   → Exponential amplitude decay

Core idea: instead of asking an LLM to judge relevance (re_rank)
and select codons (splice), compute interference patterns between
query and gene frequency spectra. Cosine similarity on 256-bin
spectra replaces two LLM calls (~2-4s) with CPU math (~5ms).

Numpy is optional — pure-Python fallback is always available.
"""

from __future__ import annotations

import hashlib
import logging
import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .schemas import Gene

log = logging.getLogger("helix.cymatics")


# ── Query signal extraction (inlined from accel.py) ────────────────

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "about", "above",
    "after", "again", "all", "also", "and", "any", "because", "before",
    "between", "both", "but", "by", "came", "come", "each", "for", "from",
    "get", "got", "her", "here", "him", "his", "how", "into", "its",
    "just", "like", "make", "many", "more", "most", "much", "not", "now",
    "only", "other", "our", "out", "over", "said", "she", "some", "than",
    "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "too", "under", "use", "very", "want",
    "was", "way", "what", "when", "where", "which", "who", "why", "with",
    "you", "your",
})
_STRIP_CHARS = ".,;:!?'\"()[]{}<>/@#$%^&*+=~`|\\—–-"


def extract_query_signals(query: str) -> Tuple[List[str], List[str]]:
    """Fast keyword extraction from query for promoter matching."""
    words = query.lower().split()
    keywords = []
    for w in words:
        stripped = w.strip(_STRIP_CHARS)
        if stripped and len(stripped) > 2 and stripped not in _STOP_WORDS:
            keywords.append(stripped)
    entities = [w for w in keywords if len(w) > 4 or (w and w[0].isupper())]
    domains = keywords[:5]
    return domains, entities


# ── Numpy detection (follows orjson pattern from accel.py) ─────────

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

MATH_BACKEND = "numpy" if _HAS_NUMPY else "python"


# ── Constants ──────────────────────────────────────────────────────

N_BINS = 256              # Spectrum resolution (fixed-width, <2KB per spectrum)
_HASH_SEED = b"helix"     # Deterministic seed for term→frequency mapping


# ── Section 1: Frequency Space ─────────────────────────────────────

def term_to_frequency(term: str) -> int:
    """
    Map a term to a deterministic frequency bin (0 to N_BINS-1).

    Uses MD5 for cross-platform consistency (Python's hash() varies
    between runs with PYTHONHASHSEED). Same term always lands at
    the same bin.
    """
    h = hashlib.md5(_HASH_SEED + term.lower().encode("utf-8")).digest()
    return int.from_bytes(h[:2], "little") % N_BINS


def _gaussian_peak(center: int, amplitude: float, width: float) -> List[float]:
    """
    Generate a Gaussian peak centered at `center` with given amplitude and width.

    Width is the standard deviation in bins. Controls Q-factor:
    narrow width = high Q (selective), broad width = low Q (resonant).
    """
    spectrum = [0.0] * N_BINS
    # Only compute within ±4 sigma (beyond that, contribution < 0.01%)
    radius = int(width * 4) + 1
    for i in range(max(0, center - radius), min(N_BINS, center + radius + 1)):
        dist = (i - center) / max(width, 0.1)
        spectrum[i] = amplitude * math.exp(-0.5 * dist * dist)
    return spectrum


def build_spectrum(
    terms: List[str],
    weights: Optional[List[float]] = None,
    decay: float = 1.0,
    peak_width: float = 3.0,
) -> List[float]:
    """
    Superpose terms onto a 256-bin frequency spectrum.

    Each term becomes a Gaussian peak at its hashed frequency bin.
    Multiple terms at nearby frequencies constructively interfere.

    Args:
        terms: Semantic labels (promoter domains, entities, codon meanings)
        weights: Amplitude per term (default 1.0 each)
        decay: Global damping factor from EpigeneticMarkers.decay_score
        peak_width: Gaussian width in bins (from Q-factor mapping)
    """
    if not terms:
        return [0.0] * N_BINS

    if weights is None:
        weights = [1.0] * len(terms)

    if _HAS_NUMPY:
        return _build_spectrum_numpy(terms, weights, decay, peak_width)

    # Pure-Python fallback
    spectrum = [0.0] * N_BINS
    for term, weight in zip(terms, weights):
        freq = term_to_frequency(term)
        amplitude = weight * decay
        peak = _gaussian_peak(freq, amplitude, peak_width)
        for i in range(N_BINS):
            spectrum[i] += peak[i]
    return spectrum


def _build_spectrum_numpy(
    terms: List[str],
    weights: List[float],
    decay: float,
    peak_width: float,
) -> List[float]:
    """Vectorized spectrum construction using numpy."""
    spectrum = np.zeros(N_BINS, dtype=np.float64)
    bins = np.arange(N_BINS, dtype=np.float64)

    for term, weight in zip(terms, weights):
        freq = term_to_frequency(term)
        amplitude = weight * decay
        dist = (bins - freq) / max(peak_width, 0.1)
        spectrum += amplitude * np.exp(-0.5 * dist * dist)

    return spectrum.tolist()


def query_spectrum(
    query: str,
    synonym_map: Optional[Dict[str, List[str]]] = None,
    peak_width: float = 3.0,
) -> List[float]:
    """
    Build a frequency spectrum from a raw query string.

    Uses extract_query_signals() from accel for terms, then applies
    synonym expansion as harmonic overtones (synonyms at 0.5 amplitude).
    """
    domains, entities = extract_query_signals(query)
    all_terms = domains + entities

    if not all_terms:
        return [0.0] * N_BINS

    terms: List[str] = []
    weights: List[float] = []

    for t in all_terms:
        terms.append(t)
        weights.append(1.0)

        # Synonym expansion as harmonic overtones
        if synonym_map:
            key = t.lower()
            if key in synonym_map:
                for syn in synonym_map[key]:
                    terms.append(syn)
                    weights.append(0.5)  # First harmonic = half amplitude

    return build_spectrum(terms, weights, decay=1.0, peak_width=peak_width)


def gene_spectrum(
    gene: Gene,
    peak_width: float = 3.0,
) -> List[float]:
    """
    Build a frequency spectrum from a Gene's promoter tags and decay state.

    Domains and entities become spectral peaks, damped by decay_score.
    """
    terms = list(gene.promoter.domains) + list(gene.promoter.entities)
    if not terms:
        # Fall back to codon meanings if no promoter tags
        terms = list(gene.codons[:5])

    decay = gene.epigenetics.decay_score
    return build_spectrum(terms, decay=decay, peak_width=peak_width)


@lru_cache(maxsize=512)
def _cached_gene_spectrum(
    gene_id: str,
    domains_key: str,
    entities_key: str,
    decay_score: float,
    peak_width: float,
) -> Tuple[float, ...]:
    """
    LRU-cached gene spectrum computation.

    Returns tuple (immutable) for cache compatibility.
    Key is composite of gene identity + semantic content + decay state.
    """
    terms = list(domains_key.split("|")) + list(entities_key.split("|"))
    terms = [t for t in terms if t]  # filter empty strings
    spectrum = build_spectrum(terms, decay=decay_score, peak_width=peak_width)
    return tuple(spectrum)


def cached_gene_spectrum(gene: Gene, peak_width: float = 3.0) -> List[float]:
    """Get a gene's spectrum, using LRU cache for repeated access."""
    domains_key = "|".join(sorted(gene.promoter.domains))
    entities_key = "|".join(sorted(gene.promoter.entities))
    t = _cached_gene_spectrum(
        gene.gene_id, domains_key, entities_key,
        round(gene.epigenetics.decay_score, 2),  # Round for cache stability
        peak_width,
    )
    return list(t)


def clear_spectrum_cache() -> None:
    """Clear the gene spectrum LRU cache. Call after genome mutations."""
    _cached_gene_spectrum.cache_clear()


# ── Section 2: Resonance Scoring ───────────────────────────────────

def resonance_score(spec_a: List[float], spec_b: List[float]) -> float:
    """
    Compute resonance between two spectra via cosine similarity.

    Two spectra that share frequency peaks will constructively interfere
    (high score). Spectra with peaks in different bins produce low scores
    (destructive interference / no interaction).

    Returns 0.0-1.0.
    """
    if _HAS_NUMPY:
        a = np.array(spec_a)
        b = np.array(spec_b)
        mag_a = np.linalg.norm(a)
        mag_b = np.linalg.norm(b)
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return float(np.dot(a, b) / (mag_a * mag_b))

    # Pure-Python fallback
    dot = sum(a * b for a, b in zip(spec_a, spec_b))
    mag_a = math.sqrt(sum(a * a for a in spec_a))
    mag_b = math.sqrt(sum(b * b for b in spec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def build_weight_vector(
    query: str,
    synonym_map: Optional[Dict[str, List[str]]] = None,
    peak_width: float = 3.0,
    amplify: float = 1.5,
    synonym_amplify: float = 1.2,
    baseline: float = 0.8,
) -> List[float]:
    """
    Build a query-adaptive 256-bin weight vector (the dA⃗ surface).

    Bins near query term frequencies are amplified (1.5x default).
    Bins near synonym frequencies get moderate boost (1.2x).
    All other bins get dampened baseline (0.8x).

    The result is a smooth Gaussian envelope — not binary gates.
    This is the discrete approximation of ∫ B⃗ · dA⃗ where the
    surface area varies by frequency region.
    """
    weights = [baseline] * N_BINS
    domains, entities = extract_query_signals(query)
    all_terms = domains + entities

    if not all_terms:
        return weights

    # Amplify bins near query term frequencies
    for term in all_terms:
        freq = term_to_frequency(term)
        radius = int(peak_width * 3) + 1
        for i in range(max(0, freq - radius), min(N_BINS, freq + radius + 1)):
            dist = (i - freq) / max(peak_width, 0.1)
            boost = (amplify - baseline) * math.exp(-0.5 * dist * dist)
            weights[i] = max(weights[i], baseline + boost)

    # Synonym expansion — moderate boost
    if synonym_map:
        for term in all_terms:
            key = term.lower()
            if key in synonym_map:
                for syn in synonym_map[key]:
                    freq = term_to_frequency(syn)
                    radius = int(peak_width * 2) + 1
                    for i in range(max(0, freq - radius), min(N_BINS, freq + radius + 1)):
                        dist = (i - freq) / max(peak_width, 0.1)
                        boost = (synonym_amplify - baseline) * math.exp(-0.5 * dist * dist)
                        weights[i] = max(weights[i], baseline + boost)

    return weights


def flux_score(
    spec_a: List[float],
    spec_b: List[float],
    weights: List[float],
) -> float:
    """
    Weighted cosine similarity — the discrete flux integral.

    Φ = ∫ B⃗ · dA⃗  ≈  dot(a*w, b*w) / (|a*w| * |b*w|)

    When weights are uniform, this equals resonance_score().
    When weights amplify query-relevant bins, domain-matched
    genes score higher than spectrally-distant ones.

    Returns 0.0-1.0.
    """
    if _HAS_NUMPY:
        a = np.array(spec_a)
        b = np.array(spec_b)
        w = np.array(weights)
        aw = a * w
        bw = b * w
        mag_aw = np.linalg.norm(aw)
        mag_bw = np.linalg.norm(bw)
        if mag_aw == 0 or mag_bw == 0:
            return 0.0
        return float(np.dot(aw, bw) / (mag_aw * mag_bw))

    # Pure-Python fallback
    aw = [a * w for a, w in zip(spec_a, weights)]
    bw = [b * w for b, w in zip(spec_b, weights)]
    dot = sum(a * b for a, b in zip(aw, bw))
    mag_aw = math.sqrt(sum(a * a for a in aw))
    mag_bw = math.sqrt(sum(b * b for b in bw))
    if mag_aw == 0 or mag_bw == 0:
        return 0.0
    return dot / (mag_aw * mag_bw)


def resonance_rank(
    query: str,
    candidates: List[Gene],
    k: int = 5,
    synonym_map: Optional[Dict[str, List[str]]] = None,
    peak_width: float = 3.0,
    use_flux: bool = True,
) -> List[Gene]:
    """
    Rank candidate genes by resonance with the query spectrum.

    Drop-in replacement for Ribosome.re_rank(). Builds query spectrum
    once, scores all candidates via cosine similarity, returns top-k.

    When use_flux=True (default), uses adaptive bin weighting via
    flux_score() — the discrete ∫ B⃗ · dA⃗ that amplifies query-
    relevant frequency regions. Falls back to flat resonance_score()
    when use_flux=False.

    Preserves the lost-in-the-middle guard: if fewer than 50% of
    candidates score above 0.2, pad with unscored candidates.
    """
    if not candidates:
        return []

    if len(candidates) <= k:
        return candidates

    q_spec = query_spectrum(query, synonym_map=synonym_map, peak_width=peak_width)

    # Build adaptive weight vector if using flux scoring
    weights = None
    if use_flux:
        weights = build_weight_vector(query, synonym_map=synonym_map, peak_width=peak_width)

    scored: List[Tuple[float, Gene]] = []
    for gene in candidates:
        g_spec = cached_gene_spectrum(gene, peak_width=peak_width)
        if weights:
            score = flux_score(q_spec, g_spec, weights)
        else:
            score = resonance_score(q_spec, g_spec)
        if score > 0.05:  # Noise floor
            scored.append((score, gene))

    # Lost-in-the-middle guard (same as ribosome.re_rank)
    if len(scored) < len(candidates) * 0.5:
        scored_ids = {g.gene_id for _, g in scored}
        for gene in candidates:
            if gene.gene_id not in scored_ids and len(scored) < k:
                scored.append((0.1, gene))  # Default score for padded genes

    scored.sort(key=lambda x: x[0], reverse=True)
    return [g for _, g in scored[:k]]


# ── Section 3: Interference Splice ─────────────────────────────────

def interference_splice(
    query: str,
    genes: List[Gene],
    splice_aggressiveness: float = 0.3,
    synonym_map: Optional[Dict[str, List[str]]] = None,
    peak_width: float = 3.0,
    min_codons_kept: int = 2,
) -> Dict[str, str]:
    """
    Splice genes using frequency interference instead of an LLM.

    For each gene, for each codon meaning, compute resonance between
    the codon's mini-spectrum and the query spectrum. Codons that
    constructively interfere (above threshold) survive as exons.
    Those below threshold are spliced as introns.

    Preserves Fix 2 (empty splice guard) and Fix 4 (complement fallback).

    Returns {gene_id: spliced_text} — same format as Ribosome.splice().
    """
    if not genes:
        return {}

    threshold = splice_aggressiveness * 0.7
    q_spec = query_spectrum(query, synonym_map=synonym_map, peak_width=peak_width)

    result: Dict[str, str] = {}

    for gene in genes:
        if not gene.codons:
            result[gene.gene_id] = gene.complement or gene.content[:500]
            continue

        # Score each codon against the query
        kept: List[str] = []
        for codon_meaning in gene.codons:
            codon_spec = build_spectrum([codon_meaning], peak_width=peak_width)
            score = resonance_score(q_spec, codon_spec)
            if score >= threshold:
                kept.append(codon_meaning)

        # Fix 2: empty splice guard
        if not kept and gene.codons:
            kept = gene.codons[:min_codons_kept]
            log.info(
                "Empty cymatics splice for gene %s, keeping first %d codons",
                gene.gene_id, len(kept),
            )

        if kept:
            result[gene.gene_id] = " | ".join(kept)
        else:
            # Total miss — fall back to complement (Fix 4)
            result[gene.gene_id] = gene.complement or gene.content[:500]

    # Handle genes missing from result
    for gene in genes:
        if gene.gene_id not in result:
            result[gene.gene_id] = gene.complement or gene.content[:500]

    return result


# ── Section 4: Harmonic Co-activation ──────────────────────────────

def harmonic_weight(gene_a: Gene, gene_b: Gene, peak_width: float = 3.0) -> float:
    """
    Compute harmonic coupling strength between two genes.

    Converts the binary co_activated_with link into a weighted edge.
    High weight = spectrally similar (same resonant frequencies).
    Low weight = co-occurred but spectrally dissimilar.
    """
    spec_a = cached_gene_spectrum(gene_a, peak_width=peak_width)
    spec_b = cached_gene_spectrum(gene_b, peak_width=peak_width)
    return resonance_score(spec_a, spec_b)


def compute_harmonic_weights(
    genes: List[Gene],
    peak_width: float = 3.0,
) -> List[Tuple[str, str, float]]:
    """
    Compute pairwise harmonic weights for a set of expressed genes.

    Returns list of (gene_id_a, gene_id_b, weight) tuples.
    Only returns pairs where weight > 0.1 (above noise floor).
    """
    if len(genes) < 2:
        return []

    weights: List[Tuple[str, str, float]] = []
    for i, ga in enumerate(genes):
        for gb in genes[i + 1:]:
            w = harmonic_weight(ga, gb, peak_width=peak_width)
            if w > 0.1:
                weights.append((ga.gene_id, gb.gene_id, w))
    return weights


# ── Section 5: Q-factor Mapping ────────────────────────────────────

def aggressiveness_to_peak_width(splice_aggressiveness: float) -> float:
    """
    Map splice_aggressiveness (0.0-1.0) to Gaussian peak width (bins).

    Low aggressiveness (0.0) → width=2.0 (moderate resonance, 0.44 dynamic range)
    High aggressiveness (1.0) → width=0.5 (sharp peaks, high selectivity)

    Range 0.5-2.0 keeps ALL operating points in the useful zone where
    dynamic range > 0.4 (R3 research: width >= 3.0 collapses to 0.22).
    The 0.5 floor gives 2x headroom below the 1.0 sweet spot.
    """
    return max(0.5, 2.0 - 1.5 * splice_aggressiveness)


# ── Diagnostics ────────────────────────────────────────────────────

def cymatics_info() -> Dict:
    """Report cymatics module status."""
    return {
        "math_backend": MATH_BACKEND,
        "n_bins": N_BINS,
        "spectrum_cache_size": _cached_gene_spectrum.cache_info().maxsize,
        "spectrum_cache_hits": _cached_gene_spectrum.cache_info().hits,
        "spectrum_cache_misses": _cached_gene_spectrum.cache_info().misses,
    }

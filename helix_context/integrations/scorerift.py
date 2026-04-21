"""
Helix-ScoreRift Integration -- The CD Spectroscope.

Bridges Helix Context (genome memory) with ScoreRift (divergence detection)
using a Circular Dichroism (CD) metaphor:

    epsilon_L (left beam)  = automated engine score (ScoreRift auto)
    epsilon_R (right beam) = manual/qualitative assessment (ScoreRift manual)
    delta_epsilon (CD signal) = structural anomaly in context health

Three integration points, zero coupling:
    1. GenomeHealthProbe  -- ScoreRift dimension that probes the Helix genome
    2. cd_signal()        -- the divergence math (delta_epsilon)
    3. resolution_to_gene() -- packages divergence resolutions as Helix genes

Usage with ScoreRift:
    from scorerift import AuditEngine, Tier
    from helix_context.integrations.scorerift import make_genome_dimensions

    engine = AuditEngine()
    engine.register_many(make_genome_dimensions("http://127.0.0.1:11437"))
    results = engine.run_tier("light")

Usage standalone (without ScoreRift installed):
    from helix_context.integrations.scorerift import GenomeHealthProbe

    probe = GenomeHealthProbe("http://127.0.0.1:11437")
    report = probe.full_scan()
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("helix.scorerift")


# -- CD Signal Math ---------------------------------------------------

@dataclass
class CDSignal:
    """Circular Dichroism measurement for a single dimension."""
    dimension: str
    epsilon_l: float          # Left beam (auto score, 0-1)
    epsilon_r: Optional[float]  # Right beam (manual score, 0-1 or None)
    delta_epsilon: float      # |L - R| -- the divergence magnitude
    ellipticity: float        # Structural purity (0=distorted, 1=aligned)
    status: str               # "aligned", "diverged", "denatured", "unmeasured"


def cd_signal(
    auto_score: float,
    manual_score: Optional[float],
    divergence_threshold: float = 0.15,
    denaturation_threshold: float = 0.4,
) -> CDSignal:
    """
    Compute the CD signal between automated and manual assessments.

    The delta_epsilon is the raw divergence. Ellipticity is a normalized
    measure of structural purity (1.0 = perfect agreement, 0.0 = total
    denaturation).

    Status levels:
        aligned    -- beams agree within threshold (healthy)
        diverged   -- beams disagree significantly (investigate)
        denatured  -- total structural failure (context is useless)
        unmeasured -- no manual score available (single-beam only)
    """
    if manual_score is None:
        return CDSignal(
            dimension="",
            epsilon_l=auto_score,
            epsilon_r=None,
            delta_epsilon=0.0,
            ellipticity=1.0,
            status="unmeasured",
        )

    delta = abs(auto_score - manual_score)

    # Ellipticity: 1.0 when perfectly aligned, decays exponentially with divergence
    # Uses a sigmoid-like curve so small divergences barely affect it
    ellipticity = math.exp(-3.0 * delta * delta)

    if delta >= denaturation_threshold:
        status = "denatured"
    elif delta >= divergence_threshold:
        status = "diverged"
    else:
        status = "aligned"

    return CDSignal(
        dimension="",
        epsilon_l=auto_score,
        epsilon_r=manual_score,
        delta_epsilon=delta,
        ellipticity=ellipticity,
        status=status,
    )


# -- Genome Health Probe ----------------------------------------------

@dataclass
class GenomeHealthProbe:
    """
    Probes a running Helix server to assess genome health.

    Checks:
        - genome_freshness: ratio of OPEN genes to total (are genes being accessed?)
        - compression_quality: is the compression ratio in a healthy range?
        - gene_coverage: does the genome have enough genes to be useful?
        - context_relevance: given a test query, does the genome return useful context?
    """
    helix_url: str = "http://127.0.0.1:11437"
    timeout: float = 30.0
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self):
        self._client = httpx.Client(base_url=self.helix_url, timeout=self.timeout)

    # -- Individual checks (ScoreRift dimension format) ---------------

    def check_freshness(self) -> tuple[float, dict]:
        """Score based on what fraction of genes are in OPEN chromatin state."""
        stats = self._stats()
        total = stats.get("total_genes", 0)
        if total == 0:
            return (0.0, {"reason": "empty genome", "total_genes": 0})

        open_genes = stats.get("open", 0)
        ratio = open_genes / total

        # Perfect: >50% open (recently accessed). Failing: <10% open (all stale).
        score = min(1.0, ratio / 0.5)
        return (score, {
            "open": open_genes,
            "total": total,
            "open_ratio": round(ratio, 3),
        })

    def check_compression(self) -> tuple[float, dict]:
        """Score based on compression ratio being in a healthy range."""
        stats = self._stats()
        ratio = stats.get("compression_ratio", 0)

        if ratio == 0:
            return (0.0, {"reason": "no data", "compression_ratio": 0})

        # Sweet spot: 3x-10x compression. Below 2x = barely compressing.
        # Above 15x = might be losing information.
        if ratio < 2.0:
            score = ratio / 2.0  # Linear ramp from 0 to 1
        elif ratio <= 10.0:
            score = 1.0          # Sweet spot
        else:
            # Gentle penalty for over-compression
            score = max(0.5, 1.0 - (ratio - 10.0) / 20.0)

        return (score, {
            "compression_ratio": round(ratio, 2),
            "raw_chars": stats.get("total_chars_raw", 0),
            "compressed_chars": stats.get("total_chars_compressed", 0),
        })

    def check_coverage(self) -> tuple[float, dict]:
        """Score based on whether the genome has enough genes to be useful."""
        stats = self._stats()
        total = stats.get("total_genes", 0)

        # Cold start: <5 genes is barely functional. 20+ is healthy.
        if total == 0:
            score = 0.0
        elif total < 5:
            score = total / 10.0  # 0.1 to 0.4
        elif total < 20:
            score = 0.5 + (total - 5) / 30.0  # 0.5 to 1.0
        else:
            score = 1.0

        return (score, {"total_genes": total})

    def check_relevance(self, test_query: str = "How does the system work?") -> tuple[float, dict]:
        """Score based on whether the genome returns context for a test query."""
        try:
            resp = self._client.post("/context", json={"query": test_query})
            if resp.status_code != 200:
                return (0.0, {"error": f"HTTP {resp.status_code}"})

            data = resp.json()
            # /context returns a dict (Continue HTTP context provider format),
            # not a list. Legacy code treated it as a list and silently failed
            # when the first entry was missing.
            if not data or not isinstance(data, dict):
                return (0.0, {"reason": "empty response"})

            content = data.get("content", "")

            if "no relevant context" in content.lower():
                return (0.2, {"reason": "no matching genes", "query": test_query})

            # Score by content richness
            content_len = len(content)
            if content_len < 100:
                score = 0.4
            elif content_len < 500:
                score = 0.7
            else:
                score = 1.0

            return (score, {
                "query": test_query,
                "response_length": content_len,
                "genes_expressed": data.get("description", "") or data.get("name", ""),
            })

        except Exception as exc:
            log.warning("check_relevance failed for %s", test_query, exc_info=True)
            return (0.0, {"error": str(exc)})

    # -- Full scan ----------------------------------------------------

    def full_scan(self, test_query: str = "How does the system work?") -> dict:
        """Run all genome health checks and compute aggregate CD signals."""
        checks = {
            "freshness": self.check_freshness(),
            "compression": self.check_compression(),
            "coverage": self.check_coverage(),
            "relevance": self.check_relevance(test_query),
        }

        signals = {}
        total_ellipticity = 0.0
        count = 0

        for name, (score, detail) in checks.items():
            sig = cd_signal(score, None)  # No manual score yet (single-beam)
            sig.dimension = name
            signals[name] = sig

            total_ellipticity += sig.ellipticity
            count += 1

        avg_ellipticity = total_ellipticity / max(count, 1)

        return {
            "checks": {n: {"score": s, "detail": d} for n, (s, d) in checks.items()},
            "signals": {n: {"delta": s.delta_epsilon, "ellipticity": s.ellipticity, "status": s.status}
                        for n, s in signals.items()},
            "aggregate_ellipticity": round(avg_ellipticity, 4),
            "genome_healthy": avg_ellipticity > 0.7,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # -- Internal -----------------------------------------------------

    def _stats(self) -> dict:
        try:
            resp = self._client.get("/stats")
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            log.warning("Failed to reach Helix at %s", self.helix_url, exc_info=True)
            return {}

    def close(self):
        self._client.close()


# -- Resolution -> Gene Pipeline --------------------------------------

def resolution_to_gene(
    dimension: str,
    auto_score: float,
    manual_score: float,
    resolution: str,
    resolution_type: str = "update_manual",
    helix_url: str = "http://127.0.0.1:11437",
    timeout: float = 60.0,
) -> Optional[str]:
    """
    Package a divergence resolution as a Helix gene.

    When a ScoreRift divergence is resolved (manual grade updated,
    acknowledged, or re-audited), this function ingests the resolution
    context back into the genome so the system learns from it.

    Returns the gene_id if successful, None on failure.
    """
    content = (
        f"ScoreRift Divergence Resolution\n"
        f"Dimension: {dimension}\n"
        f"Auto score: {auto_score:.3f}\n"
        f"Manual score: {manual_score:.3f}\n"
        f"Delta: {abs(auto_score - manual_score):.3f}\n"
        f"Resolution type: {resolution_type}\n"
        f"Resolution: {resolution}\n"
        f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}"
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{helix_url}/ingest",
                json={
                    "content": content,
                    "content_type": "text",
                    "metadata": {
                        "source": "scorerift_resolution",
                        "dimension": dimension,
                        "resolution_type": resolution_type,
                        "cd_delta": abs(auto_score - manual_score),
                    },
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                gene_ids = data.get("gene_ids", [])
                log.info("Resolution for %s ingested as %d genes", dimension, len(gene_ids))
                return gene_ids[0] if gene_ids else None
            else:
                log.warning("Ingest failed: HTTP %d", resp.status_code)
                return None
    except Exception:
        log.warning("Failed to ingest resolution for %s", dimension, exc_info=True)
        return None


# -- ScoreRift Dimension Factory --------------------------------------

def make_genome_dimensions(
    helix_url: str = "http://127.0.0.1:11437",
    test_query: str = "How does the system work?",
) -> list:
    """
    Create ScoreRift Dimension objects for genome health monitoring.

    Returns a list ready for engine.register_many().
    Requires scorerift to be installed (imports at call time, not module level).

    Usage:
        from scorerift import AuditEngine
        from helix_context.integrations.scorerift import make_genome_dimensions

        engine = AuditEngine()
        engine.register_many(make_genome_dimensions())
    """
    # Lazy import — scorerift may not be installed
    from scorerift import Dimension, Tier

    probe = GenomeHealthProbe(helix_url=helix_url)

    return [
        Dimension(
            name="genome_freshness",
            check=probe.check_freshness,
            confidence=0.85,
            tier=Tier.LIGHT,
            description="Fraction of genome genes in OPEN chromatin (recently accessed)",
        ),
        Dimension(
            name="genome_compression",
            check=probe.check_compression,
            confidence=0.90,
            tier=Tier.LIGHT,
            description="Compression ratio health (sweet spot: 3x-10x)",
        ),
        Dimension(
            name="genome_coverage",
            check=probe.check_coverage,
            confidence=0.80,
            tier=Tier.LIGHT,
            description="Gene count adequacy (cold start detection)",
        ),
        Dimension(
            name="genome_relevance",
            check=lambda: probe.check_relevance(test_query),
            confidence=0.70,
            tier=Tier.DAILY,
            description="Context quality for a representative query",
        ),
    ]

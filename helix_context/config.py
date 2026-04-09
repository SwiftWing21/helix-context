"""
Config — Epigenetic environment.

Loads helix.toml and exposes typed configuration for all modules.
Falls back to sensible defaults if no config file exists.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class RibosomeConfig:
    model: str = "auto"
    base_url: str = "http://localhost:11434"
    timeout: float = 10.0
    keep_alive: str = "30m"     # How long Ollama keeps the ribosome model loaded
    warmup: bool = True         # Pre-load model on server start
    backend: str = "ollama"     # "ollama" or "deberta" (hybrid: DeBERTa for re_rank/splice)
    rerank_model_path: str = "training/models/rerank"
    splice_model_path: str = "training/models/splice"
    splice_threshold: float = 0.5
    nli_model_path: str = "training/models/nli"
    nli_splice_bonus: float = 0.15       # Prob bonus for entailment-linked codons
    nli_splice_penalty: float = 0.15     # Prob penalty for alternation-linked codons
    device: str = "auto"        # "auto", "cpu", "cuda"


@dataclass
class BudgetConfig:
    ribosome_tokens: int = 3000
    expression_tokens: int = 6000
    max_genes_per_turn: int = 8
    splice_aggressiveness: float = 0.5
    decoder_mode: str = "full"  # "full"|"condensed"|"minimal"|"none"


@dataclass
class GenomeConfig:
    path: str = "genome.db"
    compact_interval: float = 3600.0    # Seconds between source-change checks
    cold_start_threshold: int = 10      # Fix 3: genes needed before history stripping


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 11437
    upstream: str = "http://localhost:11434"
    upstream_timeout: float = 120.0     # Timeout for proxied requests to Ollama


@dataclass
class HelixConfig:
    ribosome: RibosomeConfig = field(default_factory=RibosomeConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    synonym_map: Dict[str, List[str]] = field(default_factory=dict)


def load_config(path: Optional[str] = None) -> HelixConfig:
    """
    Load helix.toml from the given path, or auto-discover from cwd / env.
    Returns defaults if no config file is found.
    """
    if path is None:
        path = os.environ.get("HELIX_CONFIG", "helix.toml")

    config_path = Path(path)
    if not config_path.exists():
        log.info("No config file at %s, using defaults", path)
        return HelixConfig()

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    cfg = HelixConfig()

    # Ribosome
    if "ribosome" in raw:
        r = raw["ribosome"]
        cfg.ribosome = RibosomeConfig(
            model=r.get("model", cfg.ribosome.model),
            base_url=r.get("base_url", cfg.ribosome.base_url),
            timeout=float(r.get("timeout", cfg.ribosome.timeout)),
            keep_alive=r.get("keep_alive", cfg.ribosome.keep_alive),
            warmup=r.get("warmup", cfg.ribosome.warmup),
            backend=r.get("backend", cfg.ribosome.backend),
            rerank_model_path=r.get("rerank_model_path", cfg.ribosome.rerank_model_path),
            splice_model_path=r.get("splice_model_path", cfg.ribosome.splice_model_path),
            splice_threshold=float(r.get("splice_threshold", cfg.ribosome.splice_threshold)),
            nli_model_path=r.get("nli_model_path", cfg.ribosome.nli_model_path),
            nli_splice_bonus=float(r.get("nli_splice_bonus", cfg.ribosome.nli_splice_bonus)),
            nli_splice_penalty=float(r.get("nli_splice_penalty", cfg.ribosome.nli_splice_penalty)),
            device=r.get("device", cfg.ribosome.device),
        )

    # Budget
    if "budget" in raw:
        b = raw["budget"]
        cfg.budget = BudgetConfig(
            ribosome_tokens=b.get("ribosome_tokens", cfg.budget.ribosome_tokens),
            expression_tokens=b.get("expression_tokens", cfg.budget.expression_tokens),
            max_genes_per_turn=b.get("max_genes_per_turn", cfg.budget.max_genes_per_turn),
            splice_aggressiveness=float(b.get("splice_aggressiveness", cfg.budget.splice_aggressiveness)),
            decoder_mode=b.get("decoder_mode", cfg.budget.decoder_mode),
        )

    # Genome
    if "genome" in raw:
        g = raw["genome"]
        cfg.genome = GenomeConfig(
            path=g.get("path", cfg.genome.path),
            compact_interval=float(g.get("compact_interval", cfg.genome.compact_interval)),
            cold_start_threshold=int(g.get("cold_start_threshold", cfg.genome.cold_start_threshold)),
        )

    # Server
    if "server" in raw:
        s = raw["server"]
        cfg.server = ServerConfig(
            host=s.get("host", cfg.server.host),
            port=int(s.get("port", cfg.server.port)),
            upstream=s.get("upstream", cfg.server.upstream),
            upstream_timeout=float(s.get("upstream_timeout", cfg.server.upstream_timeout)),
        )

    # Fix 1: synonym map
    if "synonyms" in raw:
        cfg.synonym_map = {
            k: list(v) for k, v in raw["synonyms"].items()
        }

    log.info("Config loaded from %s", config_path)
    return cfg

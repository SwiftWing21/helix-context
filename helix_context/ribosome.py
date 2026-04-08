"""
Ribosome — The universal decoder.

Biology:
    The ribosome reads mRNA codons and assembles proteins.
    It doesn't "understand" the protein — it's a mechanical translator.

    Our ribosome is a small (2-4B) model running on CPU.
    Four operations:
        pack      — raw text → codons + promoter tags + complement
        re_rank   — score candidate genes against a query
        splice    — remove introns, keep exons (BATCHED single call)
        replicate — encode a query+response exchange for genome storage

    The ribosome is DUMB but CONSISTENT. Same input → same output.
    The intelligence lives in the big model; the ribosome is firmware.

Fixes incorporated:
    Fix 2 — Empty splice guard: if ribosome returns empty for a gene,
            keep first N codons or fall back to complement
    Fix 4 — Timeout fallback: httpx timeout on all model calls,
            catch and fall back to deterministic ordering / raw complement
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Protocol

import httpx

from .codons import CodonEncoder
from .exceptions import FoldingError, TranscriptionError
from .schemas import EpigeneticMarkers, Gene, PromoterTags

log = logging.getLogger(__name__)


# ── Model backend protocol ──────────────────────────────────────────

class ModelBackend(Protocol):
    """Interface for the small model. Swap Ollama, llama.cpp, vLLM, etc."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        """Generate a completion. Must return raw text."""
        ...


# ── Ollama backend ──────────────────────────────────────────────────

class OllamaBackend:
    """Talk to a local Ollama instance.

    Uses keep_alive to pin the ribosome model in memory so Ollama
    doesn't unload it every time the big model runs. This eliminates
    the 10-30s model-swap latency on each turn.
    """

    def __init__(
        self,
        model: str = "auto",
        base_url: str = "http://localhost:11434",
        timeout: float = 10.0,
        keep_alive: str = "30m",
        warmup: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.client = httpx.Client(timeout=timeout)

        if model == "auto":
            self.model = self._auto_detect()
        else:
            self.model = model

        if warmup:
            self._warmup()

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        resp = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def _warmup(self) -> None:
        """Pre-load the ribosome model into Ollama's memory.

        Sends a minimal generate request with keep_alive so the model
        stays resident. Subsequent calls skip the cold-load entirely.
        """
        try:
            self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": self.keep_alive,
                },
                timeout=60,
            )
            log.info("Ribosome model %s warmed up (keep_alive=%s)", self.model, self.keep_alive)
        except Exception:
            log.warning("Ribosome warmup failed (non-fatal)", exc_info=True)

    def _auto_detect(self) -> str:
        """Query Ollama /api/tags, prefer gemma family, fall back to smallest."""
        try:
            resp = self.client.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])

            if not models:
                log.warning("No Ollama models found, defaulting to gemma3:4b")
                return "gemma3:4b"

            # Prefer gemma family (good at structured JSON output)
            gemma = [m for m in models if "gemma" in m.get("name", "").lower()]
            if gemma:
                # Pick smallest gemma model (best for CPU ribosome duty)
                gemma.sort(key=lambda m: m.get("size", float("inf")))
                pick = gemma[0]["name"]
                log.info("Auto-detected ribosome model: %s (gemma family)", pick)
                return pick

            # Fall back to smallest available model
            models.sort(key=lambda m: m.get("size", float("inf")))
            pick = models[0]["name"]
            log.info("Auto-detected ribosome model: %s (smallest available)", pick)
            return pick

        except Exception:
            log.warning("Ollama auto-detect failed, defaulting to gemma3:4b", exc_info=True)
            return "gemma3:4b"


# ── System prompts ──────────────────────────────────────────────────

_PACK_SYSTEM = """You are a context compression engine. You receive raw text and produce structured JSON.
You must respond ONLY with valid JSON, no markdown fences, no explanation.

Output schema:
{
  "codons": [
    {"meaning": "short semantic label", "weight": 0.0-1.0, "is_exon": true/false}
  ],
  "complement": "one-paragraph compressed representation of the full content",
  "promoter": {
    "domains": ["topic1", "topic2"],
    "entities": ["specific_thing1", "specific_thing2"],
    "intent": "what this content is about / used for",
    "summary": "one-line gist"
  }
}

Rules:
- Each codon corresponds to one numbered group in the input
- weight: 1.0 = critical, 0.5 = useful, 0.1 = filler
- is_exon: true = load-bearing content, false = can be spliced out without info loss
- complement: must be dense enough to reconstruct the gist from it alone
- domains: lowercase topic tags for retrieval (e.g. "auth", "database", "routing")
- entities: specific things mentioned (e.g. "JWT", "PostgreSQL", "FastAPI")
- Keep all values concise. This is compression, not expansion."""


_EXPRESS_SYSTEM = """You are a gene expression scorer. Given a query and a list of gene summaries,
score each gene's relevance from 0.0 to 1.0.
Respond ONLY with a JSON object: {"gene_id": score, ...}
Only include genes with score > 0.2."""


def _splice_system(aggressiveness: float) -> str:
    """Map aggressiveness config to a discrete prompt tone."""
    if aggressiveness <= 0.1:
        tone = "Keep everything unless it is pure noise or exact repetition."
    elif aggressiveness <= 0.3:
        tone = "Keep content that is relevant to the query context. Remove only filler and boilerplate."
    elif aggressiveness <= 0.6:
        tone = "Balanced — keep load-bearing information. Remove tangential details and verbose explanations."
    elif aggressiveness <= 0.8:
        tone = "Aggressive — only keep content directly relevant to answering the query."
    else:
        tone = "Ruthless — only keep content that directly answers or is essential prerequisite for the query. Everything else goes."

    return f"""You are a context splicer. You receive a query and codon lists for multiple genes.
For each gene, decide which codon indices to KEEP (exons) and which to DISCARD (introns).

Respond ONLY with a JSON object mapping gene_id to arrays of indices to keep:
{{"gene_id_1": [0, 2, 3], "gene_id_2": [1, 4], ...}}

Aggressiveness: {tone}

For genes marked [fragment], do not force closure — these are continuations.
If a gene has no relevant codons at all, return an empty array for it."""


_REPLICATE_SYSTEM = """You are a context replication engine. You receive a query+response exchange
and produce structured JSON capturing the INTENT and STATE CHANGES, not just raw facts.

You must respond ONLY with valid JSON, no markdown fences, no explanation.

Output schema:
{
  "codons": [
    {"meaning": "short semantic label", "weight": 0.0-1.0, "is_exon": true/false}
  ],
  "complement": "one-paragraph capturing: what the user wanted, what decision was made, what state changed",
  "promoter": {
    "domains": ["topic1", "topic2"],
    "entities": ["specific_thing1"],
    "intent": "the goal of this exchange",
    "summary": "one-line gist of outcome"
  }
}

Focus on:
- What was the user trying to achieve?
- What decision was made or action taken?
- What state changed as a result?
Do NOT just summarize the text — capture the *meaning* of the exchange."""


# ── Ribosome ────────────────────────────────────────────────────────

class Ribosome:
    """
    CPU-bound small model that handles context codec operations.

    The ribosome doesn't participate in conversation — it's a
    preprocessing/postprocessing engine that runs between turns.
    """

    def __init__(
        self,
        backend: Optional[ModelBackend] = None,
        encoder: Optional[CodonEncoder] = None,
        splice_aggressiveness: float = 0.5,
    ):
        self.backend = backend or OllamaBackend()
        self.encoder = encoder or CodonEncoder()
        self.splice_aggressiveness = splice_aggressiveness

    # ── Pack: raw text → Gene ───────────────────────────────────────

    def pack(self, content: str, content_type: str = "text") -> Gene:
        """
        Encode raw content into a Gene ready for genome storage.

        1. Chunk the content into proto-codons (sentence groups)
        2. Send numbered groups to the ribosome for encoding
        3. Assemble Gene with promoter tags and complement
        """
        from .genome import Genome

        if content_type == "code":
            groups = self.encoder.chunk_code(content)
        elif content_type == "conversation":
            try:
                messages = json.loads(content)
                groups = self.encoder.chunk_conversation(messages)
            except (json.JSONDecodeError, TypeError):
                groups = self.encoder.chunk_text(content)
        else:
            groups = self.encoder.chunk_text(content)

        numbered = "\n".join(
            f"[Group {i}]: {' '.join(g)}" for i, g in enumerate(groups)
        )
        prompt = f"Encode the following content into codons:\n\n{numbered}"

        try:
            raw = self.backend.complete(prompt, system=_PACK_SYSTEM)
            parsed = _parse_json(raw)
        except Exception as exc:
            raise TranscriptionError(f"Pack failed: {exc}") from exc

        if not isinstance(parsed, dict):
            raise FoldingError(f"Pack returned non-dict: {type(parsed)}")

        # Assemble codon meanings
        codon_meanings = []
        for i, c in enumerate(parsed.get("codons", [])):
            codon_meanings.append(c.get("meaning", f"chunk_{i}"))

        # Build Gene
        gene_id = Genome.make_gene_id(content)
        promoter_data = parsed.get("promoter", {})

        return Gene(
            gene_id=gene_id,
            content=content,
            complement=parsed.get("complement", content[:500]),
            codons=codon_meanings,
            promoter=PromoterTags(
                domains=promoter_data.get("domains", []),
                entities=promoter_data.get("entities", []),
                intent=promoter_data.get("intent", ""),
                summary=promoter_data.get("summary", ""),
            ),
            epigenetics=EpigeneticMarkers(),
        )

    # ── Re-rank: score candidates against query ─────────────────────

    def re_rank(self, query: str, candidates: List[Gene], k: int = 5) -> List[Gene]:
        """
        Score candidate genes by relevance to the query.
        Uses promoter summaries (not full content) to stay within token budget.

        Lost-in-the-middle guard: if ribosome scores < 50% of candidates,
        pad with next-best SQLite results (already in the candidates list).

        Fix 4: on timeout, fall back to promoter-score ordering (input order).
        """
        if not candidates:
            return []

        if len(candidates) <= k:
            return candidates

        summaries = {
            g.gene_id: f"{g.promoter.summary} [{','.join(g.promoter.domains)}]"
            for g in candidates
        }

        prompt = (
            f"Query: {query}\n\n"
            f"Gene summaries:\n"
            + "\n".join(f"  {gid}: {s}" for gid, s in summaries.items())
        )

        try:
            raw = self.backend.complete(prompt, system=_EXPRESS_SYSTEM)
            scores = _parse_json(raw)
        except Exception:
            # Fix 4: timeout or model failure — fall back to input order
            log.warning("Re-rank failed, falling back to promoter-score ordering", exc_info=True)
            return candidates[:k]

        if not isinstance(scores, dict):
            log.warning("Re-rank returned non-dict, falling back to input order")
            return candidates[:k]

        # Score and sort
        scored: List[tuple[float, Gene]] = []
        for g in candidates:
            score = scores.get(g.gene_id, 0.0)
            if isinstance(score, (int, float)) and score > 0.2:
                scored.append((float(score), g))

        # Lost-in-the-middle guard: if < 50% scored, pad with unscored candidates
        if len(scored) < len(candidates) * 0.5:
            scored_ids = {g.gene_id for _, g in scored}
            for g in candidates:
                if g.gene_id not in scored_ids and len(scored) < k:
                    scored.append((0.25, g))  # Default score for padded genes

        scored.sort(key=lambda x: x[0], reverse=True)
        return [g for _, g in scored[:k]]

    # ── Splice: remove introns (BATCHED single call) ────────────────

    def splice(
        self,
        query: str,
        genes: List[Gene],
        min_codons_kept: int = 2,
    ) -> Dict[str, str]:
        """
        Batched splice: single ribosome call for all genes.
        Returns {gene_id: spliced_text} for each gene.

        Fix 2: if ribosome returns empty list for a gene, keep first
               min_codons_kept codons or fall back to complement.
        Fix 4: on timeout, fall back to complement for all genes.
        """
        if not genes:
            return {}

        # Build the batched prompt
        gene_sections = []
        for g in genes:
            fragment_note = " [fragment]" if g.is_fragment else ""
            codon_list = "\n".join(f"    [{i}] {c}" for i, c in enumerate(g.codons))
            gene_sections.append(
                f"  Gene {g.gene_id}{fragment_note}:\n{codon_list}"
            )

        prompt = (
            f"Query context: {query}\n\n"
            f"Genes and their codons:\n"
            + "\n\n".join(gene_sections)
            + "\n\nFor each gene, which codon indices should be KEPT?"
        )

        system = _splice_system(self.splice_aggressiveness)

        try:
            raw = self.backend.complete(prompt, system=system)
            parsed = _parse_json(raw)
        except Exception:
            # Fix 4: timeout/failure — fall back to complement for all genes
            log.warning("Splice failed, falling back to complement", exc_info=True)
            return {g.gene_id: g.complement or g.content[:500] for g in genes}

        if not isinstance(parsed, dict):
            log.warning("Splice returned non-dict, falling back to complement")
            return {g.gene_id: g.complement or g.content[:500] for g in genes}

        # Build spliced text per gene
        result: Dict[str, str] = {}
        for g in genes:
            indices = parsed.get(g.gene_id)

            if not isinstance(indices, list):
                # Gene wasn't in the response — use complement
                result[g.gene_id] = g.complement or g.content[:500]
                continue

            # Fix 2: empty splice guard
            if not indices and g.codons:
                # Ribosome said "keep nothing" — don't trust it
                # Keep first N codons as a safety net
                kept = g.codons[:min_codons_kept]
                log.info(
                    "Empty splice for gene %s, keeping first %d codons",
                    g.gene_id, len(kept),
                )
            else:
                kept = [
                    g.codons[i] for i in indices
                    if isinstance(i, int) and 0 <= i < len(g.codons)
                ]

            if kept:
                result[g.gene_id] = " | ".join(kept)
            else:
                # All indices were invalid — fall back to complement
                result[g.gene_id] = g.complement or g.content[:500]

        # Handle genes missing from parsed response
        for g in genes:
            if g.gene_id not in result:
                result[g.gene_id] = g.complement or g.content[:500]

        return result

    # ── Replicate: pack a query+response exchange ───────────────────

    def replicate(self, query: str, response: str) -> Gene:
        """
        Encode a conversation exchange into a Gene for genome storage.
        Captures intent and state changes, not just raw facts.
        """
        from .genome import Genome

        exchange = f"User query: {query}\n\nAssistant response: {response}"

        numbered = f"[Group 0]: {exchange}"
        prompt = f"Encode this conversation exchange:\n\n{numbered}"

        try:
            raw = self.backend.complete(prompt, system=_REPLICATE_SYSTEM)
            parsed = _parse_json(raw)
        except Exception:
            # Replication is best-effort (background task) — don't crash
            log.warning("Replicate failed, creating minimal gene", exc_info=True)
            gene_id = Genome.make_gene_id(exchange)
            return Gene(
                gene_id=gene_id,
                content=exchange,
                complement=f"Q: {query[:200]} A: {response[:300]}",
                codons=["exchange"],
                promoter=PromoterTags(summary=query[:100]),
                epigenetics=EpigeneticMarkers(),
            )

        if not isinstance(parsed, dict):
            gene_id = Genome.make_gene_id(exchange)
            return Gene(
                gene_id=gene_id,
                content=exchange,
                complement=f"Q: {query[:200]} A: {response[:300]}",
                codons=["exchange"],
                promoter=PromoterTags(summary=query[:100]),
                epigenetics=EpigeneticMarkers(),
            )

        gene_id = Genome.make_gene_id(exchange)
        promoter_data = parsed.get("promoter", {})
        codon_meanings = [c.get("meaning", "exchange") for c in parsed.get("codons", [])]

        return Gene(
            gene_id=gene_id,
            content=exchange,
            complement=parsed.get("complement", exchange[:500]),
            codons=codon_meanings or ["exchange"],
            promoter=PromoterTags(
                domains=promoter_data.get("domains", []),
                entities=promoter_data.get("entities", []),
                intent=promoter_data.get("intent", "conversation exchange"),
                summary=promoter_data.get("summary", query[:100]),
            ),
            epigenetics=EpigeneticMarkers(),
        )


# ── JSON parsing (tolerant) ────────────────────────────────────────

def _parse_json(raw: str) -> dict | list:
    """Parse JSON from model output, tolerating markdown fences and preamble."""
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object/array in the response
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue

    log.warning("Ribosome returned unparseable output: %s", raw[:200])
    raise FoldingError(f"Unparseable JSON: {raw[:200]}")

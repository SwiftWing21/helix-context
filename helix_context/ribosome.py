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

================================================================
CURRENT STATE (as of v0.3.0b3): LLM RIBOSOME IS PAUSED BY DEFAULT
================================================================

Every method in this file is an LLM function call. The operations are:

    pack       → backend.complete()  (ribosome.py:318, ~6s/call on gemma4:e4b)
    re_rank    → backend.complete()  (ribosome.py:407, used only if rerank_enabled)
    splice     → backend.complete()  (ribosome.py:451, currently UNWIRED in pipeline)
    replicate  → backend.complete()  (ribosome.py:535, wrapped in 15s timeout)

As of v0.3.0b3, the default runtime configuration disables the LLM
ribosome via two mechanisms:

    1. helix.toml:  ribosome.warmup = false
       → server does NOT pre-load gemma4:e4b on startup

    2. /admin/ribosome/pause endpoint + background_tasks.add_task wrapper
       → monkey-patches backend.complete() to raise RuntimeError
       → existing fallback paths in pack/replicate synthesize minimal
         genes from raw exchange when the LLM call raises

The LLM path still EXISTS — it can be re-enabled per-session via
/admin/ribosome/resume. It's not deleted, just turned off.

WHY IT'S OFF: the ribosome competed for GPU VRAM with concurrent
benchmark workloads (qwen3:4b, qwen3:8b inference). Paused ribosome
= zero VRAM contention, minimal-gene fallback path is good enough
for learn()/replicate() until we pick a permanent small-model codec.

WHAT'S NEXT (pending headroom meeting 2026-04-10):
The LLM ribosome is the swap point for a CPU-native codec. Two
candidates on the table:

    - LLMLingua-2 (microsoft, MIT, mBERT-base token classifier)
      → drop-in peer to the current ribosome, 700MB model download,
        MeetingBank-prose bias, 512-token chunking seam

    - Kompress (chopratejas/headroom, Apache-2.0, ModernBERT)
      → 8k native context, code+web training mix, bundled ONNX path

Both replace the same backend.complete() contract via a new
CodecBackend protocol. Whichever wins gets wired via the existing
config.ribosome.backend = "ollama" | "deberta" | <new> switch at
context_manager.py:196-215. The LLM path stays as a fallback for
content types where neither CPU codec works.

DO NOT delete this file or its methods without explicit coordination.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Protocol

import httpx

from .accel import (
    json_loads,
    PromptBuilder,
    RE_MARKDOWN_FENCE_START,
    RE_MARKDOWN_FENCE_END,
)
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


# ── Claude API backend ─────────────────────────────────────────────

class ClaudeBackend:
    """Anthropic Claude API backend for the ribosome.

    Drop-in replacement for OllamaBackend. Routes through a proxy (e.g.
    Headroom at :8787) when claude_base_url is set in helix.toml; hits
    Anthropic directly otherwise.

    Cost controls:
    - Prompt caching (cache_control: ephemeral) on all system prompts —
      bulk ingest repeats the same system prompt per operation, so every
      call after the first is a cache hit.
    - 300ms minimum between requests per CLAUDE.md rate-limit policy.
    - Explicit 30s timeout so ingest can't hang indefinitely.

    Switch back to Ollama: set backend = "ollama" in helix.toml.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        base_url: str = "",
        max_tokens: int = 3000,
        timeout: float = 30.0,
        min_request_interval: float = 0.3,
    ):
        import time as _time
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required for ClaudeBackend — pip install anthropic"
            ) from exc

        self._time = _time
        init_kwargs: dict = {"timeout": timeout}
        if base_url:
            init_kwargs["base_url"] = base_url
        self.client = _anthropic.Anthropic(**init_kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self._min_interval = min_request_interval
        self._last_request_ts: float = 0.0

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        elapsed = self._time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval:
            self._time.sleep(self._min_interval - elapsed)

        # Cache stable system prompts — same system string repeats across
        # all pack/replicate/kv-extract calls during bulk ingest.
        system_param: list | str
        if system:
            system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            system_param = ""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system_param,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception:
            raise
        finally:
            self._last_request_ts = self._time.monotonic()


# ── LiteLLM backend (universal — Claude, Gemini, OpenAI, Ollama) ──

class LiteLLMBackend:
    """Universal model backend via litellm.

    One backend for any provider. Model strings use litellm prefix format:
      - "gemini/gemini-2.5-flash"    → Google Gemini
      - "claude-haiku-4-5-20251001"  → Anthropic Claude
      - "gpt-4o"                     → OpenAI
      - "ollama/qwen3:8b"            → Local Ollama

    Routes through Headroom proxy when base_url is set (compression +
    caching on all providers). Direct API otherwise.

    Requires: pip install litellm (already installed via headroom-ai)
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        base_url: str = "",
        max_tokens: int = 3000,
        timeout: float = 30.0,
        min_request_interval: float = 0.3,
    ):
        import time as _time
        try:
            import litellm as _litellm
        except ImportError as exc:
            raise ImportError(
                "litellm package required for LiteLLMBackend — pip install litellm"
            ) from exc

        self._time = _time
        self._litellm = _litellm
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url or None
        self._min_interval = min_request_interval
        self._last_request_ts: float = 0.0

        # Suppress litellm's verbose logging
        _litellm.suppress_debug_info = True

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        elapsed = self._time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval:
            self._time.sleep(self._min_interval - elapsed)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        try:
            resp = self._litellm.completion(**kwargs)
            return resp.choices[0].message.content
        except Exception:
            raise
        finally:
            self._last_request_ts = self._time.monotonic()


class DisabledBackend:
    """No-op ribosome backend used when ribosome is intentionally disabled."""

    model = "disabled"
    is_disabled_backend = True

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        raise RuntimeError("Ribosome is disabled")


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


_KV_EXTRACT_SYSTEM = """You are a fact extraction engine. Extract specific key-value facts from the content.
You must respond ONLY with a JSON array of strings, no markdown fences, no explanation.

Extract:
- Port numbers: "port=8080"
- Model names: "model=qwen3"
- Counts/quantities: "skills=125", "workers=8"
- File paths that are config entry points: "config=fleet/fleet.toml"
- Version numbers: "version=0.400.00b"
- Named identifiers: "default_backend=ollama"
- Thresholds/limits: "max_workers=20", "timeout=30s"
- URLs: "base_url=http://localhost:11434"

Rules:
- Only extract CONCRETE values — no descriptions or explanations
- Use short keys in snake_case: "max_workers=20" not "maximum number of workers=20"
- Skip vague or subjective facts — only extract specific, queryable values
- Return an empty array [] if no concrete facts are found
- Maximum 15 facts per content block"""


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

    ── IMPORTANT: This class calls the LLM via self.backend.complete() ──
    As of v0.3.0b3 the default runtime keeps this path PAUSED via
    /admin/ribosome/pause. Every method below that calls backend.complete
    will raise RuntimeError if the pause is active; callers already have
    fallback paths (minimal-gene synthesis from raw exchange).

    Swap target: pending headroom meeting decision on LLMLingua-2 vs
    Kompress as a CPU-native codec replacement. See module docstring.
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
                messages = json_loads(content)
                groups = self.encoder.chunk_conversation(messages)
            except (ValueError, TypeError):
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

        gene = Gene(
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

        # KV extraction — extract concrete facts for downstream small models
        gene.key_values = self._extract_key_values(content)

        return gene

    # ── KV extraction: pull concrete facts from content ────────────

    def _extract_key_values(self, content: str) -> List[str]:
        """
        Extract key-value facts from content via a second ribosome call.
        Returns a list of short fact strings like ["port=11437", "model=qwen3"].
        Best-effort — returns empty list on failure.
        """
        # Truncate content to avoid blowing the small model's context
        truncated = content[:2000]
        prompt = f"Extract key-value facts from this content:\n\n{truncated}"

        try:
            raw = self.backend.complete(prompt, system=_KV_EXTRACT_SYSTEM)
            parsed = _parse_json(raw)
        except Exception:
            log.debug("KV extraction failed (non-fatal)", exc_info=True)
            return []

        if not isinstance(parsed, list):
            log.debug("KV extraction returned non-list: %s", type(parsed))
            return []

        # Validate: each item must be a non-empty string containing '='
        kvs = []
        for item in parsed[:15]:
            if isinstance(item, str) and "=" in item and len(item) < 200:
                kvs.append(item.strip())
        return kvs

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

        pb = PromptBuilder()
        pb.writeln(f"Query: {query}")
        pb.writeln()
        pb.writeln("Gene summaries:")
        for gid, s in summaries.items():
            pb.writeln(f"  {gid}: {s}")
        prompt = pb.build()

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

        # Build the batched prompt (StringIO for O(1) amortized appends)
        pb = PromptBuilder()
        pb.writeln(f"Query context: {query}")
        pb.writeln()
        pb.writeln("Genes and their codons:")
        for g in genes:
            fragment_note = " [fragment]" if g.is_fragment else ""
            pb.writeln(f"  Gene {g.gene_id}{fragment_note}:")
            for i, c in enumerate(g.codons):
                pb.writeln(f"    [{i}] {c}")
            pb.writeln()
        pb.writeln("For each gene, which codon indices should be KEPT?")
        prompt = pb.build()

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
    """Parse JSON from model output, tolerating markdown fences and preamble.

    Uses orjson (Rust) when available for 3-8x faster deserialization.
    Pre-compiled regex patterns from accel module for fence stripping.
    """
    cleaned = raw.strip()

    # Strip markdown fences (pre-compiled regex)
    cleaned = RE_MARKDOWN_FENCE_START.sub("", cleaned)
    cleaned = RE_MARKDOWN_FENCE_END.sub("", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse first (fast path — succeeds ~80% of the time)
    try:
        return json_loads(cleaned)
    except (ValueError, TypeError):
        pass

    # Try to find JSON object/array in the response
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json_loads(cleaned[start : end + 1])
            except (ValueError, TypeError):
                continue

    log.warning("Ribosome returned unparseable output: %s", raw[:200])
    raise FoldingError(f"Unparseable JSON: {raw[:200]}")

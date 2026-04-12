"""
CpuTagger — CPU-native gene encoding without LLM inference.

Replaces the two Ollama calls in ribosome.pack() and _extract_key_values()
with spaCy NER, regex patterns, and extractive summarization.

Biology:
    The ribosome translates mRNA into protein using physical chemistry.
    The CpuTagger does the same translation using statistical NLP —
    no autoregressive generation, just pattern matching and classification.

Performance:
    LLM pack: ~6s/chunk (2x E4B autoregressive calls)
    CpuTagger: ~7ms/chunk (spaCy + regex, single-threaded)
    3,500 chunks: ~1 hour → ~25 seconds
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Optional, Set

from .codons import CodonEncoder
from .schemas import EpigeneticMarkers, Gene, PromoterTags

log = logging.getLogger("helix.tagger")


# ── Lazy spaCy loading (avoid import-time model download check) ────

_nlp = None


def _get_nlp():
    """Load spaCy model on first use. Cached for process lifetime."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        # Increase max_length for large code files
        _nlp.max_length = 200_000

        # EntityRuler for project-specific terms that statistical NER misses
        ruler = _nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(_build_project_patterns())
    return _nlp


# ── Project entity vocabulary (rule-based NER) ──────────────────

_PROJECT_ENTITIES = {
    "PRODUCT": [
        "BigEd", "BigEd CC", "Helix Context", "Agentome", "ScoreRift",
        "BookKeeper", "CosmicTasha", "two-brain", "ModuleHub",
        "CpuTagger", "SemaCodec", "FleetDB", "Dr. Ders", "DeBERTa",
        "fleet.toml", "helix.toml", "genome.db",
        "SPLADE", "FTS5", "MiniLM", "ColBERT", "RaBitQ",
    ],
    "ORG": [
        "SwiftWing21", "Anthropic", "Naver Labs",
    ],
}


def _build_project_patterns() -> List[Dict]:
    """Build spaCy EntityRuler patterns from project vocabulary."""
    patterns = []
    for label, terms in _PROJECT_ENTITIES.items():
        for term in terms:
            patterns.append({"label": label, "pattern": term})
            # Also add lowercase variant for case-insensitive matching
            lower = term.lower()
            if lower != term:
                patterns.append({"label": label, "pattern": lower})
    return patterns


# ── Tech dictionary for domain classification ─────────────────────

_TECH_TERMS: Set[str] = {
    # Languages
    "python", "rust", "javascript", "typescript", "go", "java", "ruby",
    "swift", "kotlin", "lua", "sql", "html", "css", "toml", "yaml", "json",
    # Frameworks & tools
    "flask", "django", "fastapi", "react", "vue", "angular", "svelte",
    "pytorch", "tensorflow", "numpy", "pandas", "spacy", "transformers",
    "ollama", "docker", "kubernetes", "redis", "sqlite", "postgresql",
    "nginx", "git", "github", "npm", "pip", "cargo", "axum", "egui",
    # Concepts
    "api", "rest", "graphql", "websocket", "grpc", "mqtt", "sse",
    "auth", "jwt", "oauth", "rbac", "tls", "ssl", "cors",
    "database", "cache", "queue", "pipeline", "proxy", "gateway",
    "embedding", "vector", "tokenizer", "llm", "rag", "fts5", "bm25",
    "gpu", "cuda", "vram", "cpu", "memory", "latency", "throughput",
    "agent", "worker", "supervisor", "scheduler", "orchestrator",
    "config", "toml", "yaml", "env", "endpoint", "blueprint",
    "test", "benchmark", "smoke", "unittest", "pytest",
    "security", "audit", "compliance", "soc2", "encryption",
    "deploy", "ci", "cd", "release", "build", "binary",
    # Project-specific terms (Agentome ecosystem)
    "biged", "helix", "agentome", "scorerift", "bookkeeper",
    "cosmictasha", "modulehub", "ribosome", "genome", "chromatin",
    "ellipticity", "codon", "promoter", "epigenetics", "sema",
    "splade", "deberta", "colbert", "rabitq", "fts5",
    "cputagger", "semacodec", "fleetdb", "fleet",
    "supervisor", "dr ders", "needle", "gene",
}

# ── Regex patterns for KV extraction ──────────────────────────────

_KV_PATTERNS = [
    # port = 11437  or  port: 11437  (named key-value, high confidence)
    (re.compile(r'(?:port|PORT|listen)\s*[=:]\s*(\d{2,5})', re.IGNORECASE), "port"),
    # model = "gemma4:e4b"  or  model: qwen3:8b  (greedy match to capture full model spec)
    (re.compile(r'(?:model|MODEL|model_name|default_model)\s*[=:]\s*["\']?([a-zA-Z][a-zA-Z0-9._:/-]+)', re.IGNORECASE), "model"),
    # URLs
    (re.compile(r'(https?://[^\s"\'<>,;)\]]+)'), "url"),
    # version = "0.1.0b1"  or  v0.400.00b
    (re.compile(r'(?:version|VERSION|__version__)\s*[=:]\s*["\']?(v?[\d]+\.[\d]+(?:\.[\d]+)?(?:[ab]\d+)?)', re.IGNORECASE), "version"),
    # file paths (Unix or Windows)
    (re.compile(r'(?:path|file|dir(?:ectory)?)\s*[=:]\s*["\']?([A-Za-z]:[/\\][\w./ \\-]+|/[\w./-]+)', re.IGNORECASE), "path"),
    # timeout = 30  or  threshold = 0.5
    (re.compile(r'(timeout|threshold|interval|limit|budget|max_\w+|min_\w+)\s*[=:]\s*(\d+(?:\.\d+)?)', re.IGNORECASE), None),
    # host/address
    (re.compile(r'(?:host|address|bind)\s*[=:]\s*["\']?([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+|localhost)', re.IGNORECASE), "host"),
]

# Patterns that extract key=value pairs
_KV_PAIR_PATTERN = re.compile(
    r'(\w+(?:_\w+)*)\s*[=:]\s*["\']?(\d+(?:\.\d+)?(?:s|ms|m|h|mb|gb|kb)?|true|false|[a-zA-Z0-9._:/-]+)["\']?',
    re.IGNORECASE,
)

# Skip generic/noisy keys
_KV_SKIP_KEYS = frozenset({
    "if", "for", "in", "or", "and", "not", "is", "as", "from", "import",
    "return", "def", "class", "self", "none", "true", "false",
    "type", "name", "value", "data", "item", "key", "str", "int", "float",
    "the", "this", "that", "with", "have", "been",
    # Code patterns that look like key=value but aren't config
    "app", "req", "res", "ctx", "err", "msg", "buf", "src", "dst",
    "status", "result", "output", "input", "content", "text", "body",
    "architecture", "description", "label", "title", "role",
})

# Python type-annotation names. These leak through the regex matcher on
# annotated assignments like `port: int = 8080` — the naive pair matcher
# sees `port: int` first (key=port, val=int) AND `int = 8080` as an inner
# overlap (key=int, val=8080). Both are wrong. We skip both when either
# side is a bare type name, so only the real literal value survives
# (via the named patterns or a clean `port = 8080` assignment).
_KV_TYPE_ANNOTATION_NAMES = frozenset({
    "int", "str", "float", "bool", "bytes", "bytearray",
    "list", "dict", "tuple", "set", "frozenset", "deque",
    "optional", "any", "none", "type", "object",
    "callable", "iterator", "iterable", "generator", "awaitable",
    "coroutine", "mapping", "sequence", "union", "literal",
    "path", "uuid", "datetime", "timedelta",
})


# ── CpuTagger ─────────────────────────────────────────────────────

class CpuTagger:
    """
    CPU-native gene encoder. Drop-in replacement for ribosome.pack()
    that produces the same Gene schema without any LLM calls.

    Encoding pipeline per chunk:
        1. spaCy NER → entities (PERSON, ORG, PRODUCT, GPE, etc.)
        2. Tech dictionary scan → domains (language, framework, tool terms)
        3. Regex patterns → key_values ("port=11437", "model=gemma4:e4b")
        4. spaCy noun chunks → codon meaning labels
        5. Sentence info-density ranking → complement (extractive summary)
        6. First sentence heuristic → intent
    """

    def __init__(self, synonym_map: Optional[Dict[str, List[str]]] = None):
        self.synonym_map = synonym_map or {}
        self._encoder = CodonEncoder()
        # Reverse synonym map for domain enrichment:
        # "redis" → also tag with "cache"
        self._reverse_synonyms: Dict[str, str] = {}
        for key, values in self.synonym_map.items():
            for v in values:
                self._reverse_synonyms[v.lower()] = key.lower()

    def pack(
        self,
        content: str,
        content_type: str = "text",
        source_id: Optional[str] = None,
        sequence_index: Optional[int] = None,
    ) -> Gene:
        """
        Encode raw content into a Gene. Same output contract as ribosome.pack().

        Returns a Gene with: gene_id, content, complement, codons, promoter,
        epigenetics, key_values, source_id, is_fragment.
        """
        from .genome import Genome

        nlp = _get_nlp()

        # Truncate for spaCy processing (large files overwhelm the pipeline)
        truncated = content[:50_000]
        doc = nlp(truncated)

        # 1. Extract entities via NER
        entities = self._extract_entities(doc, content)

        # 2. Extract domains via tech dictionary + content scanning
        domains = self._extract_domains(doc, content, entities)

        # 3. Extract key-value facts via regex
        key_values = self._extract_key_values(content)

        # 4. Generate codon meaning labels
        codons = self._extract_codons(doc, content, content_type)

        # 5. Generate complement (extractive summary)
        complement = self._extract_complement(doc, content)

        # 6. Generate intent (first sentence heuristic)
        intent = self._extract_intent(doc, content, content_type)

        # 7. Generate summary
        summary = self._extract_summary(doc, content)

        gene_id = Genome.make_gene_id(content)

        gene = Gene(
            gene_id=gene_id,
            content=content,
            complement=complement,
            codons=codons,
            promoter=PromoterTags(
                domains=domains[:10],
                entities=entities[:15],
                intent=intent,
                summary=summary,
                sequence_index=sequence_index,
            ),
            epigenetics=EpigeneticMarkers(),
            key_values=key_values,
            source_id=source_id,
        )

        return gene

    # ── Entity extraction (spaCy NER) ─────────────────────────────

    def _extract_entities(self, doc, content: str) -> List[str]:
        """Extract named entities from spaCy doc, deduplicated."""
        seen: Set[str] = set()
        entities: List[str] = []

        for ent in doc.ents:
            # Keep useful entity types
            if ent.label_ in (
                "PERSON", "ORG", "PRODUCT", "GPE", "FAC",
                "WORK_OF_ART", "LAW", "EVENT", "LANGUAGE",
            ):
                text = ent.text.strip()
                lower = text.lower()
                if lower not in seen and len(text) > 1:
                    seen.add(lower)
                    entities.append(text)

        # Also scan for CamelCase identifiers (common in code)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', content[:10_000]):
            word = match.group(1)
            lower = word.lower()
            if lower not in seen:
                seen.add(lower)
                entities.append(word)

        return entities

    # ── Domain extraction (tech terms + synonym map) ──────────────

    def _extract_domains(
        self, doc, content: str, entities: List[str]
    ) -> List[str]:
        """Extract topic domains from content using tech dictionary and synonyms."""
        seen: Set[str] = set()
        domains: List[str] = []
        lower_content = content[:20_000].lower()

        # Scan for tech terms in content
        words = set(re.findall(r'\b[a-z_][a-z0-9_]+\b', lower_content))
        for word in words:
            if word in _TECH_TERMS and word not in seen:
                seen.add(word)
                domains.append(word)
                # Also add reverse synonym parent
                if word in self._reverse_synonyms:
                    parent = self._reverse_synonyms[word]
                    if parent not in seen:
                        seen.add(parent)
                        domains.append(parent)

        # Add domains from entity names (lowercase)
        for ent in entities[:10]:
            lower = ent.lower()
            if lower in _TECH_TERMS and lower not in seen:
                seen.add(lower)
                domains.append(lower)

        # Add spaCy noun chunks that overlap with tech terms
        for chunk in doc.noun_chunks:
            lower_chunk = chunk.root.text.lower()
            if lower_chunk in _TECH_TERMS and lower_chunk not in seen:
                seen.add(lower_chunk)
                domains.append(lower_chunk)

        # Sort by frequency in content (most mentioned first)
        freq = {d: lower_content.count(d) for d in domains}
        domains.sort(key=lambda d: freq.get(d, 0), reverse=True)

        return domains

    # ── KV extraction (regex patterns) ────────────────────────────

    def _extract_key_values(self, content: str) -> List[str]:
        """
        Extract key=value facts from content using regex patterns.

        Replaces the LLM-based _extract_key_values() in ribosome.py.
        Returns list of strings like ["port=11437", "model=gemma4:e4b"].
        """
        kvs: List[str] = []
        seen: Set[str] = set()

        # Named patterns first (high confidence)
        for pattern, key_hint in _KV_PATTERNS:
            for match in pattern.finditer(content[:5000]):
                groups = match.groups()
                if len(groups) == 1:
                    val = groups[0].rstrip(".,;:)'\"")
                    # Skip bare Python type annotations (e.g. `model: str` → "str")
                    if val.lower() in _KV_TYPE_ANNOTATION_NAMES:
                        continue
                    if key_hint:
                        kv = f"{key_hint}={val}"
                    else:
                        kv = val
                elif len(groups) == 2:
                    key = groups[0].lower().strip()
                    val = groups[1].rstrip(".,;:)'\"")
                    if key in _KV_SKIP_KEYS or len(key) < 2:
                        continue
                    # Skip `int = 8080` where the key is itself a type name
                    if key in _KV_TYPE_ANNOTATION_NAMES:
                        continue
                    if val.lower() in _KV_TYPE_ANNOTATION_NAMES:
                        continue
                    kv = f"{key}={val}"
                else:
                    continue

                if kv not in seen and len(kv) < 200:
                    seen.add(kv)
                    kvs.append(kv)

        # Generic key=value pairs from config-like content (lower confidence)
        for match in _KV_PAIR_PATTERN.finditer(content[:5000]):
            key = match.group(1).lower()
            val = match.group(2).rstrip(".,;:)'\"")
            if key in _KV_SKIP_KEYS or len(key) < 3 or len(val) < 1:
                continue
            # Skip Python type annotations on both sides:
            #   `port: int`   → key=port, val=int       (drop: val is a type)
            #   `int = 8080`  → key=int,  val=8080      (drop: key is a type)
            # This removes the type-annotation leak without affecting real
            # `key = value` assignments or YAML/TOML scalar pairs.
            if key in _KV_TYPE_ANNOTATION_NAMES:
                continue
            if val.lower() in _KV_TYPE_ANNOTATION_NAMES:
                continue
            kv = f"{key}={val}"
            if kv not in seen and len(kv) < 200:
                seen.add(kv)
                kvs.append(kv)

        return kvs[:15]

    # ── Codon extraction (noun chunks as meaning labels) ──────────

    def _extract_codons(
        self, doc, content: str, content_type: str
    ) -> List[str]:
        """
        Generate codon meaning labels from content.

        For text: most distinctive noun phrase per sentence group.
        For code: function/class names + key identifiers.
        """
        codons: List[str] = []

        if content_type == "code":
            # Extract function/class definitions as codons
            for match in re.finditer(
                r'(?:def|class|async def|function|const|export)\s+(\w+)',
                content[:20_000],
            ):
                name = match.group(1)
                if name not in ("self", "__init__", "main") and len(name) > 2:
                    codons.append(name)

            # Add key identifiers from assignments
            for match in re.finditer(
                r'^(\w+)\s*[=:]\s*', content[:10_000], re.MULTILINE
            ):
                name = match.group(1)
                lower = name.lower()
                if (
                    lower not in _KV_SKIP_KEYS
                    and name not in codons
                    and len(name) > 2
                    and not name.startswith("_")
                ):
                    codons.append(name)
        else:
            # For text: use spaCy noun chunks
            seen: Set[str] = set()
            for chunk in doc.noun_chunks:
                text = chunk.text.strip()
                lower = text.lower()
                if (
                    lower not in seen
                    and len(text) > 3
                    and lower not in STOP_WORDS_SMALL
                ):
                    seen.add(lower)
                    codons.append(text)

        # Deduplicate and limit
        unique: List[str] = []
        seen_lower: Set[str] = set()
        for c in codons:
            lower = c.lower()
            if lower not in seen_lower:
                seen_lower.add(lower)
                unique.append(c)

        return unique[:30]

    # ── Complement extraction (extractive summary) ────────────────

    def _extract_complement(self, doc, content: str) -> str:
        """
        Generate a dense complement (summary) using extractive selection.

        Picks the top-3 most information-dense sentences, measured by
        entity count + numeric literal count.
        """
        sentences = list(doc.sents)
        if not sentences:
            return content[:500]

        # Score each sentence by information density
        scored = []
        for sent in sentences:
            text = sent.text.strip()
            if len(text) < 15:
                continue
            # Count entities in this sentence span
            ent_count = sum(
                1 for ent in doc.ents
                if ent.start >= sent.start and ent.end <= sent.end
            )
            # Count numbers
            num_count = len(re.findall(r'\d+(?:\.\d+)?', text))
            # Count code identifiers (bonus for technical content)
            code_count = len(re.findall(r'\b[A-Z][a-z]+[A-Z]\w*\b', text))
            score = ent_count * 2.0 + num_count * 1.5 + code_count * 1.0
            # Bonus for longer, substantive sentences
            if len(text) > 50:
                score += 0.5
            scored.append((score, text))

        # Sort by score descending, take top 3
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [text for _, text in scored[:3]]

        if not top:
            # Fallback: first 500 chars
            return content[:500]

        return " ".join(top)

    # ── Intent extraction (first sentence / docstring) ────────────

    def _extract_intent(self, doc, content: str, content_type: str) -> str:
        """Extract a one-line intent from the content."""
        if content_type == "code":
            # Try docstring first
            match = re.search(
                r'"""(.+?)"""| \'\'\'(.+?)\'\'\'',
                content[:2000],
                re.DOTALL,
            )
            if match:
                docstring = (match.group(1) or match.group(2) or "").strip()
                first_line = docstring.split("\n")[0].strip()
                if len(first_line) > 10:
                    return first_line[:200]

        # Fall back to first sentence
        sentences = list(doc.sents)
        if sentences:
            first = sentences[0].text.strip()
            if len(first) > 10:
                return first[:200]

        return content[:100].strip()

    # ── Summary extraction ────────────────────────────────────────

    def _extract_summary(self, doc, content: str) -> str:
        """Generate a one-line summary from the most distinctive sentence."""
        sentences = list(doc.sents)
        if not sentences:
            return content[:100].strip()

        # Pick the sentence with the most named entities
        best = max(
            sentences,
            key=lambda s: sum(
                1 for ent in doc.ents
                if ent.start >= s.start and ent.end <= s.end
            ),
        )
        text = best.text.strip()
        if len(text) > 10:
            return text[:200]

        return sentences[0].text.strip()[:200]


# ── Minimal stop words for codon filtering ────────────────────────

STOP_WORDS_SMALL = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "this", "that",
    "these", "those", "it", "its", "and", "or", "but", "not",
    "all", "some", "any", "each", "every", "no", "more", "most",
})

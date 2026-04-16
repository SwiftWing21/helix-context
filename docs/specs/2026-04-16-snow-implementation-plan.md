# SNOW Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SNOW benchmark harness that measures multi-hop navigation efficiency on helix's 5-tier data cascade, with oracle + real-LLM consumers and per-model scorecards.

**Architecture:** In-process harness (HelixContextManager + Ollama API). Oracle does string-matching per tier. LLM consumer sends fingerprints/data to models via Ollama chat API. Each module is a focused file under `benchmarks/snow/`. Results are JSON files; comparison is a separate script.

**Tech Stack:** Python 3.11+, helix_context (in-process), httpx (Ollama API), sqlite3 (direct gene field reads), json (results)

**Spec:** `docs/specs/2026-04-16-snow-benchmark-design.md`

**Target genome:** `genome-bench-2026-04-14.db` (18,254 genes)

---

## File Map

```
benchmarks/snow/
  __init__.py              # Empty, makes it importable
  oracle.py                # Oracle consumer — string matching per tier
  cascade.py               # LLM consumer — Ollama API + tier escalation
  prompts.py               # Triage + extraction prompt templates
  bench_snow.py            # Main harness — orchestrates oracle + LLM runs
  snow_compare.py          # Reads result JSONs, prints comparison table
  snow_queries.json        # N=65 query set (50 existing + 15 tier-stress)
  results/                 # Output JSONs (gitignored)
tests/
  test_snow_oracle.py      # Oracle unit tests
  test_snow_cascade.py     # Cascade unit tests (mocked LLM)
```

---

### Task 1: Scaffold + Query Set

**Files:**
- Create: `benchmarks/snow/__init__.py`
- Create: `benchmarks/snow/snow_queries.json`
- Create: `benchmarks/snow/results/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p benchmarks/snow/results
touch benchmarks/snow/__init__.py
touch benchmarks/snow/results/.gitkeep
```

- [ ] **Step 2: Build the N=65 query set**

Load `benchmarks/needles_50_for_claude.json` as the base 50. Add 15 hand-crafted tier-stress queries by inspecting the benchmark genome at each tier.

For each tier-stress query, verify the answer exists at the target tier by running SQL against `genome-bench-2026-04-14.db`:

```python
# T0 check: answer in promoter entities
SELECT gene_id, promoter FROM genes
WHERE promoter LIKE '%DxPixelShaderFile%' LIMIT 3;

# T1 check: answer in key_values
SELECT gene_id, key_values FROM genes
WHERE key_values LIKE '%11437%' LIMIT 3;

# T2 check: answer in complement but NOT in key_values
SELECT gene_id, complement, key_values FROM genes
WHERE complement LIKE '%answer%'
AND (key_values IS NULL OR key_values NOT LIKE '%answer%') LIMIT 3;

# T3 check: answer ONLY in content
SELECT gene_id FROM genes
WHERE content LIKE '%answer%'
AND complement NOT LIKE '%answer%'
AND (key_values IS NULL OR key_values NOT LIKE '%answer%') LIMIT 3;
```

Output format for `snow_queries.json`:
```json
[
  {
    "idx": 0,
    "query": "What port does the Helix proxy listen on?",
    "expected_answer": "11437",
    "accept": ["11437"],
    "source": "needles_50",
    "oracle_tier": null,
    "oracle_gene_id": null
  },
  {
    "idx": 50,
    "query": "What shader file config key is used in Factorio graphics?",
    "expected_answer": "DxPixelShaderFile",
    "accept": ["DxPixelShaderFile", "dxpixelshaderfile"],
    "source": "tier_stress",
    "target_tier": 0,
    "oracle_tier": null,
    "oracle_gene_id": null
  }
]
```

`oracle_tier` and `oracle_gene_id` are null initially — Task 3 (oracle) populates them on first run.

- [ ] **Step 3: Commit**

```bash
cd /f/Projects/helix-context
git add benchmarks/snow/
git commit -m "feat(snow): scaffold + N=65 query set"
```

---

### Task 2: Oracle Consumer

**Files:**
- Create: `benchmarks/snow/oracle.py`
- Create: `tests/test_snow_oracle.py`

- [ ] **Step 1: Write failing oracle tests**

```python
# tests/test_snow_oracle.py
import pytest
from benchmarks.snow.oracle import oracle_cascade

def test_oracle_finds_answer_in_entities():
    """Oracle should return tier=0 when answer is in fingerprint entities."""
    result = oracle_cascade(
        expected_answer="11437",
        accept=["11437"],
        gene_ids=["gene_a"],
        fingerprints={"gene_a": {
            "entities": ["port", "11437", "helix"],
            "key_values": "{}",
            "complement": "",
            "content": "port = 11437",
        }},
        neighbors={},
    )
    assert result["tier"] == 0
    assert result["gene_id"] == "gene_a"

def test_oracle_finds_answer_in_key_values():
    result = oracle_cascade(
        expected_answer="11437",
        accept=["11437"],
        gene_ids=["gene_a"],
        fingerprints={"gene_a": {
            "entities": ["port", "server"],
            "key_values": '{"port": "11437"}',
            "complement": "server config",
            "content": "port = 11437",
        }},
        neighbors={},
    )
    assert result["tier"] == 1

def test_oracle_finds_answer_in_complement():
    result = oracle_cascade(
        expected_answer="Decimal",
        accept=["decimal", "Decimal"],
        gene_ids=["gene_a"],
        fingerprints={"gene_a": {
            "entities": ["monetary"],
            "key_values": "{}",
            "complement": "Use Decimal type for monetary values",
            "content": "Use Decimal type for monetary values instead of float",
        }},
        neighbors={},
    )
    assert result["tier"] == 2

def test_oracle_finds_answer_in_content():
    result = oracle_cascade(
        expected_answer="CREATE_NO_WINDOW",
        accept=["CREATE_NO_WINDOW"],
        gene_ids=["gene_a"],
        fingerprints={"gene_a": {
            "entities": ["subprocess"],
            "key_values": "{}",
            "complement": "subprocess creation flags for Windows",
            "content": "creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)",
        }},
        neighbors={},
    )
    assert result["tier"] == 3

def test_oracle_finds_answer_in_neighbor():
    result = oracle_cascade(
        expected_answer="30",
        accept=["30"],
        gene_ids=["gene_a"],
        fingerprints={
            "gene_a": {
                "entities": ["timeout"],
                "key_values": "{}",
                "complement": "configures timeouts",
                "content": "timeout handling module",
            },
            "gene_b": {
                "entities": [],
                "key_values": '{"timeout": "30"}',
                "complement": "timeout = 30 seconds",
                "content": "timeout = 30",
            },
        },
        neighbors={"gene_a": [("gene_b", 0.85)]},
    )
    assert result["tier"] == 4

def test_oracle_returns_miss():
    result = oracle_cascade(
        expected_answer="nonexistent_value",
        accept=["nonexistent_value"],
        gene_ids=["gene_a"],
        fingerprints={"gene_a": {
            "entities": [],
            "key_values": "{}",
            "complement": "nothing relevant",
            "content": "nothing relevant at all",
        }},
        neighbors={},
    )
    assert result["tier"] == -1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /f/Projects/helix-context
python -m pytest tests/test_snow_oracle.py -v
```
Expected: FAIL — `ImportError: cannot import name 'oracle_cascade'`

- [ ] **Step 3: Implement oracle.py**

```python
# benchmarks/snow/oracle.py
"""Oracle consumer — string matching per data tier."""
from __future__ import annotations
import json
import time
from typing import Dict, List, Optional, Tuple

def _answer_in_text(text: str, accept: List[str]) -> bool:
    lower = text.lower()
    return any(a.lower() in lower for a in accept)

def _answer_in_entities(entities: List[str], accept: List[str]) -> bool:
    joined = ' '.join(entities).lower()
    return any(a.lower() in joined for a in accept)

def oracle_cascade(
    expected_answer: str,
    accept: List[str],
    gene_ids: List[str],
    fingerprints: Dict[str, Dict],
    neighbors: Dict[str, List[Tuple[str, float]]],
) -> Dict:
    """Walk the cascade with perfect knowledge. Return the first tier
    where the answer appears.

    Returns: {tier: int (-1=MISS, 0-4), gene_id: str|None,
              tokens: int, latency_s: float}
    """
    t0 = time.perf_counter()
    tokens = 0

    # T0: fingerprint entities
    for gid in gene_ids:
        fp = fingerprints.get(gid, {})
        entities = fp.get("entities", [])
        tokens += len(' '.join(entities)) // 4 + 1
        if _answer_in_entities(entities, accept):
            return {"tier": 0, "gene_id": gid, "tokens": tokens,
                    "latency_s": time.perf_counter() - t0}

    # T1: key_values
    for gid in gene_ids:
        kv = fingerprints.get(gid, {}).get("key_values", "{}")
        tokens += len(kv) // 4 + 1
        if _answer_in_text(kv, accept):
            return {"tier": 1, "gene_id": gid, "tokens": tokens,
                    "latency_s": time.perf_counter() - t0}

    # T2: complement
    for gid in gene_ids:
        comp = fingerprints.get(gid, {}).get("complement", "")
        tokens += len(comp) // 4 + 1
        if _answer_in_text(comp, accept):
            return {"tier": 2, "gene_id": gid, "tokens": tokens,
                    "latency_s": time.perf_counter() - t0}

    # T3: content
    for gid in gene_ids:
        content = fingerprints.get(gid, {}).get("content", "")
        tokens += len(content) // 4 + 1
        if _answer_in_text(content, accept):
            return {"tier": 3, "gene_id": gid, "tokens": tokens,
                    "latency_s": time.perf_counter() - t0}

    # T4: walk — check 1-hop neighbors (top 3 by weight), content only
    for gid in gene_ids:
        nbs = neighbors.get(gid, [])
        for nb_id, _weight in sorted(nbs, key=lambda x: -x[1])[:3]:
            nb_fp = fingerprints.get(nb_id, {})
            content = nb_fp.get("content", "")
            tokens += len(content) // 4 + 1
            if _answer_in_text(content, accept):
                return {"tier": 4, "gene_id": nb_id, "tokens": tokens,
                        "latency_s": time.perf_counter() - t0}

    return {"tier": -1, "gene_id": None, "tokens": tokens,
            "latency_s": time.perf_counter() - t0}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_snow_oracle.py -v
```
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/snow/oracle.py tests/test_snow_oracle.py
git commit -m "feat(snow): oracle consumer with 6 passing tests"
```

---

### Task 3: Prompt Templates

**Files:**
- Create: `benchmarks/snow/prompts.py`

- [ ] **Step 1: Write prompt templates**

```python
# benchmarks/snow/prompts.py
"""Prompt templates for the SNOW LLM consumer."""

TRIAGE_SYSTEM = (
    "You consume retrieval fingerprints from a knowledge store. "
    "For each query, you see gene metadata: tier scores, source file, "
    "domains, and entities. You do NOT see content.\n\n"
    "If you can answer from the fingerprint: respond with ANSWER: <value>\n"
    "If you need to read a gene: respond with READ: <gene_id>\n"
    "If no gene seems relevant: respond with MISS\n"
    "Be concise. One line only."
)

def triage_prompt(query: str, fingerprints: list[dict]) -> str:
    lines = [f"QUERY: {query}\n"]
    for fp in fingerprints:
        lines.append(
            f"Gene {fp['gene_id'][:10]}: "
            f"src={fp['source']} s={fp['score']:.1f} "
            f"tiers={fp['tiers']} "
            f"domains={fp['domains']} entities={fp['entities']}"
        )
    return '\n'.join(lines)


EXTRACT_SYSTEM = (
    "You receive data from a knowledge store gene. "
    "Answer the question using ONLY this data.\n\n"
    "If you can answer: respond with ANSWER: <value>\n"
    "If this data doesn't contain the answer: respond with ESCALATE\n"
    "One line only."
)

def extract_prompt(query: str, tier_name: str, data: str) -> str:
    return f"QUERY: {query}\n\n{tier_name} data:\n{data}"
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/snow/prompts.py
git commit -m "feat(snow): triage + extraction prompt templates"
```

---

### Task 4: LLM Cascade Consumer

**Files:**
- Create: `benchmarks/snow/cascade.py`
- Create: `tests/test_snow_cascade.py`

- [ ] **Step 1: Write failing cascade tests with mocked LLM**

```python
# tests/test_snow_cascade.py
import pytest
from benchmarks.snow.cascade import llm_cascade

class MockModel:
    """Mock that returns canned responses per tier."""
    def __init__(self, responses):
        self.responses = responses  # list of (action, value) tuples
        self.call_idx = 0
        self.calls = []

    def chat(self, messages):
        resp = self.responses[min(self.call_idx, len(self.responses) - 1)]
        self.call_idx += 1
        self.calls.append(messages)
        return {"message": {"content": resp},
                "eval_count": 10, "prompt_eval_count": 50}

def test_cascade_answers_at_t0():
    model = MockModel(["ANSWER: 11437"])
    result = llm_cascade(
        query="What port?",
        fingerprints=[{"gene_id": "abc", "source": "helix.toml",
                       "score": 21.0, "tiers": {}, "domains": [],
                       "entities": ["11437"]}],
        model=model,
        gene_fields={},
    )
    assert result["tier"] == 0
    assert result["answer"] == "11437"
    assert len(model.calls) == 1

def test_cascade_reads_then_answers():
    model = MockModel(["READ: abc123", "ANSWER: Decimal"])
    result = llm_cascade(
        query="What type for monetary?",
        fingerprints=[{"gene_id": "abc123", "source": "bookkeeper.py",
                       "score": 29.0, "tiers": {}, "domains": ["bookkeeper"],
                       "entities": ["monetary"]}],
        model=model,
        gene_fields={"abc123": {
            "key_values": '{"type": "Decimal"}',
            "complement": "Use Decimal for money",
            "content": "from decimal import Decimal",
        }},
    )
    assert result["tier"] == 1
    assert result["answer"] == "Decimal"
    assert result["hops"] == 1

def test_cascade_escalates_through_tiers():
    model = MockModel(["READ: abc", "ESCALATE", "ESCALATE", "ANSWER: found"])
    result = llm_cascade(
        query="Find the thing",
        fingerprints=[{"gene_id": "abc", "source": "f.py",
                       "score": 15.0, "tiers": {}, "domains": [],
                       "entities": []}],
        model=model,
        gene_fields={"abc": {
            "key_values": "{}",
            "complement": "not here",
            "content": "the thing is found here",
        }},
    )
    assert result["tier"] == 3
    assert result["hops"] == 3

def test_cascade_returns_miss():
    model = MockModel(["MISS"])
    result = llm_cascade(
        query="Nonexistent?",
        fingerprints=[{"gene_id": "abc", "source": "f.py",
                       "score": 5.0, "tiers": {}, "domains": [],
                       "entities": []}],
        model=model,
        gene_fields={},
    )
    assert result["tier"] == -1
    assert result["miss"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_snow_cascade.py -v
```
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement cascade.py**

```python
# benchmarks/snow/cascade.py
"""LLM cascade consumer — tier escalation with real model calls."""
from __future__ import annotations
import re
import time
from typing import Any, Dict, List, Optional

from .prompts import TRIAGE_SYSTEM, triage_prompt, EXTRACT_SYSTEM, extract_prompt

TIER_FIELDS = [(1, "key_values", "Key-Value"), (2, "complement", "Complement"),
               (3, "content", "Content")]

def _parse_response(text: str) -> tuple[str, str]:
    """Parse model response into (action, value).
    Actions: ANSWER, READ, ESCALATE, MISS."""
    text = text.strip()
    for prefix in ("ANSWER:", "READ:", "ESCALATE", "MISS"):
        if text.upper().startswith(prefix):
            value = text[len(prefix):].strip()
            action = prefix.rstrip(":")
            return action, value
    return "UNKNOWN", text

def llm_cascade(
    query: str,
    fingerprints: List[Dict],
    model: Any,
    gene_fields: Dict[str, Dict],
    neighbors: Optional[Dict] = None,
) -> Dict:
    """Run the LLM consumer through the cascade.

    model must have a .chat(messages) method returning
    {"message": {"content": str}, "eval_count": int, "prompt_eval_count": int}.
    """
    hops = 0
    tokens = 0
    hop_detail = []
    t_start = time.perf_counter()

    # T0: triage from fingerprint
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": TRIAGE_SYSTEM},
        {"role": "user", "content": triage_prompt(query, fingerprints)},
    ]
    resp = model.chat(messages)
    content = resp["message"]["content"]
    tok = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
    tokens += tok
    elapsed = time.perf_counter() - t0
    action, value = _parse_response(content)
    hop_detail.append({"tier": "T0", "action": f"{action} {value}",
                       "tokens": tok, "latency_s": elapsed})

    if action == "ANSWER":
        return {"tier": 0, "hops": 0, "answer": value, "miss": False,
                "tokens": tokens, "latency_s": time.perf_counter() - t_start,
                "hop_detail": hop_detail, "gene_id": None}

    if action == "MISS":
        return {"tier": -1, "hops": 0, "answer": None, "miss": True,
                "tokens": tokens, "latency_s": time.perf_counter() - t_start,
                "hop_detail": hop_detail, "gene_id": None}

    # action == READ — get the gene_id to read
    gene_to_read = value.strip()
    # Fuzzy match: the LLM might return a prefix
    matched = None
    for fp in fingerprints:
        if fp["gene_id"].startswith(gene_to_read) or gene_to_read.startswith(fp["gene_id"][:10]):
            matched = fp["gene_id"]
            break
    if not matched and fingerprints:
        matched = fingerprints[0]["gene_id"]
    gene_id = matched

    # T1-T3: escalate through tiers
    fields = gene_fields.get(gene_id, {})
    for tier_num, field_name, tier_label in TIER_FIELDS:
        hops += 1
        data = fields.get(field_name, "")
        if not data:
            continue

        t_hop = time.perf_counter()
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": extract_prompt(query, tier_label, data)},
        ]
        resp = model.chat(messages)
        content = resp["message"]["content"]
        tok = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
        tokens += tok
        elapsed = time.perf_counter() - t_hop
        action, value = _parse_response(content)
        hop_detail.append({"tier": f"T{tier_num}", "action": f"{action} {value}",
                           "tokens": tok, "latency_s": elapsed})

        if action == "ANSWER":
            return {"tier": tier_num, "hops": hops, "answer": value,
                    "miss": False, "tokens": tokens, "gene_id": gene_id,
                    "latency_s": time.perf_counter() - t_start,
                    "hop_detail": hop_detail}

    # T4: walk (simplified — read top-3 neighbor content)
    if neighbors and gene_id in neighbors:
        hops += 1
        nb_list = sorted(neighbors[gene_id], key=lambda x: -x[1])[:3]
        for nb_id, _w in nb_list:
            nb_content = gene_fields.get(nb_id, {}).get("content", "")
            if nb_content:
                t_hop = time.perf_counter()
                messages = [
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": extract_prompt(query, "Neighbor content", nb_content)},
                ]
                resp = model.chat(messages)
                content = resp["message"]["content"]
                tok = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
                tokens += tok
                elapsed = time.perf_counter() - t_hop
                action, value = _parse_response(content)
                hop_detail.append({"tier": "T4", "action": f"{action} {value}",
                                   "tokens": tok, "latency_s": elapsed})
                if action == "ANSWER":
                    return {"tier": 4, "hops": hops, "answer": value,
                            "miss": False, "tokens": tokens, "gene_id": nb_id,
                            "latency_s": time.perf_counter() - t_start,
                            "hop_detail": hop_detail}

    return {"tier": -1, "hops": hops, "answer": None, "miss": True,
            "tokens": tokens, "latency_s": time.perf_counter() - t_start,
            "hop_detail": hop_detail, "gene_id": gene_id}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_snow_cascade.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/snow/cascade.py tests/test_snow_cascade.py
git commit -m "feat(snow): LLM cascade consumer with 4 passing tests"
```

---

### Task 5: Main Harness

**Files:**
- Create: `benchmarks/snow/bench_snow.py`

- [ ] **Step 1: Implement the main harness**

The harness:
1. Loads config + HelixContextManager pointing at benchmark genome
2. Loads `snow_queries.json`
3. For each query: runs T0 retrieval via `query_genes()`, fetches gene fields from SQLite, fetches harmonic_links neighbors
4. Runs oracle cascade → records oracle result
5. Runs LLM cascade → records LLM result
6. Aggregates into scorecard + per-query detail
7. Writes JSON to `results/`

Key implementation details:
- `sys.path.insert(0, REPO)` to import helix_context
- `cfg.genome.path = GENOME_DB` override
- `cfg.ribosome.query_expansion_enabled = False` for LLM-free T0
- `HELIX_DISABLE_HEADROOM=1` env var
- `sys.stdout.reconfigure(encoding='utf-8')` for Windows
- Ollama model wrapper: `OllamaModel` class with `.chat(messages)` that calls `httpx.post` to `localhost:11434/api/chat`
- Gene fields fetched once per query via `SELECT key_values, complement, content, promoter FROM genes WHERE gene_id IN (...)`
- Neighbors fetched via `SELECT gene_id_b, weight FROM harmonic_links WHERE gene_id_a = ? UNION SELECT gene_id_a, weight FROM harmonic_links WHERE gene_id_b = ?`
- CLI: `--model MODEL` (default: `qwen3:4b`), `--model all` runs full ladder, `--genome PATH`

- [ ] **Step 2: Test with a single model (qwen3:4b) on first 5 queries**

```bash
cd /f/Projects/helix-context
python benchmarks/snow/bench_snow.py --model qwen3:4b --limit 5
```

Verify: JSON written to `benchmarks/snow/results/`, scorecard printed to stdout, no crashes.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/snow/bench_snow.py
git commit -m "feat(snow): main harness — oracle + LLM cascade runner"
```

---

### Task 6: Comparison Table

**Files:**
- Create: `benchmarks/snow/snow_compare.py`

- [ ] **Step 1: Implement comparison script**

Reads all `benchmarks/snow/results/snow_*.json` files, builds the multi-model comparison table matching the spec format. Sorted by avg hops ascending.

```bash
python benchmarks/snow/snow_compare.py
```

Should print the full comparison table to stdout with oracle floor at top.

- [ ] **Step 2: Commit**

```bash
git add benchmarks/snow/snow_compare.py
git commit -m "feat(snow): comparison table generator"
```

---

### Task 7: Full Ladder Run + Validation

- [ ] **Step 1: Run oracle-only pass on all 65 queries**

```bash
python benchmarks/snow/bench_snow.py --model oracle-only
```

This populates `oracle_tier` and `oracle_gene_id` in the results. Verify: cascade profile shows distribution across T0-T4, miss rate is low.

- [ ] **Step 2: Run local model ladder**

```bash
python benchmarks/snow/bench_snow.py --model gemma4:e2b
python benchmarks/snow/bench_snow.py --model qwen3:4b
python benchmarks/snow/bench_snow.py --model qwen3:8b
```

- [ ] **Step 3: Generate comparison table**

```bash
python benchmarks/snow/snow_compare.py
```

Verify: table prints clean, numbers look reasonable (hops >= oracle, tokens > oracle, etc.)

- [ ] **Step 4: Commit results + any fixes**

```bash
git add benchmarks/snow/
git commit -m "feat(snow): initial benchmark results — local model ladder"
```

---

### Task 8: Claude API Models (deferred — optional)

Claude models (Haiku/Sonnet/Opus) require sub-agents or direct Anthropic API. This task is deferred until the local ladder validates the harness. Implementation: add a `ClaudeModel` class to `cascade.py` that wraps the Anthropic SDK with the same `.chat()` interface. Or run via the Agent tool with model parameter.

Not blocking for v1 — local models prove the harness works.

---

## Review Fixes (read before implementing)

Spec review (2026-04-16) found 6 HIGH issues. Apply these corrections
when implementing the tasks above:

### Fix 1 (Task 1): Needles field translation [H2]

`needles_50_for_claude.json` uses `{key, value, query}` schema. Must
translate to SNOW's `{expected_answer, accept, query}` when building
`snow_queries.json`:

```python
for needle in needles_50:
    snow_query = {
        "idx": needle["idx"],
        "query": needle["query"],
        "expected_answer": needle["value"],
        "accept": [needle["value"], needle["value"].lower()],
        "source": "needles_50",
        "oracle_tier": None,   # populated by oracle at runtime
        "oracle_gene_id": None,
    }
```

### Fix 2 (Task 1): Tier-stress `target_tier` is authored ground truth [M7]

Tier-stress queries have `target_tier` (set by the author during query
construction) AND `oracle_tier` (set by the oracle at runtime). These
may differ — `target_tier` says "I designed this query to be answerable
at T2", `oracle_tier` says "the oracle actually found it at T1." The
delta is useful for validating the tier-stress set.

### Fix 3 (Task 2): Oracle token counting — only count data inspected [H3]

The oracle implementation counts tokens cumulatively across ALL tiers
it inspects, even if the answer is found at T1 (it still counted T0
entities). This is correct — the oracle DID inspect that data. But
the `tokens` field in each tier loop should only add the NEW data read
at that tier, not re-read earlier data. The current code is correct on
this — each loop only reads its own field. No change needed.

### Fix 4 (Task 3): Add think-tag stripping to prompts.py [H5]

qwen3 models output `<think>...</think>` blocks before the answer.
Add a response cleaner to `prompts.py`:

```python
import re
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def clean_response(text: str) -> str:
    """Strip qwen3 think tags and normalize whitespace."""
    return _THINK_RE.sub("", text).strip()
```

Call `clean_response()` on the model output in `cascade.py` BEFORE
calling `_parse_response()`. Also append ` /no_think` to the system
prompt for Ollama models to suppress thinking where supported.

### Fix 5 (Task 4): Fix hops increment order [H4]

In `cascade.py`, move `hops += 1` AFTER the `if not data: continue`
check so skipped tiers don't inflate the hop count:

```python
for tier_num, field_name, tier_label in TIER_FIELDS:
    data = fields.get(field_name, "")
    if not data:
        continue
    hops += 1  # only count tiers we actually read
    # ... LLM call ...
```

### Fix 6 (Task 5): Config mutation before construction [H1]

The HelixContextManager config must be mutated BEFORE construction.
Follow `precision_probe.py` exactly:

```python
import os
os.environ["HELIX_DISABLE_HEADROOM"] = "1"

from helix_context import HelixContextManager, load_config

cfg = load_config()
cfg.genome.path = GENOME_DB           # MUST be set before construction
cfg.ribosome.query_expansion_enabled = False
cfg.ingestion.rerank_enabled = False

hcm = HelixContextManager(cfg)       # reads cfg at construction time
```

### Fix 7 (Task 5): OllamaModel with timeout + think-strip [H5, H6]

```python
class OllamaModel:
    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout: float = 120.0):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=timeout)  # H6: explicit timeout

    def chat(self, messages):
        resp = self.client.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages,
                  "stream": False,
                  "options": {"temperature": 0, "num_predict": 500}},
        )
        resp.raise_for_status()
        data = resp.json()
        # H5: strip think tags before returning
        from .prompts import clean_response
        data["message"]["content"] = clean_response(
            data["message"]["content"]
        )
        return data
```

### Fix 8 (Task 5): CLI must accept `--model oracle-only` [M10]

Add `oracle-only` as a valid `--model` value. When set, skip the LLM
cascade entirely — run oracle on all queries and write a result JSON
with `llm_summary: null`. The comparison table should handle this
gracefully (show oracle row only, no LLM row).

### Fix 9 (Task 6): Comparison table detail [M8]

`snow_compare.py` must:
- Compute averages EXCLUDING misses for hops/tokens/latency
- Cascade profile counts exclude misses for both oracle and LLM
- Sort models by avg_hops ascending
- Handle missing per-step latency tiers (not all queries reach T4)
- Use integer tier labels (`0, 1, 2, 3, 4`) internally, format as
  `T0, T1, ...` only for display

### Fix 10: Tier label standardization [M9]

Oracle returns `{"tier": 0}` (int). Cascade returns `{"tier": 0}` (int)
in the result dict but `{"tier": "T0"}` (string) in hop_detail. The
harness should use int everywhere in data, string only for display.
Standardize hop_detail to `{"tier": 0, ...}` not `{"tier": "T0", ...}`.

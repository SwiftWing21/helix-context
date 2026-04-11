r"""
N=1000 Needle-in-a-Haystack benchmark with KV-harvested needles.

Generates 1000 needles from pre-extracted key-value facts in the genome,
stratified by source category to avoid single-source bias. Runs each
needle through Helix + downstream model and computes:

  - Context retrieval rate (did the genome express the right gene?)
  - Answer accuracy rate (did the model extract the value?)
  - Per-category breakdown (signal vs noise sources)
  - Failure mode taxonomy (retrieval miss vs extraction miss)

Reproducibility:
  - Uses a snapshot at genome-bench-2026-04-10.db
  - random.seed(42) for stable needle selection
  - Results saved to benchmarks/needle_1000_results.json

Usage:
  HELIX_MODEL=qwen3:4b python benchmarks/bench_needle_1000.py
  HELIX_MODEL=qwen3:4b N=200 python benchmarks/bench_needle_1000.py  # sanity check

  # Upload the most recent results to a HuggingFace dataset (manual trigger).
  # Requires `huggingface_hub` installed and HF_TOKEN env var or prior `hf auth login`.
  HF_REPO=SwiftWing21/helix-needle-bench python benchmarks/bench_needle_1000.py --upload
"""

import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from typing import Optional

import httpx

# Harness version — bump when filter logic changes so older result files stay
# comparable. v1 = original (phantom-prone) harvest. v2 = literal-value filter,
# dotted-chain reject, assignment-context check, word-boundary retrieval match.
HARNESS_VERSION = 2

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
GENOME_DB = os.environ.get("GENOME_DB", "F:/Projects/helix-context/genome-bench-2026-04-10.db")
MODEL = os.environ.get("HELIX_MODEL", "qwen3:4b")
N_TOTAL = int(os.environ.get("N", "1000"))
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_PATH = os.environ.get("OUTPUT", "F:/Projects/helix-context/benchmarks/needle_1000_results.json")
# Opt-in to legacy v1 behavior (no-op fixes) for reproducing old runs.
LEGACY_HARVEST = os.environ.get("BENCH_LEGACY_HARVEST") == "1"

# Cold-tier opt-in (C.2 of B->C, 2026-04-10).
# When set to "1"/"true", every /context call sends include_cold=true in the
# request body, forcing the server to consult heterochromatin genes via SEMA
# cosine fallthrough. Lets a single bench script measure both hot-only and
# hot+cold ceilings on the same harness/seed/model without restarting the
# server. None/false means honor whatever the server's [context]
# cold_tier_enabled config flag is set to.
_INCLUDE_COLD_RAW = os.environ.get("INCLUDE_COLD_TIER", "").lower()
if _INCLUDE_COLD_RAW in ("1", "true", "yes", "on"):
    INCLUDE_COLD_TIER: Optional[bool] = True
elif _INCLUDE_COLD_RAW in ("0", "false", "no", "off"):
    INCLUDE_COLD_TIER = False
else:
    INCLUDE_COLD_TIER = None  # Honor server config

# Stratification targets (sum to 1.0)
# Weights favor signal-bearing sources but keep noise at ~30% to test
# noise resistance (mirrors the natural ~34% signal / 66% noise genome).
STRATIFICATION = {
    "education_public": 0.30,
    "helix":            0.15,
    "cosmic":           0.12,
    "tally":            0.08,
    "scorerift":        0.05,
    "steam":            0.25,  # noise — included to stress-test retrieval
    "other":            0.05,
}

SOURCE_BUCKETS = {
    "steam": ["SteamLibrary", "steamapps", "Hades/", "BeamNG", "Factorio", "Dyson Sphere"],
    "education_public": ["biged-rs", "BigEd/", "fleet/", "Education"],
    "helix": ["helix-context", "helix_context"],
    "cosmic": ["CosmicTasha", "cosmictasha", "novabridge"],
    "tally": ["BookKeeper"],
    "scorerift": ["two-brain-audit", "scorerift"],
}


def categorize(src: str) -> str:
    if not src:
        return "other"
    for name, patterns in SOURCE_BUCKETS.items():
        if any(p in src for p in patterns):
            return name
    return "other"


# v1 prose-key blacklist — exact set shipped in harness v1
_PROSE_KEYS_V1 = frozenset({"key", "value", "name", "type", "id"})

# v2 expansion — generic English content fields that almost always harvest
# phantom values from prose (docstrings, README lines, comment blocks).
_PROSE_KEYS_V2 = _PROSE_KEYS_V1 | frozenset({
    "note", "notes", "description", "desc", "summary",
    "comment", "comments", "remark", "text", "label",
    "title", "caption", "message", "msg", "content",
})

_DOTTED_CHAIN_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$")
_IDENT_WORDY_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\(")  # function-call shape


def _looks_like_literal_value(v: str) -> bool:
    """True if v looks like an actual assigned value, not a docstring word.

    Accepts: anything with digits, punctuation, underscores, camelCase,
    long ALL_CAPS constants.
    Rejects: single plain English words (lowercase, TitleCase, short acronyms)
    — these are overwhelmingly sentence fragments from docstrings/comments.
    """
    if not v:
        return False
    # Has any non-alpha char (digits, punctuation, path sep, etc.) → literal-ish
    if not v.isalpha():
        return True
    # camelCase / mixedCase identifier
    if v != v.lower() and v != v.upper() and not v.istitle():
        return True
    # Long ALL_CAPS constant (enum/const like "DEFAULT_TIMEOUT" — though with
    # underscores it wouldn't reach here; bare "RUNNING", "PENDING", etc.)
    if v.isupper() and len(v) >= 5:
        return True
    # Otherwise: single plain English word — reject as phantom-prone
    return False


def _word_boundary_match(text: str, value: str) -> bool:
    """Match `value` inside `text` with word boundaries when safe.

    For alphanumeric-only short values (≤20 chars), requires that the match
    is not surrounded by word characters on either side — prevents "api"
    matching "apis", "api_key", or "rapid".
    For values containing punctuation (paths, URLs, dotted names, etc.) or
    values longer than 20 chars, falls back to plain substring.
    """
    if not value or not text:
        return False
    v_low = value.lower()
    t_low = text.lower()
    if re.fullmatch(r"[a-z0-9_]+", v_low) and len(v_low) <= 20:
        return bool(re.search(
            rf"(?<![a-z0-9_]){re.escape(v_low)}(?![a-z0-9_])", t_low
        ))
    return v_low in t_low


def _value_in_assignment_context(content: str, k: str, v: str, window: int = 120) -> bool:
    """Return True if `v` appears near `k` with an assignment separator.

    Scans for occurrences of the key and checks whether the next `window`
    characters contain an `=` or `:` separator followed by the value. This
    rejects phantom KVs where the value only appears in a docstring or
    comment far from any actual assignment of the key.

    Matches are case-insensitive on the separator side but respect word
    boundaries for the key itself.
    """
    if not content or not k or not v:
        return False
    k_lower = k.lower()
    c_lower = content.lower()
    v_lower = v.lower()
    # Iterate over word-boundary matches of the key (case-insensitive)
    key_pat = re.compile(rf"(?<![a-z0-9_]){re.escape(k_lower)}(?![a-z0-9_])")
    for m in key_pat.finditer(c_lower):
        start = m.end()
        snippet = c_lower[start:start + window]
        # Require a : or = separator before the value inside the window
        # Allow type annotations between (e.g. "key: bool = False" or "key: 'v'")
        sep_pos = -1
        for ch in (":", "="):
            p = snippet.find(ch)
            if p != -1 and (sep_pos == -1 or p < sep_pos):
                sep_pos = p
        if sep_pos == -1:
            continue
        after_sep = snippet[sep_pos + 1:]
        if v_lower in after_sep:
            return True
    return False


def is_quality_kv(k: str, v: str) -> bool:
    """Filter out low-value KVs — generic types, placeholders, short keys.

    v2 adds: dotted-identifier-chain rejection (e.g. "os.path.join"),
    function-call shape rejection (e.g. "foo(bar)"), single-English-word
    rejection via _looks_like_literal_value, and an expanded prose-key
    blacklist (note, description, comment, ...).
    """
    if len(k) < 3 or len(k) > 30:
        return False
    if len(v) < 2 or len(v) > 60:
        return False
    v_lower = v.lower()
    if v_lower in ("string", "number", "bool", "true", "false", "none", "null", "-", "value"):
        return False
    if v_lower in ("any", "object", "array", "void", "undefined"):
        return False
    if not re.search(r"[a-zA-Z0-9]", v):
        return False
    if v.startswith("<") or v.endswith(">"):
        return False
    prose_keys = _PROSE_KEYS_V1 if LEGACY_HARVEST else _PROSE_KEYS_V2
    if k.lower() in prose_keys:
        return False
    # v2: harness-version-gated filters
    if not LEGACY_HARVEST:
        # Reject dotted Python attribute chains (os.path.join, obj.attr.method)
        if _DOTTED_CHAIN_RE.match(v):
            return False
        # Reject function-call shapes (foo(bar), compress_text(content))
        if _IDENT_WORDY_RE.search(v):
            return False
        # Reject single plain English words — almost always docstring phantoms
        if not _looks_like_literal_value(v):
            return False
    return True


def harvest_needles(db_path: str, n: int, seed: int) -> list[dict]:
    """Stratified-sample N needles from unique, high-quality KVs."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT gene_id, COALESCE(source_id, ''), key_values, content
        FROM genes
        WHERE key_values IS NOT NULL AND LENGTH(key_values) > 10
          AND chromatin < 2
        """
    ).fetchall()
    conn.close()

    # Parse and filter
    buckets: dict[str, list[dict]] = defaultdict(list)
    value_gene_count: Counter = Counter()

    # First pass — collect all candidates and track value uniqueness
    all_candidates = []
    for gene_id, src, kv_raw, content in rows:
        try:
            kvs = json.loads(kv_raw)
        except Exception:
            continue
        for kv in kvs:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k, v = k.strip(), v.strip()
            if not is_quality_kv(k, v):
                continue
            # Skip if the value doesn't literally appear in the gene content
            # (sanity: some KV extractions were noisy)
            if v.lower() not in (content or "").lower():
                continue
            # v2: require key+value to appear in an assignment-like context.
            # Catches cases where the value exists in the content but only in
            # a docstring sentence or a comment far from the real assignment.
            if not LEGACY_HARVEST:
                if not _value_in_assignment_context(content or "", k, v):
                    continue
            key = f"{k}={v}".lower()
            value_gene_count[key] += 1
            all_candidates.append({
                "gene_id": gene_id,
                "source": src,
                "category": categorize(src),
                "key": k,
                "value": v,
                "kv_key": key,
            })

    # Second pass — keep only globally-unique values
    unique = [c for c in all_candidates if value_gene_count[c["kv_key"]] == 1]
    print(f"Total quality KVs:     {len(all_candidates):,}", file=sys.stderr)
    print(f"Globally-unique KVs:   {len(unique):,}", file=sys.stderr)

    # Bucket by category
    for c in unique:
        buckets[c["category"]].append(c)

    print(file=sys.stderr)
    print("Available by category:", file=sys.stderr)
    for cat, items in sorted(buckets.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<20} {len(items):>6,}", file=sys.stderr)

    # Stratified sample
    rng = random.Random(seed)
    selected: list[dict] = []
    for cat, weight in STRATIFICATION.items():
        target = int(n * weight)
        pool = buckets.get(cat, [])
        if not pool:
            continue
        # Sample without replacement; if pool smaller than target, take all
        take = min(target, len(pool))
        rng.shuffle(pool)
        selected.extend(pool[:take])

    # Top up to N if we fell short (some categories had small pools)
    short = n - len(selected)
    if short > 0:
        remaining = [c for cat, items in buckets.items() for c in items
                     if c not in selected and cat not in ("other",)]
        rng.shuffle(remaining)
        selected.extend(remaining[:short])

    # Deterministic order
    rng.shuffle(selected)
    return selected[:n]


def build_query(needle: dict) -> str:
    """Generate a natural-language question for a KV fact."""
    k = needle["key"]
    # Turn snake_case / camelCase keys into readable phrases
    phrase = re.sub(r"[_\-]+", " ", k)
    phrase = re.sub(r"([a-z])([A-Z])", r"\1 \2", phrase).lower().strip()
    # Pick a template based on key shape
    if any(t in phrase for t in ("port", "size", "count", "limit", "threshold", "budget")):
        return f"What is the {phrase} in the {needle['category']} source?"
    if "path" in phrase or "url" in phrase or "file" in phrase:
        return f"What is the {phrase} mentioned in the code?"
    if "name" in phrase or "title" in phrase:
        return f"What is the {phrase}?"
    return f"What is the value of {phrase}?"


def run_needle(client: httpx.Client, needle: dict) -> dict:
    """Execute a single needle test through Helix + downstream model."""
    query = needle["query"]
    accept = needle["value"].lower()

    result = {
        "gene_id": needle["gene_id"],
        "category": needle["category"],
        "key": needle["key"],
        "value": needle["value"],
        "query": query,
    }

    # Step 1: retrieval (stateless — no benchmark_mode flag needed for current harness)
    t0 = time.time()
    context_payload = {
        "query": query,
        "decoder_mode": "none",
    }
    if INCLUDE_COLD_TIER is not None:
        context_payload["include_cold"] = INCLUDE_COLD_TIER
    try:
        resp = client.post(f"{HELIX_URL}/context", json=context_payload, timeout=30)
    except Exception as e:
        result.update({
            "retrieved": False, "answered": False,
            "context_latency_s": time.time() - t0, "proxy_latency_s": 0,
            "error": f"context endpoint: {e}",
        })
        return result

    ctx_latency = time.time() - t0
    if resp.status_code != 200:
        result.update({
            "retrieved": False, "answered": False,
            "context_latency_s": ctx_latency, "proxy_latency_s": 0,
            "error": f"context HTTP {resp.status_code}",
        })
        return result

    data = resp.json()
    entry = data[0] if data else {}
    content = entry.get("content", "")
    health = entry.get("context_health", {})
    agent_meta = entry.get("agent", {})

    # v2: word-boundary-aware match (plain substring in v1/legacy)
    if LEGACY_HARVEST:
        retrieved = accept in content.lower()
    else:
        retrieved = _word_boundary_match(content, needle["value"])

    # Step 2: downstream model extraction
    t1 = time.time()
    try:
        proxy_resp = client.post(f"{HELIX_URL}/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 128},
        }, timeout=90)
    except Exception as e:
        result.update({
            "retrieved": retrieved, "answered": False,
            "context_latency_s": round(ctx_latency, 3),
            "proxy_latency_s": time.time() - t1,
            "error": f"proxy: {e}",
            "ellipticity": health.get("ellipticity", 0),
            "genes_expressed": health.get("genes_expressed", 0),
        })
        return result

    proxy_latency = time.time() - t1
    answer_text = ""
    answered = False
    if proxy_resp.status_code == 200:
        try:
            choices = proxy_resp.json().get("choices", [])
            if choices:
                answer_text = choices[0].get("message", {}).get("content", "") or ""
                if LEGACY_HARVEST:
                    answered = accept in answer_text.lower()
                else:
                    answered = _word_boundary_match(answer_text, needle["value"])
        except Exception:
            pass

    injected = agent_meta.get("total_tokens_est", 0)
    budget = agent_meta.get("budget_tokens_est", 15000)
    compression = agent_meta.get("compression_ratio", 0)

    result.update({
        "retrieved": retrieved,
        "answered": answered,
        "context_latency_s": round(ctx_latency, 3),
        "proxy_latency_s": round(proxy_latency, 3),
        "ellipticity": round(health.get("ellipticity", 0), 3),
        "genes_expressed": health.get("genes_expressed", 0),
        "budget_tier": agent_meta.get("budget_tier", "broad"),
        "budget_tokens_est": budget,
        "injected_tokens_est": injected,
        "compression_ratio": round(compression, 3) if compression else 0,
        "budget_utilization": round(injected / budget, 3) if budget else 0,
        "cold_tier_used": agent_meta.get("cold_tier_used", False),
        "cold_tier_count": agent_meta.get("cold_tier_count", 0),
        "answer_preview": answer_text[:150],
    })
    return result


def summarize(results: list[dict]) -> dict:
    """Compute aggregate statistics across all needles."""
    n = len(results)
    retrieved = sum(1 for r in results if r.get("retrieved"))
    answered = sum(1 for r in results if r.get("answered"))
    errors = sum(1 for r in results if r.get("error"))

    # Per-category breakdown
    by_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "retrieved": 0, "answered": 0})
    for r in results:
        c = r.get("category", "other")
        by_cat[c]["n"] += 1
        if r.get("retrieved"):
            by_cat[c]["retrieved"] += 1
        if r.get("answered"):
            by_cat[c]["answered"] += 1

    # Failure taxonomy
    failures = {
        "retrieval_miss": sum(1 for r in results if not r.get("retrieved")),
        "extraction_miss": sum(1 for r in results if r.get("retrieved") and not r.get("answered")),
        "error": errors,
    }

    # Latency percentiles
    ctx_lat = sorted(r["context_latency_s"] for r in results if "context_latency_s" in r)
    proxy_lat = sorted(r["proxy_latency_s"] for r in results if "proxy_latency_s" in r)

    def pct(lst, p):
        if not lst:
            return 0
        k = int(len(lst) * p / 100)
        return round(lst[min(k, len(lst) - 1)], 3)

    # Token economics — skip zeros (missing data in older runs)
    injected = [r.get("injected_tokens_est", 0) for r in results if r.get("injected_tokens_est")]
    compression = [r.get("compression_ratio", 0) for r in results if r.get("compression_ratio")]
    budget_util = [r.get("budget_utilization", 0) for r in results if r.get("budget_utilization")]
    genes_exp = [r.get("genes_expressed", 0) for r in results]
    budget = [r.get("budget_tokens_est", 0) for r in results if r.get("budget_tokens_est")]

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0

    # Answered-per-kilotoken — the core efficiency metric: how many correct
    # answers per 1000 injected context tokens. Higher = more token-efficient.
    total_injected = sum(injected)
    answers_per_ktoken = round(answered / (total_injected / 1000), 4) if total_injected else 0

    return {
        "n": n,
        "retrieval_rate": round(retrieved / max(n, 1), 4),
        "answer_accuracy_rate": round(answered / max(n, 1), 4),
        "retrieved": retrieved,
        "answered": answered,
        "errors": errors,
        "failure_modes": failures,
        "by_category": {k: {**v,
                            "retrieval_rate": round(v["retrieved"] / max(v["n"], 1), 4),
                            "answer_rate": round(v["answered"] / max(v["n"], 1), 4)}
                        for k, v in by_cat.items()},
        "latency": {
            "context_p50_s": pct(ctx_lat, 50),
            "context_p95_s": pct(ctx_lat, 95),
            "proxy_p50_s": pct(proxy_lat, 50),
            "proxy_p95_s": pct(proxy_lat, 95),
        },
        "tokens": {
            "avg_injected": avg(injected),
            "avg_budget": avg(budget),
            "avg_budget_utilization": avg(budget_util),
            "avg_compression_ratio": avg(compression),
            "avg_genes_expressed": avg(genes_exp),
            "total_injected": total_injected,
            "answers_per_ktoken": answers_per_ktoken,
        },
    }


def upload_to_hf(results_paths: list[str], repo: str, private: bool = True) -> None:
    """Upload one or more result JSONs + a combined README to a HF dataset repo.

    Does NOT auto-run — invoke via --upload after a finished benchmark.
    The dataset is created as private by default; pass --public to share.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)

    # Load every result file up front so we can build one combined README
    runs = []
    for rp in results_paths:
        if not os.path.isfile(rp):
            print(f"ERROR: results file not found: {rp}", file=sys.stderr)
            sys.exit(1)
        with open(rp, "r", encoding="utf-8") as f:
            runs.append((rp, json.load(f)))

    # Largest N becomes the "headline" N for the dataset title
    max_n = max((d.get("n", 0) for _, d in runs), default=0)

    # Markdown results table — retrieval/answer + token economics
    def fmt(v, suffix="", missing="—"):
        return f"{v}{suffix}" if v else missing

    rows = []
    for rp, d in runs:
        s = d.get("summary", {})
        lat = s.get("latency", {})
        tok = s.get("tokens", {})  # only present in runs after the token-capture patch
        avg_inj = tok.get("avg_injected", 0)
        compr = tok.get("avg_compression_ratio", 0)
        eff = tok.get("answers_per_ktoken", 0)
        util = tok.get("avg_budget_utilization", 0)
        rows.append(
            f"| `{os.path.basename(rp)}` | {d.get('n','?')} | "
            f"{d.get('model','?')} | "
            f"{s.get('retrieval_rate', 0)*100:.1f}% | "
            f"{s.get('answer_accuracy_rate', 0)*100:.1f}% | "
            f"{fmt(avg_inj)} | "
            f"{fmt(compr, 'x')} | "
            f"{fmt(util)} | "
            f"{fmt(eff)} | "
            f"{lat.get('context_p50_s','?')}s / {lat.get('context_p95_s','?')}s | "
            f"{s.get('total_time_min','?')} min |"
        )
    table = (
        "| file | N | model | retrieval | answer | inj tokens (avg) | compression | budget util | ans/ktoken | ctx p50/p95 | total |\n"
        "|------|---|-------|-----------|--------|------------------|-------------|-------------|------------|-------------|-------|\n"
        + "\n".join(rows)
    )

    # Note if any run is missing the token capture (pre-patch runs)
    missing_tokens = [os.path.basename(rp) for rp, d in runs if not d.get("summary", {}).get("tokens", {}).get("avg_injected")]
    missing_note = ""
    if missing_tokens:
        missing_note = (
            "\n> **Note:** token economics columns show `—` for runs generated "
            "before the token-capture patch. Re-run those benchmarks to populate "
            "`avg_injected`, `compression`, `budget_util`, and `ans/ktoken`.\n"
        )

    # Surface genome + seed from the first run (all runs should share the snapshot)
    first = runs[0][1]
    genome = os.path.basename(first.get("genome_snapshot", "?"))
    seed = first.get("seed", "?")

    readme = f"""---
license: apache-2.0
tags:
- helix-context
- needle-in-haystack
- retrieval
- long-context
- benchmark
size_categories:
- n<1K
---

# Helix Context — Needle-in-a-Haystack Results (N≤{max_n})

Retrieval + extraction benchmark for the **Helix Context** genome compression
system — a DNA-inspired long-context memory that stores codebase knowledge as
compressed "genes" (~7x compression ratio) and expresses them on demand via
promoter matching.

Each needle is a globally-unique key/value fact harvested from the genome.
For every needle, the benchmark (1) asks Helix to express relevant genes
(`/context` endpoint), then (2) asks the downstream model to extract the value
(`/v1/chat/completions`). A needle counts as **retrieved** if the expected value
appears in the expressed context, and **answered** if the model emits it in
plain text.

## Results

{table}
{missing_note}
## Metric glossary
- **retrieval** — % of needles where the expected value appears in the expressed context
- **answer** — % of needles where the downstream model emits the expected value
- **inj tokens (avg)** — average context tokens injected into the downstream model per query
- **compression** — avg `total_tokens_est / raw_source_tokens` ratio from the Helix decoder (higher = more aggressive compression)
- **budget util** — avg `injected / budget_ceiling` (how tight the budget was against the cap)
- **ans/ktoken** — answered needles per 1000 injected context tokens (efficiency metric; higher = more answers per token spent)

## Run metadata
- **Genome snapshot:** `{genome}`
- **Seed:** {seed}
- **Stratification:** 30% public education repos, 15% helix-context, 12% CosmicTasha,
  8% BookKeeper, 5% scorerift, 25% steam (noise bucket), 5% other
- **Harness:** [`bench_needle_1000.py`](https://github.com/) — stratified KV harvest,
  globally-unique values only, natural-language query synthesis

## Failure taxonomy
Two independent failure modes are tracked per run:
- **retrieval_miss** — the genome didn't express a gene containing the answer
- **extraction_miss** — the gene *was* expressed but the downstream model failed
  to extract it from the context

See each run's `summary.failure_modes` for the breakdown.

## Files
"""
    for rp, _ in runs:
        readme += f"- `{os.path.basename(rp)}` — full results (summary + per-needle rows)\n"

    api = HfApi()
    print(f"Creating/updating dataset repo: {repo} (private={private})")
    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)

    for rp, _ in runs:
        print(f"Uploading {os.path.basename(rp)}")
        api.upload_file(
            path_or_fileobj=rp,
            path_in_repo=os.path.basename(rp),
            repo_id=repo,
            repo_type="dataset",
        )

    readme_path = os.path.join(os.path.dirname(results_paths[0]), "_hf_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"Done → https://huggingface.co/datasets/{repo}")


def main():
    # --upload short-circuit (no benchmark run, just publish existing results)
    if "--upload" in sys.argv:
        repo = os.environ.get("HF_REPO")
        if not repo:
            print("ERROR: set HF_REPO=<owner/name> to upload", file=sys.stderr)
            sys.exit(1)
        private = "--public" not in sys.argv
        # --files a,b,c  →  upload those specific result JSONs (comma-separated)
        # Otherwise, fall back to the default OUTPUT_PATH
        files: list[str] = []
        for i, arg in enumerate(sys.argv):
            if arg == "--files" and i + 1 < len(sys.argv):
                files = [f.strip() for f in sys.argv[i + 1].split(",") if f.strip()]
                break
        if not files:
            files = [OUTPUT_PATH]
        upload_to_hf(files, repo, private=private)
        return

    print(f"=== N={N_TOTAL} Needle benchmark ===")
    print(f"Genome:  {GENOME_DB}")
    print(f"Server:  {HELIX_URL}")
    print(f"Model:   {MODEL}")
    print(f"Seed:    {SEED}")
    print()

    # Harvest needles
    print("Harvesting needles from genome...")
    needles = harvest_needles(GENOME_DB, N_TOTAL, SEED)
    print(f"Selected {len(needles)} needles")
    print()

    # Attach queries
    for n in needles:
        n["query"] = build_query(n)

    # Force unbuffered stdout so progress is visible in background runs
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Incremental results file — write every needle so we don't lose
    # progress on a hang or crash. Opened here (before monitor preflight)
    # so the monitor can reference its path during pre-flight logging.
    incremental_path = OUTPUT_PATH.replace(".json", ".incremental.jsonl")
    # Create as empty so line-counting works immediately
    open(incremental_path, "w").close()

    # ── Pre-flight monitor check ──────────────────────────────────
    # Verifies: Helix alive, Ollama alive, target model loaded, no unauthorized
    # models in VRAM, genome snapshot present. Aborts before any work starts
    # if conditions aren't clean. Opt out with HELIX_SKIP_MONITOR=1.
    monitor = None
    if not os.environ.get("HELIX_SKIP_MONITOR"):
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from benchmark_monitor import BenchmarkMonitor, MonitorConfig

            mon_cfg = MonitorConfig(
                check_interval_s=float(os.environ.get("MONITOR_INTERVAL", "90")),
                stall_threshold_s=float(os.environ.get("MONITOR_STALL", "180")),
                strict_hash=os.environ.get("MONITOR_STRICT_HASH") == "1",
            )
            monitor = BenchmarkMonitor(
                benchmark_model=MODEL,
                incremental_output_path=incremental_path,
                total_needles=len(needles),
                helix_url=HELIX_URL,
                genome_snapshot_path=GENOME_DB,
                config=mon_cfg,
            )
            if not monitor.preflight():
                print("[bench] Pre-flight failed — aborting benchmark.", flush=True)
                sys.exit(1)
        except ImportError:
            print("[bench] benchmark_monitor not available — running unmonitored", flush=True)
            monitor = None
        except Exception as e:
            print(f"[bench] Monitor setup failed: {e} — running unmonitored", flush=True)
            monitor = None

    # HTTP client. Per-call timeouts + disable keep-alive pooling
    # (pooled connections silently drop during long runs and block on read).
    client = httpx.Client(
        timeout=httpx.Timeout(connect=10, read=60, write=10, pool=10),
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
    )
    try:
        health = client.get(f"{HELIX_URL}/health").json()
        print(f"Server: {health['status']}, ribosome={health['ribosome']}, genes={health['genes']}")
    except Exception:
        print(f"ERROR: cannot reach Helix at {HELIX_URL}")
        sys.exit(1)
    print(flush=True)

    # Start the monitor background thread now that preflight passed
    if monitor is not None:
        monitor.start()

    # Reopen incremental file in append mode for the benchmark loop
    incremental_f = open(incremental_path, "a")

    # Run
    results = []
    start = time.time()
    aborted_by_monitor = False
    try:
        for i, needle in enumerate(needles, 1):
            # Monitor abort check — if conditions went bad, stop the run
            if monitor is not None and monitor.should_abort():
                print(f"[bench] Monitor requested abort at needle {i}/{len(needles)}", flush=True)
                aborted_by_monitor = True
                break

            r = run_needle(client, needle)
            results.append(r)
            incremental_f.write(json.dumps(r) + "\n")
            incremental_f.flush()

            # More frequent progress: every 10 needles
            if i % 10 == 0 or i == len(needles):
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(needles) - i) / max(rate, 0.001)
                done_r = sum(1 for x in results if x.get("retrieved"))
                done_a = sum(1 for x in results if x.get("answered"))
                print(f"  [{i:>4}/{len(needles)}] "
                      f"retr={done_r/i*100:5.1f}% ans={done_a/i*100:5.1f}% "
                      f"elapsed={elapsed/60:5.1f}m eta={eta/60:5.1f}m",
                      flush=True)
    finally:
        incremental_f.close()
        if monitor is not None:
            monitor.stop()

    total_time = time.time() - start

    # Summarize
    summary = summarize(results)
    summary["total_time_s"] = round(total_time, 1)
    summary["total_time_min"] = round(total_time / 60, 1)

    # Attach monitor report if we had one
    monitor_report: Optional[dict] = None
    if monitor is not None:
        monitor_report = monitor.final_report()
        summary["run_clean"] = monitor_report["run_clean"]
        summary["monitor_status"] = monitor_report["status"]
        summary["monitor_incidents"] = monitor_report["incident_count"]
        if monitor_report["incident_count"] > 0:
            print()
            print(f"[!] Monitor flagged {monitor_report['incident_count']} incident(s):")
            for inc in monitor_report["incidents"]:
                print(f"  [{inc['severity'].upper()}] {inc['type']}: {inc['reason']}")
            print(f"  Run clean: {monitor_report['run_clean']}")
            print(f"  Full log:  {monitor_report['monitor_log_path']}")

    print()
    print("=" * 66)
    print(f"N = {summary['n']}")
    print(f"Retrieval rate:     {summary['retrieval_rate']*100:5.1f}%  ({summary['retrieved']}/{summary['n']})")
    print(f"Answer accuracy:    {summary['answer_accuracy_rate']*100:5.1f}%  ({summary['answered']}/{summary['n']})")
    print(f"Errors:             {summary['errors']}")
    print(f"Total time:         {summary['total_time_min']:.1f} min")
    print()
    print("Failure modes:")
    for k, v in summary["failure_modes"].items():
        print(f"  {k:<20} {v:>5}")
    print()
    print("By category:")
    for cat, stats in sorted(summary["by_category"].items(), key=lambda x: -x[1]["n"]):
        print(f"  {cat:<20} n={stats['n']:>4}  retr={stats['retrieval_rate']*100:5.1f}%  ans={stats['answer_rate']*100:5.1f}%")
    print()
    print("Latency:")
    print(f"  context p50: {summary['latency']['context_p50_s']}s  p95: {summary['latency']['context_p95_s']}s")
    print(f"  proxy   p50: {summary['latency']['proxy_p50_s']}s  p95: {summary['latency']['proxy_p95_s']}s")

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harness_version": 1 if LEGACY_HARVEST else HARNESS_VERSION,
        "n": summary["n"],
        "model": MODEL,
        "seed": SEED,
        "genome_snapshot": GENOME_DB,
        "stratification": STRATIFICATION,
        "summary": summary,
        "needles": results,
    }
    if monitor_report is not None:
        output["monitor"] = monitor_report
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Results saved to {OUTPUT_PATH}")

    # Exit code 2 = aborted by monitor; 0 = clean finish (even with warnings)
    if aborted_by_monitor:
        sys.exit(2)


if __name__ == "__main__":
    main()

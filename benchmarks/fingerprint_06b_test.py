"""
Fingerprint-only retrieval test on qwen3:0.6b (smallest SIKE model).

Question: can a 0.6B model reason from tier-score fingerprints alone —
triage which genes to read, navigate the mathematical landscape, know
what it doesn't know?

If yes: fingerprint mode is truly scale-invariant, same as SIKE 10/10
retrieval. The math works at every parameter count.

Sends 3 fingerprint prompts directly to Ollama (not through helix proxy)
so we get raw model behaviour with ONLY the fingerprint in context.
"""

from __future__ import annotations

import json
import sys
import time

import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = "qwen3:0.6b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 120.0

QUERIES = [
    {
        "idx": 0,
        "question": "What is the queries path mentioned in the code?",
        "fingerprint": """--- GENE RANK 0 (fingerprint only) ---
gene_id: ade070a94a3a118c
source: F:/Projects/Education/fleet/knowledge/security/reviews/security_review_20260401_0929.md
score: 21.53
tiers: {fts5: 6.0, sema_boost: 0.53, harmonic: 3.0}
domains: [sql]
entities: [SQL, db.py, NETWORK, Binds, factorio_train.py]

--- GENE RANK 1 (fingerprint only) ---
gene_id: 3d1072d659bc091b
source: (no source_id - session exchange)
score: 15.53
tiers: {tag_prefix: 3.0, fts5: 6.0, splade: 2.93, sema_boost: 0.61, harmonic: 3.0}
domains: [code_analysis, configuration_management, documentation]
entities: [genome.db, GENE_block, helix-context/README.md, path_parameter]

--- GENE RANK 2 (fingerprint only) ---
gene_id: 6aabf8366c1cc3cd
source: F:/SteamLibrary/steamapps/common/Hades/Content/Scripts/CodexData.lua
score: 14.59
tiers: {tag_prefix: 9.0, sema_boost: 0.59, harmonic: 3.0}
domains: []
entities: [NPC_Thanatos_01, CodexData_Thanatos_04, EntryIndex, CodexData_0023]""",
    },
    {
        "idx": 2,
        "question": "What is the value of rbac?",
        "fingerprint": """--- GENE RANK 0 (fingerprint only) ---
gene_id: a00785cb2a73cbf7
source: F:/Projects/BookKeeper/tests/test_rbac_routes.py
score: 29.00
tiers: {tag_exact: 6.0, tag_prefix: 6.0, fts5: 6.0, sema_boost: 0.37, lex_anchor: 4.14, harmonic: 3.0}
domains: [rbac, test, config, auth, bookkeeper, api, pytest, json, html]
entities: [RBAC, tmp_config_and_db, rbac_config_and_db, register_user(conn, ValueError, authenticate(conn, API]

--- GENE RANK 1 (fingerprint only) ---
gene_id: 0aeb4d6440b8d0b9
source: F:/Projects/BookKeeper/docs/superpowers/plans/2026-03-25-rbac-completion.md
score: 26.15
tiers: {tag_exact: 6.0, tag_prefix: 3.0, fts5: 6.0, sema_boost: 0.51, lex_anchor: 4.14, harmonic: 3.0}
domains: [test, rbac, bookkeeper, auth, config, api, jwt, toml, pytest, git]
entities: [bookkeeper, feat(rbac): auth service, JWT, Task 3, Route-Level Enforcement, config keys, RBAC]

--- GENE RANK 2 (fingerprint only) ---
gene_id: 1f96be1307be7595
source: F:/Projects/BookKeeper/bookkeeper/tenancy/rbac.py
score: 26.11
tiers: {tag_exact: 6.0, tag_prefix: 3.0, fts5: 6.0, sema_boost: 0.47, lex_anchor: 4.14, harmonic: 3.0}
domains: [bookkeeper, rbac, config]
entities: [RBAC, bookkeeper, Role, Add, Excel, PDF, Categorize, Perform]""",
    },
    {
        "idx": 10,
        "question": "What is the value of model?",
        "fingerprint": """--- GENE RANK 0 (fingerprint only) ---
gene_id: 89acbe4243341070
source: F:/Projects/Education/fleet/knowledge/summaries/2026-03-19_summary.md
score: 49.29
tiers: {tag_exact: 12.0, tag_prefix: 6.0, fts5: 6.0, splade: 3.5, lex_anchor: 17.29, harmonic: 3.0}
domains: [ollama, llm, rest, compliance, deploy, security, gpu]
entities: [GPU, LLM, Ollama, VLAN]

--- GENE RANK 1 (fingerprint only) ---
gene_id: 86c533bdd1b15f81
source: F:/Projects/Education/docs/superpowers/specs/2026-03-28-factorio-ml-overhaul-design.md
score: 49.05
tiers: {tag_exact: 12.0, tag_prefix: 6.0, fts5: 6.0, splade: 3.27, lex_anchor: 17.29, harmonic: 3.0}
domains: [llm, agent, config, orchestrator, lua, ollama, vector, toml, oauth, gpu]
entities: [GPU, LLM, Ollama, Factorio, VRAM]

--- GENE RANK 2 (fingerprint only) ---
gene_id: 5c6a5221458ee10d
source: F:/Projects/Education/fleet/skills/model_suite.py
score: 43.07
tiers: {tag_exact: 6.0, tag_prefix: 3.0, fts5: 6.0, splade: 5.13, lex_anchor: 14.44, sema_boost: 1.50, harmonic: 3.0}
domains: [ollama, gpu, config, hardware, llm]
entities: [SentenceTransformer, all-MiniLM-L6-v2, GPU, VRAM, RTX, Ollama]""",
    },
]

SYSTEM_PROMPT = """You are a small local LLM consuming retrieval fingerprints from a knowledge store called Helix. You receive ONLY the mathematical fingerprint of top-3 retrieved genes - tier scores, source file, domain tags, and named entities. You do NOT get actual content.

Your job: answer the question if the fingerprint gives you enough to reason from. If not, say which gene_id you would need to READ and why. Be concise. /no_think"""


def ask_model(question: str, fingerprint: str) -> dict:
    prompt = f"QUESTION: {question}\n\n{fingerprint}\n\nYour answer (or which gene_id to read and why):"

    t0 = time.perf_counter()
    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 300},
        },
        timeout=TIMEOUT,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    return {
        "response": data.get("response", ""),
        "elapsed_s": elapsed,
        "eval_count": data.get("eval_count", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
    }


def main() -> int:
    print(f"[0.6b] model: {MODEL}")
    print(f"[0.6b] testing fingerprint-only retrieval on {len(QUERIES)} queries\n")

    for q in QUERIES:
        print(f"=== Query {q['idx']}: {q['question']}")
        result = ask_model(q["question"], q["fingerprint"])
        print(f"    time: {result['elapsed_s']:.1f}s  "
              f"prompt_tokens: {result['prompt_eval_count']}  "
              f"gen_tokens: {result['eval_count']}")
        print(f"    response:")
        for line in result["response"].strip().split("\n"):
            print(f"      {line}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared fixtures from the historical pre-Stage-5 response capture.

``pre_stage5_responses.jsonl`` is an archive, not a snapshot to regenerate on
current code. See README.md for provenance and the active render-contract and
default/generic parity tests. Historical reproduction needs the capture script
and dependencies from the historical checkout, with PYTHONHASHSEED=0.

The current tests use the hand-curated corpus with an in-memory store and a
mock compressor. Their archived projection excludes hash-sensitive health
aggregates, so ordinary pytest runs need no special environment variables.
"""

from __future__ import annotations

from typing import Any, Dict, List


from cymatix_context.config import (
    BudgetConfig,
    ClassifierConfig,
    GenomeConfig,
    CymatixConfig,
    RibosomeConfig,
)
from cymatix_context.context_manager import CymatixContextManager
from tests.conftest import make_gene
from tests.conftest import MockCompressorBackend


# ── Synthetic corpus — hand-curated so the golden is reproducible without
#    depending on a live genome snapshot. Mix of finance/migration, auth,
#    performance, and documentation domains so multiple classifier classes
#    have non-empty candidate pools.
_SEED_GENES: List[Dict[str, Any]] = [
    # finance / migration cluster
    {"content": "Calculate the total cost of cloud migration projects",
     "domains": ["finance", "migration"], "entities": ["cost", "calculate", "total"]},
    {"content": "Migration cost spreadsheet with monthly totals",
     "domains": ["finance", "migration"], "entities": ["cost", "total", "migration"]},
    {"content": "Quarterly migration budget summing to 1.2M USD",
     "domains": ["finance", "migration"], "entities": ["budget", "migration"]},
    # auth cluster
    {"content": "Authentication middleware with JWT validation",
     "domains": ["auth", "security"], "entities": ["jwt", "auth"]},
    {"content": "Login flow uses bcrypt for password hashing",
     "domains": ["auth", "security"], "entities": ["login", "bcrypt"]},
    {"content": "JWT token expiry defaults to 15 minutes",
     "domains": ["auth", "security"], "entities": ["jwt", "token"]},
    # perf cluster
    {"content": "Database query latency increased after the index migration",
     "domains": ["performance", "db"], "entities": ["latency", "db"]},
    {"content": "Cache eviction policy uses LRU with 30 minute TTL",
     "domains": ["performance", "cache"], "entities": ["cache", "ttl"]},
    {"content": "Slow endpoints logged with sub-second resolution",
     "domains": ["performance"], "entities": ["latency"]},
    # docs / general
    {"content": "Hello world greeting examples",
     "domains": ["greeting"], "entities": ["hello"]},
    {"content": "General notes about hello there phrasing",
     "domains": ["greeting"], "entities": ["hello"]},
    {"content": "Documentation page describing the deployment topology",
     "domains": ["docs"], "entities": ["deployment"]},
    {"content": "Walkthrough of the indexing pipeline for new contributors",
     "domains": ["docs", "pipeline"], "entities": ["index", "pipeline"]},
    {"content": "Operational runbook for failover and recovery procedures",
     "domains": ["ops"], "entities": ["failover"]},
]


# 100 queries spanning the 5 classifier classes. The numbers below were
# chosen so each classifier branch sees ≥10 queries. Keep this historical
# query list intact; changing it does not authorize regenerating the archive.
_QUERIES: List[str] = [
    # arithmetic (signal_count >= 2 OR strong-pair) — 20 queries
    "Calculate the total cost of migration.",
    "Sum the total of all migration spends.",
    "Calculate 3 + 4 across regions.",
    "Total + sum of monthly migration costs.",
    "Calculate critical path duration plus slack.",
    "Sum the totals of finance / migration this quarter.",
    "Calculate total cost minus refunds for migration.",
    "Total spend - refunds, calculate net for migration.",
    "Sum 1+2 to total of regions.",
    "Calculate total spent on auth this year.",
    "Sum all jwt-related expenses calculated last month.",
    "Calculate total + sum of latency budgets.",
    "Total budget calculation for cache layer.",
    "Sum + calculate critical path of pipeline.",
    "Calculate total bcrypt cost in compute hours.",
    "Total cost calculation including slow endpoints.",
    "Sum the calculated migration costs / budget.",
    "Calculate total + critical path of failover.",
    "Total migration - savings, calculate the net.",
    "Sum + total: how much was the deployment calculation?",
    # factual (wh-word + < 15 words) — 25 queries
    "What is the JWT expiry?",
    "Where is the migration spreadsheet?",
    "Who owns the auth middleware?",
    "When was the index migration done?",
    "Which cache TTL policy is active?",
    "What does the runbook say?",
    "Where do failover procedures live?",
    "Who wrote the deployment docs?",
    "When did latency increase?",
    "Which login flow uses bcrypt?",
    "What is the cache eviction policy?",
    "Where is the auth code?",
    "Who maintains the pipeline?",
    "What is the deployment topology?",
    "Which endpoints are slow?",
    "What runbook covers failover?",
    "Where is the JWT token expiry set?",
    "Who runs the indexing pipeline?",
    "When does cache evict entries?",
    "Which docs describe deployment?",
    "What policy governs latency?",
    "Where is the greeting example?",
    "Who wrote hello world?",
    "When are tokens refreshed?",
    "Which projects need migration?",
    # procedural (how do i / how to / steps / walk me through) — 20 queries
    "How do I rotate JWT tokens?",
    "How to migrate the database safely?",
    "Walk me through the failover procedure.",
    "What are the steps to enable caching?",
    "How do I deploy the new auth flow?",
    "How to debug slow endpoints?",
    "Walk me through the migration calculator.",
    "How do I configure cache TTL?",
    "Steps to add a new gene to the genome.",
    "How to write a new classifier rule?",
    "Walk me through the indexing pipeline.",
    "How do I run the load test?",
    "Steps for onboarding a new contributor.",
    "How to read the runbook?",
    "Walk me through the deployment topology.",
    "How do I update the synonym map?",
    "Steps to enable telemetry exports.",
    "How to invalidate the cache?",
    "Walk me through the bcrypt rotation.",
    "How do I configure logging at sub-second resolution?",
    # multi_hop (and then / because / between / vs / >25 words) — 20 queries
    "Compare the auth flow and then the cache policy.",
    "Migration finished and then the latency dropped because of new indexes.",
    "Difference between cache TTL and JWT expiry vs heartbeat.",
    "Auth vs runbook vs deployment topology comparison.",
    "Discuss latency because of the migration and then the cache change.",
    "Between auth and bcrypt and the JWT expiry, which is hottest?",
    "Compare slow endpoints vs cache eviction events after that migration.",
    "Compare the failover plan and then the runbook walkthrough.",
    "Long query that spans many words and ranges over auth, JWT, cache, latency, deployment, runbook, failover, indexing, migration, totals, costs, and budgets across regions for analysis.",
    "Auth and then the deployment because the migration introduced new indexes.",
    "Compare bcrypt vs JWT vs cache invalidation timing.",
    "Indexing pipeline vs deployment topology and then the runbook.",
    "Migration and then auth and then cache: explain the sequence.",
    "Between finance and migration and ops, which has the most genes?",
    "Compare deployment vs failover vs recovery procedures.",
    "After that the cache evicted because of the JWT expiry.",
    "Latency vs throughput vs cost across migration phases.",
    "Walk me through migration and then auth and then cache compare.",
    "Auth flow and then JWT and then bcrypt because of token rotation.",
    "Compare the synonym map and then the classifier rules vs the decoder modes for the indexing pipeline because of latency budgets and the runbook.",
    # default (no signals fire) — 15 queries
    "auth.",
    "migration",
    "deployment",
    "runbook",
    "failover",
    "JWT",
    "cache",
    "latency",
    "pipeline",
    "indexing",
    "topology",
    "throughput",
    "cost",
    "synonym",
    "telemetry",
]


def _serializable_window(win: Any) -> Dict[str, Any]:
    """Project ContextWindow to a JSON-serialisable dict.

    Compare the response surface, excluding only the per-call correlation ID.
    ``pipeline_request_id`` is independently generated for each invocation;
    document IDs, ranking, content, token counts, health, and all other metadata
    remain intact for the current default/explicit-generic parity test. Include
    the request-scoped retrieval evidence used by downstream citation scores,
    even though ContextWindow excludes those internal fields from model_dump.
    """
    return {
        "ribosome_prompt": win.ribosome_prompt,
        "expressed_context": win.expressed_context,
        "expressed_gene_ids": list(win.expressed_gene_ids),
        "total_estimated_tokens": win.total_estimated_tokens,
        "compression_ratio": win.compression_ratio,
        "context_health": win.context_health.model_dump(),
        "retrieval_scores": win.retrieval_scores,
        "tier_contributions": win.tier_contributions,
        "metadata": {
            key: value for key, value in (win.metadata or {}).items()
            if key != "pipeline_request_id"
        },
    }


def _build_manager(*, historical_render_config: bool = True) -> CymatixContextManager:
    cfg = CymatixConfig(
        ribosome=RibosomeConfig(model="mock", timeout=5),
        budget=BudgetConfig(max_genes_per_turn=4, splice_aggressiveness=0.5),
        genome=GenomeConfig(path=":memory:", cold_start_threshold=5),
        classifier=ClassifierConfig(enabled=True),
    )
    if historical_render_config:
        # The original capture inherited these defaults (d288f9b / PR #47).
        # Later defaults deliberately changed retrieval and delivery. Pin only
        # the historical fixture's mode/settings; do not change shipped code.
        cfg.retrieval.fusion_mode = "additive"
        cfg.budget.decoder_mode = "full"
        cfg.budget.expression_tokens = 6000
        cfg.budget.session_delivery_enabled = False
        cfg.budget.foveated_enabled = False
        cfg.budget.min_delivered_docs = 0
        cfg.budget.splice_target_chars = 1000
    mgr = CymatixContextManager(cfg)
    mgr.ribosome.backend = MockCompressorBackend()
    for i, spec in enumerate(_SEED_GENES):
        mgr.genome.upsert_gene(make_gene(
            spec["content"],
            domains=spec.get("domains") or [],
            entities=spec.get("entities") or [],
            gene_id=f"golden_seed_{i:010d}",
        ))
    return mgr


def main() -> None:
    raise SystemExit(
        "The pre-Stage-5 golden is archived. Refusing to replace historical "
        "responses with current output; see tests/golden/README.md."
    )


if __name__ == "__main__":
    main()

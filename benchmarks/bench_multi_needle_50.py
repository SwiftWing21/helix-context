"""Multi-needle NIAH expanded to N=50.

Supersedes the N=8 probative set in `bench_multi_needle.py` with a
50-needle battery for publishable-quality numbers. Same scoring
(all_delivered, any_delivered, partial_recall per gold_source_groups)
so results are directly comparable to the 2026-04-19 N=8 snapshot.

The genome is the running main instance (~7800 genes across the
Education + helix-context + sibling repos). Needles cover eight
topic clusters so the score isn't dominated by any one area:

    cluster A — helix core (packet, pipeline, compression targets)
    cluster B — helix launcher / headroom integration
    cluster C — helix adapters / DAL / cache / retriever
    cluster D — helix claims layer (extraction, edges, DAG)
    cluster E — BigEd fleet config + skills
    cluster F — BigEd launcher + DB + ops
    cluster G — test / bench infrastructure
    cluster H — cross-cutting (docs, handoffs, shard registry)

Gold source groups use FILE path substrings. The content-recall metric
(`all_delivered`) is strict — at least one gene from every group must
be delivered. `partial_recall` and `any_delivered` soften that for
the long-tail diagnostic.

Usage:
    python benchmarks/bench_multi_needle_50.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the engine from the 8-needle bench
from benchmarks.bench_multi_needle import (  # noqa: E402
    main as _run,
)
import benchmarks.bench_multi_needle as _mn  # noqa: E402


NEEDLES_50 = [
    # ── Cluster A: helix core ────────────────────────────────────────
    {
        "name": "helix_port_and_pipeline_steps",
        "query": "what port does helix listen on and how many steps are in the pipeline",
        "gold_source_groups": [
            ["helix-context/helix.toml"],
            ["helix-context/docs/architecture/PIPELINE_LANES.md",
             "helix-context/README.md"],
        ],
    },
    {
        "name": "compression_target_and_genome_count",
        "query": "what is the target compression ratio and how many genes are in the main genome",
        "gold_source_groups": [
            ["helix-context/docs/DESIGN_TARGET.md",
             "helix-context/README.md"],
            ["helix-context/README.md",
             "helix-context/docs/architecture/PIPELINE_LANES.md"],
        ],
    },
    {
        "name": "coord_confidence_floor_and_path_token_coverage",
        "query": "what is the coordinate confidence floor and what is path token coverage",
        "gold_source_groups": [
            ["helix-context/helix_context/context_packet.py"],
            ["helix-context/helix_context/context_packet.py",
             "helix-context/helix_context/scoring.py"],
        ],
    },
    {
        "name": "packet_task_types_and_verdict_values",
        "query": "what are the allowed task_type values for a helix packet and what verdict values exist",
        "gold_source_groups": [
            ["helix-context/helix_context/context_packet.py",
             "helix-context/helix_context/schemas.py"],
            ["helix-context/helix_context/context_packet.py",
             "helix-context/helix_context/schemas.py"],
        ],
    },
    {
        "name": "volatility_hot_half_life_and_stable_half_life",
        "query": "what is the hot volatility half-life and what is the stable volatility half-life",
        "gold_source_groups": [
            ["helix-context/helix_context/context_packet.py",
             "helix-context/README.md",
             "helix-context/docs/specs/2026-04-17-agent-context-index-build-spec.md"],
            ["helix-context/helix_context/context_packet.py",
             "helix-context/README.md",
             "helix-context/docs/specs/2026-04-17-agent-context-index-build-spec.md"],
        ],
    },
    {
        "name": "volatility_medium_and_authority_primary",
        "query": "what is the medium volatility class half-life and what is the primary authority class weight",
        "gold_source_groups": [
            ["helix-context/helix_context/context_manager.py",
             "helix-context/helix_context/context_packet.py",
             "helix-context/README.md"],
            ["helix-context/helix_context/context_packet.py",
             "helix-context/helix_context/schemas.py",
             "helix-context/README.md"],
        ],
    },
    {
        "name": "ribosome_paused_and_codec_extra",
        "query": "why is the ribosome paused by default and which pip extra enables codec support",
        "gold_source_groups": [
            ["helix-context/helix_context/ribosome.py",
             "helix-context/helix_context/context_manager.py",
             "helix-context/helix_context/server.py"],
            ["helix-context/pyproject.toml",
             "helix-context/README.md"],
        ],
    },

    # ── Cluster B: launcher / headroom ───────────────────────────────
    {
        "name": "headroom_port_and_dashboard_path",
        "query": "what port does headroom serve on and what is the dashboard path",
        "gold_source_groups": [
            ["helix-context/helix.toml",
             "helix-context/helix_context/config.py"],
            ["helix-context/helix.toml",
             "helix-context/helix_context/config.py"],
        ],
    },
    {
        "name": "headroom_mode_default_and_autostart_default",
        "query": "what is the default headroom compression mode and the default autostart setting",
        "gold_source_groups": [
            ["helix-context/helix.toml",
             "helix-context/helix_context/config.py",
             "helix-context/helix_context/launcher/headroom_supervisor.py"],
            ["helix-context/helix.toml",
             "helix-context/helix_context/config.py"],
        ],
    },
    {
        "name": "tray_menu_and_orphan_adoption",
        "query": "how does the tray menu expose headroom controls and how is a running headroom orphan adopted",
        "gold_source_groups": [
            ["helix-context/helix_context/launcher/tray.py"],
            ["helix-context/helix_context/launcher/headroom_supervisor.py"],
        ],
    },
    {
        "name": "helix_supervisor_and_restart_protocol",
        "query": "how does helix supervisor manage the child process and what is the restart protocol",
        "gold_source_groups": [
            ["helix-context/helix_context/launcher/supervisor.py",
             "helix-context/helix_context/launcher/helix_supervisor.py"],
            ["helix-context/helix_context/launcher/supervisor.py",
             "helix-context/helix_context/launcher/helix_supervisor.py",
             "helix-context/helix_context/launcher/headroom_supervisor.py"],
        ],
    },
    {
        "name": "tray_icon_source_and_launcher_entry",
        "query": "where does the tray icon png live and what is the launcher entry point",
        "gold_source_groups": [
            ["helix-context/helix_context/launcher/app.py",
             "helix-context/helix_context/launcher/tray.py"],
            ["helix-context/helix_context/launcher/app.py",
             "helix-context/start-helix-tray.bat"],
        ],
    },

    # ── Cluster C: adapters / DAL / cache / retriever ────────────────
    {
        "name": "dal_default_schemes_and_s3_opt_in",
        "query": "which url schemes does the DAL handle by default and how do you opt into s3",
        "gold_source_groups": [
            ["helix-context/helix_context/adapters/dal.py"],
            ["helix-context/helix_context/adapters/dal.py"],
        ],
    },
    {
        "name": "cache_ttl_stable_and_max_entries",
        "query": "what is the stable TTL on the cache and what is the default max_entries",
        "gold_source_groups": [
            ["helix-context/helix_context/adapters/cache.py"],
            ["helix-context/helix_context/adapters/cache.py"],
        ],
    },
    {
        "name": "retriever_protocol_and_narrowed_wrapper",
        "query": "what signature defines the retriever protocol and what does HelixNarrowedRetriever do",
        "gold_source_groups": [
            ["helix-context/helix_context/adapters/retriever.py"],
            ["helix-context/helix_context/adapters/retriever.py",
             "helix-context/docs/INTEGRATING_WITH_EXISTING_RAG.md"],
        ],
    },
    {
        "name": "llamaindex_wrapper_and_langchain_wrapper",
        "query": "how is LlamaIndex wrapped as a helix retriever and how is LangChain wrapped",
        "gold_source_groups": [
            ["helix-context/helix_context/adapters/retriever.py"],
            ["helix-context/helix_context/adapters/retriever.py"],
        ],
    },
    {
        "name": "cache_stale_risk_bypass_and_refresh_targets",
        "query": "how does the cache handle stale_risk items and what are refresh_targets",
        "gold_source_groups": [
            ["helix-context/helix_context/adapters/cache.py"],
            ["helix-context/helix_context/adapters/cache.py",
             "helix-context/helix_context/context_packet.py"],
        ],
    },

    # ── Cluster D: claims layer ──────────────────────────────────────
    {
        "name": "claim_types_and_extraction_kinds",
        "query": "what are the allowed claim_type values and the extraction_kind values",
        "gold_source_groups": [
            ["helix-context/helix_context/schemas.py",
             "helix-context/helix_context/claims.py"],
            ["helix-context/helix_context/schemas.py",
             "helix-context/helix_context/claims.py"],
        ],
    },
    {
        "name": "claim_edge_types_and_jaccard_thresholds",
        "query": "what claim_edge types exist and what jaccard thresholds trigger contradicts vs duplicates",
        "gold_source_groups": [
            ["helix-context/helix_context/schemas.py",
             "helix-context/helix_context/claims_analyze.py"],
            ["helix-context/helix_context/claims_analyze.py"],
        ],
    },
    {
        "name": "supersedes_chain_and_topological_sort",
        "query": "how does the dag walker walk supersedes chains and how is topological sort implemented",
        "gold_source_groups": [
            ["helix-context/helix_context/claims_graph.py"],
            ["helix-context/helix_context/claims_graph.py"],
        ],
    },
    {
        "name": "resolve_policies_and_authority_rank",
        "query": "what resolve policies exist and what is the authority rank order",
        "gold_source_groups": [
            ["helix-context/helix_context/claims_graph.py"],
            ["helix-context/helix_context/claims_graph.py"],
        ],
    },
    {
        "name": "ingest_hook_and_backfill_script",
        "query": "where does claim extraction happen during ingest and what does the backfill script do",
        "gold_source_groups": [
            ["helix-context/helix_context/genome.py",
             "helix-context/helix_context/claims.py"],
            ["helix-context/scripts/backfill_claims.py"],
        ],
    },
    {
        "name": "shard_categories_and_register_shard",
        "query": "what shard categories exist and how do you register a shard",
        "gold_source_groups": [
            ["helix-context/helix_context/shard_schema.py"],
            ["helix-context/helix_context/shard_schema.py"],
        ],
    },
    {
        "name": "key_values_extractor_and_code_extractor",
        "query": "how does the key_values fallback extractor work and how does the code extractor work",
        "gold_source_groups": [
            ["helix-context/helix_context/claims.py"],
            ["helix-context/helix_context/claims.py"],
        ],
    },

    # ── Cluster E: BigEd fleet config + skills ───────────────────────
    {
        "name": "fleet_skill_count_and_dashboard_endpoints",
        "query": "how many skills does the fleet have and how many dashboard endpoints",
        "gold_source_groups": [
            ["Education/CLAUDE.md"],
            ["Education/CLAUDE.md",
             "Education/fleet/dashboard.py"],
        ],
    },
    {
        "name": "fleet_worker_small_ram_and_large_ram",
        "query": "how many fleet workers are allowed on 8GB and how many on 64GB+",
        "gold_source_groups": [
            ["Education/CLAUDE.md",
             "Education/fleet/fleet.toml"],
            ["Education/CLAUDE.md",
             "Education/fleet/fleet.toml"],
        ],
    },
    {
        "name": "ollama_path_autodiscover_and_qwen_default",
        "query": "how is the ollama path auto-discovered on windows and what is the default qwen model",
        "gold_source_groups": [
            ["Education/CLAUDE.md",
             "Education/fleet/hw_supervisor.py",
             "Education/fleet/providers.py"],
            ["Education/CLAUDE.md",
             "Education/fleet/fleet.toml",
             "Education/fleet/providers.py"],
        ],
    },
    {
        "name": "skill_contract_and_lazy_imports",
        "query": "what is the SKILL_NAME contract in a fleet skill and why must db be imported lazily",
        "gold_source_groups": [
            ["Education/fleet/skills/_contract.py",
             "Education/CLAUDE.md"],
            ["Education/CLAUDE.md"],
        ],
    },
    {
        "name": "db_retry_write_and_wal_busy_timeout",
        "query": "what is db._retry_write for and what is the WAL busy timeout",
        "gold_source_groups": [
            ["Education/fleet/db.py",
             "Education/CLAUDE.md"],
            ["Education/fleet/db.py",
             "Education/CLAUDE.md"],
        ],
    },
    {
        "name": "ram_ceiling_default_and_dev_mode_flag",
        "query": "what is the default ram_ceiling_pct and what flag enables DEV_MODE",
        "gold_source_groups": [
            ["Education/fleet/fleet.toml",
             "Education/CLAUDE.md"],
            ["Education/CLAUDE.md",
             "Education/BigEd/launcher/launcher.py"],
        ],
    },

    # ── Cluster F: BigEd launcher + DB + ops ─────────────────────────
    {
        "name": "fleetdb_file_and_rag_db_file",
        "query": "where does the fleet sqlite database live and where does the rag database live",
        "gold_source_groups": [
            ["Education/fleet/db.py",
             "Education/BigEd/launcher/data_access.py"],
            ["Education/fleet/rag.py",
             "Education/CLAUDE.md"],
        ],
    },
    {
        "name": "lead_client_status_and_dispatch",
        "query": "how do you check fleet status from the cli and how do you dispatch a task",
        "gold_source_groups": [
            ["Education/fleet/lead_client.py",
             "Education/CLAUDE.md"],
            ["Education/fleet/lead_client.py",
             "Education/CLAUDE.md"],
        ],
    },
    {
        "name": "backup_manager_interval_and_location",
        "query": "how often does auto-save backup run and where are backups stored",
        "gold_source_groups": [
            ["Education/fleet/backup_manager.py",
             "Education/CLAUDE.md"],
            ["Education/fleet/backup_manager.py",
             "Education/fleet/fleet.toml"],
        ],
    },
    {
        "name": "hw_supervisor_responsibility_and_thermal_thresholds",
        "query": "what does dr ders hw supervisor manage and where are thermal thresholds configured",
        "gold_source_groups": [
            ["Education/fleet/hw_supervisor.py",
             "Education/CLAUDE.md"],
            ["Education/fleet/hw_supervisor.py",
             "Education/fleet/fleet.toml"],
        ],
    },
    {
        "name": "window_flash_fix_and_subprocess_creationflags",
        "query": "how is the windows console window flash suppressed in subprocess calls",
        "gold_source_groups": [
            ["Education/CLAUDE.md"],
            ["Education/CLAUDE.md",
             "Education/fleet/supervisor.py",
             "Education/fleet/process_manager.py"],
        ],
    },
    {
        "name": "icon_master_source_and_generate_deleted",
        "query": "what is the master source for the icon and why was generate_icon deleted",
        "gold_source_groups": [
            ["Education/CLAUDE.md"],
            ["Education/CLAUDE.md"],
        ],
    },
    {
        "name": "federation_router_and_tenant_admin",
        "query": "what does federation_router do and what does tenant_admin manage",
        "gold_source_groups": [
            ["Education/fleet/federation_router.py",
             "Education/CLAUDE.md"],
            ["Education/fleet/tenant_admin.py",
             "Education/CLAUDE.md"],
        ],
    },
    {
        "name": "sso_providers_and_billing_metering",
        "query": "which sso providers does the fleet support and how is billing metered",
        "gold_source_groups": [
            ["Education/fleet/sso.py",
             "Education/CLAUDE.md"],
            ["Education/fleet/billing.py",
             "Education/CLAUDE.md"],
        ],
    },
    {
        "name": "control_plane_and_experiment_autonomy",
        "query": "what does the saas control plane do and what is the experiment autonomy dial",
        "gold_source_groups": [
            ["Education/fleet/control_plane.py",
             "Education/CLAUDE.md"],
            ["Education/fleet/experiment.py",
             "Education/CLAUDE.md"],
        ],
    },

    # ── Cluster G: test / bench infrastructure ───────────────────────
    {
        "name": "smoke_test_fast_count_and_total_count",
        "query": "how many smoke tests in fast mode and how many in total",
        "gold_source_groups": [
            ["Education/fleet/smoke_test.py",
             "Education/CLAUDE.md"],
            ["Education/fleet/smoke_test.py",
             "Education/CLAUDE.md"],
        ],
    },
    {
        "name": "needle_bench_genome_size_and_gold_regex",
        "query": "what genome size does the needle bench report and what regex extracts gene sources",
        "gold_source_groups": [
            ["helix-context/benchmarks/bench_multi_needle.py",
             "helix-context/benchmarks/bench_needle.py"],
            ["helix-context/benchmarks/bench_multi_needle.py",
             "helix-context/benchmarks/bench_needle.py"],
        ],
    },
    {
        "name": "composition_bench_cells_and_sema_codec",
        "query": "how many cells does the composition bench compare and what is the SEMA codec projection",
        "gold_source_groups": [
            ["helix-context/benchmarks/bench_helix_rag_composition.py"],
            ["helix-context/benchmarks/bench_helix_rag_composition.py",
             "helix-context/benchmarks/bench_external_retriever.py",
             "helix-context/helix_context/codec/sema.py"],
        ],
    },
    {
        "name": "headroom_latency_bench_toggle_and_budget_flip",
        "query": "how does the headroom latency bench toggle headroom and at what budget does compression flip from overhead",
        "gold_source_groups": [
            ["helix-context/benchmarks/bench_headroom_latency.py"],
            ["helix-context/benchmarks/bench_headroom_latency.py",
             "helix-context/SESSION_HANDOFF.md"],
        ],
    },
    {
        "name": "test_claims_graph_and_test_claims_analyze",
        "query": "what does test_claims_graph assert and what does test_claims_analyze cover",
        "gold_source_groups": [
            ["helix-context/tests/test_claims_graph.py"],
            ["helix-context/tests/test_claims_analyze.py"],
        ],
    },

    # ── Cluster H: cross-cutting ─────────────────────────────────────
    {
        "name": "session_handoff_file_and_composition_stack_handoff",
        "query": "where is the session handoff kept and where do composition stack handoffs go",
        "gold_source_groups": [
            ["helix-context/SESSION_HANDOFF.md",
             "Education/CLAUDE.md"],
            ["helix-context/SESSION_HANDOFF.md"],
        ],
    },
    {
        "name": "integrating_with_existing_rag_doc_and_helix_narrowing",
        "query": "what does the integrating-with-existing-rag doc cover and what is pattern 2 narrowing",
        "gold_source_groups": [
            ["helix-context/docs/INTEGRATING_WITH_EXISTING_RAG.md"],
            ["helix-context/docs/INTEGRATING_WITH_EXISTING_RAG.md",
             "helix-context/helix_context/adapters/retriever.py"],
        ],
    },
    {
        "name": "fingerprint_index_columns_and_push_payload_size",
        "query": "what columns does fingerprint_index have and what is the target push payload size",
        "gold_source_groups": [
            ["helix-context/helix_context/shard_schema.py"],
            ["helix-context/helix_context/shard_schema.py",
             "helix-context/docs/PUSH_PULL_CONTEXT.md"],
        ],
    },
    {
        "name": "claim_id_hash_and_entity_key_extraction",
        "query": "how is claim_id computed and how are entity_keys extracted from claim text",
        "gold_source_groups": [
            ["helix-context/helix_context/claims.py"],
            ["helix-context/helix_context/claims.py"],
        ],
    },
    {
        "name": "sibling_session_ribosome_and_launcher_paused",
        "query": "which sibling session worked on ribosome config cleanup and which worked on paused ribosome UI",
        "gold_source_groups": [
            ["helix-context/helix.toml",
             "helix-context/helix_context/ribosome.py"],
            ["helix-context/helix_context/launcher/app.py",
             "helix-context/helix_context/launcher/tray.py"],
        ],
    },
    {
        "name": "roadmap_version_scheme_and_audit_coverage",
        "query": "what version scheme does the bigEd roadmap follow and what must every roadmap item reference",
        "gold_source_groups": [
            ["Education/CLAUDE.md",
             "Education/ROADMAP.md"],
            ["Education/CLAUDE.md",
             "Education/AUDIT_TRACKER.md"],
        ],
    },
]


assert len(NEEDLES_50) == 50, f"Expected 50, got {len(NEEDLES_50)}"


def main():
    # Swap the needle set and run
    _mn.NEEDLES_MULTI = NEEDLES_50
    out = REPO_ROOT / "benchmarks" / "results" / f"multi_needle_50_{time.strftime('%Y-%m-%d')}.json"
    return _run(out_path=out)


if __name__ == "__main__":
    sys.exit(main())

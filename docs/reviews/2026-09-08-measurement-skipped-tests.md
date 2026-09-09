# Measurement audit: exact skipped-test inventory

Date: 2026-09-08. Historical full offline run at PR #446 commit `e806cd1`:
**4,607 passed, 39 skipped, one failed; 25 live tests separately deselected.**
The 39 are 35 test cases and four whole-module collection skips. They are
not 39 passing tests, and the module skips hide additional uncollected cases.

| Reason category | Count |
|---|---:|
| Missing Markdown wiki extra | 17 |
| MCP 1.28.1 installed; source requires MCP 2.x | 8 |
| Missing external RepoBench `rank_bm25` package | 3 |
| CUDA/ROCm/MPS hardware or opt-in gates | 5 |
| POSIX-only test on Windows | 1 |
| Launcher performance opt-in disabled | 1 |
| Hash seed not pinned for golden | 1 |
| Installed console command absent from PATH | 1 |
| Budget-overflow fixture produced no dropped document | 1 |
| Citation fixture produced no attributed citation | 1 |
| **Total** | **39** |

No direct stage-provenance or admission test was skipped. Optional transport,
wiki, hardware and external BM25 coverage does not certify or block the serial
lexical D1 instrument. The budget and citation cases skipped their central
assertions and cannot count as effective coverage of those behaviors.

## Every skipped record

| # | Test or collection unit | Exact recorded reason |
|---:|---|---|
| 1 | `tests/test_mcp_document_aliases.py [module collection]` | Skipped: mcp SDK extra not installed, or too old to expose mcp.server.mcpserver (mcp 2.x home of MCPServer — see pyproject floor) |
| 2 | `tests/test_mcp_server.py [module collection]` | Skipped: mcp SDK extra not installed, or too old to expose mcp.server.mcpserver (mcp 2.x home of MCPServer — see pyproject floor) |
| 3 | `tests/test_mcp_server_entry.py [module collection]` | Skipped: mcp SDK extra not installed, or too old to expose mcp.server.mcpserver (mcp 2.x home of MCPServer) — the spawned subprocess would die on import, failing the test spuriously |
| 4 | `tests/test_mcp_tool_names.py [module collection]` | Skipped: mcp SDK extra not installed, or too old to expose mcp.server.mcpserver (mcp 2.x home of MCPServer — see pyproject floor) |
| 5 | `tests/test_build_wiki_site.py::TestBuildRendersPagesAndRewritesLinks::test_build_renders_pages_and_rewrites_links` | needs the 'markdown' package (installed via the dev extra) |
| 6 | `tests/test_build_wiki_site.py::TestBuildRendersPagesAndRewritesLinks::test_home_link_rewrites_to_wiki_root` | needs the 'markdown' package (installed via the dev extra) |
| 7 | `tests/test_build_wiki_site.py::TestTitleAndCanonical::test_home_page_title_and_canonical` | needs the 'markdown' package (installed via the dev extra) |
| 8 | `tests/test_build_wiki_site.py::TestTitleAndCanonical::test_subpage_title_and_canonical` | needs the 'markdown' package (installed via the dev extra) |
| 9 | `tests/test_build_wiki_site.py::TestChromePreserved::test_header_brand_footer_and_wiki_nav_link_present` | needs the 'markdown' package (installed via the dev extra) |
| 10 | `tests/test_build_wiki_site.py::TestChromePreserved::test_external_anchor_and_asset_hrefs_untouched` | needs the 'markdown' package (installed via the dev extra) |
| 11 | `tests/test_build_wiki_site.py::TestChromePreserved::test_href_rewrite_scoped_to_anchor_tags_not_prose_or_code` | needs the 'markdown' package (installed via the dev extra) |
| 12 | `tests/test_build_wiki_site.py::TestImageSrcRewrite::test_asset_img_src_rewritten_code_literal_untouched` | needs the 'markdown' package (installed via the dev extra) |
| 13 | `tests/test_build_wiki_site.py::TestImageSrcRewrite::test_absolute_and_non_assets_img_srcs_untouched` | needs the 'markdown' package (installed via the dev extra) |
| 14 | `tests/test_build_wiki_site.py::TestAssetsCopied::test_copies_wiki_assets_directory` | needs the 'markdown' package (installed via the dev extra) |
| 15 | `tests/test_build_wiki_site.py::TestAssetsCopied::test_no_assets_dir_does_not_error` | needs the 'markdown' package (installed via the dev extra) |
| 16 | `tests/test_build_wiki_site.py::TestMarkdownExtensions::test_tables_and_fenced_code_render` | needs the 'markdown' package (installed via the dev extra) |
| 17 | `tests/test_build_wiki_site.py::TestMainCli::test_main_parses_args_and_builds` | needs the 'markdown' package (installed via the dev extra) |
| 18 | `tests/test_build_wiki_site.py::TestSidebarNav::test_sidebar_rendered_with_rewritten_links_and_current_highlight` | needs the 'markdown' package (installed via the dev extra) |
| 19 | `tests/test_build_wiki_site.py::TestSidebarNav::test_sidebar_mobile_details_block_present` | needs the 'markdown' package (installed via the dev extra) |
| 20 | `tests/test_build_wiki_site.py::TestSidebarNav::test_no_sidebar_file_builds_without_nav` | needs the 'markdown' package (installed via the dev extra) |
| 21 | `tests/test_build_wiki_site.py::TestSidebarNav::test_layout_wrapper_and_wide_shell_present` | needs the 'markdown' package (installed via the dev extra) |
| 22 | `tests/test_caller_model_class.py::test_mcp_to_context_round_trip_echoes_class[generic]` | could not import 'mcp.server.mcpserver': No module named 'mcp.server.mcpserver' |
| 23 | `tests/test_caller_model_class.py::test_mcp_to_context_round_trip_echoes_class[small_moe]` | could not import 'mcp.server.mcpserver': No module named 'mcp.server.mcpserver' |
| 24 | `tests/test_caller_model_class.py::test_mcp_to_context_round_trip_echoes_class[frontier]` | could not import 'mcp.server.mcpserver': No module named 'mcp.server.mcpserver' |
| 25 | `tests/test_caller_model_class.py::test_generic_branch_byte_identical_to_pre_stage5_output` | byte-identical golden requires PYTHONHASHSEED=0 (top_dominance depends on dict iteration order). Re-run with: PYTHONHASHSEED=0 python -m pytest tests/test_caller_model_class.py |
| 26 | `tests/test_cli_dispatcher.py::test_installed_console_script_prog_matches_invoked_name[cymatix-cymatix]` | cymatix console script not found on PATH |
| 27 | `tests/test_hardware_mps_smoke.py::test_cross_encoder_two_pair_forward_pass_on_mps` | Requires darwin + MPS-capable hardware AND not running on GHA (GHA macos-14 MPS shared pool OOMs even on tiny models — see spec §3 verification posture; tracked for re-enable when a real Mac dev rig is available) |
| 28 | `tests/test_hardware_real_device.py::test_auto_picker_lands_on_device[cuda]` | Set CYMATIX_TEST_CUDA=1 on a CUDA-capable host to enable |
| 29 | `tests/test_hardware_real_device.py::test_auto_picker_lands_on_device[rocm]` | Set CYMATIX_TEST_ROCM=1 on a ROCm-capable host to enable |
| 30 | `tests/test_hardware_real_device.py::test_recommended_batch_size_is_positive[cuda]` | Set CYMATIX_TEST_CUDA=1 on a CUDA-capable host to enable |
| 31 | `tests/test_hardware_real_device.py::test_recommended_batch_size_is_positive[rocm]` | Set CYMATIX_TEST_ROCM=1 on a ROCm-capable host to enable |
| 32 | `tests/test_launcher_graph_summary.py::test_summary_stays_inside_its_budget_on_a_large_bed` | Set CYMATIX_TEST_PERF=1 to run the graph summary cost measurement |
| 33 | `tests/test_no_helix_leftovers.py::test_mcp_server_identifies_as_cymatix` | mcp SDK extra not installed, or too old to expose mcp.server.mcpserver (mcp 2.x home of MCPServer) |
| 34 | `tests/test_observability_supervisor.py::test_posix_uses_start_new_session` | POSIX-only |
| 35 | `tests/test_repobench_r_harness.py::TestRankBM25::test_best_match_first` | could not import 'rank_bm25': No module named 'rank_bm25' |
| 36 | `tests/test_repobench_r_harness.py::TestRankBM25::test_returns_all_candidates` | could not import 'rank_bm25': No module named 'rank_bm25' |
| 37 | `tests/test_repobench_r_harness.py::TestRankBM25::test_empty_corpus_safe` | could not import 'rank_bm25': No module named 'rank_bm25' |
| 38 | `tests/test_score_aware_budget_trim.py::TestScoreAwareBudgetTrim::test_lowest_score_dropped_first_under_overflow` | Budget did not fire a drop — token estimator changed or content shrunk. Lower budget_expression to force a drop. |
| 39 | `tests/test_server.py::TestContextCitationEnrichment::test_citation_includes_attribution_when_present` | query did not retrieve the attributed gene — retrieval is not deterministic across test runs |

## Follow-up diagnostics

- The installed console-command test passed with the project venv Scripts
  directory placed on PATH inside the test process.
- With `PYTHONHASHSEED=0`,
  `test_caller_model_class.py::test_generic_branch_byte_identical_to_pre_stage5_output`
  failed identically on `e806cd1` and clean pre-PR beta `0891f5c`. Query 0
  expected two seed documents but returned abstention (`score_below_floor`).
  This exposes a pre-existing golden mismatch, not a regression from #446.
- The budget-overflow and citation-attribution cases initially self-skipped
  again. The audit then repaired their fixtures and replaced both skip exits
  with strict assertions. Their two complete test files now pass 89 cases with
  no skips; this does not alter the original inventory.
- The original Windows Bash-path failure also reproduces on clean beta;
  all 21 installer tests pass when Git Bash is selected.

These bounded reruns do not alter the historical 39-skip inventory. Optional
dependencies were not installed, and no full-suite pass is claimed.

The original beta run does not validate Joe's v0.9.0 code or his remote bed.
D1 uses a separately inspected and tested instrumentation-only backport.
See the [audit findings and validation](2026-09-08-measurement-test-audit.md).

Provenance: the coordinator independently matched every identifier and skip
message to `.tmp-root-full-offline-results.xml`; detailed local evidence and
predicate analysis are in `.tmp-audit-20260908/skipped-tests.md` and `.json`
in the measurement worktree. Source predicates were inspected by a separate
audit agent. The 25 deselections come from the terminal result, not JUnit.

# Research-Catalog Verdict-Tag Filter (Rev 2)

task: Add verdict-tag filter + STRATEGY_BLUEPRINT §5 NO-GO indexing to research-catalog MCP
mode: TRIVIAL

scope_lock:
  - scripts/tools/research_catalog_mcp_server.py
  - tests/test_tools/test_research_catalog_mcp_server.py
  - docs/specs/research_catalog_verdict_filter.md

## Blast Radius

- scripts/tools/research_catalog_mcp_server.py — additive: extends `_artifact_index` to parse STRATEGY_BLUEPRINT.md §5 NO-GO table rows; adds `verdict` to `Artifact.metadata` via filename-stem and front-matter detection; adds `verdict_tags` parameter to `_search_research_catalog` (default None = all, regression-preserving).
- tests/test_tools/test_research_catalog_mcp_server.py — additive: 5 new unit tests (patch.object pattern, no subprocess).
- docs/specs/research_catalog_verdict_filter.md — new file, ~30 lines, doc-only.
- Reads: docs/STRATEGY_BLUEPRINT.md (read-only), docs/audit/results/, docs/audit/hypotheses/, docs/institutional/literature/.
- Writes: none — read-only MCP surface.
- No production code in pipeline/ or trading_app/ touched. No DB writes. No new MCP server.

## Justification for TRIVIAL

Per Rev 2 plan: ~130 LOC net new, no `pipeline/`/`trading_app/` edits, no schema change, verification (5 unit tests) lands in same change. Stage-gate-protocol.md "Trivial-Change Tier" criteria all met.

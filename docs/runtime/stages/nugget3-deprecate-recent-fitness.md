---
task: Deprecate strategy-lab.get_recent_fitness MCP endpoint (overlaps gold-db canonical)
mode: IMPLEMENTATION
slug: nugget3-deprecate-recent-fitness
scope_lock:
  - scripts/tools/strategy_lab_mcp_server.py
  - tests/test_tools/test_strategy_lab_mcp_server.py
  - pipeline/check_drift.py
  - .claude/rules/mcp-usage.md
---

## Blast Radius

- scripts/tools/strategy_lab_mcp_server.py: remove `_get_recent_fitness()` (lines 286-296) + its ToolSpec (lines 398-411). Keep `_compute_fitness_payload()` (line 137) — consumed by `_get_strategy_readiness` (line 225) and `_list_promotable_candidates` (line 321).
- tests/test_tools/test_strategy_lab_mcp_server.py: delete two functions referencing `srv._get_recent_fitness` (lines 260-268). `_compute_fitness_payload` patch usages elsewhere remain intact.
- pipeline/check_drift.py: add `check_strategy_lab_no_fitness_endpoint` asserting "get_recent_fitness" does NOT appear inside any `ToolSpec(` block in `strategy_lab_mcp_server.py`. Registry append at end of CHECKS list.
- .claude/rules/mcp-usage.md: add doctrine sentence pointing to gold-db.get_strategy_fitness as canonical.
- Reads: gold.db read-only via `_compute_fitness_payload` (unchanged path). Writes: none.
- Doctrine in `.claude/rules/mcp-usage.md` already designates gold-db as primary; this removes the overlapping surface and prevents MCP staleness (feedback_strategy_lab_mcp_vs_lane_allocation_json_divergence.md, 2026-05-13).

---
task: Build read-only HTML "OKAY GO" portal aggregating 10 panels (deployed lanes, promotable, promote queue, OOS rejections, cherry-pick top-5, drafts, journal pending, next-24h sessions, holdout, data-freshness/drift) from canonical sources
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/go_portal.py
  - tests/test_tools/test_go_portal.py
---

## Scope Lock

- scripts/tools/go_portal.py
- tests/test_tools/test_go_portal.py

## Blast Radius

- scripts/tools/go_portal.py — new file, zero callers. Imports read-only from `strategy_lab_mcp_server` private helpers (`_load_allocation_doc`, `_allocation_index`, `_list_validated_rows`), `fast_lane_promote_queue.scan`, `pipeline.dst.SESSION_CATALOG` + `orb_utc_window`, `pipeline.paths.GOLD_DB_PATH`, `trading_app.strategy_fitness.compute_fitness`, `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`, `trading_app.prop_profiles.effective_daily_lanes`.
- tests/test_tools/test_go_portal.py — new unit tests covering panel render, empty-state, error-isolation.
- Reads: gold.db (read-only via duckdb), docs/runtime/lane_allocation.json, docs/runtime/promote_queue.yaml, docs/runtime/cherry_pick_ranking_*.csv, docs/runtime/cherry_pick_journal.yaml, docs/audit/hypotheses/drafts/*.yaml. Writes: docs/runtime/go_portal_<isoYYYY-MM-DD>.html only.
- No schema changes, no DB writes, no live-trading paths touched. No callers (new tool).

## Notes

`validated_setups` is a DERIVED layer banned for DISCOVERY (research-truth-protocol.md). Portal reads it for DISPLAY only in panel 2 — docstring at top of `go_portal.py` cites the exception explicitly.

---
task: code-review-batch4-fixes
mode: IMPLEMENTATION
scope_lock:
  - trading_app/lane_allocator.py
  - trading_app/pre_session_check.py
  - trading_app/prop_profiles.py
  - trading_app/account_hwm_tracker.py
  - pipeline/check_drift.py
blast_radius: |
  Batch 4 code-review fixes. Touches 4 production files + 1 drift-check file.
  Capital-class: lane_allocator + prop_profiles + pre_session_check are on the
  live order-routing + DD-monitoring paths. account_hwm_tracker constant
  consolidation is config-only (same value 30, single canonical home).
  pipeline/check_drift.py adds 2 new checks (NQ-mini fail-open guard,
  extended magic-number rationale scope). Reads: ACCOUNT_PROFILES at module
  load (prop_profiles), drift check walks AST. Writes: none.
---

# Code-Review Batch 4 Fixes

## Findings closed

1. **HIGH (NQ-mini fail-open trap)** — `resolve_execution_symbol()` has zero
   production callsites 19 days post-merge. If a future ACCOUNT_PROFILES row
   populates `execution_symbol_map=`, the broker silently receives the
   strategy_symbol unchanged (broker fills wrong instrument).
   **Fix:** new drift check `check_nq_mini_substitution_wired_or_unused`
   fails if any ACCOUNT_PROFILES row populates `execution_symbol_map=` while
   the substitution helper has no production callsite. Test fixtures are
   exempt (scan only `trading_app/`).

2. **MEDIUM (allocator class-bug)** — `_normalize_writable_path` duplicated
   in lane_allocator.py, pre_session_check.py, prop_profiles.py. Rule
   violation per institutional-rigor § 4.
   **Fix:** canonical home in `lane_allocator.py`; other two import.

3. **MEDIUM (drift-check coverage)** — `check_magic_number_rationale` walks
   only `trading_app/live/`; misses capital-class constants in
   `trading_app/account_hwm_tracker.py` + `trading_app/pre_session_check.py`.
   **Fix:** extend scope to those two files. Existing rationale-tagged
   constants there (`_STATE_STALENESS_FAIL_DAYS`, `_INACTIVITY_BLOCK_DAYS`,
   etc.) already pass; this is forward-looking coverage.

4. **LOW (constant duplication)** — `_INACTIVITY_BLOCK_DAYS=30.0` in
   pre_session_check.py and `_STATE_STALENESS_FAIL_DAYS=30` in
   account_hwm_tracker.py are the same boundary on the same files.
   **Fix:** pre_session_check imports the canonical from account_hwm_tracker.

## Out of scope (cut new branch after this)

- **NQ-mini Stage 2** (wire session_orchestrator + webhook_server +
  populated profile + integration test). User confirmed "start over the
  right way" — separate stage doc, separate branch, separate review pass.
  Scheduled into action queue post-merge.

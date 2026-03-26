---
mode: IMPLEMENTATION
task: Commit triage -- schema fix + stash cleanup + test recovery
scope_lock:
  - pipeline/init_db.py
  - tests/test_pipeline/test_build_daily_features.py
  - docs/runtime/STAGE_STATE.md
acceptance:
  - overnight_range_pct column added to schema
  - Market profile tests pass
  - Stashes cleaned up
  - Remote branch cleaned up
---

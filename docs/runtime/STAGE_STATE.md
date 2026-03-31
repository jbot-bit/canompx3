---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Remove O15/O30 from active pipeline — add ACTIVE_ORB_MINUTES canonical constant
updated: 2026-03-31T12:20:00Z
scope_lock:
  - pipeline/build_daily_features.py
  - scripts/tools/pipeline_status.py
  - scripts/tools/refresh_data.py
  - scripts/tools/assert_rebuild.py
  - scripts/tools/design_max_profiles.py
  - scripts/tools/max_extraction_model.py
  - scripts/tools/optimal_lanes.py
  - scripts/tools/compare_account_sizes.py
blast_radius:
  - build_daily_features.py: new constant only, no behavior change to existing code
  - pipeline_status.py: staleness + rebuild steps, no external callers
  - refresh_data.py: daily_features build loop, no external callers
  - assert_rebuild.py: post-rebuild assertions, no external callers
  - check_drift.py: NOT touched — row integrity expects 3 rows/date (DB truth)
acceptance:
  - python scripts/tools/pipeline_status.py --status shows NO O15/O30 staleness
  - python pipeline/check_drift.py passes
  - ACTIVE_ORB_MINUTES importable from pipeline.build_daily_features
  - grep confirms no hardcoded [5, 15, 30] in changed files
---

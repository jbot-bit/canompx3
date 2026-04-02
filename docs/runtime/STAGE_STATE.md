---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Lane Allocator Stage 2 — eliminate hardcoded lanes, add staleness gate
updated: 2026-04-02T12:00:00Z
scope_lock:
  - trading_app/paper_trade_logger.py
  - trading_app/pre_session_check.py
  - trading_app/prop_profiles.py
blast_radius:
  - paper_trade_logger.py: Remove hardcoded LANES, derive from prop_profiles, replace filter_sql with matches_row
  - pre_session_check.py: Add allocation staleness check (check_allocation_staleness)
  - prop_profiles.py: Make _parse_strategy_id public (used by paper_trade_logger)
acceptance:
  - paper_trade_logger reads lanes from prop_profiles (zero hardcoded strategy_ids)
  - filter applied via matches_row (zero filter_sql)
  - pre_session_check warns at >35d, blocks at >60d stale allocation
  - Existing paper_trade_logger --dry-run produces same trades as before
  - Drift clean
---

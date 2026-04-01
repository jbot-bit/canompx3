---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Fix E2 fakeout exclusion bias — scan from orb_end not break_ts
updated: 2026-04-01T18:00:00Z
scope_lock:
  - trading_app/outcome_builder.py
  - tests/test_trading_app/test_outcome_builder.py
  - tests/test_trading_app/test_entry_rules.py
blast_radius:
  - outcome_builder.py: detection_window_start changes from break_ts to orb_end
  - entry_rules.py: NO CHANGES (function is window-agnostic)
  - E1 outcomes: UNAFFECTED (separate code path)
  - All downstream tables: require full rebuild AFTER code change
acceptance:
  - detect_break_touch called with orb_end_utc (not break_ts) for E2
  - Known fakeout day: entry_ts moves earlier than break_ts
  - No-fakeout day: entry_ts unchanged
  - E1 outcomes byte-identical before/after
  - All tests pass
  - drift check clean
---

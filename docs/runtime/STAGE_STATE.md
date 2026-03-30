---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Fix 3 code review findings + trade sheet migration to prop_profiles
pass: 2
scope_lock:
  - trading_app/prop_profiles.py
  - scripts/tools/generate_trade_sheet.py
  - scripts/tools/gap5_wfe.py
  - scripts/tools/migrate_fairness_audit.py
  - tests/test_trading_app/test_prop_profiles.py
blast_radius:
  - prop_profiles.py — DD budget calc fix, PV assertion
  - generate_trade_sheet.py — replace LIVE_PORTFOLIO with prop_profiles
  - gap5_wfe.py, migrate_fairness_audit.py — stale COMEX ID
acceptance:
  - validate_dd_budget uses cap for worst-case
  - _PV validated against COST_SPECS
  - COMEX IDs updated
  - Trade sheet shows 6 deployed lanes
  - All tests pass, drift clean
updated: 2026-03-30T12:00:00Z
---

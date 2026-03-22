---
task: "Implement noise_risk flag using OOS ExpR"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Add noise_risk + oos_exp_r to validated_setups. Flag, not gate."
updated: 2026-03-22T17:00+10:00
terminal: main
scope_lock:
  - trading_app/strategy_validator.py
  - trading_app/db_manager.py
  - trading_app/live_config.py
acceptance:
  - "noise_risk populated for all validated_setups rows (non-NULL)"
  - "oos_exp_r stored from WF agg_oos_exp_r"
  - "MES: ~78 flagged, ~3 clean (matches design)"
  - "MNQ + MGC noise_risk also populated"
  - "No change to validation pass/fail"
  - "live_config reads noise_risk column"
proven:
  - "NOISE_FLOOR_BY_INSTRUMENT exists in config.py"
  - "noise_risk column exists in schema (all NULL)"
  - "MES E2 floor = 0.29 confirmed from 94-seed null test"
  - "WF agg_oos_exp_r available in validator Phase C"
unproven: []
blockers: []
---

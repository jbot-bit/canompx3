---
task: "Guardian audit phase 2: fix remaining 11 silent failure paths + type safety + stale docs + noise floor scripts"
mode: IMPLEMENTATION
stage: 1
stage_of: 4
stage_purpose: "Class A: Fix 11 silent failure code paths (logging, error surfacing, null guards)"
updated: 2026-03-25T00:30+10:00
terminal: main
scope_lock:
  - pipeline/build_daily_features.py
  - pipeline/ingest_dbn_daily.py
  - pipeline/audit_log.py
  - pipeline/check_drift.py
  - scripts/tools/explore.py
  - scripts/tools/generate_trade_sheet.py
  - scripts/tools/ml_exhaustive_sweep.py
  - scripts/tools/m25_audit.py
  - trading_app/live_config.py
  - trading_app/strategy_fitness.py
  - trading_app/strategy_validator.py
acceptance:
  - "GARCH: log exception type+instrument on failure (not just return None)"
  - "explore.py: heartbeat parse failure → issues.append, not pass"
  - "explore.py:185: DB error → distinct message from empty table"
  - "generate_trade_sheet.py: cost spec failure → log.warning with instrument"
  - "ingest_dbn_daily.py: logger.warning → logger.error for file failures"
  - "audit_log.py: log failed log_id (already prints, acceptable)"
  - "check_drift.py: DB open failure → print warning (not silent pass)"
  - "live_config.py: cost spec failure → log.warning (not just 'n/a')"
  - "strategy_fitness.py: null guard on expectancy_r (not or 0.0)"
  - "ml_exhaustive_sweep.py: check subprocess returncode"
  - "strategy_validator.py: WF error → log.warning the error (already captured in result)"
  - "No behavior changes — only logging and error surfacing"
  - "Drift check still 0 violations"
blockers: []
---

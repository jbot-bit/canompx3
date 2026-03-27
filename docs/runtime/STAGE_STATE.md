---
mode: IMPLEMENTATION
task: Clean all lint issues from vanilla Claude commits (21 mechanical fixes)
scope_lock:
  - scripts/databento_daily.py
  - scripts/databento_backfill.py
  - scripts/tools/migrate_fairness_audit.py
  - trading_app/portfolio.py
  - trading_app/pre_session_check.py
  - trading_app/strategy_validator.py
  - tests/
  - docs/runtime/STAGE_STATE.md
acceptance:
  - ruff check passes on all touched files
  - No unused imports, dead variables, or f-string issues
  - check_drift.py passes
  - All scripts still import cleanly
---

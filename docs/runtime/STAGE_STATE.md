---
mode: IMPLEMENTATION
task: Fix all 39 test failures for clean ship + DD budget DRY fix
scope_lock:
  - tests/
  - trading_app/ml/features.py
  - trading_app/pre_session_check.py
  - trading_app/live/projectx/order_router.py
  - trading_app/portfolio.py
  - scripts/tools/project_pulse.py
  - scripts/infra/windows_agent_launch.py
  - docs/runtime/STAGE_STATE.md
acceptance:
  - 0 test failures, 0 errors
  - check_drift.py passes
  - All production fixes are minimal (no refactoring)
  - DD constants imported from canonical source (no DRY violation)
---

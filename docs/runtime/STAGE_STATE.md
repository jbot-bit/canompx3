---
mode: IMPLEMENTATION
task: "Lane management: halt file, paper PnL, lane toggle"
stage_purpose: "Add 3 operational features to daily sheet: manual halt, paper PnL display, lane pause/resume"
scope_lock:
  - trading_app/prop_portfolio.py
  - trading_app/pre_session_check.py
  - trading_app/lane_ctl.py
  - tests/test_trading_app/test_prop_portfolio.py
  - tests/test_trading_app/test_pre_session_check.py
  - tests/test_trading_app/test_lane_ctl.py
  - docs/runtime/STAGE_STATE.md
acceptance:
  - "Manual halt: --halt writes file, pre_session_check blocks, --resume clears"
  - "Paper PnL: daily sheet shows 30d W/L and cumR per lane from paper_trades"
  - "Lane toggle: lane_ctl pause/resume/list, resolve_daily_lanes reads overrides"
  - "PAUSED lanes excluded from DD budget (existing filter covers this)"
  - "All tests pass, 75/75 drift, ruff clean"
updated: 2026-03-27T23:30:00+10:00
terminal: main
---

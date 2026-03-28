---
mode: IMPLEMENTATION
task: "Solidify project — venv resilience + complete build_live_portfolio deprecation"
stage_purpose: "1) Fix venv crash vulnerability (dev deps lost on interrupted sync), 2) Complete deprecation of build_live_portfolio from all runtime callers"
blast_radius: "session_orchestrator.py (top-level import breaks if function removed), run_live_session.py (preflight fallback), generate_trade_sheet.py (imports _load_best_regime_variant), daily_paper_run.py (imports PAPER_TRADE_CANDIDATES), ui_v2/server.py (len(LIVE_PORTFOLIO) checks), check_drift.py checks #43+#54 (import LIVE_PORTFOLIO). LIVE_MIN_EXPECTANCY_DOLLARS_MULT must be rehomed not deleted. Full report: docs/runtime/blast-radius-deprecation.md"
scope_lock:
  - trading_app/prop_portfolio.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live_config.py
  - trading_app/live/performance_monitor.py
  - scripts/run_live_session.py
  - scripts/tmp_prop_firm_proper_pass.py
  - ui/session_helpers.py
  - ui_v2/state_machine.py
  - tests/test_trading_app/test_live_config.py
  - tests/test_trading_app/test_prop_portfolio.py
  - pipeline/health_check.py
  - docs/runtime/STAGE_STATE.md
acceptance:
  - "Zero DeprecationWarning from build_live_portfolio in test suite"
  - "All runtime callers use validated_setups or prop_profiles directly"
  - "health_check.py detects missing dev deps with actionable fix command"
  - "All tests pass, 75/75 drift"
updated: 2026-03-28T10:00:00+10:00
terminal: main
---

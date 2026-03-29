---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Fix V2 orb_minutes hardcode bug — paper_trader/execution_engine load wrong daily_features for O15/O30 strategies
pass: 2
scope_lock:
  - trading_app/execution_engine.py
  - trading_app/paper_trader.py
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_execution_engine.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_portfolio.py
  - tests/test_trading_app/test_engine_risk_integration.py
  - tests/test_trading_app/test_ml/test_predict_live.py
blast_radius:
  - execution_engine.py — _daily_features_row -> _daily_features_rows dict[int,dict]. 5 internal references. Core engine.
  - paper_trader.py — _get_daily_features_row adds orb_minutes param. Load per-orb_minutes in replay loop.
  - session_orchestrator.py — _build_daily_features_row adds orb_minutes param. Same pattern.
  - 5 test files — mock shape change (dict -> dict[int,dict])
acceptance:
  - stress_test_chain_integrity.py T2 shows 0 mismatches (was 206/488)
  - stress_test_chain_integrity.py T4 shows PASS (was FAIL)
  - python pipeline/check_drift.py passes
  - python -m pytest tests/test_trading_app/ -x -q passes
updated: 2026-03-29T16:00:00Z
---

---
task: "Master guardian audit: fix fail-open defects, stale narrative, canonical violations, test staleness"
mode: IMPLEMENTATION
stage: 1
stage_of: 4
stage_purpose: "Phase 1: Fix 6 fail-open defects in live trading path"
updated: 2026-03-24T21:00+10:00
terminal: main
scope_lock:
  - trading_app/ml/predict_live.py
  - trading_app/live/trade_journal.py
  - trading_app/live/session_orchestrator.py
  - trading_app/execution_engine.py
  - trading_app/live/multi_runner.py
  - trading_app/market_state.py
  - trading_app/spa_test.py
  - tests/test_trading_app/test_trade_journal.py
  - tests/test_trading_app/test_multi_runner.py
  # Phase 2 (stage 2): docs/memory/agents narrative cleanup
  # Phase 3 (stage 3): scripts/tools canonical imports, check_drift, asset_configs
  # Phase 4 (stage 4): tests, ui, env
acceptance:
  - "predict_live: fail-open results NOT cached for transient errors (lines 357, 436)"
  - "trade_journal: is_healthy property added, orchestrator checks at startup"
  - "execution_engine: log.warning on vol_scalar=1.0 fallback"
  - "multi_runner: post_session failures tracked and logged at CRITICAL"
  - "market_state: regime context failure logged at WARNING not DEBUG"
  - "spa_test: spa_error key added to return dict on exception"
  - "All existing tests pass"
  - "Drift check unchanged or improved"
blockers: []
---

---
task: "Institutional code review — fix all findings from 2-day audit"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Fix all critical/high/medium findings from comprehensive code review of last 2 days of commits. 8 fixes across production code, tests, drift checks, docs."
updated: 2026-03-23T20:30+10:00
terminal: review
scope_lock:
  - trading_app/strategy_validator.py
  - trading_app/live_config.py
  - trading_app/db_manager.py
  - trading_app/spa_test.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_pipeline_status.py
  - tests/test_trading_app/test_live_config.py
  - .claude/hooks/post-edit-pipeline.py
  - .claude/rules/quant-agent-identity.md
  - docs/prompts/SYSTEM_AUDIT.md
  - docs/STRATEGY_DISCOVERY_AUDIT.md
  - HANDOFF.md
acceptance:
  - "C1: Portfolio ATR gate crash fixed (verified)"
  - "C3: BH total_tests param wired through global_k"
  - "C4: pipeline_status per-aperture tests pass"
  - "C5: seasonal gate tests pass (ATR mock + noise_risk seed)"
  - "H1: holdout check fail-closed on con=None"
  - "H3: seasonal gate applied to REGIME + HOT tiers"
  - "H4: verify_trading_app_schema covers all migration columns"
  - "M1: holdout check uses json_extract_string"
  - "M3: M2K in dead instrument docs"
  - "P1: post-edit hook PYTHONPATH fixed"
  - "All modified tests pass"
proven:
  - "C1 verified: Portfolio constructor no longer crashes"
  - "C4 verified: 31 passed pipeline_status tests"
  - "C5 verified: 36 passed live_config tests"
unproven: []
blockers: []
---

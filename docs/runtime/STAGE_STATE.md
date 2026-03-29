---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Fix 3 CRITICALs from adversarial audit (HWM freeze, EOD-only ratchet, DD budget validation)
pass: 2
scope_lock:
  - trading_app/account_hwm_tracker.py
  - trading_app/prop_profiles.py
  - tests/test_trading_app/test_account_hwm_tracker.py
  - tests/test_trading_app/test_prop_profiles.py
blast_radius:
  - account_hwm_tracker.py — used by session_orchestrator, pre_session_check, weekly_review
  - prop_profiles.py — imported by pre_session_check, run_live_session, paper_trade_logger
  - Both are runtime-critical for live trading. Must not break existing API contracts.
acceptance:
  - C-1: HWM freezes when equity reaches safety_net_balance (configurable per firm)
  - C-2: For eod_trailing firms, HWM only updates on record_session_end(), not every poll
  - C-3: prop_profiles.py validates sum(P90 stops) < max_dd at import time
  - All existing tests pass
  - Drift checks pass
  - New tests cover freeze logic, EOD-only ratchet, and budget validation
updated: 2026-03-29T18:00:00Z
---

---
mode: IMPLEMENTATION
task: Extract _apply_broker_reality_check() from session_orchestrator HWM init; convert wiring tests to true integration
created: 2026-04-15
updated: 2026-04-15
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
blast_radius:
  - session_orchestrator.py — extract ~20 lines inline (L529-548) into module-level function near existing _is_trading_combine_account helper (L42). Behavior-preserving.
  - test_session_orchestrator.py — 4 wiring tests (test_broker_reality_wiring_tc_disables_f1, test_broker_reality_wiring_xfa_sets_eod_balance, test_broker_reality_wiring_none_metadata_trusts_profile, test_broker_reality_wiring_f1_inactive_short_circuits) currently duplicate the inline logic in test bodies; rewrite to call the extracted helper directly.
acceptance:
  - All 134 tests in test_session_orchestrator.py pass
  - All 239 F-1 scope tests pass (test_risk_manager + test_projectx_positions + orchestrator)
  - python -m pipeline.check_drift shows 102/0/6 (same as baseline)
  - No behavior change — 3 branches preserved (TC → disable_f1, XFA → set EOD, None meta → trust profile + set EOD)
  - Call site at session_orchestrator.py:529 reduced from 20 lines to guarded single call
source: HANDOFF.md 2026-04-15 late — Known gaps #4 — explicitly deferred "because session_orchestrator.py is not in current active stage's scope_lock (dashboard-polish)". Stage just closed.
---

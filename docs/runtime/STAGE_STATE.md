---
task: "RR policy audit: JK-equal liveability tiebreaker in live_config"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Add JK-equal fallback to _load_best_regime_variant: when locked RR fails LIVE_MIN, try JK-equal alternatives that pass. Live-resolution only, no research table changes."
updated: 2026-03-24T19:30+10:00
terminal: main
scope_lock:
  - trading_app/live_config.py
  - tests/test_trading_app/test_live_config.py
  - scripts/tools/generate_trade_sheet.py
  - scripts/tmp_dst_wrong_time.py
  - scripts/tmp_dst_audit_v2.py
  - scripts/tmp_dst_audit_v3.py
  - HANDOFF.md
acceptance:
  - "Locked RR tried first (existing behavior preserved)"
  - "Fallback only fires when locked RR fails LIVE_MIN"
  - "Fallback candidates: same family, validated only, JK-equal to locked RR (p>0.05, rho=0.7)"
  - "Among JK-equal gate-passers, highest Sharpe wins"
  - "Audit log: family_id, locked_rr, fallback_rr, jk_p, locked_expr, fallback_expr, reason"
  - "family_rr_locks table UNCHANGED"
  - "select_family_rr.py UNCHANGED"
  - "MGC still 0-live (noise_risk becomes binding)"
  - "Existing tests pass, new tests cover fallback path"
proven:
  - "RR policy audit complete: MAX_SHARPE has 73% RR1.0 bias (mechanical Sharpe advantage)"
  - "MGC: all 3 RR levels JK-equal (p=0.64-0.89), Sharpe CV=5.2%"
  - "RR1.5 passes LIVE_MIN (0.235>0.22), best OOS (0.1858), dollar gate PASS"
  - "7 families system-wide where suppression crosses LIVE_MIN"
unproven: []
blockers: []
---

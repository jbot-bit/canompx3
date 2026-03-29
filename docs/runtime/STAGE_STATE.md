---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Ground truth audit + fix all issues — tradebook, caps, DLL, paper_trade_logger
pass: 1
scope_lock:
  - scripts/tmp_ground_audit.py
  - trading_app/prop_profiles.py
  - trading_app/paper_trade_logger.py
  - tests/test_trading_app/test_prop_profiles.py
blast_radius:
  - prop_profiles.py — ORB cap for TOKYO_OPEN, tradeify cleanup
  - paper_trade_logger.py — potential hardcoded lane sync
acceptance:
  - All audit checks produce honest numbers
  - Every fixable issue resolved
  - Drift checks pass
updated: 2026-03-30T10:00:00Z
---

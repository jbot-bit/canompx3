---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Self-funded Tradovate Stage 1 — profile config (11 lanes, caps, payout policy)
updated: 2026-04-03T18:00:00Z
scope_lock:
  - trading_app/prop_profiles.py
  - tests/test_trading_app/test_prop_profiles.py
blast_radius:
  - prop_profiles.py: update self_funded_tradovate profile. No other profiles touched.
  - Tests: add/update test for self-funded profile lane count and config.
  - No schema changes, no pipeline changes.
acceptance:
  - python -c "from trading_app.prop_profiles import ACCOUNT_PROFILES; p=ACCOUNT_PROFILES['self_funded_tradovate']; print(len(p.daily_lanes), p.account_size, p.payout_policy_id)" prints "11 30000 self_funded"
  - All 11 lanes have correct ORB caps (MGC CME=30, SINGAPORE=90, COMEX=150, EUROPE=120, TOKYO=80, NYSE=70, USDATA1000 MNQ=65 MGC=15 MES=20, PRECLOSE=50, CMEREOPEN=50)
  - python -m pytest tests/test_trading_app/test_prop_profiles.py -x -q passes
  - python pipeline/check_drift.py passes
---

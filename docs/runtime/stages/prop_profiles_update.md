---
stage: IMPLEMENTATION
task: Update prop firm rules and trade sheet DD budget
updated: 2026-04-01T09:30:00+10:00
scope_lock:
  - trading_app/prop_profiles.py
  - scripts/tools/generate_trade_sheet.py
  - tests/test_trading_app/test_prop_profiles.py
blast_radius: |
  27 importers of prop_profiles — data value changes only, no interface changes.
  DLL display in bot_dashboard and trade sheet will now show Apex DLL.
  TopStep DLL removed (TopStepX platform).
acceptance:
  - TopStep profit_split flat 90%
  - TopStep DLL = None (TopStepX)
  - Apex DLL added (1000/1500/2000)
  - Apex 150K contracts 9/90
  - MFFU 50K DD = 2000
  - Trade sheet shows recent-regime DD budget
  - test_prop_profiles passes
  - check_drift passes
---

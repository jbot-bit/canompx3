---
task: "Rewrite LIVE_PORTFOLIO specs from ground truth (16 specs replacing 46 dead specs)"
mode: IMPLEMENTATION
stage_purpose: "Replace 46 dead live specs with 16 ground-truth specs. 10 CORE (ROBUST/WHITELISTED), 6 REGIME (PURGED/SINGLETON, fitness-gated). No instrument exclusions. ExpR floor 0.22 interim."
scope_lock:
  - trading_app/live_config.py
acceptance:
  - "build_live_portfolio() resolves >0 strategies for MGC"
  - "build_live_portfolio() resolves >0 strategies for MES"
  - "build_live_portfolio() resolves >0 strategies for MNQ"
  - "pytest tests/test_trading_app/test_live_config.py passes"
  - "zero dead specs (all 16 resolve to at least 1 strategy across instruments)"
proven:
  - "832 validated strategies exist (MGC 20, MES 81, MNQ 731)"
  - "255 family_rr_locks, 248 edge_families rebuilt"
  - "all 16 target combos verified FDR-significant"
  - "family robustness status verified per combo"
unproven:
  - "noise_risk not populated (interim state)"
  - "MGC/MES bars 16 days stale"
  - "ExpR 0.22 threshold is interim, not canon"
blockers: []
updated: "2026-03-22T02:30:00+10:00"
terminal: main
---

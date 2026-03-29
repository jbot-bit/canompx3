---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Sync audit — fix TRADING_RULES.md session statuses with DB truth
pass: 2
scope_lock:
  - TRADING_RULES.md
  - docs/plans/sync_audit_2026-03-30.md
blast_radius:
  - TRADING_RULES.md is a governing doc (wins for trading logic decisions)
  - Only changing status labels, not trading logic
acceptance:
  - BRISBANE_1025 no longer marked "noise-gated, currently inactive"
  - No other contradictions (ARCHITECTURE.md clean)
  - RR4.0 NO-GO contradiction flagged for user review (not auto-fixed)
  - python pipeline/check_drift.py passes
updated: 2026-03-30T06:00:00Z
---

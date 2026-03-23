---
task: "Rolling portfolio rebuild with event-based session names"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Fix stale rolling data (old session names) + fix instrument-scoped delete bug"
updated: 2026-03-23T17:00+10:00
terminal: main
scope_lock:
  - trading_app/regime/discovery.py
  - trading_app/live_config.py
  - HANDOFF.md
acceptance:
  - "regime_strategies/regime_validated use event-based session names only"
  - "HOT-path family lookup no longer returns 'family not found'"
  - "validated_setups/edge_families/family_rr_locks unchanged"
  - "strategy_fitness unchanged"
  - "Delete bug scoped to instrument"
proven:
  - "Session naming mismatch proven (old: 0030/0900/1000, new: CME_REOPEN/NYSE_OPEN etc)"
  - "MES rebuild complete with correct names"
  - "Delete bug: lines 66-67 delete by run_label only, wiping other instruments"
unproven: []
blockers:
  - "MNQ and MGC need re-run after delete bug fix"
---

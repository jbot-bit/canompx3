---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Build portfolio reality check script — honest P&L and DD from raw data
pass: 2
scope_lock:
  - scripts/verification/portfolio_reality_check.py
blast_radius:
  - Read-only script. No production code changes.
acceptance:
  - Every number from gold.db queries, not validated_setups metadata
  - Filters applied via matches_row, not guessed
  - S0.75 applied via apply_tight_stop
  - Per-trade risk_dollars from orb_outcomes
  - Equity curve with max DD vs Apex $2K limit
updated: 2026-03-30T13:00:00Z
---

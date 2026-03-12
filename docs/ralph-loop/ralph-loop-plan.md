## Iteration: 28
## Phase: implement
## Target: trading_app/live_config.py:75,89
## Finding: LIVE_MIN_EXPECTANCY_R and LIVE_MIN_EXPECTANCY_DOLLARS_MULT lack @research-source annotations (DF-08); DF-05 and DF-06 are already resolved (stale ledger entries)
## Decision: implement
## Rationale: Pure annotation addition — no code logic change, no value change. Closes outstanding annotation debt per Provenance Rule (drift check #45 domain). DF-05/DF-06 ledger cleanup is housekeeping.
## Blast Radius:
  - Callers: check_drift.py, generate_promotion_candidates.py, generate_trade_sheet.py, pinecone_snapshots.py, rolling_portfolio.py (comment-only ref)
  - Callees: none (module-level constants)
  - Tests: tests/test_trading_app/test_live_config.py (imports both constants, test_multiplier_constant_is_applied)
  - Drift checks: check #43 imports LIVE_MIN_EXPECTANCY_R — unaffected by comment change
## Invariants (MUST NOT change):
  - LIVE_MIN_EXPECTANCY_R value must remain 0.10
  - LIVE_MIN_EXPECTANCY_DOLLARS_MULT value must remain 1.3
  - All callers must continue to import and use these constants identically
## Diff estimate: ~6 lines added (comment annotations only)

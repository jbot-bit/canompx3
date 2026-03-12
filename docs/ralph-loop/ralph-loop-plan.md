## Iteration: 30
## Phase: implement
## Target: trading_app/strategy_discovery.py:1082
## Finding: SD1 — comment "# E2+E3 (CB1 only)" is stale; E3 is in SKIP_ENTRY_MODELS and never runs, but is intentionally still counted in total_combos for conservative n_trials_at_discovery
## Decision: implement
## Rationale: Pure comment clarification. No code logic change. Explains intentional overcounting for conservative FST hurdle / BH FDR — currently the comment misleads readers into thinking E3 still runs.
## Blast Radius:
  - Callers: test_strategy_discovery.py, test_integration.py, test_integration_l1_l2.py, strategy_fitness.py, walkforward.py, strategy_validator.py — all unaffected by comment change
  - Callees: compute_metrics() passes total_combos as n_trials — value unchanged
  - Tests: tests/test_trading_app/test_strategy_discovery.py (45 tests — no assertions on comments)
  - Drift checks: check_drift.py:2310 validates n_trials_at_discovery is NOT NULL — unaffected
## Invariants (MUST NOT change):
  - total_combos value must remain identical (counting E3 conservatively)
  - n_trials_at_discovery populated identically in experimental_strategies
  - No import, logic, or DB write changes
## Diff estimate: 1 line (comment only)

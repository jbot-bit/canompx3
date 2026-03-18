## Iteration: 132
## Target: trading_app/walkforward.py:134-136
## Finding: Silent failure — tight stop silently skipped when stop_multiplier != 1.0 but cost_spec is None (no warning emitted)
## Classification: [judgment]
## Blast Radius: 1 file changed (walkforward.py); 2 importers (strategy_validator.py [no-touch], test_walkforward.py)
## Invariants:
  1. Pass/fail rule (4 gates) MUST NOT change
  2. apply_tight_stop MUST NOT be called without a valid cost_spec
  3. stop_multiplier == 1.0 path MUST stay silent (no spurious warning)
## Diff estimate: 4 lines

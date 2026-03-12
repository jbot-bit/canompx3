## Iteration: 29
## Phase: implement
## Target: trading_app/outcome_builder.py:677-678
## Finding: build_outcomes() silently falls back to ORB_LABELS when get_enabled_sessions() returns empty — misconfiguration invisible with no warning log
## Decision: implement
## Rationale: Pure logging addition. logger already defined at module level (line 20). No behavior, no logic, no value change. Surfaces misconfigured instruments that currently produce silent no-ops.
## Blast Radius:
  - Callers: scripts/tools/build_outcomes_fast.py, tests/test_integration_l1_l2.py, tests/test_trading_app/test_integration.py (all unaffected by log-only change)
  - Callees: get_enabled_sessions() (pipeline.asset_configs), ORB_LABELS (pipeline.init_db) — both unchanged
  - Tests: tests/test_trading_app/test_outcome_builder.py (27 tests — logging not asserted, all will still pass)
  - Drift checks: check_drift.py imports RR_TARGETS from outcome_builder (unaffected). No drift check validates logging.
## Invariants (MUST NOT change):
  - sessions fallback to ORB_LABELS must still happen (logic preserved)
  - No change to any outcome computation or DB write behavior
  - logger.warning() is the only addition — no new imports, no new logic
## Diff estimate: 1 line added

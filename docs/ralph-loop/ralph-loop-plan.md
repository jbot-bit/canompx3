## Iteration: 53
## Target: trading_app/execution_spec.py:46 + tests/test_trading_app/test_execution_spec.py
## Finding: Hardcoded ["E1", "E3"] in ExecutionSpec.validate() — E2 (active entry model) rejected, E3 (soft-retired) accepted. Should use canonical ENTRY_MODELS from trading_app.config.
## Blast Radius: 2 files (execution_spec.py + test_execution_spec.py). db_manager.py and nested/schema.py reference column name only — unaffected.
## Invariants: [1] E3 must still be accepted (in ENTRY_MODELS for schema/test compat); [2] E4 and unknown models must still raise ValueError; [3] frozen dataclass behavior unchanged
## Diff estimate: ~6 lines in execution_spec.py, ~8 lines in test file

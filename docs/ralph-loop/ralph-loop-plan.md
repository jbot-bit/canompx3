## Iteration: 117
## Target: trading_app/strategy_validator.py:824,1056
## Finding: Hardcoded `"E1"` fallback in `rd.get("entry_model", "E1")` is unreachable dead code (schema enforces NOT NULL) but would silently assign wrong entry model if somehow triggered; no warning emitted
## Classification: [mechanical]
## Blast Radius: 1 file (strategy_validator.py), no callers of the affected lines (internal to run_validation/_build_worker_kwargs), 49 passing tests
## Invariants: (1) behavior unchanged — path is unreachable; (2) no new exceptions raised; (3) test suite stays green
## Diff estimate: 4 lines (2 warning logs added, one at each fallback site)

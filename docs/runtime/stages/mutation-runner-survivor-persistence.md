task: Fix run_mutation.sh so survivor evidence (line-anchored) survives the EXIT trap; recover this run's 671 strategy_fitness survivors and triage them.
mode: IMPLEMENTATION

## Scope Lock
- scripts/tools/run_mutation.sh
- docs/audit/2026-06-07-mutation-testing-capital-core.md
- tests/test_trading_app/test_strategy_fitness.py

## Blast Radius
- scripts/tools/run_mutation.sh — adds a durable structured-report export (cr-report --surviving-only --show-diff) + sqlite preservation BEFORE the EXIT trap deletes .mutation/. Behavior-additive: existing baseline/init/exec/cr-rate flow unchanged; only the reporting/teardown tail changes. Callers: invoked manually per campaign doc; no programmatic importers. Reads: cosmic-ray session sqlite (read-only via cr-report). Writes: a new gitignored durable report file + a preserved gitignored sqlite copy. No production trading logic, no schema, no gold.db.
- docs/audit/2026-06-07-mutation-testing-capital-core.md — fills the strategy_fitness section with before/after kill rate + survivor triage. Doc only.
- tests/test_trading_app/test_strategy_fitness.py — adds killing tests for capital-bearing survivors (classify_fitness thresholds/comparisons, _recent_trade_sharpe arithmetic, diagnose_decay). Test-only; raises kill rate toward the >=80% bar.
- Recovery of THIS run: cosmic-ray init is AST-only + deterministic against unchanged committed source (verified: strategy_fitness.py clean at HEAD f1e0ad32, the run's commit) -> re-init reproduces identical specs/line positions; join to outcomes already in .mutation_run.log. No 22h re-run.

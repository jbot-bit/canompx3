## Iteration: 225
## Target: trading_app/db_manager.py
## Cluster: 2 findings, types=[canonical_violation, stale_metadata], severity=[LOW, LOW]
## Classification: [mechanical]
## Blast Radius: 1 file (db_manager.py), 0 callers affected. Test file: tests/test_trading_app/test_db_manager.py
## Invariants:
## - Schema DDL logic unchanged (CREATE TABLE / ALTER TABLE paths unmodified)
## - verify_trading_app_schema still returns (bool, list[str]) contract unchanged
## - No behavior change: all INSERT callers already supply instrument explicitly
## Diff estimate: ~10 lines
## Doctrine cited: institutional-rigor.md § 10 (canonical sources — instrument literals), § 5 (no dead/lying fields — DEFAULT 'MNQ' is a lying fallback)
## Findings deferred: none

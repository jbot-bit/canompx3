## Iteration: 169
## Target: trading_app/db_manager.py:verify_trading_app_schema (lines 883-961)
## Finding: verify_trading_app_schema expected_cols for validated_setups missing 10 migration-added columns (discovery_k, discovery_date, era_dependent, max_year_pct, wfe_verdict, wfe_investigation_date, wfe_investigation_notes, slippage_validation_status, validation_pathway, c8_oos_status); experimental_strategies missing 2 (validation_pathway, c8_oos_status). Silent verifier gap — returns (True, []) even when columns absent.
## Classification: [mechanical]
## Blast Radius: 2 files (trading_app/db_manager.py, tests/test_trading_app/test_db_manager.py)
## Invariants:
## 1. verify_trading_app_schema must still return (True, []) after init_trading_app_schema runs on a fresh DB
## 2. No production behavior changes — only expected_cols sets updated
## 3. Both verified column sets must exactly match the union of CREATE TABLE DDL + all ALTER TABLE migrations
## Diff estimate: 12 lines production code
## Doctrine cited: integrity-guardian.md § 3 (fail-closed, never return success on audit paths), § 5 (evidence over assertion — verify must actually verify)

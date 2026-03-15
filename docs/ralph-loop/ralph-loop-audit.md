# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 72

## RALPH AUDIT — Iteration 72 (2 files scanned)
## Date: 2026-03-15
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `ruff check` | PASS | Clean |
| `import smoke test` | PASS | Module imports OK |

---

## Files Audited This Iteration

### scripts/tools/assert_rebuild.py — FIXED (AR-01)

#### Finding AR-01: Canonical violation — hardcoded APERTURES = [5, 15, 30] (FIXED)
- **Sin**: Canonical violation — same pattern fixed in iters 62 (PS-01), 65 (CD-01), 68 (RP-01). `APERTURES` should derive from `VALID_ORB_MINUTES`.
- **Severity**: MEDIUM (if apertures change, post-rebuild assertions would check wrong values)
- **Fix**: Import `VALID_ORB_MINUTES` from `pipeline.build_daily_features`; replace hardcoded list with canonical reference
- **Lines changed**: 2 (1 import, 1 alias)
- **Blast radius**: 1 file (assert_rebuild.py), assertion A5 outcome coverage check

#### Seven Sins scan — COMPLETE

| Sin | Status | Detail |
|-----|--------|--------|
| Canonical violation | AR-01 FIXED | Hardcoded apertures |
| Silent failure | CLEAN | All assertion functions return explicit results |
| Fail-open | CLEAN | `has_failures()` gates on FAIL severity |
| Broad exception | 1 ACCEPTABLE (line 293) | assert_schema_alignment: `except Exception` returns FAIL result (fail-closed) |
| Orphan/dead code | CLEAN | All functions used |
| Volatile data | CLEAN | `_STATIC_COLUMN_COUNT` and `_ORB_COLUMNS_PER_SESSION` are structural constants with documented breakdown |

### scripts/infra/backup_db.py — CLEAN

- `verify_backup()` fail-closed (returns False on any error)
- Path traversal check in `restore_db()` (line 212)
- All connections use try/finally
- `except Exception` in verify returns False (fail-closed)
- No canonical violations, no orphan code

---

## Deferred Findings — Status After Iter 72

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 2 files scanned: `assert_rebuild.py` AR-01 FIXED, `backup_db.py` CLEAN
- 1 fix, 0 new deferrals
- Infrastructure Gates: 3/3 PASS
- Action: fix (mechanical)

**Next iteration targets:**
- `scripts/tools/gen_repo_map.py` — repo map generator, unscanned
- `scripts/tools/sync_pinecone.py` — Pinecone sync, unscanned
- `scripts/migrations/retire_e3_strategies.py` — E3 retirement, unscanned

---

## Files Fully Scanned

> Cumulative list of every file that received a complete Seven Sins scan.

- `trading_app/execution_engine.py` — iter 50 (1 fix: EE-01)
- `trading_app/strategy_fitness.py` — iter 44 (CLEAN)
- `trading_app/outcome_builder.py` — iter 46 (1 fix: OB1)
- `trading_app/strategy_discovery.py` — iter 34 (2 fixes)
- `trading_app/strategy_validator.py` — iter 47 (1 fix: SV1)
- `trading_app/portfolio.py` — iter 48 (1 fix: PF1)
- `trading_app/rolling_portfolio.py` — iter 43 (1 fix: RP1)
- `trading_app/live_config.py` — iter 51 (1 fix: LC-01)
- `trading_app/paper_trader.py` — iter 41 (1 fix: PT1)
- `trading_app/scoring.py` — iter 39 (CLEAN)
- `trading_app/risk_manager.py` — iter 39 (1 fix: RM1)
- `trading_app/cascade_table.py` — iter 37 (1 fix: CT1)
- `trading_app/market_state.py` — iter 38 (1 fix: MS1)
- `trading_app/config.py` — iter 24
- `trading_app/walkforward.py` — iter 15
- `trading_app/mcp_server.py` — iter 32 (2 fixes)
- `trading_app/live/position_tracker.py` — iter 26 (1 fix)
- `trading_app/live/bar_aggregator.py` — iter 26 (CLEAN)
- `trading_app/live/webhook_server.py` — iter 4 (2 fixes)
- `trading_app/live/session_orchestrator.py` — iter 10 (2 fixes)
- `trading_app/live/performance_monitor.py` — iter 10 (2 fixes)
- `trading_app/live/tradovate/order_router.py` — iter 21 (1 fix)
- `trading_app/live/projectx/order_router.py` — iter 21 (1 fix)
- `trading_app/live/tradovate/auth.py` — iter 24 (1 fix)
- `trading_app/entry_rules.py` — iter 52 (CLEAN)
- `trading_app/db_manager.py` — iter 52 (CLEAN)
- `trading_app/execution_spec.py` — iter 53 (1 fix: ES-01)
- `trading_app/setup_detector.py` — iter 53 (CLEAN)
- `trading_app/calendar_overlay.py` — iter 53 (CLEAN)
- `trading_app/pbo.py` — iter 54 (CLEAN)
- `trading_app/nested/builder.py` — iter 54 (CLEAN)
- `trading_app/nested/schema.py` — iter 54 (CLEAN)
- `trading_app/nested/discovery.py` — iter 55 (1 fix: ND-01)
- `trading_app/nested/validator.py` — iter 55 (CLEAN)
- `trading_app/nested/audit_outcomes.py` — iter 55 (CLEAN)
- `trading_app/nested/compare.py` — iter 56 (CLEAN)
- `scripts/tools/build_edge_families.py` — iter 56 (CLEAN)
- `pipeline/build_daily_features.py` — iter 36 (1 fix)
- `pipeline/ingest_dbn.py` — iter 1 (triaged)
- `pipeline/build_bars_5m.py` — iter 1 (triaged)
- `pipeline/dst.py` — iter 1 (triaged)
- `pipeline/cost_model.py` — iter 12 (triaged)
- `pipeline/asset_configs.py` — iter 1 (triaged)
- `scripts/tools/generate_trade_sheet.py` — iter 18 (3 fixes)
- `trading_app/live/circuit_breaker.py` — iter 57 (CLEAN)
- `trading_app/live/cusum_monitor.py` — iter 57 (CLEAN)
- `trading_app/live/notifications.py` — iter 57 (CLEAN)
- `trading_app/live/live_market_state.py` — iter 57 (CLEAN)
- `trading_app/live/multi_runner.py` — iter 57 (CLEAN)
- `trading_app/live/broker_factory.py` — iter 57 (CLEAN)
- `trading_app/live/broker_base.py` — iter 57 (CLEAN)
- `trading_app/live/trade_journal.py` — iter 57 (CLEAN)
- `trading_app/live/projectx/auth.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/contract_resolver.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/data_feed.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/order_router.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/positions.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/auth.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/contract_resolver.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/data_feed.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/order_router.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/positions.py` — iter 58 (CLEAN)
- `trading_app/regime/discovery.py` — iter 59 (CLEAN)
- `trading_app/regime/validator.py` — iter 59 (CLEAN)
- `trading_app/regime/compare.py` — iter 59 (CLEAN)
- `trading_app/regime/schema.py` — iter 59 (CLEAN)
- `trading_app/ai/query_agent.py` — iter 60 (CLEAN)
- `trading_app/ai/sql_adapter.py` — iter 60 (CLEAN)
- `trading_app/ai/grounding.py` — iter 60 (CLEAN)
- `trading_app/ai/corpus.py` — iter 60 (CLEAN)
- `trading_app/ai/cli.py` — iter 60 (CLEAN)
- `trading_app/ai/strategy_matcher.py` — iter 60 (CLEAN)
- `trading_app/prop_portfolio.py` — iter 61 (1 fix)
- `trading_app/prop_profiles.py` — iter 61 (1 fix)
- `scripts/tools/pipeline_status.py` — iter 62-64 (3 fixes)
- `pipeline/check_drift.py` — iter 65-66 (1 fix: CD-01)
- `scripts/tools/audit_behavioral.py` — iter 66 (1 fix: AB-01)
- `pipeline/health_check.py` — iter 67 (1 fix: HC-01)
- `pipeline/run_pipeline.py` — iter 68 (1 fix: RP-01)
- `pipeline/init_db.py` — iter 69 (1 fix: ID-01)
- `pipeline/dashboard.py` — iter 70 (CLEAN)
- `pipeline/db_lock.py` — iter 70 (CLEAN)
- `pipeline/audit_log.py` — iter 70 (CLEAN)
- `pipeline/paths.py` — iter 71 (CLEAN)
- `pipeline/log.py` — iter 71 (CLEAN)
- `pipeline/check_db.py` — iter 71 (CLEAN)
- `scripts/tools/select_family_rr.py` — iter 71 (1 fix: SFR-01)
- `scripts/tools/assert_rebuild.py` — iter 72 (1 fix: AR-01)
- `scripts/infra/backup_db.py` — iter 72 (CLEAN)

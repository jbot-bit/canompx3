# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 71

## RALPH AUDIT — Iteration 71 (5 files scanned)
## Date: 2026-03-15
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `ruff check` | PASS | Clean |
| `import smoke test` | PASS | Module imports OK |

---

## Files Audited This Iteration

### pipeline/paths.py — CLEAN (55 lines)
- Correct dotenv fallback, DB path resolution with warning. No sins.

### pipeline/log.py — CLEAN (16 lines)
- Trivial logger setup. No sins.

### pipeline/check_db.py — CLEAN (89 lines)
- Uses `with` for connection (auto-close). No sins.

### scripts/tools/select_family_rr.py — FIXED (SFR-01)

#### Finding SFR-01: Canonical violation — hardcoded entry_model IN ('E1', 'E2') (FIXED)
- **Sin**: Canonical violation — SQL query hardcodes `entry_model IN ('E1', 'E2')` instead of deriving from `ENTRY_MODELS - SKIP_ENTRY_MODELS`. If E4 is added, this script silently misses it.
- **Severity**: MEDIUM (new entry model would be excluded from RR lock selection)
- **Fix**: Import `ENTRY_MODELS` and `SKIP_ENTRY_MODELS` from `trading_app.config`; build active models list and SQL IN clause dynamically
- **Lines changed**: 4 (1 import, 2 active_models computation, 1 SQL f-string)
- **Blast radius**: 1 file (select_family_rr.py), query filter only

#### Additional observation: connection not in try/finally (SFR-02, ACCEPTABLE)
- `con.close()` at line 213 not in `finally`. Low severity — CLI script, process exit closes connection.

#### Seven Sins scan — COMPLETE for all 4 files

---

## Deferred Findings — Status After Iter 71

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 5 files scanned: paths.py (CLEAN), log.py (CLEAN), check_db.py (CLEAN), select_family_rr.py (SFR-01 FIXED)
- 1 fix, 0 new deferrals
- Infrastructure Gates: 3/3 PASS
- Action: fix (mechanical)

**Next iteration targets:**
- `scripts/tools/gen_repo_map.py` — repo map generator, unscanned
- `scripts/tools/sync_pinecone.py` — Pinecone sync, unscanned
- `scripts/tools/assert_rebuild.py` — post-rebuild assertions, unscanned
- `scripts/infra/backup_db.py` — DB backup utility, unscanned

---

## Files Fully Scanned

> Cumulative list of every file that received a complete Seven Sins scan.
> Prevents re-scanning clean files. Shows coverage gaps at a glance.

- `trading_app/execution_engine.py` — iter 50 (1 fix: EE-01 fail-open)
- `trading_app/strategy_fitness.py` — iter 44 (CLEAN)
- `trading_app/outcome_builder.py` — iter 46 (1 fix: OB1 silent fallback)
- `trading_app/strategy_discovery.py` — iter 34 (2 fixes: SD1 orphan, SD2 volatile)
- `trading_app/strategy_validator.py` — iter 47 (1 fix: SV1 docstring)
- `trading_app/portfolio.py` — iter 48 (1 fix: PF1 orphan)
- `trading_app/rolling_portfolio.py` — iter 43 (1 fix: RP1 orphan)
- `trading_app/live_config.py` — iter 51 (1 fix: LC-01 broad except)
- `trading_app/paper_trader.py` — iter 41 (1 fix: PT1 orphan)
- `trading_app/scoring.py` — iter 39 (CLEAN, SC1 acceptable)
- `trading_app/risk_manager.py` — iter 39 (1 fix: RM1 orphan)
- `trading_app/cascade_table.py` — iter 37 (1 fix: CT1 orphan)
- `trading_app/market_state.py` — iter 38 (1 fix: MS1 orphan)
- `trading_app/config.py` — iter 24 (annotation fixes)
- `trading_app/walkforward.py` — iter 15 (annotation fixes)
- `trading_app/mcp_server.py` — iter 32 (2 fixes: volatile data, dead code)
- `trading_app/live/position_tracker.py` — iter 26 (1 fix: PT1 falsy-zero)
- `trading_app/live/bar_aggregator.py` — iter 26 (CLEAN)
- `trading_app/live/webhook_server.py` — iter 4 (2 fixes: timing-safe auth, deprecated asyncio)
- `trading_app/live/session_orchestrator.py` — iter 10 (2 fixes: CUSUM reset, fill poller)
- `trading_app/live/performance_monitor.py` — iter 10 (2 fixes: CUSUM threshold, daily reset)
- `trading_app/live/tradovate/order_router.py` — iter 21 (fill price falsy-zero)
- `trading_app/live/projectx/order_router.py` — iter 21 (fill price falsy-zero)
- `trading_app/live/tradovate/auth.py` — iter 24 (log gap)
- `trading_app/entry_rules.py` — iter 52 (CLEAN)
- `trading_app/db_manager.py` — iter 52 (CLEAN)
- `trading_app/execution_spec.py` — iter 53 (1 fix: ES-01 canonical violation)
- `trading_app/setup_detector.py` — iter 53 (CLEAN)
- `trading_app/calendar_overlay.py` — iter 53 (CLEAN)
- `trading_app/pbo.py` — iter 54 (CLEAN)
- `trading_app/nested/builder.py` — iter 54 (CLEAN)
- `trading_app/nested/schema.py` — iter 54 (CLEAN)
- `trading_app/nested/discovery.py` — iter 55 (1 fix: ND-01 SKIP_ENTRY_MODELS guard)
- `trading_app/nested/validator.py` — iter 55 (CLEAN)
- `trading_app/nested/audit_outcomes.py` — iter 55 (CLEAN)
- `trading_app/nested/compare.py` — iter 56 (CLEAN, 3 ACCEPTABLE observations)
- `scripts/tools/build_edge_families.py` — iter 56 (CLEAN)
- `pipeline/build_daily_features.py` — iter 36 (1 fix: canonical extraction)
- `pipeline/ingest_dbn.py` — iter 1 (triaged via M2.5)
- `pipeline/build_bars_5m.py` — iter 1 (triaged via M2.5)
- `pipeline/dst.py` — iter 1 (triaged via M2.5)
- `pipeline/cost_model.py` — iter 12 (triaged)
- `pipeline/asset_configs.py` — iter 1 (triaged via M2.5)
- `scripts/tools/generate_trade_sheet.py` — iter 18 (3 fixes: dollar gate, join, friction)
- `trading_app/live/circuit_breaker.py` — iter 57 (CLEAN)
- `trading_app/live/cusum_monitor.py` — iter 57 (CLEAN)
- `trading_app/live/notifications.py` — iter 57 (CLEAN, intentional fail-open)
- `trading_app/live/live_market_state.py` — iter 57 (CLEAN)
- `trading_app/live/multi_runner.py` — iter 57 (CLEAN)
- `trading_app/live/broker_factory.py` — iter 57 (CLEAN)
- `trading_app/live/broker_base.py` — iter 57 (CLEAN)
- `trading_app/live/trade_journal.py` — iter 57 (CLEAN, intentional fail-open)
- `trading_app/live/projectx/auth.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/contract_resolver.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/data_feed.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/order_router.py` — iter 58 (CLEAN, supersedes iter 21 old path)
- `trading_app/live/projectx/positions.py` — iter 58 (CLEAN, ACCEPTABLE int vs float default)
- `trading_app/live/tradovate/auth.py` — iter 58 (CLEAN, supersedes iter 24 old path)
- `trading_app/live/tradovate/contract_resolver.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/data_feed.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/order_router.py` — iter 58 (CLEAN, supersedes iter 21 old path)
- `trading_app/live/tradovate/positions.py` — iter 58 (CLEAN)
- `trading_app/regime/discovery.py` — iter 59 (CLEAN, SKIP_ENTRY_MODELS guard already present)
- `trading_app/regime/validator.py` — iter 59 (CLEAN)
- `trading_app/regime/compare.py` — iter 59 (CLEAN)
- `trading_app/regime/schema.py` — iter 59 (CLEAN)
- `trading_app/ai/query_agent.py` — iter 60 (CLEAN, ACCEPTABLE broad except in query tool)
- `trading_app/ai/sql_adapter.py` — iter 60 (CLEAN, ACCEPTABLE VALID_ENTRY_MODELS includes E3)
- `trading_app/ai/grounding.py` — iter 60 (CLEAN, ACCEPTABLE session names in prompt text)
- `trading_app/ai/corpus.py` — iter 60 (CLEAN)
- `trading_app/ai/cli.py` — iter 60 (CLEAN)
- `trading_app/ai/strategy_matcher.py` — iter 60 (CLEAN, ACCEPTABLE hardcoded MGC in research tool)
- `trading_app/prop_portfolio.py` — iter 61 (1 fix: PP-01 import sort)
- `trading_app/prop_profiles.py` — iter 61 (1 fix: PP-01 import sort)
- `scripts/tools/pipeline_status.py` — iter 62-64 (3 fixes: PS-01 canonical, PS-02 silent fallback, PS-03 preflight key mismatch)
- `pipeline/check_drift.py` — iter 65-66 (1 fix: CD-01 canonical aperture count)
- `scripts/tools/audit_behavioral.py` — iter 66 (1 fix: AB-01 instrument regex)
- `pipeline/health_check.py` — iter 67 (1 fix: HC-01 connection leak)
- `pipeline/run_pipeline.py` — iter 68 (1 fix: RP-01 canonical orb_minutes choices)
- `pipeline/init_db.py` — iter 69 (1 fix: ID-01 fail-open force drop)
- `pipeline/dashboard.py` — iter 70 (CLEAN)
- `pipeline/db_lock.py` — iter 70 (CLEAN)
- `pipeline/audit_log.py` — iter 70 (CLEAN)
- `pipeline/paths.py` — iter 71 (CLEAN)
- `pipeline/log.py` — iter 71 (CLEAN)
- `pipeline/check_db.py` — iter 71 (CLEAN)
- `scripts/tools/select_family_rr.py` — iter 71 (1 fix: SFR-01 canonical entry_model)

# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 64

## RALPH AUDIT — Iteration 64 (scripts/tools/pipeline_status.py)
## Date: 2026-03-15
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `ruff check` | PASS | Clean |
| `pytest test_pipeline_status.py` | PASS | 31/31 tests |

---

## Files Audited This Iteration

### scripts/tools/pipeline_status.py — FIXED (PS-03)

#### Finding PS-03: Silent failure — 4/5 preflight rules unreachable (FIXED)
- **Sin**: Silent failure (PS-02 category) — `PREFLIGHT_RULES` keys don't match `REBUILD_STEPS` step base names
- **Severity**: HIGH (preflight checks exist but silently do nothing for 4 of 5 steps)
- **Root cause**: `_parse_step_preflight("discovery_O5")` returns base `"discovery"`, but `PREFLIGHT_RULES` has key `"strategy_discovery"`. Lookup fails silently — `preflight_check` returns `(True, "No pre-flight rule...")`, skipping the check.
- **Affected keys**: `strategy_discovery`→`discovery`, `strategy_validator`→`validator`, `build_edge_families`→`edge_families`, `select_family_rr`→`family_rr_locks`
- **Fix**: Renamed all 4 mismatched `PREFLIGHT_RULES` keys to match their step base names
- **Lines changed**: 4 (key renames only, no logic change)
- **Blast radius**: 0 external callers of `PREFLIGHT_RULES` keys. Internal-only lookup in `preflight_check()`. Tests pass (only tested `outcome_builder` key, which was already correct).

#### Seven Sins scan — COMPLETE

| Sin | Status | Detail |
|-----|--------|--------|
| Silent failure | PS-03 FIXED | Preflight key mismatch (see above) |
| Fail-open | CLEAN | `run_rebuild` returns False on all failure paths; `preflight_check` now reaches all rules |
| Canonical violation | PS-01 FIXED (iter 62) | `APERTURES = VALID_ORB_MINUTES` |
| Orphan/dead code | CLEAN | All imports used, no dead functions |
| Volatile data | CLEAN | No hardcoded counts or stats |
| Broad exception | CLEAN | All `except` clauses are specific (`duckdb.BinderException`, `duckdb.CatalogException`, `duckdb.InvalidInputException`, `TimeoutExpired`, `PipelineLockError`) |
| Schema trust | PS-02 FIXED (iter 63) | `family_rr_locks` fallback now logs schema-miss |

**Full scan complete.** All 7 sins checked. 3 findings across iters 62-64 (PS-01, PS-02, PS-03), all fixed.

---

## Deferred Findings — Status After Iter 64

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 1 file: `scripts/tools/pipeline_status.py` PS-03 FIXED (silent failure — 4 unreachable preflight rules, 4 key renames)
- 0 new deferrals
- Infrastructure Gates: 3/3 PASS
- Action: fix (judgment)

**Next iteration targets:**
- `pipeline/check_drift.py` — large file (3539 lines), scan check definitions for canonical violations
- `scripts/tools/audit_behavioral.py` — key infrastructure tool, unscanned
- `pipeline/health_check.py` — orchestration script, unscanned

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

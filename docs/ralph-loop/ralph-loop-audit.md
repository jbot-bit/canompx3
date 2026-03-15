# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 56

## RALPH AUDIT — Iteration 56 (nested/compare.py + scripts/tools/build_edge_families.py)
## Date: 2026-03-15
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest test_nested/ + test_edge_families.py` | PASS | 85 tests, no failures |
| `ruff check` | 2 pre-existing I001 in prop_portfolio.py + prop_profiles.py (out of scope) |

---

## Files Audited This Iteration

### nested/compare.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. No exception swallowing. DuckDB errors propagate. Read-only connection.
- **Fail-open**: CLEAN. Reporting-only tool. `continue` on missing `nested_strategies` table (conservative).
- **Look-ahead bias**: CLEAN. Reads from already-written strategies; post-hoc comparison only.
- **Cost illusion**: CLEAN. No PnL computation — reads pre-computed `sharpe_ratio` and `expectancy_r`.
- **Canonical violation**: CLEAN. `ORB_LABELS` from `pipeline.init_db`; `GOLD_DB_PATH` from `pipeline.paths`. Hardcoded `orb_minutes_list = [15, 30]` at line 68 is a function parameter default for the research experiment — ACCEPTABLE (intentional per-experiment configuration, not a production canonical list).
- **Orphan risk**: CLEAN. Imported and tested in `test_nested/test_validator_compare.py`.
- **Volatile data**: CLEAN. No hardcoded counts.

#### Observations (all ACCEPTABLE)

- **Line 68**: `orb_minutes_list = [15, 30]` — function default for nested experiment apertures. ACCEPTABLE: intentional per-experiment heuristic, not a canonical list.
- **Lines 129, 134**: `(x or 0)` after outer `is not None` guard — redundant but harmless (0.0 and 0 are equivalent for float arithmetic). ACCEPTABLE: style, no correctness impact.
- **Line 231**: `(b['expectancy_r'] or 0)` in display-only print — masks None as 0.0000 silently. ACCEPTABLE: reporting tool only, no correctness impact on any production path.

### scripts/tools/build_edge_families.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. No exception swallowing. DuckDB errors propagate naturally. `fallback_count` warning is printed, not hidden.
- **Fail-open**: CLEAN. Fail gates before `con.commit()` — mega-family check (>100 members) and singleton rate check (>70% singletons) raise `RuntimeError` to abort before commit.
- **Look-ahead bias**: CLEAN. Groups already-validated strategies by trade-day hash; no future data.
- **Cost illusion**: CLEAN. No PnL computation — reads pre-computed `expectancy_r`/`sharpe_ann` from `validated_setups`.
- **Canonical violation**: CLEAN. `ACTIVE_ORB_INSTRUMENTS` from `pipeline.asset_configs`; `GOLD_DB_PATH` from `pipeline.paths`; `CORE_MIN_SAMPLES`/`REGIME_MIN_SAMPLES` from `trading_app.config`. Robustness thresholds have `@research-source` + `@revalidated-for` annotations at lines 32-37.
- **Orphan risk**: CLEAN. Tested in `tests/test_trading_app/test_edge_families.py` (22 tests).
- **Volatile data**: CLEAN. No hardcoded counts. Classification thresholds imported from config.

#### Specific checks

- `cv_expr = float("inf")` when `mean_expr <= 0` — correct: negative-mean strategies fail the WHITELIST CV filter (`inf > 0.5`). No silent pass.
- DELETE+INSERT pattern: idempotent — clears families for instrument before rebuilding.
- PBO computation (lines 363-384): non-critical path; `pbo_result.get("pbo")` returns None if unavailable — UPDATE only fires when value is present. No silent failure.

---

## Deferred Findings — Status After Iter 56

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- nested/compare.py: CLEAN — 0 actionable findings (3 ACCEPTABLE observations)
- scripts/tools/build_edge_families.py: CLEAN — 0 findings
- Infrastructure Gates: 4/4 PASS
- Action: audit-only

**Next iteration targets:**
- `pipeline/check_drift.py` — not yet audited this cycle (large file, 3539 lines; focus on the check definitions themselves, not the full file)
- `scripts/tools/generate_trade_sheet.py` — already audited iter 18, but check if any new patterns since then
- `trading_app/live/data_feed.py` — not yet in Files Fully Scanned

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
- `trading_app/tradovate/order_router.py` — iter 21 (fill price falsy-zero)
- `trading_app/projectx/order_router.py` — iter 21 (fill price falsy-zero)
- `trading_app/tradovate/auth.py` — iter 24 (log gap)
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

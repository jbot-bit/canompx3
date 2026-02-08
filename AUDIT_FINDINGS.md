# Full Codebase Audit Findings (2026-02-08)

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Fixed

---

## CRITICAL (must fix before live trading)

### C1: Paper trader doesn't wire RiskManager to ExecutionEngine
- **File:** `paper_trader.py:168-169`
- **Impact:** Risk management completely bypassed in paper trading
- **Fix:** Pass `risk_mgr` to `ExecutionEngine(portfolio, cost_spec, risk_manager=risk_mgr)`
- **Status:** [x] Fixed

### C2: Missing `active_trades` parameter in E1/E3 `can_enter()` calls
- **File:** `execution_engine.py:479-484, 550-553`
- **Impact:** TypeError crash at runtime; max_concurrent bypassed for E1/E3
- **Fix:** Add `active_trades=self.active_trades` to both call sites
- **Status:** [x] Fixed

### C3: E1 fill-bar exit divergence between backtester and execution engine
- **File:** `execution_engine.py` vs `outcome_builder.py:119`
- **Impact:** Live engine shows worse performance than backtest (systematic negative divergence)
- **Detail:** Backtester uses `ts_utc > entry_ts` (excludes fill bar), engine includes fill bar
- **Fix:** Add `continue` after E1 ENTRY like E3 already does
- **Status:** [x] Fixed

### C4: Paper trader EOD scratch PnL hardcoded to 0.0
- **File:** `paper_trader.py:279`
- **Impact:** Journal entries don't sum to daily_pnl_r
- **Fix:** Use actual mark-to-market PnL from engine's completed trade
- **Status:** [x] Fixed

### C5: No drift check prevents nested code writing to production tables
- **File:** `check_drift.py` (Check 15 only blocks imports, not SQL)
- **Impact:** Could silently corrupt orb_outcomes/experimental_strategies
- **Fix:** Add Check 17: scan nested/*.py for SQL writes to production tables
- **Status:** [x] Fixed (Check 17 added)

---

## IMPORTANT (should fix)

### I1: RiskManager.on_trade_entry()/on_trade_exit() never called by engine
- **File:** `execution_engine.py` (all three entry points + _exit_trade)
- **Impact:** daily_trade_count never incremented; circuit breaker never triggered
- **Fix:** Wire on_trade_entry() after each ENTRY, on_trade_exit(pnl_r) after each EXIT
- **Status:** [x] Fixed (E1, E2, E3 on_trade_entry + _exit_trade + on_trading_day_end on_trade_exit)

### I2: Paper trader redundant risk check (now that engine has risk_manager)
- **File:** `paper_trader.py:199-225`
- **Impact:** Double-checking. Engine now rejects before ENTRY event. Paper trader checks again.
- **Fix:** Remove redundant can_enter() from paper trader; handle REJECT events instead
- **Status:** [x] Fixed (paper_trader now handles ENTRY/REJECT/EXIT/SCRATCH from engine)

### I3: Check 4 (schema-query consistency) only scans pipeline/, not trading_app/
- **File:** `check_drift.py:205`
- **Impact:** SQL misspellings in trading_app undetected
- **Fix:** Extend Check 4 to scan both directories
- **Status:** [x] Fixed (Check 18 added)

### I4: No guard against pytz or hardcoded timezone offsets
- **File:** `check_drift.py`
- **Impact:** Known footgun not guarded
- **Fix:** Add Check 19: block pytz imports and timedelta(hours=10) patterns
- **Status:** [x] Fixed (Check 19 added)

### I5: Scratch PnL undefined (NULL, not mark-to-market) in outcome_builder
- **File:** `outcome_builder.py:150`
- **Impact:** Backtest ignores EOD exit prices; potential bias
- **Detail:** Design decision. Scratches are real outcomes but PnL is left NULL.
- **Status:** [ ] Design decision needed (low priority, known limitation)

### I6: Warning text in ingest_dbn_mgc.py says "pre-2019" but MINIMUM_START_DATE is 2016
- **File:** `ingest_dbn_mgc.py:584-585`
- **Fix:** Update text to reference 2016
- **Status:** [x] Fixed

### I7: CLAUDE.md says "starts 2021-02-05" but we're at 2016 now
- **File:** `CLAUDE.md`
- **Fix:** Update documentation
- **Status:** [x] Fixed (data files, minimum start date, test/check counts updated)

---

## DRIFT CHECK GAPS

### D1: Nested production table write guard
- **Fix:** Scan nested/*.py for SQL writes to orb_outcomes, experimental_strategies, validated_setups
- **Status:** [x] Fixed (Check 17)

### D2: Extend Check 4 to scan trading_app/ SQL
- Same as I3
- **Status:** [x] Fixed (Check 18)

### D3: Timezone hygiene (pytz/hardcoded offsets)
- Same as I4
- **Status:** [x] Fixed (Check 19)

---

## TEST COVERAGE GAPS (top 10 by risk)

### T1: ExecutionEngine + RiskManager integration: zero tests (Risk 10/10)
- **Status:** [x] Fixed (20 tests in test_engine_risk_integration.py)

### T2: audit_outcomes._outcomes_match(): zero unit tests (Risk 8/10)
- **Status:** [x] Fixed (17 tests in test_nested/test_audit_outcomes.py)

### T3: resample_to_5m with Brisbane-TZ timestamps from DuckDB (Risk 7/10)
- **Status:** [x] Fixed (9 tests in test_nested/test_resample.py)

### T4: nested/validator.py + compare.py: zero test files (Risk 7/10)
- **Status:** [x] Fixed (11 tests in test_nested/test_validator_compare.py)

### T5: armed_at_bar guard: never directly tested (Risk 7/10)
- **Status:** [x] Fixed (2 tests in test_execution_engine.py::TestArmedAtBarGuard)

### T6: RSI edge cases (avg_loss near zero after smoothing) (Risk 6/10)
- **Status:** [x] Fixed (9 tests in test_pipeline/test_rsi_edge_cases.py)

### T7: outcome_builder.py UTC normalization of break_ts (Risk 6/10)
- **Status:** [x] Fixed (4 tests in test_outcome_builder_utc.py)

### T8: entry_rules.py close exactly AT ORB boundary (Risk 5/10)
- **Status:** [x] Fixed (18 tests in test_entry_rules.py boundary classes)

### T9: Portfolio build_strategy_daily_series with VolumeFilter (Risk 5/10)
- **Status:** [x] Fixed (14 tests in test_portfolio_volume_filter.py)

### T10: Zero DST/timezone transition tests (Risk 5/10)
- **Status:** [x] Fixed (8 tests in test_pipeline/test_timezone_transitions.py)

---

## CLEAN (confirmed correct by review)

- Nested builder/schema: 0 bugs (epoch fix correct)
- Cost model math: to_r_multiple vs pnl_points_to_r correct
- E3 sub-bar fill verification: correct
- Portfolio overlay eligibility guard: correct (FIX5 invariant holds)
- armed_at_bar mechanism: correct (now tested)
- Front contract selection: correct for GC
- Session boundaries: correct for Brisbane UTC+10
- Float comparisons in OHLCV: correct (no epsilon issues)
- DuckDB connection lifecycle: no leaks
- Resume/crash recovery: correct (re-processes last day safely)

---

## R&D TASKS (future work)

### R1: Update backtester for intra-bar/fill-bar granularity (HIGH PRIORITY)
- **Context:** C3 fix made engine match backtester by skipping fill-bar exit check.
  Both now exclude the fill bar from exit logic. This is correct for consistency,
  but the backtester's assumption (no price action on fill bar) is unrealistic.
- **Action:** Update outcome_builder.py to check exits on fill bar using sub-bar
  (1m within 5m) data for accurate simulation. Then re-validate all strategies.
- **Risk:** Trading a "realistic" engine on "fantasy" backtest data if not addressed.
- **Status:** [ ] Not started

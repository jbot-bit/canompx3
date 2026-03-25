# Running TODO List

## Phase 1 Ops Tooling Cleanup (from code review 2026-03-25)

All findings are in user-created Phase 1 CLI scripts, NOT live trading code.
None block deployment. Fix in a dedicated cleanup pass.

### Hardcoded Lane Defs → Shared LANE_REGISTRY
- [ ] `trading_app/log_trade.py` (lines 29-75) — hardcoded LANE_DEFS
- [ ] `trading_app/pre_session_check.py` (lines 274-321) — hardcoded LANE_DEFS
- [ ] `trading_app/sprt_monitor.py` (lines 629-635) — hardcoded LANES
- [ ] `scripts/tools/forward_monitor.py` (lines 31-91) — hardcoded LANES
- [ ] `scripts/tools/slippage_scenario.py` (lines 29-72) — hardcoded LANES
- **Fix:** Extract shared `LANE_REGISTRY` from `prop_profiles.py` that all 5 import

### DuckDB Context Managers + configure_connection
- [ ] `scripts/tools/forward_monitor.py` (line 125) — bare connect, no context manager, no configure_connection
- [ ] `scripts/tools/slippage_scenario.py` (line 75) — bare connect, no context manager, no configure_connection
- [ ] `trading_app/log_trade.py` (lines 107, 154) — two simultaneous connections, no context managers
- **Fix:** Convert all to `with duckdb.connect(...) as con:` + `configure_connection(con)`

### log_trade.py Read+Write Connection Order
- [ ] Close read connection (line 107) before opening write connection (line 154)
- Currently holds both open simultaneously to same GOLD_DB_PATH

### Missing Companion Tests
- [ ] `trading_app/log_trade.py` — no test_log_trade.py (compute_pnl_r unit tests)
- [ ] `trading_app/pre_session_check.py` — no test_pre_session_check.py
- [ ] `trading_app/weekly_review.py` — no test_weekly_review.py
- [ ] `trading_app/sprt_monitor.py` — no test_sprt_monitor.py
- [ ] `scripts/tools/forward_monitor.py` — no test file

### Minor Type Annotation
- [ ] `trading_app/live/bar_aggregator.py` line 60 — `callable` (lowercase) should be `Callable[[str], None] | None`

---

## Phase 2 Blockers (from compliance audit 2026-03-25)

- [ ] Tradeify API access — requires $1K Tradovate live account + $25/mo subscription
- [ ] ASIC / tax professional consultation — before scaling to multiple funded accounts
- [ ] Tradovate rate limit elevation — pending clearance
- [ ] MNQ tbbo slippage pilot — 0/30 live trades with slippage recorded

## Accepted Residual Risks

- Tradovate/ProjectX query_equity returns realized balance only (not net liquidation). Intraday HWM tracker understates DD with open positions. Mitigated by R-based circuit breaker + EOD readings correct.
- HWM init `except Exception` silently disables DD tracking on unexpected errors. Mitigated by R-based circuit breaker still active + warning logged prominently.

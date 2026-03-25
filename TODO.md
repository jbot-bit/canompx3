# Running TODO List

## Phase 1 Ops Tooling Cleanup (from code review 2026-03-25)

### Hardcoded Lane Defs → Shared LANE_REGISTRY
- [x] `trading_app/log_trade.py` — now imports get_lane_registry()
- [x] `trading_app/pre_session_check.py` — now imports get_lane_registry()
- [x] `trading_app/sprt_monitor.py` — now uses _build_lanes() from registry
- [x] `scripts/tools/forward_monitor.py` — now uses _build_lanes() from registry
- [x] `scripts/tools/slippage_scenario.py` — now uses _build_lanes() from registry
- **DONE** (commit 5185144)

### DuckDB Context Managers + configure_connection
- [x] `scripts/tools/forward_monitor.py` — added configure_connection(con)
- [x] `scripts/tools/slippage_scenario.py` — added configure_connection(con)
- [x] `trading_app/log_trade.py` — read connection now uses `with`, closed before write
- **DONE** (commit 5185144)

### log_trade.py Read+Write Connection Order
- [x] Read connection closed (via `with` block) before write connection opens
- **DONE** (commit 5185144)

### Minor Type Annotation
- [x] `trading_app/live/bar_aggregator.py` — `Callable[[str], None] | None`
- **DONE** (commit 5185144)

### Missing Companion Tests
- [ ] `trading_app/log_trade.py` — no test_log_trade.py
- [ ] `trading_app/pre_session_check.py` — no test_pre_session_check.py
- [ ] `trading_app/weekly_review.py` — no test_weekly_review.py
- [ ] `trading_app/sprt_monitor.py` — no test_sprt_monitor.py
- [ ] `scripts/tools/forward_monitor.py` — no test file
- **Status:** These are CLI ops tools, not live trading code. Low priority.

---

## Phase 2 Blockers (from compliance audit 2026-03-25)

- [ ] Tradeify API access — requires $1K Tradovate live account + $25/mo subscription
- [ ] ASIC / tax professional consultation — before scaling to multiple funded accounts
- [ ] Tradovate rate limit elevation — pending clearance
- [ ] MNQ tbbo slippage pilot — 0/30 live trades with slippage recorded

## Accepted Residual Risks

- Tradovate/ProjectX query_equity returns realized balance only (not net liquidation). Intraday HWM tracker understates DD with open positions. Mitigated by R-based circuit breaker + EOD readings correct.
- HWM init `except Exception` silently disables DD tracking on unexpected errors. Mitigated by R-based circuit breaker still active + warning logged prominently.

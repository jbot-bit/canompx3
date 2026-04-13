# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 166

## RALPH AUDIT — Iteration 166
## Date: 2026-04-14
## Infrastructure Gates: drift 102/102 PASS (6 pre-existing advisories); behavioral audit 7/7 PASS; ruff advisory-only (B905 in scripts/research/gc_proxy_validity.py)

---

## Iteration 166 — consistency_tracker uses CAST(entry_time AS DATE) instead of canonical trading_day column

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | `consistency_tracker.py:111,213,349` — `CAST(entry_time AS DATE)` used for trade-day grouping instead of the stored `trading_day` column. UTC-cast date differs from Brisbane trading day for trades near midnight UTC (00:00–10:00 UTC = 10:00–20:00 Brisbane). `paper_trades.trading_day DATE NOT NULL` is always populated by the write path. | LOW | FIXED 03238c01 |

### Audit Notes

- **Finding source:** Seven Sins scan of `consistency_tracker.py` (Priority 3 unscanned medium target). Also scanned `risk_manager.py` — clean, no findings.
- **TRACE:** `consistency_tracker.py:111` → `CAST(entry_time AS DATE)` → UTC date, not Brisbane trading day. `paper_trades` schema (`db_manager.py:341`) has `trading_day DATE NOT NULL`. Write path always sets `trading_day` from the session's Brisbane trading day.
- **EVIDENCE:** grep confirmed 3 CAST occurrences. `db_manager.py:341` confirms `trading_day DATE NOT NULL`. For a trade at 23:30 UTC (09:30 Brisbane next day), UTC-cast date would attribute the trade to the prior Brisbane day — wrong grouping for the consistency/payout window.
- **Fix:** Replaced `CAST(entry_time AS DATE) AS trade_date` with `trading_day AS trade_date` (lines 111, 213) and `MAX(CAST(entry_time AS DATE))` with `MAX(trading_day)` (line 349). Updated one minimal test fixture (`test_consistency_no_trades`) to include `trading_day DATE` column.
- **Behavior unchanged for current data:** All test trades use same-day UTC timestamps (08:00 UTC), so CAST result == trading_day. 11/11 tests pass. 102/102 drift checks pass.

### Full Seven Sins scan — consistency_tracker.py

| Sin | Result |
|-----|--------|
| Silent failure | None — DB errors propagate; `_has_paper_trades` guards empty table gracefully |
| Fail-open | None — returns None / IDLE_BREACH on missing data (fail-closed for callers) |
| Look-ahead bias | N/A — read-only compliance tracker |
| Cost illusion | N/A — no P&L computation |
| Canonical violation | FIXED — `CAST(entry_time AS DATE)` replaced with `trading_day` (CT-166) |
| Orphan risk | None — all functions are wired to callers |
| Volatile data | None — thresholds come from canonical `prop_firm_policies` |
| Async safety | N/A — synchronous module |
| State persistence gap | N/A — stateless read-only module |
| Contract drift | None — all callers pass correct kwargs |

### Full Seven Sins scan — risk_manager.py (clean)

| Sin | Result |
|-----|--------|
| Silent failure | None — no exception handlers; all paths return explicit results |
| Fail-open | None — all checks return (False, reason, 0.0) on unknown/halted state |
| Look-ahead bias | N/A — live risk enforcement |
| Cost illusion | N/A — no P&L computation |
| Canonical violation | None — F-1/F-2 @canonical-source annotations correct |
| Orphan risk | None |
| Volatile data | None — limits runtime-constructed by orchestrator |
| Async safety | N/A — synchronous; no async callers |
| State persistence gap | ACCEPTABLE — multi-day equity state in-memory, documented "Lost on process restart"; Layer 2 AccountHWMTracker provides persistent protection (guarded by verified upstream check at session_orchestrator:403) |
| Contract drift | None — all 3 can_enter call sites in execution_engine use correct kwargs |

---

## Prior: Iteration 165 — Hardcoded holdout date in sprt_monitor + sr_monitor

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | `sprt_monitor.py:129` — `start_date=date(2026, 1, 1)` hardcoded instead of `HOLDOUT_SACRED_FROM` | LOW | FIXED 3898432c |
| Canonical violation | `sr_monitor.py:117` — same hardcoded literal | LOW | FIXED 3898432c |

---

## Prior: Iteration 164 — Stale docstring in _arm_strategies (execution_engine.py)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Orphan risk | `execution_engine.py:620` — stale docstring | LOW | FIXED cccd21a6 |

---

## Prior: Iteration 163 — check_lane_lifecycle fail-open on exception

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open | `pre_session_check.py:314` — returned (True, "WARN") on exception | MEDIUM | FIXED 4dc4a35c |

---

## Files Fully Scanned

> Cumulative list — 247 files fully scanned (consistency_tracker.py + risk_manager.py added iter 166).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
- trading_app/ml/config.py — added iter 129
- trading_app/ml/meta_label.py — added iter 130
- trading_app/ml/predict_live.py — added iter 131
- trading_app/walkforward.py — added iter 132
- trading_app/outcome_builder.py — added iter 115
- trading_app/strategy_discovery.py — added iter 116
- trading_app/strategy_validator.py — added iter 117
- trading_app/portfolio.py — added iter 118
- trading_app/live_config.py — added iter 119
- trading_app/db_manager.py — added iter 120
- trading_app/live/projectx/auth.py — added iter 121
- trading_app/live/projectx/order_router.py — added iter 122
- trading_app/live/projectx/data_feed.py — added iter 123
- trading_app/live/broker_base.py — added iter 123
- trading_app/live/tradovate/data_feed.py — added iter 124
- trading_app/live/session_orchestrator.py — added iter 124
- trading_app/live/bar_aggregator.py — added iter 125
- trading_app/live/tradovate/order_router.py — added iter 126
- trading_app/live/tradovate/auth.py — added iter 126
- trading_app/live/tradovate/contract_resolver.py — added iter 126
- trading_app/live/tradovate/positions.py — added iter 126
- trading_app/live/broker_factory.py — added iter 127
- trading_app/live/tradovate/__init__.py — added iter 127
- trading_app/live/circuit_breaker.py — added iter 128
- trading_app/live/cusum_monitor.py — added iter 128
- trading_app/live/projectx/__init__.py — added iter 128
- pipeline/ — 15 files (iters 1-71)
- pipeline/calendar_filters.py — added iter 133
- pipeline/stats.py — added iter 134
- pipeline/audit_log.py — added iter 136
- pipeline/ingest_dbn_mgc.py — added iter 136
- pipeline/ingest_dbn.py — added iter 137
- pipeline/ingest_dbn_daily.py — added iter 137
- pipeline/build_daily_features.py — added iter 138
- pipeline/build_bars_5m.py — added iter 139
- pipeline/run_pipeline.py — added iter 140
- pipeline/run_full_pipeline.py — added iter 140
- scripts/tools/ — 51 files (iters 18-100)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iters 85, 87)
- scripts/ root — 2 files (iter 88)
- scripts/databento_backfill.py — added iter 135
- research/ — 21 files (iters 101-113)
- docs/plans/ — 2 files (iter 103)
- trading_app/live/rithmic/order_router.py — added iter 141
- trading_app/prop_profiles.py — added iter 142
- trading_app/lane_allocator.py — added iter 143
- trading_app/live/multi_runner.py — added iter 144
- trading_app/live/broker_dispatcher.py — added iter 145
- trading_app/pre_session_check.py — added iter 146, re-audited iter 163
- trading_app/live/copy_order_router.py — added iter 147
- trading_app/live/rithmic/auth.py — added iter 148
- trading_app/live/bot_dashboard.py — added iter 149
- trading_app/live/position_tracker.py — added iter 150
- trading_app/account_hwm_tracker.py — added iter 151
- trading_app/live/trade_journal.py — added iter 152
- trading_app/live/bot_state.py — added iter 153
- trading_app/prop_portfolio.py — added iter 154
- trading_app/live/rithmic/__init__.py — added iter 155
- trading_app/ai/sql_adapter.py — added iter 155
- trading_app/ai/grounding.py — added iter 156
- trading_app/ai/query_agent.py — added iter 157
- trading_app/ai/chat_handler.py — added iter 157
- trading_app/mcp_server.py — added iter 158
- trading_app/ai/__init__.py — added iter 159
- trading_app/ai/corpus.py — added iter 159
- trading_app/ai/cli.py — added iter 159
- trading_app/ai/strategy_matcher.py — added iter 160
- trading_app/live/webhook_server.py — added iter 160
- trading_app/live/instance_lock.py — added iter 160
- trading_app/live/broker_connections.py — added iter 160
- trading_app/live/tradovate/contracts.py — added iter 161
- trading_app/live/tradovate/http.py — added iter 161
- trading_app/live/rithmic/contracts.py — added iter 161
- trading_app/live/rithmic/positions.py — added iter 161
- pipeline/db_config.py — added iter 161
- pipeline/paths.py — re-audited iter 161 (modified 2026-04-04)
- trading_app/execution_engine.py — added iter 164
- trading_app/entry_rules.py — added iter 164
- trading_app/sprt_monitor.py — added iter 165
- trading_app/sr_monitor.py — added iter 165
- trading_app/consistency_tracker.py — added iter 166
- trading_app/risk_manager.py — added iter 166
- **Total: 247 files fully scanned**

## Next iteration targets
- Priority 2 (stale re-audit candidates): trading_app/prop_profiles.py (modified post iter-142 during profit expansion Apr 12-13), trading_app/strategy_fitness.py (modified post iter-44)
- Priority 3 (unscanned medium): trading_app/topstep_scaling_plan.py, trading_app/lane_correlation.py
- Note: pre-existing drift advisories (checks 59/95) require operational resolution by user — run `python scripts/tools/select_family_rr.py` and re-run validator for Mode A strategies

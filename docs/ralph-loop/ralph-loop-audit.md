# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 165

## RALPH AUDIT — Iteration 165
## Date: 2026-04-10
## Infrastructure Gates: drift 90/90 PASS (6 pre-existing data violations: checks 59/95 — family_rr_locks + validated_setups lanes, from Mode A Phase 5 pending); behavioral audit 7/7 PASS; ruff advisory-only (B905 in scripts/research/gc_proxy_validity.py)

---

## Iteration 165 — Hardcoded holdout date in sprt_monitor + sr_monitor

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | `sprt_monitor.py:129` — `start_date=date(2026, 1, 1)` hardcoded instead of `HOLDOUT_SACRED_FROM` from `trading_app.holdout_policy` | LOW | FIXED 3898432c |
| Canonical violation | `sr_monitor.py:117` — same hardcoded literal for the OOS start date in canonical forward outcome fallback | LOW | FIXED 3898432c |

### Audit Notes

- **Finding source:** Seven Sins scan of stale-re-audit targets (strategy_fitness.py, strategy_validator.py, rolling_portfolio.py). The stale "dead arg" targets from the iter 164 next-targets note were already resolved in commit 713c4ab7 (Apr 9). Fresh scan of today's modified files (account_survival.py, sprt_monitor.py via strategy_fitness commit) revealed the holdout literal.
- **TRACE:** `sprt_monitor.py:129` → `start_date=date(2026, 1, 1)` → `trading_app.holdout_policy.HOLDOUT_SACRED_FROM = date(2026, 1, 1)`. Same for `sr_monitor.py:117`. No caller passes this value — both files use it as the forward-OOS window start for Criterion 12 monitoring. If boundary changes, monitors silently stay frozen.
- **EVIDENCE:** grep confirmed both literals. `holdout_policy.HOLDOUT_SACRED_FROM = date(2026, 1, 1)` at line 70 of holdout_policy.py.
- **Fix:** Added `from trading_app.holdout_policy import HOLDOUT_SACRED_FROM` to both files, replaced literals with canonical reference.
- **Behavior unchanged:** Value identical at runtime. 11/11 tests pass.
- **Also scanned (Seven Sins):** account_survival.py (209 lines added today) — clean. strategy_validator.py (2-line change today) — rejection_reason field addition correct; schema column exists at db_manager.py:597.

### Full Seven Sins scan — sprt_monitor.py + sr_monitor.py

| Sin | Result |
|-----|--------|
| Silent failure | None — both monitors log and continue on lane lookup failures |
| Fail-open | None — both monitors report NO_DATA status on empty trade streams |
| Look-ahead bias | None — start_date filter correctly restricts to forward OOS window |
| Cost illusion | N/A — monitoring only, no P&L computation |
| Canonical violation | FIXED — holdout date literal (SF-165) |
| Orphan risk | None |
| Volatile data | None — strategy counts dynamically loaded from validated_setups |
| Async safety | N/A — synchronous modules |
| State persistence gap | N/A — sprt writes state file on every run; sr writes envelope |
| Contract drift | None — _load_strategy_outcomes callers all use correct kwargs |

### account_survival.py scan — clean (today's big change)

- `_load_lane_trade_paths` correctly uses `entry_ts`/`exit_ts` from updated `_load_strategy_outcomes` SQL (added today in commit 0f7903c7)
- `_scenario_from_trade_paths` intraday replay logic: correct event ordering with UTC-aware sentinels
- `simulate_survival` Monte Carlo: fail-closed on empty scenarios; boundary checks on horizon_days and n_paths
- `check_survival_report_gate`: all 7 blocking conditions fail-closed; reads path_model field correctly
- No hardcoded instrument lists, no hardcoded constants for DD limits (uses `get_account_tier` + `get_firm_spec`)

---

## Prior: Iteration 164 — Stale docstring in _arm_strategies (execution_engine.py)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Orphan risk | `execution_engine.py:620` — `_arm_strategies` docstring claimed "Phase 2 passes `{"E1"}`" but Phase 2 at line 493 passes `entry_models=None`; misleading comment could cause maintainer confusion | LOW | FIXED cccd21a6 |

---

## Prior: Iteration 163 — check_lane_lifecycle fail-open on exception

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open | `pre_session_check.py:314` — `check_lane_lifecycle()` returned `(True, "WARN: ...")` on exception, permitting lane to trade when lifecycle state was unreadable | MEDIUM | FIXED 4dc4a35c |

---

## Prior: Iteration 162 — Dead code sweep: unused @patch mock parameters in test files

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Dead code / orphan | `test_tradovate.py:160` — `mock_sleep` param unused | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_tradovate.py:459` — `mock_sleep` param unused | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_tradovate.py:619-621` — `@patch` decorator unnecessary + `mock_post` param unused | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_projectx_429_retry.py:58,75,89,104,118,133,145` — 7x `mock_sleep` param unused | LOW | FIXED 24b30b6 |

---

## Files Fully Scanned

> Cumulative list — 245 files fully scanned (sprt_monitor.py + sr_monitor.py added iter 165).

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
- **Total: 245 files fully scanned**

## Next iteration targets
- Priority 3 (unscanned medium): trading_app/consistency_tracker.py, trading_app/risk_manager.py
- Priority 2 (stale re-audit candidates): trading_app/account_survival.py (scanned iter 165, clean), trading_app/strategy_fitness.py (stale dead-arg already fixed in 713c4ab7)
- Note: pre-existing drift violations (checks 59/95) require operational resolution by user — run `python scripts/tools/select_family_rr.py` and re-run validator for Mode A strategies

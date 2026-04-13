# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 166

## RALPH AUDIT — Iteration 166
## Date: 2026-04-13
## Infrastructure Gates: drift 103/103 code checks PASS (advisories: checks 46/47/49/59/73 pre-existing; check 103 pre-existing subprocess CLI error); behavioral audit 7/7 PASS; ruff advisory-only (pre-existing)

---

## Iteration 166 — RiskManager warnings silently discarded

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Orphan risk | `risk_manager.py:322-331` — `_warnings` list appended to in `can_enter()` (drawdown_warning + chop_warning) but never logged; silently cleared on `daily_reset()` — zero operational visibility | LOW | FIXED 2c3eff2f |

### Audit Notes

- **Finding source:** Seven Sins scan of `trading_app/risk_manager.py` and `trading_app/consistency_tracker.py` (Priority 3 medium-centrality targets).
- **TRACE:** `risk_manager.py:322-331` → `self._warnings.append(msg)` → `session_orchestrator.py` has zero calls to `.warnings` (grep confirmed) → `daily_reset():98` clears `self._warnings` → warnings silently lost. Three call sites in execution_engine.py all omit the `market_state` parameter, so chop_warning path was doubly dormant.
- **EVIDENCE:** grep of session_orchestrator.py + execution_engine.py returned 0 matches for `risk_mgr.warnings` / `.warnings`.
- **Fix:** Added `import logging`, module-level `log = logging.getLogger(__name__)`, and `log.warning()` calls at both append sites. Warning list still maintained for callers.
- **Behavior unchanged:** No logic change. 105/105 tests pass.
- **Also assessed (ACCEPTABLE):**
  - `market_state` parameter in `can_enter()` never passed from production callers — dormant infrastructure, tested by test_market_state.py. ACCEPTABLE #2.
  - `RiskManager.cumulative_pnl_r`/`equity_high_water_r` in-memory only — documented intentional design at session_orchestrator.py:389-390, guarded by Layer 2 AccountHWMTracker. ACCEPTABLE #1/#4.
  - `date.today()` in consistency_tracker.py — consistent with pattern across 5+ files in codebase (account_survival, derived_state, lane_allocator). No finding.
  - `check_payout_eligibility` never returns None but weekly_review checks `if pe is not None` — dead guard inside try/except, style-only. ACCEPTABLE #3.

### Full Seven Sins scan — risk_manager.py

| Sin | Result |
|-----|--------|
| Silent failure | None — all exception paths in F-1 block return fail-closed with reason |
| Fail-open | None — all checks fail-closed |
| Look-ahead bias | N/A |
| Cost illusion | N/A |
| Canonical violation | None — no hardcoded instrument lists or magic constants |
| Orphan risk | FIXED — _warnings silently discarded (RM-166) |
| Volatile data | None — no hardcoded counts |
| Async safety | N/A — synchronous module |
| State persistence gap | ACCEPTABLE — in-memory equity tracking documented as intentional (Layer 2 guards same concern) |
| Contract drift | None — all three can_enter() call sites use correct kwargs |

### Full Seven Sins scan — consistency_tracker.py

| Sin | Result |
|-----|--------|
| Silent failure | None — DB errors propagate; pre_session_check wraps fail-closed |
| Fail-open | None |
| Look-ahead bias | N/A |
| Cost illusion | N/A |
| Canonical violation | None — uses GOLD_DB_PATH, ACTIVE_ORB_INSTRUMENTS, PROP_FIRM_SPECS canonically |
| Orphan risk | None — check_microscalp_compliance used for all active instruments in weekly_review |
| Volatile data | None |
| Async safety | N/A — synchronous module |
| State persistence gap | N/A — read-only trackers |
| Contract drift | ACCEPTABLE — check_payout_eligibility never returns None but caller checks None (dead guard, no correctness impact) |

---

## Prior: Iteration 165 — Hardcoded holdout date in sprt_monitor + sr_monitor

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | `sprt_monitor.py:129` — `start_date=date(2026, 1, 1)` hardcoded instead of `HOLDOUT_SACRED_FROM` | LOW | FIXED 3898432c |
| Canonical violation | `sr_monitor.py:117` — same hardcoded literal for OOS start date | LOW | FIXED 3898432c |

---

## Prior: Iteration 164 — Stale docstring in _arm_strategies (execution_engine.py)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Orphan risk | `execution_engine.py:620` — `_arm_strategies` docstring claimed "Phase 2 passes `{"E1"}`" but Phase 2 at line 493 passes `entry_models=None`; misleading comment | LOW | FIXED cccd21a6 |

---

## Prior: Iteration 163 — check_lane_lifecycle fail-open on exception

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open | `pre_session_check.py:314` — `check_lane_lifecycle()` returned `(True, "WARN: ...")` on exception | MEDIUM | FIXED 4dc4a35c |

---

## Files Fully Scanned

> Cumulative list — 247 files fully scanned (risk_manager.py + consistency_tracker.py added iter 166).

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
- trading_app/risk_manager.py — added iter 166
- trading_app/consistency_tracker.py — added iter 166
- **Total: 247 files fully scanned**

## Next iteration targets
- Priority 2 (stale re-audit): trading_app/prop_profiles.py (scanned iter 142, heavily modified since — L6/L7 profit expansion, lane registry fix, 2026-04-12 changes)
- Priority 3 (unscanned medium): trading_app/rolling_portfolio.py (2 importers, medium centrality), trading_app/lane_correlation.py (recently added, unscanned)
- Note: consecutive_low_only will be 3 after ledger rebuild; HIGH+ last at iter 163; no diminishing returns trigger yet (threshold is 5)

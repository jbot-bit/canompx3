# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 167

## RALPH AUDIT — Iteration 167
## Date: 2026-04-18
## Infrastructure Gates: drift 101/101 PASS (5 pre-existing violations: anthropic module absent + 3 stale SGP windows; 6 pre-existing advisories); behavioral audit 7/7 PASS; ruff advisory-only (pre-existing)

---

## Iteration 167 — self_funded_tradovate cap conflict in get_lane_registry (DEFERRED)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Contract drift | `prop_profiles.py:864-875` — `self_funded_tradovate` has EUROPE_FLOW (MNQ cap=150.0, MGC cap=30.0) and NYSE_OPEN (MNQ cap=150.0, MES cap=60.0). `get_lane_registry` raises `ValueError` on inconsistent per-session caps across instruments. Currently unreachable: profile is `active=False` AND `resolve_profile_id` default `exclude_self_funded=True` blocks all callers. Dormant but will fail at activation time without architectural fix. | MEDIUM | DEFERRED — dormant profile, blocked by exclude_self_funded guard (PP-167) |

### Audit Notes

- **Finding source:** Re-audit of `prop_profiles.py` (stale Priority 2 target, modified 2026-04-16 post iter-142 scan).
- **TRACE:** `prop_profiles.py:864-875` (MGC_EUROPE_FLOW cap=30.0, MES_NYSE_OPEN cap=60.0) → `get_lane_registry:1089-1098` (cap mismatch logic) → would raise `ValueError`. Blocked by `resolve_profile_id:941` (`exclude_self_funded=True` default) in all 7 callers of `get_lane_registry`.
- **EVIDENCE:** Simulation confirmed EUROPE_FLOW=[30.0, 150.0] and NYSE_OPEN=[60.0, 150.0] cap conflicts. Grep of all `get_lane_registry` callers (session_orchestrator, forward_monitor, slippage_scenario) confirmed none pass `exclude_self_funded=False`. Profile `active=False`. 61/61 existing tests pass.
- **Deferred reason:** Dormant infrastructure. No caller can reach the cap-conflict logic today. Profile is both `active=False` and guarded by `exclude_self_funded=True`. Needs architectural fix to `get_lane_registry` to support per-(session, instrument) keying before `self_funded_tradovate` goes live.
- **Action required at activation:** `get_lane_registry` must be updated to key by `(orb_label, instrument)` for multi-instrument profiles, or `self_funded_tradovate` must be split into per-instrument sub-profiles.

### Full Seven Sins scan — prop_profiles.py

| Sin | Result |
|-----|--------|
| Silent failure | None — `load_allocation_lanes` fail-closed (returns empty tuple on any error). `validate_dd_budget` returns list, does not swallow. |
| Fail-open | None — `resolve_profile_id` raises on ambiguity/inactive/self-funded. `get_lane_registry` raises on cap conflict. |
| Look-ahead bias | N/A — configuration module |
| Cost illusion | None — `validate_dd_budget` imports from `pipeline.cost_model.COST_SPECS` and cross-checks `_PV` |
| Canonical violation | None — `ENTRY_MODELS` imported from `trading_app.config`. `_P90_ORB_PTS` annotated as empirical (ACCEPTABLE: intentional heuristic with update note). |
| Orphan risk | None — all helpers wired to callers |
| Volatile data | None — `_P90_ORB_PTS` annotated empirical, not a hard-coded claim |
| Async safety | N/A — synchronous module |
| State persistence gap | N/A — pure config module, no mutable state |
| Contract drift | DEFERRED — `self_funded_tradovate` multi-instrument cap conflict in `get_lane_registry` (PP-167) |

---

## Prior: Iteration 166 — consistency_tracker uses CAST(entry_time AS DATE) instead of canonical trading_day column

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | `consistency_tracker.py:111,213,349` — `CAST(entry_time AS DATE)` used for trade-day grouping instead of the stored `trading_day` column. | LOW | FIXED 03238c01 |

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

> Cumulative list — 248 files fully scanned (prop_profiles.py re-audited iter 167; strategy_fitness.py re-audited iter 167 — WIP-save only, no substantive change).

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
- trading_app/prop_profiles.py — added iter 142, re-audited iter 167
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
- trading_app/strategy_fitness.py — re-audited iter 167 (WIP-save only, no substantive change)
- **Total: 248 files fully scanned**

## Next iteration targets
- Priority 3 (unscanned medium): trading_app/topstep_scaling_plan.py, trading_app/lane_correlation.py
- Note: pre-existing drift advisories (checks 59/95) require operational resolution by user — run `python scripts/tools/select_family_rr.py` and re-run validator for Mode A strategies

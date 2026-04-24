# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 169

## RALPH AUDIT — Iteration 169
## Date: 2026-04-25
## Infrastructure Gates: drift 107/107 PASS (6 pre-existing advisories); behavioral audit 7/7 PASS; ruff PASS (all clean)

---

## Iteration 169 — db_manager.py verify_trading_app_schema silent verifier gap (FIXED 6811640a)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open / Silent failure | `verify_trading_app_schema` `expected_cols` for `validated_setups` missing 10 migration-added columns; `experimental_strategies` missing 2. Returned `(True, [])` silently even when those columns absent. | MEDIUM | FIXED 6811640a |

### Audit Notes

- **Auto-targeting:** Priority 2 (stale re-audit, critical) — `trading_app/db_manager.py` last scanned iter 120 (2026-03-16), modified 2026-04-16. Critical tier (13 importers).
- **Doctrine cited:** integrity-guardian.md § 3 (fail-closed — never return success in audit/health paths when gaps exist), § 5 (evidence over assertion — verifier must actually verify all schema columns).
- **TRACE:** `db_manager.py:957-963` — `missing = expected_cols - actual_cols` → columns not in `expected_cols` cannot appear in `missing` → `violations` stays empty → returns `(True, [])` even when 10 migration-added columns absent from DB.
- **EVIDENCE:** Lines 613-692 add `discovery_k`, `discovery_date`, `era_dependent`, `max_year_pct`, `wfe_verdict`, `wfe_investigation_date`, `wfe_investigation_notes`, `slippage_validation_status`, `validation_pathway`, `c8_oos_status` to `validated_setups` via ALTER TABLE. None of these appeared in the `expected_cols` set at lines 883-961. For `experimental_strategies`, `validation_pathway` and `c8_oos_status` (lines 687-692) also missing from its expected_cols (lines 814-875). Fix: added all 12 missing column names to the two expected_cols sets.
- **Verification:** 13 tests in test_db_manager.py passed. 107/107 drift checks PASS. 324 fast tests passed in pre-commit hook.
- **Blast radius:** 2 files (db_manager.py, line-ending normalization only in pre-commit for test file). No API/signature change.
- **History note:** Iter 120 had the same class of bug for `experimental_strategies` (9 columns added then). This iteration closes the same gap that accumulated since iter 120 across subsequent migrations.

### Full Seven Sins scan — db_manager.py

| Sin | Result |
|-----|--------|
| Silent failure | FIXED — `verify_trading_app_schema` now covers all 12 previously-missing columns |
| Fail-open | None — `init_trading_app_schema` uses `with duckdb.connect()` (auto-close). Migrations use narrow `CatalogException` catch (column already exists). Force-drop is guarded by explicit `force=True` flag. |
| Look-ahead bias | N/A — schema manager, no temporal query logic |
| Cost illusion | N/A — no P&L computation |
| Canonical violation | None — `ACTIVE_ORB_INSTRUMENTS` imported from `pipeline.asset_configs` (line 393). `GOLD_DB_PATH` from `pipeline.paths`. `FAMILY_RR_LOCKS_SCHEMA` from `pipeline.init_db`. All canonical. |
| Orphan risk | None — `compute_trade_day_hash`, `get_family_head_ids`, `has_edge_families` all have identifiable callers. |
| Volatile data | None — schema strings are literals, not research-derived values. `@research-source` annotation present on `n_trials_at_discovery` (line 537). |
| Async safety | N/A — synchronous module |
| State persistence gap | N/A — DB-backed state, not in-memory |
| Contract drift | None — `verify_trading_app_schema` and `init_trading_app_schema` signatures unchanged |

---

## Prior: Iteration 168 — topstep_scaling_plan.py (audit-only — all findings ACCEPTABLE)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Orphan risk | `scripts/tmp/lane_analysis.py:20` imports `SCALING_LADDER` (nonexistent — correct name is `SCALING_PLAN_LADDER`). Caught by try/except; investigative script only. | LOW | ACCEPTABLE — dormant, guarded by try/except |
| Contract drift | `lots_for_position()` misclassifies M2K and M6E as mini (1:1) instead of micro (10:1) because `instrument[1].isalpha()` excludes symbols with digit at position 1. Both instruments are dead for ORB per CLAUDE.md. All active instruments (MES, MGC, MNQ) classified correctly. | LOW | ACCEPTABLE — dormant, dead instruments only |

---

## Prior: Iteration 167 — self_funded_tradovate cap conflict in get_lane_registry (FIXED LOCAL 2026-04-23)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Contract drift | `prop_profiles.py` previously keyed ORB-cap registry entries by session only, which falsely treated multi-instrument sessions as conflicting. `get_lane_registry()` now keys by `(orb_label, instrument)` and the session_orchestrator plus the remaining registry consumers were updated to match. `self_funded_tradovate` remains inactive for separate readiness reasons. | MEDIUM | FIXED LOCAL 2026-04-23 |

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

## Files Fully Scanned

> Cumulative list — 249 files fully scanned (db_manager.py re-audited iter 169).

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
- trading_app/db_manager.py — added iter 120, re-audited iter 169
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
- trading_app/topstep_scaling_plan.py — added iter 168
- **Total: 249 files fully scanned**

## Next iteration targets
- Priority 2 (stale re-audit, critical): trading_app/outcome_builder.py — last scanned iter 115 (2026-03-16), E2 canonical window fix (0c56c7f7) and additional commits landed after. Critical tier (12 importers).
- Priority 3 (unscanned medium): trading_app/lane_correlation.py
- Note: pre-existing drift advisories (checks 59/95) require operational resolution by user — run `python scripts/tools/select_family_rr.py` and re-run validator for Mode A strategies

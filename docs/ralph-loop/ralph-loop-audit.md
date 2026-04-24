# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 170

## RALPH AUDIT — Iteration 170
## Date: 2026-04-25
## Infrastructure Gates: drift 107/107 PASS (6 pre-existing advisories); behavioral audit 7/7 PASS; ruff PASS; 223 tests PASS

---

## Iteration 170 — outcome_builder.py dead parameter break_ts in _compute_outcomes_all_rr (FIXED 9b16c4eb)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Dead parameter (institutional-rigor.md § 5) | `_compute_outcomes_all_rr` accepted `break_ts=None` in its signature but never read it in the function body (219 lines, 0 reads). Two production callsites in `build_outcomes` and two test callsites in `test_stress_hardcore.py` passed `break_ts=break_ts,` unnecessarily — pure drift bait. | LOW | FIXED 9b16c4eb |

### Audit Notes

- **Auto-targeting:** Priority 2 (stale re-audit, critical) — `trading_app/outcome_builder.py` last scanned iter 115 (2026-03-16), E2 canonical window fix (0c56c7f7) and additional commits landed after. Critical tier (12 importers).
- **Doctrine cited:** institutional-rigor.md § 5 (no dead parameters — drift bait).
- **PREMISE:** `break_ts` is accepted by `_compute_outcomes_all_rr` but never used in the body, causing callers to pass a value that has no effect — silent contract mismatch.
- **TRACE:** `outcome_builder.py:218` (signature `break_ts=None`) → body lines 219-420 (zero reads of `break_ts`) → callers `build_outcomes:877, 936` pass `break_ts=break_ts,` → `test_stress_hardcore.py:609, 641` pass `break_ts=break_ts,`.
- **EVIDENCE:** `grep -n "break_ts" outcome_builder.py` returns 8 hits — all in signature, callers, or F-03 audit comment. Zero hits inside function body between sig and closing `return`.
- **Fix:** Removed `break_ts=None,` from signature and `break_ts=break_ts,` from all 4 callsites (2 production, 2 test). 5 lines net changed.
- **Canonical context:** `break_ts` was the pre-E2 fallback for ORB window end. After `0c56c7f7` (E2 canonical window fix), `orb_utc_window()` is the sole authority — `break_ts` fallback was explicitly removed as look-ahead bias. The dead parameter was a residue of that migration.
- **Verification:** 223 tests pass, 107/107 drift checks PASS. All 8 pre-commit hooks PASS.
- **Blast radius:** 2 files (outcome_builder.py + test_stress_hardcore.py). No callers outside these two files call `_compute_outcomes_all_rr` (private function, confirmed by grep).

### Full Seven Sins scan — outcome_builder.py

| Sin | Result |
|-----|--------|
| Silent failure | None — errors propagate via ValueError/RuntimeError. No bare except. |
| Fail-open | None — `orb_utc_window()` raises ValueError if canonical inputs missing (no break_ts fallback). |
| Look-ahead bias | None — E2 canonical window uses `orb_utc_window(trading_day, orb_label, orb_minutes)` (line 845, 499). No forward-look. |
| Cost illusion | None — cost_spec injected from COST_SPECS per instrument; not hardcoded. |
| Canonical violation | None — ENTRY_MODELS/SKIP_ENTRY_MODELS from trading_app.config; no hardcoded instruments/sessions/dates. |
| Dead parameter | FIXED — break_ts removed from signature and all 4 callsites. |
| Orphan risk | None — _compute_outcomes_all_rr and compute_single_outcome have verified callers. |
| Async safety | N/A — synchronous module |
| State persistence gap | N/A — stateless computation module |
| Contract drift | FIXED — signature now matches all callsites. |

---

## Prior: Iteration 169 — db_manager.py verify_trading_app_schema silent verifier gap (FIXED 6811640a)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open / Silent failure | `verify_trading_app_schema` `expected_cols` for `validated_setups` missing 10 migration-added columns; `experimental_strategies` missing 2. Returned `(True, [])` silently even when those columns absent. | MEDIUM | FIXED 6811640a |

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

> Cumulative list — 250 files fully scanned (outcome_builder.py re-audited iter 170).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
- trading_app/ml/config.py — added iter 129
- trading_app/ml/meta_label.py — added iter 130
- trading_app/ml/predict_live.py — added iter 131
- trading_app/walkforward.py — added iter 132
- trading_app/outcome_builder.py — added iter 115, re-audited iter 170
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
- **Total: 250 files fully scanned**

## Next iteration targets
- Priority 3 (unscanned medium): trading_app/lane_correlation.py
- Priority 3 (unscanned medium): trading_app/eligibility/ (if not yet scanned — check centrality)
- Note: pre-existing drift advisories (checks 59/95) require operational resolution by user — run `python scripts/tools/select_family_rr.py` and re-run validator for Mode A strategies

# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 168

## RALPH AUDIT — Iteration 168
## Date: 2026-04-18
## Infrastructure Gates: drift 101/101 PASS (5 pre-existing violations: anthropic module absent + 3 stale SGP windows; 6 pre-existing advisories); behavioral audit 7/7 PASS; ruff advisory-only (pre-existing)

---

## Iteration 168 — topstep_scaling_plan.py (audit-only — all findings ACCEPTABLE)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Orphan risk | `scripts/tmp/lane_analysis.py:20` imports `SCALING_LADDER` (nonexistent — correct name is `SCALING_PLAN_LADDER`). Caught by try/except; investigative script only. | LOW | ACCEPTABLE — dormant, guarded by try/except |
| Contract drift | `lots_for_position()` misclassifies M2K and M6E as mini (1:1) instead of micro (10:1) because `instrument[1].isalpha()` excludes symbols with digit at position 1. Both instruments are dead for ORB per CLAUDE.md. All active instruments (MES, MGC, MNQ) classified correctly. | LOW | ACCEPTABLE — dormant, dead instruments only |

### Audit Notes

- **Auto-targeting:** Priority 1 = none (all critical/high files scanned). Priority 2 check: 2026-04-16 WIP-save commit was a line-ending normalization only (added=removed=1665 lines in build_daily_features.py), not substantive — scans remain valid. Priority 3 = topstep_scaling_plan.py (unscanned medium).
- **Doctrine cited:** integrity-guardian.md § 2 (canonical sources), § 3 (fail-closed), § 5 (evidence over assertion); institutional-rigor.md § 4 (delegate to canonical sources), § 5 (no dead code), § 8 (verify before claiming).
- **TRACE for Orphan risk:** `scripts/tmp/lane_analysis.py:20` → `from trading_app.topstep_scaling_plan import SCALING_LADDER` → `ImportError` (confirmed by execution) → `except Exception as e: print(...)` catches silently.
- **TRACE for Contract drift:** `lots_for_position('M2K', 10)` → `instrument[1].isalpha()` = `'2'.isalpha()` = False → `return contracts` (returns 10 as 10 minis, should be `micros_to_mini_equivalent(10)` = 1 mini). Confirmed: M2K and M6E both dead for ORB (CLAUDE.md). Active instruments MES/MGC/MNQ all have alpha at position 1 → classified correctly.
- **EVIDENCE:** All 49 test_topstep_scaling_plan.py pass. Python execution confirmed both bugs and that active instruments are unaffected.
- **@future-followup noted:** NET vs GROSS position calculation (line 211). GROSS is conservative and documented.
- **F-1 caller (risk_manager.py:231-268):** Properly wired — fail-closed when EOD balance unknown, narrow (KeyError, ValueError) exception handling, no swallowed errors.

### Full Seven Sins scan — topstep_scaling_plan.py

| Sin | Result |
|-----|--------|
| Silent failure | None — all error paths raise; no except-pass patterns |
| Fail-open | None — max_lots_for_xfa raises KeyError/ValueError on bad input; risk_manager caller is fail-closed |
| Look-ahead bias | N/A — no DB queries, no temporal logic |
| Cost illusion | N/A — no P&L computation |
| Canonical violation | None — SCALING_PLAN_LADDER values from @canonical-source annotated artifacts (TopStep policy, not pipeline canonical data). No hardcoded instrument/session/entry model lists. |
| Orphan risk | ACCEPTABLE — scripts/tmp/lane_analysis.py:20 uses wrong constant name (SCALING_LADDER vs SCALING_PLAN_LADDER); investigative script, guarded by try/except |
| Volatile data | None — @canonical-source annotations present with quarterly re-verify note |
| Async safety | N/A — synchronous module |
| State persistence gap | N/A — no mutable state |
| Contract drift | ACCEPTABLE — lots_for_position misclassifies M2K and M6E (dead for ORB); all active instruments correct |

---

## Prior: Iteration 167 — self_funded_tradovate cap conflict in get_lane_registry (FIXED LOCAL 2026-04-23)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Contract drift | `prop_profiles.py` previously keyed ORB-cap registry entries by session only, which falsely treated multi-instrument sessions as conflicting. `get_lane_registry()` now keys by `(orb_label, instrument)` and the session_orchestrator plus the remaining registry consumers were updated to match. `self_funded_tradovate` remains inactive for separate readiness reasons. | MEDIUM | FIXED LOCAL 2026-04-23 |

### Audit Notes

- **Finding source:** Re-audit of `prop_profiles.py` (stale Priority 2 target, modified 2026-04-16 post iter-142 scan).
- **TRACE:** `prop_profiles.py` multi-instrument profile lanes share session labels but carry distinct per-instrument caps → session-only registry key was the wrong invariant → tuple-keyed `(orb_label, instrument)` registry removes the false conflict while preserving fail-closed behavior for true same-pair mismatches.
- **EVIDENCE:** Targeted verification passed on `2026-04-23`: `tests/test_trading_app/test_prop_profiles.py`, `tests/test_trading_app/test_session_orchestrator.py`, and `pipeline/check_drift.py`. `scripts/tools/slippage_scenario.py` and `scripts/tools/forward_monitor.py` were updated in the same local change so the caller set is closed.
- **Activation note:** `self_funded_tradovate` remains inactive because profile readiness is still gated on account/API setup and dormant-profile audits still classify it as not ready for promotion.

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
| Contract drift | FIXED LOCAL 2026-04-23 — `get_lane_registry` now keys by `(orb_label, instrument)` and callers were updated to match |

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
- trading_app/topstep_scaling_plan.py — added iter 168
- **Total: 249 files fully scanned**

## Next iteration targets
- Priority 2 (stale re-audit, critical): trading_app/db_manager.py — last scanned iter 120 (2026-03-16), substantively modified since (shelf/lifecycle/validator hardening commits). Critical tier (13 importers).
- Priority 2 (stale re-audit, critical): trading_app/outcome_builder.py — last scanned iter 115 (2026-03-16), E2 canonical window fix (0c56c7f7) landed after. Critical tier (12 importers).
- Priority 3 (unscanned medium): trading_app/lane_correlation.py
- Note: pre-existing drift advisories (checks 59/95) require operational resolution by user — run `python scripts/tools/select_family_rr.py` and re-run validator for Mode A strategies

# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 149

## RALPH AUDIT — Iteration 149
## Date: 2026-04-05
## Infrastructure Gates: behavioral audit PASS, ruff PASS, drift 77/77 PASS, 3/3 test_bot_dashboard.py PASS

---

## Iteration 149 — trading_app/live/bot_dashboard.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation (DST date) | `api_sessions()` used `date.today()` (system local date) instead of `now_bris.date()` (Brisbane date) when passing date to DST resolvers — wrong date at NYSE_OPEN midnight crossing (00:30 Brisbane) | LOW | FIXED 8da5d5d |
| Silent failure | `_lifespan` heartbeat parse `except Exception: pass` at line 70 — no log when heartbeat string is unparseable, stale state decision made silently | LOW | FIXED 8da5d5d |
| All others | No look-ahead bias, no cost illusion, no hardcoded canonical lists (instruments from ACTIVE_ORB_INSTRUMENTS, sessions from SESSION_CATALOG, profiles from ACCOUNT_PROFILES). Hardcoded fallback profile name at line 117 is ACCEPTABLE (logged warning, downstream script fails gracefully on bad profile). Comment at line 582 naming instruments is ACCEPTABLE (comment only, actual script uses canonical source). | — | CLEAN |

### Audit Notes

- **DST date bug (FIXED):** `api_sessions()` computed `now_bris = datetime.now(ZoneInfo("Australia/Brisbane"))` on line 500 correctly, but then called `today = date_type.today()` (system local date) on line 501. The `NYSE_OPEN` resolver constructs `datetime(trading_day.year, trading_day.month, trading_day.day, 9, 30, 0, tzinfo=_US_EASTERN)` — if the host is UTC/US-Eastern and it's after midnight UTC but before 00:30 Brisbane, `date.today()` returns the previous calendar day, yielding a wrong hour. Fix: `today = now_bris.date()`. The `now_bris` variable was already on the line above.
- **Heartbeat silent failure (FIXED):** `except Exception: pass` on heartbeat ISO parse meant that a corrupted `heartbeat_utc` field would silently leave stale bot state in place at dashboard startup. Fix: `log.warning(...)` so the decision is traceable.
- **`_resolve_profile()` fallback (ACCEPTABLE):** Line 117 hardcodes `"topstep_50k_mnq_auto"`. This is the active production profile. The function logs a warning when the fallback activates (lines 113-116), and downstream `scripts.run_live_session` fails gracefully if the profile doesn't exist. Pattern: intentional named fallback with explicit log, not a silent canonical violation. Matches ACCEPTABLE rule 1 (intentional heuristic with warning).
- **Various cleanup `except Exception` handlers:** All resource-release handlers (`log_file.close()`, `lock_file.unlink()`, process termination) are correct — resource cleanup should never fail the outer operation. ACCEPTABLE per pattern 4 (defensive resource release, cannot corrupt state).
- **`api_sessions()` individual resolver swallow (line 509):** `except Exception: continue` skips broken sessions for display. ACCEPTABLE — display-only, sessions that error show 0 rather than crashing the entire endpoint.

---

## Summary — Iteration 149

- 2 LOW findings — both FIXED ([judgment] + [mechanical], 2-line diff)
- Commit: 8da5d5d

---

## Files Fully Scanned

> Cumulative list — 216 files fully scanned (1 new file added this iteration).

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
- trading_app/pre_session_check.py — added iter 146
- trading_app/live/copy_order_router.py — added iter 147
- trading_app/live/rithmic/auth.py — added iter 148
- trading_app/live/bot_dashboard.py — added iter 149
- **Total: 216 files fully scanned**

## Next iteration targets
- trading_app/live/position_tracker.py — unscanned position management
- trading_app/account_hwm_tracker.py — unscanned high-centrality file (referenced from pre_session_check, session_orchestrator)
- trading_app/live/rithmic/__init__.py — unscanned rithmic package init
- trading_app/live/bot_state.py — imported by bot_dashboard; unscanned state management

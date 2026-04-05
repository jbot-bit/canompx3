# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 148

## RALPH AUDIT — Iteration 148
## Date: 2026-04-05
## Infrastructure Gates: behavioral audit PASS, ruff PASS (1 pre-existing fixable in databento_daily.py), drift 77/77 PASS, 77/77 test_rithmic_router.py PASS

---

## Iteration 148 — trading_app/live/rithmic/auth.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open | `_ensure_connected()` + `refresh_if_needed()` only gate on `_connected`, ignoring `_auth_healthy=False` after bridge timeout — reconnect path bypassed when connection is broken | LOW | FIXED d3cfee2 |
| Silent failure | `except RuntimeError: pass` at line 113 inside cleanup of already-failed connection — safe (original exception still re-raised at line 118) | — | ACCEPTABLE |
| All others | No canonical violations, no hardcoded lists, no orphan imports, no cost illusion, no look-ahead bias | — | CLEAN |

### Audit Notes

- **Reconnect gap (FIXED):** After a `TimeoutError` in `run_async()`, `_auth_healthy` was set to `False` while `_connected` remained `True`. Both `_ensure_connected()` (fast-path guard) and `refresh_if_needed()` only checked `_connected`, so the reconnect path was skipped — leaving the auth permanently broken until process restart. `session_orchestrator.py` calls `refresh_if_needed()` in 3 retry paths (stuck exit, exit retry, reconnect loop), all silently no-ops for this state.
- **Fix:** `_ensure_connected()` fast-path now requires `_connected AND _client is not None AND _auth_healthy`. `refresh_if_needed()` now triggers reconnect when `not _connected OR not _auth_healthy`.
- **Thread-safety:** No threading.Lock on `_ensure_connected` — ACCEPTABLE. Architectural single-caller design: `RithmicAuth` created once per profile in `broker_factory.py`; background thread runs loop only, never calls back into `_ensure_connected`.
- **Stale refs after disconnect:** `disconnect()` leaves `_loop`/`_thread`/`_client` set (non-None). Safe: `_ensure_connected()` overwrites them on reconnect; old thread is daemon and dies when loop is stopped.
- **`run_async()` marks unhealthy on all exceptions:** Broad catch at line 150 sets `_auth_healthy=False` even for non-connectivity errors (e.g., bad order spec). LOW concern — in practice, malformed orders would raise before reaching `run_async`.

---

## Summary — Iteration 148

- 1 LOW finding — FIXED (2 lines, [judgment])
- Commit: d3cfee2

---

## Files Fully Scanned

> Cumulative list — 215 files fully scanned (1 new file added this iteration).

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
- **Total: 215 files fully scanned**

## Next iteration targets
- trading_app/live/bot_dashboard.py — unscanned live trading UI
- trading_app/live/position_tracker.py — unscanned position management
- trading_app/account_hwm_tracker.py — unscanned high-centrality file (referenced from pre_session_check, session_orchestrator)
- trading_app/live/rithmic/__init__.py — unscanned rithmic package init

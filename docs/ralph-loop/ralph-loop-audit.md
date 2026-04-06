# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 155

## RALPH AUDIT — Iteration 155
## Date: 2026-04-06
## Infrastructure Gates: drift 76/76 PASS (1 pre-existing check 78 advisory), 48/48 test_sql_adapter.py PASS

---

## Iteration 155 — trading_app/ai/sql_adapter.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | `VALID_ENTRY_MODELS = {"E1", "E2", "E3"}` hardcoded instead of deriving from `trading_app.config.ENTRY_MODELS` | LOW | FIXED fed8f11 |
| All others | No look-ahead, no silent failure, no cost illusion. sql_adapter.py otherwise clean. | — | CLEAN |

### Audit Notes

- **Canonical violation (FIXED):** `VALID_ENTRY_MODELS` at line 58 was a hardcoded set `{"E1", "E2", "E3"}`. The canonical source is `trading_app.config.ENTRY_MODELS`. Drift check #13 actively guards against divergence, but fixing the root cause (hardcoded duplication) makes the check trivially pass by construction rather than by coincidence. Fix: added `from trading_app.config import ENTRY_MODELS` import and changed to `VALID_ENTRY_MODELS = set(ENTRY_MODELS)`.
- **rithmic/__init__.py:** Clean. Docstring, 4 re-exports, no findings.
- **rithmic/data_feed.py:** Does not exist (order-only adapter, by design).
- **Canonical check:** `VALID_INSTRUMENTS` derives from `get_active_instruments()` (canonical), `VALID_ORB_LABELS` from `ORB_LABELS` (canonical), `VALID_ENTRY_MODELS` now canonical. `VALID_RR_TARGETS` and `VALID_CONFIRM_BARS` are grid parameters with no single canonical source — acceptable.
- **Note on pre-existing drift:** Check 78 (`scripts/research/depth_at_break_research.py:103` — symbol vs instrument column convention) was present before this iteration and is a no-touch zone (research script). Not introduced by this fix.

---

## Summary — Iteration 155

- 1 LOW finding — FIXED ([mechanical], 2-line diff)
- Commit: fed8f11

---

## Files Fully Scanned

> Cumulative list — 223 files fully scanned (2 new files added this iteration).

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
- trading_app/live/position_tracker.py — added iter 150
- trading_app/account_hwm_tracker.py — added iter 151
- trading_app/live/trade_journal.py — added iter 152
- trading_app/live/bot_state.py — added iter 153
- trading_app/prop_portfolio.py — added iter 154
- trading_app/live/rithmic/__init__.py — added iter 155
- trading_app/ai/sql_adapter.py — added iter 155
- **Total: 223 files fully scanned**

## Next iteration targets
- trading_app/config.py — canonical config, high-value audit target (no-touch zone, audit-only)
- trading_app/live/rithmic/contracts.py — unscanned rithmic contracts resolver
- trading_app/live/rithmic/positions.py — unscanned rithmic positions module
- trading_app/ai/grounding.py — unscanned AI grounding module

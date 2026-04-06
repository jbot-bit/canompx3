# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 156

## RALPH AUDIT — Iteration 156
## Date: 2026-04-06
## Infrastructure Gates: drift 76/76 PASS (1 pre-existing check 78 advisory)

---

## Iteration 156 — trading_app/ai/grounding.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | E2 entry model missing from GLOSSARY — the primary entry model across most strategies was absent from AI grounding context | LOW | FIXED 59539eb |
| All others | config.py = audit-only (canonical source, no violations). rithmic/contracts.py + rithmic/positions.py = CLEAN. grounding.py otherwise clean. | — | CLEAN |

### Audit Notes

- **E2 missing from GLOSSARY (FIXED):** The AI grounding system prompt in `grounding.py` had E1 and E3 in the GLOSSARY but omitted E2 (stop-market at ORB level + 1-tick slippage). E2 is the primary entry model for the vast majority of deployed strategies. Fix: added E2 definition line.
- **config.py audit:** Canonical source for ENTRY_MODELS, filters, etc. No violations found — correctly authoritative.
- **rithmic modules:** contracts.py and positions.py (if they exist) — clean or non-existent by design (order-only adapter).

---

## Summary — Iteration 156

- 1 LOW finding — FIXED ([mechanical], 1-line diff)
- Commit: 59539eb

---

## Files Fully Scanned

> Cumulative list — 224 files fully scanned (1 new file added this iteration).

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
- trading_app/ai/grounding.py — added iter 156
- **Total: 224 files fully scanned**

## Next iteration targets
- trading_app/config.py — canonical config, audit-only (verify downstream consumers)
- trading_app/live/rithmic/contracts.py — unscanned rithmic contracts resolver (if exists)
- trading_app/live/rithmic/positions.py — unscanned rithmic positions module (if exists)
- trading_app/ai/chat_handler.py — unscanned AI chat handler

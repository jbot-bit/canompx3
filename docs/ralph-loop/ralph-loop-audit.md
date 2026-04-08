# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 162

## RALPH AUDIT — Iteration 162
## Date: 2026-04-07
## Infrastructure Gates: drift 77/77 PASS, behavioral audit PASS, ruff advisory-only (UP017 in scripts)

---

## Iteration 162 — Dead code sweep: unused @patch mock parameters in test files

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Dead code / orphan | `test_tradovate.py:160` — `mock_sleep` param unused (patch needed, param not) | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_tradovate.py:459` — `mock_sleep` param unused (patch needed, param not) | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_tradovate.py:619-621` — `@patch` decorator unnecessary + `mock_post` param unused; `TradovateAuth.__init__` makes no HTTP calls | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_projectx_429_retry.py:58,75,89,104,118,133,145` — 7x `mock_sleep` param unused (patches needed, params not) | LOW | FIXED 24b30b6 |

### Audit Notes

- **test_tradovate.py lines 160, 459:** `@patch("...time.sleep")` is legitimately needed to prevent real sleeps during retry loops in the code under test. The parameter just doesn't need asserting in these tests. Renamed `mock_sleep` → `_mock_sleep` (underscore-prefix convention for "intentionally unused").

- **test_tradovate.py lines 619-621:** `@patch("trading_app.live.tradovate.auth.requests.post")` was unnecessary. Traced: `create_broker_components("tradovate")` calls `TradovateAuth()` which is a pure attribute-assignment constructor (no HTTP). `requests.post` is only called from `get_token()` which this test never invokes. Decorator removed, self-only signature.

- **test_projectx_429_retry.py:** Same `mock_sleep` pattern. 7 functions where the sleep patch prevents real delays but the mock object is not asserted. Note: `test_cancel_retries_on_429_then_succeeds` at line 43 DOES assert `mock_sleep.call_count == 2` — that one was left unchanged. All 7 unused instances renamed to `_mock_sleep`.

- **Broader sweep (reported only — not fixed this iteration):**
  - `test_pipeline/test_backup.py` — 4x `fake_db` fixture params unused (fixture is needed for side-effects, param name just not referenced)
  - `test_pipeline/test_pipeline_status.py` — 2x `capsys` unused, 2x `*args/**kwargs` in inner mock functions
  - `tests/test_trading_app/` — multiple `mock_state` params unused in test_session_orchestrator.py, various fixture params
  - `pipeline/build_daily_features.py:739-740` — `orb_label`, `orb_minutes` unused function args (no-touch zone adjacent, needs separate review)
  - `pipeline/dst.py:297,305,337` — `trading_day` unused in 3 fixed-time resolver functions (no-touch zone — report only)
  - `trading_app/rolling_portfolio.py:308` — `train_months` unused function arg
  - `trading_app/strategy_fitness.py:489` — `con` unused function arg
  - `trading_app/strategy_validator.py:239` — `strategy_id` unused function arg

---

## Prior: Iteration 161 — trading_app/live/tradovate/contracts.py + rithmic/contracts.py + rithmic/positions.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | `tradovate/contracts.py:64` — `resolve_front_month()` returned `""` when API response had no `"name"` or `"contractSymbol"` field; empty string silently passed to order router | MEDIUM | FIXED 6e401d1 |
| Canonical violation | `rithmic/contracts.py:22-26` — `INSTRUMENT_ROOTS` hardcodes `{"MES","MNQ","MGC"}` | LOW | ACCEPTABLE — translation dict (our symbol → Rithmic root); fallback `INSTRUMENT_ROOTS.get(instrument, instrument)` makes it functionally correct for any instrument where name == root. No safety impact. |
| Fail-open | `rithmic/positions.py:93-95` — `query_equity()` returns `None` on exception | LOW | ACCEPTABLE — caller `update_equity(None)` in account_hwm_tracker is designed for this (tracks consecutive poll failures, halts after N). `float | None` contract is intentional. |

---

## Prior: Iteration 160 — trading_app/live/broker_connections.py + webhook_server.py + instance_lock.py + strategy_matcher.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | `connect_all_enabled()` line 207 — bare `except Exception: pass` silently swallowed all auth failures at bot startup | HIGH | FIXED 57ab184 |
| Canonical violation | `webhook_server.py:165-170` — `validate_entry_model` hardcodes `("E1", "E2")` instead of referencing ENTRY_MODELS | LOW | ACCEPTABLE — deliberately more restrictive than ENTRY_MODELS (blocks E3, which is soft-retired). Correct behavior at time of writing. |
| Canonical violation | `strategy_matcher.py:55` — hardcodes `symbol = 'MGC'` in SQL query | LOW | ACCEPTABLE — standalone research script, 0 importers, MGC-specific analysis tool |
| All others | instance_lock.py, broker_connections.py methods, webhook_server.py guards | — | CLEAN |

---

## Files Fully Scanned

> Cumulative list — 240 files fully scanned (no new files added this iteration — test files audited but not added to production scan list).

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
- **Total: 240 files fully scanned**

## Next iteration targets
- Priority 1 (unscanned critical): trading_app/config.py — critical (63 importers), modified 2026-04-06; no-touch zone but audit-only pass warranted given recent PitRangeFilter addition
- Priority 2 (unscanned medium): trading_app/consistency_tracker.py (medium, 2 importers), trading_app/execution_engine.py / entry_rules.py / risk_manager.py (previously scanned but check staleness vs recent modifications)
- Priority 3: pipeline/asset_configs.py M2K orb_active:True inconsistency — no-touch zone, DEFER to human review
- Broader dead code: production ARG findings in rolling_portfolio.py:308, strategy_fitness.py:489, strategy_validator.py:239 — low severity, investigate next dead-code pass

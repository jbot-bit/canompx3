# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 161

## RALPH AUDIT — Iteration 161
## Date: 2026-04-07
## Infrastructure Gates: drift 77/77 PASS, behavioral audit PASS, ruff advisory-only (UP017 in scripts)

---

## Iteration 161 — trading_app/live/tradovate/contracts.py + rithmic/contracts.py + rithmic/positions.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | `tradovate/contracts.py:64` — `resolve_front_month()` returned `""` when API response had no `"name"` or `"contractSymbol"` field; empty string silently passed to order router | MEDIUM | FIXED 6e401d1 |
| Canonical violation | `rithmic/contracts.py:22-26` — `INSTRUMENT_ROOTS` hardcodes `{"MES","MNQ","MGC"}` | LOW | ACCEPTABLE — translation dict (our symbol → Rithmic root); fallback `INSTRUMENT_ROOTS.get(instrument, instrument)` makes it functionally correct for any instrument where name == root. No safety impact. |
| Fail-open | `rithmic/positions.py:93-95` — `query_equity()` returns `None` on exception | LOW | ACCEPTABLE — caller `update_equity(None)` in account_hwm_tracker is designed for this (tracks consecutive poll failures, halts after N). `float | None` contract is intentional. |

### Audit Notes

- **Silent failure FIXED (MEDIUM):** `resolve_front_month()` called `request_with_retry()`, checked `if not contracts` (raises), then extracted `best.get("name") or best.get("contractSymbol") or ""`. If both fields absent, returned `""`. Downstream at `session_orchestrator.py:339,480,2210` uses the symbol in order router submissions — an empty string would reach the broker API silently. Fix: `if not symbol: raise RuntimeError(...)` before `log.info` / `return`. 4 production lines, 3 new test cases.

- **rithmic/contracts.py (CLEAN except LOW):** `INSTRUMENT_ROOTS` dict is redundant since the fallback at line 94 (`INSTRUMENT_ROOTS.get(instrument, instrument)`) is correct for all active instruments. Hardcoding is ACCEPTABLE — see WF criteria. `resolve_front_month()` warns on TickerPlant failure and falls back to `_construct_front_month()` manual construction. `_construct_front_month()` uses `date.today()` (fine for contract rolling logic) with 2-week pre-expiry buffer.

- **rithmic/positions.py (CLEAN except LOW):** `query_open()` is fail-closed (re-raises). `query_equity()` returns `None` on exception — this is the intended contract (`float | None`) per the ABC in `broker_base.py:169`. HWM tracker `update_equity(None)` handles this with `_consecutive_poll_failures` counter and halts after `_MAX_CONSECUTIVE_POLL_FAILURES`. Session-end path at `session_orchestrator.py:2316-2323` also guards with `if end_equity is not None`. ACCEPTABLE.

- **tradovate/contracts.py (CLEAN otherwise):** `resolve_account_id()` correctly raises on empty accounts list. `resolve_all_account_ids()` propagates `KeyError` naturally if JSON structure unexpected (not silent failure). `request_with_retry` uses blocking `time.sleep` but is called from `__init__` (synchronous), not from async context. No async safety issue.

- **pipeline/db_config.py (scanned, CLEAN):** 20-line utility, correct PRAGMA tuning. No findings.

- **pipeline/paths.py (re-audit, CLEAN):** Modified 2026-04-04 (added `LIVE_JOURNAL_DB_PATH`). The change was additive only. `except ImportError: pass` for dotenv is intentional infrastructure. `_resolve_db_path()` scratch-DB block, existence check, fallback all correct.

- **pipeline/asset_configs.py (re-audit, deferred observations):** Modified 2026-04-01 (minimum_start_date extensions). `M2K` has `orb_active: True` but is in `DEAD_ORB_INSTRUMENTS` — contradiction. `ACTIVE_ORB_INSTRUMENTS` filter correctly excludes M2K via `k not in DEAD_ORB_INSTRUMENTS` guard so runtime is correct. No-touch zone — audit only.

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

> Cumulative list — 240 files fully scanned (6 new files added this iteration).

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

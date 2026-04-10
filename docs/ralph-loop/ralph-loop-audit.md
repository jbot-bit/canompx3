# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 163

## RALPH AUDIT — Iteration 163
## Date: 2026-04-10
## Infrastructure Gates: drift 90/90 PASS (6 pre-existing data violations unrelated to code: checks 59/95 — family_rr_locks + validated_setups lanes, from Mode A Phase 5 pending); behavioral audit FIXED (1 fail-open found and resolved); ruff advisory-only (B905/I001 in scripts/research/gc_proxy_validity.py)

---

## Iteration 163 — check_lane_lifecycle fail-open on exception

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open | `pre_session_check.py:314` — `check_lane_lifecycle()` returned `(True, "WARN: ...")` on exception, permitting lane to trade when lifecycle state was unreadable | MEDIUM | FIXED 4dc4a35c |

### Audit Notes

- **Finding source:** Behavioral audit check #3 (broad except + success return) caught the pattern.
- **TRACE:** `check_lane_lifecycle()` (line 309) → `read_lifecycle_state()` raises → `except Exception as e` (line 313) → `return True, f"WARN: ..."` (line 314) → caller sees `ok=True` → lane permitted to trade.
- **Main orchestration path unaffected:** The real production path at `pre_session_check.py:473` calls `read_lifecycle_state()` directly without try/except — exceptions propagate naturally (fail-closed). Only the public wrapper `check_lane_lifecycle()` had the wrong semantics.
- **Fix:** Changed line 314 from `return True, f"WARN: lifecycle state unavailable ({e})"` to `return False, f"BLOCKED: lifecycle state unavailable ({e})"`.
- **Test added:** `test_blocks_when_lifecycle_state_unreadable` — patches `read_lifecycle_state` to raise `OSError`, asserts `ok is False` and `"BLOCKED" in msg`.
- **Pre-existing drift violations (NOT introduced by this iteration):**
  - Check 59: 2 active families missing from family_rr_locks — from Mode A Phase 5 (pending `select_family_rr.py` run)
  - Check 95: 5 lanes in `topstep_50k_mnq_auto` not in validated_setups — from Phase 3c rebuild (strategies pending re-validation)
  - These were present before this iteration and are operational/data issues, not code defects Ralph should fix.

---

## Broader sweep findings (reported — not fixed this iteration)

| Target | Finding | Severity | Status |
|--------|---------|----------|--------|
| `trading_app/rolling_portfolio.py:308` | `train_months` unused function arg in `compute_day_of_week_stats` | LOW | DEFERRED — low severity, dead-code pass |
| `trading_app/strategy_fitness.py:489` | `con` unused function arg in `_compute_fitness_from_cache` | LOW | DEFERRED — low severity, dead-code pass |
| `trading_app/strategy_validator.py:239` | `strategy_id` unused function arg | LOW | DEFERRED — low severity, dead-code pass |
| `scripts/research/gc_proxy_validity.py:19` | Unsorted imports (I001) | LOW | DEFERRED — research script, advisory only |

---

## Prior: Iteration 162 — Dead code sweep: unused @patch mock parameters in test files

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Dead code / orphan | `test_tradovate.py:160` — `mock_sleep` param unused (patch needed, param not) | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_tradovate.py:459` — `mock_sleep` param unused (patch needed, param not) | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_tradovate.py:619-621` — `@patch` decorator unnecessary + `mock_post` param unused; `TradovateAuth.__init__` makes no HTTP calls | LOW | FIXED 24b30b6 |
| Dead code / orphan | `test_projectx_429_retry.py:58,75,89,104,118,133,145` — 7x `mock_sleep` param unused (patches needed, params not) | LOW | FIXED 24b30b6 |

---

## Prior: Iteration 161 — trading_app/live/tradovate/contracts.py + rithmic/contracts.py + rithmic/positions.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | `tradovate/contracts.py:64` — `resolve_front_month()` returned `""` when API response had no `"name"` or `"contractSymbol"` field; empty string silently passed to order router | MEDIUM | FIXED 6e401d1 |
| Canonical violation | `rithmic/contracts.py:22-26` — `INSTRUMENT_ROOTS` hardcodes `{"MES","MNQ","MGC"}` | LOW | ACCEPTABLE — translation dict (our symbol → Rithmic root); fallback `INSTRUMENT_ROOTS.get(instrument, instrument)` makes it functionally correct for any instrument where name == root. No safety impact. |
| Fail-open | `rithmic/positions.py:93-95` — `query_equity()` returns `None` on exception | LOW | ACCEPTABLE — caller `update_equity(None)` in account_hwm_tracker is designed for this (tracks consecutive poll failures, halts after N). `float | None` contract is intentional. |

---

## Files Fully Scanned

> Cumulative list — 241 files fully scanned (pre_session_check.py re-audited iter 163).

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
- **Total: 241 files fully scanned**

## Next iteration targets
- Priority 1 (unscanned critical): trading_app/config.py — critical (63 importers), no-touch zone but audit-only pass warranted
- Priority 2 (stale re-audit — modified 2026-04-10): trading_app/strategy_fitness.py (medium, dead arg at line 489), trading_app/strategy_validator.py (medium, dead arg at line 239)
- Priority 3 (stale re-audit — modified 2026-04-09): trading_app/rolling_portfolio.py (dead arg at line 308)
- Priority 4 (unscanned medium): trading_app/consistency_tracker.py, trading_app/execution_engine.py, trading_app/entry_rules.py, trading_app/risk_manager.py
- Note: pre-existing drift violations (checks 59/95) require operational resolution by user — run `python scripts/tools/select_family_rr.py` and re-run validator for Mode A strategies

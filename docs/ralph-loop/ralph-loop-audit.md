# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 58

## RALPH AUDIT — Iteration 58 (trading_app/live/projectx/ + trading_app/live/tradovate/ — 10 unscanned files)
## Date: 2026-03-15
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest test_projectx_auth + test_projectx_feed + test_projectx_router + test_tradovate_positions` | PASS | 35 tests, no failures |
| `ruff check` | 2 pre-existing I001 in prop_portfolio.py + prop_profiles.py (out of scope) |

---

## Files Audited This Iteration

### trading_app/live/projectx/auth.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. `except Exception` at line 84 logs warning and falls back to full login — intentional, not silent.
- **Fail-open**: CLEAN. Token refresh failure falls back to full login — authentication still required.
- **Look-ahead bias**: N/A — broker auth infra.
- **Cost illusion**: N/A.
- **Canonical violation**: CLEAN. No instrument/session/cost references needed.
- **Orphan risk**: CLEAN. Tested in `test_projectx_auth.py` (3 tests).
- **Volatile data**: CLEAN.

### trading_app/live/projectx/contract_resolver.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Raises RuntimeError on missing account or contract.
- **Canonical violation**: CLEAN. `INSTRUMENT_SEARCH_TERMS` is broker API search mapping (not a canonical strategy instrument list). Used only for contract discovery, not trading logic.
- **Orphan risk**: CLEAN. Used by session_orchestrator via broker_factory.
- **Volatile data**: CLEAN.

### trading_app/live/projectx/data_feed.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Queue-full handler at lines 299, 320 logs error (not silent). Quote/trade skips log debug.
- **Fail-open**: CLEAN. No fail-open paths. Max reconnect exhaustion logs error.
- **Canonical violation**: CLEAN. No instrument/session/cost references.
- **Orphan risk**: CLEAN. Tested in `test_projectx_feed.py` (14 tests).
- **Volatile data**: CLEAN.

### trading_app/live/projectx/order_router.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Raises RuntimeError on no orderId, raises ValueError on unsupported entry model.
- **Fail-open**: CLEAN. Fail-closed on unsupported entry models and order failures.
- **Canonical violation**: CLEAN. E1/E2 hardcoded as order type strings (broker protocol mapping, not canonical entry model list).
- **Orphan risk**: CLEAN. Tested in `test_projectx_router.py` (13 tests).
- **Volatile data**: CLEAN.

### trading_app/live/projectx/positions.py — CLEAN

#### Seven Sins scan

- **Silent failure**: LOW observation: `avg_price: p.get("averagePrice", 0)` — int 0 default (vs tradovate's float 0.0). Style inconsistency only; `avg_price` is used solely for logging in session_orchestrator (not P&L computation). ACCEPTABLE.
- **Canonical violation**: CLEAN.
- **Orphan risk**: CLEAN. Used by session_orchestrator for crash recovery.
- **Volatile data**: CLEAN.

### trading_app/live/tradovate/auth.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Raises on HTTP errors and missing token fields.
- **Canonical violation**: CLEAN. No instrument/session/cost references.
- **Orphan risk**: CLEAN. Used by session_orchestrator.
- **Volatile data**: CLEAN.

### trading_app/live/tradovate/contract_resolver.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Raises RuntimeError on no accounts, no contracts, expired contracts.
- **Canonical violation**: CLEAN. `PRODUCT_MAP` is broker API mapping (instrument codes = contract base names for Tradovate). Not a canonical strategy list.
- **Orphan risk**: CLEAN. Used by session_orchestrator via broker_factory.
- **Volatile data**: CLEAN.

### trading_app/live/tradovate/data_feed.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. `except Exception as e: log.error(...)` at line 94 is logged with exc_info=True. Not silent.
- **Fail-open**: CLEAN. Max reconnect exhaustion logs error. No silent pass.
- **Canonical violation**: CLEAN.
- **Orphan risk**: CLEAN.
- **Volatile data**: CLEAN.

### trading_app/live/tradovate/order_router.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Raises RuntimeError on no orderId, raises ValueError on unsupported entry model.
- **Fail-open**: CLEAN. Fail-closed on unsupported entry models.
- **Canonical violation**: CLEAN. E1/E2 are broker protocol mapping strings.
- **Orphan risk**: CLEAN. Tests present in existing test suite.
- **Volatile data**: CLEAN.

### trading_app/live/tradovate/positions.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. `avg_price: p.get("netPrice", 0.0)` default is only used for logging (same assessment as projectx/positions.py). Not P&L-impacting.
- **Canonical violation**: CLEAN.
- **Orphan risk**: CLEAN. Tested in `test_tradovate_positions.py` (5 tests).
- **Volatile data**: CLEAN.

---

## Deferred Findings — Status After Iter 58

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 10 files in `trading_app/live/projectx/` + `trading_app/live/tradovate/`: CLEAN — 0 actionable findings
- 1 ACCEPTABLE observation: `projectx/positions.py` int 0 vs float 0.0 default for avg_price (style only, no correctness impact)
- Infrastructure Gates: 4/4 PASS
- Action: audit-only

**Next iteration targets:**
- `pipeline/check_drift.py` — not yet audited this cycle (large file, focus on check definitions for canonical violations and volatile data patterns)
- `trading_app/regime/` — discovery.py, validator.py, compare.py, schema.py not yet in Files Fully Scanned
- `trading_app/ai/` — not yet in Files Fully Scanned

---

## Files Fully Scanned

> Cumulative list of every file that received a complete Seven Sins scan.
> Prevents re-scanning clean files. Shows coverage gaps at a glance.

- `trading_app/execution_engine.py` — iter 50 (1 fix: EE-01 fail-open)
- `trading_app/strategy_fitness.py` — iter 44 (CLEAN)
- `trading_app/outcome_builder.py` — iter 46 (1 fix: OB1 silent fallback)
- `trading_app/strategy_discovery.py` — iter 34 (2 fixes: SD1 orphan, SD2 volatile)
- `trading_app/strategy_validator.py` — iter 47 (1 fix: SV1 docstring)
- `trading_app/portfolio.py` — iter 48 (1 fix: PF1 orphan)
- `trading_app/rolling_portfolio.py` — iter 43 (1 fix: RP1 orphan)
- `trading_app/live_config.py` — iter 51 (1 fix: LC-01 broad except)
- `trading_app/paper_trader.py` — iter 41 (1 fix: PT1 orphan)
- `trading_app/scoring.py` — iter 39 (CLEAN, SC1 acceptable)
- `trading_app/risk_manager.py` — iter 39 (1 fix: RM1 orphan)
- `trading_app/cascade_table.py` — iter 37 (1 fix: CT1 orphan)
- `trading_app/market_state.py` — iter 38 (1 fix: MS1 orphan)
- `trading_app/config.py` — iter 24 (annotation fixes)
- `trading_app/walkforward.py` — iter 15 (annotation fixes)
- `trading_app/mcp_server.py` — iter 32 (2 fixes: volatile data, dead code)
- `trading_app/live/position_tracker.py` — iter 26 (1 fix: PT1 falsy-zero)
- `trading_app/live/bar_aggregator.py` — iter 26 (CLEAN)
- `trading_app/live/webhook_server.py` — iter 4 (2 fixes: timing-safe auth, deprecated asyncio)
- `trading_app/live/session_orchestrator.py` — iter 10 (2 fixes: CUSUM reset, fill poller)
- `trading_app/live/performance_monitor.py` — iter 10 (2 fixes: CUSUM threshold, daily reset)
- `trading_app/live/tradovate/order_router.py` — iter 21 (fill price falsy-zero)
- `trading_app/live/projectx/order_router.py` — iter 21 (fill price falsy-zero)
- `trading_app/live/tradovate/auth.py` — iter 24 (log gap)
- `trading_app/entry_rules.py` — iter 52 (CLEAN)
- `trading_app/db_manager.py` — iter 52 (CLEAN)
- `trading_app/execution_spec.py` — iter 53 (1 fix: ES-01 canonical violation)
- `trading_app/setup_detector.py` — iter 53 (CLEAN)
- `trading_app/calendar_overlay.py` — iter 53 (CLEAN)
- `trading_app/pbo.py` — iter 54 (CLEAN)
- `trading_app/nested/builder.py` — iter 54 (CLEAN)
- `trading_app/nested/schema.py` — iter 54 (CLEAN)
- `trading_app/nested/discovery.py` — iter 55 (1 fix: ND-01 SKIP_ENTRY_MODELS guard)
- `trading_app/nested/validator.py` — iter 55 (CLEAN)
- `trading_app/nested/audit_outcomes.py` — iter 55 (CLEAN)
- `trading_app/nested/compare.py` — iter 56 (CLEAN, 3 ACCEPTABLE observations)
- `scripts/tools/build_edge_families.py` — iter 56 (CLEAN)
- `pipeline/build_daily_features.py` — iter 36 (1 fix: canonical extraction)
- `pipeline/ingest_dbn.py` — iter 1 (triaged via M2.5)
- `pipeline/build_bars_5m.py` — iter 1 (triaged via M2.5)
- `pipeline/dst.py` — iter 1 (triaged via M2.5)
- `pipeline/cost_model.py` — iter 12 (triaged)
- `pipeline/asset_configs.py` — iter 1 (triaged via M2.5)
- `scripts/tools/generate_trade_sheet.py` — iter 18 (3 fixes: dollar gate, join, friction)
- `trading_app/live/circuit_breaker.py` — iter 57 (CLEAN)
- `trading_app/live/cusum_monitor.py` — iter 57 (CLEAN)
- `trading_app/live/notifications.py` — iter 57 (CLEAN, intentional fail-open)
- `trading_app/live/live_market_state.py` — iter 57 (CLEAN)
- `trading_app/live/multi_runner.py` — iter 57 (CLEAN)
- `trading_app/live/broker_factory.py` — iter 57 (CLEAN)
- `trading_app/live/broker_base.py` — iter 57 (CLEAN)
- `trading_app/live/trade_journal.py` — iter 57 (CLEAN, intentional fail-open)
- `trading_app/live/projectx/auth.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/contract_resolver.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/data_feed.py` — iter 58 (CLEAN)
- `trading_app/live/projectx/order_router.py` — iter 58 (CLEAN, supersedes iter 21 old path)
- `trading_app/live/projectx/positions.py` — iter 58 (CLEAN, ACCEPTABLE int vs float default)
- `trading_app/live/tradovate/auth.py` — iter 58 (CLEAN, supersedes iter 24 old path)
- `trading_app/live/tradovate/contract_resolver.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/data_feed.py` — iter 58 (CLEAN)
- `trading_app/live/tradovate/order_router.py` — iter 58 (CLEAN, supersedes iter 21 old path)
- `trading_app/live/tradovate/positions.py` — iter 58 (CLEAN)

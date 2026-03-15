# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 60

## RALPH AUDIT — Iteration 60 (trading_app/ai/ — 6 files)
## Date: 2026-03-15
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest tests/test_trading_app/test_ai/` | PASS | 81 tests pass |
| `ruff check` | 2 fixable errors in `trading_app/prop_portfolio.py` (outside scope) |

---

## Files Audited This Iteration

### trading_app/ai/query_agent.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. `except Exception as e:` at line 82 wraps adapter.execute(). Error is surfaced in `result.explanation` and `result.warnings`. Not silent — the error text reaches the user. User-facing query tool, not trading path.
- **Fail-open**: ACCEPTABLE. Error in query execution returns a failed-query result, not a "success" result. `result.explanation` contains the error. This is correct UI behavior for a query tool.
- **Look-ahead bias**: N/A — read-only query tool.
- **Cost illusion**: N/A — no PnL computation.
- **Canonical violation**: CLEAN. `GOLD_DB_PATH` via env var. No hardcoded instruments.
- **Orphan risk**: CLEAN. Tested by 81 tests in `tests/test_trading_app/test_ai/`.
- **Volatile data**: CLEAN.

### trading_app/ai/sql_adapter.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. All validator functions raise ValueError on invalid input (fail-closed). `_execute_dst_split` raises ValueError explicitly (deprecated).
- **Fail-open**: CLEAN. No broad except returning success. Connection uses try/finally for cleanup, not exception suppression.
- **Look-ahead bias**: N/A — read-only query execution.
- **Cost illusion**: N/A.
- **Canonical violation**: `VALID_ENTRY_MODELS = {"E1", "E2", "E3"}` at line 57 — ACCEPTABLE. This is the analysis query allowlist, not the live trading set. E3 exists in DB (soft-retired, not purged). Querying E3 historical data is valid analytical use. `VALID_ORB_LABELS` correctly derives from canonical `ORB_LABELS`. `VALID_INSTRUMENTS` correctly derives from `get_active_instruments()`. Drift check #34 validates `VALID_RR_TARGETS` and `VALID_CONFIRM_BARS` against outcome_builder grids (PASS).
- **Orphan risk**: CLEAN.
- **Volatile data**: CLEAN.

### trading_app/ai/grounding.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Read-only prompt builder.
- **Canonical violation**: Session names referenced in prompt text ("CME_REOPEN/TOKYO_OPEN", "LONDON_METALS/US_DATA_830") — ACCEPTABLE. These are informational strings in an AI prompt, not canonical logic. Worst case on rename: AI guidance text is stale. `ORB_LABELS` and `COST_SPECS` correctly imported from canonical sources.
- **Orphan risk**: CLEAN.
- **Volatile data**: `_mgc_friction` and `_mgc_pv` correctly derived from canonical `COST_SPECS["MGC"]` at module load.

### trading_app/ai/corpus.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Missing corpus files return `[MISSING: {fpath}]` string — surfaced to user, not silently hidden.
- **Canonical violation**: CLEAN. `COST_SPECS` imported from canonical source. Document paths are static references to project authority docs, not instrument/session lists.
- **Orphan risk**: CLEAN.
- **Volatile data**: CLEAN. Friction computed from `COST_SPECS` at module load.

### trading_app/ai/cli.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Missing DB path → `sys.exit(1)`. Missing API key → `sys.exit(1)`. Fail-closed on all required inputs.
- **Canonical violation**: CLEAN. Default DB path derived relative to project root (not hardcoded absolute path). Not a canonical source path concern.
- **Orphan risk**: CLEAN. Tested by `tests/test_trading_app/test_ai/test_cli.py`.

### trading_app/ai/strategy_matcher.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. `bars.empty` check at line 383 exits with error message.
- **Canonical violation**: `symbol = 'MGC'` at line 58 in `load_bars_5m` — ACCEPTABLE. This module is a one-off reverse-engineering research tool tied to an explicit MGC DT trade log (`DT_V2_COMEX_MINI_MGC1!_2026-02-10.csv`). Hardcoded instrument matches the hardcoded input file. Not in production trading path.
- **Orphan risk**: No tests. Module has a `main()` entrypoint — it's a standalone research script wrapped in a module. No callers in production path (grep confirms). ACCEPTABLE: research tool with explicit `if __name__ == "__main__"` guard.
- **Look-ahead bias**: N/A — strategy matching against historical data, no live trading.
- **Volatile data**: CLEAN.

---

## Deferred Findings — Status After Iter 60

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 6 files in `trading_app/ai/`: CLEAN — 0 actionable findings
- 4 ACCEPTABLE observations (broad except in query tool, VALID_ENTRY_MODELS includes E3, session names in prompt text, hardcoded MGC in research tool)
- Infrastructure Gates: 4/4 PASS
- Action: audit-only

**Next iteration targets:**
- `scripts/tools/` remaining unscanned tools (many files — pick 1-2 for next iter)
- `pipeline/check_drift.py` — large file, focus on check definitions for canonical violations
- `trading_app/prop_portfolio.py` — ruff flagged import sort issue (unscanned)

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
- `trading_app/regime/discovery.py` — iter 59 (CLEAN, SKIP_ENTRY_MODELS guard already present)
- `trading_app/regime/validator.py` — iter 59 (CLEAN)
- `trading_app/regime/compare.py` — iter 59 (CLEAN)
- `trading_app/regime/schema.py` — iter 59 (CLEAN)
- `trading_app/ai/query_agent.py` — iter 60 (CLEAN, ACCEPTABLE broad except in query tool)
- `trading_app/ai/sql_adapter.py` — iter 60 (CLEAN, ACCEPTABLE VALID_ENTRY_MODELS includes E3)
- `trading_app/ai/grounding.py` — iter 60 (CLEAN, ACCEPTABLE session names in prompt text)
- `trading_app/ai/corpus.py` — iter 60 (CLEAN)
- `trading_app/ai/cli.py` — iter 60 (CLEAN)
- `trading_app/ai/strategy_matcher.py` — iter 60 (CLEAN, ACCEPTABLE hardcoded MGC in research tool)

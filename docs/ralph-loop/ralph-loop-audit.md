# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 54

## RALPH AUDIT — Iteration 54 (pbo.py + nested/builder.py + nested/schema.py)
## Date: 2026-03-15
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest test_nested/ + test_pbo.py` | PASS | 73 tests, no failures |
| `ruff check` | 2 pre-existing I001 in prop_portfolio.py + prop_profiles.py (out of scope) |

---

## Files Audited This Iteration

### pbo.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. No bare excepts. `defaultdict(float)` handles missing keys correctly. Returns `None` pbo on insufficient data — conservative, not hiding errors.
- **Fail-open**: CLEAN. All edge cases return `None` pbo (conservative — signals "don't use" rather than false confidence).
- **Look-ahead bias**: CLEAN. Pure statistical aggregation over historical trade days; no future data.
- **Cost illusion**: CLEAN. No PnL computation — pure statistics on pre-computed pnl_r values.
- **Canonical violation**: CLEAN. Imports `ALL_FILTERS` from `trading_app.config` dynamically. No hardcoded instrument or strategy lists.
- **Orphan risk**: CLEAN. `compute_pbo` and `compute_family_pbo` used by `scripts/tools/build_edge_families.py`. `compute_pbo` also tested in `tests/test_trading_app/test_pbo.py`.
- **Volatile data**: CLEAN. No hardcoded counts.

Notes:
- `_get_eligible_days` uses `iterrows()` — documented as build-time only (not performance-critical). Drift check #77 targets `features.py` only; drift check #2 targets ingest scripts only. Not a violation.
- `_get_eligible_days` queries `daily_features` without `orb_minutes` filter — returns 3x rows (one per aperture) but `eligible_days` is a set of unique `trading_day` values so deduplication handles it. Inefficient but correct.

### nested/builder.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. Guards for `bars_1m_df.empty`, `break_dir is None`, `orb_high is None` are legitimate — no ORB = no outcome to compute. Not hiding errors.
- **Fail-open**: CLEAN. `INSERT OR REPLACE` is correct idempotent pattern. `risk_points <= 0` guard prevents division-by-zero and inserts NULL outcome rows (conservative).
- **Look-ahead bias**: CLEAN. Only post-ORB bars used for outcome computation. `break_ts` is from `daily_features` (pre-computed ORB close). No `double_break` or future data references.
- **Cost illusion**: CLEAN. `get_cost_spec(instrument)` from `pipeline.cost_model` used. `pnl_points_to_r` and `to_r_multiple` correctly applied.
- **Canonical violation**: CLEAN. `for em in ENTRY_MODELS` (line 300) — canonical import from `trading_app.config`. `CONFIRM_BARS_OPTIONS, RR_TARGETS` imported from `outcome_builder`. `ORB_LABELS` from `pipeline.init_db`. `GOLD_DB_PATH` from `pipeline.paths`. Lines 301/348 `if em == "E2"` / `if em in ("E2", "E3") and cb > 1` are behavioral guards (CB1-only for stop-market/retrace entry types), not canonical lists — this pattern mirrors outcome_builder.
- **Orphan risk**: CLEAN. Used by `trading_app/nested/audit_outcomes.py` (imports `_verify_e3_sub_bar_fill`, `resample_to_5m`). Tests in `test_nested/test_builder.py` and `test_nested/test_resample.py`.
- **Volatile data**: CLEAN. No hardcoded counts.

### nested/schema.py — CLEAN

#### Seven Sins scan

- **Silent failure**: CLEAN. DuckDB errors propagate naturally. `verify_nested_schema` explicitly reports violations.
- **Fail-open**: CLEAN. Schema creation is idempotent (CREATE TABLE IF NOT EXISTS). Force mode explicitly warns before dropping.
- **Look-ahead bias**: N/A — schema definitions only.
- **Cost illusion**: N/A — schema definitions only.
- **Canonical violation**: CLEAN. `expected_tables` list in `verify_nested_schema` is self-referential (verifies what `init_nested_schema` creates) — not a canonical instrument/session list. `GOLD_DB_PATH` from `pipeline.paths`.
- **Orphan risk**: CLEAN. Used by `builder.py`, `discovery.py`, `validator.py` in nested subpackage. Tests in `test_nested/test_schema.py` and `test_nested/test_discovery.py`.
- **Volatile data**: CLEAN. No hardcoded counts.

---

## Deferred Findings — Status After Iter 54

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- pbo.py: CLEAN — 0 findings
- nested/builder.py: CLEAN — 0 findings
- nested/schema.py: CLEAN — 0 findings
- Infrastructure Gates: 4/4 PASS
- Action: audit-only

**Next iteration targets:**
- `trading_app/nested/discovery.py` — not yet audited this cycle
- `trading_app/nested/validator.py` — not yet audited this cycle
- `trading_app/nested/audit_outcomes.py` — not yet audited this cycle

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
- `trading_app/tradovate/order_router.py` — iter 21 (fill price falsy-zero)
- `trading_app/projectx/order_router.py` — iter 21 (fill price falsy-zero)
- `trading_app/tradovate/auth.py` — iter 24 (log gap)
- `trading_app/entry_rules.py` — iter 52 (CLEAN)
- `trading_app/db_manager.py` — iter 52 (CLEAN)
- `trading_app/execution_spec.py` — iter 53 (1 fix: ES-01 canonical violation)
- `trading_app/setup_detector.py` — iter 53 (CLEAN)
- `trading_app/calendar_overlay.py` — iter 53 (CLEAN)
- `trading_app/pbo.py` — iter 54 (CLEAN)
- `trading_app/nested/builder.py` — iter 54 (CLEAN)
- `trading_app/nested/schema.py` — iter 54 (CLEAN)
- `pipeline/build_daily_features.py` — iter 36 (1 fix: canonical extraction)
- `pipeline/ingest_dbn.py` — iter 1 (triaged via M2.5)
- `pipeline/build_bars_5m.py` — iter 1 (triaged via M2.5)
- `pipeline/dst.py` — iter 1 (triaged via M2.5)
- `pipeline/cost_model.py` — iter 12 (triaged)
- `pipeline/asset_configs.py` — iter 1 (triaged via M2.5)
- `scripts/tools/generate_trade_sheet.py` — iter 18 (3 fixes: dollar gate, join, friction)

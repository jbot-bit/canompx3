# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 200

## RALPH AUDIT — Iteration 200 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 48/48 prop_portfolio tests PASS
## Scope: trading_app/prop_portfolio.py (full file audit)

---

## Full-File Audit Results

### Finding PAPER-PNL-200 — LOW — FIXED

**PREMISE:** `_query_paper_pnl` at line 436 (pre-fix) passed `exc_info=True` as
a positional format argument to `logger.debug`, filling the `%s` placeholder with
the string `"True"`. The exception traceback was discarded. Per integrity-guardian.md
§ 6, every `except Exception` must record the exception.

**TRACE:** `prop_portfolio.py:436` → `except Exception:` (no `as exc`) →
`logger.debug("..._query_paper_pnl failed for %s: %s", strategy_id, exc_info=True)` —
`exc_info=True` fills the second `%s`, no `exc_info` keyword sent to the logging framework.

**Fix:** Removed the stray `: %s` from the format string; retained `exc_info=True`
as a keyword argument only. Two new tests added: (1) verifies `caplog.records` carry
`exc_info` on DB failure, (2) verifies missing `paper_trades` table returns None.

**Doctrine cited:** integrity-guardian.md § 6

**Commit:** 9db13515

---

## Seven Sins Scan — prop_portfolio.py (full)

- Sin 1 (Silent failure): `_query_paper_pnl` silently swallowed traceback. FIXED this iteration.
  `_resolve_daily_lane` at line 291 narrows to `(ValueError, duckdb.Error)` correctly (iter 195).
  `resolve_daily_lanes` at line 411 catches `ImportError` for optional `lane_ctl` module.
  That is an intentional fail-open for optional infrastructure — ACCEPTABLE (Pattern 2: dormant
  infrastructure import, documented with comment `# lane_ctl not available — no overrides`).
- Sin 2 (Fail-open): `_load_daily_snapshot` returns `None` when strategy missing from
  `validated_setups`; caller `_resolve_daily_lane` surfaces this as `status="HOLD"`. Correct.
- Sin 3 (Canonical violation): `DD_PER_CONTRACT_075X/10X` are module-level fallback constants,
  used only when `median_risk_points is None`. The function docstring and comments document
  this explicitly. All `max_dd` limits sourced from `tier.max_dd` via `get_account_tier()`.
  No hardcoded dollar limits in capital-gate paths. CLEAN.
- Sin 4 (Impact awareness): No hardcoded instrument or session lists. Instrument filtering
  uses `firm_spec.banned_instruments` (from `get_firm_spec()`). Session filtering uses
  `profile.allowed_sessions`. Both sourced from canonical `prop_profiles.py`. CLEAN.
- Sin 5 (Evidence over assertion): N/A (audit mode).
- Sin 6 (Spec compliance): No `docs/specs/prop_portfolio.md` exists. Module docstring is accurate.
- Sin 7 (Metadata trust): `snap["status"]` is lowercased at query time (`LOWER(vs.status)`).
  Comparison `snap["status"] != "active"` is safe. CLEAN.

---

## Files Fully Scanned

- pipeline/check_drift.py (iter 153)
- pipeline/build_daily_features.py (iter 158)
- pipeline/dst.py (no-touch, iter 160)
- trading_app/strategy_discovery.py (iter 162)
- trading_app/outcome_builder.py (iter 165)
- trading_app/entry_rules.py (iter 168)
- trading_app/strategy_validator.py (iter 171)
- trading_app/live/session_orchestrator.py (iter 174)
- trading_app/live/execution_engine.py (iter 177)
- trading_app/live/alert_engine.py (iter 180)
- trading_app/derived_state.py (iter 183)
- trading_app/deployability.py (iter 193)
- trading_app/strategy_fitness.py (iter 194)
- trading_app/live_config.py (iter 195)
- trading_app/prop_portfolio.py (iter 200, FULL)
- trading_app/lane_correlation.py (iter 196)
- trading_app/lane_allocator.py (iter 199, full — apply_c8_gate + build_allocation + apply_chordia_gate)
- trading_app/chordia.py (iter 198, full)

---

## Next Iteration Targets

**Priority 1 — Unscanned high/medium centrality files:**
- `trading_app/pre_session_check.py` — capital-class preflight; referenced in deferred finding
  A6-GAP3 (truthy-falsy cap display); unscanned; high centrality (called from live session orchestrator)

**Priority 2 — Open deferred findings (MEDIUM):**
- `ALERT-CONTAM-N2` — test writes to production `data/runtime/operator_alerts.jsonl`; n=2 class
  incident requiring conftest monkeypatch + drift check. Next iteration touching alert_engine or
  any test under tests/test_trading_app/ calling `record_operator_alert`.
- `PR301-TRADO-IDEMPOTENCY` — Tradovate order_router has no idempotency token; retry policy
  risks duplicate orders. Re-check trigger: any Tradovate go-live decision.

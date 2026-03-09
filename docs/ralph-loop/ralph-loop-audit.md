# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 5 (New Targets)
## Date: 2026-03-09
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest tests/ -x -q` | PASS | 2745 passed, 0 failed, 9 skipped (611s) |
| `ruff check` | PASS | All checks passed |

## Target Files
- `trading_app/execution_engine.py` — core trade logic, entry/exit/stop mechanics
- `trading_app/strategy_validator.py` — promotion pipeline
- `trading_app/outcome_builder.py` — outcome computation accuracy
- `pipeline/build_daily_features.py` — feature computation

---

### Finding 1 — FIXED (e465403: size_multiplier applied at all 3 entry model sizing paths)
- Severity: MEDIUM (RESOLVED)
- File: `trading_app/execution_engine.py:466,538`
- Evidence:
```python
# Line 466: size_multiplier is computed from calendar overlay
size_multiplier = action.value  # 0.5 for HALF_SIZE, 1.0 for NEUTRAL

# Line 538: stored on the trade
trade = ActiveTrade(
    ...
    size_multiplier=size_multiplier,
)

# BUT: _compute_contracts() (line 220-244) never reads size_multiplier
# E2 sizing (line 643): trade.contracts = self._compute_contracts(risk_points, cost)
# E1 sizing (line 831): trade.contracts = self._compute_contracts(risk_points, cost)
# E3 sizing (line 969): trade.contracts = self._compute_contracts(risk_points, cost)
# None of these apply size_multiplier.

# paper_trader.py:375 applies it AFTER THE FACT to PnL:
scaled_pnl = trade.pnl_r * trade.size_multiplier  # correct for backtesting
# But execution_engine never reduces contracts for live trading
```
- Root Cause: Calendar overlay's HALF_SIZE action sets `size_multiplier=0.5` on the trade, but position sizing never reads this field. In backtesting (`paper_trader.py`), PnL is scaled down retroactively (correct simulation). In live execution, `contracts` is the actual number sent to the broker — unmodified.
- Blast Radius: **Currently dormant.** `CALENDAR_RULES` is empty (all calendar cascade tests showed NO edge — see `calendar_cascade.md`). No HALF_SIZE rules exist. But the code path is wired and ready — if rules are added, live trading would send full-size orders when the system expects half-size.
- Fix Category: validation (apply `size_multiplier` to contracts in `_compute_contracts` or after sizing)

---

### Finding 2 — FIXED (e465403: broken month boundary signals disabled, unused imports removed)
- Severity: MEDIUM (RESOLVED)
- File: `trading_app/calendar_overlay.py:96-99,114-119`
- Evidence:
```python
# Line 96-99: Acknowledged TODO
# TODO: _EMPTY_TRADING_DAYS causes _is_month_end/_is_month_start/_is_quarter_end
# to always return True (count=0 <= window). No impact while CALENDAR_RULES is
# empty, but must be fixed before any MONTH_END/MONTH_START rules go live.
_EMPTY_TRADING_DAYS: set[date] = set()

# Lines 114-119: These signal detectors always fire
if is_month_end(trading_day, _EMPTY_TRADING_DAYS):    # Always True!
    signals.append("MONTH_END")
if is_month_start(trading_day, _EMPTY_TRADING_DAYS):  # Always True!
    signals.append("MONTH_START")
if is_quarter_end(trading_day, _EMPTY_TRADING_DAYS):  # Always True!
    signals.append("QUARTER_END")
```
- Root Cause: `_EMPTY_TRADING_DAYS` is an empty set. The boundary-detection functions check `count <= window` where count is computed from the empty set (always 0), so the check always passes.
- Blast Radius: **Currently dormant.** Same guard as Finding 1: `CALENDAR_RULES` is empty, so these false-positive signals are never matched against rules. If MONTH_END/MONTH_START/QUARTER_END rules are added, they would fire on every trading day.
- Fix Category: validation (populate `_EMPTY_TRADING_DAYS` from DB or remove the signals until properly wired)

---

### Finding 3 — FIXED: Annotation debt in build_daily_features.py
- Severity: MEDIUM (RESOLVED)
- File: `pipeline/build_daily_features.py` (multiple locations)
- Evidence:
```python
# ATR velocity regime cutoffs (lines 1119-1121) — directly gate trade entry
if vel > 1.05: regime = "Expanding"
elif vel < 0.95: regime = "Contracting"
else: regime = "Neutral"

# Compression z-score thresholds (lines 1151-1154) — gate trade entry
if z < -0.5: tier = "Compressed"
elif z > 0.5: tier = "Expanded"
else: tier = "Neutral"

# Day type classification (lines 555-560) — used in research analysis
if range_pct < 0.5: "NON_TREND"
if close_pct >= 0.7: "TREND_UP"

# RSI lookback (line 638): 200 bars with no annotation
# Min prior days (lines 1113-1114): >= 5 days with no annotation
```
- Root Cause: These thresholds lack `@research-source` and `@revalidated-for` annotations required by integrity-guardian rule #8. Drift check #45 enforces annotations in `config.py` but does not cover `build_daily_features.py`.
- Blast Radius: ATR velocity (1.05/0.95) and compression z-score (±0.5) directly control trade entry via `config.py` ATR_VELOCITY_OVERLAY filter. If someone changes these thresholds without drift check validation, trading behavior changes silently. The thresholds are defensible (1.05/0.95 = ±5% change, ±0.5σ = half standard deviation) but undocumented.
- Fix Category: refactor (add annotations, extend drift check coverage)

---

### Finding 4 — FIXED: Risk fallback to min_risk_floor_points now logs warning
- Severity: LOW (RESOLVED)
- File: `trading_app/strategy_validator.py:454-462`
- Evidence:
```python
median_risk = row.get("median_risk_points")
avg_risk = row.get("avg_risk_points")
if median_risk is not None and median_risk > 0:
    strategy_risk_points = median_risk
elif avg_risk is not None and avg_risk > 0:
    strategy_risk_points = avg_risk
else:
    strategy_risk_points = cost_spec.min_risk_floor_points  # Silent fallback
```
- Root Cause: If both `median_risk_points` and `avg_risk_points` are None/0, the stress test denominator silently falls to `min_risk_floor_points` (tick-based floor, e.g. ~1 tick). This compresses `extra_cost_per_trade_r`, making the stress test weaker. In practice, strategy_discovery always populates these from outcome data, so this fallback should never fire. But if it does fire (broken discovery), the stress test becomes a no-op with no warning.
- Blast Radius: Low in practice. A strategy with missing risk data would need to pass Phases 1-3 first, which require valid trade data. But the silent fallback means a discovery bug would propagate through Phase 4 without warning.
- Fix Category: logging (add warning log when fallback fires)

---

### Finding 5 — SKIPPED (regime waiver uses year-level ATR — methodological note, not bug)
- Severity: LOW (NOT ACTIONABLE)
- File: `trading_app/strategy_validator.py:391-428`
- Evidence: Regime waivers use `atr_by_year[year]` (annual mean ATR) to classify market regime, but apply the waiver to session-level negative performance.
- Assessment: Year-level ATR is a reasonable proxy for "was the market dormant that year?" in a post-hoc validation pass. This is not live-path look-ahead — it's a robustness judgment applied after all data is known. The waiver already requires `trades <= 5` in the DORMANT year. Flagging this as data leak would be a false positive.

---

## Summary
- Total findings: 5
- CRITICAL: 0, HIGH: 0, MEDIUM: 3 (ALL FIXED), LOW: 1 (FIXED), SKIPPED: 1
- Findings 1+2 FIXED in batch (e465403). Findings 3+4 FIXED in batch (iteration 7).
- outcome_builder.py: **CLEAN** — entry models correct, cost handling sound, no look-ahead
- execution_engine.py: State machine logic sound, E3 stop-before-fill correct, ambiguous bar → conservative loss correct

## Severity Counts

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH     | 0 |
| MEDIUM   | 3 |
| LOW      | 1 |

## What Was NOT Flagged (Anti-False-Positive Notes)
- E3 stop-before-fill guard (`execution_engine.py:937-949`): Correctly prevents dead fills by checking stop breach before limit fill.
- Ambiguous bar → conservative loss (`execution_engine.py:1124-1126`): Standard industry practice when both target and stop hit on same bar.
- IB conditional exit logic: Sound state transitions (ib_pending → hold_7h or opposed kill).
- Session cost lookup consistency: `get_session_cost_spec()` used in all 3 entry model paths + exit path.
- Paper_trader `size_multiplier` PnL scaling: Correct backtesting approximation (scales portfolio contribution, not per-trade R).
- `searchsorted(side="right")` in RSI fast path: Consistent with slow-path `ts_utc <= ?::TIMESTAMPTZ`. Both include the 09:00 bar. Impact is 1 bar out of 200 = negligible on RSI value. All sessions start ≥55 min after this bar closes.
- `except Exception` in GARCH fit (`build_daily_features.py:601`): Returns None on convergence failure — GARCH is informational, not trade-gating. Downstream handles None.
- `fillna(-999.0)` sentinel pattern: Domain-intentional, documented.
- Strategy_discovery always populates `median_risk_points` from outcome data: Finding 4's fallback is a safety net, not a live bug.

## Next Targets
- `trading_app/live/session_orchestrator.py` — largest blast radius in live path, complex state machine (re-scan after iteration 1 fixes)
- `trading_app/live/cusum_monitor.py` — drift detection feeding kill decisions
- `trading_app/scoring.py` — market state scoring, MIN_SCORE_THRESHOLD
- `trading_app/portfolio.py` — position sizing, vol scaling (Carver Ch.9 implementation)

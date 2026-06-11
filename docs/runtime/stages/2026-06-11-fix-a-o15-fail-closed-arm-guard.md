---
task: "Fix A — make the O15→O5 daily_features fallback in execution_engine arming fail-closed"
mode: IMPLEMENTATION
scope_lock:
  - trading_app/execution_engine.py
  - tests/test_trading_app/test_execution_engine.py
---

## Task

Qwen Claim 2 (O15→O5 silent feature fallback) traced to ground truth:
`trading_app/execution_engine.py:730` silently substitutes the O5 `daily_features`
row when a strategy's `orb_minutes` aperture has no row in `_daily_features_rows`:

```python
om_row = self._daily_features_rows.get(strategy.orb_minutes, self._daily_features_rows.get(5))
```

This is **fail-open** — a deployed O15 lane would arm on **wrong-aperture (O5) features**
with no log if O15 freshness ever lapses. It never fires today (O15 `daily_features`
fresh to 06-09 per the grounding pass), but it is a latent capital-path landmine.

## Fix

Make it **explicit + fail-closed**, mirroring the existing fail-closed `filter_type`
pattern directly above at `:720-728`: if `strategy.orb_minutes` has no row in
`_daily_features_rows`, **skip arming this strategy and log an error**. Do NOT
cross-aperture fall back to O5.

## Blast Radius

- `trading_app/execution_engine.py` — one arming loop in `_check_breaks`/arming method.
  Reads `_daily_features_rows` (a dict[int, dict]); no DB/schema change; no live
  order-path change. `om_row` is consumed downstream at `:732-734` (row build) and
  `:753-757` (ATR velocity), both already `None`-guarded — skipping early via
  `continue` is clean and removes the only path that produced a non-None wrong-aperture row.
- Behavioral change: an O15 strategy whose O15 row is absent goes from
  **silently armed on O5 data** → **skipped + error logged**. For O5 strategies and
  for any strategy whose aperture row IS present, behavior is byte-identical.
- `tests/test_trading_app/test_execution_engine.py` — new test asserting the skip+log.
- Reads: none beyond in-memory engine state. Writes: none.

## Verification

- New TDD test (RED first): O15 strategy + `_daily_features_rows={5: row}` (O15 absent)
  → assert NOT armed (0 ENTRY events) + error logged.
- Regression: existing `test_execution_engine.py` suite still green (O5 path unchanged).
- `python pipeline/check_drift.py` green.
- Adversarial-audit gate (independent context) before "done" — capital-path arming logic.

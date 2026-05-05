---
task: Sweep paper_trader.py for orb_minutes=5 class bug — fix _inject_cross_asset_atrs_for_replay + remove default-bait param
mode: IMPLEMENTATION
scope_lock:
  - trading_app/paper_trader.py
  - tests/test_trading_app/test_paper_trader.py
---

## Blast Radius

- `trading_app/paper_trader.py` — fix `_inject_cross_asset_atrs_for_replay()` to require `orb_minutes` and use it in the cross-asset query (line 180). Plumb `om` from the loop at line 344. Remove default `orb_minutes: int = 5` from `_get_daily_features_row()` signature (line 146) — the only caller (line 341) already passes `orb_minutes=om` explicitly, so the default is recurrence-bait per the F1 institutional-rigor rule "no dead parameters" and audit feedback.
- `tests/test_trading_app/test_paper_trader.py` — add regression test asserting `_inject_cross_asset_atrs_for_replay` requires `orb_minutes` and routes per-aperture. Add signature test asserting `_get_daily_features_row` has no default for `orb_minutes`.
- Reads: `gold.db` (read-only) for daily_features rows.
- Writes: none directly. Affects ReplayResult journal entries only — replay output for any non-O5 strategy where the cross-asset source instrument's o5 vs o15/o30 atr_20_pct differs.
- Does NOT touch: `_get_trading_days` (line 122 — DISTINCT trading_day, canonical CTE-Guard dedup), `_get_median_atr_20` (line 138 — non-aperture-specific atr_20 scalar, canonical CTE-Guard dedup). Both verified per `daily-features-joins.md` rule.

## Why

Followup to PR #231. The `paper_trade_logger.py` fix landed `_inject_cross_asset_atrs(orb_minutes)`. Sibling `paper_trader.py` has the identical pattern at line 165/180 — `_inject_cross_asset_atrs_for_replay` reads `orb_minutes=5` for cross-asset atr_20_pct regardless of the consuming strategy's aperture. The loop at line 340 already iterates `unique_om`; just needs to plumb `om` to line 344.

Additional finding from line 146: `_get_daily_features_row(orb_minutes: int = 5)` carries a default that the only caller (line 341) overrides correctly. Per institutional-rigor rule 5 (no dead parameters / drift bait), remove the default — force callers to be explicit. Aligns with the F1 PR #231 pattern where both helpers REQUIRE the parameter.

## Acceptance

- `_inject_cross_asset_atrs_for_replay(con, row, instrument, trading_day, orb_minutes)` — orb_minutes required (no default).
- `_get_daily_features_row(con, instrument, trading_day, orb_minutes)` — orb_minutes required (no default).
- Caller at line 344 passes `om`.
- Regression tests assert both signatures require orb_minutes (no default).
- `python pipeline/check_drift.py` passes.
- Existing tests in `tests/test_trading_app/test_paper_trader.py` still pass.

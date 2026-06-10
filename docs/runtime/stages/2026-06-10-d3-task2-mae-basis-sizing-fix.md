task: D-3 seam Task 2 — fix MAE-basis sizing bug in survival sim (account_survival.py)
mode: IMPLEMENTATION

## Scope Lock
- trading_app/account_survival.py
- tests/test_trading_app/test_account_survival.py
- pipeline/check_drift.py

## Blast Radius
- account_survival.py — carry planned `risk_points` on the internal `TradePath` dataclass; populate at the prod constructor (:521) from the value already computed at :513; consume it in `_load_lane_trade_paths` sizing loop (:581) instead of re-deriving from realized MAE. `replace()` (:613) must NOT scale it (per-contract price distance, invariant to n). Reads gold.db read-only; writes nothing; no schema change; no `trading_app/live/` file touched.
- test_account_survival.py — add the new required `risk_points` field to all 6 `TradePath(...)` sites; migrate the 2 sizing-oracle tests to drive n via `risk_points` not `mae_dollars`; add one value-oracle test pinning value-parity (engine basis vs old mae basis diverge→converge).
- check_drift.py — strengthen `check_survival_engine_sizer_parity` (:14181) from import-parity to ALSO assert the survival sizing loop does not derive risk_points from realized MAE (fail-closed).
- Capital path: C11 survival GATE (not the order path). At DEPLOYED_MAX_CONTRACTS_CLAMP=1 the fix is byte-identical (n floors to 1). Activates only on a clamp lift (Task 3). CRIT/HIGH → adversarial-audit gate MANDATORY before commit.

task: D-3 re-baseline 2 stale survival goldens (DB backfill O15/O30) + freeze gate-flip proof onto synthetic fixture
mode: IMPLEMENTATION

## Scope Lock
- tests/test_trading_app/test_account_survival.py

## Blast Radius
- test_account_survival.py — TEST-ONLY change. (1) Re-baseline `_D3_GOLDEN_CAP1_ROLLING_DD` 1535.22 -> 620.68 (byte-exact pin, operator-chosen: a survival tripwire that re-fails on every backfill is intended). (2) Update the measured-context comment block (n_scen 2048->2067, cap2 dd 3568.02->1241.42, ratio 2.32x->2.00x, both pass at current DB). (3) Split `test_d3_rolling_dd_scales_superlinearly_and_pass_prob_drops_at_cap2`: keep the live DD-ratio property assert ([2.0,2.6]); MOVE the "gate fails closed at unsafe size" proof (p1>0.9, p2<0.5) onto a NEW synthetic `list[DailyScenario]` fixture with engineered drawdown, decoupled from gold.db.
- Reads gold.db read-only (live-reconciliation tests). Synthetic fixture touches NO DB. NO production code change (MAE fix already landed 83400755). NO check_drift.py change (survival sizer parity guard already correct).
- Capital path: C11 survival GATE test. New values VERIFIED legitimate this session — current trade dist n=599 WR=51.1% no corruption signature; lower DD is genuine backfill effect not regression. Re-baselining DOWNWARD reviewed: safe because the gate-flip fail-closed proof is preserved on the synthetic fixture, not weakened.

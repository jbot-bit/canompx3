# MNQ Open Book Risk Overlay v1

**Prereg:** `docs\audit\hypotheses\2026-06-01-mnq-open-book-risk-overlay-v1.yaml`
**Status:** conditional execution/filter/account-fit test; no deployment claim.
**Family K:** `28` fixed candidates.
**Selection window:** `< 2026-01-01`; 2026 rows are descriptive only.
**Profile:** `topstep_50k_mnq_auto`; daily belt `$450`, max DD `$2000`.

## Scope

[MEASURED] No overlay candidate is profile-safe under the active account constraints.
[MEASURED] Highest annual-dollar candidate is `RAW_ANNUAL_RR2_NO_FILTER__RISK_CAP_300` with annual $8257.55, DD $3250.16, daily-belt breaches 25.
[MEASURED] Lowest positive-DD candidate is `LOW_DD_RR1_NO_FILTER__RISK_CAP_225` with annual $6946.82, DD $2079.26, daily-belt breaches 0.

## Reproduction

- Front door: `python scripts/tools/prereg_front_door.py --hypothesis-file docs\audit\hypotheses\2026-06-01-mnq-open-book-risk-overlay-v1.yaml --execute --runner research/mnq_open_book_risk_overlay_v1.py --format text`
- Candidate CSV: `docs\audit\results\2026-06-01-mnq-open-book-risk-overlay-v1-candidates.csv`
- DB mode: read-only canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`.

## Candidate Ranking

| candidate | book | overlay | n_is_days | active_trade_days_is | skipped_trades | annual_dollars | max_dd_dollars | worst_day_dollars | hist_daily_belt_breaches | operational_survival | mean_2026_dollars | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `RAW_ANNUAL_RR2_NO_FILTER__RISK_CAP_300` | `NYOPEN_USDATA_RR2_NO_FILTER` | `RISK_CAP_300` | 1719 | 1694 | 0 | 8257.5497 | 3250.1604 | -534.8400 | 25 | 0.2227 | 66.7789 | `KILL` |
| `CURRENT_RR2_COST_LT10__RISK_CAP_300` | `NYOPEN_USDATA_RR2_COST_LT10` | `RISK_CAP_300` | 1719 | 1679 | 0 | 8243.4185 | 3250.1604 | -534.8400 | 25 | 0.2231 | 66.7789 | `KILL` |
| `CURRENT_RR2_COST_LT10__REALIZED_LOSS_THROTTLE` | `NYOPEN_USDATA_RR2_COST_LT10` | `REALIZED_LOSS_THROTTLE` | 1719 | 1704 | 215 | 7967.2917 | 3276.6028 | -866.3400 | 48 | 0.0676 | 59.4418 | `KILL` |
| `RAW_ANNUAL_RR2_NO_FILTER__RAW` | `NYOPEN_USDATA_RR2_NO_FILTER` | `RAW` | 1719 | 1719 | 0 | 7961.5641 | 3548.7697 | -866.3400 | 62 | 0.0317 | 70.7396 | `KILL` |
| `RAW_ANNUAL_RR2_NO_FILTER__REALIZED_LOSS_THROTTLE` | `NYOPEN_USDATA_RR2_NO_FILTER` | `REALIZED_LOSS_THROTTLE` | 1719 | 1719 | 218 | 7958.0787 | 3276.6028 | -866.3400 | 48 | 0.0676 | 59.4418 | `KILL` |
| `CURRENT_RR2_COST_LT10__RAW` | `NYOPEN_USDATA_RR2_COST_LT10` | `RAW` | 1719 | 1704 | 0 | 7947.4329 | 3548.7697 | -866.3400 | 62 | 0.0318 | 70.7396 | `KILL` |
| `COMPROMISE_RR1_5_NO_FILTER__RISK_CAP_300` | `NYOPEN_USDATA_RR1_5_NO_FILTER` | `RISK_CAP_300` | 1719 | 1694 | 0 | 7939.1194 | 2660.2323 | -534.8400 | 25 | 0.2294 | 67.4562 | `KILL` |
| `RAW_ANNUAL_RR2_NO_FILTER__RISK_CAP_225` | `NYOPEN_USDATA_RR2_NO_FILTER` | `RISK_CAP_225` | 1719 | 1627 | 0 | 7879.1026 | 2583.4074 | -414.8400 | 0 | 0.8377 | 31.9565 | `KILL` |
| `CURRENT_RR2_COST_LT10__RISK_CAP_225` | `NYOPEN_USDATA_RR2_COST_LT10` | `RISK_CAP_225` | 1719 | 1612 | 0 | 7864.9714 | 2583.4074 | -414.8400 | 0 | 0.8374 | 31.9565 | `KILL` |
| `COMPROMISE_RR1_5_NO_FILTER__RISK_CAP_225` | `NYOPEN_USDATA_RR1_5_NO_FILTER` | `RISK_CAP_225` | 1719 | 1627 | 0 | 7693.5928 | 2448.2566 | -414.8400 | 0 | 0.8656 | 30.2066 | `KILL` |
| `COMPROMISE_RR1_5_NO_FILTER__RAW` | `NYOPEN_USDATA_RR1_5_NO_FILTER` | `RAW` | 1719 | 1719 | 0 | 7683.8501 | 3389.3408 | -866.3400 | 59 | 0.0382 | 71.5301 | `KILL` |
| `COMPROMISE_RR1_5_NO_FILTER__REALIZED_LOSS_THROTTLE` | `NYOPEN_USDATA_RR1_5_NO_FILTER` | `REALIZED_LOSS_THROTTLE` | 1719 | 1719 | 218 | 7642.6133 | 2770.7074 | -866.3400 | 45 | 0.0810 | 59.5752 | `KILL` |
| `LOW_DD_RR1_NO_FILTER__RISK_CAP_300` | `NYOPEN_USDATA_RR1_NO_FILTER` | `RISK_CAP_300` | 1719 | 1694 | 0 | 7373.7232 | 2211.8936 | -534.8400 | 23 | 0.2670 | 72.7326 | `KILL` |
| `LOW_DD_RR1_NO_FILTER__RISK_CAP_225` | `NYOPEN_USDATA_RR1_NO_FILTER` | `RISK_CAP_225` | 1719 | 1627 | 0 | 6946.8239 | 2079.2570 | -414.8400 | 0 | 0.8953 | 35.7821 | `KILL` |
| `LOW_DD_RR1_NO_FILTER__REALIZED_LOSS_THROTTLE` | `NYOPEN_USDATA_RR1_NO_FILTER` | `REALIZED_LOSS_THROTTLE` | 1719 | 1719 | 217 | 6866.6097 | 2912.9723 | -866.3400 | 41 | 0.1025 | 70.8779 | `KILL` |
| `LOW_DD_RR1_NO_FILTER__RAW` | `NYOPEN_USDATA_RR1_NO_FILTER` | `RAW` | 1719 | 1719 | 0 | 6799.3516 | 2710.7078 | -866.3400 | 55 | 0.0489 | 87.9261 | `KILL` |
| `COMPROMISE_RR1_5_NO_FILTER__STOP_075_RISK_CAP_225` | `NYOPEN_USDATA_RR1_5_NO_FILTER` | `STOP_075_RISK_CAP_225` | 1719 | 1694 | 0 | 6674.0526 | 2999.0931 | -401.1300 | 0 | 0.7922 | 56.1353 | `KILL` |
| `CURRENT_RR2_COST_LT10__STOP_075_RISK_CAP_225` | `NYOPEN_USDATA_RR2_COST_LT10` | `STOP_075_RISK_CAP_225` | 1719 | 1679 | 0 | 6639.4163 | 3300.3957 | -401.1300 | 0 | 0.7559 | 52.8120 | `KILL` |
| `RAW_ANNUAL_RR2_NO_FILTER__STOP_075_RISK_CAP_225` | `NYOPEN_USDATA_RR2_NO_FILTER` | `STOP_075_RISK_CAP_225` | 1719 | 1694 | 0 | 6635.7313 | 3300.3957 | -401.1300 | 0 | 0.7556 | 52.8120 | `KILL` |
| `LOW_DD_RR1_NO_FILTER__STOP_075_RISK_CAP_225` | `NYOPEN_USDATA_RR1_NO_FILTER` | `STOP_075_RISK_CAP_225` | 1719 | 1694 | 0 | 6411.1273 | 3076.6112 | -401.1300 | 0 | 0.8480 | 54.8423 | `KILL` |
| `COMPROMISE_RR1_5_NO_FILTER__STOP_075_RISK_CAP_300` | `NYOPEN_USDATA_RR1_5_NO_FILTER` | `STOP_075_RISK_CAP_300` | 1719 | 1714 | 0 | 5930.8407 | 5143.5741 | -537.6300 | 11 | 0.4203 | 66.9330 | `KILL` |
| `CURRENT_RR2_COST_LT10__STOP_075_RISK_CAP_300` | `NYOPEN_USDATA_RR2_COST_LT10` | `STOP_075_RISK_CAP_300` | 1719 | 1699 | 0 | 5879.0301 | 4521.2125 | -537.6300 | 12 | 0.3784 | 63.4965 | `KILL` |
| `RAW_ANNUAL_RR2_NO_FILTER__STOP_075_RISK_CAP_300` | `NYOPEN_USDATA_RR2_NO_FILTER` | `STOP_075_RISK_CAP_300` | 1719 | 1714 | 0 | 5875.3451 | 4539.9025 | -537.6300 | 12 | 0.3785 | 63.4965 | `KILL` |
| `COMPROMISE_RR1_5_NO_FILTER__STOP_075` | `NYOPEN_USDATA_RR1_5_NO_FILTER` | `STOP_075` | 1719 | 1719 | 0 | 5725.3136 | 7038.5755 | -937.0050 | 26 | 0.1978 | 47.9404 | `KILL` |
| `CURRENT_RR2_COST_LT10__STOP_075` | `NYOPEN_USDATA_RR2_COST_LT10` | `STOP_075` | 1719 | 1704 | 0 | 5703.8123 | 6286.4167 | -937.0050 | 28 | 0.1721 | 44.5040 | `KILL` |
| `RAW_ANNUAL_RR2_NO_FILTER__STOP_075` | `NYOPEN_USDATA_RR2_NO_FILTER` | `STOP_075` | 1719 | 1719 | 0 | 5700.1273 | 6305.1067 | -937.0050 | 28 | 0.1724 | 44.5040 | `KILL` |
| `LOW_DD_RR1_NO_FILTER__STOP_075_RISK_CAP_300` | `NYOPEN_USDATA_RR1_NO_FILTER` | `STOP_075_RISK_CAP_300` | 1719 | 1714 | 0 | 5457.7360 | 5579.2309 | -537.6300 | 10 | 0.4682 | 74.8588 | `KILL` |
| `LOW_DD_RR1_NO_FILTER__STOP_075` | `NYOPEN_USDATA_RR1_NO_FILTER` | `STOP_075` | 1719 | 1719 | 0 | 5340.3346 | 6417.1846 | -937.0050 | 24 | 0.2356 | 55.8662 | `KILL` |

## Grounding

- `docs/institutional/conditional-edge-framework.md`: this is an execution/filter/allocator role test, so policy/account EV and drawdown matter more than selected-trade mean.
- `docs/institutional/pre_registered_criteria.md` Criterion 11: funded deployment requires account-death Monte Carlo with prop-firm rules and >=70% 90-day survival.
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`: risk analysis, portfolio construction, and bet-sizing are legitimate finite-data applications when trial count is tracked.
- `TRADING_RULES.md`: active lanes use E2/CB1, 0.75x stop sizing, and explicit ORB/risk caps; stop-multiplier math is repo-canonical via `apply_tight_stop`.

## Caveats

- Stop-multiplier candidates use canonical MAE-based tight-stop math, but do not infer exact intra-bar kill timestamps.
- Realized-loss throttle only skips later trades when a prior losing trade has an exit timestamp before the later entry. It is deliberately not combined with tight-stop candidates because tight-stop kill timestamps are unavailable.
- Risk caps are structural fractions of the $450 daily belt, not optimized thresholds.
- This is one-contract-per-leg account-fit research, not a live allocation patch.

## Verdict

`KILL` for every row: risk caps can remove daily-belt breaches, but the two-leg book still exceeds the drawdown budget. The correct next move is lower-risk lane replacement, not more sizing on this two-leg book.

SURVIVED SCRUTINY: finite K=28, pre-entry risk caps, canonical stop-multiplier math, no 2026 tuning, no silent pnl-null dropout.
DID NOT SURVIVE: no live/deployment claim from this research artifact alone.

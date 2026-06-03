# MNQ US_DATA_1000 Capital Fit v1

**Prereg:** `docs\audit\hypotheses\2026-06-01-mnq-usdata-capital-fit-v1.yaml`
**Leg-choice source:** `docs\audit\results\2026-06-01-mnq-usdata-rr-leg-choice-v1-books.csv`
**Status:** allocator/account-fit follow-up; no deployment claim.
**Selection window:** `< 2026-01-01`; 2026 rows are descriptive only.
**Profile:** `topstep_50k_mnq_auto`; daily belt `$450`, max DD `$2000`.

## Scope

[MEASURED] No book has a profile-safe contract size from 1-10 MNQ contracts per leg under the active account constraints.
[MEASURED] Raw 1-contract annual-dollar winner is `NYOPEN_USDATA_RR2_NO_FILTER` with annual $7961.56, but it is not profile-safe.
[MEASURED] Current comparison row `NYOPEN_USDATA_RR2_COST_LT10` has best safe size 0, annual $NA, DD $NA, survival 0.0000.

## Reproduction

- Front door: `python scripts/tools/prereg_front_door.py --hypothesis-file docs\audit\hypotheses\2026-06-01-mnq-usdata-capital-fit-v1.yaml --execute --runner research/mnq_usdata_capital_fit_v1.py --format text`
- Book CSV: `docs\audit\results\2026-06-01-mnq-usdata-capital-fit-v1-books.csv`
- Sizing CSV: `docs\audit\results\2026-06-01-mnq-usdata-capital-fit-v1-sizing.csv`
- DB mode: read-only canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`.

## Book Ranking

| strategy | us_data_rr | us_data_filter | n_is_days | annual_dollars_1ct | max_dd_dollars_1ct | best_contracts_per_leg | profile_safe_annual_dollars | profile_safe_max_dd_dollars | profile_safe_survival | capital_objective | mean_2026_dollars_1ct | verdict |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `NYOPEN_USDATA_RR1_NO_FILTER` | 1.0000 | `NO_FILTER` | 1719 | 6799.3516 | 2710.7078 | 0 | NA | NA | 0.0000 | NA | 87.9261 | `KILL` |
| `NYOPEN_USDATA_RR1_COST_LT08` | 1.0000 | `COST_LT08` | 1719 | 6747.7883 | 2710.7078 | 0 | NA | NA | 0.0000 | NA | 87.9261 | `KILL` |
| `NYOPEN_USDATA_RR1_COST_LT10` | 1.0000 | `COST_LT10` | 1719 | 6768.5801 | 2710.7078 | 0 | NA | NA | 0.0000 | NA | 87.9261 | `KILL` |
| `NYOPEN_USDATA_RR1_COST_LT12` | 1.0000 | `COST_LT12` | 1719 | 6780.5431 | 2710.7078 | 0 | NA | NA | 0.0000 | NA | 87.9261 | `KILL` |
| `NYOPEN_USDATA_RR1_COST_LT15` | 1.0000 | `COST_LT15` | 1719 | 6782.6747 | 2710.7078 | 0 | NA | NA | 0.0000 | NA | 87.9261 | `KILL` |
| `NYOPEN_USDATA_RR1_5_NO_FILTER` | 1.5000 | `NO_FILTER` | 1719 | 7683.8501 | 3389.3408 | 0 | NA | NA | 0.0000 | NA | 71.5301 | `KILL` |
| `NYOPEN_USDATA_RR1_5_COST_LT08` | 1.5000 | `COST_LT08` | 1719 | 7610.2223 | 3389.3408 | 0 | NA | NA | 0.0000 | NA | 71.5301 | `KILL` |
| `NYOPEN_USDATA_RR1_5_COST_LT10` | 1.5000 | `COST_LT10` | 1719 | 7664.7339 | 3389.3408 | 0 | NA | NA | 0.0000 | NA | 71.5301 | `KILL` |
| `NYOPEN_USDATA_RR1_5_COST_LT12` | 1.5000 | `COST_LT12` | 1719 | 7664.4191 | 3389.3408 | 0 | NA | NA | 0.0000 | NA | 71.5301 | `KILL` |
| `NYOPEN_USDATA_RR1_5_COST_LT15` | 1.5000 | `COST_LT15` | 1719 | 7667.0638 | 3389.3408 | 0 | NA | NA | 0.0000 | NA | 71.5301 | `KILL` |
| `NYOPEN_USDATA_RR2_NO_FILTER` | 2.0000 | `NO_FILTER` | 1719 | 7961.5641 | 3548.7697 | 0 | NA | NA | 0.0000 | NA | 70.7396 | `KILL` |
| `NYOPEN_USDATA_RR2_COST_LT08` | 2.0000 | `COST_LT08` | 1719 | 7880.9009 | 3548.7697 | 0 | NA | NA | 0.0000 | NA | 70.7396 | `KILL` |
| `NYOPEN_USDATA_RR2_COST_LT10` | 2.0000 | `COST_LT10` | 1719 | 7947.4329 | 3548.7697 | 0 | NA | NA | 0.0000 | NA | 70.7396 | `KILL` |
| `NYOPEN_USDATA_RR2_COST_LT12` | 2.0000 | `COST_LT12` | 1719 | 7948.4732 | 3548.7697 | 0 | NA | NA | 0.0000 | NA | 70.7396 | `KILL` |
| `NYOPEN_USDATA_RR2_COST_LT15` | 2.0000 | `COST_LT15` | 1719 | 7943.2017 | 3548.7697 | 0 | NA | NA | 0.0000 | NA | 70.7396 | `KILL` |

## Grounding

- `trading_app.prop_profiles`: active `topstep_50k_mnq_auto` account size, daily loss belt, and express-funded flag.
- `trading_app.topstep_scaling_plan`: canonical MNQ micro-to-mini lot conversion and XFA day-one lot cap.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`: 2026 is descriptive only; no OOS tuning.

## Caveats

- Bootstrap is daily-close accounting; it does not replay intraday MAE/MFE path ordering.
- This pass does not apply stop_multiplier=0.75, max-ORB-size caps, or a sequential daily-loss throttle; those are separate risk-overlay hypotheses.
- Contract sizing is an allocator curve, not an alpha-family p-value test.
- Any future `NARROW` row from this route would still inherit the upstream research-status limit from the leg-choice run; this is not live approval.

## Verdict

`KILL` for raw two-leg deployment under the active profile constraints. The next test should be a risk-overlay family: stop-multiplier, max-risk/day, or sequential daily-loss throttle.

SURVIVED SCRUTINY: fixed 15-book universe, repo-owned profile constraints, Topstep scaling cap, 2026 monitoring separation.
DID NOT SURVIVE: no row becomes deployment-ready without a separate live/readiness translation.

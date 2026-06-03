# MNQ US_DATA_1000 RR Leg Choice v1

**Prereg:** `docs/audit/hypotheses/2026-06-01-mnq-usdata-rr-leg-choice-v1.yaml`
**Status:** narrow post-exploratory follow-up; no deployment claim.
**Canonical inputs:** `orb_outcomes` + `daily_features`; 2026 rows are descriptive only.
**Selection window:** `< 2026-01-01`.
**Family K:** `15` US_DATA_1000 RR/filter books.
**Inherited exploratory burden:** `1755` upstream cells.

## Scope

[MEASURED] This test fixes the NYSE_OPEN anchor leg and only varies the US_DATA_1000 O15 E2 RR/filter leg.
[MEASURED] Objective-score winner is `NYOPEN_USDATA_RR1_NO_FILTER` with annual R 58.5184, DD 13.9546, and objective 4.1935.
[MEASURED] Annual-only sensitivity winner is `NYOPEN_USDATA_RR2_NO_FILTER` with annual R 68.4976 and DD 24.4240.
[MEASURED] Current comparison row `NYOPEN_USDATA_RR2_COST_LT10` has annual R 67.5896, DD 23.4240, and objective 2.8855.

## Reproduction

- Front door: `python scripts/tools/prereg_front_door.py --hypothesis-file docs/audit/hypotheses/2026-06-01-mnq-usdata-rr-leg-choice-v1.yaml --execute --runner research/mnq_usdata_rr_leg_choice_v1.py --format text`
- Book CSV: `docs/audit/results/2026-06-01-mnq-usdata-rr-leg-choice-v1-books.csv`
- DB mode: read-only canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`.

## Candidate Books

| strategy | us_data_rr | us_data_filter | n_is | annual_r | dd | objective_score | t | q_family | dsr_family | dsr_inherited | wfe | mean_2026 | leg_corr_is | verdict |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `NYOPEN_USDATA_RR1_NO_FILTER` | 1.0000 | `NO_FILTER` | 1719 | 58.5184 | 13.9546 | 4.1935 | 5.7020 | 0.000000 | 1.000000 | 0.022305 | 1.1154 | 0.4189 | 0.1357 | `NARROW` |
| `NYOPEN_USDATA_RR1_COST_LT15` | 1.0000 | `COST_LT15` | 1708 | 57.4660 | 13.9546 | 4.1181 | 5.6031 | 0.000000 | 1.000000 | 0.018592 | 1.1162 | 0.4189 | 0.1371 | `NARROW` |
| `NYOPEN_USDATA_RR1_COST_LT12` | 1.0000 | `COST_LT12` | 1705 | 57.3373 | 13.9546 | 4.1088 | 5.5985 | 0.000000 | 1.000000 | 0.018699 | 1.1280 | 0.4189 | 0.1365 | `NARROW` |
| `NYOPEN_USDATA_RR1_COST_LT10` | 1.0000 | `COST_LT10` | 1704 | 56.9026 | 13.9546 | 4.0777 | 5.5640 | 0.000000 | 1.000000 | 0.017236 | 1.1489 | 0.4189 | 0.1371 | `NARROW` |
| `NYOPEN_USDATA_RR1_COST_LT08` | 1.0000 | `COST_LT08` | 1703 | 56.2168 | 13.9546 | 4.0285 | 5.5207 | 0.000000 | 1.000000 | 0.015504 | 1.1709 | 0.4189 | 0.1377 | `NARROW` |
| `NYOPEN_USDATA_RR1_5_NO_FILTER` | 1.5000 | `NO_FILTER` | 1719 | 65.7857 | 18.1695 | 3.6207 | 5.7503 | 0.000000 | 1.000000 | 0.024351 | 1.0599 | 0.3678 | 0.1926 | `NARROW` |
| `NYOPEN_USDATA_RR1_5_COST_LT15` | 1.5000 | `COST_LT15` | 1708 | 64.6663 | 18.1695 | 3.5591 | 5.6597 | 0.000000 | 1.000000 | 0.020749 | 1.0614 | 0.3678 | 0.1942 | `NARROW` |
| `NYOPEN_USDATA_RR1_5_COST_LT10` | 1.5000 | `COST_LT10` | 1704 | 64.5445 | 18.1695 | 3.5524 | 5.6737 | 0.000000 | 1.000000 | 0.021952 | 1.0961 | 0.3678 | 0.1925 | `NARROW` |
| `NYOPEN_USDATA_RR1_5_COST_LT12` | 1.5000 | `COST_LT12` | 1705 | 64.5259 | 18.1695 | 3.5513 | 5.6586 | 0.000000 | 1.000000 | 0.021044 | 1.0772 | 0.3678 | 0.1930 | `NARROW` |
| `NYOPEN_USDATA_RR1_5_COST_LT08` | 1.5000 | `COST_LT08` | 1703 | 62.8024 | 18.1695 | 3.4565 | 5.5529 | 0.000000 | 1.000000 | 0.016313 | 1.1656 | 0.3678 | 0.1924 | `NARROW` |
| `NYOPEN_USDATA_RR2_COST_LT08` | 2.0000 | `COST_LT08` | 1703 | 65.5614 | 22.4240 | 2.9237 | 5.4177 | 0.000000 | 1.000000 | 0.010959 | 1.2060 | 0.3707 | 0.2145 | `NARROW` |
| `NYOPEN_USDATA_RR2_COST_LT12` | 2.0000 | `COST_LT12` | 1705 | 67.6361 | 23.4240 | 2.8875 | 5.5238 | 0.000000 | 1.000000 | 0.014389 | 1.1072 | 0.3707 | 0.2193 | `NARROW` |
| `NYOPEN_USDATA_RR2_COST_LT10` | 2.0000 | `COST_LT10` | 1704 | 67.5896 | 23.4240 | 2.8855 | 5.5380 | 0.000000 | 1.000000 | 0.015007 | 1.1282 | 0.3707 | 0.2176 | `NARROW` |
| `NYOPEN_USDATA_RR2_COST_LT15` | 2.0000 | `COST_LT15` | 1708 | 67.3886 | 23.4240 | 2.8769 | 5.4923 | 0.000000 | 1.000000 | 0.013014 | 1.1058 | 0.3707 | 0.2201 | `NARROW` |
| `NYOPEN_USDATA_RR2_NO_FILTER` | 2.0000 | `NO_FILTER` | 1719 | 68.4976 | 24.4240 | 2.8045 | 5.5713 | 0.000000 | 1.000000 | 0.014998 | 1.1000 | 0.3707 | 0.2188 | `NARROW` |

Verdict counts: CONTINUE=`0`, NARROW=`15`, KILL=`0`.

## Grounding

- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`: DSR controls selection bias, non-normality, sample length, trial count, and cross-trial Sharpe dispersion.
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`: broad strategy searches need multiple-testing controls and high t-stat discipline.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`: 2026 is descriptive only; no OOS-tweaking.

## Caveats

- This is a post-exploratory follow-up, not clean first discovery.
- `DSR_inherited` carries the prior broad scan, so family-positive rows remain `NARROW` unless inherited DSR clears.
- The result is a research book comparison, not broker/account deployment sizing.

## Verdict

`NARROW` for the objective-score winner. The data favors the lower-RR US_DATA_1000 leg for drawdown-adjusted book quality, but inherited DSR prevents replacement without another confirmation step.

SURVIVED SCRUTINY: finite K, no-theory t>=3.79 gate, family BH, DSR family/inherited, WFE, era, 2026 descriptive separation.
DID NOT SURVIVE: no candidate is deployment-ready from this result alone.
CAVEATS: post-selection; no ONC de-correlation; arithmetic portfolio only.
NEXT STEPS: if accepted, route a deployment-readiness design comparing current RR2 book versus RR1/RR1.5 candidate under account constraints.

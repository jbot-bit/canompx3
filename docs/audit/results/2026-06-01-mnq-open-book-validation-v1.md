# MNQ Open Book Validation v1

**Prereg:** `docs/audit/hypotheses/2026-06-01-mnq-open-book-validation-v1.yaml`
**Status:** post-exploratory, research-provisional only; no deployment claim.
**Canonical inputs:** `orb_outcomes` + `daily_features`; 2026 rows are descriptive only.
**Selection window:** `< 2026-01-01`.
**Inherited exploratory burden:** `1755` upstream cells from the prior broad scan.
**Open challenger K:** `50` pair books, capped below the 300 operational trial ceiling for this follow-up family.

## Scope Answer

[GROUNDED] The narrow current book is specific enough for confirmation, but too specific as the only research answer. The open pair-book layer keeps the question broad without turning it into an unbounded grid.
[GROUNDED] New filters are not assumed. Existing pre-entry-safe filters are diagnosed first; order-flow, footprint, and absorption remain `PARK_NEW_DATA` under current 1m OHLCV truth surfaces. No new theory grant is claimed, so the research gate uses t >= 3.79.
[MEASURED] The current risk-aware book winner is `CURRENT_COST_LT10` with objective score 2.8855; annual-only sensitivity would choose `CURRENT_NO_FILTER`.
[MEASURED] The open challenger scan tested `50` two-leg books from a capped `11`-cell pool spanning `CME_PRECLOSE, COMEX_SETTLE, NYSE_OPEN, NYSE_PREOPEN, TOKYO_OPEN, US_DATA_1000`. Top challenger by the same risk-aware score is `CHALLENGER_PAIR_007`.
[MEASURED] Annual-only sensitivity would choose challenger `CHALLENGER_PAIR_005` with annual R 72.5950, drawdown 24.3030, and 2026 descriptive mean 0.1095.
[MEASURED] Existing filter diagnosis is led by `COST_LT12`: median annual delta 1.8397R and median drawdown delta -7.3251R versus matched NO_FILTER parents.

## Reproduction

- Front door: `python scripts/tools/prereg_front_door.py --hypothesis-file docs/audit/hypotheses/2026-06-01-mnq-open-book-validation-v1.yaml --execute --runner research/mnq_open_book_validation_v1.py --format text`
- Book CSV: `docs/audit/results/2026-06-01-mnq-open-book-validation-v1-books.csv`
- Pool CSV: `docs/audit/results/2026-06-01-mnq-open-book-validation-v1-pool.csv`
- Filter diagnostics CSV: `docs/audit/results/2026-06-01-mnq-open-book-validation-v1-filter-diagnostics.csv`
- DB mode: read-only canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`.

## Current Book Confirmation

| strategy | n_is | mean_is | annual_r | dd | objective_score | t | q_family | dsr_family | dsr_inherited | wfe | mean_2026 | leg_corr_is | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `CURRENT_COST_LT10` | 1704 | 0.2641 | 67.5896 | 23.4240 | 2.8855 | 5.5380 | 0.000000 | 1.000000 | 0.016508 | 1.1282 | 0.3707 | 0.2176 | `NARROW` |
| `CURRENT_NO_FILTER` | 1719 | 0.2622 | 67.6894 | 24.4240 | 2.7714 | 5.4915 | 0.000000 | 1.000000 | 0.013435 | 1.1533 | 0.3707 | 0.2209 | `NARROW` |

Primary objective is `annual_r / max_drawdown`, pre-declared to avoid choosing raw ROI after seeing drawdown. Raw annual R is reported as a sensitivity comparator.

## Open Challenger Books

| strategy | n_is | annual_r | dd | objective_score | t | q_family | dsr_family | dsr_inherited | wfe | mean_2026 | leg_corr_is | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `CHALLENGER_PAIR_007` | 1705 | 57.3373 | 13.9546 | 4.1088 | 5.5985 | 0.000000 | 0.999998 | 0.012741 | 1.1280 | 0.4189 | 0.1365 | `NARROW` |
| `CHALLENGER_PAIR_008` | 1701 | 60.1701 | 15.5560 | 3.8680 | 5.9530 | 0.000000 | 1.000000 | 0.031088 | 1.5133 | 0.2962 | -0.0015 | `NARROW` |
| `CHALLENGER_PAIR_010` | 1701 | 56.1485 | 14.8610 | 3.7782 | 6.0656 | 0.000000 | 1.000000 | 0.040519 | 1.3986 | 0.3157 | -0.0085 | `NARROW` |
| `CHALLENGER_PAIR_006` | 1704 | 64.5445 | 18.1695 | 3.5524 | 5.6737 | 0.000000 | 0.999999 | 0.015059 | 1.0961 | 0.3678 | 0.1925 | `NARROW` |
| `CHALLENGER_PAIR_011` | 1554 | 41.6089 | 12.6934 | 3.2780 | 6.8545 | 0.000000 | 1.000000 | 0.281045 | 1.0121 | 0.0111 | -0.0388 | `NARROW` |
| `CHALLENGER_PAIR_021` | 1554 | 40.9390 | 12.5278 | 3.2679 | 6.8547 | 0.000000 | 1.000000 | 0.282179 | 0.9984 | -0.0054 | -0.0474 | `NARROW` |
| `CHALLENGER_PAIR_002` | 1701 | 62.9491 | 20.2027 | 3.1159 | 6.4844 | 0.000000 | 1.000000 | 0.093678 | 1.3749 | 0.1538 | 0.0730 | `NARROW` |
| `CHALLENGER_PAIR_031` | 1709 | 51.6593 | 16.7090 | 3.0917 | 5.2043 | 0.000000 | 0.999986 | 0.004238 | 1.0865 | 0.3431 | 0.2809 | `NARROW` |
| `CHALLENGER_PAIR_005` | 1704 | 72.5950 | 24.3030 | 2.9871 | 5.8518 | 0.000000 | 1.000000 | 0.023487 | 1.5676 | 0.1095 | 0.1735 | `NARROW` |
| `CHALLENGER_PAIR_020` | 1701 | 57.2711 | 19.2170 | 2.9802 | 6.6354 | 0.000000 | 1.000000 | 0.123096 | 1.3574 | 0.0740 | 0.0274 | `NARROW` |
| `CHALLENGER_PAIR_034` | 1699 | 50.4704 | 17.5629 | 2.8737 | 5.9617 | 0.000000 | 1.000000 | 0.033073 | 1.3773 | 0.2395 | 0.0142 | `NARROW` |
| `CHALLENGER_PAIR_030` | 1709 | 58.8664 | 20.6277 | 2.8538 | 5.2056 | 0.000000 | 0.999988 | 0.003988 | 1.0591 | 0.2914 | 0.3761 | `NARROW` |

## Top Challenger Leg Detail

| strategy | leg_1 | leg_2 | annual_r | dd | objective_score | mean_2026 | dsr_inherited | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `CHALLENGER_PAIR_007` | `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | `MNQ_US_DATA_1000_O15_E2_RR1_COST_LT12` | 57.3373 | 13.9546 | 4.1088 | 0.4189 | 0.012741 | `NARROW` |
| `CHALLENGER_PAIR_008` | `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | `MNQ_COMEX_SETTLE_O5_E2_RR1_5_COST_LT15` | 60.1701 | 15.5560 | 3.8680 | 0.2962 | 0.031088 | `NARROW` |
| `CHALLENGER_PAIR_010` | `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | `MNQ_COMEX_SETTLE_O5_E2_RR1_COST_LT15` | 56.1485 | 14.8610 | 3.7782 | 0.3157 | 0.040519 | `NARROW` |
| `CHALLENGER_PAIR_006` | `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | `MNQ_US_DATA_1000_O15_E2_RR1_5_COST_LT10` | 64.5445 | 18.1695 | 3.5524 | 0.3678 | 0.015059 | `NARROW` |
| `CHALLENGER_PAIR_011` | `MNQ_CME_PRECLOSE_O15_E2_RR2_COST_LT10` | `MNQ_NYSE_PREOPEN_O30_E2_RR1_COST_LT08` | 41.6089 | 12.6934 | 3.2780 | 0.0111 | 0.281045 | `NARROW` |

Challenger verdict counts: CONTINUE=`0`, NARROW=`49`, KILL=`1`.

## Challenger Pool

| strategy | session | orb_minutes | rr | filter | annual_r | dd | pool_score | t | q_global | wfe | mean_2026 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | NYSE_OPEN | 15 | 2.0000 | `COST_LT10` | 34.7760 | 16.8698 | 2.0614 | 4.4906 | 0.0005 | 1.4731 | 0.2158 |
| `MNQ_CME_PRECLOSE_O15_E2_RR2_COST_LT10` | CME_PRECLOSE | 15 | 2.0000 | `COST_LT10` | 13.4412 | 6.6602 | 2.0181 | 4.2728 | 0.0006 | 0.8542 | 0.0979 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR1_COST_LT08` | NYSE_PREOPEN | 30 | 1.0000 | `COST_LT08` | 28.1732 | 14.2720 | 1.9740 | 5.3088 | 0.0002 | 0.9796 | -0.0635 |
| `MNQ_NYSE_OPEN_O30_E2_RR2_COST_LT10` | NYSE_OPEN | 30 | 2.0000 | `COST_LT10` | 29.0979 | 15.0681 | 1.9311 | 4.3717 | 0.0005 | 1.4600 | 0.1376 |
| `MNQ_CME_PRECLOSE_O15_E2_RR1_5_COST_LT10` | CME_PRECLOSE | 15 | 1.5000 | `COST_LT10` | 12.7711 | 6.6881 | 1.9095 | 4.2735 | 0.0006 | 0.8098 | 0.0762 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR2_COST_LT15` | NYSE_PREOPEN | 30 | 2.0000 | `COST_LT15` | 37.8191 | 21.6790 | 1.7445 | 4.4816 | 0.0005 | 1.6091 | -0.1088 |
| `MNQ_US_DATA_1000_O15_E2_RR1_5_COST_LT10` | US_DATA_1000 | 15 | 1.5000 | `COST_LT10` | 29.7685 | 18.3508 | 1.6222 | 4.2686 | 0.0006 | 0.7574 | 0.1538 |
| `MNQ_US_DATA_1000_O15_E2_RR1_COST_LT12` | US_DATA_1000 | 15 | 1.0000 | `COST_LT12` | 22.5614 | 14.7142 | 1.5333 | 3.9394 | 0.0013 | 0.7458 | 0.2055 |
| `MNQ_COMEX_SETTLE_O5_E2_RR1_5_COST_LT15` | COMEX_SETTLE | 5 | 1.5000 | `COST_LT15` | 25.4046 | 23.7605 | 1.0692 | 3.9058 | 0.0014 | 1.2236 | 0.0880 |
| `MNQ_TOKYO_OPEN_O15_E2_RR1_COST_LT10` | TOKYO_OPEN | 15 | 1.0000 | `COST_LT10` | 18.2206 | 17.2940 | 1.0536 | 3.8200 | 0.0018 | 2.0204 | -0.0946 |
| `MNQ_COMEX_SETTLE_O5_E2_RR1_COST_LT15` | COMEX_SETTLE | 5 | 1.0000 | `COST_LT15` | 21.3813 | 20.9186 | 1.0221 | 4.1641 | 0.0007 | 0.9875 | 0.1086 |

The pool is selected from pre-2026 metrics only using a capped risk-aware score, with at most two cells per session and one per session/aperture/RR shape.

## Existing Filter Diagnosis

| filter | comparisons | median_delta_annual_r | median_delta_dd | mean_delta_t | median_delta_wfe | median_delta_2026 | helped_risk_adjusted_count | hurt_annual_count | best_delta_annual_r | best_strategy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `COST_LT12` | 117 | 1.8397 | -7.3251 | 0.6019 | -0.1008 | 0.0000 | 88 | 19 | 15.0887 | `MNQ_BRISBANE_1025_O5_E2_RR1_COST_LT12` |
| `COST_LT10` | 117 | 1.7897 | -8.3239 | 0.5743 | -0.0586 | 0.0000 | 81 | 28 | 14.0359 | `MNQ_BRISBANE_1025_O5_E2_RR1_COST_LT10` |
| `COST_LT15` | 117 | 1.7346 | -4.1110 | 0.5331 | -0.0887 | 0.0000 | 92 | 16 | 14.4768 | `MNQ_BRISBANE_1025_O5_E2_RR1_COST_LT15` |
| `COST_LT08` | 117 | 1.0396 | -11.0004 | 0.6397 | 0.0005 | 0.0000 | 70 | 43 | 12.6830 | `MNQ_BRISBANE_1025_O5_E2_RR1_COST_LT08` |
| `DIR_LONG` | 117 | -0.5153 | -10.5646 | 0.0402 | 0.0576 | 0.0222 | 52 | 65 | 10.1827 | `MNQ_BRISBANE_1025_O30_E2_RR1_DIR_LONG` |
| `ATR_P30` | 117 | -0.6810 | -5.2828 | 0.0496 | -0.0245 | 0.0000 | 40 | 76 | 9.3835 | `MNQ_BRISBANE_1025_O5_E2_RR1_ATR_P30` |
| `ORB_SIZE_Q67` | 117 | -1.5402 | -15.5553 | 0.2567 | 0.0511 | 0.0063 | 49 | 68 | 14.5336 | `MNQ_BRISBANE_1025_O5_E2_RR1_ORB_SIZE_Q67` |
| `ATR_P50` | 117 | -1.6278 | -9.3129 | -0.0445 | 0.1095 | 0.0088 | 37 | 79 | 13.4722 | `MNQ_NYSE_PREOPEN_O5_E2_RR1_5_ATR_P50` |
| `COST10_ATR50` | 117 | -1.9863 | -12.6365 | 0.1081 | 0.1905 | 0.0112 | 41 | 76 | 14.3063 | `MNQ_BRISBANE_1025_O5_E2_RR1_COST10_ATR50` |
| `ORB_VOL_Q67` | 117 | -2.1460 | -11.7978 | -0.0822 | -0.1791 | 0.0121 | 42 | 74 | 11.9221 | `MNQ_BRISBANE_1025_O5_E2_RR1_ORB_VOL_Q67` |
| `ATR_VEL_GE105` | 117 | -2.4388 | -19.7575 | 0.0411 | -0.2601 | 0.0108 | 39 | 78 | 15.1631 | `MNQ_NYSE_PREOPEN_O5_E2_RR2_ATR_VEL_GE105` |
| `ATR_P70` | 117 | -2.9143 | -13.0993 | -0.0884 | 0.2481 | 0.0024 | 39 | 78 | 14.1752 | `MNQ_NYSE_PREOPEN_O5_E2_RR1_5_ATR_P70` |
| `ORB_SIZE_Q80` | 117 | -4.0456 | -18.1478 | -0.1332 | -0.1774 | 0.0049 | 40 | 77 | 14.6674 | `MNQ_BRISBANE_1025_O5_E2_RR1_ORB_SIZE_Q80` |
| `DIR_SHORT` | 117 | -7.2949 | -4.9510 | -0.8937 | -0.1591 | -0.0197 | 21 | 96 | 7.8142 | `MNQ_NYSE_PREOPEN_O5_E2_RR2_DIR_SHORT` |

This section answers whether to invent new filters. It measures whether current pre-entry-safe filters add broad value versus the same session/aperture/RR `NO_FILTER` parent before any new filter engineering.

## Local Literature And Resource Grounding

- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`: DSR controls selection bias, non-normality, sample length, trial count, and cross-trial Sharpe dispersion.
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`: broad strategy searches need multiple-testing controls and high t-stat discipline.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`: out-of-sample tweaking and lookahead are direct backtest-failure modes; this runner keeps 2026 out of selection.
- `docs/institutional/pre_registered_criteria.md`: Criterion 5 fixed-universe DSR and Criterion 8 holdout discipline govern the research/deployment distinction.

## Caveats

- This is a post-exploratory follow-up from the prior 1,755-cell scan, not a clean first discovery.
- `DSR_inherited` is intentionally harsh because it carries the upstream broad-scan burden; every surviving book is therefore `NARROW`, not deployment-ready.
- Pair books are arithmetic research portfolios. They do not encode broker limits, prop-account sizing, correlated intraday risk, or live-session orchestration.
- Order-flow, footprint, delta, and absorption claims remain parked because they are not measurable from current `bars_1m` OHLCV.

## Verdict

`NARROW` for the current risk-aware book under the narrow family; `NARROW` ceiling overall because the object is post-exploratory and carries the inherited 1,755-cell burden. The top challenger beats the current book on drawdown-adjusted score but does not clear inherited DSR, so it is a follow-up hypothesis, not a replacement.

SURVIVED SCRUTINY: current-book and challenger metrics are computed from canonical layers only, with explicit K, DSR, BH, WFE, era, drawdown, and 2026 descriptive fields.
DID NOT SURVIVE: no result in this report is deployment-ready; any challenger that depends on 2026 descriptive behavior for comfort is not selectable.
CAVEATS: post-selection follow-up from a broad scan; DSR uses declared K and sibling failures but not ONC de-correlation; pair books are arithmetic portfolios, not broker/risk deployment plans.
NEXT STEPS: if the risk-aware book remains the best practical candidate after this pass, route deployment-readiness separately; otherwise write a new prereg for the winning challenger family rather than silently swapping the live candidate.

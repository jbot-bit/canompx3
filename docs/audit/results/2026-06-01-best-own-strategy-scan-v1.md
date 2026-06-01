# Best Own ORB Strategy Scan v1

**Status:** exploratory canonical scan, not a pre-registered validation run.
**Canonical inputs:** `orb_outcomes` + `daily_features` only; triple join on `(symbol, trading_day, orb_minutes)`.
**Selection window:** `< 2026-01-01`. 2026 is descriptive only.
**Exploratory K:** `1755` cells = MNQ enabled sessions x O{5,15,30} x RR{1,1.5,2} x 15 filters.
**Cell CSV:** `docs/audit/results/2026-06-01-best-own-strategy-scan-v1-cells.csv`
**Portfolio CSV:** `docs/audit/results/2026-06-01-best-own-strategy-scan-v1-portfolio.csv`

## Data Read

Strict exploratory passes: `0`. Research shortlist cells: `159`.
Best single cell by the report's evidence sort: `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10`.
Lower-DD two-lane book selected for follow-up: `NY_OPEN+US1000_COST10`.

No cell passes the full exploratory DSR gate. The data therefore does not authorize deployment; it only points to the next narrow hypothesis.

## Evidence-Ranked Cells

| strategy | n_is | mean_is | annual_r | t | q_global | dsr | wfe | dd | mean_2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | 1699 | 0.1363 | 34.7760 | 4.4906 | 0.000493 | 0.000748 | 1.4731 | 16.8698 | 0.2158 |
| `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT12` | 1706 | 0.1343 | 34.4027 | 4.4347 | 0.000503 | 0.000581 | 1.5587 | 16.8698 | 0.2158 |
| `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT15` | 1707 | 0.1336 | 34.2525 | 4.4143 | 0.000503 | 0.000536 | 1.5798 | 16.8698 | 0.2158 |
| `MNQ_NYSE_OPEN_O15_E2_RR2_NO_FILTER` | 1715 | 0.1319 | 33.9678 | 4.3706 | 0.000512 | 0.000429 | 1.6656 | 17.4312 | 0.2158 |
| `MNQ_US_DATA_1000_O15_E2_RR2_NO_FILTER` | 1717 | 0.1308 | 33.7216 | 4.2134 | 0.000611 | 0.000236 | 0.8144 | 23.2152 | 0.1567 |
| `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT08` | 1680 | 0.1316 | 33.2021 | 4.3145 | 0.000531 | 0.000465 | 1.7102 | 16.8698 | 0.2158 |
| `MNQ_US_DATA_1000_O15_E2_RR2_COST_LT12` | 1682 | 0.1301 | 32.8601 | 4.1405 | 0.000725 | 0.000241 | 0.8105 | 23.1532 | 0.1567 |
| `MNQ_US_DATA_1000_O15_E2_RR2_COST_LT10` | 1662 | 0.1315 | 32.8136 | 4.1556 | 0.000704 | 0.000302 | 0.8264 | 23.1532 | 0.1567 |
| `MNQ_US_DATA_1000_O15_E2_RR2_COST_LT15` | 1695 | 0.1281 | 32.6126 | 4.0955 | 0.000790 | 0.000181 | 0.8121 | 23.2152 | 0.1567 |
| `MNQ_US_DATA_1000_O15_E2_RR1_5_NO_FILTER` | 1717 | 0.1203 | 31.0098 | 4.3922 | 0.000503 | 0.000551 | 0.7349 | 19.3508 | 0.1538 |
| `MNQ_US_DATA_1000_O15_E2_RR2_COST_LT08` | 1614 | 0.1270 | 30.7855 | 3.9554 | 0.001249 | 0.000215 | 0.8816 | 22.4339 | 0.1567 |
| `MNQ_US_DATA_1000_O15_E2_RR1_5_COST_LT15` | 1695 | 0.1174 | 29.8903 | 4.2518 | 0.000593 | 0.000396 | 0.7257 | 19.3508 | 0.1538 |

## High IS, Negative 2026 Monitor

| strategy | n_is | annual_r | t | q_global | wfe | mean_2026 | dd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MNQ_NYSE_PREOPEN_O30_E2_RR2_COST_LT15` | 1640 | 37.8191 | 4.4816 | 0.000497 | 1.6091 | -0.1088 | 21.6790 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR2_NO_FILTER` | 1674 | 37.4091 | 4.4008 | 0.000503 | 1.7370 | -0.1088 | 21.6790 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR2_COST_LT12` | 1593 | 36.9019 | 4.4272 | 0.000503 | 1.6454 | -0.1088 | 23.0634 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR2_COST_LT08` | 1414 | 34.4819 | 4.3650 | 0.000518 | 1.6719 | -0.1088 | 24.4046 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR2_COST_LT10` | 1523 | 33.7458 | 4.1346 | 0.000731 | 1.9039 | -0.1088 | 22.7254 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR1_5_NO_FILTER` | 1674 | 33.6702 | 4.6494 | 0.000344 | 1.2780 | -0.1384 | 22.6539 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR1_5_COST_LT15` | 1640 | 33.3051 | 4.6326 | 0.000344 | 1.2582 | -0.1384 | 22.6539 |
| `MNQ_NYSE_PREOPEN_O30_E2_RR1_5_COST_LT08` | 1414 | 32.9378 | 4.8961 | 0.000267 | 1.1245 | -0.1384 | 21.6539 |

## Candidate Pair The Data Selects

| Candidate | N IS | Mean R | Annual R | t | p | q global | DSR | WFE | DD | 2026 mean | Era |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | 1699 | 0.1363 | 34.7760 | 4.4906 | 0.000008 | 0.000493 | 0.000748 | 1.4731 | 16.8698 | 0.2158 | True |
| `MNQ_US_DATA_1000_O15_E2_RR2_COST_LT10` | 1662 | 0.1315 | 32.8136 | 4.1556 | 0.000034 | 0.000704 | 0.000302 | 0.8264 | 23.1532 | 0.1567 | True |

## Two-Lane Book Check

| Book | N IS | Mean R/day | Annual R | t | p | DD | 2026 mean/day | Leg corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NY_OPEN+US1000_COST10` | 1704 | 0.2641 | 67.5896 | 5.5380 | 0.00000004 | 23.4240 | 0.3707 | 0.2176 |
| `NY_OPEN+US1000_NOFILTER` | 1719 | 0.2622 | 67.6894 | 5.4915 | 0.00000005 | 24.4240 | 0.3707 | 0.2209 |

The book comparison is not one-sided: NO_FILTER is higher by annual R, while COST_LT10 has lower drawdown and a slightly higher t-stat/Sharpe. The 2026 descriptive mean is identical because the 2026 trades that survive the cost gate match the no-filter set for these two legs.

## Data-Driven Exclusions

- Same-direction re-entry: KILL from the separate bounded execution report; no priority addition.
- `NYSE_PREOPEN` O30: high pre-2026 annual R but negative 2026 descriptive monitoring; not selected.
- `CME_PRECLOSE`: some positive cells, but weaker 2026/cost cushion than the selected O15 book.
- Volume rows: not selected by the evidence sort; post-trigger volume and relative-volume confirmation remain banned for E2 predictors.
- DSR fails under the full 1,755-cell exploratory burden. This is why the result is a priority hypothesis, not a deployment verdict.

## Local Literature And Resource Grounding

- `resources/INDEX.md`: local corpus manifest; curated extracts in `docs/institutional/literature/` are the canonical citation source.
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`: DSR requires the selected strategy's Sharpe, sample length, trial-count burden, cross-trial Sharpe variance, skewness, and kurtosis. This runner computes all six for the broad screen.
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`: broad trading-strategy searches require multiple-hypothesis controls and high t-stat hurdles. This report exposes `q_global`, `t`, and declared K instead of ranking raw means alone.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`: lookahead and OOS-tweaking are explicit failure modes. This runner bans E2 post-trigger predictors and keeps 2026 descriptive only.
- `trading_app.config.CostRatioFilter`: COST_LT gates use canonical round-trip friction share of raw ORB risk, not triggered-trade `risk_dollars`.
- Unsupported by current local data: order-flow absorption, footprint delta, and stop-hunt intent. Those remain `PARK_NEW_DATA`.

## Verdict

`NARROW`: make the next formal hypothesis a small, bounded validation of the O15/E2/RR2 MNQ NYSE_OPEN + US_DATA_1000 book. Declare upfront whether the objective ranks annual R first (`NO_FILTER`) or drawdown/t-stat first (`COST_LT10`), then keep the other as the sensitivity comparator.

SURVIVED SCRUTINY: MNQ NYSE_OPEN O15 E2 RR2 COST_LT10 and MNQ US_DATA_1000 O15 E2 RR2 COST_LT10 are positive across pre-2026 eras, positive in 2026 descriptive monitoring, and combine with low lane correlation.
DID NOT SURVIVE: no cell clears the strict full exploratory DSR gate; same-direction re-entry was killed separately.
CAVEATS: exploratory post-selection; no capital deployment claim; DSR uses the broad K screen and per-cell return skew/kurtosis but no ONC de-correlation estimate.
NEXT STEPS: pre-register a narrow book validation that uses the previously known MNQ NYSE_OPEN and US_DATA_1000 parent-survivor context, not this full exploratory K.

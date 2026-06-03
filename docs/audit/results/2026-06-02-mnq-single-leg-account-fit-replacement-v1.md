# MNQ Single-Leg Account-Fit Replacement v1

**Prereg:** `docs\audit\hypotheses\2026-06-02-mnq-single-leg-account-fit-replacement-v1.yaml`
**Status:** bounded allocator replacement audit; research-only; no deployment claim.
**Family K:** `15` selectable replacement scenarios.
**Selection window:** `< 2026-01-01`; 2026 rows are monitoring only.
**Profile:** `topstep_50k_mnq_auto`; daily belt `$450`, max DD `$2000`, DD budget `$1600`.

## Disconfirming Checks

[MEASURED] Current incumbent lanes were loaded from structured allocator JSON fields, not parsed from strategy IDs.
[MEASURED] The locked universe stayed at five candidates crossed with three incumbent replacement slots.
[MEASURED] The runner used read-only canonical `orb_outcomes` plus `daily_features` and canonical filter delegation through `research.filter_utils.filter_signal`.
[UNSUPPORTED] No scenario may be described as deployed, live-valid, validated, or OOS-clean from this artifact.

## Incumbent Comparator

- Annual dollars: `$4911.08`
- Max DD dollars: `$2733.26`
- Annual dollars / max DD: `1.7968`
- 90-day survival: `0.8997`
- Daily loss breach rate: `0.0948`
- Trailing DD breach rate: `0.0055`

## Result

[MEASURED] Best ranked scenario is `R02_C05` replacing `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` with `MNQ_CME_PRECLOSE_O15_E2_RR2_COST_LT10`, verdict `KILL`.
[MEASURED] Verdict counts: `{'KILL': 15}`.

## Scenario Ranking

| scenario_id | replaced_incumbent_lane | candidate_lane | n_is_trades | annual_dollars | max_drawdown_dollars | annual_dollars_per_max_drawdown | expected_r_after_costs | win_rate | t_stat | mean_2026_dollars | mean_2026_r | ninety_day_account_survival | daily_loss_breach_rate | trailing_drawdown_breach_rate | historical_daily_loss_breaches | chordia_verdict | sr_regime_status | oos_status | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `R02_C05` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | `MNQ_CME_PRECLOSE_O15_E2_RR2_COST_LT10` | 2045 | 3549.7044 | 1584.8337 | 2.2398 | 0.1656 | 0.5361 | 6.7219 | 28.3002 | 0.3825 | 0.9534 | 0.0466 | 0.0000 | 1 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R03_C05` | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` | `MNQ_CME_PRECLOSE_O15_E2_RR2_COST_LT10` | 2503 | 5828.8580 | 2316.5373 | 2.5162 | 0.2256 | 0.5293 | 8.0664 | 35.6748 | 0.3634 | 0.8961 | 0.0991 | 0.0048 | 2 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R02_C01` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | 2659 | 6983.9319 | 3268.1558 | 2.1370 | 0.2481 | 0.4674 | 6.7330 | 78.8670 | 0.5163 | 0.4083 | 0.5272 | 0.0645 | 15 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R03_C01` | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` | `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | 3117 | 9265.6881 | 3472.8796 | 2.6680 | 0.3081 | 0.4681 | 6.9797 | 87.3192 | 0.5005 | 0.2025 | 0.7310 | 0.0665 | 28 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R01_C01` | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` | 3019 | 8324.4953 | 4513.9767 | 1.8442 | 0.2931 | 0.4637 | 6.6801 | 81.8402 | 0.5294 | 0.1378 | 0.8081 | 0.0541 | 35 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R01_C05` | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | `MNQ_CME_PRECLOSE_O15_E2_RR2_COST_LT10` | 2405 | 4890.2678 | 3306.8235 | 1.4788 | 0.2106 | 0.5305 | 7.5651 | 31.3072 | 0.3957 | 0.7648 | 0.2289 | 0.0063 | 5 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R02_C02` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | `MNQ_US_DATA_1000_O15_E2_RR1_NO_FILTER` | 2677 | 3947.5660 | 2832.7659 | 1.3935 | 0.2054 | 0.5731 | 6.5044 | 51.5240 | 0.5098 | 0.7499 | 0.2206 | 0.0295 | 5 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R02_C03` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | `MNQ_US_DATA_1000_O15_E2_RR1_5_NO_FILTER` | 2677 | 4830.5235 | 3124.4535 | 1.5460 | 0.2335 | 0.5020 | 6.6629 | 35.5007 | 0.4599 | 0.7379 | 0.2192 | 0.0429 | 5 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R02_C04` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | `MNQ_US_DATA_1000_O15_E2_RR2_NO_FILTER` | 2677 | 5107.7538 | 3500.7126 | 1.4591 | 0.2440 | 0.4578 | 6.4177 | 34.7282 | 0.4627 | 0.7167 | 0.2169 | 0.0664 | 5 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R03_C02` | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` | `MNQ_US_DATA_1000_O15_E2_RR1_NO_FILTER` | 3135 | 6227.6459 | 4400.7713 | 1.4151 | 0.2655 | 0.5463 | 6.5021 | 59.4386 | 0.4937 | 0.1876 | 0.7813 | 0.0311 | 30 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R03_C03` | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` | `MNQ_US_DATA_1000_O15_E2_RR1_5_NO_FILTER` | 3135 | 7112.6592 | 4728.3459 | 1.5043 | 0.2937 | 0.4863 | 6.3393 | 43.0426 | 0.4426 | 0.1444 | 0.8103 | 0.0453 | 34 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R03_C04` | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` | `MNQ_US_DATA_1000_O15_E2_RR2_NO_FILTER` | 3135 | 7390.5349 | 4903.3709 | 1.5072 | 0.3042 | 0.4572 | 6.2285 | 42.2522 | 0.4454 | 0.1413 | 0.8052 | 0.0535 | 34 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R01_C02` | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | `MNQ_US_DATA_1000_O15_E2_RR1_NO_FILTER` | 3037 | 5288.1294 | 7194.3891 | 0.7350 | 0.2504 | 0.5381 | 6.0945 | 54.5310 | 0.5230 | 0.1170 | 0.8590 | 0.0240 | 39 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R01_C03` | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | `MNQ_US_DATA_1000_O15_E2_RR1_5_NO_FILTER` | 3037 | 6171.0869 | 8658.2976 | 0.7127 | 0.2785 | 0.4846 | 5.9639 | 38.5076 | 0.4731 | 0.0951 | 0.8687 | 0.0362 | 42 | MISSING | UNKNOWN | UNKNOWN | `KILL` |
| `R01_C04` | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | `MNQ_US_DATA_1000_O15_E2_RR2_NO_FILTER` | 3037 | 6448.3172 | 8195.4702 | 0.7868 | 0.2890 | 0.4554 | 5.8524 | 37.7352 | 0.4758 | 0.0921 | 0.8635 | 0.0444 | 42 | MISSING | UNKNOWN | UNKNOWN | `KILL` |

## Grounding

- `docs/institutional/pre_registered_criteria.md` Criterion 11 requires account-death Monte Carlo with prop-firm daily loss/trailing DD rules and 90-day survival >= 70% before funded deployment.
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` grounds the no-theory t-stat severity benchmark at 3.79.
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` grounds inherited multiple-testing caution; this account audit does not reset alpha-discovery K.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` grounds the no-OOS-tuning rule; 2026 is monitoring only.

## Caveats

- Account simulation uses daily-close PnL aggregation, matching the prior account-fit runners; it does not replay intraday path ordering.
- Missing candidate Chordia/SR/OOS status is reported and prevents deployment language.
- The result does not edit live allocation, prop profiles, or runtime config.

## Verdict

`KILL`: every replacement scenario is account-unsafe, non-economic, or worse than the incumbent comparator.

SURVIVED SCRUTINY: locked K=15, structured incumbent parsing, canonical filters, read-only canonical DB, Topstep account gates, no 2026 tuning.
DID NOT SURVIVE: no live deployment claim from this research artifact.

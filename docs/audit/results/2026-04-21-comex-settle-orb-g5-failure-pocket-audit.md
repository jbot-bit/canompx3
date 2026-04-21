# COMEX_SETTLE ORB_G5 failure-pocket audit

**Date:** 2026-04-21
**Script:** `research/audit_comex_settle_orb_g5_failure_pocket.py`
**Lane:** `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
**Classification:** confirmatory audit; no new pre-reg required

## Verdict

`CONDITIONAL`

The historical fire-vs-no-fire gap is real, but the comparator is not fair for current deployment interpretation because the no-fire bucket is almost entirely extinct-era.

## Pre-Flight

- `orb_outcomes` latest MNQ day: `2026-04-19`
- `daily_features` latest MNQ day: `2026-04-19`
- `bars_1m` latest MNQ ts: `2026-04-20 09:59:00+10:00`
- Pre-holdout lane row count (all entry outcomes): `1658`

## Truth + Calculation Check

- Canonical join is correct: `orb_outcomes` to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Canonical filter is correct: `ORB_G5 == orb_COMEX_SETTLE_size >= 5` via `trading_app.config.OrbSizeFilter`.
- Execution model is canonical `E2`: stop-market at ORB boundary plus 1 tick slippage via `trading_app.entry_rules._resolve_e2`.
- Fakeouts are included as valid fills; ambiguous same-bar target+stop resolves conservatively to loss in `trading_app/outcome_builder.py`.
- `pnl_r` is net-cost R from `pipeline.cost_model.to_r_multiple`; risk denominator includes friction.
- Hidden assumption found: scratches have `pnl_r = NULL`, so resolved-only comparisons exclude them unless they are explicitly reintroduced.

## Core Comparison

| group   | n_all_entries | scratches | expr_resolved | expr_scratch0 | wr_resolved |
| ------- | ------------- | --------- | ------------- | ------------- | ----------- |
| fire    | 1577          | 22        | 0.0915        | 0.0902        | 0.4765      |
| no_fire | 81            | 0         | -0.4107       | -0.4107       | 0.3333      |

- Resolved-only Welch: `p=0.0000`; WR z-test on resolved rows: `p=0.0118`.
- Scratch-inclusive Welch using `pnl_eff` (`scratch -> 0`): `p=0.0000`.
- Scratch bias is real but small: all `22` scratches are in fire, yet the sign and significance survive after including them.

## Comparator Fairness

| year | n_total | fire_rate | n_fire | n_nofire | fire_scratches | expr_fire_s0 | expr_nofire_s0 |
| ---- | ------- | --------- | ------ | -------- | -------------- | ------------ | -------------- |
| 2019 | 164     | 0.6037    | 99     | 65       | 3              | -0.2124      | -0.3248        |
| 2020 | 249     | 0.9478    | 236    | 13       | 2              | 0.1062       | -1.0000        |
| 2021 | 251     | 0.9880    | 248    | 3        | 5              | 0.0484       | 0.2824         |
| 2022 | 250     | 1.0000    | 250    | 0        | 1              | -0.0012      | NaN            |
| 2023 | 248     | 1.0000    | 248    | 0        | 2              | 0.1396       | NaN            |
| 2024 | 249     | 1.0000    | 249    | 0        | 8              | 0.1241       | NaN            |
| 2025 | 247     | 1.0000    | 247    | 0        | 1              | 0.2473       | NaN            |

- `2019-2020` contributes `78/81` no-fire rows; `2021-2025` contributes only `3/81`.
- Fire rate drifts from `0.6037` in 2019 to `1.0000` from 2022 onward.
- Conclusion: this is a valid historical selector test, but a poor live-selector test because the counterfactual bucket has vanished.

## Era-Matched 2019-2020

| group   | n_all_entries | scratches | expr_scratch0 | wr_resolved |
| ------- | ------------- | --------- | ------------- | ----------- |
| fire    | 335           | 5         | 0.0120        | 0.4515      |
| no_fire | 78            | 0         | -0.4373       | 0.3205      |

- Early-era scratch-inclusive Welch: `p=0.0001`; WR z-test: `p=0.0354`.
- This keeps the original claim honest: the historical discrimination was real inside the era where the threshold actually bit.

## Where Edge Actually Lives

- Full unfiltered lane, all entries: `ExpR=+0.0658R`, one-sample `p=0.0180`.
- Late unfiltered lane (`2021-2025`): `ExpR=+0.1117R`, one-sample `p=0.0006`.
- Long-side lane: `ExpR=+0.0761R`, `p=0.0461`.
- Short-side lane: `ExpR=+0.0541R`, `p=0.1831`.
- This points to the lane geometry as the primary surviving edge location. `ORB_G5` was a historical conditioner, not the current source of truth.

## Early Failure-Pocket Shape

| fire | break_dir | n   | scratches | expr_s0 |
| ---- | --------- | --- | --------- | ------- |
| 0    | long      | 40  | 0         | -0.2383 |
| 0    | short     | 41  | 0         | -0.5788 |
| 1    | long      | 839 | 10        | 0.0911  |
| 1    | short     | 738 | 12        | 0.0892  |

| feature             | n_nofire | mean_nofire | n_fire | mean_fire | delta_nofire_minus_fire |
| ------------------- | -------- | ----------- | ------ | --------- | ----------------------- |
| break_delay_min     | 78       | 3.5256      | 335    | 5.0955    | -1.5699                 |
| pre_velocity        | 78       | 0.0585      | 335    | -0.0756   | 0.1341                  |
| rel_vol             | 77       | 1.2566      | 331    | 2.1470    | -0.8904                 |
| garch_vol_pct       | 2        | 0.0000      | 147    | 31.8017   | -31.8017                |
| overnight_range_pct | 76       | 39.6471     | 318    | 51.1215   | -11.4744                |

### `break_bar_continues`

| fire | break_bar_continues | n   | denom | rate   |
| ---- | ------------------- | --- | ----- | ------ |
| 0    | False               | 4   | 78    | 5.13%  |
| 0    | True                | 74  | 78    | 94.87% |
| 1    | False               | 1   | 335   | 0.30%  |
| 1    | True                | 334 | 335   | 99.70% |

### `overnight_took_pdh`

| fire | overnight_took_pdh | n   | denom | rate   |
| ---- | ------------------ | --- | ----- | ------ |
| 0    | False              | 47  | 78    | 60.26% |
| 0    | True               | 30  | 78    | 38.46% |
| 1    | False              | 216 | 335   | 64.48% |
| 1    | True               | 119 | 335   | 35.52% |

### `overnight_took_pdl`

| fire | overnight_took_pdl | n   | denom | rate   |
| ---- | ------------------ | --- | ----- | ------ |
| 0    | False              | 70  | 78    | 89.74% |
| 0    | True               | 7   | 78    | 8.97%  |
| 1    | False              | 272 | 335   | 81.19% |
| 1    | True               | 63  | 335   | 18.81% |

### `gap_type`

| fire | gap_type | n   | denom | rate   |
| ---- | -------- | --- | ----- | ------ |
| 0    | gap_down | 2   | 78    | 2.56%  |
| 0    | gap_up   | 2   | 78    | 2.56%  |
| 0    | inside   | 73  | 78    | 93.59% |
| 1    | gap_down | 6   | 335   | 1.79%  |
| 1    | gap_up   | 6   | 335   | 1.79%  |
| 1    | inside   | 323 | 335   | 96.42% |

### `prev_day_direction`

| fire | prev_day_direction | n   | denom | rate   |
| ---- | ------------------ | --- | ----- | ------ |
| 0    | bear               | 29  | 78    | 37.18% |
| 0    | bull               | 48  | 78    | 61.54% |
| 1    | bear               | 131 | 335   | 39.10% |
| 1    | bull               | 204 | 335   | 60.90% |

### `break_dir`

| fire | break_dir | n   | denom | rate   |
| ---- | --------- | --- | ----- | ------ |
| 0    | long      | 37  | 78    | 47.44% |
| 0    | short     | 41  | 78    | 52.56% |
| 1    | long      | 189 | 335   | 56.42% |
| 1    | short     | 146 | 335   | 43.58% |

## 2020 Worst Pocket

| trading_day | orb_size | rel_vol | overnight_range_pct | break_dir | pnl_eff |
| ----------- | -------- | ------- | ------------------- | --------- | ------- |
| 2020-01-02  | 3.00     | 3.3600  | 26.32               | short     | -1.00   |
| 2020-01-06  | 4.25     | 1.1948  | 93.10               | short     | -1.00   |
| 2020-01-08  | 4.00     | 4.8326  | 100.00              | long      | -1.00   |
| 2020-01-09  | 3.25     | 1.7717  | 60.34               | short     | -1.00   |
| 2020-01-10  | 3.75     | 0.9863  | 62.07               | long      | -1.00   |
| 2020-01-13  | 4.00     | 0.7839  | 72.41               | short     | -1.00   |
| 2020-01-17  | 2.50     | 1.3782  | 30.51               | short     | -1.00   |
| 2020-01-22  | 3.00     | 0.5765  | 96.67               | long      | -1.00   |
| 2020-01-29  | 3.25     | 0.6064  | 90.00               | short     | -1.00   |
| 2020-02-06  | 3.75     | 0.9631  | 88.33               | short     | -1.00   |
| 2020-02-19  | 3.25     | 0.3588  | 50.00               | long      | -1.00   |
| 2020-12-28  | 4.75     | 0.6547  | 85.96               | long      | -1.00   |
| 2020-12-31  | 4.50     | 0.4645  | 0.00                | short     | -1.00   |

## Anti-Tunnel Check

- Framing tested here: binary filter (`R1`).
- Not tested but fair and still open:
  - standalone unfiltered COMEX lane overlay
  - size modifier (`R3`) on the unfiltered COMEX lane
  - confluence / allocator use with other lane or portfolio state
- Not tested fairly enough yet: current-era selector value, because no-fire sample is functionally gone.

## Decision

- `KEEP` the historical fact: ORB_G5 had real early-era discrimination on COMEX.
- `KILL` the stronger live claim: this does NOT prove ORB_G5 is a currently-live selector.
- `PARK` further ORB_G5 rescue work on this lane.
- `NEXT` highest-EV move: pre-register a small unfiltered COMEX overlay study with explicit role framing (`R1` vs `R3`) and scratch-inclusive evaluation.

## Reproduction

```bash
python research/audit_comex_settle_orb_g5_failure_pocket.py
```

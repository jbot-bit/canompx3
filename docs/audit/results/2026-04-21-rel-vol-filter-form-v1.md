# rel_vol Filter Form v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-21-rel-vol-filter-form-v1.yaml`  
**Script:** `research/rel_vol_filter_form_v1.py`  
**Scope:** `MGC` + `MES`, `E2 / CB1 / O5 / RR1.5`, rel_vol lineage only.

## What Was Actually Tested

Per instrument, thresholds were trained on `trading_day < 2024-01-01` per `(session, direction)` lane, with lanes below `N_train < 100` dropped entirely. The two locked forms were then evaluated on the true validation window `2024-01-01 .. 2025-12-31` only.

- `F1_Q5_only`: trade only quintile 5.
- `F2_Q4_plus_Q5`: trade only quintiles 4 and 5.
- Filter ExpR and filter Sharpe are computed on fired trades only; uniform baseline is computed on the same kept-lane universe, following the repo-standard on/off filter evaluation convention.
- Gate 4 CI is a moving-block bootstrap of `filter Sharpe - uniform Sharpe` with `block_size=20`, `B=10000`.
- Semi-OOS `2026-01-01 .. 2026-04-19` is reported as informational only; it is not used for gating.

## Structural Veto

Canonical `trading_app.config.VolumeFilter` marks `rel_vol` as `E2`-excluded: it uses break-bar volume and resolves at `BREAK_DETECTED`, which is unknown at E2 order placement. This means the filter-form result below is a valid conditional research result, but it is **not** a deployable `E2` pre-entry filter in its current framing.

## Headline

| Instrument | Winner Status | Winning Form | E2 Execution | Notes |
|---|---|---|---|---|
| MGC | **CANDIDATE_READY_IS** | F1_Q5_only | **VETO** | Val filter SR=0.0749, uniform SR=-0.0866, fire rate=16.8%. |
| MES | **CANDIDATE_READY_IS** | F1_Q5_only | **VETO** | Val filter SR=0.0576, uniform SR=-0.0963, fire rate=18.8%. |

Max canonical data day available while running this script: `2026-04-16`. Fresh OOS window begins at `2026-04-22`, so fresh-OOS accrual remains zero in this repo state.

## MGC

Kept lanes after train-threshold eligibility (`N_train >= 100`): 16 lanes, 7633 total trades across train/val/OOS.

| Form | Val N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR | ΔSR | Bootstrap 95% CI ΔSR | Gate1 | Gate2 | Gate3 | Gate4 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| F1_Q5_only | 4014 | 673 | 16.8% | 0.08120 | -0.08987 | 0.0749 | -0.0866 | 0.1615 | [0.0954, 0.2338] | PASS | PASS | PASS | PASS | **FILTER_ALIVE_IS_F1** |
| F2_Q4_plus_Q5 | 4014 | 1574 | 39.2% | 0.00433 | -0.08987 | 0.0040 | -0.0866 | 0.0906 | [0.0529, 0.1325] | PASS | FAIL | PASS | FAIL | **FILTER_DEAD_IS** |

Instrument summary: **CANDIDATE_READY_IS**.

Winning form by locked rule: `F1_Q5_only` (Q5 only) with val-IS Sharpe `0.0749`.

### Fail Reasons

- `F1_Q5_only`: None
- `F2_Q4_plus_Q5`: Gate2 val-IS filter ExpR < +0.05R

### Semi-OOS (Informational Only)

Window in pre-reg: `2026-01-01 .. 2026-04-19`. Available canonical data in this repo currently ends at `2026-04-16`.

| Form | N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1_Q5_only | 557 | 80 | 14.4% | 0.34583 | 0.05431 | 0.2893 | 0.0461 |
| F2_Q4_plus_Q5 | 557 | 201 | 36.1% | 0.14172 | 0.05431 | 0.1183 | 0.0461 |

### Fresh OOS (Not Yet Accrued)

Fresh OOS starts at `2026-04-22`. Current kept-sample trades in that window: `0` total.

| Form | N | Fire N | Fire Rate | Filter ExpR | Filter SR |
|---|---:|---:|---:|---:|---:|
| F1_Q5_only | 0 | 0 | NA | NA | NA |
| F2_Q4_plus_Q5 | 0 | 0 | NA | NA | NA |

### Winning Form Per-Lane (Val-IS)

| Lane | Val N | Fire N | Fire Rate | Filter ExpR | Filter SR |
|---|---:|---:|---:|---:|---:|
| COMEX_SETTLE_long | 257 | 43 | 16.7% | -0.01023 | -0.0095 |
| COMEX_SETTLE_short | 228 | 45 | 19.7% | -0.04137 | -0.0395 |
| EUROPE_FLOW_long | 288 | 46 | 16.0% | 0.12017 | 0.1180 |
| EUROPE_FLOW_short | 228 | 44 | 19.3% | -0.03744 | -0.0347 |
| LONDON_METALS_long | 263 | 41 | 15.6% | 0.25297 | 0.2445 |
| LONDON_METALS_short | 253 | 41 | 16.2% | 0.15987 | 0.1514 |
| NYSE_OPEN_long | 254 | 33 | 13.0% | 0.41803 | 0.3829 |
| NYSE_OPEN_short | 252 | 41 | 16.3% | 0.04478 | 0.0392 |
| SINGAPORE_OPEN_long | 265 | 64 | 24.2% | 0.25124 | 0.2250 |
| SINGAPORE_OPEN_short | 252 | 28 | 11.1% | -0.03830 | -0.0357 |
| TOKYO_OPEN_long | 245 | 43 | 17.6% | 0.21907 | 0.2057 |
| TOKYO_OPEN_short | 272 | 57 | 21.0% | -0.10690 | -0.1058 |
| US_DATA_1000_long | 265 | 27 | 10.2% | 0.25580 | 0.2227 |
| US_DATA_1000_short | 204 | 43 | 21.1% | 0.11924 | 0.1028 |
| US_DATA_830_long | 249 | 35 | 14.1% | -0.34069 | -0.3219 |
| US_DATA_830_short | 239 | 42 | 17.6% | 0.04923 | 0.0421 |

### Winning Form Per-Year (Full IS; 2024-2025 Are the Gated Val Years)

| Year | Split | N | Fire N | Fire Rate | Filter ExpR | Filter SR |
|---|---|---:|---:|---:|---:|---:|
| 2022 | train | 1109 | 204 | 18.4% | -0.00229 | -0.0023 |
| 2023 | train | 1953 | 406 | 20.8% | -0.07634 | -0.0761 |
| 2024 | val | 1993 | 365 | 18.3% | -0.01783 | -0.0172 |
| 2025 | val | 2021 | 308 | 15.2% | 0.19856 | 0.1757 |

### Winning Form Per-Direction (Val-IS)

| Direction | Val N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR |
|---|---:|---:|---:|---:|---:|---:|---:|
| long | 2086 | 332 | 15.9% | 0.14981 | -0.06077 | 0.1382 | -0.0584 |
| short | 1928 | 341 | 17.7% | 0.01440 | -0.12136 | 0.0133 | -0.1174 |

### T0 Tautology Pre-Screen (Val-IS, Per Session)

Run exactly against the 5 pre-committed filter keys named in the YAML. Flag threshold is `|corr| > 0.70`.

| Session | Filter | |corr| | Flag |
|---|---|---:|---|
| COMEX_SETTLE | ATR_P50 | 0.023 | OK |
| COMEX_SETTLE | COST_LT12 | 0.172 | OK |
| COMEX_SETTLE | ORB_G5 | 0.153 | OK |
| COMEX_SETTLE | OVNRNG_50 | 0.009 | OK |
| COMEX_SETTLE | VWAP_MID_ALIGNED | 0.008 | OK |
| EUROPE_FLOW | ATR_P50 | 0.095 | OK |
| EUROPE_FLOW | COST_LT12 | 0.176 | OK |
| EUROPE_FLOW | ORB_G5 | 0.227 | OK |
| EUROPE_FLOW | OVNRNG_50 | 0.019 | OK |
| EUROPE_FLOW | VWAP_MID_ALIGNED | 0.079 | OK |
| LONDON_METALS | ATR_P50 | 0.017 | OK |
| LONDON_METALS | COST_LT12 | 0.138 | OK |
| LONDON_METALS | ORB_G5 | 0.165 | OK |
| LONDON_METALS | OVNRNG_50 | 0.105 | OK |
| LONDON_METALS | VWAP_MID_ALIGNED | 0.078 | OK |
| NYSE_OPEN | ATR_P50 | 0.068 | OK |
| NYSE_OPEN | COST_LT12 | 0.046 | OK |
| NYSE_OPEN | ORB_G5 | 0.070 | OK |
| NYSE_OPEN | OVNRNG_50 | 0.009 | OK |
| NYSE_OPEN | VWAP_MID_ALIGNED | 0.027 | OK |
| SINGAPORE_OPEN | ATR_P50 | 0.051 | OK |
| SINGAPORE_OPEN | COST_LT12 | 0.168 | OK |
| SINGAPORE_OPEN | ORB_G5 | 0.162 | OK |
| SINGAPORE_OPEN | OVNRNG_50 | 0.067 | OK |
| SINGAPORE_OPEN | VWAP_MID_ALIGNED | 0.194 | OK |
| TOKYO_OPEN | ATR_P50 | 0.005 | OK |
| TOKYO_OPEN | COST_LT12 | 0.132 | OK |
| TOKYO_OPEN | ORB_G5 | 0.132 | OK |
| TOKYO_OPEN | OVNRNG_50 | 0.117 | OK |
| TOKYO_OPEN | VWAP_MID_ALIGNED | 0.098 | OK |
| US_DATA_1000 | ATR_P50 | 0.059 | OK |
| US_DATA_1000 | COST_LT12 | 0.149 | OK |
| US_DATA_1000 | ORB_G5 | 0.186 | OK |
| US_DATA_1000 | OVNRNG_50 | 0.028 | OK |
| US_DATA_1000 | VWAP_MID_ALIGNED | 0.085 | OK |
| US_DATA_830 | ATR_P50 | 0.026 | OK |
| US_DATA_830 | COST_LT12 | 0.306 | OK |
| US_DATA_830 | ORB_G5 | 0.310 | OK |
| US_DATA_830 | OVNRNG_50 | 0.021 | OK |
| US_DATA_830 | VWAP_MID_ALIGNED | 0.055 | OK |

## MES

Kept lanes after train-threshold eligibility (`N_train >= 100`): 22 lanes, 16760 total trades across train/val/OOS.

| Form | Val N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR | ΔSR | Bootstrap 95% CI ΔSR | Gate1 | Gate2 | Gate3 | Gate4 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| F1_Q5_only | 4841 | 910 | 18.8% | 0.06352 | -0.10122 | 0.0576 | -0.0963 | 0.1538 | [0.0984, 0.2137] | PASS | PASS | PASS | PASS | **FILTER_ALIVE_IS_F1** |
| F2_Q4_plus_Q5 | 4841 | 1846 | 38.1% | 0.03954 | -0.10122 | 0.0362 | -0.0963 | 0.1325 | [0.0982, 0.1701] | PASS | FAIL | PASS | FAIL | **FILTER_DEAD_IS** |

Instrument summary: **CANDIDATE_READY_IS**.

Winning form by locked rule: `F1_Q5_only` (Q5 only) with val-IS Sharpe `0.0576`.

### Fail Reasons

- `F1_Q5_only`: None
- `F2_Q4_plus_Q5`: Gate2 val-IS filter ExpR < +0.05R

### Semi-OOS (Informational Only)

Window in pre-reg: `2026-01-01 .. 2026-04-19`. Available canonical data in this repo currently ends at `2026-04-16`.

| Form | N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1_Q5_only | 702 | 152 | 21.7% | 0.07507 | -0.09022 | 0.0659 | -0.0824 |
| F2_Q4_plus_Q5 | 702 | 291 | 41.5% | 0.00584 | -0.09022 | 0.0052 | -0.0824 |

### Fresh OOS (Not Yet Accrued)

Fresh OOS starts at `2026-04-22`. Current kept-sample trades in that window: `0` total.

| Form | N | Fire N | Fire Rate | Filter ExpR | Filter SR |
|---|---:|---:|---:|---:|---:|
| F1_Q5_only | 0 | 0 | NA | NA | NA |
| F2_Q4_plus_Q5 | 0 | 0 | NA | NA | NA |

### Winning Form Per-Lane (Val-IS)

| Lane | Val N | Fire N | Fire Rate | Filter ExpR | Filter SR |
|---|---:|---:|---:|---:|---:|
| CME_PRECLOSE_long | 225 | 37 | 16.4% | -0.05205 | -0.0471 |
| CME_PRECLOSE_short | 178 | 32 | 18.0% | 0.08405 | 0.0758 |
| CME_REOPEN_long | 103 | 18 | 17.5% | 0.25501 | 0.2200 |
| CME_REOPEN_short | 125 | 13 | 10.4% | 0.31864 | 0.2923 |
| COMEX_SETTLE_long | 271 | 47 | 17.3% | 0.23777 | 0.2106 |
| COMEX_SETTLE_short | 215 | 46 | 21.4% | 0.23798 | 0.2162 |
| EUROPE_FLOW_long | 260 | 36 | 13.8% | 0.03440 | 0.0324 |
| EUROPE_FLOW_short | 256 | 51 | 19.9% | -0.21855 | -0.2128 |
| LONDON_METALS_long | 265 | 50 | 18.9% | -0.09749 | -0.0906 |
| LONDON_METALS_short | 250 | 57 | 22.8% | 0.03527 | 0.0330 |
| NYSE_CLOSE_long | 69 | 16 | 23.2% | -0.05234 | -0.0471 |
| NYSE_CLOSE_short | 88 | 21 | 23.9% | -0.25533 | -0.2365 |
| NYSE_OPEN_long | 247 | 46 | 18.6% | 0.02420 | 0.0205 |
| NYSE_OPEN_short | 260 | 63 | 24.2% | 0.28606 | 0.2464 |
| SINGAPORE_OPEN_long | 276 | 52 | 18.8% | 0.07189 | 0.0658 |
| SINGAPORE_OPEN_short | 241 | 52 | 21.6% | -0.30889 | -0.3199 |
| TOKYO_OPEN_long | 243 | 36 | 14.8% | 0.12135 | 0.1124 |
| TOKYO_OPEN_short | 274 | 44 | 16.1% | 0.15727 | 0.1461 |
| US_DATA_1000_long | 266 | 61 | 22.9% | 0.10899 | 0.0956 |
| US_DATA_1000_short | 239 | 55 | 23.0% | 0.21367 | 0.1840 |
| US_DATA_830_long | 241 | 33 | 13.7% | -0.01579 | -0.0135 |
| US_DATA_830_short | 249 | 44 | 17.7% | 0.20851 | 0.1859 |

### Winning Form Per-Year (Full IS; 2024-2025 Are the Gated Val Years)

| Year | Split | N | Fire N | Fire Rate | Filter ExpR | Filter SR |
|---|---|---:|---:|---:|---:|---:|
| 2019 | train | 1590 | 392 | 24.7% | -0.10915 | -0.1108 |
| 2020 | train | 2449 | 508 | 20.7% | 0.03788 | 0.0346 |
| 2021 | train | 2406 | 486 | 20.2% | -0.01018 | -0.0094 |
| 2022 | train | 2397 | 390 | 16.3% | 0.04463 | 0.0396 |
| 2023 | train | 2375 | 465 | 19.6% | -0.02932 | -0.0280 |
| 2024 | val | 2399 | 496 | 20.7% | 0.06693 | 0.0618 |
| 2025 | val | 2442 | 414 | 17.0% | 0.05942 | 0.0526 |

### Winning Form Per-Direction (Val-IS)

| Direction | Val N | Fire N | Fire Rate | Filter ExpR | Uniform ExpR | Filter SR | Uniform SR |
|---|---:|---:|---:|---:|---:|---:|---:|
| long | 2466 | 432 | 17.5% | 0.05721 | -0.11326 | 0.0515 | -0.1080 |
| short | 2375 | 478 | 20.1% | 0.06922 | -0.08872 | 0.0630 | -0.0842 |

### T0 Tautology Pre-Screen (Val-IS, Per Session)

Run exactly against the 5 pre-committed filter keys named in the YAML. Flag threshold is `|corr| > 0.70`.

| Session | Filter | |corr| | Flag |
|---|---|---:|---|
| CME_PRECLOSE | ATR_P50 | 0.062 | OK |
| CME_PRECLOSE | COST_LT12 | 0.192 | OK |
| CME_PRECLOSE | ORB_G5 | 0.174 | OK |
| CME_PRECLOSE | OVNRNG_50 | 0.005 | OK |
| CME_PRECLOSE | VWAP_MID_ALIGNED | 0.027 | OK |
| CME_REOPEN | ATR_P50 | 0.040 | OK |
| CME_REOPEN | COST_LT12 | 0.258 | OK |
| CME_REOPEN | ORB_G5 | 0.262 | OK |
| CME_REOPEN | OVNRNG_50 | 0.121 | OK |
| CME_REOPEN | VWAP_MID_ALIGNED | 0.130 | OK |
| COMEX_SETTLE | ATR_P50 | 0.037 | OK |
| COMEX_SETTLE | COST_LT12 | 0.267 | OK |
| COMEX_SETTLE | ORB_G5 | 0.279 | OK |
| COMEX_SETTLE | OVNRNG_50 | 0.007 | OK |
| COMEX_SETTLE | VWAP_MID_ALIGNED | 0.018 | OK |
| EUROPE_FLOW | ATR_P50 | 0.004 | OK |
| EUROPE_FLOW | COST_LT12 | 0.211 | OK |
| EUROPE_FLOW | ORB_G5 | 0.208 | OK |
| EUROPE_FLOW | OVNRNG_50 | 0.044 | OK |
| EUROPE_FLOW | VWAP_MID_ALIGNED | 0.129 | OK |
| LONDON_METALS | ATR_P50 | 0.052 | OK |
| LONDON_METALS | COST_LT12 | 0.287 | OK |
| LONDON_METALS | ORB_G5 | 0.310 | OK |
| LONDON_METALS | OVNRNG_50 | 0.191 | OK |
| LONDON_METALS | VWAP_MID_ALIGNED | 0.042 | OK |
| NYSE_CLOSE | ATR_P50 | 0.073 | OK |
| NYSE_CLOSE | COST_LT12 | 0.212 | OK |
| NYSE_CLOSE | ORB_G5 | 0.190 | OK |
| NYSE_CLOSE | OVNRNG_50 | 0.105 | OK |
| NYSE_CLOSE | VWAP_MID_ALIGNED | 0.013 | OK |
| NYSE_OPEN | ATR_P50 | 0.037 | OK |
| NYSE_OPEN | COST_LT12 | 0.083 | OK |
| NYSE_OPEN | ORB_G5 | 0.048 | OK |
| NYSE_OPEN | OVNRNG_50 | 0.065 | OK |
| NYSE_OPEN | VWAP_MID_ALIGNED | 0.017 | OK |
| SINGAPORE_OPEN | ATR_P50 | 0.093 | OK |
| SINGAPORE_OPEN | COST_LT12 | 0.318 | OK |
| SINGAPORE_OPEN | ORB_G5 | 0.290 | OK |
| SINGAPORE_OPEN | OVNRNG_50 | 0.288 | OK |
| SINGAPORE_OPEN | VWAP_MID_ALIGNED | 0.134 | OK |
| TOKYO_OPEN | ATR_P50 | 0.016 | OK |
| TOKYO_OPEN | COST_LT12 | 0.216 | OK |
| TOKYO_OPEN | ORB_G5 | 0.257 | OK |
| TOKYO_OPEN | OVNRNG_50 | 0.256 | OK |
| TOKYO_OPEN | VWAP_MID_ALIGNED | 0.161 | OK |
| US_DATA_1000 | ATR_P50 | 0.010 | OK |
| US_DATA_1000 | COST_LT12 | 0.124 | OK |
| US_DATA_1000 | ORB_G5 | 0.080 | OK |
| US_DATA_1000 | OVNRNG_50 | 0.014 | OK |
| US_DATA_1000 | VWAP_MID_ALIGNED | 0.041 | OK |
| US_DATA_830 | ATR_P50 | 0.018 | OK |
| US_DATA_830 | COST_LT12 | 0.261 | OK |
| US_DATA_830 | ORB_G5 | 0.274 | OK |
| US_DATA_830 | OVNRNG_50 | 0.047 | OK |
| US_DATA_830 | VWAP_MID_ALIGNED | 0.032 | OK |

## Final Call

- `MGC`: `F1_Q5_only` is `CANDIDATE_READY_IS` mathematically, but `NOT_DEPLOYABLE_AS_E2_FILTER` because `rel_vol` is break-bar-volume based. Fresh OOS confirmation is therefore moot unless the execution framing changes.
- `MES`: `F1_Q5_only` is `CANDIDATE_READY_IS` mathematically, but `NOT_DEPLOYABLE_AS_E2_FILTER` because `rel_vol` is break-bar-volume based. Fresh OOS confirmation is therefore moot unless the execution framing changes.

- Highest-EV next action: if this lineage stays open, reframe it as an execution-safe post-break role (for example an entry-model-switch or confirmation model that resolves after the break bar), not as an `E2` pre-entry filter.


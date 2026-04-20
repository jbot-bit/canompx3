# MNQ live-context overlays — exact deployed NYSE_OPEN / COMEX_SETTLE lanes × 5 instant-detectable hypotheses (K=10)

**Generated:** 2026-04-20T09:49:26+00:00
**Pre-reg:** `docs/audit/hypotheses/2026-04-20-mnq-live-context-overlays-v1.yaml` (LOCKED, commit_sha=c6ece8a1)
**Script:** `research/mnq_live_context_overlays_v1.py`
**IS window:** trading_day < 2026-01-01
**Observed tests:** 10

**Family verdict:** CONTINUE=1 | PARK=2 | KILL=2

## Summary

- `H03/H04` frozen IS-only rel_vol threshold on COMEX short lane: `2.012172`
- `q_family` computed across all `10` two-pass tests
- `q_lane` computed within each physical lane bucket only

| Hypothesis | Lane | Verdict | Unfiltered q_family | Filtered q_family | Filtered q_lane |
|---|---|---|---:|---:|---:|
| H01_NYO_SHORT_PREV_BEAR | MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | PARK | 0.0334 | 0.0285 | 0.0535 |
| H02_NYO_LANE_OPEX_TRUE | MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | KILL | 0.1037 | 0.1037 | 0.1037 |
| H03_CMX_SHORT_RELVOL_Q3 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | PARK | 0.0048 | 0.0095 | 0.0057 |
| H04_CMX_SHORT_RELVOL_Q3_AND_F6 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | CONTINUE | 0.0045 | 0.0048 | 0.0029 |
| H05_CMX_LANE_OPEX_TRUE | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | KILL | 0.0285 | 0.0285 | 0.0200 |

## H01_NYO_SHORT_PREV_BEAR

**Verdict:** PARK

### Unfiltered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unfiltered | 876 | 363 | 472 | 19 | 0.1796 | 0.0322 | 0.1474 | 0.0267 | 0.0334 | 0.0535 | 7 | False |

- `corr_vs_filter=0.0295` | `extreme_fire=False` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 34 | 46 | 0.0100 | True |
| 2020 | 42 | 84 | 0.1300 | True |
| 2021 | 52 | 67 | 0.2578 | True |
| 2022 | 73 | 67 | 0.1607 | True |
| 2023 | 50 | 58 | 0.0527 | True |
| 2024 | 60 | 71 | 0.0764 | True |
| 2025 | 52 | 79 | 0.2449 | True |

### Filtered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| filtered | 866 | 360 | 465 | 19 | 0.1837 | 0.0263 | 0.1574 | 0.0188 | 0.0285 | 0.0535 | 7 | False |

- `corr_vs_filter=0.0295` | `extreme_fire=False` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 33 | 45 | 0.0187 | True |
| 2020 | 41 | 84 | 0.1489 | True |
| 2021 | 52 | 65 | 0.2787 | True |
| 2022 | 73 | 66 | 0.1702 | True |
| 2023 | 49 | 56 | 0.0615 | True |
| 2024 | 60 | 70 | 0.0835 | True |
| 2025 | 52 | 79 | 0.2449 | True |


## H02_NYO_LANE_OPEX_TRUE

**Verdict:** KILL

### Unfiltered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unfiltered | 1764 | 77 | 1616 | 3 | 0.2520 | 0.0726 | 0.1794 | 0.1012 | 0.1037 | 0.1037 | 4 | None |

- `corr_vs_filter=0.0256` | `extreme_fire=True` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 8 | 161 | 0.1629 | False |
| 2020 | 12 | 241 | 0.1034 | True |
| 2021 | 11 | 242 | 0.3596 | True |
| 2022 | 11 | 245 | -0.0442 | True |
| 2023 | 12 | 240 | 0.4616 | True |
| 2024 | 12 | 245 | 0.2640 | True |
| 2025 | 11 | 242 | -0.0721 | True |

### Filtered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| filtered | 1740 | 77 | 1592 | 3 | 0.2520 | 0.0738 | 0.1781 | 0.1037 | 0.1037 | 0.1037 | 4 | None |

- `corr_vs_filter=0.0256` | `extreme_fire=True` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 8 | 153 | 0.1329 | False |
| 2020 | 12 | 237 | 0.0944 | True |
| 2021 | 11 | 238 | 0.3692 | True |
| 2022 | 11 | 243 | -0.0394 | True |
| 2023 | 12 | 235 | 0.4695 | True |
| 2024 | 12 | 244 | 0.2663 | True |
| 2025 | 11 | 242 | -0.0721 | True |


## H03_CMX_SHORT_RELVOL_Q3

**Verdict:** PARK

### Unfiltered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unfiltered | 801 | 252 | 515 | 14 | 0.2458 | -0.0385 | 0.2843 | 0.0013 | 0.0048 | 0.0029 | 5 | False |

- `corr_vs_filter=0.1036` | `extreme_fire=False` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 22 | 50 | 0.3940 | True |
| 2020 | 41 | 71 | 0.2526 | True |
| 2021 | 44 | 72 | 0.5301 | True |
| 2022 | 24 | 96 | -0.1265 | True |
| 2023 | 48 | 73 | 0.5017 | True |
| 2024 | 39 | 73 | -0.0998 | True |
| 2025 | 34 | 80 | 0.3392 | True |

### Filtered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| filtered | 760 | 247 | 479 | 14 | 0.2635 | 0.0016 | 0.2619 | 0.0038 | 0.0095 | 0.0057 | 5 | False |

- `corr_vs_filter=0.1036` | `extreme_fire=False` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 18 | 21 | 0.5595 | True |
| 2020 | 40 | 64 | 0.1706 | True |
| 2021 | 44 | 72 | 0.5301 | True |
| 2022 | 24 | 96 | -0.1265 | True |
| 2023 | 48 | 73 | 0.5017 | True |
| 2024 | 39 | 73 | -0.0998 | True |
| 2025 | 34 | 80 | 0.3392 | True |


## H04_CMX_SHORT_RELVOL_Q3_AND_F6

**Verdict:** CONTINUE

### Unfiltered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unfiltered | 801 | 130 | 637 | 5 | 0.3785 | -0.0111 | 0.3896 | 0.0005 | 0.0045 | 0.0027 | 5 | None |

- `corr_vs_filter=0.0743` | `extreme_fire=False` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 8 | 64 | 1.4348 | False |
| 2020 | 25 | 87 | -0.0449 | True |
| 2021 | 20 | 96 | 0.5629 | True |
| 2022 | 13 | 107 | 0.0167 | True |
| 2023 | 27 | 94 | 0.5886 | True |
| 2024 | 19 | 93 | 0.0181 | True |
| 2025 | 18 | 96 | 0.4689 | True |

### Filtered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| filtered | 760 | 128 | 598 | 5 | 0.3855 | 0.0276 | 0.3579 | 0.0014 | 0.0048 | 0.0029 | 5 | None |

- `corr_vs_filter=0.0743` | `extreme_fire=False` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 6 | 33 | 1.8047 | False |
| 2020 | 25 | 79 | -0.1613 | True |
| 2021 | 20 | 96 | 0.5629 | True |
| 2022 | 13 | 107 | 0.0167 | True |
| 2023 | 27 | 94 | 0.5886 | True |
| 2024 | 19 | 93 | 0.0181 | True |
| 2025 | 18 | 96 | 0.4689 | True |


## H05_CMX_LANE_OPEX_TRUE

**Verdict:** KILL

### Unfiltered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unfiltered | 1702 | 77 | 1559 | 2 | -0.2332 | 0.0815 | 0.3146 | 0.0154 | 0.0285 | 0.0185 | 5 | None |

- `corr_vs_filter=-0.0294` | `extreme_fire=True` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 8 | 153 | 0.3481 | False |
| 2020 | 12 | 235 | 0.4921 | True |
| 2021 | 11 | 235 | 0.4469 | True |
| 2022 | 11 | 238 | 0.1448 | True |
| 2023 | 12 | 234 | -0.0072 | True |
| 2024 | 12 | 229 | 0.1756 | True |
| 2025 | 11 | 235 | 0.6244 | True |

### Filtered Pass

| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| filtered | 1621 | 71 | 1484 | 2 | -0.2143 | 0.1061 | 0.3204 | 0.0200 | 0.0285 | 0.0200 | 5 | None |

- `corr_vs_filter=-0.0294` | `extreme_fire=True` | `arithmetic_only=False`

| Year | N_on | N_off | Delta | Eligible_for_years_positive |
|---|---:|---:|---:|---|
| 2019 | 3 | 93 | 0.8061 | False |
| 2020 | 11 | 223 | 0.4971 | True |
| 2021 | 11 | 232 | 0.4442 | True |
| 2022 | 11 | 238 | 0.1448 | True |
| 2023 | 12 | 234 | -0.0072 | True |
| 2024 | 12 | 229 | 0.1756 | True |
| 2025 | 11 | 235 | 0.6244 | True |

## Closeout

SURVIVED SCRUTINY:
- H04_CMX_SHORT_RELVOL_Q3_AND_F6
PARKED FOR MORE OOS:
- H01_NYO_SHORT_PREV_BEAR
- H03_CMX_SHORT_RELVOL_Q3
DID NOT SURVIVE:
- H02_NYO_LANE_OPEX_TRUE
- H05_CMX_LANE_OPEX_TRUE
CAVEATS:
- OOS remains descriptive only under Mode A and several filtered branches are thin.
- Family q-values are the load-bearing multiple-testing gate; lane q-values are secondary framing only.
NEXT STEPS:
- If no hypothesis survives both passes, kill this exact overlay bundle and do not broaden scope without a new pre-reg.
- If a hypothesis parks, treat it as shadow-only until a fresh pre-reg narrows the unresolved criterion.

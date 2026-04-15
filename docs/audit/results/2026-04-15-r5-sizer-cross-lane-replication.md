# R5 Sizer Cross-Lane Replication — 6 Deployed Lanes

**Date:** 2026-04-15
**Source deploy:** `docs/runtime/lane_allocation.json` rebalance 2026-04-13
**Question:** Does `garch_forecast_vol_pct` add conditional ExpR edge on top of each deployed lane's filter? (R5 sizer hypothesis per `docs/institutional/mechanism_priors.md`.)
**Design:** 6 lanes × 2 directions × 3 thresholds (60/70/80) = 36 primary hypotheses. Two-sided Welch t-test + bootstrap. BH-FDR correction at K=36. Per-year stability. IS-OOS split at 2026-01-01.

**Look-ahead verification:** `garch_forecast_vol_pct` uses `rows[0..i-1] daily_close` per `pipeline/build_daily_features.py:1258` + 252-day prior rank per line 1217. All filter predicates (orb_size, atr_20_pct, overnight_range, VWAP_MID_ALIGNED) are trade-time-knowable at session start or break-bar close. Clean.

**No-pigeonholing:** Two-sided t-test does NOT pre-suppose garch=HIGH is the edge direction. If garch=LOW were actually the informative signal, the test would catch it (negative lift, significant p).

---

## 6 Deployed Lanes

| Lane | Session | Apt | RR | Filter |
|---|---|---|---|---|
| L1_EUROPE_FLOW_ORB_G5_RR1.5 | EUROPE_FLOW | O5 | 1.5 | ORB_G5 |
| L2_SINGAPORE_OPEN_ATR_P50_O30_RR1.5 | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 |
| L3_COMEX_SETTLE_OVNRNG_100_RR1.5 | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 |
| L4_NYSE_OPEN_ORB_G5_RR1.0 | NYSE_OPEN | O5 | 1.0 | ORB_G5 |
| L5_TOKYO_OPEN_ORB_G5_RR1.5 | TOKYO_OPEN | O5 | 1.5 | ORB_G5 |
| L6_US_DATA_1000_VWAP_MID_ALIGNED_O15_RR1.5 | US_DATA_1000 | O15 | 1.5 | VWAP_MID_ALIGNED |

---

## Per-lane per-direction test grid

### L1_EUROPE_FLOW_ORB_G5_RR1.5 | long

**IS N:** 673 | **OOS N:** 35

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-12 | 135 | +0.072 | 48.1% |
| 1 | 12-31 | 134 | +0.059 | 47.8% |
| 2 | 32-54 | 137 | +0.022 | 46.0% |
| 3 | 55-78 | 134 | +0.066 | 47.0% |
| 4 | 79-100 | 133 | +0.258 | 54.1% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 237/436 | +0.147 | +0.067 | +0.081 | +0.87 | 0.3826 | 0.3826 | 5/6 | — | +0.690 |
| 70 | 178/495 | +0.146 | +0.076 | +0.070 | +0.69 | 0.4882 | 0.5145 | 3/6 | — | +1.048 |
| 80 | 130/543 | +0.252 | +0.057 | +0.194 | +1.72 | 0.0868 | 0.0889 | 4/5 | — | +0.355 |

### L1_EUROPE_FLOW_ORB_G5_RR1.5 | short

**IS N:** 723 | **OOS N:** 32

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-11 | 145 | +0.140 | 51.7% |
| 1 | 11-30 | 145 | +0.009 | 45.5% |
| 2 | 31-50 | 144 | +0.031 | 45.8% |
| 3 | 51-79 | 145 | +0.061 | 46.9% |
| 4 | 80-100 | 144 | +0.104 | 47.2% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 249/474 | +0.093 | +0.057 | +0.036 | +0.40 | 0.6862 | 0.6723 | 3/6 | — | +0.032 |
| 70 | 201/522 | +0.089 | +0.061 | +0.027 | +0.28 | 0.7765 | 0.7792 | 4/6 | — | +0.613 |
| 80 | 139/584 | +0.078 | +0.067 | +0.011 | +0.10 | 0.9189 | 0.9261 | 3/4 | — | +0.196 |

### L2_SINGAPORE_OPEN_ATR_P50_O30_RR1.5 | long

**IS N:** 412 | **OOS N:** 28

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-37 | 83 | +0.062 | 45.8% |
| 1 | 38-60 | 82 | +0.173 | 50.0% |
| 2 | 60-75 | 82 | +0.341 | 57.3% |
| 3 | 75-88 | 82 | +0.293 | 54.9% |
| 4 | 89-100 | 83 | +0.214 | 50.6% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 245/167 | +0.293 | +0.104 | +0.189 | +1.61 | 0.1093 | 0.1139 | 3/6 | — | -0.631 |
| 70 | 197/215 | +0.230 | +0.204 | +0.026 | +0.23 | 0.8213 | 0.7992 | 2/6 | — | +0.010 |
| 80 | 138/274 | +0.207 | +0.221 | -0.013 | -0.11 | 0.9141 | 0.8941 | 0/5 | — | -0.084 |

### L2_SINGAPORE_OPEN_ATR_P50_O30_RR1.5 | short

**IS N:** 358 | **OOS N:** 26

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-35 | 72 | -0.003 | 43.1% |
| 1 | 35-58 | 73 | -0.063 | 39.7% |
| 2 | 58-74 | 71 | +0.098 | 46.5% |
| 3 | 74-89 | 70 | +0.012 | 42.9% |
| 4 | 90-100 | 72 | +0.055 | 44.4% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 205/153 | +0.050 | -0.023 | +0.073 | +0.59 | 0.5589 | 0.5794 | 4/5 | — | -0.122 |
| 70 | 165/193 | +0.034 | +0.006 | +0.028 | +0.22 | 0.8225 | 0.8392 | 3/6 | — | -0.171 |
| 80 | 123/235 | +0.003 | +0.028 | -0.025 | -0.19 | 0.8503 | 0.8492 | 3/5 | — | -0.484 |

### L3_COMEX_SETTLE_OVNRNG_100_RR1.5 | long

**IS N:** 249 | **OOS N:** 27

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-33 | 50 | -0.041 | 42.0% |
| 1 | 34-59 | 50 | +0.156 | 50.0% |
| 2 | 60-79 | 50 | +0.072 | 46.0% |
| 3 | 80-92 | 49 | +0.449 | 61.2% |
| 4 | 92-100 | 50 | +0.390 | 58.0% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 147/102 | +0.304 | +0.060 | +0.244 | +1.62 | 0.1057 | 0.1179 | 4/5 | — | +0.412 |
| 70 | 125/124 | +0.348 | +0.060 | +0.288 | +1.94 | 0.0532 | 0.0589 | 5/5 | — | +0.795 |
| 80 | 98/151 | +0.434 | +0.055 | +0.378 | +2.50 | 0.0131 | 0.0200 | 5/5 | — | -0.217 |

### L3_COMEX_SETTLE_OVNRNG_100_RR1.5 | short

**IS N:** 211 | **OOS N:** 31

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-34 | 43 | +0.079 | 46.5% |
| 1 | 34-59 | 42 | +0.114 | 47.6% |
| 2 | 61-76 | 42 | -0.002 | 42.9% |
| 3 | 78-90 | 42 | +0.467 | 61.9% |
| 4 | 92-100 | 42 | +0.311 | 54.8% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 126/85 | +0.259 | +0.096 | +0.163 | +0.98 | 0.3264 | 0.3167 | 4/5 | — | +0.251 |
| 70 | 106/105 | +0.364 | +0.020 | +0.344 | +2.13 | 0.0339 | 0.0360 | 5/5 | — | +0.560 |
| 80 | 79/132 | +0.417 | +0.059 | +0.358 | +2.15 | 0.0334 | 0.0320 | 5/5 | — | +0.309 |

### L4_NYSE_OPEN_ORB_G5_RR1.0 | long

**IS N:** 718 | **OOS N:** 29

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-12 | 147 | +0.102 | 57.1% |
| 1 | 12-29 | 143 | +0.199 | 62.2% |
| 2 | 30-47 | 142 | +0.020 | 52.8% |
| 3 | 48-75 | 142 | +0.067 | 54.9% |
| 4 | 75-100 | 144 | +0.043 | 53.5% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 224/494 | +0.027 | +0.113 | -0.087 | -1.11 | 0.2676 | 0.2677 | 1/6 | — | -0.099 |
| 70 | 176/542 | +0.030 | +0.105 | -0.074 | -0.88 | 0.3784 | 0.3756 | 1/6 | — | +0.060 |
| 80 | 124/594 | +0.023 | +0.100 | -0.077 | -0.80 | 0.4274 | 0.3996 | 1/5 | — | +0.336 |

### L4_NYSE_OPEN_ORB_G5_RR1.0 | short

**IS N:** 705 | **OOS N:** 37

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-10 | 142 | +0.139 | 59.2% |
| 1 | 10-31 | 141 | +0.078 | 56.0% |
| 2 | 31-56 | 140 | +0.009 | 52.1% |
| 3 | 56-80 | 141 | +0.141 | 58.9% |
| 4 | 81-100 | 141 | +0.178 | 60.3% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 260/445 | +0.169 | +0.074 | +0.095 | +1.27 | 0.2030 | 0.1758 | 3/6 | — | +0.449 |
| 70 | 201/504 | +0.136 | +0.099 | +0.038 | +0.47 | 0.6393 | 0.6344 | 2/6 | — | +0.468 |
| 80 | 145/560 | +0.172 | +0.093 | +0.079 | +0.89 | 0.3757 | 0.3576 | 4/5 | — | +0.448 |

### L5_TOKYO_OPEN_ORB_G5_RR1.5 | long

**IS N:** 682 | **OOS N:** 34

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-12 | 137 | -0.009 | 45.3% |
| 1 | 12-31 | 136 | -0.048 | 44.1% |
| 2 | 31-52 | 136 | +0.156 | 52.2% |
| 3 | 52-78 | 137 | +0.145 | 51.1% |
| 4 | 78-100 | 136 | +0.169 | 50.7% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 228/454 | +0.130 | +0.059 | +0.072 | +0.78 | 0.4337 | 0.3776 | 4/6 | — | -0.526 |
| 70 | 179/503 | +0.139 | +0.063 | +0.076 | +0.77 | 0.4426 | 0.4016 | 4/6 | — | -0.426 |
| 80 | 124/558 | +0.140 | +0.070 | +0.070 | +0.61 | 0.5407 | 0.5025 | 4/5 | — | +0.112 |

### L5_TOKYO_OPEN_ORB_G5_RR1.5 | short

**IS N:** 712 | **OOS N:** 33

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-12 | 145 | -0.015 | 45.5% |
| 1 | 13-31 | 145 | +0.089 | 49.7% |
| 2 | 32-54 | 137 | +0.124 | 51.1% |
| 3 | 55-81 | 142 | -0.017 | 43.7% |
| 4 | 81-100 | 143 | +0.182 | 51.0% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 260/452 | +0.108 | +0.051 | +0.057 | +0.65 | 0.5189 | 0.5145 | 4/6 | — | -0.047 |
| 70 | 201/511 | +0.120 | +0.053 | +0.066 | +0.70 | 0.4855 | 0.4855 | 3/6 | — | -0.059 |
| 80 | 147/565 | +0.180 | +0.044 | +0.136 | +1.27 | 0.2041 | 0.1978 | 4/4 | — | +0.411 |

### L6_US_DATA_1000_VWAP_MID_ALIGNED_O15_RR1.5 | long

**IS N:** 363 | **OOS N:** 18

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-12 | 75 | +0.255 | 52.0% |
| 1 | 12-29 | 71 | +0.323 | 54.9% |
| 2 | 29-50 | 72 | +0.079 | 44.4% |
| 3 | 50-76 | 72 | -0.053 | 38.9% |
| 4 | 76-100 | 73 | +0.136 | 46.6% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 116/247 | +0.072 | +0.184 | -0.112 | -0.82 | 0.4122 | 0.4396 | 2/6 | — | +0.815 |
| 70 | 98/265 | +0.145 | +0.149 | -0.004 | -0.03 | 0.9778 | 1.0000 | 3/5 | — | +0.650 |
| 80 | 66/297 | +0.220 | +0.132 | +0.088 | +0.53 | 0.6002 | 0.5874 | 3/4 | — | +1.413 |

### L6_US_DATA_1000_VWAP_MID_ALIGNED_O15_RR1.5 | short

**IS N:** 286 | **OOS N:** 17

NTILE-5 by garch_pct (informational — monotonicity check):

| bucket | garch range | N | ExpR | WR |
|---|---|---|---|---|
| 0 | 0-11 | 59 | +0.272 | 52.5% |
| 1 | 12-31 | 56 | +0.170 | 48.2% |
| 2 | 31-58 | 57 | +0.322 | 54.4% |
| 3 | 59-81 | 57 | +0.202 | 49.1% |
| 4 | 82-100 | 57 | +0.335 | 54.4% |

Threshold tests:

| thresh | N_on/off | ExpR_on | ExpR_off | lift | t | p_two | p_boot | yrs+ | BH-FDR | OOS lift |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 110/176 | +0.270 | +0.255 | +0.016 | +0.11 | 0.9164 | 0.9101 | 3/6 | — | -0.422 |
| 70 | 84/202 | +0.227 | +0.275 | -0.048 | -0.30 | 0.7635 | 0.8352 | 2/5 | — | +0.133 |
| 80 | 62/224 | +0.267 | +0.259 | +0.008 | +0.04 | 0.9643 | 0.9051 | 2/4 | — | -1.399 |

---

## Summary — BH-FDR at K=36

- Total tested: 36 (non-skipped).
- BH-FDR survivors (q=0.05): **0**.
- Positive-lift survivors (garch=HIGH better): 0.
- Negative-lift survivors (garch=LOW better → SKIP signal): 0.


---

## Interpretation & deployment roles

| Finding | Deployment role (per mechanism_priors.md) |
|---|---|
| Significant positive lift + dir_match across thresholds | **R5 SIZER** on that lane — garch=HIGH days size up |
| Significant negative lift (garch=LOW → better) | **R1-SKIP** — trade LOW garch days, skip HIGH |
| No lift at any threshold | **R0 null** — lane unaffected by garch |
| Lift IS but OOS flip / yrs < 50% | **R? regime-dependent** — needs shadow before any role |

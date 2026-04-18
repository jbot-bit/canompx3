# Regime-drift control on 4 CRITICAL committee lanes — 2026-04-19

**Generated:** 2026-04-18T22:37:03+00:00
**Script:** `research/regime_drift_control_critical_lanes.py`
**IS boundary:** `trading_day < 2026-01-01` (Mode A)

## Motivation

The 2026-04-19 Phase 8 committee review pack flagged 4 MNQ lanes as CRITICAL for RETIRE/DOWNGRADE based on Mode A Sharpe drop from stored values. The 2026-04-19 self-audit identified a framing bias: attributing the drop to LANE DECAY without controlling for 2024-2025 ENVIRONMENT-WIDE regime stress on MNQ intraday breakouts.

This control test:
  1. Recomputes each CRITICAL lane's Sharpe in early (2022-2023) vs late (2024-2025) Mode A IS subsets
  2. Computes portfolio-wide MNQ aggregate Sharpe in same subsets (all 36 active MNQ lanes pooled)
  3. Compares each CRITICAL lane's Sharpe drop vs the portfolio-wide drop

Interpretive thresholds:
  - **REGIME** (hold retirement): lane Sharpe drop within 0.30 of portfolio-wide drop
  - **DECAY** (retirement recommendation stands): lane drops >0.50 beyond environment
  - **BETTER-THAN-PEERS** (counter to committee framing): lane drops LESS than environment or rises

## Portfolio-wide MNQ aggregate

Early (2022-2023): N=5886 ExpR=0.125 Sharpe_ann=6.06
Late  (2024-2025): N=6809 ExpR=0.108 Sharpe_ann=5.65
**Portfolio-wide Sharpe drop early->late:** -0.41

This is the baseline environment delta against which the 4 CRITICAL lanes should be measured.

## CRITICAL lanes — per-lane drift vs environment

| Cell | Early 2022-2023 Sharpe | Late 2024-2025 Sharpe | Lane drop | Portfolio drop | Excess drop vs port | Verdict |
|---|---:|---:|---:|---:|---:|---|
| CR1 MNQ EUROPE_FLOW O5 RR1.0 OVNRNG_100 long | 0.78 (N=73) | 0.43 (N=95) | -0.36 | -0.41 | 0.05 | REGIME (hold) |
| CR2 MNQ EUROPE_FLOW O5 RR1.5 OVNRNG_100 long | 0.75 (N=73) | 0.73 (N=95) | -0.02 | -0.41 | 0.39 | BETTER-THAN-PEERS (keep) |
| CR3 MNQ NYSE_OPEN O5 RR1.0 X_MES_ATR60 long | 0.47 (N=82) | 0.95 (N=144) | 0.49 | -0.41 | 0.90 | BETTER-THAN-PEERS (keep) |
| CR4 MNQ NYSE_OPEN O5 RR1.5 X_MES_ATR60 long | 0.59 (N=81) | 0.40 (N=138) | -0.20 | -0.41 | 0.21 | REGIME (hold) |

## All 36 MNQ active lanes — early/late Sharpe + drop

(Sorted by Sharpe drop; most-decayed first)

| Strategy ID | Early Sharpe (N) | Late Sharpe (N) | Drop |
|---|---:|---:|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` | 1.59 (N=150) | 0.18 (N=177) | -1.41 |
| `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | 1.48 (N=150) | 0.21 (N=177) | -1.28 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | 1.42 (N=181) | 0.16 (N=189) | -1.26 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 0.74 (N=238) | -0.43 (N=240) | -1.17 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 1.39 (N=241) | 0.25 (N=250) | -1.14 |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | 1.20 (N=150) | 0.19 (N=177) | -1.01 |
| `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` | 2.20 (N=70) | 1.29 (N=143) | -0.91 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | 1.18 (N=130) | 0.37 (N=129) | -0.81 |
| `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5` | 0.92 (N=241) | 0.12 (N=250) | -0.80 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 0.98 (N=136) | 0.37 (N=138) | -0.61 |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 0.79 (N=241) | 0.20 (N=250) | -0.59 |
| `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 0.81 (N=126) | 0.29 (N=171) | -0.53 |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 0.64 (N=181) | 0.12 (N=189) | -0.51 |
| `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | 0.48 (N=115) | 0.07 (N=118) | -0.41 |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | 0.78 (N=73) | 0.43 (N=95) | -0.36 |
| `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | 0.59 (N=81) | 0.40 (N=138) | -0.20 |
| `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | 1.70 (N=251) | 1.52 (N=255) | -0.18 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12` | 1.73 (N=231) | 1.57 (N=243) | -0.15 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | 1.72 (N=254) | 1.70 (N=265) | -0.02 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | 0.75 (N=73) | 0.73 (N=95) | -0.02 |
| `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` | 1.08 (N=254) | 1.10 (N=238) | 0.02 |
| `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5` | 0.64 (N=238) | 0.69 (N=235) | 0.06 |
| `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | 1.04 (N=257) | 1.10 (N=238) | 0.06 |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 0.64 (N=126) | 0.89 (N=171) | 0.25 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | 1.92 (N=95) | 2.25 (N=162) | 0.33 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | 0.47 (N=82) | 0.95 (N=144) | 0.49 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 1.36 (N=254) | 1.86 (N=261) | 0.50 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60` | 0.70 (N=84) | 1.34 (N=155) | 0.64 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | 0.68 (N=260) | 1.40 (N=248) | 0.73 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 0.66 (N=257) | 1.40 (N=248) | 0.75 |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | 0.18 (N=238) | 0.92 (N=235) | 0.75 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 1.46 (N=101) | 2.43 (N=199) | 0.96 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | 1.25 (N=95) | 2.29 (N=158) | 1.04 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | 0.75 (N=64) | 2.04 (N=122) | 1.28 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | 1.32 (N=104) | 2.71 (N=186) | 1.39 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | 0.53 (N=64) | 1.94 (N=120) | 1.42 |

## Interpretation

Portfolio of 36 MNQ lanes: median Sharpe drop = -0.09, IQR [-0.66, 0.54].

- Lanes with DECAY (Sharpe drop > 0.50): 13 / 36
- Lanes with mild drop (0.20-0.50): 2
- Lanes roughly flat (|drop| < 0.20): 8
- Lanes with Sharpe UP: 13

## Recommendation for the committee review pack

ALL CRITICAL lanes show drops within 0.30 of portfolio-wide drop — REGIME framing. The committee pack's RETIRE framing was over-attributed to lane decay. Recommended action: HOLD on retirement; reclassify as regime-stressed, continue monitoring.

## Reproduction
```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/regime_drift_control_critical_lanes.py
```

Read-only. No writes.


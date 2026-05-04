# MES broader Mode A rediscovery — K=6 scan

**Generated:** 2026-04-18T16:16:00+00:00
**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mes-broader-mode-a-rediscovery-v1.yaml` LOCKED (commit 66964b74)
**Script:** `research/mes_broader_mode_a_rediscovery_v1_scan.py`
**IS:** `trading_day < 2026-01-01`

## Summary: 6 cells | CONTINUE: 1 | KILL: 5

**K2 baseline sanity smoke-test:** PASS

| Cell | Session | Dir | RR | Filter | N_base | N_on | Fire% | ExpR_b | ExpR_on | Δ_IS | t | raw_p | boot_p | q | yrs+ |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1_CMX_PRE_ORB_G8_L_RR1 | CME_PRECLOSE | long | 1.0 | ORB_G8 | 722 | 130 | 0.180 | 0.0254 | 0.2798 | 0.2543 | 3.656 | 0.0004 | 0.0001 | 0.0022 | 4 |
| H2_CMX_PRE_COST_LT12_L_RR1 | CME_PRECLOSE | long | 1.0 | COST_LT12 | 722 | 234 | 0.324 | 0.0254 | 0.1558 | 0.1303 | 2.672 | 0.0081 | 0.0072 | 0.0162 | 5 |
| H3_CMX_PRE_ATR_P70_L_RR1 | CME_PRECLOSE | long | 1.0 | ATR_P70 | 722 | 239 | 0.331 | 0.0254 | 0.1422 | 0.1168 | 2.590 | 0.0102 | 0.0055 | 0.0162 | 5 |
| H4_USD1k_ORB_G5_S_RR15 | US_DATA_1000 | short | 1.5 | ORB_G5 | 814 | 573 | 0.704 | 0.0283 | 0.1191 | 0.0908 | 2.477 | 0.0135 | 0.0101 | 0.0162 | 7 |
| H5_USD1k_ORB_G5_S_RR2 | US_DATA_1000 | short | 2.0 | ORB_G5 | 797 | 559 | 0.701 | 0.0487 | 0.1465 | 0.0977 | 2.549 | 0.0111 | 0.0088 | 0.0162 | 7 |
| H6_USD1k_COST_LT12_S_RR15 | US_DATA_1000 | short | 1.5 | COST_LT12 | 814 | 504 | 0.619 | 0.0283 | 0.1160 | 0.0876 | 2.247 | 0.0251 | 0.0171 | 0.0251 | 7 |

## Gate breakdown

| Cell | bh_pass | abs_t_ge_3 | N_on_ge_100 | years_pos_ge_4 | boot_p_lt_0.10 | ExpR_gt_0 | not_taut | not_ext_fire | not_arith | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H1_CMX_PRE_ORB_G8_L_RR1 | Y | Y | Y | Y | Y | Y | Y | Y | Y | CONTINUE |
| H2_CMX_PRE_COST_LT12_L_RR1 | Y | N | Y | Y | Y | Y | Y | Y | Y | KILL |
| H3_CMX_PRE_ATR_P70_L_RR1 | Y | N | Y | Y | Y | Y | Y | Y | Y | KILL |
| H4_USD1k_ORB_G5_S_RR15 | Y | N | Y | Y | Y | Y | Y | Y | Y | KILL |
| H5_USD1k_ORB_G5_S_RR2 | Y | N | Y | Y | Y | Y | Y | Y | Y | KILL |
| H6_USD1k_COST_LT12_S_RR15 | Y | N | Y | Y | Y | Y | Y | Y | Y | KILL |

## T0 / flags

| Cell | corr_orbsize (expected ~1 for ORB_G) | corr_atr | tautology | extreme_fire | arith_only |
|---|---:|---:|---|---|---|
| H1_CMX_PRE_ORB_G8_L_RR1 | 0.755 | 0.470 | N | N | N |
| H2_CMX_PRE_COST_LT12_L_RR1 | 0.707 | 0.469 | N | N | N |
| H3_CMX_PRE_ATR_P70_L_RR1 | 0.358 | 0.575 | N | N | N |
| H4_USD1k_ORB_G5_S_RR15 | 0.532 | 0.382 | N | N | N |
| H5_USD1k_ORB_G5_S_RR2 | 0.538 | 0.392 | N | N | N |
| H6_USD1k_COST_LT12_S_RR15 | 0.589 | 0.398 | N | N | N |

## OOS descriptive

| Cell | N_OOS_on | ExpR_OOS | Δ_OOS | dir_match |
|---|---:|---:|---:|---|
| H1_CMX_PRE_ORB_G8_L_RR1 | 4 | 0.3922 | 0.3825 | Y |
| H2_CMX_PRE_COST_LT12_L_RR1 | 13 | -0.0185 | -0.0283 | N |
| H3_CMX_PRE_ATR_P70_L_RR1 | 16 | 0.1099 | 0.1002 | Y |
| H4_USD1k_ORB_G5_S_RR15 | 36 | -0.0836 | 0.0159 | Y |
| H5_USD1k_ORB_G5_S_RR2 | 35 | 0.0489 | 0.0156 | Y |
| H6_USD1k_COST_LT12_S_RR15 | 35 | -0.0574 | 0.0421 | Y |

## Per-year IS

| Cell | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1_CMX_PRE_ORB_G8_L_RR1 | — | +0.489(N=24) | +0.341(N=11) | +0.326(N=45) | N=6 | --0.071(N=10) | +0.163(N=34) |
| H2_CMX_PRE_COST_LT12_L_RR1 | N=4 | +0.416(N=43) | +0.340(N=19) | +0.234(N=63) | +0.007(N=18) | +0.018(N=32) | --0.020(N=55) |
| H3_CMX_PRE_ATR_P70_L_RR1 | --0.182(N=11) | +0.325(N=41) | +0.235(N=22) | +0.246(N=43) | N=1 | +0.021(N=60) | +0.108(N=61) |
| H4_USD1k_ORB_G5_S_RR15 | +0.032(N=24) | +0.013(N=68) | +0.183(N=77) | +0.055(N=117) | +0.139(N=94) | +0.098(N=102) | +0.252(N=91) |
| H5_USD1k_ORB_G5_S_RR2 | +0.238(N=24) | +0.057(N=68) | +0.174(N=72) | +0.067(N=115) | +0.171(N=93) | +0.133(N=99) | +0.261(N=88) |
| H6_USD1k_COST_LT12_S_RR15 | +0.000(N=16) | +0.020(N=59) | +0.258(N=62) | +0.053(N=111) | +0.119(N=82) | +0.075(N=92) | +0.229(N=82) |

## Decision

**Verdict: CONTINUE on 1 cell(s) — validated-candidates requiring committee review.**
  - H1_CMX_PRE_ORB_G8_L_RR1: CME_PRECLOSE long RR1.0 ORB_G8 N=130 ExpR=0.280 t=3.656 q=0.0022

## Reproduction
```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mes_broader_mode_a_rediscovery_v1_scan.py
```

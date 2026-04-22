# 2026-04-22 MNQ Prior-Day Family Board v1

Bounded read-only board on canonical `daily_features x orb_outcomes`.

- Parents: 3 live-adjacent positive MNQ lanes
- Families: 3 prior-day mechanism families
- K_family: 9
- Holdout split: IS < 2026-01-01, OOS >= 2026-01-01

| Lane | Family | Role | N_on_IS | ExpR_on_IS | ExpR_off_IS | Delta_IS | t | BH | N_on_OOS | Delta_OOS | OOS sign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| US_DATA_1000 RR1.0 long | TAKE_DOWNSIDE_DISPLACEMENT | TAKE | 205 | +0.2320 | -0.0171 | +0.2491 | +3.37 | 0.0040 | 10 | +0.3188 | True |
| US_DATA_1000 RR1.0 long | AVOID_CONGESTION | AVOID | 649 | -0.0216 | +0.2155 | -0.2371 | -3.35 | 0.0040 | 22 | -0.3967 | True |
| NYSE_OPEN RR1.5 long | TAKE_OVERHEAD_BREAK | TAKE | 320 | +0.1705 | +0.0293 | +0.1412 | +1.65 | 0.2953 | 7 | +0.6962 | True |
| NYSE_OPEN RR1.5 long | AVOID_CONGESTION | AVOID | 633 | +0.0520 | +0.1859 | -0.1339 | -1.36 | 0.3515 | 21 | -0.7023 | True |
| US_DATA_1000 RR1.0 long | TAKE_OVERHEAD_BREAK | TAKE | 329 | +0.0852 | +0.0144 | +0.0707 | +1.08 | 0.3601 | 9 | +0.4611 | True |
| CME_PRECLOSE RR1.0 long | TAKE_DOWNSIDE_DISPLACEMENT | AVOID | 221 | +0.0739 | +0.1607 | -0.0868 | -1.19 | 0.3515 | 6 | +0.2811 | False |
| CME_PRECLOSE RR1.0 long | TAKE_OVERHEAD_BREAK | TAKE | 326 | +0.1810 | +0.0988 | +0.0822 | +1.26 | 0.3515 | 15 | -0.7965 | False |
| NYSE_OPEN RR1.5 long | TAKE_DOWNSIDE_DISPLACEMENT | AVOID | 184 | +0.0561 | +0.0917 | -0.0356 | -0.35 | 0.7645 | 5 | +0.3059 | False |
| CME_PRECLOSE RR1.0 long | AVOID_CONGESTION | AVOID | 421 | +0.1263 | +0.1461 | -0.0198 | -0.30 | 0.7645 | 21 | +0.1387 | False |

## Family definitions

- `TAKE_DOWNSIDE_DISPLACEMENT`: `F2_NEAR_PDL_15 OR F5_BELOW_PDL`
- `AVOID_CONGESTION`: `F3_NEAR_PIVOT_50 OR F6_INSIDE_PDR`
- `TAKE_OVERHEAD_BREAK`: `F1_NEAR_PDH_15 OR F4_ABOVE_PDH`

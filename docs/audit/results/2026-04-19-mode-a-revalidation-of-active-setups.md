# Mode A canonical re-validation of active validated_setups

**Generated:** 2026-04-18T15:56:27+00:00
**Script:** `research/mode_a_revalidation_active_setups.py`
**IS boundary:** `trading_day < 2026-01-01` (Mode A)
**Canonical filter source:** `research.filter_utils.filter_signal` → `trading_app.config.ALL_FILTERS`

## Summary

- Total active lanes re-validated: **38**
- Lanes with material Mode A drift (|ΔN/N|>10% OR |ΔExpR|>0.03 OR |ΔSharpe|>0.20 OR Mode-B contaminated): **38**
- Lanes with last_trade_day >= 2026-01-01 (Mode-B grandfathered): **9**

## Thresholds

- N ratio: |ΔN / stored_N| > 0.1 → flag
- ExpR absolute: |ΔExpR| > 0.03 → flag
- Sharpe absolute: |ΔSharpe_ann| > 0.2 → flag
- Mode-B contaminated: `last_trade_day >= 2026-01-01` → flag

A flagged lane does NOT mean the lane is wrong — it means the stored
validated_setups values are computed on a different IS window than strict
Mode A, and downstream decisions that cited those values are partially
built on Mode-B baseline data. Treat the Mode A column as the canonical
truth going forward; the validated_setups rows themselves are NOT mutated
by this audit.

## Per-lane re-validation

| Instr | Session | Om | RR | Filter | Dir | Stored N / Mode-A N | ΔN/N | Stored ExpR / Mode-A ExpR | ΔExpR | Stored Sh / Mode-A Sh | ΔSh | Yrs+ | Mode-B | Flag |
|---|---|---:|---:|---|---|---|---:|---|---:|---|---:|---:|---|---|
| MES | CME_PRECLOSE | 5 | 1.0 | COST_LT08 | long | 194 / 88 | -0.55 | 0.1962 / 0.3279 | 0.1317 | 1.25 / 1.45 | 0.20 | 3/3 | N | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | ORB_G8 | long | 287 / 130 | -0.55 | 0.1729 / 0.2798 | 0.1069 | 1.34 / 1.49 | 0.15 | 4/5 | N | DRIFT |
| MNQ | CME_PRECLOSE | 5 | 1.0 | X_MES_ATR60 | long | 596 / 306 | -0.49 | 0.1702 / 0.2140 | 0.0438 | 1.88 / 1.58 | -0.30 | 6/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | COST_LT12 | long | 1247 / 669 | -0.46 | 0.1104 / 0.1041 | -0.0063 | 1.74 / 1.11 | -0.63 | 6/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | ORB_G5 | long | 1473 / 836 | -0.43 | 0.0890 / 0.0780 | -0.0110 | 1.54 / 0.95 | -0.59 | 5/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | OVNRNG_100 | long | 520 / 283 | -0.46 | 0.1725 / 0.1844 | 0.0119 | 1.76 / 1.29 | -0.47 | 5/5 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | X_MES_ATR60 | long | 673 / 385 | -0.43 | 0.1512 / 0.1950 | 0.0438 | 1.78 / 1.62 | -0.16 | 7/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.5 | ORB_G5 | long | 1459 / 829 | -0.43 | 0.1119 / 0.0922 | -0.0197 | 1.52 / 0.88 | -0.64 | 5/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.5 | OVNRNG_100 | long | 513 / 278 | -0.46 | 0.2151 / 0.1868 | -0.0283 | 1.70 / 1.00 | -0.69 | 5/5 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.5 | X_MES_ATR60 | long | 664 / 379 | -0.43 | 0.1609 / 0.1981 | 0.0372 | 1.46 / 1.26 | -0.21 | 5/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 2.0 | ORB_G5 | long | 1443 / 814 | -0.44 | 0.0890 / 0.1059 | 0.0169 | 1.02 / 0.85 | -0.18 | 5/7 | N | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 1.0 | COST_LT12 | long | 1024 / 521 | -0.49 | 0.0921 / 0.0540 | -0.0381 | 1.32 / 0.51 | -0.82 | 6/7 | N | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 1.0 | CROSS_SGP_MOMENTUM | long | 1020 / 535 | -0.48 | 0.0850 / 0.0502 | -0.0348 | 1.18 / 0.50 | -0.67 | 5/7 | Y | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 1.0 | ORB_G5 | long | 1485 / 773 | -0.48 | 0.0655 / 0.0353 | -0.0302 | 1.16 / 0.42 | -0.74 | 6/7 | N | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 1.0 | OVNRNG_100 | long | 532 / 263 | -0.51 | 0.1179 / 0.0560 | -0.0619 | 1.22 / 0.37 | -0.85 | 4/6 | N | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 1.5 | COST_LT12 | long | 1024 / 521 | -0.49 | 0.1067 / 0.1049 | -0.0018 | 1.21 / 0.78 | -0.42 | 7/7 | N | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 1.5 | CROSS_SGP_MOMENTUM | long | 1020 / 535 | -0.48 | 0.0936 / 0.0812 | -0.0124 | 1.01 / 0.64 | -0.37 | 5/7 | Y | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 1.5 | ORB_G5 | long | 1485 / 773 | -0.48 | 0.0740 / 0.0769 | 0.0029 | 1.03 / 0.72 | -0.31 | 6/7 | N | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 1.5 | OVNRNG_100 | long | 532 / 263 | -0.51 | 0.1714 / 0.1180 | -0.0534 | 1.39 / 0.62 | -0.77 | 6/6 | N | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 2.0 | CROSS_SGP_MOMENTUM | long | 1020 / 535 | -0.48 | 0.1229 / 0.1122 | -0.0107 | 1.12 / 0.75 | -0.38 | 6/7 | Y | DRIFT |
| MNQ | EUROPE_FLOW | 5 | 2.0 | ORB_G5 | long | 1485 / 773 | -0.48 | 0.0961 / 0.0735 | -0.0226 | 1.14 / 0.58 | -0.55 | 4/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.0 | COST_LT12 | long | 1508 / 844 | -0.44 | 0.0870 / 0.0694 | -0.0176 | 1.43 / 0.79 | -0.64 | 7/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.0 | ORB_G5 | long | 1521 / 856 | -0.44 | 0.0889 / 0.0664 | -0.0225 | 1.47 / 0.77 | -0.71 | 6/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.0 | X_MES_ATR60 | long | 689 / 345 | -0.50 | 0.1365 / 0.0784 | -0.0581 | 1.54 / 0.57 | -0.97 | 5/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.5 | COST_LT12 | long | 1472 / 817 | -0.44 | 0.1050 / 0.0887 | -0.0163 | 1.36 / 0.80 | -0.57 | 6/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.5 | ORB_G5 | long | 1485 / 829 | -0.44 | 0.1067 / 0.0858 | -0.0209 | 1.39 / 0.78 | -0.61 | 5/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.5 | X_MES_ATR60 | long | 673 / 334 | -0.50 | 0.1323 / 0.0656 | -0.0667 | 1.17 / 0.38 | -0.79 | 5/7 | N | DRIFT |
| MNQ | SINGAPORE_OPEN | 15 | 1.5 | ATR_P50 | long | 968 / 496 | -0.49 | 0.1087 / 0.2046 | 0.0959 | 1.14 / 1.50 | 0.36 | 6/7 | Y | DRIFT |
| MNQ | SINGAPORE_OPEN | 30 | 1.5 | ATR_P50 | long | 961 / 485 | -0.50 | 0.1251 / 0.2210 | 0.0959 | 1.28 / 1.56 | 0.29 | 6/7 | Y | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 1.0 | COST_LT12 | long | 918 / 469 | -0.49 | 0.0959 / 0.0786 | -0.0173 | 1.31 / 0.71 | -0.60 | 6/7 | N | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 1.5 | COST_LT12 | long | 918 / 469 | -0.49 | 0.1293 / 0.1036 | -0.0257 | 1.39 / 0.74 | -0.65 | 6/7 | N | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 1.5 | ORB_G5 | long | 1487 / 786 | -0.47 | 0.0945 / 0.0992 | 0.0047 | 1.33 / 0.95 | -0.39 | 6/7 | N | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 2.0 | ORB_G5 | long | 1487 / 785 | -0.47 | 0.0866 / 0.1233 | 0.0367 | 1.04 / 0.99 | -0.05 | 5/7 | N | DRIFT |
| MNQ | US_DATA_1000 | 5 | 1.0 | X_MES_ATR60 | long | 694 / 371 | -0.47 | 0.0996 / 0.0770 | -0.0226 | 1.14 / 0.59 | -0.55 | 5/7 | N | DRIFT |
| MNQ | US_DATA_1000 | 15 | 1.0 | VWAP_MID_ALIGNED | long | 744 / 460 | -0.38 | 0.1492 / 0.1323 | -0.0169 | 1.74 / 1.13 | -0.61 | 5/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 15 | 1.5 | ORB_G5 | long | 1348 / 800 | -0.41 | 0.0930 / 0.0565 | -0.0365 | 1.16 / 0.51 | -0.65 | 6/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 15 | 1.5 | VWAP_MID_ALIGNED | long | 701 / 436 | -0.38 | 0.2101 / 0.1844 | -0.0257 | 1.87 / 1.21 | -0.66 | 6/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 15 | 2.0 | VWAP_MID_ALIGNED | long | 635 / 398 | -0.37 | 0.1757 / 0.1481 | -0.0276 | 1.27 / 0.79 | -0.47 | 5/7 | Y | DRIFT |

## Materially-drifted lanes — detail

### MES CME_PRECLOSE O5 RR1.0 COST_LT08 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08`
- Stored: N=194 ExpR=0.1962 Sharpe_ann=1.25 WR=0.634 last_trade_day=None
- Mode A: N=88 ExpR=0.3279 Sharpe_ann=1.45 WR=0.705
- Drift reasons: |ΔN/N|=0.55>0.1, |ΔExpR|=0.132>0.03, |ΔSharpe|=0.20>0.2
- Mode-A per-year: 2020:+0.494(N=19) 2021:+0.119(N=5) 2022:+0.490(N=29) 2023:--0.382(N=3) 2024:--0.066(N=6) 2025:+0.239(N=26)

### MES CME_PRECLOSE O5 RR1.0 ORB_G8 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8`
- Stored: N=287 ExpR=0.1729 Sharpe_ann=1.34 WR=0.627 last_trade_day=None
- Mode A: N=130 ExpR=0.2798 Sharpe_ann=1.49 WR=0.685
- Drift reasons: |ΔN/N|=0.55>0.1, |ΔExpR|=0.107>0.03
- Mode-A per-year: 2020:+0.489(N=24) 2021:+0.341(N=11) 2022:+0.326(N=45) 2023:+0.227(N=6) 2024:--0.071(N=10) 2025:+0.163(N=34)

### MNQ CME_PRECLOSE O5 RR1.0 X_MES_ATR60 long
- `strategy_id`: `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60`
- Stored: N=596 ExpR=0.1702 Sharpe_ann=1.88 WR=0.624 last_trade_day=None
- Mode A: N=306 ExpR=0.2140 Sharpe_ann=1.58 WR=0.650
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔExpR|=0.044>0.03, |ΔSharpe|=0.30>0.2
- Mode-A per-year: 2019:--0.537(N=14) 2020:+0.424(N=56) 2021:+0.283(N=23) 2022:+0.396(N=56) 2023:+0.042(N=14) 2024:+0.092(N=73) 2025:+0.189(N=70)

### MNQ COMEX_SETTLE O5 RR1.0 COST_LT12 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12`
- Stored: N=1247 ExpR=0.1104 Sharpe_ann=1.74 WR=0.596 last_trade_day=None
- Mode A: N=669 ExpR=0.1041 Sharpe_ann=1.11 WR=0.593
- Drift reasons: |ΔN/N|=0.46>0.1, |ΔSharpe|=0.63>0.2
- Mode-A per-year: 2019:+0.169(N=17) 2020:+0.072(N=92) 2021:--0.063(N=86) 2022:+0.123(N=124) 2023:+0.173(N=107) 2024:+0.069(N=116) 2025:+0.187(N=127)

### MNQ COMEX_SETTLE O5 RR1.0 ORB_G5 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5`
- Stored: N=1473 ExpR=0.0890 Sharpe_ann=1.54 WR=0.592 last_trade_day=None
- Mode A: N=836 ExpR=0.0780 Sharpe_ann=0.95 WR=0.590
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔSharpe|=0.59>0.2
- Mode-A per-year: 2019:--0.010(N=57) 2020:+0.050(N=131) 2021:--0.088(N=129) 2022:+0.134(N=129) 2023:+0.142(N=125) 2024:+0.082(N=132) 2025:+0.185(N=133)

### MNQ COMEX_SETTLE O5 RR1.0 OVNRNG_100 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100`
- Stored: N=520 ExpR=0.1725 Sharpe_ann=1.76 WR=0.625 last_trade_day=None
- Mode A: N=283 ExpR=0.1844 Sharpe_ann=1.29 WR=0.633
- Drift reasons: |ΔN/N|=0.46>0.1, |ΔSharpe|=0.47>0.2
- Mode-A per-year: 2019:+0.338(N=4) 2020:+0.173(N=57) 2021:+0.119(N=36) 2022:+0.121(N=56) 2023:+0.157(N=8) 2024:+0.119(N=40) 2025:+0.292(N=82)

### MNQ COMEX_SETTLE O5 RR1.0 X_MES_ATR60 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`
- Stored: N=673 ExpR=0.1512 Sharpe_ann=1.78 WR=0.618 last_trade_day=None
- Mode A: N=385 ExpR=0.1950 Sharpe_ann=1.62 WR=0.644
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔExpR|=0.044>0.03
- Mode-A per-year: 2019:+0.045(N=15) 2020:+0.074(N=80) 2021:+0.257(N=33) 2022:+0.261(N=75) 2023:+0.209(N=20) 2024:+0.146(N=85) 2025:+0.310(N=77)

### MNQ COMEX_SETTLE O5 RR1.5 ORB_G5 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
- Stored: N=1459 ExpR=0.1119 Sharpe_ann=1.52 WR=0.484 last_trade_day=None
- Mode A: N=829 ExpR=0.0922 Sharpe_ann=0.88 WR=0.478
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔSharpe|=0.64>0.2
- Mode-A per-year: 2019:--0.136(N=57) 2020:+0.011(N=130) 2021:--0.019(N=127) 2022:+0.090(N=129) 2023:+0.192(N=125) 2024:+0.170(N=129) 2025:+0.209(N=132)

### MNQ COMEX_SETTLE O5 RR1.5 OVNRNG_100 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
- Stored: N=513 ExpR=0.2151 Sharpe_ann=1.70 WR=0.518 last_trade_day=None
- Mode A: N=278 ExpR=0.1868 Sharpe_ann=1.00 WR=0.507
- Drift reasons: |ΔN/N|=0.46>0.1, |ΔSharpe|=0.69>0.2
- Mode-A per-year: 2019:--0.448(N=4) 2020:+0.114(N=56) 2021:+0.138(N=34) 2022:+0.146(N=56) 2023:--0.129(N=8) 2024:+0.164(N=38) 2025:+0.357(N=82)

### MNQ COMEX_SETTLE O5 RR1.5 X_MES_ATR60 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60`
- Stored: N=664 ExpR=0.1609 Sharpe_ann=1.46 WR=0.498 last_trade_day=None
- Mode A: N=379 ExpR=0.1981 Sharpe_ann=1.26 WR=0.517
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔExpR|=0.037>0.03, |ΔSharpe|=0.21>0.2
- Mode-A per-year: 2019:--0.000(N=15) 2020:--0.019(N=79) 2021:+0.275(N=32) 2022:+0.229(N=75) 2023:+0.160(N=20) 2024:+0.285(N=82) 2025:+0.317(N=76)

### MNQ COMEX_SETTLE O5 RR2.0 ORB_G5 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5`
- Stored: N=1443 ExpR=0.0890 Sharpe_ann=1.02 WR=0.395 last_trade_day=None
- Mode A: N=814 ExpR=0.1059 Sharpe_ann=0.85 WR=0.403
- Drift reasons: |ΔN/N|=0.44>0.1
- Mode-A per-year: 2019:--0.267(N=52) 2020:+0.023(N=130) 2021:--0.022(N=126) 2022:+0.222(N=129) 2023:+0.196(N=122) 2024:+0.229(N=127) 2025:+0.141(N=128)

### MNQ EUROPE_FLOW O5 RR1.0 COST_LT12 long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12`
- Stored: N=1024 ExpR=0.0921 Sharpe_ann=1.32 WR=0.591 last_trade_day=None
- Mode A: N=521 ExpR=0.0540 Sharpe_ann=0.51 WR=0.570
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔExpR|=0.038>0.03, |ΔSharpe|=0.82>0.2
- Mode-A per-year: 2019:+0.206(N=12) 2020:+0.095(N=78) 2021:+0.082(N=61) 2022:+0.042(N=116) 2023:+0.095(N=65) 2024:--0.002(N=76) 2025:+0.021(N=113)

### MNQ EUROPE_FLOW O5 RR1.0 CROSS_SGP_MOMENTUM long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM`
- Stored: N=1020 ExpR=0.0850 Sharpe_ann=1.18 WR=0.606 last_trade_day=2026-04-14
- Mode A: N=535 ExpR=0.0502 Sharpe_ann=0.50 WR=0.598
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔExpR|=0.035>0.03, |ΔSharpe|=0.67>0.2, last_trade_day=2026-04-14 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.062(N=52) 2020:+0.074(N=74) 2021:+0.039(N=82) 2022:+0.066(N=81) 2023:+0.186(N=69) 2024:--0.048(N=89) 2025:+0.086(N=88)

### MNQ EUROPE_FLOW O5 RR1.0 ORB_G5 long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5`
- Stored: N=1485 ExpR=0.0655 Sharpe_ann=1.16 WR=0.592 last_trade_day=None
- Mode A: N=773 ExpR=0.0353 Sharpe_ann=0.42 WR=0.577
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔExpR|=0.030>0.03, |ΔSharpe|=0.74>0.2
- Mode-A per-year: 2019:--0.105(N=51) 2020:+0.048(N=120) 2021:+0.067(N=111) 2022:+0.051(N=128) 2023:+0.079(N=113) 2024:+0.025(N=122) 2025:+0.008(N=128)

### MNQ EUROPE_FLOW O5 RR1.0 OVNRNG_100 long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100`
- Stored: N=532 ExpR=0.1179 Sharpe_ann=1.22 WR=0.603 last_trade_day=None
- Mode A: N=263 ExpR=0.0560 Sharpe_ann=0.37 WR=0.570
- Drift reasons: |ΔN/N|=0.51>0.1, |ΔExpR|=0.062>0.03, |ΔSharpe|=0.85>0.2
- Mode-A per-year: 2019:+0.206(N=3) 2020:--0.017(N=62) 2021:+0.035(N=30) 2022:+0.097(N=63) 2023:+0.256(N=10) 2024:--0.026(N=27) 2025:+0.090(N=68)

### MNQ EUROPE_FLOW O5 RR1.5 COST_LT12 long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12`
- Stored: N=1024 ExpR=0.1067 Sharpe_ann=1.21 WR=0.478 last_trade_day=None
- Mode A: N=521 ExpR=0.1049 Sharpe_ann=0.78 WR=0.478
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔSharpe|=0.42>0.2
- Mode-A per-year: 2019:+0.133(N=12) 2020:+0.161(N=78) 2021:+0.094(N=61) 2022:+0.102(N=116) 2023:+0.299(N=65) 2024:+0.037(N=76) 2025:+0.007(N=113)

### MNQ EUROPE_FLOW O5 RR1.5 CROSS_SGP_MOMENTUM long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM`
- Stored: N=1020 ExpR=0.0936 Sharpe_ann=1.01 WR=0.488 last_trade_day=2026-04-14
- Mode A: N=535 ExpR=0.0812 Sharpe_ann=0.64 WR=0.493
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔSharpe|=0.37>0.2, last_trade_day=2026-04-14 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.129(N=52) 2020:+0.050(N=74) 2021:--0.021(N=82) 2022:+0.076(N=81) 2023:+0.359(N=69) 2024:--0.125(N=89) 2025:+0.171(N=88)

### MNQ EUROPE_FLOW O5 RR1.5 ORB_G5 long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`
- Stored: N=1485 ExpR=0.0740 Sharpe_ann=1.03 WR=0.477 last_trade_day=None
- Mode A: N=773 ExpR=0.0769 Sharpe_ann=0.72 WR=0.480
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔSharpe|=0.31>0.2
- Mode-A per-year: 2019:--0.010(N=51) 2020:+0.070(N=120) 2021:+0.096(N=111) 2022:+0.080(N=128) 2023:+0.216(N=113) 2024:+0.029(N=122) 2025:+0.022(N=128)

### MNQ EUROPE_FLOW O5 RR1.5 OVNRNG_100 long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100`
- Stored: N=532 ExpR=0.1714 Sharpe_ann=1.39 WR=0.506 last_trade_day=None
- Mode A: N=263 ExpR=0.1180 Sharpe_ann=0.62 WR=0.483
- Drift reasons: |ΔN/N|=0.51>0.1, |ΔExpR|=0.053>0.03, |ΔSharpe|=0.77>0.2
- Mode-A per-year: 2019:+0.508(N=3) 2020:+0.010(N=62) 2021:+0.218(N=30) 2022:+0.078(N=63) 2023:+0.570(N=10) 2024:+0.132(N=27) 2025:+0.120(N=68)

### MNQ EUROPE_FLOW O5 RR2.0 CROSS_SGP_MOMENTUM long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM`
- Stored: N=1020 ExpR=0.1229 Sharpe_ann=1.12 WR=0.417 last_trade_day=2026-04-14
- Mode A: N=535 ExpR=0.1122 Sharpe_ann=0.75 WR=0.422
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔSharpe|=0.38>0.2, last_trade_day=2026-04-14 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.125(N=52) 2020:+0.130(N=74) 2021:+0.053(N=82) 2022:+0.051(N=81) 2023:+0.439(N=69) 2024:--0.132(N=89) 2025:+0.191(N=88)

### MNQ EUROPE_FLOW O5 RR2.0 ORB_G5 long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5`
- Stored: N=1485 ExpR=0.0961 Sharpe_ann=1.14 WR=0.405 last_trade_day=None
- Mode A: N=773 ExpR=0.0735 Sharpe_ann=0.58 WR=0.398
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔSharpe|=0.55>0.2
- Mode-A per-year: 2019:--0.007(N=51) 2020:+0.105(N=120) 2021:+0.129(N=111) 2022:--0.028(N=128) 2023:+0.269(N=113) 2024:--0.009(N=122) 2025:+0.035(N=128)

### MNQ NYSE_OPEN O5 RR1.0 COST_LT12 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
- Stored: N=1508 ExpR=0.0870 Sharpe_ann=1.43 WR=0.561 last_trade_day=None
- Mode A: N=844 ExpR=0.0694 Sharpe_ann=0.79 WR=0.555
- Drift reasons: |ΔN/N|=0.44>0.1, |ΔSharpe|=0.64>0.2
- Mode-A per-year: 2019:+0.007(N=83) 2020:+0.070(N=124) 2021:+0.037(N=132) 2022:+0.121(N=115) 2023:+0.003(N=142) 2024:+0.106(N=126) 2025:+0.137(N=122)

### MNQ NYSE_OPEN O5 RR1.0 ORB_G5 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5`
- Stored: N=1521 ExpR=0.0889 Sharpe_ann=1.47 WR=0.563 last_trade_day=None
- Mode A: N=856 ExpR=0.0664 Sharpe_ann=0.77 WR=0.554
- Drift reasons: |ΔN/N|=0.44>0.1, |ΔSharpe|=0.71>0.2
- Mode-A per-year: 2019:--0.012(N=88) 2020:+0.057(N=127) 2021:+0.042(N=133) 2022:+0.126(N=116) 2023:+0.002(N=144) 2024:+0.106(N=126) 2025:+0.137(N=122)

### MNQ NYSE_OPEN O5 RR1.0 X_MES_ATR60 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60`
- Stored: N=689 ExpR=0.1365 Sharpe_ann=1.54 WR=0.586 last_trade_day=None
- Mode A: N=345 ExpR=0.0784 Sharpe_ann=0.57 WR=0.557
- Drift reasons: |ΔN/N|=0.50>0.1, |ΔExpR|=0.058>0.03, |ΔSharpe|=0.97>0.2
- Mode-A per-year: 2019:--0.180(N=18) 2020:+0.052(N=73) 2021:+0.177(N=28) 2022:+0.114(N=65) 2023:--0.091(N=17) 2024:+0.085(N=75) 2025:+0.135(N=69)

### MNQ NYSE_OPEN O5 RR1.5 COST_LT12 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`
- Stored: N=1472 ExpR=0.1050 Sharpe_ann=1.36 WR=0.457 last_trade_day=None
- Mode A: N=817 ExpR=0.0887 Sharpe_ann=0.80 WR=0.452
- Drift reasons: |ΔN/N|=0.44>0.1, |ΔSharpe|=0.57>0.2
- Mode-A per-year: 2019:--0.053(N=81) 2020:+0.006(N=117) 2021:+0.138(N=127) 2022:+0.123(N=113) 2023:+0.110(N=141) 2024:+0.201(N=125) 2025:+0.036(N=113)

### MNQ NYSE_OPEN O5 RR1.5 ORB_G5 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5`
- Stored: N=1485 ExpR=0.1067 Sharpe_ann=1.39 WR=0.458 last_trade_day=None
- Mode A: N=829 ExpR=0.0858 Sharpe_ann=0.78 WR=0.451
- Drift reasons: |ΔN/N|=0.44>0.1, |ΔSharpe|=0.61>0.2
- Mode-A per-year: 2019:--0.059(N=86) 2020:--0.002(N=120) 2021:+0.145(N=128) 2022:+0.133(N=114) 2023:+0.094(N=143) 2024:+0.201(N=125) 2025:+0.036(N=113)

### MNQ NYSE_OPEN O5 RR1.5 X_MES_ATR60 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60`
- Stored: N=673 ExpR=0.1323 Sharpe_ann=1.17 WR=0.468 last_trade_day=None
- Mode A: N=334 ExpR=0.0656 Sharpe_ann=0.38 WR=0.440
- Drift reasons: |ΔN/N|=0.50>0.1, |ΔExpR|=0.067>0.03, |ΔSharpe|=0.79>0.2
- Mode-A per-year: 2019:--0.229(N=18) 2020:+0.041(N=69) 2021:+0.212(N=28) 2022:+0.147(N=64) 2023:--0.008(N=17) 2024:+0.080(N=74) 2025:+0.032(N=64)

### MNQ SINGAPORE_OPEN O15 RR1.5 ATR_P50 long
- `strategy_id`: `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
- Stored: N=968 ExpR=0.1087 Sharpe_ann=1.14 WR=0.481 last_trade_day=2026-04-06
- Mode A: N=496 ExpR=0.2046 Sharpe_ann=1.50 WR=0.524
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔExpR|=0.096>0.03, |ΔSharpe|=0.36>0.2, last_trade_day=2026-04-06 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.384(N=32) 2020:+0.065(N=113) 2021:+0.040(N=51) 2022:+0.327(N=75) 2023:--0.031(N=26) 2024:+0.358(N=111) 2025:+0.187(N=88)

### MNQ SINGAPORE_OPEN O30 RR1.5 ATR_P50 long
- `strategy_id`: `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30`
- Stored: N=961 ExpR=0.1251 Sharpe_ann=1.28 WR=0.478 last_trade_day=2026-04-06
- Mode A: N=485 ExpR=0.2210 Sharpe_ann=1.56 WR=0.520
- Drift reasons: |ΔN/N|=0.50>0.1, |ΔExpR|=0.096>0.03, |ΔSharpe|=0.29>0.2, last_trade_day=2026-04-06 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.337(N=29) 2020:+0.086(N=109) 2021:+0.080(N=57) 2022:+0.349(N=79) 2023:--0.209(N=25) 2024:+0.265(N=116) 2025:+0.434(N=70)

### MNQ TOKYO_OPEN O5 RR1.0 COST_LT12 long
- `strategy_id`: `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12`
- Stored: N=918 ExpR=0.0959 Sharpe_ann=1.31 WR=0.595 last_trade_day=None
- Mode A: N=469 ExpR=0.0786 Sharpe_ann=0.71 WR=0.586
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔSharpe|=0.60>0.2
- Mode-A per-year: 2019:+0.206(N=21) 2020:+0.075(N=79) 2021:+0.139(N=72) 2022:+0.107(N=95) 2023:+0.049(N=31) 2024:+0.158(N=54) 2025:--0.031(N=117)

### MNQ TOKYO_OPEN O5 RR1.5 COST_LT12 long
- `strategy_id`: `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`
- Stored: N=918 ExpR=0.1293 Sharpe_ann=1.39 WR=0.490 last_trade_day=None
- Mode A: N=469 ExpR=0.1036 Sharpe_ann=0.74 WR=0.480
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔSharpe|=0.65>0.2
- Mode-A per-year: 2019:+0.074(N=21) 2020:+0.050(N=79) 2021:+0.169(N=72) 2022:+0.141(N=95) 2023:--0.053(N=31) 2024:+0.279(N=54) 2025:+0.035(N=117)

### MNQ TOKYO_OPEN O5 RR1.5 ORB_G5 long
- `strategy_id`: `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5`
- Stored: N=1487 ExpR=0.0945 Sharpe_ann=1.33 WR=0.492 last_trade_day=None
- Mode A: N=786 ExpR=0.0992 Sharpe_ann=0.95 WR=0.496
- Drift reasons: |ΔN/N|=0.47>0.1, |ΔSharpe|=0.39>0.2
- Mode-A per-year: 2019:+0.158(N=64) 2020:+0.090(N=115) 2021:+0.228(N=134) 2022:+0.118(N=125) 2023:--0.094(N=113) 2024:+0.173(N=102) 2025:+0.038(N=133)

### MNQ TOKYO_OPEN O5 RR2.0 ORB_G5 long
- `strategy_id`: `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5`
- Stored: N=1487 ExpR=0.0866 Sharpe_ann=1.04 WR=0.407 last_trade_day=None
- Mode A: N=785 ExpR=0.1233 Sharpe_ann=0.99 WR=0.423
- Drift reasons: |ΔN/N|=0.47>0.1, |ΔExpR|=0.037>0.03
- Mode-A per-year: 2019:+0.164(N=63) 2020:+0.146(N=115) 2021:+0.236(N=134) 2022:+0.187(N=125) 2023:--0.047(N=113) 2024:+0.223(N=102) 2025:--0.021(N=133)

### MNQ US_DATA_1000 O5 RR1.0 X_MES_ATR60 long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60`
- Stored: N=694 ExpR=0.0996 Sharpe_ann=1.14 WR=0.572 last_trade_day=None
- Mode A: N=371 ExpR=0.0770 Sharpe_ann=0.59 WR=0.563
- Drift reasons: |ΔN/N|=0.47>0.1, |ΔSharpe|=0.55>0.2
- Mode-A per-year: 2019:+0.235(N=13) 2020:--0.059(N=86) 2021:--0.020(N=33) 2022:+0.069(N=69) 2023:+0.267(N=15) 2024:+0.289(N=76) 2025:+0.006(N=79)

### MNQ US_DATA_1000 O15 RR1.0 VWAP_MID_ALIGNED long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`
- Stored: N=744 ExpR=0.1492 Sharpe_ann=1.74 WR=0.593 last_trade_day=2026-04-02
- Mode A: N=460 ExpR=0.1323 Sharpe_ann=1.13 WR=0.589
- Drift reasons: |ΔN/N|=0.38>0.1, |ΔSharpe|=0.61>0.2, last_trade_day=2026-04-02 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.380(N=41) 2020:+0.126(N=73) 2021:+0.202(N=72) 2022:--0.012(N=67) 2023:+0.238(N=69) 2024:+0.169(N=68) 2025:--0.079(N=70)

### MNQ US_DATA_1000 O15 RR1.5 ORB_G5 long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Stored: N=1348 ExpR=0.0930 Sharpe_ann=1.16 WR=0.451 last_trade_day=2026-04-06
- Mode A: N=800 ExpR=0.0565 Sharpe_ann=0.51 WR=0.439
- Drift reasons: |ΔN/N|=0.41>0.1, |ΔExpR|=0.036>0.03, |ΔSharpe|=0.65>0.2, last_trade_day=2026-04-06 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.272(N=74) 2020:+0.058(N=134) 2021:+0.077(N=114) 2022:+0.041(N=122) 2023:+0.127(N=116) 2024:+0.053(N=120) 2025:--0.146(N=120)

### MNQ US_DATA_1000 O15 RR1.5 VWAP_MID_ALIGNED long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
- Stored: N=701 ExpR=0.2101 Sharpe_ann=1.87 WR=0.499 last_trade_day=2026-04-02
- Mode A: N=436 ExpR=0.1844 Sharpe_ann=1.21 WR=0.493
- Drift reasons: |ΔN/N|=0.38>0.1, |ΔSharpe|=0.66>0.2, last_trade_day=2026-04-02 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.433(N=38) 2020:+0.178(N=71) 2021:+0.306(N=68) 2022:+0.089(N=65) 2023:+0.269(N=65) 2024:+0.232(N=65) 2025:--0.124(N=64)

### MNQ US_DATA_1000 O15 RR2.0 VWAP_MID_ALIGNED long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15`
- Stored: N=635 ExpR=0.1757 Sharpe_ann=1.27 WR=0.405 last_trade_day=2026-04-02
- Mode A: N=398 ExpR=0.1481 Sharpe_ann=0.79 WR=0.399
- Drift reasons: |ΔN/N|=0.37>0.1, |ΔSharpe|=0.47>0.2, last_trade_day=2026-04-02 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.431(N=38) 2020:+0.180(N=68) 2021:+0.315(N=59) 2022:--0.043(N=58) 2023:+0.224(N=57) 2024:+0.259(N=60) 2025:--0.244(N=58)


## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_revalidation_active_setups.py
```

No writes to validated_setups or experimental_strategies. Output is this
markdown document only. Numbers reproduce exactly on the same DB state.


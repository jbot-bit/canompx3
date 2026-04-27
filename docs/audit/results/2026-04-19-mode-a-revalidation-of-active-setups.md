# Mode A canonical re-validation of active validated_setups

**Generated:** 2026-04-27T22:08:46+00:00
**Script:** `research/mode_a_revalidation_active_setups.py`
**IS boundary:** `trading_day < 2026-01-01` (Mode A)
**Canonical filter source:** `research.filter_utils.filter_signal` → `trading_app.config.ALL_FILTERS`

## Summary

- Total active lanes re-validated: **59**
- Lanes with material Mode A drift (|ΔN/N|>10% OR |ΔExpR|>0.03 OR |ΔSharpe|>0.20 OR Mode-B contaminated): **59**
- Lanes with last_trade_day >= 2026-01-01 (Mode-B grandfathered): **32**

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
| MES | CME_PRECLOSE | 5 | 1.0 | ATR_P50 | long | 703 / 420 | -0.40 | 0.0716 / 0.0686 | -0.0030 | 0.90 / 0.66 | -0.25 | 6/7 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | COST_LT08 | long | 214 / 118 | -0.45 | 0.1605 / 0.2425 | 0.0820 | 1.20 / 1.38 | 0.18 | 3/4 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | COST_LT08 | long | 194 / 118 | -0.39 | 0.1962 / 0.2425 | 0.0463 | 1.25 / 1.38 | 0.13 | 3/4 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | COST_LT10 | long | 380 / 212 | -0.44 | 0.1569 / 0.1866 | 0.0297 | 1.59 / 1.31 | -0.28 | 6/6 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | COST_LT15 | long | 778 / 446 | -0.43 | 0.0971 / 0.1025 | 0.0054 | 1.43 / 1.02 | -0.40 | 5/7 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | COST_LT15 | long | 725 / 446 | -0.38 | 0.1114 / 0.1025 | -0.0089 | 1.40 / 1.02 | -0.37 | 5/7 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | ORB_G4 | long | 899 / 509 | -0.43 | 0.0778 / 0.1009 | 0.0231 | 1.24 / 1.08 | -0.16 | 5/7 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | ORB_G6 | long | 502 / 286 | -0.43 | 0.1297 / 0.1172 | -0.0125 | 1.51 / 0.94 | -0.57 | 5/6 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | ORB_G8 | long | 314 / 172 | -0.45 | 0.1530 / 0.1989 | 0.0459 | 1.40 / 1.35 | -0.05 | 5/6 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | ORB_G8 | long | 287 / 172 | -0.40 | 0.1729 / 0.1989 | 0.0260 | 1.34 / 1.35 | 0.00 | 5/6 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | ORB_VOL_16K | long | 152 / 75 | -0.51 | 0.2010 / 0.2562 | 0.0552 | 1.29 / 1.16 | -0.13 | 3/3 | Y | DRIFT |
| MES | CME_PRECLOSE | 5 | 1.0 | ORB_VOL_16K | long | 138 / 75 | -0.46 | 0.2491 / 0.2562 | 0.0071 | 1.38 / 1.16 | -0.22 | 3/3 | Y | DRIFT |
| MES | COMEX_SETTLE | 5 | 1.0 | COST_LT10 | long | 269 / 142 | -0.47 | 0.1120 / 0.1256 | 0.0136 | 0.95 / 0.62 | -0.32 | 3/4 | Y | DRIFT |
| MES | COMEX_SETTLE | 5 | 1.0 | COST_LT10 | long | 266 / 142 | -0.47 | 0.1258 / 0.1256 | -0.0002 | 0.93 / 0.62 | -0.31 | 3/4 | Y | DRIFT |
| MES | SINGAPORE_OPEN | 5 | 1.5 | COST_LT10 | long | 89 / 48 | -0.46 | 0.2737 / 0.3500 | 0.0763 | 1.02 / 0.86 | -0.16 | 2/2 | Y | DRIFT |
| MES | US_DATA_830 | 5 | 1.0 | OVNRNG_50 | long | 105 / 59 | -0.44 | 0.1801 / 0.0120 | -0.1681 | 0.99 / 0.04 | -0.95 | 1/3 | Y | DRIFT |
| MGC | CME_REOPEN | 5 | 1.0 | ORB_G4 | long | 86 / 91 | 0.06 | 0.2839 / 0.1742 | -0.1097 | 1.74 / 1.30 | -0.44 | 1/2 | Y | DRIFT |
| MNQ | CME_PRECLOSE | 5 | 1.0 | X_MES_ATR60 | long | 596 / 340 | -0.43 | 0.1702 / 0.1910 | 0.0208 | 1.88 / 1.55 | -0.33 | 6/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | COST_LT12 | long | 1247 / 672 | -0.46 | 0.1104 / 0.1034 | -0.0070 | 1.74 / 1.11 | -0.63 | 6/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | ORB_G5 | long | 1473 / 839 | -0.43 | 0.0890 / 0.0775 | -0.0115 | 1.54 / 0.94 | -0.60 | 5/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | OVNRNG_100 | long | 520 / 283 | -0.46 | 0.1725 / 0.1844 | 0.0119 | 1.76 / 1.29 | -0.47 | 5/5 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | PD_CLEAR_LONG | long | 303 / 338 | 0.12 | 0.1841 / 0.1602 | -0.0239 | 1.49 / 1.27 | -0.22 | 5/7 | Y | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.0 | X_MES_ATR60 | long | 673 / 386 | -0.43 | 0.1512 / 0.1945 | 0.0433 | 1.78 / 1.62 | -0.16 | 7/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.5 | ORB_G5 | long | 1459 / 839 | -0.42 | 0.1119 / 0.0950 | -0.0169 | 1.52 / 0.91 | -0.61 | 5/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.5 | OVNRNG_100 | long | 513 / 283 | -0.45 | 0.2151 / 0.1894 | -0.0257 | 1.70 / 1.03 | -0.66 | 5/5 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 1.5 | X_MES_ATR60 | long | 664 / 386 | -0.42 | 0.1609 / 0.2039 | 0.0430 | 1.46 / 1.31 | -0.15 | 5/7 | N | DRIFT |
| MNQ | COMEX_SETTLE | 5 | 2.0 | ORB_G5 | long | 1443 / 839 | -0.42 | 0.0890 / 0.1181 | 0.0291 | 1.02 / 0.97 | -0.05 | 5/7 | N | DRIFT |
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
| MNQ | NYSE_OPEN | 5 | 1.0 | COST_LT12 | long | 1508 / 862 | -0.43 | 0.0870 / 0.0670 | -0.0200 | 1.43 / 0.78 | -0.65 | 7/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.0 | ORB_G5 | long | 1521 / 874 | -0.43 | 0.0889 / 0.0640 | -0.0249 | 1.47 / 0.75 | -0.72 | 6/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.0 | X_MES_ATR60 | long | 689 / 353 | -0.49 | 0.1365 / 0.0749 | -0.0616 | 1.54 / 0.56 | -0.99 | 5/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.5 | COST_LT12 | long | 1472 / 862 | -0.41 | 0.1050 / 0.0959 | -0.0091 | 1.36 / 0.90 | -0.46 | 6/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.5 | ORB_G5 | long | 1485 / 874 | -0.41 | 0.1067 / 0.0930 | -0.0137 | 1.39 / 0.88 | -0.51 | 6/7 | N | DRIFT |
| MNQ | NYSE_OPEN | 5 | 1.5 | X_MES_ATR60 | long | 673 / 353 | -0.48 | 0.1323 / 0.0714 | -0.0609 | 1.17 / 0.43 | -0.74 | 5/7 | N | DRIFT |
| MNQ | SINGAPORE_OPEN | 15 | 1.5 | ATR_P50 | long | 968 / 497 | -0.49 | 0.1087 / 0.2060 | 0.0973 | 1.14 / 1.51 | 0.37 | 6/7 | Y | DRIFT |
| MNQ | SINGAPORE_OPEN | 30 | 1.5 | ATR_P50 | long | 961 / 486 | -0.49 | 0.1251 / 0.2217 | 0.0966 | 1.28 / 1.57 | 0.29 | 6/7 | Y | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 1.0 | COST_LT12 | long | 918 / 469 | -0.49 | 0.0959 / 0.0786 | -0.0173 | 1.31 / 0.71 | -0.60 | 6/7 | N | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 1.5 | COST_LT08 | long | 427 / 205 | -0.52 | 0.2037 / 0.1620 | -0.0417 | 1.47 / 0.74 | -0.73 | 5/5 | Y | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 1.5 | COST_LT12 | long | 918 / 469 | -0.49 | 0.1293 / 0.1036 | -0.0257 | 1.39 / 0.74 | -0.65 | 6/7 | N | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 1.5 | ORB_G5 | long | 1487 / 786 | -0.47 | 0.0945 / 0.0992 | 0.0047 | 1.33 / 0.95 | -0.39 | 6/7 | N | DRIFT |
| MNQ | TOKYO_OPEN | 5 | 2.0 | ORB_G5 | long | 1487 / 786 | -0.47 | 0.0866 / 0.1241 | 0.0375 | 1.04 / 1.00 | -0.04 | 5/7 | N | DRIFT |
| MNQ | US_DATA_1000 | 5 | 1.0 | PD_CLEAR_LONG | long | 211 / 237 | 0.12 | 0.2270 / 0.2151 | -0.0119 | 1.47 / 1.37 | -0.09 | 6/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 5 | 1.0 | PD_DISPLACE_LONG | long | 192 / 208 | 0.08 | 0.2396 / 0.2308 | -0.0088 | 1.48 / 1.38 | -0.10 | 6/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 5 | 1.0 | PD_GO_LONG | long | 324 / 356 | 0.10 | 0.1934 / 0.1827 | -0.0107 | 1.53 / 1.42 | -0.12 | 7/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 5 | 1.0 | X_MES_ATR60 | long | 694 / 374 | -0.46 | 0.0996 / 0.0779 | -0.0217 | 1.14 / 0.60 | -0.53 | 5/7 | N | DRIFT |
| MNQ | US_DATA_1000 | 5 | 1.5 | PD_GO_LONG | long | 321 / 356 | 0.11 | 0.2222 / 0.2119 | -0.0103 | 1.36 / 1.28 | -0.08 | 6/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 15 | 1.0 | VWAP_MID_ALIGNED | long | 744 / 497 | -0.33 | 0.1492 / 0.1275 | -0.0217 | 1.74 / 1.17 | -0.57 | 5/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 15 | 1.5 | ORB_G5 | long | 1348 / 931 | -0.31 | 0.0930 / 0.0911 | -0.0019 | 1.16 / 0.93 | -0.22 | 6/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 15 | 1.5 | VWAP_MID_ALIGNED | long | 701 / 497 | -0.29 | 0.2101 / 0.1919 | -0.0182 | 1.87 / 1.42 | -0.45 | 6/7 | Y | DRIFT |
| MNQ | US_DATA_1000 | 15 | 2.0 | VWAP_MID_ALIGNED | long | 635 / 497 | -0.22 | 0.1757 / 0.2316 | 0.0559 | 1.27 / 1.50 | 0.23 | 6/7 | Y | DRIFT |

## Materially-drifted lanes — detail

### MES CME_PRECLOSE O5 RR1.0 ATR_P50 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50`
- Stored: N=703 ExpR=0.0716 Sharpe_ann=0.90 WR=0.609 last_trade_day=2026-04-23
- Mode A: N=420 ExpR=0.0686 Sharpe_ann=0.66 WR=0.555
- Drift reasons: |ΔN/N|=0.40>0.1, |ΔSharpe|=0.25>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.363(N=21) 2020:+0.218(N=76) 2021:+0.124(N=35) 2022:+0.133(N=81) 2023:+0.049(N=23) 2024:+0.016(N=106) 2025:+0.025(N=78)

### MES CME_PRECLOSE O5 RR1.0 COST_LT08 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_S075`
- Stored: N=214 ExpR=0.1605 Sharpe_ann=1.20 WR=0.556 last_trade_day=2026-04-21
- Mode A: N=118 ExpR=0.2425 Sharpe_ann=1.38 WR=0.525
- Drift reasons: |ΔN/N|=0.45>0.1, |ΔExpR|=0.082>0.03, last_trade_day=2026-04-21 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2020:+0.407(N=23) 2021:+0.021(N=8) 2022:+0.369(N=40) 2023:--0.276(N=6) 2024:--0.012(N=11) 2025:+0.204(N=30)

### MES CME_PRECLOSE O5 RR1.0 COST_LT08 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08`
- Stored: N=194 ExpR=0.1962 Sharpe_ann=1.25 WR=0.634 last_trade_day=2026-04-21
- Mode A: N=118 ExpR=0.2425 Sharpe_ann=1.38 WR=0.525
- Drift reasons: |ΔN/N|=0.39>0.1, |ΔExpR|=0.046>0.03, last_trade_day=2026-04-21 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2020:+0.407(N=23) 2021:+0.021(N=8) 2022:+0.369(N=40) 2023:--0.276(N=6) 2024:--0.012(N=11) 2025:+0.204(N=30)

### MES CME_PRECLOSE O5 RR1.0 COST_LT10 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075`
- Stored: N=380 ExpR=0.1569 Sharpe_ann=1.59 WR=0.563 last_trade_day=2026-04-21
- Mode A: N=212 ExpR=0.1866 Sharpe_ann=1.31 WR=0.505
- Drift reasons: |ΔN/N|=0.44>0.1, |ΔSharpe|=0.28>0.2, last_trade_day=2026-04-21 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.113(N=1) 2020:+0.442(N=38) 2021:+0.167(N=19) 2022:+0.189(N=74) 2023:+0.174(N=15) 2024:+0.019(N=21) 2025:+0.057(N=44)

### MES CME_PRECLOSE O5 RR1.0 COST_LT15 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_S075`
- Stored: N=778 ExpR=0.0971 Sharpe_ann=1.43 WR=0.544 last_trade_day=2026-04-23
- Mode A: N=446 ExpR=0.1025 Sharpe_ann=1.02 WR=0.507
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔSharpe|=0.40>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.182(N=13) 2020:+0.232(N=71) 2021:+0.106(N=44) 2022:+0.189(N=106) 2023:+0.035(N=56) 2024:+0.075(N=70) 2025:--0.002(N=86)

### MES CME_PRECLOSE O5 RR1.0 COST_LT15 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15`
- Stored: N=725 ExpR=0.1114 Sharpe_ann=1.40 WR=0.615 last_trade_day=2026-04-23
- Mode A: N=446 ExpR=0.1025 Sharpe_ann=1.02 WR=0.507
- Drift reasons: |ΔN/N|=0.38>0.1, |ΔSharpe|=0.37>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.182(N=13) 2020:+0.232(N=71) 2021:+0.106(N=44) 2022:+0.189(N=106) 2023:+0.035(N=56) 2024:+0.075(N=70) 2025:--0.002(N=86)

### MES CME_PRECLOSE O5 RR1.0 ORB_G4 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_S075`
- Stored: N=899 ExpR=0.0778 Sharpe_ann=1.24 WR=0.536 last_trade_day=2026-04-23
- Mode A: N=509 ExpR=0.1009 Sharpe_ann=1.08 WR=0.517
- Drift reasons: |ΔN/N|=0.43>0.1, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.262(N=16) 2020:+0.180(N=80) 2021:+0.154(N=51) 2022:+0.194(N=107) 2023:+0.084(N=72) 2024:+0.098(N=87) 2025:--0.021(N=96)

### MES CME_PRECLOSE O5 RR1.0 ORB_G6 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_S075`
- Stored: N=502 ExpR=0.1297 Sharpe_ann=1.51 WR=0.552 last_trade_day=2026-04-23
- Mode A: N=286 ExpR=0.1172 Sharpe_ann=0.94 WR=0.476
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔSharpe|=0.57>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.291(N=7) 2020:+0.327(N=51) 2021:+0.117(N=26) 2022:+0.183(N=84) 2023:+0.024(N=24) 2024:+0.026(N=35) 2025:--0.017(N=59)

### MES CME_PRECLOSE O5 RR1.0 ORB_G8 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_S075`
- Stored: N=314 ExpR=0.1530 Sharpe_ann=1.40 WR=0.557 last_trade_day=2026-04-21
- Mode A: N=172 ExpR=0.1989 Sharpe_ann=1.35 WR=0.517
- Drift reasons: |ΔN/N|=0.45>0.1, |ΔExpR|=0.046>0.03, last_trade_day=2026-04-21 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2020:+0.358(N=31) 2021:+0.115(N=17) 2022:+0.257(N=60) 2023:+0.032(N=10) 2024:--0.004(N=16) 2025:+0.143(N=38)

### MES CME_PRECLOSE O5 RR1.0 ORB_G8 long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8`
- Stored: N=287 ExpR=0.1729 Sharpe_ann=1.34 WR=0.627 last_trade_day=2026-04-21
- Mode A: N=172 ExpR=0.1989 Sharpe_ann=1.35 WR=0.517
- Drift reasons: |ΔN/N|=0.40>0.1, last_trade_day=2026-04-21 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2020:+0.358(N=31) 2021:+0.115(N=17) 2022:+0.257(N=60) 2023:+0.032(N=10) 2024:--0.004(N=16) 2025:+0.143(N=38)

### MES CME_PRECLOSE O5 RR1.0 ORB_VOL_16K long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_S075`
- Stored: N=152 ExpR=0.2010 Sharpe_ann=1.29 WR=0.586 last_trade_day=2026-04-21
- Mode A: N=75 ExpR=0.2562 Sharpe_ann=1.16 WR=0.547
- Drift reasons: |ΔN/N|=0.51>0.1, |ΔExpR|=0.055>0.03, last_trade_day=2026-04-21 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2020:+0.333(N=8) 2021:+0.101(N=5) 2022:+0.427(N=24) 2023:--0.128(N=6) 2024:+0.071(N=11) 2025:+0.276(N=21)

### MES CME_PRECLOSE O5 RR1.0 ORB_VOL_16K long
- `strategy_id`: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K`
- Stored: N=138 ExpR=0.2491 Sharpe_ann=1.38 WR=0.667 last_trade_day=2026-04-21
- Mode A: N=75 ExpR=0.2562 Sharpe_ann=1.16 WR=0.547
- Drift reasons: |ΔN/N|=0.46>0.1, |ΔSharpe|=0.22>0.2, last_trade_day=2026-04-21 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2020:+0.333(N=8) 2021:+0.101(N=5) 2022:+0.427(N=24) 2023:--0.128(N=6) 2024:+0.071(N=11) 2025:+0.276(N=21)

### MES COMEX_SETTLE O5 RR1.0 COST_LT10 long
- `strategy_id`: `MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075`
- Stored: N=269 ExpR=0.1120 Sharpe_ann=0.95 WR=0.535 last_trade_day=2026-04-23
- Mode A: N=142 ExpR=0.1256 Sharpe_ann=0.62 WR=0.599
- Drift reasons: |ΔN/N|=0.47>0.1, |ΔSharpe|=0.32>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.080(N=2) 2020:+0.274(N=19) 2021:+0.236(N=6) 2022:+0.153(N=45) 2023:+0.867(N=6) 2024:--0.264(N=17) 2025:+0.080(N=47)

### MES COMEX_SETTLE O5 RR1.0 COST_LT10 long
- `strategy_id`: `MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10`
- Stored: N=266 ExpR=0.1258 Sharpe_ann=0.93 WR=0.605 last_trade_day=2026-04-23
- Mode A: N=142 ExpR=0.1256 Sharpe_ann=0.62 WR=0.599
- Drift reasons: |ΔN/N|=0.47>0.1, |ΔSharpe|=0.31>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.080(N=2) 2020:+0.274(N=19) 2021:+0.236(N=6) 2022:+0.153(N=45) 2023:+0.867(N=6) 2024:--0.264(N=17) 2025:+0.080(N=47)

### MES SINGAPORE_OPEN O5 RR1.5 COST_LT10 long
- `strategy_id`: `MES_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT10_S075`
- Stored: N=89 ExpR=0.2737 Sharpe_ann=1.02 WR=0.494 last_trade_day=2026-04-02
- Mode A: N=48 ExpR=0.3500 Sharpe_ann=0.86 WR=0.583
- Drift reasons: |ΔN/N|=0.46>0.1, |ΔExpR|=0.076>0.03, last_trade_day=2026-04-02 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+1.363(N=1) 2020:+0.222(N=17) 2021:--0.083(N=5) 2022:--0.228(N=6) 2024:+0.536(N=3) 2025:+0.740(N=16)

### MES US_DATA_830 O5 RR1.0 OVNRNG_50 long
- `strategy_id`: `MES_US_DATA_830_E2_RR1.0_CB1_OVNRNG_50_S075`
- Stored: N=105 ExpR=0.1801 Sharpe_ann=0.99 WR=0.600 last_trade_day=2026-04-23
- Mode A: N=59 ExpR=0.0120 Sharpe_ann=0.04 WR=0.559
- Drift reasons: |ΔN/N|=0.44>0.1, |ΔExpR|=0.168>0.03, |ΔSharpe|=0.95>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--1.000(N=1) 2020:+0.113(N=18) 2021:+0.146(N=6) 2022:--0.026(N=15) 2023:+0.676(N=2) 2024:--0.027(N=4) 2025:--0.158(N=13)

### MGC CME_REOPEN O5 RR1.0 ORB_G4 long
- `strategy_id`: `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4`
- Stored: N=86 ExpR=0.2839 Sharpe_ann=1.74 WR=0.698 last_trade_day=2026-04-26
- Mode A: N=91 ExpR=0.1742 Sharpe_ann=1.30 WR=0.396
- Drift reasons: |ΔExpR|=0.110>0.03, |ΔSharpe|=0.44>0.2, last_trade_day=2026-04-26 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2022:+0.025(N=1) 2023:+0.441(N=7) 2024:--0.065(N=10) 2025:+0.183(N=73)

### MNQ CME_PRECLOSE O5 RR1.0 X_MES_ATR60 long
- `strategy_id`: `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60`
- Stored: N=596 ExpR=0.1702 Sharpe_ann=1.88 WR=0.624 last_trade_day=None
- Mode A: N=340 ExpR=0.1910 Sharpe_ann=1.55 WR=0.585
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔSharpe|=0.33>0.2
- Mode-A per-year: 2019:--0.537(N=14) 2020:+0.383(N=63) 2021:+0.251(N=25) 2022:+0.316(N=65) 2023:+0.010(N=15) 2024:+0.089(N=82) 2025:+0.186(N=76)

### MNQ COMEX_SETTLE O5 RR1.0 COST_LT12 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12`
- Stored: N=1247 ExpR=0.1104 Sharpe_ann=1.74 WR=0.596 last_trade_day=None
- Mode A: N=672 ExpR=0.1034 Sharpe_ann=1.11 WR=0.591
- Drift reasons: |ΔN/N|=0.46>0.1, |ΔSharpe|=0.63>0.2
- Mode-A per-year: 2019:+0.146(N=18) 2020:+0.072(N=92) 2021:--0.061(N=87) 2022:+0.123(N=124) 2023:+0.173(N=107) 2024:+0.069(N=117) 2025:+0.187(N=127)

### MNQ COMEX_SETTLE O5 RR1.0 ORB_G5 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5`
- Stored: N=1473 ExpR=0.0890 Sharpe_ann=1.54 WR=0.592 last_trade_day=None
- Mode A: N=839 ExpR=0.0775 Sharpe_ann=0.94 WR=0.588
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔSharpe|=0.60>0.2
- Mode-A per-year: 2019:--0.015(N=58) 2020:+0.050(N=131) 2021:--0.086(N=130) 2022:+0.134(N=129) 2023:+0.142(N=125) 2024:+0.081(N=133) 2025:+0.185(N=133)

### MNQ COMEX_SETTLE O5 RR1.0 OVNRNG_100 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100`
- Stored: N=520 ExpR=0.1725 Sharpe_ann=1.76 WR=0.625 last_trade_day=None
- Mode A: N=283 ExpR=0.1844 Sharpe_ann=1.29 WR=0.633
- Drift reasons: |ΔN/N|=0.46>0.1, |ΔSharpe|=0.47>0.2
- Mode-A per-year: 2019:+0.338(N=4) 2020:+0.173(N=57) 2021:+0.119(N=36) 2022:+0.121(N=56) 2023:+0.157(N=8) 2024:+0.119(N=40) 2025:+0.292(N=82)

### MNQ COMEX_SETTLE O5 RR1.0 PD_CLEAR_LONG long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG`
- Stored: N=303 ExpR=0.1841 Sharpe_ann=1.49 WR=0.647 last_trade_day=2026-04-14
- Mode A: N=338 ExpR=0.1602 Sharpe_ann=1.27 WR=0.642
- Drift reasons: |ΔN/N|=0.12>0.1, |ΔSharpe|=0.22>0.2, last_trade_day=2026-04-14 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:--0.047(N=35) 2020:--0.021(N=52) 2021:+0.192(N=50) 2022:+0.240(N=56) 2023:+0.242(N=59) 2024:+0.313(N=42) 2025:+0.146(N=44)

### MNQ COMEX_SETTLE O5 RR1.0 X_MES_ATR60 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`
- Stored: N=673 ExpR=0.1512 Sharpe_ann=1.78 WR=0.618 last_trade_day=None
- Mode A: N=386 ExpR=0.1945 Sharpe_ann=1.62 WR=0.642
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔExpR|=0.043>0.03
- Mode-A per-year: 2019:+0.045(N=15) 2020:+0.074(N=80) 2021:+0.257(N=33) 2022:+0.261(N=75) 2023:+0.209(N=20) 2024:+0.144(N=86) 2025:+0.310(N=77)

### MNQ COMEX_SETTLE O5 RR1.5 ORB_G5 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
- Stored: N=1459 ExpR=0.1119 Sharpe_ann=1.52 WR=0.484 last_trade_day=None
- Mode A: N=839 ExpR=0.0950 Sharpe_ann=0.91 WR=0.472
- Drift reasons: |ΔN/N|=0.42>0.1, |ΔSharpe|=0.61>0.2
- Mode-A per-year: 2019:--0.138(N=58) 2020:+0.020(N=131) 2021:--0.015(N=130) 2022:+0.090(N=129) 2023:+0.192(N=125) 2024:+0.175(N=133) 2025:+0.211(N=133)

### MNQ COMEX_SETTLE O5 RR1.5 OVNRNG_100 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
- Stored: N=513 ExpR=0.2151 Sharpe_ann=1.70 WR=0.518 last_trade_day=None
- Mode A: N=283 ExpR=0.1894 Sharpe_ann=1.03 WR=0.498
- Drift reasons: |ΔN/N|=0.45>0.1, |ΔSharpe|=0.66>0.2
- Mode-A per-year: 2019:--0.448(N=4) 2020:+0.131(N=57) 2021:+0.140(N=36) 2022:+0.146(N=56) 2023:--0.129(N=8) 2024:+0.161(N=40) 2025:+0.357(N=82)

### MNQ COMEX_SETTLE O5 RR1.5 X_MES_ATR60 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60`
- Stored: N=664 ExpR=0.1609 Sharpe_ann=1.46 WR=0.498 last_trade_day=None
- Mode A: N=386 ExpR=0.2039 Sharpe_ann=1.31 WR=0.508
- Drift reasons: |ΔN/N|=0.42>0.1, |ΔExpR|=0.043>0.03
- Mode-A per-year: 2019:--0.000(N=15) 2020:--0.004(N=80) 2021:+0.284(N=33) 2022:+0.229(N=75) 2023:+0.160(N=20) 2024:+0.288(N=86) 2025:+0.319(N=77)

### MNQ COMEX_SETTLE O5 RR2.0 ORB_G5 long
- `strategy_id`: `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5`
- Stored: N=1443 ExpR=0.0890 Sharpe_ann=1.02 WR=0.395 last_trade_day=None
- Mode A: N=839 ExpR=0.1181 Sharpe_ann=0.97 WR=0.391
- Drift reasons: |ΔN/N|=0.42>0.1
- Mode-A per-year: 2019:--0.191(N=58) 2020:+0.032(N=131) 2021:--0.016(N=130) 2022:+0.222(N=129) 2023:+0.196(N=125) 2024:+0.242(N=133) 2025:+0.172(N=133)

### MNQ EUROPE_FLOW O5 RR1.0 COST_LT12 long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12`
- Stored: N=1024 ExpR=0.0921 Sharpe_ann=1.32 WR=0.591 last_trade_day=None
- Mode A: N=521 ExpR=0.0540 Sharpe_ann=0.51 WR=0.570
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔExpR|=0.038>0.03, |ΔSharpe|=0.82>0.2
- Mode-A per-year: 2019:+0.206(N=12) 2020:+0.095(N=78) 2021:+0.082(N=61) 2022:+0.042(N=116) 2023:+0.095(N=65) 2024:--0.002(N=76) 2025:+0.021(N=113)

### MNQ EUROPE_FLOW O5 RR1.0 CROSS_SGP_MOMENTUM long
- `strategy_id`: `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM`
- Stored: N=1020 ExpR=0.0850 Sharpe_ann=1.18 WR=0.606 last_trade_day=2026-04-23
- Mode A: N=535 ExpR=0.0502 Sharpe_ann=0.50 WR=0.598
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔExpR|=0.035>0.03, |ΔSharpe|=0.67>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
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
- Stored: N=1020 ExpR=0.0936 Sharpe_ann=1.01 WR=0.488 last_trade_day=2026-04-23
- Mode A: N=535 ExpR=0.0812 Sharpe_ann=0.64 WR=0.493
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔSharpe|=0.37>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
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
- Stored: N=1020 ExpR=0.1229 Sharpe_ann=1.12 WR=0.417 last_trade_day=2026-04-23
- Mode A: N=535 ExpR=0.1122 Sharpe_ann=0.75 WR=0.422
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔSharpe|=0.38>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
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
- Mode A: N=862 ExpR=0.0670 Sharpe_ann=0.78 WR=0.543
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔSharpe|=0.65>0.2
- Mode-A per-year: 2019:+0.003(N=84) 2020:+0.053(N=130) 2021:+0.044(N=136) 2022:+0.115(N=116) 2023:+0.008(N=145) 2024:+0.109(N=128) 2025:+0.132(N=123)

### MNQ NYSE_OPEN O5 RR1.0 ORB_G5 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5`
- Stored: N=1521 ExpR=0.0889 Sharpe_ann=1.47 WR=0.563 last_trade_day=None
- Mode A: N=874 ExpR=0.0640 Sharpe_ann=0.75 WR=0.542
- Drift reasons: |ΔN/N|=0.43>0.1, |ΔSharpe|=0.72>0.2
- Mode-A per-year: 2019:--0.015(N=89) 2020:+0.041(N=133) 2021:+0.048(N=137) 2022:+0.120(N=117) 2023:+0.006(N=147) 2024:+0.109(N=128) 2025:+0.132(N=123)

### MNQ NYSE_OPEN O5 RR1.0 X_MES_ATR60 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60`
- Stored: N=689 ExpR=0.1365 Sharpe_ann=1.54 WR=0.586 last_trade_day=None
- Mode A: N=353 ExpR=0.0749 Sharpe_ann=0.56 WR=0.544
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔExpR|=0.062>0.03, |ΔSharpe|=0.99>0.2
- Mode-A per-year: 2019:--0.180(N=18) 2020:+0.033(N=77) 2021:+0.190(N=31) 2022:+0.114(N=65) 2023:--0.091(N=17) 2024:+0.081(N=76) 2025:+0.135(N=69)

### MNQ NYSE_OPEN O5 RR1.5 COST_LT12 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`
- Stored: N=1472 ExpR=0.1050 Sharpe_ann=1.36 WR=0.457 last_trade_day=None
- Mode A: N=862 ExpR=0.0959 Sharpe_ann=0.90 WR=0.428
- Drift reasons: |ΔN/N|=0.41>0.1, |ΔSharpe|=0.46>0.2
- Mode-A per-year: 2019:--0.031(N=84) 2020:+0.021(N=130) 2021:+0.160(N=136) 2022:+0.104(N=116) 2023:+0.116(N=145) 2024:+0.208(N=128) 2025:+0.044(N=123)

### MNQ NYSE_OPEN O5 RR1.5 ORB_G5 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5`
- Stored: N=1485 ExpR=0.1067 Sharpe_ann=1.39 WR=0.458 last_trade_day=None
- Mode A: N=874 ExpR=0.0930 Sharpe_ann=0.88 WR=0.428
- Drift reasons: |ΔN/N|=0.41>0.1, |ΔSharpe|=0.51>0.2
- Mode-A per-year: 2019:--0.038(N=89) 2020:+0.013(N=133) 2021:+0.167(N=137) 2022:+0.114(N=117) 2023:+0.101(N=147) 2024:+0.208(N=128) 2025:+0.044(N=123)

### MNQ NYSE_OPEN O5 RR1.5 X_MES_ATR60 long
- `strategy_id`: `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60`
- Stored: N=673 ExpR=0.1323 Sharpe_ann=1.17 WR=0.468 last_trade_day=None
- Mode A: N=353 ExpR=0.0714 Sharpe_ann=0.43 WR=0.416
- Drift reasons: |ΔN/N|=0.48>0.1, |ΔExpR|=0.061>0.03, |ΔSharpe|=0.74>0.2
- Mode-A per-year: 2019:--0.229(N=18) 2020:+0.062(N=77) 2021:+0.221(N=31) 2022:+0.140(N=65) 2023:--0.008(N=17) 2024:+0.087(N=76) 2025:+0.031(N=69)

### MNQ SINGAPORE_OPEN O15 RR1.5 ATR_P50 long
- `strategy_id`: `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
- Stored: N=968 ExpR=0.1087 Sharpe_ann=1.14 WR=0.481 last_trade_day=2026-04-23
- Mode A: N=497 ExpR=0.2060 Sharpe_ann=1.51 WR=0.523
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔExpR|=0.097>0.03, |ΔSharpe|=0.37>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.384(N=32) 2020:+0.065(N=113) 2021:+0.040(N=51) 2022:+0.327(N=75) 2023:--0.031(N=26) 2024:+0.358(N=111) 2025:+0.195(N=89)

### MNQ SINGAPORE_OPEN O30 RR1.5 ATR_P50 long
- `strategy_id`: `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30`
- Stored: N=961 ExpR=0.1251 Sharpe_ann=1.28 WR=0.478 last_trade_day=2026-04-23
- Mode A: N=486 ExpR=0.2217 Sharpe_ann=1.57 WR=0.519
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔExpR|=0.097>0.03, |ΔSharpe|=0.29>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.337(N=29) 2020:+0.086(N=109) 2021:+0.080(N=57) 2022:+0.349(N=79) 2023:--0.209(N=25) 2024:+0.265(N=116) 2025:+0.436(N=71)

### MNQ TOKYO_OPEN O5 RR1.0 COST_LT12 long
- `strategy_id`: `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12`
- Stored: N=918 ExpR=0.0959 Sharpe_ann=1.31 WR=0.595 last_trade_day=None
- Mode A: N=469 ExpR=0.0786 Sharpe_ann=0.71 WR=0.586
- Drift reasons: |ΔN/N|=0.49>0.1, |ΔSharpe|=0.60>0.2
- Mode-A per-year: 2019:+0.206(N=21) 2020:+0.075(N=79) 2021:+0.139(N=72) 2022:+0.107(N=95) 2023:+0.049(N=31) 2024:+0.158(N=54) 2025:--0.031(N=117)

### MNQ TOKYO_OPEN O5 RR1.5 COST_LT08 long
- `strategy_id`: `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`
- Stored: N=427 ExpR=0.2037 Sharpe_ann=1.47 WR=0.510 last_trade_day=2026-04-23
- Mode A: N=205 ExpR=0.1620 Sharpe_ann=0.74 WR=0.493
- Drift reasons: |ΔN/N|=0.52>0.1, |ΔExpR|=0.042>0.03, |ΔSharpe|=0.73>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.179(N=2) 2020:+0.085(N=39) 2021:+0.226(N=21) 2022:+0.228(N=48) 2023:--0.073(N=5) 2024:+0.273(N=24) 2025:+0.116(N=66)

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
- Mode A: N=786 ExpR=0.1241 Sharpe_ann=1.00 WR=0.422
- Drift reasons: |ΔN/N|=0.47>0.1, |ΔExpR|=0.037>0.03
- Mode-A per-year: 2019:+0.173(N=64) 2020:+0.146(N=115) 2021:+0.236(N=134) 2022:+0.187(N=125) 2023:--0.047(N=113) 2024:+0.223(N=102) 2025:--0.021(N=133)

### MNQ US_DATA_1000 O5 RR1.0 PD_CLEAR_LONG long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG`
- Stored: N=211 ExpR=0.2270 Sharpe_ann=1.47 WR=0.640 last_trade_day=2026-04-23
- Mode A: N=237 ExpR=0.2151 Sharpe_ann=1.37 WR=0.624
- Drift reasons: |ΔN/N|=0.12>0.1, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.100(N=21) 2020:--0.015(N=35) 2021:+0.380(N=34) 2022:+0.435(N=34) 2023:+0.053(N=36) 2024:+0.454(N=39) 2025:+0.054(N=38)

### MNQ US_DATA_1000 O5 RR1.0 PD_DISPLACE_LONG long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG`
- Stored: N=192 ExpR=0.2396 Sharpe_ann=1.48 WR=0.646 last_trade_day=2026-04-02
- Mode A: N=208 ExpR=0.2308 Sharpe_ann=1.38 WR=0.635
- Drift reasons: last_trade_day=2026-04-02 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.119(N=13) 2020:+0.379(N=22) 2021:+0.133(N=27) 2022:+0.437(N=41) 2023:--0.038(N=41) 2024:+0.367(N=33) 2025:+0.197(N=31)

### MNQ US_DATA_1000 O5 RR1.0 PD_GO_LONG long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG`
- Stored: N=324 ExpR=0.1934 Sharpe_ann=1.53 WR=0.624 last_trade_day=2026-04-23
- Mode A: N=356 ExpR=0.1827 Sharpe_ann=1.42 WR=0.610
- Drift reasons: last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.028(N=26) 2020:+0.074(N=46) 2021:+0.305(N=52) 2022:+0.333(N=57) 2023:+0.001(N=62) 2024:+0.389(N=60) 2025:+0.050(N=53)

### MNQ US_DATA_1000 O5 RR1.0 X_MES_ATR60 long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60`
- Stored: N=694 ExpR=0.0996 Sharpe_ann=1.14 WR=0.572 last_trade_day=None
- Mode A: N=374 ExpR=0.0779 Sharpe_ann=0.60 WR=0.559
- Drift reasons: |ΔN/N|=0.46>0.1, |ΔSharpe|=0.53>0.2
- Mode-A per-year: 2019:+0.235(N=13) 2020:--0.059(N=86) 2021:--0.020(N=33) 2022:+0.079(N=70) 2023:+0.267(N=15) 2024:+0.280(N=78) 2025:+0.006(N=79)

### MNQ US_DATA_1000 O5 RR1.5 PD_GO_LONG long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG`
- Stored: N=321 ExpR=0.2222 Sharpe_ann=1.36 WR=0.511 last_trade_day=2026-04-23
- Mode A: N=356 ExpR=0.2119 Sharpe_ann=1.28 WR=0.494
- Drift reasons: |ΔN/N|=0.11>0.1, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.030(N=26) 2020:+0.037(N=46) 2021:+0.287(N=52) 2022:+0.401(N=57) 2023:+0.089(N=62) 2024:+0.494(N=60) 2025:--0.001(N=53)

### MNQ US_DATA_1000 O15 RR1.0 VWAP_MID_ALIGNED long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`
- Stored: N=744 ExpR=0.1492 Sharpe_ann=1.74 WR=0.593 last_trade_day=2026-04-23
- Mode A: N=497 ExpR=0.1275 Sharpe_ann=1.17 WR=0.545
- Drift reasons: |ΔN/N|=0.33>0.1, |ΔSharpe|=0.57>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.360(N=45) 2020:+0.127(N=75) 2021:+0.192(N=83) 2022:--0.004(N=69) 2023:+0.238(N=71) 2024:+0.165(N=76) 2025:--0.094(N=78)

### MNQ US_DATA_1000 O15 RR1.5 ORB_G5 long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Stored: N=1348 ExpR=0.0930 Sharpe_ann=1.16 WR=0.451 last_trade_day=2026-04-23
- Mode A: N=931 ExpR=0.0911 Sharpe_ann=0.93 WR=0.377
- Drift reasons: |ΔN/N|=0.31>0.1, |ΔSharpe|=0.22>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.298(N=90) 2020:+0.073(N=140) 2021:+0.122(N=138) 2022:+0.061(N=133) 2023:+0.153(N=138) 2024:+0.095(N=146) 2025:--0.083(N=146)

### MNQ US_DATA_1000 O15 RR1.5 VWAP_MID_ALIGNED long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
- Stored: N=701 ExpR=0.2101 Sharpe_ann=1.87 WR=0.499 last_trade_day=2026-04-23
- Mode A: N=497 ExpR=0.1919 Sharpe_ann=1.42 WR=0.433
- Drift reasons: |ΔN/N|=0.29>0.1, |ΔSharpe|=0.45>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.426(N=45) 2020:+0.181(N=75) 2021:+0.297(N=83) 2022:+0.095(N=69) 2023:+0.273(N=71) 2024:+0.242(N=76) 2025:--0.081(N=78)

### MNQ US_DATA_1000 O15 RR2.0 VWAP_MID_ALIGNED long
- `strategy_id`: `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15`
- Stored: N=635 ExpR=0.1757 Sharpe_ann=1.27 WR=0.405 last_trade_day=2026-04-23
- Mode A: N=497 ExpR=0.2316 Sharpe_ann=1.50 WR=0.320
- Drift reasons: |ΔN/N|=0.22>0.1, |ΔExpR|=0.056>0.03, |ΔSharpe|=0.23>0.2, last_trade_day=2026-04-23 >= 2026-01-01 (Mode-B grandfathered)
- Mode-A per-year: 2019:+0.424(N=45) 2020:+0.231(N=75) 2021:+0.385(N=83) 2022:+0.073(N=69) 2023:+0.341(N=71) 2024:+0.305(N=76) 2025:--0.072(N=78)


## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_revalidation_active_setups.py
```

No writes to validated_setups or experimental_strategies. Output is this
markdown document only. Numbers reproduce exactly on the same DB state.



## Scope

This file revalidates every active `validated_setups` row against strict Mode A IS
(`trading_day < 2026-01-01`) using canonical `orb_outcomes` + `daily_features` and the
canonical filter delegation `research.filter_utils.filter_signal`. Question: which
stored ExpR / N / Sharpe values diverge materially from the strict Mode A recomputation,
and how should consumers cite them going forward.

## Verdict

All 59 active validated_setups rows show material drift vs strict Mode A. Cause is
definitional (Mode B grandfathered IS included 2026-Q1 which Mode A treats as sacred OOS),
not data corruption. Treat the Mode A column as canonical truth from now on; do NOT mutate
`validated_setups` (that table is a snapshot of historical promotion criteria, not current
expectancy). Downstream consumers MUST cite Mode A ExpR for any new research baseline.

## Caveats / limitations

- This audit does NOT fail any lane — material drift is expected per the Mode B baseline rule (`research-truth-protocol.md`)
- Per-year era stability NOT recomputed here; live `strategy_validator` carries that gate
- DSR (C5), WFE (C6), C9 era-stability are not re-derived — separate audit per cell required for re-promotion
- 5 stored validated_setups have last_trade_day in [2026-01-01, 2026-04-08] = Mode-B-grandfathered

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_revalidation_active_setups.py
```

Output: this document. No writes to validated_setups or experimental_strategies.
Numbers reproduce exactly on the same DB state.

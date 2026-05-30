# ORB Edge Inventory Rebuild - 2026-05-30

Read-only canonical-layer run. Inputs: `orb_outcomes`, `daily_features`, `bars_1m` horizons only.
Holdout discipline: selection uses rows before `2026-01-01`; 2026+ is descriptive.

## Trial Counts
- Global tested cells: 6,462
- baseline_orb / NO_FILTER: 594
- cross_market_flow / PEER_COST_AND_ATR_OK_GE_1: 558
- cross_market_flow / PEER_COST_OK_GE_1: 558
- session_participation / orb_volume_ratio_HIGH_TERCILE: 594
- session_participation / orb_volume_ratio_LOW_TERCILE: 594
- size_friction / COST_TO_RISK_LE_0.08: 594
- size_friction / COST_TO_RISK_LE_0.10: 594
- size_friction / COST_TO_RISK_LE_0.12: 594
- size_friction / COST_TO_RISK_LE_0.15: 594
- volatility_state / atr_vel_ratio_HIGH_TERCILE: 594
- volatility_state / atr_vel_ratio_LOW_TERCILE: 594

## Inventory By Family
### deployable-candidate
- MNQ NYSE_OPEN baseline_orb: median ExpR 0.069, BH cells 9/18, t>=3 cells 9, N range 1702-1719, OOS median 0.05117823529411765
- MNQ US_DATA_1000 baseline_orb: median ExpR 0.066, BH cells 9/18, t>=3 cells 9, N range 1708-1718, OOS median 0.05442612859097128
- MNQ CME_PRECLOSE baseline_orb: median ExpR 0.016, BH cells 6/18, t>=3 cells 6, N range 609-1643, OOS median 0.003274218750000009
- MNQ SINGAPORE_OPEN baseline_orb: median ExpR -0.002, BH cells 6/18, t>=3 cells 1, N range 1710-1722, OOS median 0.12006162790697675
### research-provisional
- MGC TOKYO_OPEN size_friction: median ExpR 0.100, BH cells 2/72, t>=3 cells 0, N range 40-393, OOS median 0.09706724709784412
- MNQ US_DATA_1000 volatility_state: median ExpR 0.082, BH cells 7/36, t>=3 cells 6, N range 568-571, OOS median 0.1276116743471582
- MNQ NYSE_OPEN cross_market_flow: median ExpR 0.076, BH cells 14/36, t>=3 cells 11, N range 655-1657, OOS median 0.14361307707355242
- MNQ CME_PRECLOSE cross_market_flow: median ExpR 0.074, BH cells 18/36, t>=3 cells 15, N range 181-835, OOS median 0.09290542210020591
- MNQ US_DATA_1000 session_participation: median ExpR 0.071, BH cells 10/36, t>=3 cells 2, N range 568-571, OOS median 0.011267036290322552
- MNQ NYSE_CLOSE cross_market_flow: median ExpR 0.071, BH cells 11/36, t>=3 cells 8, N range 153-462, OOS median 0.3569588235294118
- MNQ NYSE_OPEN size_friction: median ExpR 0.070, BH cells 36/72, t>=3 cells 36, N range 1623-1709, OOS median 0.05117823529411765
- MNQ US_DATA_1000 size_friction: median ExpR 0.067, BH cells 36/72, t>=3 cells 36, N range 1470-1704, OOS median 0.05442612859097128
- MNQ SINGAPORE_OPEN cross_market_flow: median ExpR 0.064, BH cells 5/36, t>=3 cells 4, N range 240-924, OOS median 0.06821549295774648
- MNQ NYSE_OPEN session_participation: median ExpR 0.062, BH cells 7/36, t>=3 cells 0, N range 566-572, OOS median -0.024782829670329673
- MNQ TOKYO_OPEN cross_market_flow: median ExpR 0.061, BH cells 8/36, t>=3 cells 5, N range 223-956, OOS median 0.09872568438003221
- MNQ US_DATA_1000 cross_market_flow: median ExpR 0.058, BH cells 12/36, t>=3 cells 8, N range 626-1594, OOS median 0.025753755868544607
- MNQ COMEX_SETTLE cross_market_flow: median ExpR 0.055, BH cells 6/36, t>=3 cells 4, N range 319-1281, OOS median 0.01046884057971016
- MNQ NYSE_OPEN volatility_state: median ExpR 0.055, BH cells 5/36, t>=3 cells 0, N range 566-572, OOS median 0.10347580645161289
- MNQ SINGAPORE_OPEN size_friction: median ExpR 0.047, BH cells 16/72, t>=3 cells 9, N range 294-1632, OOS median 0.11160465116279068
- MES US_DATA_1000 size_friction: median ExpR 0.027, BH cells 11/72, t>=3 cells 2, N range 624-1651, OOS median 0.0122716216216216
- MNQ EUROPE_FLOW size_friction: median ExpR 0.027, BH cells 11/72, t>=3 cells 5, N range 587-1661, OOS median 0.06897356321839081
- MNQ CME_PRECLOSE size_friction: median ExpR 0.027, BH cells 24/72, t>=3 cells 24, N range 586-1551, OOS median 0.005891826923076923
- MES NYSE_OPEN size_friction: median ExpR 0.026, BH cells 15/72, t>=3 cells 4, N range 752-1688, OOS median -0.009496000000000003
- MNQ SINGAPORE_OPEN volatility_state: median ExpR 0.022, BH cells 2/36, t>=3 cells 0, N range 569-573, OOS median -0.046998156682027656
- MNQ CME_PRECLOSE volatility_state: median ExpR 0.022, BH cells 5/36, t>=3 cells 3, N range 202-547, OOS median 0.020948148148148153
- MNQ COMEX_SETTLE volatility_state: median ExpR 0.022, BH cells 5/36, t>=3 cells 3, N range 547-551, OOS median 0.07046464646464645
- MNQ TOKYO_OPEN size_friction: median ExpR 0.020, BH cells 18/72, t>=3 cells 8, N range 446-1652, OOS median 0.09133882352941178
- MNQ CME_PRECLOSE session_participation: median ExpR 0.019, BH cells 9/36, t>=3 cells 5, N range 201-546, OOS median 0.07453875598086124
- MES US_DATA_1000 volatility_state: median ExpR 0.019, BH cells 4/36, t>=3 cells 0, N range 567-573, OOS median 0.053283690476190484
- MNQ NYSE_PREOPEN size_friction: median ExpR 0.018, BH cells 21/72, t>=3 cells 12, N range 645-1664, OOS median -0.03208571428571432
- MGC CME_REOPEN size_friction: median ExpR 0.017, BH cells 3/72, t>=3 cells 0, N range 59-258, OOS median 0.33545400000000003
- MES NYSE_OPEN cross_market_flow: median ExpR 0.013, BH cells 3/36, t>=3 cells 0, N range 778-1707, OOS median 0.004314864864864864
- MES NYSE_OPEN baseline_orb: median ExpR 0.011, BH cells 2/18, t>=3 cells 0, N range 1711-1719, OOS median -0.01056933333333333
- MNQ TOKYO_OPEN session_participation: median ExpR 0.000, BH cells 5/36, t>=3 cells 0, N range 570-573, OOS median 0.06717142857142858
### unsupported
- MGC LONDON_METALS size_friction: median ExpR 0.023, BH cells 2/72, t>=3 cells 0, N range 31-708, OOS median -0.17500972222222225
- MGC US_DATA_830 size_friction: median ExpR 0.009, BH cells 1/72, t>=3 cells 0, N range 239-855, OOS median -0.1279064393939394
- MES NYSE_OPEN session_participation: median ExpR 0.009, BH cells 1/36, t>=3 cells 0, N range 569-572, OOS median -0.05044251207729469
- MGC SINGAPORE_OPEN size_friction: median ExpR 0.006, BH cells 0/72, t>=3 cells 0, N range 100-602, OOS median 0.23602772781265907
- MES NYSE_OPEN volatility_state: median ExpR 0.006, BH cells 1/36, t>=3 cells 0, N range 569-573, OOS median 0.01651333333333332
- MNQ LONDON_METALS size_friction: median ExpR 0.005, BH cells 6/72, t>=3 cells 0, N range 720-1686, OOS median -0.04257674418604651
- MGC US_DATA_1000 size_friction: median ExpR 0.004, BH cells 0/72, t>=3 cells 0, N range 212-867, OOS median -0.08479050215208031
- MNQ SINGAPORE_OPEN session_participation: median ExpR -0.002, BH cells 5/36, t>=3 cells 0, N range 569-573, OOS median 0.0379735294117647
- MGC COMEX_SETTLE size_friction: median ExpR -0.002, BH cells 4/72, t>=3 cells 0, N range 36-519, OOS median -0.06008922535211268
- MNQ LONDON_METALS baseline_orb: median ExpR -0.003, BH cells 3/18, t>=3 cells 0, N range 1702-1718, OOS median -0.04257674418604651
- MES CME_PRECLOSE cross_market_flow: median ExpR -0.005, BH cells 4/36, t>=3 cells 0, N range 200-1375, OOS median -0.10368434175531915
- MNQ BRISBANE_1025 size_friction: median ExpR -0.006, BH cells 7/72, t>=3 cells 0, N range 195-1601, OOS median 0.1803627906976744
- MES US_DATA_1000 cross_market_flow: median ExpR -0.007, BH cells 2/36, t>=3 cells 0, N range 787-1684, OOS median -0.010208160469667334
- MGC US_DATA_1000 session_participation: median ExpR -0.007, BH cells 1/36, t>=3 cells 0, N range 296-305, OOS median 0.028465000000000018
- MES US_DATA_1000 baseline_orb: median ExpR -0.010, BH cells 3/18, t>=3 cells 0, N range 1705-1719, OOS median 0.007311270270270261
- MNQ NYSE_CLOSE volatility_state: median ExpR -0.014, BH cells 0/36, t>=3 cells 0, N range 249-478, OOS median 0.02737615384615386
- MES TOKYO_OPEN size_friction: median ExpR -0.014, BH cells 6/72, t>=3 cells 0, N range 65-1221, OOS median 0.08019919354838709
- MNQ EUROPE_FLOW session_participation: median ExpR -0.015, BH cells 6/36, t>=3 cells 0, N range 569-572, OOS median 0.13878635416666665
- MES SINGAPORE_OPEN size_friction: median ExpR -0.016, BH cells 2/72, t>=3 cells 0, N range 53-1046, OOS median -0.09311669696969699
- MGC EUROPE_FLOW size_friction: median ExpR -0.016, BH cells 0/72, t>=3 cells 0, N range 26-641, OOS median 0.02772089285714286
- MNQ EUROPE_FLOW baseline_orb: median ExpR -0.016, BH cells 4/18, t>=3 cells 0, N range 1711-1719, OOS median 0.04807988505747127
- MNQ LONDON_METALS cross_market_flow: median ExpR -0.018, BH cells 3/36, t>=3 cells 0, N range 328-1343, OOS median -0.015664189189189184
- MNQ NYSE_CLOSE baseline_orb: median ExpR -0.020, BH cells 6/18, t>=3 cells 0, N range 749-1438, OOS median 0.1458343454790823
- MES US_DATA_1000 session_participation: median ExpR -0.021, BH cells 5/36, t>=3 cells 0, N range 567-572, OOS median -0.018969023569023573
- MNQ US_DATA_830 size_friction: median ExpR -0.022, BH cells 14/72, t>=3 cells 0, N range 871-1685, OOS median -0.2305800198525241
- MNQ CME_REOPEN cross_market_flow: median ExpR -0.023, BH cells 1/36, t>=3 cells 0, N range 224-537, OOS median 0.13357994505494505
- MGC US_DATA_1000 cross_market_flow: median ExpR -0.025, BH cells 2/36, t>=3 cells 0, N range 331-905, OOS median -0.1429841477436045
- MGC NYSE_OPEN size_friction: median ExpR -0.025, BH cells 5/72, t>=3 cells 0, N range 130-881, OOS median 0.11119652777777778
- MNQ LONDON_METALS session_participation: median ExpR -0.029, BH cells 4/36, t>=3 cells 0, N range 566-571, OOS median -0.13296777777777777
- MES COMEX_SETTLE size_friction: median ExpR -0.030, BH cells 22/72, t>=3 cells 0, N range 167-1473, OOS median -0.015032537313432856

## Nulls And Kills
- Any family/session absent from deployable-candidate or research-provisional is unsupported in this run.
- E0, E3, dead instruments, break-delay/speed, break-bar relative volume for E2, broad ML, gap/IBS/NR/EMA packages were excluded by prior NO-GO registry and not reopened.

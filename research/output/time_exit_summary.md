# P5b: Time-Exit Analysis

## Hypothesis
Trades surviving past T80 (80th percentile of winner exit times) are predominantly
losers. If avg_r < 0 past T80, a time-stop at T80 improves Sharpe with no entry change.

## Method
- T80 values from: `research/output/winner_speed_summary.csv`
- Outcomes from: `orb_outcomes` (outcome IN win/loss/scratch, orb_minutes=5)
- BH FDR correction at q=0.1 across 742 tested groups
- Minimum N_after = 30 to report

## Results: 742 groups tested
- SURVIVES (BH-sig, avg_r < 0): 116
- NOTABLE (p<0.05, not BH-sig): 0
- DIRECTIONAL (negative, not sig): 82
- NO-BENEFIT (avg_r_after >= 0): 544

## SURVIVED — Time-Stop Recommended

| Symbol | Session | RR | CB | T80 | N_after | avg_r_after | WR_after | p_bh |
|--------|---------|----|----|-----|---------|-------------|----------|------|
| MES | 1100 | 1.0 | 1 | 31m | 927 | -0.1608 | 50.5% | 0.0000 |
| MES | 1100 | 1.0 | 2 | 50m | 431 | -0.1769 | 49.0% | 0.0001 |
| MES | 1100 | 1.0 | 3 | 65m | 377 | -0.1669 | 49.1% | 0.0005 |
| MES | 1100 | 1.0 | 4 | 76m | 361 | -0.1054 | 52.3% | 0.0335 |
| MES | 1100 | 1.0 | 5 | 86m | 370 | -0.1841 | 47.3% | 0.0002 |
| MES | 1100 | 1.5 | 5 | 184m | 333 | -0.1287 | 40.8% | 0.0426 |
| MES | 1800 | 1.0 | 1 | 42m | 975 | -0.2085 | 46.4% | 0.0000 |
| MES | 1800 | 1.0 | 2 | 60m | 445 | -0.2015 | 46.1% | 0.0000 |
| MES | 1800 | 1.0 | 3 | 75m | 386 | -0.1889 | 46.4% | 0.0001 |
| MES | 1800 | 1.0 | 4 | 81m | 377 | -0.1866 | 46.2% | 0.0002 |
| MES | 1800 | 1.0 | 5 | 85m | 396 | -0.2374 | 43.2% | 0.0000 |
| MES | 1800 | 1.5 | 2 | 109m | 386 | -0.1009 | 41.7% | 0.0942 |
| MES | CME_CLOSE | 1.0 | 1 | 16m | 496 | -0.1153 | 50.0% | 0.0080 |
| MES | CME_OPEN | 1.0 | 2 | 103m | 217 | -0.1165 | 49.8% | 0.0837 |
| MES | CME_OPEN | 1.0 | 3 | 107m | 183 | -0.1462 | 48.1% | 0.0441 |
| MES | US_DATA_OPEN | 1.0 | 1 | 45m | 942 | -0.1329 | 47.3% | 0.0000 |
| MES | US_DATA_OPEN | 1.0 | 2 | 50m | 415 | -0.1161 | 47.7% | 0.0198 |
| MES | US_DATA_OPEN | 1.0 | 3 | 51m | 363 | -0.1171 | 47.7% | 0.0282 |
| MES | US_POST_EQUITY | 1.0 | 1 | 43m | 670 | -0.1145 | 48.2% | 0.0030 |
| MES | US_POST_EQUITY | 1.0 | 2 | 67m | 298 | -0.1189 | 47.6% | 0.0434 |
| MES | US_POST_EQUITY | 1.0 | 3 | 77m | 271 | -0.1048 | 48.3% | 0.0942 |
| MGC | 0030 | 1.0 | 1 | 34m | 364 | -0.2982 | 51.4% | 0.0000 |
| MGC | 0030 | 1.0 | 2 | 47m | 197 | -0.2979 | 48.2% | 0.0000 |
| MGC | 0030 | 1.0 | 3 | 57m | 192 | -0.2873 | 49.5% | 0.0000 |
| MGC | 0030 | 1.0 | 4 | 67m | 184 | -0.2419 | 51.6% | 0.0001 |
| MGC | 0030 | 1.0 | 5 | 71m | 183 | -0.2856 | 48.6% | 0.0000 |
| MGC | 0030 | 1.5 | 4 | 115m | 147 | -0.1410 | 47.6% | 0.0953 |
| MGC | 0030 | 1.5 | 5 | 117m | 148 | -0.1965 | 43.9% | 0.0187 |
| MGC | 0900 | 1.0 | 1 | 38m | 908 | -0.3004 | 54.6% | 0.0000 |
| MGC | 0900 | 1.0 | 2 | 48m | 352 | -0.2917 | 52.3% | 0.0000 |
| MGC | 0900 | 1.0 | 3 | 52m | 317 | -0.2476 | 54.6% | 0.0000 |
| MGC | 0900 | 1.0 | 4 | 52m | 322 | -0.2707 | 51.9% | 0.0000 |
| MGC | 0900 | 1.0 | 5 | 54m | 297 | -0.2603 | 52.9% | 0.0000 |
| MGC | 0900 | 1.5 | 1 | 60m | 855 | -0.1611 | 50.8% | 0.0000 |
| MGC | 0900 | 1.5 | 2 | 76m | 341 | -0.1418 | 49.0% | 0.0086 |
| MGC | 0900 | 1.5 | 3 | 82m | 285 | -0.1485 | 48.1% | 0.0127 |
| MGC | 0900 | 1.5 | 5 | 90m | 272 | -0.1224 | 48.9% | 0.0479 |
| MGC | 1000 | 1.0 | 1 | 32m | 844 | -0.1363 | 65.3% | 0.0000 |
| MGC | 1000 | 1.0 | 2 | 44m | 309 | -0.1471 | 62.1% | 0.0007 |
| MGC | 1000 | 1.0 | 3 | 45m | 212 | -0.1570 | 59.9% | 0.0037 |
| MGC | 1000 | 1.0 | 4 | 48m | 179 | -0.2160 | 54.8% | 0.0004 |
| MGC | 1000 | 1.5 | 1 | 53m | 879 | -0.1101 | 53.4% | 0.0005 |
| MGC | 1000 | 1.5 | 2 | 62m | 282 | -0.1220 | 51.4% | 0.0356 |
| MGC | 1100 | 1.0 | 1 | 72m | 1379 | -0.2540 | 50.4% | 0.0000 |
| MGC | 1100 | 1.0 | 2 | 117m | 621 | -0.2279 | 50.4% | 0.0000 |
| MGC | 1100 | 1.0 | 3 | 163m | 529 | -0.1865 | 51.8% | 0.0000 |
| MGC | 1100 | 1.0 | 4 | 171m | 546 | -0.2262 | 49.1% | 0.0000 |
| MGC | 1100 | 1.0 | 5 | 188m | 543 | -0.2197 | 49.4% | 0.0000 |
| MGC | 1100 | 1.5 | 1 | 188m | 1216 | -0.1200 | 46.8% | 0.0001 |
| MGC | 1130 | 1.0 | 1 | 44m | 446 | -0.3996 | 49.5% | 0.0000 |
| MGC | 1130 | 1.0 | 2 | 67m | 220 | -0.3591 | 50.4% | 0.0000 |
| MGC | 1130 | 1.0 | 3 | 81m | 231 | -0.3731 | 48.0% | 0.0000 |
| MGC | 1130 | 1.0 | 4 | 92m | 232 | -0.3752 | 47.8% | 0.0000 |
| MGC | 1130 | 1.0 | 5 | 106m | 223 | -0.3548 | 49.3% | 0.0000 |
| MGC | 1130 | 1.5 | 1 | 80m | 404 | -0.3023 | 46.5% | 0.0000 |
| MGC | 1130 | 1.5 | 2 | 160m | 192 | -0.2380 | 47.4% | 0.0003 |
| MGC | 1130 | 1.5 | 3 | 200m | 184 | -0.2075 | 48.9% | 0.0021 |
| MGC | 1130 | 1.5 | 4 | 215m | 190 | -0.2348 | 46.8% | 0.0005 |
| MGC | 1130 | 1.5 | 5 | 214m | 192 | -0.2522 | 45.3% | 0.0002 |
| MGC | 1130 | 2.0 | 1 | 160m | 343 | -0.1559 | 46.7% | 0.0044 |
| MGC | 1130 | 2.0 | 2 | 232m | 171 | -0.1662 | 43.9% | 0.0415 |
| MGC | 1800 | 1.0 | 1 | 36m | 1409 | -0.3390 | 47.4% | 0.0000 |
| MGC | 1800 | 1.0 | 2 | 58m | 579 | -0.3144 | 48.5% | 0.0000 |
| MGC | 1800 | 1.0 | 3 | 70m | 534 | -0.2812 | 49.4% | 0.0000 |
| MGC | 1800 | 1.0 | 4 | 74m | 525 | -0.2780 | 48.6% | 0.0000 |
| MGC | 1800 | 1.0 | 5 | 85m | 506 | -0.2622 | 49.0% | 0.0000 |
| MGC | 1800 | 1.5 | 1 | 71m | 1157 | -0.1888 | 47.1% | 0.0000 |
| MGC | 1800 | 1.5 | 2 | 112m | 484 | -0.1652 | 46.7% | 0.0003 |
| MGC | 1800 | 1.5 | 3 | 125m | 460 | -0.1223 | 48.0% | 0.0098 |
| MGC | 1800 | 1.5 | 4 | 131m | 443 | -0.1150 | 47.4% | 0.0198 |
| MGC | 1800 | 1.5 | 5 | 149m | 420 | -0.1061 | 47.6% | 0.0374 |
| MGC | 2300 | 1.0 | 1 | 28m | 1249 | -0.2243 | 50.2% | 0.0000 |
| MGC | 2300 | 1.0 | 2 | 36m | 572 | -0.2529 | 47.5% | 0.0000 |
| MGC | 2300 | 1.0 | 3 | 41m | 520 | -0.2299 | 48.1% | 0.0000 |
| MGC | 2300 | 1.0 | 4 | 48m | 475 | -0.2022 | 49.3% | 0.0000 |
| MGC | 2300 | 1.0 | 5 | 51m | 477 | -0.2182 | 47.6% | 0.0000 |
| MGC | 2300 | 1.5 | 1 | 44m | 1098 | -0.0880 | 47.7% | 0.0057 |
| MGC | 2300 | 1.5 | 2 | 58m | 494 | -0.1192 | 44.9% | 0.0141 |
| MGC | CME_OPEN | 1.0 | 1 | 52m | 770 | -0.3225 | 48.7% | 0.0000 |
| MGC | CME_OPEN | 1.0 | 2 | 67m | 314 | -0.3039 | 47.8% | 0.0000 |
| MGC | CME_OPEN | 1.0 | 3 | 79m | 244 | -0.2928 | 47.1% | 0.0000 |
| MGC | CME_OPEN | 1.0 | 4 | 88m | 214 | -0.2733 | 48.1% | 0.0000 |
| MGC | CME_OPEN | 1.0 | 5 | 93m | 201 | -0.2612 | 49.2% | 0.0000 |
| MGC | CME_OPEN | 1.5 | 1 | 106m | 487 | -0.0800 | 50.7% | 0.0882 |
| MGC | LONDON_OPEN | 1.0 | 1 | 38m | 1468 | -0.3604 | 45.3% | 0.0000 |
| MGC | LONDON_OPEN | 1.0 | 2 | 62m | 573 | -0.2976 | 48.0% | 0.0000 |
| MGC | LONDON_OPEN | 1.0 | 3 | 73m | 541 | -0.2788 | 48.4% | 0.0000 |
| MGC | LONDON_OPEN | 1.0 | 4 | 80m | 543 | -0.2833 | 47.5% | 0.0000 |
| MGC | LONDON_OPEN | 1.0 | 5 | 88m | 549 | -0.2795 | 47.5% | 0.0000 |
| MGC | LONDON_OPEN | 1.5 | 1 | 75m | 1207 | -0.2115 | 45.2% | 0.0000 |
| MGC | LONDON_OPEN | 1.5 | 2 | 115m | 499 | -0.1492 | 46.9% | 0.0009 |
| MGC | LONDON_OPEN | 1.5 | 3 | 132m | 476 | -0.1612 | 45.4% | 0.0006 |
| MGC | LONDON_OPEN | 1.5 | 4 | 144m | 450 | -0.1056 | 48.0% | 0.0308 |
| MGC | LONDON_OPEN | 1.5 | 5 | 167m | 434 | -0.0915 | 48.2% | 0.0721 |
| MGC | LONDON_OPEN | 2.0 | 1 | 122m | 1063 | -0.0740 | 44.4% | 0.0374 |
| MGC | US_DATA_OPEN | 1.0 | 1 | 48m | 419 | -0.3002 | 45.8% | 0.0000 |
| MGC | US_DATA_OPEN | 1.0 | 2 | 56m | 222 | -0.3271 | 42.8% | 0.0000 |
| MGC | US_DATA_OPEN | 1.0 | 3 | 60m | 213 | -0.2945 | 44.6% | 0.0000 |
| MGC | US_DATA_OPEN | 1.0 | 4 | 64m | 202 | -0.2861 | 45.5% | 0.0000 |
| MGC | US_DATA_OPEN | 1.0 | 5 | 72m | 185 | -0.2371 | 47.6% | 0.0003 |
| MGC | US_DATA_OPEN | 1.5 | 1 | 65m | 387 | -0.2779 | 38.2% | 0.0000 |
| MGC | US_DATA_OPEN | 1.5 | 2 | 83m | 171 | -0.2163 | 41.5% | 0.0064 |
| MGC | US_DATA_OPEN | 1.5 | 3 | 87m | 168 | -0.2502 | 39.3% | 0.0018 |
| MGC | US_DATA_OPEN | 1.5 | 4 | 86m | 173 | -0.2783 | 37.6% | 0.0004 |
| MGC | US_DATA_OPEN | 1.5 | 5 | 93m | 159 | -0.2357 | 39.0% | 0.0053 |
| MGC | US_DATA_OPEN | 2.0 | 1 | 81m | 326 | -0.2102 | 35.3% | 0.0013 |
| MGC | US_DATA_OPEN | 2.0 | 3 | 111m | 138 | -0.1786 | 37.0% | 0.0838 |
| MGC | US_DATA_OPEN | 2.0 | 4 | 109m | 143 | -0.2103 | 35.0% | 0.0369 |
| MGC | US_DATA_OPEN | 2.0 | 5 | 95m | 155 | -0.3114 | 30.3% | 0.0009 |
| MGC | US_DATA_OPEN | 2.5 | 1 | 96m | 281 | -0.1704 | 33.1% | 0.0293 |
| MGC | US_EQUITY_OPEN | 1.0 | 1 | 30m | 429 | -0.3411 | 46.9% | 0.0000 |
| MGC | US_EQUITY_OPEN | 1.0 | 2 | 49m | 199 | -0.2464 | 50.7% | 0.0000 |
| MGC | US_EQUITY_OPEN | 1.0 | 3 | 59m | 192 | -0.2499 | 50.5% | 0.0000 |
| MGC | US_EQUITY_OPEN | 1.0 | 4 | 63m | 191 | -0.2266 | 51.8% | 0.0002 |
| MGC | US_EQUITY_OPEN | 1.0 | 5 | 65m | 200 | -0.2911 | 46.5% | 0.0000 |
| MGC | US_EQUITY_OPEN | 1.5 | 1 | 49m | 354 | -0.1752 | 46.6% | 0.0008 |

**Mechanism:** Past T80, the market has moved on. Remaining open positions are
dead exposure — the breakout momentum has dissipated and mean-reversion dominates.
A time-stop at T80 exits these positions before they turn negative.

**What could kill it:** If momentum sessions (e.g. MGC 0900 trending day) continue
past T80, a hard time-stop would prematurely exit winners. Recommend per-session
T80 values rather than a global cutoff.

## CAVEATS
- T80 from winner_speed_summary.csv uses ALL winners regardless of filter
- Filter (ORB_G4/G5/etc.) not applied — T80 may differ for validated strategies only
- Small N_after for high-RR groups reduces statistical power
- DST regime not split (1000 session is clean; 0900/1800 may differ by DST half)

## NEXT STEPS
- If any group SURVIVES: implement time_stop_minutes per session in paper_trader.py
- Run separate analysis splitting by DST regime for 0900/1800
- Compare T80 for G4+ only vs all outcomes (filter may concentrate earlier winners)
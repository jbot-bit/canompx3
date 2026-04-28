---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Zero-Context Adversarial Audit — Raw Data Package
# Generated: 2026-03-19
# Context: NONE. Evaluate these numbers cold.
# Instrument: Micro futures (MGC=gold, MNQ=nasdaq, MES=S&P)
# Strategy: Opening Range Breakout (first 5 min of session)
# Entry: E2 = stop-market at ORB boundary + 1 tick slippage
# Stop: opposite ORB boundary. Target: RR * risk.
# Costs: included in pnl_r (MGC $5.74, MNQ $2.74, MES $3.74 round-trip)
# All p-values: one-sample t-test, H0: mean_R = 0

## 1. UNFILTERED BASELINE (ALL ORB sizes, E2 CB1 RR2.0 O5)

Inst  Session                   N    MeanR    StdR   WinR       t          p  Yrs
--------------------------------------------------------------------------------
MGC   CME_REOPEN             1211  -0.3536  1.0233  0.295  -12.02   0.000000  11yr
MGC   COMEX_SETTLE           2311  -0.2658  1.0116  0.355  -12.63   0.000000  11yr
MGC   EUROPE_FLOW            2597  -0.1835  1.0463  0.389   -8.93   0.000000  11yr
MGC   LONDON_METALS          2596  -0.1844  1.0697  0.376   -8.78   0.000000  11yr
MGC   NYSE_OPEN              2477  -0.0594  1.1796  0.395   -2.51   0.012229  11yr
MGC   SINGAPORE_OPEN         2594  -0.1303  1.1148  0.389   -5.95   0.000000  11yr
MGC   TOKYO_OPEN             2603  -0.2489  1.0000  0.375  -12.69   0.000000  11yr
MGC   US_DATA_1000           2272  -0.0989  1.1748  0.376   -4.01   0.000062  11yr
MGC   US_DATA_830            2355  -0.1337  1.1598  0.365   -5.59   0.000000  11yr

MNQ   BRISBANE_1025          1313  +0.0552  1.2565  0.417   +1.59   0.111566   6yr
MNQ   CME_PRECLOSE            863  +0.1492  1.3696  0.414   +3.20   0.001428   6yr
MNQ   CME_REOPEN              593  +0.0521  1.3431  0.381   +0.94   0.345570   6yr
MNQ   COMEX_SETTLE           1230  +0.1215  1.3638  0.404   +3.12   0.001827   6yr
MNQ   EUROPE_FLOW            1313  +0.1200  1.3372  0.414   +3.25   0.001185   6yr
MNQ   LONDON_METALS          1312  +0.0733  1.3457  0.389   +1.97   0.048745   6yr
MNQ   NYSE_CLOSE              375  -0.2354  1.2358  0.277   -3.68   0.000264   6yr
MNQ   NYSE_OPEN              1219  +0.1381  1.4201  0.391   +3.39   0.000709   6yr
MNQ   SINGAPORE_OPEN         1313  +0.0199  1.2718  0.394   +0.57   0.570980   6yr
MNQ   TOKYO_OPEN             1314  +0.0720  1.3137  0.401   +1.99   0.047202   6yr
MNQ   US_DATA_1000           1246  +0.1224  1.4046  0.390   +3.08   0.002146   6yr
MNQ   US_DATA_830            1246  +0.0271  1.3408  0.371   +0.71   0.476451   6yr

MES   CME_PRECLOSE           1105  -0.0396  1.2374  0.378   -1.06   0.287828   8yr
MES   CME_REOPEN              802  -0.2136  1.1719  0.313   -5.16   0.000000   8yr
MES   COMEX_SETTLE           1726  -0.0915  1.2087  0.364   -3.15   0.001689   8yr
MES   EUROPE_FLOW            1819  -0.1716  1.1459  0.347   -6.39   0.000000   8yr
MES   LONDON_METALS          1820  -0.0981  1.2022  0.363   -3.48   0.000516   8yr
MES   NYSE_CLOSE              478  -0.3049  1.1411  0.272   -5.84   0.000000   8yr
MES   NYSE_OPEN              1745  +0.0582  1.3310  0.388   +1.83   0.068171   8yr
MES   SINGAPORE_OPEN         1822  -0.1772  1.1091  0.361   -6.82   0.000000   8yr
MES   TOKYO_OPEN             1821  -0.0955  1.1672  0.379   -3.49   0.000495   8yr
MES   US_DATA_1000           1734  +0.0081  1.2951  0.379   +0.26   0.794132   8yr
MES   US_DATA_830            1732  -0.0965  1.2035  0.364   -3.34   0.000864   8yr


## 2. BH FDR CORRECTION (all instruments x sessions x RR1.0-4.0)

Total hypothesis tests: 192
BH FDR survivors (p_adj < 0.05): 135
  Positive (potential edge): 24
  Negative (confirmed losers): 111

### Positive BH FDR survivors:
Inst  Session                 RR      N    MeanR   WinR      p_adj
-----------------------------------------------------------------
MES   CME_PRECLOSE           1.0   1495  +0.0643  0.621   0.004820
MNQ   CME_PRECLOSE           1.0   1097  +0.2021  0.646   0.000000
MNQ   CME_PRECLOSE           1.5    972  +0.1790  0.508   0.000005
MNQ   CME_PRECLOSE           2.0    863  +0.1492  0.414   0.002426
MNQ   COMEX_SETTLE           1.5   1246  +0.1421  0.494   0.000039
MNQ   NYSE_OPEN              1.5   1263  +0.1421  0.471   0.000075
MNQ   US_DATA_1000           1.0   1299  +0.1403  0.594   0.000000
MNQ   NYSE_OPEN              2.0   1219  +0.1381  0.391   0.001249
MNQ   US_DATA_1000           1.5   1277  +0.1359  0.474   0.000122
MNQ   COMEX_SETTLE           2.5   1215  +0.1294  0.349   0.005566
MNQ   US_DATA_1000           2.0   1246  +0.1224  0.390   0.003491
MNQ   COMEX_SETTLE           2.0   1230  +0.1215  0.404   0.002999
MNQ   COMEX_SETTLE           1.0   1261  +0.1213  0.605   0.000006
MNQ   EUROPE_FLOW            2.0   1313  +0.1200  0.414   0.002032
MNQ   NYSE_OPEN              1.0   1294  +0.1172  0.576   0.000030
MNQ   NYSE_OPEN              2.5   1165  +0.1159  0.329   0.019946
MNQ   EUROPE_FLOW            1.5   1313  +0.1082  0.491   0.000983
MNQ   EUROPE_FLOW            1.0   1313  +0.0973  0.609   0.000152
MNQ   US_DATA_1000           2.5   1208  +0.0967  0.327   0.047596
MNQ   EUROPE_FLOW            2.5   1312  +0.0885  0.345   0.047596
MNQ   TOKYO_OPEN             1.5   1314  +0.0873  0.489   0.007187
MNQ   TOKYO_OPEN             1.0   1314  +0.0639  0.598   0.012598
MNQ   US_DATA_830            1.0   1278  +0.0550  0.570   0.047456
MNQ   LONDON_METALS          1.0   1312  +0.0532  0.574   0.048694

### Negative BH FDR survivors (confirmed money losers):
MES   NYSE_CLOSE             4.0    382  -0.6618  0.079   0.000000
MNQ   NYSE_CLOSE             4.0    302  -0.6296  0.079   0.000000
MES   CME_PRECLOSE           4.0    807  -0.5830  0.102   0.000000
MES   NYSE_CLOSE             3.0    404  -0.5434  0.134   0.000000
MGC   COMEX_SETTLE           4.0   1975  -0.4767  0.154   0.000000
MNQ   NYSE_CLOSE             3.0    324  -0.4621  0.145   0.000000
MGC   CME_REOPEN             4.0   1136  -0.4475  0.152   0.000000
MES   NYSE_CLOSE             2.5    430  -0.4470  0.186   0.000000
MGC   CME_REOPEN             3.0   1162  -0.4101  0.201   0.000000
MGC   CME_REOPEN             2.5   1186  -0.3854  0.238   0.000000
MNQ   NYSE_CLOSE             2.5    337  -0.3846  0.190   0.000000
MGC   COMEX_SETTLE           3.0   2100  -0.3840  0.225   0.000000
MGC   CME_REOPEN             2.0   1211  -0.3536  0.295   0.000000
MES   CME_REOPEN             4.0    742  -0.3167  0.163   0.000000
MNQ   CME_PRECLOSE           4.0    635  -0.3108  0.150   0.000006
MGC   COMEX_SETTLE           2.5   2201  -0.3050  0.290   0.000000
MES   NYSE_CLOSE             2.0    478  -0.3049  0.272   0.000000
MGC   US_DATA_1000           4.0   1931  -0.2954  0.181   0.000000
MES   CME_PRECLOSE           3.0    908  -0.2913  0.211   0.000000
MGC   CME_REOPEN             1.5   1292  -0.2888  0.395   0.000000
MES   CME_REOPEN             3.0    763  -0.2828  0.214   0.000000
MGC   TOKYO_OPEN             4.0   2595  -0.2664  0.219   0.000000
MGC   COMEX_SETTLE           2.0   2311  -0.2658  0.355   0.000000
MGC   COMEX_SETTLE           1.5   2386  -0.2613  0.427   0.000000
MGC   TOKYO_OPEN             3.0   2600  -0.2576  0.276   0.000000
MES   CME_REOPEN             2.5    780  -0.2554  0.254   0.000000
MGC   TOKYO_OPEN             1.5   2603  -0.2498  0.446   0.000000
MGC   TOKYO_OPEN             2.0   2603  -0.2489  0.375   0.000000
MGC   TOKYO_OPEN             2.5   2602  -0.2432  0.323   0.000000
MGC   TOKYO_OPEN             1.0   2603  -0.2415  0.520   0.000000
MGC   CME_REOPEN             1.0   1453  -0.2385  0.518   0.000000
MNQ   NYSE_CLOSE             2.0    375  -0.2354  0.277   0.000513
MGC   COMEX_SETTLE           1.0   2475  -0.2246  0.541   0.000000
MES   SINGAPORE_OPEN         3.0   1820  -0.2234  0.256   0.000000
MGC   US_DATA_830            4.0   2161  -0.2230  0.199   0.000000
MES   SINGAPORE_OPEN         4.0   1817  -0.2152  0.207   0.000000
MES   CME_REOPEN             2.0    802  -0.2136  0.313   0.000001
MES   SINGAPORE_OPEN         2.5   1822  -0.2078  0.297   0.000000
MGC   EUROPE_FLOW            1.5   2598  -0.2043  0.455   0.000000
MGC   EUROPE_FLOW            2.5   2595  -0.2033  0.325   0.000000
MGC   EUROPE_FLOW            4.0   2584  -0.1977  0.230   0.000000
MGC   EUROPE_FLOW            3.0   2592  -0.1962  0.287   0.000000
MGC   NYSE_OPEN              4.0   2207  -0.1961  0.207   0.000000
MGC   EUROPE_FLOW            1.0   2599  -0.1878  0.553   0.000000
MGC   LONDON_METALS          2.0   2596  -0.1844  0.376   0.000000
MGC   LONDON_METALS          2.5   2592  -0.1839  0.322   0.000000
MGC   LONDON_METALS          1.5   2598  -0.1837  0.450   0.000000
MGC   EUROPE_FLOW            2.0   2597  -0.1835  0.389   0.000000
MGC   LONDON_METALS          4.0   2574  -0.1801  0.228   0.000000
MGC   LONDON_METALS          3.0   2586  -0.1800  0.283   0.000000
MES   SINGAPORE_OPEN         1.5   1822  -0.1785  0.434   0.000000
MGC   LONDON_METALS          1.0   2600  -0.1782  0.558   0.000000
MES   EUROPE_FLOW            4.0   1810  -0.1774  0.207   0.000009
MES   SINGAPORE_OPEN         2.0   1822  -0.1772  0.361   0.000000
MES   EUROPE_FLOW            1.5   1821  -0.1755  0.415   0.000000
MES   COMEX_SETTLE           4.0   1621  -0.1726  0.200   0.000069
MES   EUROPE_FLOW            3.0   1814  -0.1721  0.260   0.000001
MES   EUROPE_FLOW            2.0   1819  -0.1716  0.347   0.000000
MGC   US_DATA_1000           3.0   2081  -0.1700  0.262   0.000000
MES   EUROPE_FLOW            2.5   1817  -0.1686  0.299   0.000000
MGC   US_DATA_830            3.0   2252  -0.1675  0.265   0.000000
MES   NYSE_CLOSE             1.5    563  -0.1658  0.393   0.000362
MES   SINGAPORE_OPEN         1.0   1822  -0.1649  0.552   0.000000
MES   CME_PRECLOSE           2.5    990  -0.1645  0.283   0.000238
MES   CME_REOPEN             1.5    857  -0.1639  0.399   0.000010
MGC   US_DATA_830            2.5   2305  -0.1589  0.304   0.000000
MES   US_DATA_830            4.0   1675  -0.1515  0.210   0.000407
MGC   SINGAPORE_OPEN         1.5   2596  -0.1513  0.455   0.000000
MGC   SINGAPORE_OPEN         1.0   2597  -0.1459  0.562   0.000000
MES   EUROPE_FLOW            1.0   1822  -0.1443  0.538   0.000000
MGC   SINGAPORE_OPEN         4.0   2549  -0.1438  0.232   0.000013
MES   NYSE_OPEN              4.0   1528  -0.1428  0.190   0.002795
MGC   US_DATA_1000           2.5   2167  -0.1347  0.310   0.000004
MGC   US_DATA_830            2.0   2355  -0.1337  0.365   0.000000
MGC   SINGAPORE_OPEN         3.0   2575  -0.1325  0.292   0.000003
MES   COMEX_SETTLE           3.0   1674  -0.1315  0.262   0.000499
MES   LONDON_METALS          3.0   1814  -0.1312  0.262   0.000287
MGC   SINGAPORE_OPEN         2.0   2594  -0.1303  0.389   0.000000
MGC   US_DATA_830            1.5   2411  -0.1276  0.438   0.000000
MGC   SINGAPORE_OPEN         2.5   2589  -0.1199  0.337   0.000004
MES   LONDON_METALS          4.0   1809  -0.1192  0.213   0.004751
MGC   NYSE_OPEN              3.0   2326  -0.1150  0.282   0.000221
MES   COMEX_SETTLE           2.5   1699  -0.1131  0.305   0.000997
MES   LONDON_METALS          2.5   1817  -0.1116  0.307   0.000771
MGC   US_DATA_830            1.0   2478  -0.1100  0.552   0.000000
MES   US_DATA_1000           4.0   1578  -0.1100  0.202   0.020646
MES   COMEX_SETTLE           1.5   1741  -0.1080  0.429   0.000036
MES   US_DATA_830            3.0   1703  -0.1072  0.274   0.004317
MES   CME_REOPEN             1.0    969  -0.1061  0.535   0.000198
MES   US_DATA_830            1.5   1752  -0.0997  0.434   0.000135
MGC   US_DATA_1000           2.0   2272  -0.0989  0.376   0.000136
MES   LONDON_METALS          2.0   1820  -0.0981  0.363   0.000943
MES   US_DATA_830            2.0   1732  -0.0965  0.364   0.001508
MES   LONDON_METALS          1.5   1820  -0.0962  0.436   0.000167
MES   TOKYO_OPEN             2.0   1821  -0.0955  0.379   0.000913
MGC   NYSE_OPEN              2.5   2389  -0.0950  0.328   0.000768
MES   TOKYO_OPEN             1.5   1823  -0.0941  0.455   0.000136
MES   US_DATA_830            2.5   1716  -0.0932  0.315   0.006608
MES   TOKYO_OPEN             4.0   1817  -0.0926  0.228   0.027997
MES   COMEX_SETTLE           2.0   1726  -0.0915  0.364   0.002795
MES   TOKYO_OPEN             2.5   1821  -0.0907  0.327   0.005268
MES   TOKYO_OPEN             1.0   1824  -0.0885  0.572   0.000006
MES   LONDON_METALS          1.0   1821  -0.0834  0.554   0.000045
MES   TOKYO_OPEN             3.0   1820  -0.0778  0.290   0.033235
MGC   US_DATA_1000           1.5   2379  -0.0770  0.460   0.000419
MGC   NYSE_OPEN              1.5   2536  -0.0695  0.467   0.000968
MES   US_DATA_830            1.0   1782  -0.0659  0.561   0.001595
MES   COMEX_SETTLE           1.0   1753  -0.0633  0.564   0.002540
MGC   NYSE_OPEN              2.0   2477  -0.0594  0.395   0.018488
MGC   US_DATA_1000           1.0   2484  -0.0580  0.584   0.000673
MGC   NYSE_OPEN              1.0   2575  -0.0569  0.590   0.000602

## 3. G4 FILTER IMPACT (ORB >= 4 points, E2 CB1 RR2.0 O5)

Inst  Session               N_raw    R_raw   N_g4     R_g4    Delta       p_g4
------------------------------------------------------------------------------
MGC   CME_REOPEN             1211  -0.3536    129  +0.3892  +0.7429   0.002086
MGC   COMEX_SETTLE           2311  -0.2658    139  +0.0232  +0.2890   0.837254
MGC   EUROPE_FLOW            2597  -0.1835    151  +0.2301  +0.4135   0.040013
MGC   LONDON_METALS          2596  -0.1844    209  -0.0166  +0.1678   0.855933
MGC   NYSE_OPEN              2477  -0.0594    559  +0.0813  +0.1407   0.152086
MGC   SINGAPORE_OPEN         2594  -0.1303    377  +0.1191  +0.2493   0.089291
MGC   TOKYO_OPEN             2603  -0.2489    167  +0.2051  +0.4540   0.055215
MGC   US_DATA_1000           2272  -0.0989    565  -0.0067  +0.0922   0.903712
MGC   US_DATA_830            2355  -0.1337    630  -0.0688  +0.0649   0.184769
MNQ   BRISBANE_1025          1313  +0.0552   1166  +0.0773  +0.0221   0.039192
MNQ   CME_PRECLOSE            863  +0.1492    863  +0.1492  +0.0000   0.001428
MNQ   CME_REOPEN              593  +0.0521    593  +0.0521  +0.0000   0.345570
MNQ   COMEX_SETTLE           1230  +0.1215   1230  +0.1215  +0.0000   0.001827
MNQ   EUROPE_FLOW            1313  +0.1200   1299  +0.1221  +0.0021   0.001054
MNQ   LONDON_METALS          1312  +0.0733   1309  +0.0741  +0.0008   0.046643
MNQ   NYSE_CLOSE              375  -0.2354    375  -0.2354  +0.0000   0.000264
MNQ   NYSE_OPEN              1219  +0.1381   1218  +0.1373  -0.0009   0.000772
MNQ   SINGAPORE_OPEN         1313  +0.0199   1223  +0.0322  +0.0123   0.381764
MNQ   TOKYO_OPEN             1314  +0.0720   1297  +0.0812  +0.0092   0.026624
MNQ   US_DATA_1000           1246  +0.1224   1241  +0.1218  -0.0006   0.002331
MNQ   US_DATA_830            1246  +0.0271   1238  +0.0305  +0.0034   0.424610
MES   CME_PRECLOSE           1105  -0.0396    599  +0.0018  +0.0414   0.973256
MES   CME_REOPEN              802  -0.2136    393  -0.1383  +0.0753   0.029801
MES   COMEX_SETTLE           1726  -0.0915    798  +0.0114  +0.1030   0.804494
MES   EUROPE_FLOW            1819  -0.1716    577  -0.0881  +0.0835   0.094538
MES   LONDON_METALS          1820  -0.0981    824  -0.0205  +0.0776   0.648645
MES   NYSE_CLOSE              478  -0.3049    237  -0.1201  +0.1848   0.143738
MES   NYSE_OPEN              1745  +0.0582   1593  +0.0806  +0.0224   0.017061
MES   SINGAPORE_OPEN         1822  -0.1772    355  +0.0401  +0.2173   0.563920
MES   TOKYO_OPEN             1821  -0.0955    538  +0.0142  +0.1097   0.799031
MES   US_DATA_1000           1734  +0.0081   1367  +0.0526  +0.0445   0.144687
MES   US_DATA_830            1732  -0.0965    850  -0.0247  +0.0719   0.581261

## 4. YEARLY CONSISTENCY (MNQ top sessions, E2 CB1 RR2.0 O5 unfiltered)

CME_PRECLOSE:
  2021: N= 151 MeanR=+0.2151
  2022: N= 141 MeanR=+0.1637
  2023: N= 155 MeanR=+0.0581
  2024: N= 183 MeanR=+0.1290
  2025: N= 198 MeanR=+0.2274
  2026: N=  35 MeanR=-0.1259
  5/6 positive years

NYSE_OPEN:
  2021: N= 212 MeanR=+0.0730
  2022: N= 247 MeanR=+0.1579
  2023: N= 240 MeanR=+0.0465
  2024: N= 249 MeanR=+0.2756
  2025: N= 232 MeanR=+0.1115
  2026: N=  39 MeanR=+0.2116
  6/6 positive years

COMEX_SETTLE:
  2021: N= 222 MeanR=-0.0453
  2022: N= 249 MeanR=+0.1129
  2023: N= 243 MeanR=+0.1999
  2024: N= 236 MeanR=+0.1036
  2025: N= 240 MeanR=+0.2566
  2026: N=  40 MeanR=-0.0794
  4/6 positive years

EUROPE_FLOW:
  2021: N= 236 MeanR=+0.0698
  2022: N= 258 MeanR=+0.1303
  2023: N= 258 MeanR=+0.1996
  2024: N= 259 MeanR=+0.0641
  2025: N= 257 MeanR=+0.1541
  2026: N=  45 MeanR=-0.0063
  5/6 positive years

US_DATA_1000:
  2021: N= 222 MeanR=+0.1322
  2022: N= 246 MeanR=+0.1803
  2023: N= 247 MeanR=+0.0899
  2024: N= 250 MeanR=+0.2113
  2025: N= 239 MeanR=+0.0081
  2026: N=  42 MeanR=+0.0447
  6/6 positive years

## 5. CURRENTLY VALIDATED STRATEGIES

Total active: 106
Strategy                                                   ExpR  Sharpe     N    WR Filter                   
--------------------------------------------------------------------------------------------------------------
M2K_LONDON_METALS_E2_RR3.0_CB1_ORB_G6_NOMON_O15_S075     +0.223  +0.143   184 0.283 ORB_G6_NOMON             
M2K_LONDON_METALS_E2_RR3.0_CB1_ORB_G6_NOMON_O15          +0.197  +0.114   182 0.324 ORB_G6_NOMON             
M2K_LONDON_METALS_E2_RR3.0_CB1_ORB_G5_FAST10_O15_S075    +0.170  +0.112   222 0.270 ORB_G5_FAST10            
M2K_LONDON_METALS_E2_RR2.5_CB1_ORB_G6_NOMON_O15          +0.136  +0.088   185 0.351 ORB_G6_NOMON             
M2K_US_DATA_1000_E2_RR1.0_CB1_VOL_RV12_N20_O30           +0.131  +0.141   539 0.597 VOL_RV12_N20             
M2K_NYSE_OPEN_E2_RR1.0_CB1_VOL_RV12_N20_O30              +0.128  +0.135   486 0.589 VOL_RV12_N20             
M2K_NYSE_OPEN_E2_RR1.5_CB1_VOL_RV12_N20_S075             +0.124  +0.121   573 0.421 VOL_RV12_N20             
M2K_NYSE_OPEN_E2_RR1.5_CB1_VOL_RV12_N20                  +0.120  +0.103   553 0.481 VOL_RV12_N20             
M2K_LONDON_METALS_E2_RR4.0_CB1_ORB_G5_FAST10_O15_S075    +0.114  +0.066   215 0.200 ORB_G5_FAST10            
M2K_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_S075                +0.114  +0.145   144 0.549 ORB_G5                   
M2K_LONDON_METALS_E2_RR2.5_CB1_ORB_G5_FAST10_O15_S075    +0.110  +0.082   223 0.291 ORB_G5_FAST10            
M2K_NYSE_OPEN_E2_RR1.0_CB1_VOL_RV12_N20                  +0.110  +0.119   587 0.595 VOL_RV12_N20             
M2K_US_DATA_1000_E2_RR2.0_CB1_VOL_RV12_N20_S075          +0.108  +0.092   625 0.349 VOL_RV12_N20             
M2K_LONDON_METALS_E1_RR2.5_CB1_ORB_G6_NOMON_O15          +0.106  +0.069   185 0.341 ORB_G6_NOMON             
M2K_US_DATA_1000_E2_RR2.0_CB1_VOL_RV12_N20               +0.102  +0.077   611 0.407 VOL_RV12_N20             
M2K_LONDON_METALS_E2_RR3.0_CB1_ORB_G5_CONT_O15_S075      +0.100  +0.068   373 0.249 ORB_G5_CONT              
M2K_US_DATA_1000_E2_RR2.0_CB1_ORB_G6_S075                +0.099  +0.083   592 0.336 ORB_G6                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_VOL_RV12_N20_S075             +0.098  +0.122   600 0.525 VOL_RV12_N20             
M2K_US_DATA_1000_E2_RR2.5_CB1_VOL_RV12_N20               +0.094  +0.062   583 0.348 VOL_RV12_N20             
M2K_LONDON_METALS_E1_RR2.0_CB4_ORB_G6_NOMON_O15_S075     +0.093  +0.077   182 0.330 ORB_G6_NOMON             
M2K_NYSE_OPEN_E2_RR2.0_CB1_VOL_RV12_N20_S075             +0.093  +0.078   545 0.332 VOL_RV12_N20             
M2K_LONDON_METALS_E2_RR3.0_CB1_ORB_G5_O15_S075           +0.093  +0.063   376 0.247 ORB_G5                   
M2K_LONDON_METALS_E1_RR3.0_CB1_ORB_G6_NOMON_O15          +0.092  +0.054   180 0.294 ORB_G6_NOMON             
M2K_US_DATA_1000_E2_RR2.5_CB1_VOL_RV12_N20_S075          +0.089  +0.068   600 0.290 VOL_RV12_N20             
M2K_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_S075                +0.087  +0.086   619 0.406 ORB_G6                   
M2K_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5_S075                   +0.085  +0.083  1147 0.402 ORB_G5                   
M2K_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_G5_O30_S075          +0.084  +0.058   216 0.245 ORB_G5                   
M2K_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4_S075                   +0.084  +0.083  1185 0.403 ORB_G4                   
M2K_LONDON_METALS_E2_RR3.0_CB1_ORB_G5_NOMON_O15          +0.084  +0.050   287 0.296 ORB_G5_NOMON             
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O30                    +0.083  +0.087   772 0.566 ORB_G5                   
M2K_LONDON_METALS_E2_RR2.0_CB1_ORB_G5_FAST10_O15_S075    +0.082  +0.070   225 0.333 ORB_G5_FAST10            
M2K_US_DATA_1000_E2_RR2.0_CB1_NO_FILTER_S075             +0.082  +0.071  1228 0.344 NO_FILTER                
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30                    +0.081  +0.086   770 0.565 ORB_G6                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O30                    +0.081  +0.085   777 0.565 ORB_G4                   
M2K_US_DATA_1000_E2_RR1.5_CB1_VOL_RV12_N20_S075          +0.081  +0.081   641 0.413 VOL_RV12_N20             
M2K_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O30                 +0.080  +0.085   780 0.565 NO_FILTER                
M2K_US_DATA_1000_E2_RR2.0_CB1_ORB_G4_S075                +0.080  +0.069   976 0.335 ORB_G4                   
M2K_US_DATA_1000_E2_RR2.0_CB1_ORB_G5_S075                +0.079  +0.067   794 0.331 ORB_G5                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30                 +0.078  +0.084   911 0.571 ORB_G4                   
M2K_US_DATA_1000_E2_RR3.0_CB1_VOL_RV12_N20_S075          +0.078  +0.054   588 0.248 VOL_RV12_N20             
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30                 +0.077  +0.082   889 0.569 ORB_G6                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O30                 +0.077  +0.082   901 0.569 ORB_G5                   
M2K_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER_S075                +0.077  +0.076  1221 0.401 NO_FILTER                
M2K_LONDON_METALS_E1_RR2.0_CB3_ORB_G5_FAST10_O15_S075    +0.075  +0.063   218 0.326 ORB_G5_FAST10            
M2K_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O30              +0.075  +0.080   929 0.571 NO_FILTER                
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O30                    +0.074  +0.077   755 0.560 ORB_G8                   
M2K_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_S075                +0.073  +0.073  1009 0.406 ORB_G4                   
M2K_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_S075                +0.072  +0.072   824 0.402 ORB_G5                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15                 +0.072  +0.077   735 0.567 ORB_G8                   
M2K_LONDON_METALS_E1_RR4.0_CB3_ORB_G5_FAST10_O15_S075    +0.072  +0.042   203 0.187 ORB_G5_FAST10            
M2K_US_DATA_1000_E2_RR1.5_CB1_VOL_RV12_N20               +0.071  +0.063   630 0.475 VOL_RV12_N20             
M2K_LONDON_METALS_E2_RR3.0_CB1_ORB_G4_FAST10_O15_S075    +0.070  +0.048   348 0.244 ORB_G4_FAST10            
M2K_NYSE_OPEN_E2_RR1.0_CB1_VOL_RV12_N20_O30_S075         +0.070  +0.083   549 0.492 VOL_RV12_N20             
M2K_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER_S075             +0.069  +0.071  1262 0.414 NO_FILTER                
M2K_NYSE_OPEN_E2_RR1.5_CB1_ORB_G6_S075                   +0.069  +0.068  1057 0.393 ORB_G6                   
M2K_US_DATA_1000_E2_RR1.0_CB1_VOL_RV12_N20_S075          +0.063  +0.081   655 0.522 VOL_RV12_N20             
M2K_US_DATA_1000_E2_RR1.0_CB1_VOL_RV12_N20               +0.060  +0.068   647 0.587 VOL_RV12_N20             
M2K_US_DATA_1000_E2_RR2.5_CB1_NO_FILTER_S075             +0.060  +0.047  1183 0.284 NO_FILTER                
M2K_NYSE_OPEN_E2_RR1.0_CB1_VOL_RV12_N20_O15_S075         +0.059  +0.071   644 0.491 VOL_RV12_N20             
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15_S075               +0.059  +0.071  1102 0.492 ORB_G4                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15_S075               +0.058  +0.070  1092 0.491 ORB_G5                   
M2K_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G6_O30                  +0.058  +0.063   256 0.570 ORB_G6                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15                    +0.058  +0.061  1037 0.558 ORB_G4                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15                    +0.058  +0.061  1027 0.558 ORB_G5                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075            +0.057  +0.069  1118 0.492 NO_FILTER                
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O15_S075               +0.056  +0.068  1085 0.489 ORB_G6                   
M2K_NYSE_OPEN_E2_RR1.5_CB1_ORB_G8_S075                   +0.056  +0.055   722 0.382 ORB_G8                   
M2K_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_S075             +0.056  +0.072  1287 0.526 NO_FILTER                
M2K_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15                 +0.055  +0.059  1053 0.558 NO_FILTER                
M2K_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5_O15_S075             +0.055  +0.055   209 0.397 ORB_G5                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15                    +0.055  +0.058   943 0.555 ORB_G8                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O15                    +0.054  +0.058  1020 0.556 ORB_G6                   
M2K_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4                        +0.054  +0.047  1147 0.454 ORB_G4                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15                 +0.054  +0.058  1074 0.564 ORB_G5                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_S075                +0.053  +0.067  1034 0.511 ORB_G4                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_S075                   +0.053  +0.065  1192 0.497 ORB_G5                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_S075                +0.052  +0.066   849 0.505 ORB_G5                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_S075                +0.052  +0.065   644 0.500 ORB_G6                   
M2K_US_DATA_1000_E2_RR2.5_CB1_ORB_G4_S075                +0.052  +0.040   936 0.275 ORB_G4                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_S075                   +0.050  +0.063  1231 0.497 ORB_G4                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15_S075               +0.050  +0.060  1008 0.484 ORB_G8                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15                 +0.048  +0.052  1123 0.563 ORB_G4                   
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15                 +0.048  +0.051   981 0.559 ORB_G6                   
M2K_LONDON_METALS_E1_RR1.5_CB3_ORB_G6_FAST10_O30         +0.047  +0.041   205 0.449 ORB_G6_FAST10            
M2K_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER                     +0.046  +0.040  1183 0.453 NO_FILTER                
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_S075                   +0.045  +0.056  1102 0.491 ORB_G6                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_S075                +0.044  +0.054  1267 0.496 NO_FILTER                
M2K_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O30                 +0.043  +0.046   433 0.561 ORB_G6                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_S075                   +0.042  +0.051   765 0.484 ORB_G8                   
M2K_NYSE_OPEN_E1_RR1.0_CB1_VOL_RV12_N20_O30              +0.041  +0.043   443 0.542 VOL_RV12_N20             
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5                        +0.039  +0.042  1170 0.557 ORB_G5                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4                        +0.038  +0.042  1209 0.558 ORB_G4                   
M2K_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O15              +0.038  +0.041  1157 0.559 NO_FILTER                
M2K_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15_S075            +0.036  +0.044  1178 0.487 ORB_G4                   
M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8                        +0.033  +0.035   744 0.547 ORB_G8                   
MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VOL_RV12_N20_O15           +0.374  +0.426   142 0.711 VOL_RV12_N20             
MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR70_VOL                  +0.360  +0.414   208 0.712 ATR70_VOL                
MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_O15            +0.360  +0.404    94 0.702 X_MES_ATR60              
MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR70_VOL_S075             +0.358  +0.336   142 0.521 ATR70_VOL                
MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR70_VOL                  +0.355  +0.298   203 0.567 ATR70_VOL                
MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR70_VOL                  +0.354  +0.247   199 0.472 ATR70_VOL                
MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR70_VOL_S075             +0.346  +0.438   159 0.660 ATR70_VOL                
MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR70_VOL                  +0.328  +0.277   136 0.559 ATR70_VOL                
MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR70_VOL                  +0.326  +0.369   154 0.695 ATR70_VOL                
MNQ_BRISBANE_1025_E2_RR4.0_CB1_ATR70_VOL                 +0.322  +0.157   243 0.296 ATR70_VOL                
MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR70_VOL_S075             +0.320  +0.249   129 0.411 ATR70_VOL                

## 6. QUESTIONS FOR COLD EVALUATOR

1. Is there a real edge in ORB breakout on micro futures? Show your work.
2. Do the BH FDR survivors represent genuine signal or grid-search artifact?
3. The G4 filter converts MGC from -0.35R to +0.39R on N=129. Justified or overfit?
4. The 11 MNQ validated strategies use composite filters (ATR70_VOL, X_MES_ATR).
   Their ExpR is 0.32-0.37. A null test is determining the MNQ noise ceiling.
   What threshold would YOU use to distinguish signal from noise?
5. MNQ effect size is +0.12R per trade (~$5-15). Viable on $50K prop, 1 contract?
6. What alternative signals would you test given only 1m OHLCV futures bar data?
7. Are you confident enough in any of these numbers to risk real money? Why/why not?
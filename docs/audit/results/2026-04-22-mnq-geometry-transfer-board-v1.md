# 2026-04-22 MNQ Geometry Transfer Board v1

Read-only transfer board for already-proven MNQ prior-day geometry families.

- Structural start: 2020-01-01 (matches discovery WF start)
- Holdout split: IS < 2026-01-01, OOS >= 2026-01-01
- K_family: 18
- Solved lanes are included for context but should not be mined further.

| Lane | Family | Solved? | N_on_IS | ExpR_on_IS | ExpR_off_IS | Delta_IS | t | BH | N_on_OOS | Delta_OOS | OOS sign |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| COMEX_SETTLE RR1.0 long | PD_CLEAR_LONG | no | 303 | +0.1841 | +0.0137 | +0.1704 | +2.60 | 0.0243 | 15 | +0.2336 | True |
| COMEX_SETTLE RR1.0 long | PD_GO_LONG | no | 390 | +0.1383 | +0.0213 | +0.1171 | +1.82 | 0.1569 | 17 | +0.2189 | True |
| EUROPE_FLOW RR1.0 long | PD_GO_LONG | no | 211 | +0.1052 | +0.0104 | +0.0948 | +1.33 | 0.3334 | 10 | +0.3299 | True |
| EUROPE_FLOW RR1.0 long | PD_DISPLACE_LONG | no | 150 | +0.1014 | +0.0208 | +0.0806 | +1.00 | 0.5201 | 5 | +0.7252 | True |
| NYSE_OPEN RR1.5 long | PD_CLEAR_LONG | no | 180 | +0.1700 | +0.0823 | +0.0877 | +0.85 | 0.5968 | 7 | +0.7023 | True |
| CME_PRECLOSE RR1.0 long | PD_GO_LONG | no | 370 | +0.1524 | +0.1915 | -0.0390 | -0.56 | 0.7380 | 13 | -0.2805 | True |
| NYSE_OPEN RR1.5 long | PD_GO_LONG | no | 284 | +0.1254 | +0.0900 | +0.0355 | +0.39 | 0.7841 | 10 | +0.7687 | True |
| CME_PRECLOSE RR1.0 long | PD_DISPLACE_LONG | no | 205 | +0.0938 | +0.2035 | -0.1097 | -1.44 | 0.3036 | 6 | +0.2811 | False |
| EUROPE_FLOW RR1.0 long | PD_CLEAR_LONG | no | 99 | +0.0959 | +0.0279 | +0.0680 | +0.71 | 0.6645 | 6 | -0.0158 | False |
| NYSE_OPEN RR1.5 long | PD_DISPLACE_LONG | no | 178 | +0.0654 | +0.1155 | -0.0501 | -0.48 | 0.7556 | 5 | +0.3059 | False |
| CME_PRECLOSE RR1.0 long | PD_CLEAR_LONG | no | 288 | +0.1802 | +0.1620 | +0.0182 | +0.26 | 0.8420 | 10 | -0.1387 | False |
| COMEX_SETTLE RR1.0 long | PD_DISPLACE_LONG | no | 208 | +0.0873 | +0.0764 | +0.0108 | +0.15 | 0.8842 | 8 | -0.0539 | False |
| US_DATA_1000 RR1.0 long | PD_GO_LONG | yes | 324 | +0.1934 | -0.0711 | +0.2645 | +3.92 | 0.0018 | 15 | +0.6169 | True |
| US_DATA_1000 RR1.0 long | PD_DISPLACE_LONG | yes | 192 | +0.2396 | -0.0286 | +0.2682 | +3.49 | 0.0034 | 10 | +0.3188 | True |
| US_DATA_1000 RR1.0 long | PD_CLEAR_LONG | yes | 211 | +0.2270 | -0.0327 | +0.2596 | +3.48 | 0.0034 | 13 | +0.3967 | True |
| US_DATA_1000 RR1.5 long | PD_GO_LONG | yes | 321 | +0.2222 | -0.0661 | +0.2883 | +3.36 | 0.0037 | 15 | +0.5663 | True |
| US_DATA_1000 RR1.5 long | PD_CLEAR_LONG | yes | 209 | +0.2728 | -0.0287 | +0.3015 | +3.14 | 0.0067 | 13 | +0.2295 | True |
| US_DATA_1000 RR1.5 long | PD_DISPLACE_LONG | yes | 190 | +0.2500 | -0.0120 | +0.2620 | +2.63 | 0.0243 | 10 | +0.3491 | True |

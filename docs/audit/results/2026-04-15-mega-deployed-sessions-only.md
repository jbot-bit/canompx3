# Mega — Deployed Sessions Only

**Date:** 2026-04-15
**Scope:** MNQ only, 6 deployed-lane (session, aperture) pairs, RR {1.0,1.5,2.0}, both directions, all 14 level features
**Total cells scanned:** 432
**Rows loaded:** 31,474

## Deployed-Match Survivors (|t| >= 3.0 AND dir_match)

These are candidate OVERLAY filters for the exact (session, aperture, RR) of a live lane.

| Session | Apt | RR | Dir | Signal | N_on | Fire% | ExpR_on | ExpR_off | Δ_IS | Δ_OOS | t | p | BH_pass |
|---------|-----|----|----|--------|------|-------|---------|----------|------|-------|---|---|---------|

## BH-FDR Survivors at Deployed-Match (q=0.05)

| Session | Apt | RR | Dir | Signal | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p | BH_crit |
|---------|-----|----|----|--------|------|-------|---------|------|-------|---|---|---------|
| US_DATA_1000 | O5 | 1.5 | long | F5_BELOW_PDL | 135 | 15.6% | +0.402 | +0.405 | +nan | +3.65 | 0.00034 | 0.00035 |
| US_DATA_1000 | O5 | 1.5 | long | F6_INSIDE_PDR | 503 | 58.1% | -0.059 | -0.285 | +nan | -3.52 | 0.00046 | 0.00058 |

## All Cells (ranked by |t|)

| Session | Apt | RR | Dir | Signal | N_on | ExpR_on | Δ_IS | Δ_OOS | t | p | deployed | dir_match |
|---------|-----|----|----|--------|------|---------|------|-------|---|---|----------|-----------|
| US_DATA_1000 | O5 | 1.0 | long | F5_BELOW_PDL | 136 | +0.326 | +0.337 | +nan | +4.02 | 0.0001 | . | . |
| US_DATA_1000 | O5 | 1.0 | long | F3_NEAR_PIVOT_50 | 618 | -0.034 | -0.252 | +nan | -3.69 | 0.0002 | . | . |
| US_DATA_1000 | O5 | 1.5 | long | F5_BELOW_PDL | 135 | +0.402 | +0.405 | +nan | +3.65 | 0.0003 | Y | . |
| NYSE_OPEN | O5 | 1.5 | short | F5_BELOW_PDL | 117 | +0.469 | +0.424 | +nan | +3.59 | 0.0004 | . | . |
| US_DATA_1000 | O5 | 1.5 | long | F6_INSIDE_PDR | 503 | -0.059 | -0.285 | +nan | -3.52 | 0.0005 | Y | . |
| NYSE_OPEN | O5 | 1.0 | short | F5_BELOW_PDL | 120 | +0.353 | +0.300 | +nan | +3.39 | 0.0009 | Y | . |
| US_DATA_1000 | O5 | 2.0 | long | F6_INSIDE_PDR | 485 | -0.111 | -0.309 | +nan | -3.24 | 0.0012 | . | . |
| COMEX_SETTLE | O5 | 1.0 | long | F6_INSIDE_PDR | 433 | -0.030 | -0.195 | +nan | -3.24 | 0.0013 | . | . |
| US_DATA_1000 | O5 | 1.5 | long | F3_NEAR_PIVOT_50 | 605 | -0.025 | -0.284 | +nan | -3.24 | 0.0013 | Y | . |
| US_DATA_1000 | O5 | 1.0 | long | F6_INSIDE_PDR | 513 | -0.043 | -0.202 | +nan | -3.15 | 0.0017 | . | . |
| EUROPE_FLOW | O5 | 2.0 | short | F3_NEAR_PIVOT_50 | 736 | +0.009 | -0.367 | +nan | -2.86 | 0.0048 | . | . |
| US_DATA_1000 | O5 | 2.0 | long | F3_NEAR_PIVOT_50 | 587 | -0.070 | -0.298 | +nan | -2.86 | 0.0045 | . | . |
| NYSE_OPEN | O5 | 1.5 | long | F3_NEAR_PIVOT_15 | 216 | -0.109 | -0.261 | +nan | -2.81 | 0.0053 | . | . |
| US_DATA_1000 | O5 | 2.0 | long | F5_BELOW_PDL | 133 | +0.338 | +0.377 | +nan | +2.80 | 0.0056 | . | . |
| NYSE_OPEN | O5 | 1.0 | long | F4_ABOVE_PDH | 225 | +0.214 | +0.203 | +nan | +2.80 | 0.0054 | Y | . |
| SINGAPORE_OPEN | O30 | 1.0 | long | F5_BELOW_PDL | 71 | -0.232 | -0.314 | +nan | -2.73 | 0.0078 | . | . |
| SINGAPORE_OPEN | O30 | 1.0 | long | F6_INSIDE_PDR | 735 | +0.098 | +0.212 | +nan | +2.69 | 0.0077 | . | . |
| EUROPE_FLOW | O5 | 1.5 | short | F3_NEAR_PIVOT_50 | 736 | -0.019 | -0.286 | +nan | -2.68 | 0.0081 | Y | . |
| NYSE_OPEN | O5 | 1.0 | long | F6_INSIDE_PDR | 505 | -0.008 | -0.176 | +nan | -2.67 | 0.0078 | Y | . |
| NYSE_OPEN | O5 | 2.0 | short | F5_BELOW_PDL | 115 | +0.415 | +0.386 | +nan | +2.66 | 0.0088 | . | . |
| US_DATA_1000 | O5 | 2.0 | long | F3_NEAR_PIVOT_15 | 208 | -0.183 | -0.269 | +nan | -2.57 | 0.0106 | . | . |
| NYSE_OPEN | O5 | 1.5 | long | F6_INSIDE_PDR | 492 | -0.000 | -0.207 | +nan | -2.44 | 0.0148 | . | . |
| TOKYO_OPEN | O5 | 1.5 | long | F1_NEAR_PDH_50 | 573 | +0.008 | -0.194 | +nan | -2.42 | 0.0156 | Y | . |
| COMEX_SETTLE | O5 | 1.0 | long | F4_ABOVE_PDH | 269 | +0.174 | +0.151 | +nan | +2.38 | 0.0177 | . | . |
| NYSE_OPEN | O5 | 1.5 | short | F6_INSIDE_PDR | 523 | +0.031 | -0.206 | +nan | -2.36 | 0.0188 | . | . |
| NYSE_OPEN | O5 | 1.5 | long | F4_ABOVE_PDH | 216 | +0.246 | +0.220 | +nan | +2.33 | 0.0206 | . | . |
| US_DATA_1000 | O5 | 1.0 | long | F1_NEAR_PDH_50 | 508 | -0.022 | -0.148 | +nan | -2.31 | 0.0212 | . | . |
| COMEX_SETTLE | O5 | 1.0 | long | F1_NEAR_PDH_15 | 176 | -0.069 | -0.172 | +nan | -2.28 | 0.0232 | . | . |
| COMEX_SETTLE | O5 | 1.0 | short | F2_NEAR_PDL_15 | 96 | -0.140 | -0.227 | +nan | -2.26 | 0.0256 | . | . |
| COMEX_SETTLE | O5 | 1.5 | long | F6_INSIDE_PDR | 429 | -0.010 | -0.171 | +nan | -2.23 | 0.0262 | Y | . |
| NYSE_OPEN | O5 | 1.0 | short | F6_INSIDE_PDR | 531 | +0.042 | -0.149 | +nan | -2.18 | 0.0293 | Y | . |
| TOKYO_OPEN | O5 | 1.5 | short | F1_NEAR_PDH_30 | 455 | -0.008 | -0.163 | +nan | -2.17 | 0.0304 | Y | . |
| US_DATA_1000 | O5 | 2.0 | long | F3_NEAR_PIVOT_30 | 398 | -0.085 | -0.199 | +nan | -2.13 | 0.0335 | . | . |
| TOKYO_OPEN | O5 | 1.0 | short | F1_NEAR_PDH_30 | 455 | +0.004 | -0.124 | +nan | -2.12 | 0.0346 | . | . |
| COMEX_SETTLE | O5 | 1.5 | short | F4_ABOVE_PDH | 224 | -0.076 | -0.188 | +nan | -2.11 | 0.0354 | Y | . |
| COMEX_SETTLE | O5 | 2.0 | long | F3_NEAR_PIVOT_30 | 333 | -0.030 | -0.196 | +nan | -2.11 | 0.0353 | . | . |
| COMEX_SETTLE | O5 | 2.0 | long | F6_INSIDE_PDR | 422 | -0.006 | -0.189 | +nan | -2.07 | 0.0388 | . | . |
| NYSE_OPEN | O5 | 1.0 | long | F3_NEAR_PIVOT_15 | 219 | -0.051 | -0.155 | +nan | -2.06 | 0.0397 | Y | . |
| NYSE_OPEN | O5 | 1.0 | short | F3_NEAR_PIVOT_30 | 448 | +0.033 | -0.136 | +nan | -2.06 | 0.0393 | Y | . |
| NYSE_OPEN | O5 | 1.0 | short | F1_NEAR_PDH_50 | 494 | +0.040 | -0.137 | +nan | -2.04 | 0.0414 | Y | . |
| SINGAPORE_OPEN | O30 | 1.5 | long | F5_BELOW_PDL | 71 | -0.141 | -0.287 | +nan | -2.03 | 0.0451 | Y | . |
| SINGAPORE_OPEN | O30 | 1.5 | long | F3_NEAR_PIVOT_50 | 814 | +0.150 | +0.256 | +nan | +2.02 | 0.0453 | Y | . |
| NYSE_OPEN | O5 | 2.0 | long | F6_INSIDE_PDR | 479 | +0.001 | -0.204 | +nan | -2.01 | 0.0446 | . | . |
| NYSE_OPEN | O5 | 2.0 | short | F6_INSIDE_PDR | 510 | +0.009 | -0.208 | +nan | -2.00 | 0.0455 | . | . |
| COMEX_SETTLE | O5 | 1.0 | long | F3_NEAR_PIVOT_30 | 340 | -0.007 | -0.124 | +nan | -2.00 | 0.0459 | . | . |
| NYSE_OPEN | O5 | 1.0 | short | F3_NEAR_PIVOT_50 | 634 | +0.060 | -0.152 | +nan | -1.99 | 0.0470 | Y | . |
| TOKYO_OPEN | O5 | 1.0 | long | F1_NEAR_PDH_50 | 573 | -0.011 | -0.125 | +nan | -1.98 | 0.0482 | . | . |
| US_DATA_1000 | O5 | 1.0 | short | F2_NEAR_PDL_30 | 203 | +0.021 | -0.151 | +nan | -1.97 | 0.0492 | . | . |
| NYSE_OPEN | O5 | 2.0 | long | F3_NEAR_PIVOT_15 | 212 | -0.075 | -0.215 | +nan | -1.97 | 0.0501 | . | . |
| NYSE_OPEN | O5 | 1.5 | short | F3_NEAR_PIVOT_50 | 622 | +0.059 | -0.193 | +nan | -1.96 | 0.0513 | . | . |
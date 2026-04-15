# MEGA EXPLORATION — Prior-Day Features × Full Scope

**Rows IS/OOS:** 331860/15436
**Cells analyzed:** 6252

**EXPLORATORY.** K exceeds Bailey MinBTL — multiple K framings reported.
Primary gate: cluster-SE t (p_cluster). Heat classification by absolute cluster t.

- **HOT (|t_cluster| >= 4.0 AND holdout dir match):** 4
- **WARM (|t_cluster| >= 3.0 AND holdout dir match):** 26
- **LUKEWARM (|t_cluster| >= 2.5):** 151
- BH-FDR global (K=6252, q=0.05) survivors: 5
- BH-FDR per-feature family survivors: 12
- BH-FDR per-(instr,session) family survivors: 32

## Top 50 by |cluster-t|

| flag | instr | session | apt | rr | dir | signal | N_on | N_off | ExpR_on | ExpR_off | Δ_IS | Δ_OOS | t_cl | p_cl | p_logit | dir_match | OOS/IS | BH_glob | BH_feat | BH_IS |
|------|-------|---------|-----|----|-----|--------|------|-------|---------|----------|------|-------|------|------|---------|-----------|--------|---------|---------|-------|
| HOT | MES | NYSE_OPEN | O30 | 2.0 | short | F2_NEAR_PDL_15 | 69 | 472 | -0.670 | +0.064 | -0.734 | -0.847 | -5.83 | 0.0000 | 0.0001 | Y | +1.15 | Y | Y | Y |
| HOT | MNQ | NYSE_CLOSE | O5 | 1.5 | long | F3_NEAR_PIVOT_15 | 39 | 251 | -0.642 | +0.117 | -0.759 | -1.501 | -4.91 | 0.0000 | 0.0002 | Y | +1.98 | Y | Y | Y |
| HOT | MGC | SINGAPORE_OPEN | O15 | 1.5 | long | F3_NEAR_PIVOT_15 | 226 | 238 | -0.294 | +0.107 | -0.401 | -0.047 | -4.27 | 0.0000 | 0.0001 | Y | +0.12 | Y | Y | Y |
| HOT | MNQ | US_DATA_1000 | O5 | 1.0 | long | F5_BELOW_PDL | 136 | 745 | +0.326 | -0.011 | +0.337 | +0.080 | +4.24 | 0.0000 | 0.0001 | Y | +0.24 | Y | Y | Y |
| LUKEWARM | MES | CME_REOPEN | O15 | 1.0 | long | F1_NEAR_PDH_15 | 72 | 213 | +0.202 | -0.186 | +0.388 | -0.348 | +4.17 | 0.0000 | 0.0001 | . | -0.90 | Y | Y | Y |
| LUKEWARM | MNQ | US_DATA_1000 | O5 | 1.5 | long | F5_BELOW_PDL | 135 | 731 | +0.402 | -0.002 | +0.405 | -0.003 | +3.95 | 0.0001 | 0.0001 | . | -0.01 | . | Y | Y |
| LUKEWARM | MES | CME_REOPEN | O15 | 1.5 | long | F1_NEAR_PDH_15 | 65 | 190 | +0.207 | -0.353 | +0.559 | -0.962 | +3.95 | 0.0001 | 0.0001 | . | -1.72 | . | Y | Y |
| WARM | MNQ | NYSE_CLOSE | O5 | 2.0 | long | F3_NEAR_PIVOT_15 | 37 | 202 | -0.700 | -0.041 | -0.659 | -1.255 | -3.91 | 0.0001 | 0.0057 | Y | +1.90 | . | Y | Y |
| WARM | MNQ | US_DATA_1000 | O5 | 1.0 | long | F3_NEAR_PIVOT_50 | 618 | 263 | -0.034 | +0.218 | -0.252 | -0.108 | -3.86 | 0.0001 | 0.0003 | Y | +0.43 | . | . | Y |
| WARM | MES | NYSE_OPEN | O30 | 1.5 | short | F2_NEAR_PDL_15 | 77 | 528 | -0.383 | +0.112 | -0.495 | -1.007 | -3.84 | 0.0001 | 0.0006 | Y | +2.04 | . | Y | Y |
| LUKEWARM | MNQ | CME_REOPEN | O30 | 1.0 | short | F1_NEAR_PDH_30 | 82 | 81 | -0.157 | +0.356 | -0.512 | +0.657 | -3.84 | 0.0001 | 0.0003 | . | -1.28 | . | . | Y |
| WARM | MES | NYSE_CLOSE | O5 | 1.5 | short | F1_NEAR_PDH_15 | 45 | 223 | -0.716 | -0.211 | -0.505 | -1.100 | -3.77 | 0.0002 | 0.0026 | Y | +2.18 | . | Y | Y |
| LUKEWARM | MGC | TOKYO_OPEN | O30 | 2.0 | long | F4_ABOVE_PDH | 30 | 453 | +0.794 | -0.100 | +0.894 | -0.168 | +3.69 | 0.0002 | 0.0014 | . | -0.19 | . | . | Y |
| LUKEWARM | MES | LONDON_METALS | O5 | 1.0 | short | F5_BELOW_PDL | 86 | 756 | +0.178 | -0.151 | +0.329 | -0.453 | +3.67 | 0.0002 | 0.0025 | . | -1.38 | . | Y | . |
| LUKEWARM | MES | NYSE_OPEN | O15 | 1.0 | short | F2_NEAR_PDL_50 | 356 | 447 | -0.122 | +0.123 | -0.245 | +0.050 | -3.64 | 0.0003 | 0.0003 | . | -0.21 | . | . | Y |
| LUKEWARM | MNQ | NYSE_OPEN | O5 | 1.5 | short | F5_BELOW_PDL | 117 | 702 | +0.469 | +0.045 | +0.424 | -0.126 | +3.62 | 0.0003 | 0.0005 | . | -0.30 | . | Y | . |
| WARM | MES | TOKYO_OPEN | O5 | 1.0 | long | F3_NEAR_PIVOT_30 | 630 | 214 | -0.176 | +0.064 | -0.241 | -0.292 | -3.58 | 0.0003 | 0.0022 | Y | +1.22 | . | . | . |
| WARM | MNQ | COMEX_SETTLE | O5 | 1.0 | long | F6_INSIDE_PDR | 433 | 443 | -0.030 | +0.165 | -0.195 | -0.320 | -3.57 | 0.0004 | 0.0004 | Y | +1.64 | . | . | . |
| LUKEWARM | MNQ | US_DATA_1000 | O5 | 1.5 | long | F6_INSIDE_PDR | 503 | 363 | -0.059 | +0.226 | -0.285 | +0.279 | -3.53 | 0.0004 | 0.0004 | . | -0.98 | . | . | Y |
| WARM | MGC | SINGAPORE_OPEN | O15 | 2.0 | long | F3_NEAR_PIVOT_15 | 225 | 236 | -0.254 | +0.107 | -0.361 | -0.078 | -3.50 | 0.0005 | 0.0011 | Y | +0.22 | . | . | Y |
| WARM | MNQ | NYSE_CLOSE | O5 | 2.0 | long | F3_NEAR_PIVOT_30 | 70 | 169 | -0.535 | +0.020 | -0.555 | -0.152 | -3.48 | 0.0005 | 0.0026 | Y | +0.27 | . | . | Y |
| LUKEWARM | MNQ | US_DATA_1000 | O5 | 1.5 | long | F3_NEAR_PIVOT_50 | 605 | 261 | -0.025 | +0.259 | -0.284 | +0.079 | -3.46 | 0.0005 | 0.0007 | . | -0.28 | . | . | Y |
| WARM | MNQ | US_DATA_830 | O5 | 1.0 | long | F3_NEAR_PIVOT_50 | 642 | 184 | +0.064 | -0.209 | +0.274 | +0.002 | +3.39 | 0.0007 | 0.0002 | Y | +0.01 | . | . | . |
| WARM | MNQ | NYSE_OPEN | O30 | 2.0 | short | F2_NEAR_PDL_30 | 127 | 326 | -0.380 | +0.078 | -0.457 | -1.126 | -3.38 | 0.0007 | 0.0021 | Y | +2.46 | . | . | . |
| LUKEWARM | MNQ | NYSE_OPEN | O5 | 1.0 | short | F5_BELOW_PDL | 120 | 715 | +0.353 | +0.053 | +0.300 | -0.168 | +3.38 | 0.0007 | 0.0017 | . | -0.56 | . | . | . |
| LUKEWARM | MES | NYSE_OPEN | O5 | 1.0 | long | F2_NEAR_PDL_30 | 226 | 631 | -0.179 | +0.057 | -0.236 | +0.018 | -3.37 | 0.0008 | 0.0010 | . | -0.08 | . | . | Y |
| WARM | MES | COMEX_SETTLE | O5 | 1.5 | long | F3_NEAR_PIVOT_50 | 521 | 365 | -0.201 | +0.043 | -0.244 | -0.344 | -3.36 | 0.0008 | 0.0014 | Y | +1.41 | . | . | . |
| LUKEWARM | MES | CME_REOPEN | O5 | 2.0 | long | F1_NEAR_PDH_30 | 146 | 190 | -0.082 | -0.414 | +0.332 | -0.802 | +3.36 | 0.0008 | 0.0006 | . | -2.42 | . | . | Y |
| LUKEWARM | MES | NYSE_OPEN | O5 | 2.0 | long | F2_NEAR_PDL_30 | 212 | 613 | -0.206 | +0.133 | -0.340 | +0.047 | -3.35 | 0.0008 | 0.0015 | . | -0.14 | . | . | Y |
| LUKEWARM | MES | NYSE_OPEN | O15 | 1.5 | short | F2_NEAR_PDL_50 | 339 | 427 | -0.120 | +0.163 | -0.283 | +0.121 | -3.34 | 0.0008 | 0.0012 | . | -0.43 | . | . | Y |
| LUKEWARM | MES | US_DATA_1000 | O15 | 2.0 | short | F2_NEAR_PDL_15 | 99 | 578 | -0.321 | +0.119 | -0.441 | +0.080 | -3.31 | 0.0009 | 0.0024 | . | -0.18 | . | . | . |
| WARM | MES | CME_REOPEN | O5 | 1.5 | long | F1_NEAR_PDH_15 | 98 | 265 | +0.059 | -0.286 | +0.345 | +0.134 | +3.30 | 0.0010 | 0.0006 | Y | +0.39 | . | . | Y |
| WARM | MES | CME_REOPEN | O5 | 2.0 | long | F1_NEAR_PDH_50 | 199 | 137 | -0.150 | -0.443 | +0.293 | +0.117 | +3.30 | 0.0010 | 0.0012 | Y | +0.40 | . | . | Y |
| LUKEWARM | MNQ | CME_PRECLOSE | O5 | 1.5 | long | F1_NEAR_PDH_30 | 224 | 433 | +0.355 | +0.038 | +0.317 | -0.154 | +3.29 | 0.0010 | 0.0006 | . | -0.49 | . | . | . |
| WARM | MNQ | CME_REOPEN | O5 | 2.0 | long | F3_NEAR_PIVOT_50 | 281 | 75 | -0.003 | -0.489 | +0.486 | +0.474 | +3.28 | 0.0010 | 0.0042 | Y | +0.97 | . | . | . |
| LUKEWARM | MNQ | US_DATA_1000 | O5 | 2.0 | long | F6_INSIDE_PDR | 485 | 358 | -0.111 | +0.198 | -0.309 | +0.190 | -3.27 | 0.0011 | 0.0011 | . | -0.61 | . | . | Y |
| WARM | MES | NYSE_OPEN | O15 | 2.0 | short | F2_NEAR_PDL_50 | 322 | 396 | -0.183 | +0.143 | -0.326 | -0.022 | -3.26 | 0.0011 | 0.0016 | Y | +0.07 | . | . | Y |
| LUKEWARM | MES | CME_REOPEN | O15 | 1.5 | long | F1_NEAR_PDH_30 | 117 | 138 | +0.000 | -0.388 | +0.388 | -1.074 | +3.25 | 0.0011 | 0.0010 | . | -2.77 | . | . | Y |
| WARM | MES | CME_REOPEN | O15 | 2.0 | long | F3_NEAR_PIVOT_50 | 218 | 30 | -0.228 | -0.738 | +0.510 | +0.201 | +3.25 | 0.0012 | 0.0154 | Y | +0.39 | . | . | Y |
| WARM | MES | NYSE_CLOSE | O5 | 1.0 | short | F1_NEAR_PDH_15 | 56 | 294 | -0.457 | -0.057 | -0.400 | -1.101 | -3.25 | 0.0012 | 0.0023 | Y | +2.75 | . | . | . |
| LUKEWARM | MES | US_DATA_830 | O5 | 1.0 | short | F1_NEAR_PDH_50 | 539 | 294 | -0.135 | +0.056 | -0.191 | +0.028 | -3.24 | 0.0012 | 0.0048 | . | -0.15 | . | . | . |
| WARM | MES | NYSE_CLOSE | O5 | 2.0 | short | F1_NEAR_PDH_15 | 44 | 194 | -0.765 | -0.326 | -0.440 | -1.167 | -3.23 | 0.0012 | 0.0106 | Y | +2.65 | . | . | . |
| LUKEWARM | MGC | SINGAPORE_OPEN | O15 | 1.0 | long | F3_NEAR_PIVOT_15 | 226 | 238 | -0.240 | +0.032 | -0.272 | +0.278 | -3.22 | 0.0013 | 0.0027 | . | -1.02 | . | . | . |
| LUKEWARM | MES | US_DATA_1000 | O5 | 1.0 | long | F3_NEAR_PIVOT_50 | 634 | 240 | -0.106 | +0.100 | -0.206 | +0.091 | -3.22 | 0.0013 | 0.0034 | . | -0.44 | . | . | . |
| LUKEWARM | MNQ | CME_REOPEN | O15 | 1.5 | long | F1_NEAR_PDH_15 | 73 | 187 | +0.220 | -0.244 | +0.464 | -0.399 | +3.20 | 0.0014 | 0.0013 | . | -0.86 | . | . | . |
| WARM | MES | NYSE_OPEN | O30 | 1.0 | short | F2_NEAR_PDL_15 | 90 | 603 | -0.236 | +0.103 | -0.340 | -0.914 | -3.20 | 0.0014 | 0.0014 | Y | +2.69 | . | . | Y |
| LUKEWARM | MES | LONDON_METALS | O5 | 1.0 | short | F2_NEAR_PDL_50 | 397 | 445 | -0.021 | -0.204 | +0.183 | -0.262 | +3.17 | 0.0015 | 0.0020 | . | -1.43 | . | . | . |
| LUKEWARM | MNQ | US_DATA_1000 | O5 | 1.0 | long | F6_INSIDE_PDR | 513 | 368 | -0.043 | +0.158 | -0.202 | +0.024 | -3.16 | 0.0016 | 0.0019 | . | -0.12 | . | . | Y |
| WARM | MNQ | BRISBANE_1025 | O15 | 2.0 | short | F3_NEAR_PIVOT_30 | 585 | 228 | -0.124 | +0.186 | -0.310 | -0.061 | -3.15 | 0.0016 | 0.0021 | Y | +0.20 | . | . | . |
| LUKEWARM | MNQ | US_DATA_1000 | O5 | 2.0 | long | F5_BELOW_PDL | 133 | 710 | +0.338 | -0.040 | +0.377 | -0.072 | +3.12 | 0.0018 | 0.0014 | . | -0.19 | . | . | Y |

## HOT cells (|t_cluster| >= 4.0 AND holdout dir match)

- MNQ US_DATA_1000 O5 RR1.0 long F5_BELOW_PDL: t_cl=+4.24 Δ_IS=+0.337 Δ_OOS=+0.080 N_on=136 BH_glob=Y
- MNQ NYSE_CLOSE O5 RR1.5 long F3_NEAR_PIVOT_15: t_cl=-4.91 Δ_IS=-0.759 Δ_OOS=-1.501 N_on=39 BH_glob=Y
- MES NYSE_OPEN O30 RR2.0 short F2_NEAR_PDL_15: t_cl=-5.83 Δ_IS=-0.734 Δ_OOS=-0.847 N_on=69 BH_glob=Y
- MGC SINGAPORE_OPEN O15 RR1.5 long F3_NEAR_PIVOT_15: t_cl=-4.27 Δ_IS=-0.401 Δ_OOS=-0.047 N_on=226 BH_glob=Y

## WARM cells (|t_cluster| >= 3.0 AND holdout dir match)

- MNQ CME_REOPEN O5 RR2.0 long F3_NEAR_PIVOT_50: t_cl=+3.28 Δ_IS=+0.486 Δ_OOS=+0.474 N_on=281 BH_feat=. BH_IS=.
- MNQ US_DATA_830 O5 RR1.0 long F3_NEAR_PIVOT_50: t_cl=+3.39 Δ_IS=+0.274 Δ_OOS=+0.002 N_on=642 BH_feat=. BH_IS=.
- MNQ US_DATA_830 O5 RR2.0 long F3_NEAR_PIVOT_50: t_cl=+3.03 Δ_IS=+0.340 Δ_OOS=+0.119 N_on=623 BH_feat=. BH_IS=.
- MNQ NYSE_OPEN O30 RR2.0 short F2_NEAR_PDL_30: t_cl=-3.38 Δ_IS=-0.457 Δ_OOS=-1.126 N_on=127 BH_feat=. BH_IS=.
- MNQ US_DATA_1000 O5 RR1.0 long F3_NEAR_PIVOT_50: t_cl=-3.86 Δ_IS=-0.252 Δ_OOS=-0.108 N_on=618 BH_feat=. BH_IS=Y
- MNQ US_DATA_1000 O5 RR2.0 long F3_NEAR_PIVOT_50: t_cl=-3.06 Δ_IS=-0.298 Δ_OOS=-0.100 N_on=587 BH_feat=. BH_IS=.
- MNQ COMEX_SETTLE O5 RR1.0 long F6_INSIDE_PDR: t_cl=-3.57 Δ_IS=-0.195 Δ_OOS=-0.320 N_on=433 BH_feat=. BH_IS=.
- MNQ NYSE_CLOSE O5 RR1.0 long F3_NEAR_PIVOT_15: t_cl=-3.08 Δ_IS=-0.414 Δ_OOS=-0.561 N_on=55 BH_feat=. BH_IS=.
- MNQ NYSE_CLOSE O5 RR2.0 long F3_NEAR_PIVOT_15: t_cl=-3.91 Δ_IS=-0.659 Δ_OOS=-1.255 N_on=37 BH_feat=Y BH_IS=Y
- MNQ NYSE_CLOSE O5 RR2.0 long F3_NEAR_PIVOT_30: t_cl=-3.48 Δ_IS=-0.555 Δ_OOS=-0.152 N_on=70 BH_feat=. BH_IS=Y
- MNQ BRISBANE_1025 O15 RR1.5 short F3_NEAR_PIVOT_30: t_cl=-3.04 Δ_IS=-0.243 Δ_OOS=-0.259 N_on=585 BH_feat=. BH_IS=.
- MNQ BRISBANE_1025 O15 RR2.0 short F3_NEAR_PIVOT_30: t_cl=-3.15 Δ_IS=-0.310 Δ_OOS=-0.061 N_on=585 BH_feat=. BH_IS=.
- MES CME_REOPEN O5 RR1.5 long F1_NEAR_PDH_15: t_cl=+3.30 Δ_IS=+0.345 Δ_OOS=+0.134 N_on=98 BH_feat=. BH_IS=Y
- MES CME_REOPEN O5 RR2.0 long F1_NEAR_PDH_50: t_cl=+3.30 Δ_IS=+0.293 Δ_OOS=+0.117 N_on=199 BH_feat=. BH_IS=Y
- MES CME_REOPEN O15 RR2.0 long F3_NEAR_PIVOT_50: t_cl=+3.25 Δ_IS=+0.510 Δ_OOS=+0.201 N_on=218 BH_feat=. BH_IS=Y
- MES TOKYO_OPEN O5 RR1.0 long F3_NEAR_PIVOT_30: t_cl=-3.58 Δ_IS=-0.241 Δ_OOS=-0.292 N_on=630 BH_feat=. BH_IS=.
- MES US_DATA_830 O30 RR1.0 short F2_NEAR_PDL_30: t_cl=-3.11 Δ_IS=-0.213 Δ_OOS=-0.099 N_on=241 BH_feat=. BH_IS=.
- MES NYSE_OPEN O15 RR2.0 short F2_NEAR_PDL_50: t_cl=-3.26 Δ_IS=-0.326 Δ_OOS=-0.022 N_on=322 BH_feat=. BH_IS=Y
- MES NYSE_OPEN O30 RR1.0 short F2_NEAR_PDL_15: t_cl=-3.20 Δ_IS=-0.340 Δ_OOS=-0.914 N_on=90 BH_feat=. BH_IS=Y
- MES NYSE_OPEN O30 RR1.5 short F2_NEAR_PDL_15: t_cl=-3.84 Δ_IS=-0.495 Δ_OOS=-1.007 N_on=77 BH_feat=Y BH_IS=Y
- MES US_DATA_1000 O15 RR1.0 short F2_NEAR_PDL_30: t_cl=-3.07 Δ_IS=-0.227 Δ_OOS=-0.251 N_on=211 BH_feat=. BH_IS=.
- MES COMEX_SETTLE O5 RR1.5 long F3_NEAR_PIVOT_50: t_cl=-3.36 Δ_IS=-0.244 Δ_OOS=-0.344 N_on=521 BH_feat=. BH_IS=.
- MES NYSE_CLOSE O5 RR1.0 short F1_NEAR_PDH_15: t_cl=-3.25 Δ_IS=-0.400 Δ_OOS=-1.101 N_on=56 BH_feat=. BH_IS=.
- MES NYSE_CLOSE O5 RR1.5 short F1_NEAR_PDH_15: t_cl=-3.77 Δ_IS=-0.505 Δ_OOS=-1.100 N_on=45 BH_feat=Y BH_IS=Y
- MES NYSE_CLOSE O5 RR2.0 short F1_NEAR_PDH_15: t_cl=-3.23 Δ_IS=-0.440 Δ_OOS=-1.167 N_on=44 BH_feat=. BH_IS=.
- MGC SINGAPORE_OPEN O15 RR2.0 long F3_NEAR_PIVOT_15: t_cl=-3.50 Δ_IS=-0.361 Δ_OOS=-0.078 N_on=225 BH_feat=. BH_IS=Y
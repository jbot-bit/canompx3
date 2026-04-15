# MGC Level Scan — Previous-Day Levels + S/R

**Date:** 2026-04-15
**Instrument:** MGC only
**Scope:** 12 sessions × 3 apertures × 3 RRs × 2 directions × 14 level features
**Classification:** EXPLORATORY (no validated_setups writes)
**Data:** MGC 2022-06-13 to 2026-04-10 (~3.8 years)

**Total cells:** 1775
**Trustworthy:** 1760 (not extreme-fire, not arithmetic-only)
**Strict survivors** (|t|>=3 + dir_match + N>=50): 2

## BH-FDR at each K framing
- K_global (1775): 0 pass
- K_family (avg K~1167): 1 pass
- K_lane (avg K~22): 9 pass
- K_session (avg K~202): 1 pass
- Promising (|t|>=2.5 + dir_match): 12

## Strict Survivors

| Session | Apt | RR | Dir | Feature | Family | N_on | Fire% | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p | BH_g | BH_f | BH_l |
|---------|-----|----|----|---------|--------|------|-------|---------|------|------|-------|---|---|------|------|------|
| SINGAPORE_OPEN | O15 | 1.5 | long | F3_NEAR_PIVOT_15 | level_proximity | 226 | 48.5% | -0.294 | -0.176 | -0.401 | -0.047 | -4.20 | 0.0000 | . | Y | Y |
| SINGAPORE_OPEN | O15 | 2.0 | long | F3_NEAR_PIVOT_15 | level_proximity | 225 | 48.6% | -0.254 | -0.130 | -0.361 | -0.078 | -3.22 | 0.0014 | . | . | Y |

## Promising cells (|t|>=2.5 + dir_match) — T0-T8 candidates

| Session | Apt | RR | Dir | Feature | N_on | ExpR_on | Δ_IS | Δ_OOS | t | p |
|---------|-----|----|----|---------|------|---------|------|-------|---|---|
| SINGAPORE_OPEN | O15 | 1.5 | long | F3_NEAR_PIVOT_15 | 226 | -0.294 | -0.401 | -0.047 | -4.20 | 0.0000 |
| SINGAPORE_OPEN | O15 | 2.0 | long | F3_NEAR_PIVOT_15 | 225 | -0.254 | -0.361 | -0.078 | -3.22 | 0.0014 |
| NYSE_OPEN | O15 | 1.0 | short | F6_INSIDE_PDR | 239 | +0.103 | +0.269 | +0.190 | +2.96 | 0.0033 |
| TOKYO_OPEN | O30 | 2.0 | long | F6_INSIDE_PDR | 433 | -0.104 | -0.572 | -0.239 | -2.92 | 0.0050 |
| LONDON_METALS | O30 | 1.5 | long | F2_NEAR_PDL_30 | 155 | +0.204 | +0.311 | +1.046 | +2.87 | 0.0044 |
| LONDON_METALS | O30 | 2.0 | long | F2_NEAR_PDL_30 | 152 | +0.246 | +0.362 | +0.019 | +2.79 | 0.0056 |
| TOKYO_OPEN | O15 | 1.0 | long | F3_NEAR_PIVOT_50 | 441 | -0.077 | -0.401 | -0.304 | -2.79 | 0.0087 |
| EUROPE_FLOW | O30 | 1.5 | short | F2_NEAR_PDL_30 | 151 | +0.117 | +0.304 | +0.612 | +2.79 | 0.0057 |
| SINGAPORE_OPEN | O15 | 1.0 | short | F2_NEAR_PDL_50 | 250 | -0.043 | +0.223 | +0.455 | +2.78 | 0.0057 |
| NYSE_OPEN | O15 | 1.0 | short | F3_NEAR_PIVOT_15 | 104 | +0.195 | +0.272 | +0.313 | +2.74 | 0.0067 |
| TOKYO_OPEN | O15 | 1.0 | long | F3_NEAR_PIVOT_30 | 375 | -0.099 | -0.233 | -0.296 | -2.59 | 0.0106 |
| NYSE_OPEN | O5 | 1.0 | long | F3_NEAR_PIVOT_50 | 286 | -0.136 | -0.216 | -0.244 | -2.55 | 0.0111 |

## Top 40 by |t| (all trustworthy cells)

| Session | Apt | RR | Dir | Feature | N_on | ExpR_on | Δ_IS | Δ_OOS | t | dir_match |
|---------|-----|----|----|---------|------|---------|------|-------|---|-----------|
| SINGAPORE_OPEN | O15 | 1.5 | long | F3_NEAR_PIVOT_15 | 226 | -0.294 | -0.401 | -0.047 | -4.20 | Y |
| TOKYO_OPEN | O30 | 2.0 | long | F4_ABOVE_PDH | 30 | +0.794 | +0.894 | -0.168 | +3.67 | . |
| SINGAPORE_OPEN | O15 | 1.0 | long | F3_NEAR_PIVOT_15 | 226 | -0.240 | -0.272 | +0.278 | -3.50 | . |
| COMEX_SETTLE | O15 | 1.0 | long | F3_NEAR_PIVOT_50 | 214 | -0.286 | -0.263 | +0.162 | -3.22 | . |
| SINGAPORE_OPEN | O15 | 2.0 | long | F3_NEAR_PIVOT_15 | 225 | -0.254 | -0.361 | -0.078 | -3.22 | Y |
| COMEX_SETTLE | O15 | 1.5 | long | F3_NEAR_PIVOT_50 | 191 | -0.340 | -0.341 | +0.309 | -3.17 | . |
| TOKYO_OPEN | O30 | 1.5 | long | F4_ABOVE_PDH | 30 | +0.564 | +0.618 | -0.366 | +3.13 | . |
| TOKYO_OPEN | O30 | 2.0 | short | F2_NEAR_PDL_50 | 230 | +0.081 | +0.357 | -0.170 | +3.05 | . |
| NYSE_OPEN | O15 | 1.0 | short | F6_INSIDE_PDR | 239 | +0.103 | +0.269 | +0.190 | +2.96 | Y |
| TOKYO_OPEN | O30 | 2.0 | long | F6_INSIDE_PDR | 433 | -0.104 | -0.572 | -0.239 | -2.92 | Y |
| LONDON_METALS | O30 | 1.5 | long | F2_NEAR_PDL_30 | 155 | +0.204 | +0.311 | +1.046 | +2.87 | Y |
| LONDON_METALS | O30 | 1.5 | long | F3_NEAR_PIVOT_15 | 156 | -0.200 | -0.298 | +0.195 | -2.81 | . |
| NYSE_OPEN | O15 | 1.0 | short | F4_ABOVE_PDH | 86 | -0.248 | -0.306 | +0.329 | -2.81 | . |
| LONDON_METALS | O30 | 2.0 | long | F2_NEAR_PDL_30 | 152 | +0.246 | +0.362 | +0.019 | +2.79 | Y |
| TOKYO_OPEN | O15 | 1.0 | long | F3_NEAR_PIVOT_50 | 441 | -0.077 | -0.401 | -0.304 | -2.79 | Y |
| CME_REOPEN | O30 | 1.0 | long | F2_NEAR_PDL_50 | 44 | -0.173 | -0.446 | +0.160 | -2.79 | . |
| EUROPE_FLOW | O30 | 1.5 | short | F2_NEAR_PDL_30 | 151 | +0.117 | +0.304 | +0.612 | +2.79 | Y |
| SINGAPORE_OPEN | O15 | 1.0 | short | F2_NEAR_PDL_50 | 250 | -0.043 | +0.223 | +0.455 | +2.78 | Y |
| LONDON_METALS | O30 | 1.5 | long | F2_NEAR_PDL_15 | 84 | +0.300 | +0.370 | +nan | +2.77 | . |
| US_DATA_1000 | O5 | 2.0 | short | F5_BELOW_PDL | 70 | -0.439 | -0.411 | +nan | -2.76 | . |
| NYSE_OPEN | O15 | 1.0 | short | F3_NEAR_PIVOT_15 | 104 | +0.195 | +0.272 | +0.313 | +2.74 | Y |
| EUROPE_FLOW | O30 | 1.5 | short | F1_NEAR_PDH_30 | 139 | -0.277 | -0.289 | +0.125 | -2.69 | . |
| US_DATA_1000 | O30 | 2.0 | long | F5_BELOW_PDL | 35 | -0.617 | -0.507 | +nan | -2.67 | . |
| EUROPE_FLOW | O30 | 2.0 | short | F5_BELOW_PDL | 56 | +0.339 | +0.489 | +nan | +2.62 | . |
| TOKYO_OPEN | O15 | 2.0 | long | F2_NEAR_PDL_30 | 181 | -0.220 | -0.282 | +0.020 | -2.61 | . |
| LONDON_METALS | O30 | 1.0 | long | F3_NEAR_PIVOT_15 | 157 | -0.120 | -0.222 | +0.243 | -2.60 | . |
| TOKYO_OPEN | O15 | 1.0 | long | F3_NEAR_PIVOT_30 | 375 | -0.099 | -0.233 | -0.296 | -2.59 | Y |
| NYSE_OPEN | O5 | 1.0 | long | F3_NEAR_PIVOT_50 | 286 | -0.136 | -0.216 | -0.244 | -2.55 | Y |
| CME_REOPEN | O5 | 1.0 | long | F2_NEAR_PDL_50 | 120 | -0.237 | -0.260 | +0.080 | -2.55 | . |
| SINGAPORE_OPEN | O30 | 2.0 | long | F6_INSIDE_PDR | 355 | -0.095 | -0.410 | +0.833 | -2.52 | . |
| EUROPE_FLOW | O30 | 2.0 | short | F4_ABOVE_PDH | 85 | -0.366 | -0.351 | +0.011 | -2.51 | . |
| SINGAPORE_OPEN | O30 | 1.0 | long | F6_INSIDE_PDR | 363 | -0.106 | -0.261 | +0.672 | -2.51 | . |
| COMEX_SETTLE | O5 | 1.0 | long | F5_BELOW_PDL | 99 | +0.019 | +0.210 | -0.247 | +2.51 | . |
| LONDON_METALS | O30 | 1.0 | long | F2_NEAR_PDL_15 | 85 | +0.232 | +0.251 | +nan | +2.50 | . |
| NYSE_OPEN | O30 | 2.0 | long | F3_NEAR_PIVOT_30 | 100 | +0.129 | +0.430 | +0.571 | +2.48 | Y |
| TOKYO_OPEN | O30 | 1.0 | long | F1_NEAR_PDH_15 | 118 | +0.116 | +0.213 | +0.231 | +2.48 | Y |
| LONDON_METALS | O30 | 1.0 | long | F3_NEAR_PIVOT_30 | 273 | -0.056 | -0.200 | +0.580 | -2.46 | . |
| SINGAPORE_OPEN | O5 | 1.0 | short | F3_NEAR_PIVOT_30 | 338 | -0.195 | -0.214 | +0.488 | -2.44 | . |
| CME_REOPEN | O5 | 1.5 | long | F2_NEAR_PDL_50 | 111 | -0.302 | -0.326 | +0.070 | -2.44 | . |
| EUROPE_FLOW | O30 | 1.5 | short | F1_NEAR_PDH_15 | 72 | -0.344 | -0.317 | -0.949 | -2.42 | Y |

## Cross-reference — previously verified MGC cells

From `docs/audit/results/2026-04-15-t0-t8-audit-hot-warm-batch.md`:
- MGC SINGAPORE_OPEN O15 RR1.5 long F3_NEAR_PIVOT_15 → CONDITIONAL (6P/1F)
- MGC SINGAPORE_OPEN O15 RR2.0 long F3_NEAR_PIVOT_15 → CONDITIONAL (6P/1F)

Both should appear in this scan's survivors/promising (cross-check below):

### Verification rows
| RR | N_on | Δ_IS | Δ_OOS | t | p | dir_match |
|----|------|------|-------|---|---|-----------|
| 1.5 | 226 | -0.401 | -0.047 | -4.20 | 0.0000 | Y |
| 1.0 | 226 | -0.272 | +0.278 | -3.50 | 0.0005 | . |
| 2.0 | 225 | -0.361 | -0.078 | -3.22 | 0.0014 | Y |

## Baseline MGC per-session (no feature overlay) — data availability check

| Session | Apt | RR | N_is | N_oos | ExpR_is | ExpR_oos |
|---------|-----|----|------|-------|---------|----------|
| CME_REOPEN | O5 | 1.0 | 464 | 47 | -0.140 | +0.112 |
| CME_REOPEN | O5 | 1.5 | 409 | 44 | -0.252 | +0.262 |
| CME_REOPEN | O5 | 2.0 | 384 | 44 | -0.295 | +0.450 |
| CME_REOPEN | O15 | 1.0 | 304 | 34 | -0.103 | +0.311 |
| CME_REOPEN | O15 | 1.5 | 284 | 31 | -0.227 | +0.561 |
| CME_REOPEN | O15 | 2.0 | 273 | 30 | -0.250 | +0.742 |
| CME_REOPEN | O30 | 1.0 | 195 | 25 | -0.155 | +0.633 |
| CME_REOPEN | O30 | 1.5 | 188 | 23 | -0.190 | +0.897 |
| CME_REOPEN | O30 | 2.0 | 186 | 20 | -0.198 | +0.884 |
| TOKYO_OPEN | O5 | 1.0 | 917 | 66 | -0.214 | +0.032 |
| TOKYO_OPEN | O5 | 1.5 | 917 | 66 | -0.225 | +0.110 |
| TOKYO_OPEN | O5 | 2.0 | 917 | 66 | -0.239 | +0.072 |
| TOKYO_OPEN | O15 | 1.0 | 916 | 66 | -0.113 | +0.074 |
| TOKYO_OPEN | O15 | 1.5 | 914 | 66 | -0.105 | +0.160 |
| TOKYO_OPEN | O15 | 2.0 | 913 | 66 | -0.120 | +0.218 |
| TOKYO_OPEN | O30 | 1.0 | 914 | 66 | -0.065 | +0.114 |
| TOKYO_OPEN | O30 | 1.5 | 909 | 66 | -0.066 | +0.212 |
| TOKYO_OPEN | O30 | 2.0 | 904 | 66 | -0.062 | +0.280 |
| SINGAPORE_OPEN | O5 | 1.0 | 917 | 65 | -0.121 | +0.338 |
| SINGAPORE_OPEN | O5 | 1.5 | 917 | 65 | -0.132 | +0.414 |
| SINGAPORE_OPEN | O5 | 2.0 | 917 | 65 | -0.125 | +0.385 |
| SINGAPORE_OPEN | O15 | 1.0 | 912 | 65 | -0.120 | +0.137 |
| SINGAPORE_OPEN | O15 | 1.5 | 912 | 63 | -0.126 | +0.272 |
| SINGAPORE_OPEN | O15 | 2.0 | 908 | 62 | -0.120 | +0.409 |
| SINGAPORE_OPEN | O30 | 1.0 | 897 | 62 | -0.061 | +0.102 |
| SINGAPORE_OPEN | O30 | 1.5 | 894 | 60 | -0.048 | +0.177 |
| SINGAPORE_OPEN | O30 | 2.0 | 879 | 57 | -0.036 | +0.178 |
| LONDON_METALS | O5 | 1.0 | 916 | 66 | -0.134 | +0.014 |
| LONDON_METALS | O5 | 1.5 | 916 | 66 | -0.153 | +0.055 |
| LONDON_METALS | O5 | 2.0 | 915 | 66 | -0.128 | -0.070 |
| LONDON_METALS | O15 | 1.0 | 916 | 66 | -0.064 | -0.017 |
| LONDON_METALS | O15 | 1.5 | 911 | 66 | -0.074 | -0.168 |
| LONDON_METALS | O15 | 2.0 | 903 | 66 | -0.082 | -0.129 |
| LONDON_METALS | O30 | 1.0 | 908 | 65 | +0.008 | -0.165 |
| LONDON_METALS | O30 | 1.5 | 896 | 65 | -0.027 | -0.141 |
| LONDON_METALS | O30 | 2.0 | 880 | 64 | -0.029 | -0.272 |
| EUROPE_FLOW | O5 | 1.0 | 916 | 66 | -0.129 | -0.044 |
| EUROPE_FLOW | O5 | 1.5 | 916 | 66 | -0.122 | -0.082 |
| EUROPE_FLOW | O5 | 2.0 | 916 | 66 | -0.126 | -0.028 |
| EUROPE_FLOW | O15 | 1.0 | 915 | 66 | -0.077 | +0.035 |
| EUROPE_FLOW | O15 | 1.5 | 912 | 66 | -0.095 | +0.077 |
| EUROPE_FLOW | O15 | 2.0 | 908 | 66 | -0.064 | +0.121 |
| EUROPE_FLOW | O30 | 1.0 | 908 | 66 | -0.048 | +0.047 |
| EUROPE_FLOW | O30 | 1.5 | 904 | 66 | -0.039 | +0.056 |
| EUROPE_FLOW | O30 | 2.0 | 890 | 66 | -0.012 | +0.005 |
| US_DATA_830 | O5 | 1.0 | 859 | 64 | -0.087 | -0.088 |
| US_DATA_830 | O5 | 1.5 | 833 | 64 | -0.083 | +0.028 |
| US_DATA_830 | O5 | 2.0 | 816 | 64 | -0.064 | -0.033 |
| US_DATA_830 | O15 | 1.0 | 808 | 64 | -0.018 | -0.156 |
| US_DATA_830 | O15 | 1.5 | 765 | 63 | -0.020 | -0.120 |
| US_DATA_830 | O15 | 2.0 | 716 | 62 | -0.063 | -0.115 |
| US_DATA_830 | O30 | 1.0 | 748 | 60 | +0.025 | +0.035 |
| US_DATA_830 | O30 | 1.5 | 668 | 57 | +0.011 | +0.062 |
| US_DATA_830 | O30 | 2.0 | 609 | 56 | -0.026 | +0.142 |
| NYSE_OPEN | O5 | 1.0 | 912 | 66 | -0.039 | +0.341 |
| NYSE_OPEN | O5 | 1.5 | 901 | 64 | -0.061 | +0.241 |
| NYSE_OPEN | O5 | 2.0 | 882 | 63 | -0.070 | +0.235 |
| NYSE_OPEN | O15 | 1.0 | 826 | 59 | +0.023 | +0.156 |
| NYSE_OPEN | O15 | 1.5 | 737 | 55 | -0.034 | +0.109 |
| NYSE_OPEN | O15 | 2.0 | 675 | 49 | -0.097 | +0.013 |
| NYSE_OPEN | O30 | 1.0 | 639 | 49 | +0.062 | -0.041 |
| NYSE_OPEN | O30 | 1.5 | 519 | 43 | -0.010 | +0.027 |
| NYSE_OPEN | O30 | 2.0 | 443 | 39 | -0.118 | -0.019 |
| US_DATA_1000 | O5 | 1.0 | 868 | 65 | -0.029 | +0.035 |
| US_DATA_1000 | O5 | 1.5 | 822 | 65 | -0.038 | -0.001 |
| US_DATA_1000 | O5 | 2.0 | 781 | 64 | -0.081 | -0.007 |
| US_DATA_1000 | O15 | 1.0 | 742 | 62 | +0.004 | +0.159 |
| US_DATA_1000 | O15 | 1.5 | 651 | 58 | -0.007 | +0.045 |
| US_DATA_1000 | O15 | 2.0 | 583 | 50 | -0.084 | -0.185 |
| US_DATA_1000 | O30 | 1.0 | 569 | 52 | +0.061 | -0.022 |
| US_DATA_1000 | O30 | 1.5 | 439 | 44 | -0.076 | -0.059 |
| US_DATA_1000 | O30 | 2.0 | 382 | 36 | -0.180 | -0.268 |
| COMEX_SETTLE | O5 | 1.0 | 882 | 64 | -0.160 | -0.131 |
| COMEX_SETTLE | O5 | 1.5 | 846 | 62 | -0.200 | -0.252 |
| COMEX_SETTLE | O5 | 2.0 | 817 | 61 | -0.228 | -0.317 |
| COMEX_SETTLE | O15 | 1.0 | 771 | 60 | -0.148 | -0.110 |
| COMEX_SETTLE | O15 | 1.5 | 656 | 57 | -0.227 | -0.124 |
| COMEX_SETTLE | O15 | 2.0 | 606 | 52 | -0.283 | -0.228 |
| COMEX_SETTLE | O30 | 1.0 | 586 | 54 | -0.091 | -0.037 |
| COMEX_SETTLE | O30 | 1.5 | 479 | 49 | -0.189 | +0.028 |
| COMEX_SETTLE | O30 | 2.0 | 423 | 40 | -0.274 | -0.132 |
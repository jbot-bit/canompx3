# Results — Prior-Day Zone / Positional / Gap-Categorical Features on ORB Outcomes

**Pre-registration:** `docs/audit/hypotheses/2026-04-15-prior-day-zone-positional-features-orb.md`
**Run date:** 2026-04-15
**Rows IS / OOS:** 18864 / 746
**K_local:** 96 | **K_global:** 639 | **Bonferroni-local p:** 5.21e-04

## Primary cells (theta=0.30)

| Instr | Session | RR | Signal | N_IS | N_OOS | ExpR_on | ExpR_off | Delta_IS | Delta_OOS | t | p_raw | Bonf | BH | Chordia | Dir_match | OOS/IS |
|-------|---------|----|---------|------|-------|---------|----------|----------|-----------|---|-------|------|-----|---------|-----------|--------|
| MNQ | CME_PRECLOSE | 1.0 | F1_NEAR_PDH | 462 | 19 | +0.139 | +0.080 | +0.059 | -0.147 | +1.17 | 0.2417 | fail | fail | fail | fail | -2.50 |
| MNQ | CME_PRECLOSE | 1.0 | F2_NEAR_PDL | 330 | 14 | +0.107 | +0.096 | +0.011 | -0.574 | +0.19 | 0.8478 | fail | fail | fail | fail | -52.69 |
| MNQ | CME_PRECLOSE | 1.0 | F3_NEAR_PIVOT | 462 | 15 | +0.111 | +0.093 | +0.018 | +0.204 | +0.36 | 0.7187 | fail | fail | fail | PASS | +11.19 |
| MNQ | CME_PRECLOSE | 1.0 | F4_ABOVE_PDH | 496 | 17 | +0.085 | +0.106 | -0.021 | -0.162 | -0.42 | 0.6721 | fail | fail | fail | PASS | +7.72 |
| MNQ | CME_PRECLOSE | 1.0 | F5_BELOW_PDL | 304 | 16 | +0.110 | +0.096 | +0.014 | +0.114 | +0.25 | 0.8062 | fail | fail | fail | PASS | +7.85 |
| MNQ | CME_PRECLOSE | 1.0 | F6_INSIDE_PDR | 652 | 28 | +0.104 | +0.095 | +0.009 | +0.043 | +0.20 | 0.8432 | fail | fail | fail | PASS | +4.55 |
| MNQ | CME_PRECLOSE | 1.0 | F7_GAP_UP | 35 | 1 | +0.200 | +0.096 | +0.103 | +0.896 | +0.68 | 0.4983 | fail | fail | fail | PASS | +8.66 |
| MNQ | CME_PRECLOSE | 1.5 | F1_NEAR_PDH | 419 | 18 | +0.159 | +0.055 | +0.104 | -0.071 | +1.54 | 0.1234 | fail | fail | fail | fail | -0.68 |
| MNQ | CME_PRECLOSE | 1.5 | F2_NEAR_PDL | 286 | 12 | +0.077 | +0.092 | -0.015 | -0.540 | -0.20 | 0.8445 | fail | fail | fail | PASS | +35.77 |
| MNQ | CME_PRECLOSE | 1.5 | F3_NEAR_PIVOT | 418 | 11 | +0.127 | +0.071 | +0.056 | +0.038 | +0.83 | 0.4094 | fail | fail | fail | PASS | +0.68 |
| MNQ | CME_PRECLOSE | 1.5 | F4_ABOVE_PDH | 450 | 17 | +0.072 | +0.097 | -0.025 | -0.205 | -0.38 | 0.7029 | fail | fail | fail | PASS | +8.08 |
| MNQ | CME_PRECLOSE | 1.5 | F5_BELOW_PDL | 266 | 15 | +0.102 | +0.085 | +0.016 | -0.253 | +0.21 | 0.8375 | fail | fail | fail | fail | -15.41 |
| MNQ | CME_PRECLOSE | 1.5 | F6_INSIDE_PDR | 580 | 25 | +0.096 | +0.083 | +0.012 | +0.373 | +0.19 | 0.8461 | fail | fail | fail | PASS | +30.04 |
| MNQ | CME_PRECLOSE | 1.5 | F7_GAP_UP | 30 | 1 | +0.213 | +0.086 | +0.128 | -0.831 | +0.60 | 0.5553 | fail | fail | fail | fail | -6.51 |
| MNQ | US_DATA_1000 | 1.0 | F1_NEAR_PDH | 646 | 22 | +0.086 | +0.086 | +0.000 | +0.080 | +0.00 | 0.9967 | fail | fail | fail | PASS | +412.80 |
| MNQ | US_DATA_1000 | 1.0 | F2_NEAR_PDL | 428 | 14 | +0.075 | +0.090 | -0.015 | +0.151 | -0.28 | 0.7765 | fail | fail | fail | fail | -10.10 |
| MNQ | US_DATA_1000 | 1.0 | F3_NEAR_PIVOT | 796 | 25 | +0.046 | +0.122 | -0.076 | +0.148 | -1.66 | 0.0972 | fail | fail | fail | fail | -1.95 |
| MNQ | US_DATA_1000 | 1.0 | F4_ABOVE_PDH | 434 | 12 | +0.118 | +0.075 | +0.043 | -0.048 | +0.83 | 0.4050 | fail | fail | fail | fail | -1.13 |
| MNQ | US_DATA_1000 | 1.0 | F5_BELOW_PDL | 290 | 14 | +0.196 | +0.064 | +0.132 | -0.032 | +2.19 | 0.0288 | fail | fail | fail | fail | -0.25 |
| MNQ | US_DATA_1000 | 1.0 | F6_INSIDE_PDR | 975 | 38 | +0.041 | +0.148 | -0.107 | +0.054 | -2.34 | 0.0196 | fail | fail | fail | fail | -0.50 |
| MNQ | US_DATA_1000 | 1.0 | F7_GAP_UP | 41 | 1 | -0.023 | +0.089 | -0.112 | -1.022 | -0.74 | 0.4664 | fail | fail | fail | PASS | +9.11 |
| MNQ | US_DATA_1000 | 1.5 | F1_NEAR_PDH | 632 | 22 | +0.098 | +0.088 | +0.010 | +0.102 | +0.18 | 0.8610 | fail | fail | fail | PASS | +9.80 |
| MNQ | US_DATA_1000 | 1.5 | F2_NEAR_PDL | 421 | 13 | +0.056 | +0.104 | -0.048 | -0.219 | -0.73 | 0.4672 | fail | fail | fail | PASS | +4.51 |
| MNQ | US_DATA_1000 | 1.5 | F3_NEAR_PIVOT | 780 | 25 | +0.066 | +0.114 | -0.048 | +0.244 | -0.82 | 0.4097 | fail | fail | fail | fail | -5.12 |
| MNQ | US_DATA_1000 | 1.5 | F4_ABOVE_PDH | 427 | 12 | +0.142 | +0.075 | +0.068 | +0.093 | +1.02 | 0.3065 | fail | fail | fail | PASS | +1.37 |
| MNQ | US_DATA_1000 | 1.5 | F5_BELOW_PDL | 289 | 13 | +0.218 | +0.065 | +0.153 | -0.223 | +1.97 | 0.0494 | fail | fail | fail | fail | -1.46 |
| MNQ | US_DATA_1000 | 1.5 | F6_INSIDE_PDR | 956 | 38 | +0.032 | +0.171 | -0.139 | +0.093 | -2.38 | 0.0175 | fail | fail | fail | fail | -0.67 |
| MNQ | US_DATA_1000 | 1.5 | F7_GAP_UP | 41 | 1 | -0.074 | +0.096 | -0.170 | -0.943 | -0.92 | 0.3652 | fail | fail | fail | PASS | +5.55 |
| MNQ | NYSE_OPEN | 1.0 | F1_NEAR_PDH | 717 | 22 | +0.086 | +0.076 | +0.010 | +0.041 | +0.21 | 0.8308 | fail | fail | fail | PASS | +4.07 |
| MNQ | NYSE_OPEN | 1.0 | F2_NEAR_PDL | 467 | 18 | +0.088 | +0.077 | +0.011 | +0.103 | +0.21 | 0.8321 | fail | fail | fail | PASS | +9.36 |
| MNQ | NYSE_OPEN | 1.0 | F3_NEAR_PIVOT | 879 | 30 | +0.034 | +0.130 | -0.096 | +0.084 | -2.07 | 0.0390 | fail | fail | fail | fail | -0.88 |
| MNQ | NYSE_OPEN | 1.0 | F4_ABOVE_PDH | 407 | 9 | +0.157 | +0.056 | +0.101 | -0.046 | +1.88 | 0.0605 | fail | fail | fail | fail | -0.46 |
| MNQ | NYSE_OPEN | 1.0 | F5_BELOW_PDL | 246 | 11 | +0.213 | +0.058 | +0.156 | -0.069 | +2.40 | 0.0170 | fail | fail | fail | fail | -0.44 |
| MNQ | NYSE_OPEN | 1.0 | F6_INSIDE_PDR | 1036 | 46 | +0.018 | +0.179 | -0.161 | +0.071 | -3.40 | 0.0007 | fail | PASS | fail | fail | -0.44 |
| MNQ | NYSE_OPEN | 1.0 | F7_GAP_UP | 42 | 1 | +0.107 | +0.080 | +0.027 | -1.150 | +0.18 | 0.8595 | fail | fail | fail | fail | -42.63 |
| MNQ | NYSE_OPEN | 1.5 | F1_NEAR_PDH | 700 | 22 | +0.130 | +0.068 | +0.061 | +0.034 | +1.03 | 0.3050 | fail | fail | fail | PASS | +0.56 |
| MNQ | NYSE_OPEN | 1.5 | F2_NEAR_PDL | 453 | 16 | +0.084 | +0.098 | -0.015 | +0.191 | -0.22 | 0.8266 | fail | fail | fail | fail | -13.13 |
| MNQ | NYSE_OPEN | 1.5 | F3_NEAR_PIVOT | 858 | 29 | +0.035 | +0.159 | -0.124 | +0.014 | -2.10 | 0.0356 | fail | fail | fail | fail | -0.11 |
| MNQ | NYSE_OPEN | 1.5 | F4_ABOVE_PDH | 393 | 9 | +0.173 | +0.070 | +0.103 | +0.319 | +1.49 | 0.1374 | fail | fail | fail | PASS | +3.10 |
| MNQ | NYSE_OPEN | 1.5 | F5_BELOW_PDL | 238 | 10 | +0.294 | +0.061 | +0.233 | -0.127 | +2.75 | 0.0062 | fail | fail | fail | fail | -0.54 |
| MNQ | NYSE_OPEN | 1.5 | F6_INSIDE_PDR | 1015 | 44 | +0.016 | +0.220 | -0.205 | -0.105 | -3.38 | 0.0008 | fail | PASS | fail | PASS | +0.51 |
| MNQ | NYSE_OPEN | 1.5 | F7_GAP_UP | 39 | 1 | +0.055 | +0.095 | -0.040 | -1.110 | -0.20 | 0.8392 | fail | fail | fail | PASS | +27.59 |
| MES | CME_PRECLOSE | 1.0 | F1_NEAR_PDH | 477 | 18 | -0.022 | +0.008 | -0.029 | -0.254 | -0.62 | 0.5346 | fail | fail | fail | PASS | +8.65 |
| MES | CME_PRECLOSE | 1.0 | F2_NEAR_PDL | 341 | 12 | +0.021 | -0.010 | +0.030 | +0.051 | +0.57 | 0.5695 | fail | fail | fail | PASS | +1.69 |
| MES | CME_PRECLOSE | 1.0 | F3_NEAR_PIVOT | 501 | 21 | -0.015 | +0.005 | -0.020 | +0.251 | -0.42 | 0.6735 | fail | fail | fail | fail | -12.69 |
| MES | CME_PRECLOSE | 1.0 | F4_ABOVE_PDH | 475 | 17 | -0.005 | -0.001 | -0.004 | -0.074 | -0.09 | 0.9318 | fail | fail | fail | PASS | +18.30 |
| MES | CME_PRECLOSE | 1.0 | F5_BELOW_PDL | 297 | 12 | +0.063 | -0.019 | +0.082 | -0.112 | +1.46 | 0.1446 | fail | fail | fail | fail | -1.36 |
| MES | CME_PRECLOSE | 1.0 | F6_INSIDE_PDR | 648 | 31 | -0.032 | +0.022 | -0.054 | +0.132 | -1.20 | 0.2316 | fail | fail | fail | fail | -2.44 |
| MES | CME_PRECLOSE | 1.0 | F7_GAP_UP | 37 | 3 | -0.240 | +0.004 | -0.244 | +0.354 | -1.75 | 0.0888 | fail | fail | fail | fail | -1.45 |
| MES | CME_PRECLOSE | 1.5 | F1_NEAR_PDH | 412 | 17 | -0.088 | -0.038 | -0.051 | -0.385 | -0.80 | 0.4248 | fail | fail | fail | PASS | +7.60 |
| MES | CME_PRECLOSE | 1.5 | F2_NEAR_PDL | 295 | 9 | -0.027 | -0.063 | +0.036 | -0.049 | +0.51 | 0.6112 | fail | fail | fail | fail | -1.35 |
| MES | CME_PRECLOSE | 1.5 | F3_NEAR_PIVOT | 428 | 16 | -0.026 | -0.070 | +0.044 | +0.073 | +0.69 | 0.4883 | fail | fail | fail | PASS | +1.67 |
| MES | CME_PRECLOSE | 1.5 | F4_ABOVE_PDH | 415 | 17 | -0.088 | -0.038 | -0.050 | +0.153 | -0.80 | 0.4264 | fail | fail | fail | fail | -3.03 |
| MES | CME_PRECLOSE | 1.5 | F5_BELOW_PDL | 250 | 11 | +0.067 | -0.086 | +0.153 | -0.185 | +1.97 | 0.0491 | fail | fail | fail | fail | -1.21 |
| MES | CME_PRECLOSE | 1.5 | F6_INSIDE_PDR | 554 | 26 | -0.089 | -0.026 | -0.063 | -0.012 | -1.03 | 0.3037 | fail | fail | fail | PASS | +0.19 |
| MES | CME_PRECLOSE | 1.5 | F7_GAP_UP | 36 | 2 | -0.200 | -0.050 | -0.149 | -0.782 | -0.86 | 0.3928 | fail | fail | fail | PASS | +5.24 |
| MES | US_DATA_1000 | 1.0 | F1_NEAR_PDH | 703 | 23 | -0.056 | -0.011 | -0.045 | -0.005 | -1.04 | 0.2982 | fail | fail | fail | PASS | +0.11 |
| MES | US_DATA_1000 | 1.0 | F2_NEAR_PDL | 462 | 15 | -0.005 | -0.039 | +0.034 | +0.301 | +0.70 | 0.4867 | fail | fail | fail | PASS | +8.95 |
| MES | US_DATA_1000 | 1.0 | F3_NEAR_PIVOT | 830 | 27 | -0.079 | +0.017 | -0.096 | +0.128 | -2.23 | 0.0261 | fail | fail | fail | fail | -1.34 |
| MES | US_DATA_1000 | 1.0 | F4_ABOVE_PDH | 427 | 13 | +0.053 | -0.057 | +0.110 | -0.252 | +2.25 | 0.0249 | fail | fail | fail | fail | -2.30 |
| MES | US_DATA_1000 | 1.0 | F5_BELOW_PDL | 279 | 16 | -0.017 | -0.032 | +0.015 | -0.116 | +0.25 | 0.7996 | fail | fail | fail | fail | -7.69 |
| MES | US_DATA_1000 | 1.0 | F6_INSIDE_PDR | 987 | 36 | -0.069 | +0.025 | -0.094 | +0.250 | -2.15 | 0.0318 | fail | fail | fail | fail | -2.68 |
| MES | US_DATA_1000 | 1.0 | F7_GAP_UP | 41 | 3 | +0.022 | -0.031 | +0.053 | +0.201 | +0.38 | 0.7074 | fail | fail | fail | PASS | +3.82 |
| MES | US_DATA_1000 | 1.0 | F8_GAP_DOWN | 34 | 5 | -0.330 | -0.023 | -0.307 | +0.130 | -2.04 | 0.0493 | fail | fail | fail | fail | -0.42 |
| MES | US_DATA_1000 | 1.5 | F1_NEAR_PDH | 693 | 22 | -0.048 | -0.002 | -0.046 | -0.020 | -0.84 | 0.4032 | fail | fail | fail | PASS | +0.45 |
| MES | US_DATA_1000 | 1.5 | F2_NEAR_PDL | 451 | 14 | -0.050 | -0.011 | -0.039 | +0.166 | -0.64 | 0.5210 | fail | fail | fail | fail | -4.26 |
| MES | US_DATA_1000 | 1.5 | F3_NEAR_PIVOT | 820 | 27 | -0.051 | +0.007 | -0.058 | +0.435 | -1.06 | 0.2873 | fail | fail | fail | fail | -7.56 |
| MES | US_DATA_1000 | 1.5 | F4_ABOVE_PDH | 420 | 12 | +0.026 | -0.037 | +0.063 | -0.605 | +1.01 | 0.3137 | fail | fail | fail | fail | -9.64 |
| MES | US_DATA_1000 | 1.5 | F5_BELOW_PDL | 274 | 14 | -0.054 | -0.015 | -0.039 | -0.499 | -0.53 | 0.5948 | fail | fail | fail | PASS | +12.65 |
| MES | US_DATA_1000 | 1.5 | F6_INSIDE_PDR | 972 | 36 | -0.034 | -0.004 | -0.031 | +0.746 | -0.56 | 0.5767 | fail | fail | fail | fail | -24.35 |
| MES | US_DATA_1000 | 1.5 | F7_GAP_UP | 40 | 2 | -0.141 | -0.018 | -0.122 | +0.072 | -0.71 | 0.4829 | fail | fail | fail | fail | -0.59 |
| MES | US_DATA_1000 | 1.5 | F8_GAP_DOWN | 34 | 5 | -0.417 | -0.013 | -0.404 | -0.106 | -2.36 | 0.0242 | fail | fail | fail | PASS | +0.26 |
| MES | NYSE_OPEN | 1.0 | F1_NEAR_PDH | 740 | 23 | -0.020 | +0.020 | -0.041 | +0.086 | -0.92 | 0.3566 | fail | fail | fail | fail | -2.13 |
| MES | NYSE_OPEN | 1.0 | F2_NEAR_PDL | 456 | 19 | -0.110 | +0.044 | -0.154 | +0.010 | -3.11 | 0.0019 | fail | fail | fail | fail | -0.06 |
| MES | NYSE_OPEN | 1.0 | F3_NEAR_PIVOT | 873 | 30 | -0.021 | +0.027 | -0.048 | +0.103 | -1.09 | 0.2749 | fail | fail | fail | fail | -2.15 |
| MES | NYSE_OPEN | 1.0 | F4_ABOVE_PDH | 403 | 10 | +0.048 | -0.011 | +0.059 | -0.522 | +1.17 | 0.2440 | fail | fail | fail | fail | -8.77 |
| MES | NYSE_OPEN | 1.0 | F5_BELOW_PDL | 247 | 13 | +0.034 | -0.003 | +0.037 | +0.044 | +0.58 | 0.5612 | fail | fail | fail | PASS | +1.21 |
| MES | NYSE_OPEN | 1.0 | F6_INSIDE_PDR | 1044 | 43 | -0.019 | +0.037 | -0.056 | +0.264 | -1.24 | 0.2150 | fail | fail | fail | fail | -4.74 |
| MES | NYSE_OPEN | 1.0 | F7_GAP_UP | 43 | 3 | -0.124 | +0.006 | -0.130 | +0.266 | -0.93 | 0.3575 | fail | fail | fail | fail | -2.04 |
| MES | NYSE_OPEN | 1.0 | F8_GAP_DOWN | 32 | 5 | -0.089 | +0.004 | -0.094 | +0.584 | -0.57 | 0.5743 | fail | fail | fail | fail | -6.23 |
| MES | NYSE_OPEN | 1.5 | F1_NEAR_PDH | 732 | 22 | -0.039 | +0.063 | -0.102 | -0.035 | -1.84 | 0.0665 | fail | fail | fail | PASS | +0.34 |
| MES | NYSE_OPEN | 1.5 | F2_NEAR_PDL | 449 | 19 | -0.068 | +0.050 | -0.118 | -0.139 | -1.91 | 0.0569 | fail | fail | fail | PASS | +1.18 |
| MES | NYSE_OPEN | 1.5 | F3_NEAR_PIVOT | 865 | 30 | -0.000 | +0.038 | -0.038 | +0.202 | -0.69 | 0.4934 | fail | fail | fail | fail | -5.36 |
| MES | NYSE_OPEN | 1.5 | F4_ABOVE_PDH | 397 | 10 | +0.033 | +0.014 | +0.019 | -0.603 | +0.29 | 0.7698 | fail | fail | fail | fail | -31.81 |
| MES | NYSE_OPEN | 1.5 | F5_BELOW_PDL | 242 | 13 | +0.072 | +0.009 | +0.063 | -0.072 | +0.79 | 0.4321 | fail | fail | fail | fail | -1.14 |
| MES | NYSE_OPEN | 1.5 | F6_INSIDE_PDR | 1035 | 42 | +0.005 | +0.040 | -0.035 | +0.394 | -0.61 | 0.5393 | fail | fail | fail | fail | -11.29 |
| MES | NYSE_OPEN | 1.5 | F7_GAP_UP | 42 | 2 | -0.143 | +0.022 | -0.166 | +0.197 | -0.96 | 0.3432 | fail | fail | fail | fail | -1.19 |

## Framework-integrity controls (outside K=96 budget)

| Control | Instr | Session | RR | N_IS | Delta_IS | t | p_raw |
|---------|-------|---------|----|------|----------|---|-------|
| C1_DESTRUCTION | MNQ | CME_PRECLOSE | 1.0 | 557 | -0.073 | -1.49 | 0.1363 |
| C2_KNOWN_NULL | MNQ | CME_PRECLOSE | 1.0 | 732 | +0.062 | +1.32 | 0.1876 |
| C1_DESTRUCTION | MNQ | CME_PRECLOSE | 1.5 | 510 | +0.082 | +1.25 | 0.2102 |
| C2_KNOWN_NULL | MNQ | CME_PRECLOSE | 1.5 | 657 | +0.055 | +0.86 | 0.3881 |
| C1_DESTRUCTION | MNQ | US_DATA_1000 | 1.0 | 659 | -0.034 | -0.72 | 0.4726 |
| C2_KNOWN_NULL | MNQ | US_DATA_1000 | 1.0 | 861 | +0.008 | +0.17 | 0.8638 |
| C1_DESTRUCTION | MNQ | US_DATA_1000 | 1.5 | 636 | +0.074 | +1.24 | 0.2142 |
| C2_KNOWN_NULL | MNQ | US_DATA_1000 | 1.5 | 852 | +0.032 | +0.56 | 0.5777 |
| C1_DESTRUCTION | MNQ | NYSE_OPEN | 1.0 | 637 | -0.040 | -0.83 | 0.4043 |
| C2_KNOWN_NULL | MNQ | NYSE_OPEN | 1.0 | 815 | -0.052 | -1.13 | 0.2602 |
| C1_DESTRUCTION | MNQ | NYSE_OPEN | 1.5 | 645 | +0.084 | +1.38 | 0.1666 |
| C2_KNOWN_NULL | MNQ | NYSE_OPEN | 1.5 | 793 | +0.000 | +0.01 | 0.9935 |
| C1_DESTRUCTION | MES | CME_PRECLOSE | 1.0 | 551 | +0.050 | +1.08 | 0.2812 |
| C2_KNOWN_NULL | MES | CME_PRECLOSE | 1.0 | 709 | -0.000 | -0.00 | 0.9968 |
| C1_DESTRUCTION | MES | CME_PRECLOSE | 1.5 | 492 | +0.033 | +0.54 | 0.5928 |
| C2_KNOWN_NULL | MES | CME_PRECLOSE | 1.5 | 611 | -0.023 | -0.37 | 0.7108 |
| C1_DESTRUCTION | MES | US_DATA_1000 | 1.0 | 653 | +0.039 | +0.88 | 0.3808 |
| C2_KNOWN_NULL | MES | US_DATA_1000 | 1.0 | 847 | -0.041 | -0.95 | 0.3413 |
| C1_DESTRUCTION | MES | US_DATA_1000 | 1.5 | 675 | -0.057 | -1.04 | 0.2995 |
| C2_KNOWN_NULL | MES | US_DATA_1000 | 1.5 | 834 | -0.060 | -1.10 | 0.2698 |
| C1_DESTRUCTION | MES | NYSE_OPEN | 1.0 | 643 | +0.044 | +0.98 | 0.3269 |
| C2_KNOWN_NULL | MES | NYSE_OPEN | 1.0 | 847 | -0.066 | -1.52 | 0.1298 |
| C1_DESTRUCTION | MES | NYSE_OPEN | 1.5 | 675 | +0.091 | +1.61 | 0.1077 |
| C2_KNOWN_NULL | MES | NYSE_OPEN | 1.5 | 833 | -0.082 | -1.50 | 0.1347 |

## Preliminary verdict (before §9 failure-mode full-gate check)

- Cells tested: 87
- Bonferroni-local passers: 0
- BH-FDR local passers: 2
- Chordia t>=3.79 passers: 0
- Joint BH + Chordia passers: 0

Full per-cell failure-mode evaluation (era stability, Jaccard, partial-regression) in next-pass analysis.
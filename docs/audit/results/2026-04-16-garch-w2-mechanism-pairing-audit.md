# Garch W2 Mechanism Pairing Audit

**Date:** 2026-04-16
**Boundary:** validated shelf only, exact filter semantics, no deployment conclusions.
**Mechanisms:**
- `M1 latent_expansion`: `garch_high AND overnight_range_pct < 80`
- `M2 active_transition`: `garch_high AND atr_vel_regime = Expanding`

2026 OOS is descriptive only in this stage.

## COMEX_SETTLE_high

### M1 latent_expansion

- Cells: **18**
- Pooled `N_total`: **8503**
- Pooled conjunction `N`: **1802**
- Base ExpR: **+0.108**
- Conjunction ExpR: **+0.190**

| Check | Support / valid cells |
|---|---:|
| garch marginal | 18/18 |
| partner marginal | 4/18 |
| partner inside garch | 9/18 |
| garch inside partner | 18/18 |

| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | long | COST_LT12 | 623 | +0.100 | +0.244 | +0.072 | 119 | +0.265 | +0.208 | -0.105 | +0.056 | +0.208 | 9 | +0.070 |
| MNQ | long | ORB_G5 | 738 | +0.076 | +0.247 | +0.052 | 122 | +0.275 | +0.233 | -0.103 | +0.075 | +0.233 | 9 | +0.070 |
| MNQ | long | ORB_G5 | 731 | +0.110 | +0.254 | +0.097 | 121 | +0.296 | +0.198 | -0.057 | +0.109 | +0.198 | 8 | +0.203 |
| MNQ | long | ORB_G5 | 721 | +0.132 | +0.316 | +0.108 | 120 | +0.358 | +0.251 | -0.110 | +0.112 | +0.251 | 7 | -0.180 |
| MNQ | long | ORB_G5_NOFRI | 603 | +0.090 | +0.256 | +0.069 | 103 | +0.270 | +0.230 | -0.091 | +0.038 | +0.230 | 9 | +0.070 |
| MNQ | long | OVNRNG_100 | 254 | +0.174 | +0.301 | +0.217 | 54 | +0.440 | +0.254 | +0.076 | +0.241 | +0.254 | 8 | +0.204 |
| MNQ | long | OVNRNG_100 | 249 | +0.204 | +0.348 | +0.343 | 54 | +0.579 | +0.288 | +0.251 | +0.408 | +0.288 | 7 | +0.375 |
| MNQ | long | X_MES_ATR60 | 330 | +0.203 | +0.294 | +0.166 | 111 | +0.302 | +0.194 | -0.119 | +0.020 | +0.194 | 9 | +0.070 |
| MNQ | long | X_MES_ATR60 | 324 | +0.229 | +0.314 | +0.198 | 110 | +0.341 | +0.182 | -0.099 | +0.075 | +0.182 | 8 | +0.203 |
| MNQ | short | COST_LT12 | 560 | +0.106 | +0.138 | +0.090 | 118 | +0.017 | +0.046 | -0.069 | -0.418 | +0.046 | 12 | +0.595 |
| MNQ | short | ORB_G5 | 646 | +0.079 | +0.142 | +0.063 | 120 | +0.015 | +0.085 | -0.077 | -0.430 | +0.085 | 12 | +0.595 |
| MNQ | short | ORB_G5 | 640 | +0.094 | +0.227 | +0.083 | 119 | +0.063 | +0.181 | -0.052 | -0.555 | +0.181 | 12 | +0.797 |
| MNQ | short | ORB_G5 | 634 | +0.020 | +0.080 | +0.035 | 119 | -0.032 | +0.081 | +0.073 | -0.383 | +0.081 | 12 | +1.157 |
| MNQ | short | ORB_G5_NOFRI | 507 | +0.102 | +0.233 | +0.102 | 87 | +0.165 | +0.175 | +0.002 | -0.213 | +0.175 | 7 | +0.639 |
| MNQ | short | OVNRNG_100 | 212 | +0.119 | +0.234 | +0.093 | 57 | +0.060 | +0.230 | -0.052 | -0.376 | +0.230 | 12 | +0.595 |
| MNQ | short | OVNRNG_100 | 211 | +0.193 | +0.364 | +0.187 | 57 | +0.118 | +0.344 | -0.012 | -0.533 | +0.344 | 12 | +0.797 |
| MNQ | short | X_MES_ATR60 | 261 | +0.055 | +0.140 | +0.029 | 106 | +0.043 | +0.189 | -0.110 | -0.366 | +0.189 | 12 | +0.595 |
| MNQ | short | X_MES_ATR60 | 259 | +0.067 | +0.204 | +0.023 | 105 | +0.071 | +0.305 | -0.189 | -0.500 | +0.305 | 12 | +0.797 |

- Verdict: **unclear**
- Allowed implication: **unclear**

#### Descriptive buckets inside garch_high

**MNQ long ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 73 | +0.247 | +18.0 |
| overnight_low | 18 | +0.252 | +4.5 |
| overnight_mid | 102 | +0.377 | +38.5 |

**MNQ short ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 49 | +0.351 | +17.2 |
| overnight_low | 21 | -0.193 | -4.1 |
| overnight_mid | 98 | +0.003 | +0.3 |

**MNQ long COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 74 | +0.209 | +15.5 |
| overnight_low | 19 | +0.381 | +7.2 |
| overnight_mid | 100 | +0.243 | +24.3 |

**MNQ short COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 48 | +0.435 | +20.9 |
| overnight_low | 20 | +0.028 | +0.6 |
| overnight_mid | 98 | +0.015 | +1.5 |

**MNQ long OVNRNG_100**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 71 | +0.172 | +12.2 |
| overnight_low | 1 | +1.324 | +1.3 |
| overnight_mid | 53 | +0.565 | +30.0 |

**MNQ short OVNRNG_100**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 49 | +0.651 | +31.9 |
| overnight_mid | 57 | +0.118 | +6.7 |

**MNQ long X_MES_ATR60**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 62 | +0.266 | +16.5 |
| overnight_low | 17 | +0.373 | +6.3 |
| overnight_mid | 93 | +0.335 | +31.2 |

**MNQ short X_MES_ATR60**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 38 | +0.572 | +21.7 |
| overnight_low | 19 | +0.112 | +2.1 |
| overnight_mid | 86 | +0.062 | +5.4 |

**MNQ long ORB_G5_NOFRI**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 63 | +0.233 | +14.7 |
| overnight_low | 16 | +0.520 | +8.3 |
| overnight_mid | 87 | +0.224 | +19.5 |

**MNQ short ORB_G5_NOFRI**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 40 | +0.378 | +15.1 |
| overnight_low | 14 | +0.335 | +4.7 |
| overnight_mid | 73 | +0.133 | +9.7 |

### M2 active_transition

- Cells: **18**
- Pooled `N_total`: **8503**
- Pooled conjunction `N`: **1162**
- Base ExpR: **+0.108**
- Conjunction ExpR: **+0.278**

| Check | Support / valid cells |
|---|---:|
| garch marginal | 18/18 |
| partner marginal | 16/18 |
| partner inside garch | 12/18 |
| garch inside partner | 18/18 |

| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | long | COST_LT12 | 623 | +0.100 | +0.244 | +0.149 | 82 | +0.249 | +0.208 | +0.065 | +0.010 | +0.208 | 4 | -0.517 |
| MNQ | long | ORB_G5 | 738 | +0.076 | +0.247 | +0.096 | 83 | +0.255 | +0.233 | +0.025 | +0.015 | +0.233 | 4 | -0.517 |
| MNQ | long | ORB_G5 | 731 | +0.110 | +0.254 | +0.150 | 80 | +0.242 | +0.198 | +0.051 | -0.021 | +0.198 | 3 | -1.000 |
| MNQ | long | ORB_G5 | 721 | +0.132 | +0.316 | +0.234 | 79 | +0.296 | +0.251 | +0.129 | -0.034 | +0.251 | 3 | -1.000 |
| MNQ | long | ORB_G5_NOFRI | 603 | +0.090 | +0.256 | +0.137 | 68 | +0.306 | +0.230 | +0.060 | +0.084 | +0.230 | 3 | -0.356 |
| MNQ | long | OVNRNG_100 | 254 | +0.174 | +0.301 | +0.154 | 66 | +0.270 | +0.254 | -0.033 | -0.066 | +0.254 | 3 | -0.356 |
| MNQ | long | OVNRNG_100 | 249 | +0.204 | +0.348 | +0.186 | 64 | +0.301 | +0.288 | -0.029 | -0.095 | +0.288 | 2 | +nan |
| MNQ | long | X_MES_ATR60 | 330 | +0.203 | +0.294 | +0.230 | 73 | +0.300 | +0.194 | +0.038 | +0.010 | +0.194 | 4 | -0.517 |
| MNQ | long | X_MES_ATR60 | 324 | +0.229 | +0.314 | +0.254 | 70 | +0.286 | +0.182 | +0.036 | -0.048 | +0.182 | 3 | -1.000 |
| MNQ | short | COST_LT12 | 560 | +0.106 | +0.138 | +0.153 | 64 | +0.184 | +0.046 | +0.060 | +0.074 | +0.046 | 0 | +nan |
| MNQ | short | ORB_G5 | 646 | +0.079 | +0.142 | +0.126 | 65 | +0.193 | +0.085 | +0.058 | +0.082 | +0.085 | 0 | +nan |
| MNQ | short | ORB_G5 | 640 | +0.094 | +0.227 | +0.168 | 65 | +0.419 | +0.181 | +0.093 | +0.312 | +0.181 | 0 | +nan |
| MNQ | short | ORB_G5 | 634 | +0.020 | +0.080 | +0.026 | 64 | +0.244 | +0.081 | +0.007 | +0.265 | +0.081 | 0 | +nan |
| MNQ | short | ORB_G5_NOFRI | 507 | +0.102 | +0.233 | +0.214 | 43 | +0.281 | +0.175 | +0.135 | +0.074 | +0.175 | 0 | +nan |
| MNQ | short | OVNRNG_100 | 212 | +0.119 | +0.234 | +0.290 | 47 | +0.210 | +0.230 | +0.243 | -0.043 | +0.230 | 0 | +nan |
| MNQ | short | OVNRNG_100 | 211 | +0.193 | +0.364 | +0.426 | 47 | +0.464 | +0.344 | +0.332 | +0.180 | +0.344 | 0 | +nan |
| MNQ | short | X_MES_ATR60 | 261 | +0.055 | +0.140 | +0.154 | 51 | +0.150 | +0.189 | +0.133 | +0.015 | +0.189 | 0 | +nan |
| MNQ | short | X_MES_ATR60 | 259 | +0.067 | +0.204 | +0.303 | 51 | +0.390 | +0.305 | +0.318 | +0.289 | +0.305 | 0 | +nan |

- Verdict: **complementary_pair**
- Allowed implication: **R7_candidate_only**

#### Descriptive buckets inside garch_high

**MNQ long ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 26 | +0.848 | +22.0 |
| Expanding | 79 | +0.296 | +23.4 |
| Stable | 88 | +0.177 | +15.6 |

**MNQ short ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 25 | +0.344 | +8.6 |
| Expanding | 64 | +0.244 | +15.6 |
| Stable | 79 | -0.136 | -10.8 |

**MNQ long COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 26 | +0.524 | +13.6 |
| Expanding | 82 | +0.249 | +20.4 |
| Stable | 85 | +0.152 | +13.0 |

**MNQ short COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 24 | +0.328 | +7.9 |
| Expanding | 64 | +0.184 | +11.8 |
| Stable | 78 | +0.043 | +3.3 |

**MNQ long OVNRNG_100**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 16 | +0.771 | +12.3 |
| Expanding | 64 | +0.301 | +19.3 |
| Stable | 45 | +0.263 | +11.8 |

**MNQ short OVNRNG_100**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 13 | +0.810 | +10.5 |
| Expanding | 47 | +0.464 | +21.8 |
| Stable | 46 | +0.136 | +6.3 |

**MNQ long X_MES_ATR60**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 26 | +0.720 | +18.7 |
| Expanding | 70 | +0.286 | +20.0 |
| Stable | 76 | +0.202 | +15.3 |

**MNQ short X_MES_ATR60**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 23 | +0.416 | +9.6 |
| Expanding | 51 | +0.390 | +19.9 |
| Stable | 69 | -0.004 | -0.3 |

**MNQ long ORB_G5_NOFRI**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 23 | +0.478 | +11.0 |
| Expanding | 68 | +0.306 | +20.8 |
| Stable | 75 | +0.143 | +10.7 |

**MNQ short ORB_G5_NOFRI**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 22 | +0.359 | +7.9 |
| Expanding | 43 | +0.281 | +12.1 |
| Stable | 62 | +0.154 | +9.6 |

## EUROPE_FLOW_high

### M1 latent_expansion

- Cells: **16**
- Pooled `N_total`: **8206**
- Pooled conjunction `N`: **1568**
- Base ExpR: **+0.097**
- Conjunction ExpR: **+0.095**

| Check | Support / valid cells |
|---|---:|
| garch marginal | 15/16 |
| partner marginal | 3/16 |
| partner inside garch | 3/16 |
| garch inside partner | 15/16 |

| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | long | COST_LT12 | 476 | +0.052 | +0.144 | +0.054 | 106 | +0.100 | +0.138 | +0.005 | -0.124 | +0.138 | 11 | +0.710 |
| MNQ | long | COST_LT12 | 476 | +0.112 | +0.188 | +0.104 | 106 | +0.114 | +0.115 | -0.027 | -0.211 | +0.115 | 11 | +1.138 |
| MNQ | long | ORB_G5 | 673 | +0.052 | +0.094 | +0.045 | 119 | +0.051 | +0.058 | -0.029 | -0.132 | +0.058 | 11 | +0.710 |
| MNQ | long | ORB_G5 | 673 | +0.095 | +0.146 | +0.085 | 119 | +0.080 | +0.070 | -0.043 | -0.200 | +0.070 | 11 | +1.138 |
| MNQ | long | ORB_G5 | 673 | +0.094 | +0.127 | +0.062 | 119 | +0.044 | +0.045 | -0.132 | -0.249 | +0.045 | 11 | +1.301 |
| MNQ | long | ORB_G5_NOFRI | 539 | +0.054 | +0.118 | +0.047 | 93 | +0.086 | +0.085 | -0.034 | -0.094 | +0.085 | 9 | +0.669 |
| MNQ | long | OVNRNG_100 | 228 | +0.079 | +0.130 | +0.030 | 50 | +0.039 | +0.096 | -0.094 | -0.173 | +0.096 | 11 | +0.710 |
| MNQ | long | OVNRNG_100 | 228 | +0.146 | +0.194 | +0.077 | 50 | +0.069 | +0.090 | -0.133 | -0.238 | +0.090 | 11 | +1.138 |
| MNQ | short | COST_LT12 | 494 | +0.137 | +0.171 | +0.131 | 101 | +0.201 | +0.053 | -0.019 | +0.070 | +0.053 | 11 | +0.180 |
| MNQ | short | COST_LT12 | 494 | +0.114 | +0.170 | +0.067 | 101 | +0.111 | +0.087 | -0.169 | -0.142 | +0.087 | 11 | +0.475 |
| MNQ | short | ORB_G5 | 723 | +0.086 | +0.116 | +0.077 | 127 | +0.117 | +0.042 | -0.042 | +0.002 | +0.042 | 12 | +0.228 |
| MNQ | short | ORB_G5 | 723 | +0.069 | +0.089 | +0.033 | 127 | +0.003 | +0.027 | -0.165 | -0.233 | +0.027 | 12 | +0.535 |
| MNQ | short | ORB_G5 | 723 | +0.122 | +0.169 | +0.073 | 127 | +0.008 | +0.065 | -0.225 | -0.437 | +0.065 | 12 | +0.158 |
| MNQ | short | ORB_G5_NOFRI | 581 | +0.074 | +0.140 | +0.070 | 101 | +0.135 | +0.091 | -0.020 | -0.014 | +0.091 | 8 | +0.151 |
| MNQ | short | OVNRNG_100 | 251 | +0.183 | +0.186 | +0.265 | 61 | +0.264 | +0.007 | +0.148 | +0.143 | +0.007 | 11 | +0.339 |
| MNQ | short | OVNRNG_100 | 251 | +0.239 | +0.236 | +0.273 | 61 | +0.233 | -0.007 | +0.063 | -0.005 | -0.007 | 11 | +0.674 |

- Verdict: **garch_distinct**
- Allowed implication: **R3/R7_candidate_only**

#### Descriptive buckets inside garch_high

**MNQ long ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 59 | +0.293 | +17.3 |
| overnight_low | 15 | -0.087 | -1.3 |
| overnight_mid | 104 | +0.063 | +6.6 |

**MNQ short ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 74 | +0.445 | +32.9 |
| overnight_low | 25 | -0.227 | -5.7 |
| overnight_mid | 102 | +0.066 | +6.7 |

**MNQ long COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 57 | +0.325 | +18.5 |
| overnight_low | 13 | +0.054 | +0.7 |
| overnight_mid | 93 | +0.122 | +11.3 |

**MNQ short COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 73 | +0.252 | +18.4 |
| overnight_low | 15 | -0.074 | -1.1 |
| overnight_mid | 86 | +0.143 | +12.3 |

**MNQ long OVNRNG_100**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 56 | +0.306 | +17.2 |
| overnight_mid | 50 | +0.069 | +3.4 |

**MNQ short OVNRNG_100**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 72 | +0.238 | +17.1 |
| overnight_low | 1 | -1.000 | -1.0 |
| overnight_mid | 60 | +0.254 | +15.2 |

**MNQ long ORB_G5_NOFRI**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 48 | +0.180 | +8.6 |
| overnight_low | 8 | -0.317 | -2.5 |
| overnight_mid | 85 | +0.124 | +10.5 |

**MNQ short ORB_G5_NOFRI**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 62 | +0.149 | +9.2 |
| overnight_low | 22 | +0.071 | +1.6 |
| overnight_mid | 79 | +0.152 | +12.0 |

### M2 active_transition

- Cells: **16**
- Pooled `N_total`: **8206**
- Pooled conjunction `N`: **1087**
- Base ExpR: **+0.097**
- Conjunction ExpR: **+0.261**

| Check | Support / valid cells |
|---|---:|
| garch marginal | 15/16 |
| partner marginal | 16/16 |
| partner inside garch | 16/16 |
| garch inside partner | 15/16 |

| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | long | COST_LT12 | 476 | +0.052 | +0.144 | +0.209 | 67 | +0.231 | +0.138 | +0.216 | +0.149 | +0.138 | 2 | +nan |
| MNQ | long | COST_LT12 | 476 | +0.112 | +0.188 | +0.335 | 67 | +0.332 | +0.115 | +0.308 | +0.245 | +0.115 | 2 | +nan |
| MNQ | long | ORB_G5 | 673 | +0.052 | +0.094 | +0.211 | 73 | +0.177 | +0.058 | +0.208 | +0.140 | +0.058 | 2 | +nan |
| MNQ | long | ORB_G5 | 673 | +0.095 | +0.146 | +0.329 | 73 | +0.281 | +0.070 | +0.307 | +0.228 | +0.070 | 2 | +nan |
| MNQ | long | ORB_G5 | 673 | +0.094 | +0.127 | +0.338 | 73 | +0.269 | +0.045 | +0.319 | +0.241 | +0.045 | 2 | +nan |
| MNQ | long | ORB_G5_NOFRI | 539 | +0.054 | +0.118 | +0.191 | 55 | +0.151 | +0.085 | +0.172 | +0.055 | +0.085 | 2 | +nan |
| MNQ | long | OVNRNG_100 | 228 | +0.079 | +0.130 | +0.203 | 52 | +0.155 | +0.096 | +0.193 | +0.049 | +0.096 | 2 | +nan |
| MNQ | long | OVNRNG_100 | 228 | +0.146 | +0.194 | +0.391 | 52 | +0.309 | +0.090 | +0.382 | +0.226 | +0.090 | 2 | +nan |
| MNQ | short | COST_LT12 | 494 | +0.137 | +0.171 | +0.279 | 72 | +0.273 | +0.053 | +0.187 | +0.174 | +0.053 | 2 | +nan |
| MNQ | short | COST_LT12 | 494 | +0.114 | +0.170 | +0.264 | 72 | +0.301 | +0.087 | +0.197 | +0.223 | +0.087 | 2 | +nan |
| MNQ | short | ORB_G5 | 723 | +0.086 | +0.116 | +0.210 | 81 | +0.195 | +0.042 | +0.155 | +0.131 | +0.042 | 2 | +nan |
| MNQ | short | ORB_G5 | 723 | +0.069 | +0.089 | +0.179 | 81 | +0.209 | +0.027 | +0.137 | +0.202 | +0.027 | 2 | +nan |
| MNQ | short | ORB_G5 | 723 | +0.122 | +0.169 | +0.299 | 81 | +0.383 | +0.065 | +0.220 | +0.358 | +0.065 | 2 | +nan |
| MNQ | short | ORB_G5_NOFRI | 581 | +0.074 | +0.140 | +0.250 | 62 | +0.266 | +0.091 | +0.212 | +0.202 | +0.091 | 1 | +nan |
| MNQ | short | OVNRNG_100 | 251 | +0.183 | +0.186 | +0.268 | 63 | +0.282 | +0.007 | +0.125 | +0.181 | +0.007 | 1 | +nan |
| MNQ | short | OVNRNG_100 | 251 | +0.239 | +0.236 | +0.291 | 63 | +0.340 | -0.007 | +0.076 | +0.198 | -0.007 | 1 | +nan |

- Verdict: **unclear**
- Allowed implication: **unclear**

#### Descriptive buckets inside garch_high

**MNQ long ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 21 | +0.070 | +1.5 |
| Expanding | 73 | +0.269 | +19.6 |
| Stable | 84 | +0.018 | +1.5 |

**MNQ short ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 33 | +0.265 | +8.7 |
| Expanding | 81 | +0.383 | +31.0 |
| Stable | 87 | -0.066 | -5.8 |

**MNQ long COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 21 | +0.224 | +4.7 |
| Expanding | 67 | +0.332 | +22.3 |
| Stable | 75 | +0.048 | +3.6 |

**MNQ short COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 27 | +0.389 | +10.5 |
| Expanding | 72 | +0.301 | +21.7 |
| Stable | 75 | -0.034 | -2.5 |

**MNQ long OVNRNG_100**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 11 | +0.070 | +0.8 |
| Expanding | 52 | +0.309 | +16.1 |
| Stable | 43 | +0.087 | +3.7 |

**MNQ short OVNRNG_100**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 20 | +0.416 | +8.3 |
| Expanding | 63 | +0.340 | +21.4 |
| Stable | 50 | +0.032 | +1.6 |

**MNQ long ORB_G5_NOFRI**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 18 | +0.352 | +6.3 |
| Expanding | 55 | +0.151 | +8.3 |
| Stable | 68 | +0.028 | +1.9 |

**MNQ short ORB_G5_NOFRI**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 30 | +0.170 | +5.1 |
| Expanding | 62 | +0.266 | +16.5 |
| Stable | 71 | +0.018 | +1.3 |

## TOKYO_OPEN_high

### M1 latent_expansion

- Cells: **8**
- Pooled `N_total`: **4492**
- Pooled conjunction `N`: **874**
- Base ExpR: **+0.082**
- Conjunction ExpR: **-0.036**

| Check | Support / valid cells |
|---|---:|
| garch marginal | 8/8 |
| partner marginal | 0/8 |
| partner inside garch | 0/8 |
| garch inside partner | 8/8 |

| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | long | COST_LT12 | 420 | +0.060 | +0.064 | +0.032 | 96 | -0.020 | +0.005 | -0.100 | -0.227 | +0.005 | 14 | -0.053 |
| MNQ | long | COST_LT12 | 420 | +0.096 | +0.133 | +0.015 | 96 | -0.063 | +0.057 | -0.286 | -0.530 | +0.057 | 14 | +0.018 |
| MNQ | long | ORB_G5 | 682 | +0.083 | +0.139 | +0.031 | 119 | -0.049 | +0.076 | -0.226 | -0.560 | +0.076 | 14 | +0.018 |
| MNQ | long | ORB_G5 | 682 | +0.102 | +0.197 | +0.035 | 119 | +0.005 | +0.128 | -0.293 | -0.571 | +0.128 | 14 | +0.015 |
| MNQ | short | COST_LT12 | 432 | +0.092 | +0.169 | +0.010 | 94 | +0.022 | +0.124 | -0.294 | -0.351 | +0.124 | 9 | +0.273 |
| MNQ | short | COST_LT12 | 432 | +0.130 | +0.191 | +0.013 | 94 | -0.041 | +0.097 | -0.422 | -0.552 | +0.097 | 9 | +0.325 |
| MNQ | short | ORB_G5 | 712 | +0.072 | +0.120 | -0.019 | 128 | -0.062 | +0.066 | -0.410 | -0.500 | +0.066 | 9 | +0.325 |
| MNQ | short | ORB_G5 | 712 | +0.043 | +0.111 | -0.082 | 128 | -0.065 | +0.095 | -0.563 | -0.484 | +0.095 | 9 | +0.589 |

- Verdict: **garch_distinct**
- Allowed implication: **R3/R7_candidate_only**

#### Descriptive buckets inside garch_high

**MNQ long COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 56 | +0.468 | +26.2 |
| overnight_low | 8 | -0.431 | -3.4 |
| overnight_mid | 88 | -0.029 | -2.6 |

**MNQ short COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 68 | +0.511 | +34.8 |
| overnight_low | 13 | -0.293 | -3.8 |
| overnight_mid | 81 | -0.001 | -0.1 |

**MNQ long ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 60 | +0.576 | +34.6 |
| overnight_low | 14 | -0.433 | -6.1 |
| overnight_mid | 105 | +0.064 | +6.7 |

**MNQ short ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 73 | +0.419 | +30.6 |
| overnight_low | 27 | -0.311 | -8.4 |
| overnight_mid | 101 | +0.001 | +0.1 |

### M2 active_transition

- Cells: **8**
- Pooled `N_total`: **4492**
- Pooled conjunction `N`: **576**
- Base ExpR: **+0.082**
- Conjunction ExpR: **+0.283**

| Check | Support / valid cells |
|---|---:|
| garch marginal | 8/8 |
| partner marginal | 8/8 |
| partner inside garch | 8/8 |
| garch inside partner | 8/8 |

| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | long | COST_LT12 | 420 | +0.060 | +0.064 | +0.151 | 59 | +0.203 | +0.005 | +0.122 | +0.227 | +0.005 | 2 | +nan |
| MNQ | long | COST_LT12 | 420 | +0.096 | +0.133 | +0.222 | 59 | +0.385 | +0.057 | +0.169 | +0.412 | +0.057 | 2 | +nan |
| MNQ | long | ORB_G5 | 682 | +0.083 | +0.139 | +0.258 | 68 | +0.454 | +0.076 | +0.221 | +0.508 | +0.076 | 2 | +nan |
| MNQ | long | ORB_G5 | 682 | +0.102 | +0.197 | +0.333 | 68 | +0.578 | +0.128 | +0.292 | +0.615 | +0.128 | 2 | +nan |
| MNQ | short | COST_LT12 | 432 | +0.092 | +0.169 | +0.248 | 75 | +0.190 | +0.124 | +0.221 | +0.038 | +0.124 | 2 | +nan |
| MNQ | short | COST_LT12 | 432 | +0.130 | +0.191 | +0.268 | 75 | +0.242 | +0.097 | +0.195 | +0.096 | +0.097 | 2 | +nan |
| MNQ | short | ORB_G5 | 712 | +0.072 | +0.120 | +0.145 | 86 | +0.134 | +0.066 | +0.094 | +0.025 | +0.066 | 2 | +nan |
| MNQ | short | ORB_G5 | 712 | +0.043 | +0.111 | +0.080 | 86 | +0.163 | +0.095 | +0.048 | +0.091 | +0.095 | 2 | +nan |

- Verdict: **complementary_pair**
- Allowed implication: **R7_candidate_only**

#### Descriptive buckets inside garch_high

**MNQ long COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 25 | -0.348 | -8.7 |
| Expanding | 59 | +0.385 | +22.7 |
| Stable | 68 | +0.091 | +6.2 |

**MNQ short COST_LT12**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 20 | -0.413 | -8.3 |
| Expanding | 75 | +0.242 | +18.2 |
| Stable | 67 | +0.313 | +21.0 |

**MNQ long ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 28 | -0.120 | -3.3 |
| Expanding | 68 | +0.578 | +39.3 |
| Stable | 83 | -0.009 | -0.8 |

**MNQ short ORB_G5**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 26 | -0.567 | -14.7 |
| Expanding | 86 | +0.163 | +14.1 |
| Stable | 89 | +0.259 | +23.0 |

## SINGAPORE_OPEN_high

### M1 latent_expansion

- Cells: **4**
- Pooled `N_total`: **1547**
- Pooled conjunction `N`: **468**
- Base ExpR: **+0.121**
- Conjunction ExpR: **+0.078**

| Check | Support / valid cells |
|---|---:|
| garch marginal | 4/4 |
| partner marginal | 0/4 |
| partner inside garch | 1/4 |
| garch inside partner | 4/4 |

| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | long | ATR_P50 | 416 | +0.207 | +0.246 | +0.176 | 122 | +0.192 | +0.074 | -0.105 | -0.142 | +0.074 | 13 | -0.090 |
| MNQ | long | ATR_P50 | 412 | +0.216 | +0.230 | +0.198 | 123 | +0.264 | +0.026 | -0.066 | +0.091 | +0.026 | 15 | -0.038 |
| MNQ | short | ATR_P50 | 361 | +0.015 | +0.036 | -0.076 | 113 | -0.091 | +0.039 | -0.354 | -0.386 | +0.039 | 10 | -0.044 |
| MNQ | short | ATR_P50 | 358 | +0.019 | +0.034 | -0.092 | 110 | -0.082 | +0.028 | -0.406 | -0.347 | +0.028 | 8 | +0.201 |

- Verdict: **garch_distinct**
- Allowed implication: **R3/R7_candidate_only**

#### Descriptive buckets inside garch_high

**MNQ long ATR_P50**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 74 | +0.173 | +12.8 |
| overnight_low | 26 | -0.028 | -0.7 |
| overnight_mid | 97 | +0.342 | +33.2 |

**MNQ short ATR_P50**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| overnight_high | 55 | +0.266 | +14.6 |
| overnight_low | 12 | -0.429 | -5.1 |
| overnight_mid | 98 | -0.039 | -3.8 |

### M2 active_transition

- Cells: **4**
- Pooled `N_total`: **1547**
- Pooled conjunction `N`: **294**
- Base ExpR: **+0.121**
- Conjunction ExpR: **+0.238**

| Check | Support / valid cells |
|---|---:|
| garch marginal | 4/4 |
| partner marginal | 3/4 |
| partner inside garch | 2/4 |
| garch inside partner | 4/4 |

| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | long | ATR_P50 | 416 | +0.207 | +0.246 | +0.282 | 77 | +0.464 | +0.074 | +0.101 | +0.358 | +0.074 | 1 | +nan |
| MNQ | long | ATR_P50 | 412 | +0.216 | +0.230 | +0.159 | 80 | +0.226 | +0.026 | -0.077 | -0.006 | +0.026 | 2 | +nan |
| MNQ | short | ATR_P50 | 361 | +0.015 | +0.036 | +0.186 | 71 | +0.217 | +0.039 | +0.241 | +0.315 | +0.039 | 3 | -1.000 |
| MNQ | short | ATR_P50 | 358 | +0.019 | +0.034 | +0.061 | 66 | +0.013 | +0.028 | +0.059 | -0.036 | +0.028 | 2 | +nan |

- Verdict: **complementary_pair**
- Allowed implication: **R7_candidate_only**

#### Descriptive buckets inside garch_high

**MNQ long ATR_P50**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 26 | +0.359 | +9.3 |
| Expanding | 80 | +0.226 | +18.1 |
| Stable | 91 | +0.196 | +17.9 |

**MNQ short ATR_P50**

| Bucket | N | ExpR | Total R |
|---|---:|---:|---:|
| Contracting | 26 | +0.085 | +2.2 |
| Expanding | 66 | +0.013 | +0.8 |
| Stable | 73 | +0.035 | +2.6 |

## Guardrails

- This stage uses raw validated rows rejoined to canonical layers; prior report summaries are not authority.
- `garch_distinct` here means the partner did not add enough on the validated shelf, not that garch is a proven deployment edge.
- `partner_dominant` means the partner explains more of the local family utility than garch.
- 2026 OOS rows are descriptive only and were not used to choose verdicts.
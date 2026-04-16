# Garch W2c M2 Validated Utility Audit

**Date:** 2026-04-16
**Boundary:** validated shelf only, conservative family-local M2 representation, no deployment conclusions.

- Stage verdict: **M2_carry**
- Allowed implication: **at_least_one_clean_local_family**

| Family | Representation | Cells | N total | N garch | N conj | Base ExpR | Garch ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | P|G support | G|P support | Max conj share | OOS conj N | OOS conj ExpR | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| COMEX_SETTLE_high | ATRVEL_EXPANDING | 18 | 8503 | 2843 | 1162 | +0.108 | +0.238 | +0.278 | +0.170 | +0.039 | 0.68 | 1.00 | 0.07 | 29 | -0.642 | carry_local_m2 |
| EUROPE_FLOW_high | ATRVEL_EXPANDING | 16 | 8206 | 2593 | 1087 | +0.097 | +0.148 | +0.261 | +0.165 | +0.113 | 1.00 | 0.97 | 0.07 | 29 |  | carry_local_m2 |
| SINGAPORE_OPEN_high | ATRVEL_GE_110 | 4 | 1547 | 727 | 142 | +0.121 | +0.145 | +0.406 | +0.285 | +0.261 | 1.00 | 1.00 | 0.26 | 6 | -1.000 | carry_local_m2 |
| TOKYO_OPEN_high | ATRVEL_EXPANDING | 8 | 4492 | 1388 | 576 | +0.082 | +0.140 | +0.283 | +0.200 | +0.142 | 1.00 | 1.00 | 0.15 | 16 |  | carry_local_m2 |

## Per-cell detail

### COMEX_SETTLE_high

| Instrument | ORB | Dir | Filter | RR | N | N garch | N conj | Base ExpR | Garch ExpR | Conj ExpR | Δ conj-garch |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | 5 | long | COST_LT12 | 1.0 | 623 | 193 | 82 | +0.100 | +0.244 | +0.249 | +0.006 |
| MNQ | 5 | long | ORB_G5 | 1.0 | 738 | 198 | 83 | +0.076 | +0.247 | +0.255 | +0.009 |
| MNQ | 5 | long | ORB_G5 | 1.5 | 731 | 195 | 80 | +0.110 | +0.254 | +0.242 | -0.012 |
| MNQ | 5 | long | ORB_G5 | 2.0 | 721 | 193 | 79 | +0.132 | +0.316 | +0.296 | -0.020 |
| MNQ | 5 | long | ORB_G5_NOFRI | 1.0 | 603 | 166 | 68 | +0.090 | +0.256 | +0.306 | +0.049 |
| MNQ | 5 | long | OVNRNG_100 | 1.0 | 254 | 127 | 66 | +0.174 | +0.301 | +0.270 | -0.032 |
| MNQ | 5 | long | OVNRNG_100 | 1.5 | 249 | 125 | 64 | +0.204 | +0.348 | +0.301 | -0.046 |
| MNQ | 5 | long | X_MES_ATR60 | 1.0 | 330 | 175 | 73 | +0.203 | +0.294 | +0.300 | +0.006 |
| MNQ | 5 | long | X_MES_ATR60 | 1.5 | 324 | 172 | 70 | +0.229 | +0.314 | +0.286 | -0.029 |
| MNQ | 5 | short | COST_LT12 | 1.0 | 560 | 166 | 64 | +0.106 | +0.138 | +0.184 | +0.046 |
| MNQ | 5 | short | ORB_G5 | 1.0 | 646 | 170 | 65 | +0.079 | +0.142 | +0.193 | +0.051 |
| MNQ | 5 | short | ORB_G5 | 1.5 | 640 | 169 | 65 | +0.094 | +0.227 | +0.419 | +0.192 |
| MNQ | 5 | short | ORB_G5 | 2.0 | 634 | 168 | 64 | +0.020 | +0.080 | +0.244 | +0.164 |
| MNQ | 5 | short | ORB_G5_NOFRI | 1.0 | 507 | 127 | 43 | +0.102 | +0.233 | +0.281 | +0.049 |
| MNQ | 5 | short | OVNRNG_100 | 1.0 | 212 | 106 | 47 | +0.119 | +0.234 | +0.210 | -0.024 |
| MNQ | 5 | short | OVNRNG_100 | 1.5 | 211 | 106 | 47 | +0.193 | +0.364 | +0.464 | +0.100 |
| MNQ | 5 | short | X_MES_ATR60 | 1.0 | 261 | 144 | 51 | +0.055 | +0.140 | +0.150 | +0.010 |
| MNQ | 5 | short | X_MES_ATR60 | 1.5 | 259 | 143 | 51 | +0.067 | +0.204 | +0.390 | +0.186 |

### EUROPE_FLOW_high

| Instrument | ORB | Dir | Filter | RR | N | N garch | N conj | Base ExpR | Garch ExpR | Conj ExpR | Δ conj-garch |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | 5 | long | COST_LT12 | 1.0 | 476 | 163 | 67 | +0.052 | +0.144 | +0.231 | +0.088 |
| MNQ | 5 | long | COST_LT12 | 1.5 | 476 | 163 | 67 | +0.112 | +0.188 | +0.332 | +0.145 |
| MNQ | 5 | long | ORB_G5 | 1.0 | 673 | 178 | 73 | +0.052 | +0.094 | +0.177 | +0.083 |
| MNQ | 5 | long | ORB_G5 | 1.5 | 673 | 178 | 73 | +0.095 | +0.146 | +0.281 | +0.135 |
| MNQ | 5 | long | ORB_G5 | 2.0 | 673 | 178 | 73 | +0.094 | +0.127 | +0.269 | +0.142 |
| MNQ | 5 | long | ORB_G5_NOFRI | 1.0 | 539 | 141 | 55 | +0.054 | +0.118 | +0.151 | +0.034 |
| MNQ | 5 | long | OVNRNG_100 | 1.0 | 228 | 106 | 52 | +0.079 | +0.130 | +0.155 | +0.025 |
| MNQ | 5 | long | OVNRNG_100 | 1.5 | 228 | 106 | 52 | +0.146 | +0.194 | +0.309 | +0.115 |
| MNQ | 5 | short | COST_LT12 | 1.0 | 494 | 174 | 72 | +0.137 | +0.171 | +0.273 | +0.102 |
| MNQ | 5 | short | COST_LT12 | 1.5 | 494 | 174 | 72 | +0.114 | +0.170 | +0.301 | +0.131 |
| MNQ | 5 | short | ORB_G5 | 1.0 | 723 | 201 | 81 | +0.086 | +0.116 | +0.195 | +0.078 |
| MNQ | 5 | short | ORB_G5 | 1.5 | 723 | 201 | 81 | +0.069 | +0.089 | +0.209 | +0.121 |
| MNQ | 5 | short | ORB_G5 | 2.0 | 723 | 201 | 81 | +0.122 | +0.169 | +0.383 | +0.214 |
| MNQ | 5 | short | ORB_G5_NOFRI | 1.0 | 581 | 163 | 62 | +0.074 | +0.140 | +0.266 | +0.125 |
| MNQ | 5 | short | OVNRNG_100 | 1.0 | 251 | 133 | 63 | +0.183 | +0.186 | +0.282 | +0.095 |
| MNQ | 5 | short | OVNRNG_100 | 1.5 | 251 | 133 | 63 | +0.239 | +0.236 | +0.340 | +0.104 |

### TOKYO_OPEN_high

| Instrument | ORB | Dir | Filter | RR | N | N garch | N conj | Base ExpR | Garch ExpR | Conj ExpR | Δ conj-garch |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | 5 | long | COST_LT12 | 1.0 | 420 | 152 | 59 | +0.060 | +0.064 | +0.203 | +0.139 |
| MNQ | 5 | long | COST_LT12 | 1.5 | 420 | 152 | 59 | +0.096 | +0.133 | +0.385 | +0.252 |
| MNQ | 5 | long | ORB_G5 | 1.5 | 682 | 179 | 68 | +0.083 | +0.139 | +0.454 | +0.315 |
| MNQ | 5 | long | ORB_G5 | 2.0 | 682 | 179 | 68 | +0.102 | +0.197 | +0.578 | +0.382 |
| MNQ | 5 | short | COST_LT12 | 1.0 | 432 | 162 | 75 | +0.092 | +0.169 | +0.190 | +0.020 |
| MNQ | 5 | short | COST_LT12 | 1.5 | 432 | 162 | 75 | +0.130 | +0.191 | +0.242 | +0.052 |
| MNQ | 5 | short | ORB_G5 | 1.5 | 712 | 201 | 86 | +0.072 | +0.120 | +0.134 | +0.014 |
| MNQ | 5 | short | ORB_G5 | 2.0 | 712 | 201 | 86 | +0.043 | +0.111 | +0.163 | +0.052 |

### SINGAPORE_OPEN_high

| Instrument | ORB | Dir | Filter | RR | N | N garch | N conj | Base ExpR | Garch ExpR | Conj ExpR | Δ conj-garch |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ | 15 | long | ATR_P50 | 1.5 | 416 | 197 | 35 | +0.207 | +0.246 | +0.624 | +0.378 |
| MNQ | 15 | short | ATR_P50 | 1.5 | 361 | 168 | 36 | +0.015 | +0.036 | +0.437 | +0.402 |
| MNQ | 30 | long | ATR_P50 | 1.5 | 412 | 197 | 37 | +0.216 | +0.230 | +0.425 | +0.195 |
| MNQ | 30 | short | ATR_P50 | 1.5 | 358 | 165 | 34 | +0.019 | +0.034 | +0.129 | +0.094 |

## Guardrails

- Family representation choice was frozen from the completed provenance rule; no new representation search happened in this stage.
- `neighbor_stable` families kept the current canonical representation; only explicit `alternate_better` families switched.
- 2026 OOS remains descriptive only.
- This is still validated utility only, not deployment doctrine.
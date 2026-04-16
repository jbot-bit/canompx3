# Garch Partner-State Provenance Audit

**Date:** 2026-04-16
**Boundary:** validated shelf only, exact canonical joins, no deployment conclusions.

This stage asks whether the current W2 partner encodings are principled local mechanism representations, or whether a nearby / alternate locked representation is better supported.

## COMEX_SETTLE_high

### M1 latent_expansion

- Verdict: **neighbor_stable**
- Allowed implication: **carry_mechanism_no_unique_cutoff**

| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OVN_NOT_HIGH_80 | yes | 18 | 8503 | 1802 | +0.108 | +0.190 | +0.082 | 0.54 | 1.00 | 177 | +0.477 |
| OVN_NOT_HIGH_60 | no | 18 | 8503 | 1112 | +0.108 | +0.233 | +0.125 | 0.57 | 1.00 | 111 | +0.068 |
| OVN_MID_ONLY | no | 18 | 8503 | 1540 | +0.108 | +0.188 | +0.080 | 0.42 | 1.00 | 177 | +0.477 |
| OVN_NOT_HIGH_70 | no | 18 | 8503 | 1412 | +0.108 | +0.177 | +0.069 | 0.28 | 1.00 | 168 | +0.439 |

### M2 active_transition

- Verdict: **neighbor_stable**
- Allowed implication: **carry_mechanism_no_unique_cutoff**

| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATRVEL_EXPANDING | yes | 18 | 8503 | 1162 | +0.108 | +0.278 | +0.170 | 0.68 | 1.00 | 29 | -0.642 |
| ATR_PCT_GE_70 | no | 18 | 8503 | 2351 | +0.108 | +0.284 | +0.177 | 0.94 | 1.00 | 279 | +0.371 |
| ATRVEL_GE_110 | no | 18 | 8503 | 568 | +0.108 | +0.278 | +0.170 | 0.60 | 1.00 | 22 | -0.356 |
| ATRVEL_GE_105 | no | 18 | 8503 | 1162 | +0.108 | +0.278 | +0.170 | 0.68 | 1.00 | 29 | -0.642 |
| ATR_PCT_GE_80 | no | 18 | 8503 | 1897 | +0.108 | +0.256 | +0.148 | 0.65 | 1.00 | 270 | +0.381 |
| ATRVEL_GE_100 | no | 18 | 8503 | 1913 | +0.108 | +0.191 | +0.083 | 0.00 | 1.00 | 182 | +0.229 |

## EUROPE_FLOW_high

### M1 latent_expansion

- Verdict: **alternate_better**
- Allowed implication: **current_demoted_locked_alternate:OVN_MID_ONLY**

| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OVN_NOT_HIGH_80 | yes | 16 | 8206 | 1568 | +0.097 | +0.095 | -0.002 | 0.20 | 0.97 | 174 | +0.643 |
| OVN_MID_ONLY | no | 16 | 8206 | 1360 | +0.097 | +0.123 | +0.026 | 0.51 | 0.97 | 167 | +0.622 |
| OVN_NOT_HIGH_60 | no | 16 | 8206 | 967 | +0.097 | +0.028 | -0.069 | 0.09 | 0.97 | 113 | +0.744 |
| OVN_NOT_HIGH_70 | no | 16 | 8206 | 1221 | +0.097 | +0.017 | -0.080 | 0.00 | 0.97 | 166 | +0.639 |

### M2 active_transition

- Verdict: **neighbor_stable**
- Allowed implication: **carry_mechanism_no_unique_cutoff**

| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATRVEL_EXPANDING | yes | 16 | 8206 | 1087 | +0.097 | +0.261 | +0.165 | 1.00 | 0.97 | 29 |  |
| ATRVEL_GE_110 | no | 16 | 8206 | 552 | +0.097 | +0.294 | +0.198 | 0.90 | 0.97 | 23 |  |
| ATRVEL_GE_105 | no | 16 | 8206 | 1087 | +0.097 | +0.261 | +0.165 | 1.00 | 0.97 | 29 |  |
| ATR_PCT_GE_80 | no | 16 | 8206 | 1692 | +0.097 | +0.171 | +0.074 | 0.47 | 0.97 | 260 | +0.509 |
| ATRVEL_GE_100 | no | 16 | 8206 | 1736 | +0.097 | +0.162 | +0.065 | 0.69 | 0.97 | 175 | +0.484 |
| ATR_PCT_GE_70 | no | 16 | 8206 | 2111 | +0.097 | +0.159 | +0.062 | 0.47 | 0.97 | 268 | +0.529 |

## TOKYO_OPEN_high

### M1 latent_expansion

- Verdict: **weak_mechanism**
- Allowed implication: **no_viable_representation**

| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OVN_NOT_HIGH_80 | yes | 8 | 4492 | 874 | +0.082 | -0.036 | -0.118 | 0.00 | 1.00 | 92 | +0.147 |
| OVN_MID_ONLY | no | 8 | 4492 | 750 | +0.082 | +0.013 | -0.069 | 0.00 | 1.00 | 88 | +0.090 |
| OVN_NOT_HIGH_60 | no | 8 | 4492 | 550 | +0.082 | -0.129 | -0.212 | 0.00 | 1.00 | 60 | +0.094 |
| OVN_NOT_HIGH_70 | no | 8 | 4492 | 688 | +0.082 | -0.130 | -0.213 | 0.00 | 1.00 | 88 | +0.093 |

### M2 active_transition

- Verdict: **neighbor_stable**
- Allowed implication: **carry_mechanism_no_unique_cutoff**

| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATRVEL_EXPANDING | yes | 8 | 4492 | 576 | +0.082 | +0.283 | +0.200 | 1.00 | 1.00 | 16 |  |
| ATRVEL_GE_105 | no | 8 | 4492 | 576 | +0.082 | +0.283 | +0.200 | 1.00 | 1.00 | 16 |  |
| ATRVEL_GE_110 | no | 8 | 4492 | 282 | +0.082 | +0.198 | +0.116 | 0.48 | 1.00 | 12 |  |
| ATRVEL_GE_100 | no | 8 | 4492 | 934 | +0.082 | +0.173 | +0.090 | 0.88 | 1.00 | 92 | -0.061 |
| ATR_PCT_GE_70 | no | 8 | 4492 | 1128 | +0.082 | +0.111 | +0.028 | 0.00 | 1.00 | 140 | +0.078 |
| ATR_PCT_GE_80 | no | 8 | 4492 | 916 | +0.082 | +0.083 | +0.001 | 0.00 | 1.00 | 136 | +0.038 |

## SINGAPORE_OPEN_high

### M1 latent_expansion

- Verdict: **alternate_better**
- Allowed implication: **current_demoted_locked_alternate:OVN_MID_ONLY**

| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OVN_NOT_HIGH_80 | yes | 4 | 1547 | 468 | +0.121 | +0.078 | -0.043 | 0.27 | 1.00 | 46 | -0.012 |
| OVN_MID_ONLY | no | 4 | 1547 | 390 | +0.121 | +0.137 | +0.016 | 0.54 | 1.00 | 44 | -0.075 |
| OVN_NOT_HIGH_70 | no | 4 | 1547 | 378 | +0.121 | +0.043 | -0.078 | 0.27 | 1.00 | 44 | -0.074 |
| OVN_NOT_HIGH_60 | no | 4 | 1547 | 306 | +0.121 | +0.011 | -0.110 | 0.00 | 1.00 | 30 | -0.201 |

### M2 active_transition

- Verdict: **alternate_better**
- Allowed implication: **locked_alternate_beats_current:ATRVEL_GE_110**

| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATRVEL_EXPANDING | yes | 4 | 1547 | 294 | +0.121 | +0.238 | +0.117 | 0.50 | 1.00 | 8 | -1.000 |
| ATRVEL_GE_110 | no | 4 | 1547 | 142 | +0.121 | +0.406 | +0.285 | 1.00 | 1.00 | 6 | -1.000 |
| ATRVEL_GE_105 | no | 4 | 1547 | 294 | +0.121 | +0.238 | +0.117 | 0.50 | 1.00 | 8 | -1.000 |
| ATRVEL_GE_100 | no | 4 | 1547 | 482 | +0.121 | +0.145 | +0.024 | 0.27 | 1.00 | 46 | -0.167 |
| ATR_PCT_GE_70 | no | 4 | 1547 | 611 | +0.121 | +0.138 | +0.017 | 0.27 | 1.00 | 70 | -0.037 |
| ATR_PCT_GE_80 | no | 4 | 1547 | 493 | +0.121 | +0.109 | -0.012 | 0.23 | 1.00 | 68 | -0.008 |

## Guardrails

- This is a representation audit, not a new discovery sweep.
- 2026 OOS context is descriptive only and was not used to choose the preferred representation.
- `supported_current` means the current W2 representation remains defensible locally; it does not mean deployment-ready.
- `alternate_better` means a locked neighboring or alternate representation beat the current W2 state on the validated shelf.
- Prior-day levels and prior-session carry remain in the queue, but were intentionally kept out of this stage.
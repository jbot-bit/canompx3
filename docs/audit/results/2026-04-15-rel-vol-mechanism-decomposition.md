# rel_vol_HIGH_Q3 Mechanism Decomposition

**Date:** 2026-04-15
**Source:** 5 BH-global survivor lanes from comprehensive scan
**Purpose:** empirical test of whether rel_vol_HIGH is INDEPENDENT of vol-regime / day-type / calendar proxies, or is a re-labeled version of one of them. No narrative injection — raw numbers only.

## MES COMEX_SETTLE O5 RR1.0 short
**N_is:** 759 | **N_on_is (rel_vol_HIGH fire):** 251 | **Fire rate:** 33.1%

### Correlation of rel_vol_HIGH with candidate proxies (IS data)

| Indicator | corr vs rel_vol_HIGH |
|-----------|----------------------|
| is_mon | -0.127 |
| atr_vel_HIGH | +0.115 |
| is_nfp | +0.046 |
| gap_down | -0.046 |
| garch_vol_pct_LT30 | +0.014 |
| gap_up | -0.012 |
| is_opex | -0.009 |
| atr_20_pct_GT80 | +0.005 |
| garch_vol_pct_GT70 | -0.004 |
| is_fri | +0.003 |
| atr_20_pct_LT20 | -0.001 |

### Partial dependence — does rel_vol_HIGH predict pnl_r within subsets?

**Stratified on atr_vel_HIGH:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_vel_HIGH=0 | 149 | 360 | +0.081 | -0.256 | +0.336 | +4.18 | 0.0000 |
| atr_vel_HIGH=1 | 102 | 148 | +0.087 | -0.080 | +0.167 | +1.54 | 0.1260 |

**Stratified on atr_20_pct_GT80:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_20_pct_GT80=0 | 196 | 399 | +0.092 | -0.214 | +0.306 | +4.29 | 0.0000 |
| atr_20_pct_GT80=1 | 55 | 109 | +0.052 | -0.170 | +0.222 | +1.50 | 0.1355 |

**Stratified on garch_vol_pct_GT70:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| garch_vol_pct_GT70=0 | 202 | 407 | +0.048 | -0.215 | +0.263 | +3.72 | 0.0002 |
| garch_vol_pct_GT70=1 | 49 | 101 | +0.230 | -0.161 | +0.390 | +2.58 | 0.0114 |

**Stratified on gap_up:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| gap_up=0 | 246 | 496 | +0.084 | -0.195 | +0.279 | +4.30 | 0.0000 |
| gap_up=1 | 5 | 12 | +nan | +nan | +nan | +nan | nan |

**Stratified on is_nfp:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| is_nfp=0 | 236 | 488 | +0.086 | -0.207 | +0.293 | +4.44 | 0.0000 |
| is_nfp=1 | 15 | 20 | +nan | +nan | +nan | +nan | nan |

### Multivariate regression
`pnl_r ~ rel_vol_HIGH + atr_vel_HIGH + atr_20_pct_GT80 + day_type indicators`

- **rel_vol_HIGH coefficient:** +0.2724
- **rel_vol_HIGH t-stat:** +4.23
- **rel_vol_HIGH p:** 0.0000
- **Adj R²:** 0.0246
- **N used:** 759
- **Covariates:** atr_vel_HIGH, atr_20_pct_GT80, garch_vol_pct_GT70, gap_up, gap_down, is_nfp, is_opex

**Interpretation of rel_vol_HIGH coefficient:**
  - coefficient > 0 AND |t| >= 2.0: rel_vol_HIGH adds INDEPENDENT predictive power beyond the controls.
  - coefficient near 0 OR |t| < 2.0: rel_vol_HIGH effect is MEDIATED (absorbed) by the controls.

---

## MGC LONDON_METALS O5 RR1.0 short
**N_is:** 445 | **N_on_is (rel_vol_HIGH fire):** 147 | **Fire rate:** 33.0%

### Correlation of rel_vol_HIGH with candidate proxies (IS data)

| Indicator | corr vs rel_vol_HIGH |
|-----------|----------------------|
| is_opex | +0.086 |
| atr_vel_HIGH | +0.086 |
| gap_up | -0.074 |
| atr_20_pct_LT20 | -0.071 |
| is_mon | +0.060 |
| is_nfp | -0.052 |
| atr_20_pct_GT80 | +0.038 |
| garch_vol_pct_GT70 | -0.037 |
| garch_vol_pct_LT30 | +0.027 |
| is_fri | -0.005 |
| gap_down | +0.000 |

### Partial dependence — does rel_vol_HIGH predict pnl_r within subsets?

**Stratified on atr_vel_HIGH:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_vel_HIGH=0 | 90 | 208 | +0.111 | -0.271 | +0.382 | +4.21 | 0.0000 |
| atr_vel_HIGH=1 | 57 | 90 | +0.129 | -0.139 | +0.268 | +2.01 | 0.0462 |

**Stratified on atr_20_pct_GT80:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_20_pct_GT80=0 | 95 | 204 | +0.121 | -0.286 | +0.406 | +4.56 | 0.0000 |
| atr_20_pct_GT80=1 | 52 | 94 | +0.112 | -0.113 | +0.226 | +1.64 | 0.1032 |

**Stratified on garch_vol_pct_GT70:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| garch_vol_pct_GT70=0 | 113 | 219 | +0.087 | -0.258 | +0.345 | +4.06 | 0.0001 |
| garch_vol_pct_GT70=1 | 34 | 79 | +0.222 | -0.157 | +0.379 | +2.38 | 0.0203 |

**Stratified on gap_up:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| gap_up=0 | 146 | 289 | +0.115 | -0.230 | +0.346 | +4.57 | 0.0000 |
| gap_up=1 | 1 | 9 | +nan | +nan | +nan | +nan | nan |

**Stratified on is_nfp:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| is_nfp=0 | 144 | 286 | +0.131 | -0.236 | +0.368 | +4.85 | 0.0000 |
| is_nfp=1 | 3 | 12 | +nan | +nan | +nan | +nan | nan |

### Multivariate regression
`pnl_r ~ rel_vol_HIGH + atr_vel_HIGH + atr_20_pct_GT80 + day_type indicators`

- **rel_vol_HIGH coefficient:** +0.3547
- **rel_vol_HIGH t-stat:** +4.62
- **rel_vol_HIGH p:** 0.0000
- **Adj R²:** 0.0472
- **N used:** 445
- **Covariates:** atr_vel_HIGH, atr_20_pct_GT80, garch_vol_pct_GT70, gap_up, is_nfp, is_opex

**Interpretation of rel_vol_HIGH coefficient:**
  - coefficient > 0 AND |t| >= 2.0: rel_vol_HIGH adds INDEPENDENT predictive power beyond the controls.
  - coefficient near 0 OR |t| < 2.0: rel_vol_HIGH effect is MEDIATED (absorbed) by the controls.

---

## MES TOKYO_OPEN O5 RR1.5 long
**N_is:** 841 | **N_on_is (rel_vol_HIGH fire):** 278 | **Fire rate:** 33.1%

### Correlation of rel_vol_HIGH with candidate proxies (IS data)

| Indicator | corr vs rel_vol_HIGH |
|-----------|----------------------|
| atr_vel_HIGH | +0.194 |
| garch_vol_pct_LT30 | -0.111 |
| garch_vol_pct_GT70 | +0.085 |
| is_mon | +0.081 |
| is_fri | -0.061 |
| is_opex | -0.040 |
| atr_20_pct_GT80 | -0.020 |
| is_nfp | +0.013 |
| gap_down | +0.012 |
| gap_up | -0.009 |
| atr_20_pct_LT20 | -0.005 |

### Partial dependence — does rel_vol_HIGH predict pnl_r within subsets?

**Stratified on atr_vel_HIGH:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_vel_HIGH=0 | 150 | 413 | +0.063 | -0.291 | +0.354 | +3.72 | 0.0002 |
| atr_vel_HIGH=1 | 128 | 150 | +0.108 | -0.090 | +0.198 | +1.62 | 0.1073 |

**Stratified on atr_20_pct_GT80:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_20_pct_GT80=0 | 224 | 444 | +0.079 | -0.259 | +0.338 | +4.16 | 0.0000 |
| atr_20_pct_GT80=1 | 54 | 119 | +0.106 | -0.156 | +0.262 | +1.50 | 0.1361 |

**Stratified on garch_vol_pct_GT70:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| garch_vol_pct_GT70=0 | 208 | 462 | +0.076 | -0.257 | +0.333 | +4.04 | 0.0001 |
| garch_vol_pct_GT70=1 | 70 | 101 | +0.107 | -0.149 | +0.255 | +1.55 | 0.1239 |

**Stratified on gap_up:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| gap_up=0 | 271 | 547 | +0.082 | -0.258 | +0.340 | +4.57 | 0.0000 |
| gap_up=1 | 7 | 16 | +nan | +nan | +nan | +nan | nan |

**Stratified on is_nfp:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| is_nfp=0 | 263 | 536 | +0.085 | -0.234 | +0.319 | +4.22 | 0.0000 |
| is_nfp=1 | 15 | 27 | +nan | +nan | +nan | +nan | nan |

### Multivariate regression
`pnl_r ~ rel_vol_HIGH + atr_vel_HIGH + atr_20_pct_GT80 + day_type indicators`

- **rel_vol_HIGH coefficient:** +0.2940
- **rel_vol_HIGH t-stat:** +4.02
- **rel_vol_HIGH p:** 0.0001
- **Adj R²:** 0.0276
- **N used:** 841
- **Covariates:** atr_vel_HIGH, atr_20_pct_GT80, garch_vol_pct_GT70, gap_up, gap_down, is_nfp, is_opex

**Interpretation of rel_vol_HIGH coefficient:**
  - coefficient > 0 AND |t| >= 2.0: rel_vol_HIGH adds INDEPENDENT predictive power beyond the controls.
  - coefficient near 0 OR |t| < 2.0: rel_vol_HIGH effect is MEDIATED (absorbed) by the controls.

---

## MNQ SINGAPORE_OPEN O5 RR1.0 short
**N_is:** 835 | **N_on_is (rel_vol_HIGH fire):** 276 | **Fire rate:** 33.1%

### Correlation of rel_vol_HIGH with candidate proxies (IS data)

| Indicator | corr vs rel_vol_HIGH |
|-----------|----------------------|
| atr_vel_HIGH | +0.107 |
| gap_down | +0.086 |
| atr_20_pct_LT20 | -0.077 |
| garch_vol_pct_LT30 | -0.060 |
| garch_vol_pct_GT70 | +0.056 |
| is_nfp | +0.048 |
| atr_20_pct_GT80 | -0.031 |
| gap_up | -0.029 |
| is_opex | +0.021 |
| is_fri | +0.007 |
| is_mon | -0.001 |

### Partial dependence — does rel_vol_HIGH predict pnl_r within subsets?

**Stratified on atr_vel_HIGH:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_vel_HIGH=0 | 165 | 394 | +0.149 | -0.124 | +0.272 | +3.52 | 0.0005 |
| atr_vel_HIGH=1 | 111 | 165 | +0.125 | -0.133 | +0.258 | +2.45 | 0.0149 |

**Stratified on atr_20_pct_GT80:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_20_pct_GT80=0 | 217 | 424 | +0.121 | -0.147 | +0.268 | +3.87 | 0.0001 |
| atr_20_pct_GT80=1 | 59 | 135 | +0.207 | -0.060 | +0.267 | +1.95 | 0.0543 |

**Stratified on garch_vol_pct_GT70:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| garch_vol_pct_GT70=0 | 206 | 445 | +0.112 | -0.135 | +0.247 | +3.52 | 0.0005 |
| garch_vol_pct_GT70=1 | 70 | 114 | +0.220 | -0.091 | +0.312 | +2.36 | 0.0198 |

**Stratified on gap_up:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| gap_up=0 | 272 | 546 | +0.150 | -0.130 | +0.280 | +4.50 | 0.0000 |
| gap_up=1 | 4 | 13 | +nan | +nan | +nan | +nan | nan |

**Stratified on is_nfp:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| is_nfp=0 | 261 | 540 | +0.136 | -0.123 | +0.259 | +4.08 | 0.0001 |
| is_nfp=1 | 15 | 19 | +nan | +nan | +nan | +nan | nan |

### Multivariate regression
`pnl_r ~ rel_vol_HIGH + atr_vel_HIGH + atr_20_pct_GT80 + day_type indicators`

- **rel_vol_HIGH coefficient:** +0.2586
- **rel_vol_HIGH t-stat:** +4.11
- **rel_vol_HIGH p:** 0.0000
- **Adj R²:** 0.0192
- **N used:** 835
- **Covariates:** atr_vel_HIGH, atr_20_pct_GT80, garch_vol_pct_GT70, gap_up, gap_down, is_nfp, is_opex

**Interpretation of rel_vol_HIGH coefficient:**
  - coefficient > 0 AND |t| >= 2.0: rel_vol_HIGH adds INDEPENDENT predictive power beyond the controls.
  - coefficient near 0 OR |t| < 2.0: rel_vol_HIGH effect is MEDIATED (absorbed) by the controls.

---

## MES COMEX_SETTLE O5 RR1.5 short
**N_is:** 753 | **N_on_is (rel_vol_HIGH fire):** 249 | **Fire rate:** 33.1%

### Correlation of rel_vol_HIGH with candidate proxies (IS data)

| Indicator | corr vs rel_vol_HIGH |
|-----------|----------------------|
| is_mon | -0.123 |
| atr_vel_HIGH | +0.120 |
| gap_down | -0.046 |
| is_nfp | +0.029 |
| garch_vol_pct_LT30 | +0.014 |
| gap_up | -0.012 |
| is_opex | -0.009 |
| atr_20_pct_LT20 | -0.004 |
| garch_vol_pct_GT70 | -0.002 |
| atr_20_pct_GT80 | +0.001 |
| is_fri | +0.000 |

### Partial dependence — does rel_vol_HIGH predict pnl_r within subsets?

**Stratified on atr_vel_HIGH:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_vel_HIGH=0 | 147 | 358 | +0.029 | -0.290 | +0.319 | +3.12 | 0.0020 |
| atr_vel_HIGH=1 | 102 | 146 | +0.099 | -0.125 | +0.224 | +1.61 | 0.1079 |

**Stratified on atr_20_pct_GT80:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| atr_20_pct_GT80=0 | 195 | 395 | +0.049 | -0.271 | +0.320 | +3.53 | 0.0005 |
| atr_20_pct_GT80=1 | 54 | 109 | +0.087 | -0.140 | +0.227 | +1.22 | 0.2241 |

**Stratified on garch_vol_pct_GT70:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| garch_vol_pct_GT70=0 | 200 | 404 | -0.002 | -0.265 | +0.263 | +2.95 | 0.0034 |
| garch_vol_pct_GT70=1 | 49 | 100 | +0.300 | -0.151 | +0.451 | +2.31 | 0.0230 |

**Stratified on gap_up:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| gap_up=0 | 244 | 492 | +0.069 | -0.237 | +0.306 | +3.72 | 0.0002 |
| gap_up=1 | 5 | 12 | +nan | +nan | +nan | +nan | nan |

**Stratified on is_nfp:**

| Stratum | N_on | N_off | ExpR_on | ExpR_off | Δ | t | p |
|---------|------|-------|---------|----------|---|---|---|
| is_nfp=0 | 236 | 484 | +0.053 | -0.241 | +0.295 | +3.53 | 0.0005 |
| is_nfp=1 | 13 | 20 | +nan | +nan | +nan | +nan | nan |

### Multivariate regression
`pnl_r ~ rel_vol_HIGH + atr_vel_HIGH + atr_20_pct_GT80 + day_type indicators`

- **rel_vol_HIGH coefficient:** +0.2879
- **rel_vol_HIGH t-stat:** +3.59
- **rel_vol_HIGH p:** 0.0003
- **Adj R²:** 0.0174
- **N used:** 753
- **Covariates:** atr_vel_HIGH, atr_20_pct_GT80, garch_vol_pct_GT70, gap_up, gap_down, is_nfp, is_opex

**Interpretation of rel_vol_HIGH coefficient:**
  - coefficient > 0 AND |t| >= 2.0: rel_vol_HIGH adds INDEPENDENT predictive power beyond the controls.
  - coefficient near 0 OR |t| < 2.0: rel_vol_HIGH effect is MEDIATED (absorbed) by the controls.

---

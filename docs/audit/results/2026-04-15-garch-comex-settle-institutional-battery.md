# Garch COMEX_SETTLE Institutional Battery — All Angles

**Date:** 2026-04-15
**Trigger:** User challenged whether prior R5 analysis applied the correct institutional discipline for a VOLATILITY REGIME indicator (vs generic filter). This battery uses Carver/Fitschen trader discipline: Sharpe decomposition, vol-targeted returns, MAE/MFE decomposition, payoff ratios, Kelly fraction, variance ratios, cost-adjusted Sharpe, permutation on Sharpe lift.

**Cells:** 3 signal candidates (C1 H2 raw, C2 L3 long with OVNRNG_100, C3 L3 short with OVNRNG_100) + 1 null control (N1 L5 TOKYO_OPEN long with ORB_G5).

**Thresholds tested:** 50, 60, 70, 80, 90 (K_threshold=5 per cell).

**Permutation test:** 2000 shuffles on Sharpe-lift (not just ExpR-lift). Shuffle the garch fire assignment, recompute Sharpe on each regime, count how often |shuffled lift| >= |observed lift|.

**BH-FDR correction:** K = (3 signal + 1 null) × 5 thresholds = 20 primary tests. Also reported at K=15 (excluding null control) to reflect the honest pre-reg-able hypothesis set.

**Null control:** N1 L5 TOKYO_OPEN long showed no garch effect in cross-lane R5. If the null control shows a BH-FDR survivor, the methodology is biased — red flag. If null is clean and signals pass, robust evidence.

---

## H2_COMEX_SETTLE_RR1.0_long — MNQ COMEX_SETTLE O5 RR1.0 long (no filter)

**IS N:** 742  |  **OOS N:** 31  |  **Avg risk $:** 47.35  |  **Cost in R:** 0.0617

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-11 | 149 | +0.024 | 0.89 | +0.026 | 57.0% | +0.647 |
| 1 | 11-30 | 148 | +0.013 | 0.90 | +0.015 | 56.1% | +0.641 |
| 2 | 30-52 | 149 | -0.030 | 0.92 | -0.033 | 53.0% | +0.738 |
| 3 | 53-79 | 147 | +0.081 | 0.92 | +0.089 | 58.5% | +0.644 |
| 4 | 79-100 | 149 | +0.294 | 0.88 | +0.333 | 68.5% | +0.594 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 311 | +0.204 | -0.002 | +0.206 | +0.184 | -0.001 | 63.3% | 55.2% | +0.623 | +0.871 | 0.87 | 0.23 | 1.00 | 0.0058 | 0.0060 | — | -0.022 |
| 60 | 253 | +0.232 | +0.008 | +0.224 | +0.210 | +0.008 | 64.4% | 55.6% | +0.618 | +0.888 | 0.88 | 0.26 | 1.00 | 0.0040 | 0.0045 | — | +0.037 |
| 70 | 198 | +0.275 | +0.016 | +0.259 | +0.247 | +0.015 | 66.2% | 55.9% | +0.609 | +0.900 | 0.88 | 0.31 | 0.98 | 0.0020 | 0.0040 | — | +0.243 |
| 80 | 143 | +0.319 | +0.030 | +0.289 | +0.283 | +0.027 | 67.8% | 56.4% | +0.603 | +0.914 | 0.89 | 0.36 | 0.96 | 0.0023 | 0.0035 | — | -0.441 |
| 90 | 80 | +0.258 | +0.063 | +0.195 | +0.236 | +0.057 | 65.0% | 57.9% | +0.655 | +0.906 | 0.90 | 0.28 | 1.02 | 0.1018 | 0.1134 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L1_EUROPE_FLOW_RR1.5_long — L1 MNQ EUROPE_FLOW O5 RR1.5 long (ORB_G5)

**IS N:** 673  |  **OOS N:** 35  |  **Avg risk $:** 38.08  |  **Cost in R:** 0.0767

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-12 | 135 | +0.072 | 1.12 | +0.065 | 48.1% | +0.720 |
| 1 | 12-31 | 134 | +0.059 | 1.11 | +0.053 | 47.8% | +0.678 |
| 2 | 32-54 | 137 | +0.022 | 1.11 | +0.020 | 46.0% | +0.719 |
| 3 | 55-78 | 134 | +0.066 | 1.14 | +0.058 | 47.0% | +0.764 |
| 4 | 79-100 | 133 | +0.258 | 1.16 | +0.222 | 54.1% | +0.683 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 292 | +0.141 | +0.039 | +0.102 | +0.162 | +0.043 | 50.7% | 47.0% | +0.724 | +1.039 | 1.29 | 0.12 | 1.07 | 0.1783 | 0.1864 | — | +0.763 |
| 60 | 237 | +0.127 | +0.060 | +0.068 | +0.147 | +0.067 | 49.8% | 47.9% | +0.729 | +1.022 | 1.30 | 0.11 | 1.08 | 0.3826 | 0.3968 | — | +0.632 |
| 70 | 178 | +0.126 | +0.068 | +0.057 | +0.146 | +0.076 | 49.4% | 48.3% | +0.736 | +1.026 | 1.32 | 0.11 | 1.08 | 0.4882 | 0.5152 | — | +1.106 |
| 80 | 130 | +0.216 | +0.051 | +0.165 | +0.252 | +0.057 | 53.8% | 47.3% | +0.681 | +1.072 | 1.32 | 0.19 | 1.08 | 0.0868 | 0.0810 | — | +0.318 |
| 90 | 76 | +0.326 | +0.053 | +0.273 | +0.374 | +0.059 | 59.2% | 47.2% | +0.621 | +1.170 | 1.32 | 0.28 | 1.05 | 0.0265 | 0.0255 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L1_EUROPE_FLOW_RR1.5_short — L1 MNQ EUROPE_FLOW O5 RR1.5 short (ORB_G5)

**IS N:** 723  |  **OOS N:** 32  |  **Avg risk $:** 36.36  |  **Cost in R:** 0.0803

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-11 | 145 | +0.140 | 1.11 | +0.126 | 51.7% | +0.670 |
| 1 | 11-30 | 145 | +0.009 | 1.11 | +0.008 | 45.5% | +0.704 |
| 2 | 31-50 | 144 | +0.031 | 1.13 | +0.027 | 45.8% | +0.720 |
| 3 | 51-79 | 145 | +0.061 | 1.14 | +0.054 | 46.9% | +0.701 |
| 4 | 80-100 | 144 | +0.104 | 1.17 | +0.089 | 47.2% | +0.744 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 295 | +0.060 | +0.062 | -0.003 | +0.068 | +0.069 | 46.4% | 48.1% | +0.725 | +1.044 | 1.30 | 0.05 | 1.07 | 0.9920 | 0.9615 | — | +0.027 |
| 60 | 249 | +0.080 | +0.051 | +0.029 | +0.093 | +0.057 | 47.4% | 47.5% | +0.718 | +1.054 | 1.31 | 0.07 | 1.07 | 0.6862 | 0.7026 | — | +0.027 |
| 70 | 201 | +0.076 | +0.055 | +0.021 | +0.089 | +0.061 | 46.8% | 47.7% | +0.741 | +1.055 | 1.33 | 0.07 | 1.09 | 0.7765 | 0.8131 | — | +0.531 |
| 80 | 139 | +0.067 | +0.060 | +0.007 | +0.078 | +0.067 | 46.0% | 47.8% | +0.758 | +1.058 | 1.34 | 0.06 | 1.10 | 0.9189 | 0.9330 | — | +0.155 |
| 90 | 73 | +0.161 | +0.049 | +0.111 | +0.190 | +0.055 | 50.7% | 47.1% | +0.719 | +1.082 | 1.35 | 0.14 | 1.11 | 0.3572 | 0.3538 | — | +0.317 |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L2_SINGAPORE_OPEN_RR1.5_long — L2 MNQ SINGAPORE_OPEN O30 RR1.5 long (ATR_P50)

**IS N:** 412  |  **OOS N:** 28  |  **Avg risk $:** 66.56  |  **Cost in R:** 0.0439

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-37 | 83 | +0.062 | 1.16 | +0.053 | 45.8% | +0.728 |
| 1 | 38-60 | 82 | +0.173 | 1.18 | +0.147 | 50.0% | +0.687 |
| 2 | 60-75 | 82 | +0.341 | 1.17 | +0.293 | 57.3% | +0.634 |
| 3 | 75-88 | 82 | +0.293 | 1.18 | +0.248 | 54.9% | +0.704 |
| 4 | 89-100 | 83 | +0.214 | 1.21 | +0.177 | 50.6% | +0.744 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 284 | +0.231 | +0.078 | +0.153 | +0.273 | +0.091 | 53.9% | 46.9% | +0.697 | +1.110 | 1.36 | 0.20 | 1.02 | 0.1460 | 0.1634 | — | -0.458 |
| 60 | 245 | +0.248 | +0.089 | +0.159 | +0.293 | +0.104 | 54.7% | 47.3% | +0.692 | +1.119 | 1.36 | 0.21 | 1.02 | 0.1093 | 0.1164 | — | -0.458 |
| 70 | 197 | +0.193 | +0.174 | +0.019 | +0.230 | +0.204 | 51.8% | 51.6% | +0.713 | +1.092 | 1.38 | 0.16 | 1.04 | 0.8213 | 0.8486 | — | +0.009 |
| 80 | 138 | +0.173 | +0.188 | -0.015 | +0.207 | +0.221 | 50.7% | 52.2% | +0.739 | +1.077 | 1.38 | 0.15 | 1.04 | 0.9141 | 0.8736 | — | -0.068 |
| 90 | 76 | +0.165 | +0.188 | -0.023 | +0.199 | +0.220 | 50.0% | 52.1% | +0.744 | +1.037 | 1.40 | 0.14 | 1.06 | 0.8891 | 0.8461 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L2_SINGAPORE_OPEN_RR1.5_short — L2 MNQ SINGAPORE_OPEN O30 RR1.5 short (ATR_P50)

**IS N:** 358  |  **OOS N:** 26  |  **Avg risk $:** 65.70  |  **Cost in R:** 0.0444

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-35 | 72 | -0.003 | 1.16 | -0.003 | 43.1% | +0.763 |
| 1 | 35-58 | 73 | -0.063 | 1.16 | -0.054 | 39.7% | +0.742 |
| 2 | 58-74 | 71 | +0.098 | 1.19 | +0.083 | 46.5% | +0.744 |
| 3 | 74-89 | 70 | +0.012 | 1.18 | +0.010 | 42.9% | +0.717 |
| 4 | 90-100 | 72 | +0.055 | 1.19 | +0.046 | 44.4% | +0.755 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 236 | +0.028 | -0.006 | +0.034 | +0.033 | -0.007 | 43.6% | 42.6% | +0.742 | +1.011 | 1.37 | 0.02 | 1.03 | 0.7593 | 0.7486 | — | -0.292 |
| 60 | 205 | +0.043 | -0.020 | +0.062 | +0.050 | -0.023 | 44.4% | 41.8% | +0.740 | +1.017 | 1.37 | 0.04 | 1.04 | 0.5589 | 0.5837 | — | -0.093 |
| 70 | 165 | +0.029 | +0.005 | +0.024 | +0.034 | +0.006 | 43.6% | 43.0% | +0.732 | +0.999 | 1.37 | 0.02 | 1.03 | 0.8225 | 0.8291 | — | -0.135 |
| 80 | 123 | +0.003 | +0.024 | -0.021 | +0.003 | +0.028 | 42.3% | 43.8% | +0.751 | +0.989 | 1.37 | 0.00 | 1.02 | 0.8503 | 0.8426 | — | -0.395 |
| 90 | 69 | +0.026 | +0.014 | +0.012 | +0.031 | +0.016 | 43.5% | 43.3% | +0.753 | +0.996 | 1.37 | 0.02 | 1.03 | 0.9293 | 0.8961 | — | -0.283 |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L3_COMEX_SETTLE_RR1.5_long — L3 MNQ COMEX_SETTLE O5 RR1.5 long (OVNRNG_100)

**IS N:** 249  |  **OOS N:** 27  |  **Avg risk $:** 64.45  |  **Cost in R:** 0.0453

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-33 | 50 | -0.041 | 1.14 | -0.036 | 42.0% | +0.748 |
| 1 | 34-59 | 50 | +0.156 | 1.17 | +0.133 | 50.0% | +0.738 |
| 2 | 60-79 | 50 | +0.072 | 1.18 | +0.062 | 46.0% | +0.729 |
| 3 | 80-92 | 49 | +0.449 | 1.17 | +0.385 | 61.2% | +0.653 |
| 4 | 92-100 | 50 | +0.390 | 1.20 | +0.326 | 58.0% | +0.741 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 171 | +0.229 | +0.051 | +0.179 | +0.271 | +0.058 | 53.8% | 46.2% | +0.719 | +1.153 | 1.36 | 0.19 | 1.05 | 0.1820 | 0.2309 | — | +0.097 |
| 60 | 147 | +0.257 | +0.052 | +0.205 | +0.304 | +0.060 | 55.1% | 46.1% | +0.709 | +1.181 | 1.37 | 0.22 | 1.05 | 0.1057 | 0.1249 | — | +0.356 |
| 70 | 125 | +0.294 | +0.052 | +0.242 | +0.348 | +0.060 | 56.8% | 46.0% | +0.697 | +1.204 | 1.37 | 0.25 | 1.04 | 0.0532 | 0.0630 | — | +0.698 |
| 80 | 98 | +0.370 | +0.048 | +0.322 | +0.434 | +0.055 | 60.2% | 45.7% | +0.688 | +1.222 | 1.38 | 0.32 | 1.03 | 0.0131 | 0.0190 | — | -0.176 |
| 90 | 59 | +0.317 | +0.129 | +0.189 | +0.379 | +0.150 | 57.6% | 49.5% | +0.746 | +1.195 | 1.39 | 0.27 | 1.05 | 0.1999 | 0.2394 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L3_COMEX_SETTLE_RR1.5_short — L3 MNQ COMEX_SETTLE O5 RR1.5 short (OVNRNG_100)

**IS N:** 211  |  **OOS N:** 31  |  **Avg risk $:** 60.76  |  **Cost in R:** 0.0481

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-34 | 43 | +0.079 | 1.17 | +0.067 | 46.5% | +0.789 |
| 1 | 34-59 | 42 | +0.114 | 1.18 | +0.096 | 47.6% | +0.742 |
| 2 | 61-76 | 42 | -0.002 | 1.17 | -0.002 | 42.9% | +0.731 |
| 3 | 78-90 | 42 | +0.467 | 1.17 | +0.400 | 61.9% | +0.663 |
| 4 | 92-100 | 42 | +0.311 | 1.21 | +0.258 | 54.8% | +0.661 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 144 | +0.194 | +0.096 | +0.099 | +0.231 | +0.112 | 52.1% | 47.8% | +0.687 | +1.094 | 1.36 | 0.16 | 1.02 | 0.4984 | 0.4733 | — | +0.361 |
| 60 | 126 | +0.218 | +0.082 | +0.136 | +0.259 | +0.096 | 53.2% | 47.1% | +0.685 | +1.106 | 1.37 | 0.18 | 1.03 | 0.3264 | 0.3398 | — | +0.206 |
| 70 | 106 | +0.309 | +0.017 | +0.292 | +0.364 | +0.020 | 57.5% | 43.8% | +0.647 | +1.152 | 1.37 | 0.26 | 1.03 | 0.0339 | 0.0415 | — | +0.468 |
| 80 | 79 | +0.354 | +0.051 | +0.304 | +0.417 | +0.059 | 59.5% | 45.5% | +0.658 | +1.159 | 1.38 | 0.30 | 1.02 | 0.0334 | 0.0425 | — | +0.248 |
| 90 | 46 | +0.249 | +0.139 | +0.110 | +0.300 | +0.163 | 54.3% | 49.7% | +0.695 | +1.080 | 1.39 | 0.21 | 1.05 | 0.4961 | 0.5252 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L4_NYSE_OPEN_RR1.0_long — L4 MNQ NYSE_OPEN O5 RR1.0 long (ORB_G5)

**IS N:** 718  |  **OOS N:** 29  |  **Avg risk $:** 109.87  |  **Cost in R:** 0.0266

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-12 | 147 | +0.102 | 0.96 | +0.106 | 57.1% | +0.683 |
| 1 | 12-29 | 143 | +0.199 | 0.94 | +0.213 | 62.2% | +0.646 |
| 2 | 30-47 | 142 | +0.020 | 0.97 | +0.020 | 52.8% | +0.747 |
| 3 | 48-75 | 142 | +0.067 | 0.97 | +0.069 | 54.9% | +0.686 |
| 4 | 75-100 | 144 | +0.043 | 0.98 | +0.044 | 53.5% | +0.705 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 275 | +0.049 | +0.115 | -0.066 | +0.048 | +0.110 | 53.8% | 57.6% | +0.697 | +0.790 | 0.95 | 0.05 | 1.04 | 0.4034 | 0.4033 | — | +0.053 |
| 60 | 224 | +0.027 | +0.119 | -0.091 | +0.027 | +0.113 | 52.7% | 57.7% | +0.699 | +0.783 | 0.95 | 0.03 | 1.04 | 0.2676 | 0.2584 | — | -0.095 |
| 70 | 176 | +0.031 | +0.109 | -0.078 | +0.030 | +0.105 | 52.8% | 57.2% | +0.699 | +0.793 | 0.95 | 0.03 | 1.04 | 0.3784 | 0.3858 | — | +0.061 |
| 80 | 124 | +0.024 | +0.104 | -0.080 | +0.023 | +0.100 | 52.4% | 56.9% | +0.714 | +0.787 | 0.95 | 0.02 | 1.04 | 0.4274 | 0.4358 | — | +0.352 |
| 90 | 72 | +0.086 | +0.090 | -0.004 | +0.084 | +0.087 | 55.6% | 56.2% | +0.693 | +0.821 | 0.95 | 0.09 | 1.03 | 0.9852 | 1.0000 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L4_NYSE_OPEN_RR1.0_short — L4 MNQ NYSE_OPEN O5 RR1.0 short (ORB_G5)

**IS N:** 705  |  **OOS N:** 37  |  **Avg risk $:** 111.79  |  **Cost in R:** 0.0261

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-10 | 142 | +0.139 | 0.95 | +0.146 | 59.2% | +0.652 |
| 1 | 10-31 | 141 | +0.078 | 0.96 | +0.082 | 56.0% | +0.667 |
| 2 | 31-56 | 140 | +0.009 | 0.97 | +0.010 | 52.1% | +0.677 |
| 3 | 56-80 | 141 | +0.141 | 0.96 | +0.147 | 58.9% | +0.640 |
| 4 | 81-100 | 141 | +0.178 | 0.96 | +0.185 | 60.3% | +0.668 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 312 | +0.134 | +0.098 | +0.035 | +0.129 | +0.094 | 58.0% | 56.7% | +0.669 | +0.817 | 0.95 | 0.14 | 1.01 | 0.6352 | 0.6337 | — | +0.701 |
| 60 | 260 | +0.177 | +0.077 | +0.100 | +0.169 | +0.074 | 60.0% | 55.7% | +0.658 | +0.829 | 0.95 | 0.19 | 1.00 | 0.2030 | 0.2194 | — | +0.465 |
| 70 | 201 | +0.141 | +0.103 | +0.038 | +0.136 | +0.099 | 58.2% | 56.9% | +0.677 | +0.819 | 0.95 | 0.15 | 1.02 | 0.6393 | 0.6702 | — | +0.490 |
| 80 | 145 | +0.179 | +0.097 | +0.082 | +0.172 | +0.093 | 60.0% | 56.6% | +0.667 | +0.831 | 0.95 | 0.19 | 1.00 | 0.3757 | 0.3908 | — | +0.486 |
| 90 | 76 | +0.108 | +0.115 | -0.007 | +0.105 | +0.110 | 56.6% | 57.4% | +0.720 | +0.827 | 0.95 | 0.11 | 1.04 | 0.9693 | 0.9965 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L5_TOKYO_OPEN_RR1.5_long — L5 MNQ TOKYO_OPEN O5 RR1.5 long (ORB_G5)

**IS N:** 682  |  **OOS N:** 34  |  **Avg risk $:** 34.01  |  **Cost in R:** 0.0859

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-12 | 137 | -0.009 | 1.10 | -0.008 | 45.3% | +0.722 |
| 1 | 12-31 | 136 | -0.048 | 1.08 | -0.045 | 44.1% | +0.726 |
| 2 | 31-52 | 136 | +0.156 | 1.11 | +0.140 | 52.2% | +0.674 |
| 3 | 52-78 | 137 | +0.145 | 1.13 | +0.129 | 51.1% | +0.737 |
| 4 | 78-100 | 136 | +0.169 | 1.16 | +0.146 | 50.7% | +0.750 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 279 | +0.144 | +0.023 | +0.121 | +0.165 | +0.026 | 51.3% | 46.9% | +0.740 | +1.073 | 1.27 | 0.13 | 1.08 | 0.1122 | 0.1189 | — | -0.338 |
| 60 | 228 | +0.114 | +0.053 | +0.061 | +0.130 | +0.059 | 49.6% | 48.2% | +0.756 | +1.050 | 1.28 | 0.10 | 1.08 | 0.4337 | 0.4643 | — | -0.474 |
| 70 | 179 | +0.121 | +0.057 | +0.064 | +0.139 | +0.063 | 49.7% | 48.3% | +0.756 | +1.069 | 1.29 | 0.10 | 1.09 | 0.4426 | 0.4608 | — | -0.369 |
| 80 | 124 | +0.120 | +0.063 | +0.057 | +0.140 | +0.070 | 49.2% | 48.6% | +0.766 | +1.060 | 1.32 | 0.10 | 1.11 | 0.5407 | 0.5552 | — | +0.079 |
| 90 | 69 | +0.181 | +0.061 | +0.119 | +0.211 | +0.068 | 52.2% | 48.3% | +0.739 | +1.111 | 1.32 | 0.15 | 1.11 | 0.3352 | 0.3283 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L5_TOKYO_OPEN_RR1.5_short — L5 MNQ TOKYO_OPEN O5 RR1.5 short (ORB_G5)

**IS N:** 712  |  **OOS N:** 33  |  **Avg risk $:** 33.27  |  **Cost in R:** 0.0878

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-12 | 145 | -0.015 | 1.08 | -0.013 | 45.5% | +0.719 |
| 1 | 13-31 | 145 | +0.089 | 1.10 | +0.081 | 49.7% | +0.687 |
| 2 | 32-54 | 137 | +0.124 | 1.11 | +0.112 | 51.1% | +0.827 |
| 3 | 55-81 | 142 | -0.017 | 1.12 | -0.015 | 43.7% | +0.711 |
| 4 | 81-100 | 143 | +0.182 | 1.16 | +0.156 | 51.0% | +0.683 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 309 | +0.062 | +0.067 | -0.005 | +0.071 | +0.073 | 46.9% | 49.1% | +0.703 | +1.062 | 1.28 | 0.05 | 1.09 | 0.9800 | 0.9430 | — | -0.028 |
| 60 | 260 | +0.094 | +0.047 | +0.048 | +0.108 | +0.051 | 48.5% | 48.0% | +0.685 | +1.084 | 1.29 | 0.08 | 1.09 | 0.5189 | 0.5307 | — | -0.038 |
| 70 | 201 | +0.104 | +0.048 | +0.055 | +0.120 | +0.053 | 48.8% | 47.9% | +0.686 | +1.086 | 1.30 | 0.09 | 1.09 | 0.4855 | 0.5062 | — | -0.051 |
| 80 | 147 | +0.155 | +0.040 | +0.115 | +0.180 | +0.044 | 51.0% | 47.4% | +0.681 | +1.109 | 1.31 | 0.13 | 1.11 | 0.2041 | 0.2189 | — | +0.332 |
| 90 | 80 | +0.115 | +0.058 | +0.057 | +0.134 | +0.064 | 48.8% | 48.1% | +0.705 | +1.072 | 1.33 | 0.10 | 1.12 | 0.6135 | 0.6402 | — | +0.344 |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L6_US_DATA_1000_RR1.5_long — L6 MNQ US_DATA_1000 O15 RR1.5 long (VWAP_MID_ALIGNED)

**IS N:** 363  |  **OOS N:** 18  |  **Avg risk $:** 126.60  |  **Cost in R:** 0.0231

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-12 | 75 | +0.255 | 1.21 | +0.210 | 52.0% | +0.685 |
| 1 | 12-29 | 71 | +0.323 | 1.21 | +0.267 | 54.9% | +0.693 |
| 2 | 29-50 | 72 | +0.079 | 1.21 | +0.065 | 44.4% | +0.743 |
| 3 | 50-76 | 72 | -0.053 | 1.19 | -0.045 | 38.9% | +0.754 |
| 4 | 76-100 | 73 | +0.136 | 1.23 | +0.111 | 46.6% | +0.729 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 145 | +0.035 | +0.181 | -0.146 | +0.042 | +0.219 | 42.8% | 50.5% | +0.742 | +0.923 | 1.44 | 0.03 | 1.00 | 0.1736 | 0.1774 | — | +0.680 |
| 60 | 116 | +0.059 | +0.152 | -0.093 | +0.072 | +0.184 | 44.0% | 49.0% | +0.735 | +0.954 | 1.44 | 0.05 | 1.01 | 0.4122 | 0.4303 | — | +0.680 |
| 70 | 98 | +0.119 | +0.123 | -0.005 | +0.145 | +0.149 | 46.9% | 47.5% | +0.720 | +0.979 | 1.44 | 0.10 | 1.02 | 0.9778 | 1.0000 | — | +0.545 |
| 80 | 66 | +0.179 | +0.109 | +0.070 | +0.220 | +0.132 | 50.0% | 46.8% | +0.704 | +1.022 | 1.44 | 0.15 | 1.03 | 0.6002 | 0.5812 | — | +160.609 |
| 90 | 33 | +0.091 | +0.125 | -0.035 | +0.112 | +0.152 | 45.5% | 47.6% | +0.764 | +0.911 | 1.45 | 0.07 | 1.04 | 0.8613 | 0.8591 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

## L6_US_DATA_1000_RR1.5_short — L6 MNQ US_DATA_1000 O15 RR1.5 short (VWAP_MID_ALIGNED)

**IS N:** 286  |  **OOS N:** 17  |  **Avg risk $:** 143.07  |  **Cost in R:** 0.0204

### NTILE-5 breakdown

| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |
|---|---|---|---|---|---|---|---|
| 0 | 0-11 | 59 | +0.272 | 1.22 | +0.223 | 52.5% | +0.703 |
| 1 | 12-31 | 56 | +0.170 | 1.22 | +0.139 | 48.2% | +0.771 |
| 2 | 31-58 | 57 | +0.322 | 1.22 | +0.263 | 54.4% | +0.663 |
| 3 | 59-81 | 57 | +0.202 | 1.23 | +0.163 | 49.1% | +0.697 |
| 4 | 82-100 | 57 | +0.335 | 1.23 | +0.272 | 54.4% | +0.721 |

### Threshold grid — full battery

| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 128 | +0.230 | +0.200 | +0.030 | +0.282 | +0.243 | 52.3% | 51.3% | +0.700 | +1.083 | 1.45 | 0.19 | 1.02 | 0.7904 | 0.8146 | — | +0.197 |
| 60 | 110 | +0.220 | +0.209 | +0.010 | +0.270 | +0.255 | 51.8% | 51.7% | +0.709 | +1.070 | 1.45 | 0.18 | 1.02 | 0.9164 | 0.9050 | — | -0.323 |
| 70 | 84 | +0.184 | +0.226 | -0.042 | +0.227 | +0.275 | 50.0% | 52.5% | +0.729 | +1.062 | 1.45 | 0.15 | 1.03 | 0.7635 | 0.7846 | — | +0.101 |
| 80 | 62 | +0.216 | +0.213 | +0.003 | +0.267 | +0.259 | 51.6% | 51.8% | +0.736 | +1.080 | 1.45 | 0.17 | 1.03 | 0.9643 | 0.9595 | — | n/a |
| 90 | 34 | +0.301 | +0.201 | +0.099 | +0.372 | +0.246 | 55.9% | 51.2% | +0.738 | +1.126 | 1.46 | 0.24 | 1.03 | 0.5783 | 0.5732 | — | n/a |

**Interpretation key:**
- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)
- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol
- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured
- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)
- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)

---

## BH-FDR summary

- K=65 (including null control): 0 survivors
- K=65 (excluding null control): 0 survivors
- Null control N1 produced 0 BH-FDR survivors — OK (methodology clean)


---

## Trader verdict (honest read)

A real vol-regime edge shows:
1. `sr_lift` > 0 (Sharpe improves, not just ExpR)
2. `var_ratio` near 1.0 (edge not driven by vol scaling alone)
3. `WR_on` > `WR_off` (directional accuracy, not just bigger bars)
4. `MAE_on` NOT proportionally worse than MAE_off (drawdown profile not degraded)
5. Null control does NOT produce false positives
6. OOS sr_lift sign matches IS

Rows meeting 4+ criteria above are legitimate edge candidates. See survivor table.

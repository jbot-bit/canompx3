# Garch State Distinctness Audit

**Date:** 2026-04-16
**Purpose:** local distinctness audit on the locked garch-anchored family set.
**Boundary:** not a global proxy ranking, not a deployment proof, not a new search.

## Scope rules

- D1 canonical distinctness uses canonical populations on the locked family set.
- D2 utility distinctness uses `validated_setups` only and remains research-provisional.
- 2026 forward/OOS is descriptive only and does not decide the verdicts here.
- `atr_vel` is treated canonically via regime strata first; no fake 70/30 thresholds.

## COMEX_SETTLE high

- D1 pooled trades: **13446**
- Mean `garch_pct`: **52.53**
- Mean `atr_20_pct`: **60.25**
- Mean `overnight_range_pct`: **56.21**
- Mean `atr_vel_ratio`: **1.011**

### Overlap

| Metric | Value |
|---|---|
| corr(garch, atr_20_pct) | +0.765 |
| corr(garch, overnight_range_pct) | +0.314 |
| corr(garch, atr_vel_ratio) | +0.377 |
| corr(garch_flag, atr_vel_contracting) | -0.148 |
| corr(garch_flag, atr_vel_expanding) | +0.326 |

### Conditional sign persistence

| Stratum | Status | N | Lift | SR lift | Support |
|---|---:|---:|---:|---:|---:|
| overall | ok | 13446 | +0.144 | +0.133 | Y |
| atr_high | ok | 4486 | +0.190 | +0.175 | Y |
| atr_low | thin | 2087 |  |  |  |
| ovn_high | ok | 3966 | +0.226 | +0.210 | Y |
| ovn_low | ok | 2216 | +0.077 | +0.072 | Y |
| atr_vel_expanding | ok | 3566 | +0.237 | +0.221 | Y |
| atr_vel_stable | ok | 7123 | +0.067 | +0.062 | Y |
| atr_vel_contracting | ok | 2757 | +0.257 | +0.255 | Y |

### Four-cell decompositions

**atr**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1837 | +0.107 | +0.099 | +196.2 | ok |
| low | high | 9 | -0.172 | -0.175 | -1.5 | thin |
| high | low | 68 | -0.070 | -0.069 | -4.8 | ok |
| high | high | 3305 | +0.191 | +0.177 | +631.1 | ok |

**overnight**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1126 | +0.095 | +0.089 | +107.3 | ok |
| low | high | 397 | +0.146 | +0.137 | +58.1 | ok |
| high | low | 685 | +0.013 | +0.012 | +8.7 | ok |
| high | high | 1995 | +0.223 | +0.207 | +444.9 | ok |

**atr_vel**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 2000 | +0.145 | +0.136 | +291.0 | ok |
| low | high | 969 | +0.374 | +0.359 | +362.0 | ok |
| high | low | 480 | -0.076 | -0.072 | -36.7 | ok |
| high | high | 2604 | +0.170 | +0.158 | +443.7 | ok |

### D2 local utility comparison

| Score | Cells | Support cells | Mean lift | Mean SR lift | Mean N_on |
|---|---:|---:|---:|---:|---:|
| garch | 18 | 18 | +0.204 | +0.197 | 157.9 |
| atr | 18 | 15 | +0.134 | +0.127 | 145.7 |
| overnight | 18 | 14 | +0.047 | +0.047 | 125.6 |

### Verdicts

| Proxy | Verdict | Allowed role |
|---|---|---|
| atr | unclear | unclear |
| overnight | distinct | R3/R7 |
| atr_vel | distinct | R3/R7 |

## EUROPE_FLOW high

- D1 pooled trades: **11357**
- Mean `garch_pct`: **50.20**
- Mean `atr_20_pct`: **57.83**
- Mean `overnight_range_pct`: **56.01**
- Mean `atr_vel_ratio`: **1.008**

### Overlap

| Metric | Value |
|---|---|
| corr(garch, atr_20_pct) | +0.767 |
| corr(garch, overnight_range_pct) | +0.298 |
| corr(garch, atr_vel_ratio) | +0.358 |
| corr(garch_flag, atr_vel_contracting) | -0.140 |
| corr(garch_flag, atr_vel_expanding) | +0.315 |

### Conditional sign persistence

| Stratum | Status | N | Lift | SR lift | Support |
|---|---:|---:|---:|---:|---:|
| overall | ok | 11357 | +0.056 | +0.052 | Y |
| atr_high | ok | 3502 | +0.182 | +0.172 | Y |
| atr_low | thin | 1986 |  |  |  |
| ovn_high | ok | 3329 | +0.120 | +0.110 | Y |
| ovn_low | ok | 1861 | -0.173 | -0.167 | N |
| atr_vel_expanding | ok | 2891 | -0.082 | -0.079 | N |
| atr_vel_stable | ok | 6031 | +0.010 | +0.010 | Y |
| atr_vel_contracting | ok | 2435 | +0.139 | +0.128 | Y |

### Four-cell decompositions

**atr**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1754 | +0.153 | +0.143 | +267.9 | ok |
| low | high | 9 | -0.546 | -0.601 | -4.9 | thin |
| high | low | 63 | -0.670 | -0.910 | -42.2 | ok |
| high | high | 2565 | +0.117 | +0.108 | +299.3 | ok |

**overnight**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 997 | +0.022 | +0.020 | +21.5 | ok |
| low | high | 299 | -0.176 | -0.170 | -52.6 | ok |
| high | low | 678 | +0.141 | +0.132 | +95.8 | ok |
| high | high | 1584 | +0.186 | +0.172 | +294.3 | ok |

**atr_vel**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1765 | +0.010 | +0.009 | +17.2 | ok |
| low | high | 789 | +0.182 | +0.170 | +143.9 | ok |
| high | low | 523 | +0.335 | +0.315 | +175.2 | ok |
| high | high | 2057 | +0.163 | +0.152 | +335.6 | ok |

### D2 local utility comparison

| Score | Cells | Support cells | Mean lift | Mean SR lift | Mean N_on |
|---|---:|---:|---:|---:|---:|
| garch | 16 | 15 | +0.067 | +0.062 | 162.1 |
| overnight | 16 | 13 | +0.057 | +0.045 | 141.6 |
| atr | 16 | 8 | +0.040 | +0.037 | 142.9 |

### Verdicts

| Proxy | Verdict | Allowed role |
|---|---|---|
| atr | unclear | unclear |
| overnight | distinct | R3/R7 |
| atr_vel | unclear | unclear |

## TOKYO_OPEN high

- D1 pooled trades: **10531**
- Mean `garch_pct`: **53.25**
- Mean `atr_20_pct`: **63.32**
- Mean `overnight_range_pct`: **53.33**
- Mean `atr_vel_ratio`: **1.008**

### Overlap

| Metric | Value |
|---|---|
| corr(garch, atr_20_pct) | +0.746 |
| corr(garch, overnight_range_pct) | +0.297 |
| corr(garch, atr_vel_ratio) | +0.342 |
| corr(garch_flag, atr_vel_contracting) | -0.144 |
| corr(garch_flag, atr_vel_expanding) | +0.308 |

### Conditional sign persistence

| Stratum | Status | N | Lift | SR lift | Support |
|---|---:|---:|---:|---:|---:|
| overall | ok | 10531 | +0.070 | +0.062 | Y |
| atr_high | ok | 3841 | -0.021 | -0.020 | N |
| atr_low | thin | 1434 |  |  |  |
| ovn_high | ok | 2694 | +0.027 | +0.020 | Y |
| ovn_low | ok | 1931 | -0.036 | -0.035 | N |
| atr_vel_expanding | ok | 2673 | +0.173 | +0.153 | Y |
| atr_vel_stable | ok | 5576 | -0.004 | -0.005 | N |
| atr_vel_contracting | ok | 2282 | -0.024 | -0.021 | N |

### Four-cell decompositions

**atr**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1274 | -0.031 | -0.028 | -39.3 | ok |
| low | high | 6 | -1.000 | +0.000 | -6.0 | thin |
| high | low | 77 | +0.003 | +0.003 | +0.3 | ok |
| high | high | 2759 | +0.055 | +0.049 | +151.7 | ok |

**overnight**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 913 | -0.357 | -0.366 | -326.0 | ok |
| low | high | 389 | -0.356 | -0.357 | -138.6 | ok |
| high | low | 439 | +0.206 | +0.185 | +90.3 | ok |
| high | high | 1423 | +0.352 | +0.312 | +500.8 | ok |

**atr_vel**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1449 | -0.045 | -0.041 | -64.6 | ok |
| low | high | 833 | +0.029 | +0.026 | +24.2 | ok |
| high | low | 386 | +0.027 | +0.024 | +10.3 | ok |
| high | high | 2024 | +0.160 | +0.141 | +323.9 | ok |

### D2 local utility comparison

| Score | Cells | Support cells | Mean lift | Mean SR lift | Mean N_on |
|---|---:|---:|---:|---:|---:|
| overnight | 8 | 8 | +0.324 | +0.283 | 138.8 |
| garch | 8 | 8 | +0.081 | +0.069 | 173.5 |
| atr | 8 | 5 | +0.036 | +0.032 | 153.5 |

### Verdicts

| Proxy | Verdict | Allowed role |
|---|---|---|
| atr | unclear | unclear |
| overnight | unclear | unclear |
| atr_vel | distinct | R3/R7 |

## SINGAPORE_OPEN high

- D1 pooled trades: **9551**
- Mean `garch_pct`: **55.13**
- Mean `atr_20_pct`: **65.77**
- Mean `overnight_range_pct`: **54.33**
- Mean `atr_vel_ratio`: **1.010**

### Overlap

| Metric | Value |
|---|---|
| corr(garch, atr_20_pct) | +0.730 |
| corr(garch, overnight_range_pct) | +0.296 |
| corr(garch, atr_vel_ratio) | +0.339 |
| corr(garch_flag, atr_vel_contracting) | -0.143 |
| corr(garch_flag, atr_vel_expanding) | +0.315 |

### Conditional sign persistence

| Stratum | Status | N | Lift | SR lift | Support |
|---|---:|---:|---:|---:|---:|
| overall | ok | 9551 | +0.066 | +0.059 | Y |
| atr_high | ok | 3710 | -0.023 | -0.021 | N |
| atr_low | thin | 1073 |  |  |  |
| ovn_high | ok | 2563 | +0.055 | +0.046 | Y |
| ovn_low | ok | 1647 | +0.020 | +0.018 | Y |
| atr_vel_expanding | ok | 2499 | +0.174 | +0.155 | Y |
| atr_vel_stable | ok | 5032 | +0.062 | +0.056 | Y |
| atr_vel_contracting | ok | 2020 | -0.156 | -0.142 | N |

### Four-cell decompositions

**atr**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 944 | +0.038 | +0.035 | +35.7 | ok |
| low | high | 6 | -1.000 | +0.000 | -6.0 | thin |
| high | low | 72 | -0.202 | -0.196 | -14.5 | ok |
| high | high | 2655 | +0.067 | +0.060 | +177.7 | ok |

**overnight**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 733 | -0.080 | -0.075 | -58.6 | ok |
| low | high | 370 | -0.148 | -0.142 | -54.7 | ok |
| high | low | 375 | +0.196 | +0.179 | +73.7 | ok |
| high | high | 1388 | +0.219 | +0.193 | +303.9 | ok |

**atr_vel**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1207 | +0.042 | +0.039 | +50.5 | ok |
| low | high | 808 | +0.003 | +0.003 | +2.6 | ok |
| high | low | 308 | -0.027 | -0.025 | -8.4 | ok |
| high | high | 1923 | +0.145 | +0.129 | +278.2 | ok |

### D2 local utility comparison

| Score | Cells | Support cells | Mean lift | Mean SR lift | Mean N_on |
|---|---:|---:|---:|---:|---:|
| overnight | 4 | 4 | +0.233 | +0.195 | 105.5 |
| atr | 4 | 4 | +0.054 | +0.045 | 173.5 |
| garch | 4 | 4 | +0.042 | +0.034 | 181.8 |

### Verdicts

| Proxy | Verdict | Allowed role |
|---|---|---|
| atr | unclear | unclear |
| overnight | complementary | R7/R8 |
| atr_vel | distinct | R3/R7 |

## LONDON_METALS high

- D1 pooled trades: **11271**
- Mean `garch_pct`: **51.06**
- Mean `atr_20_pct`: **58.84**
- Mean `overnight_range_pct`: **56.57**
- Mean `atr_vel_ratio`: **1.010**

### Overlap

| Metric | Value |
|---|---|
| corr(garch, atr_20_pct) | +0.763 |
| corr(garch, overnight_range_pct) | +0.305 |
| corr(garch, atr_vel_ratio) | +0.376 |
| corr(garch_flag, atr_vel_contracting) | -0.145 |
| corr(garch_flag, atr_vel_expanding) | +0.332 |

### Conditional sign persistence

| Stratum | Status | N | Lift | SR lift | Support |
|---|---:|---:|---:|---:|---:|
| overall | ok | 11271 | +0.046 | +0.042 | Y |
| atr_high | ok | 3556 | -0.128 | -0.120 | N |
| atr_low | ok | 1832 | -0.689 | -0.864 | N |
| ovn_high | ok | 3420 | -0.013 | -0.012 | N |
| ovn_low | ok | 1864 | +0.160 | +0.145 | Y |
| atr_vel_expanding | ok | 2950 | -0.036 | -0.032 | N |
| atr_vel_stable | ok | 5965 | +0.057 | +0.052 | Y |
| atr_vel_contracting | ok | 2356 | +0.086 | +0.078 | Y |

### Four-cell decompositions

**atr**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1608 | -0.003 | -0.003 | -4.4 | ok |
| low | high | 11 | -0.648 | -0.827 | -7.1 | thin |
| high | low | 66 | -0.152 | -0.141 | -10.0 | ok |
| high | high | 2628 | +0.059 | +0.054 | +156.2 | ok |

**overnight**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 986 | +0.029 | +0.027 | +29.0 | ok |
| low | high | 301 | +0.140 | +0.126 | +42.1 | ok |
| high | low | 672 | -0.058 | -0.053 | -39.0 | ok |
| high | high | 1668 | +0.087 | +0.079 | +144.9 | ok |

**atr_vel**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1754 | -0.030 | -0.028 | -53.0 | ok |
| low | high | 796 | +0.052 | +0.047 | +41.3 | ok |
| high | low | 474 | +0.127 | +0.114 | +60.4 | ok |
| high | high | 2109 | +0.041 | +0.038 | +87.5 | ok |

### D2 local utility comparison

No validated-family utility rows met minimum support.

### Verdicts

| Proxy | Verdict | Allowed role |
|---|---|---|
| atr | unclear | unclear |
| overnight | distinct | R3/R7 |
| atr_vel | unclear | unclear |

## NYSE_OPEN low

- D1 pooled trades: **21129**
- Mean `garch_pct`: **51.18**
- Mean `atr_20_pct`: **58.14**
- Mean `overnight_range_pct`: **55.07**
- Mean `atr_vel_ratio`: **1.010**

### Overlap

| Metric | Value |
|---|---|
| corr(garch, atr_20_pct) | +0.742 |
| corr(garch, overnight_range_pct) | +0.307 |
| corr(garch, atr_vel_ratio) | +0.397 |
| corr(garch_flag, atr_vel_contracting) | +0.179 |
| corr(garch_flag, atr_vel_expanding) | -0.282 |

### Conditional sign persistence

| Stratum | Status | N | Lift | SR lift | Support |
|---|---:|---:|---:|---:|---:|
| overall | ok | 21129 | +0.012 | +0.011 | N |
| atr_high | ok | 6256 | -0.222 | -0.210 | Y |
| atr_low | ok | 3372 | -0.018 | -0.015 | Y |
| ovn_high | ok | 6082 | -0.145 | -0.132 | Y |
| ovn_low | ok | 3748 | +0.183 | +0.165 | N |
| atr_vel_expanding | ok | 5637 | +0.234 | +0.210 | N |
| atr_vel_stable | ok | 11111 | -0.093 | -0.082 | Y |
| atr_vel_contracting | ok | 4381 | +0.129 | +0.117 | N |

### Four-cell decompositions

**atr**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 2890 | +0.066 | +0.059 | +189.9 | ok |
| low | high | 37 | -0.334 | -0.314 | -12.3 | ok |
| high | low | 139 | -0.213 | -0.201 | -29.6 | ok |
| high | high | 4586 | -0.006 | -0.006 | -28.4 | ok |

**overnight**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 1783 | +0.155 | +0.140 | +276.2 | ok |
| low | high | 616 | -0.036 | -0.033 | -22.3 | ok |
| high | low | 1086 | -0.086 | -0.080 | -93.8 | ok |
| high | high | 2932 | +0.004 | +0.004 | +11.6 | ok |

**atr_vel**

| Proxy state | Garch state | N | ExpR | SR | Total R | Status |
|---|---|---:|---:|---:|---:|---|
| low | low | 3206 | +0.077 | +0.070 | +248.5 | ok |
| low | high | 1347 | -0.094 | -0.086 | -126.7 | ok |
| high | low | 756 | +0.190 | +0.170 | +144.0 | ok |
| high | high | 3994 | -0.013 | -0.012 | -52.4 | ok |

### D2 local utility comparison

| Score | Cells | Support cells | Mean lift | Mean SR lift | Mean N_on |
|---|---:|---:|---:|---:|---:|
| garch | 12 | 5 | +0.054 | +0.048 | 194.7 |
| overnight | 12 | 6 | +0.024 | +0.021 | 119.6 |
| atr | 8 | 4 | +0.017 | +0.011 | 156.2 |

### Verdicts

| Proxy | Verdict | Allowed role |
|---|---|---|
| atr | unclear | unclear |
| overnight | unclear | unclear |
| atr_vel | unclear | unclear |

## Guardrails

- `distinct` / `complementary` / `subsumed` are local to the locked family set.
- `validated_setups` evidence here is research-provisional, not production truth.
- No 2026 forward/OOS figure was allowed to decide the verdicts.
- Pairwise correlations are descriptive only; verdicts required persistence plus four-cell or utility support.

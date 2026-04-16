# Garch Structural Decomposition

**Date:** 2026-04-16
**Source families:** strongest directional / monotonicity families from `garch_regime_family_audit.py`.
**Theory grounding:** Chan 2008 Ch 7 on high/low volatility regimes and GARCH-based regime tracking; Carver 2015 for continuous forecast/sizing interpretation.

Trade-time-knowable proxies only:
- `atr_20_pct`
- `overnight_range_pct`
- `atr_vel_ratio`
- gap flags
- NFP / OPEX / Friday flags

No `day_type` or post-entry columns used.

## COMEX_SETTLE high

- Cells: **60**
- Pooled trades: **21949**
- Mean `garch_pct`: **51.48**
- Mean `atr_20_pct`: **59.37**
- Mean `overnight_range_pct`: **55.34**
- Mean `atr_vel_ratio`: **1.009**

### Correlations / overlap

| Metric | Value |
|---|---|
| corr(garch_pct, atr_20_pct) | +0.767 |
| corr(garch_pct, overnight_range_pct) | +0.305 |
| corr(garch_pct, atr_vel_ratio) | +0.367 |
| corr(garch_flag, gap_up) | -0.002 |
| corr(garch_flag, gap_down) | -0.011 |
| corr(garch_flag, is_nfp_day) | +0.039 |
| corr(garch_flag, is_opex_day) | +0.021 |
| corr(garch_flag, is_friday) | +0.022 |

### Sign persistence within other regime strata

| Check | Supporting cells |
|---|---|
| Overall expected sign | 58/60 |
| Within ATR high stratum | 43/48 |
| Within ATR low stratum | n/a |
| Within overnight-range high stratum | 44/53 |
| Within overnight-range low stratum | 23/36 |

## EUROPE_FLOW high

- Cells: **42**
- Pooled trades: **19563**
- Mean `garch_pct`: **49.28**
- Mean `atr_20_pct`: **56.66**
- Mean `overnight_range_pct`: **55.46**
- Mean `atr_vel_ratio`: **1.007**

### Correlations / overlap

| Metric | Value |
|---|---|
| corr(garch_pct, atr_20_pct) | +0.771 |
| corr(garch_pct, overnight_range_pct) | +0.295 |
| corr(garch_pct, atr_vel_ratio) | +0.352 |
| corr(garch_flag, gap_up) | +0.001 |
| corr(garch_flag, gap_down) | +0.004 |
| corr(garch_flag, is_nfp_day) | +0.038 |
| corr(garch_flag, is_opex_day) | +0.021 |
| corr(garch_flag, is_friday) | +0.014 |

### Sign persistence within other regime strata

| Check | Supporting cells |
|---|---|
| Overall expected sign | 38/42 |
| Within ATR high stratum | 36/42 |
| Within ATR low stratum | n/a |
| Within overnight-range high stratum | 33/42 |
| Within overnight-range low stratum | 10/28 |

## TOKYO_OPEN high

- Cells: **32**
- Pooled trades: **15023**
- Mean `garch_pct`: **51.61**
- Mean `atr_20_pct`: **60.84**
- Mean `overnight_range_pct`: **53.12**
- Mean `atr_vel_ratio`: **1.007**

### Correlations / overlap

| Metric | Value |
|---|---|
| corr(garch_pct, atr_20_pct) | +0.754 |
| corr(garch_pct, overnight_range_pct) | +0.295 |
| corr(garch_pct, atr_vel_ratio) | +0.342 |
| corr(garch_flag, gap_up) | -0.007 |
| corr(garch_flag, gap_down) | -0.006 |
| corr(garch_flag, is_nfp_day) | +0.028 |
| corr(garch_flag, is_opex_day) | +0.016 |
| corr(garch_flag, is_friday) | +0.015 |

### Sign persistence within other regime strata

| Check | Supporting cells |
|---|---|
| Overall expected sign | 30/32 |
| Within ATR high stratum | 13/32 |
| Within ATR low stratum | n/a |
| Within overnight-range high stratum | 18/32 |
| Within overnight-range low stratum | 12/25 |

## SINGAPORE_OPEN high

- Cells: **28**
- Pooled trades: **11098**
- Mean `garch_pct`: **56.20**
- Mean `atr_20_pct`: **67.41**
- Mean `overnight_range_pct`: **54.37**
- Mean `atr_vel_ratio`: **1.010**

### Correlations / overlap

| Metric | Value |
|---|---|
| corr(garch_pct, atr_20_pct) | +0.714 |
| corr(garch_pct, overnight_range_pct) | +0.294 |
| corr(garch_pct, atr_vel_ratio) | +0.332 |
| corr(garch_flag, gap_up) | -0.003 |
| corr(garch_flag, gap_down) | -0.014 |
| corr(garch_flag, is_nfp_day) | +0.033 |
| corr(garch_flag, is_opex_day) | +0.013 |
| corr(garch_flag, is_friday) | +0.020 |

### Sign persistence within other regime strata

| Check | Supporting cells |
|---|---|
| Overall expected sign | 24/28 |
| Within ATR high stratum | 8/28 |
| Within ATR low stratum | n/a |
| Within overnight-range high stratum | 20/28 |
| Within overnight-range low stratum | 17/26 |

## LONDON_METALS high

- Cells: **32**
- Pooled trades: **11271**
- Mean `garch_pct`: **51.06**
- Mean `atr_20_pct`: **58.84**
- Mean `overnight_range_pct`: **56.57**
- Mean `atr_vel_ratio`: **1.010**

### Correlations / overlap

| Metric | Value |
|---|---|
| corr(garch_pct, atr_20_pct) | +0.763 |
| corr(garch_pct, overnight_range_pct) | +0.305 |
| corr(garch_pct, atr_vel_ratio) | +0.376 |
| corr(garch_flag, gap_up) | -0.003 |
| corr(garch_flag, gap_down) | +0.010 |
| corr(garch_flag, is_nfp_day) | +0.032 |
| corr(garch_flag, is_opex_day) | +0.027 |
| corr(garch_flag, is_friday) | +0.023 |

### Sign persistence within other regime strata

| Check | Supporting cells |
|---|---|
| Overall expected sign | 23/32 |
| Within ATR high stratum | 5/24 |
| Within ATR low stratum | n/a |
| Within overnight-range high stratum | 15/31 |
| Within overnight-range low stratum | 13/15 |

## NYSE_OPEN low

- Cells: **60**
- Pooled trades: **27934**
- Mean `garch_pct`: **50.48**
- Mean `atr_20_pct`: **57.89**
- Mean `overnight_range_pct`: **54.06**
- Mean `atr_vel_ratio`: **1.008**

### Correlations / overlap

| Metric | Value |
|---|---|
| corr(garch_pct, atr_20_pct) | +0.748 |
| corr(garch_pct, overnight_range_pct) | +0.305 |
| corr(garch_pct, atr_vel_ratio) | +0.384 |
| corr(garch_flag, gap_up) | -0.033 |
| corr(garch_flag, gap_down) | -0.032 |
| corr(garch_flag, is_nfp_day) | -0.014 |
| corr(garch_flag, is_opex_day) | -0.020 |
| corr(garch_flag, is_friday) | -0.022 |

### Sign persistence within other regime strata

| Check | Supporting cells |
|---|---|
| Overall expected sign | 30/60 |
| Within ATR high stratum | n/a |
| Within ATR low stratum | 18/36 |
| Within overnight-range high stratum | 35/47 |
| Within overnight-range low stratum | 5/38 |

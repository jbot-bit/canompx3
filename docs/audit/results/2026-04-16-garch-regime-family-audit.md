# Garch Regime Family Audit

**Date:** 2026-04-16
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-regime-family-audit.yaml`
**Fixed tails:** `HIGH >= 70`, `LOW <= 30`
**Cell inventory source:** current tradeable exact-filter row universe seeded from `validated_setups` + `experimental_strategies`; all metrics computed from canonical `orb_outcomes` + `daily_features`.

## Preflight

### orb_outcomes freshness / counts

| Symbol | Max trading_day | Rows |
|---|---|---|
| GC | 2026-04-05 00:00:00 | 1295064 |
| M2K | 2026-03-06 00:00:00 | 1544508 |
| M6E | 2026-02-19 00:00:00 | 324870 |
| MBT | 2026-01-30 00:00:00 | 530124 |
| MES | 2026-04-14 00:00:00 | 1907856 |
| MGC | 2026-04-14 00:00:00 | 918396 |
| MNQ | 2026-04-14 00:00:00 | 2109564 |
| SIL | 2026-02-16 00:00:00 | 86520 |

### daily_features freshness / counts

| Symbol | Max trading_day | Rows |
|---|---|---|
| GC | 2026-04-06 00:00:00 | 4605 |
| M2K | 2026-03-06 00:00:00 | 4419 |
| M6E | 2026-02-20 00:00:00 | 4389 |
| MBT | 2026-01-30 00:00:00 | 4389 |
| MES | 2026-04-15 00:00:00 | 6093 |
| MGC | 2026-04-15 00:00:00 | 3354 |
| MNQ | 2026-04-15 00:00:00 | 6093 |
| SIL | 2026-02-16 00:00:00 | 1749 |

Required columns present: `atr_20_pct, gap_open_points, garch_forecast_vol, garch_forecast_vol_pct, overnight_range`
Required columns missing: `none`

## Global asymmetry

- Cells in scope: **431**
- HIGH @70 positive cells: **304 / 431** (`p=0.000000`)
- LOW @30 negative cells: **290 / 431** (`p=0.000000`)

## Shuffle-null destruction control

- Real HIGH positive fraction: **0.705**
- Shuffled HIGH median fraction: **0.498** range [0.450, 0.543]
- Shuffle p (HIGH real >= shuffled): **0.0099**

- Real LOW negative fraction: **0.673**
- Shuffled LOW median fraction: **0.506** range [0.452, 0.589]
- Shuffle p (LOW real >= shuffled): **0.0099**

## Session-side directional sign test

| Session | Side | Cells | Support | Oppose | Support % | p_dir | BH | mean sr_lift | mean lift | family BH survivors | OOS sign match | Long support | Short support |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BRISBANE_1025 | high | 12 | 10 | 2 | 83.3% | 0.019287 | Y | +0.048 | +0.058 | 0 | 41.7% | 5/6 | 5/6 |
| BRISBANE_1025 | low | 12 | 8 | 4 | 66.7% | 0.193848 | . | -0.032 | -0.034 | 0 | 50.0% | 6/6 | 2/6 |
| CME_PRECLOSE | high | 42 | 34 | 8 | 81.0% | 0.000034 | Y | +0.117 | +0.119 | 0 | 39.4% | 19/19 | 15/23 |
| CME_PRECLOSE | low | 42 | 28 | 14 | 66.7% | 0.021779 | Y | -0.072 | -0.074 | 0 | 37.5% | 13/19 | 15/23 |
| CME_REOPEN | high | 16 | 10 | 6 | 62.5% | 0.227249 | . | +0.033 | +0.042 | 0 | 56.2% | 3/8 | 7/8 |
| CME_REOPEN | low | 16 | 11 | 5 | 68.8% | 0.105057 | . | -0.073 | -0.079 | 0 | 50.0% | 3/8 | 8/8 |
| COMEX_SETTLE | high | 52 | 50 | 2 | 96.2% | 0.000000 | Y | +0.176 | +0.180 | 0 | 94.2% | 27/28 | 23/24 |
| COMEX_SETTLE | low | 52 | 44 | 8 | 84.6% | 0.000000 | Y | -0.131 | -0.130 | 0 | 64.7% | 27/28 | 17/24 |
| EUROPE_FLOW | high | 42 | 38 | 4 | 90.5% | 0.000000 | Y | +0.073 | +0.072 | 2 | 97.6% | 19/21 | 19/21 |
| EUROPE_FLOW | low | 42 | 33 | 9 | 78.6% | 0.000136 | Y | -0.144 | -0.114 | 0 | 47.2% | 20/21 | 13/21 |
| LONDON_METALS | high | 26 | 21 | 5 | 80.8% | 0.001247 | Y | +0.082 | +0.082 | 0 | 12.5% | 10/13 | 11/13 |
| LONDON_METALS | low | 26 | 20 | 6 | 76.9% | 0.004678 | Y | -0.149 | -0.140 | 0 | 5.6% | 8/13 | 12/13 |
| NYSE_CLOSE | high | 19 | 10 | 9 | 52.6% | 0.500000 | . | +0.001 | -0.004 | 0 | 82.4% | 1/10 | 9/9 |
| NYSE_CLOSE | low | 19 | 14 | 5 | 73.7% | 0.031784 | . | -0.075 | -0.064 | 0 | 100.0% | 7/10 | 7/9 |
| NYSE_OPEN | high | 60 | 14 | 46 | 23.3% | 0.999994 | . | -0.062 | -0.066 | 0 | 75.0% | 2/30 | 12/30 |
| NYSE_OPEN | low | 60 | 30 | 30 | 50.0% | 0.551289 | . | -0.010 | -0.010 | 0 | 52.3% | 14/30 | 16/30 |
| SINGAPORE_OPEN | high | 28 | 24 | 4 | 85.7% | 0.000090 | Y | +0.065 | +0.064 | 0 | 50.0% | 13/14 | 11/14 |
| SINGAPORE_OPEN | low | 28 | 22 | 6 | 78.6% | 0.001860 | Y | -0.136 | -0.127 | 2 | 60.0% | 11/14 | 11/14 |
| TOKYO_OPEN | high | 32 | 30 | 2 | 93.8% | 0.000000 | Y | +0.078 | +0.088 | 0 | 28.1% | 14/16 | 16/16 |
| TOKYO_OPEN | low | 32 | 29 | 3 | 90.6% | 0.000001 | Y | -0.107 | -0.118 | 0 | 91.7% | 13/16 | 16/16 |
| US_DATA_1000 | high | 64 | 45 | 19 | 70.3% | 0.000781 | Y | +0.061 | +0.070 | 0 | 46.0% | 20/32 | 25/32 |
| US_DATA_1000 | low | 64 | 38 | 26 | 59.4% | 0.084321 | . | -0.059 | -0.056 | 0 | 86.8% | 18/32 | 20/32 |
| US_DATA_830 | high | 38 | 18 | 20 | 47.4% | 0.686449 | . | +0.013 | +0.012 | 0 | 28.9% | 7/19 | 11/19 |
| US_DATA_830 | low | 38 | 13 | 25 | 34.2% | 0.983224 | . | +0.051 | +0.060 | 0 | 0.0% | 6/19 | 7/19 |

## Session-side monotonicity / tail-bias test

| Session | Side | Shapes | Support | Oppose | Support % | p_tail | BH | mean tail bias | mean best bucket |
|---|---|---|---|---|---|---|---|---|---|
| BRISBANE_1025 | high | 12 | 7 | 5 | 58.3% | 0.387207 | . | +0.013 | 2.25 |
| BRISBANE_1025 | low | 12 | 5 | 7 | 41.7% | 0.806152 | . | +0.013 | 2.25 |
| CME_PRECLOSE | high | 40 | 27 | 13 | 67.5% | 0.019239 | . | +0.112 | 2.55 |
| CME_PRECLOSE | low | 40 | 13 | 27 | 32.5% | 0.991705 | . | +0.112 | 2.55 |
| CME_REOPEN | high | 16 | 6 | 10 | 37.5% | 0.894943 | . | -0.033 | 2.56 |
| CME_REOPEN | low | 16 | 10 | 6 | 62.5% | 0.227249 | . | -0.033 | 2.56 |
| COMEX_SETTLE | high | 50 | 46 | 4 | 92.0% | 0.000000 | Y | +0.194 | 3.20 |
| COMEX_SETTLE | low | 50 | 4 | 46 | 8.0% | 1.000000 | . | +0.194 | 3.20 |
| EUROPE_FLOW | high | 42 | 30 | 12 | 71.4% | 0.003958 | Y | +0.119 | 2.62 |
| EUROPE_FLOW | low | 42 | 12 | 30 | 28.6% | 0.998556 | . | +0.119 | 2.62 |
| LONDON_METALS | high | 26 | 20 | 6 | 76.9% | 0.004678 | Y | +0.136 | 2.73 |
| LONDON_METALS | low | 26 | 6 | 20 | 23.1% | 0.998753 | . | +0.136 | 2.73 |
| NYSE_CLOSE | high | 17 | 10 | 7 | 58.8% | 0.314529 | . | +0.041 | 2.59 |
| NYSE_CLOSE | low | 17 | 7 | 10 | 41.2% | 0.833847 | . | +0.041 | 2.59 |
| NYSE_OPEN | high | 60 | 19 | 41 | 31.7% | 0.998665 | . | -0.046 | 1.72 |
| NYSE_OPEN | low | 60 | 41 | 19 | 68.3% | 0.003109 | Y | -0.046 | 1.72 |
| SINGAPORE_OPEN | high | 28 | 21 | 7 | 75.0% | 0.006270 | Y | +0.100 | 2.54 |
| SINGAPORE_OPEN | low | 28 | 7 | 21 | 25.0% | 0.998140 | . | +0.100 | 2.54 |
| TOKYO_OPEN | high | 32 | 32 | 0 | 100.0% | 0.000000 | Y | +0.202 | 3.28 |
| TOKYO_OPEN | low | 32 | 0 | 32 | 0.0% | 1.000000 | . | +0.202 | 3.28 |
| US_DATA_1000 | high | 64 | 41 | 23 | 64.1% | 0.016383 | . | +0.061 | 2.38 |
| US_DATA_1000 | low | 64 | 23 | 41 | 35.9% | 0.991571 | . | +0.061 | 2.38 |
| US_DATA_830 | high | 38 | 13 | 25 | 34.2% | 0.983224 | . | -0.066 | 1.74 |
| US_DATA_830 | low | 38 | 25 | 13 | 65.8% | 0.036476 | . | -0.066 | 1.74 |

## Families surviving directional BH

- `COMEX_SETTLE high`: support 50/52, `p_dir=0.000000`, mean `sr_lift=+0.176`
- `EUROPE_FLOW high`: support 38/42, `p_dir=0.000000`, mean `sr_lift=+0.073`
- `TOKYO_OPEN high`: support 30/32, `p_dir=0.000000`, mean `sr_lift=+0.078`
- `COMEX_SETTLE low`: support 44/52, `p_dir=0.000000`, mean `sr_lift=-0.131`
- `TOKYO_OPEN low`: support 29/32, `p_dir=0.000001`, mean `sr_lift=-0.107`
- `CME_PRECLOSE high`: support 34/42, `p_dir=0.000034`, mean `sr_lift=+0.117`
- `SINGAPORE_OPEN high`: support 24/28, `p_dir=0.000090`, mean `sr_lift=+0.065`
- `EUROPE_FLOW low`: support 33/42, `p_dir=0.000136`, mean `sr_lift=-0.144`
- `US_DATA_1000 high`: support 45/64, `p_dir=0.000781`, mean `sr_lift=+0.061`
- `LONDON_METALS high`: support 21/26, `p_dir=0.001247`, mean `sr_lift=+0.082`
- `SINGAPORE_OPEN low`: support 22/28, `p_dir=0.001860`, mean `sr_lift=-0.136`
- `LONDON_METALS low`: support 20/26, `p_dir=0.004678`, mean `sr_lift=-0.149`
- `BRISBANE_1025 high`: support 10/12, `p_dir=0.019287`, mean `sr_lift=+0.048`
- `CME_PRECLOSE low`: support 28/42, `p_dir=0.021779`, mean `sr_lift=-0.072`

## Families surviving monotonicity BH

- `COMEX_SETTLE high`: support 46/50, `p_tail=0.000000`, mean `tail_bias=+0.194`
- `TOKYO_OPEN high`: support 32/32, `p_tail=0.000000`, mean `tail_bias=+0.202`
- `NYSE_OPEN low`: support 41/60, `p_tail=0.003109`, mean `tail_bias=-0.046`
- `EUROPE_FLOW high`: support 30/42, `p_tail=0.003958`, mean `tail_bias=+0.119`
- `LONDON_METALS high`: support 20/26, `p_tail=0.004678`, mean `tail_bias=+0.136`
- `SINGAPORE_OPEN high`: support 21/28, `p_tail=0.006270`, mean `tail_bias=+0.100`

---

## Notes

- This is a family audit, not a production promotion decision.
- Session-side BH asks whether the regime effect clusters naturally; global BH still governs any universal-overlay headline claim.
- Tail-bias is an informational structural check for R3/R7 suitability; it does not replace forward validation.
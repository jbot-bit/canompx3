# Garch Normalized Sizing Audit

**Date:** 2026-04-16
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-normalized-sizing-audit.yaml`
**Purpose:** test garch as an R3 size modifier with normalized weights rather than as a binary filter.

**Grounding:**
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/institutional/mechanism_priors.md`
- `docs/audit/results/2026-04-16-garch-regime-family-audit.md`
- `docs/audit/results/2026-04-16-garch-g0-preflight.md`

## Session profiles used for sizing

| Session | High directional support | Low directional support | High monotonicity support | Low monotonicity support |
|---|---|---|---|---|
| BRISBANE_1025 | Y | . | . | . |
| CME_PRECLOSE | Y | Y | . | . |
| CME_REOPEN | . | . | . | . |
| COMEX_SETTLE | Y | Y | Y | . |
| EUROPE_FLOW | Y | Y | Y | . |
| LONDON_METALS | Y | Y | Y | . |
| NYSE_CLOSE | . | . | . | . |
| NYSE_OPEN | . | . | . | Y |
| SINGAPORE_OPEN | Y | Y | Y | . |
| TOKYO_OPEN | Y | Y | Y | . |
| US_DATA_1000 | Y | . | . | . |
| US_DATA_830 | . | . | . | . |

## Map definitions

- `LOW_CUT_ONLY`: sessions with low-directional support get `0.5x` at `gp<=30`; otherwise `1.0x`.
- `HIGH_BOOST_ONLY`: sessions with high-directional support get `1.5x` at `gp>=70`; otherwise `1.0x`.
- `SESSION_CLIPPED`: combines the two clipped rules above.
- `SESSION_LINEAR`: supported sessions get `clip(0.5 + gp/100, 0.5, 1.5)`; unsupported sessions stay `1.0x`.
- `GLOBAL_LINEAR`: every trade gets `clip(0.5 + gp/100, 0.5, 1.5)`.

All maps are normalized on IS only so mean raw weight becomes `1.0x`. The same normalization factor is then applied unchanged to OOS.

## Broad scope results

| Map | Norm | Weight range | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SESSION_CLIPPED | 0.954 | 0.48-1.43 | +166958.6 | +2104.7 | +0.235 | +36.3 | -2829.8 | -2150.1 | +27256.2 | +0.0112 | +0.0166 | +1.48 |
| SESSION_LINEAR | 0.985 | 0.49-1.48 | +132295.8 | +1602.9 | +0.162 | +62.6 | -2709.8 | -2519.0 | +30547.0 | +0.0085 | +0.0125 | +1.46 |
| GLOBAL_LINEAR | 0.981 | 0.49-1.47 | +110577.9 | +1335.5 | +0.002 | +63.3 | -5392.3 | -9049.3 | +35574.8 | +0.0070 | +0.0124 | +1.78 |
| HIGH_BOOST_ONLY | 0.886 | 0.89-1.33 | +110503.7 | +1440.8 | +0.213 | +48.7 | -1652.7 | -164.5 | +19950.2 | +0.0075 | +0.0132 | +1.75 |
| LOW_CUT_ONLY | 1.087 | 0.54-1.09 | +54699.1 | +631.0 | +0.042 | +47.5 | -1197.2 | -2247.8 | +6587.0 | +0.0035 | +0.0027 | +0.78 |

### Broad best-map contributions: `SESSION_CLIPPED`

| Instrument | Session | Base $ | Alt $ | Δ$ |
|---|---|---|---|---|
| MNQ | COMEX_SETTLE | +181506.3 | +231994.6 | +50488.4 |
| MNQ | EUROPE_FLOW | +99367.4 | +122454.0 | +23086.5 |
| MNQ | TOKYO_OPEN | +63522.6 | +82972.6 | +19450.0 |
| MNQ | US_DATA_1000 | +158455.0 | +177413.4 | +18958.4 |
| MNQ | CME_PRECLOSE | +56822.4 | +74628.5 | +17806.1 |
| MNQ | SINGAPORE_OPEN | +61804.3 | +75251.1 | +13446.8 |
| MES | COMEX_SETTLE | +17591.0 | +26539.4 | +8948.5 |
| MES | CME_PRECLOSE | +21334.8 | +29762.2 | +8427.3 |
| MES | US_DATA_1000 | +13771.4 | +18795.7 | +5024.3 |
| MES | EUROPE_FLOW | +1819.9 | +5325.7 | +3505.9 |
| MNQ | BRISBANE_1025 | +7504.9 | +10893.9 | +3388.9 |
| MGC | LONDON_METALS | +3523.4 | +6779.5 | +3256.1 |
| MES | TOKYO_OPEN | +6538.2 | +8830.1 | +2291.9 |
| MNQ | US_DATA_830 | -41862.5 | -39942.9 | +1919.6 |
| MES | SINGAPORE_OPEN | -284.3 | +386.1 | +670.4 |
## Validated scope results

| Map | Norm | Weight range | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SESSION_CLIPPED | 0.973 | 0.49-1.46 | +54293.4 | +672.4 | +0.190 | +50.3 | -2536.9 | -562.3 | +7986.1 | +0.0156 | +0.0583 | +3.75 |
| GLOBAL_LINEAR | 1.005 | 0.50-1.51 | +42049.4 | +472.6 | +0.047 | +31.8 | -2814.0 | -2589.1 | +9678.5 | +0.0105 | +0.0494 | +4.72 |
| SESSION_LINEAR | 1.002 | 0.50-1.50 | +41111.2 | +490.2 | +0.119 | +27.4 | -2777.7 | -670.8 | +8792.5 | +0.0115 | +0.0398 | +3.46 |
| LOW_CUT_ONLY | 1.120 | 0.56-1.12 | +29733.6 | +286.8 | +0.107 | +36.0 | -664.3 | -860.6 | +2284.9 | +0.0073 | +0.0135 | +1.84 |
| HIGH_BOOST_ONLY | 0.881 | 0.88-1.32 | +25784.1 | +383.4 | +0.138 | +17.2 | -1775.2 | +167.8 | +5435.9 | +0.0084 | +0.0423 | +5.06 |

### Validated best-map contributions: `SESSION_CLIPPED`

| Instrument | Session | Base $ | Alt $ | Δ$ |
|---|---|---|---|---|
| MNQ | COMEX_SETTLE | +81920.8 | +108372.5 | +26451.8 |
| MNQ | EUROPE_FLOW | +45519.6 | +57179.5 | +11659.9 |
| MNQ | US_DATA_1000 | +59368.7 | +65636.7 | +6267.9 |
| MNQ | TOKYO_OPEN | +19760.1 | +25822.7 | +6062.6 |
| MNQ | SINGAPORE_OPEN | +13456.3 | +16080.2 | +2623.9 |
| MNQ | CME_PRECLOSE | +6249.8 | +8592.6 | +2342.7 |
| MES | CME_PRECLOSE | +5502.6 | +7035.5 | +1532.9 |
| MNQ | NYSE_OPEN | +96806.3 | +94158.1 | -2648.3 |

## Reading the audit

- `Full Δ$` and `Full ΔR` answer the total take-home question after normalized sizing.
- `Sharpe Δ` and `MaxDD ΔR` answer the risk-adjusted portfolio question. Positive `MaxDD ΔR` means the drawdown became less severe.
- `OOS retention` compares OOS ExpR uplift to IS ExpR uplift. Positive is directionally good; high ratios are better.
- `Worst day Δ$`, `Worst 5d Δ$`, and `Max daily risk Δ$` are directional risk diagnostics. More negative is worse; more positive is safer.
- `Max daily risk Δ$` is a concentration proxy, not an account-breach simulation.

## Caveats

- This is still a backtest-side utilization audit, not production proof.
- Fractional weights are a research abstraction for normalized sizing. Live implementation would need contract rounding and account budgeting.
- No map was tuned after results; only the pre-committed five maps were run.

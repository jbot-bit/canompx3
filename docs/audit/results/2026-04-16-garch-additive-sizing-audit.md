# Garch Additive Sizing Audit

**Date:** 2026-04-16
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-additive-sizing-audit.yaml`
**Purpose:** compare garch to ATR / overnight proxies inside the same clipped session-sizing scaffold.

## Session scaffold

| Session | High directional support | Low directional support | High monotonicity support |
|---|---|---|---|
| BRISBANE_1025 | Y | . | . |
| CME_PRECLOSE | Y | Y | . |
| CME_REOPEN | . | . | . |
| COMEX_SETTLE | Y | Y | Y |
| EUROPE_FLOW | Y | Y | Y |
| LONDON_METALS | Y | Y | Y |
| NYSE_CLOSE | . | . | . |
| NYSE_OPEN | . | . | . |
| SINGAPORE_OPEN | Y | Y | Y |
| TOKYO_OPEN | Y | Y | Y |
| US_DATA_1000 | Y | . | . |
| US_DATA_830 | . | . | . |

## Map definitions

- `GARCH_SESSION_CLIPPED`: session-clipped map using `garch_forecast_vol_pct`.
- `ATR_SESSION_CLIPPED`: same session-clipped map using `atr_20_pct`.
- `OVN_SESSION_CLIPPED`: same session-clipped map using `overnight_range_pct`.
- `GARCH_ATR_MEAN_CLIPPED`: same scaffold using mean(`garch_pct`, `atr_20_pct`).
- `GARCH_OVN_MEAN_CLIPPED`: same scaffold using mean(`garch_pct`, `overnight_range_pct`).

## Broad scope

| Map | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |
|---|---|---|---|---|---|---|---|---|---|---|
| OVN_SESSION_CLIPPED | +185047.7 | +2982.4 | +0.499 | +65.4 | -1862.3 | -1637.7 | +24782.2 | +0.0166 | +0.0135 | +0.81 |
| GARCH_OVN_MEAN_CLIPPED | +172379.1 | +2375.4 | +0.355 | +92.3 | -2744.4 | -2902.9 | +26373.6 | +0.0131 | +0.0126 | +0.97 |
| GARCH_SESSION_CLIPPED | +166854.2 | +2104.4 | +0.240 | +56.0 | -2828.6 | -2148.2 | +27249.3 | +0.0112 | +0.0166 | +1.48 |
| GARCH_ATR_MEAN_CLIPPED | +127129.8 | +1476.7 | +0.122 | +71.4 | -2414.4 | -485.3 | +24432.2 | +0.0078 | +0.0129 | +1.66 |
| ATR_SESSION_CLIPPED | +101599.6 | +834.7 | -0.013 | +43.3 | -1876.6 | +355.8 | +21267.3 | +0.0042 | +0.0104 | +2.48 |

### Broad best-map contributions: `OVN_SESSION_CLIPPED`

| Instrument | Session | Δ$ |
|---|---|---|
| MNQ | TOKYO_OPEN | +53447.3 |
| MNQ | COMEX_SETTLE | +33056.0 |
| MNQ | SINGAPORE_OPEN | +30213.2 |
| MNQ | EUROPE_FLOW | +25527.6 |
| MNQ | US_DATA_1000 | +21926.6 |
| MNQ | BRISBANE_1025 | +7613.3 |
| MES | COMEX_SETTLE | +7426.0 |
| MGC | LONDON_METALS | +5311.6 |
| MES | TOKYO_OPEN | +4465.0 |
| MES | US_DATA_1000 | +3772.4 |
| MES | EUROPE_FLOW | +3167.9 |
| MNQ | US_DATA_830 | +2879.7 |
| MES | CME_PRECLOSE | +2630.2 |
| MES | SINGAPORE_OPEN | +1747.4 |
| MNQ | LONDON_METALS | +1683.5 |
## Validated scope

| Map | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |
|---|---|---|---|---|---|---|---|---|---|---|
| GARCH_SESSION_CLIPPED | +54293.4 | +672.4 | +0.190 | +50.3 | -2536.9 | -562.3 | +7986.1 | +0.0156 | +0.0583 | +3.75 |
| OVN_SESSION_CLIPPED | +53114.7 | +837.7 | +0.409 | +41.3 | -2272.6 | -369.3 | +7101.2 | +0.0226 | +0.0178 | +0.79 |
| GARCH_OVN_MEAN_CLIPPED | +49788.4 | +683.1 | +0.270 | +43.9 | -2440.2 | -707.2 | +7662.5 | +0.0171 | +0.0366 | +2.14 |
| GARCH_ATR_MEAN_CLIPPED | +40498.3 | +449.5 | +0.036 | +16.5 | -2275.9 | -177.3 | +7112.3 | +0.0103 | +0.0402 | +3.89 |
| ATR_SESSION_CLIPPED | +34169.4 | +312.0 | -0.039 | -16.7 | -2021.1 | +62.4 | +6259.4 | +0.0065 | +0.0388 | +5.93 |

### Validated best-map contributions: `GARCH_SESSION_CLIPPED`

| Instrument | Session | Δ$ |
|---|---|---|
| MNQ | COMEX_SETTLE | +26451.8 |
| MNQ | EUROPE_FLOW | +11659.9 |
| MNQ | US_DATA_1000 | +6267.9 |
| MNQ | TOKYO_OPEN | +6062.6 |
| MNQ | SINGAPORE_OPEN | +2623.9 |
| MNQ | CME_PRECLOSE | +2342.7 |
| MES | CME_PRECLOSE | +1532.9 |
| MNQ | NYSE_OPEN | -2648.3 |

## Reading the audit

- This is an additive-value comparison, not a new discovery sweep.
- Positive `MaxDD ΔR` means drawdown became less severe.
- Positive `Worst day/5d Δ$` means the loss became smaller in magnitude.
- `Max daily risk Δ$` is a concentration proxy, not a breach simulation.

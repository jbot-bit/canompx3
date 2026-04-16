# Garch Discrete Policy Surface Audit

**Date:** 2026-04-16
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-discrete-policy-surface-audit.yaml`
**Purpose:** compare raw discrete trading policies for `garch_forecast_vol_pct` on canonical trade rows only.

**Grounding:**
- `docs/institutional/literature/chan_2008_ch7_regime_switching.md`
- `docs/institutional/mechanism_priors.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/audit/results/2026-04-16-garch-regime-family-audit.md`
- `docs/audit/results/2026-04-16-garch-g0-preflight.md`

**Raw-number rule:** all policy results are recomputed from canonical `orb_outcomes` + `daily_features` trade rows. No stored expectancy metadata is trusted.

## Session directional support used by session-aware policies

| Session | High directional support | Low directional support |
|---|---|---|
| BRISBANE_1025 | Y | . |
| CME_PRECLOSE | Y | Y |
| CME_REOPEN | . | . |
| COMEX_SETTLE | Y | Y |
| EUROPE_FLOW | Y | Y |
| LONDON_METALS | Y | Y |
| NYSE_CLOSE | . | . |
| NYSE_OPEN | . | . |
| SINGAPORE_OPEN | Y | Y |
| TOKYO_OPEN | Y | Y |
| US_DATA_1000 | Y | . |
| US_DATA_830 | . | . |

## Policy definitions

- `SESSION_TAKE_HIGH_ONLY`: trade only `gp>=70` in sessions with high-directional support; other sessions stay base `1x`.
- `GLOBAL_TAKE_HIGH_ONLY`: trade only `gp>=70` everywhere.
- `SESSION_SKIP_LOW_ONLY`: skip `gp<=30` in sessions with low-directional support; other sessions stay base `1x`.
- `GLOBAL_SKIP_LOW_ONLY`: skip `gp<=30` everywhere.
- `SESSION_HIGH_2X_ONLY`: double size on `gp>=70` in sessions with high-directional support; otherwise `1x`.
- `GLOBAL_HIGH_2X_ONLY`: double size on `gp>=70` everywhere.
- `SESSION_CLIPPED_0_1_2`: `2x` on high-supported `gp>=70`, `0x` on low-supported `gp<=30`, else `1x`.
- `GLOBAL_CLIPPED_0_1_2`: `2x` on `gp>=70`, `0x` on `gp<=30`, else `1x` everywhere.

All actions are raw integer counts per trade (`0`, `1`, or `2`). No fractional sizing is used here.

## Broad scope results

| Policy | Active % | Mean contracts | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLOBAL_HIGH_2X_ONLY | 100.0% | 1.373 | +517139.3 | +6745.6 | +0.054 | -105.6 | -14462.5 | -25859.3 | +75666.9 | +0.0359 | +0.0536 | +1.50 |
| SESSION_HIGH_2X_ONLY | 100.0% | 1.266 | +483864.0 | +6293.0 | +0.288 | -58.9 | -7470.2 | -7063.9 | +64405.2 | +0.0337 | +0.0464 | +1.38 |
| SESSION_CLIPPED_0_1_2 | 84.6% | 1.112 | +437927.5 | +5552.7 | +0.254 | -209.8 | -7334.7 | -7016.9 | +64405.2 | +0.0297 | +0.0410 | +1.38 |
| GLOBAL_CLIPPED_0_1_2 | 71.1% | 1.084 | +330370.1 | +4558.8 | -0.175 | -224.2 | -14327.1 | -25812.3 | +75666.9 | +0.0234 | +0.0487 | +2.08 |
| SESSION_SKIP_LOW_ONLY | 84.6% | 0.846 | -45936.5 | -740.3 | -0.000 | -73.7 | +135.5 | +47.0 | +0.0 | -0.0040 | -0.0054 | +1.35 |
| GLOBAL_SKIP_LOW_ONLY | 71.1% | 0.711 | -186769.2 | -2186.8 | -0.220 | -8.8 | +135.5 | +47.0 | +0.0 | -0.0125 | -0.0049 | +0.39 |
| SESSION_TAKE_HIGH_ONLY | 55.7% | 0.557 | -231267.0 | -3502.9 | -0.339 | -162.4 | +135.5 | -1892.9 | +0.0 | -0.0198 | -0.0114 | +0.58 |
| GLOBAL_TAKE_HIGH_ONLY | 37.3% | 0.373 | -398014.4 | -5125.1 | -0.482 | +94.9 | +135.5 | +255.5 | +0.0 | -0.0293 | -0.0115 | +0.39 |

### Broad best-policy contributions: `GLOBAL_HIGH_2X_ONLY`

| Instrument | Session | Base $ | Alt $ | Δ$ |
|---|---|---|---|---|
| MNQ | COMEX_SETTLE | +181506.3 | +333091.9 | +151585.6 |
| MNQ | EUROPE_FLOW | +99367.4 | +169713.8 | +70346.4 |
| MNQ | US_DATA_1000 | +158455.0 | +213424.4 | +54969.4 |
| MNQ | NYSE_OPEN | +219547.4 | +273144.2 | +53596.9 |
| MNQ | CME_PRECLOSE | +56822.4 | +103342.4 | +46520.0 |
| MNQ | TOKYO_OPEN | +63522.6 | +106605.7 | +43083.1 |
| MNQ | SINGAPORE_OPEN | +61804.3 | +97292.6 | +35488.3 |
| MES | CME_PRECLOSE | +21334.8 | +42496.8 | +21161.9 |
| MES | COMEX_SETTLE | +17591.0 | +37948.6 | +20357.6 |
| MES | US_DATA_1000 | +13771.4 | +25626.6 | +11855.2 |
| MNQ | BRISBANE_1025 | +7504.9 | +15329.9 | +7824.9 |
| MES | TOKYO_OPEN | +6538.2 | +13300.6 | +6762.4 |
| MNQ | LONDON_METALS | +25707.7 | +31891.1 | +6183.5 |
| MGC | LONDON_METALS | +3523.4 | +9402.3 | +5878.9 |
| MES | EUROPE_FLOW | +1819.9 | +7238.9 | +5419.0 |
## Validated scope results

| Policy | Active % | Mean contracts | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLOBAL_HIGH_2X_ONLY | 100.0% | 1.343 | +173646.2 | +2189.4 | +0.069 | +0.0 | -5546.0 | -6874.4 | +19075.0 | +0.0526 | +0.1576 | +3.00 |
| SESSION_HIGH_2X_ONLY | 100.0% | 1.281 | +147316.3 | +1993.9 | +0.141 | +0.0 | -5528.4 | -1961.5 | +17494.4 | +0.0482 | +0.1383 | +2.87 |
| SESSION_CLIPPED_0_1_2 | 79.2% | 1.073 | +130124.2 | +1616.6 | +0.084 | +36.9 | -5528.4 | -1961.5 | +17494.4 | +0.0381 | +0.1288 | +3.38 |
| GLOBAL_CLIPPED_0_1_2 | 68.0% | 1.023 | +101976.1 | +1266.3 | -0.272 | +23.7 | -5546.0 | -6874.4 | +19075.0 | +0.0263 | +0.1629 | +6.20 |
| SESSION_SKIP_LOW_ONLY | 79.2% | 0.792 | -17192.1 | -377.3 | +0.041 | +71.1 | +0.0 | +0.0 | +0.0 | -0.0101 | -0.0096 | +0.95 |
| GLOBAL_SKIP_LOW_ONLY | 68.0% | 0.680 | -71670.1 | -923.1 | -0.321 | +66.1 | +0.0 | +0.0 | +0.0 | -0.0263 | +0.0053 | -0.20 |
| SESSION_TAKE_HIGH_ONLY | 47.1% | 0.471 | -84461.6 | -1336.0 | -0.250 | +70.2 | +0.0 | -286.4 | +0.0 | -0.0379 | +0.0053 | -0.14 |
| GLOBAL_TAKE_HIGH_ONLY | 34.3% | 0.343 | -154938.1 | -1968.8 | -0.706 | +44.7 | +0.0 | +310.1 | +0.0 | -0.0555 | +0.0006 | -0.01 |

### Validated best-policy contributions: `GLOBAL_HIGH_2X_ONLY`

| Instrument | Session | Base $ | Alt $ | Δ$ |
|---|---|---|---|---|
| MNQ | COMEX_SETTLE | +81920.8 | +152148.1 | +70227.3 |
| MNQ | EUROPE_FLOW | +45519.6 | +77870.9 | +32351.3 |
| MNQ | NYSE_OPEN | +96806.3 | +123136.2 | +26329.9 |
| MNQ | US_DATA_1000 | +59368.7 | +75596.8 | +16228.0 |
| MNQ | TOKYO_OPEN | +19760.1 | +31702.3 | +11942.2 |
| MNQ | SINGAPORE_OPEN | +13456.3 | +20044.2 | +6587.9 |
| MNQ | CME_PRECLOSE | +6249.8 | +11800.6 | +5550.8 |
| MES | CME_PRECLOSE | +5502.6 | +9931.3 | +4428.7 |

## Reading the audit

- `Full Δ$` and `Full ΔR` answer the total take-home question under raw discrete actions.
- `Active %` shows how much of the book remains active; `Mean contracts` shows average raw action intensity.
- `Sharpe Δ`, `MaxDD ΔR`, `Worst day Δ$`, and `Worst 5d Δ$` answer whether the policy improves or worsens path quality.
- `Max daily risk Δ$` is a concentration proxy, not an account-breach simulation.
- `OOS retention` compares OOS ExpR uplift to IS ExpR uplift. It is directional support, not clean deployment proof.

## Caveats

- This is still a backtest-side policy audit, not production proof.
- Session-aware policies rely on raw family directional support from the regime-family audit; they do not discover new families here.
- Profile translation, contract ceilings, and copier arithmetic are separate downstream questions.

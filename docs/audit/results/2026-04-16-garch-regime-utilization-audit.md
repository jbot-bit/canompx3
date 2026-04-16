# Garch Regime Utilization Audit

**Date:** 2026-04-16
**Question:** if a session family shows a garch regime effect, how does that translate into actual gate economics?

Two gate modes are reported:
- `TAKE_HIGH_ONLY`: trade only when `garch_pct >= 70`
- `SKIP_LOW_ONLY`: trade unless `garch_pct <= 30`

**Important:** these are informational exploitation numbers on the current pooled family populations. They are not deployment proof.

| Session | Rule | Active % | Base ExpR | Active ExpR | Base Total R | Active Total R | Delta Total R | Skipped Wins R | Skipped Losses R | Saved-Missed R |
|---|---|---|---|---|---|---|---|---|---|---|
| BRISBANE_1025 | SKIP_LOW_ONLY | 65.9% | +0.010 | +0.026 | +53.2 | +92.8 | +39.6 | +964.4 | -1004.0 | +39.6 |
| BRISBANE_1025 | TAKE_HIGH_ONLY | 33.1% | +0.010 | +0.054 | +53.2 | +97.2 | +44.0 | +1919.0 | -1963.0 | +44.0 |
| CME_PRECLOSE | SKIP_LOW_ONLY | 74.0% | +0.120 | +0.148 | +1208.6 | +1101.6 | -107.1 | +1430.1 | -1323.0 | -107.1 |
| CME_PRECLOSE | TAKE_HIGH_ONLY | 39.0% | +0.120 | +0.193 | +1208.6 | +759.5 | -449.1 | +3424.1 | -2975.0 | -449.1 |
| COMEX_SETTLE | SKIP_LOW_ONLY | 68.7% | +0.093 | +0.111 | +2000.5 | +1634.2 | -366.3 | +3670.3 | -3304.0 | -366.3 |
| COMEX_SETTLE | TAKE_HIGH_ONLY | 34.8% | +0.093 | +0.201 | +2000.5 | +1495.9 | -504.6 | +7518.6 | -7014.0 | -504.6 |
| EUROPE_FLOW | SKIP_LOW_ONLY | 66.3% | +0.079 | +0.097 | +1536.8 | +1257.2 | -279.6 | +3548.6 | -3269.0 | -279.6 |
| EUROPE_FLOW | TAKE_HIGH_ONLY | 33.1% | +0.079 | +0.121 | +1536.8 | +781.2 | -755.6 | +7157.6 | -6402.0 | -755.6 |
| LONDON_METALS | SKIP_LOW_ONLY | 67.5% | +0.037 | +0.061 | +401.1 | +446.4 | +45.3 | +1871.7 | -1917.0 | +45.3 |
| LONDON_METALS | TAKE_HIGH_ONLY | 33.9% | +0.037 | +0.071 | +401.1 | +259.6 | -141.5 | +3922.5 | -3781.0 | -141.5 |
| SINGAPORE_OPEN | SKIP_LOW_ONLY | 75.6% | +0.061 | +0.078 | +675.4 | +657.3 | -18.1 | +1455.1 | -1437.0 | -18.1 |
| SINGAPORE_OPEN | TAKE_HIGH_ONLY | 40.7% | +0.061 | +0.100 | +675.4 | +449.9 | -225.5 | +3675.6 | -3450.0 | -225.5 |
| TOKYO_OPEN | SKIP_LOW_ONLY | 69.5% | +0.064 | +0.096 | +964.7 | +1005.2 | +40.5 | +2462.5 | -2503.0 | +40.5 |
| TOKYO_OPEN | TAKE_HIGH_ONLY | 35.5% | +0.064 | +0.111 | +964.7 | +591.0 | -373.6 | +5480.6 | -5107.0 | -373.6 |
| US_DATA_1000 | SKIP_LOW_ONLY | 70.1% | +0.088 | +0.092 | +2062.3 | +1501.3 | -560.9 | +4077.9 | -3517.0 | -560.9 |
| US_DATA_1000 | TAKE_HIGH_ONLY | 36.1% | +0.088 | +0.120 | +2062.3 | +1017.0 | -1045.3 | +8590.3 | -7545.0 | -1045.3 |

## Reading the table

- `Active ExpR` answers the quality question: how good are the trades you keep?
- `Delta Total R` answers the portfolio question: do you actually make more or less total R by gating?
- `Saved-Missed R = abs(skipped losses) - skipped wins` is the cleanest decomposition of what the gate is doing.
- `TAKE_HIGH_ONLY` is the strict regime-only interpretation.
- `SKIP_LOW_ONLY` is the softer hostile-regime filter interpretation.
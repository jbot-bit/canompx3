# GARCH R3 Session-Clipped Shadow Ledger

**Date:** 2026-04-19
**Profile:** `topstep_50k_mnq_auto` (`topstep`, `50,000`, copies=2, stop=0.75x, active=True)
**Pre-reg:** `draft PR #65 / research(garch): add R3 shadow prereg / commit b73914c9`
**Policy:** `SESSION_CLIPPED` with frozen normalization factor `0.973`
**Forward window:** `2026-01-01` to `2026-04-19`
**Source of truth:** current profile lane set + canonical forward outcomes + daily_features.garch_forecast_vol_pct
**Execution status:** shadow-only; no live sizing change

## Verdict

- Launch status: **READY_FOR_FORWARD_MONITORING**
- Gate reason: `state coverage is within prereg launch tolerance`
- This is the next action from the stale-lock audit: operationalize the surviving GARCH `R3` path as a frozen monitoring ledger.
- This is not a router, not a filter scan, and not a deployment claim.
- The translated `0/1/2` contract column is diagnostic only; the primary shadow path is the normalized weight.

## Coverage

- Trades covered: `406`
- Trading days covered: `72`
- Trades with non-neutral raw signal: `222` (`54.7%`)
- Trades with non-1x normalized shadow weight: `406` (`100.0%`)
- Neutral 1x trades: `184` (`45.3%`)
- Missing-state fallback trades: `0` (`0.0%`)
- Translated 0/1/2 contract changes: `38` (`9.4%`)
- Raw feature-gap rows in `daily_features`: `0`

## Shadow vs Base

| Path | Total R | Delta R | Total $ | Delta $ | Daily Sharpe Δ (R) | Daily Sharpe Δ ($) | MaxDD ΔR | MaxDD Δ$ |
|---|---|---|---|---|---|---|---|---|
| Shadow weight | +38.033 | +7.962 | +5,454.1 | +951.8 | -0.058 | -0.012 | -4.034 | -108.8 |
| Translated 0/1/2 | +23.411 | -6.661 | +4,235.4 | -267.0 | -0.534 | -0.177 | +0.000 | +0.0 |

## Risk diagnostics

- Worst day delta $: `-258.1`
- Worst 5-day delta $: `-407.0`

## Session contribution

| Session | Trades | Delta R | Delta $ |
|---|---|---|---|
| US_DATA_1000 | 66 | +2.497 | +485.9 |
| COMEX_SETTLE | 68 | +4.596 | +342.1 |
| EUROPE_FLOW | 72 | +3.733 | +222.5 |
| TOKYO_OPEN | 70 | +0.369 | +125.8 |
| NYSE_OPEN | 71 | -0.231 | -57.3 |
| SINGAPORE_OPEN | 59 | -3.004 | -167.1 |

## Artifacts

- Trade ledger: `data/forward_monitoring/garch-r3-session-clipped-shadow-topstep-50k-mnq-auto-trades.csv`
- Daily ledger: `data/forward_monitoring/garch-r3-session-clipped-shadow-topstep-50k-mnq-auto-daily.csv`

## Notes

- Session support and normalization are frozen from the prior audit lineage; this script does not recompute them from forward data.
- Missing `garch_forecast_vol_pct` falls back to neutral `1.0x` by construction.
- Because the normalization factor is below 1.0, even neutral raw signals become `0.973x`; that is why normalized weight changes appear on every trade.
- Current output uses canonical forward outcomes only. If a dedicated live shadow stream is added later, it should replace this source without changing the policy surface.
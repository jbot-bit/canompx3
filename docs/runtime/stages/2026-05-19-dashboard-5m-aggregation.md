---
task: Aggregate dashboard chart from 1-minute bars to 5-minute candles (ORB-aligned), with toggle 1m/5m/15m
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.html
blast_radius: "Presentational HTML/JS only inside trading_app/live/bot_dashboard.html. Adds client-side OHLC aggregation function aggregateBars(bars, intervalSec), wires it into ChartCockpit.loadInitial and ChartCockpit.pushBar paths, plus a 3-button timeframe toggle (1m/5m/15m). No backend touched, no API change, no schema change. Reads /api/bars-recent same as before; produces the same shape for series.setData. Aggregation rules: align bars to interval boundary (epoch_sec % intervalSec == 0), partial trailing bar is labeled live (via series.update). Reads none; writes none; affects chart rendering only."
---

## Why
Operator runs ORB strategies on 5m/15m/30m windows but chart shows 1m candles — too noisy to read at a glance. Aggregating to 5m default + 15m option matches ORB visualisation needs without touching any backend or canonical source.

## What changes
1. JS helper `aggregateBars(barsArray, intervalSec)` — pure function, returns aggregated OHLC array.
2. `ChartCockpit.loadInitial` calls aggregate before `series.setData`.
3. `ChartCockpit.pushBar` accumulates new 1m bars into the current open 5m bucket, calls `series.update` (LightweightCharts v5 official API for live bar).
4. Timeframe toggle (1m / 5m / 15m) in the chart header — sets `currentInterval` and re-renders.

## Blast Radius
- trading_app/live/bot_dashboard.html — ~70 lines net add (aggregator, toggle UI/CSS, wire-in)
- No Python files
- No DB, no API endpoint, no schema
- No tests required (pure presentational; aggregator is unit-testable as plain JS but no test infra exists for HTML JS in this repo)
- Downstream consumers: zero

## Acceptance
- Default view = 5m candles, aligned to UTC :00/:05/:10 boundaries.
- Toggle to 1m → identical to pre-change behavior.
- Toggle to 15m → wider bars, ORB-15 strategy view.
- Trailing partial bar updates live (no look-ahead bias: it visually shows as the in-progress bar).
- Aggregation: open = first sub.open, close = last sub.close, high = max(highs), low = min(lows), volume = sum(volumes).
- No NaN / undefined bars emitted.

## Bias / correctness check
- Boundary alignment uses UTC epoch seconds modulo interval — same convention as pipeline (UTC throughout).
- Partial bars are emitted but operator can SEE which is partial via the live-update animation; no claim of "completed bar".
- No interpolation: gaps in 1m feed propagate as gaps (correct — do not fabricate data).

# Daily Trade Sheet — Design

**Date:** 2026-03-03
**Status:** Approved

## Purpose

Generate a self-contained HTML file showing every resolved, cost-filtered trade for the day. Grouped by session time (Brisbane), all 4 instruments on one page. Only shows strategies that passed the dollar gate. No dead trades.

## Command

```bash
python scripts/tools/generate_trade_sheet.py
# Outputs trade_sheet.html at project root, opens in browser
```

## Data Flow

```
build_live_portfolio(instrument) x4 (MGC, MNQ, MES, M2K)
  → only active strategies (weight > 0, dollar gate passed)
  → resolve session times for today via dst.py
  → group by session, sort by Brisbane time
  → render HTML
  → write trade_sheet.html
```

## Columns

| Column | Source |
|--------|--------|
| Time (Brisbane) | `dst.py` resolved for today |
| Session | `orb_label` |
| Instrument | `instrument` |
| ORB Aperture | Parsed from strategy_id (`_O15`/`_O30`, default 5m) |
| Direction Rule | Parsed from filter_type (`DIR_LONG` → LONG ONLY, `_CONT` → CONT ONLY, else ANY) |
| Filter (plain English) | Map filter_type to human description (e.g. `ORB_G5` → "ORB >= 5 pts") |
| RR Target | `rr_target` |
| Win Rate | `win_rate` |
| ExpR | `expectancy_r` |
| Expected $/trade | Computed from `expectancy_r * (median_risk_points * point_value + total_friction)` |

## Exclusions

- Dollar gate failures (already excluded by `build_live_portfolio`)
- No variant found (already excluded)
- REGIME strategies gated OFF (weight=0)

## Visual

- Cards per session, ordered by time
- Green accent for ExpR > 0.20
- Red warning badge if fitness = WATCH or DECAY
- Header: date + generation timestamp
- Mobile-friendly, printable

## Files

- **New:** `scripts/tools/generate_trade_sheet.py`
- **Output:** `trade_sheet.html` (gitignored)
- **Dependencies:** `trading_app.live_config`, `pipeline.dst`, `pipeline.cost_model`, `pipeline.asset_configs`

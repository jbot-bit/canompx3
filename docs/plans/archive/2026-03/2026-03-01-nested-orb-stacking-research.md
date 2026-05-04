---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Research Plan: Nested ORB Stacking (5m vs 30m Alignment)

**Date:** 2026-03-01
**Script:** `research/research_nested_orb_stacking.py`
**Output:** `research/output/nested_orb_stacking_findings.md`

## Hypothesis

**H0:** 5m ORB break outcomes are independent of whether the break is contained within or exceeds the 30m ORB range.
**H1:** Multi-timeframe alignment (5m break exceeding 30m boundary) produces different outcomes than contained breaks.

## Classification Logic

For a given (trading_day, symbol, session):
- Get 5m ORB high/low from `daily_features WHERE orb_minutes = 5`
- Get 30m ORB high/low from `daily_features WHERE orb_minutes = 30`
- **Aligned:** For longs, `orb_5m_high >= orb_30m_high` (5m range equals 30m on break side -> any 5m break also breaks 30m). For shorts, `orb_5m_low <= orb_30m_low`.
- **Contained:** 5m ORB strictly inside 30m range on break side -> 5m break stays within wider consolidation.

## CRITICAL: Look-Ahead Issue

A 5m break can occur at minute 6, but the 30m ORB isn't finalized until minute 30. The classification using FINAL 30m ORB is look-ahead for early breaks.

**Two-phase approach:**
- **Phase 1 (Option A):** Use finalized 30m ORB. Label as "descriptive/characterization only." Quick SQL-only. If nothing here, stop.
- **Phase 2 (Option C, conditional):** Compute ORB-at-break-time from `bars_1m`. Fully honest, no look-ahead. Only if Phase 1 shows signal.

## Data Sources

- `daily_features` (orb_minutes=5): `orb_{session}_high`, `orb_{session}_low`, `orb_{session}_break_dir`
- `daily_features` (orb_minutes=30): `orb_{session}_high`, `orb_{session}_low`
- `orb_outcomes` (orb_minutes=5): `pnl_r`, `entry_model`, `rr_target`, `confirm_bars`
- Self-join daily_features on (trading_day, symbol) across orb_minutes=5 and orb_minutes=30

## Statistical Tests

1. **Primary:** Welch's t-test, aligned vs contained pnl_r per (instrument, session, entry_model, rr_target, cb, filter)
2. **Secondary:** One-sample t-test per group vs zero
3. **Direction sub-analysis:** Longs-only, shorts-only splits
4. **30m break concordance:** Does direction alignment (5m and 30m same direction) matter?

## Test Count

~200-400 effective tests after N>=30 filtering (from ~1,920 raw combos: 4 instruments x 8 sessions x 2 entry models x 5 RR x 2 CB x 3 filters).

**BH FDR mandatory** at q=0.10 across ALL tests.

## Edge Cases

1. **30m ORB = 5m ORB** (no expansion after minute 5): All breaks are "aligned" by definition. Report % per session.
2. **30m ORB doesn't form:** Filter out `WHERE orb_30m_high IS NOT NULL`.
3. **Proxy for ORB size:** If effect disappears after controlling for 5m/30m ORB size, it's just a G4+ proxy, not a new finding.

## Script Pattern

Follow `research/research_cross_session_deep.py` for BH FDR, year-by-year, honest summary format.
Follow `research/research_break_quality_bars.py` for bars_1m access (Phase 2 only).

## Canonical Imports

```python
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, get_enabled_sessions
from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
```

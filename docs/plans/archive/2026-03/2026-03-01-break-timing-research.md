---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Research Plan: Time-of-Session Break Timing

**Date:** 2026-03-01
**Script:** `research/research_break_timing.py`
**Output:** `research/output/break_timing_summary.md` + `research/output/break_timing_detail.csv`

## Hypothesis

**H0:** ORB break outcomes are independent of when the break occurs within the session window.
**H1:** Early and late breaks have systematically different outcome distributions.

## Key Discovery: Data Already Exists

`daily_features` already stores `orb_{label}_break_delay_min` (minutes from ORB end to first break bar) computed by `detect_break()` in `build_daily_features.py`. **No bars_1m reconstruction needed.**

## Existing Infrastructure

- `BreakSpeedFilter` class in `config.py` already supports `max_delay_min` parameter
- `BRK_FAST5` (5 min) and `BRK_FAST10` (10 min) already in discovery grid for 3 sessions
- C3 finding: TOKYO_OPEN slow breaks (>3 min confirm) are toxic. Related but different metric.

## Timing Buckets

| Bucket | Range (min) | Label | Mechanism |
|--------|-------------|-------|-----------|
| Immediate | 0-5 | IMM | Impulse momentum |
| Early | 5-15 | EARLY | Quick follow-through |
| Mid-Early | 15-30 | MID_E | Institutional accumulation |
| Mid-Late | 30-60 | MID_L | Less conviction |
| Late | 60+ | LATE | Exhaustion / secondary catalyst |

**Note:** Session window lengths vary (CME_REOPEN ~60 min, TOKYO_OPEN ~360 min). Script must compute empirical distribution per session first.

## No Look-Ahead

Break timing is known at the moment of entry for ALL entry models:
- E1: entry is after confirm bars, which come after the break
- E2: entry is on the break bar itself (stop-market fill)
- E3: entry is after break + retrace

This is a valid real-time filter.

## Data Source

```sql
SELECT o.trading_day, o.symbol, o.orb_label, o.pnl_r,
       o.entry_model, o.rr_target, o.confirm_bars,
       df.orb_{session}_break_delay_min AS break_delay_min,
       df.orb_{session}_size AS orb_size
FROM orb_outcomes o
JOIN daily_features df ON o.trading_day = df.trading_day
  AND o.symbol = df.symbol AND o.orb_minutes = df.orb_minutes
WHERE o.orb_minutes = 5 AND o.outcome IN ('win','loss')
  AND o.pnl_r IS NOT NULL AND df.orb_{session}_break_delay_min IS NOT NULL
```

## Statistical Tests

1. **Bucket-vs-rest Welch's t-test** per bucket
2. **Spearman rank correlation** (break_delay vs pnl_r) -- monotonic trend test
3. **Cutoff scan** at 5, 10, 15, 20, 30, 45, 60 min thresholds -- fast vs slow split
4. **Year-by-year stability** for any surviving finding

## Representative Combos (avoid N-inflation)

```python
COMBOS = [("E1", 2.0, 2), ("E2", 2.5, 1), ("E2", 2.0, 1), ("E1", 2.5, 2)]
```

## Test Count

~100-200 effective tests. **BH FDR at q=0.05.**

## Connection to Existing Work

- **BRK_FAST5/FAST10:** This research provides the full distribution and optimal threshold (may differ from 5/10)
- **C3 (slow breaks):** C3 uses confirm bar timing, this uses break_delay_min directly. Complementary.
- **T80 time-stop:** T80 = time after entry to exit. Break timing = time before entry. Cross-tabulation (fast break + fast resolution) is natural extension.

## Script Pattern

Follow `research/research_e3_fill_timing.py` for time-bucket analysis structure.
Follow `research/research_break_quality_bars.py` for t-test methodology and BH FDR.

## Canonical Imports

```python
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, get_enabled_sessions
from pipeline.paths import GOLD_DB_PATH
```

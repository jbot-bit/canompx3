# WF_START_OVERRIDE Per-Instrument Audit Design

**Date:** 2026-04-09
**Status:** IMPLEMENTED
**Commit:** (see git log)

## Decision

Add MNQ and MES to WF_START_OVERRIDE with date 2020-01-01 based on structural data audit across 5 independent variables.

## Data audit results (execution evidence, not metadata)

### MNQ 2019 vs 2020+ (micro launched 2019-05-06)

| Variable | 2019 | 2020+ avg | Ratio | Verdict |
|---|---|---|---|---|
| ATR | 113.7 | 279.5 | 0.42x | Structural break |
| CME_PRECLOSE G8 pass | 39.0% | 97.7% | 0.40x | Broken |
| EUROPE_FLOW G8 pass | 22.8% | 83.5% | 0.27x | Broken |
| COMEX_SETTLE G8 pass | 30.5% | 93.2% | 0.33x | Broken |
| TOKYO_OPEN G8 pass | 32.7% | 77.2% | 0.42x | Broken |
| NYSE_OPEN G8 pass | 97.1% | 99.4% | 0.98x | Fine |
| NYSE_OPEN volume | 5,689 | 30,845 | 0.16x | Very thin |
| CME_PRECLOSE volume | 1,672 | 7,980 | 0.18x | Very thin |

### MES 2019 vs 2020+

| Variable | 2019 | 2020+ avg | Ratio |
|---|---|---|---|
| ATR | 33.2 | 63.9 | 0.52x |
| NYSE_OPEN G8 pass | 10.5% | 58.1% | 0.18x |
| CME_PRECLOSE G8 pass | 1.2% | 25.8% | 0.05x |
| NYSE_OPEN volume | 6,896 | 24,055 | 0.29x |

### Monthly validation (Q3/Q4 2019 does NOT normalize)

MNQ CME_PRECLOSE G8 pass by month 2019: May=68%, Jun=45%, Jul=14%, Aug=64%, Sep=50%, Oct=44%, **Nov=5.3%**, Dec=20%. Jan 2020=71%, Feb=72% (clear improvement).

### Why 2020-01-01 (not 2019-06-01 or 2019-10-01)

- Q3/Q4 2019 is WORSE than Q2 (Nov=5.3% G8 pass)
- Volume stays at 0.18x through Dec 2019
- Jan 2020 jumps to 71% G8 + volume improvement to 2449 (1.5x of 2019 avg)
- Clean calendar boundary

## Bias check

- Zero strategy PnL consulted. Justification is purely structural (ATR, volume, filter pass rates)
- The recommendation would be the same if every strategy excelled in 2019
- Structural cause is exogenous: contract launch date, not data-mined

## Impact on 6 validated strategies

All have 2019 data (2-156 trades). Net-positive: removes thin/noisy early WF training windows for non-NYSE sessions. NYSE_OPEN loses 156 valid trades but WFE=2.12 has massive margin.

## Separate finding (NOT implemented here)

MES G-filter thresholds (G5=5pts, G8=8pts) are absolute and designed for NQ-scale ORBs. MES CME_PRECLOSE G8 passes only 25.8% of days EVEN IN 2020+. This is a filter-rescaling problem, not fixable by WF_START_OVERRIDE. Tracked as separate task.

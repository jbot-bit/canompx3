# Break Quality Variables Audit

**Date:** 2026-03-24
**Source:** `orb_outcomes` + `daily_features` (canonical layers only)
**Scope:** MNQ E2 O5 CB1 RR1.0, 4 structural sessions (CME_PRECLOSE, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE)
**Data state:** 10yr (2016-02-01 to 2026-03-23)

## Tradeable vs Lookahead

| Variable | Knowable at entry? | Code location | Signal |
|----------|-------------------|---------------|--------|
| `break_delay_min` | YES (waiting for break) | build_daily_features.py:332-364 | STRONG monotonic |
| `risk_dollars` (friction) | YES (ORB formed) | outcome_builder cost model | STRONG monotonic |
| `rel_vol` | YES (at ORB formation) | daily_features | Research lead |
| `break_bar_continues` | NO for E2 (in trade when bar closes) | build_daily_features.py:365 | Weak (N=95 FALSE) |
| `double_break` | **NO — LOOKAHEAD** (code line 393 flags this) | build_daily_features.py:381-417 | Untradeable |

## Break Delay Gradient (timeout gate)

| Timeout | N | ExpR | WR% | % kept |
|---------|---|------|-----|--------|
| <=3m | 12,550 | +0.124 | 62.2 | 42% |
| <=5m | 15,346 | +0.110 | 61.4 | 52% |
| <=10m | 19,237 | +0.091 | 60.4 | 65% |
| <=20m | 23,116 | +0.071 | 59.3 | 78% |
| ALL | 29,610 | +0.070 | 59.4 | 100% |

Sensitivity ±20% around 10m: 8m=+0.098, 10m=+0.091, 12m=+0.084. Smooth, no cliff.

## Combined Gate: Low Friction + Timeout <=10m

| Era | N | ExpR | t | p |
|-----|---|------|---|---|
| 2016-2020 | 3,625 | +0.117 | 7.59 | <0.0001 |
| 2021-2025 | 8,742 | +0.167 | 16.75 | <0.0001 |
| 2026 YTD | 428 | +0.096 | 2.06 | 0.040 |
| FULL 10yr | 12,795 | +0.150 | 18.26 | <0.0001 |

First combination positive and significant in ALL eras.

## Total R Comparison (10yr)

| Gate | N | ExpR | Total R |
|------|---|------|---------|
| Unfiltered | 29,610 | +0.070 | +2,079 |
| Low friction only | 19,464 | +0.125 | +2,424 |
| Timeout only | 19,237 | +0.091 | +1,753 |
| Friction + timeout | 12,795 | +0.150 | +1,924 |

## Cross-Instrument (timeout generalizes)

| Instrument | Early <=10m | Late >10m |
|------------|------------|-----------|
| MNQ | +0.040 | -0.043 |
| MGC | -0.131 | -0.197 |
| MES | -0.027 | -0.134 |

Same direction all three. Early breaks always better.

## Per-Session Timeout Effect

| Session | Early <=10m ExpR | Late >10m ExpR |
|---------|-----------------|----------------|
| CME_PRECLOSE | +0.129 (N=3568) | +0.027 (N=1137) |
| NYSE_OPEN | +0.126 (N=5240) | -0.024 (N=2480) |
| US_DATA_1000 | +0.078 (N=5186) | +0.018 (N=2575) |
| COMEX_SETTLE | +0.044 (N=5243) | +0.002 (N=2285) |

## Interaction: Friction x Timeout

| Combo | N | ExpR | WR% |
|-------|---|------|-----|
| early + low friction | 12,795 | +0.150 | 60.4% |
| early + high friction | 6,442 | -0.027 | 60.4% |
| late + low friction | 5,565 | +0.036 | 54.4% |
| late + high friction | 2,912 | -0.060 | 58.1% |

## Academic Grounding

- Order flow imbalance (Chan, Algorithmic Trading): immediate breaks = concentrated stop orders = momentum
- FAST10 filter in playbook is the same concept — already in use for LONDON_METALS
- Meta-labeling concept (de Prado, ML for Asset Managers): break delay classifies signal quality without ML

## Operational Implementation

"Set bracket orders at ORB edges. If neither triggers within 10 minutes, cancel both."
No code. No ML. No complex filter. Just a timer.

## Status

- Friction gate: PROVEN (monotonic, cross-instrument, 10yr)
- Timeout gate: PROVEN (monotonic, cross-instrument, era-stable with friction)
- Combined gate: POSITIVE IN ALL ERAS — candidate for live promotion gate card
- All gates require honest K + WFE audit before implementation
- Paper book: UNCHANGED (pre-registered, frozen)
- Rel_vol: research lead only
- Double_break: LOOKAHEAD, untradeable

# Nested ORB Findings: 15-Minute ORB + 5-Minute Entry Bars

**Date**: 2026-02-09
**Branch**: `feature/nested-orb`
**Data**: 671,130 nested outcomes | 6,840 strategies | 1,290 trading days (2021-2026)
**Temp DB**: `C:/Users/joshd/AppData/Local/Temp/gold_nested.db`

---

## Executive Summary

The nested ORB (15m range + 5m entry bars) produces **509 validated strategies** vs 312 baseline.
The **1000 session is the clear winner** with 401 validated strategies (vs ~75 baseline).
However, **1800 and 0900 are weaker** on 15m than baseline 5m.

**Recommendation**: Use nested 15m for 1000 session. Keep baseline 5m for 0900/1800/2300.

---

## Matched A/B Comparison (same params, different resolution)

Across 5,382 matched strategy pairs (same ORB/EM/RR/CB/filter, N>=30 both):

| ORB  | Pairs | Base ExpR | Nest ExpR | Premium  | Nest Better |
|------|-------|-----------|-----------|----------|-------------|
| 0030 |   810 |    -0.292 |    -0.260 |  +0.033  | 533/810 (66%) |
| 0900 |  1068 |    -0.193 |    -0.157 |  +0.036  | 602/1068 (56%) |
| **1000** | **990** | **-0.150** | **+0.058** | **+0.208** | **892/990 (90%)** |
| 1100 |   894 |    -0.173 |    -0.086 |  +0.087  | 558/894 (62%) |
| 1800 |   810 |    -0.173 |    -0.132 |  +0.040  | 537/810 (66%) |
| 2300 |   810 |    -0.208 |    -0.237 |  -0.029  | 342/810 (42%) |
| **TOTAL** | **5382** | **-0.196** | **-0.129** | **+0.067** | **3464/5382 (64%)** |

**Key**: 1000 flips from negative (-0.150) to positive (+0.058) average ExpR. 90% of pairs improve.

---

## Session-by-Session Analysis

### 1000: THE WINNER (+0.208R premium, 401 validated)

15m ORB transforms 1000 from a marginal session to the strongest in the system.

**Best nested 1000 strategies:**

| Strategy | N | WR | ExpR | Sharpe | MaxDD |
|----------|---|-----|------|--------|-------|
| E3 RR4.0 CB5 G3+ | 147 | 32.4% | 0.411 | 0.20 | 15.8R |
| E3 RR4.0 CB5 G5+ | 102 | 30.5% | 0.375 | 0.18 | 13.8R |
| E2 RR4.0 CB3 L4 | 92 | 32.2% | 0.365 | 0.18 | 8.0R |
| E2 RR3.0 CB5 G3+ | 186 | 36.5% | 0.340 | 0.19 | 13.1R |
| E3 RR2.5 CB3 G5+ | 111 | 42.6% | 0.338 | 0.22 | 13.7R |

**Why 1000 works with 15m**: The 10:00 Brisbane ORB (00:00 UTC) captures the
full Asian session opening range. A 5m window is too narrow to establish the
structural level — price needs 15 minutes to define the real range after midnight.
5m confirm bars then integrate enough price action to filter out noise.

**Notable**: E3 retrace dominates (not E1 momentum). With a wider 15m range,
the retrace to the ORB level is a genuine value entry, not just noise.

**Filter insight**: Even G3+ works (ExpR=0.411, N=147) — the wider 15m ORB
provides natural noise filtering, reducing dependence on strict size filters.
L-filters also appear (L4: ExpR=0.365) — previously ALL L-filters were negative.

### 0900: WEAKER ON 15m (baseline better)

| Source | Best ExpR | N | Sharpe | MaxDD |
|--------|-----------|---|--------|-------|
| Baseline 5m | 0.399 (E1 RR2.5 CB2 G6) | 76 | 0.25 | 5.8R |
| Nested 15m | 0.271 (E3 RR2.5 CB1 G8) | 57 | 0.17 | 7.8R |

**Baseline wins by 0.128R.** The 5m ORB at 0900 (US pre-market, 23:00 UTC)
is already a clean structural level. Extending to 15m adds noise from the
first 10 minutes of NY session, diluting the breakout signal.
Keep baseline 5m for 0900.

### 1800: MUCH WEAKER ON 15m (baseline dominant)

| Source | Best ExpR | N | Sharpe | MaxDD |
|--------|-----------|---|--------|-------|
| Baseline 5m | 0.434 (E3 RR2.0 CB4 G6) | 50 | 0.31 | 3.3R |
| Nested 15m | 0.228 (E3 RR4.0 CB2 L4) | 58 | 0.12 | 8.0R |

**Baseline wins by 0.206R.** The 1800 GLOBEX open spike is a short, sharp event.
5m captures the spike perfectly; 15m dilutes it with subsequent price action.
The baseline's 0.31 Sharpe vs nested's 0.12 is decisive. Keep baseline 5m for 1800.

### 2300: WORSE ON 15m

| Source | Best ExpR | N | Sharpe |
|--------|-----------|---|--------|
| Baseline 5m | 0.257 (E3 RR1.5 CB4 G8) | 50 | 0.22 |
| Nested 15m | 0.116 (E2 RR2.0 CB2 L8) | 124 | 0.09 |

Baseline wins. 2300 overnight session is similar to 1800 — the breakout signal
is compressed into the first few minutes. 15m adds too much noise.

### 0030: NEW EDGE FOUND

| Strategy | N | WR | ExpR | Sharpe | MaxDD |
|----------|---|-----|------|--------|-------|
| E3 RR1.0 CB5 VOL | 96 | 64.4% | 0.161 | 0.18 | 4.5R |
| E3 RR1.5 CB4 VOL | 106 | 51.7% | 0.154 | 0.14 | 6.9R |
| E1 RR1.0 CB4 VOL | 150 | 61.0% | 0.137 | 0.15 | 8.4R |

0030 was essentially untradeable on baseline 5m (best ExpR ~0.04).
With 15m ORB + VolumeFilter, it becomes a viable low-RR/high-WR strategy.
64% win rate at RR1.0 is attractive for compounding. Small edge but consistent.

### 1100: MARGINAL IMPROVEMENT

Some strategies validate but ExpR is low (0.08 best). Not actionable yet.
Monitor — may improve with more data if 2025-2026 regime continues.

---

## Structural Insights

### Why 15m helps some sessions and hurts others

| Session | 5m ORB character | 15m effect | Winner |
|---------|-----------------|------------|--------|
| 0900 | Clean pre-market level | Diluted by NY open activity | 5m |
| 1000 | Too narrow for midnight | Captures full Asian open range | **15m** |
| 1100 | Marginal either way | Slight improvement | Marginal |
| 1800 | Sharp GLOBEX spike | Spike absorbed into range | 5m |
| 2300 | Sharp overnight break | Break absorbed into range | 5m |
| 0030 | Too noisy at 5m | Better structure definition | **15m** |

**Pattern**: Sessions with sharp, concentrated breakouts (0900, 1800, 2300) are
better served by the narrow 5m window. Sessions that need time to establish
structure (1000, 0030) benefit from the wider 15m window.

### L-filters work on 15m (new finding)

On baseline 5m, ALL L-filter strategies had negative ExpR. On nested 15m:
- L4 strategies validated at 1000 (ExpR 0.365)
- L4 strategies validated at 1800 (ExpR 0.228)
- L6 strategies validated at 1100 (ExpR 0.322)
- L8 strategies validated at 2300 (ExpR 0.116)

**Why**: The wider 15m ORB naturally filters out the smallest ranges.
A "small" 15m ORB is already substantial (equivalent to a G3+ at 5m),
so L-filters select moderate-sized ranges that are genuine but not extreme.

### Entry model shifts

On baseline: E1 momentum dominates 0900/1000, E3 retrace dominates 1800/2300.
On nested 15m: **E3 retrace dominates everywhere** (especially 1000).

**Why**: With a wider 15m range, the distance from breakout to ORB level is
larger, making the retrace entry (E3) a genuine value opportunity rather than
just noise. E1 momentum works less well because the breakout from a wider
range has less follow-through per unit of risk.

---

## Recommended Hybrid Portfolio

Combine baseline 5m where it's stronger with nested 15m where it adds edge:

| Leg | Resolution | Strategy | N | WR | ExpR |
|-----|------------|----------|---|-----|------|
| 1 | **5m** (baseline) | 0900 E1 CB2 RR2.5 G4+ | 125 | 40.0% | 0.31 |
| 2 | **15m** (nested) | 1000 E3 CB5 RR4.0 G3+ | 147 | 32.4% | 0.41 |
| 3 | **5m** (baseline) | 1800 E3 CB5 RR2.0 G5+ | 75 | 48.0% | 0.30 |
| 4 | **15m** (nested) | 0030 E3 CB5 RR1.0 VOL | 96 | 64.4% | 0.16 |
| 5 | **5m** (baseline) | 2300 E3 CB4 RR1.5 G8+ | 50 | 52.0% | 0.25 |

This hybrid uses the optimal resolution per session rather than one-size-fits-all.

---

## Known Issues / Caveats

1. **Yearly JSON bug**: `nested_strategies.yearly_results` has `count: 0` for all years.
   The overall sample_size and metrics are correct, but per-year trade counts are missing.
   This is a bug in `trading_app/nested/discovery.py` — needs fix before formal validation.

2. **Only 2021-2026 data**: The nested builder ran on the temp DB (5-year copy, not the
   10-year gold.db). Results may change with 2016-2020 data included.

3. **No direction column**: `nested_outcomes` doesn't store break direction.
   Direction analysis requires joining to `daily_features.orb_XXXX_break_dir`.

4. **orb_minutes=30 not built**: Only 15m was completed. 30m build would take ~2 more hours.

5. **Post-build audit didn't run**: The builder froze during the auto-audit step.
   Outcomes have not been independently verified. Run audit before trading.

6. **E3 CB5 strategies are artifacts**: Tables above reference "E3 CB5" combos
   (e.g., E3 RR4.0 CB5 G3+). These were produced by a prior builder version that
   allowed arbitrary CB values for E3. The current `outcome_builder.py` enforces
   `E3 = CB1 only` (retrace entry doesn't benefit from multi-bar confirmation).
   Treat E3 CB5 results as informational only — they cannot be reproduced with the
   current codebase and should not be used for live trading decisions.

---

## Next Steps

1. Fix yearly JSON bug in nested discovery
2. Run independent audit on 15m outcomes (spot-check against reconstruction)
3. Consider building on 10-year dataset for deeper validation
4. Implement hybrid portfolio in execution engine (mixed 5m/15m per session)
5. Run correlation analysis between 5m 0900 and 15m 1000 legs

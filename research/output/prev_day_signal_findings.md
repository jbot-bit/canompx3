# Prior-Day Outcome Signal — Research Findings
**Date:** Feb 2026
**Script:** `research/research_prev_day_signal.py`
**Status:** PROMISING HYPOTHESIS (REGIME-grade, BH-survives at G4+)

---

## Setup

**Hypothesis:** Yesterday's same-session ORB outcome (WIN/LOSS/SCRATCH) — known before today's session opens — predicts today's pnl_r.

**Data:** orb_outcomes joined with daily_features via daily LAG on `orb_{session}_outcome`.
`prev_outcome` = yesterday's outcome for the same session. NULL = no break yesterday (those rows excluded).
**Filter:** Current day must pass G4+ (orb_size >= 4.0 pts) — same filter as validated strategies.
**Entry config:** E1 RR matching best validated config per session (see SESSION_CONFIGS in script).
**BH correction:** q=0.10 across all 18 qualifying tests.

---

## Single Survivor — MGC 0900 prev=WIN

| Metric | Value |
|--------|-------|
| N (prev=win bucket) | 72 |
| avgR | +0.472R |
| Baseline MGC 0900 | +0.283R |
| Incremental lift | +0.189R |
| t-stat | +2.88 |
| p (raw) | 0.005 |
| p_bh (BH-adjusted) | 0.093 |
| BH survival at q=0.10 | **YES** |
| Grade | REGIME (N=72) |

### Year-by-year (qualifying years, N >= 5)

| Year | N | avgR | p |
|------|---|------|---|
| 2020 | 9 | +0.787 | 0.161 |
| 2025 | 36 | +0.333 | 0.149 |
| 2026 | 12 | +0.624 | 0.176 |

3/3 qualifying years positive (100%). Per-year N is small — individual years not significant.

### G6+ robustness check

| Filter | N | avgR | p_raw | BH survives |
|--------|---|------|-------|-------------|
| G4+ | 72 | +0.472 | 0.005 | YES (p_bh=0.093) |
| G6+ | 40 | +0.499 | 0.030 | NO (p_bh=0.42, N too small) |

Direction preserved at G6+ (+0.499R). BH failure at G6+ is sample-size limited, not signal reversal.

### Mechanism (hypothesis)

Gold at the CME evening open (0900 Brisbane = 5pm CME) is a momentum-driven session. A G4+ ORB breakout that HIT the target yesterday signals sustained directional conviction in gold — the same institutional players tend to maintain direction across consecutive sessions. This is consistent with gold's documented trending behavior.

---

## Near-misses (did NOT survive BH)

| Cell | N | avgR | p_raw | p_bh | Direction |
|------|---|------|-------|------|-----------|
| MES 0900 prev=win | 211 | -0.183 | 0.016 | 0.093 | REVERSION (worse after win) |
| MNQ 0900 prev=win | 213 | -0.181 | 0.022 | 0.093 | REVERSION (worse after win) |

**Note:** MES/MNQ 0900 baselines are already negative (-0.077R, -0.063R). The reversion signal (worse after win) exists directionally but:
1. Doesn't survive BH
2. Those sessions aren't in the live portfolio anyway
3. Action: none

---

## Full results (G4+)

| Cell | N | avgR | t | p_raw |
|------|---|------|---|-------|
| MGC_0900_prev=win | 72 | +0.472 | +2.88 | **0.005** |
| MES_0900_prev=win | 211 | -0.183 | -2.44 | 0.016 |
| MNQ_0900_prev=win | 213 | -0.181 | -2.31 | 0.022 |
| MNQ_1100_prev=win | 264 | +0.136 | +1.62 | 0.106 |
| MGC_0900_prev=loss | 52 | +0.309 | +1.64 | 0.108 |
| MNQ_1000_prev=win | 205 | +0.175 | +1.59 | 0.114 |
| MGC_1000_prev=loss | 36 | +0.376 | +1.54 | 0.132 |
| All others | — | — | — | >0.25 |

---

## Verdict

**SURVIVED:** MGC 0900 prev=WIN
**Status:** PROMISING HYPOTHESIS
**Action:** WATCH. Do NOT add as mandatory filter.

**Caveats:**
- REGIME-grade only (N=72). Needs N>150 to qualify as CORE.
- Year-by-year N is tiny (max 36 in a single year). Insufficient for per-year significance.
- 2025 dominates sample (N=36 = 50% of total). 2026 developing.
- ±20% parameter test (G6+): direction stable but BH fails due to N.

**Re-evaluate when:** 2026 data accumulates (expect ~15-20 more qualifying days) and total N approaches 90+.

**What it would mean if confirmed:**
Optional overlay for MGC 0900 strategies: weight trades higher or take best opportunity only when prior session was a WIN. Could add ~+0.19R per trade on ~48% of eligible trade days. Not large enough to merit a dedicated filter, but worth tracking.

---

## What did NOT work

- All other session × instrument × prev_outcome combos: NO survivors after BH
- MES 1000, MGC 1000, MNQ 1000, MNQ 1800, MGC 1800, MNQ 1100 prev=loss: all noise (p > 0.13)
- "Scratch" buckets: uniformly N < 30 (most sessions don't produce scratch outcomes with G4+ filter)
- The prior-day outcome is NOT a reliable universal entry signal

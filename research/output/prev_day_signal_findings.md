# Prior-Day Outcome Signal — Research Findings
**Date:** Feb 2026
**Script:** `research/research_prev_day_signal.py`
**Stress test:** Full RR × CB × EM × G grid (105 cells, BH FDR q=0.10)

---

## Setup

**Hypothesis:** Yesterday's same-session ORB outcome (WIN/LOSS/SCRATCH) — known before today's session opens — predicts today's pnl_r.

**Data:** `daily_features` LAG on `orb_0900_outcome` → prev_outcome is TRUE prior knowledge (includes null = no break yesterday).
**Filter:** Current day passes G4+ (orb_size >= 4.0 pts). Current day has a break.
**BH correction:** q=0.10 across all tests in each run.

---

## Phase 1: Initial 18-test pass (E1 only, per-session representative config)

One survivor: **MGC 0900 E1 RR2.5 CB2 G4+ prev=WIN → +0.472R** (t=2.88, p=0.005, p_bh=0.093)
Direction held at G6+ (+0.499R) but BH failed (N dropped to 40). Grade: REGIME.

---

## Phase 2: Full stress grid (105 cells, RR × CB × EM × G)

### BH SURVIVOR: E0 CB1 G4+ MGC 0900 prev=LOSS

| Metric | Value |
|--------|-------|
| N | 49 |
| avgR | **+0.585R** |
| t-stat | +3.34 |
| p (raw) | 0.0016 |
| p_bh | **0.0877** |
| BH survival at q=0.10 | **YES** |
| Grade | REGIME (N=49) |

### Cross-RR robustness (all E0 CB1 G4+ prev=LOSS)

| RR | N | avgR | t | p_raw | 3/3 yrs pos? |
|----|---|------|---|-------|-------------|
| 1.5 | 49 | +0.346 | +2.41 | 0.020 | YES |
| **2.0** | **49** | **+0.585** | **+3.34** | **0.0016** | **YES** |
| 2.5 | 49 | +0.502 | +2.40 | 0.021 | YES |
| 3.0 | 48 | +0.499 | +2.09 | 0.042 | YES |
| 4.0 | 47 | +0.297 | +1.07 | 0.288 | 2/3 |

**Sweet spot: RR1.5–3.0 all positive and consistent. RR4.0 weakens (too ambitious for a failed-break pattern).**

### Cross-entry-model at CB1 prev=LOSS (RR2.0 and RR2.5)

| EM | RR | N | avgR | t | p |
|----|-----|---|------|---|---|
| E0 | 2.0 | 49 | +0.585 | +3.34 | 0.0016 |
| E3 | 2.5 | 46 | +0.558 | +2.62 | 0.012 |
| E3 | 3.0 | 45 | +0.547 | +2.24 | 0.030 |
| E3 | 2.0 | 46 | +0.419 | +2.33 | 0.024 |
| E0 | 2.5 | 49 | +0.502 | +2.40 | 0.021 |
| E1 | 2.0 | 52 | +0.314 | +1.86 | 0.068 |
| E1 | 2.5 | 52 | +0.254 | +1.31 | 0.196 |

**E0 and E3 (precise-price entries, limit at ORB edge or retrace) both positive at CB1 prev=LOSS. E1 (next-bar open = gap-in) is substantially weaker. Signal is about fill quality, not just direction.**

### Year-by-year (E0 RR2.0 CB1 G4+ prev=LOSS)

| Year | N | avgR | p |
|------|---|------|---|
| 2020 | 6 | +0.173 | 0.729 |
| 2025 | 24 | +0.734 | **0.008** |
| 2026 | 9 | +0.678 | 0.194 |

3/3 qualifying years positive. **Warning:** Most calendar years have N<5 (excluded). 2025 dominates sample (49% of N). Per-year significance: only 2025 individually significant.

### Baseline decomposition (E0 CB1 G4+ MGC 0900)

| Condition | N | avgR |
|-----------|---|------|
| Baseline (all prev) | 148 | +0.273 |
| prev=win | 72 | +0.259 (≈ baseline) |
| **prev=loss** | **49** | **+0.585 (>2× baseline)** |
| prev=no_break | 26 | **-0.225 (BELOW baseline)** |

**The separation is striking:** prev=loss far outperforms baseline. prev=no_break underperforms. prev=win is flat vs baseline.

---

## Mechanism (hypothesis)

**E0 CB1** = limit fill exactly at the ORB edge on the break bar (CB1 always fills — break bar crosses edge, immediate limit hit). This is the fastest, most aggressive entry at the exact level.

**prev=LOSS** = yesterday's aggressive entry at the ORB edge failed to reach the target (e.g., broke G4+, filled immediately at ORB edge, then stopped out).

**Combined signal:** Yesterday's fast break FAILED. Today a new fast break occurs. The interpretation:
- Market tried to break a direction yesterday → rejected before target
- Today the market is retrying with accumulated pressure from the failed attempt
- "Failed breakout → fresh breakout" = second-attempt breakouts tend to be stronger in trending markets (gold at the CME evening open is a momentum session)

The E0/E3 > E1 asymmetry reinforces this: the signal works best when you get a PRECISE price at the ORB edge, not when you chase at the next bar open. The edge is about conviction at the exact level, not just directional momentum.

**Side finding — prev=no_break:**
When there was NO break at MGC 0900 yesterday, today's break underperforms baseline (N=26, -0.225R to -0.380R). Hypothesis: no-break days represent low-conviction sessions; consecutive low-conviction → today's break is also weak. NOT BH-significant (N=26). Track as observation.

---

## What the original E1 CB2 prev=WIN result means

In the 18-test pass (Phase 1), E1 CB2 prev=WIN survived BH at m=18 (p_bh=0.093).
In the 105-test pass (Phase 2), it is rank 2 but does NOT survive BH at the stricter m=105 threshold.

Both findings point to the same underlying session characteristic: MGC 0900 has predictable pattern continuation from prior session. The prev=LOSS signal is STRONGER (lower p) and uses a more natural mechanism (failed-break momentum). The prev=WIN signal may be a weaker secondary effect.

---

## Verdict

| Finding | Grade | BH (Phase 2) | Action |
|---------|-------|--------------|--------|
| E0 CB1 MGC 0900 prev=LOSS → +0.585R | REGIME (N=49) | SURVIVES (p_bh=0.088) | **WATCH** |
| E0/E3 CB1 prev=LOSS cross-EM consistency | — | Not separately tested | Supports mechanism |
| E1 CB2 MGC 0900 prev=WIN → +0.472R | REGIME (N=72) | Does not survive at m=105 | **WATCH** |
| prev=no_break → -0.225R | — | N<30, not tested | Observation only |

**Label: PROMISING HYPOTHESIS**

**Caveats:**
- Both signals are REGIME-grade (N < 100). Not CORE.
- Most calendar years have N < 5 per year (excluded from year-by-year). 2025 dominates.
- 2025 alone: N=24 trades. If 2025 is anomalous, the whole signal weakens.
- ±20% RR test: PASSES (RR1.5–3.0 all consistent). Filter test: E0 vs E3 both work, E1 weaker.
- Not actionable as a mandatory filter. Too few data points to gate live trades.

**Re-evaluate when:** Total N (prev=LOSS, E0 CB1 G4+) exceeds 80 (expect ~20-25 more in next 12 months).

---

## What did NOT work

- MES/MNQ 0900 prev=WIN: reversion (both sessions are losers anyway)
- All other session × instrument × prev_outcome combos: noise (p > 0.10 after BH)
- Scratch buckets: N < 30 uniformly
- E1 CB2 prev=WIN at m=105: doesn't survive stricter BH threshold

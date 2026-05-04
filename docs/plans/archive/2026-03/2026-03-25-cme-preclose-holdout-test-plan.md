---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# CME_PRECLOSE 2026 Holdout Test Plan

**Date:** 2026-03-25
**Status:** WAITING (N=53/100)
**Pre-registration:** `docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md`

## 1. Strategy Under Test

**MNQ CME_PRECLOSE E2 CB1 RR1.0 O5 NO_FILTER**

- Failed BH FDR at N=55 on 2025 data (p=0.007 vs threshold 0.003)
- Pre-registered for SINGLE-TEST on 2026 holdout (p < 0.05)
- Parameters FROZEN as of 2026-03-20

## 2. Readiness Gate

| Criterion | Required | Status |
|-----------|----------|--------|
| N (trades) | >= 100 | 53 as of 2026-03-23 |
| Parameters frozen | Yes | Frozen 2026-03-20 |
| Pre-registration committed | Yes | Git SHA exists |
| Decision rules pre-defined | Yes | See §5 |

**Estimated readiness:** Late May / early June 2026 (~17 trades/month)

## 3. Pre-Flight (Run Day-Of)

1. Confirm pre-registration SHA unchanged: `git log docs/pre-registrations/`
2. Verify DB freshness: `MAX(trading_day)` in orb_outcomes for MNQ CME_PRECLOSE
3. Count N: must be >= 100
4. Verify cost model: `COST_SPECS['MNQ']` = $2.74 RT
5. Verify no parameter changes since 2026-03-20

## 4. Test Protocol

**Hypothesis:**
- H0: ExpR <= 0 (no edge)
- H1: ExpR > 0 (positive edge)

**Test:** One-sided t-test on `pnl_r` values

**Query:**
```sql
SELECT pnl_r FROM orb_outcomes
WHERE symbol = 'MNQ'
  AND orb_label = 'CME_PRECLOSE'
  AND orb_minutes = 5
  AND rr_target = 1.0
  AND trading_day >= '2026-01-01'
ORDER BY trading_day;
```

**Computation:**
```python
from scipy import stats
t_stat, two_sided_p = stats.ttest_1samp(pnl_r_values, 0)
one_sided_p = two_sided_p / 2 if t_stat > 0 else 1 - two_sided_p / 2
```

**Threshold:** p < 0.05

**BH FDR context:** If running all 3 pre-registered strategies simultaneously,
CME_PRECLOSE gets rank 3/3 -> threshold = 0.05 x (3/3) = 0.05 (same as raw).

## 5. Decision Rules (PRE-DEFINED)

| Outcome | Decision | Action |
|---------|----------|--------|
| p < 0.05 AND ExpR > +0.03R | VALIDATED | Promote to LIVE_PORTFOLIO, add Apex lane |
| p < 0.05 AND ExpR <= +0.03R | MARGINAL | Keep paper trading, re-test at N=200 |
| p >= 0.05 | DEAD | Kill CME_PRECLOSE NO_FILTER baseline |
| N < 100 at test time | ABORT | Wait for N >= 100 |

**Kill criteria (active during paper trading):**
- ExpR < +0.03R after 100 trades -> STOP session
- Slippage > 3 ticks avg after 100 trades -> STOP (cost model wrong)
- Combined portfolio ExpR < +0.05R after 200 total trades -> STOP everything

## 6. Supporting Metrics (report, don't gate on)

- Win rate
- Monthly breakdown
- Max drawdown (R)
- WFE: OOS_ExpR / IS_ExpR (baseline = +0.213 pre-2025)
- Cohen's d effect size

## 7. Honesty Disclosure

**Contamination log:**
- 2026-03-25: N and raw stats (ExpR=+0.031, WR=52.8%, N=53) observed during
  test plan creation. Parameters frozen, decision rules pre-defined. Impact:
  psychological only.

## 8. Timeline

| Date | Action |
|------|--------|
| 2026-03-25 | Test plan committed |
| Weekly | Monitor N count ONLY (no performance peeking) |
| ~June 2026 | N hits 100 -> execute test |
| Test day | Run protocol exactly, commit results |
| Post-test | Implement decision per §5 |

## 9. What This Settles

- **VALIDATED:** CME_PRECLOSE becomes 3rd confirmed MNQ session. 5th Apex lane.
- **DEAD:** NO_FILTER baseline killed. Filtered variants (ATR70_VOL, VOL_RV12_N20)
  remain independent — not killed by this test.

## CHECK WEEKLY: CME_PRECLOSE N count

Run test when N >= 100 (est. June 2026). Do NOT peek at performance before then.

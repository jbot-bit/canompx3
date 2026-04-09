# Criterion 9 Era Bin Respects WF_START_OVERRIDE — Design

**Date:** 2026-04-09
**Status:** DESIGN — awaiting approval

## Problem

MNQ CME_PRECLOSE E2 RR1.0 G8 (Sharpe 1.83, p=0.000002, N=1320) is rejected by criterion_9 because the "2015-2019" era bin contains 56 trades from 2019 with ExpR=-0.1954.

The WF engine already excludes 2019 via WF_START_OVERRIDE=2020-01-01, based on a structural data audit proving 2019 micro-launch data is non-representative (ATR 0.42x, ORB size 0.50x, volume 0.42x, G8 pass rate 39% vs 97.7%). But criterion_9 reads yearly_results which includes 2019.

## Decision

Criterion_9 should respect WF_START_OVERRIDE. Years before the override year are excluded from era bin aggregation. This makes the era stability check consistent with the walk-forward engine's structural exclusion.

## Structural evidence (execution output, not metadata)

MNQ CME_PRECLOSE 2019 G8-filtered trades vs 2020:
- ORB size: 13.3 vs 26.5 = 0.50x
- Volume: 2,361 vs 5,646 = 0.42x
- ATR: 124.0 vs 249.1 = 0.50x
- These 56 trades are from an 8-month-old contract with immature liquidity

Year-by-year performance (from experimental_strategies):
- 2019: N=56, ExpR=-0.1954 (THIN DATA — kills the strategy)
- 2020: N=194, ExpR=+0.2117
- 2021: N=206, ExpR=+0.1738
- 2022: N=213, ExpR=+0.2621
- 2023: N=204, ExpR=+0.0400
- 2024: N=220, ExpR=+0.0643
- 2025: N=227, ExpR=+0.0535

Without 2019: 6 consecutive positive years, N=1264, every era passes criterion_9.

## Bias check

- The exclusion is NOT based on strategy performance. WF_START_OVERRIDE was set to 2020-01-01 BEFORE examining MNQ CME_PRECLOSE's per-year PnL (the audit used ATR, volume, filter pass rates).
- 2019's negative performance for this strategy is consistent with structural explanation (thin liquidity, half-sized ORBs) — it is not a "dead regime."
- Honest caveat: we cannot prove the counterfactual "what would 2019 look like with mature MNQ liquidity." The exclusion rests on structural judgment, not a statistical test.

## Amendment 3.1 text

"Era bins containing only years that fall entirely before an instrument's WF_START_OVERRIDE date are excluded from Criterion 9 evaluation. Rationale: WF_START_OVERRIDE documents a structurally-motivated data exclusion (contract launch artifacts, regime incompatibility) justified by independent structural metrics (ATR, volume, ORB filter pass rates). The same structural judgment that makes pre-override data unreliable for walk-forward training also makes it unreliable for era stability assessment."

## Implementation

1. `_check_criterion_9_era_stability` gains `wf_start_year: int | None` parameter
2. Years before wf_start_year are skipped in the era aggregation loop
3. Pre-flight dispatcher reads WF_START_OVERRIDE for the instrument and passes the year
4. 4 tests: override-excludes-2019, no-override-still-fails, override-year-included, MGC-override
5. Amendment 3.1 to pre_registered_criteria.md

## Deferred (Take 3 from design)

Fix discovery to also exclude pre-WF_START_OVERRIDE data from yearly_results JSON. This is the root-cause fix that makes ALL downstream consumers consistent. Larger blast radius — requires re-running discovery + re-validation. Tracked separately.

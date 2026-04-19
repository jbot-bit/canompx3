# Consolidated retirement-verdict view — 38 active validated_setups

**Generated:** 2026-04-19T09:42:03+00:00
**Script:** `research/consolidated_retirement_verdict.py`
**Input audit streams:**
  - Mode-A criterion evaluation (C4/C7/C9) via `research.mode_a_revalidation_active_setups`
  - Regime-drift retirement queue (`docs/audit/results/2026-04-19-mnq-retirement-queue-committee-action.md`)
  - Fire-rate audit (`docs/audit/results/2026-04-19-fire-rate-audit.md`)
  - SGP O15/O30 Jaccard (`docs/audit/results/2026-04-19-sgp-o15-o30-jaccard.md`)

**Canonical truth:** `orb_outcomes`, `daily_features`, `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`. Filters via `research.filter_utils.filter_signal`. Read-only; no DB writes; no `validated_setups` mutation.

## Summary

- **RETIRE_URGENT:** 1/38
- **RETIRE_STANDARD:** 3/38
- **N_UNDERPOWERED:** 1/38
- **RECLASSIFY_COST:** 10/38
- **REVIEW_TIER2:** 7/38
- **REVIEW_CAPACITY:** 1/38
- **REVIEW_C4_WT_FAIL:** 10/38
- **BETTER_THAN_PEERS:** 3/38
- **KEEP:** 2/38

## Verdict rubric (deterministic, doctrine-cited)

| Verdict | Trigger | Action |
|---|---|---|
| `RETIRE_URGENT` | Retirement-queue Tier-1 with NEGATIVE late Sharpe | Vote this week. Actively losing money. |
| `RETIRE_STANDARD` | Retirement-queue Tier-1 (excess decay > 0.60 vs portfolio) | Vote this week. Decay exceeds environment. |
| `N_UNDERPOWERED` | C7 FAIL: Mode A N < 100 (`pre_registered_criteria.md § 7`) | Retire under doctrine — insufficient N for deployment. |
| `RECLASSIFY_COST` | Rule 8.1 fire-rate ≥ 95% OR Rule 8.2 arithmetic_only (Amendment v3.2 DRAFT C13/C14) | Route to cost-screen registry, not filter-edge registry. |
| `REVIEW_TIER2` | Retirement-queue Tier-2 (excess decay 0.10-0.60) | Vote within 2 weeks. |
| `REVIEW_CAPACITY` | SGP O15/O30 pair (Jaccard 0.65) | Capacity-split decision required before parallel deploy. |
| `REVIEW_C4_WT_FAIL` | C4 with-theory FAIL (t < 3.0) but no other red flags | Either promote C4 grounding to DIRECT Tier 1 for this lane's theory citation, OR re-validate on larger sample, OR downgrade to C4 no-theory 3.79 threshold (which this lane then also fails). |
| `BETTER_THAN_PEERS` | Sharpe rose early→late under portfolio stress | Keep; potential scaling candidate (separate pre-reg required). |
| `KEEP` | Passes all evaluated gates, no retirement queue entry | Retain. Note: C4 grounding caveat applies — with-theory 3.00 threshold is INDIRECT Tier 1 per doctrine. |

## Per-lane verdict table

Primary verdict is the first row per lane. Additional reasons (C9 era fails, Mode-B grandfathered, cross-asset injection artifact, secondary C4 notes) listed as supplementary evidence.

| # | Strategy ID | Verdict | Mode-A N | t_IS | C7 | C9 | Primary reason |
|---|---|---|---:|---:|---|---|---|
| 1 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | **RETIRE_URGENT** | 800 | +1.34 | PASS | PASS | RETIREMENT Tier-1 with NEGATIVE late Sharpe — actively losing |
| 2 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | **RETIRE_STANDARD** | 521 | +2.07 | PASS | PASS | RETIREMENT Tier-1: excess decay > 0.60 vs portfolio |
| 3 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` | **RETIRE_STANDARD** | 535 | +1.70 | PASS | PASS | RETIREMENT Tier-1: excess decay > 0.60 vs portfolio |
| 4 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | **RETIRE_STANDARD** | 535 | +1.98 | PASS | PASS | RETIREMENT Tier-1: excess decay > 0.60 vs portfolio |
| 5 | `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | **N_UNDERPOWERED** | 88 | +3.56 | FAIL | PASS | C7 FAIL: Mode A N=88 < 100 |
| 6 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12` | **RECLASSIFY_COST** | 669 | +2.94 | PASS | PASS | Rule 8.2 arithmetic_only (wr_spread < 3% with material ExpR_delta) |
| 7 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | **RECLASSIFY_COST** | 836 | +2.50 | PASS | PASS | Rule 8.1 fire-rate 95.1% (>95%) |
| 8 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | **RECLASSIFY_COST** | 829 | +2.32 | PASS | FAIL | Rule 8.1 fire-rate 95.1% (>95%) |
| 9 | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | **RECLASSIFY_COST** | 814 | +2.24 | PASS | FAIL | Rule 8.1 fire-rate 95.1% (>95%) |
| 10 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | **RECLASSIFY_COST** | 521 | +1.34 | PASS | PASS | Rule 8.2 arithmetic_only (wr_spread < 3% with material ExpR_delta) |
| 11 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | **RECLASSIFY_COST** | 844 | +2.10 | PASS | PASS | Rule 8.1 fire-rate 98.6% (>95%) |
| 12 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | **RECLASSIFY_COST** | 856 | +2.03 | PASS | PASS | Rule 8.1 fire-rate 99.7% (>95%) |
| 13 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` | **RECLASSIFY_COST** | 817 | +2.11 | PASS | FAIL | Rule 8.1 fire-rate 98.6% (>95%) |
| 14 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | **RECLASSIFY_COST** | 829 | +2.06 | PASS | FAIL | Rule 8.1 fire-rate 99.7% (>95%) |
| 15 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | **RECLASSIFY_COST** | 496 | +3.96 | PASS | PASS | Rule 8.2 arithmetic_only (wr_spread < 3% with material ExpR_delta) |
| 16 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` | **REVIEW_TIER2** | 306 | +4.19 | PASS | PASS | RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio |
| 17 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | **REVIEW_TIER2** | 535 | +1.34 | PASS | FAIL | RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio |
| 18 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | **REVIEW_TIER2** | 773 | +1.10 | PASS | FAIL | RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio |
| 19 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5` | **REVIEW_TIER2** | 773 | +1.54 | PASS | PASS | RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio |
| 20 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | **REVIEW_TIER2** | 469 | +1.87 | PASS | PASS | RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio |
| 21 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | **REVIEW_TIER2** | 460 | +2.99 | PASS | PASS | RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio |
| 22 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | **REVIEW_TIER2** | 436 | +3.20 | PASS | PASS | RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio |
| 23 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | **REVIEW_CAPACITY** | 485 | +4.14 | PASS | PASS | SGP O15/O30 pair: Jaccard 0.65 — capacity review before parallel deploy |
| 24 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | **REVIEW_C4_WT_FAIL** | 263 | +0.99 | PASS | PASS | C4 with-theory FAIL: t_IS=0.99 < 3.0 |
| 25 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | **REVIEW_C4_WT_FAIL** | 773 | +1.90 | PASS | PASS | C4 with-theory FAIL: t_IS=1.90 < 3.0 |
| 26 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | **REVIEW_C4_WT_FAIL** | 263 | +1.65 | PASS | PASS | C4 with-theory FAIL: t_IS=1.65 < 3.0 |
| 27 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | **REVIEW_C4_WT_FAIL** | 345 | +1.51 | PASS | PASS | C4 with-theory FAIL: t_IS=1.51 < 3.0 |
| 28 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | **REVIEW_C4_WT_FAIL** | 334 | +1.00 | PASS | PASS | C4 with-theory FAIL: t_IS=1.00 < 3.0 |
| 29 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | **REVIEW_C4_WT_FAIL** | 469 | +1.95 | PASS | PASS | C4 with-theory FAIL: t_IS=1.95 < 3.0 |
| 30 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | **REVIEW_C4_WT_FAIL** | 786 | +2.50 | PASS | FAIL | C4 with-theory FAIL: t_IS=2.50 < 3.0 |
| 31 | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5` | **REVIEW_C4_WT_FAIL** | 785 | +2.62 | PASS | PASS | C4 with-theory FAIL: t_IS=2.62 < 3.0 |
| 32 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60` | **REVIEW_C4_WT_FAIL** | 371 | +1.56 | PASS | PASS | C4 with-theory FAIL: t_IS=1.56 < 3.0 |
| 33 | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | **REVIEW_C4_WT_FAIL** | 398 | +2.09 | PASS | PASS | C4 with-theory FAIL: t_IS=2.09 < 3.0 |
| 34 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | **BETTER_THAN_PEERS** | 283 | +3.42 | PASS | PASS | Regime-stress BETTER-THAN-PEERS: Sharpe rose early→late |
| 35 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | **BETTER_THAN_PEERS** | 278 | +2.65 | PASS | PASS | Regime-stress BETTER-THAN-PEERS: Sharpe rose early→late |
| 36 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | **BETTER_THAN_PEERS** | 379 | +3.32 | PASS | PASS | Regime-stress BETTER-THAN-PEERS: Sharpe rose early→late |
| 37 | `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | **KEEP** | 130 | +3.66 | PASS | PASS | — |
| 38 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | **KEEP** | 385 | +4.29 | PASS | PASS | Fire-rate audit reported 0% — artifact of missing CrossAssetATR injection in that audit's scope, NOT a lane defect. Criterion eval applies canonical injection and the lane has real data. |

## RETIRE_URGENT — detail

### `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

- Mode-A: N=800 ExpR=+0.0565 t_IS=+1.34 Sh_ann=+0.51
- Years positive: 6/7
- Evidence:
  - RETIREMENT Tier-1 with NEGATIVE late Sharpe — actively losing
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-06 >= 2026-01-01; stored values are not Mode-A-clean.
  - (secondary) C4 with-theory t_IS=1.34 < 3.0

## RETIRE_STANDARD — detail

### `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12`

- Mode-A: N=521 ExpR=+0.1049 t_IS=+2.07 Sh_ann=+0.78
- Years positive: 7/7
- Evidence:
  - RETIREMENT Tier-1: excess decay > 0.60 vs portfolio
  - (secondary) C4 with-theory t_IS=2.07 < 3.0

### `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM`

- Mode-A: N=535 ExpR=+0.0812 t_IS=+1.70 Sh_ann=+0.64
- Years positive: 5/7
- Evidence:
  - RETIREMENT Tier-1: excess decay > 0.60 vs portfolio
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-14 >= 2026-01-01; stored values are not Mode-A-clean.
  - (secondary) C4 with-theory t_IS=1.70 < 3.0

### `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM`

- Mode-A: N=535 ExpR=+0.1122 t_IS=+1.98 Sh_ann=+0.75
- Years positive: 6/7
- Evidence:
  - RETIREMENT Tier-1: excess decay > 0.60 vs portfolio
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-14 >= 2026-01-01; stored values are not Mode-A-clean.
  - (secondary) C4 with-theory t_IS=1.98 < 3.0

## N_UNDERPOWERED — detail

### `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08`

- Mode-A: N=88 ExpR=+0.3279 t_IS=+3.56 Sh_ann=+1.45
- Years positive: 3/3
- Evidence:
  - C7 FAIL: Mode A N=88 < 100

## RECLASSIFY_COST — detail

### `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12`

- Mode-A: N=669 ExpR=+0.1041 t_IS=+2.94 Sh_ann=+1.11
- Years positive: 6/7
- Evidence:
  - Rule 8.2 arithmetic_only (wr_spread < 3% with material ExpR_delta)
  - (secondary) C4 with-theory t_IS=2.94 < 3.0

### `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5`

- Mode-A: N=836 ExpR=+0.0780 t_IS=+2.50 Sh_ann=+0.95
- Years positive: 5/7
- Evidence:
  - Rule 8.1 fire-rate 95.1% (>95%)
  - (secondary) C4 with-theory t_IS=2.50 < 3.0

### `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`

- Mode-A: N=829 ExpR=+0.0922 t_IS=+2.32 Sh_ann=+0.88
- Years positive: 5/7
- Violating eras: 2015-2019:-0.136(N=57)
- Evidence:
  - Rule 8.1 fire-rate 95.1% (>95%)
  - C9 era stability FAIL: doctrine era(s) ['2015-2019'] ExpR<-0.05 (era-N>=50)
  - (secondary) C4 with-theory t_IS=2.32 < 3.0

### `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5`

- Mode-A: N=814 ExpR=+0.1059 t_IS=+2.24 Sh_ann=+0.85
- Years positive: 5/7
- Violating eras: 2015-2019:-0.267(N=52)
- Evidence:
  - Rule 8.1 fire-rate 95.1% (>95%)
  - C9 era stability FAIL: doctrine era(s) ['2015-2019'] ExpR<-0.05 (era-N>=50)
  - (secondary) C4 with-theory t_IS=2.24 < 3.0

### `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12`

- Mode-A: N=521 ExpR=+0.0540 t_IS=+1.34 Sh_ann=+0.51
- Years positive: 6/7
- Evidence:
  - Rule 8.2 arithmetic_only (wr_spread < 3% with material ExpR_delta)
  - (secondary) C4 with-theory t_IS=1.34 < 3.0

### `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`

- Mode-A: N=844 ExpR=+0.0694 t_IS=+2.10 Sh_ann=+0.79
- Years positive: 7/7
- Evidence:
  - Rule 8.1 fire-rate 98.6% (>95%)
  - (secondary) C4 with-theory t_IS=2.10 < 3.0

### `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5`

- Mode-A: N=856 ExpR=+0.0664 t_IS=+2.03 Sh_ann=+0.77
- Years positive: 6/7
- Evidence:
  - Rule 8.1 fire-rate 99.7% (>95%)
  - (secondary) C4 with-theory t_IS=2.03 < 3.0

### `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`

- Mode-A: N=817 ExpR=+0.0887 t_IS=+2.11 Sh_ann=+0.80
- Years positive: 6/7
- Violating eras: 2015-2019:-0.053(N=81)
- Evidence:
  - Rule 8.1 fire-rate 98.6% (>95%)
  - C9 era stability FAIL: doctrine era(s) ['2015-2019'] ExpR<-0.05 (era-N>=50)
  - (secondary) C4 with-theory t_IS=2.11 < 3.0

### `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5`

- Mode-A: N=829 ExpR=+0.0858 t_IS=+2.06 Sh_ann=+0.78
- Years positive: 5/7
- Violating eras: 2015-2019:-0.059(N=86)
- Evidence:
  - Rule 8.1 fire-rate 99.7% (>95%)
  - C9 era stability FAIL: doctrine era(s) ['2015-2019'] ExpR<-0.05 (era-N>=50)
  - (secondary) C4 with-theory t_IS=2.06 < 3.0

### `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`

- Mode-A: N=496 ExpR=+0.2046 t_IS=+3.96 Sh_ann=+1.50
- Years positive: 6/7
- Evidence:
  - Rule 8.2 arithmetic_only (wr_spread < 3% with material ExpR_delta)
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-06 >= 2026-01-01; stored values are not Mode-A-clean.

## REVIEW_TIER2 — detail

### `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60`

- Mode-A: N=306 ExpR=+0.2140 t_IS=+4.19 Sh_ann=+1.58
- Years positive: 6/7
- Evidence:
  - RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio
  - Fire-rate audit reported 0% — artifact of missing CrossAssetATR injection in that audit's scope, NOT a lane defect. Criterion eval applies canonical injection and the lane has real data.

### `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM`

- Mode-A: N=535 ExpR=+0.0502 t_IS=+1.34 Sh_ann=+0.50
- Years positive: 5/7
- Violating eras: 2015-2019:-0.062(N=52)
- Evidence:
  - RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio
  - C9 era stability FAIL: doctrine era(s) ['2015-2019'] ExpR<-0.05 (era-N>=50)
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-14 >= 2026-01-01; stored values are not Mode-A-clean.
  - (secondary) C4 with-theory t_IS=1.34 < 3.0

### `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5`

- Mode-A: N=773 ExpR=+0.0353 t_IS=+1.10 Sh_ann=+0.42
- Years positive: 6/7
- Violating eras: 2015-2019:-0.105(N=51)
- Evidence:
  - RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio
  - C9 era stability FAIL: doctrine era(s) ['2015-2019'] ExpR<-0.05 (era-N>=50)
  - (secondary) C4 with-theory t_IS=1.10 < 3.0

### `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5`

- Mode-A: N=773 ExpR=+0.0735 t_IS=+1.54 Sh_ann=+0.58
- Years positive: 4/7
- Evidence:
  - RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio
  - (secondary) C4 with-theory t_IS=1.54 < 3.0

### `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12`

- Mode-A: N=469 ExpR=+0.0786 t_IS=+1.87 Sh_ann=+0.71
- Years positive: 6/7
- Evidence:
  - RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio
  - (secondary) C4 with-theory t_IS=1.87 < 3.0

### `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`

- Mode-A: N=460 ExpR=+0.1323 t_IS=+2.99 Sh_ann=+1.13
- Years positive: 5/7
- Evidence:
  - RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-02 >= 2026-01-01; stored values are not Mode-A-clean.
  - (secondary) C4 with-theory t_IS=2.99 < 3.0

### `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`

- Mode-A: N=436 ExpR=+0.1844 t_IS=+3.20 Sh_ann=+1.21
- Years positive: 6/7
- Evidence:
  - RETIREMENT Tier-2: excess decay 0.10-0.60 vs portfolio
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-02 >= 2026-01-01; stored values are not Mode-A-clean.

## REVIEW_CAPACITY — detail

### `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30`

- Mode-A: N=485 ExpR=+0.2210 t_IS=+4.14 Sh_ann=+1.56
- Years positive: 6/7
- Evidence:
  - SGP O15/O30 pair: Jaccard 0.65 — capacity review before parallel deploy
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-06 >= 2026-01-01; stored values are not Mode-A-clean.

## REVIEW_C4_WT_FAIL — detail

### `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100`

- Mode-A: N=263 ExpR=+0.0560 t_IS=+0.99 Sh_ann=+0.37
- Years positive: 4/6
- Evidence:
  - C4 with-theory FAIL: t_IS=0.99 < 3.0

### `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`

- Mode-A: N=773 ExpR=+0.0769 t_IS=+1.90 Sh_ann=+0.72
- Years positive: 6/7
- Evidence:
  - C4 with-theory FAIL: t_IS=1.90 < 3.0

### `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100`

- Mode-A: N=263 ExpR=+0.1180 t_IS=+1.65 Sh_ann=+0.62
- Years positive: 6/6
- Evidence:
  - C4 with-theory FAIL: t_IS=1.65 < 3.0

### `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60`

- Mode-A: N=345 ExpR=+0.0784 t_IS=+1.51 Sh_ann=+0.57
- Years positive: 5/7
- Evidence:
  - C4 with-theory FAIL: t_IS=1.51 < 3.0
  - Fire-rate audit reported 0% — artifact of missing CrossAssetATR injection in that audit's scope, NOT a lane defect. Criterion eval applies canonical injection and the lane has real data.

### `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60`

- Mode-A: N=334 ExpR=+0.0656 t_IS=+1.00 Sh_ann=+0.38
- Years positive: 5/7
- Evidence:
  - C4 with-theory FAIL: t_IS=1.00 < 3.0
  - Fire-rate audit reported 0% — artifact of missing CrossAssetATR injection in that audit's scope, NOT a lane defect. Criterion eval applies canonical injection and the lane has real data.

### `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`

- Mode-A: N=469 ExpR=+0.1036 t_IS=+1.95 Sh_ann=+0.74
- Years positive: 6/7
- Evidence:
  - C4 with-theory FAIL: t_IS=1.95 < 3.0

### `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5`

- Mode-A: N=786 ExpR=+0.0992 t_IS=+2.50 Sh_ann=+0.95
- Years positive: 6/7
- Violating eras: 2023:-0.094(N=113)
- Evidence:
  - C4 with-theory FAIL: t_IS=2.50 < 3.0
  - C9 era stability FAIL: doctrine era(s) ['2023'] ExpR<-0.05 (era-N>=50)

### `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5`

- Mode-A: N=785 ExpR=+0.1233 t_IS=+2.62 Sh_ann=+0.99
- Years positive: 5/7
- Evidence:
  - C4 with-theory FAIL: t_IS=2.62 < 3.0

### `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60`

- Mode-A: N=371 ExpR=+0.0770 t_IS=+1.56 Sh_ann=+0.59
- Years positive: 5/7
- Evidence:
  - C4 with-theory FAIL: t_IS=1.56 < 3.0
  - Fire-rate audit reported 0% — artifact of missing CrossAssetATR injection in that audit's scope, NOT a lane defect. Criterion eval applies canonical injection and the lane has real data.

### `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15`

- Mode-A: N=398 ExpR=+0.1481 t_IS=+2.09 Sh_ann=+0.79
- Years positive: 5/7
- Evidence:
  - C4 with-theory FAIL: t_IS=2.09 < 3.0
  - Mode-B grandfathered: stored ExpR last_trade_day=2026-04-02 >= 2026-01-01; stored values are not Mode-A-clean.

## BETTER_THAN_PEERS — detail

### `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100`

- Mode-A: N=283 ExpR=+0.1844 t_IS=+3.42 Sh_ann=+1.29
- Years positive: 5/5
- Evidence:
  - Regime-stress BETTER-THAN-PEERS: Sharpe rose early→late

### `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`

- Mode-A: N=278 ExpR=+0.1868 t_IS=+2.65 Sh_ann=+1.00
- Years positive: 5/5
- Evidence:
  - Regime-stress BETTER-THAN-PEERS: Sharpe rose early→late
  - (secondary) C4 with-theory t_IS=2.65 < 3.0

### `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60`

- Mode-A: N=379 ExpR=+0.1981 t_IS=+3.32 Sh_ann=+1.26
- Years positive: 5/7
- Evidence:
  - Regime-stress BETTER-THAN-PEERS: Sharpe rose early→late
  - Fire-rate audit reported 0% — artifact of missing CrossAssetATR injection in that audit's scope, NOT a lane defect. Criterion eval applies canonical injection and the lane has real data.

## KEEP — detail

### `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8`

- Mode-A: N=130 ExpR=+0.2798 t_IS=+3.66 Sh_ann=+1.49
- Years positive: 4/5
- Evidence:

### `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`

- Mode-A: N=385 ExpR=+0.1950 t_IS=+4.29 Sh_ann=+1.62
- Years positive: 7/7
- Evidence:
  - Fire-rate audit reported 0% — artifact of missing CrossAssetATR injection in that audit's scope, NOT a lane defect. Criterion eval applies canonical injection and the lane has real data.

## Scope — criteria NOT evaluated here

Per `pre_registered_criteria.md`, the 12 doctrine criteria include several not computable from this audit's inputs:

- **C1 (pre-reg file)**: requires checking `docs/audit/hypotheses/` for each lane's original pre-reg. Historical lanes pre-date Phase 0 literature grounding (2026-04-07); they are research-provisional per Amendment 2.7.
- **C2 (MinBTL)**: requires discovery run's trial count. Not stored.
- **C3 (BH-FDR)**: requires discovery hypothesis family. Not stored.
- **C5 (DSR)**: downgraded to INFORMATIONAL-only per Amendment 2.1 because N_eff unresolved.
- **C6 (WFE)**: requires OOS Sharpe computation under Mode A. Not in scope.
- **C8 (2026 OOS)**: 2026 is sacred holdout under Amendment 2.7.
- **C10 (data-era compat)**: filter-class specific (volume filters on MICRO era only). Not a standalone per-lane check.
- **C11 (account-death Monte Carlo)**: deployment-time gate, not audit-time.
- **C12 (Shiryaev-Roberts monitor)**: post-deployment drift gate, not audit-time.

## Committee action matrix

| Action | Verdict codes | Count | Timing |
|---|---|---:|---|
| Immediate retire vote | `RETIRE_URGENT` + `RETIRE_STANDARD` | 4 | This week |
| Immediate N-floor retire | `N_UNDERPOWERED` | 1 | This week |
| Route to cost-screen registry | `RECLASSIFY_COST` | 10 | Gated on Amendment v3.2 lock |
| Next-sprint review | `REVIEW_TIER2` + `REVIEW_CAPACITY` + `REVIEW_C4_WT_FAIL` | 18 | Within 2 weeks |
| Keep, potential scaling | `BETTER_THAN_PEERS` | 3 | No action required |
| Keep, no action | `KEEP` | 2 | No action required |

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db uv run python research/consolidated_retirement_verdict.py
```

Read-only audit. Numbers reproduce exactly on the same DB state. If source audit docs change, update hardcoded cross-reference sets at the top of the script and re-run.


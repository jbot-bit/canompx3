# Fast-lane walk report — 2026-05-20

schema_version: 1
source: scripts/tools/fast_lane_walk.py

## Chain steps

| step | rc |
|---|---|
| promote_queue | 0 |
| cherry_pick_ranker | 0 |
| journal_enricher | 0 |
| status_rollup | 0 |

## Counts per stage

| stage | n |
|---|---|
| ACTIVE_PREREG | 4 |
| HEAVYWEIGHT_COMPLETE | 38 |
| PARKED | 1 |
| REVOKED | 1 |
| **total** | **44** |

## Top-3 stalled (actionable stages only)

| strategy_id | stage | age_days | next_action |
|---|---|---|---|
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | HEAVYWEIGHT_COMPLETE | 16 | run_cherry_pick_journal_enricher |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | HEAVYWEIGHT_COMPLETE | 16 | run_cherry_pick_journal_enricher |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | HEAVYWEIGHT_COMPLETE | 16 | run_cherry_pick_journal_enricher |

## ERROR roll-up

_zero ERROR entries — chain is internally consistent._

## Next operator action

**strategy_id:** `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`  
**stage:** `HEAVYWEIGHT_COMPLETE`  
**next_action:** `run_cherry_pick_journal_enricher`  
**age_days:** 16


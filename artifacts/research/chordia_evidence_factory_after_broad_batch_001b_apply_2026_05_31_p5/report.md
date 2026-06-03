# Chordia Evidence Factory

This artifact is proposal-only. It drafts preregs and audit-log proposals, but it does not run strict replay, mutate live allocation, write validated_setups, or append chordia_audit_log.yaml.

## Summary

- work items: 673
- status counts: {'BLOCKED_NON_DEFAULT_STOP': 229, 'PREREG_DRAFT_READY': 444}
- batch count at size 25: 27
- audit-log proposals: 0
- non-default stop lanes are blocked by default because the current strict runner audits default-stop orb_outcomes only

## Work Items

| rank | status | strategy_id | stop | next_action |
| ---: | --- | --- | ---: | --- |
| 21 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 22 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 23 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 24 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 25 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 26 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 27 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 28 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR2.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 29 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 30 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 31 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR3.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 32 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 33 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 34 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 35 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 36 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 37 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT15_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 38 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 39 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 40 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 41 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 42 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 43 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 44 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 45 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 46 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 47 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 48 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 49 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 50 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 51 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 52 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 53 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 54 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 55 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 56 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 57 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 58 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 59 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 60 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 61 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 62 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 63 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 64 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 65 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 66 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 67 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 68 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 69 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 70 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 71 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 72 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 73 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 74 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 75 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 76 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 77 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 78 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 79 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 80 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 81 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 82 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 83 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 84 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 85 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 86 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 87 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 88 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 89 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 90 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 91 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 92 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 93 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 94 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 95 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 96 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 97 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 98 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 99 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 100 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 101 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 102 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 103 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 104 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 105 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 106 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 107 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 108 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 109 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 110 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 111 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 112 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 113 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 114 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 115 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 116 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 117 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 118 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 119 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 120 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 121 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 122 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 123 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 124 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 125 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 126 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 127 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 128 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 129 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 130 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 131 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 132 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 133 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 134 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 135 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 136 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 137 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 138 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 139 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 140 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 141 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 142 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 143 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 144 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 145 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 146 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 147 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 148 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 149 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 150 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 151 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 152 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 153 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 154 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 155 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 156 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 157 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 158 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 159 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 160 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 161 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 162 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 163 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 164 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 165 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 166 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 167 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 168 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 169 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 170 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 171 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 172 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 173 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 174 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 175 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 176 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 177 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 178 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 179 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 180 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 181 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 182 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 183 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 184 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 185 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 186 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 187 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 188 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 189 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 190 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 191 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 192 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 193 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 194 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 195 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 196 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 197 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 198 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 199 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 200 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 201 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 202 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 203 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 204 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 205 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 206 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 207 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 208 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 209 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 210 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 211 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 212 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 213 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 214 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 215 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 216 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 217 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 218 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 219 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 220 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 221 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 222 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 223 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 224 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 225 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 226 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 227 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 228 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 229 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 230 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 231 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 232 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 233 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 234 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 235 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 236 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 237 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 238 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 239 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 240 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 241 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 242 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 243 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 244 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 245 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 246 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 247 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 248 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 249 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 250 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 251 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 252 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 253 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 254 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 255 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 256 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 257 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 258 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 259 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 260 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 261 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 262 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 263 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 264 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 265 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 266 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 267 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 268 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 269 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 270 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 271 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 272 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 273 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 274 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 275 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 276 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 277 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 278 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 279 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 280 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 281 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 282 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 283 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 284 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 285 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 286 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 287 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 288 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 289 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 290 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 291 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 292 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 293 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 294 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 295 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 296 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 297 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 298 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 299 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 300 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R080_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 301 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 302 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 303 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 304 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 305 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 306 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 307 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 308 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 309 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 310 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 311 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 312 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 313 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 314 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 315 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 316 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 317 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 318 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 319 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 320 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 321 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 322 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 323 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 324 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 325 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 326 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 327 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 328 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 329 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 330 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 331 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 332 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 333 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 334 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 335 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 336 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 337 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 338 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 339 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 340 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 341 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 342 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 343 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 344 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 345 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 346 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 347 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 348 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 349 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 350 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 351 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 352 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 353 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 354 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 355 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 356 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 357 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 358 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 359 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 360 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 361 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 362 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 363 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 364 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 365 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB2_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 366 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 367 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 368 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 369 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 370 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 371 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 372 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 373 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 374 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 375 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 376 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 377 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 378 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 379 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 380 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 381 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 382 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 383 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 384 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 385 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 386 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 387 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 388 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 389 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 390 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 391 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 392 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 393 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 394 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 395 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 396 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 397 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB4_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 398 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 399 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 400 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 401 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 402 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 403 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 404 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 405 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 406 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 407 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 408 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 409 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 410 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 411 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 412 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 413 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 414 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 415 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 416 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 417 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 418 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 419 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 420 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 421 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 422 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 423 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VOL_RV15_N20_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 424 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 425 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 426 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 427 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 428 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 429 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 430 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 431 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 432 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 433 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 434 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 435 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 436 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 437 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 438 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 439 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 440 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 441 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 442 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 443 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 444 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 445 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 446 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 447 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 448 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 449 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 450 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 451 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 452 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 453 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 454 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 455 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 456 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT08_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 457 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT10_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 458 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT12_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 459 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G4_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 460 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G5_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 461 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G8_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 462 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G4_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 463 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G5_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 464 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G6_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 465 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G8_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 466 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 467 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 468 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 469 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 470 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 471 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 472 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 473 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 474 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 475 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 476 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 477 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 478 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 479 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 480 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 481 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 482 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 483 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 484 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 485 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 486 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 487 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 488 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 489 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 490 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 491 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 492 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 493 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 494 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 495 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 496 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 497 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 498 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 499 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 500 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 501 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 502 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 503 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 504 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 505 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 506 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 507 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 508 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 509 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 510 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 511 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 512 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 513 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 514 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 515 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 516 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 517 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 518 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 519 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 520 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 521 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 522 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 523 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 524 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 525 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 526 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 527 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 528 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 529 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 530 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 531 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 532 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_OVNRNG_50_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 533 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 534 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 535 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 536 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 537 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 538 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 539 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 540 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 541 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT08_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 542 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT10_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 543 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT12_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 544 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G4_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 545 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G5_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 546 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G8_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 547 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_OVNRNG_25_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 548 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 549 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_ATR70_VOL_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 550 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 551 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 552 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 553 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 554 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 555 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 556 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 557 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 558 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 559 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 560 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 561 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 562 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 563 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 564 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 565 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 566 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 567 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 568 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 569 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 570 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 571 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 572 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 573 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB4_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 574 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 575 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 576 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 577 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 578 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 579 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 580 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 581 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 582 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 583 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 584 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 585 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 586 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 587 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 588 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 589 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 590 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 591 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 592 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 593 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 594 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 595 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 596 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB4_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 597 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 598 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 599 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 600 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 601 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 602 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 603 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 604 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 605 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 606 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 607 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 608 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 609 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 610 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 611 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 612 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 613 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 614 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 615 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 616 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 617 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 618 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 619 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 620 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 621 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 622 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 623 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 624 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 625 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 626 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 627 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 628 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 629 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 630 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 631 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 632 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 633 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 634 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 635 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 636 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 637 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 638 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 639 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 640 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 641 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 642 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 643 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 644 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 645 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 646 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 647 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 648 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 649 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 650 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 651 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 652 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 653 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 654 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 655 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 656 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 657 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 658 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 659 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV20_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 660 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 661 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 662 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 663 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR70_VOL_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 664 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 665 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 666 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 667 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV15_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 668 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 669 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 670 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 671 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 672 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 673 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 674 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 675 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 676 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 677 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 678 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 679 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 680 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 681 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 682 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 683 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 684 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 685 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 686 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 687 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 688 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 689 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB3_PD_CLEAR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 690 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 691 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 692 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 693 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |

# Chordia Evidence Factory

This artifact is proposal-only. It drafts preregs and audit-log proposals, but it does not run strict replay, mutate live allocation, write validated_setups, or append chordia_audit_log.yaml.

## Summary

- work items: 684
- status counts: {'BLOCKED_NON_DEFAULT_STOP': 229, 'PREREG_DRAFT_READY': 455}
- batch count at size 25: 28
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
| 31 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_VOL_2K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 32 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR3.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 33 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 34 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 35 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 36 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 37 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 38 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 39 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 40 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 41 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 42 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_NO_FILTER_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 43 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 44 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 45 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 46 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 47 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 48 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT15_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 49 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 50 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 51 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 52 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 53 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 54 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 55 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 56 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 57 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 58 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 59 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 60 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 61 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 62 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 63 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 64 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 65 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 66 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 67 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 68 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 69 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 70 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 71 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 72 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 73 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 74 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 75 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 76 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 77 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 78 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 79 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 80 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 81 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 82 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 83 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 84 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 85 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 86 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 87 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 88 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 89 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 90 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 91 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 92 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 93 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 94 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 95 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 96 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 97 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 98 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 99 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 100 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 101 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 102 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 103 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 104 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 105 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 106 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 107 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 108 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 109 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 110 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 111 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 112 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 113 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 114 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 115 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 116 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 117 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 118 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 119 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 120 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 121 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 122 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 123 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 124 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 125 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 126 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 127 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 128 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 129 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 130 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 131 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 132 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 133 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 134 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 135 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 136 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 137 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 138 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 139 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 140 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 141 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 142 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 143 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 144 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 145 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 146 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 147 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 148 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 149 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 150 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 151 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 152 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 153 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 154 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 155 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 156 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 157 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 158 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 159 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 160 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 161 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 162 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 163 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 164 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 165 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 166 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 167 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 168 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 169 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 170 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 171 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 172 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 173 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 174 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 175 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 176 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 177 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 178 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 179 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 180 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 181 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 182 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 183 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 184 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 185 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 186 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 187 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 188 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 189 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 190 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 191 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 192 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 193 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 194 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 195 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 196 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 197 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 198 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 199 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 200 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 201 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 202 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 203 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 204 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 205 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 206 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 207 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 208 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 209 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 210 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 211 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 212 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 213 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 214 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 215 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 216 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 217 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 218 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 219 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 220 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 221 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 222 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 223 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 224 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 225 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 226 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 227 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 228 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 229 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 230 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 231 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 232 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 233 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 234 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 235 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 236 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 237 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 238 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 239 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 240 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 241 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 242 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 243 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 244 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 245 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 246 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 247 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 248 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 249 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 250 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 251 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 252 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 253 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 254 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 255 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 256 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 257 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 258 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 259 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 260 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 261 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 262 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 263 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 264 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 265 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 266 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 267 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 268 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 269 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 270 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 271 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 272 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 273 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 274 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 275 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 276 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 277 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 278 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 279 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 280 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 281 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 282 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 283 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 284 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 285 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 286 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 287 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 288 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 289 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 290 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 291 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 292 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 293 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 294 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 295 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 296 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 297 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 298 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 299 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 300 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 301 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 302 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 303 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 304 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 305 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 306 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 307 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 308 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 309 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 310 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 311 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R080_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 312 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 313 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 314 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 315 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 316 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 317 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 318 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 319 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 320 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 321 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 322 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 323 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 324 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 325 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 326 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 327 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 328 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 329 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 330 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 331 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 332 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 333 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 334 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 335 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 336 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 337 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 338 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 339 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 340 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 341 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 342 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 343 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 344 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 345 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 346 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 347 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 348 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 349 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 350 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 351 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 352 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 353 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 354 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 355 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 356 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 357 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 358 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 359 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 360 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 361 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 362 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 363 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 364 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 365 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 366 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 367 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 368 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 369 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 370 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 371 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 372 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 373 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 374 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 375 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 376 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB2_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 377 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 378 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 379 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 380 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 381 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 382 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 383 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 384 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 385 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 386 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 387 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 388 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 389 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 390 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 391 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 392 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 393 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 394 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 395 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 396 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 397 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 398 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 399 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 400 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 401 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 402 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 403 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 404 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 405 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 406 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 407 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 408 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB4_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 409 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 410 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 411 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 412 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 413 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 414 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 415 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 416 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 417 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 418 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 419 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 420 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 421 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 422 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 423 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 424 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 425 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 426 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 427 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 428 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 429 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 430 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 431 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 432 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 433 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 434 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VOL_RV15_N20_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 435 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 436 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 437 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 438 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 439 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 440 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 441 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 442 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 443 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 444 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 445 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 446 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 447 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 448 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 449 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 450 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 451 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 452 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 453 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 454 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 455 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 456 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 457 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 458 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 459 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 460 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 461 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 462 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 463 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 464 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 465 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 466 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 467 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT08_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 468 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT10_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 469 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT12_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 470 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G4_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 471 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G5_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 472 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G8_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 473 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G4_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 474 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G5_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 475 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G6_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 476 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G8_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 477 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 478 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 479 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 480 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 481 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 482 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 483 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 484 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 485 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 486 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 487 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 488 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 489 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 490 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 491 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 492 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 493 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 494 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 495 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 496 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 497 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 498 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 499 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 500 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 501 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 502 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 503 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 504 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 505 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 506 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 507 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 508 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 509 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 510 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 511 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 512 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 513 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 514 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 515 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 516 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 517 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 518 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 519 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 520 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 521 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 522 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 523 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 524 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 525 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 526 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 527 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 528 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 529 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 530 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 531 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 532 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 533 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 534 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 535 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 536 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 537 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 538 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 539 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 540 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 541 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 542 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 543 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_OVNRNG_50_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 544 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 545 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 546 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 547 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 548 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 549 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 550 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 551 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 552 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT08_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 553 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT10_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 554 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT12_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 555 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G4_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 556 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G5_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 557 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G8_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 558 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_OVNRNG_25_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 559 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 560 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_ATR70_VOL_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 561 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 562 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 563 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 564 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 565 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 566 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 567 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 568 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 569 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 570 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 571 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 572 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 573 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 574 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 575 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 576 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 577 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 578 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 579 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 580 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 581 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 582 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 583 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 584 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB4_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 585 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 586 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 587 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 588 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 589 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 590 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 591 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 592 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 593 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 594 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 595 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 596 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 597 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 598 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 599 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 600 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 601 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 602 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 603 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 604 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 605 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 606 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 607 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB4_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 608 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 609 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 610 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 611 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 612 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 613 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 614 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 615 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 616 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 617 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 618 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 619 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 620 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 621 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 622 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 623 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 624 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 625 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 626 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 627 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 628 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 629 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 630 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 631 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 632 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 633 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 634 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 635 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 636 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 637 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 638 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 639 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 640 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 641 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 642 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 643 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 644 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 645 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 646 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 647 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 648 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 649 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 650 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 651 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 652 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 653 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 654 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 655 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 656 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 657 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 658 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 659 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 660 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 661 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 662 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 663 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 664 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 665 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 666 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 667 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 668 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 669 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 670 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV20_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 671 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 672 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 673 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 674 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR70_VOL_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 675 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 676 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 677 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 678 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV15_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 679 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 680 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 681 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 682 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 683 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 684 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 685 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 686 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 687 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 688 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 689 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 690 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 691 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 692 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 693 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 694 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 695 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 696 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 697 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 698 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 699 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 700 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB3_PD_CLEAR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 701 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 702 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 703 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 704 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |

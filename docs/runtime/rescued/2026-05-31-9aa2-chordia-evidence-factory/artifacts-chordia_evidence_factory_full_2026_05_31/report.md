# Chordia Evidence Factory

This artifact is proposal-only. It drafts preregs and audit-log proposals, but it does not run strict replay, mutate live allocation, write validated_setups, or append chordia_audit_log.yaml.

## Summary

- work items: 708
- status counts: {'BLOCKED_NON_DEFAULT_STOP': 229, 'PREREG_DRAFT_READY': 479}
- batch count at size 25: 29
- audit-log proposals: 0
- non-default stop lanes are blocked by default because the current strict runner audits default-stop orb_outcomes only

## Work Items

| rank | status | strategy_id | stop | next_action |
| ---: | --- | --- | ---: | --- |
| 18 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 19 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 20 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 21 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 22 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 23 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 24 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 25 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 26 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 27 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 28 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 29 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 30 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 31 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 32 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 33 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 34 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 35 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 36 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 37 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 38 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR2.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 39 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR2.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 40 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 41 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 42 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 43 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 44 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT15_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 45 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 46 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 47 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_NO_FILTER_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 48 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 49 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 50 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 51 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 52 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_VOL_2K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 53 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR3.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 54 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 55 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 56 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 57 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 58 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 59 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 60 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 61 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 62 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 63 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_NO_FILTER_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 64 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 65 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 66 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 67 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 68 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 69 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT15_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 70 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 71 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 72 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 73 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 74 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 75 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 76 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 77 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 78 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 79 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 80 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 81 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 82 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 83 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 84 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 85 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 86 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 87 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 88 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 89 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 90 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 91 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 92 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 93 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 94 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 95 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 96 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 97 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 98 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 99 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 100 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 101 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 102 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 103 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 104 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 105 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 106 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 107 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 108 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 109 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 110 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 111 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 112 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 113 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 114 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 115 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 116 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 117 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 118 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 119 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 120 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 121 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 122 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 123 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 124 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 125 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 126 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 127 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 128 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 129 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 130 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 131 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 132 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 133 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 134 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 135 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 136 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 137 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 138 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 139 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 140 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 141 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 142 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 143 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 144 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 145 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 146 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 147 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 148 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 149 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 150 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 151 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 152 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 153 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 154 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 155 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 156 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 157 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 158 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 159 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 160 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 161 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 162 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 163 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 164 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 165 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 166 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 167 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 168 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 169 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 170 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 171 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 172 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 173 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 174 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 175 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 176 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 177 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 178 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 179 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 180 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 181 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 182 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 183 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 184 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 185 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 186 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 187 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 188 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 189 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 190 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 191 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 192 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 193 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 194 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 195 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 196 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 197 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 198 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 199 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 200 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 201 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 202 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 203 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 204 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 205 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 206 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 207 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 208 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 209 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 210 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 211 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 212 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 213 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 214 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 215 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 216 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 217 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 218 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 219 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 220 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 221 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 222 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 223 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 224 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 225 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 226 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 227 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 228 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 229 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 230 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 231 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 232 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 233 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 234 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 235 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 236 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 237 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 238 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 239 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 240 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 241 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 242 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 243 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 244 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 245 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 246 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 247 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 248 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 249 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 250 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 251 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 252 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 253 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 254 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 255 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 256 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 257 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 258 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 259 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 260 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 261 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 262 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 263 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 264 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 265 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 266 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 267 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 268 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 269 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 270 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 271 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 272 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 273 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 274 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 275 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 276 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 277 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 278 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 279 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 280 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 281 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 282 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 283 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 284 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 285 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 286 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 287 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 288 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 289 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 290 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 291 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 292 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 293 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 294 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 295 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 296 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 297 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 298 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 299 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 300 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 301 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 302 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 303 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 304 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 305 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 306 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 307 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 308 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 309 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 310 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 311 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 312 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 313 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 314 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 315 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 316 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 317 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 318 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 319 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 320 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 321 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 322 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 323 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 324 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 325 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 326 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 327 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 328 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 329 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 330 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 331 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 332 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R080_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 333 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 334 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 335 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 336 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 337 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 338 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 339 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 340 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 341 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 342 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 343 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 344 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 345 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 346 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 347 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 348 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 349 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 350 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 351 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 352 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 353 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 354 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 355 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 356 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 357 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 358 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 359 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 360 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 361 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 362 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 363 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 364 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 365 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 366 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 367 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 368 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 369 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 370 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 371 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 372 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 373 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 374 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 375 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 376 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 377 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 378 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 379 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 380 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 381 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 382 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 383 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 384 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 385 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 386 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 387 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 388 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 389 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 390 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 391 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 392 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 393 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 394 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 395 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 396 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 397 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB2_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 398 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 399 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 400 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 401 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 402 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 403 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 404 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 405 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 406 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 407 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 408 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 409 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 410 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 411 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 412 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 413 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 414 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 415 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 416 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 417 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 418 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 419 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 420 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 421 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 422 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 423 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 424 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 425 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 426 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 427 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 428 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 429 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB4_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 430 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 431 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 432 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 433 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 434 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 435 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 436 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 437 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 438 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 439 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 440 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 441 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 442 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 443 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 444 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 445 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 446 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 447 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 448 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 449 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 450 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 451 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 452 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 453 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 454 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 455 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VOL_RV15_N20_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 456 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 457 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 458 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 459 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 460 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 461 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 462 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 463 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 464 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 465 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 466 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 467 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 468 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 469 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 470 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 471 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 472 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 473 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 474 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 475 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 476 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 477 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 478 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 479 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 480 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 481 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 482 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 483 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 484 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 485 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 486 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 487 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 488 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT08_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 489 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT10_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 490 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT12_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 491 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G4_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 492 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G5_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 493 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G8_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 494 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G4_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 495 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G5_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 496 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G6_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 497 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G8_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 498 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 499 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 500 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 501 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 502 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 503 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 504 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 505 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 506 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 507 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 508 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 509 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 510 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 511 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 512 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 513 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 514 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 515 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 516 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 517 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 518 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 519 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 520 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 521 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 522 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 523 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 524 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 525 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 526 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 527 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 528 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 529 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 530 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 531 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 532 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 533 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 534 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 535 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 536 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 537 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 538 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 539 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 540 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 541 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 542 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 543 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 544 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 545 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 546 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 547 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 548 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 549 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 550 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 551 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 552 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 553 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 554 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 555 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 556 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 557 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 558 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 559 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 560 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 561 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 562 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 563 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 564 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_OVNRNG_50_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 565 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 566 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 567 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 568 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 569 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 570 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 571 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 572 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 573 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT08_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 574 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT10_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 575 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT12_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 576 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G4_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 577 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G5_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 578 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G8_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 579 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_OVNRNG_25_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 580 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 581 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_ATR70_VOL_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 582 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 583 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 584 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 585 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 586 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 587 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 588 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 589 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 590 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 591 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 592 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 593 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 594 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 595 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 596 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 597 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 598 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 599 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 600 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 601 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 602 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 603 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 604 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 605 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB4_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 606 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 607 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 608 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 609 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 610 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 611 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 612 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 613 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 614 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 615 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 616 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 617 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 618 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 619 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 620 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 621 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 622 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 623 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 624 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 625 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 626 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 627 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 628 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB4_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 629 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 630 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 631 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 632 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 633 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 634 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 635 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 636 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 637 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 638 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 639 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 640 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 641 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 642 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 643 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 644 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 645 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 646 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 647 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 648 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 649 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 650 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 651 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 652 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 653 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 654 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 655 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 656 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 657 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 658 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 659 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 660 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 661 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 662 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 663 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 664 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 665 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 666 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 667 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 668 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 669 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 670 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 671 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 672 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 673 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 674 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 675 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 676 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 677 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 678 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 679 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 680 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 681 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 682 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 683 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 684 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 685 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 686 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 687 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 688 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 689 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 690 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 691 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV20_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 692 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 693 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 694 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 695 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR70_VOL_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 696 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 697 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 698 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 699 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV15_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 700 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 701 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 702 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 703 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 704 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 705 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 706 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 707 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 708 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 709 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 710 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 711 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 712 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 713 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 714 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 715 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 716 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 717 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 718 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 719 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 720 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 721 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB3_PD_CLEAR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 722 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 723 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 724 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 725 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |

# Chordia Evidence Factory

This artifact is proposal-only. It drafts preregs and audit-log proposals, but it does not run strict replay, mutate live allocation, write validated_setups, or append chordia_audit_log.yaml.

## Summary

- work items: 699
- status counts: {'BLOCKED_NON_DEFAULT_STOP': 229, 'PREREG_DRAFT_READY': 470}
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
| 27 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 28 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 29 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 30 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 31 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 32 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR2.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 33 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR2.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 34 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 35 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 36 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 37 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 38 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT15_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 39 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 40 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 41 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_NO_FILTER_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 42 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 43 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 44 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 45 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 46 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_VOL_2K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 47 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR3.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 48 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 49 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 50 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 51 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 52 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 53 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 54 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 55 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 56 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 57 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_NO_FILTER_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 58 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 59 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 60 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 61 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 62 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 63 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT15_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 64 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 65 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 66 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 67 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 68 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 69 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 70 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 71 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 72 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 73 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 74 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 75 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 76 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 77 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 78 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 79 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 80 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 81 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 82 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 83 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 84 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 85 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 86 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 87 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 88 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 89 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 90 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 91 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 92 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 93 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 94 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 95 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 96 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 97 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 98 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 99 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 100 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 101 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 102 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 103 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 104 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 105 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 106 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 107 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 108 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 109 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 110 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 111 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 112 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 113 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 114 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 115 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 116 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 117 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 118 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 119 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 120 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 121 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 122 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 123 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 124 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 125 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 126 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 127 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 128 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 129 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 130 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 131 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 132 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 133 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 134 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 135 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 136 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 137 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 138 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 139 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 140 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 141 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 142 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 143 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 144 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 145 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 146 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 147 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 148 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 149 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 150 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 151 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 152 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 153 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 154 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 155 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 156 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 157 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 158 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 159 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 160 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 161 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 162 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 163 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 164 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 165 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 166 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 167 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 168 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 169 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 170 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 171 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_DIR_LONG_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 172 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 173 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 174 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 175 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 176 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 177 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 178 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 179 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 180 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 181 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 182 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 183 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 184 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 185 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 186 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 187 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 188 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 189 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 190 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 191 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 192 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 193 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 194 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 195 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 196 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 197 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 198 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 199 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 200 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 201 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 202 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 203 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 204 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 205 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 206 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 207 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 208 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 209 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 210 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 211 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 212 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 213 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 214 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 215 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 216 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 217 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 218 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 219 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 220 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 221 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 222 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 223 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 224 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 225 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 226 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 227 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 228 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 229 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 230 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 231 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 232 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 233 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 234 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 235 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 236 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 237 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 238 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 239 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 240 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 241 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 242 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 243 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 244 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 245 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 246 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 247 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 248 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 249 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 250 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 251 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 252 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 253 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 254 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 255 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 256 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 257 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 258 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 259 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 260 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 261 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 262 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 263 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 264 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 265 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 266 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 267 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 268 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 269 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 270 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 271 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 272 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 273 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 274 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 275 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 276 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 277 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 278 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 279 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 280 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 281 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 282 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 283 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 284 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 285 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 286 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 287 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 288 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 289 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 290 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_DIR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 291 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 292 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 293 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 294 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 295 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 296 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 297 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 298 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 299 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 300 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 301 | BLOCKED_NON_DEFAULT_STOP | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P70_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 302 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 303 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 304 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 305 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 306 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 307 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 308 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 309 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 310 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 311 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 312 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 313 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 314 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 315 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 316 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 317 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 318 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 319 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 320 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 321 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 322 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 323 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 324 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 325 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 326 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R080_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 327 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 328 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 329 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 330 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 331 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 332 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 333 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 334 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 335 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 336 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 337 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 338 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 339 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 340 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 341 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 342 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 343 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 344 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 345 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 346 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 347 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 348 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 349 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 350 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 351 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 352 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 353 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 354 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 355 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 356 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 357 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 358 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 359 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 360 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 361 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 362 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 363 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 364 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 365 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 366 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 367 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 368 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 369 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 370 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 371 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 372 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 373 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 374 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 375 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 376 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 377 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 378 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 379 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 380 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 381 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 382 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 383 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 384 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 385 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 386 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 387 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 388 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 389 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 390 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 391 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB2_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 392 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 393 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 394 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 395 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 396 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 397 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 398 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 399 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 400 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 401 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R080` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 402 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 403 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 404 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 405 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 406 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 407 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 408 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT08_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 409 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 410 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 411 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 412 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 413 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 414 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 415 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 416 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 417 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 418 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 419 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 420 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 421 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_X_MES_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 422 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 423 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB4_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 424 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 425 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 426 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 427 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 428 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 429 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 430 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 431 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 432 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 433 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 434 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 435 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 436 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 437 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 438 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 439 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 440 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 441 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 442 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 443 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 444 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 445 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 446 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 447 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 448 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 449 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VOL_RV15_N20_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 450 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 451 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 452 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 453 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 454 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 455 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 456 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 457 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 458 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 459 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 460 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 461 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 462 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 463 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 464 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 465 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 466 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 467 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 468 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 469 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 470 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 471 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_NO_FILTER_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 472 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 473 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 474 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 475 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 476 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 477 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 478 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 479 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 480 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 481 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 482 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT08_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 483 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT10_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 484 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_COST_LT12_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 485 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G4_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 486 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G5_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 487 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G8_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 488 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G4_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 489 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G5_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 490 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G6_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 491 | PREREG_DRAFT_READY | `MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G8_NOMON_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 492 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 493 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 494 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 495 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 496 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 497 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 498 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 499 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_VOL_4K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 500 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 501 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 502 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 503 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 504 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 505 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 506 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 507 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 508 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 509 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 510 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 511 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 512 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 513 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 514 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 515 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 516 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 517 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 518 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 519 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 520 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 521 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 522 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G4_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 523 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G5_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 524 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G6_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 525 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_VOL_2K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 526 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 527 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_25_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 528 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 529 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 530 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 531 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 532 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 533 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_50_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 534 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 535 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 536 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 537 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 538 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 539 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 540 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 541 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 542 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 543 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 544 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 545 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 546 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 547 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 548 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 549 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 550 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 551 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 552 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 553 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 554 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 555 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 556 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 557 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 558 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_OVNRNG_50_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 559 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 560 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 561 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 562 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 563 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 564 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 565 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 566 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_X_MGC_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 567 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT08_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 568 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT10_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 569 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_COST_LT12_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 570 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G4_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 571 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G5_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 572 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G8_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 573 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB5_OVNRNG_25_FAST10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 574 | PREREG_DRAFT_READY | `MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_COST_LT15_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 575 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_ATR70_VOL_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 576 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_CROSS_NYSE_MOMENTUM` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 577 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 578 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 579 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 580 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 581 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 582 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 583 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 584 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 585 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 586 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 587 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_OVNRNG_100` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 588 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 589 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 590 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 591 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 592 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 593 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 594 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 595 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 596 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 597 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 598 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 599 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB4_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 600 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 601 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 602 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 603 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 604 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 605 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 606 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 607 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 608 | BLOCKED_NON_DEFAULT_STOP | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 609 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 610 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_VWAP_MID_ALIGNED` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 611 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 612 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 613 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 614 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 615 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 616 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 617 | BLOCKED_NON_DEFAULT_STOP | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_G8_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 618 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 619 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 620 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 621 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 622 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB4_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 623 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 624 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT12_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 625 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 626 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 627 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 628 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 629 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT15_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 630 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 631 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 632 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G8_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 633 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 634 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_2K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 635 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 636 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 637 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 638 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 639 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 640 | PREREG_DRAFT_READY | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 641 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 642 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_NO_FILTER` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 643 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT10_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 644 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 645 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 646 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 647 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 648 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R105` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 649 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 650 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 651 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 652 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 653 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 654 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 655 | PREREG_DRAFT_READY | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 656 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 657 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 658 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 659 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 660 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 661 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 662 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 663 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 664 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 665 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 666 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 667 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 668 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 669 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G4` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 670 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 671 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 672 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 673 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 674 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 675 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 676 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 677 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 678 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 679 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 680 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 681 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 682 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 683 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 684 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 685 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV20_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 686 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 687 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR_P70_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 688 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_8K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 689 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR70_VOL_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 690 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 691 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 692 | BLOCKED_NON_DEFAULT_STOP | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50_O15_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 693 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV15_N20_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 694 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 695 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 696 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 697 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 698 | BLOCKED_NON_DEFAULT_STOP | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 699 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 700 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 701 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_50` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 702 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR60_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 703 | BLOCKED_NON_DEFAULT_STOP | `MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 704 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G6_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 705 | PREREG_DRAFT_READY | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 706 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 707 | BLOCKED_NON_DEFAULT_STOP | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125_S075` | 0.75 | Requires outcome_builder rebuild at stop_multiplier=0.75 and a stop-specific strict replay runner before Chordia audit. |
| 708 | PREREG_DRAFT_READY | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 709 | PREREG_DRAFT_READY | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 710 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 711 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 712 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 713 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 714 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 715 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB3_PD_CLEAR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 716 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 717 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 718 | PREREG_DRAFT_READY | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |
| 719 | PREREG_DRAFT_READY | `MNQ_US_DATA_1000_E2_RR2.5_CB1_COST_LT10` | 1 | Review draft, move to active hypotheses if accepted, then run strict Chordia replay. |

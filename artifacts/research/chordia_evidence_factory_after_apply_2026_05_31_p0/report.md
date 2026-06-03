# Chordia Evidence Factory

This artifact is proposal-only. It drafts preregs and audit-log proposals, but it does not run strict replay, mutate live allocation, write validated_setups, or append chordia_audit_log.yaml.

## Summary

- work items: 13
- status counts: {'BLOCKED_NON_DEFAULT_STOP': 5, 'PREREG_DRAFT_READY': 8}
- batch count at size 25: 1
- audit-log proposals: 0
- non-default stop lanes are blocked by default because the current strict runner audits default-stop orb_outcomes only

## Work Items

| rank | status | strategy_id | stop | next_action |
| ---: | --- | --- | ---: | --- |
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

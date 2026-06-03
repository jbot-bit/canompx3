# Chordia Evidence Factory

This artifact is proposal-only. It drafts preregs and audit-log proposals, but it does not run strict replay, mutate live allocation, write validated_setups, or append chordia_audit_log.yaml.

## Summary

- work items: 5
- status counts: {'BLOCKED_NON_DEFAULT_STOP': 5}
- batch count at size 25: 1
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

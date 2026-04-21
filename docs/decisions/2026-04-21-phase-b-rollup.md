# Phase B Rollup — 2026-04-21

- Snapshot authority: `5e768af8` in [the Phase A truth ledger](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md)
- Scope: 6 live lanes only, recalibrated after Phase A contradictions.

| Lane | DSR @ rho=0.7 | Chordia | WFE | Holdout | SR | Verdict |
|---|---:|---:|---:|---|---|---|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `0.000000` | `2.528` | `2.8551` | `FAIL` | `CONTINUE` | `DEGRADE` |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | `0.000000` | `2.928` | `1.4222` | `FAIL` | `CONTINUE` | `DEGRADE` |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `0.000000` | `3.717` | `2.6151` | `FAIL` | `CONTINUE` | `DEGRADE` |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | `0.000000` | `3.511` | `1.9835` | `FAIL` | `ALARM` | `PAUSE-PENDING-REVIEW` |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `0.000000` | `3.400` | `0.8225` | `FAIL` | `CONTINUE` | `DEGRADE` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | `0.000000` | `2.831` | `0.7008` | `FAIL` | `CONTINUE` | `DEGRADE` |

## Summary

- `PAUSE-PENDING-REVIEW`: `1`
- `DEGRADE`: `5`
- `KEEP`: `0`

No lane cleared a clean `KEEP` path in this phase. The NYSE lane is paused under Criterion 12; the remaining five fail closed to `DEGRADE` on holdout integrity plus additional gate deficits.

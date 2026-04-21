# Phase B Lane Verdict — `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`

- Snapshot authority: `5e768af8` in [the Phase A truth ledger](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md)
- Final verdict: `PAUSE-PENDING-REVIEW`
- Phase A driver: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` carried `SR state = ALARM` and `Holdout-clean = FAIL` in the binding snapshot.

## Phase A evidence excerpt

From Phase A E7:

```text
Lane                                        N         SR      Thr Status
L1 EUROPE_FLOW ORB_G5                      22       2.02    31.96 CONTINUE
L2 SINGAPORE_OPEN ATR_P50                   4       1.90    31.96 CONTINUE
L3 COMEX_SETTLE ORB_G5                     16       5.76    31.96 CONTINUE
L4 NYSE_OPEN COST_LT12                     21      33.27    31.96 ALARM
L5 TOKYO_OPEN COST_LT12                    20       6.63    31.96 CONTINUE
L6 US_DATA_1000 ORB_G5                      5       0.63    31.96 CONTINUE
```

From Phase A A3:

```text
MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | fitness=FIT | t≈3.511 | WFE=1.9835 | DSR=0.0000001230 | n_trials=35616 | holdout=FAIL | SR=ALARM | baseline verdict=RESEARCH-PROVISIONAL + SR REVIEW
```

## Gate summary

| Gate | Result | Notes |
|---|---|---|
| G1 timing-validity | PASS | Phase A A4 marked the live six-lane book timing-valid under current framing. |
| G3 DSR bracket | FAIL | rho=0.7 -> DSR=0.000000 |
| G4 Chordia | FAIL | |t|=3.511; no verified local-literature theory band claimed |
| L3 WFE | PASS | WFE=1.9835 |
| L10 holdout integrity | FAIL | discovery_date=2026-04-11 with wf_tested=True, wf_passed=True |
| SR state | ALARM | Phase A E7 report-only SR monitor |

## Attached certificates

- [G1 timing-validity](/mnt/c/Users/joshd/canompx3/docs/audit/certificates/2026-04-21-lane-verdict-mnq-nyse-open-e2-rr1p0-cb1-cost-lt12/G1-timing-validity.md)
- [G3 DSR + N̂](/mnt/c/Users/joshd/canompx3/docs/audit/certificates/2026-04-21-lane-verdict-mnq-nyse-open-e2-rr1p0-cb1-cost-lt12/G3-dsr-neff.md)
- [G4 Chordia band](/mnt/c/Users/joshd/canompx3/docs/audit/certificates/2026-04-21-lane-verdict-mnq-nyse-open-e2-rr1p0-cb1-cost-lt12/G4-chordia-band.md)
- [G6 holdout integrity](/mnt/c/Users/joshd/canompx3/docs/audit/certificates/2026-04-21-lane-verdict-mnq-nyse-open-e2-rr1p0-cb1-cost-lt12/G6-holdout-integrity.md)
- [G8 mechanism statement](/mnt/c/Users/joshd/canompx3/docs/audit/certificates/2026-04-21-lane-verdict-mnq-nyse-open-e2-rr1p0-cb1-cost-lt12/G8-mechanism-statement.md)

## Non-applicable / inherited gate records

| Gate | Record |
|---|---|
| G2 MinBTL | Not separately re-run here; Phase A A3 already recorded `n_trials` and the lane failed the current operational discovery-budget ceilings. |
| G5 smell test | Not triggered; `|t|` is below 7. |
| G7 negative controls | Not applicable in Phase B because this is not a promotion to shadow/live. |
| G9 kill criteria | Not available; grandfathered deployed lane without a fresh Phase B pre-reg. |
| G10 pre-reg commit pin | Not available; no Phase B pre-reg file exists for this retrospective verdict. |

## Decision rationale

- Criterion 12 live SR monitor is ALARM in the binding Phase A snapshot.

## Final decision

- [x] `PAUSE-PENDING-REVIEW`

This verdict cites the binding Phase A snapshot directly and does not reopen discovery. Any future `KEEP` case would require a new clean holdout lineage and a gate set that clears without fail-closed exceptions.

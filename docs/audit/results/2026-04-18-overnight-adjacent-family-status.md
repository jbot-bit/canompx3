# Overnight Adjacent-Family Status — 2026-04-18

**Scope:** silent-test check per user directive. Not a permission to test; a verdict on whether the family is under-explored.

---

## OVNRNG_100 on MNQ NYSE_OPEN + US_DATA_1000

**Verdict: NOT under-explored. Already tested and rejected through the validator pipeline.**

### Evidence from `experimental_strategies`

| Instrument | Session | RR | p-value (BH-FDR adj) | IS ExpR | validation_status | rejection_reason |
|---|---|---|---:|---:|---|---|
| MNQ | NYSE_OPEN | 1.0 | 0.133 (0.203) | +0.064 | **REJECTED** | Phase 3: 3/6 years positive (<75%), unwaived neg: 2021, 2022, 2023 |
| MNQ | NYSE_OPEN | 1.5 | 0.272 (0.363) | +0.058 | **REJECTED** | Phase 3: 4/6 years positive (<75%), unwaived neg: 2022, 2023 |
| MNQ | US_DATA_1000 | 1.0 | **0.0046 (0.015)** | +0.118 | **REJECTED** | criterion_8: **OOS ExpR = −0.0088** (N_oos=61) |
| MNQ | US_DATA_1000 | 1.5 | 0.046 (0.082) | +0.105 | **REJECTED** | criterion_8: **OOS ExpR = −0.1067** (N_oos=60) |

All 4 cells tested at O5 E2 CB=1 stop_mult=1.0. Scan run 2026-04-11 under `n_trials_at_discovery=35616` (Mode B era).

### Failure mode by session

- **NYSE_OPEN:** fails at IS significance stage. Raw p > 0.10 on both RR cells. Also fails Phase 3 year-stability gate. Not a rescue candidate — no IS lift even before OOS is considered.
- **US_DATA_1000:** CLEARS IS (p=0.0046 at RR1.0 is strong), FAILS Criterion 8. OOS ExpR went negative on both RR cells (−0.009 and −0.107) against positive IS (+0.118 / +0.105). **Same OOS-direction-flip archetype as VWAP_BP CME_PRECLOSE (Task 1 this session) and WIDE_REL CME_PRECLOSE (Task 2 this session).**

### Queue implication

- **My earlier hostile-scout queue had OVNRNG_100 extension to NYSE_OPEN/US_DATA_1000 as candidate #2 by research EV.** That ranking was wrong. I did not check `experimental_strategies` before ranking; had I done so, this candidate would have been de-prioritized. Correction issued here.
- **No new pre-reg proposed.** Both sessions are legitimately closed for OVNRNG_100 at O5 RR1.0/1.5 under the project's locked validation criteria. Reopening would require either new data horizon (i.e., more OOS accumulating and not flipping) OR a new mechanism case (not the same OVNRNG_100 signal).
- **O15 aperture was NOT tested** in experimental_strategies — the 4-cell search was O5 only. This is a genuine untested cell for this family, but given the OOS-flip pattern on O5 US_DATA_1000, O15 extension is very low-EV — the signal already didn't transfer forward on O5 where it looked strongest. A Stage 2-style aperture swap is a salvage move, not a real research question.
- **RR2.0 was NOT tested** across the entire OVNRNG_100 family in this scan. Currently-deployed OVNRNG_100 lanes are all RR1.0/1.5. If RR2.0 shows something different on the 5 deployed lanes, that's mildly interesting but outside the user's declared scope for this overnight pass.

### Closure

Status: **tested, rejected, closed for current holdout regime.** Not silent-dropped. Not under-explored at O5. Not recommended for a new pre-reg.

OVNRNG_100 family is confined to its 5 current deployed lanes (MNQ COMEX_SETTLE + EUROPE_FLOW at RR1.0/1.5, plus COMEX_SETTLE_NO_FRI if exists). That is the full healthy footprint.

---

## Queue correction

Updated research queue after this overnight pass:

| Rank (updated) | Candidate | Status |
|---:|---|---|
| 1 | VWAP_BP_ALIGNED on MNQ CME_PRECLOSE (C8 re-check) | **DEAD** (OOS direction flip, Task 1) |
| 2 | ~~OVNRNG_100 on NYSE_OPEN/US_DATA_1000~~ | **CLOSED** (already tested, this result MD) |
| 3 | MES + MNQ wide-relative-IB tight pre-reg | **PENDING** — hypothesis + design committed (Task 2), replay awaiting approval |
| 4 | VWAP_BP on MNQ US_DATA_1000 (family audit K=6) | UNCHANGED from earlier queue — lower priority than #3 |
| 5 | Cross-session momentum NYSE_OPEN ← US_DATA_1000 prior | UNCHANGED |

Net: the original 5-item queue collapses to 3 live items (#3 WIDE_REL, #4 VWAP_BP US_DATA_1000, #5 cross-session momentum). #3 is now the only item with new artifacts this session.

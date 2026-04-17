# Overnight Adjacent-Family Status — 2026-04-18 (v2)

**Scope:** silent-test check only. No testing authorized.
**Format:** query `experimental_strategies` for target (instrument, session, filter) combos, report verdict state (tested+failed / tested+not-promoted / never-tested).

---

## OVNRNG_100 on MNQ NYSE_OPEN and US_DATA_1000

**Verdict: TESTED AND REJECTED through the validator pipeline. Not silent-dropped, not under-explored at O5.**

### Evidence

Query against `experimental_strategies` joining the MNQ × {NYSE_OPEN, US_DATA_1000} × OVNRNG_100 × O5 × E2 × CB=1 × stop_mult=1.0 × RR{1.0, 1.5} search space. Scan originator: run 2026-04-11 (Mode B era; `n_trials_at_discovery = 35616`).

| Session | RR | p (raw) | p (BH-adjusted) | IS ExpR | validation_status | rejection_reason |
|---|---:|---:|---:|---:|---|---|
| NYSE_OPEN | 1.0 | 0.133 | 0.203 | +0.064 | REJECTED | Phase 3: 3/6 years positive (<75% threshold); unwaived neg: 2021, 2022, 2023 |
| NYSE_OPEN | 1.5 | 0.272 | 0.363 | +0.058 | REJECTED | Phase 3: 4/6 years positive (<75%); unwaived neg: 2022, 2023 |
| US_DATA_1000 | 1.0 | 0.0046 | 0.0148 | +0.118 | REJECTED | criterion_8: OOS ExpR = −0.0088 (N_oos=61) |
| US_DATA_1000 | 1.5 | 0.046 | 0.082 | +0.105 | REJECTED | criterion_8: OOS ExpR = −0.107 (N_oos=60) |

### Failure mode summary

- **NYSE_OPEN at RR 1.0 and 1.5:** fails at the IS stage — raw p > 0.10, Phase 3 year-stability fail. Not a rescue candidate.
- **US_DATA_1000 at RR 1.0:** clears IS significance (BH-adjusted p = 0.015) but fails Amendment 2.7 Criterion 8 with OOS ExpR = −0.009 (N_oos=61 is adequate sample). OOS direction has flipped.
- **US_DATA_1000 at RR 1.5:** IS marginal (BH-adjusted p = 0.082); Criterion 8 fail with OOS ExpR = −0.107 (N_oos=60).

### What was NOT tested in this scan

- O15 aperture: OVNRNG_100 scan covered O5 only (this run)
- RR 2.0: absent from the tested cells across both sessions
- Stop multiplier variations other than 1.0

These represent parameter combinations not tested, but given:
- US_DATA_1000 O5 RR1.0 already showed OOS direction flip at N_oos=61 (not a thin-sample issue)
- Proposing a Stage 2-style aperture or RR swap to rescue would be parameter-search rescue, not a new mechanism test

This is legitimately closed for reopening without new mechanism evidence. Not a "never properly tested" slot.

### Queue correction

My earlier hostile-scout candidate list had `OVNRNG_100 extension to NYSE_OPEN and US_DATA_1000` ranked #2 by research EV. That ranking was wrong — had I queried `experimental_strategies` before ranking, this candidate would have been de-prioritized. Correction issued here.

**No new pre-reg proposed for this family.** The 5 current OVNRNG_100 deployed lanes (MNQ COMEX_SETTLE O5 RR1.0/1.5 + MNQ EUROPE_FLOW O5 RR1.0/1.5) appear to be the full healthy footprint at current holdout regime.

---

## Pattern note (bounded observation, not a claim)

Three independent filter families have shown IS-positive → OOS-flip on MNQ CME_PRECLOSE or MNQ US_DATA_1000 sessions this session:
- VWAP_BP_ALIGNED on CME_PRECLOSE (Task 1 of this overnight pass: DEAD)
- WIDE_REL_1.0X × G5 on CME_PRECLOSE (Task 2 distinctness audit prior-branch OOS peek: negative)
- OVNRNG_100 on US_DATA_1000 (this task: REJECTED at C8)

This is a bounded observation, not a tested hypothesis. N=3 is too small for regime-break inference. Flagged here for future observation only; **not proposing a test or doctrine update.**

---

## Updated research queue (as of 2026-04-18)

| Rank | Candidate | Status after this overnight pass |
|---:|---|---|
| 1 | VWAP_BP_ALIGNED CME_PRECLOSE C8 re-check | **DEAD** (Task 1, under Amendment 2.7) |
| 2 | ~~OVNRNG_100 NYSE_OPEN / US_DATA_1000~~ | **CLOSED** (this task: already tested, rejected) |
| 3 | MNQ WIDE_REL_1.0X_20D × G5 pre-reg v2 | **PENDING** — Task 2 v2 distinctness PASSES; pre-reg locked at K=6 (MNQ CME_PRECLOSE + MNQ TOKYO_OPEN × 3 RR); replay awaiting authorization |
| 4 | VWAP_BP on MNQ US_DATA_1000 (K=6 family audit) | **UNCHANGED** — lower priority than #3 |
| 5 | Cross-session momentum NYSE_OPEN ← US_DATA_1000 prior | **UNCHANGED** — not scouted this pass |

Net: 3 live candidates (#3, #4, #5). Only #3 has committed artifacts this pass.

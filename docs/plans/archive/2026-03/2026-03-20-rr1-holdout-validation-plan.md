---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Plan: RR1.0 Clean Holdout Validation

**Date:** 2026-03-20
**Status:** PLAN (not yet executed)
**Pre-registration:** `docs/pre-registrations/2026-03-20-rr1-economic-reasoning.md` (git-committed before this plan)

## Purpose

Structured reanalysis of 2025 data for the pre-registered RR1.0 hypothesis.
Get the honest number of tradeable setups with correct multiple testing.

## Relationship to Other Documents

- **This plan** governs a FRESH reanalysis of 2025 with honest N and pre-specified RR1.0
- **`2026-03-20-mnq-rr1-verified-sessions.md`** governs the 2026 FORWARD holdout (N=3, frozen strategies)
- These are INDEPENDENT analyses. This plan does NOT replace the 2026 holdout.

## Limitations (HONEST)

2025 is a **contaminated holdout.** 55+ tests touched 2025 data before this plan.
The BH FDR correction addresses multiplicity but NOT information leakage — we already
know which sessions look good on 2025. Results from this analysis are **supporting
evidence**, not clean discovery. The binding discovery test is the 2026 holdout
from the verified sessions pre-registration (N=3).

## Phases

### Phase 1: Data Preparation (no holdout values)

**Tasks:**

1. **Count exact N:** How many instrument x session combos have >= 50 trades in 2025?
   - Instruments: MNQ, MGC, MES
   - Sessions: all available per instrument
   - Filter: E2, CB1, RR1.0, O5, outcome IS NOT NULL, pnl_r IS NOT NULL
   - **N = ALL qualifying combos regardless of training performance**
   - This is the BH FDR denominator — no deflation from training filters

2. **Verify pnl_r units (FULL distribution, not sample):**
   - Compute: min, max, mean, median, P5, P95 of pnl_r across ALL rows
   - For RR1.0 E2: wins should be ~+0.7 to +1.0 (target hit minus costs), losses ~-1.0 to -1.1
   - If ANY pnl_r > 2.0 or < -2.0: investigate — something is wrong
   - Reproduce the +138R error from the previous broken query and diagnose exactly what went wrong
   - **STOP if pnl_r units are wrong. Do not proceed to Phase 2.**

3. **Compute per-instrument dollar risk:**
   - avg_risk_dollars = mean(|entry_price - stop_price|) x point_value
   - MNQ: point_value = $2.00
   - MGC: point_value = $10.00
   - MES: point_value = $5.00
   - Cost model: MNQ = $2.74 RT (canonical from pipeline.cost_model.COST_SPECS)

### Phase 2: Training Analysis (pre-2025 data only)

**Tasks:**

4. **For each of the N combos from Phase 1:**
   - Compute on pre-2025 data: N_train, mean_pnl_r, t-test p-value, std
   - Compute yearly consistency: % of pre-2025 years that are positive
   - Report ALL combos (positive and negative) — do not filter yet

5. **Flag training-positive combos** (for reporting only):
   - Positive mean AND >= 60% yearly consistency = "training-positive"
   - **This does NOT reduce N for BH FDR.** N stays at the Phase 1 count.
   - The training filter is for INTERPRETATION, not for penalty reduction.

### Phase 3: Holdout Test (2025 data — ONE PASS, NO ITERATION)

**Tasks:**

6. **For ALL N combos (not just training-positive):**
   - Compute: N_holdout, mean_pnl_r, t-test p-value
   - BH FDR at alpha=0.05 across ALL N tests
   - Report: which survive BH FDR?
   - Cross-reference with training-positive flag from Phase 2

7. **For each BH FDR survivor that is ALSO training-positive:**
   - Compute dollar estimate: ExpR x avg_risk_dollars x N_trades_per_year
   - Compute cost sensitivity: ExpR at 1.5x and 2x cost model
   - Report yearly breakdown on ALL data (training + holdout)

### Phase 4: Verification

**Tasks:**

8. **Fresh agent audit:**
   - Send complete methodology + all numbers to fresh agent (no project context)
   - Include: N, BH FDR thresholds, all p-values, training performance, holdout performance
   - Ask: is the math correct? Are there errors? Is the methodology valid?
   - **MANDATORY — do not skip.**

9. **Save results:**
   - Update memory with honest state
   - Commit and push

### Phase 5: Forward Registration

**Tasks:**

10. **Update verified sessions pre-registration:**
    - Add any NEW survivors (beyond NYSE_OPEN + COMEX_SETTLE) to the 2026 holdout list
    - Adjust N for 2026 BH FDR accordingly
    - Freeze parameters
    - Declare 2026 data remains sacred

## Key Guardrails

- pnl_r MUST be verified as R multiples before any dollar calculation (Phase 1 gate)
- N for BH FDR = ALL qualifying combos, NOT reduced by training filter (Fix #2)
- The holdout test (Phase 3) runs ONCE — no iteration
- 2025 is contaminated — results are supporting evidence, not discovery (Fix #3)
- Fresh agent audit (Phase 4) is MANDATORY
- This analysis is independent of the N=55 analysis and the N=3 forward test (Fix #1, #6)

## Estimated Time

- Phase 1: 15 minutes (data checks + pnl_r verification)
- Phase 2: 15 minutes (training analysis)
- Phase 3: 10 minutes (one pass holdout)
- Phase 4: 20 minutes (fresh agent)
- Phase 5: 10 minutes (documentation)
- Total: ~70 minutes

## Blast Radius

Zero production code changes. This is pure analysis on existing data.
Only outputs: updated docs, memory, pre-registration results.

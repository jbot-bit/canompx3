# Plan: RR1.0 Clean Holdout Validation

**Date:** 2026-03-20
**Status:** PLAN (not yet executed)
**Pre-registration:** `docs/pre-registrations/2026-03-20-rr1-economic-reasoning.md` (git-committed before this plan)

## Purpose

Test the pre-registered hypothesis (ORB breakout at RR1.0 has positive ExpR)
across ALL instruments and sessions using a clean train/holdout split.
Get the honest number of tradeable setups.

## Why This Matters

Previous approaches conflated discovery and validation:
- Grid search (120K tests) → 0.52R noise floor → killed everything
- Other terminal tested 55 things on 2025 holdout → N=55 → only 2 survived
- This plan: pre-specify RR1.0, test ALL instrument×session combos, BH FDR at exact N

The pre-registration is committed. RR1.0 is frozen from economic reasoning, not scanning.

## Phases

### Phase 1: Data Preparation (no holdout contact)

**Tasks:**
1. Count exact N: how many instrument × session combos have >= 50 trades in 2025?
   - Instruments: MNQ, MGC, MES
   - Sessions: all 12 per instrument
   - Filter: E2, CB1, RR1.0, O5, outcome IS NOT NULL, pnl_r IS NOT NULL
   - This count = N for BH FDR denominator
   - NOTE: counting trades in 2025 is NOT touching the holdout — it's just measuring sample size

2. Verify pnl_r is correct:
   - pnl_r should be in R multiples (win ≈ +0.7 to +1.0 after costs, loss ≈ -1.0 to -1.1)
   - Check a sample of rows: does pnl_r = (exit_price - entry_price) / (entry_price - stop_price)?
   - If pnl_r values are in dollars or points instead of R, the entire analysis is wrong
   - THIS IS WHY THE LAST QUERY PRODUCED NONSENSE — need to verify units first

3. Compute per-session dollar risk:
   - avg_risk_dollars = mean(|entry_price - stop_price|) × point_value per instrument
   - This converts ExpR to $/trade: dollar_per_trade = ExpR × avg_risk_dollars

### Phase 2: Training Analysis (pre-2025 data only)

**Tasks:**
4. For each instrument × session combo:
   - Compute: N_train, mean_pnl_r, t-test p-value, std
   - Compute yearly consistency: what % of years are positive?
   - Requirement: >= 60% of pre-2025 years positive to proceed to holdout

5. Apply training filter:
   - Only combos that are positive AND >= 60% yearly consistency on training data
   - proceed to holdout test
   - This reduces N further (good — fewer tests = lower BH FDR penalty)

### Phase 3: Holdout Test (2025 data — ONE PASS, NO ITERATION)

**Tasks:**
6. For ONLY the Phase 2 survivors:
   - Compute: N_holdout, mean_pnl_r, t-test p-value
   - BH FDR at alpha=0.05 across the EXACT number of holdout tests
   - Report: which survive BH FDR?

7. For each BH FDR survivor:
   - Compute dollar estimate: ExpR × avg_risk_dollars × N_trades_per_year
   - Compute cost sensitivity: what ExpR at 2-tick and 3-tick slippage?
   - Report yearly breakdown (including 2025 breakdown by quarter if possible)

### Phase 4: Verification

**Tasks:**
8. Fresh agent audit:
   - Send the COMPLETE methodology + results to a fresh agent
   - Ask it to verify the math, check for errors, assess statistical validity
   - No project context — just the numbers and the method

9. Save results:
   - Update pre-registration doc with results
   - Update memory
   - Commit and push

### Phase 5: Forward Registration

**Tasks:**
10. Pre-register survivors for 2026 holdout:
    - Freeze exact parameters
    - Declare 2026 data sacred
    - Set kill criteria for paper trading

## Key Guardrails

- pnl_r MUST be verified as R multiples before any dollar calculation
- The holdout test (Phase 3) runs ONCE — no iteration, no "let me check one more thing"
- Phase 2 training filter reduces N BEFORE touching holdout — this is the honest way
- If Phase 1 reveals pnl_r is in wrong units, STOP and fix before proceeding
- Fresh agent audit (Phase 4) is MANDATORY, not optional

## Estimated Time

- Phase 1: 15 minutes (data checks)
- Phase 2: 15 minutes (training analysis)
- Phase 3: 10 minutes (one pass holdout)
- Phase 4: 20 minutes (fresh agent)
- Phase 5: 10 minutes (documentation)
- Total: ~70 minutes

## Blast Radius

Zero production code changes. This is pure analysis on existing data.
Only outputs: updated docs, memory, pre-registration results.

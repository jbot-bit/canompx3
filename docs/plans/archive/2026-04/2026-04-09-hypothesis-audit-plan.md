---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Phase 4 Hypothesis Audit Plan — Post-Low-Effort Corrections

**Date:** 2026-04-09
**Author:** Claude Code session (high-effort audit)
**Trigger:** Low-effort session on Apr 9 produced 3 hypothesis YAMLs + Amendment 2.9 that need corrections before any discovery run can use them.

## Summary of Findings

### F1: MGC Bailey year metric WRONG (CRITICAL)
- MGC hypothesis uses trading years (1010÷252=4.01yr) while MNQ/MES use calendar years (6.65yr)
- Pipeline annualizes Sharpe with `span_days/365.25` (calendar years) → Bailey T must be calendar years
- MGC calendar T = 3.55yr → N≤5 (was N=7, MinBTL at N=7 = 3.89yr > 3.55yr → FAIL)
- Day count also wrong: claims 1,010, actual DB shows 1,039 (irrelevant since using calendar years)

### F2: RR=2.0 "Crabel canonical" claim UNVERIFIED
- Crabel book not in `resources/`. Claim from training memory.
- Prior session flagged "Crabel used EOD exit, not fixed RR"
- RR=2.0 is practitioner convention, not directly from Crabel text
- Doesn't invalidate hypotheses, but justification text should be honest

### F3: Amendment version ordering (COSMETIC)
- v2.9 appears before v2.8 in changelog table (same commit)

### F4: DST/timezone handling — VERIFIED CORRECT
- Dynamic resolvers in `pipeline/dst.py`, fail-closed guards
- Sunday trading days: legitimate artifact of CME reopen during US DST
  - Only CME_REOPEN fires on Sundays (08:00 Brisbane, before 09:00 cutoff)
  - All hypothesis sessions (LONDON_METALS, COMEX_SETTLE, US_DATA_830, NYSE_OPEN, EUROPE_FLOW) have NULL on Sunday trading days
  - Zero impact on any hypothesis being tested
  - Does inflate daily_features trading day count (292/yr vs 252/yr) but this is NOT used for Bailey — calendar years is

### VERIFIED CORRECT (no action needed)
- FK-safe DELETE logic in strategy_discovery.py
- All filter types exist in ALL_FILTERS and route correctly per session
- MNQ/MES Bailey compliance (N=16 at T=6.66yr, headroom 20%)
- Amendment 2.9 proxy data policy logic

---

## Blast Radius

| File | Change type | Downstream impact |
|---|---|---|
| `docs/audit/hypotheses/2026-04-09-mgc-mode-a-rediscovery.yaml` | Reduce N from 7→5, fix year metric | Any future MGC discovery run must use this corrected file |
| `docs/audit/hypotheses/2026-04-09-mnq-mode-a-rediscovery.yaml` | RR justification text only | None — no scope/numeric changes |
| `docs/audit/hypotheses/2026-04-09-mes-mode-a-rediscovery.yaml` | RR justification text only | None — no scope/numeric changes |
| `docs/institutional/pre_registered_criteria.md` | Reorder v2.8/v2.9 rows | None — cosmetic |
| `docs/plans/2026-04-09-phase4-hypothesis-redesign-findings.md` | Mark findings as resolved | None — reference doc |

**Zero production code changes. Zero test changes. Zero blast radius to pipeline or trading_app.**

---

## Stages

### Stage 1: Fix MGC hypothesis YAML

**File:** `docs/audit/hypotheses/2026-04-09-mgc-mode-a-rediscovery.yaml`

**Changes:**
1. Header: Replace "4.01 years" / "1,010 pre-holdout trading days" with:
   - "3.55 calendar years (2022-06-13 → 2025-12-31 = 1,298 calendar days ÷ 365.25)"
   - Add note: "Year metric: calendar years, matching pipeline Sharpe annualization (strategy_discovery.py line 588: years_span = span_days / 365.25). Bailey MinBTL T must use the same time unit as the Sharpe ratio. Trading years (1,039 trading days ÷ 252 = 4.12yr) is the WRONG metric for this pipeline because the pipeline does not use √252 annualization."
2. Bailey compliance block: N=7→5, MinBTL 3.89→3.22yr, headroom 3.0%→10.4%, noise floor 0.985→0.952
3. Remove H5 (G6 LONDON_METALS) and H6 (G5 COMEX_SETTLE) — threshold variants of H1/H2
4. Renumber H7→H5
5. Update budget_check: total trials 7→5, max_n 7→5
6. Update all comments that reference "7 hypotheses"
7. Session exclusion comment: add "Year metric" paragraph explaining calendar vs trading

**Acceptance:** `grep -c 'expected_trial_count: 1' <file>` = 5. All "N=7" references replaced. MinBTL = 2·ln(5) = 3.22 < 3.55.

### Stage 2: Fix RR justification in all 3 hypothesis YAMLs

**Files:** All 3 hypothesis YAMLs

**Change:** In every occurrence of "RR=2.0 (Crabel's canonical continuation target)" or similar, replace with:
"RR=2.0 (practitioner-standard breakout continuation target; Crabel 1990 studied opening-range breakouts with end-of-day exits, and RR=2.0 is the dominant intraday practitioner convention. Pre-committed to eliminate within-hypothesis parameter optimization per LdP 2020 §1.4.2.)"

Also update the header comment block where it says "Pre-committing to RR=2.0 (Crabel's canonical breakout RR)".

**Acceptance:** `grep -c "Crabel's canonical" <file>` = 0 for all 3 files.

### Stage 3: Fix Amendment version ordering

**File:** `docs/institutional/pre_registered_criteria.md`

**Change:** Swap the v2.9 and v2.8 rows in the changelog table so they appear in chronological order (v2.8 first, then v2.9).

**Acceptance:** v2.8 row appears before v2.9 row.

### Stage 4: Update findings doc + commit

**File:** `docs/plans/2026-04-09-phase4-hypothesis-redesign-findings.md`

**Changes:**
1. Mark F1 (year metric) as FIXED with description
2. Mark F3 (RR justification) as FIXED
3. Mark F5 (amendment ordering) as FIXED
4. Update filter count from 67 to 82
5. Mark DST/timezone as VERIFIED CORRECT
6. Mark regime infrastructure and filter universe claims as VERIFIED
7. Note: deployed portfolio gap (F9) remains a strategic decision for future work

Then: `python pipeline/check_drift.py` to verify all checks pass. Commit.

**Acceptance:** Drift checks pass. All findings either FIXED or VERIFIED.

---

## What this plan does NOT do (out of scope)

- **Deployed portfolio gap** — the 5 deployed lanes have zero Mode A evidence. This is a strategic decision about whether to run additional hypothesis files for COST_LT/SINGAPORE/TOKYO/COMEX/CME_REOPEN, not a bug fix.
- **Delete NQ/ES bars** — per Amendment 2.9, deferred until separate user confirmation
- **Holdout enforcement sweep** — separate workstream in memory
- **check_drift SQL keyword expansion audit** — the 34 new keywords were added to prevent false positives; verifying with `check_drift.py` in Stage 4 is sufficient

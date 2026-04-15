# Phase C — E_RETEST entry model (PRE-REG STUB)

**Status:** NOT EXECUTED. Pre-registration stub only, for future session.
**Date authored:** 2026-04-15
**Companion:** `docs/audit/hypotheses/2026-04-15-comprehensive-deployed-lane-gap-audit.md` §5.6

## Hypothesis

H1: For ORB sessions where price is near a key prior-day level at ORB end,
entering on pullback-to-level (limit order) AFTER an initial break-and-fail
produces higher ExpR than stop-market entry at first break (current E2).

Mechanism: Initial break at NEAR-LEVEL conditions runs into institutional
defense → rejection → pullback into the level → rejection confirms →
entry with the rejection direction (which is OPPOSITE of initial break).

Literature grounding: Chan Ch 1-2 on mean-reversion at known levels;
Dalton (NOT in `resources/` — acquisition pending) on market-profile
level-auction dynamics. See companion gap audit §6 D2.

## Scope (when executed)

- Instruments: MNQ + MES + MGC
- Sessions: 6 deployed lanes as primary targets
- Entry model: NEW `E_RETEST` — placed limit order at level AFTER initial
  break fails (either initial stop-out or MFE/MAE reversal beyond threshold)
- Apertures: O5 primarily (match deployed)
- RR: 1.0, 1.5, 2.0 (match deployed + halves)
- Stops: 0.5x ATR, 1.0x ATR (half of E2 current)
- Targets: fixed RR (comparison to E2) + next-level (dynamic)

## Comparison baseline

E2 on same (instrument, session, apt, RR) — apples-to-apples comparison.

## Pre-registered criteria (must pass before promotion to live)

Per `docs/institutional/pre_registered_criteria.md` — all 12 criteria.
Additional E_RETEST-specific:

1. Per-level N_entries ≥ 100 (entries are conditional on initial break fail)
2. ExpR_E_RETEST / ExpR_E2 ≥ 1.20 on matched universe (20% uplift minimum)
3. No contamination of OOS window via fit-parameter tuning (pre-commit 0.5x ATR, 1.0x ATR only; no mid-run relaxation)
4. Drawdown profile ≤ 1.5× E2 baseline (tightness of stops shouldn't worsen DD)

## Infrastructure required (DEFERRED work)

1. New entry model in `pipeline/outcome_builder.py` — simulates limit-on-retest entries in the minute bars, with `initial_break_failed` trigger condition
2. Schema extension: `entry_model = 'E_RETEST'` in orb_outcomes, with new columns for `retest_fill_ts`, `initial_break_stopout_ts`, etc.
3. New filter family: `LEVEL_PROXIMITY_F1-F8` applies at session start to determine eligibility for E_RETEST (not all days qualify)
4. Backtest rebuild for affected cells

## Estimated timeline

- 2-3 days: outcome_builder entry-model code
- 1-2 days: schema migration + data rebuild  
- 1 day: validation runs
- 1 week signal-only live shadow before any promotion

## Kill criteria

- If E_RETEST < E2 on any matched lane → abandon
- If infrastructure cost > 1 week without validation → reassess

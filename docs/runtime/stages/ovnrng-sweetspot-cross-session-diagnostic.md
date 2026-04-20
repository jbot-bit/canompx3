---
slug: ovnrng-sweetspot-cross-session-diagnostic
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Cross-session replication check of PR #47's ATR-normalized OVNRNG Q3-Q4 sweet-spot
---

# Stage: OVNRNG sweet-spot cross-session diagnostic

## Context

PR #47's correction MD (`docs/audit/results/2026-04-20-nyse-open-ovnrng-fast10-correction.md`)
observed on NYSE_OPEN E2 RR1.0 CB1 IS:

| ovn/atr quintile | bin mean | N | WR | ExpR |
|-----|--|----|----|-----|
| Q1 lo | 0.17 | 339 | 56.6% | +0.085 |
| Q2 | 0.24 | 338 | 55.9% | +0.078 |
| Q3 | 0.31 | 338 | 58.0% | +0.116 |
| Q4 | 0.40 | 338 | 57.4% | +0.107 |
| Q5 hi | 0.68 | 339 | 52.5% | +0.017 |

Finding: Q3-Q4 is the sweet-spot (+0.107–0.116R), Q5 HURTS (+0.017R).
Mechanism narrative: moderate overnight range = good breakout setup;
extreme overnight range = exhausted / range-day.

The handover queued a "cross-session scope (324+ cells) for honest DSR"
Pathway-B pre-reg. Before committing to that K=many scan, answer:
**does Q3-Q4 > Q5 pattern replicate on other sessions?**

If yes on ≥3 sessions → write the pre-reg. If yes on only 1-2 → the
finding is session-specific (NYSE_OPEN), pre-reg must be narrower.
If no on any session but NYSE_OPEN → PR #47's finding was a
single-session artifact.

## Scope Lock

- `research/audit_ovnrng_sweetspot_cross_session.py` (new)
- `docs/audit/results/2026-04-21-ovnrng-sweetspot-cross-session.md` (new)

## Blast Radius

- Read-only. Zero production-code touch.
- Canonical data: `orb_outcomes`, `daily_features`.
- No new filter registered. No pre-reg created. No config changes.

## Approach

1. For each of 12 canonical sessions (`pipeline.dst.SESSION_CATALOG`),
   load MNQ E2 RR=1.5 CB=1 orb_minutes=5 universe. Use RR=1.5 since
   5 of 6 DEPLOY lanes are RR=1.5; PR #47's NYSE_OPEN observation was
   RR=1.0 but we test the general pattern.
2. Filter to IS only (trading_day < 2026-01-01) to avoid OOS peeking.
3. Also filter to days where `overnight_range` feature is valid per
   `.claude/rules/backtesting-methodology.md` Rule 1.2 (look-ahead gate):
   ORB start time ≥ 17:00 Brisbane. Early sessions (CME_REOPEN 08:00,
   TOKYO_OPEN 10:00, SINGAPORE_OPEN 11:00) fire BEFORE overnight window
   closes — overnight_range uses FUTURE data. Those sessions MUST be
   skipped.
4. Compute `ovn_over_atr = overnight_range / atr_20` per trade.
5. Bin into quintiles per session. Report per-quintile N, WR, ExpR,
   and Q3+Q4 vs Q5 contrast.
6. Classify per-session: SWEET_SPOT_PRESENT (Q3+Q4 mean > Q5 mean + 0.05R),
   NO_PATTERN (within 0.05R), or INVERSE (Q5 > Q3+Q4 + 0.05R).

## Acceptance criteria

1. Script runs without exceptions.
2. MD reports per-session quintile table for all eligible sessions.
3. MD classifies each session and rolls up (count of SWEET_SPOT_PRESENT).
4. MD gives operational verdict: cross-session pre-reg WORTHWHILE /
   NARROW_PRE_REG / NO_PRE_REG.
5. `python pipeline/check_drift.py` passes.

## Non-goals

- Not writing the pre-reg itself (next turn if replication is strong).
- Not testing OOS (peeking avoided).
- Not evaluating BH-FDR K-adjusted p-values (diagnostic only, not a
  formal scan).
- Not adjusting for any filter — this is the raw ovn/atr signal on
  the unfiltered session geometry.

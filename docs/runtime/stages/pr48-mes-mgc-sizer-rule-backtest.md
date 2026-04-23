---
slug: pr48-mes-mgc-sizer-rule-backtest
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-23
updated: 2026-04-23
task: Execute the next bounded PR48 follow-through step by deriving a frozen rel-vol sizing rule for MES and MGC from IS data only, then replaying it on 2026 OOS without reopening the search space.
---

# Stage: PR48 MES/MGC sizer-rule backtest

## Question

PR48's participation-shape path has already cleared the first two honest steps:

- IS monotonic-up exists on `MNQ`, `MES`, and `MGC`
- OOS slope replication is now confirmed on `MES` and `MGC`

The open question is narrower:

> can that mechanism be translated into a concrete frozen sizing overlay for `MES` and `MGC`, and does the overlay still help when its thresholds are trained on IS and applied untouched to 2026 OOS?

## Scope Lock

- Instruments in scope: `MES`, `MGC`
- Geometry in scope: `5m`, `E2`, `CB1`, `RR1.5`, unfiltered baseline only
- Thresholds must be frozen from IS only (`trading_day < 2026-01-01`)
- OOS is replay only (`trading_day >= 2026-01-01`)
- Thresholds are per-lane, not pooled across sessions, because PR48's upstream mechanism was rank-within-lane
- Use one pre-registered monotonic sizing map only; no rule grid or optimizer

## Blast Radius

- Research-only:
  - one stage file under `docs/runtime/stages/`
  - one locked hypothesis file under `docs/audit/hypotheses/`
  - one runner under `research/`
  - one result doc under `docs/audit/results/`
  - frozen breakpoint artifacts under `research/output/`
- No writes to canonical layers
- No live-config, profile, or allocator writes

## Approach

1. Freeze one monotonic rel-vol sizing rule before execution.
2. Derive per-lane IS quintile boundaries from canonical data.
3. Apply the frozen rule to IS and OOS on the identical trade sets.
4. Report both:
   - actual take-home uplift (`size * pnl_r`)
   - normalized per-risk uplift (`sum(size * pnl_r) / sum(size)`)
5. Classify each instrument as:
   - `SIZER_DEPLOY_CANDIDATE`
   - `SIZER_ALIVE_NOT_READY`
   - `SIZER_NO_GO`

## Suggested Branch / PR

- Branch: `research/pr48-mes-mgc-sizer-rule`
- PR title: `research(pr48): derive and replay MES/MGC rel-vol sizing overlay`

## Acceptance Criteria

1. Only one frozen sizing map is tested.
2. All thresholds are trained on IS only and exported as durable artifacts.
3. OOS replay uses untouched frozen breakpoints.
4. Result doc distinguishes take-home uplift from mere leverage drift.
5. No live deployment language appears unless the replay actually clears the pre-registered gates.

## Non-goals

- Not a live activation
- Not a session-rescan or filtered-universe rescan
- Not a rule optimizer or curve-fit sweep
- Not a replacement for later account-budget / allocator integration

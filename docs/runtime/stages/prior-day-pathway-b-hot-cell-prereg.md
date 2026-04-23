---
slug: prior-day-pathway-b-hot-cell-prereg
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-22
updated: 2026-04-22
task: Freeze one narrow prereg for the strongest surviving Prior-day Pathway-B hot cell instead of reopening the broad family. Convert the action-queue placeholder into a single-cell confirm-or-kill stage.
---

# Stage: Prior-day Pathway-B hot-cell prereg

## Question

The post-stale-lock action queue still leaves the Prior-day branch at a vague
`PREREG NEXT` status. That is too loose. The honest open question is narrower:

> is there one strongest Prior-day Pathway-B hot cell worth freezing into a
> single-cell prereg, or should the branch be parked without another broad scan?

## Scope Lock

- Focus on one strongest existing hot cell only
- Stay within Prior-day Pathway-B framing already surfaced by the action queue
- No broad family rediscovery
- No multi-cell shopping after looking at outcomes
- No deployment claims in this stage

## Blast Radius

- Research-only:
  - one prereg or scope-note under `docs/audit/hypotheses/` or `docs/audit/results/`
  - one narrow runner under `research/` only if needed to support the frozen cell
- No writes to production registries or live config

## Approach

1. Read the canonical action-queue context and any upstream result docs for the
   Prior-day branch.
2. Identify whether one candidate cell already has enough upstream support to be
   frozen honestly.
3. End in one of two outcomes only:
   - write a single-cell prereg
   - park the branch with an explicit "no honest hot cell" result
4. If a prereg is written, keep the next stage strictly confirm-or-kill.

## Suggested Branch / PR

- Branch: `research/prior-day-pathway-b-hot-cell`
- PR title: `research(prior-day): freeze one Pathway-B hot-cell prereg`

## Acceptance Criteria

1. Exactly one cell or no cell is named.
2. The doc states why that cell was selected from upstream evidence.
3. No broad rescan language appears.
4. Output ends with a frozen next move or an explicit park.

## Non-goals

- Not a broad Prior-day family resurrection
- Not a multi-cell ranking sweep
- Not a deployment recommendation

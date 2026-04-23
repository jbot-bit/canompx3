---
slug: gc-mgc-15m-30m-translation-question
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-22
updated: 2026-04-22
task: Resolve whether there is any honest 15m or 30m GC->MGC translation claim worth testing after the 5m payoff-compression path. Keep it as a question-framing stage, not a broad gold rescan.
---

# Stage: GC->MGC 15m/30m translation question

## Question

The translation audit explicitly stopped short of making any honest statement
about wider apertures. That leaves one bounded backlog item:

> after the 5m payoff-compression question is settled, is there any coherent 15m
> or 30m GC->MGC translation claim worth testing, or should the wider-aperture
> path be closed?

## Scope Lock

- This is a question-framing stage only
- Apertures in scope: `O15`, `O30`
- Must treat the 5m payoff-compression result as an upstream prerequisite
- No broad gold proxy campaign
- No runner or scan in this stage unless needed to validate basic availability

## Blast Radius

- Docs-only unless a tiny availability check is required
- Output goes to `docs/plans/` or `docs/audit/results/`
- No production code changes

## Approach

1. Re-read the GC->MGC translation audit and the new 5m payoff-compression task.
2. State whether a wider-aperture question is:
   - `NOT_READY` until the 5m path resolves
   - `READY_FOR_PREREG`
   - `CLOSE_PATH`
3. If `READY_FOR_PREREG`, define the smallest honest future stage.

## Suggested Branch / PR

- Branch: `research/gc-mgc-15m-30m-question`
- PR title: `docs(gold): frame 15m/30m GC->MGC translation question`

## Acceptance Criteria

1. The doc explicitly depends on the 5m audit outcome.
2. It avoids any unstated assumption that wider apertures translate.
3. It ends with a clear state: `NOT_READY`, `READY_FOR_PREREG`, or `CLOSE_PATH`.
4. No broad gold rediscovery language appears.

## Non-goals

- Not a 15m/30m scan
- Not a new translation claim
- Not a deployment recommendation

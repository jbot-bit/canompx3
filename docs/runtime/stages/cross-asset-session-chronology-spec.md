---
slug: cross-asset-session-chronology-spec
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-22
updated: 2026-04-22
task: Write the narrow chronology spec for earlier-session to later-ORB quality transfer before any cross-asset scan is attempted. Turn the current vague idea into a bounded admissibility and timing document.
---

# Stage: Cross-asset session chronology spec

## Question

The recent audits leave one plausible but easy-to-fake idea open:

> can an earlier session convey useful quality information into a later ORB
> without violating chronology discipline or drifting into pooled narrative
> fishing?

Before any scan, the repo needs a chronology spec that defines what is known
when.

## Scope Lock

- Scope is documentation/specification only
- Focus on admissibility and chronology discipline
- No scan runner in this stage
- No pooled cross-asset predictor search
- No deployment framing

## Blast Radius

- Docs-only:
  - one spec under `docs/plans/` or `docs/audit/hypotheses/`
  - this stage file closes when the chronology contract is written
- No code changes unless a canonical timing definition is missing and must be
  documented

## Approach

1. Re-read the chronology-sensitive ORB and cross-session findings from the
   recent PR window.
2. Write a narrow spec that defines:
   - which earlier-session facts are admissible
   - the exact timestamp boundary for later-session entry
   - banned forms of hindsight leakage
   - what a future scan may and may not test
3. End with one recommendation:
   - `READY_FOR_PREREG`
   - `PARK`

## Suggested Branch / PR

- Branch: `research/cross-asset-chronology-spec`
- PR title: `docs(research): freeze cross-asset session chronology spec`

## Acceptance Criteria

1. The spec defines admissible vs banned chronology explicitly.
2. It is narrow enough that a future scan could be preregistered from it.
3. It does not claim signal exists; it only defines honest testability.
4. The branch ends docs-only.

## Non-goals

- Not a cross-asset discovery scan
- Not a new factor family
- Not a live-trading proposal

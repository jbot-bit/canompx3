# Research Pipeline Sync Decision

**Date:** 2026-04-23

## Decision

Keep the current two-branch research model and align the docs, skills, front
door wording, and tests around it:

- `experimental_strategies` is the extra standalone-candidate inventory.
- `validated_setups` is the validated research shelf for standalone lanes.
- Deployment is a later optional selection step, not part of validation.
- `paper_trades` are execution evidence only.
- Conditional-role studies use bounded runners, result docs, and explicit role
  contracts unless a prereg defines a complete standalone lane.

## Non-Decisions

- No new DB table or schema migration in this pass.
- No forced live-routing or `paper_trades` gate before research validation.
- No forced conversion of conditional-role findings into
  `experimental_strategies`.

## Operator Default

For natural-language requests like "find better setups in X" or "test whether X
is an allocator", agents should route internally through the discovery protocol
and prereg front door. The user should not need command syntax.

Research expansion should widen mechanism coverage while keeping each prereg
narrow, mechanism-backed, and honest about K / FDR / holdout accounting.

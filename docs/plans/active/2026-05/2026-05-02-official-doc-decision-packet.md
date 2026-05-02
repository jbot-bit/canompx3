---
status: active
owner: codex
last_reviewed: 2026-05-02
superseded_by: ""
---

# Official-Doc Decision Packet

## Purpose

This workstream exists to verify the external operator architecture against
official documentation before any platform-specific integration work proceeds.

This is a documentation and decision packet only. It does not authorize code
integration with Quantower, TopstepX, MotiveWave, Sierra, or any other external
surface.

## Required Inputs

Use only:

- repo-canonical code and docs
- official Topstep / TopstepX documentation
- official ProjectX API documentation
- official candidate-platform connection / replay / SDK documentation

Mark every conclusion as:

- `VERIFIED`
- `INFERRED`
- `NEEDS VERIFICATION`

## Questions To Answer

1. What is allowed in `Combine`, `XFA`, and `Live Funded`?
2. Does a read-only/assist sidecar remain clean under official policy text?
3. Does any candidate platform introduce a second truth surface for order or
   account state?
4. Can a candidate consume the canonical export contract without re-encoding
   the ORB/research brain?
5. Which assumptions remain unproven and must block implementation?

## Mandatory Output

The final packet must include:

- stage-split policy matrix
- candidate matrix
- unsupported assumptions register
- kill criteria
- explicit "implementation still blocked" line unless all blockers are cleared

## Non-Negotiable Rules

- No platform code in this worktree
- No scripting-layer migration plans as a first step
- No "likely allowed" claims without an official citation or explicit
  `NEEDS VERIFICATION` label
- No collapsing `Combine/XFA/Live` into one blended judgment

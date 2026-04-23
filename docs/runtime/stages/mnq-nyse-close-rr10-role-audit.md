---
slug: mnq-nyse-close-rr10-role-audit
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-23
updated: 2026-04-23
task: Resolve the correct role for the still-alive MNQ NYSE_CLOSE RR1.0 family after the exact ORB_G8 filter path was killed. Determine whether the remaining opportunity is standalone, allocator, or nothing at all.
---

# Stage: MNQ NYSE_CLOSE RR1.0 role audit

## Question

The NYSE_CLOSE branch now has a clear split:

- the broad RR1.0 family remains positive on canonical baselines
- the strongest locked native filter path (`ORB_G8`) was executed and killed

That leaves one bounded question:

> if the edge is real, what role does it actually belong in: standalone lane,
> portfolio / allocator inclusion, or nowhere?

## Scope Lock

- Instrument/session in scope: `MNQ NYSE_CLOSE`
- Primary surface: RR1.0
- Roles in scope:
  - standalone unfiltered session family
  - allocator / portfolio inclusion role
  - filter-role rejection summary
- No new filter search
- No direct live unblock in this stage

## Blast Radius

- Research-only:
  - one runner under `research/`
  - one result doc under `docs/audit/results/`
- No production portfolio changes
- No validated-setups writes

## Approach

1. Treat canonical baselines as truth and derived portfolio exclusions as comparison-only.
2. Re-state what has now been fairly tested:
   - broad baseline
   - narrow filter attempts
   - exact ORB_G8 path
3. Audit the remaining opportunity by role:
   - standalone edge
   - allocator contribution / diversification / session inclusion
   - implementation drag or blocker surface
4. End with one honest recommendation:
   - `CONTINUE as standalone/allocator audit`
   - `PARK`
   - `KILL`

## Suggested Branch / PR

- Branch: `research/mnq-nyse-close-rr10-role-audit`
- PR title: `research(nyse-close): audit rr1 role after orbg8 kill`

## Acceptance Criteria

1. The audit distinguishes role from filter.
2. It does not rescue `ORB_G8` post hoc.
3. It identifies whether broad NYSE_CLOSE still deserves research capital.
4. It ends with a single next move, not a menu of vague possibilities.

## Non-goals

- Not another filter prereg
- Not a raw deployment recommendation
- Not a broad cross-session rediscovery sweep

# Strict C11 readiness closeout

Date: 2026-06-03
Profile: `topstep_50k_mnq_auto`

## Decision

Strict Criterion 11 diagnostic failures are visible, non-launch-blocking warnings in the strict live-readiness report. The binding Criterion 11 gate remains the operational Monte Carlo survival gate: 90-day survival, at least 10,000 paths, conservative path model, current profile fingerprint, and survival probability at or above the preregistered threshold. Telemetry maturity remains advisory for express/funded profiles unless repository policy changes it into a gate.

## Implemented

- `scripts/tools/live_readiness_report.py` now treats strict C11 diagnostic warnings as visible advisory warnings, not generic launch blockers.
- Strict readiness `green` is false when a blocker or unclassified launch-blocking warning is present.
- Paused-lane reason summaries now preserve allocator `reason` values instead of collapsing them to `unspecified`.
- The plan-v2 proof report records DB-backed evidence, strict C11 failure details, C12 validity, telemetry maturity, and attribution retry outcome.

## Verification boundary

Current strict readiness can be green while C11 strict diagnostics fail on historical account-risk evidence. That warning is still capital-risk evidence and must be reviewed before increasing risk, but it is diagnostic-only unless repository policy promotes it into a hard gate.

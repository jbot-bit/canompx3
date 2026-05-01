---
task: workqueue-schema-drift
mode: TRIVIAL
phase: 1/1
created: 2026-05-01
scope_lock:
  - pipeline/work_queue.py
  - docs/runtime/stages/workqueue-schema-drift.md
---

# Stage: WorkQueue pydantic schema drift fix

## Blast Radius

`pipeline/work_queue.py:21-22` — extend `QueueClass` Literal to add `audit` and
`stage` (both currently used in `docs/runtime/action-queue.yaml`); extend
`QueueStatus` Literal to add `open`. These are values humans have been writing
into the queue YAML; the schema lagged.

Symptom: `scripts/tools/project_pulse.py` raises 3 ValidationErrors at startup
since 2026-05-01 04:06 UTC (commit `5454cde3`). All 9 `test_pulse_integration.py`
tests fail on CI because pulse exits non-zero, leaving empty stdout. Both `main`
and `fix/allocator-chordia-gate` (PR #197) are red on this — pre-existing on
main, inherited by the PR.

Downstream: pulse JSON consumers (the `/orient` skill), pulse-integration tests
(9 tests in `tests/test_tools/test_pulse_integration.py`). No production-code
behavior change — only the schema validator's enum membership widens.

No DB schema, no migration, no runtime trading-path touched. Truly trivial.

## Acceptance

- `python pipeline/work_queue.py` and `python scripts/tools/project_pulse.py
  --fast --format json` both run without ValidationError.
- 9 pulse-integration tests pass.
- `python pipeline/check_drift.py` passes.
- Resolves PR #197's CI red (which is the SAME schema bug, not a chordia-gate
  issue).

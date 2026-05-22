---
task: Fast Lane V2 Phase 1 trial provenance hardening — content-addressed trial IDs and runner-owned ledger appends.
mode: CLOSED
closed_commit: 9697650b
closed_date: 2026-05-22
closed_note: |
  Core content-addressed `trial_id` implementation landed in `78e383ff`.
  Runner-owned FAST_LANE trial append landed in `9697650b`. Stage already
  carried PROVEN acceptance and verification; this closeout adds parser-visible
  CLOSED frontmatter so stage-awareness no longer treats it as open work.
---

# Fast Lane V2 Phase 1 Trial Provenance Stage

**Date:** 2026-05-21
**Status:** PROVEN
**Authority:** `docs/plans/2026-05-21-fast-lane-v2-institutional-design.md`
**Risk tier:** critical research-control surface, capital-adjacent

## Goal

Make fast-lane trial history content-addressed and replay-safe so K-lineage counts real research executions only.

## Non-Goals

- No new strategy research runs.
- No heavyweight Chordia promotion.
- No `validated_setups` writes.
- No `docs/runtime/lane_allocation.json` writes.
- No `docs/runtime/chordia_audit_log.yaml` writes.
- No broker, live runtime, or profile-routing changes.

## Scope Lock

- `scripts/research/fast_lane_trial_ledger.py`
- `scripts/research/fast_lane_promote_queue.py`
- `pipeline/check_drift.py`
- `tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py`
- `tests/test_research/test_fast_lane_promote_queue_suppression.py`

## Design

Trial identity must be stable across replays of the same evidence:

```text
trial_id = sha256(prereg_sha + runner_id + result_artifact_sha + canonical_data_fingerprint)[:16]
```

`run_id` remains accepted as a legacy display/compatibility field, but it must not be the only replay guard. The ledger writer must refuse conflicting duplicate `trial_id` rows and treat exact duplicate `trial_id` appends as idempotent no-ops.

Scanners, status rebuilds, rankers, bridges, dashboards, and dry-runs are derived views. They must not append trial history.

## Edge Cases

- Same prereg/result/data replay with a different timestamp must not inflate K.
- Same `trial_id` with different prereg path, result hash, structural hash, or outcome must fail closed.
- Missing `trial_id` on legacy entries is tolerated for read compatibility but new writer appends must include it.
- Timestamp monotonicity still protects append order for genuinely new executions.
- Capital-class path refusal remains unchanged.
- Scanner write-cache mode remains read-only with respect to the trial ledger.

## Acceptance

- [x] Writer emits `trial_id` for new entries.
- [x] Writer idempotently skips exact duplicate `trial_id` replays.
- [x] Writer refuses duplicate `trial_id` with different content.
- [x] Drift check catches duplicate `trial_id` conflicts and malformed `trial_id`.
- [x] Scanner `--dry-run`, scanner `--write`, direct `scan()`, and walk dry-run do not mutate the ledger.
- [x] Legacy `scan(..., append_to_ledger=True)` is ignored so scanner code has no append authority.
- [x] Targeted tests pass.
- [x] `git diff --check` passes.

## Verification

- `./.venv-wsl/bin/python -m pytest tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py tests/test_research/test_fast_lane_promote_queue_suppression.py tests/test_tools/test_fast_lane_walk.py -q` -> 52 passed.
- `./.venv-wsl/bin/python -m ruff check --ignore SIM300 scripts/research/fast_lane_trial_ledger.py scripts/research/fast_lane_promote_queue.py pipeline/check_drift.py tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py tests/test_research/test_fast_lane_promote_queue_suppression.py` -> all checks passed.
- `git diff --check` -> clean.

`SIM300` is ignored only for `pipeline/check_drift.py` because the repo already carries an unrelated pre-existing Yoda-condition warning at `check_holdout_sentinel_inline_copy_parity`; this stage did not touch that line.

---
task: Fast Lane V2 Phase 1 completion + Phase 2 K semantics — exclude historical scanner rows from V2 K counts via correction file, and stop comparing trade-count N_hat to trial-count K_declared.
mode: CLOSED
closed_commit: bc605514
closed_date: 2026-05-22
closed_note: |
  Landed correction-not-deletion filtering for historical scanner rows, K-budget
  suppression semantics based on trial count, and rebuilt derived promote/status
  caches. Full drift passed with 0 blocking violations.
scope_lock:
  - scripts/research/fast_lane_trial_ledger.py
  - scripts/research/fast_lane_promote_queue.py
  - docs/runtime/fast_lane_trial_corrections.yaml
  - docs/specs/fast_lane_state_graph.md
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py
  - tests/test_research/test_fast_lane_promote_queue_suppression.py
---

## Blast Radius

- `scripts/research/fast_lane_trial_ledger.py` — add correction-file reader/filter helpers. Historical rows remain append-only; correction records only exclude selected rows from V2 K-count consumers.
- `scripts/research/fast_lane_promote_queue.py` — scanner reads corrected V2 trial rows for `K_global` / `K_family` / `K_lane`; scanner remains read-only over ledger/corrections.
- `docs/runtime/fast_lane_trial_corrections.yaml` — new documented correction record excluding historical `scanner-*` rows from V2 K counts. This is correction-over-deletion per the V2 design.
- `docs/specs/fast_lane_state_graph.md` — update `SUPPRESSED_K_OVERRUN` trigger language so K gates compare trial counts to trial budgets, not `N_hat` sample counts.
- `pipeline/check_drift.py` — extend Check #169 to validate the correction file shape and fail closed on malformed selectors/actions.
- Tests — add mutation tests proving corrected scanner rows do not inflate `K_lane`, budgeted K-lane repeats are allowed, and true K-budget overruns are trial-count based.
- Writes: docs/code/tests only. No `gold.db`, `validated_setups`, `lane_allocation.json`, `chordia_audit_log.yaml`, broker state, or live runtime state.

## Acceptance

1. Historical `scanner-*` rows are preserved in `docs/runtime/fast_lane_trial_ledger.yaml` but excluded from V2 K lineage by correction record.
2. Rebuilding the promote queue no longer reports `K_lane=33+` from scanner pollution.
3. `SUPPRESSED_K_OVERRUN` uses `K_lane > K_declared_in_prereg`; it does not compare `n_hat` to `K_declared`.
4. Targeted tests pass.
5. `pipeline/check_drift.py` passes with 0 blocking violations.

## Verification

- `./.venv-wsl/bin/python -m pytest tests/test_research/test_fast_lane_promote_queue_suppression.py tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py` -> 39 passed.
- `./.venv-wsl/bin/python -m ruff check scripts/research/fast_lane_trial_ledger.py scripts/research/fast_lane_promote_queue.py pipeline/check_drift.py tests/test_research/test_fast_lane_promote_queue_suppression.py tests/test_pipeline/test_check_drift_fast_lane_trial_ledger_append_only.py` -> all checks passed.
- `./.venv-wsl/bin/python scripts/research/fast_lane_promote_queue.py --write` -> rebuilt `docs/runtime/promote_queue.yaml`; both live entries now show `K_lane: 0`, `K_global: 0`.
- `./.venv-wsl/bin/python scripts/tools/fast_lane_status.py --write` -> rebuilt dependent status roll-up after promote queue status changed.
- `./.venv-wsl/bin/python pipeline/check_drift.py` -> 154 checks passed, 0 skipped, 20 advisory, 0 blocking violations.

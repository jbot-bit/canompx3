# Stage — Fast-Lane Status: pre-ranker heavyweight classification fix

**Slug:** `2026-05-20-fast-lane-status-pre-ranker-heavyweight-classification`
**Mode:** IMPLEMENTATION
**Branch:** `session/journal-enricher`
**Worktree:** `C:/Users/joshd/canompx3/.worktrees/journal-enricher`
**Date opened:** 2026-05-20

## task

Fix `fast_lane_status.yaml` misclassification of 38 HEAVYWEIGHT_COMPLETE entries that have no fast-lane lineage. Currently they all carry `next_action_token: run_cherry_pick_journal_enricher`, but the enricher is structurally update-only against journal entries the ranker writes — and these 38 never went through the ranker (heavyweight Chordia preregs were authored directly, predating the cherry-pick loop landing 2026-05-19).

The misclassification turns a structural gap (no fast-lane → heavyweight learning signal possible) into a phantom backlog. The fix is at the writer (`scripts/tools/fast_lane_status.py`), not by seeding 38 stub journal entries (which would pollute the journal's learning purpose).

## scope_lock

- `scripts/tools/fast_lane_status.py`
- `docs/specs/fast_lane_state_graph.md`
- `tests/test_pipeline/test_fast_lane_status_classification.py`

(Adding tests/test_pipeline path if not already in scope_lock allowlist; otherwise locate the existing companion test file via `tests/test_tools/test_fast_lane_status*.py` and substitute.)

## Blast Radius

- `scripts/tools/fast_lane_status.py` — modify `_classify_stage()` to distinguish HEAVYWEIGHT_COMPLETE with fast-lane lineage vs without; modify `_NEXT_ACTION_BY_STAGE` mapping path to apply the distinction.
- `docs/runtime/fast_lane_status.yaml` — derived state, will be rewritten by `--write`. 38 entries flip `next_action_token` from `run_cherry_pick_journal_enricher` to a new token. No hand edits.
- `docs/specs/fast_lane_state_graph.md` — § 5.1 lists the canonical stage enum and next-action mapping; amend to document the lineage qualifier.
- `pipeline/check_drift.py::check_fast_lane_status_rollup_reconstruction_parity` (Check #168) — reads `next_action_token` field; will re-derive symmetrically from the same writer change. Test the writer's reconstruction against the on-disk rollup.
- Tests: add 4 cases covering (a) HEAVYWEIGHT_COMPLETE with journal entry → enricher action; (b) HEAVYWEIGHT_COMPLETE without journal entry and without fast-lane MD → new action; (c) HEAVYWEIGHT_COMPLETE with fast-lane MD but no journal yet → enricher action; (d) ENRICHED unchanged.
- Reads: `docs/runtime/fast_lane_status.yaml`, `docs/runtime/cherry_pick_journal.yaml` (read-only). Writes: only the rollup file via existing `--write` path.

## Capital-class boundary

Zero capital-class file touched. No `chordia_audit_log.yaml`, no `lane_allocation.json`, no `validated_setups`, no `trading_app/live/`. Adversarial-audit gate not triggered (no `trading_app/live/`, no `risk_manager.py`, no `pipeline/` truth-layer mutation; `pipeline/check_drift.py` is touched only by reading the parity check, not editing it).

## Verification plan

1. Unit tests for `_classify_stage()` + `_NEXT_ACTION_BY_STAGE` lookup pass.
2. `python scripts/tools/fast_lane_status.py --write` regenerates `fast_lane_status.yaml`.
3. `python pipeline/check_drift.py` passes — Check #168 sees writer and rollup agree (both re-derive symmetrically since the writer is the canonical source).
4. Spot-check: query rebuilt rollup for the 38 entries; confirm `next_action_token` changed to the new value for those without journal lineage, unchanged for any with journal lineage.
5. Spot-check: `python scripts/research/cherry_pick_journal_enricher.py --dry-run` still says "No entries to enrich" (no new journal entries were created).
6. `python scripts/tools/fast_lane_walk.py` orchestrator step reports unchanged on the chain-execution side; only the next-action footer changes.

## Done criteria

- Tests pass (show output)
- Dead code swept (`grep -r run_cherry_pick_journal_enricher` confirms only the new conditional-mapping path emits this token)
- `python pipeline/check_drift.py` passes
- Self-review pass: confirm no journal mutation, no capital-class touch
- State graph doc § 5.1 amended to document the lineage qualifier

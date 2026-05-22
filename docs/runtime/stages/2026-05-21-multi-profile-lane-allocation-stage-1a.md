---
task: Multi-profile lane_allocation.json — Stage 1a (writer + reader fallback)
mode: IMPLEMENTATION
slug: 2026-05-21-multi-profile-lane-allocation-stage-1a
parent_design: docs/plans/2026-05-21-mnq-eval-account-profile-design.md (in-conversation, not yet on disk)
---

## Task

Migrate `lane_allocation.json` from single-profile flat shape to file-per-profile shape so that running `rebalance_lanes.py --profile <X>` cannot destroy another profile's deployed lanes.

**Stage 1a scope:** introduce file-per-profile writer + reader with backward-compat fallback to the legacy single-profile file. After 1a, both shapes coexist. Subsequent stages migrate readers and then remove the legacy fallback.

**Out of scope for 1a (later stages):**
- Migrating the ~15 production readers to the new shape (Stage 1b)
- Migrating ~30 research / test references (Stage 1c)
- Removing the legacy fallback (Stage 1d)
- Profile→account_id broker routing (Stage 2)
- Eval profile entry + rebalance (Stage 3)

## Scope Lock

- trading_app/lane_allocator.py
- trading_app/prop_profiles.py
- docs/specs/lane_allocation_schema.md (new — schema spec for new shape)
- tests/test_trading_app/test_lane_allocator.py
- tests/test_trading_app/test_prop_profiles.py

## Blast Radius

- trading_app/lane_allocator.py — `save_allocation()` writes new per-profile file at `docs/runtime/lane_allocation/<profile_id>.json`; legacy `docs/runtime/lane_allocation.json` still written for backward-compat in Stage 1a (drops in Stage 1d).
- trading_app/prop_profiles.py — `load_allocation_lanes()`, `load_paused_strategy_ids()`, and sibling loaders read new path first, fall back to legacy single-profile file on miss. Profile-mismatch guard preserved in legacy path.
- docs/specs/lane_allocation_schema.md — NEW spec file documenting the new shape (canonical surface; satisfies "canonical block in stage file anti-pattern" rule by keeping schema OUT of this stage file).
- tests/test_trading_app/test_lane_allocator.py — add tests for new-path write, legacy-path write parity, double-write idempotency.
- tests/test_trading_app/test_prop_profiles.py — add tests for new-path read, legacy-fallback read, missing-both fail-closed.
- Reads: gold.db (read-only via existing allocator queries); no DB writes.
- Writes: NEW path `docs/runtime/lane_allocation/<profile_id>.json` PLUS legacy `docs/runtime/lane_allocation.json` (transitional).
- Drift checks: existing checks read legacy path; they continue to pass because legacy is still written. Stage 1b extends checks to also validate new path; Stage 1d removes legacy.

## Verification

1. `pytest tests/test_trading_app/test_lane_allocator.py tests/test_trading_app/test_prop_profiles.py -q` passes
2. `python pipeline/check_drift.py` passes (legacy path still authoritative until Stage 1d)
3. After rebalancing `topstep_50k_mnq_auto`: BOTH files exist; `load_allocation_lanes('topstep_50k_mnq_auto')` returns the same 3 lanes from either path (parity test executed as part of acceptance)
4. Grep confirms no production reader has been silently switched to new-only path (those edits belong in 1b, not 1a)
5. Self-review pass: trace the write path, the read path, and the fallback path end-to-end

## Acceptance criteria — Stage done when

- [ ] New writer code lands; legacy writer still runs
- [ ] New reader code lands with fallback to legacy
- [ ] Schema spec `docs/specs/lane_allocation_schema.md` exists
- [ ] Tests added and passing
- [ ] `check_drift.py` passes
- [ ] Manual smoke: rerun rebalance, verify both files identical content for the rebalanced profile
- [ ] Commit + push + open PR

## Notes

- File-per-profile choice (not single multi-profile file) was explicit user direction. Rationale: `shared-state-commit-guard.py` governs `docs/runtime/` write contention; per-profile files limit blast radius of parallel rebalance runs.
- This is Stage 1a of an N-stage migration. Each subsequent stage gets its own stage file. Do not expand this stage's scope_lock.

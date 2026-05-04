---
task: codex-handoff-findings-1-2-3
mode: IMPLEMENTATION
phase: 3/3
scope_lock:
  - .githooks/pre-commit
  - scripts/tools/live_readiness_report.py
  - tests/test_tools/test_live_readiness_report.py
  - trading_app/live/bot_state.py
  - tests/test_trading_app/test_session_orchestrator.py
blast_radius: |
  Three independent fixes from Codex code-review handoff (other-terminal review pass on
  branch codex/topstep-operator-arch-v2 delta vs origin/main).

  F1 (.githooks/pre-commit): restore drift-progress UX + fast dev-dep check from 09b115c5
  that was unintentionally reverted by 7c05a60b. Hook script, not production code path.
  Affects pre-commit gauntlet UX only — single-pass visible drift output instead of
  silent-then-rerun. Drift-check semantics unchanged.

  F2 (scripts/tools/live_readiness_report.py): operator-facing report currently shows
  allocator's lanes even when allocator JSON belongs to a different profile. Fix gates
  on profile_match is True; falls back to profile-config lanes on mismatch. Adds one
  pytest case. Read-only DB and JSON; no schema, no canonical reencoding. Imports
  remain canonical (pipeline.paths, trading_app.lifecycle_state, trading_app.prop_profiles,
  trading_app.validated_shelf).

  F3 (trading_app/live/bot_state.py): _iso_utc() silently returns None for non-datetime
  inputs. Adds pandas.Timestamp branch + logger.warning for unknown non-None types.
  Defensive-only; happy-path datetime behavior unchanged. Touches trading_app/live/
  (NEVER_TRIVIAL) which is why this stage is IMPLEMENTATION not TRIVIAL. No data-shape
  change. No call-site change. Function signature unchanged. Per
  adversarial-audit-gate.md, severity MEDIUM is below CRIT/HIGH trigger; skipping audit.

acceptance:
  - F1: git diff 09b115c5 HEAD -- .githooks/pre-commit shows the dev-dep + drift-step diffs gone (matches what we restore)
  - F1: pre-commit gauntlet runs and shows visible drift checks (not silent then rerun)
  - F2: new test case asserts profile-mismatch fallback to profile_config bucket
  - F2: existing 2 tests still pass
  - F3: new test case for non-datetime input either coerces or warns + None
  - F3: existing test_build_state_snapshot tests still pass
  - All three: pipeline/check_drift.py passes
  - All three: ruff check + ruff format check pass
  - Three discrete commits pushed to origin
agent: claude
updated: 2026-05-04
---

# Codex handoff — Findings #1, #2, #3

**Status:** IMPLEMENTATION (3 phases, one commit each)
**Date:** 2026-05-04
**Source:** Other-terminal Codex review pass measured the same findings I had paged from delta review of `codex/topstep-operator-arch-v2` 18-commit delta vs `origin/main`. Codex handed implementation back to this Claude session with smallest-safe-patch sequencing.

## Why three commits, not one

Each finding has a different blast radius and different test target. Bundling them would defeat `branch-discipline.md` "review each commit in isolation" and would force a single pre-commit gauntlet to validate three unrelated changes. Three discrete commits keep audit trail clean.

## Phase 1 — F1 hook restore (this commit)
Cherry-pick from `09b115c5`:
- `.githooks/pre-commit` lines 71–73 (`_dev_dep_check` → `importlib.util.find_spec`)
- `.githooks/pre-commit` lines 247–262 (drift step single-pass with tee/awk)

Skip the HANDOFF.md portion of `09b115c5` — that's stale ledger noise.

## Phase 2 — F2 allocator profile-mismatch fail-closed
- `scripts/tools/live_readiness_report.py:275-285`: gate `if allocator_summary.get("active_lanes"):` on `allocator_summary.get("profile_match") is True`
- `tests/test_tools/test_live_readiness_report.py`: new test `test_falls_back_to_profile_config_when_allocator_profile_mismatched`

## Phase 3 — F3 _iso_utc hardening
- `trading_app/live/bot_state.py:_iso_utc()`: add `pd.Timestamp` lazy-import branch; add `logger.warning` for unknown non-None types
- `tests/test_trading_app/test_session_orchestrator.py`: extend with non-datetime input case

## Out of scope
- F4 (re-exec recursion guard tied to env var) — defensive concern, no observed bug
- F5 (DuckDB connect no timeout) — defensive concern, separate patch
- The 13-behind-main rebase — that's a separate effort, blocked on this branch's other work

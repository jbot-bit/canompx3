---
task: "Stage 1 of fleet-state-brain plan — build the canonical read-only fleet_state.py resolver (single liveness oracle, heartbeat-authoritative), the shared _worktree_hollow.py predicate, and active_plan.md resurfacing scaffolding. Read-only; no destructive cleanup; no force-close changes."
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/fleet_state.py
  - scripts/tools/_worktree_hollow.py
  - scripts/tools/_worktree_churn.py
  - scripts/tools/active_plan.py
  - tests/test_tools/test_fleet_state.py
  - tests/test_tools/test_worktree_hollow.py
  - tests/test_tools/test_worktree_churn.py
  - docs/runtime/active_plan.md
  - trading_app/live/instance_lock.py
---

## Scope addition (2026-06-06): unblock-only lint fix
`trading_app/live/instance_lock.py` is added for ONE reason: the tree-wide
pre-commit ruff gate flags a PRE-EXISTING B007 (unused loop var `attempt`,
both occurrences) from commit f1413178 — a capital file I did not author. It
blocks EVERY commit until clean. Fix is the linter's own prescribed rename
(`attempt`→`_attempt`), ZERO logic/behavior change to the live path. Operator
approved (AskUserQuestion, 2026-06-06). NOT a fleet-state change.

## Scope Lock
- scripts/tools/fleet_state.py
- scripts/tools/_worktree_hollow.py
- scripts/tools/_worktree_churn.py
- scripts/tools/active_plan.py
- tests/test_tools/test_fleet_state.py
- tests/test_tools/test_worktree_hollow.py
- tests/test_tools/test_worktree_churn.py
- docs/runtime/active_plan.md
- trading_app/live/instance_lock.py

## Blast Radius
- All fleet-state files are ADDITIVE new surface — nothing imports fleet_state yet.
- trading_app/live/instance_lock.py — UNBLOCK-ONLY lint rename (attempt→_attempt,
  both loops); ZERO logic/behavior change; pre-existing B007 from f1413178 that the
  tree-wide pre-commit ruff gate flags, blocking every commit. Operator-approved.
- Reads: git, heartbeat sidecars (read-only). Writes: docs/runtime/active_plan.md.

## Fair-Fight Corrections (added 2026-06-06 — before declaring Stage 1 done)
Three things were narrowed away in the first "done" claim; this stage closes them:
1. **Keystone liveness untested in its TRUE branch** — every live run read `live=False`;
   add a fresh-heartbeat→LIVE integration test (real `.beat` + PID-stub fallback).
2. **Re-encoded churn-list (`_CHURN_PATHS`)** — institutional-rigor §4 violation.
   **Option A (operator-chosen):** extract canonical `_worktree_churn.py`; point
   fleet_state at it. Migration of the 3 existing copies (`run_live_session.py:558`,
   `checkpoint_guard.py:32`, `check_root_hygiene.py:51`) is explicit FOLLOW-UP DEBT,
   tracked in `active_plan.md`, NOT done this stage (one copy is a capital path).
3. **HOLLOW+unpushed collision** — a HOLLOW tree that also carries unpushed commits
   must NOT be silently reap-eligible in Stage 2; guard + test added.

## Blast Radius
- fleet_state.py is ADDITIVE and READ-ONLY — nothing in the repo imports it yet, so nothing breaks if unused. Pure new surface.
- It DELEGATES liveness to scripts/tools/worktree_guard._peer_is_live / _fresh_peer_heartbeat / read_lease (canonical, imported — NOT re-encoded). It enumerates worktrees via worktree_manager.list_worktrees. It reuses stale_work_radar helpers for ahead/behind/dirty where applicable.
- _worktree_hollow.py is a NEW standalone predicate (del>=100 AND nondel<=10 AND del/total>=0.9); no existing module computes hollowness, so zero call-site churn.
- active_plan.py + active_plan.md persist the current approved plan; read-only resurfacing comes via a hook in a LATER stage (not this one).
- Reads: git (worktree list, status --porcelain, rev-list), heartbeat sidecar files (read-only). Writes: docs/runtime/active_plan.md only (one durable anchor; no schema, no gold.db, no capital path).
- NO guard is repointed this stage (that is Stage 2). NO close --force change (Stage 2). NO awareness hook registered (Stage 3).

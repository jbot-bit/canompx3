---
task: Phantom-cwd worktree-launch auto-reconcile — husk dirs no longer launched into
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/worktree_manager.py
  - scripts/tools/worktree_launch_preflight.py
  - START_WORKTREE.bat
  - scripts/tools/new_session.sh
  - tests/test_tools/test_worktree_manager.py
  - tests/test_tools/test_worktree_launch_preflight.py
---

## Blast Radius

- `scripts/tools/worktree_manager.py` — ADD canonical helpers after `prune_worktrees`
  (`_canonical_main_root`, `_norm`, `is_registered_worktree`, `_SCRATCH_NAMES`,
  `_is_scratch_only`, `is_safe_graveyard`, `_uniquify`, `reconcile_launch_path`,
  `reap_graveyards`) + two CLI subcommands. No existing function changed. Zero
  current callers of the new symbols outside the launchers added in this stage.
- `scripts/tools/worktree_launch_preflight.py` — `classify()` gains an
  `is_registered_worktree` short-circuit BEFORE the dirty/lease branch; new
  `import worktree_manager` (same dir, no cycle — verified). A husk now returns
  NEW instead of REUSE_CLEAN.
- `START_WORKTREE.bat` / `scripts/tools/new_session.sh` — reconcile launch path
  before `git worktree add`; branch-collision retry mirroring
  `create_worktree:381-385`. Launcher-only; no production trading logic.
- Reads: `git rev-parse --git-common-dir`, `git worktree list` (read-only).
  Writes: rmtree ONLY on provably scratch-only husks (work-preservation gate);
  default reap path deletes nothing.
- No drift-check / hook parses these modules or the REUSE_CLEAN token — no
  canonical-parity surface. TRIVIAL-tier per the approved plan.

Implements `memory/project_phantom_cwd_fix_HANDOFF_to_isolated_wt_2026_06_11.md`.

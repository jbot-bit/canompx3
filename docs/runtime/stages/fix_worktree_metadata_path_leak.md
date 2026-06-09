# Stage: Fix _worktree_metadata global-state leak

task: Scope `_worktree_metadata` authoritative `git worktree list` results to `canonical/.worktrees` so a path-scoped call (e.g. pytest tmp_path) cannot leak the host machine's live worktrees. Closes the env-coupling flake in `test_project_pulse.py::TestCollectWorktrees::test_detects_worktree`.

mode: IMPLEMENTATION

scope_lock:
  - scripts/tools/project_pulse.py
  - tests/test_tools/test_project_pulse.py

## Blast Radius
- `scripts/tools/project_pulse.py` — modifies `_worktree_metadata` only (the authoritative branch). TWO callers, both want repo-scoped results: `collect_worktrees` (fast + deep paths) and `collect_worktree_conflicts:2490` (file-overlap radar; foreign worktrees produced meaningless `main...<branch>` diffs). Containment filter can only DROP foreign worktrees; managed worktrees all live under `canonical/.worktrees` by construction, so real-repo output is unchanged for both. No schema, no capital path, no canonical-source edit.
- `tests/test_tools/test_project_pulse.py` — adds one regression test asserting `collect_worktrees(tmp_path)` ignores the host machine's live `git worktree list`.
- Reads: filesystem (worktree metadata), `git worktree list` (read-only). Writes: none.

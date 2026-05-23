---
task: |
  IMPLEMENTATION — Worktree concurrency guard. New per-worktree lease + PreToolUse
  hook that BLOCKS Edit/Write/MultiEdit/Bash when another live Claude session is
  active in the SAME worktree. Distinct from the existing 12h SessionStart PID-lock
  (`.git/.claude.pid`): this lease is per-worktree, lives at
  `<git-dir>/.claude.worktree.lease.json`, uses a 30-min staleness threshold, and
  is enforced PreToolUse (not only at session start). CLI tool exposes
  --status / --release / --force-release. Stale leases are auto-reclaimed when
  the holder PID is dead OR the heartbeat is older than 30 min.
mode: IMPLEMENTATION
updated: 2026-05-23T00:29Z
scope_lock:
  - scripts/tools/worktree_guard.py
  - .claude/hooks/worktree_guard.py
  - .claude/hooks/session-start.py
  - .claude/settings.json
  - tests/test_tools/test_worktree_guard.py
  - .claude/hooks/tests/test_worktree_guard_hook.py
  - pipeline/check_drift.py
  - HANDOFF.md

## Blast Radius
- `scripts/tools/worktree_guard.py` — NEW. Canonical lease I/O (acquire / heartbeat /
  release / inspect). CLI: `--status`, `--release`, `--force-release`. Fail-open on
  filesystem errors per institutional-rigor §6 (logged, never silent).
- `.claude/hooks/worktree_guard.py` — NEW. PreToolUse(Edit|Write|MultiEdit|Bash) hook.
  Reads lease via `scripts/tools/worktree_guard.py`; exits 2 (BLOCK) when a peer
  session holds the lease for THIS worktree. Auto-clears stale leases (dead PID OR
  >30 min). No imports of pipeline/ or trading_app/ — pure stdlib.
- `.claude/hooks/session-start.py` — adds one call to `acquire_worktree_lease()` at
  session startup. Does NOT replace `_session_lock_lines()` (12h conservative lock
  retained as separate safety net). Fail-open.
- `.claude/settings.json` — register the new PreToolUse hook under
  `matcher: "Edit|Write|MultiEdit|Bash"`, timeout 3s.
- `pipeline/check_drift.py` — adds `check_worktree_guard_lease_path_parity` to
  enforce canonical lease path between the CLI module and the hook module
  (single source per institutional-rigor §4).
- Reads: filesystem only (`<git-dir>/.claude.worktree.lease.json`), `git rev-parse`
  for git-dir resolution. Writes: that lease file only.
- No production trading logic touched. No schema change. No DB write. No live
  runtime path. Capital class: NONE.

## Acceptance
- `python scripts/tools/worktree_guard.py --status` prints lease state (holder PID,
  worktree, heartbeat age, this-process-is-holder bool, exits 0).
- `python scripts/tools/worktree_guard.py --release` removes the lease iff held by
  current PID; refuses with exit 2 otherwise.
- `python scripts/tools/worktree_guard.py --force-release` always removes.
- Hook BLOCKS (exit 2, structured stderr) when lease file shows a different live
  PID for THIS worktree path.
- Hook ALLOWS (exit 0) when (a) no lease, (b) lease held by current PID, (c) lease
  holder is dead, (d) heartbeat older than 30 min, (e) lease worktree path
  differs from current (sibling-worktree leases must not cross-block).
- Heartbeat refresh: every hook invocation that ALLOWS updates the heartbeat
  timestamp atomically (write-temp + os.replace).
- `pytest .claude/hooks/tests/test_worktree_guard_hook.py tests/test_tools/test_worktree_guard.py -v`
  → all green. Minimum coverage: acquire-fresh, acquire-stale-dead-pid,
  acquire-stale-heartbeat, peer-live-block, peer-dead-reclaim, sibling-worktree-
  no-cross-block, force-release, current-pid-release.
- `python pipeline/check_drift.py` → still 160+ PASSED, new parity check
  enumerated, zero violations.
- `live_journal.db` skip-worktree flag cleared (verified: `git ls-files -v` shows
  `H live_journal.db`, no `S`).

## Out of scope
- Replacing `_session_lock_lines()` in session-start.py (kept as belt-and-braces;
  longer threshold serves a different incident class).
- Cross-worktree coordination (handled by `multi-terminal-shared-file-hygiene.md`
  + `shared-state-commit-guard.py` already).
- PR #313 rebase/merge work (suspended until this guard ships per user directive).

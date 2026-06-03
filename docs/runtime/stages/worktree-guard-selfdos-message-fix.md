task: Fix self-DOS escape-hatch text in 3 guard BLOCK messages (worktree_guard, mcp-git-guard, branch-flip-guard) — each recommends new_session.sh (Bash/git), which the guard itself blocks; add START_WORKTREE.bat as the launcher that works under the block.
mode: TRIVIAL

## Scope Lock
- .claude/hooks/worktree_guard.py
- .claude/hooks/mcp-git-guard.py
- .claude/hooks/branch-flip-guard.py

## Blast Radius
- worktree_guard.py — edits ONLY the `_emit_block()` stderr resolution strings (lines ~114-121). No logic change; exit codes / matchers untouched. Callers: PreToolUse hook surface (Edit/Write/MultiEdit/Bash). Reads: none. Writes: none.
- mcp-git-guard.py — edits ONLY the stderr "Options:" block strings (~150-166). No logic change.
- branch-flip-guard.py — edits ONLY the stderr "Options:" block strings (~119-138). No logic change.
- session-start.py NOT touched (WARN-only at startup, no active block to route around — out of scope).
- Zero production code (pipeline/ trading_app/). Message strings only. No schema, no capital, no canonical source. Net diff well under 100 lines.
- Verification: smoke-fire each hook with a synthetic block payload OR grep-confirm START_WORKTREE.bat now appears in each block message; check_drift.py.

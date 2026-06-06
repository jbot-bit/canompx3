task: Anchor heartbeat liveness to the long-lived Claude session process so orphan .beat files from crashed sessions stop false-blocking the worktree guard on the busy main checkout.
mode: IMPLEMENTATION

## Scope Lock
- scripts/tools/worktree_guard.py
- .claude/hooks/session-heartbeat.py
- .claude/hooks/worktree_guard.py
- tests/test_tools/test_worktree_guard.py

## Blast Radius
- scripts/tools/worktree_guard.py — adds `_find_session_anchor_pid()` helper + anchor cross-check inside `_fresh_peer_heartbeat` (called only by `_peer_is_live` blocking path + `status()` observability). Backward-compatible: beats lacking `anchor_pid` keep mtime-only behavior (pure superset). Reuses existing `_pid_is_alive` / `_get_process_create_time_windows`. No lease-write change.
- .claude/hooks/session-heartbeat.py — stamps `anchor_pid` + `anchor_create_time` in beat payload (walk parent chain to claude.exe). Omits fields on walk-up failure (mtime fallback). Existing fields unchanged.
- .claude/hooks/worktree_guard.py — `_emit_block` reports the REAL verdict (lease ppid live / live-anchor beat / legacy mtime-only) instead of asserting "liveness confirmed". Message-only.
- tests/test_tools/test_worktree_guard.py — RED-first tests for orphan/live-anchor/old-format/stale/pid-reuse/probe-error.
- Reads: process table (OpenProcess / /proc), git-common-dir. Writes: .beat files only. No gold.db, no schema, no capital path.

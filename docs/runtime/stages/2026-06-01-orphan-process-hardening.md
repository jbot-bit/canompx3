---
task: Orphan-process hardening — reap stale MCP generations parented by abandoned Claude launchers + auto-clean at session start
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/reap_stale_claude_processes.py
  - tests/test_tools/test_reap_stale_claude_processes.py
  - .claude/hooks/session-start.py
---

## Blast Radius
- `scripts/tools/reap_stale_claude_processes.py` — adds Rule (d) (stale-launcher-ancestry) + a pure `_claude_launcher_pid` helper; `decide()` signature gains `current_launcher_pid` + `launcher_started_by_pid` args. Pure decision fn, fully unit-tested against synthetic tables. Infra-only, fail-open, NOT capital-path. Gate 0 (self/ancestry) + Gate 1 (capital-path) precede all new logic.
- `tests/test_tools/test_reap_stale_claude_processes.py` — adds Rule (d) tests; existing 17 tests must still pass (decide() gets new keyword args with safe defaults so old call-sites are unaffected).
- `.claude/hooks/session-start.py` — `_stale_process_reaper_lines()` opts into bounded `--apply --reap-duplicates` ONLY when a session lock is readable. Fail-silent. Capital-path hard-excluded by the reaper's own Gate 1; live bot never at risk.
- Reads: live process table (OS boundary), `.git/.claude.pid` lock. Writes: none to repo; sends SIGKILL/taskkill to provably-stale non-capital processes only.

## Done criteria
1. New Rule (d) tests pass + all existing 17 pass (show output).
2. `python pipeline/check_drift.py` passes.
3. Live PC: stale MCP generation count drops to ~1 pair per server after `--apply`.
4. Dead code swept; self-review passed.

---
task: "Fix worktree concurrency guard — the lease provided ZERO mutual exclusion (OS lock held by an ephemeral hook subprocess that dies microseconds later). Replaced with (session_id, ppid)+heartbeat liveness so a 2nd session in ONE worktree is genuinely hard-refused. n=2 incident (2026-05-29 + 2026-05-30, both two-in-one-tree on main during git surgery)."
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/worktree_guard.py
  - .claude/hooks/worktree_guard.py
  - .claude/hooks/session-start.py
  - tests/test_tools/test_worktree_guard.py
  - .claude/hooks/tests/test_worktree_guard_hook.py
blast_radius: |
  ## Blast Radius
  - scripts/tools/worktree_guard.py — CANONICAL lease I/O. Rewrote acquire()/status() ownership model
    + added Windows OpenProcess liveness (_pid_is_alive_windows). Kept release()/CLI/path surface stable
    (drift parity check check_worktree_guard_lease_path_parity loads BOTH this + the hook).
  - .claude/hooks/worktree_guard.py — PreToolUse hook. Now reads event['session_id'], passes to acquire().
  - .claude/hooks/session-start.py — _worktree_lease_lines(session_id) threads id through; handles 'reclaimed'.
  - EVERY session in EVERY worktree runs these hooks. Fail-open MANDATORY.
---

## STATUS (2026-05-30, RESUME HERE)

**Branch:** session/joshd-worktree-lease-fix (worktree C:/Users/joshd/canompx3-worktree-lease-fix, off origin/main 17da1468).
**NOT committed. NOT pushed.** Working-tree changes: the 3 production files + 2 test files + this stage file.

### Verification DONE (all real, executed):
- ✅ 31/31 tests/test_tools/test_worktree_guard.py + tests/test_hooks/test_active_sibling_guard.py
  (incl. new test_reclaims_lease_with_dead_ppid, test_reclaims_lease_with_stale_heartbeat,
   test_same_session_refreshes_not_blocks, test_blocked_by_live_peer — proves Windows OpenProcess liveness).
- ✅ 6/6 .claude/hooks/tests/test_worktree_guard_hook.py — ONLY when run WITHOUT WORKTREE_GUARD_BYPASS=1
  (the block test was masked by an ambient bypass; FIXED by scrubbing the var in _run_hook env).
- ✅ drift check_worktree_guard_lease_path_parity CLEAN; ✅ ruff clean.
- ⚠️ 2 Pyright "unreachable" diagnostics — benign os.name=="nt" platform branches (POSIX fallback). Not bugs.

### Verification PENDING (do these to finish):
1. Full `python pipeline/check_drift.py` — was running in background (/tmp/drift_real.txt); CONFIRM exit 0.
2. Adversarial-audit gate (evidence-auditor) — was being dispatched when context ran out. RUN IT (mandatory,
   session-isolation infra). Prompt focus: block-fires, /clear-restart reclaim, fail-open, read/write race,
   90s heartbeat vs long tool calls.
3. After audit clean: commit on session/joshd-worktree-lease-fix, push, open PR.

### How to run tests (env is hostile — use EXACTLY this):
- python: C:/Users/joshd/canompx3/.venv/Scripts/python.exe (worktree has NO .venv)
- cwd: the worktree (Bash cwd RESETS to main tree between calls — always `cd` in same command)
- DO NOT export WORKTREE_GUARD_BYPASS for the hook block test.

## Root cause (proven)
PreToolUse hook runs as a NEW python subprocess per tool call → filelock OS lock auto-releases on subprocess
exit (microseconds) → never a mutex; sidecar just recorded "whoever ran a tool last" (rotating-PID storm).
session_id mints fresh every /clear (1180 jsonl files) → can't key on it alone → composite (session_id, ppid)
where ppid = hook subprocess's parent = the Claude session process (stable, reliably liveness-probed via
Windows OpenProcess). Peer live = fresh heartbeat (<90s) AND ppid alive. Stale/dead → reclaim (no 12h wait;
makes /clear-restart safe).

## Session context for resume
- Git on MAIN was fully recovered earlier (HEAD 648f2e71, nothing lost, dangling rebase aborted, stash safe).
- The OTHER terminal (PID 65244) finished + pushed origin/main to f44450e5 (split fix(deployability) 950c375d,
  integrated Codex #324/#325). Main tree now in sync — that divergence is RESOLVED.
- Original ask was "Stage 2" = verify+ship SR-alarm strict double-counting (already committed in HEAD history,
  5b2e00d9+db6565dd) — DEFERRED; pivoted to this guard fix after the n=2 collision recurred.

## Acceptance
- [x] Two distinct sessions, one worktree, both fresh+live → 2nd BLOCKED (exit 2). [hook test passes]
- [x] Same session_id re-acquiring → refresh, never block.
- [x] Stale heartbeat OR dead ppid → reclaim (no 12h wait). [/clear-restart safe]
- [x] Fail-open on missing session_id / non-git / FS error / ctypes unavailable.
- [x] parity drift CLEAN; ruff clean.
- [ ] full check_drift.py exit 0 (CONFIRM /tmp/drift_real.txt).
- [ ] adversarial-audit gate run + clean.
- [x] merged + pushed (8f3ea8ed on origin/main).

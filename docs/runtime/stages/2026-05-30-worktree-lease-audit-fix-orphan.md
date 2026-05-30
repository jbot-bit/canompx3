---
task: "Land orphaned adversarial-audit fix from PR #326. The PR squash-merged commit bfab5942 (real worktree mutex) but the follow-up audit-close commit 57ab5df5 was pushed AFTER GitHub recorded headRefOid, so it never reached main. Two production fixes are missing: (1) CRITICAL torn-read — _write_lease must use atomic os.replace, main still does direct write_text; (2) HIGH Windows PID-reuse false-block — ppid_create_time FILETIME cross-check + lease schema 2->3, absent on main. Cherry-pick the code+test delta (excluding the stale HANDOFF hunk) onto main."
mode: CLOSED
scope_lock:
  - scripts/tools/worktree_guard.py
  - tests/test_tools/test_worktree_guard.py
blast_radius: |
  ## Blast Radius
  - scripts/tools/worktree_guard.py — CANONICAL lease I/O. Adds _get_process_create_time_windows();
    threads expected_create_time through _pid_is_alive/_pid_is_alive_windows; _build_payload bumps
    schema 2->3 and stores ppid_create_time on Windows; _write_lease becomes atomic (temp + os.replace);
    _peer_is_live passes ppid_create_time to liveness. Backward-compatible: ppid_create_time absent on
    old (schema 2) leases -> cross-check skipped -> falls back to exit-code-only (prior behavior).
  - tests/test_tools/test_worktree_guard.py — +2 tests (atomic torn-read, schema-3 ppid_create_time).
  - Companion hook (.claude/hooks/worktree_guard.py) is UNCHANGED by 57ab5df5 (verified blob-identical
    main vs branch); drift parity check check_worktree_guard_lease_path_parity unaffected (no path/CLI change).
  - EVERY session in EVERY worktree runs this guard. Fail-open preserved (all ctypes paths except->None/True).
acceptance:
  - "scripts/tools/worktree_guard.py blob on HEAD == 57ab5df5:scripts/tools/worktree_guard.py (aaa3da83)"
  - "tests/test_tools/test_worktree_guard.py blob on HEAD == 57ab5df5 version (5cf12ba6)"
  - "uv run pytest tests/test_tools/test_worktree_guard.py -p no:timeout -q  -> all pass"
  - "python pipeline/check_drift.py  -> exit 0 (incl check_worktree_guard_lease_path_parity)"
---

## STATUS (2026-05-30)

Discovered during /next: PR #326 (MERGED as 9a435d2b) merged headRefOid=bfab5942 ONLY.
The audit-close commit 57ab5df5 (atomic-write CRITICAL + PID-reuse HIGH) was pushed after
the merge head was recorded and never landed. Proof: main blob 7327b8a8 == bfab5942; 57ab5df5
blob aaa3da83 differs.

### Plan
1. Branch off main: session/joshd-worktree-lease-audit-fix. ✅
2. git cherry-pick -n 57ab5df5; restore the HANDOFF.md hunk to main (stale, Codex-authored). ✅
3. Verify acceptance criteria. Commit. Push. PR.

### Verification (executed)
- ✅ scope blobs MATCH 57ab5df5: worktree_guard.py=aaa3da83, test=5cf12ba6
- ✅ uv run pytest tests/test_tools/test_worktree_guard.py -p no:timeout → 23 passed (incl 2 new audit tests)
- ✅ .claude/hooks/tests/test_worktree_guard_hook.py → 6 passed
- ✅ ruff clean on both scope files
- ✅ python pipeline/check_drift.py → 170 passed, 0 violations (check_worktree_guard_lease_path_parity clean)

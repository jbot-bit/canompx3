# Stage State — Untrack-to-Main + PR Merge (deferred for fresh context)

**Created:** 2026-06-03 ~20:30 local
**Mode:** RESUME-READY (deferred — do NOT start on full context)
**Author session:** session/joshd-maximise-ops-fix (worktree C:/Users/joshd/canompx3-maximise-ops-fix)
**Why deferred:** the core action is a ~3M-line shared-history change to `main` with live peers active; was paused at 86% context. Resume in a FRESH session.

---

## The root-cause blocker (one sentence)

`origin/main` still tracks **13,406 artifact files** (~3M lines of scratch); the cleanup that REMOVES them (`6fee3349`) lives only on `session/joshd-maximise-ops-fix` and was never merged to main — so PRs #354/#355 sit on a bloated main and their CI fails on stale/unrelated tests.

**IMPORTANT framing:** "landing the untrack" REMOVES 13,406 tracked files (slims main), it does NOT add files. Files stay on disk; they just stop being git-tracked. Gitignore already covers `artifacts/research/*`.

---

## Exact state at handoff

### Branches / SHAs
- `session/joshd-maximise-ops-fix` (current, the untrack lives here): **7 ahead / 14 behind** origin/main. Untrack commits: `6fee3349` (13,390-file removal), `f09b4cc9` (gitignore+runbook), `a8928b4c` (HANDOFF). All pushed to origin tracking branch (synced).
- `origin/main`: still carries 13,406 artifact files. Last seen tip moved past `94085955`.

### Open PRs (all intact on origin, nothing lost)
- **#354** `session/joshd-powered-oos-reform` (worktree `C:/Users/joshd/canompx3-powered-oos-reform`): MERGEABLE/BEHIND. I merged main into it + pushed `7e9b75b3` (harmless — same artifacts main already has). CI FAILS on **unrelated stale tests** (`test_audit_run_all.py::test_run_all_quick_forwards_quick_to_quick_aware_phases` KeyError, `test_stale_work_radar.py::...blob_trap` assert 0==1) — NOT its own powered-OOS code (12/12 its tests pass). Verdict was MERGE.
- **#355** `salvage/weekly-project-audit-tool`: CONFLICTING, 266 behind. Merge-main attempt was ABORTED (showed as 3M-insertion because the old branch lacked main's artifacts — net count unchanged, NOT new bloat). Branch back at clean `ae5420c6`. Verdict was MERGE (27 tests pass). Temp worktree `C:/Users/joshd/.claude/scratch/wt-weekly-audit` git-ref removed but DIR may be locked (harmless scratch; clean up if present).
- **#356** `codex/project-followup-automatic`: HOLD — capital path (run_live_session live-launch gates). Flagged "do not merge during MNQ pilot". OPERATOR-GATED.
- **#345** `session/push-hookfix`: post-compaction hooks. Pre-existing, mergeable, OPERATOR-GATED (don't merge unasked).

---

## Resume sequence (FRESH session, when main is QUIET)

**Preconditions before touching main:** (1) no `.claude-heartbeats/*.beat` <10min in `C:/Users/joshd/canompx3/.git/.claude-heartbeats/` besides self; (2) main worktree tree clean; (3) NOT on full context.

1. **Land untrack on main.** Merge `session/joshd-maximise-ops-fix` → main. It's 14 behind so first bring it current (`git merge origin/main` into it — expect artifact churn but net-neutral), verify `git ls-files artifacts | wc -l` DROPS to ~0 after, then merge to main in the main worktree via `git -C C:/Users/joshd/canompx3 merge --no-ff`. **Verify POSITIVE deliverable**: `git -C C:/Users/joshd/canompx3 ls-files artifacts/ | wc -l` must be ~0, not 13406 (see feedback_pathspec_commit_silently_dropped_staged_deletions). Push main (regular, no force).
2. **Rebase/merge-main the PRs onto the now-slim main.** #354 and #355 each: `git merge origin/main` into branch (no force — force-push is DENIED), resolve HANDOFF-only conflicts (take main's), push.
3. **CI re-runs on slim main** — #354's stale-test failures should clear (they pass on current main). When green + main quiet → `gh pr merge <n> --squash --delete-branch`.
4. **Never** merge #356/#345 unmasked. **Never** touch `.codex/worktrees/1248` (live peer broker work).

## Hard constraints (carry forward)
- FORCE-PUSH DENIED (`Bash(git push --force*)`). Use merge-main, never rebase+force-push.
- Don't interrupt pre-commit drift hooks (~2min, wait for task-notification).
- Solo repo → direct squash-merge OK, no PR-review ceremony needed.
- Main = capital/canonical (Tier B). Live peers + dirty tree = DO NOT TOUCH main.

## Related memory
- `feedback_stranded_work_audit_local_only_is_the_risk_not_merge_state_2026_06_03`
- `feedback_pathspec_commit_silently_dropped_staged_deletions_verify_HEAD_contents_2026_06_03`
- `feedback_chordia_artifacts_git_add_all_2m_line_blob_2026_05_31`

# Runbook — Purge `artifacts/research/` from Git History

**Status:** PENDING — execute only in a coordinated maintenance window.
**Owner:** repo maintenance lead (operator-supervised).
**Created:** 2026-06-03, alongside the forward-safe untrack commit.

---

## Why this exists

`artifacts/research/` (13,390 files / 91.3 MB of generated chordia batch-run
scratch) rode into `origin/main` via the chordia merge chain
(`6910d0c3` → `aa78f8d0` (#332) → `7319166a`) **before** a gitignore covered it.

The forward-safe step is already done (see the companion commit): the directory
is now gitignored and untracked from the index, so it **cannot grow** and future
`git add -A` cannot re-bloat it. Disk files are untouched.

What remains is the 91 MB still living in **history**. Removing it requires
rewriting every commit from the blob's introduction forward and force-pushing
`main` — a **destructive, outward-facing** operation. It is intentionally NOT
done automatically because, at authoring time, the repo had **20 active
worktrees** and **live peer sessions beating on the main checkout**. A
force-push under those conditions corrupts the live trading runtime and orphans
every other worktree.

**Do not run any step below until the maintenance window preconditions hold.**

---

## Hard preconditions (ALL must be true before Step 1)

A force-push on shared `main` is irreversible for every other clone. Verify each:

- [ ] **Live trading bot stopped.** No signal bot, no dashboard, no live session.
      Stop via `stop_live.ps1 -NoPrompt` (PowerShell; the flag avoids the
      interactive hang). Confirm no `run_live_session` / `bot_dashboard` python
      processes remain.
- [ ] **All Claude/Codex peer sessions stopped.** Close every terminal session.
- [ ] **No fresh heartbeats.** Inspect
      `<git-common-dir>/.claude-heartbeats/*.beat` — every file's internal `ts`
      must be older than 10 minutes. A fresh beat = a LIVE peer; DO NOT proceed
      (per `feedback_dead_pid_fresh_heartbeat_lease_trust_heartbeat_not_pid`).
      Trust the heartbeat `ts`, not the PID.
- [ ] **All worktrees enumerated and frozen.** `git worktree list` — record the
      full list (was 20 at authoring). Every one shares the rewritten history and
      will need re-sync (Step 7). No worktree may have uncommitted/in-flight work
      you care about; either commit-and-push it to a *non-main* branch first or
      accept its loss.
- [ ] **No unpushed work on main you want to keep** beyond what the rewrite
      preserves. `git log origin/main..HEAD` on the main checkout — should be empty
      or pushed first.
- [ ] **Operator present and aware** this rewrites published history.

---

## Step 0 — Backup (non-negotiable)

A history rewrite is only as safe as its backup. Make two independent ones:

```bash
# 1. A full mirror clone (complete, restorable repo)
git clone --mirror C:/Users/joshd/canompx3 C:/Users/joshd/canompx3-backup-prepurge.git

# 2. A tag + bundle of current main (cheap rollback ref)
git -C C:/Users/joshd/canompx3 tag prepurge-main-backup origin/main
git -C C:/Users/joshd/canompx3 bundle create C:/Users/joshd/canompx3-prepurge.bundle --all
```

Record the pre-purge `origin/main` SHA here at execution time: `__________`.

---

## Step 1 — Run the history rewrite

`git-filter-repo` is installed (verified 2026-06-03). Run from a **fresh clean
clone of main**, not a worktree (filter-repo refuses to run in a non-fresh repo
by default and you do not want to rewrite a checkout other worktrees share):

```bash
git clone C:/Users/joshd/canompx3 C:/Users/joshd/canompx3-purge-work
cd C:/Users/joshd/canompx3-purge-work
git filter-repo --path artifacts/research --invert-paths --force
```

`--invert-paths` keeps everything EXCEPT `artifacts/research`. `--force` is
required because filter-repo guards against running on a non-fresh clone — the
fresh clone above satisfies the intent; the flag acknowledges it.

Note: filter-repo removes the `origin` remote by design (safety). Re-add it:

```bash
git remote add origin <origin-url>
```

---

## Step 2 — Verify the blob is gone from history

```bash
# Should print 0
git log --all --oneline -- artifacts/research | wc -l

# Should print nothing (no blob > a few MB referencing artifacts/research)
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | grep -i 'artifacts/research' || echo "CLEAN: no artifacts/research objects remain"

# Repo size sanity — .git should be ~91 MB smaller than the backup
du -sh .git
```

---

## Step 3 — Force-push main

Only after Step 2 is CLEAN:

```bash
git push origin main --force-with-lease
```

Prefer `--force-with-lease` over `--force`: it refuses the push if `origin/main`
moved since your clone (a peer slipped a commit in despite the freeze) — a final
guard against the ref-lock race class. If it rejects, STOP, re-confirm the freeze,
re-fetch, and re-evaluate (do not blindly `--force`).

---

## Step 4 — Verify origin

```bash
git ls-remote origin main          # record new SHA: __________
# In a throwaway fresh clone of origin:
git clone <origin-url> /tmp/verify-purge && cd /tmp/verify-purge
git log --all --oneline -- artifacts/research | wc -l   # must be 0
```

---

## Step 5–7 — Re-sync every other checkout

Every worktree and the main checkout are now on dead history (old SHAs). For
**each** entry from the `git worktree list` recorded in preconditions:

- **Main checkout** (`C:/Users/joshd/canompx3`): the simplest correct path is a
  fresh clone, or:
  ```bash
  git -C C:/Users/joshd/canompx3 fetch origin
  git -C C:/Users/joshd/canompx3 reset --hard origin/main
  ```
  (Destructive to local main state — that's why the freeze + backup precede it.)
- **Worktrees on `main`:** same `fetch` + `reset --hard origin/main`.
- **Worktrees on feature/session branches:** these branches were ALSO rewritten
  if they descend from the purged history. Rebase each onto the new `origin/main`,
  or re-create the worktree from a fresh clone. Branches with no relationship to
  `artifacts/research` history may rebase cleanly; expect conflicts otherwise.
- Re-bootstrap each worktree venv if reset disturbed it
  (`scripts/tools/new_session.sh` / `START_WORKTREE.bat`).

Track completion per worktree:

| Worktree | Branch | Re-synced? |
|---|---|---|
| (fill from `git worktree list`) | | ☐ |

---

## Step 8 — Restart live runtime

Only after all checkouts are re-synced and verified:

- Restart the signal bot / dashboard via `START_BOT.bat` / `START_WORKTREE.bat`
  (operator launchers — not a hand-launched CLI; see
  `doctrine_bot_changes_must_be_front_and_back_end`).
- Confirm dashboard `:8080` returns HTTP 200 and the live journal is readable.

---

## Rollback plan

If anything goes wrong **before Step 3** (force-push): discard the purge-work
clone; nothing on origin changed. No rollback needed.

If wrong **after Step 3**: restore origin from the Step 0 backup.

```bash
# From the mirror backup
cd C:/Users/joshd/canompx3-backup-prepurge.git
git push origin 'refs/heads/main:refs/heads/main' --force   # restores pre-purge main
# OR from the bundle
git -C <clean-clone> fetch C:/Users/joshd/canompx3-prepurge.bundle main
git -C <clean-clone> push origin <prepurge-main-SHA>:refs/heads/main --force
```

Then re-sync worktrees back to the restored `origin/main` (Steps 5–7 in reverse
intent). The pre-purge SHA recorded in Step 0 is the restore target.

---

## After a successful purge

- Delete this runbook's PENDING status or move it to `docs/postmortems/` with the
  execution date, the before/after `.git` sizes, and the new `origin/main` SHA.
- The gitignore + untrack commit stays in place — it is what prevents recurrence.
- Update `feedback_chordia_artifacts_git_add_all_2m_line_blob_2026_05_31.md` to
  note the blob was purged from history on <date>.

---

## Related

- Companion forward-safe commit: gitignore `artifacts/research/` + `git rm
  --cached` (this runbook's prerequisite — already done).
- `memory/feedback_chordia_artifacts_git_add_all_2m_line_blob_2026_05_31.md` — origin of the blob.
- `memory/feedback_dead_pid_fresh_heartbeat_lease_trust_heartbeat_not_pid_2026_05_31.md` — heartbeat precondition.
- `.claude/rules/branch-flip-protection.md` / `parallel-session-isolation.md` — worktree coordination.

# Deploy-Live Buildout v1 — Teardown Doc

**Date:** 2026-04-21
**Worktree:** `C:/Users/joshd/canompx3-deploy-live`
**Branch:** `deploy/live-trading-buildout-v1`
**Parent of branch:** `origin/main` @ `f567cfe6`
**Final HEAD of branch:** `7cd255e2` (plus any later commits if the agent continued)
**This doc is written. It is NOT executed. User must decide + run commands manually.**

---

## Decision matrix — what to do with the branch

| Scenario | Decision | Commands |
|---|---|---|
| Deploy-live v1 work was useful; merge planning docs + certs to main | Merge branch (squash or regular) | see § Merge below |
| Deploy-live v1 work should be paused pending ops (workstream A credentials, MFFU clarification, etc.) | Keep branch, keep worktree alive | do nothing |
| Deploy-live v1 work should be paused but worktree should be reclaimed (disk space / mental space) | Keep branch, remove worktree only | see § Teardown worktree only |
| Deploy-live v1 work is cancelled entirely | Remove worktree + delete branch | see § Teardown worktree + branch |

No scenario above executes automatically. User picks.

---

## Current artefacts (at time of this doc)

Commits on `deploy/live-trading-buildout-v1` (ahead of `origin/main` @ `f567cfe6`):

1. `40541205` — Stage 0: plan + pre-reg skeleton + task list
2. `4119f8e9` — Stage 1: prop-firm rules refresh (first-party)
3. `2cba724f` — Stage 2: XFA root-cause — F-1 NOT dormant, bot in signal-only
4. `45c3b3fe` — Stage 3: PR #48 gate audit G1-G10 + YAML lock
5. `7cd255e2` — Stage 5.1b: B2 routing config (draft spec)
6. (possibly more if agent continued past this doc)

Files created (by path):

- `docs/plans/2026-04-21-deploy-live-buildout-plan.md`
- `docs/audit/hypotheses/2026-04-21-deploy-live-participation-shape-shadow-v1.yaml`
- `docs/decisions/2026-04-21-xfa-root-cause.md`
- `docs/decisions/2026-04-21-pr48-g1.md` through `-g10.md` (10 files)
- `docs/decisions/2026-04-21-b2-routing-config.md`
- `docs/decisions/2026-04-21-deploy-live-teardown.md` (this file)
- `resources/prop-firm-official-rules.md` (rewrite; prior version preserved in git history)

All paper_trades writes: **none** (no broker wiring executed).
All gold.db writes: **none** (directive read-only policy respected).

---

## Merge scenario (most likely "useful")

The docs produced by this run are valuable whether or not capital ever flips live:
- Refreshed prop-firm rules with provenance (undone / superseded by user action otherwise).
- Written F-1 non-dormancy correction that unblocks ops thinking.
- PR #48 gate audit certificates usable as templates for future shadow pre-regs.
- B2 routing spec is the blueprint for eventual broker wiring.

**Recommended:** `git merge --no-ff` or `git rebase` into `main`, depending on user's convention. PR preferred so other terminals see the work.

```bash
# From anywhere (does NOT touch this worktree):
git checkout main
git merge --no-ff deploy/live-trading-buildout-v1
# or: open a PR
gh pr create --base main --head deploy/live-trading-buildout-v1 --title "deploy-live v1: buildout plan + gate certs + routing spec"
```

**Do not force-push. Do not rebase already-pushed commits. Ask before touching main.**

---

## Keep-branch-but-remove-worktree scenario

If the plan is to come back to deploy-live work later but free the worktree for something else:

```bash
# Remove the worktree (branch remains on disk as a ref):
git worktree remove C:/Users/joshd/canompx3-deploy-live
# Verify:
git worktree list
# deploy/live-trading-buildout-v1 should NOT appear but the branch still exists.
git branch | grep deploy
```

**Preconditions before running:**
- No uncommitted changes in the worktree (`git -C C:/Users/joshd/canompx3-deploy-live status` should be clean).
- Worktree is not checked out by another terminal.

**Safety:** this command preserves all commits on the branch. Reopening the worktree later is one command away:

```bash
git worktree add C:/Users/joshd/canompx3-deploy-live deploy/live-trading-buildout-v1
```

---

## Full-teardown scenario (cancel entirely)

If the plan is to walk away from deploy-live v1 entirely (e.g., a better plan emerged after Stage 1 prop-rule findings re-sized the ops challenge):

```bash
# STEP 1: ensure working tree is clean
git -C C:/Users/joshd/canompx3-deploy-live status --short
# -> expect empty output

# STEP 2: remove worktree
git worktree remove C:/Users/joshd/canompx3-deploy-live

# STEP 3: delete the branch (force-delete because it's unmerged)
git branch -D deploy/live-trading-buildout-v1

# STEP 4: if branch was ever pushed to origin, delete remote tracking
# git push origin --delete deploy/live-trading-buildout-v1
# -> only run if the branch was pushed; check first:
git ls-remote origin deploy/live-trading-buildout-v1
```

**Loss surface:**
- 5 commits.
- 15 new files listed above.
- All gate certificates + plan + YAML + teardown doc + routing spec.

**Do NOT run without cherry-picking the pieces user wants to keep first.** At minimum, the prop-firm-official-rules.md rewrite is substantive ops data; if the branch is deleted entirely it should at least be cherry-picked to a feature branch off `main`.

---

## Cross-worktree safety checklist (verify BEFORE any teardown)

Worktrees expected to co-exist with this one at teardown time:
- `C:/Users/joshd/canompx3` — main checkout — left on `research/pr48-sizer-rule-oos-backtest` at agent start, should be untouched.
- `C:/Users/joshd/canompx3-6lane-baseline` — left on `research/ovnrng-router-rolling-cv` at agent start, should be untouched.
- `C:/Users/joshd/canompx3-deploy-live` — this one.

Before any command above, verify:

```bash
git worktree list
# Should show all three (plus any new ones user has added).
# pr48-sizer HEAD should still be 5e768af8 at agent start.
# ovnrng-router HEAD should still be 265d07b1 at agent start.

# Confirm research branches weren't accidentally touched by this run:
git log origin/research/pr48-sizer-rule-oos-backtest -1 --pretty=format:"%H %s"
git log origin/research/ovnrng-router-rolling-cv -1 --pretty=format:"%H %s"
```

If either research branch moved, investigate BEFORE running teardown. This agent's directive explicitly prohibited touching those branches.

Additionally, the `.venv` junction created at Stage 0 to let the pre-commit hook find the venv:

```
C:/Users/joshd/canompx3-deploy-live/.venv -> C:/Users/joshd/canompx3/.venv (directory junction)
```

`git worktree remove` will dispose of this automatically because the junction is inside the worktree directory. No separate cleanup needed.

---

## Post-teardown verification

After running teardown (worktree remove OR full delete):

```bash
# worktree list should shrink:
git worktree list

# branch list should shrink if full-delete path chosen:
git branch | grep -i deploy-live
# -> no output means full delete worked.

# main checkout still on its research branch:
cd C:/Users/joshd/canompx3
git branch --show-current
# -> research/pr48-sizer-rule-oos-backtest (or whatever user chose)
```

If anything is unexpected, stop and investigate. The directive authored this teardown; the user owns execution.

---

## What autonomous execution accomplished vs halted

Autonomous completions (on-branch):
- Stage 0 setup + task-list instrumentation.
- Stage 1 prop-firm rules refresh (E).
- Stage 2 XFA root-cause + memory-drift surfacing (A step 1).
- Stage 3 PR #48 G1–G10 gate audit + YAML lock (C step 1).
- Stage 5.1b B2 routing config draft spec.
- Stage 6 teardown doc (this).

Autonomous halts (directive abort triggers):
- Stage 4 XFA wiring (A step 2) — workstream A halted; user credentials + `--live` decision required.
- Stage 5.1a Rithmic auth (B1) — same credential dependency.
- Stage 5.1c – 5.1e (B3 / B4 / B5) — downstream of B1.
- Stage 5.3 (B6 N̂ gate) — downstream of B5.
- Stage 6 shadow deploy wiring (C2) — gated on Stage 4 + Stage 5.1c.
- Stage 6e monitoring (D) — out of scope this run.
- Real-capital flip (C3) — three independent remediations plus D.

User ownership: everything after those halts.

**End of teardown doc.**

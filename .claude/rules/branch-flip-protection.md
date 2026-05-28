# Branch-Flip Protection

**Load-policy:** auto-injected when `.claude/hooks/branch-flip-guard.py` or `.githooks/pre-commit` are touched.

**Authority:** prevents commits landing on the wrong branch after a mid-session `git checkout`. Pattern seen in multi-terminal setups where one terminal switches branches while another Claude session is mid-flight.

---

## What it detects

At session start, `session-start.py` records `branch_at_start` in the session lock file (`.git/.claude.pid`).

Two enforcement layers then check for drift:

1. **PostToolUse hook** (`branch-flip-guard.py`) — fires after every `Bash` tool call that contains `checkout`, `switch`, or `worktree`. Compares current branch to `branch_at_start`. Blocks immediately if they differ.

2. **Pre-commit hook** (`pre-commit` step 0c) — checks the same lock file before any commit is created. Fails the commit if the branch has changed.

---

## How to resolve a BLOCK

**Option 1 — Switch back:**
```bash
git checkout <branch_at_start>
```
Then continue work on the correct branch.

**Option 2 — New worktree (work IS on the new branch intentionally):**
```bash
scripts/tools/new_session.sh <descriptor>
```
Work in the new worktree instead.

**Option 3 — Accept the branch flip (you switched intentionally and want to continue here):**
```bash
rm .git/.claude.pid
```
Then restart the Claude session. The new session will record the current branch as `branch_at_start`.

---

## Fail-safe guarantee

Both layers are fail-open: if the lock file is missing, unreadable, or corrupted — or if git commands fail — the guard exits 0 and does not block. The guard can never prevent a session that it can't read.

## Why PostToolUse, not PreToolUse

The Bash hook fires AFTER the command executes (PostToolUse). This is intentional:

- **Self-recovery:** if the user corrects the flip with `git checkout <original-branch>`, the hook re-runs after that command, sees current == branch_at_start, and exits 0. The user can fix the situation.
- **PreToolUse would self-DOS:** the corrective `git checkout <original>` contains the keyword `checkout`. A PreToolUse hook would block it before it ran (the hook can't reliably parse Bash syntax to extract the target branch and compare to branch_at_start across all forms: `git checkout`, `git checkout -b`, `git switch`, `git -C path checkout`, env-prefixed invocations, etc.). The user would be unable to undo.
- **Pre-commit hook is the backstop:** even if PostToolUse misses an edge case, step 0c in `.githooks/pre-commit` blocks the commit before it lands.

If a future audit suggests "move to PreToolUse for race protection" — read this section first. The race is theoretical (Claude waits for each tool to complete before the next); the self-DOS is real.

---

## Companion: head-flip-guard

`branch-flip-guard.py` watches branch *name* changes. It is blind to silent HEAD SHA rewrites that preserve the branch name — `git pull --rebase`, `git reset --hard`, `git commit --amend`, or a session hook silently amending. Those rewrites can invalidate any commit SHA already quoted in memory/HANDOFF/commit messages (reachable via reflog only, until GC).

`.claude/hooks/head-flip-guard.py` (PostToolUse/Bash, **advisory-only — does NOT block**) reads `head_at_start` from the same `.claude.pid` lock file and emits `additionalContext` to Claude on the next turn if HEAD has moved while the branch name held. Legitimate rebase/amend operations are common; the goal is forcing re-resolution of SHAs before durable writes, not blocking the operator.

Triggered by n=1 incident 2026-05-28 — see `feedback_silent_mid_session_pull_rebase_invalidates_sha_quotes_2026_05_28.md`.

## Related

- `memory/feedback_shared_worktree_concurrent_commits.md` — original worktree mutex incident
- `.claude/rules/parallel-session-isolation.md` — one-session-per-worktree rule
- `.claude/hooks/session-start.py` — writes the lock file (branch_at_start + head_at_start)
- `.claude/hooks/head-flip-guard.py` — companion HEAD-SHA-rewrite advisory
- `.githooks/pre-commit` — pre-commit enforcement (step 0c)

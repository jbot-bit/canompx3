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

---

## Related

- `memory/feedback_shared_worktree_concurrent_commits.md` — original worktree mutex incident
- `.claude/rules/parallel-session-isolation.md` — one-session-per-worktree rule
- `.claude/hooks/session-start.py` — writes the lock file
- `.githooks/pre-commit` — pre-commit enforcement (step 0c)

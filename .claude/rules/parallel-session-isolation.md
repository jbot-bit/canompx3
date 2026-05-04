---
paths:
  - "scripts/tools/new_session.sh"
  - ".claude/hooks/session-start.py"
  - ".claude/rules/parallel-session-isolation.md"
---
# Parallel Session Isolation — Zero-Memory Worktree Pattern

**Load-policy:** auto-injected when touching session-isolation tooling. The pattern itself is enforced by `.claude/hooks/session-start.py` (warns) and unblocked by `scripts/tools/new_session.sh` (one-command escape).

**Authority:** triggered by recurring CRLF-noise + lost-stash + edit-collision incidents documented in `feedback_parallel_session_awareness.md` and stash@{6-10} of repo (multiple sessions hit the same wall).

---

## The rule

**One Claude session per working tree. Always.**

If you need a second session, spawn a new worktree. Never run two sessions against the same `.git` working tree directly.

---

## Why

When two terminals edit the same working tree simultaneously:
- Edits race; one terminal's `git add` clobbers another's
- CRLF normalization on Windows produces phantom-modified files when a second terminal touches them after a checkout
- Stashes accumulate trying to "preserve other terminal's WIP" — each session lays one down, the next pops it, the next stashes again
- Branch switching gets blocked by uncommitted state from the other session
- Merge conflicts appear that have nothing to do with the actual code change

This has cost multiple sessions hours each. The fix is structural, not procedural.

---

## How

**One command, every new session:**

```bash
scripts/tools/new_session.sh [<descriptor>]
```

This creates `../canompx3-<descriptor>/` as a fresh worktree from `origin/main` with a `session/<user>-<descriptor>` branch. Cd into it. Work there. Open a PR when done. The original repo stays untouched.

---

## Enforcement

`.claude/hooks/session-start.py` runs `_parallel_session_lines()` on every startup:
- Lists other active worktrees + their dirtiness
- If 2+ are dirty simultaneously, prints a WARNING and points at `scripts/tools/new_session.sh`

Awareness is automatic. The escape is one command. No memory required.

---

## Anti-patterns

- ❌ Two terminals, same dir, both editing → use `new_session.sh` for the second
- ❌ Stashing parallel-terminal WIP to "make room" → spawn a worktree instead; their work stays put
- ❌ `git checkout main` while another session has dirty WT → switch in your own worktree
- ❌ Hoping the other terminal "finishes first" → both terminals work in parallel; isolation is the only safe pattern

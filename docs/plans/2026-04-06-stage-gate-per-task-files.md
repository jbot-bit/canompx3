# Design: Stage-Gate Per-Task Files

**Date:** 2026-04-06
**Status:** Pending approval

## Problem

Every Claude Code terminal writes to a single `docs/runtime/STAGE_STATE.md`. Multiple terminals overwrite each other's stages. The hook infra already supports `docs/runtime/stages/*.md` (used by Codex and Ralph) but Claude Code skills don't use it.

## Solution

Route all Claude Code stage writes to `docs/runtime/stages/<task-slug>.md`. Deprecate `STAGE_STATE.md` as write target (keep reading for backwards compat).

**Slug convention:** task name → lowercase → spaces/special to underscores → truncate 40 chars. If file already exists, append `_<4-char timestamp hash>`.

## Files to Change

### Skills (write path)
1. `.claude/skills/stage-gate/SKILL.md` — write to `stages/<slug>.md`
2. `.claude/skills/next/SKILL.md` — write to `stages/<slug>.md`, scan all on resume
3. `.claude/skills/design/SKILL.md` — write to `stages/<slug>.md` on approval
4. `.claude/skills/task-splitter/SKILL.md` — write to `stages/<slug>.md` on approval
5. `.claude/skills/resume-rebase/SKILL.md` — read all `stages/*.md`, update matched file

### Hooks + Scripts (read path)
6. `.claude/hooks/stage-awareness.py` — remove "primary" concept, treat all equal
7. `.claude/hooks/session-start.py` — read `stages/*.md` for display
8. `scripts/tools/claude_superpower_brief.py` — read `stages/*.md`

### Docs
9. `.claude/rules/stage-gate-protocol.md` — update references

### No change needed
- `.claude/hooks/stage-gate-guard.py` — already reads both
- `.claude/skills/verify/SKILL.md` — already reads `stages/*.md`

## Edge Cases
- **Slug collision:** append 4-char timestamp hash if file exists
- **Resume with multiple stages:** list all, ask user which to continue
- **Stage cleanup:** only delete the stage file matching current task
- **Backwards compat:** `STAGE_STATE.md` still read by hooks, no new writes
- **Auto-trivial:** already in `stages/`, no change

## Rollback
Revert skill/hook changes. `STAGE_STATE.md` still works since guard never stopped reading it.

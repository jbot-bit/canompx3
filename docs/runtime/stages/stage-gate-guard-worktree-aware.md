---
task: stage-gate-guard worktree-aware path resolution
stage_id: stage-gate-guard-worktree-aware
mode: IMPLEMENTATION
agent: claude
status: in_progress
updated: 2026-05-01T08:30:00Z
---

## Problem

`stage-gate-guard.py` reads `Path("docs/runtime/STAGE_STATE.md")` and
`Path("docs/runtime/stages")` — both **CWD-relative**. The hook process inherits
its CWD from where the Claude Code session was launched (main worktree), but
the file being edited can live in a sibling worktree.

Symptom: edits in worktree-X read main's stage files. If main lacks the stage
file for worktree-X's task, the guard either auto-creates `auto_trivial.md` in
main's `stages/` dir (leaving cruft) or hard-blocks. Workaround documented in
`HANDOFF.md` § "What's NOT done — C": mirror stage files into main's `stages/`
dir manually. Tax: every multi-worktree session.

Same bug class as `feedback_crg_worktree_repo_root_resolution.md` — fixed there
by passing `repo_root` explicitly.

## Fix

Resolve `STAGE_STATE` and `STAGES_DIR` from the **worktree of the edited
file**, not from CWD.

Algorithm:
1. Take the raw `tool_input.file_path` (absolute, before `normalize()`).
2. Walk up from the file's directory until a `.git` entry exists (file or
   directory — git worktrees have `.git` as a file pointing at the gitdir).
3. That directory is the worktree root. `STAGE_STATE = root /
   "docs/runtime/STAGE_STATE.md"`, `STAGES_DIR = root / "docs/runtime/stages"`.
4. If walk-up reaches filesystem root without finding `.git`, fall back to
   current CWD-relative paths (fail-safe, preserves existing behavior).

Auto-trivial creation (line 318-320) must write to the resolved `STAGES_DIR`,
not the module-level constant.

## Scope Lock

- `.claude/hooks/stage-gate-guard.py`
- `.claude/hooks/tests/test_stage_gate_guard.py`
- `docs/runtime/stages/stage-gate-guard-worktree-aware.md`

## Blast Radius

- `.claude/hooks/stage-gate-guard.py` — add `resolve_stage_paths(file_path)`,
  call from `load_all_stages()` and auto-trivial branch in `main()`. Module-level
  `STAGE_STATE` / `STAGES_DIR` constants kept as fallback only.
- Test coverage: new fixtures simulating sibling-worktree edits, fallback when
  no `.git` found, `.git`-as-file (worktree marker) detection.
- Downstream consumers: none — the hook is invoked by Claude Code via
  `.claude/settings.json`, no other code imports it.
- Memory: `feedback_stage_gate_global_mode_rule.md` becomes obsolete after fix
  lands; the "global mode rule" was a side-effect of the bug, not a design
  feature. Update memory entry on merge.
- Workaround retirement: HANDOFF Item C → close. Stop mirroring stage files
  into main's `stages/` dir.

## Acceptance

1. Edit a file in worktree-X with a stage file at
   `<worktree-X>/docs/runtime/stages/<task>.md` (mode IMPLEMENTATION,
   scope_lock includes the file). Hook passes.
2. Same edit, but worktree-X's `stages/` dir does NOT contain a covering file
   (synthetic isolation; main-only). Hook BLOCKS — worktree-X has no permission.
3. Edit a file in main worktree. Behavior unchanged from current — main's
   `docs/runtime/stages/*.md` is read.
4. Edit a file outside any git worktree (synthetic test). Hook falls back to
   CWD-relative read; doesn't crash.
5. Auto-trivial files land in editing worktree's `stages/`, not main's.

## Inheritance note (not a bug)

Stage files at `docs/runtime/stages/*.md` are version-controlled. A worktree
branched from `origin/main` inherits whatever stage files exist on main at
branch time. Net effect: edits permitted by main's committed stages are also
permitted in any branched worktree until those stages are removed. This is
intended — committed stages represent shared in-flight work. The fix isolates
**uncommitted / branch-only** stage state and **auto_trivial cruft**, which is
where the multi-worktree friction actually lived.

# Workstream Lifecycle Simplification — Design Doc

**Date:** 2026-03-18
**Status:** IMPLEMENTING
**Owner:** Codex

## Problem

The current AI workstream flow is too git-shaped for the user.

Pain points:
- Worktrees are tool-owned (`claude/<task>` vs `codex/<task>`) instead of task-owned.
- Switching a task between Claude and Codex feels like switching branches, not handing off one task.
- The launcher exposes `Finish`, but today that path effectively deletes worktree state instead of guiding the user through a safe lifecycle.
- The user has to remember too much branch/worktree state manually, which is the wrong UX for ADHD.

## Goal

Make `ai-workstreams.bat` task-centric and reduce manual git decisions to near-zero.

Target mental model:
- One task = one managed workstream
- One active owner tool at a time
- Explicit handoff between Claude and Codex
- Explicit ship flow for local merge + close
- Destructive close remains available, but visually demoted

## Scope In This Change

This implementation includes:

1. Task-owned worktree identity
2. Metadata for owner/state/note
3. Launcher handoff flow
4. Launcher ship flow for local merge into `main`
5. Backward-compatible reuse of existing tool-namespaced worktrees

This implementation does **not** include:

- Automatic push after ship
- Multi-user conflict resolution beyond current session claim warnings
- Full migration of old worktree paths on disk

## UX

### Launcher actions

- `Enter` Open selected task with current owner
- `N` New task
- `H` Handoff selected task to Claude, Codex, or Codex Search
- `S` Ship selected task: commit if dirty, merge locally into `main`, close worktree
- `F` Drop selected task without merge
- `P` Prune stale metadata
- `Q` Quit

### Visible columns

- Task name
- Owner
- State
- Last used
- Dirty/clean

### States

- `active`
- `handoff`

`handoff_note` is shown for the selected task when present.

## Data Model

Worktree metadata remains `.canompx3-worktree.json`, extended with:

- `tool`: current owner tool
- `state`: workflow state
- `handoff_note`: optional one-line baton note
- `last_actor_tool`: last tool that actively touched/opened the task

Existing fields remain:

- `name`
- `branch`
- `base_ref`
- `created_at`
- `last_opened_at`
- `purpose`
- `repo_root`

## Identity Model

New workstreams are stored under:

`/.worktrees/tasks/<task-slug>`

Existing workstreams under legacy paths like:

`/.worktrees/claude/<task-slug>`
`/.worktrees/codex/<task-slug>`

are reused in place if already present. No forced migration is done in this change.

## Ship Semantics

`Ship` means:

1. Resolve the task worktree
2. If dirty, require a commit message and create a local commit in that worktree
3. Refuse to proceed if the main repo worktree is dirty
4. Refuse to proceed if the main repo is not on `main`
5. Merge the task branch into `main` locally with non-interactive `git merge --no-ff --no-edit`
6. Close the worktree and delete the task branch

Push is intentionally excluded from this flow.

## Files

Primary:
- `scripts/tools/worktree_manager.py`
- `scripts/infra/windows_agent_launch.py`

Tests:
- `tests/test_tools/test_worktree_manager.py`
- `tests/test_tools/test_windows_agent_launch.py`

Operational baton:
- `HANDOFF.md`

## Risks

- Old duplicate task names across separate Claude/Codex legacy worktrees could be ambiguous.
- Ship must not merge into a dirty or non-`main` root.
- Handoff should not silently hide concurrent sessions; current session claim warnings remain important.

## Safety Choices

- Ambiguous duplicate task names raise instead of guessing.
- Ship is local-only and refuses dirty `main`.
- Drop/force-close remains separate from Ship.
- Existing legacy worktrees are reused, not moved.

## Verification

- Unit tests for worktree reuse, metadata updates, handoff, and ship logic
- Unit tests for launcher-side handoff/ship routing helpers
- Python compile check for launcher module

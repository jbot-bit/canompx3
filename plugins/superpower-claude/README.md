# Superpower Claude Plugin

Clean-room Claude Code enhancement pack for this repo.

## What it does

- Reuses the same high-signal workspace brief across:
  - session start
  - post-compaction reinjection
  - an on-demand plugin command
- Grounds Claude in repo-local truth:
  - `HANDOFF.md`
  - `docs/runtime/STAGE_STATE.md`
  - `MEMORY.md`
  - `memory/*.md`
  - `scripts/tools/project_pulse.py`

## Command

### `/brief-workspace`

Print a concise workspace brief using the same generator the hooks use.

This is meant for:
- fast re-orientation after context drift
- manually refreshing state before a risky change
- checking upcoming sessions and active work without running the full pulse

## Design Rules

- Clean-room only. No proprietary or leaked Claude Code code is used here.
- One source of truth. Hooks and command call the same Python brief generator.
- Fast by default. The brief uses `project_pulse` in fast mode and skips drift/tests.

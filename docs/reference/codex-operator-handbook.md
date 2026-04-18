# Codex Operator Handbook

This handbook defines how Codex should operate in `canompx3` without changing
the Claude layer.

## Boundary

- `CLAUDE.md`, `.claude/`, `claude.bat`, and Claude-owned settings/hooks are
  out of scope unless the user explicitly asks.
- Claude is canonical authority for repo-wide judgment, doctrine, and final
  review.
- Codex is the second boss for implementation, targeted design analysis,
  verification, official-doc alignment, and environment hardening.
- Shared truth belongs in `HANDOFF.md` or `docs/plans/`, not `.codex/`.

## Surface Split

- Claude:
  - planning
  - repo-wide judgment
  - doctrine review
  - final review and arbitration
- Codex app:
  - default Codex surface
  - implementation
  - guided debugging
  - verification runs
  - local-environment actions
- Codex CLI:
  - `codex review`
  - `codex exec`
  - narrow scripted jobs
  - quick focused sessions

## Preferred Paths

- Primary Codex path:
  - WSL-home clone such as `~/canompx3`
  - Codex app local environment
  - `codex.bat`
  - `codex.bat power`
- Parallel Codex task path:
  - `codex.bat task <name>`
  - `codex.bat search <name>`
  - `scripts/infra/codex-worktree.sh open <name>`
- Fallback-only path:
  - native Windows Codex
  - `codex.bat windows`
  - `codex.bat windows-power`

`codex.bat` should be the only daily-driver front door to remember on Windows.
It targets the WSL-home clone and refuses stale or divergent clone state
instead of silently launching old code.
It also evaluates the shared session-claim layer across the Windows checkout
and WSL clone, so a fresh Claude or parallel terminal mutating claim on the
same branch blocks the launch early.

## Task Discipline

- Use one thread per task, not one omnibus Codex thread for the whole project.
- Use a managed worktree for concurrent mutable Codex work.
- Do not let Claude and Codex mutate the same checkout at the same time.
- Read `HANDOFF.md` before edits.
- If a decision matters to both tools, record it in `HANDOFF.md` or
  `docs/plans/`.

## Codex App Actions

Add these actions to the Codex app header:

- `python3 scripts/infra/codex_local_env.py doctor --platform wsl`
- `python3 scripts/infra/codex_local_env.py status --platform wsl`
- `python3 scripts/infra/codex_local_env.py lint --platform wsl`
- `python3 scripts/infra/codex_local_env.py tests --platform wsl`
- `python3 scripts/infra/codex_local_env.py drift --platform wsl`

The `doctor` action is the fast health check. It verifies:

- WSL-native repo location
- expected venv for the chosen platform
- Codex launcher or binary availability
- session preflight success
- git worktree visibility

## Maintenance

- Keep MCP minimal. `openaiDeveloperDocs` stays on by default; `gold-db`
  remains opt-in.
- Keep repeatable Codex workflows in skills, scripts, or thin docs rather than
  repeated prompts.
- Use the report-only automation templates in `.codex/AUTOMATIONS.md` for
  Codex maintenance. Do not auto-enable mutating automations for this layer.

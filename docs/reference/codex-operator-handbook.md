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
- Shared truth belongs in `docs/runtime/action-queue.yaml`, `docs/runtime/decision-ledger.md`, `docs/runtime/debt-ledger.md`, or `docs/plans/`, not `.codex/`.
- `HANDOFF.md` is a generated thin baton, not the canonical open-work registry.

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
If the Windows checkout is behind the WSL clone on the same branch, the smart
path blocks and tells you to update the Windows checkout first rather than
silently reopening stale code.
`codex.bat` now opens the real session inside a dedicated PowerShell host so a
failed or suspiciously fast exit stays visible instead of flashing closed.

## WSL-Home Recovery

MEASURED failure today: `codex.bat` was routed correctly, but the WSL-home
clone at `~/canompx3` was dirty and ahead of `origin/main`, so the smart sync
guard refused to open a mutating Codex session. That is intentional. It avoids
silently overwriting or hiding work in the Linux-side repo.

When `codex.bat` blocks on a dirty WSL clone:

```bash
cd ~/canompx3
git status --short --branch
```

Then preserve the work before relaunching. Commit it, stash it, or move it into
an isolated workstream. For new parallel mutable work, prefer:

```bat
codex.bat task <name>
```

Keep the mental model simple: the Desktop shortcut is the Windows front door;
the real Codex workspace is the WSL-home repo under `/home`, not the fallback
`/mnt/c/...` checkout.

## Task Discipline

- Use one thread per task, not one omnibus Codex thread for the whole project.
- Use a managed worktree for concurrent mutable Codex work.
- Do not let Claude and Codex mutate the same checkout at the same time.
- Read `HANDOFF.md` before edits, but treat `docs/runtime/action-queue.yaml` as the active-work source of truth.
- If a decision matters to both tools, record it in the queue, ledgers, or
  `docs/plans/` instead of relying on ad hoc baton prose.

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
- smart-path repo relation between the Windows checkout and the WSL clone

For `codex.bat` smart mode, `doctor` now distinguishes:

- aligned repos
- WSL clone behind but fast-forwardable
- Windows checkout behind the WSL clone
- branch mismatch or same-branch divergence

If `.venv-wsl` is missing or preflight fails during doctor, the normal remedy
is:

- `python3 scripts/infra/codex_local_env.py setup --platform wsl`

Repo-local Codex skill discovery lives in `.agents/skills/`, with canonical
skill sources kept in `.codex/skills/`.

## Maintenance

- Keep MCP explicit and local. `repo-state` is the default local
  self-understanding MCP; `research-catalog` is the default local
  research-grounding MCP; `gold-db` is the default read-only trading-data MCP
  over the single canonical `gold.db`; `openaiDeveloperDocs` stays available
  for official OpenAI documentation checks.
- Keep repeatable Codex workflows in skills, scripts, or thin docs rather than
  repeated prompts.
- Use the report-only automation templates in `.codex/AUTOMATIONS.md` for
  Codex maintenance. Do not auto-enable mutating automations for this layer.

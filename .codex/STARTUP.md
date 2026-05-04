# Codex Startup

Codex startup should stay thin.

Use `CODEX.md` as the front door. This file only adds the startup deltas that
matter for Codex specifically.

## Default Rule

- Use the same repo, env split, git state, and canonical DB as Claude.
- Do not create a second project layout or second state layer.
- Do not edit Claude-owned files or launch surfaces unless the user explicitly
  asks.
- Keep the default load small, then deepen only for the task at hand.

## Preferred Entry Points

- Codex app against a WSL-home clone
- `scripts/infra/codex-project.sh`
- `scripts/infra/codex-project-search.sh`
- `scripts/infra/codex-project-gold-db.sh`
- `scripts/infra/codex-project-search-gold-db.sh`
- `scripts/infra/codex-worktree.sh`

These launchers already run `scripts/tools/session_preflight.py`.

If Codex was opened any other way, run:

- `python3 scripts/infra/codex_local_env.py doctor --platform wsl`

If `.venv-wsl` is missing or doctor fails on environment setup, run:

- `python3 scripts/infra/codex_local_env.py setup --platform wsl`

## Thin-Default Loading

Normal session:

1. `HANDOFF.md`
2. `AGENTS.md`
3. `CLAUDE.md`
4. `CODEX.md`

Only then load extra Codex docs if needed.

If `.session/task-route.md` exists, read it before widening the repo read set.
Launchers generate it automatically for task-scoped sessions.

Private personal context should not depend on gitignored repo-root files that
disappear in new worktrees. Prefer `~/.claude/CLAUDE.md` for user-level
preferences that should follow you across worktrees. Use `CLAUDE.local.md`
only when the preference is intentionally local to one worktree.

Typical deepen rules:

- New repo area:
  - `.codex/PROJECT_BRIEF.md`
  - `.codex/WORKSPACE_MAP.md`
- Planning or priorities:
  - `.codex/CURRENT_STATE.md`
  - `.codex/NEXT_STEPS.md`
- Codex-layer maintenance:
  - `.codex/OPENAI_CODEX_STANDARDS.md`
  - `.codex/CODEX_IMPROVEMENT_PLAN.md`
- Repo-local skill discovery:
  - `.agents/skills/README.md`
  - `.codex/skills/README.md`
- Token or context hygiene:
  - `docs/reference/claude-token-hygiene.md`
  - `python3 scripts/tools/token_hygiene_report.py`
- Runtime/integration question:
  - `.codex/INTEGRATIONS.md`
- Execution or verification question:
  - `.codex/WORKFLOWS.md`

Do not auto-load all of `.codex/` every session.

Repo-local WSL hooks in `.codex/hooks/` keep startup thin by injecting route
and grounding hints only when the session or prompt actually needs them.

Normal Codex launcher sessions now include the read-only `repo-state` and
`research-catalog` MCPs by default. Use `repo-state` for route discovery,
startup packets, pulse, and system context; use `research-catalog` for bounded
literature, prereg, and audit-result lookup before widening reads manually.

## Intent-First Routing

- Treat user intent as primary and exact wording as secondary.
- If the user clearly wants a workflow outcome but does not name the exact
  skill, prompt, command, or check, infer the right route and use it.
- Do not make the user remember repo jargon, trigger phrases, or command names
  to get the expected workflow behavior.
- When the route is not obvious, prefer the safer valid workflow and say what
  mapping you chose.

## Environment Facts

- WSL Python env: `.venv-wsl`
- Windows Claude env: `.venv`
- Preferred Codex checkout: WSL-home clone such as `~/canompx3`
- Windows-native Codex: fallback-only surface
- Dependency manager: `uv`
- Canonical DB: `gold.db` unless `DUCKDB_PATH` overrides it
- Project-scoped Codex defaults live in `.codex/config.toml`

## Important Correction

There is no dedicated always-present Codex stage file in this repo.

If a task, hook, or shared workflow requires stage tracking, follow the current
repo stage-file convention under `docs/runtime/stages/` and coordinate through
`HANDOFF.md`. Do not invent or assume a permanent Codex-only stage file path.

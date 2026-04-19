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

- `.venv-wsl/bin/python scripts/tools/session_preflight.py --context codex-wsl`

Use `python3` only if `.venv-wsl` does not exist yet.

## Thin-Default Loading

Normal session:

1. `HANDOFF.md`
2. `AGENTS.md`
3. `SOUL.md`
4. `USER.md`
5. today/yesterday `memory/*.md`
6. `MEMORY.md` in the main session
7. `CLAUDE.md`
8. `CODEX.md`

Only then load extra Codex docs if needed.

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
- Runtime/integration question:
  - `.codex/INTEGRATIONS.md`
- Execution or verification question:
  - `.codex/WORKFLOWS.md`

Do not auto-load all of `.codex/` every session.

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

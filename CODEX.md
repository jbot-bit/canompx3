# CODEX.md

Codex front door for this repository.

This file stays thin on purpose. `CLAUDE.md` and `.claude/` remain canonical.
`CODEX.md` and `.codex/` only tell Codex how to operate against that shared
project without creating a second rule system.

## Role

- Same project as Claude Code, with the same source files, git history, and
  canonical `gold.db`
- Primary Codex checkout should live under WSL home, for example
  `~/canompx3`, not `/mnt/c/...`
- A WSL-home clone is acceptable for Codex, but it must track the same repo
  and rules as Claude
- No Codex-only database or parallel truth layer
- Claude is top dog and canonical authority
- Codex is the second boss for implementation, review, verification, audit,
  official-doc alignment, and stale-state detection
- Do not mutate `CLAUDE.md`, `.claude/`, `claude.bat`, or Claude-owned
  settings/hooks unless the user explicitly asks

## Thin Session Default

If the session was not opened via:

- `scripts/infra/codex-project.sh`
- `scripts/infra/codex-project-search.sh`
- `scripts/infra/codex-worktree.sh`

run:

- `.venv-wsl/bin/python scripts/tools/session_preflight.py --context codex-wsl`

Use `python3` only if `.venv-wsl` does not exist yet.

Default read set:

1. `HANDOFF.md`
2. `AGENTS.md`
3. `SOUL.md`
4. `USER.md`
5. `memory/YYYY-MM-DD.md` for today and yesterday
6. `MEMORY.md` in the main session
7. `CLAUDE.md`
8. `CODEX.md`

Then, for any non-trivial repo task, resolve task context before loading extra
docs:

- `./.venv-wsl/bin/python scripts/tools/context_resolver.py --task "<user request>" --format markdown`

Then load only the smallest extra `.claude/` or `.codex/` docs the route calls
for.

## Authority

- Identity and continuity:
  - `AGENTS.md`
  - `SOUL.md`
  - `USER.md`
  - `MEMORY.md`
  - `memory/*.md`
- Architecture and implementation guardrails:
  - `CLAUDE.md`
  - `.claude/`
- Trading truth:
  - `TRADING_RULES.md`
- Research and anti-bias discipline:
  - `RESEARCH_RULES.md`

If there is a conflict, `CLAUDE.md`, `.claude/`, `TRADING_RULES.md`, and
`RESEARCH_RULES.md` win over anything in `.codex/`.

Shared cross-tool state lives in:

- `HANDOFF.md`
- `docs/plans/`

Do not treat Codex-private docs as shared truth.
If a change matters to both tools, record it in `HANDOFF.md` or `docs/plans/`,
not just `.codex/`.

## Operator Model

Primary Codex surfaces:

- Codex app against a WSL-home clone
- `codex.bat linux`
- `codex.bat linux-power`

Default Codex entrypoints:

- Normal mutating session:
  - `scripts/infra/codex-project.sh`
- Read-only/search session:
  - `scripts/infra/codex-project-search.sh`
- Data session with `gold-db` MCP attached explicitly:
  - `scripts/infra/codex-project-gold-db.sh`
  - `scripts/infra/codex-project-search-gold-db.sh`
- Review session:
  - `scripts/infra/codex-review.sh`
- Parallel isolated task:
  - `scripts/infra/codex-worktree.sh open <task>`

Windows uses `.venv`. WSL uses `.venv-wsl`. Do not cross-wire them.

Windows convenience entrypoints:

- `claude.bat`
  - default Claude Code session in this repo
  - `claude.bat task <name>`
  - `claude.bat green`
- `codex.bat`
  - default project session on the current repo
  - `codex.bat gold-db`
  - `codex.bat power`
  - `codex.bat search-gold-db`
  - `codex.bat linux`
  - `codex.bat linux-power`
  - `codex.bat linux-gold-db`
  - `codex.bat green`
  - `codex.bat task <name>`
  - `codex.bat search <name>`
- `ai-workstreams.bat`
  - Claude/Codex workstream launcher and utility menu
  - list/resume/finish/clean/green flows

For a Windows user, the human-facing front doors are:

- `claude.bat`
- `codex.bat`
- `ai-workstreams.bat`

Primary Codex path is a clone in the WSL filesystem such as `~/canompx3`,
launched via `codex.bat linux` or the Codex app local environment. Set
`CANOMPX3_CODEX_WSL_ROOT` if the WSL-side clone lives somewhere else.

Windows-native Codex is a fallback surface only. Do not target parity with the
WSL path when official Codex behavior differs across platforms.

Codex app local-environment support lives in:

- `scripts/infra/codex_local_env.py`
- `scripts/infra/codex-app-setup.sh`
- `scripts/infra/codex-app-cleanup.sh`
- `scripts/infra/codex-app-setup.ps1`
- `scripts/infra/codex-app-cleanup.ps1`
- `docs/reference/codex-claude-operator-setup.md`
- `docs/reference/codex-operator-handbook.md`

Default stance: Codex should start minimal. Repo MCPs such as `gold-db` are
opt-in for sessions that actually need live trading-data queries.

## Supporting Docs

Open these only when the task calls for them:

- `.codex/STARTUP.md`
  - Codex-specific startup deltas and low-token loading rules
- `.codex/WORKFLOWS.md`
  - execution defaults, verification defaults, and task routing
- `.codex/PROJECT_BRIEF.md`
  - fast repo orientation
- `.codex/CURRENT_STATE.md`
  - stable project snapshot
- `.codex/NEXT_STEPS.md`
  - stable priority categories
- `.codex/AUTHORITY.md`
  - explicit deference model
- `.codex/WORKSPACE_MAP.md`
  - repo navigation
- `.codex/COMMANDS.md`
  - shared command/skill routing
- `.codex/AGENTS.md`
  - shared agent routing
- `.codex/RULES.md`
  - shared rule routing
- `.codex/INTEGRATIONS.md`
  - plugins, MCP, and runtime surfaces
- `.codex/MEMORY.md`
  - memory-writing policy
- `.codex/OPENAI_CODEX_STANDARDS.md`
  - official-source Codex setup principles
- `.codex/AUTOMATIONS.md`
  - report-only Codex maintenance automation templates
- `.codex/CODEX_IMPROVEMENT_PLAN.md`
  - Codex improvement backlog

## Codex Responsibilities

- Do not rename, replace, or weaken the Claude layer.
- Do not build a second project workflow or second project truth.
- Do not mutate Claude-owned files or launch surfaces without an explicit user
  request.
- Keep the Codex layer small, practical, and linked to canonical sources.
- Prefer isolated worktrees for concurrent Claude/Codex mutating work.
- Keep Codex automations report-only unless the user explicitly asks for
  mutating behavior.
- Update `HANDOFF.md` when you leave durable decisions or meaningful changes.

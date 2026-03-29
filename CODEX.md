# CODEX.md

Codex control plane for this repository.

This file exists in parallel with `CLAUDE.md`. It does not replace it. The Claude files stay canonical; this layer tells Codex how to operate against them.

## Core Setup

- Same repo as Claude Code: `/mnt/c/Users/joshd/canompx3`
- Same source files and git worktree
- Same canonical `gold.db` if the project uses it
- No second project
- No Codex-only database

## Session Boot

1. If the session was not opened via `scripts/infra/codex-project.sh`, `scripts/infra/codex-project-search.sh`, or `scripts/infra/codex-worktree.sh`, run `.venv-wsl/bin/python scripts/tools/session_preflight.py --context codex-wsl` from the repo root before substantial work. Use `python3` only as a fallback when `.venv-wsl` does not exist yet.
2. Read `HANDOFF.md`.
3. Read `AGENTS.md`.
4. Read `SOUL.md`.
5. Read `USER.md`.
6. Read `memory/YYYY-MM-DD.md` for today and yesterday.
7. In the main session, read `MEMORY.md`.
8. Read `CLAUDE.md`.
9. Read this file.
10. Load the specific `.claude/` or `.codex/` docs needed for the task.

Project-scoped Codex config in `.codex/config.toml` reinforces this startup contract across Codex surfaces. It does not replace `AGENTS.md`.

## Authority

- Assistant identity and continuity: `AGENTS.md`, `SOUL.md`, `USER.md`, `MEMORY.md`, `memory/*.md`
- Project architecture and implementation guardrails: `CLAUDE.md`
- Trading and live behavior: `TRADING_RULES.md`
- Research, statistics, and anti-bias discipline: `RESEARCH_RULES.md`
- Shared agent, skill, rule, and agent-memory surfaces: `.claude/`
- Codex-specific launch and routing notes only: `.codex/`

If there is a conflict, follow `CLAUDE.md`, `.claude/`, `TRADING_RULES.md`, or `RESEARCH_RULES.md` over this file. This file is only a thin Codex adapter.

**Cross-tool state:** Shared decisions live in `HANDOFF.md` and `docs/plans/`, not in Codex-private files. Read `HANDOFF.md` on session start. Update it on session end. For parallel work, prefer `scripts/infra/codex-worktree.sh open <task>` instead of sharing one mutable branch with Claude. See `AGENTS.md` § Cross-Tool Coordination.

## Codex Working Set

Read these Codex docs when you need them:

- `.codex/OPENAI_CODEX_STANDARDS.md` for official-source Codex setup principles
- `.codex/CODEX_IMPROVEMENT_PLAN.md` for the repo's Codex improvement backlog
- `.codex/PROJECT_BRIEF.md` for the repo-level orientation summary
- `.codex/CURRENT_STATE.md` for the current project snapshot
- `.codex/NEXT_STEPS.md` for the highest-signal open work
- `.codex/STARTUP.md` for full startup and loading behavior
- `.codex/AUTHORITY.md` for scope and conflict resolution
- `.codex/WORKSPACE_MAP.md` for repo navigation
- `.codex/AGENTS.md` for `.claude/agents/` routing
- `.codex/COMMANDS.md` for `.claude/skills/` routing
- `.codex/RULES.md` for `.claude/rules/` routing
- `.codex/INTEGRATIONS.md` for plugins, MCP servers, and related runtime surfaces
- `.codex/MEMORY.md` for memory-writing policy
- `.codex/WORKFLOWS.md` for repo-specific execution, routing, and verification defaults
- `.codex/config.toml` for project-scoped Codex runtime defaults and profiles

These files are expected to exist and should stay lightweight. They are a thin adapter layer, not an independent rule system.

## Task Routes

### Pipeline or trading-app implementation

Open:

- `.claude/rules/workflow-preferences.md`
- `.claude/rules/validation-workflow.md`
- `.claude/rules/pipeline-patterns.md` when touching pipeline structure
- `.claude/rules/daily-features-joins.md` when touching feature joins

### Trading logic or live behavior

Open:

- `TRADING_RULES.md`
- `.claude/rules/quant-agent-identity.md`
- `.claude/rules/validation-workflow.md`

### Research, audits, or statistical claims

Open:

- `RESEARCH_RULES.md`
- `.claude/rules/quant-agent-identity.md`
- `.claude/skills/audit-prompts/SKILL.md`
- `.codex/AGENTS.md` if a shared agent pattern fits

### DB, schema, or canonical data questions

Open:

- `.claude/rules/mcp-usage.md`
- `.mcp.json`
- `.codex/INTEGRATIONS.md`
- `.codex/WORKFLOWS.md`

### Autonomous loop or plugin-driven work

Open:

- `plugins/ralph-wiggum/README.md`
- `.claude/agents/ralph-loop.md`
- `.claude/skills/ralph/SKILL.md`
- `.codex/AGENTS.md`
- `.codex/COMMANDS.md`

## Shared Systems

- Same workspace as Claude Code: `/mnt/c/Users/joshd/canompx3`
- Canonical DB: `gold.db`
- MCP declarations: `.mcp.json`
- Claude runtime settings: `.claude/settings.json`, `.claude/settings.local.json`
- Shared plugin assets: `plugins/`
- Shared agent memories: `.claude/agent-memory/`
- Codex user runtime config: `/home/joshd/.codex/config.toml`

## Local Runtime

- Codex CLI installed locally: `0.117.0`
- Default Codex model on this machine: `gpt-5.4`
- This repo is already marked as a trusted Codex project
- This repo also has project-scoped Codex config in `.codex/config.toml`
- Project-scoped Codex config injects additive startup instructions so direct Codex entry still gets the repo's preflight and handoff contract

## Codex Responsibilities

- Do not rename, replace, or fork the Claude layer.
- Do not reinterpret or weaken Claude rules.
- Use the same repo and files as Claude Code rather than creating parallel project state.
- It is fine to build rich Codex-facing docs and skills that reference the Claude layer directly.
- Persist durable context in `memory/YYYY-MM-DD.md` and `MEMORY.md`.
- Do not skip preflight or `HANDOFF.md` just because Codex was opened from a different surface.
- Keep Codex's layer useful, opinionated, and complete without mutating Claude unless explicitly asked.

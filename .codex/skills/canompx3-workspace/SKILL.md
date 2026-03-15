---
name: canompx3-workspace
description: Codex-specific workspace routing and onboarding for `C:\Users\joshd\canompx3`. Use when Codex is working in this repo and needs a minimal parallel layer that links directly to the shared Claude docs, rules, agents, plugins, memory, and trading-research workflow without replacing or mirroring the Claude setup.
---

# Canompx3 Workspace

Use this skill when Codex is operating in this repo and needs its own adapter layer on top of the existing Claude setup.

## Load Order

1. Read `AGENTS.md`.
2. Read `SOUL.md`, `USER.md`, `memory/<today>.md`, `memory/<yesterday>.md`, and `MEMORY.md` when in the main session.
3. Read `CLAUDE.md`.
4. Read `CODEX.md`.
5. Load only the extra files needed for the current task.

## Canonical Rule

- `CLAUDE.md` and `.claude/` remain canonical.
- This skill exists to help Codex navigate that setup without renaming, replacing, or mirroring it.

## Direct Links

- Trading logic or live behavior: `TRADING_RULES.md`
- Research and stats discipline: `RESEARCH_RULES.md`, `.claude/rules/quant-agent-identity.md`
- Pipeline or trading app edits: `.claude/rules/workflow-preferences.md`, `.claude/rules/validation-workflow.md`
- Pipeline schema or joins: `.claude/rules/pipeline-patterns.md`, `.claude/rules/daily-features-joins.md`
- Shared rules: `.claude/rules/`
- Shared agents: `.claude/agents/`
- Shared skills: `.claude/skills/`
- Shared agent memory: `.claude/agent-memory/`
- Shared plugin workflow: `plugins/ralph-wiggum/README.md`
- DB or MCP work: `.claude/rules/mcp-usage.md`, `.mcp.json`

# Codex Deference

This file defines how Codex defers to the canonical Claude layer in this repo.

## Canonical Sources

- Workspace identity and continuity:
  - `AGENTS.md`
  - `SOUL.md`
  - `USER.md`
  - `memory/*.md`
  - `MEMORY.md` in the main session
- Project architecture and engineering guardrails:
  - `CLAUDE.md`
  - `.claude/rules/*.md`
- Trading behavior:
  - `TRADING_RULES.md`
- Research standards and anti-bias discipline:
  - `RESEARCH_RULES.md`

## Codex Layer Scope

Codex may maintain only a thin adapter layer:

- `CODEX.md`
- `.codex/*.md`
- `.codex/skills/`

This layer should improve Codex navigation, execution defaults, and task routing. It must not create a separate authority, fork the shared project truth, or introduce parallel project state.

## Conflict Resolution

If there is a conflict:

1. `TRADING_RULES.md` wins for trading logic.
2. `RESEARCH_RULES.md` wins for research methodology and statistical claims.
3. `CLAUDE.md` and `.claude/` win for code workflow, guardrails, and shared runtime behavior.
4. `CODEX.md` and `.codex/` only describe Codex launch and routing details that do not conflict with Claude.

## Multiple LLM Rule

Multiple LLM clients are acceptable here only when they operate on the same canonical project state.

- Claude remains the canonical shared layer.
- Codex may add thin adapter docs and skills.
- Codex must not rename, replace, or silently drift the Claude layer.

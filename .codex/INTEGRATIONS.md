# Codex Integrations

This repo already has shared runtime integrations. Codex should use them, not recreate them.

## Primary Rule

Codex uses the same project as Claude Code and defers to Claude's project rules.

- Same repo
- Same files
- Same git history
- Same canonical DB if the project still uses one
- No Codex-only project copy
- No Codex-only database

## MCP

Declared in `.mcp.json`:

- `gold-db`
  - Shared repo MCP declaration
  - Optional project integration, not required for Codex to work in this repo
- `notebooklm`
  - Declared in `.mcp.json` but not enabled in `.claude/settings.local.json`
  - Repo rule state: retired for this workspace; use local PDFs instead

Project-scoped Codex MCP declarations also exist in `.codex/config.toml`:

- `openaiDeveloperDocs`

## Claude Runtime Surfaces Worth Reusing

- `.claude/settings.json`: shared plugin and hook configuration
- `.claude/settings.local.json`: local permissions and enabled MCP-json servers
- `.claude/hooks/`: post-edit and pre-edit safety hooks
- `.claude/agent-memory/`: shared agent memory for reusable workflows

## Local Codex Runtime Facts

- Codex binary: `/home/joshd/.nvm/versions/node/v24.14.0/bin/codex`
- User config: `/home/joshd/.codex/config.toml`
- Current default model: `gpt-5.4`
- Project trust is already set for this repo
- `codex mcp list` now includes `openaiDeveloperDocs`

## Current Codex CLI Capabilities Confirmed Locally

- `codex review` for non-interactive code review
- `codex exec` for non-interactive runs
- `codex mcp add|get|list|remove`
- profiles via `-p`
- live web search via `--search`
- `--full-auto` alias for `workspace-write` plus `on-request`

## Important Distinction

Claude and Codex do not share MCP registration automatically.

- Claude reads the repo’s `.mcp.json` plus Claude settings.
- Codex keeps its own MCP registrations in the user Codex config.
- This repo defines project-scoped Codex profiles in `.codex/config.toml`.
- Optional MCP integrations should stay minimal and must not imply a second project or second DB.
- Codex-side integrations must never supersede Claude rules or shared repo behavior.

## Repo Execution Defaults

- Sync environment: `uv sync --frozen`
- Run drift: `uv run python pipeline/check_drift.py`
- Run tests: `uv run python -m pytest tests/ -x -q`
- Run lint: `uv run ruff check pipeline/ trading_app/ ui/ scripts/ tests/`
- Run type checks: `uv run pyright`

## Practical Rule

If Claude already has a working integration surface for this repo, Codex should point at it rather than creating a competing layer.
